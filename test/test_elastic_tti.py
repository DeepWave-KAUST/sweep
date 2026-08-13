import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.equations import ElasticTTI, ElasticTTISG, supports_torch_binding
from sweep.equations.elastic_tti import STIFFNESS_KEYS
from sweep.operators.rsg import RSGDerivative, _build_Lx_kernels, _build_Lz_kernels
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch


def _torch_rsg(spatial_order=4, spacing=(10.0, 15.0)):
    op = RSGDerivative(spatial_order=spatial_order, backend="torch")
    op.to_backend(lambda arr, backend, device: torch.from_numpy(arr).float())
    op.set_spacing(spacing)
    return op


def test_rsg_constant_field():
    op = _torch_rsg()
    u = torch.ones(1, 1, 64, 64)
    interior = (slice(None), slice(None), slice(8, -8), slice(8, -8))

    for fn in (op.lx_fwd, op.lz_fwd, op.lx_bwd, op.lz_bwd):
        assert torch.allclose(fn(u)[interior], torch.zeros_like(u[interior]), atol=1e-5)


def test_rsg_linear_fields():
    dz, dx = 10.0, 15.0
    op = _torch_rsg(spacing=(dz, dx))
    z = torch.arange(64, dtype=torch.float32).view(1, 1, 64, 1).expand(1, 1, 64, 64) * dz
    x = torch.arange(64, dtype=torch.float32).view(1, 1, 1, 64).expand(1, 1, 64, 64) * dx
    interior = (slice(None), slice(None), slice(8, -8), slice(8, -8))

    assert torch.allclose(op.lx_fwd(x)[interior], torch.ones_like(x[interior]), atol=1e-4)
    assert torch.allclose(op.lz_fwd(z)[interior], torch.ones_like(z[interior]), atol=1e-4)
    assert torch.allclose(op.lx_fwd(z)[interior], torch.zeros_like(z[interior]), atol=1e-4)
    assert torch.allclose(op.lz_fwd(x)[interior], torch.zeros_like(x[interior]), atol=1e-4)

    assert torch.allclose(op.lx_bwd(x)[interior], torch.ones_like(x[interior]), atol=1e-4)
    assert torch.allclose(op.lz_bwd(z)[interior], torch.ones_like(z[interior]), atol=1e-4)
    assert torch.allclose(op.lx_bwd(z)[interior], torch.zeros_like(z[interior]), atol=1e-4)
    assert torch.allclose(op.lz_bwd(x)[interior], torch.zeros_like(x[interior]), atol=1e-4)


def test_rsg_coefficients_match_thesis_linear_system():
    from sweep.scalars import staggered_grid_coes

    for half_order in (1, 2, 3, 4):
        matrix = np.array(
            [[(2 * m + 1) ** (2 * k + 1) for m in range(half_order)] for k in range(half_order)],
            dtype=np.float64,
        )
        rhs = np.zeros(half_order)
        rhs[0] = 1.0
        expected = np.linalg.solve(matrix, rhs)
        assert np.allclose(staggered_grid_coes(half_order), expected, atol=1e-5)


def test_rsg_forward_backward_shift_relation():
    for half_order in (1, 2, 3, 4):
        for build in (_build_Lx_kernels, _build_Lz_kernels):
            fwd, bwd = build(half_order)
            shifted = np.zeros_like(fwd)
            shifted[:-1, :-1] = fwd[1:, 1:]
            assert np.allclose(shifted, bwd)


def test_elastic_tti_exports_and_binding_flag():
    equation = ElasticTTI(spatial_order=4, device="cpu", backend="torch")
    equation_sg = ElasticTTISG(spatial_order=4, device="cpu", backend="torch")

    assert equation.models == ["vp0", "vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"]
    assert len(equation.wavefields) == 20
    assert equation.default_source_fields == ["sxx", "szz"]
    assert equation.default_receiver_fields == ["vx", "vz"]
    assert equation_sg.models == equation.models
    assert equation_sg.wavefields == equation.wavefields
    assert equation_sg.default_source_fields == ["sxx", "szz"]
    assert equation_sg.default_receiver_fields == ["vx", "vz"]
    assert supports_torch_binding("ElasticTTI") is False
    assert supports_torch_binding("ElasticTTISG") is True


def test_elastic_tti_bond_rotation_preserves_vti_at_zero_tilt():
    shape = (3, 4)
    C11 = torch.full(shape, 10.0)
    C13 = torch.full(shape, 3.0)
    C33 = torch.full(shape, 8.0)
    C44 = torch.full(shape, 2.0)
    C66 = torch.full(shape, 4.0)
    theta = torch.zeros(shape)
    phi = torch.full(shape, 0.7)

    stiffness = ElasticTTI._bond_rotate(C11, C13, C33, C44, C66, theta, phi)
    expected = {
        "C11": C11,
        "C13": C13,
        "C33": C33,
        "C44": C44,
        "C55": C44,
        "C66": C66,
    }

    for key in STIFFNESS_KEYS:
        reference = expected.get(key, torch.zeros_like(C11))
        assert torch.allclose(stiffness[key], reference, atol=1e-5)


def _elastic_tti_problem(equation_cls=ElasticTTI, *, free_surface=False):
    nt = 10
    dt = 5e-4
    dh = 10.0
    shape = (20, 24)
    dev = torch.device("cpu")
    is_sg = equation_cls is ElasticTTISG
    equation = equation_cls(spatial_order=4, device=dev, backend="torch")
    solver = PropTorch(
        equation,
        shape=shape,
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        abcn=4,
        free_surface=free_surface,
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vy", "vz"],
        pml_type="cpmls" if is_sg else "cpmlr",
        backend="torch",
        impl="eager",
        eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
    )

    t = np.arange(nt, dtype=np.float32) * dt - 0.003
    wavelet = 1e5 * (1.0 - 2.0 * (np.pi * 25.0 * t) ** 2) * np.exp(-(np.pi * 25.0 * t) ** 2)
    source_z = 5 if free_surface else shape[0] // 2
    sources = np.array([[shape[1] // 2, source_z]], dtype=np.int64)
    rec_x = np.arange(4, shape[1] - 4, 4, dtype=np.int64)
    rec_z = np.full_like(rec_x, source_z)
    receivers = np.stack([rec_x, rec_z], axis=1)[None]

    models = [
        torch.full(shape, 2400.0, requires_grad=True),
        torch.full(shape, 1200.0, requires_grad=True),
        torch.full(shape, 2000.0, requires_grad=True),
        torch.full(shape, 0.12),
        torch.full(shape, 0.04),
        torch.full(shape, 0.08),
        torch.full(shape, 0.25),
        torch.full(shape, 0.15),
    ]
    return solver, wavelet.astype(np.float32), sources, receivers, models


@pytest.mark.parametrize("equation_cls", [ElasticTTI, ElasticTTISG])
def test_elastic_tti_free_surface_raises(equation_cls):
    """The isotropic image method is wrong for an anisotropic medium; the
    propagator refuses free_surface on TTI (see test_aniso_free_surface_guard)."""
    with pytest.raises(NotImplementedError, match="anisotropic"):
        _elastic_tti_problem(equation_cls, free_surface=True)


@pytest.mark.parametrize("equation_cls", [ElasticTTI, ElasticTTISG])
@pytest.mark.parametrize("free_surface", [False])
def test_elastic_tti_short_forward_and_gradient_are_finite(equation_cls, free_surface):
    solver, wavelet, sources, receivers, models = _elastic_tti_problem(equation_cls, free_surface=free_surface)

    record = solver(wavelet, sources, receivers, models=models)
    assert record.shape == (1, wavelet.shape[0], receivers.shape[1], 3)
    assert torch.isfinite(record).all()

    loss = record.square().sum()
    loss.backward()
    assert models[0].grad is not None
    assert torch.isfinite(models[0].grad).all()
    assert models[0].grad.abs().max() > 0

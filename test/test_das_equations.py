import numpy as np
import pytest
import torch

from sweep.equations import (
    DAS,               # unified DAS facade (was DASModeler)
    DASElastic,
    DASElastic3D,
    DASModeler,        # back-compat alias of DAS (kept for the rename test below)
    DASMu,
    DASMu3D,
    DASZhao,
    DASZhao3D,
    gauge_average,
    helical_das_response,
)
from sweep.propagator.torch import PropTorch


def _ricker(nt, dt, fm=12.0, delay=0.04):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def test_helical_response_weights_and_gauge_average():
    exx = torch.tensor([1.0, 2.0, 4.0])
    eyy = torch.tensor([10.0, 20.0, 40.0])
    ezz = torch.tensor([100.0, 200.0, 400.0])

    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=35.3), exx + eyy + ezz)
    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="x"), 4 * exx + eyy + ezz)
    assert torch.allclose(helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="z"), exx + eyy + 4 * ezz)

    averaged = gauge_average(torch.arange(5.0), gauge_cells=3)
    expected = torch.tensor([(0 + 0 + 1) / 3, 1.0, 2.0, 3.0, (3 + 4 + 4) / 3])
    assert torch.allclose(averaged, expected)


def test_das_equations_are_exported_with_cuda_binding():
    import sweep.equations as eq

    classes = eq._equation_classes()
    assert classes["DASElastic"] is DASElastic
    assert classes["DASElastic3D"] is DASElastic3D
    assert classes["DASMu"] is DASMu
    assert classes["DASMu3D"] is DASMu3D
    assert classes["DASZhao"] is DASZhao
    assert classes["DASZhao3D"] is DASZhao3D
    # DAS is now the unified facade (formerly DASModeler), not an equation
    # class.  DASElastic / DASElastic3D remain as raw equation-class aliases.
    assert DASModeler is DAS
    assert DASElastic.supports_torch_binding()
    assert DASElastic3D.supports_torch_binding()
    assert DASMu.supports_torch_binding()
    assert DASMu3D.supports_torch_binding()


def test_das_modeler_zhao_uses_standard_record_layout():
    device = torch.device("cpu")
    nt = 18
    dt = 0.001
    shape = (22, 24)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 5]]], dtype=np.int32)
    receivers = np.array([[[8, 6], [14, 7], [18, 8]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32)
    vs = torch.full(shape, 1200.0, dtype=torch.float32)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    modeler = DAS(
        method="zhao",
        ndim=2,
        spatial_order=2,
        shape=shape,
        receiver_type=["exx_t", "ezz_t", "das35_t"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = modeler(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 3)
    assert modeler.channels == {"exx_t": 0, "ezz_t": 1, "das35_t": 2}
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def test_das_modeler_mu_outputs_physical_strain_rate_and_das_fields():
    device = torch.device("cpu")
    nt = 18
    dt = 0.001
    shape = (24, 28)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 6]]], dtype=np.int32)
    receivers = np.array([[[8, 7], [14, 8], [20, 9]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    modeler = DAS(
        method="mu",
        ndim=2,
        spatial_order=2,
        shape=shape,
        receiver_type=["vx", "vz", "exx_t", "ezz_t", "das35_t"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = modeler(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 5)
    assert modeler.channels == {"vx": 0, "vz": 1, "exx_t": 2, "ezz_t": 3, "das35_t": 4}
    assert modeler.solver_receiver_type == ("vx", "vz", "exx", "ezz")
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out[..., [modeler.channels["exx_t"], modeler.channels["ezz_t"], modeler.channels["das35_t"]]].pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_das_mu_supports_all_physical_fields_as_source_receiver_with_free_surface():
    device = torch.device("cpu")
    nt = 18
    dt = 0.001
    shape = (24, 28)
    fields = ["vx", "vz", "sxx", "szz", "sxz", "exx", "ezz", "exz"]
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 6]]], dtype=np.int32)
    receivers = np.array([[[8, 7], [14, 8], [20, 9]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASMu(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=fields,
        receiver_type=fields,
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        free_surface=True,
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], len(fields))
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out[..., fields.index("exz")].pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_das_modeler_mu_3d_outputs_physical_strain_rate_and_das_fields():
    device = torch.device("cpu")
    nt = 10
    dt = 0.001
    shape = (8, 8, 8)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.004).reshape(1, nt)
    sources = np.array([[[4, 4, 3]]], dtype=np.int32)
    receivers = np.array([[[3, 3, 3], [5, 5, 4]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    modeler = DAS(
        method="mu",
        ndim=3,
        spatial_order=2,
        shape=shape,
        receiver_type=["vx", "vy", "vz", "exx_t", "eyy_t", "ezz_t", "das35_t", "das54y_t"],
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = modeler(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 8)
    assert modeler.channels == {
        "vx": 0,
        "vy": 1,
        "vz": 2,
        "exx_t": 3,
        "eyy_t": 4,
        "ezz_t": 5,
        "das35_t": 6,
        "das54y_t": 7,
    }
    assert modeler.solver_receiver_type == ("vx", "vy", "vz", "exx", "eyy", "ezz")
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out[..., [modeler.channels["exx_t"], modeler.channels["eyy_t"], modeler.channels["ezz_t"]]].pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_das_mu3d_supports_all_physical_fields_as_source_receiver_with_free_surface():
    device = torch.device("cpu")
    nt = 8
    dt = 0.001
    shape = (8, 8, 8)
    fields = ["vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz", "exx", "eyy", "ezz", "exy", "exz", "eyz"]
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.004).reshape(1, nt)
    sources = np.array([[[4, 4, 3]]], dtype=np.int32)
    receivers = np.array([[[3, 3, 3], [5, 5, 4]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASMu3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=fields,
        receiver_type=fields,
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        free_surface=True,
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], len(fields))
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out[..., fields.index("exz")].pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


def test_das_modeler_mu_variant():
    device = torch.device("cpu")
    nt = 16
    dt = 0.001
    shape = (22, 26)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 5]]], dtype=np.int32)
    receivers = np.array([[[8, 6], [14, 7]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32)
    vs = torch.full(shape, 1200.0, dtype=torch.float32)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    modeler = DAS(
        method="mu",
        ndim=2,
        spatial_order=2,
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz", "exx", "ezz", "exz"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = modeler(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 5)
    assert modeler.channels == {"vx": 0, "vz": 1, "exx": 2, "ezz": 3, "exz": 4}
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def test_das_elastic_2d_eager_forward_and_gradient():
    device = torch.device("cpu")
    nt = 28
    dt = 0.001
    shape = (24, 28)
    wavelet = _ricker(nt, dt).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 3]]], dtype=np.int32)
    receivers = np.stack(
        [
            np.arange(4, shape[1] - 4, 6, dtype=np.int32),
            np.full(len(np.arange(4, shape[1] - 4, 6)), 5, dtype=np.int32),
        ],
        axis=-1,
    )[None, ...]

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    solver = PropTorch(
        DASElastic(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=["exx_t", "ezz_t", "das35_t", "das54x_t"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 4)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    loss = out.pow(2).mean()
    loss.backward()
    assert vp.grad is not None and torch.isfinite(vp.grad).all()
    assert vs.grad is not None and torch.isfinite(vs.grad).all()


def test_das_elastic_3d_field_metadata():
    equation = DASElastic3D(spatial_order=2, backend="torch")
    assert equation.models == ["vp", "vs", "rho"]
    assert equation.default_source_fields == ["sxx", "syy", "szz"]
    assert equation.default_receiver_fields == ["exx_t", "eyy_t", "ezz_t"]
    assert "das54z_t" in equation.wavefields
    assert "m_sxx_xf" in equation.wavefields
    assert "m_tzz_yb" in equation.wavefields


def test_das_elastic_3d_eager_forward_smoke():
    device = torch.device("cpu")
    nt = 8
    dt = 0.001
    shape = (8, 8, 8)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.003).reshape(1, nt)
    sources = np.array([[[4, 4, 3]]], dtype=np.int32)
    receivers = np.array([[[3, 3, 3], [5, 5, 3]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32)
    vs = torch.full(shape, 1200.0, dtype=torch.float32)
    rho = torch.full(shape, 2100.0, dtype=torch.float32)

    solver = PropTorch(
        DASElastic3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["exx_t", "eyy_t", "ezz_t", "das35_t", "das54x_t"],
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 5)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


def _assert_nonzero_finite_gradient(*models):
    for model in models:
        assert model.grad is not None
        assert torch.isfinite(model.grad).all()
        assert model.grad.abs().max() > 0


def test_das_final_channels_backward_2d():
    device = torch.device("cpu")
    nt = 20
    dt = 0.001
    shape = (24, 28)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[shape[1] // 2, 5]]], dtype=np.int32)
    receivers = np.array([[[8, 6], [14, 6], [20, 8], [14, 12]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASElastic(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "szz"],
        receiver_type=["das35_t", "das54x_t", "das54z_t"],
        abcn=4,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 3)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out.pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    ("name", "boundary_saving_config"),
    [
        ("full", {"enabled": False}),
        ("bs_gpu", {"enabled": True, "storage": "gpu"}),
    ],
)
def test_das_elastic_2d_cuda_backward_matches_eager_with_random_adjoint(name, boundary_saving_config):
    device = torch.device("cuda:0")
    nt = 18
    dt = 0.001
    shape = (28, 32)
    wavelet = torch.as_tensor(_ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt), device=device)
    sources = np.array([[[shape[1] // 2, 7]]], dtype=np.int32)
    receivers = np.array([[[9, 8], [15, 9], [21, 10]]], dtype=np.int32)
    receiver_type = ["exx_t", "ezz_t", "das35_t", "das54x_t", "das54z_t"]

    def run(backend, adjoint_weight=None):
        vp = torch.full(shape, 2200.0, dtype=torch.float32, device=device, requires_grad=True)
        vs = torch.full(shape, 1200.0, dtype=torch.float32, device=device, requires_grad=True)
        rho = torch.full(shape, 2100.0, dtype=torch.float32, device=device, requires_grad=True)
        solver = PropTorch(
            DASElastic(spatial_order=2, device=device, backend="torch"),
            shape=shape,
            source_type=["sxx", "szz"],
            receiver_type=receiver_type,
            abcn=4,
            dh=10.0,
            dt=dt,
            dev=device,
            pml_type="cpmls",
            use_ckpt=False,
            backend=backend,
        )
        out = solver(
            wavelet,
            sources=sources,
            receivers=receivers,
            models=[vp, vs, rho],
            boundary_saving_config={"enabled": False} if backend == "eager" else boundary_saving_config,
        )
        # Both backends return the canonical (B, nt, nrec, nfield) record;
        # the legacy `permute(1, 3, 2, 0)` here was a remnant from when the
        # CUDA wrapper still emitted raw (nfield, B, nrec, nt) and is no
        # longer needed.
        if adjoint_weight is None:
            torch.manual_seed(2026)
            adjoint_weight = torch.randn_like(out)
        loss = (out * adjoint_weight).sum()
        loss.backward()
        torch.cuda.synchronize(device)
        return out.detach(), vp.grad.detach(), vs.grad.detach(), rho.grad.detach(), adjoint_weight

    eager_out, eager_gvp, eager_gvs, eager_grho, adjoint_weight = run("eager")
    cuda_out, cuda_gvp, cuda_gvs, cuda_grho, _ = run("cuda", adjoint_weight)

    torch.testing.assert_close(cuda_out, eager_out, rtol=2e-4, atol=1e-12, msg=f"{name} forward mismatch")
    torch.testing.assert_close(cuda_gvp, eager_gvp, rtol=2e-3, atol=1e-14, msg=f"{name} vp gradient mismatch")
    torch.testing.assert_close(cuda_grho, eager_grho, rtol=2e-3, atol=1e-14, msg=f"{name} rho gradient mismatch")

    # vs gradient: exactly one cell breaks the tight tolerance, and it is the
    # source cell.  There the float32 *eager* reference is itself inaccurate —
    # that is where the wavefield dynamic range peaks, and
    # grad_vs = 2*rho*vs*(grad_mu - 2*grad_lambda) subtracts two nearly equal
    # terms, so the long float32 autograd chain loses digits.  Adjudicated
    # against a float64 eager run (same discretisation, same adjoint weights):
    # over the whole grid the CUDA adjoint is off by 7.2e-7 relative and float32
    # eager by 3.1e-4; at the source cell float64 says 8.9234e-14, CUDA says
    # 8.9224e-14 (1.1e-5 off) and float32 eager says 1.0373e-13 — 16% off.  The
    # reference, not the CUDA kernel, is the imprecise side, so the source
    # neighbourhood is compared loosely while the rest of the grid keeps the
    # tight elementwise tolerance.
    halo = 1  # spatial_order // 2
    src_mask = torch.zeros_like(eager_gvs, dtype=torch.bool)
    for sx, sz in sources[0]:
        z0, z1 = max(int(sz) - halo, 0), min(int(sz) + halo + 1, src_mask.shape[0])
        x0, x1 = max(int(sx) - halo, 0), min(int(sx) + halo + 1, src_mask.shape[1])
        src_mask[z0:z1, x0:x1] = True
    torch.testing.assert_close(
        cuda_gvs[~src_mask], eager_gvs[~src_mask], rtol=2e-3, atol=1e-14,
        msg=f"{name} vs gradient mismatch away from the source",
    )
    torch.testing.assert_close(
        cuda_gvs[src_mask], eager_gvs[src_mask], rtol=1e-1, atol=1e-14,
        msg=f"{name} vs gradient mismatch on the source stencil",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
@pytest.mark.parametrize(
    ("name", "boundary_saving_config"),
    [
        ("full", {"enabled": False}),
        ("bs_gpu", {"enabled": True, "storage": "gpu"}),
    ],
)
def test_das_elastic_3d_cuda_backward_matches_eager_with_encoded_wavelet(name, boundary_saving_config):
    device = torch.device("cuda:0")
    nt = 16
    dt = 0.001
    shape = (14, 12, 14)
    wavelet = torch.as_tensor(_ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt), device=device)
    sources = np.array([[[shape[2] // 2, shape[1] // 2, 5]]], dtype=np.int32)
    receivers = np.array([[[5, 5, 5], [7, 6, 5], [9, 7, 6], [7, 5, 7]]], dtype=np.int32)
    receiver_type = [
        "exx_t",
        "eyy_t",
        "ezz_t",
        "das35_t",
        "das54x_t",
        "das54y_t",
        "das54z_t",
    ]

    def run(backend, adjoint_weight=None):
        vp = torch.full(shape, 2200.0, dtype=torch.float32, device=device, requires_grad=True)
        vs = torch.full(shape, 1200.0, dtype=torch.float32, device=device, requires_grad=True)
        rho = torch.full(shape, 2100.0, dtype=torch.float32, device=device, requires_grad=True)
        solver = PropTorch(
            DASElastic3D(spatial_order=2, device=device, backend="torch"),
            shape=shape,
            source_type=["sxx", "syy", "szz"],
            receiver_type=receiver_type,
            abcn=3,
            dh=10.0,
            dt=dt,
            dev=device,
            pml_type="cpmls",
            use_ckpt=False,
            backend=backend,
        )
        out = solver(
            wavelet,
            sources=sources,
            receivers=receivers,
            models=[vp, vs, rho],
            boundary_saving_config={"enabled": False} if backend == "eager" else boundary_saving_config,
        )
        # See note on the 2-D variant: CUDA record is already canonical.
        if adjoint_weight is None:
            torch.manual_seed(2027)
            adjoint_weight = torch.randn_like(out)
        loss = (out * adjoint_weight).sum()
        loss.backward()
        torch.cuda.synchronize(device)
        return out.detach(), vp.grad.detach(), vs.grad.detach(), rho.grad.detach(), adjoint_weight

    eager_out, eager_gvp, eager_gvs, eager_grho, adjoint_weight = run("eager")
    cuda_out, cuda_gvp, cuda_gvs, cuda_grho, _ = run("cuda", adjoint_weight)

    torch.testing.assert_close(cuda_out, eager_out, rtol=5e-4, atol=1e-12, msg=f"{name} forward mismatch")
    torch.testing.assert_close(cuda_gvp, eager_gvp, rtol=3e-3, atol=1e-14, msg=f"{name} vp gradient mismatch")
    torch.testing.assert_close(cuda_gvs, eager_gvs, rtol=5e-3, atol=1e-14, msg=f"{name} vs gradient mismatch")
    torch.testing.assert_close(cuda_grho, eager_grho, rtol=3e-3, atol=1e-14, msg=f"{name} rho gradient mismatch")


def test_das_final_channels_backward_3d():
    device = torch.device("cpu")
    nt = 14
    dt = 0.001
    shape = (10, 10, 10)
    wavelet = _ricker(nt, dt, fm=15.0, delay=0.006).reshape(1, nt)
    sources = np.array([[[5, 5, 5]]], dtype=np.int32)
    receivers = np.array([[[4, 5, 5], [6, 5, 5], [5, 4, 5], [5, 5, 6]]], dtype=np.int32)

    vp = torch.full(shape, 2200.0, dtype=torch.float32, requires_grad=True)
    vs = torch.full(shape, 1200.0, dtype=torch.float32, requires_grad=True)
    rho = torch.full(shape, 2100.0, dtype=torch.float32, requires_grad=True)

    solver = PropTorch(
        DASElastic3D(spatial_order=2, device=device, backend="torch"),
        shape=shape,
        source_type=["sxx", "syy", "szz"],
        receiver_type=["das35_t", "das54x_t", "das54y_t", "das54z_t"],
        abcn=2,
        dh=10.0,
        dt=dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
    )

    out = solver(wavelet, sources=sources, receivers=receivers, models=[vp, vs, rho])
    assert out.shape == (1, nt, receivers.shape[1], 4)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0

    out.pow(2).mean().backward()
    _assert_nonzero_finite_gradient(vp, vs, rho)

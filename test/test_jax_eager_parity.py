"""PropJax steps the same discrete equation as the eager PropTorch.

Four ways the jax path solved a different problem (or none), all found by the
jax-vs-eager column of a gradient verification run:

* ``operators.jax.gradient`` flipped the first-derivative kernel before a
  cross-correlation, i.e. NEGATED every first derivative taken with the
  equation's FD kernels.  Where a first derivative multiplies another one
  (VRZ's ``grad vp . grad p``, the CPML's ``d(a psi)``) the signs cancelled;
  a lone one -- AcousticVRR's ``-2 vp^2 r . grad p``, the CPML ``db/dz du/dz``
  term -- ran with the wrong sign: AcousticVRR records 3e-2 off eager and its
  ``rx`` / ``rz`` gradients reversed (cos -0.998), VRZ / qP 1e-3.
* ``AcousticVRZ3D`` had no first-derivative kernels on jax, so its interior
  VRZ terms fell back to ``jnp.gradient`` (second order): gradients 13% / 37%
  off at spatial_order=4.
* ``ElasticVRR`` called ``float()`` on the spacing, a traced array on jax
  (TypeError), and kept an ``id()``-keyed cache of traced arrays.
* ``AcousticCurvilinear`` got its metrics as torch tensors (TypeError), and the
  jax path neither sampled the models on the boundary-fitted rows nor placed
  sources and receivers on them (records 40% off once that ran).
  ``ElasticCurvilinear``'s geometry is torch code: jax is refused up front.
* The qP S_n quotient ``num / (den + 1e-26)``: jax differentiates a division
  through ``x / y**2``, and ``y**2`` underflows to 0 in fp32, so wherever the
  field is quiet the epsilon / delta / theta gradients were NaN (and with them
  every model's under s_gradient='exact').  The quotient now carries torch's
  rule, ``(dn - S d) / (den + 1e-26)``, as a custom JVP.

CPU, fp32, small grids; the tolerances sit two orders below the defects.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from sweep.equations import (AcousticCurvilinear, AcousticVRR, AcousticVRZ,  # noqa: E402
                             AcousticVRZ3D, AcousticVTI, ElasticCurvilinear, ElasticVRR)
from sweep.equations.qP_vti import s_quotient  # noqa: E402
from sweep.propagator.jax import PropJax  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DH, DT, F0, SO = 10.0, 1e-3, 15.0, 4
VALUES = {"vp": 2500.0, "vs": 1400.0, "z": 5.0e6, "rx": 1e-4, "rz": 1e-4,
          "Rp_x": 0.01, "Rp_z": 0.01, "Rs_x": 0.01, "Rs_z": 0.01, "epsilon": 0.1, "delta": 0.05}


def _wavelet(nt):
    t = np.arange(nt) * DT - 1.2 / F0
    a = (np.pi * F0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def _case(shape, nt, three=False, topography=None, abcn=12):
    r = np.broadcast_to(np.linspace(0, 1, shape[0]).reshape((-1,) + (1,) * (len(shape) - 1)), shape)
    box = np.zeros(shape)
    box[tuple(slice(s // 3, 2 * s // 3) for s in shape)] = 1.0
    if three:
        nz, ny, nx = shape
        src = np.array([[nx // 2, ny // 2, nz // 3]], np.int64)
        gy, gx = np.meshgrid(np.arange(3, ny - 3, 4), np.arange(3, nx - 3, 4), indexing="ij")
        rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 4)], -1)[None].astype(np.int64)
    else:
        nz, nx = shape
        src = np.array([[nx // 2, nz // 3]], np.int64)
        xs = np.arange(3, nx - 3, 3)
        rec = np.stack([xs, np.full(xs.size, 6)], -1)[None].astype(np.int64)
    kw = {} if topography is None else {"topography": topography}
    return dict(shape=shape, nt=nt, r=r, box=box, src=src, rec=rec, kw=kw, abcn=abcn)


def _compare(cls, case, eq_kw=None):
    """(record rel L2, [gradient rel L2 per model]) of PropJax vs eager."""
    eq_kw = eq_kw or {}
    shape, nt, abcn = case["shape"], case["nt"], case["abcn"]
    eqt = cls(spatial_order=SO, device="cpu", backend="torch", **eq_kw)
    names = [n if isinstance(n, str) else n.name for n in eqt.models]
    init = [(VALUES[n] * (1 + 0.05 * case["r"])).astype(np.float32) for n in names]
    true = [(VALUES[n] * (1 + 0.05 * case["r"] + 0.03 * case["box"])).astype(np.float32) for n in names]
    w = _wavelet(nt)
    src, rec = case["src"], case["rec"]
    pt = PropTorch(eqt, shape=shape, dh=DH, dt=DT, nt=nt, abcn=abcn, dev="cpu", impl="eager", B=1,
                   memory=Full(), eager_options=EagerOptions(use_compile=False), **case["kw"])
    with torch.no_grad():
        scale = float(pt(torch.tensor(w), src, rec, models=[torch.tensor(m) for m in init]).abs().max())
    w = (w / scale).astype(np.float32)
    with torch.no_grad():
        obs_t = pt(torch.tensor(w), src, rec, models=[torch.tensor(m) for m in true]).double()
    ms = [torch.tensor(m, requires_grad=True) for m in init]
    rec_t = pt(torch.tensor(w), src, rec, models=ms)
    (0.5 * (rec_t.double() - obs_t).pow(2).sum()).backward()
    gt = [m.grad.double().numpy() for m in ms]

    eqj = cls(spatial_order=SO, device="cpu", backend="jax", **eq_kw)
    pj = PropJax(eqj, shape, dh=DH, dt=DT, abcn=abcn, nt=nt, B=1, use_ckpt=False,
                 pml_type=eqj.default_pml_type, **case["kw"])
    srcj = src[None]
    obs_j = pj(w[None], srcj, rec, models=[jnp.asarray(m) for m in true])
    rec_j = pj(w[None], srcj, rec, models=[jnp.asarray(m) for m in init])

    def loss(mm):
        return 0.5 * jnp.sum((pj(w[None], srcj, rec, models=list(mm)) - obs_j) ** 2)

    gj = [np.asarray(g, np.float64) for g in jax.grad(loss)([jnp.asarray(m) for m in init])]
    rt = rec_t.detach().double().numpy().ravel()
    rj = np.asarray(rec_j, np.float64).ravel()
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    return rel(rj, rt), [rel(a, b) for a, b in zip(gj, gt)]


def test_first_derivative_has_the_torch_sign():
    """The jax first derivative with the equation's kernels is the torch one
    (it was its negative)."""
    u = np.random.default_rng(0).standard_normal((1, 1, 20, 24)).astype(np.float32)
    et = AcousticVRR(spatial_order=SO, device="cpu", backend="torch")
    ej = AcousticVRR(spatial_order=SO, device="cpu", backend="jax")
    for axis in (-1, -2):
        gt = et.gradient(torch.tensor(u), DH, axis, kernels=et.grad_kernels).numpy()
        gj = np.asarray(ej.gradient(jnp.asarray(u), DH, axis, kernels=ej.grad_kernels))
        np.testing.assert_allclose(gj, gt, rtol=0, atol=1e-6 * np.abs(gt).max())


def test_vrz3d_first_derivatives_take_the_fd_order():
    """3-D kernels on jax too: the same 4th-order derivative as torch."""
    u = np.random.default_rng(1).standard_normal((1, 1, 12, 10, 14)).astype(np.float32)
    et = AcousticVRZ3D(spatial_order=SO, device="cpu", backend="torch")
    ej = AcousticVRZ3D(spatial_order=SO, device="cpu", backend="jax")
    assert ej.grad_kernels is not None
    for axis in (-1, -2, -3):
        gt = et.gradient(torch.tensor(u), DH, axis, kernels=et.grad_kernels).numpy()
        gj = np.asarray(ej.gradient(jnp.asarray(u), DH, axis, kernels=ej.grad_kernels))
        np.testing.assert_allclose(gj, gt, rtol=0, atol=1e-6 * np.abs(gt).max())


@pytest.mark.parametrize("cls", [AcousticVRR, AcousticVRZ, ElasticVRR], ids=lambda c: c.__name__)
def test_jax_matches_eager_2d(cls):
    rec, grads = _compare(cls, _case((32, 36), 220))
    assert rec < 1e-4, f"{cls.__name__}: record {rec:.2e}"
    assert max(grads) < 1e-3, f"{cls.__name__}: gradients {grads}"


def test_jax_matches_eager_vrz3d():
    rec, grads = _compare(AcousticVRZ3D, _case((20, 18, 20), 120, three=True, abcn=8))
    assert rec < 1e-4, f"record {rec:.2e}"
    assert max(grads) < 1e-3, f"gradients {grads}"


def test_jax_matches_eager_curvilinear_topography():
    nx = 36
    x = np.arange(nx)
    topo = np.round(3.0 * np.exp(-((x - nx / 2) / (nx / 6)) ** 2)).astype(np.int64)
    rec, grads = _compare(AcousticCurvilinear, _case((32, nx), 220, topography=topo))
    assert rec < 1e-4, f"record {rec:.2e}"
    assert max(grads) < 1e-3, f"gradients {grads}"


def test_elastic_curvilinear_refuses_jax():
    with pytest.raises(NotImplementedError, match="backend='torch' only"):
        ElasticCurvilinear(spatial_order=8, device="cpu", backend="jax")


def test_s_quotient_gradient_is_finite_where_the_field_is_quiet():
    """Zero numerator and denominator (a quiet cell): S = 0, derivative 0."""
    g = jax.grad(lambda d: jnp.sum(s_quotient(jnp.zeros(3, jnp.float32), d)))(jnp.zeros(3, jnp.float32))
    assert np.isfinite(np.asarray(g)).all() and float(jnp.abs(g).max()) == 0.0
    # and the derivative itself, away from zero, against the quotient rule
    n, d = jnp.float32(0.3), jnp.float32(2.0)
    gn, gd = jax.grad(lambda a, b: s_quotient(a, b), argnums=(0, 1))(n, d)
    np.testing.assert_allclose([float(gn), float(gd)], [1 / 2.0, -0.3 / 4.0], rtol=1e-6)


def test_qp_frozen_parameter_gradients_finite_and_match_eager():
    rec, grads = _compare(AcousticVTI, _case((32, 36), 220), eq_kw={"s_gradient": "frozen"})
    assert np.isfinite(grads).all(), f"gradients {grads}"
    assert rec < 1e-4 and max(grads) < 1e-2, f"record {rec:.2e}, gradients {grads}"

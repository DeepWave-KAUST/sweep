"""Receiver-side adjoint bugs in the compiled elastic backward (impl='c').

Two defects, both invisible to the canonical gradient suite because it only
ever records the default (velocity) receivers:

1. **Stress receivers negated every model gradient.**  The compiled elastic
   adjoint carries the stress multipliers with the opposite sign of the true
   adjoint (the -1 folded into the lambda/mu imaging terms), so a residual
   recorded on ``sxx`` / ``szz`` (3-D also ``syy``) has to be injected negated.
   It was injected raw, exactly as for a velocity receiver, and the whole
   d(loss)/d(vp,vs,rho) came out with cos = -1 against eager — correct
   magnitude, wrong sign.  An FWI driven by pressure/stress data would have
   walked uphill.

2. **rho gradient was wrong AT the receiver cells.**  The rho imaging
   correlates the adjoint velocity with the stored difference v(it) - v(it+1),
   and the residual for a velocity receiver had already been injected into that
   adjoint velocity when the imaging ran.  The discrete adjoint has no such
   term, so the imaging over-counted by resid(it) * (v(it) - v(it+1)) / rho at
   every receiver cell — a finite-difference check put impl='c' 16.5% above the
   true derivative there while eager matched it to 2.5e-5.

Both were mode-independent (full, boundary-saving and both checkpoint flavours
were equally wrong) and appeared for every source loading type.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

DT, DH, SO, NT = 8e-4, 10.0, 4, 300


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _binding_ready(),
                                reason="CUDA + compiled sweep._C required")


def _ramp(shape, top, bottom):
    nz = shape[0]
    r = np.linspace(top, bottom, nz, dtype=np.float32)
    return np.broadcast_to(r.reshape((nz,) + (1,) * (len(shape) - 1)), shape).copy()


def _models(shape):
    vp = _ramp(shape, 1800.0, 2400.0)
    vp[shape[0] // 3:shape[0] // 3 + 4] += 180.0
    vs = (vp / 1.73).astype(np.float32)
    rho = _ramp(shape, 1000.0, 1300.0)
    rho[shape[0] // 2:shape[0] // 2 + 3] += 150.0
    return vp, vs, rho


def _geom(shape):
    """Sources (nshots, ndim) and receivers (nshots, nrec, ndim), plus the
    receiver cell indices used to probe the rho gradient locally."""
    rec_z = 3
    if len(shape) == 2:
        nz, nx = shape
        src = np.array([[nx // 2, nz // 4]], np.int64)
        xs = np.arange(4, nx - 4, 6, dtype=np.int64)
        rec = np.stack([xs, np.full_like(xs, rec_z)], -1)[None]
        cells = (np.full_like(xs, rec_z), xs)
    else:
        nz, ny, nx = shape
        src = np.array([[nx // 2, ny // 2, nz // 4]], np.int64)
        xs = np.arange(4, nx - 4, 8, dtype=np.int64)
        ys = np.arange(4, ny - 4, 8, dtype=np.int64)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        rec = np.stack([gx, gy, np.full_like(gx, rec_z)], -1)[None]
        cells = (np.full_like(gx, rec_z), gy, gx)
    return src, rec, cells


def _grads(shape, impl, source_type, receiver_type, mode="full"):
    from sweep.propagator.options import CkptOptions, CUDAOptions, EagerOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch
    if len(shape) == 2:
        from sweep.equations import Elastic as EQ
    else:
        from sweep.equations import Elastic3D as EQ

    dev = "cuda"
    eq = EQ(spatial_order=SO, device=dev, backend="torch")
    kw = dict(backend="torch", shape=shape, abcn=16, dh=DH, dt=DT, dev=dev,
              pml_type="cpmls", nt=NT, B=1, free_surface=False,
              source_type=list(source_type), receiver_type=list(receiver_type))
    if impl == "eager":
        p = PropTorch(eq, impl="eager", use_ckpt=False,
                      eager_options=EagerOptions(use_compile=False), **kw)
    elif mode == "full":
        p = PropTorch(eq, impl="c", use_ckpt=False,
                      boundary_saving_config={"enabled": False}, **kw)
    elif mode == "bs":
        p = PropTorch(eq, impl="c", use_ckpt=False,
                      boundary_saving_config={"enabled": True, "storage": "gpu"}, **kw)
    elif mode == "ckpt":
        p = PropTorch(eq, impl="c", cuda_options=CUDAOptions(
            memory=MemoryOptions(strategy="ckpt",
                                 ckpt=CkptOptions(mode="chunk", chunks=20))), **kw)
    else:
        raise ValueError(mode)

    t = np.arange(NT, dtype=np.float32) * DT - 0.12
    x = np.pi * 8.0 * t
    wav = torch.tensor((1e3 * (1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32),
                       device=dev)
    src, rec, _ = _geom(shape)
    models = [torch.tensor(a, device=dev, requires_grad=True) for a in _models(shape)]
    p(wav, src.copy(), rec.copy(), models=models).pow(2).sum().backward()
    return [m.grad.detach().double().cpu() for m in models]


def _cos(a, b):
    return float(torch.dot(a.ravel() / a.norm().clamp_min(1e-30),
                           b.ravel() / b.norm().clamp_min(1e-30)))


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# --------------------------------------------------------------------------- #
# 1. stress receivers must not flip the sign of the gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["full", "bs", "ckpt"])
@pytest.mark.parametrize("receiver_type", [["sxx"], ["szz"], ["sxx", "szz"],
                                           ["vz", "sxx"]])
def test_elastic2d_stress_receiver_gradient_sign(mode, receiver_type):
    shape = (48, 56)
    ref = _grads(shape, "eager", ["vz"], receiver_type)
    got = _grads(shape, "c", ["vz"], receiver_type, mode=mode)
    for name, a, b in zip(("vp", "vs", "rho"), got, ref):
        cos = _cos(a, b)
        assert cos > 0.999, (
            f"receiver_type={receiver_type} mode={mode}: {name} cos={cos:.6f} "
            f"(cos ~ -1 means the adjoint residual was injected into the stress "
            f"field without the sign flip the compiled adjoint convention needs)")


@pytest.mark.parametrize("receiver_type", [["szz"], ["sxx", "syy", "szz"]])
def test_elastic3d_stress_receiver_gradient_sign(receiver_type):
    shape = (24, 20, 24)
    ref = _grads(shape, "eager", ["vz"], receiver_type)
    for mode in ("full", "bs"):
        got = _grads(shape, "c", ["vz"], receiver_type, mode=mode)
        for name, a, b in zip(("vp", "vs", "rho"), got, ref):
            cos = _cos(a, b)
            assert cos > 0.999, (f"receiver_type={receiver_type} mode={mode}: "
                                 f"{name} cos={cos:.6f}")


def test_elastic2d_stress_source_and_stress_receiver():
    """The sign is a property of the RECEIVER field, not of the source."""
    shape = (48, 56)
    ref = _grads(shape, "eager", ["sxx", "szz"], ["sxx", "szz"])
    got = _grads(shape, "c", ["sxx", "szz"], ["sxx", "szz"])
    for name, a, b in zip(("vp", "vs", "rho"), got, ref):
        assert _cos(a, b) > 0.999, f"{name} cos={_cos(a, b):.6f}"


# --------------------------------------------------------------------------- #
# 2. rho gradient at the receiver cells
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["full", "bs", "ckpt"])
def test_elastic2d_receiver_cell_rho_gradient(mode):
    shape = (48, 56)
    _, _, cells = _geom(shape)
    ref = _grads(shape, "eager", ["vz"], ["vx", "vz"])[2]
    got = _grads(shape, "c", ["vz"], ["vx", "vz"], mode=mode)[2]
    # Probe the receiver cells directly: the bug lived entirely there and was
    # diluted ~20x in the whole-field norm.  (The whole field is NOT asserted
    # here — it still carries a separate, pre-existing ~0.4% residual at the
    # body-force SOURCE cell, see test_body_force_rho_grad.py.)
    rel = _rel(got[cells], ref[cells])
    assert rel < 1e-3, (
        f"mode={mode}: rho gradient at the receiver cells rel={rel:.2e} "
        f"(the just-injected adjoint residual is being correlated with "
        f"v(it) - v(it+1) there; the discrete adjoint has no such term)")


def test_elastic3d_receiver_cell_rho_gradient():
    shape = (24, 20, 24)
    _, _, cells = _geom(shape)
    ref = _grads(shape, "eager", ["vz"], ["vx", "vy", "vz"])[2]
    for mode in ("full", "bs"):
        got = _grads(shape, "c", ["vz"], ["vx", "vy", "vz"], mode=mode)[2]
        rel = _rel(got[cells], ref[cells])
        assert rel < 1e-3, f"mode={mode}: receiver-cell rho rel={rel:.2e}"


@pytest.mark.parametrize("source_type", [["sxx", "szz"], ["sxz"], ["vx"]])
def test_elastic2d_receiver_cell_rho_gradient_source_types(source_type):
    """The receiver-cell term is independent of how the source is loaded."""
    shape = (48, 56)
    _, _, cells = _geom(shape)
    ref = _grads(shape, "eager", source_type, ["vx", "vz"])[2]
    got = _grads(shape, "c", source_type, ["vx", "vz"])[2]
    rel = _rel(got[cells], ref[cells])
    assert rel < 1e-3, f"source_type={source_type}: receiver-cell rho rel={rel:.2e}"

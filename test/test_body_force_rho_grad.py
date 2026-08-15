"""Body-force (velocity) sources and the rho gradient on impl='c'.

The rho-gradient imaging correlates the adjoint velocity with stored
wavefield differences v(it) - v(it+1).  At a source cell those differences
still contain the raw injected amplitude, which the true derivative does not
(the injection is rho-independent), so before the fix the CUDA adjoint's rho
gradient was wrong AT THE SOURCE CELL whenever a velocity source was used:
elastic2d c-vs-eager rho cos 0.648 (rel 0.85) on a heterogeneous model, and
FD arbitration showed impl='c' captured only ~12% of the true derivative at
the source cell while eager matched FD to 1e-5.  Stress sources were always
fine (their injection enters the lambda/mu imaging through recomputed
velocity derivatives, not stored differences).

These tests pin c-vs-eager rho agreement with vz/vx sources on heterogeneous
models (red before the fix), for full and boundary-saving backward paths, in
2D and 3D, elastic and DAS-mu.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

DT, DH, SO = 8e-4, 10.0, 4


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
    if len(shape) == 2:
        nz, nx = shape
        src = np.array([[nx // 2, 3]], np.int64)
        xs = np.arange(3, nx - 3, 3, dtype=np.int64)
        rec = np.stack([xs, np.full_like(xs, 1)], -1)[None]
    else:
        nz, ny, nx = shape
        src = np.array([[nx // 2, ny // 2, 3]], np.int64)
        xs = np.arange(3, nx - 3, 3, dtype=np.int64)
        rec = np.stack([xs, np.full_like(xs, ny // 2), np.full_like(xs, 1)], -1)[None]
    return src, rec


def _grads(cls, shape, st, impl, bs=False, rt=None):
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions

    dev, nt = "cuda", 300
    eq = cls(spatial_order=SO, device=dev, backend="torch")
    kw = dict(backend="torch", shape=shape,
              free_surface=["top"] if len(shape) == 2 else True,
              abcn=16, dh=DH, dt=DT, source_type=st,
              receiver_type=rt if rt is not None else st)
    if impl == "eager":
        p = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                      use_ckpt=False, dev=dev, nt=nt, B=1, **kw)
    else:
        cfg = {"enabled": True, "storage": "gpu"} if bs else {"enabled": False}
        p = PropTorch(eq, impl="c", use_ckpt=False, boundary_saving_config=cfg, **kw)
    t = np.arange(nt, dtype=np.float32) * DT - 0.12
    x = np.pi * 8.0 * t
    wav = torch.tensor((1e3 * (1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32),
                       device=dev)
    src, rec = _geom(shape)
    m = [torch.tensor(a, device="cuda", requires_grad=True) for a in _models(shape)]
    p(wav, src.copy(), rec.copy(), models=m).pow(2).sum().backward()
    return [t.grad.detach().double() for t in m]


def _cos(a, b):
    return float(torch.dot(a.ravel() / a.norm().clamp_min(1e-30),
                           b.ravel() / b.norm().clamp_min(1e-30)))


def _case(cls, shape, st, rt=None):
    ge = _grads(cls, shape, st, "eager", rt=rt)
    for bs in (False, True):
        gc = _grads(cls, shape, st, "c", bs=bs, rt=rt)
        for name, a, b in zip(("vp", "vs", "rho"), gc, ge):
            cos = _cos(a, b)
            rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
            assert cos > 0.999 and rel < 5e-3, \
                f"{cls.__name__} {st} bs={bs}: {name} cos={cos:.4f} rel={rel:.2e}"


def test_elastic2d_vz_source_rho_grad():
    from sweep.equations import Elastic
    _case(Elastic, (48, 56), ["vz"])


def test_elastic2d_vx_source_rho_grad():
    from sweep.equations import Elastic
    _case(Elastic, (48, 56), ["vx"])


def test_elastic3d_vz_source_rho_grad():
    from sweep.equations import Elastic3D
    _case(Elastic3D, (24, 20, 24), ["vz"])


def test_das_mu2d_vz_source_rho_grad():
    from sweep.equations import DASMu
    _case(DASMu, (48, 56), ["vz"])


def test_elastic2d_shear_and_mixed_sources_rho_grad():
    from sweep.equations import Elastic
    _case(Elastic, (48, 56), ["sxz"], rt=["vz"])
    _case(Elastic, (48, 56), ["vz", "sxx"], rt=["vz"])


def test_elastic2d_multi_source_encoded_rho_grad():
    """Four simultaneous vz sources (source-encoding mode).  The first-order
    injection term is corrected per source; a small cross-source second-order
    residual remains (~0.4% of the rho-gradient norm, cos 1.0000 — vs 85%
    before the fix), so the tolerance here is looser than the single-source
    tests but still ~5x tighter than the canonical suite's rho baseline."""
    from sweep.equations import Elastic

    ge = _grads_multi(Elastic, (48, 56), ["vz"], "eager")
    gc = _grads_multi(Elastic, (48, 56), ["vz"], "c")
    cos = _cos(gc[2], ge[2])
    rel = float((gc[2] - ge[2]).norm() / ge[2].norm().clamp_min(1e-30))
    assert cos > 0.9999 and rel < 2e-2, f"multi-source rho cos={cos:.4f} rel={rel:.2e}"


def _grads_multi(cls, shape, st, impl, nsrc=4):
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions

    dev, nt = "cuda", 300
    nz, nx = shape
    eq = cls(spatial_order=SO, device=dev, backend="torch")
    kw = dict(backend="torch", shape=shape, free_surface=["top"], abcn=16,
              dh=DH, dt=DT, source_type=st, receiver_type=st)
    if impl == "eager":
        p = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                      use_ckpt=False, dev=dev, nt=nt, B=1, **kw)
    else:
        p = PropTorch(eq, impl="c", use_ckpt=False,
                      boundary_saving_config={"enabled": False}, **kw)
    t = np.arange(nt, dtype=np.float32) * DT - 0.12
    x = np.pi * 8.0 * t
    w1 = (1e3 * (1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32)
    wav = torch.tensor(np.stack([w1 * (1 + 0.3 * k) for k in range(nsrc)]), device="cuda")
    src = np.array([[nx // 2 + 6 * k, 3] for k in range(nsrc)], np.int64)[None]
    xs = np.arange(3, nx - 3, 3, dtype=np.int64)
    rec = np.stack([xs, np.full_like(xs, 1)], -1)[None]
    m = [torch.tensor(a, device="cuda", requires_grad=True) for a in _models(shape)]
    p(wav, src.copy(), rec.copy(), models=m).pow(2).sum().backward()
    return [t.grad.detach().double() for t in m]

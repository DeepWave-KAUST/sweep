"""Adjoint defects in the compiled AcousticVTI1st backward (impl='c').

Before these fixes the compiled gradient disagreed with eager under **every**
legal source and receiver condition, including the equation's own defaults, in
every implemented memory mode.  A finite-difference check put eager right
(rel ~1e-4 of a central difference of the forward-only loss) and impl='c' 0.7%
to 20% off.  Three separate causes:

1. **The operator adjoint multiplied by the material coefficient on the wrong
   side of the derivative.**  The forward stress update is
   ``s += dt * c11 * Db_x(v_x)``, so as an operator on ``v_x`` its adjoint is
   ``-Df_x(c11 * lambda_s)`` — the stiffness multiplies the adjoint stress AT
   THE STRESS LOCATION and only then is differentiated.  The kernel computed
   ``-c11[idx] * Df_x(lambda_s)`` instead, which is the same thing only for a
   homogeneous model; for a heterogeneous one the two differ by a grad(c) term
   that accumulates once per step.  Both half-steps were affected (the
   stiffnesses in stress->velocity, inv_rho in velocity->stress), in 2-D and
   3-D.  This was the dominant error and it is what made the rho gradient look
   ~10x worse than vp: rho is a difference of two nearly cancelling terms, so
   it amplifies a small absolute error in the shared factor.

2. **The imaging ran before the stress->velocity half-step.**  The inv_rho
   gradient at step ``it`` needs the COMPLETE adjoint velocity, which only
   exists once that half-step has applied this step's ds(it)/dv(it) coupling.
   That half-step writes lambda_v and only reads lambda_s, so imaging between
   the two halves fixes the inv_rho operand and leaves the stiffness operand
   bit-identical.

3. **Chunk checkpointing substituted zero for the previous stress state at
   chunk boundaries** (an acknowledged TODO in the source).  The state at
   ``start-1`` is exactly the checkpoint the chunk was replayed from.

The equation's own docstring blamed the untracked adjoint CPML memory variables
and advised masking the PML band.  That was wrong: with source and receivers
80 cells from the absorbing layer and the run stopped before anything could
reach it, the gap was fully present, and it is unchanged by abcn.  After these
fixes the whole-grid agreement is cos = 1.0000000 to nt = 400, PML band
included.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

DT, DH, SO, NT, FREQ, DELAY = 1.5e-3, 10.0, 4, 120, 10.0, 0.06
ABCN2D, ABCN3D = 20, 12
SHAPE2D, SHAPE3D = (100, 110), (48, 44, 48)


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


def _ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)).astype(np.float32)


def _models(shape):
    """Heterogeneous on purpose: a homogeneous model hides defect 1 entirely."""
    ramp = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)
    ramp = np.broadcast_to(ramp.reshape((shape[0],) + (1,) * (len(shape) - 1)),
                           shape).copy()
    vp = (1800.0 + 600.0 * ramp).astype(np.float32)
    vp[shape[0] // 3: shape[0] // 3 + 3] += 180.0
    rho = (1000.0 + 300.0 * ramp).astype(np.float32)
    rho[shape[0] // 2: shape[0] // 2 + 2] += 150.0
    eps = (0.10 + 0.04 * ramp).astype(np.float32)
    dlt = (0.05 + 0.02 * ramp).astype(np.float32)
    return [vp, eps, dlt, rho]


def _geometry(shape, abcn):
    """Source and receivers inside the physical region (coords are x[,y],z)."""
    m = abcn + 6
    if len(shape) == 2:
        _, nx = shape
        src = np.array([[nx // 2, m + 6]], np.int64)
        rx = np.arange(m, nx - m, 6, dtype=np.int64)
        rec = np.stack([rx, np.full_like(rx, m)], -1)[None]
    else:
        _, ny, nx = shape
        src = np.array([[nx // 2, ny // 2, m + 6]], np.int64)
        rx = np.arange(m, nx - m, 6, dtype=np.int64)
        ry = np.arange(m, ny - m, 6, dtype=np.int64)
        gy, gx = np.meshgrid(ry, rx, indexing="ij")
        rec = np.stack([gx.reshape(-1), gy.reshape(-1),
                        np.full(gx.size, m, np.int64)], -1)[None]
    assert rec.shape[1] > 0
    return src, rec


def _solver(shape, abcn, mode, source_type, receiver_type):
    from sweep.propagator.options import (CkptOptions, CUDAOptions, EagerOptions,
                                          MemoryOptions)
    from sweep.propagator.torch import PropTorch
    if len(shape) == 2:
        from sweep.equations import AcousticVTI1st as EQ
    else:
        from sweep.equations import AcousticVTI1st3D as EQ

    dev = torch.device("cuda")
    eq = EQ(spatial_order=SO, device=dev, backend="torch")
    kw = dict(backend="torch", shape=shape, abcn=abcn, dh=DH, dt=DT, dev=dev,
              pml_type="cpmls", nt=NT, B=1, source_type=list(source_type),
              receiver_type=list(receiver_type), free_surface=False)
    if mode == "eager":
        return PropTorch(eq, impl="eager", use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False), **kw)
    if mode == "full":
        return PropTorch(eq, impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **kw)
    if mode == "bs":
        return PropTorch(eq, impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": True,
                                                 "storage": "gpu"}, **kw)
    if mode == "ckpt":
        return PropTorch(eq, impl="c", cuda_options=CUDAOptions(
            memory=MemoryOptions(strategy="ckpt",
                                 ckpt=CkptOptions(mode="chunk", chunks=24))), **kw)
    raise ValueError(mode)


def _grads(shape, abcn, mode, source_type, receiver_type):
    dev = torch.device("cuda")
    solver = _solver(shape, abcn, mode, source_type, receiver_type)
    wavelet = torch.tensor(_ricker(NT, DT, FREQ, DELAY), device=dev)
    src, rec = _geometry(shape, abcn)
    models = [torch.tensor(a, device=dev, requires_grad=True) for a in _models(shape)]
    solver(wavelet, src.copy(), rec.copy(), models=models).pow(2).mean().backward()
    return [m.grad.detach().cpu() for m in models]


def _metrics(cand, ref):
    x = cand.double().reshape(-1)
    y = ref.double().reshape(-1)
    xn, yn = float(x.norm()), float(y.norm())
    if yn <= 1e-300 or xn <= 1e-300:
        return float("nan"), float("nan")
    return float(torch.dot(x, y) / (xn * yn)), float((x - y).norm() / yn)


NAMES = ("vp", "epsilon", "delta", "rho")

# Before the fixes every one of these was off by 0.7%-20%; a 1e-3 bar is far
# below the defect and far above fp32 accumulation noise (measured <= 1e-5).
COS_MIN, REL_MAX = 0.99999, 1e-3


@pytest.mark.parametrize("mode", ["full", "bs", "ckpt"])
@pytest.mark.parametrize("receiver_type", [["vz"], ["vx"], ["sH"], ["sV"],
                                           ["vx", "vz"]])
@pytest.mark.parametrize("source_type", [["sH", "sV"], ["sH"], ["sV"]])
def test_vti2d_matches_eager(mode, source_type, receiver_type):
    ref = _grads(SHAPE2D, ABCN2D, "eager", source_type, receiver_type)
    got = _grads(SHAPE2D, ABCN2D, mode, source_type, receiver_type)
    for name, g, r in zip(NAMES, got, ref):
        cos, rel = _metrics(g, r)
        assert cos > COS_MIN and rel < REL_MAX, (
            f"{name}: cos={cos:.8f} rel={rel:.3e} "
            f"(mode={mode}, src={source_type}, rec={receiver_type})")


@pytest.mark.parametrize("mode", ["full", "bs", "ckpt"])
@pytest.mark.parametrize("receiver_type", [["vz"], ["vy"], ["sH"]])
def test_vti3d_matches_eager(mode, receiver_type):
    source_type = ["sH", "sV"]
    ref = _grads(SHAPE3D, ABCN3D, "eager", source_type, receiver_type)
    got = _grads(SHAPE3D, ABCN3D, mode, source_type, receiver_type)
    for name, g, r in zip(NAMES, got, ref):
        cos, rel = _metrics(g, r)
        assert cos > COS_MIN and rel < REL_MAX, (
            f"{name}: cos={cos:.8f} rel={rel:.3e} "
            f"(mode={mode}, rec={receiver_type})")


def test_heterogeneity_is_what_exposes_the_operator_adjoint():
    """Defect 1 is invisible on a homogeneous model — guard the guard.

    `c11[idx] * Df_x(lambda)` and `Df_x(c11 * lambda)` agree exactly when c11 is
    constant, so a suite that only ever ran homogeneous models would have called
    the old kernel correct.  Assert the test models really do vary.
    """
    vp, eps, dlt, rho = _models(SHAPE2D)
    for name, m in (("vp", vp), ("epsilon", eps), ("delta", dlt), ("rho", rho)):
        assert float(m.max() - m.min()) > 0.0, f"{name} is homogeneous"

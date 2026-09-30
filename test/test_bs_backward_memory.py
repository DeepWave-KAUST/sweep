"""Boundary saving must not pay for a forward history in the backward.

The point of boundary saving is that the backward rebuilds the forward state
from the saved strips instead of holding it: what it allocates may not grow
with nt like a stack of full grids.  2026-09-30: the backward bound a
per-call "transient replay" of ``cuda_layout.checkpoint_replay_shapes`` at
seg = nt whenever an equation's declaration ignored the mode (VRZ 2-D/3-D,
VTI first-order 2-D/3-D, ElasticTTI2nd, DASZhao) although no boundary-saving
driver reads it -- nb17 (VRZ Marmousi) asked for 68.6 GiB and died on a 32 GB
V100.  Acoustic, whose declaration is guarded, is the control.

Measured here: the backward's peak allocation above what the forward left
behind, at two nt; its growth must stay far below one padded grid per step.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# class, pml_type, receiver_type, source_type, ndim
CASES = [
    ("Acoustic", "cpmlr", None, None, 2),
    ("AcousticVRZ", "cpmlr", None, None, 2),
    ("AcousticVRZ3D", "cpmlr", None, None, 3),
    ("AcousticVTI1st", "cpmls", ["sH", "sV"], ["sV"], 2),
    ("AcousticVTI1st3D", "cpmls", ["sH", "sV"], ["sV"], 3),
    ("ElasticTTI2nd", "cpmls", ["ux", "uz"], ["uz"], 2),
    ("DASZhao", "cpmls", ["exx_t", "ezz_t", "das35_t"], None, 2),
]
SHAPE = {2: (48, 64), 3: (20, 16, 24)}
ABCN, DH, DT = 8, 10.0, 1e-3
NT_SHORT, NT_LONG = 150, 450
# model values (a mild linear gradient on each)
VALUES = {"vp": (2200, 60), "vs": (1200, 30), "rho": (2100, 20), "z": (4.6e6, 1e4),
          "vp0": (2200, 60), "vs0": (1200, 30), "vh": (2400, 60), "epsilon": (0.08, 0.04),
          "delta": (0.03, 0.02), "eta": (0.04, 0.02), "gamma": (0.05, 0.02),
          "theta": (0.25, 0.10), "phi": (0.15, 0.08)}


def _cuda_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


def _backward_peak(cls_name, pml, rtype, stype, nd, nt):
    import sweep.equations as E
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    eq = getattr(E, cls_name)(spatial_order=4, device="cuda", backend="torch")
    shape = SHAPE[nd]
    kw = dict(backend="torch", impl="c", shape=shape, abcn=ABCN, dh=DH, dt=DT, nt=nt,
              cuda_options=CUDAOptions(memory=MemoryOptions(
                  strategy="boundary", boundary=BoundaryOptions(storage="gpu", storage_dtype="fp32"))))
    if pml:
        kw["pml_type"] = pml
    if rtype:
        kw["receiver_type"] = list(rtype)
    if stype:
        kw["source_type"] = list(stype)
    prop = PropTorch(eq, **kw)
    assert prop.impl == "c", prop.impl
    g = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    models = [torch.tensor((VALUES[s.name][0] + VALUES[s.name][1] * g).astype(np.float32),
                           device="cuda", requires_grad=True) for s in eq.MODEL_SPECS]
    nx = shape[-1]
    xs = np.arange(2, nx - 2, 4, dtype=np.int64)
    if nd == 2:
        src = np.array([[nx // 2, shape[0] // 2]], np.int64)
        rec = np.stack([xs, np.full_like(xs, 6)], -1)[None]
    else:
        src = np.array([[nx // 2, shape[1] // 2, shape[0] // 2]], np.int64)
        rec = np.stack([xs, np.full_like(xs, shape[1] // 2), np.full_like(xs, 6)], -1)[None]
    t = np.arange(nt) * DT
    arg = (np.pi * 12.0 * (t - 0.08)) ** 2
    wav = torch.tensor(((1 - 2 * arg) * np.exp(-arg)).astype(np.float32), device="cuda")
    out = prop(wav, src, rec, models=models)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out.pow(2).sum().backward()
    torch.cuda.synchronize()
    assert all(torch.isfinite(m.grad).all() for m in models)
    padded_cells = int(np.prod(prop.shape_cuda))
    return torch.cuda.max_memory_allocated() - before, padded_cells


@pytest.mark.skipif(not _cuda_ready(), reason="CUDA + the compiled core required")
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_bs_backward_does_not_hold_a_forward_history(case):
    short, cells = _backward_peak(*case, NT_SHORT)
    long_, _ = _backward_peak(*case, NT_LONG)
    one_history = (NT_LONG - NT_SHORT) * cells * 4      # one padded fp32 grid per extra step
    grew = long_ - short
    print(f"{case[0]}: bs backward peak {short / 2**20:.2f} -> {long_ / 2**20:.2f} MiB "
          f"(+{grew / 2**20:.2f}; one grid per step would be +{one_history / 2**20:.2f})")
    assert grew < 0.1 * one_history, (
        f"{case[0]}: the boundary-saving backward's peak grew {grew / 2**20:.1f} MiB for "
        f"{NT_LONG - NT_SHORT} more steps -- {grew / one_history:.2f} padded grids per step; "
        f"it is holding a forward history it should rebuild")

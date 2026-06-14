"""Single-GPU, no-NCCL repro of the py>=3 boundary-saving FORWARD crash.

Faithfully rebuilds ONE crashing middle rank's tile (px=2, py=4, rank=3: one
x-face cut + BOTH y-faces cut -> thin y) and runs its *local* public forward
(== DDPropagator._capture's forward, which needs no NCCL). We drive the C
forward param's cut_face_mask directly:

  MASK=0     -> capture as it WAS (mask unset) -> expect CUDA invalid config
  MASK=auto  -> capture as FIXED (mask = prop._dd_cut_mask) -> expect OK

Prints the cut mask + the implied phys-y extent / compact boundary count so the
sign-flip is visible.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sweep.equations import Acoustic3D, Elastic3D
from sweep.parallel import MeshTopology
from sweep.propagator.torch import PropTorch

FAM = os.environ.get("FAM", "acoustic")

DEV = torch.device("cuda")
PX = int(os.environ.get("PX", "2"))
PY = int(os.environ.get("PY", "4"))
RANK = int(os.environ.get("RANK", "3"))   # a middle y-rank (both y-faces cut)
NZ, NY, NXG = 48, int(os.environ.get("NY", "20")), 28 * PX   # global (nxp=28 per x-tile)
NT, DT, SO, ABCN = 12, 0.0015, 4, 10
M = SO // 2
MASK = os.environ.get("MASK", "0")


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    topo = MeshTopology(py=PY, px=PX, shot_groups=1, world_size=PX * PY, rank=RANK)
    local_shape, offsets = topo.local_extent((NZ, NY, NXG))
    print(f"rank={RANK} local_shape={local_shape} offsets={offsets}")
    nz, nyp, nxp = local_shape

    if FAM == "elastic":
        eq = Elastic3D(spatial_order=SO, device=DEV, backend="torch")
        st, rt, pml = ["sxx", "syy", "szz"], ["vx", "vy", "vz"], "cpmls"
    else:
        eq = Acoustic3D(spatial_order=SO, device=DEV, backend="torch")
        st, rt, pml = ["h1"], ["h1"], "cpmlr"
    prop = PropTorch(eq, backend="torch", impl="c", shape=tuple(local_shape), dev=DEV,
                     dh=10.0, dt=DT, source_type=st, receiver_type=rt,
                     abcn=ABCN, free_surface=False, pml_type=pml, nt=NT, B=1,
                     use_ckpt=False, model_parallel=topo,
                     boundary_saving_config={"enabled": True, "storage": "gpu",
                                             "transfer_interval": 1, "pinned_memory": False})
    impl = prop._backend_impl
    cm = getattr(impl, "_dd_cut_mask", 0)
    pad = ABCN + M
    pady = prop.padding[2:4]
    ny_cuda = nyp + pady[0] + pady[1] + SO        # shape_cuda = shape + so
    print(f"_dd_cut_mask={cm} (y_lo={'cut' if cm&16 else 'PML'}, y_hi={'cut' if cm&32 else 'PML'})"
          f"  pad_y={pady} ny_cuda={ny_cuda}")
    for mk in (0, cm):
        p_y0 = M if (mk & 16) else pad
        p_y1 = ny_cuda - (M if (mk & 32) else pad)
        print(f"  cut_mask={mk:>2}: phys_y0={p_y0} phys_y1={p_y1} ny_phys={p_y1 - p_y0}")

    use_mask = cm if MASK == "auto" else int(MASK)
    do_bwd = os.environ.get("BWD", "0") == "1"
    orig_f, orig_b = impl.forward_func, impl.backward_bs_func

    def fwrap(p):
        p.cut_face_mask = use_mask     # mimics DDPropagator._capture.fwrap
        return orig_f(p)

    def bwrap(p):
        p.cut_face_mask = use_mask     # mimics DDPropagator._capture.bwrap
        return orig_b(p)

    impl.forward_func, impl.backward_bs_func = fwrap, bwrap

    wav = ricker(NT, DT)
    src = np.array([[[nxp // 2, nyp // 2, nz // 4]]], dtype=np.int32)
    rec = np.array([[[nxp // 2, nyp // 2, 2]]], dtype=np.int32)
    ramp = np.broadcast_to(np.linspace(0, 1, nz, dtype=np.float32)[:, None, None],
                           (nz, nyp, nxp)).copy()
    if FAM == "elastic":
        mods = [2200.0 + 400.0 * ramp, 1200.0 + 200.0 * ramp, 2000.0 + 100.0 * ramp]
    else:
        mods = [1800.0 + 600.0 * ramp]
    m = [torch.tensor(a, device=DEV, requires_grad=True) for a in mods]
    print(f"--- forward with cut_face_mask={use_mask} (MASK={MASK!r}) ---")
    try:
        syn = prop(wav, src, rec, models=m)
        torch.cuda.synchronize()
        r = syn[0] if isinstance(syn, (tuple, list)) else syn
        print(f"FORWARD OK  record {tuple(r.shape)} sum {float(r.detach().sum()):.3e}")
        if do_bwd:
            r.sum().backward()
            torch.cuda.synchronize()
            print(f"BACKWARD OK grad_vp sum {float(m[0].grad.sum()):.3e}")
    except Exception as e:
        print("FORWARD RAISED:", type(e).__name__, str(e)[:200])
        sys.exit(3)


if __name__ == "__main__":
    main()

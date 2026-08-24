"""Isolated cut-aware-pad memory A/B (single GPU).

For one physical tile, measure peak GPU memory of a bs-forward two ways:
  * SYMMETRIC : model_parallel=None  -> pad abcn+M on every side (legacy)
  * CUT-AWARE : model_parallel=mesh  -> pad M on cut faces (this branch)
Same physical tile size, so the only difference is the per-side pad. Peak is
per-process (torch.cuda.max_memory_allocated), so it is NOT contaminated by
other jobs sharing the GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D, Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DEV = "cuda:0"
ABCN = 20
SO = 4
NT = 30


def ricker(nt, dt, fm=10.0, delay=0.04, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return (scale * (1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def peak(eqc, shp, mp, elastic):
    nz, ny, nxp = shp
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    eq = eqc(spatial_order=SO, device=DEV, backend="torch")
    st = ["sxx", "syy", "szz"] if elastic else ["h1"]
    rt = ["vx", "vy", "vz"] if elastic else ["h1"]
    p = PropTorch(eq, backend="torch", impl="c", shape=(nz, ny, nxp), dev=DEV,
                  dh=10.0, dt=0.0005, source_type=st, receiver_type=rt,
                  abcn=ABCN, free_surface=False, nt=NT, B=1, use_ckpt=False,
                  boundary_saving_config={"enabled": True, "storage": "gpu",
                                          "transfer_interval": 1, "pinned_memory": False},
                  **({"model_parallel": mp} if mp else {}))
    if elastic:
        mm = [np.full((nz, ny, nxp), v, np.float32) for v in (3000., 1730., 2200.)]
        wav = ricker(NT, 0.0005) * 1e6
    else:
        mm = [np.full((nz, ny, nxp), 2500., np.float32)]
        wav = ricker(NT, 0.0005)
    m = [torch.tensor(a, device=DEV) for a in mm]
    src = np.array([[[nxp // 2, ny // 2, nz // 4]]], np.int32)
    rec = np.array([[[nxp // 2, ny // 2, 2]]], np.int32)
    with torch.no_grad():
        p(wav, src, rec, models=m)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 30, p.shape_cuda


def main():
    x3 = MeshTopology(py=1, px=3, shot_groups=1, world_size=3, rank=1)   # interior, both x cut
    q2 = MeshTopology(py=3, px=3, shot_groups=1, world_size=9, rank=4)   # interior, x+y cut (2x2)
    cases = [
        ("acoustic3d 160^3  x-cut", Acoustic3D, (160, 160, 160), x3, False),
        ("acoustic3d nxp64  x-cut", Acoustic3D, (160, 160, 64), x3, False),
        ("acoustic3d nxp64  2x2  ", Acoustic3D, (160, 64, 64), q2, False),
        ("elastic3d  nxp64  x-cut", Elastic3D, (96, 96, 64), x3, True),
        ("elastic3d  nxp64  2x2  ", Elastic3D, (96, 64, 64), q2, True),
    ]
    print(f"{'case':28s} {'sym GB':>8s} {'cut GB':>8s} {'reduction':>10s}  shape_cuda(cut)")
    for name, eqc, shp, mp, el in cases:
        ps, _ = peak(eqc, shp, None, el)
        pa, sc = peak(eqc, shp, mp, el)
        print(f"{name:28s} {ps:8.3f} {pa:8.3f} {100 * (1 - pa / ps):9.1f}%  {sc}")


if __name__ == "__main__":
    main()

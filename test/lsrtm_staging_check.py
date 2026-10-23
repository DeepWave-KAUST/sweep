#!/usr/bin/env python3
"""AcousticLSRTM / AcousticLSRTM3D boundary staging: storage='cpu' against
storage='gpu' (and against itself, run twice), on one card, through the plain
single-domain PropTorch and through a single-tile ModelParallel (the stepped DD
path without a cut), with the mp gradient alone (one boundary field) and with
the RWI vp gradient too (two boundary fields: background and scattered).

Staging only moves where the boundary lives, so every comparison must be
bitwise in fp32.

    PYTHONPATH=src python test/lsrtm_staging_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sweep.equations import AcousticLSRTM, AcousticLSRTM3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

dev = torch.device("cuda:0")
dh, dt, nt, abcn, order = 10.0, 0.001, 140, 12, 8


def setup(dim):
    shape = (40, 40) if dim == 2 else (40, 40, 40)
    nz = shape[0]
    idx = np.indices(shape).astype(np.float32)
    xs = idx[-1] / (shape[-1] - 1)
    vp = (2000.0 + 700.0 * idx[0] / (nz - 1)).astype(np.float32)
    mp = (0.08 * np.exp(-((idx[0] - (0.45 + 0.2 * xs) * nz) / 2.0) ** 2)
          * (1.0 + 0.3 * np.sin(6.0 * xs))).astype(np.float32)
    t = np.arange(nt) * dt
    a = np.pi * 12.0 * (t - 0.06)
    wav = ((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32)
    nx = shape[-1]
    if dim == 2:
        src = np.array([[[nx // 2, 1]]], np.int32)
        rx = np.arange(2, nx, 4)
        rec = np.stack([rx, np.ones(rx.size, np.int32)], -1)[None].astype(np.int32)
    else:
        ny = shape[1]
        src = np.array([[[nx // 2, ny // 2, 1]]], np.int32)
        gx, gy = np.meshgrid(np.arange(2, nx, 4), np.arange(2, ny, 4), indexing="xy")
        rec = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.int32)], -1)[None].astype(np.int32)
    return shape, vp, mp, wav, src, rec


def run(dim, path, storage, rwi):
    shape, vp, mp, wav, src, rec = setup(dim)
    eq = (AcousticLSRTM if dim == 2 else AcousticLSRTM3D)(spatial_order=order, device=dev)
    prop = PropTorch(eq, backend="torch", impl="c", shape=shape, dh=dh, dt=dt, nt=nt, abcn=abcn,
                     source_type=["h1"], receiver_type=["sh1"], dev=dev,
                     boundary_saving_config={"enabled": True, "storage": storage,
                                             "storage_dtype": "fp32"})
    if path == "dd1":
        prop = ModelParallel(prop, MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0))
    w = torch.tensor(wav, device=dev)
    ms = [torch.tensor(vp, device=dev, requires_grad=rwi),
          torch.tensor(0.5 * mp, device=dev, requires_grad=True)]
    (prop(w, src, rec, models=ms).double() ** 2).sum().backward()
    torch.cuda.synchronize()
    return torch.cat([m.grad.flatten() for m in ms if m.requires_grad])


def main():
    for dim in (2, 3):
        for path in ("single", "dd1"):
            for rwi in (False, True):
                g = run(dim, path, "gpu", rwi)
                c1 = run(dim, path, "cpu", rwi)
                c2 = run(dim, path, "cpu", rwi)
                scale = float(g.abs().max())
                d1 = float((c1 - g).abs().max()) / scale
                d2 = float((c2 - c1).abs().max()) / scale
                print(f"{dim}-D {path:6s} {'mp+vp' if rwi else 'mp   '}  cpu vs gpu: bitwise "
                      f"{bool(torch.equal(c1, g))!s:5s} max|d|/max|g| {d1:.1e}   cpu twice: bitwise "
                      f"{bool(torch.equal(c1, c2))!s:5s} {d2:.1e}", flush=True)


if __name__ == "__main__":
    main()

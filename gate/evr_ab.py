"""A/B bit-exactness harness for elastic_vr2d (ElasticVRR).

The solver_gradient_mode_suite has no ElasticVRR entry, so bitgate never
exercises the compiled elastic_vr2d bindings.  This standalone harness runs
impl='c' forward+backward in the four backward modes (full / boundary-saving /
chunk ckpt / recursive ckpt), each with free_surface off and on, and collects
the synthetic record plus all six model gradients.

Usage:
    python gate/evr_ab.py --out results.pt
    python gate/evr_ab.py --out results.pt --compare base.pt   # bitwise gate

Deterministic by construction: layered models, Ricker wavelet, fixed
geometry, zero target -- no RNG anywhere.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from sweep.equations import ElasticVRR, compute_vector_reflectivity
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda"
NZ, NX = 48, 56
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
SO = 4
FREQ = 10.0
DELAY = 0.06

GRAD_NAMES = ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z")

CASES = [
    ("full",           {"use_ckpt": False}, {}),
    ("bs",             {"use_ckpt": False}, {"use_boundary_saving": True}),
    ("ckpt_chunk",     {"use_ckpt": True, "ckpt_mode": "chunk", "ckpt_chunks": 16}, {}),
    ("ckpt_recursive", {"use_ckpt": True, "ckpt_mode": "recursive", "ckpt_num": 5}, {}),
]


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry():
    sources = np.array([[NX // 2, NZ // 4]], dtype=np.int64)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    receivers = np.stack([rec_x, np.full_like(rec_x, 2)], axis=-1)[None, ...]
    return sources, receivers


def _layered_models_with_grad():
    z = torch.arange(NZ, device=DEVICE, dtype=torch.float32).view(NZ, 1)
    vp = (1800.0 + 8.0 * z).expand(NZ, NX).contiguous()
    vs = vp / 1.73
    rho = (1000.0 + 5.0 * z).expand(NZ, NX).contiguous().clone()
    rho[NZ // 2 :, :] += 200.0
    Rp_x, Rp_z, Rs_x, Rs_z = compute_vector_reflectivity(vp, vs, rho, h=DH)
    models = [vp.clone(), vs.clone(),
              Rp_x.clone(), Rp_z.clone(), Rs_x.clone(), Rs_z.clone()]
    for m in models:
        m.requires_grad_(True)
    return models


def run_case(mode_kwargs, call_kwargs, free_surface):
    wavelet = _wavelet()
    sources, receivers = _geometry()
    models = _layered_models_with_grad()
    eq = ElasticVRR(spatial_order=SO, device=DEVICE, backend="torch")
    prop = PropTorch(
        eq, shape=(NZ, NX),
        abcn=ABCN, dh=DH, dt=DT,
        impl="c", free_surface=free_surface,
        **mode_kwargs,
    )
    syn = prop(wavelet, sources, receivers, models=models, **call_kwargs)
    loss = syn.pow(2).sum()
    loss.backward()
    out = {"syn": syn.detach().cpu()}
    for name, m in zip(GRAD_NAMES, models):
        out[f"grad_{name}"] = m.grad.detach().cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    results = {}
    for name, mode_kwargs, call_kwargs in CASES:
        for fs in (False, True):
            key = f"{name}|fs={int(fs)}"
            case = run_case(mode_kwargs, call_kwargs, fs)
            for tname, t in case.items():
                results[f"{key}|{tname}"] = t
            print(f"ran {key}", flush=True)

    torch.save(results, args.out)
    print(f"saved {len(results)} tensors -> {args.out}")

    if args.compare:
        base = torch.load(args.compare, weights_only=True)
        n_pass = n_fail = 0
        for k in sorted(base):
            if k not in results:
                print(f"MISSING {k}")
                n_fail += 1
                continue
            if torch.equal(base[k], results[k]):
                n_pass += 1
            else:
                d = (base[k] - results[k]).abs().max().item()
                r = ((base[k] - results[k]).norm() /
                     base[k].norm().clamp_min(1e-30)).item()
                print(f"FAIL {k}: max_abs_diff={d:.6e} rel_l2={r:.6e}")
                n_fail += 1
        print(f"BITWISE: PASS {n_pass}   FAIL {n_fail}")
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

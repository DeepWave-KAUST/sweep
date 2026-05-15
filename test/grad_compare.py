#!/usr/bin/env python3
"""Gradient consistency reusing solver_gradient_mode_suite.py's model + parameters.

Setup (matches test/solver_gradient_mode_suite.py defaults for acoustic2d):
  shape (nz, nx) = (48, 56)
  nt=120, dt=1.5e-3, dh=10.0, freq=10.0, delay=0.06
  spatial_order = 4, abcn=30, pml_type='cpmlr'
  vp_init = depth ramp 1800→2400 m/s
  vp_true = vp_init + 180 m/s box (the perturbation we invert for)
  Source @ (nx//2, nz//4) = (28, 12); receivers along z=2 at stride=6.
  Observed = eager forward on vp_true (no grad).
  Loss = MSE(record(vp_init) - observed) → ∂loss/∂vp.

Variants compared:
  - eager           (autograd, gold-standard reference)
  - c               (CUDA implementation in the active sweep install)
  - c, dev pristine (re-run via PYTHONPATH=<install> if --include-dev)

Per-config: rel_l2, cosine, diff_linf vs eager (per solver_gradient_mode_suite.metric_pair).
Plot:  test/grad_compare.png  — 1 row × N variants (single spatial order).
       Each subplot uses its own np.percentile(g, [2, 98]) scale.
"""
from __future__ import annotations
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Match solver_gradient_mode_suite.py exactly
NZ, NX = 48, 56
NT = 120
DT = 1.5e-3
DH = 10.0
FREQ = 10.0
DELAY = 0.06
SPATIAL_ORDER = 4
ABCN = 30
PML_TYPE = 'cpmlr'
RECEIVER_STRIDE = 6


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def depth_ramp(shape, top, bottom):
    nz = shape[0]
    depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    ramp = top + (bottom - top) * depth
    return np.broadcast_to(ramp[:, None], shape).astype(np.float32).copy()


def add_box(arr, value):
    out = arr.copy()
    nz, nx = out.shape
    out[nz // 3:max(nz // 3 + 2, (2 * nz) // 3),
        nx // 4:max(nx // 4 + 2, (3 * nx) // 4)] += value
    return out


def make_setup():
    shape = (NZ, NX)
    vp_init = depth_ramp(shape, 1800.0, 2400.0)
    vp_true = add_box(vp_init, 180.0)

    radius = SPATIAL_ORDER // 2
    source_z = max(1, min(NZ - 1, NZ // 4))
    receiver_z = max(1, radius)
    source_x = NX // 2
    margin = max(2, radius)
    sources = np.array([[source_x, source_z]], dtype=np.int32)
    rec_x = np.arange(margin, max(margin + 1, NX - margin), RECEIVER_STRIDE, dtype=np.int32)
    rec_z = np.full(rec_x.size, receiver_z, dtype=np.int32)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
    return shape, sources, receivers, vp_init, vp_true


def normalize_record(record, nshots, nrecv, nt):
    if tuple(record.shape) == (nshots, nrecv, nt):
        return record
    if tuple(record.shape) == (nshots, nt, nrecv):
        return record.transpose(1, 2)
    if record.ndim == 4 and record.shape[-1] == 1:
        sq = record[..., 0]
        if tuple(sq.shape) == (nshots, nrecv, nt):
            return sq
        if tuple(sq.shape) == (nshots, nt, nrecv):
            return sq.transpose(1, 2)
    raise RuntimeError(f"unexpected record shape {tuple(record.shape)}")


def make_prop(impl, *, free_surface, device):
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions
    from sweep.equations import Acoustic

    eq = Acoustic(spatial_order=SPATIAL_ORDER, device=device, backend='torch')
    common = dict(
        shape=(NZ, NX), dev=device, dh=DH, dt=DT, nt=NT,
        source_type=['h1'], receiver_type=['h1'],
        pml_type=PML_TYPE, abcn=ABCN, free_surface=free_surface,
        backend='torch', impl=impl,
    )
    if impl == 'eager':
        return PropTorch(eq, eager_options=EagerOptions(use_compile=False),
                          use_ckpt=False, **common)
    return PropTorch(eq, **common)


def run_forward(prop, wavelet, sources, receivers, vp_np, device):
    vp_t = torch.tensor(vp_np, device=device, dtype=torch.float32)
    with torch.no_grad():
        rec = prop(wavelet, sources.copy(), receivers.copy(), models=[vp_t])
        rec = normalize_record(rec, sources.shape[0], receivers.shape[1], NT)
        torch.cuda.synchronize(device)
    return rec.detach().cpu()


def run_gradient(prop, wavelet, sources, receivers, observed, vp_np, device):
    vp_t = torch.tensor(vp_np, device=device, dtype=torch.float32).requires_grad_(True)
    rec = prop(wavelet, sources.copy(), receivers.copy(), models=[vp_t])
    rec = normalize_record(rec, sources.shape[0], receivers.shape[1], NT)
    loss = (rec - observed.to(device)).pow(2).mean()
    loss.backward()
    torch.cuda.synchronize(device)
    return float(loss.detach().cpu().item()), rec.detach().cpu(), vp_t.grad.detach().cpu()


def metric_pair(ref, cand):
    r = ref.reshape(-1).to(torch.float64).numpy()
    c = cand.reshape(-1).to(torch.float64).numpy()
    diff = c - r
    ref_l2 = float(np.linalg.norm(r))
    cand_l2 = float(np.linalg.norm(c))
    diff_l2 = float(np.linalg.norm(diff))
    denom = max(ref_l2, 1e-30)
    cos = math.nan if ref_l2 <= 0 or cand_l2 <= 0 else float(np.dot(r, c) / (ref_l2 * cand_l2))
    return {
        'rel_l2': diff_l2 / denom,
        'cosine': cos,
        'diff_linf': float(np.max(np.abs(diff))),
        'ref_l2': ref_l2,
    }


def percentile_limit(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    vmax = max(abs(lo), abs(hi), 1e-30)
    return -vmax, vmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--free-surface', action='store_true', help='Use free_surface=True scenario.')
    p.add_argument('--include-dev', action='store_true',
                    help='Also subprocess into another sweep install (default /tmp/sweep_dev_install) for dev comparison.')
    p.add_argument('--dev-pythonpath', default='/tmp/sweep_dev_install')
    p.add_argument('--out-png', default='/tmp/regression/grad_compare.png')
    p.add_argument('--out-npz', default='/tmp/regression/grad_compare.npz')
    p.add_argument('--snapshot-out', default=None,
                    help='If set, only run eager + c-impl and save to this .npz, then exit. '
                         'Used internally by --include-dev to invoke a different sweep install.')
    args = p.parse_args()

    device = torch.device('cuda')
    shape, sources, receivers, vp_init, vp_true = make_setup()
    wavelet = torch.tensor(ricker(NT, DT, FREQ, DELAY), device=device)
    free_surface = args.free_surface

    import sweep
    print(f"sweep: {sweep.__file__}")

    if args.snapshot_out is not None:
        # Snapshot mode: minimal run for the dev subprocess.
        prop_eager = make_prop('eager', free_surface=free_surface, device=device)
        observed = run_forward(prop_eager, wavelet, sources, receivers, vp_true, device)
        prop_c = make_prop('c', free_surface=free_surface, device=device)
        _, _, grad = run_gradient(prop_c, wavelet, sources, receivers, observed, vp_init, device)
        np.savez_compressed(args.snapshot_out, grad=grad.numpy(),
                             observed=observed.numpy(),
                             sweep_file=str(sweep.__file__))
        print(f"snapshot saved {args.snapshot_out}")
        return

    # 1) eager reference
    prop_eager = make_prop('eager', free_surface=free_surface, device=device)
    observed = run_forward(prop_eager, wavelet, sources, receivers, vp_true, device)
    loss_e, _, grad_eager = run_gradient(prop_eager, wavelet, sources, receivers, observed, vp_init, device)

    # 2) c-impl (active sweep install)
    prop_c = make_prop('c', free_surface=free_surface, device=device)
    loss_c, _, grad_c = run_gradient(prop_c, wavelet, sources, receivers, observed, vp_init, device)

    grads = {'eager': grad_eager.numpy(), 'c': grad_c.numpy()}
    losses = {'eager': loss_e, 'c': loss_c}

    # 3) optional dev pristine via subprocess
    if args.include_dev:
        snap_path = '/tmp/regression/_dev_snapshot.npz'
        cmd = ['/home/wangs0j/miniconda3/envs/ifwitorch/bin/python', __file__,
                '--snapshot-out', snap_path]
        if free_surface:
            cmd.append('--free-surface')
        env = os.environ.copy()
        env['PYTHONPATH'] = args.dev_pythonpath
        print(f"[dev subprocess] PYTHONPATH={args.dev_pythonpath}")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(f"dev subprocess failed:\n{r.stderr}")
        else:
            snap = np.load(snap_path, allow_pickle=True)
            grads['c, dev pristine'] = snap['grad']
            losses['c, dev pristine'] = float('nan')

    # Metrics
    print()
    print("=" * 100)
    print(f"{'variant':<26} {'loss':<14} {'rel_l2':<12} {'cosine':<10} {'diff_linf':<12}")
    print("=" * 100)
    ref_t = torch.from_numpy(grads['eager'])
    for name, g in grads.items():
        m = metric_pair(ref_t, torch.from_numpy(g))
        loss_s = f"{losses[name]:.6e}" if not math.isnan(losses[name]) else "(n/a)"
        print(f"{name:<26} {loss_s:<14} {m['rel_l2']:<12.4e} "
              f"{m['cosine']:<10.6f} {m['diff_linf']:<12.4e}")
    print("=" * 100)

    if 'c, dev pristine' in grads:
        print("\nvs c-impl current sweep (modification-vs-baseline check):")
        ref_c = torch.from_numpy(grads['c'])
        for name, g in grads.items():
            if name in ('eager', 'c'):
                continue
            m = metric_pair(ref_c, torch.from_numpy(g))
            print(f"  {name:<26} rel_l2={m['rel_l2']:.4e} diff_linf={m['diff_linf']:.4e}")

    # Plot 1×N
    names = list(grads)
    fig, axes = plt.subplots(1, len(names),
                              figsize=(4.0 * len(names), 4.0), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0][j]
        g = grads[name]
        lo, hi = percentile_limit(g)
        im = ax.imshow(g, cmap='seismic', origin='upper', aspect='auto', vmin=lo, vmax=hi)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("ix"); ax.set_ylabel("iz")
        # Overlay source (cyan star) + receivers (open circles)
        ax.scatter(receivers[0, :, 0], receivers[0, :, 1], s=12,
                    marker='o', facecolors='none', edgecolors='black', linewidths=0.6)
        ax.scatter(sources[:, 0], sources[:, 1], s=80, marker='*',
                    c='cyan', edgecolors='black', linewidths=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title = ("vp gradient — solver_gradient_mode_suite setup "
              f"(order={SPATIAL_ORDER}, free_surface={free_surface}, "
              f"shape={NZ}×{NX}, nt={NT})")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_png, dpi=140)
    print(f"\nSaved {args.out_png}")
    np.savez_compressed(args.out_npz, **{k.replace(', ', '__').replace(' ', '_').replace('=','-'): v
                                          for k, v in grads.items()})
    print(f"Saved {args.out_npz}")


if __name__ == '__main__':
    main()

"""Visualise eager-vs-CUDA backward gradients for AcousticVTI1st (Phase 1).

Uses the canonical 2-D sizing from `sweep/test/solver_gradient_mode_suite.py`
so results line up with the gradient-consistency suite used for the other
equations (Acoustic / Elastic / DAS / ElasticTTISG ...).

  * nz × nx     = 48 × 56     (interior; abcn=30 added around)
  * dh          = 10.0 m
  * dt          = 0.0015 s
  * nt          = 120
  * Ricker      = f=10 Hz, delay=0.06 s
  * source      = (nx//2, nz//4)         — interior scenario
  * receivers   = horizontal line at z=2, x-stride 6
  * vp model    = depth ramp 1800→2400 + box +180 (true vs init)
  * rho model   = depth ramp 1000→1200 + box +60
  * eps, delta  = 0.05, 0.03 (near-isotropic so the test stays well-posed)

Produces two PNGs in outputs/:
  * vti_gradients_eager_vs_cuda.png — 4×3 grid of full-field gradient maps
  * vti_gradients_xsection.png      — 1D cross-section through receiver depth

Colour-map limits use a **signed-data percentile clip** (no abs).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ----------------------------------------------------------------------
# Canonical sizing (mirrors solver_gradient_mode_suite.py defaults)
# ----------------------------------------------------------------------
NZ, NX = 48, 56
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
FREQ = 10.0
DELAY = 0.06
SPATIAL_ORDER = 4
RADIUS = SPATIAL_ORDER // 2
REC_STRIDE = 6

device = "cuda"
shape = (NZ, NX)

src_x = NX // 2
src_z = max(1, min(NZ - 1, NZ // 4))
rec_z = max(1, RADIUS)
rec_x = np.arange(max(2, RADIUS),
                  max(max(2, RADIUS) + 1, NX - max(2, RADIUS)),
                  REC_STRIDE, dtype=np.int64)

src_pos = np.array([[src_x, src_z]], dtype=np.int64)[None]
rec_pos = np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=-1)[None]

t = np.arange(NT, dtype=np.float32) * DT - DELAY
# Ricker amplitude scaled to lift the receiver record out of the float32
# round-off floor; the canonical suite uses an unscaled ricker but its
# metric_pair runs in float64.  Our pytest also accumulates the metric in
# float64, so any scaling that keeps the wavelet well above ~1e-3 gives a
# stable signal.  1e6 matches the magnitude used by other example scripts.
WAVELET_AMPL = 1e6
wavelet = torch.tensor((WAVELET_AMPL * ricker(t, f=FREQ)).astype(np.float32))[None, None, :].to(device)


def _ramp(top, bottom):
    depth = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    return np.broadcast_to((top + (bottom - top) * depth)[:, None], shape).astype(np.float32).copy()


def _box(arr, val):
    out = arr.copy()
    out[NZ // 3: max(NZ // 3 + 2, (2 * NZ) // 3),
        NX // 4: max(NX // 4 + 2, (3 * NX) // 4)] += val
    return out


vp_init  = _ramp(1800.0, 2400.0)
vp_true  = _box(vp_init, 180.0)
rho_init = _ramp(1000.0, 1200.0)
rho_true = _box(rho_init, 60.0)
eps      = np.full(shape, 0.05, dtype=np.float32)
delta    = np.full(shape, 0.03, dtype=np.float32)


def _models(vp_arr, rho_arr, req_grad=False):
    return [
        torch.tensor(vp_arr,  dtype=torch.float32, device=device, requires_grad=req_grad),
        torch.tensor(eps,     dtype=torch.float32, device=device, requires_grad=req_grad),
        torch.tensor(delta,   dtype=torch.float32, device=device, requires_grad=req_grad),
        torch.tensor(rho_arr, dtype=torch.float32, device=device, requires_grad=req_grad),
    ]


def _target_record():
    eq = AcousticVTI1st(spatial_order=SPATIAL_ORDER, device=device, backend="torch")
    prop = PropTorch(eq, shape, source_type=["sH", "sV"], receiver_type=["vz"],
                     abcn=ABCN, dh=DH, dt=DT, nt=NT, device=device,
                     impl="eager", use_ckpt=False)
    with torch.no_grad():
        return prop(wavelet, src_pos, rec_pos, models=_models(vp_true, rho_true)).detach()


def _grads(impl, target):
    eq = AcousticVTI1st(spatial_order=SPATIAL_ORDER, device=device, backend="torch")
    prop = PropTorch(eq, shape, source_type=["sH", "sV"], receiver_type=["vz"],
                     abcn=ABCN, dh=DH, dt=DT, nt=NT, device=device,
                     impl=impl, use_ckpt=False)
    models = _models(vp_init, rho_init, req_grad=True)
    rec_syn = prop(wavelet, src_pos, rec_pos, models=models)
    loss = ((rec_syn - target) ** 2).mean()
    grads = torch.autograd.grad(loss, models)
    return [g.detach().cpu().numpy() for g in grads], float(loss.detach())


def main():
    print("Canonical 2-D gradient-consistency setup:")
    print(f"  shape (interior): {shape}, dh = {DH} m, abcn = {ABCN}")
    print(f"  source : (ix={src_x}, iz={src_z}) → ({src_x*DH:.0f}, {src_z*DH:.0f}) m")
    print(f"  rec    : {rec_x.size} at z={rec_z*DH:.0f} m")
    print(f"  NT={NT}, dt={DT*1e3:.1f} ms")
    print(f"  wavelet: ricker f={FREQ} Hz, delay={DELAY*1e3:.0f} ms")
    print(f"  models : vp ramp 1800→2400 (+180 box true), ρ ramp 1000→1200 (+60 box true),")
    print(f"           ε={eps[0,0]}, δ={delta[0,0]}\n")

    target = _target_record()
    print(f"target record range: [{float(target.min()):.3e}, {float(target.max()):.3e}]")

    g_eager, loss_e = _grads("eager", target)
    print(f"eager loss = {loss_e:.4e}")
    g_cuda,  loss_c = _grads("c",     target)
    print(f"cuda  loss = {loss_c:.4e}\n")

    names = ["vp", "epsilon", "delta", "rho"]
    units = ["m/s", "—", "—", "kg/m³"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: 4×3 grid of full-field gradient maps
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    ext = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]
    sx = src_x * DH
    sz = src_z * DH
    rec_xm = rec_x * DH
    rec_zm = rec_z * DH

    for i, (name, unit) in enumerate(zip(names, units)):
        ge = g_eager[i]
        gc = g_cuda[i]
        diff = ge - gc

        joint = np.concatenate([ge.ravel(), gc.ravel()])
        vmin_j, vmax_j = percentile_clip(joint, percentiles=(2, 98))
        vmin_d, vmax_d = percentile_clip(diff, percentiles=(2, 98))

        for j, (data, title, (vmin, vmax)) in enumerate([
            (ge,   f"eager   grad_{name}",  (vmin_j, vmax_j)),
            (gc,   f"CUDA    grad_{name}",  (vmin_j, vmax_j)),
            (diff, f"diff (eager − CUDA)",  (vmin_d, vmax_d)),
        ]):
            ax = axes[i, j]
            im = ax.imshow(data, cmap="seismic", aspect="auto",
                           extent=ext, vmin=vmin, vmax=vmax)
            ax.plot(sx, sz, "k*", markersize=12)
            ax.plot(rec_xm, [rec_zm] * len(rec_xm), "kv", markersize=4)
            ax.set_title(
                f"{title}\n(data range [{data.min():.2e}, {data.max():.2e}])",
                fontsize=9,
            )
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        "AcousticVTI1st backward gradients — eager autograd vs CUDA Phase 1\n"
        f"canonical 2-D test: nz={NZ}, nx={NX}, nt={NT}, dh={DH} m, abcn={ABCN}",
        fontsize=11,
    )
    fig.tight_layout()
    out_grid = OUTPUT_DIR / "vti_gradients_eager_vs_cuda.png"
    fig.savefig(out_grid, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_grid}")

    # ------------------------------------------------------------------
    # Figure 2: 1D cross-section through the receiver depth row
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    iz_xs = rec_z
    x_axis = np.arange(NX) * DH

    for i, (name, ax) in enumerate(zip(names, axes.flat)):
        ge_row = g_eager[i][iz_xs, :]
        gc_row = g_cuda[i][iz_xs, :]
        ax.plot(x_axis, ge_row, "k-",  lw=2,   label="eager (truth)")
        ax.plot(x_axis, gc_row, "r--", lw=1.2, label="CUDA Phase 1")
        ax.axvline(sx, color="green", linestyle=":", alpha=0.7, label="source x")
        ax.set_title(f"grad_{name}  at z = {iz_xs*DH:.0f} m  ({units[i]})",
                     fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel(f"∂L / ∂{name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Gradient cross-section at receiver depth (z = {iz_xs*DH:.0f} m)",
                 fontsize=11)
    fig.tight_layout()
    out_xs = OUTPUT_DIR / "vti_gradients_xsection.png"
    fig.savefig(out_xs, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_xs}")

    # ------------------------------------------------------------------
    # Numeric summary — full field, matching solver_gradient_mode_suite
    # (no margin stripping; the suite compares on the whole tensor).
    # ------------------------------------------------------------------
    print("\nFull-field gradient comparison (matches solver_gradient_mode_suite metrics):")
    print(f"  {'param':<8} {'eager range':<26} {'cuda range':<26} "
          f"{'rel L2':<10} {'cos sim':<10}")
    for i, name in enumerate(names):
        ge_flat = g_eager[i].astype(np.float64).ravel()
        gc_flat = g_cuda[i].astype(np.float64).ravel()
        ref_l2 = np.linalg.norm(ge_flat)
        cand_l2 = np.linalg.norm(gc_flat)
        diff_l2 = np.linalg.norm(ge_flat - gc_flat)
        rel_l2 = diff_l2 / max(ref_l2, 1e-30)
        cos_sim = (np.dot(ge_flat, gc_flat) / (ref_l2 * cand_l2 + 1e-30))
        e_rng = f"[{g_eager[i].min():.2e},{g_eager[i].max():.2e}]"
        c_rng = f"[{g_cuda[i].min():.2e},{g_cuda[i].max():.2e}]"
        print(f"  {name:<8} {e_rng:<26} {c_rng:<26} {rel_l2:<10.3e} {cos_sim:<10.4f}")


if __name__ == "__main__":
    main()

"""Visualize EVR gradients (eager autograd vs CUDA full-mode backward)
on a layered model so the patterns are clean and the eye can pick out
where the two backends agree or disagree.

Outputs a 6-rows x 3-cols figure (one row per model parameter, columns
``eager | c | c - eager``) with shared per-row percentile colour scaling,
plus a per-row title showing ``cosine`` and ``rel_l2`` so the metric maps
back to the spatial pattern.

Run:
    PYTHONPATH=src python test/evr_gradient_visualize.py
"""

from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from sweep.equations import (
    ElasticVRR,
    compute_vector_reflectivity,
)
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Bigger grid than the gradient-consistency test so the layered patterns
# are visible across the figure (Phase-3 test used 48x56).
NZ, NX = 80, 96
DH = 10.0
DT = 1.5e-3
NT = 240
ABCN = 30
SO = 4
FREQ = 12.0
DELAY = 0.06

OUT_DIR = os.environ.get(
    "EVR_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs", "evr_grad_layered"),
)
os.makedirs(OUT_DIR, exist_ok=True)


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry():
    # Centre-shot, line of receivers near the top
    sources = np.array([[NX // 2, NZ // 6]], dtype=np.int64)
    rec_x = np.arange(4, NX - 4, 4, dtype=np.int64)
    receivers = np.stack([rec_x, np.full_like(rec_x, 2)], axis=-1)[None, ...]
    return sources, receivers


def _layered_model():
    """Three-layer model with clean horizontal reflectors. Layer 1 (top
    third) is the slow layer, layer 2 (middle third) is intermediate,
    layer 3 (bottom) is fast. Density also varies per layer.
    """
    vp = torch.full((NZ, NX), 1800.0, device=DEVICE)
    vs = torch.full((NZ, NX), 1100.0, device=DEVICE)
    rho = torch.full((NZ, NX), 1000.0, device=DEVICE)

    # Layer 2 (middle): faster + denser
    z1, z2 = NZ // 3, 2 * NZ // 3
    vp[z1:z2, :] = 2200.0
    vs[z1:z2, :] = 1350.0
    rho[z1:z2, :] = 1200.0

    # Layer 3 (bottom): fastest
    vp[z2:, :] = 2700.0
    vs[z2:, :] = 1600.0
    rho[z2:, :] = 1400.0

    return vp, vs, rho


def _models_with_grad():
    vp, vs, rho = _layered_model()
    Rp_x, Rp_z, Rs_x, Rs_z = compute_vector_reflectivity(vp, vs, rho, h=DH)
    ms = [vp.clone(), vs.clone(),
          Rp_x.clone(), Rp_z.clone(), Rs_x.clone(), Rs_z.clone()]
    for m in ms:
        m.requires_grad_(True)
    return ms


def _make_prop(impl):
    eq = ElasticVRR(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(
        eq, shape=(NZ, NX),
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl=impl,
        eager_options={"use_compile": False} if impl == "eager" else None,
    )


def _compute(impl, wavelet, sources, receivers, target):
    models = _models_with_grad()
    prop = _make_prop(impl)
    syn = prop(wavelet, sources, receivers, models=models)
    loss = (syn - target).pow(2).sum()
    loss.backward()
    g = {n: m.grad.detach().cpu().numpy()
         for n, m in zip(("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z"), models)}
    return g, syn.detach()


def _signed_pct(a, b, pct=(2.0, 98.0)):
    """Joint percentile range on the signed (eager, c) concatenation.
    Returns (vmin, vmax). Per MEMORY.md: don't abs, don't force symmetry.
    """
    arr = np.concatenate([a.ravel(), b.ravel()])
    return float(np.percentile(arr, pct[0])), float(np.percentile(arr, pct[1]))


def _cos(a, b):
    af, bf = a.ravel(), b.ravel()
    n = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(af @ bf) / max(n, 1e-30)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-30)


def main():
    print(f"Device: {DEVICE}, grid: {NZ}x{NX}, nt: {NT}")
    wavelet = _wavelet()
    sources, receivers = _geometry()

    # Make a target shot from a perturbed model
    t_models = _models_with_grad()
    with torch.no_grad():
        t_models[0] += 60.0   # vp perturb
        t_models[1] += 30.0   # vs perturb
    target = _make_prop("eager")(wavelet, sources, receivers, models=t_models).detach()

    print("Computing eager grads...")
    eager_g, _ = _compute("eager", wavelet, sources, receivers, target)
    print("Computing CUDA grads...")
    cuda_g, _ = _compute("c", wavelet, sources, receivers, target)

    # ----- Figure layout -----
    names = ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z")
    rows = len(names)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 2.2 * rows),
                             constrained_layout=True)
    cmap = "RdBu_r"

    extent = [0, NX * DH, NZ * DH, 0]  # (left, right, bottom, top)

    for r, name in enumerate(names):
        eg = eager_g[name].squeeze()
        cg = cuda_g[name].squeeze()
        diff = cg - eg

        # Per-row percentile clipping for eager vs c columns (shared scale).
        vmin, vmax = _signed_pct(eg, cg, pct=(1.0, 99.0))
        # Difference column gets its own scale to highlight residuals.
        dvmin, dvmax = _signed_pct(diff, diff, pct=(1.0, 99.0))

        cos = _cos(cg, eg)
        rl2 = _rel_l2(cg, eg)

        im0 = axes[r, 0].imshow(eg, vmin=vmin, vmax=vmax, cmap=cmap,
                                extent=extent, aspect="auto")
        axes[r, 0].set_title(f"eager  {name}", fontsize=10)
        im1 = axes[r, 1].imshow(cg, vmin=vmin, vmax=vmax, cmap=cmap,
                                extent=extent, aspect="auto")
        axes[r, 1].set_title(
            f"c (full)  {name}   cos={cos:+.4f}  rel_l2={rl2:.3f}",
            fontsize=10,
        )
        im2 = axes[r, 2].imshow(diff, vmin=dvmin, vmax=dvmax, cmap=cmap,
                                extent=extent, aspect="auto")
        axes[r, 2].set_title(f"c - eager  {name}", fontsize=10)

        # Colorbars per row (compact)
        for ax, im in zip(axes[r, :], (im0, im1, im2)):
            ax.set_xlabel("x [m]", fontsize=8)
            ax.set_ylabel("z [m]", fontsize=8)
            ax.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"EVR gradients: eager autograd vs CUDA full-mode backward\n"
        f"layered model {NZ}x{NX}, nt={NT}, freq={FREQ}Hz",
        fontsize=11,
    )

    out_png = os.path.join(OUT_DIR, "evr_grad_eager_vs_c.png")
    fig.savefig(out_png, dpi=140)
    print(f"\nSaved: {out_png}")

    # Also save the model and the synthetic shot for context
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    vp_np = _layered_model()[0].cpu().numpy()
    vs_np = _layered_model()[1].cpu().numpy()
    rho_np = _layered_model()[2].cpu().numpy()
    for ax, m, label in zip(ax2, (vp_np, vs_np, rho_np),
                            ("vp [m/s]", "vs [m/s]", "rho [kg/m^3]")):
        im = ax.imshow(m, cmap="viridis", extent=extent, aspect="auto")
        ax.set_title(label)
        # mark source + receiver row
        sx_m = sources[0, 0] * DH
        sz_m = sources[0, 1] * DH
        ax.plot(sx_m, sz_m, "r*", markersize=10, label="source")
        rec_z_m = float(receivers[0, 0, 1]) * DH
        ax.axhline(rec_z_m, color="white", linestyle="--", linewidth=0.8,
                   label="receivers")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        fig2.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    ax2[0].legend(loc="lower right", fontsize=8)
    out_model = os.path.join(OUT_DIR, "evr_layered_model.png")
    fig2.savefig(out_model, dpi=140)
    print(f"Saved: {out_model}")

    # Print numeric summary
    print("\n  param      cos        rel_l2     eager_max     c_max")
    for name in names:
        eg = eager_g[name].squeeze()
        cg = cuda_g[name].squeeze()
        print(f"  {name:<7s} {_cos(cg, eg):+8.4f}   "
              f"{_rel_l2(cg, eg):8.4f}   {np.abs(eg).max():.4e}   "
              f"{np.abs(cg).max():.4e}")


if __name__ == "__main__":
    main()

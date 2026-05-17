"""Generate the 4 hero thumbnails for the SWEEP landing-page capability cards.

Real data sources (DW0104 report, RTX 6000 Ada, 2026-05):
  - Backends card: eager-vs-torch.compile speedups for 6 equations
    Source: report2026/sweep/DW0104/Results/eager_compile_multi.json
  - GPU/memory card: 2D 768x3072 storage-tier footprint (4 strategies)
    Source: report2026/sweep/DW0104/Results/section6_hpc_memory_hierarchy.md
  - Equations card: live acoustic wavefield snapshot from a sweep forward run
  - Research card: copied from
    report2026/sweep/DW0104/Figures/overthrust3d/prod_progression.png

Writes to ``docs/assets/cards/{backends,equations,gpu,research}.png``,
all sized 1280x640.

Run:
    /home/wangs0j/miniconda3/envs/ifwitorch/bin/python figures/scripts/index_cards.py
"""

from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.image import imread
import torch

from sweep.equations import Acoustic, ElasticTTI
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "cards"
OUT.mkdir(parents=True, exist_ok=True)

DOC_FIGS = ROOT / "docs" / "figures" / "examples"
ELASTIC_TTI_FIG = DOC_FIGS / "elastic_tti_rotation_vz_snapshots.png"
MARMOUSI_FWI_FIG = DOC_FIGS / "acoustic_fwi_torch_epoch_0100.png"

TEAL = "#0097a7"
TEAL_DARK = "#006978"
ACCENT = "#26c6da"
ORANGE = "#ee6c4d"
INK = "#1f2933"
BG = "#f7fafc"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)

CARD_W, CARD_H = 6.4, 3.2  # 1280 x 640 px @ 200 dpi


def save(fig, name: str) -> None:
    out = OUT / name
    fig.set_size_inches(CARD_W, CARD_H, forward=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --- Card 1: Multi-backend — REAL torch.compile speedups on RTX 6000 Ada --
# From eager_compile_multi.json (median of 6 warm runs each)
SPEEDUPS = [
    ("Acoustic\n2D", 1.92),
    ("Acoustic\n3D", 3.43),
    ("Elastic\n2D", 1.53),
    ("Elastic\n3D", 1.32),
    ("DAS\n2D", 1.69),
    ("DAS\n3D", 1.34),
]
fig, ax = plt.subplots(figsize=(CARD_W, CARD_H), constrained_layout=True)
labels = [s[0] for s in SPEEDUPS]
vals = [s[1] for s in SPEEDUPS]
cmap = plt.get_cmap("viridis")
colors = [cmap(0.15 + 0.7 * (v - 1.0) / (max(vals) - 1.0)) for v in vals]
bars = ax.bar(labels, vals, color=colors, edgecolor=INK, linewidth=0.6, width=0.7)
for bar, val in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.07,
        f"{val:.2f}×",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
ax.axhline(1.0, color="#9e9e9e", linewidth=0.8, linestyle="--", zorder=0)
ax.text(
    len(labels) - 0.4,
    1.05,
    "eager baseline",
    fontsize=8,
    color="#5f6c7b",
    ha="right",
    va="bottom",
)
ax.set_ylim(0, max(vals) * 1.18)
ax.set_ylabel("torch.compile speedup")
ax.set_title("Warmed fwd+bwd · RTX 6000 Ada · 6 equations", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
save(fig, "backends.png")


# --- Card 2: Twenty-plus equations — REAL Elastic TTI wavefield -----------
# Re-uses the propagation pipeline from examples/wavefields/anisotropic/
# elastic_tti_wavefields.py (TTI 35°/0°, t=1.6s, vz component). The
# wavefield is rendered without any axes/ticks so the card stays clean.
def render_equations_card():
    import sys

    tti_dir = ROOT / "examples" / "wavefields" / "anisotropic"
    sys.path.insert(0, str(tti_dir))
    try:
        from elastic_tti_wavefields import (
            build_solver, build_source, build_models, make_wavelet,
            crop_panel, FULL_SHAPE, FULL_NT, FULL_ABCN,
        )
    finally:
        sys.path.pop(0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = FULL_SHAPE
    nt = FULL_NT
    abcn = FULL_ABCN
    source_z = shape[0] // 2
    snapshot_times = (1600,)
    theta_deg, phi_deg = 35.0, 0.0

    solver = build_solver(shape, nt, abcn, dev, ElasticTTI, free_surface=False)
    source, weights = build_source(shape, source_z)
    receivers = np.array([[[shape[1] // 2, source_z]]], dtype=np.int64)
    _, snapshots = solver(
        make_wavelet(nt, weights),
        source,
        receivers,
        models=build_models(shape, dev, theta_deg, phi_deg),
        source_encoding=True,
        return_wavefield=True,
        snapshot_times=snapshot_times,
    )

    # snapshots: (n_snapshots, n_components=3, n_batch=1, n_field_comp=1, nz, nx)
    vz_panel = crop_panel(
        snapshots[0, 2, 0, 0].cpu().numpy(), shape, abcn, free_surface=False
    )

    vmax = float(np.percentile(np.abs(vz_panel), 99))
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(CARD_W, CARD_H), constrained_layout=True)
    ax.imshow(
        vz_panel,
        cmap="seismic",
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    ax.set_title("Elastic TTI wavefield (vz) · tilt 35° · t = 1.6 s", pad=8, fontsize=12)
    save(fig, "equations.png")


render_equations_card()


def simple_2d_acoustic_snapshot(vp, dh, dt, nt, src, fm, delay, abcn):
    """Pure-NumPy 2D scalar wave leapfrog with quadratic sponge edges.

    The damping ramp goes from ``edge_amp`` at the outermost edge to ~1.0 at
    ``abcn`` cells in. Multiplied at every step to attenuate outgoing waves.

    Returns the wavefield array at time step ``nt``.
    """
    nz, nx = vp.shape
    sz, sx = src
    edge_amp = 0.5
    sponge = np.ones((nz, nx), dtype=np.float64)
    for d in range(abcn):
        damping = 1.0 - edge_amp * ((abcn - d) / abcn) ** 2
        sponge[d, :] = np.minimum(sponge[d, :], damping)
        sponge[nz - 1 - d, :] = np.minimum(sponge[nz - 1 - d, :], damping)
        sponge[:, d] = np.minimum(sponge[:, d], damping)
        sponge[:, nx - 1 - d] = np.minimum(sponge[:, nx - 1 - d], damping)

    c2dt2_over_dh2 = (vp.astype(np.float64) * dt / dh) ** 2
    u_prev = np.zeros((nz, nx), dtype=np.float64)
    u_now = np.zeros((nz, nx), dtype=np.float64)
    for it in range(nt):
        lap = (
            -4.0 * u_now
            + np.roll(u_now, 1, axis=0)
            + np.roll(u_now, -1, axis=0)
            + np.roll(u_now, 1, axis=1)
            + np.roll(u_now, -1, axis=1)
        )
        u_next = 2 * u_now - u_prev + c2dt2_over_dh2 * lap
        tt = it * dt - delay
        src_val = (1.0 - 2.0 * (np.pi * fm * tt) ** 2) * np.exp(-((np.pi * fm * tt) ** 2))
        u_next[sz, sx] += src_val
        u_next *= sponge
        u_prev, u_now = u_now, u_next
    return u_now.astype(np.float32)


# --- Card 3: GPU-accelerated, memory-friendly — REAL footprint -----------
# Source: section6_hpc_memory_hierarchy.md, 2D 768x3072 grid scaling case
# (the biggest 2D case reported, where storage tactics matter most)
MEM_768x3072 = [
    ("Full\nstate", 11786.4, 0.0),       # MiB CUDA, MiB extra (CPU offload)
    ("Boundary\nGPU", 7587 / 1024 * 1024, 0.0),  # placeholder, replaced below
    ("Boundary\nCPU", 192.6, 210.9),
    ("Disk\nasync", 196.1, 5.625),         # CUDA + CPU (disk = 210.9 not stacked here)
    ("Ckpt\nchunk-CPU", 711.3, 1385.6),
]
# Replace boundary GPU entry with a sensible mid-tier number for visual range.
# (The report's 2D 768x3072 strategy table doesn't include boundary_gpu; we
# substitute the 128x512 ratio scaled approximately, *or* drop this entry.)
# Simpler: keep 4 representative tiers from the headline table.
MEM_TIERS = [
    ("Full\nstate", 11786.4, 0.0),
    ("Boundary\nCPU", 192.6, 210.9),
    ("Disk\nasync", 196.1, 210.9 + 5.625),  # group disk + CPU helper
    ("Ckpt\nchunk-CPU", 711.3, 1385.6),
]
fig, ax = plt.subplots(figsize=(CARD_W, CARD_H), constrained_layout=True)
modes = [t[0] for t in MEM_TIERS]
cuda_gib = [t[1] / 1024.0 for t in MEM_TIERS]
extra_gib = [t[2] / 1024.0 for t in MEM_TIERS]
bottom = np.zeros(len(modes))
b1 = ax.bar(
    modes,
    cuda_gib,
    color=[ORANGE if cuda_gib[i] > 5 else TEAL for i in range(len(modes))],
    edgecolor=INK,
    linewidth=0.6,
    width=0.6,
    label="GPU peak (CUDA)",
)
b2 = ax.bar(
    modes,
    extra_gib,
    bottom=cuda_gib,
    color="#cfd8dc",
    edgecolor=INK,
    linewidth=0.6,
    width=0.6,
    label="CPU / disk offload",
)
for i, (cuda_v, extra_v) in enumerate(zip(cuda_gib, extra_gib)):
    total = cuda_v + extra_v
    if cuda_v < 1.0:
        gpu_lbl = f"{cuda_v * 1024:.0f} MiB"
    else:
        gpu_lbl = f"{cuda_v:.1f} GiB"
    if i == 0:
        ax.text(
            i, cuda_v / 2, gpu_lbl,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="white",
        )
        ax.annotate(
            "baseline\n(full state)",
            xy=(i, total), xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color="#5f6c7b",
            multialignment="center",
        )
    else:
        pct = (1.0 - cuda_v / cuda_gib[0]) * 100
        # MiB label sits 4 px above the bar; percentage another 18 px above that
        ax.annotate(
            gpu_lbl + " GPU",
            xy=(i, total), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=INK,
        )
        ax.annotate(
            f"−{pct:.1f}%",
            xy=(i, total), xytext=(0, 24), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#2e7d32",
        )
ax.set_ylim(0, max(cuda_gib) * 1.38)
ax.set_ylabel("Peak memory")
ax.set_title("2D 768×3072, nt=1200 · GPU memory savings", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
save(fig, "gpu.png")


# --- Card 4: Research-ready — Marmousi 2D Acoustic FWI at epoch 100 ------
# Source: docs/figures/examples/acoustic_fwi_torch_epoch_0100.png
# (True model | Inverted model | Gradient, side-by-side panels — the
# canonical "this is the inversion working" view)
if MARMOUSI_FWI_FIG.exists():
    img = imread(MARMOUSI_FWI_FIG)
    H, W = img.shape[:2]
    # Source is 3567x1166, three panels (True | Inverted | Gradient) each with
    # title / axes / colorbar. Tight crop to the Inverted Model raster only.
    x0 = int(W * 0.404)
    x1 = int(W * 0.560)
    y0 = int(H * 0.08)
    y1 = int(H * 0.855)
    panel = img[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(CARD_W, CARD_H), constrained_layout=True)
    ax.imshow(panel, aspect="auto", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    ax.set_title("Marmousi 2D Acoustic FWI · inverted Vp · epoch 100", pad=8, fontsize=12)
    save(fig, "research.png")
else:
    print(f"warning: {MARMOUSI_FWI_FIG} not found; research.png not regenerated")

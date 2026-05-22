"""Acoustic 2-D vp gradient under an irregular free-surface topography.

Builds the same undulating-surface model as ``acoustic2d_hill_demo.py``, plants
a buried velocity anomaly in the "true" model, and back-propagates an
FWI-style L2 mis-fit through the eager topography path. Outputs:

    outputs/acoustic2d_hill_gradient.png      anomaly (top) + gradient (bottom)

Run:

    python examples/wavefields/topography/acoustic2d_hill_gradient_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# Same geometry as the forward demo so the GIF and gradient line up.
NZ, NX = 160, 320
DH = 10.0
DT = 1.0e-3
NT = 900
ABCN = 40
SO = 4
FREQ = 8.0
DELAY = 0.12

# Multi-shot illumination so the gradient banana isn't too one-sided.
N_SHOTS = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32)
    peak_a = 22.0 * np.exp(-((x - nx * 0.30) ** 2) / (2.0 * 22.0**2))
    peak_b = 16.0 * np.exp(-((x - nx * 0.68) ** 2) / (2.0 * 30.0**2))
    ripple = 3.0 * np.sin(2.0 * np.pi * x / 60.0)
    elevation = peak_a + peak_b + ripple
    surface_row = np.round(28.0 - elevation).astype(np.int64)
    return np.clip(surface_row, 0, nx)


def build_background_vp(nz: int, nx: int, topo: np.ndarray) -> np.ndarray:
    vp = np.full((nz, nx), 2800.0, dtype=np.float32)
    near_surface_thickness = 6
    middle_thickness = 35
    for ix in range(nx):
        s = topo[ix]
        vp[s : s + near_surface_thickness, ix] = 1800.0
        vp[s + near_surface_thickness : s + near_surface_thickness + middle_thickness, ix] = 2300.0
    return vp


def add_anomaly(vp_bg: np.ndarray) -> np.ndarray:
    """Insert a +400 m/s box anomaly mid-depth (well below the topography)."""
    vp = vp_bg.copy()
    vp[NZ // 2 - 6 : NZ // 2 + 6, NX // 3 : (2 * NX) // 3] += 400.0
    return vp


def build_wavelet() -> torch.Tensor:
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))


def build_geometry(topo: np.ndarray):
    """Multi-shot acquisition: N_SHOTS surface sources spread across x, all
    sharing the same surface receiver line."""
    src_xs = np.linspace(NX * 0.18, NX * 0.82, N_SHOTS).astype(np.int64)
    src_zs = (topo[src_xs] + 2).astype(np.int64)
    sources = np.stack([src_xs, src_zs], axis=-1)  # (N_SHOTS, 2)
    rec_x = np.arange(6, NX - 6, 3, dtype=np.int64)
    rec_z = (topo[rec_x] + 1).astype(np.int64)
    # Per-shot receivers (all shots share the same receiver layout).
    receivers = np.broadcast_to(
        np.stack([rec_x, rec_z], axis=-1)[None, ...],
        (N_SHOTS, rec_x.size, 2),
    ).copy()
    return (
        torch.from_numpy(sources),
        torch.from_numpy(receivers),
        src_xs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[topo-grad] device = {device}")

    topo = build_topography(NX)
    vp_bg = build_background_vp(NZ, NX, topo)
    vp_true_np = add_anomaly(vp_bg)
    vp_init_np = vp_bg

    vp_true = torch.from_numpy(vp_true_np).to(device)
    vp_param = torch.from_numpy(vp_init_np).clone().to(device).requires_grad_(True)

    wavelet = build_wavelet().to(device)
    sources, receivers, src_xs = build_geometry(topo)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = Acoustic(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation,
        shape=(NZ, NX),
        free_surface=True,
        topography=topo,
        abcn=ABCN,
        dh=DH,
        dt=DT,
        use_ckpt=False,
        impl="eager",
    )

    # Wavelet broadcast across shots so each source gets the same pulse.
    # Eager backend infers per-shot batching from sources.shape[0].
    print(f"[topo-grad] forward (obs) — {N_SHOTS} shots")
    with torch.no_grad():
        obs = prop(wavelet, sources, receivers, models=[vp_true])
    print(f"[topo-grad] forward (syn) + backward")
    syn = prop(wavelet, sources, receivers, models=[vp_param])
    loss = 0.5 * (syn - obs).pow(2).sum()
    loss.backward()

    grad = vp_param.grad.detach().cpu().numpy()
    print(f"[topo-grad] loss = {loss.item():.3e}; "
          f"grad |max|={np.abs(grad).max():.3e}; "
          f"finite={np.isfinite(grad).all()}")

    # ---- plot ----
    anomaly = vp_true_np - vp_init_np
    # Air mask for cosmetic overlay.
    z_idx = np.arange(NZ)[:, None]
    air_mask = z_idx < topo[None, :]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    topo_km = topo * DH * 1e-3
    x_km = np.arange(NX) * DH * 1e-3

    # Panel 1: anomaly (vp_true − vp_init)
    a = anomaly.copy()
    a[air_mask] = np.nan
    a_lim = np.nanmax(np.abs(a))
    norm_a = TwoSlopeNorm(vmin=-a_lim, vcenter=0.0, vmax=a_lim)
    im_a = axes[0].imshow(a, cmap="seismic", norm=norm_a, extent=extent, aspect="auto")
    axes[0].plot(x_km, topo_km, "k-", lw=1.2)
    axes[0].fill_between(x_km, 0, topo_km, color="white")
    axes[0].plot(src_xs * DH * 1e-3, topo[src_xs] * DH * 1e-3, "r*", ms=10, label="sources")
    axes[0].set_title("Target perturbation  $v_p^{\\rm true} - v_p^{\\rm init}$  (m/s)")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("z (km)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.colorbar(im_a, ax=axes[0], shrink=0.85, label="m/s")

    # Panel 2: gradient ∂L/∂vp (signed; use percentile of the signed array)
    g = grad.copy()
    g[air_mask] = np.nan
    g_lo, g_hi = np.nanpercentile(g, [2.0, 98.0])
    g_lim = max(abs(g_lo), abs(g_hi))
    norm_g = TwoSlopeNorm(vmin=-g_lim, vcenter=0.0, vmax=g_lim)
    im_g = axes[1].imshow(g, cmap="seismic", norm=norm_g, extent=extent, aspect="auto")
    axes[1].plot(x_km, topo_km, "k-", lw=1.2)
    axes[1].fill_between(x_km, 0, topo_km, color="white")
    axes[1].plot(src_xs * DH * 1e-3, topo[src_xs] * DH * 1e-3, "r*", ms=10)
    axes[1].set_title(r"FWI gradient  $\partial L / \partial v_p$  (signed)")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("z (km)")
    fig.colorbar(im_g, ax=axes[1], shrink=0.85, label=r"$\partial L/\partial v_p$")

    out_path = OUTPUT_DIR / "acoustic2d_hill_gradient.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[topo-grad] wrote {out_path}")


if __name__ == "__main__":
    main()

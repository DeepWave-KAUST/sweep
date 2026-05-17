"""Regenerate the SWEEP hero figure: model, observed gather, and Vp gradient.

This is a publication-styled version of the README's "30-line example" — same
2-layer toy problem the project has shipped with, with proper axes in physical
units, balanced colormaps, and a clean three-panel layout. Run with the
``ifwitorch`` env active:

    /home/wangs0j/miniconda3/envs/ifwitorch/bin/python figures/scripts/hero_grad_vp.py

Writes ``figures/grad_vp.png``.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


def main() -> None:
    nt = 1500
    dt = 0.002
    dh = 10.0
    delay = 0.1
    fm = 5.0
    spatial_order = 8
    shape = (100, 100)
    abcn = 20

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vp_np = np.full(shape, 1500.0, dtype=np.float32)
    vp_np[50:, :] = 2000.0

    equation = Acoustic(spatial_order=spatial_order, device=dev, backend="torch")
    solver = PropTorch(
        equation,
        shape=shape,
        dev=dev,
        dh=dh,
        dt=dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        pml_type="cpmlr",
        free_surface=False,
        backend="torch",
        impl="eager",
    )

    t = np.arange(nt, dtype=np.float32) * dt
    wave = ricker(t - delay, f=fm).astype(np.float32)

    sources = np.array([[1, 1]], dtype=np.int32)
    receivers = np.array([[[99, 1]]], dtype=np.int32)

    vp = torch.from_numpy(vp_np).to(dev).requires_grad_(True)
    obs = solver(wave, sources, receivers, models=[vp])
    obs.pow(2).sum().backward()

    obs_arr = obs.detach().cpu().numpy().squeeze()
    grad_arr = vp.grad.detach().cpu().numpy()

    nz, nx = shape
    extent_km = (0.0, nx * dh / 1000.0, nz * dh / 1000.0, 0.0)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), constrained_layout=True)

    im0 = axes[0].imshow(vp_np, extent=extent_km, cmap="viridis", aspect="auto")
    axes[0].plot(
        sources[0, 1] * dh / 1000.0,
        sources[0, 0] * dh / 1000.0,
        marker="*",
        color="white",
        markersize=14,
        markeredgecolor="black",
        linestyle="none",
        label="source",
    )
    axes[0].plot(
        receivers[0, :, 1] * dh / 1000.0,
        receivers[0, :, 0] * dh / 1000.0,
        marker="v",
        color="white",
        markersize=9,
        markeredgecolor="black",
        linestyle="none",
        label="receiver",
    )
    axes[0].set_title("True velocity model")
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.02)
    cb0.set_label("Vp (m/s)")

    obs_1d = obs_arr.reshape(-1)
    time_axis = np.arange(obs_1d.size) * dt
    axes[1].plot(time_axis, obs_1d, color="#cf3030", linewidth=1.2)
    axes[1].set_title("Observed receiver trace")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    g_abs = float(np.percentile(np.abs(grad_arr), 99))
    if g_abs == 0.0:
        g_abs = 1.0
    norm = TwoSlopeNorm(vmin=-g_abs, vcenter=0.0, vmax=g_abs)
    im2 = axes[2].imshow(
        grad_arr, extent=extent_km, cmap="RdBu_r", aspect="auto", norm=norm
    )
    axes[2].set_title(r"Gradient $\partial L / \partial v_p$")
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Depth (km)")
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.9, pad=0.02)
    cb2.set_label("dL / dVp")

    out = Path(__file__).resolve().parent.parent / "grad_vp.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

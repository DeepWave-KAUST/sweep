"""Render a 3D cutaway view of the Overthrust velocity model as a card cover.

Loads ``examples/models/overthrust/true_3d.npy`` and produces a 1280x640 PNG
with three orthogonal velocity slices arranged as a 3D "L"-block cutaway,
revealing the model's internal structure. No axes, no colorbar.

Run inside the ``ifwitorch`` env:

    /home/wangs0j/miniconda3/envs/ifwitorch/bin/python figures/scripts/overthrust_3d_render.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


ROOT = Path(__file__).resolve().parents[2]
MODEL_NPY = ROOT / "examples" / "models" / "overthrust" / "true_3d.npy"
OUT = ROOT / "docs" / "assets" / "cards" / "overthrust_3d.png"
CARD_W, CARD_H = 9.6, 3.2  # ≈ 3:1 to match other gallery covers
DPI = 200
BG = "#f7fafc"


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample so the 3D card render stays light."""
    sl = tuple(slice(0, (s // factor) * factor) for s in arr.shape)
    cropped = arr[sl]
    new_shape = []
    for s in cropped.shape:
        new_shape.extend([s // factor, factor])
    return cropped.reshape(new_shape).mean(axis=tuple(range(1, len(new_shape), 2)))


def main() -> None:
    vp = np.load(MODEL_NPY).astype(np.float32)
    print(f"loaded {MODEL_NPY.name}, shape={vp.shape}")

    # Convention: model is stored (nx, ny, nz). Convert to (nz, ny, nx) for an
    # imshow-friendly orientation if shape suggests depth is the last axis.
    if vp.shape[-1] < vp.shape[0] // 2 and vp.shape[-1] < vp.shape[1] // 2:
        nz, ny, nx = vp.shape[2], vp.shape[1], vp.shape[0]
        # transpose to (nz, ny, nx)
        vp = vp.transpose(2, 1, 0)
    nz, ny, nx = vp.shape
    print(f"oriented as (nz={nz}, ny={ny}, nx={nx})")

    # Downsample so each face has ~ few hundred pixels — fast to render.
    vp_ds = downsample(vp, 3)
    nz_d, ny_d, nx_d = vp_ds.shape
    print(f"downsampled to (nz={nz_d}, ny={ny_d}, nx={nx_d})")

    vmin = float(np.percentile(vp_ds, 2))
    vmax = float(np.percentile(vp_ds, 98))
    cmap = plt.colormaps["viridis"]

    def to_rgb(slice2d: np.ndarray) -> np.ndarray:
        clipped = np.clip((slice2d - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        return cmap(clipped)[..., :3]

    # Three exposed faces (after cutting one corner away):
    # - top face   : depth=0, full xy
    # - front face : y=0,     full xz
    # - side face  : x=nx-1,  full yz
    # We render each as a 3D plane in matplotlib via plot_surface.
    fig = plt.figure(figsize=(CARD_W, CARD_H), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    # World coordinates so the aspect feels right (Overthrust is roughly
    # 20 x 20 x 4.65 km).
    Lx = 20.0
    Ly = 20.0
    Lz = nz_d / ny_d * Lx  # keep depth proportional

    # --- Top face (z = 0): the surface velocity map
    xs = np.linspace(0, Lx, nx_d)
    ys = np.linspace(0, Ly, ny_d)
    X, Y = np.meshgrid(xs, ys)
    Z_top = np.zeros_like(X)
    top_face = to_rgb(vp_ds[0, :, :])
    ax.plot_surface(
        X, Y, Z_top,
        facecolors=top_face, rstride=1, cstride=1, shade=False,
        linewidth=0, antialiased=False,
    )

    # --- Front face (y = 0): xz vertical section
    zs = np.linspace(0, Lz, nz_d)
    X2, Z2 = np.meshgrid(xs, zs)
    Y_front = np.zeros_like(X2)
    front_face = to_rgb(vp_ds[:, 0, :])
    ax.plot_surface(
        X2, Y_front, Z2,
        facecolors=front_face, rstride=1, cstride=1, shade=False,
        linewidth=0, antialiased=False,
    )

    # --- Side face (x = Lx): yz vertical section
    Y3, Z3 = np.meshgrid(ys, zs)
    X_side = np.full_like(Y3, Lx)
    side_face = to_rgb(vp_ds[:, :, -1])
    ax.plot_surface(
        X_side, Y3, Z3,
        facecolors=side_face, rstride=1, cstride=1, shade=False,
        linewidth=0, antialiased=False,
    )

    # Cosmetics: clean off-axis 3D look. Boost the depth so the cutaway has
    # presence; geophysically Overthrust is shallow (~4.65 km vs 20 km wide)
    # but for the card we exaggerate vertical scale 3x.
    ax.set_box_aspect((Lx, Ly, Lz * 3.0))
    ax.view_init(elev=22, azim=-55)
    ax.invert_zaxis()  # depth grows downwards
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(Lz, 0)

    # Make the 3D axes fill the canvas (no matplotlib outer padding).
    ax.set_position([0.0, 0.0, 1.0, 0.92])

    # Title bar
    fig.text(
        0.5, 0.96,
        "SEG/EAGE Overthrust · 3D acoustic velocity model",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#1f2933",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

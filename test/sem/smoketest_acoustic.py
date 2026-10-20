"""Visual + numerical smoketest for the SEM acoustic 2-D PoC.

Runs a Ricker source in a uniform vp medium, with Stacey absorbing on
left/right/bottom and free top. Saves:
  - snapshot grid (PNG) showing the wave at several times
  - one trace at a receiver to inspect the time series

Run from the worktree root:
    PYTHONPATH=src python test/sem/smoketest_acoustic.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import AcousticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t: torch.Tensor, f0: float, delay: float) -> torch.Tensor:
    a = (torch.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def percentile_clip(arr: np.ndarray, lo=2.0, hi=98.0) -> tuple[float, float]:
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    device = "cpu"

    # --- mesh & model --------------------------------------------------
    x_max, z_max = 1000.0, 750.0
    nex, nez = 20, 15                # dh = 50 m
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=4, dtype=dtype, device=device)
    print(f"mesh: E={mesh.n_elements}, ngll={mesh.ngll}, "
          f"n_global={mesh.n_global}, dh={mesh.dx_e}")

    vp_const = 2000.0
    vp_e = vp_const * torch.ones(mesh.n_elements, mesh.ngll, mesh.ngll,
                                 dtype=dtype, device=device)

    eq = AcousticSEM2D.build(mesh, vp_e)

    # --- time discretization ------------------------------------------
    dt = 0.0008
    nt = 900
    t = torch.arange(nt, dtype=dtype) * dt
    f0 = 15.0
    delay = 0.08
    src = ricker(t, f0=f0, delay=delay)

    # Source near the top (typical surface acquisition geometry).
    src_xz = [(500.0, 100.0)]
    # Receivers: line at z=100 m, every 50 m.
    rec_xz = [(x, 100.0) for x in np.arange(50.0, 1000.0, 50.0)]

    prop = SEMPropagator.build(eq, dt=dt, nt=nt,
                               source_xz=src_xz, receiver_xz=rec_xz)
    print(f"CFL estimate (informational): dt_max ≈ {prop.cfl_estimate():.4g} s, "
          f"chosen dt = {dt}")

    seis_t, snaps_t = prop.forward(src, record_field_every=50)
    seis = seis_t[..., 0].T.numpy()                         # (n_rec, nt)
    snaps = snaps_t[:, 0].numpy()                           # (n_snap, n_global)
    snap_t = np.arange(0, nt, 50)

    # --- snapshots -----------------------------------------------------
    # Reshape n_global -> (nz_global, nx_global) (mesh.x_global is row-major)
    nx_global = nex * mesh.n_order + 1
    nz_global = nez * mesh.n_order + 1
    grid = snaps.reshape(-1, nz_global, nx_global)

    ncols = 4
    nrows = int(np.ceil(grid.shape[0] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    vmin, vmax = percentile_clip(grid, 1.0, 99.0)
    for i, ax in enumerate(axes.flat):
        if i >= grid.shape[0]:
            ax.axis("off")
            continue
        im = ax.imshow(
            grid[i],
            extent=[0, x_max, z_max, 0],  # z increases downward
            cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal",
        )
        ax.set_title(f"t = {snap_t[i] * dt:.3f} s")
        ax.plot([s[0] for s in src_xz], [s[1] for s in src_xz], "ko", ms=4)
        ax.plot([r[0] for r in rec_xz], [r[1] for r in rec_xz], "k.", ms=2)
    fig.tight_layout()
    snap_path = out / "sem_acoustic_snapshots.png"
    fig.savefig(snap_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {snap_path.resolve()}")

    # --- gather --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.percentile(np.abs(seis), 99.0)
    ax.imshow(seis.T, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
              extent=[rec_xz[0][0], rec_xz[-1][0], t[-1].item(), 0])
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title("SEM acoustic — surface gather")
    fig.tight_layout()
    gath_path = out / "sem_acoustic_gather.png"
    fig.savefig(gath_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {gath_path.resolve()}")

    # --- trace at one receiver -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3))
    idx = len(rec_xz) // 2
    ax.plot(t.numpy(), seis[idx], lw=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("pressure")
    ax.set_title(f"trace at x={rec_xz[idx][0]:.0f}, z={rec_xz[idx][1]:.0f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    trc_path = out / "sem_acoustic_trace.png"
    fig.savefig(trc_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {trc_path.resolve()}")

    # --- numerical sanity ---------------------------------------------
    # Direct arrival time at receiver `idx`: offset = |x_r - x_s|, vp known
    src_x, src_z = src_xz[0]
    rec_x, rec_z = rec_xz[idx]
    offset = np.hypot(rec_x - src_x, rec_z - src_z)
    expected_t = offset / vp_const + delay
    # locate peak of |trace| in a window around expected_t
    iw = (np.abs(t.numpy() - expected_t) < 4 / f0)  # 4-period window
    peak_t = t.numpy()[iw][np.argmax(np.abs(seis[idx][iw]))]
    print(f"Direct arrival check at offset {offset:.1f} m: "
          f"expected peak ≈ {expected_t:.4f} s, observed ≈ {peak_t:.4f} s, "
          f"diff = {1e3 * (peak_t - expected_t):.2f} ms")


if __name__ == "__main__":
    main(os.environ.get("OUTDIR", "_smoketest"))

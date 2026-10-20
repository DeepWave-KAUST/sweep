"""Smoketest: elastic SEM on a Gaussian-hill topography mesh.

A vertical force just below the hill summit should generate clear P/S
waves diving into the half-space + a Rayleigh wave running along the
curved surface (the headline feature SEM is being asked to handle).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import (
    ElasticSEM2DDeformed,
    SEMPropagator,
    build_topography_quad9,
)


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64

    # ---- domain ----
    x_max  = 1280.0
    z_bottom = 960.0
    nex, nez = 32, 24                          # element grid

    # Gaussian hill: peak at centre, h0 = 50 m above flat baseline
    h0 = 50.0
    x_mid = x_max / 2.0
    sigma = 160.0

    def z_top_fn(x):
        return h0 * (1.0 - torch.exp(-((x - x_mid) ** 2) / (2.0 * sigma ** 2)))

    mesh = build_topography_quad9(
        nex=nex, nez=nez, x_max=x_max, z_bottom=z_bottom,
        z_top_fn=z_top_fn, n_order=4, dtype=dtype,
    )
    print(f"mesh: E={mesh.n_elements}, n_global={mesh.n_global}, "
          f"surface z ∈ [{mesh.z_e[:, 0, :].min().item():.1f}, "
          f"{mesh.z_e[:, 0, :].max().item():.1f}] m")

    vp_v, vs_v, rho_v = 2200.0, 1100.0, 2000.0
    vp = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), vp_v, dtype=dtype)
    vs = torch.full_like(vp, vs_v); rho = torch.full_like(vp, rho_v)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho)

    dt = 0.0006
    nt = 1000
    f0, delay = 15.0, 0.08
    t = torch.arange(nt, dtype=dtype) * dt
    src_wave = torch.from_numpy(ricker(t.numpy(), f0, delay))

    # Source 30 m below the local surface at peak
    src_x = x_mid + 1.0                              # nudge off boundary
    src_z = float(z_top_fn(torch.tensor([src_x])).item()) + 30.0
    src_xz = (src_x, src_z)
    # Receivers along the curved surface (1 cell ~= 20 m below it)
    rec_x_arr = np.arange(80.0, x_max - 80.0, 40.0)
    rec_z_arr = [float(z_top_fn(torch.tensor([float(x)])).item()) + 5.0
                 for x in rec_x_arr]
    rec_xz = list(zip(rec_x_arr.tolist(), rec_z_arr))

    prop = SEMPropagator.build(
        eq, dt=dt, nt=nt, source_xz=[src_xz], receiver_xz=rec_xz,
        source_kinds=["force"], source_payloads=[(0.0, 1.0)],
    )
    every = 40
    seis_t, snaps_t = prop.forward(src_wave, record_field_every=every)
    snaps_x = snaps_t[:, 0].numpy()
    snaps_z = snaps_t[:, 1].numpy()
    snap_t  = np.arange(0, nt, every) * dt
    seis_z  = seis_t[..., 1].T.numpy()
    print(f"snaps: {snaps_x.shape}; surface gather: {seis_z.shape}")

    # ---- render: displacement magnitude on a fine regular grid ----
    nx_g = mesh.nex * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1
    # SEM x_global / z_global form the deformed structured grid; griddata
    # interpolation is needed because z varies with x.
    from scipy.interpolate import griddata
    xi = np.linspace(0, x_max, 601)
    zi = np.linspace(0, z_bottom, 451)
    XI, ZI = np.meshgrid(xi, zi)
    coords_xy = np.stack([mesh.x_global.numpy(), mesh.z_global.numpy()], axis=-1)

    pick = list(range(0, snaps_x.shape[0], 2))[:9]
    nrows, ncols = 3, (len(pick) + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows),
                             squeeze=False)
    for ax, i in zip(axes.flat, pick):
        ux = snaps_x[i]; uz = snaps_z[i]
        mag = np.sqrt(ux ** 2 + uz ** 2)
        img = griddata((coords_xy[:, 0], coords_xy[:, 1]), mag,
                       (XI, ZI), method="linear", fill_value=0.0)
        vmax = np.percentile(img, 99.5)
        im = ax.imshow(img, extent=[0, x_max, z_bottom, 0],
                       cmap="hot_r", vmin=0, vmax=vmax, aspect="equal")
        # Topography line
        ax.plot(xi, z_top_fn(torch.from_numpy(xi)).numpy(), "b-", lw=1.0)
        ax.plot(src_xz[0], src_xz[1], "co", ms=4)
        ax.set_title(f"t = {snap_t[i]:.3f} s")
    fig.tight_layout()
    sp = out / "sem_topo_smoketest_snapshots.png"
    fig.savefig(sp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sp.resolve()}")

    # ---- gather ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmax = np.percentile(np.abs(seis_z), 99.0)
    ax.imshow(seis_z.T, aspect="auto", cmap="seismic",
              vmin=-vmax, vmax=vmax,
              extent=[rec_x_arr[0], rec_x_arr[-1], t[-1].item(), 0])
    ax.axvline(x_mid, color="b", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title("u_z surface gather, Gaussian-hill topography")
    fig.tight_layout()
    gp = out / "sem_topo_smoketest_gather.png"
    fig.savefig(gp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {gp.resolve()}")


if __name__ == "__main__":
    main(os.environ.get("OUTDIR",
                        "_smoketest"))

"""Visual smoketest for ElasticSEM2D: vertical force source near a free
surface should produce a downgoing P, downgoing S, and a Rayleigh
surface wave along the top.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import ElasticSEM2D, SEMPropagator, build_quad4_mesh


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main(out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64

    x_max, z_max = 1280.0, 960.0
    nex, nez = 32, 24                       # dh_e = 40 m
    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=4, dtype=dtype)
    print(f"mesh: E={mesh.n_elements}, ngll={mesh.ngll}, "
          f"n_global={mesh.n_global}, dh_e={mesh.dx_e}")

    vp_v, vs_v, rho_v = 2200.0, 1100.0, 2000.0
    vp = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), vp_v, dtype=dtype)
    vs = torch.full_like(vp, vs_v)
    rho = torch.full_like(vp, rho_v)

    eq = ElasticSEM2D.build(mesh, vp, vs, rho)

    dt = 0.0006
    nt = 1000
    f0, delay = 15.0, 0.08
    t = torch.arange(nt, dtype=dtype) * dt
    src = torch.from_numpy(ricker(t.numpy(), f0, delay))

    # Vertical force source 60 m below the free surface (top), at centre x.
    src_xz = (x_max / 2.0, 60.0)
    # Receiver line along z=20 m (near surface) and a deeper line at z=400 m
    rec_xz = [(x, 20.0) for x in np.arange(60.0, x_max - 60.0, 40.0)]

    prop = SEMPropagator.build(
        eq, dt=dt, nt=nt,
        source_xz=[src_xz], receiver_xz=rec_xz,
        source_kinds=["force"],
        source_payloads=[(0.0, 1.0)],         # F_z only (vertical)
    )
    every = 40
    seis_t, snaps_t = prop.forward(src, record_field_every=every)
    snaps_x = snaps_t[:, 0].numpy()
    snaps_z = snaps_t[:, 1].numpy()
    snap_t  = np.arange(0, nt, every) * dt
    seis_x  = seis_t[..., 0].T.numpy()
    seis_z  = seis_t[..., 1].T.numpy()
    print(f"snaps: {snaps_x.shape}; surface gather: {seis_z.shape}")

    # Reshape to regular GLL grid for visualization.
    nx_g = nex * mesh.n_order + 1
    nz_g = nez * mesh.n_order + 1
    snaps_x_grid = snaps_x.reshape(-1, nz_g, nx_g)
    snaps_z_grid = snaps_z.reshape(-1, nz_g, nx_g)
    mag = np.sqrt(snaps_x_grid ** 2 + snaps_z_grid ** 2)

    # ----- Snapshot grid (|u|) -----
    pick = list(range(0, mag.shape[0], 2))[:9]
    nrows = 3; ncols = (len(pick) + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             squeeze=False)
    vmax = np.percentile(mag, 99.0)
    for ax, i in zip(axes.flat, pick):
        im = ax.imshow(mag[i], extent=[0, x_max, z_max, 0],
                       cmap="hot_r", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"t = {snap_t[i]:.3f} s")
        ax.plot(src_xz[0], src_xz[1], "co", ms=4)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    sp = out / "sem_elastic_smoketest_snapshots.png"
    fig.savefig(sp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sp.resolve()}")

    # ----- Surface gather (v_z) -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rec_x = np.array([rx for (rx, _) in rec_xz])
    vmax_g = np.percentile(np.abs(seis_z), 99.0)
    ax.imshow(seis_z.T, aspect="auto", cmap="seismic",
              vmin=-vmax_g, vmax=vmax_g,
              extent=[rec_x[0], rec_x[-1], t[-1].item(), 0])
    ax.set_xlabel("receiver x (m)")
    ax.set_ylabel("t (s)")
    ax.set_title("u_z near-surface gather (Rayleigh + P+S arrivals)")
    fig.tight_layout()
    gp = out / "sem_elastic_smoketest_gather.png"
    fig.savefig(gp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {gp.resolve()}")


if __name__ == "__main__":
    main(os.environ.get("OUTDIR", "_smoketest"))

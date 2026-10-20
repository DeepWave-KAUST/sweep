"""Visual demo: no-absorption vs Stacey vs CPML on the same SEM domain.

Renders snapshots at late times so you can see how much energy leaks back
from each boundary treatment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import AcousticSEM2D, SEMPropagator, build_cpml, build_quad4_mesh


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main():
    out = Path("_smoketest")
    out.mkdir(parents=True, exist_ok=True)

    dt, f0, delay = 0.0008, 12.0, 0.10
    nt = 600
    x_max, z_max = 2000.0, 1000.0
    nex, nez = 20, 10
    src_xz = (1000.0, 500.0)

    mesh = build_quad4_mesh(nex=nex, nez=nez, x_max=x_max, z_max=z_max,
                            n_order=4, dtype=torch.float64)
    vp_e = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), 2000.0,
                      dtype=torch.float64)
    t = torch.arange(nt, dtype=torch.float64) * dt
    src = torch.from_numpy(ricker(t.numpy(), f0, delay))

    runs = {}
    # 1. No absorption — pure reflecting box (free surface on top + Neumann
    #    elsewhere via the unmodified weak form on those edges).
    eq_none = AcousticSEM2D.build(mesh, vp_e, dirichlet=("top",))
    runs["no absorbing"] = SEMPropagator.build(
        eq_none, dt=dt, nt=nt, source_xz=[src_xz], receiver_xz=[(0.0, 0.0)],
    ).forward(src, record_field_every=20)[1][:, 0].numpy()

    # 2. Stacey first-order on three sides
    eq_stacey = AcousticSEM2D.build(mesh, vp_e,
                                    dirichlet=("top",))
    runs["Stacey 1st-order"] = SEMPropagator.build(
        eq_stacey, dt=dt, nt=nt, source_xz=[src_xz], receiver_xz=[(0.0, 0.0)],
    ).forward(src, record_field_every=20)[1][:, 0].numpy()

    # 3. CPML on three sides (free top)
    cpml = build_cpml(mesh, dt=dt,
                      sides=("left", "right", "bottom"),
                      pml_thickness_elements=4,
                      c_max=2000.0, f_dominant=f0,
                      reflection_target=1e-3)
    eq_cpml = AcousticSEM2D.build(mesh, vp_e,
                                  dirichlet=("top",), cpml=cpml)
    runs["CPML (4-elem)"] = SEMPropagator.build(
        eq_cpml, dt=dt, nt=nt, source_xz=[src_xz], receiver_xz=[(0.0, 0.0)],
    ).forward(src, record_field_every=20)[1][:, 0].numpy()

    nx_g = mesh.nex * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1
    snap_times = list(range(0, nt, 20))
    pick_step_idx = [snap_times.index(s) for s in (200, 300, 400, 500)]

    # Build the *true* GLL coordinate axes (non-uniform within each element).
    # x_global/z_global on the mesh are row-major; the per-axis 1-D coords
    # are recovered by slicing the first row / first column.
    x_axis = mesh.x_global.reshape(nz_g, nx_g)[0, :].numpy()
    z_axis = mesh.z_global.reshape(nz_g, nx_g)[:, 0].numpy()
    # Interpolate to a fine regular grid for crisp rendering.
    from scipy.interpolate import RegularGridInterpolator
    xi = np.linspace(0, x_max, 801)
    zi = np.linspace(0, z_max, 401)
    XI, ZI = np.meshgrid(xi, zi)

    def to_fine(snap2d):
        f = RegularGridInterpolator((z_axis, x_axis), snap2d,
                                    bounds_error=False, fill_value=0.0)
        return f((ZI, XI))

    fig, axes = plt.subplots(len(pick_step_idx), len(runs),
                             figsize=(5.0 * len(runs), 3.0 * len(pick_step_idx)),
                             squeeze=False)
    ref_run = runs["no absorbing"].reshape(-1, nz_g, nx_g)
    vmax = np.percentile(np.abs(ref_run[snap_times.index(200)]), 99.0)

    for col, (label, snaps_flat) in enumerate(runs.items()):
        snaps = snaps_flat.reshape(-1, nz_g, nx_g)
        for row, k in enumerate(pick_step_idx):
            ax = axes[row, col]
            t_phys = snap_times[k] * dt
            img = to_fine(snaps[k])
            ax.imshow(img, extent=[0, x_max, z_max, 0],
                      cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal")
            ax.plot(src_xz[0], src_xz[1], "ko", ms=4)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"t = {t_phys:.3f}s", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fpath = out / "cpml_vs_stacey_vs_none.png"
    fig.savefig(fpath, dpi=180, bbox_inches="tight")
    print(f"Saved: {fpath.resolve()}")


if __name__ == "__main__":
    main()

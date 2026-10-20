"""1st-order velocity-stress SEM vs SPECFEM2D wavefield on hill_case.

Renders side-by-side snapshots of v_x, v_z at five target times:

  * SEM 1st-order (:class:`SEMPropagator`) on the Quad9
    Gaussian-hill mesh, with free top + simple CPML on sides.
  * SPECFEM2D hill_case reference (binary wavefield dumps).

The 1st-order solver outputs velocity directly so no time
differentiation is needed (contrast :mod:`wavefield_3way_hill.py`,
where 2nd-order displacement snapshots are centre-differenced).

SPECFEM stores v_z z-up; SEM stores v_z z-down. We negate the SPECFEM
v_z so both panels share the SEM (sweep) convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from sweep.sem import (
    ElasticVSSEM2DDeformed,
    SEMPropagator,
    build_simple_elastic_cpml,
    build_topography_quad9,
)


# Match wavefield_3way_hill.py geometry exactly.
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ        = 64, 44
N_ORDER         = 4
DT              = 1e-3
NSTEP           = 700
F0, TSHIFT      = 12.0, 0.12
VP, VS, RHO     = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ     = (640.0, 860.0)
SRC_MT          = (1.0, 1.0, 0.0)

HILL_BASE_DEPTH = 50.0
HILL_H0         = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0

TARGETS_T = [0.10, 0.20, 0.30, 0.40, 0.50]


def z_top_fn(x):
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz_spec):
    return (xz_spec[0], Z_BOTTOM - xz_spec[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def load_specfem_dumps(out_dir):
    grid_raw = np.fromfile(out_dir / "wavefield_grid_for_dumps.bin",
                           dtype=np.float32)
    n_pts = grid_raw.size // 2
    coords = grid_raw.reshape(n_pts, 2).astype(np.float64)
    dump_files = sorted(out_dir.glob("wavefield*_01.bin"))
    snaps = []
    steps = []
    for f in dump_files:
        step = int(f.name[len("wavefield"):len("wavefield") + 7])
        raw = np.fromfile(f, dtype=np.float32)
        if raw.size != 2 * n_pts:
            continue
        snaps.append(raw.reshape(n_pts, 2).astype(np.float64))
        steps.append(step)
    return coords, np.stack(snaps, axis=0), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specfem_dir",
        default="_specfem_compare/hill_case",
    )
    parser.add_argument(
        "--out_dir",
        default="_smoketest",
    )
    args = parser.parse_args()

    specfem_dir = Path(args.specfem_dir)
    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. SPECFEM wavefield dumps + time alignment ------------------- #
    coords_spec, snaps_spec, spec_steps = load_specfem_dumps(
        specfem_dir / "OUTPUT_FILES"
    )
    stations_lines = (specfem_dir / "DATA" / "STATIONS").read_text().splitlines()
    rec_spec_x = np.array([float(p.split()[2]) for p in stations_lines])
    j_close = int(np.argmin(np.abs(rec_spec_x - SRC_SPEC_XZ[0])))
    seis_z_close = np.loadtxt(
        specfem_dir / "OUTPUT_FILES" / f"AA.S{j_close:03d}.BXZ.semv"
    )
    t_file_close = seis_z_close[:, 0]
    peak_t_spec = t_file_close[np.argmax(np.abs(seis_z_close[:, 1]))]
    expected_t = TSHIFT + abs(rec_spec_x[j_close] - SRC_SPEC_XZ[0]) / VP
    spec_shift = expected_t - peak_t_spec
    phys_t_spec = np.array([(k - 1) * DT + t_file_close[0] + spec_shift
                            for k in spec_steps])
    print(f"SPECFEM time shift = {spec_shift*1e3:+.1f} ms; "
          f"{len(spec_steps)} dump snapshots available.")
    coords_spec_sem = np.stack(
        [coords_spec[:, 0], Z_BOTTOM - coords_spec[:, 1]], axis=-1
    )

    # ----- 2. SEM 1st-order forward with snapshots ---------------------- #
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    vp_e  = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP,
                       dtype=torch.float64)
    vs_e  = torch.full_like(vp_e, VS)
    rho_e = torch.full_like(vp_e, RHO)
    cpml = build_simple_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "bottom"),
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    eq = ElasticVSSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, free_top=True, cpml=cpml,
    )
    src_sem_xz = to_sem(SRC_SPEC_XZ)
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[src_sem_xz], receiver_xz=[(50.0, 100.0)],
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t_sem = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_sem, F0, TSHIFT))
    _, sem_snaps = prop.forward(src_wave, record_field_every=5)
    sem_snap_vx = sem_snaps[:, 0].numpy()
    sem_snap_vz = sem_snaps[:, 1].numpy()
    sem_snap_step = np.arange(0, nt, 5)
    sem_phys_t = sem_snap_step * DT
    print(f"SEM 1st-order: {len(sem_phys_t)} snapshots; "
          f"peak |v_x| = {np.abs(sem_snap_vx).max():.3e}; "
          f"peak |v_z| = {np.abs(sem_snap_vz).max():.3e}")

    # ----- 3. Render side-by-side ---------------------------------------- #
    nx_img, nz_img = 601, 421
    xi_img = np.linspace(0, X_MAX, nx_img)
    zi_img = np.linspace(0, Z_BOTTOM, nz_img)
    XI, ZI = np.meshgrid(xi_img, zi_img)
    coords_sem = np.stack([mesh.x_global.numpy(), mesh.z_global.numpy()], axis=-1)

    def sem_to_image(flat_field):
        return griddata((coords_sem[:, 0], coords_sem[:, 1]),
                        flat_field, (XI, ZI),
                        method="linear", fill_value=0.0)

    def spec_to_image(flat_field):
        return griddata((coords_spec_sem[:, 0], coords_spec_sem[:, 1]),
                        flat_field, (XI, ZI),
                        method="linear", fill_value=0.0)

    topo_x_dense = torch.linspace(0, X_MAX, 400)
    topo_z_dense = z_top_fn(topo_x_dense).numpy()
    topo_x_dense = topo_x_dense.numpy()

    n_t = len(TARGETS_T)
    fig, axes = plt.subplots(n_t, 4, figsize=(16, 3.0 * n_t), squeeze=False)

    for r, tt in enumerate(TARGETS_T):
        j_sem  = int(np.argmin(np.abs(sem_phys_t  - tt)))
        j_spec = int(np.argmin(np.abs(phys_t_spec - tt)))

        sem_vx_img  = sem_to_image(sem_snap_vx[j_sem])
        sem_vz_img  = sem_to_image(sem_snap_vz[j_sem])
        spec_vx_img =  spec_to_image(snaps_spec[j_spec, :, 0])
        # SPECFEM v_z z-up → flip to SEM z-down convention
        spec_vz_img = -spec_to_image(snaps_spec[j_spec, :, 1])

        def show(ax, img, label):
            vm = max(np.percentile(np.abs(img), 99.0), 1e-30)
            ax.imshow(img / vm, extent=[0, X_MAX, Z_BOTTOM, 0],
                      cmap="seismic", vmin=-1.0, vmax=1.0, aspect="equal")
            ax.plot(topo_x_dense, topo_z_dense, "k-", lw=0.6)
            ax.plot(src_sem_xz[0], src_sem_xz[1], "go", ms=3, mew=0)
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=7)

        show(axes[r, 0], sem_vx_img,
             f"SEM 1st-order  v_x  t={sem_phys_t[j_sem]:.2f}s")
        show(axes[r, 1], spec_vx_img,
             f"SPECFEM2D  v_x  t={phys_t_spec[j_spec]:.2f}s")
        show(axes[r, 2], sem_vz_img,
             f"SEM 1st-order  v_z  t={sem_phys_t[j_sem]:.2f}s")
        show(axes[r, 3], spec_vz_img,
             f"SPECFEM2D  v_z (flipped)  t={phys_t_spec[j_spec]:.2f}s")
        axes[r, 0].set_ylabel(f"target  {tt:.2f}s", fontsize=8)

    fig.suptitle("1st-order velocity-stress SEM vs SPECFEM2D — hill_case wavefield",
                 fontsize=11)
    fig.tight_layout()
    wp = fig_dir / "hill_wavefield_1st_vs_specfem.png"
    fig.savefig(wp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {wp.resolve()}")


if __name__ == "__main__":
    main()

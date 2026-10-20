"""Three-way wavefield snapshot comparison on the Gaussian-hill case:

  * SEM Q9 (this PoC) — conforming deformed mesh, free top = natural BC.
  * sweep FD elastic_curvilinear_hr — coordinate-transform FD on
    structured grid, dh=5 m (pre-computed: hill_case/OUTPUT_FILES_sweep
    /curvilinear_hr_dh5.npz).
  * SPECFEM2D hill_case — reference (binary wavefield dumps).

All three are shown at five physical times (0.10, 0.20, 0.30, 0.40,
0.50 s) and both velocity components (v_x, v_z). The topography curve
``z_top(x)`` is overlaid in every panel for visual reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import (
    ElasticSEM2DDeformed,
    SEMPropagator,
    build_topography_quad9,
)


# Physical / discretization constants — match run_compare_hill.py
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ = 64, 44
N_ORDER = 4
DT = 1e-3
NSTEP = 700
F0, TSHIFT = 12.0, 0.12
VP, VS, RHO = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ = (640.0, 860.0)
SRC_MT      = (1.0, 1.0, 0.0)

# Gaussian hill (z_down convention; peak at z=0, flat at z=HILL_BASE_DEPTH)
HILL_BASE_DEPTH = 50.0
HILL_H0         = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0


def z_top_fn(x: torch.Tensor) -> torch.Tensor:
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz_spec):
    return (xz_spec[0], Z_BOTTOM - xz_spec[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def load_specfem_dumps(out_dir: Path, n_pts_expect: int):
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


def load_curvilinear_npz(path: Path):
    d = np.load(path)
    snaps = d["snaps"]           # (n_snap, n_field, B=1, C=1, NZ, NX)
    snap_idx = d["snap_idx"]
    topo_row = d["topo_row"]
    dh = float(d["dh"])
    dt = float(d["dt"])
    NZ = int(d["NZ"]); NX = int(d["NX"])
    # FIELD_SPECS order: index 0 = vx, 1 = vz
    vx = snaps[:, 0, 0, 0]       # (n_snap, NZ, NX)
    vz = snaps[:, 1, 0, 0]
    phys_t = snap_idx * dt
    return {
        "vx": vx, "vz": vz, "phys_t": phys_t, "topo_row": topo_row,
        "dh": dh, "NZ": NZ, "NX": NX,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specfem_dir",
        default="_specfem_compare/hill_case")
    parser.add_argument("--curv_npz",
        default="_specfem_compare/hill_case/OUTPUT_FILES_sweep/curvilinear_hr_dh5.npz")
    parser.add_argument("--out_dir",
        default="_smoketest")
    args = parser.parse_args()

    specfem_dir = Path(args.specfem_dir)
    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # ============== 1. SPECFEM wavefield dumps ==============
    coords_spec, snaps_spec, spec_steps = load_specfem_dumps(
        specfem_dir / "OUTPUT_FILES", n_pts_expect=45489
    )
    # SPECFEM peak time → shift to align with SEM physical axis.
    # (Use S010 — closest to source.)
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

    # SPECFEM z is up-positive; convert to SEM down-positive.
    coords_spec_sem = np.stack(
        [coords_spec[:, 0], Z_BOTTOM - coords_spec[:, 1]], axis=-1
    )

    # ============== 2. Sweep curvilinear FD snapshots ==============
    curv = load_curvilinear_npz(Path(args.curv_npz))
    print(f"Sweep curvilinear FD: snapshots at {curv['phys_t']} s; "
          f"grid {curv['NZ']}×{curv['NX']}, dh={curv['dh']}")
    # Coordinates of curvilinear grid:
    nx_c = curv["NX"]; nz_c = curv["NZ"]; dh_c = curv["dh"]
    topo_row = curv["topo_row"]
    # SEM coords for the curvilinear grid: x = ix*dh, z = (iz - peak_topo_row)*dh
    # to align peak surface with SEM z=0. Note SWEEP topo_row[ix] is the row
    # of the SURFACE for that column; peak surface = MIN(topo_row).
    peak_row = int(topo_row.min())
    curv_x_axis = np.arange(nx_c) * dh_c
    curv_z_axis = (np.arange(nz_c) - peak_row) * dh_c
    print(f"Sweep curvilinear physical extents: "
          f"x ∈ [{curv_x_axis[0]:.0f}, {curv_x_axis[-1]:.0f}], "
          f"z ∈ [{curv_z_axis[0]:.0f}, {curv_z_axis[-1]:.0f}] (z=0 = SEM peak)")

    # ============== 3. SEM Q9 forward (with snapshots) ==============
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    vp_e  = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP,  dtype=torch.float64)
    vs_e  = torch.full_like(vp_e, VS); rho_e = torch.full_like(vp_e, RHO)
    eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e)
    src_sem_xz = to_sem(SRC_SPEC_XZ)
    rec_xz_dummy = [(50.0, 100.0)]
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[src_sem_xz], receiver_xz=rec_xz_dummy,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t_sem = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_sem, F0, TSHIFT))
    sem_out = prop.forward(src_wave, record_field_every=5)
    sem_snap_ux  = sem_snaps[:, 0].numpy()
    sem_snap_uz  = sem_snaps[:, 1].numpy()
    sem_snap_step = np.arange(0, SEM_NT, SEM_SNAP_EVERY)
    sem_phys_t = sem_snap_step * DT
    # Centred difference into velocity: stride is 5*DT
    sem_snap_vx = np.zeros_like(sem_snap_ux)
    sem_snap_vz = np.zeros_like(sem_snap_uz)
    sem_snap_vx[1:-1] = (sem_snap_ux[2:] - sem_snap_ux[:-2]) / (2 * 5 * DT)
    sem_snap_vz[1:-1] = (sem_snap_uz[2:] - sem_snap_uz[:-2]) / (2 * 5 * DT)
    print(f"SEM Q9: {len(sem_phys_t)} snapshots recorded.")

    # ============== 4. Render side-by-side ==============
    targets_t = curv["phys_t"]                     # [0.10, 0.20, 0.30, 0.40, 0.50]
    nx_g = mesh.nex * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1

    # SEM Q9: deformed structured grid → griddata to a regular image
    from scipy.interpolate import griddata
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

    # Build figure: 5 rows (times) × 6 cols (SEM v_x, sweep v_x, SPECFEM v_x,
    # SEM v_z, sweep v_z, SPECFEM v_z).
    n_t = len(targets_t)
    fig, axes = plt.subplots(n_t, 6, figsize=(18, 2.8 * n_t), squeeze=False)
    topo_x_dense = torch.linspace(0, X_MAX, 400)
    topo_z_dense = z_top_fn(topo_x_dense).numpy()
    topo_x_dense = topo_x_dense.numpy()

    for r, tt in enumerate(targets_t):
        # ---- SEM Q9 ----
        j_sem = int(np.argmin(np.abs(sem_phys_t - tt)))
        sem_vx_img = sem_to_image(sem_snap_vx[j_sem])
        sem_vz_img = sem_to_image(sem_snap_vz[j_sem])
        # ---- sweep curvilinear ----
        j_curv = int(np.argmin(np.abs(curv["phys_t"] - tt)))
        curv_vx = curv["vx"][j_curv]               # (NZ, NX) in (iz, ix)
        curv_vz = curv["vz"][j_curv]
        # ---- SPECFEM ----
        j_spec = int(np.argmin(np.abs(phys_t_spec - tt)))
        spec_vx_img = spec_to_image(snaps_spec[j_spec, :, 0])
        spec_vz_img = spec_to_image(snaps_spec[j_spec, :, 1])

        def show(ax, img, extent, label):
            vm = max(np.percentile(np.abs(img), 99.0), 1e-30)
            ax.imshow(img / vm, extent=extent,
                      cmap="seismic", vmin=-1.0, vmax=1.0, aspect="equal")
            ax.plot(topo_x_dense, topo_z_dense, "k-", lw=0.6)
            ax.plot(src_sem_xz[0], src_sem_xz[1], "go", ms=3, mew=0)
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=7)

        sem_extent = [0, X_MAX, Z_BOTTOM, 0]
        curv_extent = [curv_x_axis[0], curv_x_axis[-1],
                       curv_z_axis[-1], curv_z_axis[0]]
        spec_extent = sem_extent

        show(axes[r, 0], sem_vx_img,  sem_extent,
             f"SEM Q9  v_x  t={sem_phys_t[j_sem]:.2f}s")
        show(axes[r, 1], curv_vx,     curv_extent,
             f"sweep curvilinear FD  v_x  t={curv['phys_t'][j_curv]:.2f}s")
        show(axes[r, 2], spec_vx_img, spec_extent,
             f"SPECFEM2D  v_x  t={phys_t_spec[j_spec]:.2f}s")
        show(axes[r, 3], sem_vz_img,  sem_extent,
             f"SEM Q9  v_z  t={sem_phys_t[j_sem]:.2f}s")
        show(axes[r, 4], curv_vz,     curv_extent,
             f"sweep curvilinear FD  v_z  t={curv['phys_t'][j_curv]:.2f}s")
        show(axes[r, 5], spec_vz_img, spec_extent,
             f"SPECFEM2D  v_z  t={phys_t_spec[j_spec]:.2f}s")
        axes[r, 0].set_ylabel(f"target  {tt:.2f}s", fontsize=8)

    fig.tight_layout()
    wp = fig_dir / "hill_wavefield_3way.png"
    fig.savefig(wp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {wp.resolve()}")


if __name__ == "__main__":
    main()

"""Compare SEM ElasticSEM2D against SPECFEM2D's flat_case (elastic).

Matches the existing setup in
``_specfem_compare/flat_case``:

  * Domain 1270 × 890 m, 64 × 44 quad elements, N = 4 GLL.
  * Material: vp = 2200, vs = 1100, ρ = 2000 (uniform).
  * Source: explosive moment-tensor (Mxx = Mzz = 1, Mxz = 0)
    at SPECFEM coords (640, 860). Ricker, f0 = 12 Hz, tshift = 0.12 s.
  * Receivers: surface line at SPECFEM z = 880 (10 m below the free
    surface), x = 60..600 in 30 m steps.
  * DT = 1e-3, NSTEP = 950.

SPECFEM outputs velocity (``seismotype = 2``) in ``AA.S*.BXX.semv``
(``v_x``) and ``AA.S*.BXZ.semv`` (``v_z``). Our SEM evolves
displacement ``u``; we differentiate numerically to compare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import ElasticSEM2D, SEMPropagator, build_quad4_mesh


# ----- physical setup (frozen to match SPECFEM flat_case) -----
X_MAX, Z_MAX = 1270.0, 890.0           # SPECFEM coords; SEM uses z-down
NEX, NEZ = 64, 44
N_ORDER = 4
DT = 1e-3
NSTEP = 950
F0, TSHIFT = 12.0, 0.12
VP, VS, RHO = 2200.0, 1100.0, 2000.0

# SPECFEM SOURCE: (xs=640, zs=860) with Mxx = Mzz = 1, Mxz = 0
SRC_SPEC_XZ = (640.0, 860.0)
SRC_MT      = (1.0, 1.0, 0.0)          # (Mxx, Mzz, Mxz)

# Convert to SEM (z-down). z_sem = Z_MAX − z_spec.
def to_sem(xz_spec):
    return (xz_spec[0], Z_MAX - xz_spec[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def load_specfem_seismograms(out_dir: Path, n_rec: int):
    """Returns vx[n_rec, nt], vz[n_rec, nt], t_file[nt]."""
    vx_list, vz_list = [], []
    t_file = None
    for j in range(n_rec):
        dx = np.loadtxt(out_dir / f"AA.S{j:03d}.BXX.semv")
        dz = np.loadtxt(out_dir / f"AA.S{j:03d}.BXZ.semv")
        if t_file is None:
            t_file = dx[:, 0]
        vx_list.append(dx[:, 1])
        vz_list.append(dz[:, 1])
    return np.stack(vx_list, axis=0), np.stack(vz_list, axis=0), t_file


def load_specfem_wavefield_dumps(out_dir: Path):
    """Read SPECFEM2D vector wavefield dumps + grid file.

    Layouts (verified empirically on the flat_case dump):

      * ``wavefield_grid_for_dumps.bin``: interleaved (x, z) float32
        pairs, walking element-by-element, then row-major within each
        element. ``n_pts = 45489`` unique deduped GLL nodes.
      * ``wavefield<step>_01.bin``: interleaved (vx, vz) float32 pairs
        in the SAME order as the grid file.

    Returns ``(coords[N, 2], snaps[T, N, 2], step_indices)`` with z in
    SPECFEM (z-up) coordinates.
    """
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
            print(f"skip {f.name}: size {raw.size} vs expected {2*n_pts}")
            continue
        snaps.append(raw.reshape(n_pts, 2).astype(np.float64))
        steps.append(step)
    return coords, np.stack(snaps, axis=0), steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specfem_dir",
        default="_specfem_compare/flat_case",
    )
    parser.add_argument(
        "--out_dir",
        default="_smoketest",
    )
    args = parser.parse_args()

    specfem_out = Path(args.specfem_dir) / "OUTPUT_FILES"
    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- receiver coords from STATIONS -----
    stations_lines = (Path(args.specfem_dir) / "DATA" / "STATIONS").read_text().splitlines()
    rec_spec = [(float(p.split()[2]), float(p.split()[3])) for p in stations_lines]
    rec_sem  = [to_sem(rx) for rx in rec_spec]
    n_rec = len(rec_sem)
    print(f"{n_rec} receivers, x ∈ [{rec_spec[0][0]:.0f}, {rec_spec[-1][0]:.0f}] m, "
          f"surface z = {rec_spec[0][1]:.0f} m (SPECFEM)")

    # ----- SPECFEM seismograms (velocity) -----
    spec_vx, spec_vz, t_spec_file = load_specfem_seismograms(specfem_out, n_rec)
    print(f"SPECFEM velocity gather: vx {spec_vx.shape}, vz {spec_vz.shape}")

    # SPECFEM SEM convention applies its own t0 offset. Re-anchor by locating
    # the v_z peak at the near-source receiver (S010 at x=360 m — closest to
    # source at 640 m on the half-line that exists in STATIONS).
    # Actually the STATIONS list above only spans x=60..600 (offset 40..580 m
    # from src x=640). Pick the closest one and use it.
    j_close = int(np.argmin(np.abs(np.array([rx for rx, _ in rec_spec]) - SRC_SPEC_XZ[0])))
    rx_close = rec_spec[j_close][0]
    offset_close = abs(rx_close - SRC_SPEC_XZ[0])
    # P arrival ≈ tshift + offset/vp (SPECFEM internally shifts but the
    # response peak gives us the empirical offset). Use this to fit shift.
    peak_t_spec = t_spec_file[np.argmax(np.abs(spec_vz[j_close]))]
    expected_t  = TSHIFT + offset_close / VP
    spec_shift  = expected_t - peak_t_spec
    print(f"SPECFEM nearest receiver x={rx_close} m, peak at file-t = "
          f"{peak_t_spec:.4f} s; expected ≈ {expected_t:.4f} s. "
          f"Aligning shift = {spec_shift * 1e3:+.1f} ms")
    t_spec_phys = t_spec_file + spec_shift

    # ----- SEM run -----
    mesh = build_quad4_mesh(nex=NEX, nez=NEZ, x_max=X_MAX, z_max=Z_MAX,
                            n_order=N_ORDER, dtype=torch.float64)
    print(f"SEM mesh: E={mesh.n_elements}, ngll={mesh.ngll}, "
          f"n_global={mesh.n_global}, dh_e = {mesh.dx_e:.2f} m")

    vp_e  = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP,  dtype=torch.float64)
    vs_e  = torch.full_like(vp_e, VS)
    rho_e = torch.full_like(vp_e, RHO)
    eq = ElasticSEM2D.build(mesh, vp_e, vs_e, rho_e)

    src_sem_xz = to_sem(SRC_SPEC_XZ)
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[src_sem_xz], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"],
        source_payloads=[SRC_MT],
    )
    t_sem = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_sem, F0, TSHIFT))
    # Record SEM more densely than SPECFEM dump cadence (every 5 steps)
    # so we can centred-difference into velocity to match SPECFEM imagetype.
    sem_seis, sem_snaps = prop.forward(src_wave, record_field_every=5)
    sem_ux   = sem_seis[..., 0].T.numpy()
    sem_uz   = sem_seis[..., 1].T.numpy()
    sem_snap_ux = sem_snaps[:, 0].numpy()
    sem_snap_uz = sem_snaps[:, 1].numpy()
    sem_snap_step = np.arange(0, nt, 5)
    # Centred-difference displacement snapshots to velocity (sample stride
    # is 5 · DT, so ∂u/∂t ≈ (snap[i+1] − snap[i−1]) / (10 · DT))
    sem_snap_vx = np.zeros_like(sem_snap_ux)
    sem_snap_vz = np.zeros_like(sem_snap_uz)
    sem_snap_vx[1:-1] = (sem_snap_ux[2:] - sem_snap_ux[:-2]) / (2 * 5 * DT)
    sem_snap_vz[1:-1] = (sem_snap_uz[2:] - sem_snap_uz[:-2]) / (2 * 5 * DT)

    # ----- Numerically differentiate displacement → velocity -----
    sem_vx = np.gradient(sem_ux, DT, axis=1)
    sem_vz = np.gradient(sem_uz, DT, axis=1)

    # ----- 2-component gather comparison -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", sharey="row")
    rec_x_arr = np.array([rx for rx, _ in rec_spec])

    def panel(ax, t_axis, seis, title, vmax_frac=0.2):
        m = np.abs(seis).max()
        vm = vmax_frac * max(m, 1e-30)
        im = ax.imshow(
            seis, aspect="auto", cmap="seismic", vmin=-vm, vmax=vm,
            extent=[t_axis[0], t_axis[-1], rec_x_arr[-1], rec_x_arr[0]],
        )
        ax.set_title(title, fontsize=10)
        return im

    panel(axes[0, 0], t_sem,        sem_vx, "SEM  v_x  (∂u_x/∂t)")
    panel(axes[0, 1], t_spec_phys,  spec_vx, "SPECFEM2D  v_x")
    panel(axes[1, 0], t_sem,        sem_vz, "SEM  v_z  (∂u_z/∂t)")
    panel(axes[1, 1], t_spec_phys,  spec_vz, "SPECFEM2D  v_z")
    for ax in axes[:, 0]:
        ax.set_ylabel("receiver x (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("t (s)")
    fig.tight_layout()
    gp = fig_dir / "elastic_flat_gather_2way.png"
    fig.savefig(gp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {gp.resolve()}")

    # ----- Trace overlay at 3 receivers -----
    pick_offsets = [60.0, 300.0, 580.0]
    pick_x = [SRC_SPEC_XZ[0] - off for off in pick_offsets]
    pick_idx = [int(np.argmin(np.abs(rec_x_arr - x))) for x in pick_x]

    def norm(a):
        return a / max(np.max(np.abs(a)), 1e-30)

    fig, axes = plt.subplots(len(pick_idx), 2,
                             figsize=(14, 2.6 * len(pick_idx)),
                             sharex=True)
    for row, j in enumerate(pick_idx):
        rx = rec_x_arr[j]; offset = abs(rx - SRC_SPEC_XZ[0])
        axes[row, 0].plot(t_sem, norm(sem_vx[j]),  "C0",  lw=1.4, label="SEM v_x")
        axes[row, 0].plot(t_spec_phys, norm(spec_vx[j]), "C3", lw=1.0,
                          label="SPECFEM v_x")
        axes[row, 0].set_title(f"v_x at x={rx:.0f} m (offset {offset:.0f} m)",
                               fontsize=10)
        axes[row, 1].plot(t_sem, norm(sem_vz[j]),  "C0",  lw=1.4, label="SEM v_z")
        axes[row, 1].plot(t_spec_phys, norm(spec_vz[j]), "C3", lw=1.0,
                          label="SPECFEM v_z")
        axes[row, 1].set_title(f"v_z at x={rx:.0f} m (offset {offset:.0f} m)",
                               fontsize=10)
        for c in (0, 1):
            axes[row, c].grid(alpha=0.3); axes[row, c].legend(fontsize=8)
    axes[-1, 0].set_xlabel("t (s)"); axes[-1, 1].set_xlabel("t (s)")
    fig.tight_layout()
    tp = fig_dir / "elastic_flat_traces_2way.png"
    fig.savefig(tp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tp.resolve()}")

    # ----- Wavefield snapshot comparison -----
    coords_spec, snaps_spec, spec_steps = load_specfem_wavefield_dumps(specfem_out)
    print(f"SPECFEM dumps: {len(spec_steps)} snapshots, {snaps_spec.shape[1]} pts each")
    # Convert SPECFEM z-up to SEM z-down
    coords_sem_grid = np.stack(
        [coords_spec[:, 0], Z_MAX - coords_spec[:, 1]], axis=-1
    )
    # Physical time of SPECFEM dump step k:
    #   file_time(k)  = (k − 1) · DT + t_spec_file[0]      (t_spec_file[0]<0)
    #   phys_time     = file_time(k)  +  spec_shift
    phys_t_spec = np.array([(k - 1) * DT + t_spec_file[0] + spec_shift
                            for k in spec_steps])

    # Pick a handful of times where both backends have a snapshot.
    targets_t = np.array([0.10, 0.18, 0.26, 0.34, 0.42])
    nx_g = NEX * N_ORDER + 1
    nz_g = NEZ * N_ORDER + 1
    sem_x_axis = mesh.x_global.reshape(nz_g, nx_g)[0, :].numpy()
    sem_z_axis = mesh.z_global.reshape(nz_g, nx_g)[:, 0].numpy()
    from scipy.interpolate import RegularGridInterpolator, griddata

    def sem_snap_to_image(snap_flat, nx_img=601, nz_img=421):
        """Interpolate SEM GLL grid (non-uniform tensor product) to fine regular."""
        snap2d = snap_flat.reshape(nz_g, nx_g)
        xi = np.linspace(0, X_MAX, nx_img)
        zi = np.linspace(0, Z_MAX, nz_img)
        f = RegularGridInterpolator((sem_z_axis, sem_x_axis), snap2d,
                                    bounds_error=False, fill_value=0.0)
        XI, ZI = np.meshgrid(xi, zi)
        return f((ZI, XI)), xi, zi

    def spec_snap_to_image(snap_vec, nx_img=601, nz_img=421):
        """Scattered SPECFEM GLL → regular grid via linear griddata."""
        xi = np.linspace(0, X_MAX, nx_img)
        zi = np.linspace(0, Z_MAX, nz_img)
        XI, ZI = np.meshgrid(xi, zi)
        img = griddata(
            (coords_sem_grid[:, 0], coords_sem_grid[:, 1]),
            snap_vec, (XI, ZI), method="linear", fill_value=0.0,
        )
        return img, xi, zi

    nrows = len(targets_t)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 3.0 * nrows),
                             squeeze=False)
    for r, tt in enumerate(targets_t):
        j_sem  = int(np.argmin(np.abs(sem_snap_step * DT - tt)))
        j_spec = int(np.argmin(np.abs(phys_t_spec - tt)))
        spec_t_picked = phys_t_spec[j_spec]
        sem_t_picked  = sem_snap_step[j_sem] * DT

        sem_img_vx, xi, zi = sem_snap_to_image(sem_snap_vx[j_sem])
        sem_img_vz, _, _   = sem_snap_to_image(sem_snap_vz[j_sem])
        spec_img_vx, _, _  = spec_snap_to_image(snaps_spec[j_spec, :, 0])
        spec_img_vz, _, _  = spec_snap_to_image(snaps_spec[j_spec, :, 1])

        # Normalize each PANEL by its own 99-percentile so absolute
        # amplitude differences don't wash the field out.
        def panel(ax, img, label):
            vm = max(np.percentile(np.abs(img), 99.0), 1e-30)
            ax.imshow(img / vm,
                      extent=[0, X_MAX, Z_MAX, 0],
                      cmap="seismic", vmin=-1.0, vmax=1.0, aspect="equal")
            ax.set_title(label, fontsize=9)
            ax.plot(*to_sem(SRC_SPEC_XZ), "ko", ms=3)
            ax.tick_params(labelsize=7)

        panel(axes[r, 0], sem_img_vx,  f"SEM v_x  (t={sem_t_picked:.3f}s)")
        panel(axes[r, 1], spec_img_vx, f"SPECFEM v_x (t={spec_t_picked:.3f}s)")
        panel(axes[r, 2], sem_img_vz,  f"SEM v_z")
        panel(axes[r, 3], spec_img_vz, f"SPECFEM v_z")
        axes[r, 0].set_ylabel(f"target {tt:.2f}s", fontsize=8)

    fig.tight_layout()
    wp = fig_dir / "elastic_flat_wavefield_2way.png"
    fig.savefig(wp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {wp.resolve()}")


if __name__ == "__main__":
    main()

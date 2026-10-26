"""Compare SEM ElasticSEM2DDeformed against SPECFEM2D hill_case.

Setup (mirrors ``_specfem_compare/hill_case``):

  * Domain 1270 × 890 m, 64 × 44 deformed quad elements (NGNOD=9).
  * Gaussian hill: 50 m peak at centre, σ = 160 m.
  * vp = 2200, vs = 1100, ρ = 2000 (uniform).
  * Moment-tensor source ``M_xx = M_zz = 1`` at (640, 860) in SPECFEM
    coords; Ricker f0 = 12 Hz, tshift = 0.12 s. DT = 1 ms, NSTEP = 700.
  * Receivers along the curved surface (1 cell below local z_top).
  * SPECFEM outputs velocity (``seismotype = 2``).

We numerically differentiate our displacement output to compare to
SPECFEM's velocity record.
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


# Physical / discretization constants (frozen to match SPECFEM hill_case)
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ = 64, 44
N_ORDER = 4
DT = 1e-3
NSTEP = 700
F0, TSHIFT = 12.0, 0.12
VP, VS, RHO = 2200.0, 1100.0, 2000.0

# Source: SPECFEM coords (xs, zs) → SEM coords (x, z_max − z_specfem)
SRC_SPEC_XZ = (640.0, 860.0)
SRC_MT      = (1.0, 1.0, 0.0)

# Gaussian-hill parameters (matches build_inputs.py in hill_case)
HILL_BASE_DEPTH = 50.0          # surface sits 50 m below SEM top at flat sides
HILL_H0         = 50.0          # 50 m peak height (above flat baseline)
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0


def z_top_fn(x: torch.Tensor) -> torch.Tensor:
    """SEM-coord (z-down) topography z_top(x).

    z_top = HILL_BASE_DEPTH at flat sides, → 0 at peak (peak is highest, so
    smallest SEM z).
    """
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz_spec):
    return (xz_spec[0], Z_BOTTOM - xz_spec[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def load_specfem_seismograms(out_dir: Path, n_rec: int):
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
    specfem_out = specfem_dir / "OUTPUT_FILES"
    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- receivers from STATIONS -----
    stations_lines = (specfem_dir / "DATA" / "STATIONS").read_text().splitlines()
    rec_spec = [(float(p.split()[2]), float(p.split()[3])) for p in stations_lines]
    rec_sem  = [to_sem(rx) for rx in rec_spec]
    n_rec = len(rec_sem)
    print(f"{n_rec} receivers, x ∈ [{rec_spec[0][0]:.0f}, {rec_spec[-1][0]:.0f}] m")

    # ----- SPECFEM seismograms -----
    spec_vx, spec_vz, t_spec_file = load_specfem_seismograms(specfem_out, n_rec)
    print(f"SPECFEM gather: vx {spec_vx.shape}, vz {spec_vz.shape}")
    # Time alignment via near-source receiver (closest to src in x).
    j_close = int(np.argmin(np.abs(np.array([rx for rx, _ in rec_spec])
                                   - SRC_SPEC_XZ[0])))
    rx_close = rec_spec[j_close][0]
    offset_close = abs(rx_close - SRC_SPEC_XZ[0])
    peak_t_spec = t_spec_file[np.argmax(np.abs(spec_vz[j_close]))]
    expected_t  = TSHIFT + offset_close / VP
    spec_shift  = expected_t - peak_t_spec
    print(f"SPECFEM peak at x={rx_close:.0f}: file-t={peak_t_spec:.4f}, "
          f"expected={expected_t:.4f}; shift={spec_shift*1e3:+.1f} ms")
    t_spec_phys = t_spec_file + spec_shift

    # ----- SEM run -----
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    print(f"SEM mesh: E={mesh.n_elements}, n_global={mesh.n_global}, "
          f"surface z ∈ [{mesh.z_e[:NEX, 0, :].min().item():.1f}, "
          f"{mesh.z_e[:NEX, 0, :].max().item():.1f}] m")
    vp_e  = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP,  dtype=torch.float64)
    vs_e  = torch.full_like(vp_e, VS)
    rho_e = torch.full_like(vp_e, RHO)
    eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e)

    src_sem_xz = to_sem(SRC_SPEC_XZ)
    print(f"Source SEM = {src_sem_xz} (SPECFEM = {SRC_SPEC_XZ})")
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[src_sem_xz], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"],
        source_payloads=[SRC_MT],
    )
    t_sem = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_sem, F0, TSHIFT))
    sem_out = prop.forward(src_wave)
    sem_ux = sem_out[..., 0].T.numpy()
    sem_uz = sem_out[..., 1].T.numpy()
    sem_vx = np.gradient(sem_ux, DT, axis=1)
    sem_vz = np.gradient(sem_uz, DT, axis=1)

    # ----- 2-way gather -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", sharey="row")
    rec_x_arr = np.array([rx for rx, _ in rec_spec])

    def panel(ax, t_axis, seis, title, vmax_frac=0.2):
        m = np.abs(seis).max()
        vm = vmax_frac * max(m, 1e-30)
        ax.imshow(seis, aspect="auto", cmap="seismic", vmin=-vm, vmax=vm,
                  extent=[t_axis[0], t_axis[-1],
                          rec_x_arr[-1], rec_x_arr[0]])
        ax.set_title(title, fontsize=10)

    panel(axes[0, 0], t_sem,       sem_vx, "SEM v_x (Q9 hill)")
    panel(axes[0, 1], t_spec_phys, spec_vx, "SPECFEM2D v_x (hill_case)")
    panel(axes[1, 0], t_sem,       sem_vz, "SEM v_z (Q9 hill)")
    panel(axes[1, 1], t_spec_phys, spec_vz, "SPECFEM2D v_z (hill_case)")
    for ax in axes[:, 0]:
        ax.set_ylabel("receiver x (m)")
    for ax in axes[-1, :]:
        ax.set_xlabel("t (s)")
    fig.tight_layout()
    gp = fig_dir / "hill_gather_2way.png"
    fig.savefig(gp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {gp.resolve()}")

    # ----- trace overlay at 3 receivers -----
    pick_offsets = [60.0, 200.0, 500.0]
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
        axes[row, 0].set_title(f"v_x at x={rx:.0f} (offset {offset:.0f} m)",
                               fontsize=10)
        axes[row, 1].plot(t_sem, norm(sem_vz[j]),  "C0",  lw=1.4, label="SEM v_z")
        axes[row, 1].plot(t_spec_phys, norm(spec_vz[j]), "C3", lw=1.0,
                          label="SPECFEM v_z")
        axes[row, 1].set_title(f"v_z at x={rx:.0f} (offset {offset:.0f} m)",
                               fontsize=10)
        for c in (0, 1):
            axes[row, c].grid(alpha=0.3); axes[row, c].legend(fontsize=8)
    axes[-1, 0].set_xlabel("t (s)"); axes[-1, 1].set_xlabel("t (s)")
    fig.tight_layout()
    tp = fig_dir / "hill_traces_2way.png"
    fig.savefig(tp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tp.resolve()}")


if __name__ == "__main__":
    main()

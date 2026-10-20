"""Long-nt boundary-reflection comparison on the Gaussian-hill case:
SEM Q9 (Stacey) vs SPECFEM2D (Stacey), NSTEP = 2500.

vp = 2200 m/s, deepest point ~860 m below source. Bottom round-trip
≈ 0.78 s; in 2.5 s we see three full bottom bounces. If our Stacey
reflects with the same amplitude and timing as SPECFEM's Stacey, that
validates the SEM implementation and quantifies how much energy leaks
back into the interior over realistic FWI-length simulations.
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


X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ = 64, 44
N_ORDER = 4
DT = 1e-3
NSTEP = 2500
F0, TSHIFT = 12.0, 0.12
VP, VS, RHO = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ = (640.0, 860.0)
SRC_MT      = (1.0, 1.0, 0.0)

HILL_BASE_DEPTH = 50.0
HILL_H0         = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0


def z_top_fn(x: torch.Tensor) -> torch.Tensor:
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz):
    return (xz[0], Z_BOTTOM - xz[1])


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
        default="_smoketest/specfem_hill_longnt")
    parser.add_argument(
        "--out_dir",
        default="_smoketest")
    args = parser.parse_args()

    specfem_dir = Path(args.specfem_dir)
    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    stations_lines = (specfem_dir / "DATA" / "STATIONS").read_text().splitlines()
    rec_spec = [(float(p.split()[2]), float(p.split()[3])) for p in stations_lines]
    rec_sem  = [to_sem(rx) for rx in rec_spec]
    n_rec = len(rec_sem)
    rec_x_arr = np.array([rx for rx, _ in rec_spec])

    spec_vx, spec_vz, t_spec_file = load_specfem_seismograms(
        specfem_dir / "OUTPUT_FILES", n_rec
    )
    # Time alignment: SPECFEM peak at source-co-located receiver →
    # expected_t = tshift + offset/vp ; shift moves SPECFEM file-time
    # axis to SEM physical-time axis.
    j_close = int(np.argmin(np.abs(rec_x_arr - SRC_SPEC_XZ[0])))
    peak_t_spec = t_spec_file[np.argmax(np.abs(spec_vz[j_close]))]
    expected_t = TSHIFT + abs(rec_x_arr[j_close] - SRC_SPEC_XZ[0]) / VP
    spec_shift = expected_t - peak_t_spec
    print(f"SPECFEM shift = {spec_shift*1e3:+.1f} ms; "
          f"trace length {len(t_spec_file)} samples")
    t_spec_phys = t_spec_file + spec_shift

    # ----- SEM Q9 forward (long nt) -----
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    vp_e = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP,
                     dtype=torch.float64)
    vs_e = torch.full_like(vp_e, VS); rho_e = torch.full_like(vp_e, RHO)
    eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e)
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[to_sem(SRC_SPEC_XZ)], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t_sem = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_sem, F0, TSHIFT))
    sem_out = prop.forward(src_wave)
    sem_ux = sem_out[..., 0].T.numpy()
    sem_uz = sem_out[..., 1].T.numpy()
    # Velocity (∂u/∂t) — match SPECFEM seismotype=2
    sem_vx = np.gradient(sem_ux, DT, axis=1)
    # Flip SEM v_z to match SPECFEM convention (z-up vs z-down)
    sem_vz = -np.gradient(sem_uz, DT, axis=1)
    print(f"SEM gather: {sem_vx.shape}")

    def gather_panel(ax, t_axis, seis, title, vmax=0.15, per_trace=True):
        s = seis.copy()
        if per_trace:
            peaks = np.maximum(np.abs(s).max(axis=1, keepdims=True), 1e-30)
            s = s / peaks
        else:
            s = s / max(np.max(np.abs(s)), 1e-30)
        ax.imshow(s, aspect="auto", cmap="seismic",
                  vmin=-vmax, vmax=vmax,
                  extent=[t_axis[0], t_axis[-1],
                          rec_x_arr[-1], rec_x_arr[0]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("receiver x (m)")
        ax.tick_params(labelsize=8)

    # ----- Full gather (per-trace normalize so reflections are visible) -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex="col", sharey="row")
    gather_panel(axes[0, 0], t_sem,       sem_vx,  "SEM Q9 v_x (Stacey)",  vmax=0.15)
    gather_panel(axes[0, 1], t_spec_phys, spec_vx, "SPECFEM2D v_x (Stacey)", vmax=0.15)
    gather_panel(axes[1, 0], t_sem,       sem_vz,  "SEM Q9 −v_z (z-flip to match)", vmax=0.15)
    gather_panel(axes[1, 1], t_spec_phys, spec_vz, "SPECFEM2D v_z", vmax=0.15)
    fig.suptitle("Long-nt boundary-reflection comparison (per-trace norm, clip ±15%)",
                 fontsize=11)
    fig.tight_layout()
    g1 = fig_dir / "hill_longnt_gather_full.png"
    fig.savefig(g1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g1.resolve()}")

    # ----- Late-window gather (t > 0.6 s, global norm, tight clip) -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex="col", sharey="row")
    gather_panel(axes[0, 0], t_sem,       sem_vx,  "SEM Q9 v_x — late window",
                 vmax=0.03, per_trace=False)
    gather_panel(axes[0, 1], t_spec_phys, spec_vx, "SPECFEM2D v_x — late",
                 vmax=0.03, per_trace=False)
    gather_panel(axes[1, 0], t_sem,       sem_vz,  "SEM Q9 −v_z — late",
                 vmax=0.03, per_trace=False)
    gather_panel(axes[1, 1], t_spec_phys, spec_vz, "SPECFEM2D v_z — late",
                 vmax=0.03, per_trace=False)
    for ax in axes.flat:
        ax.set_xlim(0.6, t_sem[-1])
    fig.suptitle("Late-window: each panel normalized by its own global peak; "
                 "anything visible here is Stacey leak",
                 fontsize=11)
    fig.tight_layout()
    g2 = fig_dir / "hill_longnt_gather_late.png"
    fig.savefig(g2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g2.resolve()}")

    # ----- Quantitative leak per offset -----
    print()
    print(f"{'offset':>7s}  {'SEM peak':>10s}  {'SEM late%':>10s}  "
          f"{'SPECFEM peak':>14s}  {'SPECFEM late%':>14s}")
    late_start = int(0.6 / DT)
    late_spec_mask = t_spec_phys > 0.6
    for j in range(0, n_rec, 5):
        rx = rec_x_arr[j]; offset = abs(rx - SRC_SPEC_XZ[0])
        sem_peak = np.max(np.abs(sem_vz[j]))
        sem_late = np.max(np.abs(sem_vz[j, late_start:]))
        sem_pct  = sem_late / max(sem_peak, 1e-30)
        spec_peak = np.max(np.abs(spec_vz[j]))
        spec_late = np.max(np.abs(spec_vz[j][late_spec_mask]))
        spec_pct  = spec_late / max(spec_peak, 1e-30)
        print(f"{offset:7.0f}  {sem_peak:10.3e}  {sem_pct:9.2%}  "
              f"{spec_peak:14.3e}  {spec_pct:13.2%}")


if __name__ == "__main__":
    main()

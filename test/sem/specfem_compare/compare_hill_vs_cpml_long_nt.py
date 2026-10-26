"""Long-nt CPML stability: 1st-order velocity-stress on hill_case.

Demonstrates that the 1st-order velocity-stress SEM with FD-style
single-stage CPML stays stable over long simulations (NSTEP=2000+),
where the 2nd-order displacement form's SPECFEM-style CPML develops
the documented elastic late-time instability around t ~ 1 s.

Three runs on the same hill_case geometry (Gaussian hill, NSTEP=2000):

  1. SEM 2nd-order displacement + Stacey         — reference (stable, partial absorption)
  2. SEM 2nd-order displacement + SPECFEM CPML   — known late-time instability
  3. SEM 1st-order velocity-stress + simple CPML — should stay stable

Saves seismic gathers, plus a quantitative 'late-window energy ratio'
table at receivers far from / near the boundary.
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
    ElasticVSSEM2DDeformed,
    build_elastic_cpml,
    build_simple_elastic_cpml,
    build_topography_quad9,
)


# Match compare_hill_sem_cpml_vs_stacey.py so geometry/source are identical.
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ        = 64, 44
N_ORDER         = 4
DT              = 1e-3
NSTEP           = 2000          # long enough to expose CPML failure modes
F0, TSHIFT      = 12.0, 0.12
VP, VS, RHO     = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ     = (640.0, 860.0)
SRC_MT          = (1.0, 1.0, 0.0)

HILL_BASE_DEPTH = 50.0
HILL_H0         = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0


def z_top_fn(x):
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz):
    return (xz[0], Z_BOTTOM - xz[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _build_mesh_and_models():
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp_e  = torch.full(e, VP,  dtype=torch.float64)
    vs_e  = torch.full(e, VS,  dtype=torch.float64)
    rho_e = torch.full(e, RHO, dtype=torch.float64)
    return mesh, vp_e, vs_e, rho_e


def run_2nd_stacey(mesh, vp_e, vs_e, rho_e, stations):
    eq = ElasticSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e,
    )
    rec_sem = [to_sem(rx) for rx in stations]
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[to_sem(SRC_SPEC_XZ)], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    # 2nd-order outputs displacement; differentiate for velocity comparison.
    vx = np.gradient(out[..., 0].T.numpy(), DT, axis=1)
    vz = -np.gradient(out[..., 1].T.numpy(), DT, axis=1)
    return vx, vz


def run_2nd_specfem_cpml(mesh, vp_e, vs_e, rho_e, stations):
    cpml = build_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "bottom"),
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    eq = ElasticSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, cpml=cpml,
    )
    rec_sem = [to_sem(rx) for rx in stations]
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[to_sem(SRC_SPEC_XZ)], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    vx = np.gradient(out[..., 0].T.numpy(), DT, axis=1)
    vz = -np.gradient(out[..., 1].T.numpy(), DT, axis=1)
    return vx, vz


def run_1st_simple_cpml(mesh, vp_e, vs_e, rho_e, stations):
    cpml = build_simple_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "bottom"),
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    eq = ElasticVSSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, free_top=False, cpml=cpml,
    )
    rec_sem = [to_sem(rx) for rx in stations]
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[to_sem(SRC_SPEC_XZ)], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    # 1st-order outputs velocity directly; flip v_z sign to match the 2nd-order
    # convention used in the other compare scripts (sweep z-down vs SEM gradient).
    return out[..., 0].T.numpy(), -out[..., 1].T.numpy()


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

    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    stations_lines = (Path(args.specfem_dir) / "DATA" / "STATIONS").read_text().splitlines()
    stations = [(float(p.split()[2]), float(p.split()[3])) for p in stations_lines]
    rec_x_arr = np.array([rx for rx, _ in stations])
    t_sem = np.arange(NSTEP) * DT

    mesh, vp_e, vs_e, rho_e = _build_mesh_and_models()

    print(f"NSTEP={NSTEP}, total time = {NSTEP*DT:.2f} s")
    print(f"Running 2nd-order + Stacey…")
    s_vx, s_vz = run_2nd_stacey(mesh, vp_e, vs_e, rho_e, stations)
    print(f"  v_z peak = {np.abs(s_vz).max():.3e}")

    print(f"Running 2nd-order + SPECFEM CPML…")
    c2_vx, c2_vz = run_2nd_specfem_cpml(mesh, vp_e, vs_e, rho_e, stations)
    print(f"  v_z peak = {np.abs(c2_vz).max():.3e}")

    print(f"Running 1st-order + simple CPML…")
    c1_vx, c1_vz = run_1st_simple_cpml(mesh, vp_e, vs_e, rho_e, stations)
    print(f"  v_z peak = {np.abs(c1_vz).max():.3e}")

    # ----- gather plot (per-trace normalised) ------------------------------ #
    def panel(ax, t_axis, seis, title, vmax=0.15, per_trace=True):
        s = seis.copy()
        if per_trace:
            peaks = np.maximum(np.abs(s).max(axis=1, keepdims=True), 1e-30)
            s = s / peaks
        else:
            s = s / max(np.abs(s).max(), 1e-30)
        ax.imshow(s, aspect="auto", cmap="seismic",
                  vmin=-vmax, vmax=vmax,
                  extent=[t_axis[0], t_axis[-1], rec_x_arr[-1], rec_x_arr[0]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel("receiver x (m)")
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10),
                             sharex="col", sharey="row")
    panel(axes[0, 0], t_sem, s_vx,  "2nd-order Stacey v_x", vmax=0.15)
    panel(axes[0, 1], t_sem, s_vz,  "2nd-order Stacey −v_z", vmax=0.15)
    panel(axes[1, 0], t_sem, c2_vx, "2nd-order CPML (SPECFEM-style) v_x", vmax=0.15)
    panel(axes[1, 1], t_sem, c2_vz, "2nd-order CPML (SPECFEM-style) −v_z", vmax=0.15)
    panel(axes[2, 0], t_sem, c1_vx, "1st-order CPML (FD-style) v_x", vmax=0.15)
    panel(axes[2, 1], t_sem, c1_vz, "1st-order CPML (FD-style) −v_z", vmax=0.15)
    fig.suptitle(f"hill_case, NSTEP={NSTEP}: three boundary treatments",
                 fontsize=11)
    fig.tight_layout()
    g1 = fig_dir / "hill_three_bc_long_nt.png"
    fig.savefig(g1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g1.resolve()}")

    # ----- late-window absolute amplitude ---------------------------------- #
    fig, axes = plt.subplots(3, 2, figsize=(15, 10),
                             sharex="col", sharey="row")
    panel(axes[0, 0], t_sem, s_vx,  "2nd-order Stacey v_x late",
          vmax=0.03, per_trace=False)
    panel(axes[0, 1], t_sem, s_vz,  "2nd-order Stacey −v_z late",
          vmax=0.03, per_trace=False)
    panel(axes[1, 0], t_sem, c2_vx, "2nd-order CPML v_x late",
          vmax=0.03, per_trace=False)
    panel(axes[1, 1], t_sem, c2_vz, "2nd-order CPML −v_z late",
          vmax=0.03, per_trace=False)
    panel(axes[2, 0], t_sem, c1_vx, "1st-order CPML v_x late",
          vmax=0.03, per_trace=False)
    panel(axes[2, 1], t_sem, c1_vz, "1st-order CPML −v_z late",
          vmax=0.03, per_trace=False)
    t_late = 0.6
    for ax in axes.flat:
        ax.set_xlim(t_late, t_sem[-1])
    fig.tight_layout()
    g2 = fig_dir / "hill_three_bc_long_nt_late.png"
    fig.savefig(g2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g2.resolve()}")

    # ----- quantitative late-window energy --------------------------------- #
    late_start = int(t_late / DT)
    print()
    print(f"Late-window v_z amplitude (t > {t_late} s) as fraction of peak:")
    print(f"{'offset':>7s}  {'Stacey':>10s}  "
          f"{'2nd CPML':>12s}  {'1st CPML':>12s}")
    for j in range(0, len(stations), 5):
        rx = stations[j][0]
        off = abs(rx - SRC_SPEC_XZ[0])
        s_pct  = (np.abs(s_vz[j,  late_start:]).max()
                  / max(np.abs(s_vz[j]).max(),  1e-30))
        c2_pct = (np.abs(c2_vz[j, late_start:]).max()
                  / max(np.abs(c2_vz[j]).max(), 1e-30))
        c1_pct = (np.abs(c1_vz[j, late_start:]).max()
                  / max(np.abs(c1_vz[j]).max(), 1e-30))
        print(f"{off:7.0f}  {s_pct:10.2%}  {c2_pct:12.2%}  {c1_pct:12.2%}")

    # ----- absolute peak across full simulation ---------------------------- #
    print()
    print(f"Stability check — peak |v_z| at three time windows:")
    print(f"{'window':>15s}  {'Stacey':>12s}  {'2nd CPML':>12s}  {'1st CPML':>12s}")
    for t0, t1, label in [(0.0, 0.5, "  early [0,0.5]"),
                          (0.5, 1.0, "  mid   [0.5,1]"),
                          (1.0, NSTEP*DT, " late   [1,end]")]:
        i0, i1 = int(t0/DT), int(t1/DT)
        s_peak  = np.abs(s_vz[:,  i0:i1]).max()
        c2_peak = np.abs(c2_vz[:, i0:i1]).max()
        c1_peak = np.abs(c1_vz[:, i0:i1]).max()
        print(f"{label:>15s}  {s_peak:12.3e}  {c2_peak:12.3e}  "
              f"{c1_peak:12.3e}")


if __name__ == "__main__":
    main()

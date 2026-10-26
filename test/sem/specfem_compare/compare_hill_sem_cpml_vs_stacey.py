"""SEM elastic Stacey vs SEM elastic CPML on the hill_case, NSTEP = 700.

Shows that the new SPECFEM-style elastic CPML (proper algorithm at the
displacement-derivative level) attenuates boundary reflections better
than 1st-order Stacey, on the original hill_case setup that matches
SPECFEM2D.
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
    build_elastic_cpml,
    build_topography_quad9,
)

X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ = 64, 44
N_ORDER = 4
DT = 1e-3
NSTEP = 700
F0, TSHIFT = 12.0, 0.12
VP, VS, RHO = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ = (640.0, 860.0)
SRC_MT = (1.0, 1.0, 0.0)

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
    return (1 - 2 * a) * np.exp(-a)


def run(absorb_mode: str, stations):
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    vp_e = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), VP, dtype=torch.float64)
    vs_e = torch.full_like(vp_e, VS); rho_e = torch.full_like(vp_e, RHO)
    if absorb_mode == "stacey":
        eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e)
    elif absorb_mode == "cpml":
        cpml = build_elastic_cpml(
            mesh, dt=DT, sides=("left", "right", "bottom"),
            pml_thickness_elements=8, c_max=VP, f_dominant=F0,
        )
        eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e,
                                        cpml=cpml)
    else:
        raise ValueError(absorb_mode)
    rec_sem = [to_sem(rx) for rx in stations]
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[to_sem(SRC_SPEC_XZ)], receiver_xz=rec_sem,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    return (
        np.gradient(out[..., 0].T.numpy(), DT, axis=1),
       -np.gradient(out[..., 1].T.numpy(), DT, axis=1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specfem_dir",
        default="_specfem_compare/hill_case")
    parser.add_argument("--out_dir",
        default="_smoketest")
    args = parser.parse_args()

    fig_dir = Path(args.out_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    stations_lines = (Path(args.specfem_dir) / "DATA" / "STATIONS").read_text().splitlines()
    stations = [(float(p.split()[2]), float(p.split()[3])) for p in stations_lines]
    rec_x_arr = np.array([rx for rx, _ in stations])
    t_sem = np.arange(NSTEP) * DT

    print("Running SEM Stacey…")
    s_vx, s_vz = run("stacey", stations)
    print("Running SEM CPML (new SPECFEM-style)…")
    c_vx, c_vz = run("cpml", stations)
    print(f"Stacey v_z peak = {np.abs(s_vz).max():.3e}")
    print(f"CPML   v_z peak = {np.abs(c_vz).max():.3e}")

    def panel(ax, t_axis, seis, title, vmax=0.15, per_trace=True):
        s = seis.copy()
        if per_trace:
            peaks = np.maximum(np.abs(s).max(axis=1, keepdims=True), 1e-30)
            s = s / peaks
        else:
            s = s / max(np.abs(s).max(), 1e-30)
        ax.imshow(s, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
                  extent=[t_axis[0], t_axis[-1], rec_x_arr[-1], rec_x_arr[0]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel("receiver x (m)")
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex="col", sharey="row")
    panel(axes[0, 0], t_sem, s_vx, "SEM Stacey v_x", vmax=0.15)
    panel(axes[0, 1], t_sem, c_vx, "SEM CPML v_x (SPECFEM-style)", vmax=0.15)
    panel(axes[1, 0], t_sem, s_vz, "SEM Stacey −v_z", vmax=0.15)
    panel(axes[1, 1], t_sem, c_vz, "SEM CPML −v_z", vmax=0.15)
    fig.suptitle("hill_case, NSTEP=700: SEM Stacey vs SEM SPECFEM-style CPML",
                 fontsize=11)
    fig.tight_layout()
    g1 = fig_dir / "hill_sem_stacey_vs_cpml_proper.png"
    fig.savefig(g1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g1.resolve()}")

    # Late-window (t > 0.4s) absolute amplitude
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex="col", sharey="row")
    panel(axes[0, 0], t_sem, s_vx, "Stacey v_x late — global norm",
          vmax=0.03, per_trace=False)
    panel(axes[0, 1], t_sem, c_vx, "CPML v_x late",
          vmax=0.03, per_trace=False)
    panel(axes[1, 0], t_sem, s_vz, "Stacey −v_z late",
          vmax=0.03, per_trace=False)
    panel(axes[1, 1], t_sem, c_vz, "CPML −v_z late",
          vmax=0.03, per_trace=False)
    for ax in axes.flat:
        ax.set_xlim(0.4, t_sem[-1])
    fig.tight_layout()
    g2 = fig_dir / "hill_sem_stacey_vs_cpml_proper_late.png"
    fig.savefig(g2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g2.resolve()}")

    # Quantitative metric
    late_start = int(0.4 / DT)
    print()
    print(f"{'offset':>7s}  {'Stacey late%':>14s}  {'CPML late%':>12s}")
    for j in range(0, len(stations), 5):
        rx = stations[j][0]; off = abs(rx - SRC_SPEC_XZ[0])
        s_pct = np.abs(s_vz[j, late_start:]).max() / max(np.abs(s_vz[j]).max(), 1e-30)
        c_pct = np.abs(c_vz[j, late_start:]).max() / max(np.abs(c_vz[j]).max(), 1e-30)
        print(f"{off:7.0f}  {s_pct:14.2%}  {c_pct:12.2%}")


if __name__ == "__main__":
    main()

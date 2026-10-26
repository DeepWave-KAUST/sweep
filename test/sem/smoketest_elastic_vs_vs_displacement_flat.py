"""Self-consistency: 1st-order velocity-stress vs 2nd-order displacement.

Both solve the same elastic PDE; they should agree to spectral accuracy
in the interior (no PML, no free surface). The 2nd-order output is
displacement ``u(t)``; we differentiate by ``dt`` to recover ``v(t)``
and compare against the 1st-order direct velocity output.

Run::

    PYTHONPATH=src python test/sem/smoketest_elastic_vs_vs_displacement_flat.py

Saves the comparison plot to ``_smoketest/vs_vs_disp_flat.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import (
    ElasticSEM2DDeformed,
    SEMPropagator,
    ElasticVSSEM2DDeformed,
    build_axis_aligned_quad9,
)


# ----- domain & physics --------------------------------------------------- #
X_MAX, Z_MAX = 1270.0, 890.0
NEX, NEZ     = 32, 22                       # coarser than hill_case for speed
N_ORDER      = 4
DT           = 1e-3
NSTEP        = 600
F0, TSHIFT   = 12.0, 0.12
VP, VS, RHO  = 2200.0, 1100.0, 2000.0

SRC_XZ       = (X_MAX / 2.0, Z_MAX / 2.0)   # well inside, far from edges
SRC_MT       = (1.0, 1.0, 0.0)              # explosion-like

RECS = [
    (X_MAX / 2.0 - 200.0, Z_MAX / 2.0),     # 200 m left
    (X_MAX / 2.0 + 200.0, Z_MAX / 2.0),     # 200 m right
    (X_MAX / 2.0,         Z_MAX / 2.0 - 150.0),  # 150 m above
]


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _make_models(mesh):
    e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp_e  = torch.full(e, VP,  dtype=torch.float64)
    vs_e  = torch.full(e, VS,  dtype=torch.float64)
    rho_e = torch.full(e, RHO, dtype=torch.float64)
    return vp_e, vs_e, rho_e


def run_displacement_2nd_order(mesh):
    vp_e, vs_e, rho_e = _make_models(mesh)
    # No absorbing boundary: pure interior test.
    eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e)
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[SRC_XZ], receiver_xz=RECS,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    # Displacement → numerical velocity via centered gradient
    vx = np.gradient(out[..., 0].T.numpy(), DT, axis=1)
    vz = np.gradient(out[..., 1].T.numpy(), DT, axis=1)
    return vx, vz


def run_velocity_stress_1st_order(mesh):
    vp_e, vs_e, rho_e = _make_models(mesh)
    eq = ElasticVSSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e, free_top=False)
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[SRC_XZ], receiver_xz=RECS,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t, F0, TSHIFT))
    out = prop.forward(src)
    return out[..., 0].T.numpy(), out[..., 1].T.numpy()


def main():
    mesh = build_axis_aligned_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_max=Z_MAX,
        n_order=N_ORDER, dtype=torch.float64,
    )
    print(f"Mesh: {mesh.n_elements} elements, ngll={mesh.ngll}, "
          f"n_global={mesh.n_global}")

    print("Running 2nd-order displacement form…")
    vx_2nd, vz_2nd = run_displacement_2nd_order(mesh)
    print(f"  v_x peak = {np.abs(vx_2nd).max():.3e}, "
          f"v_z peak = {np.abs(vz_2nd).max():.3e}")

    print("Running 1st-order velocity-stress form…")
    vx_1st, vz_1st = run_velocity_stress_1st_order(mesh)
    # NOTE: 1st-order v is at t_{n+1/2}; flip sign of v_z to match the
    # 2nd-order convention (z-down in sweep, gradient picks sign).
    vx_1st_signed = vx_1st
    vz_1st_signed = vz_1st
    print(f"  v_x peak = {np.abs(vx_1st_signed).max():.3e}, "
          f"v_z peak = {np.abs(vz_1st_signed).max():.3e}")

    # ----- save plots ----------------------------------------------------- #
    out_dir = Path("_smoketest")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_axis = np.arange(NSTEP) * DT

    fig, axes = plt.subplots(len(RECS), 2, figsize=(13, 2.5 * len(RECS)),
                             sharex=True)
    if len(RECS) == 1:
        axes = axes[None, :]
    for i, rx_z in enumerate(RECS):
        ax_x, ax_z = axes[i]
        ax_x.plot(t_axis, vx_2nd[i], "k-",  lw=1.2, label="2nd-order (d/dt u_x)")
        ax_x.plot(t_axis, vx_1st_signed[i], "r--", lw=1.0, label="1st-order v_x")
        ax_x.set_title(f"rec {i} @ {rx_z} — v_x", fontsize=10)
        ax_x.legend(fontsize=8); ax_x.tick_params(labelsize=8)

        ax_z.plot(t_axis, vz_2nd[i], "k-",  lw=1.2, label="2nd-order (d/dt u_z)")
        ax_z.plot(t_axis, vz_1st_signed[i], "r--", lw=1.0, label="1st-order v_z")
        ax_z.set_title(f"rec {i} — v_z", fontsize=10)
        ax_z.legend(fontsize=8); ax_z.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle(f"1st-order velocity-stress vs 2nd-order displacement "
                 f"(flat Q9 mesh, no PML)", fontsize=11)
    fig.tight_layout()
    png = out_dir / "vs_vs_disp_flat.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png.resolve()}")

    # ----- numeric error summary ----------------------------------------- #
    # Use receiver 0 (200 m offset). Restrict to t > arrival time (~0.05s)
    # and t < edge-reflection time (~0.3s) to compare pure interior wave.
    t_win_lo = int(0.02 / DT)
    t_win_hi = int(0.30 / DT)
    print(f"\nMax-norm error over t∈[{t_win_lo*DT}, {t_win_hi*DT}] s "
          f"(interior, pre-reflection):")
    for i in range(len(RECS)):
        peak_x = np.abs(vx_2nd[i, t_win_lo:t_win_hi]).max()
        err_x  = np.abs(vx_2nd[i, t_win_lo:t_win_hi]
                        - vx_1st_signed[i, t_win_lo:t_win_hi]).max()
        peak_z = np.abs(vz_2nd[i, t_win_lo:t_win_hi]).max()
        err_z  = np.abs(vz_2nd[i, t_win_lo:t_win_hi]
                        - vz_1st_signed[i, t_win_lo:t_win_hi]).max()
        print(f"  rec {i}: v_x rel-err = {err_x/peak_x:6.2%}   "
              f"v_z rel-err = {err_z/peak_z:6.2%}")


if __name__ == "__main__":
    main()

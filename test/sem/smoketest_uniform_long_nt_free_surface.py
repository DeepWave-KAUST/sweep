"""Uniform model, free top surface, long nt — boundary + stability sweep.

Four configurations on a uniform vp/vs/rho medium with simple CPML on
left/right/bottom and a free surface at the top edge:

  1. flat axis-aligned mesh + natural BC      (no σ zeroing)
  2. flat axis-aligned mesh + local-coord BC  (σ_nn=σ_nt=0; on a flat
                                                surface this reduces to
                                                pointwise σ_zz=σ_xz=0)
  3. hill topography mesh    + natural BC
  4. hill topography mesh    + local-coord BC

NSTEP large (4 s). Receivers along the surface and at mid depth.
Each config reports peak |v_z| in successive 1-s windows so we can
read off late-time growth, and we plot a gather + a couple of single
traces to visualise residual boundary reflections.

A uniform medium has no internal scatterers, so anything we see in the
gather after the first wavefront has cleared is CPML leakage or BC
instability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import (
    ElasticVSSEM2DDeformed,
    SEMPropagator,
    build_axis_aligned_quad9,
    build_simple_elastic_cpml,
    build_topography_quad9,
)


# ----- geometry ---------------------------------------------------------- #
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ        = 64, 44
N_ORDER         = 4
DT              = 1e-3
NSTEP           = 4000
F0, TSHIFT      = 12.0, 0.12
VP, VS, RHO     = 2200.0, 1100.0, 2000.0

# Source near the centre of the medium so the wavefront has clear room
# to expand before hitting any boundary.
SRC_XZ = (X_MAX / 2.0, Z_BOTTOM / 2.0)
SRC_MT = (1.0, 1.0, 0.0)

# Receivers along the top (just below the free surface) plus a couple
# in the interior.
SURFACE_DEPTH = 60.0    # depth (z value) of the surface-line receivers
REC_TOP = [(x, SURFACE_DEPTH) for x in np.linspace(120.0, X_MAX - 120.0, 7)]
REC_INT = [(X_MAX / 2.0, 300.0), (X_MAX / 2.0, 700.0)]
RECS    = REC_TOP + REC_INT

# Gaussian hill (matches earlier scripts).
HILL_BASE_DEPTH = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0


def z_top_fn(x):
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _build(mesh_kind: str):
    if mesh_kind == "flat":
        mesh = build_axis_aligned_quad9(
            nex=NEX, nez=NEZ, x_max=X_MAX, z_max=Z_BOTTOM,
            n_order=N_ORDER, dtype=torch.float64,
        )
    elif mesh_kind == "hill":
        mesh = build_topography_quad9(
            nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
            z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
        )
    else:
        raise ValueError(mesh_kind)
    e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp_e  = torch.full(e, VP,  dtype=torch.float64)
    vs_e  = torch.full(e, VS,  dtype=torch.float64)
    rho_e = torch.full(e, RHO, dtype=torch.float64)
    return mesh, vp_e, vs_e, rho_e


def run(mesh_kind: str, free_top: bool):
    mesh, vp_e, vs_e, rho_e = _build(mesh_kind)
    cpml = build_simple_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "bottom"),
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    eq = ElasticVSSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, free_top=free_top, cpml=cpml,
    )
    prop = SEMPropagator.build(
        eq, dt=DT, nt=NSTEP,
        source_xz=[SRC_XZ], receiver_xz=RECS,
        source_kinds=["moment_tensor"], source_payloads=[SRC_MT],
    )
    t_arr = np.arange(NSTEP) * DT
    src = torch.from_numpy(ricker(t_arr, F0, TSHIFT))
    out = prop.forward(src)
    return out[..., 0].T.numpy(), -out[..., 1].T.numpy()


def main():
    out_dir = Path("_smoketest")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("flat",  False, "flat / natural BC"),
        ("flat",  True,  "flat / local-coord BC"),
        ("hill",  False, "hill / natural BC"),
        ("hill",  True,  "hill / local-coord BC"),
    ]
    results = {}
    for mesh_kind, free_top, label in configs:
        print(f"Running: {label}  (NSTEP={NSTEP}, {NSTEP*DT:.1f} s)")
        vx, vz = run(mesh_kind, free_top)
        results[label] = (vx, vz)
        print(f"  peak |v_x| = {np.abs(vx).max():.3e}, "
              f"peak |v_z| = {np.abs(vz).max():.3e}")

    # ----- stability table ----------------------------------------------- #
    print()
    print(f"Peak |v_z| in successive 1-s windows:")
    print(f"{'window':>10s}  " +
          "  ".join(f"{lbl:>22s}" for _, _, lbl in configs))
    last_t = NSTEP * DT
    windows = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, last_t)]
    for t0, t1 in windows:
        i0, i1 = int(t0/DT), int(t1/DT)
        cells = []
        for _, _, lbl in configs:
            vz = results[lbl][1]
            cells.append(f"{np.abs(vz[:, i0:i1]).max():22.3e}")
        print(f"  [{t0:.1f},{t1:.1f}]s  " + "  ".join(cells))

    # ----- gather panels ------------------------------------------------- #
    rec_x_arr = np.array([rx for rx, _ in RECS])
    t_axis    = np.arange(NSTEP) * DT

    def panel(ax, seis, title, vmax=0.05, per_trace=False):
        s = seis.copy()
        if per_trace:
            peaks = np.maximum(np.abs(s).max(axis=1, keepdims=True), 1e-30)
            s = s / peaks
        else:
            s = s / max(np.abs(s).max(), 1e-30)
        ax.imshow(s, aspect="auto", cmap="seismic",
                  vmin=-vmax, vmax=vmax,
                  extent=[t_axis[0], t_axis[-1], rec_x_arr[-1], rec_x_arr[0]])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("t (s)"); ax.set_ylabel("rec x (m)")
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12),
                             sharex="col", sharey="row")
    for i, (_, _, lbl) in enumerate(configs):
        vx, vz = results[lbl]
        panel(axes[i, 0], vx, f"{lbl}  v_x  (global norm, vmax=5%)",
              vmax=0.05, per_trace=False)
        panel(axes[i, 1], vz, f"{lbl}  −v_z  (global norm, vmax=5%)",
              vmax=0.05, per_trace=False)
    fig.suptitle(f"Uniform model, free top + CPML on l/r/bot — "
                 f"NSTEP={NSTEP} ({NSTEP*DT:.1f} s)", fontsize=11)
    fig.tight_layout()
    g = out_dir / "uniform_long_nt_gathers.png"
    fig.savefig(g, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g.resolve()}")

    # ----- per-receiver line traces (highlight #3 surface, #8 interior) -- #
    highlight = [3, 8]    # mid-surface receiver, mid-depth interior
    fig, axes = plt.subplots(len(highlight), 2,
                             figsize=(14, 3.0 * len(highlight)),
                             sharex=True)
    for ri, j in enumerate(highlight):
        rxz = RECS[j]
        for _, _, lbl in configs:
            vx, vz = results[lbl]
            axes[ri, 0].plot(t_axis, vx[j], lw=0.9, label=lbl)
            axes[ri, 1].plot(t_axis, vz[j], lw=0.9, label=lbl)
        axes[ri, 0].set_title(f"rec #{j} @ {rxz} — v_x", fontsize=10)
        axes[ri, 1].set_title(f"rec #{j} — v_z", fontsize=10)
        axes[ri, 0].legend(fontsize=7); axes[ri, 1].legend(fontsize=7)
        axes[ri, 0].tick_params(labelsize=8); axes[ri, 1].tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle(f"Uniform model — single-trace seismograms "
                 f"(NSTEP={NSTEP})", fontsize=11)
    fig.tight_layout()
    g = out_dir / "uniform_long_nt_traces.png"
    fig.savefig(g, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {g.resolve()}")


if __name__ == "__main__":
    main()

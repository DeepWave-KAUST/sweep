"""Long-nt seismograms from 1st-order velocity-stress SEM on hill_case.

Runs the 1st-order solver to NSTEP=3000 (3 s, ~4× longer than the
SPECFEM2D reference) so we can see boundary-reflection behaviour over
the full 'late-time' window where the 2nd-order SPECFEM-style CPML
blows up. The 2nd-order Stacey reference is also plotted for context;
SPECFEM is overlaid where it exists (t ≤ 0.7 s, the limit of the
pre-computed binaries).

Two output figures:

  * ``hill_1st_long_nt_traces.png``  — single-receiver line plots at a
    handful of stations, both v_x and v_z, overlay of 1st-order CPML
    vs 2nd-order Stacey vs SPECFEM where available.
  * ``hill_1st_long_nt_gather.png``  — full gather images for both
    backends, with the SPECFEM cut-off marked.
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
    build_simple_elastic_cpml,
    build_topography_quad9,
)


# Match the wavefield / compare scripts so geometries align.
X_MAX, Z_BOTTOM = 1270.0, 890.0
NEX, NEZ        = 64, 44
N_ORDER         = 4
DT              = 1e-3
NSTEP           = 3000
F0, TSHIFT      = 12.0, 0.12
VP, VS, RHO     = 2200.0, 1100.0, 2000.0
SRC_SPEC_XZ     = (640.0, 860.0)
SRC_MT          = (1.0, 1.0, 0.0)

HILL_BASE_DEPTH = 50.0
HILL_H0         = 50.0
HILL_X0         = X_MAX / 2.0
HILL_SIG        = 160.0

# Single-trace receivers to highlight (picked across offsets, mix of
# interior and near-boundary). Indices into the STATIONS file.
HIGHLIGHT_IDX = [5, 13, 20, 26, 33]


def z_top_fn(x):
    return HILL_BASE_DEPTH * (
        1.0 - torch.exp(-((x - HILL_X0) ** 2) / (2.0 * HILL_SIG ** 2))
    )


def to_sem(xz_spec):
    return (xz_spec[0], Z_BOTTOM - xz_spec[1])


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _build_mesh_and_models():
    mesh = build_topography_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_bottom=Z_BOTTOM,
        z_top_fn=z_top_fn, n_order=N_ORDER, dtype=torch.float64,
    )
    e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    return (
        mesh,
        torch.full(e, VP,  dtype=torch.float64),
        torch.full(e, VS,  dtype=torch.float64),
        torch.full(e, RHO, dtype=torch.float64),
    )


def run_1st_cpml(mesh, vp_e, vs_e, rho_e, stations, *, free_top=False):
    cpml = build_simple_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "bottom"),
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    # NOTE: free_top + CPML at left/right corners is a known
    # late-time-instability regime (Festa & Vilotte 2005). The
    # 1st-order CPML stays stable indefinitely with free_top=False;
    # add it later via M-PML or top-edge mini-CPML if surface-wave
    # physics on the receivers becomes critical.
    eq = ElasticVSSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, free_top=free_top, cpml=cpml,
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
    # SEM v_z z-down → flip to match the convention used in earlier
    # comparison scripts (so the SPECFEM overlay lines up).
    return out[..., 0].T.numpy(), -out[..., 1].T.numpy()


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
    # u → v by gradient, then sign-flip v_z
    vx = np.gradient(out[..., 0].T.numpy(), DT, axis=1)
    vz = -np.gradient(out[..., 1].T.numpy(), DT, axis=1)
    return vx, vz


def load_specfem_traces(specfem_dir, n_stations):
    """Load all SPECFEM .semv traces; return arrays + time axis."""
    bxx = []
    bxz = []
    t0 = None
    nt_spec = None
    for j in range(n_stations):
        ax = np.loadtxt(specfem_dir / "OUTPUT_FILES"
                        / f"AA.S{j:03d}.BXX.semv")
        az = np.loadtxt(specfem_dir / "OUTPUT_FILES"
                        / f"AA.S{j:03d}.BXZ.semv")
        if t0 is None:
            t0 = float(ax[0, 0])
            nt_spec = ax.shape[0]
        bxx.append(ax[:, 1])
        bxz.append(az[:, 1])
    return np.stack(bxx), np.stack(bxz), np.array([t0 + k * DT for k in range(nt_spec)])


def compute_specfem_shift(stations, specfem_dir):
    """Use a near-source receiver's peak v_z time to align SPECFEM to
    the SEM time axis (same procedure as wavefield_3way_hill.py)."""
    rec_x = np.array([rx for rx, _ in stations])
    j_close = int(np.argmin(np.abs(rec_x - SRC_SPEC_XZ[0])))
    seis_z = np.loadtxt(specfem_dir / "OUTPUT_FILES"
                        / f"AA.S{j_close:03d}.BXZ.semv")
    t_file = seis_z[:, 0]
    peak_t_spec = t_file[np.argmax(np.abs(seis_z[:, 1]))]
    expected_t = TSHIFT + abs(rec_x[j_close] - SRC_SPEC_XZ[0]) / VP
    return expected_t - peak_t_spec, t_file


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
    specfem_dir = Path(args.specfem_dir)

    stations_lines = (specfem_dir / "DATA" / "STATIONS").read_text().splitlines()
    stations = [(float(p.split()[2]), float(p.split()[3]))
                for p in stations_lines]
    rec_x_arr = np.array([rx for rx, _ in stations])
    n_stations = len(stations)
    print(f"{n_stations} receivers from STATIONS")

    mesh, vp_e, vs_e, rho_e = _build_mesh_and_models()
    print(f"NSTEP={NSTEP}, total sim time = {NSTEP*DT:.2f} s")

    print("Running 1st-order velocity-stress + simple CPML…")
    c1_vx, c1_vz = run_1st_cpml(mesh, vp_e, vs_e, rho_e, stations)
    print(f"  peak |v_z| = {np.abs(c1_vz).max():.3e}; "
          f"late-window peak (t>1s) = "
          f"{np.abs(c1_vz[:, int(1.0/DT):]).max():.3e}")

    print("Running 2nd-order displacement + Stacey…")
    s_vx, s_vz = run_2nd_stacey(mesh, vp_e, vs_e, rho_e, stations)
    print(f"  peak |v_z| = {np.abs(s_vz).max():.3e}; "
          f"late-window peak (t>1s) = "
          f"{np.abs(s_vz[:, int(1.0/DT):]).max():.3e}")

    # ----- SPECFEM reference (only available for t ≤ 0.7 s) ------------- #
    spec_vx_raw, spec_vz_raw, t_spec_axis = load_specfem_traces(
        specfem_dir, n_stations
    )
    spec_shift, _ = compute_specfem_shift(stations, specfem_dir)
    t_spec_phys = t_spec_axis + spec_shift
    # SPECFEM v_z z-up convention → flip to match SEM
    spec_vz_raw = -spec_vz_raw
    print(f"SPECFEM time shift = {spec_shift*1e3:+.1f} ms; "
          f"SPECFEM cut-off = {t_spec_phys[-1]:.2f} s")

    t_sem = np.arange(NSTEP) * DT

    # ----- 1. Single-trace line plots ----------------------------------- #
    n_hi = len(HIGHLIGHT_IDX)
    fig, axes = plt.subplots(n_hi, 2, figsize=(15, 2.2 * n_hi),
                             sharex=True)
    for ri, j in enumerate(HIGHLIGHT_IDX):
        rx = stations[j][0]
        off = abs(rx - SRC_SPEC_XZ[0])
        ax_x, ax_z = axes[ri]
        # v_x
        ax_x.plot(t_sem,        c1_vx[j], "C2-",  lw=1.0, label="1st CPML")
        ax_x.plot(t_sem,        s_vx[j],  "C1--", lw=0.9, label="2nd Stacey")
        ax_x.plot(t_spec_phys,  spec_vx_raw[j], "k:", lw=0.9, label="SPECFEM2D")
        ax_x.axvline(t_spec_phys[-1], color="gray", lw=0.5,
                     linestyle="-", alpha=0.5)
        ax_x.set_title(f"rec #{j}  x={rx:.0f}  offset={off:.0f} m  —  v_x",
                       fontsize=9)
        ax_x.legend(fontsize=7, loc="upper right")
        ax_x.tick_params(labelsize=8)
        # v_z
        ax_z.plot(t_sem,        c1_vz[j], "C2-",  lw=1.0, label="1st CPML")
        ax_z.plot(t_sem,        s_vz[j],  "C1--", lw=0.9, label="2nd Stacey")
        ax_z.plot(t_spec_phys,  spec_vz_raw[j], "k:", lw=0.9, label="SPECFEM2D")
        ax_z.axvline(t_spec_phys[-1], color="gray", lw=0.5,
                     linestyle="-", alpha=0.5)
        ax_z.set_title(f"rec #{j} — v_z", fontsize=9)
        ax_z.legend(fontsize=7, loc="upper right")
        ax_z.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle(f"hill_case NSTEP={NSTEP} ({NSTEP*DT:.1f} s) — single-trace seismograms",
                 fontsize=11)
    fig.tight_layout()
    g_traces = fig_dir / "hill_1st_long_nt_traces.png"
    fig.savefig(g_traces, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_traces.resolve()}")

    # ----- 2. Full gather (per-trace normalised, side by side) ---------- #
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
        ax.axvline(t_spec_phys[-1], color="g", lw=0.6, linestyle="--",
                   alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t (s)"); ax.set_ylabel("receiver x (m)")
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8),
                             sharex="col", sharey="row")
    panel(axes[0, 0], t_sem, c1_vx, "1st-order CPML v_x", vmax=0.15)
    panel(axes[0, 1], t_sem, s_vx,  "2nd-order Stacey v_x", vmax=0.15)
    panel(axes[1, 0], t_sem, c1_vz, "1st-order CPML −v_z", vmax=0.15)
    panel(axes[1, 1], t_sem, s_vz,  "2nd-order Stacey −v_z", vmax=0.15)
    fig.suptitle(
        f"hill_case NSTEP={NSTEP} — green dashed = SPECFEM cut-off "
        f"(t={t_spec_phys[-1]:.2f} s)",
        fontsize=11,
    )
    fig.tight_layout()
    g_gather = fig_dir / "hill_1st_long_nt_gather.png"
    fig.savefig(g_gather, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {g_gather.resolve()}")

    # ----- 3. Quantitative stability check ----------------------------- #
    print()
    print(f"Stability — peak |v_z| at successive 1-s windows "
          f"(SPECFEM cut-off at {t_spec_phys[-1]:.2f} s):")
    print(f"{'window':>16s}  {'1st CPML':>12s}  {'2nd Stacey':>12s}")
    last_t = NSTEP * DT
    for t0, t1 in [(0.0, 1.0), (1.0, 2.0), (2.0, last_t)]:
        i0, i1 = int(t0/DT), int(t1/DT)
        c1_p = np.abs(c1_vz[:, i0:i1]).max()
        s_p  = np.abs(s_vz[:,  i0:i1]).max()
        print(f"  [{t0:.1f}, {t1:.1f}] s   {c1_p:12.3e}  {s_p:12.3e}")


if __name__ == "__main__":
    main()

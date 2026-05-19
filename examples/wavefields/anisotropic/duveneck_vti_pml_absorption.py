"""AcousticVTI1st CPML absorbing-boundary test (two scenarios).

Demonstrates that the CPML correctly absorbs the P-wavefront on all four
sides.  The slow source-generated S-artefact intrinsic to the V_S = 0
acoustic VTI system can mask a global "interior peak" metric, so we run
two clean configurations:

  Case A — isotropic medium (ε=δ=0)
    No S-artefact at all.  Single P front, with horizontal/vertical
    velocity identical to V_P.  This is the cleanest CPML stress test.

  Case B — VTI medium (ε=0.25, δ=0.0) with δ=ε source-region taper
    The S-artefact is suppressed via smooth_delta_to_epsilon_disk so the
    interior peak is again dominated by the outgoing P front.

For each case we capture the wavefield at:
  t1 = 0.30 s  (wave entirely inside model)
  t2 = 0.55 s  (P front touching boundary)
  t3 = 0.80 s  (P front inside PML)
  t4 = 1.10 s  (P should be absorbed)
  t5 = 1.50 s  (any residual = leakage / reflection)

Pass criterion: interior peak |vz| drops ≥100× between t1 and t5.
"""

import numpy as np
import torch

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st
from sweep.equations._anisotropy_utils import smooth_delta_to_epsilon_disk
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
VP = 1500.0
RHO = 1000.0

NZ = NX = 401
DH = 5.0
DT = 1.0e-3
DOM_FREQ = 25.0
ABCN = 50

SNAPSHOT_T = [0.30, 0.55, 0.80, 1.10, 1.50]


def _make_wavelet(nt):
    t = np.arange(nt, dtype=np.float32) * DT - 0.15
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def _run(eps_field, delta_field, label):
    shape = (NZ, NX)
    src_ix = NX // 2
    src_iz = NZ // 2
    src_pos = torch.tensor(np.array([[src_ix, src_iz]], dtype=np.int64)[None])

    models = [
        torch.full(shape, VP, dtype=torch.float32),
        torch.tensor(eps_field, dtype=torch.float32),
        torch.tensor(delta_field, dtype=torch.float32),
        torch.full(shape, RHO, dtype=torch.float32),
    ]

    nt = int(max(SNAPSHOT_T) / DT) + 1
    snap_indices = [int(t / DT) for t in SNAPSHOT_T]

    eq = AcousticVTI1st(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(
        eq, shape,
        source_type=["sH", "sV"], receiver_type=["vz"],
        abcn=ABCN, dh=DH, dt=DT, nt=nt, device="cpu",
        use_ckpt=False,
    )
    wavelet = torch.tensor(_make_wavelet(nt))[None, None, :]

    print(f"[{label}] running {nt} steps, snapshots at {snap_indices} ...")
    with torch.no_grad():
        _, wf = prop(
            wavelet, src_pos, src_pos, models=models,
            return_wavefield=True, snapshot_times=snap_indices,
        )

    idx_vz = eq.wavefields.index("vz")
    snaps = []
    interior_peaks = []
    boundary_peaks = []
    for s in range(len(SNAPSHOT_T)):
        vz = prop.crop(wf[s, idx_vz])[0, 0].numpy()
        snaps.append(vz)

        margin = 30
        interior = vz[margin:-margin, margin:-margin]
        interior_peaks.append(float(np.abs(interior).max()))

        # boundary band: 5 outermost interior rows on each side
        boundary_band = np.concatenate([
            vz[:5, :].ravel(), vz[-5:, :].ravel(),
            vz[:, :5].ravel(), vz[:, -5:].ravel(),
        ])
        boundary_peaks.append(float(np.abs(boundary_band).max()))

    return snaps, interior_peaks, boundary_peaks


def _print_table(label, peaks_int, peaks_bnd):
    ref = peaks_int[0]
    print(f"\n=== Case {label}: interior-region peak |vz| ===")
    print(f"{'t (s)':<7} {'peak|vz|':<14} {'/peak0':<10} {'bdy peak':<14}")
    for t, pi, pb in zip(SNAPSHOT_T, peaks_int, peaks_bnd):
        print(f"{t:<7.3f} {pi:<14.3e} {pi/ref:<10.3e} {pb:<14.3e}")
    drop = ref / max(peaks_int[-1], 1e-30)
    print(f"  → residual-drop factor (t1 → t5) = {drop:.1f}x  "
          f"{'PASS' if drop >= 100 else 'WARN'}")
    return drop


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = (NZ, NX)
    src_ix = NX // 2
    src_iz = NZ // 2

    # ----- Case A: isotropic medium (ε=δ=0) -----
    eps_A = np.zeros(shape, np.float32)
    dlt_A = np.zeros(shape, np.float32)
    snaps_A, ipk_A, bpk_A = _run(eps_A, dlt_A, "isotropic")
    drop_A = _print_table("A (isotropic)", ipk_A, bpk_A)

    # ----- Case B: VTI with source-region δ=ε taper -----
    eps_B = np.full(shape, 0.25, np.float32)
    dlt_B = np.zeros(shape, np.float32)
    dlt_B = smooth_delta_to_epsilon_disk(
        eps_B, dlt_B, source_grid_pos=(src_ix, src_iz), r_taper_grid=8
    )
    snaps_B, ipk_B, bpk_B = _run(eps_B, dlt_B, "VTI+taper")
    drop_B = _print_table("B (VTI + δ=ε taper)", ipk_B, bpk_B)

    # ----- Plot 2 rows (cases) × 5 columns (snapshots) -----
    fig, axes = plt.subplots(2, len(SNAPSHOT_T), figsize=(3.5 * len(SNAPSHOT_T), 7))
    ext = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]
    vlim_A = max(np.abs(snaps_A[0]).max(), 1e-30) * 0.4
    vlim_B = max(np.abs(snaps_B[0]).max(), 1e-30) * 0.4

    for c, t in enumerate(SNAPSHOT_T):
        ax = axes[0, c]
        ax.imshow(snaps_A[c], cmap="seismic", aspect="auto",
                  extent=ext, vmin=-vlim_A, vmax=vlim_A)
        ax.set_title(f"A iso | t={t:.2f}s\npeak={ipk_A[c]:.2e}", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

        ax = axes[1, c]
        ax.imshow(snaps_B[c], cmap="seismic", aspect="auto",
                  extent=ext, vmin=-vlim_B, vmax=vlim_B)
        ax.set_title(f"B VTI+taper | t={t:.2f}s\npeak={ipk_B[c]:.2e}", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.suptitle(
        f"AcousticVTI1st CPML absorption test — v_z snapshots\n"
        f"Case A: isotropic (ε=δ=0), drop = {drop_A:.0f}x   |   "
        f"Case B: VTI ε=0.25,δ=0 + r=8 taper, drop = {drop_B:.0f}x\n"
        f"PML = {ABCN} cells, dh = {DH} m, dt = {DT*1e3:.1f} ms",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "duveneck_vti_pml_absorption.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return drop_A, drop_B


if __name__ == "__main__":
    main()

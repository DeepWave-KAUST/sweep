"""Side-by-side comparison of Stage-1 staircase (``Acoustic``) and
Stage-3 curvilinear (``AcousticCurvilinear``) free-surface topography
on identical model + acquisition geometry.

What it shows:
  - Two shot-gather records (raw amplitude, same colormap), one per
    method.
  - A signed difference panel (staircase − curvilinear) using the same
    clip — large near hilltop receivers where the staircase corner
    diffraction is strongest, near zero in the bulk.
  - A single-trace overlay at one mid-array receiver.
  - A wavefield-snapshot side-by-side at one time, plus an air-mask
    overlay so the topography is clearly readable.

Outputs (in ``outputs/``):
    acoustic2d_compare_record.png      4-panel record comparison
    acoustic2d_compare_wavefield.png   2-panel snapshot comparison
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Acoustic, AcousticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# Identical to the two stand-alone demos so the comparison is
# apples-to-apples.
NZ, NX = 160, 320
DH = 10.0
DT = 1.0e-3
NT = 900            # use the staircase demo's NT (curvilinear handles it easily)
ABCN = 40
SO = 4
FREQ = 8.0
DELAY = 0.12

SNAP_T = 350        # which snapshot index (in NT steps) to compare visually
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    x = np.arange(nx, dtype=np.float32)
    peak_a = 22.0 * np.exp(-((x - nx * 0.30) ** 2) / (2.0 * 22.0**2))
    peak_b = 16.0 * np.exp(-((x - nx * 0.68) ** 2) / (2.0 * 30.0**2))
    ripple = 3.0 * np.sin(2.0 * np.pi * x / 60.0)
    return np.clip(np.round(28.0 - (peak_a + peak_b + ripple)), 0, nx).astype(np.int64)


def build_velocity(nz: int, nx: int, topo: np.ndarray) -> np.ndarray:
    vp = np.full((nz, nx), 2800.0, dtype=np.float32)
    for ix in range(nx):
        s = topo[ix]
        vp[s : s + 6, ix] = 1800.0
        vp[s + 6 : s + 41, ix] = 2300.0
    return vp


def build_wavelet() -> torch.Tensor:
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))


def run_staircase(topo, vp_np, src_x_phys, src_z_phys, rec_x, rec_z_phys, device):
    """``Acoustic`` (vacuum / staircase free surface) on the PHYSICAL grid."""
    eq = Acoustic(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        eq, shape=(NZ, NX), free_surface=True, topography=topo,
        abcn=ABCN, dh=DH, dt=DT, use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )
    sources = torch.from_numpy(np.array([[src_x_phys, src_z_phys]], dtype=np.int64)).to(device)
    receivers = torch.from_numpy(
        np.stack([rec_x, rec_z_phys], axis=-1)[None, ...]
    ).to(device)
    vp = torch.from_numpy(vp_np).to(device)
    snap_times = list(range(0, NT, NT // 30))
    with torch.no_grad():
        record, snaps = prop(
            build_wavelet().to(device), sources, receivers, models=[vp],
            return_wavefield=True, snapshot_times=snap_times,
        )
    p_snap = snaps[:, 0, 0, 0].detach().cpu().numpy()
    return record.detach().cpu().numpy(), p_snap, snap_times


def run_curvilinear(topo, vp_np, src_x_phys, src_z_offset, rec_x, rec_z_offset, device):
    """``AcousticCurvilinear`` on the COMPUTATIONAL grid (η = 0 = surface)."""
    eq = AcousticCurvilinear(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        eq, shape=(NZ, NX), free_surface=True, topography=topo,
        abcn=ABCN, dh=DH, dt=DT, use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )
    sources = torch.from_numpy(
        np.array([[src_x_phys, src_z_offset]], dtype=np.int64)
    ).to(device)
    rec_z_comp = np.full_like(rec_x, rec_z_offset)
    receivers = torch.from_numpy(
        np.stack([rec_x, rec_z_comp], axis=-1)[None, ...]
    ).to(device)
    vp = torch.from_numpy(vp_np).to(device)
    snap_times = list(range(0, NT, NT // 30))
    with torch.no_grad():
        record, snaps = prop(
            build_wavelet().to(device), sources, receivers, models=[vp],
            return_wavefield=True, snapshot_times=snap_times,
        )
    p_snap = snaps[:, 0, 0, 0].detach().cpu().numpy()
    return record.detach().cpu().numpy(), p_snap, snap_times


def resample_curvilinear_to_physical(p_snap_comp, topo):
    """Map ``p_snap_comp`` (shape (n_snap, nz_pad, nx_pad)) from the
    computational (ξ, η) grid back to the physical (X, Z) grid. Cells
    above ``topo[ix]`` stay zero. ``nz_pad`` is expected to equal
    ``NZ + ABCN`` for free-surface."""
    n_snap, nz_pad, nx_pad = p_snap_comp.shape
    p_phys_grid = np.zeros_like(p_snap_comp)
    # Crop PML margins so we operate on physical (NZ, NX).
    p_comp_phys = p_snap_comp[:, :NZ, ABCN : ABCN + NX]
    p_phys = np.zeros((n_snap, NZ, NX), dtype=np.float32)
    eta_comp = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    for ix in range(NX):
        s = int(topo[ix])
        depth = NZ - 1 - s
        if depth <= 0:
            continue
        eta_phys = np.linspace(0.0, 1.0, depth + 1, dtype=np.float32)
        for t in range(n_snap):
            p_phys[t, s : NZ, ix] = np.interp(eta_phys, eta_comp, p_comp_phys[t, :, ix])
    return p_phys


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[compare] device = {device}")
    topo = build_topography(NX)
    vp_np = build_velocity(NZ, NX, topo)

    # Identical PHYSICAL source / receiver positions.
    src_x = NX // 3                  # column index
    src_offset = 2                   # cells below the local surface
    rec_x = np.arange(6, NX - 6, 3, dtype=np.int64)
    rec_offset = 1                   # 1 cell below the local surface

    # Staircase: physical z-row = topo[ix] + offset
    src_z_phys_stair = int(topo[src_x]) + src_offset
    rec_z_phys_stair = (topo[rec_x] + rec_offset).astype(np.int64)
    print("[compare] running staircase...")
    rec_stair, snap_stair, snap_t = run_staircase(
        topo, vp_np, src_x, src_z_phys_stair, rec_x, rec_z_phys_stair, device
    )

    # Curvilinear: computational z-row = offset (surface is η=0)
    print("[compare] running curvilinear...")
    rec_curv, snap_curv_comp, _ = run_curvilinear(
        topo, vp_np, src_x, src_offset, rec_x, rec_offset, device
    )
    # Resample curvilinear snapshot back to physical grid for visual compare
    snap_curv_phys = resample_curvilinear_to_physical(snap_curv_comp, topo)
    # Staircase snapshot is already on the (NZ_pad, NX_pad) grid in
    # physical coords with vacuum cells above topo; crop to (NZ, NX).
    snap_stair_phys = snap_stair[:, :NZ, ABCN : ABCN + NX]

    # ----- Record comparison figure -----
    rec_s = np.squeeze(rec_stair)    # (nt, nrec)
    rec_c = np.squeeze(rec_curv)
    diff = rec_s - rec_c
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, NT * DT, 0]
    t_axis = np.arange(NT) * DT

    # Each record gets its OWN 99-pct clip so amplitude differences
    # don't squash the weaker panel — the goal is structural compare.
    pct_s = float(np.percentile(np.abs(rec_s), 99.0))
    pct_c = float(np.percentile(np.abs(rec_c), 99.0))
    pct_shared = float(np.percentile(np.abs(np.concatenate([rec_s, rec_c])), 99.0))

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85])

    # Row 0: independently normalised records — STRUCTURAL view
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.imshow(rec_s, cmap="seismic",
                norm=TwoSlopeNorm(vmin=-pct_s, vcenter=0.0, vmax=pct_s),
                extent=extent, aspect="auto")
    ax_s.set_title(f"Staircase  (own-clip ±{pct_s:.0f})")
    ax_s.set_xlabel("receiver x (km)"); ax_s.set_ylabel("time (s)")

    ax_c = fig.add_subplot(gs[0, 1])
    ax_c.imshow(rec_c, cmap="seismic",
                norm=TwoSlopeNorm(vmin=-pct_c, vcenter=0.0, vmax=pct_c),
                extent=extent, aspect="auto")
    ax_c.set_title(f"Curvilinear  (own-clip ±{pct_c:.0f})")
    ax_c.set_xlabel("receiver x (km)"); ax_c.set_ylabel("time (s)")

    # Normalised-residual: residual of trace-normalised records
    # (so we compare WAVE SHAPE not amplitude).
    def _trace_norm(x):
        s = np.sqrt((x**2).mean(axis=0)) + 1e-30
        return x / s
    res_norm = _trace_norm(rec_s) - _trace_norm(rec_c)
    pct_r = float(np.percentile(np.abs(res_norm), 99.0))
    ax_r = fig.add_subplot(gs[0, 2])
    ax_r.imshow(res_norm, cmap="seismic",
                norm=TwoSlopeNorm(vmin=-pct_r, vcenter=0.0, vmax=pct_r),
                extent=extent, aspect="auto")
    ax_r.set_title("Diff of trace-RMS-normalised records\n(structural mismatch only)")
    ax_r.set_xlabel("receiver x (km)"); ax_r.set_ylabel("time (s)")

    # Row 1: shared-scale + raw diff + spectra
    ax_ss = fig.add_subplot(gs[1, 0])
    ax_ss.imshow(rec_s, cmap="seismic",
                 norm=TwoSlopeNorm(vmin=-pct_shared, vcenter=0.0, vmax=pct_shared),
                 extent=extent, aspect="auto")
    ax_ss.set_title("Staircase  (shared clip)")
    ax_ss.set_xlabel("receiver x (km)"); ax_ss.set_ylabel("time (s)")

    ax_cs = fig.add_subplot(gs[1, 1])
    ax_cs.imshow(rec_c, cmap="seismic",
                 norm=TwoSlopeNorm(vmin=-pct_shared, vcenter=0.0, vmax=pct_shared),
                 extent=extent, aspect="auto")
    ax_cs.set_title("Curvilinear  (shared clip)")
    ax_cs.set_xlabel("receiver x (km)"); ax_cs.set_ylabel("time (s)")

    # Frequency-domain comparison at mid-array receiver
    trace_idx = len(rec_x) // 2
    fs = 1.0 / DT
    freqs = np.fft.rfftfreq(NT, DT)
    spec_s = np.abs(np.fft.rfft(rec_s[:, trace_idx]))
    spec_c = np.abs(np.fft.rfft(rec_c[:, trace_idx]))
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.semilogy(freqs, spec_s, "C0", lw=1.2, label="staircase")
    ax_f.semilogy(freqs, spec_c, "C1--", lw=1.0, label="curvilinear")
    ax_f.set_xlim(0, min(60, freqs.max()))
    ax_f.set_xlabel("frequency (Hz)")
    ax_f.set_ylabel("|FFT(p)|")
    ax_f.set_title(f"Amplitude spectrum  (rec x={rec_x[trace_idx] * DH * 1e-3:.2f} km)")
    ax_f.legend(); ax_f.grid(True, alpha=0.3)

    # Row 2: trace overlays (raw + each-normalised)
    ax_t1 = fig.add_subplot(gs[2, 0])
    ax_t1.plot(t_axis, rec_s[:, trace_idx], "C0", lw=1.0, label="staircase")
    ax_t1.plot(t_axis, rec_c[:, trace_idx], "C1--", lw=1.0, label="curvilinear")
    ax_t1.set_xlabel("time (s)"); ax_t1.set_ylabel("p")
    ax_t1.set_title("Raw trace overlay  (true amplitudes)")
    ax_t1.legend(); ax_t1.grid(True, alpha=0.3)

    ax_t2 = fig.add_subplot(gs[2, 1])
    tr_s_n = rec_s[:, trace_idx] / (np.abs(rec_s[:, trace_idx]).max() + 1e-30)
    tr_c_n = rec_c[:, trace_idx] / (np.abs(rec_c[:, trace_idx]).max() + 1e-30)
    ax_t2.plot(t_axis, tr_s_n, "C0", lw=1.0, label="staircase (peak-norm)")
    ax_t2.plot(t_axis, tr_c_n, "C1--", lw=1.0, label="curvilinear (peak-norm)")
    ax_t2.set_xlabel("time (s)"); ax_t2.set_ylabel("p / |p|max")
    ax_t2.set_title("Peak-normalised trace overlay  (shape compare)")
    ax_t2.legend(); ax_t2.grid(True, alpha=0.3)

    ax_t3 = fig.add_subplot(gs[2, 2])
    ax_t3.plot(t_axis, tr_s_n - tr_c_n, "k", lw=0.8)
    ax_t3.set_xlabel("time (s)"); ax_t3.set_ylabel("Δ(p/|p|max)")
    ax_t3.set_title("Peak-normalised residual")
    ax_t3.grid(True, alpha=0.3)

    fig.suptitle(
        "Staircase vs curvilinear — same model + acquisition.\n"
        "Top row: each panel scaled to its OWN peak (compares STRUCTURE). "
        "Middle row: shared scale (shows amplitude inflation in staircase). "
        "Bottom row: representative trace.",
        fontsize=11,
    )
    out_rec = OUTPUT_DIR / "acoustic2d_compare_record.png"
    fig.savefig(out_rec, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {out_rec}")

    # ----- Wavefield snapshot comparison -----
    snap_idx_close = int(np.argmin(np.abs(np.array(snap_t) - SNAP_T)))
    t_snap = snap_t[snap_idx_close] * DT
    s_s = snap_stair_phys[snap_idx_close]
    s_c = snap_curv_phys[snap_idx_close]
    pct_s = float(np.percentile(np.abs(np.concatenate([s_s, s_c])), 99.0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    ex = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    norm_s = TwoSlopeNorm(vmin=-pct_s, vcenter=0.0, vmax=pct_s)
    for ax, snap, title in [
        (axes[0], s_s, f"Staircase  p  @ t={t_snap:.3f} s"),
        (axes[1], s_c, f"Curvilinear  p  @ t={t_snap:.3f} s"),
    ]:
        im = ax.imshow(snap, cmap="seismic", norm=norm_s, extent=ex, aspect="auto")
        ax.plot(np.arange(NX) * DH * 1e-3, topo * DH * 1e-3, "k-", lw=1.2)
        ax.fill_between(np.arange(NX) * DH * 1e-3, 0, topo * DH * 1e-3, color="white")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.85, label="p")
    out_snap = OUTPUT_DIR / "acoustic2d_compare_wavefield.png"
    fig.savefig(out_snap, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {out_snap}")

    # Quick summary statistics
    rel_max = np.abs(diff).max() / max(np.abs(rec_s).max(), 1e-30)
    rms_s = np.sqrt((rec_s**2).mean())
    rms_d = np.sqrt((diff**2).mean())
    print(f"[compare] |peak diff| / |peak stair| = {rel_max:.3e}")
    print(f"[compare] RMS staircase = {rms_s:.3e}, RMS diff = {rms_d:.3e}, "
          f"RMS_diff/RMS_stair = {rms_d/rms_s:.3e}")


if __name__ == "__main__":
    main()

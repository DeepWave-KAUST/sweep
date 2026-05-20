"""Homogeneous 3-D VTI wavefield snapshot (Duveneck 2008 — 3-D analog of Fig 1).

Centred Ricker source in σ_H + σ_V on a homogeneous VTI medium.  Snapshot
at t ≈ 0.4 s.  Two figures are produced:

  * duveneck_vti3d_wavefield_xz.png — XZ slice through the source y_src
    showing four fields (-σ_V, -σ_H, v_x, v_z), same layout as the 2-D
    companion script.
  * duveneck_vti3d_wavefield_vz_orth.png — three orthogonal slices of v_z
    (XZ, YZ, XY) through the source.

The V_S = 0 source-generated diamond shear artefact (Duveneck 2008
discussion) is visible on all four-field panels — the rotational invariance
about the vertical axis means the diamond shows as a square in horizontal
slices (XY) and the standard diamond in vertical slices (XZ / YZ).

References
----------
Duveneck et al. (2008) DOI: 10.1190/1.3059320
"""

import numpy as np

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st3D
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Physical parameters — mirror the 2-D Duveneck Fig 1 setup
# ---------------------------------------------------------------------------
VP = 1500.0       # m/s vertical P velocity
EPS = 0.25        # Thomsen ε  (large → diamond clearly visible)
DELTA = 0.0       # Thomsen δ  (ε ≠ δ → artefact present)
RHO = 1000.0      # kg/m³

# Mirror the 2-D Fig 1 setup exactly (dh=5 m, dt=1 ms, dom_freq=25 Hz,
# 401 cells per side, abcn=50, snapshot at t=0.6 s).  With the PML padding
# the per-tensor footprint is 505³ × 4 B ≈ 515 MB; the eager-3-D step
# pipeline touches roughly 28 tensors at peak (state + intermediates +
# new outputs), so total GPU memory peaks around 20–25 GB.  Needs an
# essentially empty 48 GB card — if another process is using more than
# ~20 GB this will OOM; drop to 321³ in that case.
NZ = NY = NX = 401
DH = 5.0
DT = 1.0e-3
SNAPSHOT_T = 0.6
DOM_FREQ = 25.0
ABCN = 50


def _check_cfl():
    dt_max = AcousticVTI1st3D.recommended_dt(VP, EPS, DH)
    if DT > dt_max * 1.05:
        raise RuntimeError(
            f"DT={DT:.4f}s exceeds CFL limit {dt_max:.4f}s for "
            f"VP={VP}, EPS={EPS}, DH={DH}."
        )
    print(f"CFL check passed: dt={DT:.4f}s <= {dt_max:.4f}s")


def _make_wavelet(nt):
    # Delay matches the 2-D companion (0.15 s for f=25 Hz) so the pulse is
    # fully launched before the snapshot is taken.
    t = np.arange(nt, dtype=np.float32) * DT - 0.15
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    _check_cfl()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    shape = (NZ, NY, NX)
    src_ix = NX // 2
    src_iy = NY // 2
    src_iz = NZ // 2
    # SWEEP 3-D source coords are (ix, iy, iz)
    src_pos = torch.tensor(
        np.array([[src_ix, src_iy, src_iz]], dtype=np.int64)[None]
    )

    models = [
        torch.full(shape, VP, dtype=torch.float32, device=device),
        torch.full(shape, EPS, dtype=torch.float32, device=device),
        torch.full(shape, DELTA, dtype=torch.float32, device=device),
        torch.full(shape, RHO, dtype=torch.float32, device=device),
    ]

    eq = AcousticVTI1st3D(spatial_order=4, device=device, backend="torch")
    nt = int(SNAPSHOT_T / DT) + 1
    wavelet = torch.tensor(_make_wavelet(nt))[None, None, :].to(device)

    prop = PropTorch(
        eq, shape,
        source_type=["sH", "sV"],
        receiver_type=["vz"],
        abcn=ABCN, dh=DH, dt=DT, nt=nt, device=device,
        # eager autograd with snapshot_times keeps memory bounded; the
        # compiled C path stores save_all_wavefields = full (nt, 5, ...) which
        # would be far too big for a 121³ × 267 step run.  So we pin
        # impl='eager' here.  `use_compile=False` skips torch.compile/inductor
        # because `sweep.__init__._LAZY_SUBMODULES` still lists 'torch' (a
        # stale lazy alias removed by an earlier refactor), and inductor's
        # FxGraphCache hits that during pickle hashing — unrelated to the
        # 3-D kernel itself.
        eager_options={"use_compile": False},
        impl="eager",
        use_ckpt=False,
    )

    print(f"Forward propagation: nt={nt} steps on {shape} grid ...", flush=True)
    with torch.no_grad():
        _, wf = prop(
            wavelet, src_pos, src_pos,
            models=models,
            return_wavefield=True,
            snapshot_times=[nt - 1],
        )

    # wf shape (eager 3-D): (n_snaps, n_fields, B, C, nz_pad, ny_pad, nx_pad)
    field_names = eq.wavefields
    print("field names:", field_names)
    print("raw wf shape:", tuple(wf.shape))

    def get_field(name):
        idx = field_names.index(name)
        full = wf[0, idx]                              # (B, C, nz_pad, ny_pad, nx_pad)
        cropped = prop.crop(full)                      # (B, C, nz, ny, nx)
        return cropped[0, 0].cpu().numpy()              # (nz, ny, nx)

    vx_snap = get_field("vx")
    vy_snap = get_field("vy")
    vz_snap = get_field("vz")
    sH_snap = get_field("sH")
    sV_snap = get_field("sV")
    print("cropped slice shape:", vx_snap.shape)

    # ----------------------------------------------------------------
    # Figure 1 — XZ slice through y_src (mirror of 2-D Duveneck Fig 1)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    panels_xz = [
        (-sV_snap[:, src_iy, :], "-σ_V"),
        (-sH_snap[:, src_iy, :], "-σ_H"),
        ( vx_snap[:, src_iy, :],  "v_x"),
        ( vz_snap[:, src_iy, :],  "v_z"),
    ]
    ext_xz = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]
    for ax, (data, title) in zip(axes, panels_xz):
        vmin, vmax = percentile_clip(data, (2, 98))
        ax.imshow(data, cmap="seismic", aspect="auto", extent=ext_xz,
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.suptitle(
        f"Homogeneous 3-D VTI — XZ slice at y={src_iy*DH:.0f} m, t={SNAPSHOT_T} s\n"
        f"V_P={VP} m/s, ε={EPS}, δ={DELTA}, ρ={RHO} kg/m³  "
        f"(diamond S-artefact visible)",
        fontsize=10,
    )
    fig.tight_layout()
    out1 = OUTPUT_DIR / "duveneck_vti3d_wavefield_xz.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ----------------------------------------------------------------
    # Figure 2 — three orthogonal slices of v_z through the source
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels_orth = [
        (vz_snap[:, src_iy, :], f"XZ at y={src_iy*DH:.0f} m",
         [0, (NX - 1) * DH, (NZ - 1) * DH, 0], "X (m)", "Z (m)"),
        (vz_snap[:, :, src_ix], f"YZ at x={src_ix*DH:.0f} m",
         [0, (NY - 1) * DH, (NZ - 1) * DH, 0], "Y (m)", "Z (m)"),
        (vz_snap[src_iz, :, :], f"XY at z={src_iz*DH:.0f} m",
         [0, (NX - 1) * DH, (NY - 1) * DH, 0], "X (m)", "Y (m)"),
    ]
    # Use SAME (vmin, vmax) across the three v_z slices so the eye can
    # compare magnitudes — pick the joint percentile.
    all_vz_slices = np.concatenate([
        vz_snap[:, src_iy, :].ravel(),
        vz_snap[:, :, src_ix].ravel(),
        vz_snap[src_iz, :, :].ravel(),
    ])
    vmin, vmax = percentile_clip(all_vz_slices, (2, 98))

    for ax, (data, title, ext, xlab, ylab) in zip(axes, panels_orth):
        ax.imshow(data, cmap="seismic", aspect="auto", extent=ext,
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    fig.suptitle(
        f"Homogeneous 3-D VTI — v_z, three orthogonal slices, t={SNAPSHOT_T} s\n"
        f"V_P={VP} m/s, ε={EPS}, δ={DELTA}, ρ={RHO} kg/m³",
        fontsize=10,
    )
    fig.tight_layout()
    out2 = OUTPUT_DIR / "duveneck_vti3d_wavefield_vz_orth.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # Diagnostics
    print(f"\nWavefield magnitudes at snapshot:")
    print(f"  |v_x|_max = {np.abs(vx_snap).max():.3e}")
    print(f"  |v_y|_max = {np.abs(vy_snap).max():.3e}")
    print(f"  |v_z|_max = {np.abs(vz_snap).max():.3e}")
    print(f"  |σ_H|_max = {np.abs(sH_snap).max():.3e}")
    print(f"  |σ_V|_max = {np.abs(sV_snap).max():.3e}")


if __name__ == "__main__":
    main()

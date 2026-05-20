"""Homogeneous 3-D VTI wavefield WITH shear-artifact suppression
(Duveneck 2008 — 3-D analog of Fig 2).

Same setup as duveneck_vti_3d_wavefield.py, but `δ` is smoothly tapered
toward `ε` inside a small ball around the source location via the
`smooth_delta_to_epsilon_disk` helper.  This makes the medium locally
isotropic at the source — the rotational invariance of the resulting
constitutive operator kills the V_S = 0 source-generated diamond shear
artifact, while leaving the outer P-wavefront essentially unchanged.

Produces two figures in outputs/:

  * duveneck_vti3d_shear_suppression_xz.png — XZ slice through the source y
    showing the 4 fields (-σ_V, -σ_H, v_x, v_z) side-by-side with the
    un-suppressed wavefield from duveneck_vti_3d_wavefield.py.
  * duveneck_vti3d_shear_suppression_vz_orth.png — three orthogonal v_z
    slices, suppressed run alone.

Prints the diamond/P amplitude-ratio drop achieved by the taper.

References
----------
Duveneck et al. (2008) DOI: 10.1190/1.3059320
"""

import numpy as np

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st3D
from sweep.equations._anisotropy_utils import smooth_delta_to_epsilon_disk
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Physical parameters — match duveneck_vti_3d_wavefield.py
# ---------------------------------------------------------------------------
VP = 1500.0
EPS = 0.25
DELTA = 0.0
RHO = 1000.0
# Same resolution as the 3-D Fig 1 analog (dh=5 m, dt=1 ms, 25 Hz Ricker)
# and the same 401³ + abcn=50 grid as the 2-D Fig 2 reproduction.  This
# script runs the forward TWICE (once without taper, once with), so peak
# memory must accommodate both runs sequentially — eager-3-D at 401³
# touches ~20 GB on the GPU, so this needs a mostly empty 48 GB card.
NZ = NY = NX = 401
DH = 5.0
DT = 1.0e-3
SNAPSHOT_T = 0.6
DOM_FREQ = 25.0
ABCN = 50

R_TAPER = 8     # 8 cells × 5 m = 40 m physical radius, matches the 2-D
                # default in duveneck_vti_shear_suppression.py


def _make_wavelet(nt):
    # Same delay as 2-D Fig 2 reproduction: 0.15 s for f=25 Hz.
    t = np.arange(nt, dtype=np.float32) * DT - 0.15
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def _amplitude_ratio_inside_diamond(vz_slice, src_iz, src_ix, p_radius_cells=165,
                                    diamond_radius_cells=50):
    """Quick proxy for "diamond / P-front" amplitude ratio.

    The P-wavefront is on a sphere of radius ≈ v_P · t ≈ 1500 · 0.4 = 600 m
    → 60 cells at dh=10 m.  We use a thin annulus around that radius for
    the P-band.  The diamond artefact sits much closer to the source on a
    diagonal "smaller" shape; we use a small box of radius `diamond_radius_cells`
    around the source for the diamond band.  Both numbers are heuristic but
    sufficient to score "diamond suppressed by N×".
    """
    nz, nx = vz_slice.shape
    iz_grid, ix_grid = np.ogrid[0:nz, 0:nx]
    r = np.sqrt((ix_grid - src_ix) ** 2 + (iz_grid - src_iz) ** 2)
    diamond_mask = r <= diamond_radius_cells
    p_mask = (r >= p_radius_cells - 4) & (r <= p_radius_cells + 4)
    diamond_amp = float(np.abs(vz_slice[diamond_mask]).max())
    p_amp = float(np.abs(vz_slice[p_mask]).max())
    return diamond_amp, p_amp


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    shape = (NZ, NY, NX)
    src_ix = NX // 2
    src_iy = NY // 2
    src_iz = NZ // 2

    src_pos = torch.tensor(
        np.array([[src_ix, src_iy, src_iz]], dtype=np.int64)[None]
    )

    eps_arr   = np.full(shape, EPS,   dtype=np.float32)
    delta_arr = np.full(shape, DELTA, dtype=np.float32)

    # δ → ε taper inside a sphere of radius R_TAPER around the source.
    # smooth_delta_to_epsilon_disk takes source coords in (ix, iy, iz) order.
    delta_taper = smooth_delta_to_epsilon_disk(
        eps_arr, delta_arr,
        source_grid_pos=(src_ix, src_iy, src_iz),
        r_taper_grid=R_TAPER,
    )

    print(f"taper applied at source ({src_ix},{src_iy},{src_iz}) with r={R_TAPER} cells")
    print(f"  δ range before taper: [{delta_arr.min():.3f}, {delta_arr.max():.3f}]")
    print(f"  δ range after  taper: [{delta_taper.min():.3f}, {delta_taper.max():.3f}]")

    def _run(delta_field):
        models = [
            torch.full(shape, VP,  dtype=torch.float32, device=device),
            torch.full(shape, EPS, dtype=torch.float32, device=device),
            torch.tensor(delta_field, dtype=torch.float32, device=device),
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
            eager_options={"use_compile": False},
            impl="eager",
            use_ckpt=False,
        )
        with torch.no_grad():
            _, wf = prop(
                wavelet, src_pos, src_pos,
                models=models,
                return_wavefield=True,
                snapshot_times=[nt - 1],
            )
        field_names = eq.wavefields

        def get_field(name):
            idx = field_names.index(name)
            full = wf[0, idx]
            return prop.crop(full)[0, 0].cpu().numpy()

        return {n: get_field(n) for n in ("vx", "vy", "vz", "sH", "sV")}

    print("\nForward 1/2: WITHOUT taper (δ = 0 everywhere) ...", flush=True)
    snap_orig = _run(delta_arr)
    print("Forward 2/2: WITH δ→ε taper around source ...", flush=True)
    snap_supp = _run(delta_taper)

    # Diamond/P amplitude diagnostics on the XZ vz slice at y=src_iy.
    d_orig, p_orig = _amplitude_ratio_inside_diamond(
        snap_orig["vz"][:, src_iy, :], src_iz, src_ix)
    d_supp, p_supp = _amplitude_ratio_inside_diamond(
        snap_supp["vz"][:, src_iy, :], src_iz, src_ix)

    print("\nDiamond/P amplitude (proxy from v_z XZ slice):")
    print(f"  without taper : diamond={d_orig:.3e}  P={p_orig:.3e}  ratio={d_orig/max(p_orig,1e-30):.3e}")
    print(f"  with    taper : diamond={d_supp:.3e}  P={p_supp:.3e}  ratio={d_supp/max(p_supp,1e-30):.3e}")
    print(f"  diamond suppression factor: {d_orig/max(d_supp,1e-30):.2f}×")
    print(f"  P-amplitude change         : {p_supp/max(p_orig,1e-30):.4f}× (should be ~1)")

    # ----------------------------------------------------------------
    # Figure 1 — side-by-side XZ slices, four-field row each.
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    ext_xz = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]
    snaps = [snap_orig, snap_supp]
    row_titles = ["No taper (Fig 1 analog)",
                  f"δ→ε taper r={R_TAPER} cells (Fig 2 analog)"]
    panel_specs = [
        ("sV", lambda d: -d, "-σ_V"),
        ("sH", lambda d: -d, "-σ_H"),
        ("vx", lambda d:  d,  "v_x"),
        ("vz", lambda d:  d,  "v_z"),
    ]
    for r, (snap, row_title) in enumerate(zip(snaps, row_titles)):
        for c, (field, sign, label) in enumerate(panel_specs):
            data = sign(snap[field][:, src_iy, :])
            vmin, vmax = percentile_clip(data, (2, 98))
            ax = axes[r, c]
            ax.imshow(data, cmap="seismic", aspect="auto", extent=ext_xz,
                      vmin=vmin, vmax=vmax)
            if c == 0:
                ax.set_ylabel(f"{row_title}\nZ (m)")
            else:
                ax.set_ylabel("Z (m)")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("X (m)")
    fig.suptitle(
        f"Homogeneous 3-D VTI — shear-artifact suppression, XZ slice at y={src_iy*DH:.0f} m\n"
        f"V_P={VP} m/s, ε={EPS}, δ={DELTA}, ρ={RHO} kg/m³  "
        f"|  diamond drop: {d_orig/max(d_supp,1e-30):.1f}×",
        fontsize=10,
    )
    out1 = OUTPUT_DIR / "duveneck_vti3d_shear_suppression_xz.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ----------------------------------------------------------------
    # Figure 2 — three orthogonal slices of v_z with the taper applied.
    # ----------------------------------------------------------------
    vz = snap_supp["vz"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels_orth = [
        (vz[:, src_iy, :], f"XZ at y={src_iy*DH:.0f} m",
         [0, (NX - 1) * DH, (NZ - 1) * DH, 0], "X (m)", "Z (m)"),
        (vz[:, :, src_ix], f"YZ at x={src_ix*DH:.0f} m",
         [0, (NY - 1) * DH, (NZ - 1) * DH, 0], "Y (m)", "Z (m)"),
        (vz[src_iz, :, :], f"XY at z={src_iz*DH:.0f} m",
         [0, (NX - 1) * DH, (NY - 1) * DH, 0], "X (m)", "Y (m)"),
    ]
    all_vz_slices = np.concatenate([
        vz[:, src_iy, :].ravel(),
        vz[:, :, src_ix].ravel(),
        vz[src_iz, :, :].ravel(),
    ])
    vmin, vmax = percentile_clip(all_vz_slices, (2, 98))
    for ax, (data, title, ext, xlab, ylab) in zip(axes, panels_orth):
        ax.imshow(data, cmap="seismic", aspect="auto", extent=ext,
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    fig.suptitle(
        f"Homogeneous 3-D VTI WITH δ→ε taper (r={R_TAPER}) — v_z, three orthogonal slices, t={SNAPSHOT_T} s\n"
        f"V_P={VP} m/s, ε={EPS}, δ={DELTA}, ρ={RHO} kg/m³",
        fontsize=10,
    )
    fig.tight_layout()
    out2 = OUTPUT_DIR / "duveneck_vti3d_shear_suppression_vz_orth.png"
    fig.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

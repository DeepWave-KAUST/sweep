"""Duveneck 2008 Figure 1: homogeneous VTI wavefield snapshots (no suppression).

Reproduces Figure 1 from Duveneck et al. (2008) qualitatively.  A Ricker
source is injected into both σ_H and σ_V at the grid centre of a homogeneous
VTI medium.  The snapshot at t = 0.6 s shows four wavefields:
  -σ_V, -σ_H, v_x, v_z.

The diamond-shaped slow front (the V_S = 0 source-generated shear artefact)
is clearly visible on all panels.  This is expected — it demonstrates the
artefact that ``duveneck_vti_shear_suppression.py`` will suppress.

Reference
---------
Duveneck, E., Milcik, P., Bakker, P. M., Perkins, C. (2008),
"Acoustic VTI wave equations and their application for anisotropic
reverse-time migration", SEG Las Vegas 2008 Annual Meeting, pp. 2186–2190.
DOI: 10.1190/1.3059320

Usage
-----
  python duveneck_vti_wavefield.py          # uses installed sweep
  python duveneck_vti_wavefield.py --import-mode=source  # uses src/
"""

import numpy as np

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

# ---------------------------------------------------------------------------
# Physical parameters — Duveneck 2008 Fig 1
# ---------------------------------------------------------------------------
VP = 1500.0       # m/s  vertical P velocity
EPS = 0.25        # Thomsen ε
DELTA = 0.0       # Thomsen δ  (ε ≠ δ → artefact visible)
RHO = 1000.0      # kg/m³

NZ = NX = 401
DH = 5.0          # m   grid spacing
DT = 1.0e-3       # s   time step
SNAPSHOT_T = 0.6  # s   snapshot time
DOM_FREQ = 25.0   # Hz  Ricker dominant frequency
ABCN = 50         # PML thickness (grid cells)


def _check_cfl():
    dt_max = AcousticVTI1st.recommended_dt(VP, EPS, DH)
    if DT > dt_max * 1.05:
        raise RuntimeError(
            f"DT={DT:.4f} s exceeds CFL limit {dt_max:.4f} s for "
            f"VP={VP}, EPS={EPS}, DH={DH}."
        )
    print(f"CFL check passed: dt={DT:.4f} s <= {dt_max:.4f} s")


def _make_wavelet(nt):
    t = np.arange(nt, dtype=np.float32) * DT - 0.15
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    _check_cfl()

    shape = (NZ, NX)
    src_ix = NX // 2
    src_iz = NZ // 2
    src_pos = torch.tensor(
        np.array([[src_ix, src_iz]], dtype=np.int64)[None]
    )  # (1, 1, 2)

    models = [
        torch.full(shape, VP, dtype=torch.float32),
        torch.full(shape, EPS, dtype=torch.float32),
        torch.full(shape, DELTA, dtype=torch.float32),
        torch.full(shape, RHO, dtype=torch.float32),
    ]

    eq = AcousticVTI1st(spatial_order=4, device="cpu", backend="torch")
    nt = int(SNAPSHOT_T / DT) + 1
    wavelet = torch.tensor(_make_wavelet(nt))[None, None, :]

    prop = PropTorch(
        eq, shape,
        source_type=["sH", "sV"],
        receiver_type=["vz"],
        abcn=ABCN, dh=DH, dt=DT, nt=nt, device="cpu",
        use_ckpt=False,
    )

    with torch.no_grad():
        _, wf = prop(
            wavelet, src_pos, src_pos,
            models=models,
            return_wavefield=True,
            snapshot_times=[nt - 1],
        )

    # wf: (n_snaps, n_fields, B, 1, nz_pad, nx_pad)
    field_names = eq.wavefields

    def get_field(name):
        idx = field_names.index(name)
        full = wf[0, idx]                       # (B, 1, nz_pad, nx_pad)
        cropped = prop.crop(full)               # (B, 1, nz, nx)
        return cropped[0, 0].numpy()

    vx_snap = get_field("vx")
    vz_snap = get_field("vz")
    sH_snap = get_field("sH")
    sV_snap = get_field("sV")

    panels = [(-sV_snap, "-σ_V"), (-sH_snap, "-σ_H"),
              (vx_snap, "v_x"), (vz_snap, "v_z")]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ext = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]
    for ax, (data, title) in zip(axes, panels):
        vmin, vmax = percentile_clip(data, (2, 98))
        ax.imshow(data, cmap="seismic", aspect="auto", extent=ext, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.suptitle(
        f"Duveneck 2008 Fig 1 — homogeneous VTI, t={SNAPSHOT_T} s\n"
        f"VP={VP} m/s, ε={EPS}, δ={DELTA}, ρ={RHO} kg/m³  "
        f"(diamond S-artefact visible)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "duveneck_vti_wavefield.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return vx_snap, vz_snap, sH_snap, sV_snap


if __name__ == "__main__":
    main()

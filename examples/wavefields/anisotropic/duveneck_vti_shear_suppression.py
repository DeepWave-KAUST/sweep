"""Duveneck 2008 Figure 2: shear-artefact suppression via δ = ε near source.

Reproduces Figure 2 from Duveneck et al. (2008).  The medium, source, and
grid are identical to ``duveneck_vti_wavefield.py``, but a raised-cosine disk
of radius 8 grid cells is applied around the source to taper δ → ε before
constructing the propagator.

At t = 0.6 s the four panels (-σ_V, -σ_H, v_x, v_z) show that:
  * The outer elliptical P-wavefront is unchanged.
  * The diamond-shaped slow S-artefact is suppressed by at least a factor
    of 5 in energy compared to the unsuppressed run.

The suppression ratio is printed at the end of the script.

Reference
---------
Duveneck, E., Milcik, P., Bakker, P. M., Perkins, C. (2008),
"Acoustic VTI wave equations and their application for anisotropic
reverse-time migration", SEG Las Vegas 2008 Annual Meeting, pp. 2186–2190.
DOI: 10.1190/1.3059320
"""

import numpy as np

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip

from sweep.equations import AcousticVTI1st
from sweep.equations._anisotropy_utils import smooth_delta_to_epsilon_disk
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

# ---------------------------------------------------------------------------
# Physical parameters — identical to duveneck_vti_wavefield.py
# ---------------------------------------------------------------------------
VP = 1500.0
EPS = 0.25
DELTA = 0.0       # background δ; will be tapered to ε near source
RHO = 1000.0

NZ = NX = 401
DH = 5.0
DT = 1.0e-3
SNAPSHOT_T = 0.6
DOM_FREQ = 25.0
ABCN = 50

R_TAPER = 8       # taper radius in grid cells


def _make_wavelet(nt):
    t = np.arange(nt, dtype=np.float32) * DT - 0.15
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def _annulus_peak(vz, src_iz, src_ix, r_inner, r_outer):
    nz, nx = vz.shape
    iz, ix = np.ogrid[0:nz, 0:nx]
    dist_m = np.sqrt(((iz - src_iz) * DH) ** 2 + ((ix - src_ix) * DH) ** 2)
    r_P = VP * SNAPSHOT_T
    r_norm = dist_m / r_P
    mask = (r_norm > r_inner) & (r_norm < r_outer)
    return float(np.abs(vz[mask]).max()) if mask.any() else 0.0


def _run_one(delta_field, src_pos):
    import torch
    shape = (NZ, NX)
    models = [
        torch.full(shape, VP, dtype=torch.float32),
        torch.full(shape, EPS, dtype=torch.float32),
        torch.tensor(delta_field, dtype=torch.float32),
        torch.full(shape, RHO, dtype=torch.float32),
    ]
    eq = AcousticVTI1st(spatial_order=4, device="cpu", backend="torch")
    nt = int(SNAPSHOT_T / DT) + 1
    wavelet = torch.tensor(_make_wavelet(nt))[None, None, :]
    prop = PropTorch(
        eq, shape,
        source_type=["sH", "sV"], receiver_type=["vz"],
        abcn=ABCN, dh=DH, dt=DT, nt=nt, device="cpu",
        use_ckpt=False,
    )
    with torch.no_grad():
        _, wf = prop(wavelet, src_pos, src_pos, models=models,
                     return_wavefield=True, snapshot_times=[nt - 1])

    field_names = eq.wavefields

    def get_field(name):
        idx = field_names.index(name)
        return prop.crop(wf[0, idx])[0, 0].numpy()

    return {
        "vx": get_field("vx"),
        "vz": get_field("vz"),
        "sH": get_field("sH"),
        "sV": get_field("sV"),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    shape = (NZ, NX)
    src_ix = NX // 2
    src_iz = NZ // 2
    src_pos = torch.tensor(np.array([[src_ix, src_iz]], dtype=np.int64)[None])

    eps_field = np.full(shape, EPS, dtype=np.float32)

    # background delta (uniform = DELTA)
    delta_bg = np.full(shape, DELTA, dtype=np.float32)

    # tapered: δ → ε inside the disk of radius R_TAPER around source
    delta_tapered = smooth_delta_to_epsilon_disk(
        eps_field, delta_bg.copy(),
        source_grid_pos=(src_ix, src_iz),
        r_taper_grid=R_TAPER,
    )

    print("Running unsuppressed propagation ...")
    unsup = _run_one(delta_bg, src_pos)
    print("Running suppressed propagation ...")
    sup = _run_one(delta_tapered, src_pos)

    # quantitative pass criterion (Spec §9.2): ≥5× drop in diamond/P ratio
    def metric(snap):
        diamond = _annulus_peak(snap["vz"], src_iz, src_ix, 0.2, 0.7)
        p_front = _annulus_peak(snap["vz"], src_iz, src_ix, 0.85, 1.10)
        return diamond, p_front

    d_u, p_u = metric(unsup)
    d_s, p_s = metric(sup)
    ratio_u = d_u / max(p_u, 1e-30)
    ratio_s = d_s / max(p_s, 1e-30)
    factor = ratio_u / max(ratio_s, 1e-30)

    print(f"  Unsuppressed: diamond_peak={d_u:.3e}, P_peak={p_u:.3e}, ratio={ratio_u:.3f}")
    print(f"  Suppressed  : diamond_peak={d_s:.3e}, P_peak={p_s:.3e}, ratio={ratio_s:.3f}")
    print(f"  Suppression factor (unsup/sup ratio) = {factor:.2f}x")
    if factor < 5.0:
        print(f"  WARNING: factor {factor:.2f} < 5 (spec criterion).")
    else:
        print(f"  Pass: factor >= 5x.")

    # Side-by-side comparison plot (top row: unsuppressed, bottom: suppressed)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ext = [0, (NX - 1) * DH, (NZ - 1) * DH, 0]

    titles = ["-σ_V", "-σ_H", "v_x", "v_z"]
    for row, (label, snap) in enumerate([("unsuppressed", unsup),
                                          ("suppressed", sup)]):
        panels = [-snap["sV"], -snap["sH"], snap["vx"], snap["vz"]]
        for ax, data, title in zip(axes[row], panels, titles):
            vmin, vmax = percentile_clip(data, (2, 98))
            ax.imshow(data, cmap="seismic", aspect="auto", extent=ext, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} | {title}", fontsize=11)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")

    fig.suptitle(
        f"Duveneck 2008 Fig 2 — δ=ε disk (r={R_TAPER} cells) suppresses S-artefact\n"
        f"VP={VP} m/s, ε={EPS}, δ={DELTA} (outside disk), t={SNAPSHOT_T} s, "
        f"suppression factor = {factor:.2f}x",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "duveneck_vti_shear_suppression.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return factor


if __name__ == "__main__":
    main()

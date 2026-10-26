"""Stage-4c gate: our coupled AcousticElasticSEM2D vs SPECFEM2D on the SAME
flat water-over-solid model (see build_inputs_fluidsolid.py).

Compares, on identical mesh / dt / source / receivers / (matched) BCs:
  * hydrophone pressure in the water : SPECFEM .semp  vs  ours p = -chi_ddot
  * geophone displacement in the solid: SPECFEM .semd  vs  ours (u_x, u_z)

ONE global scalar (SPECFEM's acoustic source is Lagrange-stamped and divided by
kappa; ours is a unit node) is fitted on the HYDROPHONES ONLY and applied
unchanged to the geophones — so the geophone match tests the fluid->solid
coupling amplitude, not the fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_inputs_fluidsolid import FS

from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


# ------------------------------------------------------------------ #
def run_ours(s: FS):
    m = build_axis_aligned_quad9(nex=s.nex, nez=s.nez, x_max=s.x_max,
                                 z_max=s.z_max, n_order=s.n_order, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    fluid = (zc < s.z_if).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fluid, s.vp_w, s.vp_s).to(DT).contiguous()
    vs = torch.where(fluid, 0.0, s.vs_s).to(DT).contiguous()
    rho = torch.where(fluid, s.rho_w, s.rho_s).to(DT).contiguous()
    # matched BCs: SPECFEM pressure-releases every non-absorbing fluid boundary
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho,
                                    dirichlet=("top", "left", "right"))

    xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    node = lambda x, z: int(((xg - x) ** 2 + (zg - z) ** 2).argmin())
    src = node(*s.src_xz)
    hyd = [node(x, s.z_hydro) for x in s.rec_x]
    geo = [node(x, s.z_geo) for x in s.rec_x]
    assert float((xg[src] - s.src_xz[0]).abs() + (zg[src] - s.src_xz[1]).abs()) < 1e-9

    pat = torch.zeros(m.n_global, dtype=DT); pat[src] = 1.0
    zero = torch.zeros(m.n_global, dtype=DT)
    t = np.arange(s.nstep) * s.dt
    wav = torch.tensor(ricker(t, s.f0, s.t0_internal), dtype=DT)

    f = eq.init_state(s.dt, "cpu", DT)
    chi_tr, ux_tr, uz_tr = [], [], []
    for n in range(s.nstep):
        f = eq.step(f, [pat * wav[n], zero, zero], s.dt)
        chi_tr.append(f[0][hyd].clone())
        ux_tr.append(f[2][geo].clone())
        uz_tr.append(f[3][geo].clone())
    chi = torch.stack(chi_tr).numpy()            # (nt, nrec) chi at level n+1
    ux = torch.stack(ux_tr).numpy()
    uz = torch.stack(uz_tr).numpy()
    # pressure via the exact leapfrog identity (fp64 second difference)
    p = np.full_like(chi, np.nan)
    p[1:-1] = -(chi[2:] - 2 * chi[1:-1] + chi[:-2]) / (s.dt ** 2)
    # our sample k (level k+1) is at physical time (k+1)*dt - t0
    t_ours = (np.arange(s.nstep) + 1) * s.dt - s.t0_internal
    return t_ours, p, ux, uz


# ------------------------------------------------------------------ #
def load_specfem(out_dir: Path, s: FS):
    n = len(s.rec_x)
    def rd(k, ch, ext):
        d = np.loadtxt(out_dir / f"AA.S{k:03d}.{ch}.{ext}")
        return d[:, 0], d[:, 1]
    t_sp, _ = rd(0, "PRE", "semp")
    P = np.stack([rd(k, "PRE", "semp")[1] for k in range(n)], 1)
    UX = np.stack([rd(k + n, "BXX", "semd")[1] for k in range(n)], 1)
    # z-axis convention: our mesh is z-DOWN, SPECFEM is z-UP -> negate BXZ.
    UZ = -np.stack([rd(k + n, "BXZ", "semd")[1] for k in range(n)], 1)
    return t_sp, P, UX, UZ


def align(t_ours, A_ours, t_sp):
    """Interpolate each of our traces onto SPECFEM's time axis (overlap only)."""
    ok = np.isfinite(A_ours).all(axis=1)
    out = np.stack([np.interp(t_sp, t_ours[ok], A_ours[ok, j],
                              left=np.nan, right=np.nan)
                    for j in range(A_ours.shape[1])], 1)
    return out


def metrics(spec, ours, w):
    """Per-trace cosine + rel-L2 on ``w``; near-zero traces (symmetry nulls,
    e.g. Ux under the source) are masked out of the stats."""
    cos, rel, amp = [], [], []
    for j in range(spec.shape[1]):
        a, b = spec[w, j], ours[w, j]
        amp.append(np.linalg.norm(a))
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)))
        rel.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-300)))
    cos, rel, amp = map(np.array, (cos, rel, amp))
    live = amp > 0.02 * amp.max()
    return cos[live], rel[live]


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir",
                    default="_smoketest/specfem_fluidsolid")
    args = ap.parse_args()
    root = Path(args.workdir)
    s = FS()

    t_sp, P_sp, UX_sp, UZ_sp = load_specfem(root / "OUTPUT_FILES", s)
    t_ours, P_o, UX_o, UZ_o = run_ours(s)

    P_i = align(t_ours, P_o, t_sp)
    UX_i = align(t_ours, UX_o, t_sp)
    UZ_i = align(t_ours, UZ_o, t_sp)
    w = np.isfinite(P_i).all(axis=1) & (t_sp > 0.0)

    # ONE global scalar from the hydrophones; applied unchanged to geophones.
    alpha = float((P_sp[w] * P_i[w]).sum() / (P_i[w] ** 2).sum())
    P_i, UX_i, UZ_i = alpha * P_i, alpha * UX_i, alpha * UZ_i

    cP, rP = metrics(P_sp, P_i, w)
    cX, rX = metrics(UX_sp, UX_i, w)
    cZ, rZ = metrics(UZ_sp, UZ_i, w)
    print(f"global scale alpha = {alpha:.6e}  (fitted on hydrophones only)")
    for nm, c, r in (("hydro P ", cP, rP), ("geo   Ux", cX, rX), ("geo   Uz", cZ, rZ)):
        print(f"{nm}: cos min/med {c.min():.5f}/{np.median(c):.5f}   "
              f"relL2 med {np.median(r):.4f}")

    # ---------------- gather overlay figure ---------------- #
    fig, axes = plt.subplots(1, 3, figsize=(19, 9), sharey=True)
    offs = np.array(s.rec_x)
    tt = t_sp[w]
    panels = [("Hydrophone pressure (water, z=400 m)", P_sp, P_i),
              ("Geophone Ux (solid, z=1600 m)", UX_sp, UX_i),
              ("Geophone Uz (solid, z=1600 m)", UZ_sp, UZ_i)]
    for ax, (title, S, O) in zip(axes, panels):
        gain = 0.45 * (offs[1] - offs[0]) / (np.percentile(np.abs(S[w]), 99) + 1e-300)
        for j in range(S.shape[1]):
            ax.plot(offs[j] + gain * S[w, j], tt, "k", lw=0.9,
                    label="SPECFEM2D" if j == 0 else None)
            ax.plot(offs[j] + gain * O[w, j], tt, "r--", lw=0.9,
                    label="sweep SEM" if j == 0 else None)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (m)")
        ax.invert_yaxis() if ax is axes[0] else None
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("t (s)")
    axes[0].set_ylim(tt[-1], tt[0])
    fig.suptitle("Fluid-solid coupling: SPECFEM2D vs sweep SEM "
                 "(same mesh/dt/source/BCs; one global scale from hydrophones)",
                 fontsize=12)
    fig.tight_layout()
    out_png = root / "gathers_fluidsolid_compare.png"
    fig.savefig(out_png, dpi=220)
    print(f"figure: {out_png}")

    np.savez(root / "compare_metrics.npz", alpha=alpha, cos_P=cP, cos_UX=cX,
             cos_UZ=cZ, rel_P=rP, rel_UX=rX, rel_UZ=rZ,
             t=t_sp[w], P_sp=P_sp[w], P_ours=P_i[w],
             UX_sp=UX_sp[w], UX_ours=UX_i[w], UZ_sp=UZ_sp[w], UZ_ours=UZ_i[w])
    print(f"data  : {root / 'compare_metrics.npz'}")


if __name__ == "__main__":
    main()

"""Wavefield-snapshot comparison for the fluid-solid gate: SPECFEM2D pressure
dumps vs our AcousticElasticSEM2D, same model/mesh/source (see
build_inputs_fluidsolid.py). Pressure everywhere: p = -chi_ddot in the fluid,
p = -(lam+mu) div u in the solid (== SPECFEM's elastic display pressure
-(sigma_xx+sigma_zz)/2 for isotropic P-SV)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_inputs_fluidsolid import FS

from sweep.sem._assembly import phys_grads
from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
ROOT = Path("_smoketest/specfem_fluidsolid")


def _read_f32(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size >= 2 and int(raw.view(np.int32)[0]) == (raw.size - 2) * 4:
        return raw[1:-1].copy()
    return raw


def load_specfem_dumps(out_dir: Path):
    grid = _read_f32(out_dir / "wavefield_grid_for_dumps.bin").reshape(-1, 2)
    n = grid.shape[0]
    steps, snaps = [], []
    for p in sorted(out_dir.glob("wavefield*_01.bin")):
        v = _read_f32(p)
        if v.size != n:
            continue
        steps.append(int(p.name[len("wavefield"):len("wavefield") + 7]))
        snaps.append(v.astype(np.float64))
    return grid.astype(np.float64), np.stack(snaps), steps


def run_ours(s: FS, want_steps):
    m = build_axis_aligned_quad9(nex=s.nex, nez=s.nez, x_max=s.x_max,
                                 z_max=s.z_max, n_order=s.n_order,
                                 dtype=DT, device=DEV)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    fluid = (zc < s.z_if).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fluid, s.vp_w, s.vp_s).to(DT).contiguous()
    vs = torch.where(fluid, 0.0, s.vs_s).to(DT).contiguous()
    rho = torch.where(fluid, s.rho_w, s.rho_s).to(DT).contiguous()
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho,
                                    dirichlet=("top", "left", "right"))
    xg = torch.zeros(m.n_global, dtype=DT, device=DEV); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT, device=DEV); zg[m.ibool] = m.z_e
    src = int(((xg - s.src_xz[0]) ** 2 + (zg - s.src_xz[1]) ** 2).argmin())
    pat = torch.zeros(m.n_global, dtype=DT, device=DEV); pat[src] = 1.0
    zero = torch.zeros(m.n_global, dtype=DT, device=DEV)
    t = np.arange(s.nstep) * s.dt
    a = (np.pi * s.f0 * (t - s.t0_internal)) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT, device=DEV)

    # our level k sits at physical time (k+1)*dt - t0; SPECFEM dump at solver
    # step "it" ~ physical it*dt - t0  ->  keep chi at k = it-2, it-1, it for
    # the centred second difference at level it-1.
    keep = {}
    for it in want_steps:
        for k in (it - 2, it - 1, it):
            keep.setdefault(k, None)
    lam, mu = eq.el._lam, eq.el._mu
    f = eq.init_state(s.dt, DEV, DT)
    with torch.no_grad():
        for n in range(s.nstep):
            f = eq.step(f, [pat * wav[n], zero, zero], s.dt)
            if n in keep:
                keep[n] = (f[0].clone(), f[2].clone(), f[3].clone())

    fields = {}
    fl_node = eq.domain.node_in_acoustic
    for it in want_steps:
        chi_m, _, _ = keep[it - 2]
        chi_0, ux, uz = keep[it - 1]
        chi_p, _, _ = keep[it]
        p_fluid = -(chi_p - 2 * chi_0 + chi_m) / (s.dt ** 2)
        dux_dx, _, _, duz_dz = phys_grads(
            eq.el._D, m.ibool, eq.el._ixx, eq.el._ixz, eq.el._iex, eq.el._iez,
            ux, uz)
        p_sol_e = -(lam + mu) * (dux_dx + duz_dz)          # (E, ngll, ngll)
        p_sol = torch.zeros(m.n_global, dtype=DT, device=DEV)
        p_sol[m.ibool] = p_sol_e                            # last-write display map
        p = torch.where(fl_node, p_fluid, p_sol)
        fields[it] = p.cpu().numpy()
    return m, (xg.cpu().numpy(), zg.cpu().numpy()), fields


def main():
    s = FS()
    grid, snaps, steps = load_specfem_dumps(ROOT / "OUTPUT_FILES")
    pick = [st for st in steps if st in (300, 600, 900, 1200)]
    m, (xg, zg), ours = run_ours(s, pick)

    from scipy.interpolate import griddata
    nxi, nzi = 400, 250
    gx, gz = np.meshgrid(np.linspace(0, s.x_max, nxi),
                         np.linspace(0, s.z_max, nzi))
    fig, axes = plt.subplots(len(pick), 2, figsize=(15, 3.6 * len(pick)))
    for r, it in enumerate(pick):
        t_phys = it * s.dt - s.t0_internal
        sp = snaps[steps.index(it)]
        # SPECFEM z is up: convert to our z-down frame for display
        img_sp = griddata((grid[:, 0], s.z_max - grid[:, 1]), sp, (gx, gz),
                          method="linear")
        img_us = griddata((xg, zg), ours[it], (gx, gz), method="linear")
        # one global scale (source-injection constant) via lstsq on the frame
        ok = np.isfinite(img_sp) & np.isfinite(img_us)
        alpha = float((img_sp[ok] * img_us[ok]).sum() / (img_us[ok] ** 2).sum())
        img_us = alpha * img_us
        vmax = np.nanpercentile(np.abs(img_sp), 99)
        for c, (img, ttl) in enumerate([(img_us, "sweep SEM"),
                                        (img_sp, "SPECFEM2D")]):
            ax = axes[r, c]
            ax.imshow(img, extent=[0, s.x_max, s.z_max, 0], cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, aspect="equal")
            ax.axhline(s.z_if, color="k", lw=0.6, ls="--")
            ax.plot(*s.src_xz, "k*", ms=8)
            ax.set_title(f"{ttl}  t={t_phys:.2f} s", fontsize=10)
            if c == 0:
                ax.set_ylabel("z (m)")
            if r == len(pick) - 1:
                ax.set_xlabel("x (m)")
        cos = float(np.nansum(img_sp[ok] * img_us[ok])
                    / (np.sqrt(np.nansum(img_sp[ok]**2))
                       * np.sqrt(np.nansum(img_us[ok]**2)) + 1e-300))
        print(f"t={t_phys:.2f}s: field cosine {cos:.5f}")
    fig.suptitle("Pressure wavefield: sweep SEM vs SPECFEM2D "
                 "(fluid over solid; dashed = interface)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "wavefield_fluidsolid_compare.png"
    fig.savefig(out, dpi=200)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()

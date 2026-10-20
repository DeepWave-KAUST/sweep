"""Stability test: free surface at BOTTOM of a uniform model.

Uniform medium on a flat axis-aligned Quad9 mesh. CPML on top, left,
right. Free surface at the bottom. We run two variants:

  * 'natural'  — drop the boundary integral (free_top=False analogue
                 in the bottom direction); no explicit σ zeroing
  * 'essential' — zero σ_zz, σ_xz at the bottom GLL row each step
                  (the bottom-flipped analogue of free_top=True)

We then look at: (a) seismograms at receivers just inside the free
surface, (b) peak |v_z| over successive 1-s windows to detect late-time
instability.

The geometry is intentionally symmetric to the hill_case-with-free-top
setup so any corner-instability the latter showed should reappear at
the BOTTOM-LEFT / BOTTOM-RIGHT corners here. If it does, the issue is
PML-corner-+-essential-BC and not specific to having the free surface
on the top edge.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.sem import (
    ElasticVSSEM2DDeformed,
    SEMPropagator,
    build_simple_elastic_cpml,
    build_axis_aligned_quad9,
)
from sweep.sem.propagator import build_point_stamp


# ----- geometry & physics ------------------------------------------------- #
X_MAX, Z_MAX = 1270.0, 890.0
NEX, NEZ     = 64, 44
N_ORDER      = 4
DT           = 1e-3
NSTEP        = 3000
F0, TSHIFT   = 12.0, 0.12
VP, VS, RHO  = 2200.0, 1100.0, 2000.0

# Source near the TOP of the domain — wave travels down to the free bottom.
SRC_XZ = (X_MAX / 2.0, 60.0)
SRC_MT = (1.0, 1.0, 0.0)
# Receivers along the bottom (z = Z_MAX - 10) so we measure the surface
# wave running along the free bottom; also a couple in the interior.
RECS = (
    [(x, Z_MAX - 10.0) for x in np.linspace(100.0, X_MAX - 100.0, 7)]
    + [(X_MAX / 2.0, 400.0), (X_MAX / 2.0, 700.0)]
)


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _build_mesh_and_models(dtype, device):
    mesh = build_axis_aligned_quad9(
        nex=NEX, nez=NEZ, x_max=X_MAX, z_max=Z_MAX,
        n_order=N_ORDER, dtype=dtype, device=device,
    )
    e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp_e  = torch.full(e, VP,  dtype=dtype, device=device)
    vs_e  = torch.full(e, VS,  dtype=dtype, device=device)
    rho_e = torch.full(e, RHO, dtype=dtype, device=device)
    return mesh, vp_e, vs_e, rho_e


def run(mode: str, mesh, vp_e, vs_e, rho_e):
    """mode: 'natural' or 'essential' — controls bottom σ zeroing."""
    cpml = build_simple_elastic_cpml(
        mesh, dt=DT, sides=("left", "right", "top"),    # ← CPML on TOP now
        pml_thickness_elements=8, c_max=VP, f_dominant=F0,
    )
    # free_top=False so the class never touches σ at the TOP edge.
    # We handle the BOTTOM ourselves inside this script.
    eq = ElasticVSSEM2DDeformed.build(
        mesh, vp_e, vs_e, rho_e, free_top=False, cpml=cpml,
    )

    # Bottom element ids (analogue of top_elem_ids but at ez = nez - 1).
    bottom_elem_ids = torch.arange(
        (mesh.nez - 1) * mesh.nex, mesh.nez * mesh.nex,
        device=mesh.x_e.device, dtype=torch.long,
    )
    # GLL row α=ngll-1 corresponds to η=+1, the bottom of the reference
    # element. For axis-aligned Q9 this is the bottom of the mesh.
    bottom_alpha = mesh.ngll - 1

    # ---- propagator inputs (replicate SEMPropagator logic) ----
    sources   = [build_point_stamp(mesh, *SRC_XZ)]
    receivers = [build_point_stamp(mesh, *r) for r in RECS]

    dtype  = eq.mass_rho.dtype
    device = eq.mass_rho.device
    inv_M  = 1.0 / eq.mass_rho

    v_x = torch.zeros(mesh.n_global, dtype=dtype, device=device)
    v_z = torch.zeros_like(v_x)
    shape_e = (mesh.n_elements, mesh.ngll, mesh.ngll)
    sigma_xx = torch.zeros(shape_e, dtype=dtype, device=device)
    sigma_zz = torch.zeros_like(sigma_xx)
    sigma_xz = torch.zeros_like(sigma_xx)

    cp = eq.cpml
    m_sxx_x = torch.zeros_like(sigma_xx)
    m_sxz_z = torch.zeros_like(sigma_xx)
    m_sxz_x = torch.zeros_like(sigma_xx)
    m_szz_z = torch.zeros_like(sigma_xx)
    m_vxx = torch.zeros_like(sigma_xx)
    m_vzz = torch.zeros_like(sigma_xx)
    m_vxz = torch.zeros_like(sigma_xx)
    m_vzx = torch.zeros_like(sigma_xx)

    t_arr = np.arange(NSTEP) * DT
    src_wave = torch.from_numpy(ricker(t_arr, F0, TSHIFT)).to(dtype=dtype, device=device)

    n_rec = len(receivers)
    seis_vx = torch.zeros(n_rec, NSTEP, dtype=dtype, device=device)
    seis_vz = torch.zeros(n_rec, NSTEP, dtype=dtype, device=device)

    for n in range(NSTEP):
        b_x = torch.zeros_like(v_x)
        b_z = torch.zeros_like(v_z)
        s_val = src_wave[n]
        for src in sources:
            stamp_x, stamp_z = src.moment_tensor_stamp(
                SRC_MT[0] * s_val, SRC_MT[1] * s_val, SRC_MT[2] * s_val
            )
            g = mesh.ibool[src.elem_id].reshape(-1)
            b_x.index_add_(0, g, stamp_x.reshape(-1))
            b_z.index_add_(0, g, stamp_z.reshape(-1))

        # ----- velocity update -----
        m_sxx_x = cp.a_x * m_sxx_x + cp.b_x * sigma_xx
        m_sxz_z = cp.a_z * m_sxz_z + cp.b_z * sigma_xz
        m_sxz_x = cp.a_x * m_sxz_x + cp.b_x * sigma_xz
        m_szz_z = cp.a_z * m_szz_z + cp.b_z * sigma_zz
        sxx_xpart = sigma_xx + m_sxx_x
        sxz_zpart = sigma_xz + m_sxz_z
        sxz_xpart = sigma_xz + m_sxz_x
        szz_zpart = sigma_zz + m_szz_z
        R_x, R_z = eq.assemble_stress_divergence(
            sxx_xpart, sxz_zpart, sxz_xpart, szz_zpart
        )
        v_x = v_x + DT * inv_M * (-R_x + b_x)
        v_z = v_z + DT * inv_M * (-R_z + b_z)

        # ----- stress update -----
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = eq.velocity_gradients(v_x, v_z)
        m_vxx = cp.a_x * m_vxx + cp.b_x * d_vx_dx
        m_vzz = cp.a_z * m_vzz + cp.b_z * d_vz_dz
        m_vxz = cp.a_z * m_vxz + cp.b_z * d_vx_dz
        m_vzx = cp.a_x * m_vzx + cp.b_x * d_vz_dx
        d_vx_dx_s = d_vx_dx + m_vxx
        d_vz_dz_s = d_vz_dz + m_vzz
        d_vx_dz_s = d_vx_dz + m_vxz
        d_vz_dx_s = d_vz_dx + m_vzx
        sigma_xx = sigma_xx + DT * (eq._lam2mu * d_vx_dx_s + eq._lam   * d_vz_dz_s)
        sigma_zz = sigma_zz + DT * (eq._lam   * d_vx_dx_s + eq._lam2mu * d_vz_dz_s)
        sigma_xz = sigma_xz + DT * eq._mu * (d_vx_dz_s + d_vz_dx_s)

        # ----- bottom free surface BC -----
        if mode == "essential":
            sigma_zz[bottom_elem_ids, bottom_alpha, :] = 0.0
            sigma_xz[bottom_elem_ids, bottom_alpha, :] = 0.0
        # 'natural' leaves σ alone — the dropped boundary integral
        # already enforces σ·n=0 weakly.

        # Receivers
        for ridx, r in enumerate(receivers):
            ibe = mesh.ibool[r.elem_id]
            seis_vx[ridx, n] = r.gather(v_x[ibe])
            seis_vz[ridx, n] = r.gather(v_z[ibe])

    return seis_vx.cpu().numpy(), seis_vz.cpu().numpy()


def main():
    out_dir = Path("_smoketest")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, vp_e, vs_e, rho_e = _build_mesh_and_models(
        dtype=torch.float64, device="cpu"
    )
    print(f"Mesh: {mesh.n_elements} elements, n_global={mesh.n_global}")
    print(f"NSTEP={NSTEP} ({NSTEP*DT:.2f} s sim time)")
    print()

    print("Running natural BC (drop boundary integral, no explicit zeroing)…")
    nx, nz = run("natural", mesh, vp_e, vs_e, rho_e)
    print(f"  peak |v_z| = {np.abs(nz).max():.3e}")

    print("Running essential BC (zero σ_zz, σ_xz at bottom GLL row)…")
    ex_, ez_ = run("essential", mesh, vp_e, vs_e, rho_e)
    print(f"  peak |v_z| = {np.abs(ez_).max():.3e}")

    # ----- stability check ----------------------------------------------- #
    print()
    print(f"Stability — peak |v_z| in successive 1-s windows:")
    print(f"{'window':>14s}  {'natural':>12s}  {'essential':>12s}  {'ratio':>10s}")
    for t0, t1 in [(0.0, 1.0), (1.0, 2.0), (2.0, NSTEP*DT)]:
        i0, i1 = int(t0/DT), int(t1/DT)
        n_pk = np.abs(nz[:, i0:i1]).max()
        e_pk = np.abs(ez_[:, i0:i1]).max()
        print(f"  [{t0:.1f}, {t1:.1f}] s  {n_pk:12.3e}  {e_pk:12.3e}  "
              f"{e_pk/max(n_pk,1e-300):10.2e}")

    # ----- per-receiver traces ------------------------------------------- #
    t_axis = np.arange(NSTEP) * DT
    fig, axes = plt.subplots(len(RECS), 2, figsize=(14, 1.8 * len(RECS)),
                             sharex=True)
    for ri, rxz in enumerate(RECS):
        ax_x, ax_z = axes[ri]
        ax_x.plot(t_axis, nx[ri],  "C2-",  lw=1.0, label="natural")
        ax_x.plot(t_axis, ex_[ri], "C3--", lw=0.9, label="essential")
        ax_x.set_title(f"rec {rxz} — v_x", fontsize=9)
        ax_x.legend(fontsize=7); ax_x.tick_params(labelsize=8)
        ax_z.plot(t_axis, nz[ri],  "C2-",  lw=1.0, label="natural")
        ax_z.plot(t_axis, ez_[ri], "C3--", lw=0.9, label="essential")
        ax_z.set_title(f"rec {rxz} — v_z", fontsize=9)
        ax_z.legend(fontsize=7); ax_z.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle(f"Uniform model + free bottom + CPML on top/left/right — "
                 f"NSTEP={NSTEP}", fontsize=11)
    fig.tight_layout()
    png = out_dir / "free_bottom_stability_traces.png"
    fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {png.resolve()}")


if __name__ == "__main__":
    main()

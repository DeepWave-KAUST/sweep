"""Stage-1 gates: SPECFEM-normalized variable-density acoustic (potential form).

Two gates, neither needs SPECFEM:

(1a) **Degeneracy** — for constant (κ, ρ) the new operators M=∫κ⁻¹, K=∫ρ⁻¹ give
     M⁻¹K = c²(∫φφ)⁻¹∫∇∇, identical to AcousticSEM2DDeformed's constant-density
     M=∫φφ, K=∫c²∇∇. So the homogeneous evolution must agree to roundoff.

(1b) **Variable density does real work** — a *pure density contrast* (same c,
     different ρ) reflects with R = (ρ₂c−ρ₁c)/(ρ₂c+ρ₁c) = (ρ₂−ρ₁)/(ρ₂+ρ₁).
     A constant-density formulation cannot even express this. We measure R in a
     **1-D waveguide** (one element wide, natural ∂χ/∂n=0 side walls, plane
     source) so there is NO geometrical spreading and R is exact — no far-field
     correction, no SPECFEM.
"""
import numpy as np
import pytest
import torch

from sweep.sem.equations.acoustic2d_deformed import AcousticSEM2DDeformed
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64


def _ricker(nt, dt, f0, t0):
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)


# --------------------------------------------------------------------------- #
# (1a) degeneracy at constant (kappa, rho)
# --------------------------------------------------------------------------- #
def test_constant_kappa_rho_reproduces_constant_density_class():
    m = build_axis_aligned_quad9(nex=6, nez=5, x_max=600.0, z_max=500.0,
                                 n_order=4, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    vp = torch.full(shp, 2000.0, dtype=DT)
    rho = torch.full(shp, 1000.0, dtype=DT)

    new = AcousticPotentialSEM2D.build(m, vp, rho, dirichlet=("top",))
    old = AcousticSEM2DDeformed.build(m, vp, dirichlet=("top",), cpml=None)

    # same smooth initial bump, zero initial velocity, no source
    xc, zc = 300.0, 250.0
    r2 = (m.x_e - xc) ** 2 + (m.z_e - zc) ** 2
    bump_e = torch.exp(-r2 / (2 * 80.0 ** 2))
    b0 = torch.zeros(m.n_global, dtype=DT)
    b0[m.ibool] = bump_e
    zero = torch.zeros(m.n_global, dtype=DT)
    dt = 3e-4

    f_new = (b0.clone(), b0.clone(), zero.clone(), zero.clone(),
             zero.clone(), zero.clone())
    f_old = (b0.clone(), b0.clone(), zero.clone(), zero.clone(),
             zero.clone(), zero.clone())
    for _ in range(60):
        f_new = new.step(f_new, [zero], dt)
        f_old = old.step(f_old, [zero], dt)

    amp = f_old[0].abs().max()
    assert amp > 1e-3                                   # it actually propagated
    rel = (f_new[0] - f_old[0]).norm() / f_old[0].norm()
    assert rel < 1e-12, f"degeneracy broken: rel={rel:.2e}"


# --------------------------------------------------------------------------- #
# (1b) pure density contrast in a 1-D waveguide -> exact R
# --------------------------------------------------------------------------- #
def _waveguide_reflection(rho_top, rho_bot):
    """1-D waveguide, plane source, pure density contrast. Returns (R, x_var)."""
    DE, NEZ, C = 50.0, 60, 2000.0
    Z_I, Z_S, Z_R = 2000.0, 1000.0, 1500.0          # interface / source / receiver
    m = build_axis_aligned_quad9(nex=2, nez=NEZ, x_max=2 * DE, z_max=NEZ * DE,
                                 n_order=4, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    vp = torch.full(shp, C, dtype=DT)                # constant c -> only rho differs
    zc = m.z_e.flatten(1).mean(1)                    # element centroid depth
    rho = torch.where(zc < Z_I, rho_top, rho_bot).reshape(-1, 1, 1).expand(shp)
    rho = rho.contiguous().to(DT)

    # natural (rigid) side walls everywhere -> 1-D waveguide; no free surface
    eq = AcousticPotentialSEM2D.build(m, vp, rho, dirichlet=())

    node_at = lambda z: m.ibool[(m.z_e - z).abs() < 1e-6].unique()
    src_nodes, rec_nodes = node_at(Z_S), node_at(Z_R)
    assert src_nodes.numel() > 0 and rec_nodes.numel() > 0

    dt, nt, f0 = 4e-4, 2500, 15.0
    wav = _ricker(nt, dt, f0, 0.12)
    # b proportional to the nodal mass -> inv_M*b uniform along the line -> a
    # pure plane (k_x = 0) mode, so there is no geometrical spreading at all.
    pat = torch.zeros(m.n_global, dtype=DT)
    pat[src_nodes] = eq.mass[src_nodes]

    f = eq.init_state(dt, "cpu", DT)
    trace, xspread = [], []
    for n in range(nt):
        f = eq.step(f, [pat * wav[n]], dt)
        chi_r = f[0][rec_nodes]
        trace.append(chi_r.mean())
        xspread.append((chi_r.max() - chi_r.min()).abs())
    tr = torch.stack(trace)
    t = np.arange(nt) * dt
    # direct: (Z_R-Z_S)/c = 0.25 s ; interface reflection: ((Z_I-Z_S)+(Z_I-Z_R))/c
    # = 0.75 s ; top "ghost" (source radiates up too): (Z_S+Z_R)/c = 1.25 s -> the
    # 0.6-0.95 window is clean.
    w_dir = (t > 0.12 + 0.15) & (t < 0.12 + 0.45)
    w_ref = (t > 0.12 + 0.60) & (t < 0.12 + 0.95)
    A_dir = tr[w_dir].abs().max()
    A_ref = tr[w_ref].abs().max()
    # signed: compare the extremum of matching polarity
    i_d = tr[w_dir].abs().argmax(); i_r = tr[w_ref].abs().argmax()
    sgn = torch.sign(tr[w_dir][i_d]) * torch.sign(tr[w_ref][i_r])
    return (sgn * A_ref / A_dir).item(), max(xspread).item() / A_dir.item()


def test_pure_density_contrast_reflection_coefficient():
    """R must equal (rho2-rho1)/(rho2+rho1) exactly (1-D waveguide, no spreading).
    A constant-density formulation would give R = 0 here — c is the same on both
    sides, so ONLY the density makes this interface reflect."""
    R, xvar = _waveguide_reflection(1000.0, 2000.0)
    assert xvar < 1e-8, f"not a pure plane mode (x-variation {xvar:.1e})"
    R_exact = (2000.0 - 1000.0) / (2000.0 + 1000.0)          # = 1/3
    assert abs(abs(R) - R_exact) / R_exact < 0.02, f"|R|={abs(R):.4f} vs 1/3"


def test_density_contrast_sign_flips_with_the_contrast():
    """Convention-independent check that R tracks the impedance jump."""
    R_up, _ = _waveguide_reflection(1000.0, 2000.0)     # denser below
    R_dn, _ = _waveguide_reflection(2000.0, 1000.0)     # lighter below
    assert abs(abs(R_up) - abs(R_dn)) < 0.02             # same magnitude
    assert R_up * R_dn < 0, f"sign did not flip: {R_up:.3f}, {R_dn:.3f}"

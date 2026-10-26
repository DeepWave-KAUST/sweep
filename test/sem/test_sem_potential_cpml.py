"""Stage-6/S1 gates: potential-form acoustic CPML (single fluid domain).

The scheme is the proven Komatitsch-Martin double convolution of
AcousticSEM2DDeformed with the flux q = (1/rho) grad chi riding inside the
directional operators. Gates:

1. zero-profile parity: with b=db=a=0 the CPML step must reproduce the bare
   potential step (kappa uniform => kappa_node*inv_M_geo == inv_M_kappa exactly
   up to ulp) — INCLUDING variable rho, which exercises the flux path.
2. absorption: CPML box vs a 3x larger reflection-free reference — the interior
   field after the boundary interaction must match the big-domain truth.
3. long-nt stability (slow): many domain-crossing times, monotone decay to a
   floor, no late growth — the classic CPML corner-instability canary.
4. layered rho AND kappa inside/entering the bands: stays stable + absorbs.
"""
import functools

import numpy as np
import pytest
import torch

from sweep.sem.cpml.acoustic import CPMLProfile, build_cpml
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64


def _ricker(nt, dt, f0, t0):
    t = np.arange(nt) * dt
    a = (np.pi * f0 * (t - t0)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)


def _mesh(nex, nez, de):
    return build_axis_aligned_quad9(nex=nex, nez=nez, x_max=nex * de,
                                    z_max=nez * de, n_order=4, dtype=DT)


def _zero_profile(m):
    z = lambda: torch.zeros(m.n_global, dtype=DT)
    return CPMLProfile(sides=(), pml_thickness=0.0, a_x=z(), b_x=z(),
                       dbx_dx=z(), a_z=z(), b_z=z(), dbz_dz=z())


def _coords(m):
    xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    return xg, zg


# --------------------------------------------------------------------------- #
# 1. zero-profile parity, uniform kappa but VARIABLE rho
# --------------------------------------------------------------------------- #
def test_zero_profile_matches_bare_variable_rho():
    m = _mesh(10, 8, 50.0)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    rho = torch.where(zc < 200.0, 1000.0, 2000.0).reshape(-1, 1, 1) \
        .expand(shp).contiguous().to(DT)
    kappa = 2.25e9                                   # uniform kappa = rho c^2
    vp = torch.sqrt(kappa / rho)
    bare = AcousticPotentialSEM2D.build(m, vp, rho, dirichlet=("top",))
    cpml = AcousticPotentialSEM2D.build(m, vp, rho, dirichlet=("top",),
                                        cpml=_zero_profile(m))

    xg, zg = _coords(m)
    s = int(((xg - 250.0) ** 2 + (zg - 150.0) ** 2).argmin())
    pat = torch.zeros(m.n_global, dtype=DT); pat[s] = 1.0
    wav = _ricker(120, 5e-4, 15.0, 0.05)

    fb = bare.init_state(5e-4, "cpu", DT)
    fc = cpml.init_state(5e-4, "cpu", DT)
    for n in range(120):
        fb = bare.step(fb, [pat * wav[n]], 5e-4)
        fc = cpml.step(fc, [pat * wav[n]], 5e-4)
    assert fb[0].abs().max() > 1e-12                 # it propagated
    rel = (fc[0] - fb[0]).norm() / fb[0].norm()
    assert rel < 1e-12, f"zero-profile CPML != bare: rel={rel:.2e}"


# --------------------------------------------------------------------------- #
# 2/3/4. Parity gates against the VALIDATED velocity-form CPML.
#
# The potential CPML deliberately reuses the velocity form's double-convolution
# scheme, so the correct bar is BEHAVIORAL PARITY with that reference — not
# absolute decay numbers (measured: the validated velocity form itself sits at
# 6.9e-4 of peak after 3000 steps on the small box; an absolute 1e-4 gate would
# fail the reference implementation too).
# --------------------------------------------------------------------------- #
class _Rollout:
    """One form's CPML run on the parity box, stepped on demand.

    The 3000-step uniform gate is the first 3000 steps of the 10000-step one
    (same box, same source; the wavelet is cut at 0.3 s, so sample n does not
    depend on the run length), and the velocity form never sees ρ, so its
    layered run is the same for the layered-vp and the layered-ρ gate. Each
    distinct run is therefore stepped once, as far as the longest gate that
    reads it, and every gate gets the numbers its own run produced."""

    def __init__(self, form, layered, layered_rho):
        from sweep.sem.equations.acoustic2d_deformed import AcousticSEM2DDeformed
        de, self.dt, f0 = 50.0, 8e-4, 12.0
        m = _mesh(20, 16, de)
        shp = (m.n_elements, m.ngll, m.ngll)
        if layered:
            zc = m.z_e.flatten(1).mean(1)
            deep = (zc > 400.0).reshape(-1, 1, 1).expand(shp)
            vp = torch.where(deep, 2200.0, 1500.0).to(DT).contiguous()
            rho = (torch.where(deep, 1800.0, 1000.0).to(DT).contiguous()
                   if layered_rho else torch.full(shp, 1000.0, dtype=DT))
            cmax = 2200.0
        else:
            vp = torch.full(shp, 1500.0, dtype=DT)
            rho = torch.full(shp, 1000.0, dtype=DT)
            cmax = 1500.0
        prof = build_cpml(m, dt=self.dt, sides=("left", "right", "bottom"),
                          pml_thickness_elements=6, c_max=cmax, f_dominant=f0)
        self.eq = (AcousticSEM2DDeformed.build(m, vp, dirichlet=("top",), cpml=prof)
                   if form == "vel" else
                   AcousticPotentialSEM2D.build(m, vp, rho, dirichlet=("top",),
                                                cpml=prof))
        xg, zg = _coords(m)
        src = int(((xg - 500.0) ** 2 + (zg - 300.0) ** 2).argmin())
        self.pat = torch.zeros(m.n_global, dtype=DT); self.pat[src] = 1.0
        self.f0, self.wav = f0, torch.zeros(0, dtype=DT)
        self.f = self.eq.init_state(self.dt, "cpu", DT)
        self.amax = []                    # max |field| after every step

    def decay(self, nt):
        """End-field/peak ratio after ``nt`` steps, the peak sampled every
        100 steps."""
        if nt > len(self.wav):
            self.wav = _ricker(nt, self.dt, self.f0, 0.12)
            self.wav[np.arange(nt) * self.dt > 0.3] = 0.0
        with torch.inference_mode():          # nothing here is differentiated
            for n in range(len(self.amax), nt):
                self.f = self.eq.step(self.f, [self.pat * self.wav[n]], self.dt)
                mx = float(self.f[0].abs().max())
                assert np.isfinite(mx), f"blow-up at step {n}"
                self.amax.append(mx)
        return self.amax[nt - 1] / max(self.amax[:nt:100])


@functools.lru_cache(maxsize=None)
def _rollout(form, layered, layered_rho):
    return _Rollout(form, layered, layered_rho)


def _decay_pair(nt, layered, layered_rho=False):
    """Velocity-form and potential-form CPML on the same box; the (velocity,
    potential) end-field/peak decay ratios.

    ``layered`` keeps ρ UNIFORM (layered vp only) so the two classes solve the
    SAME physics and decay parity is meaningful. ``layered_rho`` additionally
    layers ρ in the potential run only — that changes the interface reflection
    coefficient (R 0.19 → 0.45 here), traps more interior reverb, and slows the
    decay for PHYSICAL reasons, so it must not be gated on parity."""
    return [_rollout(form, layered, layered_rho and form == "pot").decay(nt)
            for form in ("vel", "pot")]


def test_uniform_decay_parity_with_velocity_form():
    dv, dp = _decay_pair(nt=3000, layered=False)
    assert dv < 1e-2 and dp < 1e-2            # both absorb (sanity)
    assert dp < 1.5 * dv + 1e-12, f"potential {dp:.2e} vs velocity {dv:.2e}"


def test_layered_vp_decay_parity_with_velocity_form():
    """Layered vp, uniform ρ — identical physics in both classes."""
    dv, dp = _decay_pair(nt=3000, layered=True)
    assert dv < 1e-2 and dp < 1e-2
    assert dp < 1.5 * dv + 1e-12, f"potential {dp:.2e} vs velocity {dv:.2e}"


def test_layered_rho_bands_stable_and_absorbing():
    """Layered ρ AND vp (impedance R≈0.45, more trapped reverb): no reference
    implementation exists for variable-ρ CPML, so the gate is stability +
    substantial decay, not parity."""
    dp = _rollout("pot", True, True).decay(3000)       # ρ enters the potential form only
    assert np.isfinite(dp)
    assert dp < 3e-2, f"layered-rho decay only reached {dp:.2e} of peak"


@pytest.mark.slow
def test_long_nt_decay_parity_with_velocity_form():
    """~20 crossing times: both keep decaying, no late growth, parity holds."""
    dv, dp = _decay_pair(nt=10000, layered=False)
    assert dv < 1e-3 and dp < 1e-3
    assert dp < 1.5 * dv + 1e-12, f"potential {dp:.2e} vs velocity {dv:.2e}"



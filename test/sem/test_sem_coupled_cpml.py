"""Stage-6/S3 gates: interface CPML on the coupled fluid-solid equation.

Marine geometry: water over solid, the interface crossing the left/right
X-only absorbing bands (unavoidable — the water layer reaches the model edge),
bottom band entirely in the solid, free surface on top.

G2 (the hard gate): long-nt stability with a near-interface source (excites
    interface/Scholte energy — the plain coupling term inside the X-only bands
    is the stability question). Finite, decays, no late growth.
G3: interface-leakage budget — late-window field on Γ vs its peak. The plain
    coupling costs absorption quality AT Γ; this quantifies it.
Also: the corner rule fires for this geometry only when the interface is
    pushed into the bottom band (guards the build-time check end-to-end).
"""
import functools

import numpy as np
import pytest
import torch

from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64
NEX, NEZ, DE = 20, 16, 50.0
Z_IF = 8 * DE                                    # 400 m: mid-height, no corner


def _build(z_if=Z_IF, pml_elems=4, dt=6e-4):
    m = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE,
                                 z_max=NEZ * DE, n_order=4, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    fl = (zc < z_if).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fl, 1500.0, 3000.0).to(DT).contiguous()
    vs = torch.where(fl, 0.0, 1732.0).to(DT).contiguous()
    rho = torch.where(fl, 1000.0, 2200.0).to(DT).contiguous()
    sides = ("left", "right", "bottom")
    cp_ac = build_cpml(m, dt=dt, sides=sides, pml_thickness_elements=pml_elems,
                       c_max=1500.0, f_dominant=12.0)       # fluid speed
    cp_el = build_elastic_cpml(m, dt=dt, sides=sides,
                               pml_thickness_elements=pml_elems,
                               c_max=3000.0, f_dominant=12.0)  # solid speed
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho, dirichlet=("top",),
                                    cpml_ac=cp_ac, cpml_el=cp_el)
    return m, eq


class _Rollout:
    """Near-interface water source (Scholte excitation) on ``_build()``,
    stepped on demand and monitored every ``monitor_every`` steps.

    G2 (3000 steps), G3 (6000) and the long gate (20000) used to run this
    same mesh, model, source and wavelet three times over. Their records are
    prefixes of ONE trajectory -- the wavelet is cut at 0.3 s, so sample n
    does not depend on the run length -- so each gate reads the first ``nt``
    steps of a single rollout: the numbers its own run produced, at the cost
    of the longest gate alone."""

    def __init__(self, dt=6e-4, src_z=350.0, monitor_every=200):
        self.m, self.eq = _build(dt=dt)
        m = self.m
        xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
        zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
        s = int(((xg - 500.0) ** 2 + (zg - src_z) ** 2).argmin())
        self.pat = torch.zeros(m.n_global, dtype=DT); self.pat[s] = 1.0
        self.zero = torch.zeros(m.n_global, dtype=DT)
        self.on_if = (zg - Z_IF).abs() < 1e-6
        self.dt, self.every = dt, monitor_every
        self.wav = self._wavelet(0)
        self.f = self.eq.init_state(dt, "cpu", DT)
        self.n, self.blowup = 0, None
        self.rec = []                     # (step, max |field|, max |u_z on Γ|)

    def _wavelet(self, nt):
        t = np.arange(nt) * self.dt
        a = (np.pi * 12.0 * (t - 0.12)) ** 2
        wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)
        wav[t > 0.3] = 0.0
        return wav

    def window(self, nt):
        """``(peak, series, peak_if, series_if)`` of the first ``nt`` steps:
        the peaks, and the monitored series after the source is off."""
        assert self.blowup is None or nt <= self.blowup, \
            f"blow-up at step {self.blowup} (t={self.blowup*self.dt:.2f}s)"
        if nt > len(self.wav):
            self.wav = self._wavelet(nt)
        dt = self.dt
        with torch.inference_mode():          # nothing here is differentiated
            for n in range(self.n, nt):
                self.f = f = self.eq.step(
                    self.f, [self.pat * self.wav[n], self.zero, self.zero], dt)
                self.n = n + 1
                if n % self.every == 0:
                    mx = max(float(f[0].abs().max()),
                             float(f[2].abs().max()), float(f[3].abs().max()))
                    if not np.isfinite(mx):
                        self.blowup = n
                        raise AssertionError(
                            f"blow-up at step {n} (t={n*dt:.2f}s)")
                    self.rec.append((n, mx, float(f[3][self.on_if].abs().max())))
        peak = peak_if = 0.0
        series, series_if = [], []
        for n, mx, uz_if in self.rec:
            if n >= nt:
                break
            peak, peak_if = max(peak, mx), max(peak_if, uz_if)
            if n * dt > 0.5:
                series.append(mx)
                series_if.append(uz_if)
        return peak, series, peak_if, series_if


@functools.lru_cache(maxsize=None)
def _rollout():
    return _Rollout()


def test_g2_stability_interface_through_xonly_bands():
    """3000 steps (~5 crossings): decays, no growth."""
    peak, series, *_ = _rollout().window(3000)
    assert peak > 0
    assert series[-1] < 3e-2 * peak, f"weak decay: {series[-1]/peak:.2e}"


@pytest.mark.slow
def test_g2_long_nt_hard_gate():
    """20000 steps (~35 crossing times) — the CPML corner-instability canary
    for the coupled system with plain coupling inside the X-only bands."""
    peak, series, *_ = _rollout().window(20000)
    assert peak > 0
    tail = series[len(series) * 3 // 4:]
    assert max(tail) < 1e-3 * peak, f"late field {max(tail)/peak:.2e} of peak"
    assert tail[-1] <= 3.0 * tail[0] + 1e-300, "late-time growth"


def test_g3_interface_leakage_budget():
    """Scholte/interface energy at Γ after the reverberations have been
    absorbed: the plain-coupling penalty, quantified. Where the interface
    crosses the X-only absorbing bands the coupling term is left PLAIN (no
    stretch) — SPECFEM's choice: it is unconditionally stable but caps the Γ
    absorption around −38 dB (measured). The gate guards against gross leakage
    or a slow interface-guided instability, not against that floor: the late Γ
    field must sit ≥34 dB below its own peak (amplitude ratio < 2e-2)."""
    _, _, peak_if, series_if = _rollout().window(6000)
    assert peak_if > 0
    late = max(series_if[len(series_if) * 3 // 4:])
    ratio = late / peak_if
    assert ratio < 2e-2, f"Γ leakage {20*np.log10(ratio + 1e-300):.1f} dB " \
                         f"(budget −34 dB; plain-coupling floor ≈ −38 dB)"


def test_corner_rule_fires_end_to_end():
    with pytest.raises(ValueError, match="CORNER absorbing region"):
        _build(z_if=(NEZ - 2) * DE)          # interface inside the bottom band


def test_profiles_must_come_in_pairs():
    m = build_axis_aligned_quad9(nex=4, nez=4, x_max=200.0, z_max=200.0,
                                 n_order=4, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    vp = torch.full(shp, 1500.0, dtype=DT)
    vs = torch.zeros(shp, dtype=DT); vs[8:] = 1732.0
    rho = torch.full(shp, 1000.0, dtype=DT)
    cp = build_cpml(m, dt=6e-4, sides=("left",), pml_thickness_elements=2,
                    c_max=1500.0, f_dominant=12.0)
    with pytest.raises(ValueError, match="both cpml_ac and cpml_el"):
        AcousticElasticSEM2D.build(m, vp, vs, rho, cpml_ac=cp)

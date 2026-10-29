"""The C-PML band split of the eager 2-D elastic steps (``sweep.sem.cpml.band``).

``ElasticSEM2DDeformed``, ``ElasticVSSEM2DDeformed`` (and ``ElasticVTISEM2DDeformed``
through it) and ``ElasticVSSEM3D`` (and ``ElasticVSSEM3DDeformed`` through it)
run the bare operators on the interior elements, the C-PML terms on
the band elements only, and keep their memory fields on the band rows
(``band2`` / ``band``). Outside the band every C-PML coefficient is an exact
no-op, so this is the full-mesh step it replaced, rearranged: checked here
against that step, kept below as the oracle, to fp64 round-off -- records and
every model gradient, full tape and boundary saving, flat and with topography,
and the essential free surface of the VS form.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt, Full
from sweep.sem import (PropSEM, ElasticSEM2DDeformed, ElasticVSSEM2DDeformed,
                       ElasticVTISEM2DDeformed, ElasticVSSEM3D, ElasticVSSEM3DDeformed)
from sweep.sem._assembly import assemble
from sweep.sem.fields import FieldSpec

SHAPE, NT, DT = (25, 41), 140, 8e-4


def _on_every_element(specs):
    return [FieldSpec(s.name, s.supports_source, s.supports_receiver, s.internal,
                      s.boundary_related, s.seis_key, s.snap_key,
                      {"band": "element", "band2": "element2"}.get(s.layout, s.layout))
            for s in specs]


class _FullMesh(ElasticSEM2DDeformed):
    """The full-mesh formulation (memories on every element), as it was."""

    field_specs = _on_every_element(ElasticSEM2DDeformed.field_specs)

    def _step_cpml(self, u_x_now, u_z_now, u_x_prev, u_z_prev,
                   r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
                   r_displ_x, r_displ_z, b_x, b_z, dt):
        cp = self.cpml
        dt2, inv_M, ib = dt * dt, self._inv_M, self.mesh.ibool
        ixx, ixz, iex, iez = self._ixx, self._ixz, self._iex, self._iez
        U_e = torch.stack([u_x_now, u_z_now, u_x_prev, u_z_prev], dim=-1)[ib]
        dU_dxi = torch.einsum("bg,eagk->eabk", self._D, U_e)
        dU_deta = torch.einsum("ag,egbk->eabk", self._D, U_e)
        dU_dx = ixx.unsqueeze(-1) * dU_dxi + iex.unsqueeze(-1) * dU_deta
        dU_dz = ixz.unsqueeze(-1) * dU_dxi + iez.unsqueeze(-1) * dU_deta
        dux_dx, duz_dx = dU_dx[..., 0], dU_dx[..., 1]
        dux_dz, duz_dz = dU_dz[..., 0], dU_dz[..., 1]
        dux_dx_o, duz_dx_o = dU_dx[..., 2], dU_dx[..., 3]
        dux_dz_o, duz_dz_o = dU_dz[..., 2], dU_dz[..., 3]
        ux_e, uz_e, ux_e_o, uz_e_o = (U_e[..., k] for k in range(4))
        r_dux_dx = self._conv_update(r_dux_dx, cp.zx1, cp.zx2, dux_dx, dux_dx_o)
        r_duz_dx = self._conv_update(r_duz_dx, cp.zx1, cp.zx2, duz_dx, duz_dx_o)
        r_dux_dz = self._conv_update(r_dux_dz, cp.xz1, cp.xz2, dux_dz, dux_dz_o)
        r_duz_dz = self._conv_update(r_duz_dz, cp.xz1, cp.xz2, duz_dz, duz_dz_o)
        dux_dx_s = cp.Sx0 * dux_dx + cp.Sx1 * r_dux_dx[..., 0] + cp.Sx2 * r_dux_dx[..., 1]
        duz_dx_s = cp.Sx0 * duz_dx + cp.Sx1 * r_duz_dx[..., 0] + cp.Sx2 * r_duz_dx[..., 1]
        dux_dz_s = cp.Sz0 * dux_dz + cp.Sz1 * r_dux_dz[..., 0] + cp.Sz2 * r_dux_dz[..., 1]
        duz_dz_s = cp.Sz0 * duz_dz + cp.Sz1 * r_duz_dz[..., 0] + cp.Sz2 * r_duz_dz[..., 1]
        sig_xx = self._lam2mu * dux_dx_s + self._lam * duz_dz
        sig_zz = self._lam2mu * duz_dz_s + self._lam * dux_dx
        sig_zx = self._mu * (duz_dx + dux_dz_s)
        sig_xz = self._mu * (dux_dz + duz_dx_s)
        R_x, R_z = self._assemble_stress(sig_xx, sig_zz, sig_xz, sig_zx, u_x_now)
        r_displ_x = self._conv_update(r_displ_x, cp.l1, cp.l2, ux_e, ux_e_o)
        r_displ_z = self._conv_update(r_displ_z, cp.l1, cp.l2, uz_e, uz_e_o)
        v_x_e, v_z_e = (ux_e - ux_e_o) / dt, (uz_e - uz_e_o) / dt
        wrhoJ = self._w_rho_J
        aPML_x_e = wrhoJ * (cp.La1 * v_x_e + cp.La2 * ux_e
                            + cp.La3 * r_displ_x[..., 0] + cp.La4 * r_displ_x[..., 1])
        aPML_z_e = wrhoJ * (cp.La1 * v_z_e + cp.La2 * uz_e
                            + cp.La3 * r_displ_z[..., 0] + cp.La4 * r_displ_z[..., 1])
        aPML_x, aPML_z = assemble(torch.stack([aPML_x_e, aPML_z_e], dim=-1), ib, u_x_now)
        u_x_next = 2.0 * u_x_now - u_x_prev + dt2 * inv_M * (-R_x - aPML_x + b_x)
        u_z_next = 2.0 * u_z_now - u_z_prev + dt2 * inv_M * (-R_z - aPML_z + b_z)
        return (u_x_next, u_z_next, r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
                r_displ_x, r_displ_z)


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _run(eq, memory=Full(), topo=False, fs=True, split=None):
    x = np.arange(SHAPE[1])
    kw = dict(shape=SHAPE, dh=10.0, dt=DT, n_order=4, abcn=8, f_dominant=15.0, dev="cpu",
              dtype=torch.float64, impl="eager", source_type=["sxx", "szz"],
              receiver_type=["vx", "vz"], free_surface=fs, memory=memory)
    if split is not None:
        kw["equation_options"] = {"band_split": split}
    if topo:
        kw["topography"] = 3.0 - 3.0 * np.exp(-((x - SHAPE[1] / 2) / (SHAPE[1] / 8)) ** 2)
    p = PropSEM(eq, **kw)
    g = torch.Generator().manual_seed(1)
    ms = [(v * (1 + 0.05 * torch.rand(SHAPE, generator=g, dtype=torch.float64))).requires_grad_(True)
          for v in (2400.0, 1300.0, 2000.0)]
    t = torch.arange(NT, dtype=torch.float64) * DT - 0.08
    w = (1 - 2 * (math.pi * 15 * t) ** 2) * torch.exp(-(math.pi * 15 * t) ** 2)
    rx = np.arange(3, 38, 3)
    r = p(w, np.array([[12, 5.0]]), np.stack([rx, np.full_like(rx, 4)], 1)[None], models=ms)
    (r / r.detach().abs().max()).square().sum().backward()
    return r.detach(), [m.grad for m in ms]


def _rel(a, b):
    return float((a - b).norm() / b.norm())


_ORACLE_RUNS = {}


def _oracle(run, *args):
    """``run(*args)`` of an oracle, once per configuration: the split and the
    unsplit test of a configuration compare against the same run."""
    key = (run.__name__,) + tuple(a.__name__ if isinstance(a, type) else
                                  type(a).__name__ if isinstance(a, (Full, BoundarySaving)) else a
                                  for a in args)
    if key not in _ORACLE_RUNS:
        _ORACLE_RUNS[key] = run(*args)
    return _ORACLE_RUNS[key]


@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
@pytest.mark.parametrize("memory", [Full(), BoundarySaving()], ids=["full", "boundary_saving"])
@pytest.mark.parametrize("topo, fs", [(False, True), (True, True), (False, ("top", "bottom"))],
                         ids=["flat", "topography", "free_top_and_bottom"])
def test_equals_the_full_mesh_step(memory, topo, fs, split):
    r1, g1 = _run(ElasticSEM2DDeformed, memory, topo, fs, split)
    r0, g0 = _oracle(_run, _FullMesh, memory, topo, fs)
    assert _rel(r1, r0) < 1e-12
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-12


def test_the_split():
    from sweep.sem import build_axis_aligned_quad9
    from sweep.sem.cpml.elastic import build_elastic_cpml
    m = build_axis_aligned_quad9(nex=20, nez=10, x_max=2000.0, z_max=1000.0)
    cp = build_elastic_cpml(m, dt=6e-4, sides=("left", "right", "bottom"),
                            pml_thickness_elements=3, c_max=3000.0, f_dominant=14.0)
    full = lambda v: torch.full((m.n_elements, 5, 5), v, dtype=m.w_jac.dtype)
    # this mesh is far below SPLIT_MIN_INTERIOR: unsplit unless asked
    auto = ElasticSEM2DDeformed.build(m, full(2500.0), full(1400.0), full(2000.0), cpml=cp)
    assert auto.band_elements.numel() == m.n_elements and auto.interior_elements.numel() == 0
    eq = ElasticSEM2DDeformed.build(m, full(2500.0), full(1400.0), full(2000.0), cpml=cp,
                                    band_split=True)
    band, inner = eq.band_elements, eq.interior_elements
    assert band.numel() + inner.numel() == m.n_elements
    assert not bool(torch.isin(band, inner).any())
    assert band.numel() == 20 * 10 - (20 - 6) * (10 - 3)       # the three sides' band
    # the exactness the split rests on: no-op coefficients on every interior element
    for n in ("Sx0", "Sz0"):
        assert bool((getattr(cp, n)[inner] == 1).all())
    for n in ("Sx1", "Sx2", "Sz1", "Sz2", "La1", "La2", "La3", "La4"):
        assert bool((getattr(cp, n)[inner] == 0).all())
    for n in ("zx1", "zx2", "xz1", "xz2", "l1", "l2"):
        assert all(bool((t[inner] == 0).all()) for t in getattr(cp, n))
    st = eq.init_state(6e-4, "cpu", torch.float64)
    assert all(tuple(st[k].shape) == (band.numel(), 5, 5, 2) for k in range(4, 10))


class _FullMeshVSStep:
    """The velocity-stress step on every element, as it was."""

    field_specs = _on_every_element(ElasticVSSEM2DDeformed.field_specs)

    def reverse_step(self, fields, src, dt):
        out = ElasticVSSEM2DDeformed.reverse_step(self, fields, src, dt)
        return out[:5] + tuple(torch.zeros_like(fields[2]) for _ in range(8))

    def step(self, fields, src, dt):
        (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_vxx, m_vzz, m_vxz, m_vzx) = fields
        cp = self.cpml
        inv_M = self._inv_M
        b_x, b_z = src[0], src[1]
        m_sxx_x = cp.a_x * m_sxx_x + cp.b_x * sigma_xx
        m_sxz_z = cp.a_z * m_sxz_z + cp.b_z * sigma_xz
        m_sxz_x = cp.a_x * m_sxz_x + cp.b_x * sigma_xz
        m_szz_z = cp.a_z * m_szz_z + cp.b_z * sigma_zz
        R_x, R_z = self.assemble_stress_divergence(
            sigma_xx + m_sxx_x, sigma_xz + m_sxz_z, sigma_xz + m_sxz_x, sigma_zz + m_szz_z)
        v_x = v_x + dt * inv_M * (-R_x + b_x)
        v_z = v_z + dt * inv_M * (-R_z + b_z)
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = self.velocity_gradients(v_x, v_z)
        m_vxx = cp.a_x * m_vxx + cp.b_x * d_vx_dx
        m_vzz = cp.a_z * m_vzz + cp.b_z * d_vz_dz
        m_vxz = cp.a_z * m_vxz + cp.b_z * d_vx_dz
        m_vzx = cp.a_x * m_vzx + cp.b_x * d_vz_dx
        inc_xx, inc_zz, inc_xz = self._stress_increment(
            d_vx_dx + m_vxx, d_vx_dz + m_vxz, d_vz_dx + m_vzx, d_vz_dz + m_vzz, dt)
        sigma_xx, sigma_zz, sigma_xz = sigma_xx + inc_xx, sigma_zz + inc_zz, sigma_xz + inc_xz
        self.apply_free_top(sigma_xx, sigma_zz, sigma_xz)
        if self.free_top:
            for m in (m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_vxx, m_vzz, m_vxz, m_vzx):
                m[self.top_elem_ids, 0, :] = 0.0
        return (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_vxx, m_vzz, m_vxz, m_vzx)


class _FullMeshVS(_FullMeshVSStep, ElasticVSSEM2DDeformed):
    pass


class _FullMeshVTI(_FullMeshVSStep, ElasticVTISEM2DDeformed):
    pass


def _run_vs(eq, vals, memory=Full(), topo=False, split=None):
    x = np.arange(SHAPE[1])
    kw = dict(shape=SHAPE, dh=10.0, dt=6e-4, n_order=4, abcn=8, f_dominant=15.0, dev="cpu",
              dtype=torch.float64, impl="eager", source_type=["sxx", "szz"],
              receiver_type=["vx", "vz"], free_surface=True, memory=memory)
    if split is not None:
        kw["equation_options"] = {"band_split": split}
    if topo:
        kw["topography"] = 3.0 - 3.0 * np.exp(-((x - SHAPE[1] / 2) / (SHAPE[1] / 8)) ** 2)
    p = PropSEM(eq, **kw)
    g = torch.Generator().manual_seed(2)
    ms = [(v * (1 + 0.05 * torch.rand(SHAPE, generator=g, dtype=torch.float64))).requires_grad_(True)
          for v in vals]
    t = torch.arange(160, dtype=torch.float64) * 6e-4 - 0.08
    w = (1 - 2 * (math.pi * 15 * t) ** 2) * torch.exp(-(math.pi * 15 * t) ** 2)
    rx = np.arange(3, 38, 3)
    r = p(w, np.array([[12, 5.0]]), np.stack([rx, np.full_like(rx, 4)], 1)[None], models=ms)
    (r / r.detach().abs().max()).square().sum().backward()
    return r.detach(), [m.grad for m in ms]


@pytest.mark.parametrize("eq, oracle, vals", [
    (ElasticVSSEM2DDeformed, _FullMeshVS, (2400.0, 1300.0, 2000.0)),
    (ElasticVTISEM2DDeformed, _FullMeshVTI, (2400.0, 1300.0, 2000.0, 0.1, 0.05))], ids=["vs", "vti"])
@pytest.mark.parametrize("memory", [Full(), BoundarySaving()], ids=["full", "boundary_saving"])
@pytest.mark.parametrize("topo", [False, True], ids=["flat", "topography"])
@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
def test_vs_equals_the_full_mesh_step(eq, oracle, vals, memory, topo, split):
    r1, g1 = _run_vs(eq, vals, memory, topo, split)
    r0, g0 = _oracle(_run_vs, oracle, vals, memory, topo)
    assert _rel(r1, r0) < 1e-12
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-12


@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
def test_vs_essential_free_surface(split):
    """free_top=True: the stress BC on the top row and the memory taper there,
    which addresses the top elements through their band rows."""
    from sweep.sem import build_axis_aligned_quad9
    from sweep.sem.cpml.elastic_vs import build_simple_elastic_cpml
    m = build_axis_aligned_quad9(nex=12, nez=8, x_max=1200.0, z_max=800.0, dtype=torch.float64)
    cp = build_simple_elastic_cpml(m, dt=5e-4, sides=("left", "right", "bottom"),
                                   pml_thickness_elements=3, c_max=2500.0, f_dominant=15.0)
    full = lambda v: torch.full((m.n_elements, 5, 5), v, dtype=torch.float64)
    pat = torch.zeros(m.n_global, dtype=torch.float64)
    pat[m.ibool[40, 2, 2]] = 1.0
    out = {}
    for cls in (ElasticVSSEM2DDeformed, _FullMeshVS):
        vp = full(2400.0).requires_grad_(True)
        eq = cls.build(m, vp, full(1300.0), full(2000.0), free_top=True, cpml=cp,
                       band_split=split if cls is ElasticVSSEM2DDeformed else False)
        st = eq.init_state(5e-4, "cpu", torch.float64)
        loss = 0
        for n in range(200):
            s = math.exp(-((n * 5e-4 - 0.03) / 0.008) ** 2)
            st = eq.step(st, [pat * s, pat * 0.5 * s], 5e-4)
            loss = loss + st[0][m.ibool[2, 0, :]].square().sum()     # a corner element's top row
        loss.backward()
        out[cls] = (torch.stack(st[:2]).detach(), torch.stack(st[2:5]).detach(), vp.grad)
    for a, b in zip(out[ElasticVSSEM2DDeformed], out[_FullMeshVS]):
        assert _rel(a, b) < 1e-12


class _FullMeshVS3DStep:
    """The 3-D velocity-stress step on every element, as it was."""

    field_specs = _on_every_element(ElasticVSSEM3D.field_specs)

    def reverse_step(self, fields, src, dt):
        out = ElasticVSSEM3D.reverse_step(self, fields, src, dt)
        return out[:9] + tuple(torch.zeros_like(fields[3]) for _ in range(18))

    def step(self, fields, src, dt):
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, *m) = fields
        cp, inv_M = self.cpml, self._inv_M
        ax, bx, ay, by, az, bz = cp.a_x, cp.b_x, cp.a_y, cp.b_y, cp.a_z, cp.b_z
        coef = [(ax, bx), (ay, by), (az, bz), (ax, bx), (ay, by), (az, bz), (ax, bx), (ay, by), (az, bz)]
        sig = [sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz]
        ms = [a * mm + b * s for (a, b), mm, s in zip(coef, m[:9], sig)]
        R = self.assemble_stress_divergence(*(s + mm for s, mm in zip(sig, ms)))
        vx = vx + dt * inv_M * (-R[0] + src[0])
        vy = vy + dt * inv_M * (-R[1] + src[1])
        vz = vz + dt * inv_M * (-R[2] + src[2])
        g = self.velocity_gradients(vx, vy, vz)
        gcoef = [(ax, bx), (ay, by), (az, bz), (ax, bx), (ay, by), (ax, bx), (az, bz), (ay, by), (az, bz)]
        mv = [a * mm + b * gg for (a, b), mm, gg in zip(gcoef, m[9:], g)]
        inc = self._stress_increment(tuple(gg + mm for gg, mm in zip(g, mv)), dt)
        return (vx, vy, vz, sxx + inc[0], syy + inc[1], szz + inc[2], sxy + inc[3],
                sxz + inc[4], syz + inc[5], *ms, *mv)


class _FullMeshVS3D(_FullMeshVS3DStep, ElasticVSSEM3D):
    pass


class _FullMeshVS3DDeformed(_FullMeshVS3DStep, ElasticVSSEM3DDeformed):
    pass


def _run_vs3d(eq, memory=Full(), topo=False, split=None):
    shape = (13, 13, 17)
    nz, ny, nx = shape
    kw = dict(shape=shape, dh=10.0, dt=8e-4, n_order=4, abcn=8, f_dominant=18.0, dev="cpu",
              dtype=torch.float64, impl="eager", free_surface=True, memory=memory)
    if split is not None:
        kw["equation_options"] = {"band_split": split}
    if topo:
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        kw["topography"] = 2.0 - 2.0 * np.exp(-(((xx - nx / 2) / 4.0) ** 2 + ((yy - ny / 2) / 4.0) ** 2))
    p = PropSEM(eq, **kw)
    g = torch.Generator().manual_seed(3)
    ms = [(v * (1 + 0.05 * torch.rand(shape, generator=g, dtype=torch.float64))).requires_grad_(True)
          for v in (2400.0, 1300.0, 2000.0)]
    t = torch.arange(70, dtype=torch.float64) * 8e-4 - 0.06
    w = (1 - 2 * (math.pi * 18 * t) ** 2) * torch.exp(-(math.pi * 18 * t) ** 2)
    rx = np.arange(3, nx - 3, 2)
    rec = np.stack([rx, np.full_like(rx, ny // 2), np.full_like(rx, 3)], 1)[None]
    r = p(w, np.array([[nx // 2, ny // 2, 4.0]]), rec, models=ms)
    (r / r.detach().abs().max()).square().sum().backward()
    return r.detach(), [m.grad for m in ms]


@pytest.mark.parametrize("eq, oracle, memory, topo", [
    (ElasticVSSEM3D, _FullMeshVS3D, Full(), False),
    (ElasticVSSEM3D, _FullMeshVS3D, BoundarySaving(), False),
    (ElasticVSSEM3DDeformed, _FullMeshVS3DDeformed, Full(), True)],
    ids=["hex8", "hex8_boundary_saving", "hex27_topography"])
@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
def test_vs3d_equals_the_full_mesh_step(eq, oracle, memory, topo, split):
    r1, g1 = _run_vs3d(eq, memory, topo, split)
    r0, g0 = _oracle(_run_vs3d, oracle, memory, topo)
    assert _rel(r1, r0) < 1e-12
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-12


def _split_hints(monkeypatch, compile_step, memory, dev="cpu", nshots=1, per_shot=False, **kw):
    """The ``band_split`` PropSEM passes to every ``build`` of one call."""
    import functools
    seen = []
    real = ElasticSEM2DDeformed.build.__func__

    @functools.wraps(real)                 # PropSEM reads build's signature
    def build(cls, *a, **k):
        seen.append(k.get("band_split", "absent"))
        return real(cls, *a, **k)

    monkeypatch.setattr(ElasticSEM2DDeformed, "build", classmethod(build))
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    dtype = torch.float64 if dev == "cpu" else torch.float32
    kw.setdefault("impl", "eager")
    p = PropSEM(ElasticSEM2DDeformed, shape=SHAPE, dh=10.0, dt=DT, n_order=4, abcn=8,
                f_dominant=15.0, dev=dev, dtype=dtype, source_type=["sxx", "szz"],
                receiver_type=["vx", "vz"], free_surface=True, memory=memory, **kw)
    w = torch.zeros(16, dtype=dtype, device=dev)
    w[2] = 1.0
    rx = np.arange(3, 38, 3)
    lead = (nshots,) if per_shot else ()
    with torch.no_grad():
        p(w, np.array([[12 + 8 * b, 5.0] for b in range(nshots)]),
          np.stack([rx, np.full_like(rx, 4)], 1)[None],
          models=[torch.full(lead + SHAPE, v, dtype=dtype, device=dev)
                  for v in (2400.0, 1300.0, 2000.0)],
          compile_step=compile_step)
    assert seen
    return set(seen)


@pytest.mark.parametrize("compile_step, memory", [
    (False, Full()), (True, Full()), (False, Ckpt(chunks=8))], ids=["plain", "compiled", "ckpt"])
def test_propsem_leaves_the_split_to_the_equation(monkeypatch, compile_step, memory):
    """However the eager step runs, the split is the equation's choice by mesh
    size: compiled it costs at medium sizes too, and replayed as CUDA graphs it
    loses on the velocity-stress forms (sweep.sem.cpml.band)."""
    assert _split_hints(monkeypatch, compile_step, memory) == {"absent"}


def test_propsem_passes_a_requested_split(monkeypatch):
    assert _split_hints(monkeypatch, False, Full(), equation_options={"band_split": True}) == {True}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA-graph rollout")
@pytest.mark.parametrize("nshots, per_shot", [(1, False), (2, True), (2, False)],
                         ids=["one_shot", "per_shot_models", "shared_model_shots"])
def test_propsem_leaves_the_split_to_the_equation_on_cuda(monkeypatch, nshots, per_shot):
    assert _split_hints(monkeypatch, True, Ckpt(chunks=8), "cuda", nshots, per_shot) == {"absent"}


def _sem_c():
    try:
        from sweep.sem.prop_c import sem_c_available
        return torch.cuda.is_available() and sem_c_available()
    except Exception:
        return False


@pytest.mark.skipif(not _sem_c(), reason="the SEM CUDA backend")
def test_impl_c_builds_the_equation_unsplit(monkeypatch):
    """The CUDA kernels split on their own; the eager split would be built for nothing."""
    assert _split_hints(monkeypatch, False, Full(), "cuda", impl="c") == {False}

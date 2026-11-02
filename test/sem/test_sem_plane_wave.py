"""Plane-wave incidence on the 2-D displacement SEM (sweep.sem.plane_wave).

test/sem/specfem_compare/run_compare_planewave.py checks the runs against SPECFEM2D's
``initialfield``; here: the plane wave of the background is SPECFEM2D's formulas below the
critical angle, solves the wave equation and leaves the surface free past it and as a Rayleigh
wave, and an SH wave reflects whole; a run in the flat half-space carries it inside the box and
nothing outside (P-SV and SH), a vertical SV or SH wave doubles at the surface, the mirror image
of a wave is the mirror image of its run, the memory modes run the same loads, the gradients
exist, and what the method cannot do is refused.
"""
import math

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.sem import (ElasticSEM2DDeformed, ElasticSHSEM2DDeformed, ElasticVSSEM2DDeformed,
                       PlaneWave, PropSEM, ViscoElasticSEM2DDeformed)
from sweep.sem.plane_wave import FreeField, _band_residual, rayleigh_speed

F64 = torch.float64
SHAPE, DH, DT, NT, F0 = (49, 97), 5.0, 4e-4, 1500, 10.0
VP, VS, RHO = 2000.0, 1000.0, 2000.0
# a box with a one-element margin: the band is the second ring of elements in
BOX = ((8, 0), (88, 40))
INSIDE = [(24.0, 0.0), (48.0, 0.0), (72.0, 0.0), (48.0, 16.0), (30.0, 30.0)]
OUTSIDE = [(2.0, 10.0), (94.0, 10.0), (48.0, 46.0)]


def _ricker(t, f0):
    a = (math.pi * f0) ** 2
    return (1.0 - 2.0 * a * t * t) * np.exp(-a * t * t)


def _ricker_tt(t, f0):
    a = (math.pi * f0) ** 2
    return -2.0 * a * (3.0 - 12.0 * a * t * t + 4.0 * a * a * t ** 4) * np.exp(-a * t * t)


def _wavelet(delay=1.5 / F0, n=NT):
    return torch.from_numpy(_ricker(np.arange(n) * DT - delay, F0))


def _prop(equation=ElasticSEM2DDeformed, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 16,
          "free_surface": True, "receiver_type": ["u_x", "u_z"], **kw}
    return PropSEM(equation, shape=SHAPE, dh=DH, dt=DT, n_order=4, f_dominant=F0, dev="cpu",
                   dtype=F64, impl="eager", **kw)


def _models(vp=VP, grad=False):
    return [torch.full(SHAPE, v, dtype=F64, requires_grad=grad) for v in (vp, VS, RHO)]


def _sh_prop(**kw):
    return _prop(ElasticSHSEM2DDeformed, receiver_type=["u_y"], **kw)


def _run(prop, wave, models=None, w=None, rec=INSIDE + OUTSIDE):
    models = _models() if models is None else models
    w = _wavelet() if w is None else w
    out = prop(w, wave, torch.tensor([rec], dtype=F64), models=models)[0]    # (nt, nrec, 2)
    return out, prop.free_field(w, wave, torch.tensor(rec, dtype=F64), models)


# --------------------------------------------------------------------------- #
# The plane wave of the background
# --------------------------------------------------------------------------- #
def _specfem(kind, ang, vp, vs, x, zup, t, x0, z0, f0):
    """prepare_initial_field.F90's displacement (z up): kind 1 P, 2 SV, 4 and 5 without the
    reflected waves."""
    th = abs(math.radians(ang))
    if kind in (1, 4):
        p, ci, cr = math.sin(th) / vp, vp, vs
        rf = math.asin(p * cr)
        d = math.cos(2 * rf) ** 2 / vs ** 3 + 4 * p * p * math.cos(th) * math.cos(rf) / vp
        PP = (-math.cos(2 * rf) ** 2 / vs ** 3 + 4 * p * p * math.cos(th) * math.cos(rf) / vp) / d
        PS = 4 * p * math.cos(th) * math.cos(2 * rf) / (vs ** 2 * d)
        A, B = [math.sin(th), math.cos(th)], [PP * math.sin(th), -PP * math.cos(th)]
        C = [PS * math.cos(rf), PS * math.sin(rf)]
    else:
        p, ci, cr = math.sin(th) / vs, vs, vp
        rf = math.asin(min(p * cr, 1.0)) if kind == 2 else 0.0
        d = math.cos(2 * th) ** 2 / vs ** 3 + 4 * p * p * math.cos(th) * math.cos(rf) / vp
        SS = (math.cos(2 * th) ** 2 / vs ** 3 - 4 * p * p * math.cos(th) * math.cos(rf) / vp) / d
        SP = 4 * p * math.cos(th) * math.cos(2 * th) / (vp * vs * d)
        A, B = [math.cos(th), -math.sin(th)], [SS * math.cos(th), SS * math.sin(th)]
        C = [SP * math.sin(rf), -SP * math.cos(rf)]
    if ang < 0:
        A[0], B[0], C[0] = -A[0], -B[0], -C[0]
    if kind in (4, 5):
        B, C = [0.0, 0.0], [0.0, 0.0]
    zz, xx = z0 - zup, (x - x0) if ang >= 0 else (x0 - x)
    ti = t - math.sin(th) * xx / ci + math.cos(th) * zz / ci
    tb = t - math.sin(th) * xx / ci - math.cos(th) * zz / ci
    tc = t - math.sin(rf) * xx / cr - math.cos(rf) * zz / cr
    r = [_ricker(s, f0) for s in (ti, tb, tc)]
    return (A[0] * r[0] + B[0] * r[1] + C[0] * r[2], A[1] * r[0] + B[1] * r[1] + C[1] * r[2])


@pytest.mark.parametrize("kind, ang, vp, vs", [
    (1, 23.0, 6500.0, 3700.0), (1, -23.0, 6500.0, 3700.0), (1, 0.0, 3000.0, 1500.0),
    (2, 20.0, 2.0, 1.0), (2, -20.0, 2.0, 1.0), (2, 30.0, 2.0, 1.0), (2, 0.0, 2.0, 1.0),
    (4, 15.0, 3950.0, 2250.0), (5, -35.0, 3950.0, 2250.0)])
def test_the_plane_wave_is_specfem2ds_below_the_critical_angle(kind, ang, vp, vs):
    """Its incident, reflected and converted waves (none in a whole space), to the
    band-limited delay of the samples; SV at 30 degrees with vp = 2 vs sits at the critical
    angle, its P wave along the surface."""
    f0, L = 1.0, 10.0 * vs
    dt, t0 = 1.0 / (200.0 * f0), 1.2 / f0 + 15.0 * L / (10.0 * vs)
    nt = int(6.0 * L / vs / dt)
    t = np.arange(nt) * dt - t0
    rng = np.random.default_rng(0)
    xs, zs = rng.uniform(0.0, L, 32), rng.uniform(0.0, L, 32)
    wave = PlaneWave("P" if kind in (1, 4) else "SV", ang)
    ff = FreeField(wave, vp, vs, 1.0, 0.0 if kind in (1, 2) else None, (0.37 * L, 0.0))
    u = ff.displacement(xs, zs, torch.from_numpy(_ricker(t, f0)), dt, nt).numpy()
    ref = np.stack([np.stack(_specfem(kind, ang, vp, vs, x, -z, t, 0.37 * L, 0.0, f0))
                    for x, z in zip(xs, zs)])
    ref[:, 1] *= -1.0                                                   # z up -> z down
    assert np.abs(u - ref).max() < 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize("ang", [0.0, 25.0, -40.0])
def test_an_sh_wave_reflects_whole(ang):
    """``sigma_yz = mu du_y/dz = 0`` at the surface: the reflected SH wave is the incident one
    mirrored in the surface, same sign; ``u_y`` keeps its sign under the mirror image."""
    vs, f0, dt, nt = 1.0, 1.0, 1.0 / 200.0, 4000
    t = np.arange(nt) * dt - 8.0
    ff = FreeField(PlaneWave("SH", ang), None, vs, 1.0, 0.0, (1.0, 0.0))
    rng = np.random.default_rng(3)
    xs, zs = rng.uniform(0.0, 6.0, 16), rng.uniform(0.0, 4.0, 16)
    u = ff.displacement(xs, zs, torch.from_numpy(_ricker(t, f0)), dt, nt).numpy()[:, 0]
    th = math.radians(abs(ang))
    xx = (xs - 1.0) if ang >= 0 else (1.0 - xs)
    ref = np.stack([_ricker(t - (math.sin(th) * x - math.cos(th) * z) / vs, f0)
                    + _ricker(t - (math.sin(th) * x + math.cos(th) * z) / vs, f0)
                    for x, z in zip(xx, zs)])
    assert np.abs(u - ref).max() < 1e-12 * np.abs(ref).max()


def _residuals(wave, vp, vs, rho, f0, h_frac):
    """The wave equation's and the surface tractions' residuals of the field, by central
    differences of step ``h_frac`` wavelengths, relative."""
    lam, mu = rho * (vp * vp - 2 * vs * vs), rho * vs * vs
    dt, nt = 1.0 / (400.0 * f0), 3000
    t = np.arange(nt) * dt - 3.0 / f0
    w = torch.from_numpy(_ricker(t, f0))
    w_tt = torch.from_numpy(_ricker_tt(t, f0))          # u_tt: the field of the wavelet's
    ff = FreeField(wave, vp, vs, rho, 0.0, (0.0, 0.0))
    lam_s = vs / f0
    h = h_frac * lam_s
    xc, zc = 0.3 * lam_s, 0.25 * lam_s
    X = np.array([xc + i * h for j in (-1, 0, 1) for i in (-1, 0, 1)])
    Z = np.array([zc + j * h for j in (-1, 0, 1) for i in (-1, 0, 1)])
    u = ff.displacement(X, Z, w, dt, nt).numpy().reshape(3, 3, 2, nt)       # [z, x, comp, t]
    a = ff.displacement(X[4:5], Z[4:5], w_tt, dt, nt).numpy()[0]           # at the centre
    dxx = lambda f: (f[1, 2] - 2 * f[1, 1] + f[1, 0]) / h ** 2
    dzz = lambda f: (f[2, 1] - 2 * f[1, 1] + f[0, 1]) / h ** 2
    dxz = lambda f: (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / (4 * h * h)
    ux, uz = u[:, :, 0], u[:, :, 1]
    fx = (lam + 2 * mu) * dxx(ux) + mu * dzz(ux) + (lam + mu) * dxz(uz)
    fz = (lam + 2 * mu) * dzz(uz) + mu * dxx(uz) + (lam + mu) * dxz(ux)
    wave_eq = max(np.abs(rho * a[0] - fx).max() / np.abs(fx).max(),
                  np.abs(rho * a[1] - fz).max() / np.abs(fz).max())
    # the surface z = 0, one-sided in z: an evanescent wave is not defined above it
    s = ff.displacement(np.array([-h, 0.0, h, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, h, 2 * h]),
                        w, dt, nt).numpy()
    dux_x, duz_x = (s[2, 0] - s[0, 0]) / (2 * h), (s[2, 1] - s[0, 1]) / (2 * h)
    dux_z = (-3 * s[1, 0] + 4 * s[3, 0] - s[4, 0]) / (2 * h)
    duz_z = (-3 * s[1, 1] + 4 * s[3, 1] - s[4, 1]) / (2 * h)
    scale = mu * max(np.abs(d).max() for d in (dux_x, duz_x, dux_z, duz_z))
    traction = max(np.abs(lam * dux_x + (lam + 2 * mu) * duz_z).max(),
                   np.abs(mu * (dux_z + duz_x)).max()) / scale
    return wave_eq, traction


@pytest.mark.parametrize("wave, vp, vs, rho", [
    (PlaneWave("SV", 40.0), 2.0, 1.0, 1.0), (PlaneWave("SV", -50.0), 2.0, 1.0, 1.0),
    (PlaneWave("SV", 45.0), 2.0, 1.0, 1.0), (PlaneWave("SV", 30.0), 2.0, 1.0, 1.0),
    (PlaneWave("P", 23.0), 6500.0, 3700.0, 2800.0),
    (PlaneWave("Rayleigh", 90.0), 2.0, 1.0, 1.0),
    (PlaneWave("Rayleigh", -90.0), 3000.0, 1700.0, 2200.0)])
def test_the_plane_wave_solves_the_wave_equation_and_frees_the_surface(wave, vp, vs, rho):
    """Past the critical angle and as a Rayleigh wave too: both residuals are the central
    differences' own, falling as h^2."""
    coarse = _residuals(wave, vp, vs, rho, 1.0, 2e-3)
    fine = _residuals(wave, vp, vs, rho, 1.0, 1e-3)
    for c, f in zip(coarse, fine):
        assert f < 2e-4 and f < 0.3 * c, (coarse, fine)


def _crb(bealf):
    """SPECFEM2D's CRB (paco_beyond_critical.f90): the Rayleigh speed over vs."""
    ba2 = bealf * bealf
    P, Q = 8.0 / 3.0 - 16.0 * ba2, 272.0 / 27.0 - 80.0 / 3.0 * ba2
    find = Q * Q / 4.0 + P ** 3 / 27.0
    if find >= 0.0:
        cube = lambda v: math.copysign(abs(v) ** (1.0 / 3.0), v)
        return math.sqrt(cube(math.sqrt(find) - Q / 2) + cube(-math.sqrt(find) - Q / 2) + 8 / 3)
    f1 = math.sqrt(-27.0 * Q * Q / (4.0 * P ** 3))
    f1 = math.cos((math.pi - math.acos(f1)) / 3.0) if Q < 0.0 else math.cos(math.acos(f1) / 3.0)
    return math.sqrt(-2.0 * f1 * math.sqrt(-P / 3.0) + 8.0 / 3.0)


@pytest.mark.parametrize("ratio", [1.5, math.sqrt(3.0), 2.0, 3.0])
def test_the_rayleigh_speed_is_specfem2ds(ratio):
    assert abs(rayleigh_speed(ratio * 700.0, 700.0) / 700.0 - _crb(1.0 / ratio)) < 1e-14


def test_the_band_operator_is_the_stiffness():
    """The P-SV classes' and the SH class's, over every element."""
    g = torch.Generator().manual_seed(1)
    for eq in (_prop()._build_eq(_models()),
               _sh_prop()._build_eq(_models()[1:])):
        k = 1 if isinstance(eq, ElasticSHSEM2DDeformed) else 2
        u = torch.randn(eq.mesh.n_global, k, 3, dtype=F64, generator=g)
        E = eq.mesh.n_elements
        R = _band_residual(eq, torch.arange(E), u[eq.mesh.ibool])
        K = torch.zeros_like(u).index_add(0, eq.mesh.ibool.reshape(-1), R.reshape(-1, k, 3))
        for j in range(3):
            ref = (eq.stiffness_apply(u[:, 0, j])[:, None] if k == 1 else
                   torch.stack(eq.stiffness_apply(u[:, 0, j], u[:, 1, j]), -1))
            assert torch.allclose(K[..., j], ref, rtol=0.0, atol=1e-12 * float(ref.abs().max()))


# --------------------------------------------------------------------------- #
# Runs in the flat half-space
# --------------------------------------------------------------------------- #
def _check(out, ff, inside=3e-3, outside=1e-3):
    n = len(INSIDE)
    peak = float(ff[:n].abs().max())
    err = float((out[:, :n] - ff[:n].transpose(0, 1)).abs().max()) / peak
    leak = float(out[:, n:].abs().max()) / peak
    assert err < inside, err
    assert leak < outside, leak
    return err, leak


def test_a_vertical_sv_wave_doubles_at_the_surface_and_stays_in_the_box():
    out, ff = _run(_prop(), PlaneWave("SV", 0.0, box=BOX))
    _check(out, ff)
    assert abs(float(out[:, :3, 0].abs().max()) - 2.0) < 3e-3


@pytest.mark.parametrize("wave, delay", [
    (PlaneWave("P", 23.0, box=BOX), 1.5), (PlaneWave("SV", 40.0, box=BOX), 1.5),
    (PlaneWave("SV", -30.0, box=BOX), 1.5), (PlaneWave("Rayleigh", 90.0, box=BOX), 4.5)])
def test_oblique_past_critical_and_rayleigh_waves_stay_in_the_box(wave, delay):
    """SV at 40 degrees reflects an evanescent P wave, at -30 the critical one; the Rayleigh
    wave's wavelet starts later: its vertical motion is the wavelet's Hilbert transform."""
    _check(*_run(_prop(), wave, w=_wavelet(delay / F0)))


@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_a_plane_sh_wave(angle):
    """On the SH class: whole at the surface (twice the incident wave when vertical), the
    plane wave inside the box and nothing outside."""
    out, ff = _run(_sh_prop(), PlaneWave("SH", angle, box=BOX), models=_models()[1:])
    _check(out, ff)
    if angle == 0.0:
        assert abs(float(out[:, :3, 0].abs().max()) - 2.0) < 3e-3


def test_the_mirror_image_of_the_wave_runs_the_mirror_image():
    """The box and receivers symmetric about the model's centre line, the origins the band's
    two bottom corners: u_x flips."""
    rec = [(24.0, 0.0), (72.0, 0.0), (30.0, 30.0), (66.0, 30.0)]
    a, _ = _run(_prop(), PlaneWave("SV", 25.0, box=BOX), rec=rec)
    b, _ = _run(_prop(), PlaneWave("SV", -25.0, box=BOX), rec=rec)
    swap, peak = [1, 0, 3, 2], float(a.abs().max())
    # the mesh is the mirror image to round-off (2e-13 m), not bit for bit
    assert float((a[:, :, 0] + b[:, swap, 0]).abs().max()) < 1e-8 * peak
    assert float((a[:, :, 1] - b[:, swap, 1]).abs().max()) < 1e-8 * peak


@pytest.mark.parametrize("wave, delay", [(PlaneWave("SV", 20.0, box=((8, 8), (88, 40))), 1.5),
                                         (PlaneWave("Rayleigh", 90.0, box=((8, 8), (88, 40))), 4.5)])
def test_a_box_below_the_surface(wave, delay):
    """A closed band under the free surface: the surface strip above it is outside, and
    quiet; the Rayleigh wave's reference sits on the surface above the band's first node.
    The strip is one element thick, and the leak on its surface the leapfrog's error: 1.5e-3
    of the peak here, 5e-5 at half the step."""
    rec = [(24.0, 20.0), (48.0, 12.0), (72.0, 30.0), (40.0, 20.0), (60.0, 24.0)]
    _check(*_run(_prop(), wave, w=_wavelet(delay / F0),
                 rec=rec + [(48.0, 0.0), (24.0, 2.0), (94.0, 10.0)]), outside=3e-3)


def test_under_an_absorbing_top_it_is_the_whole_space_wave():
    """No free surface: the box closed, the incident wave alone."""
    prop = _prop(free_surface=False)
    rec = [(24.0, 20.0), (48.0, 12.0), (72.0, 30.0), (40.0, 20.0), (60.0, 24.0)]
    out, ff = _run(prop, PlaneWave("P", 15.0, box=((8, 8), (88, 40))),
                   rec=rec + [(48.0, 2.0), (2.0, 10.0), (94.0, 10.0)])
    # no reflections: the incident wave's vertical displacement alone
    assert abs(float(ff[..., 1].abs().max()) - math.cos(math.radians(15.0))) < 1e-3
    _check(out, ff)


@pytest.mark.parametrize("memory", [Ckpt(chunks=200), BoundarySaving()])
def test_the_memory_modes_run_the_same_loads(memory):
    wave = PlaneWave("SV", 20.0, box=BOX)
    ref, _ = _run(_prop(), wave)
    out, _ = _run(_prop(memory=memory), wave)
    assert torch.allclose(out, ref, rtol=0.0, atol=1e-12)


def test_the_gradients_exist():
    """In the model inside the box (the loads are a fixed source: none outside) and in the
    wavelet, which the loads are linear in."""
    prop = _prop()
    models = _models(grad=True)
    w = _wavelet().requires_grad_(True)
    out = prop(w, PlaneWave("SV", 20.0, box=BOX), torch.tensor([INSIDE]), models=models)
    out.square().sum().backward()
    g = models[0].grad
    assert torch.isfinite(g).all() and float(g[4:36, 12:84].abs().max()) > 0.0
    assert torch.isfinite(w.grad).all() and float(w.grad.abs().max()) > 0.0
    # linear in the wavelet: the directional derivative is the difference quotient
    v = torch.randn(NT, dtype=F64, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        f = lambda s: prop(_wavelet() + s * v, PlaneWave("SV", 20.0, box=BOX),
                           torch.tensor([INSIDE]), models=_models()).square().sum()
        quotient = float((f(1e-3) - f(-1e-3)) / 2e-3)
    assert abs(quotient - float(w.grad @ v)) < 1e-8 * abs(quotient)


def test_attenuation_inside_the_box_and_not_outside_it():
    prop = _prop(ViscoElasticSEM2DDeformed, equation_options={"f_ref": F0})
    inf = torch.full(prop._gll_shape, float("inf"), dtype=F64)
    q = inf.clone()
    eq = prop._build_eq(_models() + [inf, inf])
    inner = prop._plane_wave_band(eq, PlaneWave("SV", 0.0, box=BOX)).interior
    q[inner] = 20.0
    out, ff = _run(prop, PlaneWave("SV", 0.0, box=BOX), models=_models() + [inf, q])
    assert float(out[:, :3, 0].abs().max()) < 0.95 * float(ff[:3].abs().max())    # it damps
    with pytest.raises(ValueError, match="Q must be infinite outside"):
        _run(prop, PlaneWave("SV", 0.0, box=BOX), models=_models() + [inf, q.fill_(20.0)])


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_what_a_plane_wave_is():
    with pytest.raises(ValueError, match="wave must be one of"):
        PlaneWave("S", 0.0)
    with pytest.raises(ValueError, match=r"\|angle\| < 90"):
        PlaneWave("SV", 90.0)
    with pytest.raises(ValueError, match="angle = 90"):
        PlaneWave("Rayleigh", 45.0)


def test_the_background_outside_the_box_must_be_uniform_and_flat():
    vp = torch.full(SHAPE, VP, dtype=F64)
    vp[46, 50] = 2100.0                                    # in the margin outside the box
    with pytest.raises(ValueError, match="vp ranges over"):
        _run(_prop(), PlaneWave("SV", 0.0, box=BOX), models=[vp] + _models()[1:])
    x = torch.arange(SHAPE[1], dtype=F64)
    topo = 2.0 * torch.exp(-((x - 2.0) / 2.0) ** 2)         # a bump at the left edge
    with pytest.raises(ValueError, match="flat free surface"):
        _run(_prop(topography=topo), PlaneWave("SV", 0.0, box=BOX))


def test_the_box_must_sit_on_element_edges_with_its_band_in_the_model():
    with pytest.raises(ValueError, match="not on an element edge"):
        _run(_prop(), PlaneWave("SV", 0.0, box=((6, 0), (88, 40))))
    with pytest.raises(ValueError, match="leaves no room for its band"):
        _run(_prop(), PlaneWave("SV", 0.0, box=((0, 0), (88, 40))))


def test_the_wave_must_reach_the_band_after_the_start():
    with pytest.raises(ValueError, match="already on the injection band"):
        _run(_prop(), PlaneWave("SV", 0.0, box=BOX), w=_wavelet(0.0))
    with pytest.raises(ValueError, match="Hilbert transform"):
        _run(_prop(), PlaneWave("Rayleigh", 90.0, box=BOX), w=_wavelet(1.5 / F0))


def test_what_a_plane_wave_call_does_not_take():
    rec = torch.tensor([INSIDE])
    with pytest.raises(ValueError, match="build the propagator without source_type"):
        _prop(source_type=["vz"])(_wavelet(), PlaneWave(), rec, models=_models())
    with pytest.raises(ValueError, match="source_vector"):
        _prop()(_wavelet(), PlaneWave(), rec, models=_models(), source_vector=(0.0, 1.0))
    with pytest.raises(ValueError, match="one shot"):
        _prop()(_wavelet(), PlaneWave(), rec.expand(2, -1, -1), models=_models())
    with pytest.raises(ValueError, match="one \\(nt,\\) wavelet"):
        _prop()(_wavelet().expand(2, -1), PlaneWave(), rec, models=_models())
    with pytest.raises(ValueError, match="needs a free surface"):
        _prop(free_surface=False)(_wavelet(4.5 / F0), PlaneWave("Rayleigh", 90.0), rec,
                                  models=_models())
    prop = _prop()
    prop._use_c = True                                  # as impl='c' on CUDA builds it
    with pytest.raises(NotImplementedError, match="impl='eager'"):
        prop(_wavelet(), PlaneWave(), rec, models=_models())
    with pytest.raises(ValueError, match="an in-plane class takes P, SV and Rayleigh"):
        _prop()(_wavelet(), PlaneWave("SH"), rec, models=_models())
    with pytest.raises(ValueError, match="an SH class takes an SH wave"):
        _sh_prop()(_wavelet(), PlaneWave("SV"), rec, models=_models()[1:])
    with pytest.raises(NotImplementedError, match="2-D displacement classes"):
        PropSEM(ElasticVSSEM2DDeformed, shape=SHAPE, dh=DH, dt=DT, n_order=4, abcn=16,
                free_surface=True, f_dominant=F0, dev="cpu", dtype=F64, impl="eager",
                memory=Full(), eager_options=EagerOptions(use_compile=False),
                receiver_type=["vx", "vz"])(_wavelet(), PlaneWave(), rec, models=_models())

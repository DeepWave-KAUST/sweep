"""Biot poroelastic equation (eager): undrained limit, wave speeds, viscous analytic solution,
open-pore free surface, gradient, refusals."""

import math

import numpy as np
import pytest
import torch

from sweep.equations import Biot, Elastic
from sweep.equations.biot import biot_from_biot_constants, biot_from_rock, biot_wave_velocities
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEV = torch.device("cpu")

# Dai, Vafidis & Kanasewich (1995), Table 1, layer 1 (Yew & Jogi 1978): inviscid here.
DAI1 = dict(P=2.0332e10, Q=0.0953e10, R=0.0331e10, N=0.684e10, rho11=2167.0, rho12=-83.0, rho22=191.0)


def _wavelet(nt, dt, f0):
    t = np.arange(nt) * dt - 1.2 / f0
    return torch.from_numpy(ricker(t, f=f0))


def _prop(eq, shape, dh, dt, src, rec, abcn=20, free_surface=False):
    return PropTorch(eq, shape=shape, dev=DEV, dh=dh, dt=dt, source_type=src, receiver_type=rec,
                     abcn=abcn, free_surface=free_surface, backend="torch", impl="eager")


def _fields(coef, shape):
    """Coefficient list -> float32 arrays of ``shape`` (scalars broadcast)."""
    return [c.float().expand(shape).contiguous() if isinstance(c, torch.Tensor) else torch.full(shape, float(c))
            for c in coef]


def _layered_rock(shape):
    nz, nx = shape
    z = torch.arange(nz, dtype=torch.float32)[:, None].expand(nz, nx)
    phi = 0.25 - 0.1 * (z > nz // 2).float()
    k_d = 8e9 + 6e9 * (z > nz // 2).float()
    mu = 6e9 + 5e9 * (z > nz // 2).float()
    return dict(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, k_d=k_d, mu=mu, phi=phi,
                kappa=1e-13, tortuosity=2.0)


def test_derived_velocities_match_dai_table_1():
    models = biot_from_biot_constants(**DAI1, b=0.0, phi=0.3)
    vf, vslow, vs = biot_wave_velocities(models)
    assert vf == pytest.approx(3210, rel=2e-3)
    assert vslow == pytest.approx(1180, rel=2e-3)
    assert vs == pytest.approx(1790, rel=2e-3)


def test_no_relative_flow_limit_is_undrained_elastic():
    # b -> infinity freezes q, so v and the total stress obey elastic waves in the
    # undrained medium; compare the records with Elastic directly
    shape, dh, dt, nt = (60, 70), 10.0, 1e-3, 300
    coef = _fields(biot_from_rock(**_layered_rock(shape)), shape)
    coef[7] = torch.full(shape, 1e16)                      # b = eta / kappa, kappa -> 0
    h, mu, rho = coef[0], coef[1], coef[4]
    vp, vs = torch.sqrt(h / rho), torch.sqrt(mu / rho)
    src = np.array([[35, 20]])
    rec = np.stack([np.arange(5, 65, 3), np.full(20, 40)], 1)[None]
    w = _wavelet(nt, dt, 12.0)
    db = _prop(Biot(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz"])(w, src, rec, models=coef)
    de = _prop(Elastic(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz"])(w, src, rec, models=[vp, vs, rho])
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-4, err


def test_inviscid_fast_p_and_s_speeds():
    # far-field arrival-time differences between two receivers on a line
    dh, dt, nt, f0 = 2.0, 2e-4, 1400, 60.0
    shape = (160, 260)
    coef = _fields(biot_from_biot_constants(**DAI1, b=0.0, phi=0.3), shape)
    vf, _, vs = biot_wave_velocities([float(c[0, 0]) for c in coef])
    w = _wavelet(nt, dt, f0)
    rec = np.stack([np.array([90, 230]), np.array([80, 80])], 1)[None]
    t = np.arange(nt) * dt

    def first_break(trace):
        a = np.abs(trace)
        return t[np.argmax(a > 0.1 * a.max())]

    d = _prop(Biot(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx"])(w, np.array([[40, 80]]), rec, models=coef)
    tr = d[0, :, :, 0].numpy()
    v_meas = (230 - 90) * dh / (first_break(tr[:, 1]) - first_break(tr[:, 0]))
    assert v_meas == pytest.approx(vf, rel=0.03), (v_meas, vf)

    d = _prop(Biot(device=DEV), shape, dh, dt, ["vz"], ["vz"])(w, np.array([[40, 80]]), rec, models=coef)
    tr = d[0, :, :, 0].numpy()
    # a vertical force radiates S along x; the P lobe is nodal there
    v_meas = (230 - 90) * dh / (first_break(tr[:, 1]) - first_break(tr[:, 0]))
    assert v_meas == pytest.approx(vs, rel=0.03), (v_meas, vs)


@pytest.mark.parametrize("free_surface", [False, True])
def test_gradient_matches_finite_difference_inside(free_surface):
    torch.manual_seed(0)
    shape, dh, dt, nt = (40, 48), 10.0, 1e-3, 220
    coef = _fields(biot_from_rock(**_layered_rock(shape)), shape)
    src = np.array([[24, 10]])
    rec = np.stack([np.arange(8, 40, 4), np.full(8, 30)], 1)[None]
    w = _wavelet(nt, dt, 10.0)
    prop = _prop(Biot(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz", "p"], free_surface=free_surface)
    obs = prop(w, src, rec, models=coef).detach()

    def J(mu):
        return 0.5 * (prop(w, src, rec, models=[coef[0], mu, *coef[2:]]) - 1.05 * obs).double().square().sum()

    mu0 = coef[1].clone().requires_grad_()
    (g,) = torch.autograd.grad(J(mu0), mu0)
    # smooth direction, zero on the outer cells (their gradient is a crop of the padded region)
    dm = torch.zeros(shape)
    zz, xx = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    dm += torch.exp(-((zz - 22.0) ** 2 + (xx - 20.0) ** 2) / 40.0)
    dm *= 2e8
    pred = float((g * dm).sum())
    eps = 1.0
    with torch.no_grad():
        fd = (float(J(coef[1] + eps * dm)) - float(J(coef[1] - eps * dm))) / (2 * eps)
    assert fd == pytest.approx(pred, rel=2e-2), (fd, pred)


def test_free_surface_no_relative_flow_limit_is_elastic():
    # b -> infinity: no fluid crosses the open-pore surface, so the half-space is Elastic's
    # (undrained moduli) with its image-method free surface; source near it, receivers on it
    shape, dh, dt, nt = (60, 70), 10.0, 1e-3, 300
    coef = _fields(biot_from_rock(**_layered_rock(shape)), shape)
    coef[7] = torch.full(shape, 1e16)
    h, mu, rho = coef[0], coef[1], coef[4]
    vp, vs = torch.sqrt(h / rho), torch.sqrt(mu / rho)
    src = np.array([[35, 3]])
    rec = np.concatenate([np.stack([np.arange(5, 65, 3), np.zeros(20, int)], 1),
                          np.stack([np.arange(5, 65, 3), np.full(20, 12)], 1)])[None]
    w = _wavelet(nt, dt, 12.0)
    db = _prop(Biot(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz"], free_surface=True)(
        w, src, rec, models=coef)
    de = _prop(Elastic(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz"], free_surface=True)(
        w, src, rec, models=[vp, vs, rho])
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-4, err


def test_decoupled_pore_fluid_sees_a_pressure_release_surface():
    # C = 0 and rho_f = 0 decouple the pore fluid into an acoustic medium (modulus M, density m);
    # p = 0 on the open-pore surface makes its field the free-space one minus the image source's
    from scipy.special import hankel2e

    dx, dt, nt, f0, t0 = 2.0, 2e-4, 900, 30.0, 0.04
    shape = (70, 130)
    models = [2e10, 6e9, 0.0, 4e9, 2300.0, 0.0, 2000.0, 0.0]
    coef = [torch.full(shape, float(c)) for c in models]
    xs, zs = 30, 15
    rec_xz = np.array([[60, 4], [90, 4], [60, 30], [90, 30]])
    t = np.arange(nt) * dt

    def s_time(tt):
        return np.exp(-(math.pi * f0 * (tt - t0)) ** 2)

    w = torch.from_numpy(dt * np.gradient(s_time(t), t) / dx**2).float()
    prop = _prop(Biot(device=DEV), shape, dx, dt, ["p"], ["p"], abcn=30, free_surface=True)
    with torch.no_grad():
        num = prop(-w, np.array([[xs, zs]]), rec_xz[None], models=coef)[0, :, :, 0].double().numpy()

    n = 16 * nt
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))
    om = 2 * math.pi * np.fft.rfftfreq(n, dt)[1:]   # k^2 H0(k r) -> 0 at w = 0
    k = om * math.sqrt(models[6] / models[3])
    # p is recorded on the integer time level, (i + 1) dt, for a source injected after the step
    # (effectively delayed by dt / 2): sample i is the analytic p at (i + 1/2) dt
    s_hat = s_hat[1:] * np.exp(0.5j * om * dt)
    for i, (xr, zr) in enumerate(rec_xz):
        r1, r2 = dx * math.hypot(xr - xs, zr - zs), dx * math.hypot(xr - xs, zr + zs)
        green = hankel2e(0, k * r1) * np.exp(-1j * k * r1) - hankel2e(0, k * r2) * np.exp(-1j * k * r2)
        ana = np.fft.irfft(np.concatenate([[0.0], k**2 * (-0.25j) * green * s_hat]), n)[:nt]
        err = np.linalg.norm(num[:, i] - ana) / np.linalg.norm(ana)
        assert err < 5e-3, (xr, zr, err)


def test_boundary_saving_is_refused():
    # dissipative drag: no reverse-time reconstruction, so a boundary-saving request raises
    from sweep.propagator.options import BoundarySaving

    shape = (30, 30)
    coef = _fields(biot_from_biot_constants(**DAI1, b=3e3, phi=0.3), shape)
    coef[1].requires_grad_()
    prop = PropTorch(Biot(device=DEV), shape=shape, dev=DEV, dh=10.0, dt=1e-3, abcn=10, free_surface=False,
                     backend="torch", impl="eager", memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="not supported for Biot"):
        prop(_wavelet(50, 1e-3, 10.0), np.array([[15, 15]]), np.array([[[10, 15]]]), models=coef)


def test_per_edge_free_surface_is_refused():
    shape = (30, 30)
    with pytest.raises(NotImplementedError, match="per-edge"):
        PropTorch(Biot(device=DEV), shape=shape, dev=DEV, dh=10.0, dt=1e-3, abcn=10,
                  free_surface=["top", "left"], backend="torch", impl="eager")


def _analytic_line_source_vz(models, t, x, z, s_time, qpart, phi):
    """Solid vz of a homogeneous viscous Biot medium for the line source (total force grad Phi,
    (qpart / phi) grad Phi in the relative-flow equation, Phi = S(t) delta), exp(+i w t):
    K lap Psi + w^2 R(w) Psi = -F Phi with R_22 = m - i b / w, split into fast and slow modes."""
    from scipy.special import hankel2e

    h, mu, c, biot_m, rho, rho_f, m, b = (float(v) for v in models)
    dt, nt = t[1] - t[0], len(t)
    n = 16 * nt
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))
    w = 2 * math.pi * np.fft.rfftfreq(n, dt)
    r = math.hypot(x, z)
    K = np.array([[h, c], [c, biot_m]], dtype=complex)
    F = np.array([1.0, qpart / phi], dtype=complex)
    v_hat = np.zeros_like(s_hat)
    for i in np.nonzero(np.abs(s_hat) > 1e-14 * np.abs(s_hat).max())[0]:
        if w[i] == 0:
            continue
        s2, E = np.linalg.eig(np.linalg.solve(K, np.array([[rho, rho_f], [rho_f, m - 1j * b / w[i]]])))
        s = np.sqrt(s2)
        s = np.where(s.real < 0, -s, s)
        a = np.linalg.solve(K @ E, F)
        k = w[i] * s
        # scaled Hankel: a diffusive slow mode underflows cleanly (unscaled hankel2 of scipy 1.14
        # returns O(1e-2) garbage near the underflow edge, e.g. at k r = 686 - 686i)
        hk = hankel2e(1, k * r) * np.exp(-1j * k * r)
        v_hat[i] = -(w[i] / 4) * (z / r) * s_hat[i] * np.sum(E[0] * a * k * hk)
    return np.fft.irfft(v_hat, n)[:nt]


@pytest.mark.parametrize("b_dai", [0.0, 7e4, 9e8])  # inviscid, f_c in band, Darcy (gamma dt >> 1)
def test_viscous_line_source_matches_analytic(b_dai):
    dx, dt, nt, f0, t0, qpart, phi = 4.0, 2e-4, 800, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    off, pad = 15, 15
    shape = (2 * pad + off + 1,) * 2
    coef = [torch.full(shape, float(c)) for c in models]
    t = np.arange(nt) * dt

    def s_time(tt):
        return np.exp(-(math.pi * f0 * (tt - t0)) ** 2)

    w = torch.from_numpy(dt * np.gradient(s_time(t), t) / dx**2).float()
    src, rec = np.array([[pad, pad]]), np.array([[[pad + off, pad + off]]])
    num = 0.0
    for fields, scale in ((["sxx", "szz"], 1.0), (["p"], -qpart / phi)):
        prop = _prop(Biot(device=DEV), shape, dx, dt, fields, ["vz"], abcn=30)
        with torch.no_grad():
            num = num + prop(w * scale, src, rec, models=coef)[0, :, 0, 0].double().numpy()
    ana = _analytic_line_source_vz(models, t, off * dx, off * dx + 0.5 * dx, s_time, qpart, phi)
    err = np.linalg.norm(num - ana) / np.linalg.norm(ana)
    assert err < 0.02, err


def _dwn_halfspace(models, force, zs, rec_xz, t, s_time, L=4000.0, kx_max=2.0):
    """Solid (vx, vz) at ``rec_xz`` = [(x, z), ...] of a P-type line source at (0, zs) in a
    homogeneous Biot half-space z > 0 with an open-pore surface: discrete-wavenumber integration
    (Bouchon 1981) at the complex frequency w - i wI.  Each full-space P mode j is
    a_j S exp(-i nu_j |z - zs|) / (2 i nu_j) per horizontal wavenumber; the up-going fast and slow
    P waves reflect into down-going fast P, slow P and S (u_S = (kz, -kx) D), the three amplitudes
    fixed by sigma_zz = sigma_xz = p = 0 at z = 0."""
    h, mu, c, biot_m, rho, rho_f, m, b = (float(v) for v in models)
    dt, nt = t[1] - t[0], len(t)
    n = 2 * nt
    tt = np.arange(n) * dt
    w_i = 2.0 * math.pi / (n * dt)
    s_hat = np.fft.rfft(s_time(tt) * np.exp(-w_i * tt))
    w = 2.0 * math.pi * np.fft.rfftfreq(n, dt)
    kx = 2.0 * math.pi / L * np.arange(-int(kx_max * L / (2 * math.pi)), int(kx_max * L / (2 * math.pi)) + 1)
    K = np.array([[h, c], [c, biot_m]], dtype=complex)
    v_hat = np.zeros((2, len(w), len(rec_xz)), complex)

    def nu_of(arg):
        nu = np.sqrt(arg.astype(complex))
        return np.where(nu.imag > 0, -nu, nu)

    for i in np.nonzero(np.abs(s_hat) > 1e-12 * np.abs(s_hat).max())[0]:
        wc = w[i] - 1j * w_i
        m_hat = m - 1j * b / wc
        s2, E = np.linalg.eig(np.linalg.solve(K, np.array([[rho, rho_f], [rho_f, m_hat]], dtype=complex)))
        k2 = wc**2 * s2
        a = np.linalg.solve(K @ E, np.asarray(force, dtype=complex))
        eu, ew = E[0], E[1]
        nu = nu_of(k2[None, :] - kx[:, None] ** 2)
        nu_s = nu_of(wc**2 * (rho - rho_f**2 / m_hat) / mu - kx**2)
        szz_p = -(((h - 2.0 * mu) * eu + c * ew) * k2)[None, :] - 2.0 * mu * eu[None, :] * nu**2
        sxz_down = -2.0 * mu * eu[None, :] * kx[:, None] * nu
        p_p = np.broadcast_to(((c * eu + biot_m * ew) * k2)[None, :], nu.shape)
        amp = a[None, :] * s_hat[i] * np.exp(-1j * nu * zs) / (2j * nu)
        mat = np.zeros((len(kx), 3, 3), complex)
        mat[:, 0, :2], mat[:, 0, 2] = szz_p, 2j * mu * kx * nu_s
        mat[:, 1, :2], mat[:, 1, 2] = sxz_down, 1j * mu * (kx**2 - nu_s**2)
        mat[:, 2, :2] = p_p
        rhs = -np.stack([(szz_p * amp).sum(1), (-sxz_down * amp).sum(1), (p_p * amp).sum(1)], 1)
        sol = np.linalg.solve(mat, rhs[..., None])[..., 0]
        for r, (x, z) in enumerate(rec_xz):
            chi = a[None, :] * s_hat[i] * np.exp(-1j * nu * abs(z - zs)) / (2j * nu)
            ref = sol[:, :2] * np.exp(-1j * nu * z)
            s_wave = sol[:, 2] * np.exp(-1j * nu_s * z)
            kz_dir = -nu if z < zs else nu
            ux = (eu * (-1j * kx[:, None]) * (chi + ref)).sum(1) + nu_s * s_wave
            uz = (eu * (-1j * kz_dir * chi - 1j * nu * ref)).sum(1) - kx * s_wave
            shift = np.exp(-1j * kx * x) / L
            v_hat[0, i, r] = 1j * wc * (ux * shift).sum()
            v_hat[1, i, r] = 1j * wc * (uz * shift).sum()
    v = np.fft.irfft(v_hat, n, axis=1) * np.exp(w_i * tt)[None, :, None]
    return v[0, :nt], v[1, :nt]


# Tolerances sit between this surface treatment's error at dx = 2 m and that of the two
# single-limit alternatives (surface sxx always undrained: 7.6% / 2.8% / 1.0%; always drained:
# 3.4% / 1.2% / 2.7%): each of those fails one case, the extrapolated flux gradient passes all.
@pytest.mark.parametrize("b_dai, tol", [(0.0, 0.04), (7e5, 0.02), (9e8, 0.013)])
def test_free_surface_matches_wavenumber_integration(b_dai, tol):
    # line source 20 m under the open-pore surface (Dai's split source), solid vx and vz on the
    # surface 100 m away: slow wave propagating, diffusive, and the Darcy limit
    dx, dt, nt, f0, t0, qpart, phi = 2.0, 1e-4, 2500, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    shape = (126, 161)
    xs, zs, xr = 30, 10, 80
    t = np.arange(nt) * dt

    def s_time(tt):
        return np.exp(-(math.pi * f0 * (tt - t0)) ** 2)

    w = torch.from_numpy(dt * np.gradient(s_time(t), t) / dx**2).float()
    coef = [torch.full(shape, float(c)) for c in models]
    num = 0.0
    for fields, scale in ((["sxx", "szz"], 1.0), (["p"], -qpart / phi)):
        prop = _prop(Biot(device=DEV), shape, dx, dt, fields, ["vx", "vz"], abcn=30, free_surface=True)
        with torch.no_grad():
            num = num + prop(w * scale, np.array([[xs, zs]]), np.array([[[xr, 0]]]), models=coef)[0, :, 0].double().numpy()
    # vx sits at (x + dx/2, 0), vz at (x, dx/2) on the staggered grid
    off = (xr - xs) * dx
    ref_vx, _ = _dwn_halfspace(models, (1.0, qpart / phi), zs * dx, [(off + 0.5 * dx, 0.0)], t, s_time)
    _, ref_vz = _dwn_halfspace(models, (1.0, qpart / phi), zs * dx, [(off, 0.5 * dx)], t, s_time)
    for k, ref in enumerate((ref_vx[:, 0], ref_vz[:, 0])):
        err = np.linalg.norm(num[:, k] - ref) / np.linalg.norm(ref)
        assert err < tol, (("vx", "vz")[k], err)

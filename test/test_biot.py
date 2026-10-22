"""Biot poroelastic equation (eager): undrained limit, wave speeds, gradient, refusals."""

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


def _prop(eq, shape, dh, dt, src, rec, abcn=20):
    return PropTorch(eq, shape=shape, dev=DEV, dh=dh, dt=dt, source_type=src, receiver_type=rec,
                     abcn=abcn, free_surface=False, backend="torch", impl="eager")


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


def test_gradient_matches_finite_difference_inside():
    torch.manual_seed(0)
    shape, dh, dt, nt = (40, 48), 10.0, 1e-3, 220
    coef = _fields(biot_from_rock(**_layered_rock(shape)), shape)
    src = np.array([[24, 10]])
    rec = np.stack([np.arange(8, 40, 4), np.full(8, 30)], 1)[None]
    w = _wavelet(nt, dt, 10.0)
    prop = _prop(Biot(device=DEV), shape, dh, dt, ["sxx", "szz"], ["vx", "vz", "p"])
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


def test_free_surface_is_refused():
    shape = (30, 30)
    coef = _fields(biot_from_biot_constants(**DAI1, b=0.0, phi=0.3), shape)
    prop = PropTorch(Biot(device=DEV), shape=shape, dev=DEV, dh=10.0, dt=1e-3, abcn=10, free_surface=True,
                     backend="torch", impl="eager")
    with pytest.raises(NotImplementedError, match="free surface"):
        prop(_wavelet(50, 1e-3, 10.0), np.array([[15, 15]]), np.array([[[10, 15]]]), models=coef)


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

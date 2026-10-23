"""BiotVSSEM2DDeformed (2-D velocity-stress Biot SEM): the undrained limit, the analytic line
source, the open-pore surface, gradient, sources and receivers, refusals.

The references are the FD ``Biot`` tests' (test/test_biot.py). Time levels: a velocity sample
``n`` is ``v^{n+1/2}``, a stress / pressure sample ``sigma^{n+1}``, and a stress or pressure
source's integrated wavelet leads its rate by ``dt / 2`` -- so the records sit ``dt`` (velocity)
and ``3 dt / 2`` (pressure) after the step's time ``n dt``.
"""

import math

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_fast_p_velocity, biot_from_biot_constants, biot_from_rock
from sweep.sem import BiotVSSEM2DDeformed, ElasticVSSEM2DDeformed, PropSEM
from test_biot import DAI1, _analytic_line_source_vz, _dwn_halfspace, _layered_rock

NO = 4


def _prop(shape, dh, dt, src, rec, *, eq=BiotVSSEM2DDeformed, dtype=torch.float32, f0=30.0,
          impl="eager", dev="cpu", **kw):
    return PropSEM(eq, shape=shape, dh=dh, dt=dt, n_order=NO, f_dominant=f0, dev=dev,
                   impl=impl, dtype=dtype, source_type=src, receiver_type=rec, **kw)


def _fields(coef, shape, dtype=torch.float32):
    """Coefficient list -> arrays of ``shape`` (scalars broadcast)."""
    return [torch.as_tensor(c, dtype=dtype).expand(shape).contiguous() for c in coef]


def _gauss(f0, t0):
    return lambda tt: np.exp(-(math.pi * f0 * (tt - t0)) ** 2)


def _rate(s_time, t, dtype=torch.float32):
    """The wavelet of a stress / pressure source whose history is ``s_time``: PropSEM
    integrates it (the moment rate), as PropTorch's stress injection does."""
    return torch.from_numpy(np.gradient(s_time(t), t)).to(dtype)


def _ricker(nt, dt, f0, dtype=torch.float32):
    a = (math.pi * f0 * (np.arange(nt) * dt - 1.2 / f0)) ** 2
    return torch.from_numpy((1 - 2 * a) * np.exp(-a)).to(dtype)


# ------------------------------------------------------------------------------------------ #
# limits and analytic solutions
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("hill", [False, True])
@pytest.mark.parametrize("source_type", [["sxx", "szz"], ["vz"]])
def test_no_relative_flow_limit_is_the_undrained_elastic(source_type, hill):
    # b -> infinity freezes q: v and the total stress obey ElasticVSSEM2DDeformed in the
    # undrained medium, open-pore surface and all (no fluid crosses it), flat or under a
    # topography. Two layers, the models given on the GLL nodes so that both equations see
    # the same moduli. Not to round-off: the alpha^2 M part of the undrained modulus passes
    # through the continuous p, a weighted mean of div v over the elements at a node, where
    # the elastic equation keeps every element's own -- 4e-3 for the explosion on 40 m
    # elements, 1e-4 on 10 m ones.
    shape, dh, dt, nt = (41, 49), 10.0, 5e-4, 500
    f64 = torch.float64
    src = np.array([[24, 18]])
    rec = np.stack([np.arange(4, 46, 3), np.full(14, 30)], 1)[None]
    w = _ricker(nt, dt, 12.0, f64)
    rows = 2.0 + 4.0 * np.exp(-((np.arange(shape[1]) - 24.0) / 8.0) ** 2) if hill else None
    kw = dict(abcn=0, free_surface=True, topography=rows, dtype=f64)   # no C-PML for both
    biot = _prop(shape, dh, dt, source_type, ["vx", "vz"], **kw)
    deep = (biot.mesh.z_e.mean((1, 2), keepdim=True) > 200.0).to(f64).expand(biot.mesh.z_e.shape)
    rock = dict(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, k_d=8e9 + 6e9 * deep,
                mu=6e9 + 5e9 * deep, phi=0.25 - 0.1 * deep, kappa=1e-13, tortuosity=2.0)
    coef = [torch.as_tensor(c, dtype=f64).expand(deep.shape).contiguous()
            for c in biot_from_rock(**rock)]
    coef[7] = torch.full(deep.shape, 1e16, dtype=f64)
    h, mu, rho = coef[0], coef[1], coef[4]
    db = biot(w, src, rec, models=coef)
    de = _prop(shape, dh, dt, source_type, ["vx", "vz"], eq=ElasticVSSEM2DDeformed, **kw)(
        w, src, rec, models=[torch.sqrt(h / rho), torch.sqrt(mu / rho), rho])
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-2, err


def analytic_line_source_p(models, t, x, z, s_time, force):
    """Pore pressure of a line source in a homogeneous Biot medium, the source as in
    :func:`_analytic_line_source_vz` with forces ``force = (total, relative) * grad Phi``,
    ``Phi = S(t) delta``: with ``u = grad Psi_u``, ``w = grad Psi_w`` and
    ``p = -C div u - M div w``, each mode adds ``k^2 (C E_u + M E_w) a S (-i/4) H0(k r)``."""
    from scipy.special import hankel2e

    h, mu, c, biot_m, rho, rho_f, m, b = (float(v) for v in models)
    dt, nt = t[1] - t[0], len(t)
    n = 16 * nt
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))
    w = 2 * math.pi * np.fft.rfftfreq(n, dt)
    r = math.hypot(x, z)
    K = np.array([[h, c], [c, biot_m]], dtype=complex)
    F = np.asarray(force, dtype=complex)
    p_hat = np.zeros_like(s_hat)
    for i in np.nonzero(np.abs(s_hat) > 1e-14 * np.abs(s_hat).max())[0]:
        if w[i] == 0:
            continue
        s2, E = np.linalg.eig(np.linalg.solve(K, np.array([[rho, rho_f], [rho_f, m - 1j * b / w[i]]])))
        s = np.sqrt(s2)
        s = np.where(s.real < 0, -s, s)
        a = np.linalg.solve(K @ E, F)
        k = w[i] * s
        h0 = hankel2e(0, k * r) * np.exp(-1j * k * r)
        p_hat[i] = -0.25j * s_hat[i] * np.sum(k**2 * (c * E[0] + biot_m * E[1]) * a * h0)
    return np.fft.irfft(p_hat, n)[:nt]


def _split_source(shape, dh, dt, nt, models, src, rec, rec_types, *, qpart, phi, f0, t0,
                  dev="cpu", **kw):
    """The record of Dai's split source: total stress plus ``qpart / phi`` in the relative
    flow, both of history ``exp(-(pi f0 (t - t0))^2)``, as two PropSEM calls."""
    t = np.arange(nt) * dt
    w = _rate(_gauss(f0, t0), t)
    coef = _fields(models, shape)
    num = 0.0
    for fields, scale in ((["sxx", "szz"], 1.0), (["p"], -qpart / phi)):
        prop = _prop(shape, dh, dt, fields, rec_types, dev=dev, **kw)
        with torch.no_grad():
            num = num + prop(w * scale, src, rec, models=coef)[0].double().cpu().numpy()
    return t, num


def line_source_case(b_dai, dev="cpu"):
    """``(t, sem (nt, 2), analytic (nt, 2))``: solid vz and pore pressure of the split source
    in Dai's medium, 60 m off in x and z. The model keeps the C-PML's echoes out of the record
    (they reach a few percent of the direct waves)."""
    dh, dt, nt, f0, t0, qpart, phi = 4.0, 2e-4, 800, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    t, num = _split_source((121, 121), dh, dt, nt, models, np.array([[45, 45]]),
                           np.array([[[60, 60]]]), ["vz", "p"], qpart=qpart, phi=phi, f0=f0,
                           t0=t0, dev=dev)
    s_time = _gauss(f0, t0)
    ana = np.stack([_analytic_line_source_vz(models, t, 60.0, 60.0, lambda tt: s_time(tt + dt),
                                             qpart, phi),
                    analytic_line_source_p(models, t, 60.0, 60.0,
                                           lambda tt: s_time(tt + 1.5 * dt),
                                           (1.0, qpart / phi))], 1)
    return t, num[:, 0], ana


def open_pore_case(b_dai, dev="cpu"):
    """``(t, sem (nt, 2), wavenumber integration (nt, 2))``: solid vx, vz on the open-pore
    surface 100 m from the split source 20 m below it, with no C-PML echo in the record."""
    dh, dt, nt, f0, t0, qpart, phi = 2.0, 1e-4, 1600, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    xs, zs, xr = 80, 10, 130
    t, num = _split_source((125, 221), dh, dt, nt, models, np.array([[xs, zs]]),
                           np.array([[[xr, 0]]]), ["vx", "vz"], qpart=qpart, phi=phi, f0=f0,
                           t0=t0, dev=dev, free_surface=True)
    s_time = _gauss(f0, t0)
    refs = _dwn_halfspace(models, (1.0, qpart / phi), zs * dh, [((xr - xs) * dh, 0.0)], t,
                          lambda tt: s_time(tt + dt))
    return t, num[:, 0], np.stack([r[:, 0] for r in refs], 1)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("b_dai, tol", [(0.0, 0.01), (7e4, 0.005), (9e8, 0.005)])
def test_viscous_line_source_matches_analytic(b_dai, tol):
    # slow wave propagating, critically damped, diffusive
    _, num, ana = line_source_case(b_dai)
    for k, comp in enumerate(("vz", "p")):
        assert _rel(num[:, k], ana[:, k]) < tol, (comp, _rel(num[:, k], ana[:, k]))


# In the Darcy limit the surface's drainage layer sqrt(2 D / w) is ~7 cm, far below the GLL
# spacing, and the nodes there are fully drained: 1.1% on 8 m elements (FD Biot: 1.3% on 2 m
# cells). With the layer resolved (b = 7e5) the error falls to 1e-4 under refinement.
@pytest.mark.parametrize("b_dai, tol", [(0.0, 0.005), (7e5, 0.005), (9e8, 0.02)])
def test_open_pore_surface_matches_wavenumber_integration(b_dai, tol):
    # sigma . n = 0, p = 0 on the surface: slow wave propagating, diffusive, Darcy limit
    _, num, ref = open_pore_case(b_dai)
    for k in range(2):
        assert _rel(num[:, k], ref[:, k]) < tol, (("vx", "vz")[k], _rel(num[:, k], ref[:, k]))


# the pore fluid on its own: C = 0 and rho_f = 0 leave an acoustic medium (modulus M,
# density m), excited by a pore-pressure source
FLUID = [2e10, 6e9, 0.0, 4e9, 2300.0, 0.0, 2000.0, 0.0]


def pressure_source_case(shape, src, rec, free_surface, dh=2.0, dt=2e-4, nt=900, dev="cpu"):
    """``(t, sem p (nt, nrec), analytic (nt, nrec))`` of a pore pressure ``S(t)`` injected
    in :data:`FLUID`: the free-space field, minus its image in an open-pore surface."""
    from scipy.special import hankel2e

    f0, t0 = 30.0, 0.04
    t = np.arange(nt) * dt
    s_time = _gauss(f0, t0)
    prop = _prop(shape, dh, dt, ["p"], ["p"], free_surface=free_surface, dev=dev)
    with torch.no_grad():
        num = prop(_rate(s_time, t), np.array([src]), np.array([rec]),
                   models=_fields(FLUID, shape))[0, :, :, 0].double().cpu().numpy()
    n = 16 * nt
    om = 2 * math.pi * np.fft.rfftfreq(n, dt)[1:]            # k^2 H0(k r) -> 0 at w = 0
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))[1:] * np.exp(1.5j * om * dt)
    k = om * math.sqrt(FLUID[6] / FLUID[3])
    ana = []
    for xr, zr in rec:
        r1 = dh * math.hypot(xr - src[0], zr - src[1])
        green = hankel2e(0, k * r1) * np.exp(-1j * k * r1)
        if free_surface:
            r2 = dh * math.hypot(xr - src[0], zr + src[1])
            green = green - hankel2e(0, k * r2) * np.exp(-1j * k * r2)
        ana.append(np.fft.irfft(np.concatenate([[0.0], 0.25j * k**2 * green * s_hat]), n)[:nt])
    return t, num, np.stack(ana, 1)


def test_decoupled_pore_fluid_sees_a_pressure_release_surface():
    # p = 0 on the open-pore surface: the free-space field minus the image source's, on
    # receivers one element under the surface and deeper
    _, num, ana = pressure_source_case((101, 257), [94, 15],
                                       [[124, 4], [154, 4], [124, 30], [154, 30]], True)
    for i in range(num.shape[1]):
        assert _rel(num[:, i], ana[:, i]) < 0.005, (i, _rel(num[:, i], ana[:, i]))


@pytest.mark.parametrize("dz", [0, 15])
def test_pressure_has_no_grid_line_modes(dz):
    # a point source excites nothing that runs along the grid lines through it: a receiver
    # on the source's row records the free-space pressure like one off it (with a continuous
    # flux and an element-local p, the row's record was 14x the analytic at dh = 2 and grew
    # like 1 / dh)
    _, num, ana = pressure_source_case((161, 161), [65, 80], [[95, 80 + dz]], False)
    assert _rel(num[:, 0], ana[:, 0]) < 0.005, _rel(num[:, 0], ana[:, 0])


def test_matches_the_fd_biot():
    # PropTorch's Biot and PropSEM on the same call: the FD stress injection is a moment of
    # dh^2 per unit wavelet sum, the SEM one of dt, and FD's vz sits half a cell down
    from sweep.equations import Biot
    from sweep.propagator.torch import PropTorch

    dh, dt, nt, f0 = 2.0, 2e-4, 900, 25.0
    shape = (81, 101)
    models = biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)
    w = _ricker(nt, dt, f0)
    src, rec = np.array([[30, 30]]), np.array([[[70, 50]]])
    fd = PropTorch(Biot(device="cpu"), shape=shape, dev="cpu", dh=dh, dt=dt, source_type=["sxx", "szz"],
                   receiver_type=["vz"], abcn=40, backend="torch", impl="eager")
    sem = _prop(shape, dh, dt, ["sxx", "szz"], ["vz"])
    with torch.no_grad():
        a = fd(w, src, rec, models=_fields(models, shape))[0, :, 0, 0].double().numpy()
        b = sem(w, src, rec + np.array([0, 0.5]), models=_fields(models, shape))[0, :, 0, 0].double().numpy()
    b = b * dh**2 / dt
    corr = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    assert corr > 0.995, corr
    assert np.linalg.norm(b) / np.linalg.norm(a) == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------------------------------------ #
# gradient
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("free_surface", [False, True])
@pytest.mark.parametrize("which", [1, 3, 7])   # mu, M, b
def test_gradient_matches_finite_difference(free_surface, which):
    shape, dh, dt, nt = (33, 41), 10.0, 5e-4, 360
    f64 = torch.float64
    rock = _layered_rock(shape)
    rock["kappa"] = 1e-10                          # gamma dt ~ 1: the drag's exponential matters
    coef = _fields(biot_from_rock(**rock), shape, f64)
    src = np.array([[20, 10]])
    rec = np.stack([np.arange(6, 36, 4), np.full(8, 24)], 1)[None]
    w = _ricker(nt, dt, 12.0, f64)
    prop = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz", "p"], dtype=f64, f0=12.0,
                 free_surface=free_surface)
    obs = prop(w, src, rec, models=coef).detach()

    def J(m):
        models = list(coef)
        models[which] = m
        return 0.5 * (prop(w, src, rec, models=models) - 1.05 * obs).square().sum()

    m0 = coef[which].clone().requires_grad_()
    (g,) = torch.autograd.grad(J(m0), m0)
    zz, xx = torch.meshgrid(torch.arange(shape[0], dtype=f64), torch.arange(shape[1], dtype=f64),
                            indexing="ij")
    dm = 1e-3 * coef[which] * torch.exp(-((zz - 18.0) ** 2 + (xx - 16.0) ** 2) / 30.0)
    pred = float((g * dm).sum())
    with torch.no_grad():
        fd = (float(J(coef[which] + dm)) - float(J(coef[which] - dm))) / 2.0
    assert fd == pytest.approx(pred, rel=1e-5), (fd, pred)


# ------------------------------------------------------------------------------------------ #
# sources, receivers, wavefields
# ------------------------------------------------------------------------------------------ #
def _small(src=None, rec=None, **kw):
    shape = (21, 25)
    prop = _prop(shape, 10.0, 5e-4, src, rec, f0=12.0, **kw)
    coef = _fields(biot_from_biot_constants(**DAI1, b=3e3, phi=0.3), shape)
    return prop, coef


def test_source_names_are_fd_biots():
    p, _ = _small()
    assert p.source_type == ["sxx", "szz"] and p._src_moment == (1.0, 1.0, 0.0)
    p, _ = _small(["sxz"])
    assert p._src_moment == (0.0, 0.0, 1.0)
    p, _ = _small(["p"])
    assert p._src_moment is None and p._source_vector() == (0.0, 0.0, 1.0)
    p, _ = _small(["vz", "p"])
    assert p._source_vector() == (0.0, 1.0, 1.0)


@pytest.mark.parametrize("names, match", [
    (["vz", "sxx"], "mixes an injected field"),
    (["sxx", "szz", "p"], "mixes an injected field"),
    (["vz", "v_z"], "same field"),
    (["u_z"], "Unknown source_type entry 'u_z'"),
    (["sxx", "sxx"], "duplicate"),
])
def test_bad_source_type_is_refused(names, match):
    with pytest.raises(ValueError, match=match):
        _small(names)


@pytest.mark.parametrize("name", ["qx", "q_z"])
def test_a_flux_source_is_refused(name):
    with pytest.raises(NotImplementedError, match="element-local"):
        _small([name])


@pytest.mark.parametrize("field", [0, 1])
def test_a_velocity_source_moves_only_the_solid(field):
    # one step from rest: vx / vz inject dt src / M_rho into v, the flux does not move,
    # whatever the drag
    p, coef = _small(["vz"])
    eq = p._build_eq(coef)
    assert eq.source_mode == "velocity"
    state = eq.init_state(p.dt, "cpu", torch.float32)
    load = torch.zeros(eq.mesh.n_global).index_fill(0, torch.tensor([eq.mesh.n_global // 2]), 1.0)
    src = [torch.zeros_like(load) for _ in range(3)]
    src[field] = load
    out = eq.step(state, src, p.dt)
    torch.testing.assert_close(out[field], p.dt * eq._inv_M * load, rtol=0, atol=0)
    torch.testing.assert_close(out[1 - field], torch.zeros_like(load), rtol=0, atol=0)
    for k in (3, 4):                                      # q_x, q_z
        assert not out[k].any()


def test_a_stress_source_is_a_load():
    p, coef = _small(["sxz"])
    assert p._build_eq(coef).source_mode == "load"


def test_default_receivers_are_the_solid_velocity():
    p, coef = _small()
    assert p.receiver_type == ["v_x", "v_z"]
    out = p(_ricker(80, 5e-4, 12.0), np.array([[12, 10]]), np.array([[[6, 6], [18, 14]]]), models=coef)
    assert out.shape == (1, 80, 2, 2)


def test_pressure_flux_and_stress_receivers_on_shots_and_snapshots():
    # the flux and the stress are element-local: they are gathered from their own layout,
    # in the vmapped multi-shot batch exactly as in the one-shot loop, and snapshotted on
    # the grid with the continuous p and v
    p, coef = _small(["sxx", "szz"], ["p", "sxx", "qz", "vz"])
    w = _ricker(120, 5e-4, 12.0)
    src = np.array([[12, 10], [8, 6]])
    rec = np.array([[[6, 7], [18, 14], [10, 3]]])
    with torch.no_grad():
        both = p(w, src, rec, models=coef)
        one = torch.cat([p(w, src[i:i + 1], rec, models=coef) for i in range(2)])
    assert both.shape == (2, 120, 3, 4) and bool((both.abs().amax((0, 1, 2)) > 0).all())
    torch.testing.assert_close(both, one, rtol=0, atol=0)
    with torch.no_grad():
        seis, snaps = p(w, src[:1], rec, models=coef, return_wavefield=True,
                        snapshot_times=[40, 100])
    assert snaps.shape == (2, 4, 1, 1) + p._grid_shape()
    torch.testing.assert_close(seis, one[:1], rtol=0, atol=0)


def test_boundary_saving_and_cuda_are_refused():
    from sweep.propagator.options import BoundarySaving
    with pytest.raises(NotImplementedError, match="no boundary saving"):
        _small(memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="no CUDA kernel"):
        _small(impl="c")


def test_source_vector_takes_one_entry_per_source_field():
    p, coef = _small(["vz"])
    rec = np.array([[[6, 6]]])
    with pytest.raises(ValueError, match="one entry per source field"):
        p(_ricker(40, 5e-4, 12.0), np.array([[12, 10]]), rec, models=coef, source_vector=(0.0, 1.0))
    a = p(_ricker(40, 5e-4, 12.0), np.array([[12, 10]]), rec, models=coef,
          source_vector=(0.0, 1.0, 0.0))
    b = p(_ricker(40, 5e-4, 12.0), np.array([[12, 10]]), rec, models=coef)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_pml_is_designed_for_the_fast_p_wave():
    # not for the first model, which is a modulus here
    p, coef = _small()
    assert p._pml_speed(coef) == pytest.approx(biot_fast_p_velocity(coef))
    assert 3000.0 < p._pml_speed(coef) < 3500.0

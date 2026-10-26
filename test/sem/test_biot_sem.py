"""BiotVSSEM2DDeformed (2-D velocity-stress Biot SEM): the undrained limit, the analytic line
source, the open-pore surface, gradient, sources and receivers, refusals.

The references are the FD ``Biot`` tests' (test/test_biot.py). Time levels: a velocity sample
``n`` is ``v^{n+1/2}``, a stress / pressure sample ``sigma^{n+1}``, and a stress or pressure
source's integrated wavelet leads its rate by ``dt / 2`` -- so the records sit ``dt`` (velocity)
and ``3 dt / 2`` (pressure) after the step's time ``n dt``.
"""

import functools
import math

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_fast_p_velocity, biot_from_biot_constants, biot_from_rock
from sweep.sem import (BiotAnisoPermSEM2DDeformed, BiotVSSEM2DDeformed,
                       ElasticVSSEM2DDeformed, PropSEM)
from test_biot import DAI1, _dwn_halfspace, _layered_rock

NO = 4
# Two elements of C-PML instead of PropSEM's ten, for tests whose C-PML is not under test: a
# gradient against finite differences (the discrete adjoint, whatever the layer), records that
# end before the first echo; on these small models the layer is most of the elements.
THIN_ABCN = 2 * NO


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
@pytest.mark.parametrize("source_type, elastic_source", [
    (["sxx", "szz"], ["sxx", "szz"]), (["vz"], ["vz"]), (["fz"], ["vz"])])
def test_no_relative_flow_limit_is_the_undrained_elastic(source_type, elastic_source, hill):
    # b -> infinity freezes q: v and the total stress obey ElasticVSSEM2DDeformed in the
    # undrained medium, open-pore surface and all (no fluid crosses it), flat or under a
    # topography; a force on both the solid and the fluid (fz) is then a force on the
    # solid. Two layers, the models given on the GLL nodes so that both equations see
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
    de = _prop(shape, dh, dt, elastic_source, ["vx", "vz"], eq=ElasticVSSEM2DDeformed, **kw)(
        w, src, rec, models=[torch.sqrt(h / rho), torch.sqrt(mu / rho), rho])
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-2, err


def analytic_line_source(models, t, x, z, s_time, force, b_of_w=None):
    """``(vz, p)`` of a line source in a homogeneous Biot medium, the source as in
    :func:`_analytic_line_source_vz` with forces ``force = (total, relative) * grad Phi``,
    ``Phi = S(t) delta``, exp(+i w t). With ``u = grad Psi_u``, ``w = grad Psi_w`` and
    ``p = -C div u - M div w``, mode ``k`` adds ``E_u a k S (z / r) (-w / 4) H1(k r)`` to vz
    and ``k^2 (C E_u + M E_w) a S (-i/4) H0(k r)`` to p. ``b_of_w(w)`` replaces the drag
    (complex, frequency dependent); default the constant ``b``."""
    from scipy.special import hankel2e

    h, mu, c, biot_m, rho, rho_f, m, b = (float(v) for v in models)
    dt, nt = t[1] - t[0], len(t)
    n = 16 * nt
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))
    w = 2 * math.pi * np.fft.rfftfreq(n, dt)
    r = math.hypot(x, z)
    K = np.array([[h, c], [c, biot_m]], dtype=complex)
    F = np.asarray(force, dtype=complex)
    v_hat, p_hat = np.zeros_like(s_hat), np.zeros_like(s_hat)
    for i in np.nonzero(np.abs(s_hat) > 1e-14 * np.abs(s_hat).max())[0]:
        if w[i] == 0:
            continue
        bw = b if b_of_w is None else b_of_w(w[i])
        s2, E = np.linalg.eig(np.linalg.solve(K, np.array([[rho, rho_f], [rho_f, m - 1j * bw / w[i]]])))
        s = np.sqrt(s2)
        s = np.where(s.real < 0, -s, s)
        a = np.linalg.solve(K @ E, F)
        k = w[i] * s
        v_hat[i] = -(w[i] / 4) * (z / r) * s_hat[i] * np.sum(
            E[0] * a * k * hankel2e(1, k * r) * np.exp(-1j * k * r))
        p_hat[i] = -0.25j * s_hat[i] * np.sum(
            k**2 * (c * E[0] + biot_m * E[1]) * a * hankel2e(0, k * r) * np.exp(-1j * k * r))
    return np.fft.irfft(v_hat, n)[:nt], np.fft.irfft(p_hat, n)[:nt]


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


@functools.lru_cache(maxsize=None)
def line_source_case(b_dai, dev="cpu", viscous_q=None):
    """``(t, sem (nt, 2), analytic (nt, 2))``: solid vz and pore pressure of the split source
    in Dai's medium, 60 m off in x and z. The model keeps the C-PML's echoes out of the record
    (they reach a few percent of the direct waves; six elements of it instead of ten add at
    most 6e-4 to the errors). ``viscous_q`` turns on the viscous attenuation, the analytic drag
    then ``b (1 + i w tau_e) / (1 + i w tau_s)``. Cached: the plain b = 7e4 case serves three
    tests."""
    from sweep.sem.equations.biot_vs_deformed import viscous_relaxation_times

    dh, dt, nt, f0, t0, qpart, phi = 4.0, 2e-4, 800, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    kw = {} if viscous_q is None else {"equation_options": {"viscous_q": viscous_q}}
    t, num = _split_source((121, 121), dh, dt, nt, models, np.array([[45, 45]]),
                           np.array([[[60, 60]]]), ["vz", "p"], qpart=qpart, phi=phi, f0=f0,
                           t0=t0, dev=dev, abcn=6 * NO, **kw)
    b_of_w = None
    if viscous_q is not None:
        tau_e, tau_s = viscous_relaxation_times(*viscous_q)
        b = float(models[7])
        b_of_w = lambda w: b * (1 + 1j * w * tau_e) / (1 + 1j * w * tau_s)
    s_time = _gauss(f0, t0)
    vz, _ = analytic_line_source(models, t, 60.0, 60.0, lambda tt: s_time(tt + dt),
                                 (1.0, qpart / phi), b_of_w)
    _, p = analytic_line_source(models, t, 60.0, 60.0, lambda tt: s_time(tt + 1.5 * dt),
                                (1.0, qpart / phi), b_of_w)
    return t, num[:, 0], np.stack([vz, p], 1)


def open_pore_case(b_dai, dev="cpu"):
    """``(t, sem (nt, 2), wavenumber integration (nt, 2))``: solid vx, vz on the open-pore
    surface 60 m from the split source 20 m below it, the model just large enough that the
    C-PML's echoes reach no more than the end of the record."""
    dh, dt, nt, f0, t0, qpart, phi = 2.0, 1e-4, 1200, 30.0, 0.04, 0.1, 0.3
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    xs, zs, xr = 65, 10, 95
    t, num = _split_source((85, 161), dh, dt, nt, models, np.array([[xs, zs]]),
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
# spacing, and the nodes there are fully drained: 0.9% on 8 m elements 60 m from the source,
# 1.2% at 100 m (FD Biot there: 1.3% on 2 m cells). With the layer resolved (b = 7e5) the error
# falls to 1e-4 under refinement.
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
    prop = _prop(shape, dh, dt, ["p"], ["p"], free_surface=free_surface, dev=dev,
                 abcn=THIN_ABCN)
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


def test_pressure_has_no_grid_line_modes():
    # a point source excites nothing that runs along the grid lines through it: a receiver
    # on the source's row records the free-space pressure like one off it (with a continuous
    # flux and an element-local p, the row's record was 14x the analytic at dh = 2 and grew
    # like 1 / dh)
    _, num, ana = pressure_source_case((161, 161), [65, 80], [[95, 80], [95, 95]], False)
    for i in range(2):
        assert _rel(num[:, i], ana[:, i]) < 0.005, (i, _rel(num[:, i], ana[:, i]))


def test_matches_the_fd_biot():
    # PropTorch's Biot and PropSEM on the same call: the FD stress injection is a moment of
    # dh^2 per unit wavelet sum, the SEM one of dt, and FD's vz sits half a cell down
    from sweep.equations import Biot
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    dh, dt, nt, f0 = 2.0, 2e-4, 900, 25.0
    shape = (81, 101)
    models = biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)
    w = _ricker(nt, dt, f0)
    src, rec = np.array([[30, 30]]), np.array([[[70, 50]]])
    # uncompiled: no other test steps this grid, which a cold compile makes 8 - 20 s, not 1.3 s
    fd = PropTorch(Biot(device="cpu"), shape=shape, dev="cpu", dh=dh, dt=dt, source_type=["sxx", "szz"],
                   receiver_type=["vz"], abcn=40, backend="torch", impl="eager",
                   eager_options=EagerOptions(use_compile=False))
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
# The finite-difference checks: a central difference of a 1e-5 relative step against the
# adjoint, to 1e-6 and no absolute floor (|J' dm| is 1e-21 - 1e-16 here: pytest.approx's default
# abs = 1e-12 would pass any gradient). Measured 1e-10 - 1e-7, the largest the step's own
# truncation (M in the water); a 1e-3 step was 1e-3 off, edge cells 2e-4, a moving C-PML 2e-3.
FD_STEP = 1e-5


def _fd_matches(J, coef, which, grad, dm):
    pred = float((grad * dm).sum())
    with torch.no_grad():
        fd = (float(J([c + dm if k == which else c for k, c in enumerate(coef)]))
              - float(J([c - dm if k == which else c for k, c in enumerate(coef)]))) / 2.0
    assert fd == pytest.approx(pred, rel=1e-6, abs=0), (which, fd, pred)


def _interior_bump(shape):
    """A Gaussian bump on the model grid, zero on its outermost cells: the C-PML copies those
    outward and the gradient is the physical box's, so theirs leaves the copies out."""
    zz, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij")
    bump = torch.exp(-((zz - 18.0) ** 2 + (xx - 16.0) ** 2) / 30.0)
    bump[[0, -1]] = 0.0
    bump[:, [0, -1]] = 0.0
    return bump


@functools.lru_cache(maxsize=None)
def _gradient_case(free_surface=False, variant=None):
    """``(coef, J, grads)``: the misfit of a layered rock and its gradient to every
    coefficient, one backward for every coefficient checked. The C-PML is laid out for the
    model's fast P speed, a float the gradient does not follow: held, or a step in M would
    move the layer too."""
    shape, dh, dt, nt = (33, 41), 10.0, 5e-4, 360
    f64 = torch.float64
    rock = _layered_rock(shape)
    rock["kappa"] = 1e-10                          # gamma dt ~ 1: the drag's exponential matters
    coef = _fields(biot_from_rock(**rock), shape, f64)
    eq, kw = BiotVSSEM2DDeformed, {}
    if variant == "tensor":
        eq = BiotAnisoPermSEM2DDeformed
        coef = coef[:7] + [coef[7], 0.3 * coef[7], 2.0 * coef[7]]
    elif variant == "viscous":
        kw["equation_options"] = {"viscous_q": (5.0, 12.0)}
    src = np.array([[20, 10]])
    rec = np.stack([np.arange(6, 36, 4), np.full(8, 24)], 1)[None]
    w = _ricker(nt, dt, 12.0, f64)
    prop = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz", "p"], eq=eq, dtype=f64, f0=12.0,
                 free_surface=free_surface, abcn=THIN_ABCN, **kw)
    c_max = prop._pml_speed([prop._interp_to_gll(c) for c in coef])
    prop._pml_speed = lambda models: c_max
    obs = prop(w, src, rec, models=coef).detach()

    def J(models):
        return 0.5 * (prop(w, src, rec, models=models) - 1.05 * obs).square().sum()

    m0 = [c.clone().requires_grad_() for c in coef]
    return coef, J, torch.autograd.grad(J(m0), m0)


@pytest.mark.parametrize("free_surface", [False, True])
@pytest.mark.parametrize("which", [1, 3, 7])   # mu, M, b
def test_gradient_matches_finite_difference(free_surface, which):
    coef, J, grads = _gradient_case(free_surface)
    dm = FD_STEP * coef[which] * _interior_bump(coef[which].shape)
    _fd_matches(J, coef, which, grads[which], dm)


@pytest.mark.parametrize("variant", ["tensor", "viscous"])
def test_gradient_through_the_drag_tensor_and_the_viscous_memory(variant):
    # the per-node matrix exponential and the memory variable carry the gradient too
    coef, J, grads = _gradient_case(variant=variant)
    which = 8 if variant == "tensor" else 7                         # b_xz, b
    _fd_matches(J, coef, which, grads[which], FD_STEP * coef[7] * _interior_bump(coef[7].shape))


# ------------------------------------------------------------------------------------------ #
# sources, receivers, wavefields
# ------------------------------------------------------------------------------------------ #
def _small(src=None, rec=None, **kw):
    shape = (21, 25)
    prop = _prop(shape, 10.0, 5e-4, src, rec, f0=12.0, **kw)
    coef = _fields(biot_from_biot_constants(**DAI1, b=3e3, phi=0.3), shape)
    return prop, coef


def test_source_names_are_fd_biots_and_specfems():
    # payloads over (v_x, v_z, p, q_x, q_z); the stress names are a moment tensor
    p, _ = _small()
    assert p.source_type == ["sxx", "szz"] and p._src_moment == (1.0, 1.0, 0.0)
    assert p._src_mode == "load"
    for names, payload, mode in ((["p"], (0, 0, 1, 0, 0), "velocity"),
                                 (["vz", "p", "qx"], (0, 1, 1, 1, 0), "velocity"),
                                 (["fz"], (0, 1, 0, 0, 1), "load"),
                                 (["fx", "p"], (1, 0, 1, 1, 0), "load")):
        p, _ = _small(names)
        assert p._src_moment is None and p._source_vector() == payload, names
        assert p._src_mode == mode, names


@pytest.mark.parametrize("names, match", [
    (["vz", "sxx"], "mixes a stress source"),
    (["sxx", "szz", "p"], "mixes a stress source"),
    (["vz", "v_z"], "both drive v_z"),
    (["fz", "qz"], "both drive q_z"),
    (["fx", "vz"], "mixes an injected velocity with a force"),
    (["u_z"], "Unknown source_type entry 'u_z'"),
    (["sxx", "sxx"], "duplicate"),
])
def test_bad_source_type_is_refused(names, match):
    with pytest.raises(ValueError, match=match):
        _small(names)


def _one_step(names, field):
    """One step from rest with a unit load on the node mid-mesh in source field ``field``."""
    p, coef = _small(names)
    eq = p._build_eq(coef)
    state = eq.init_state(p.dt, "cpu", torch.float32)
    load = torch.zeros(eq.mesh.n_global).index_fill(0, torch.tensor([eq.mesh.n_global // 2]), 1.0)
    src = [torch.zeros_like(load) for _ in range(5)]
    src[field] = load
    return p, eq, load, eq.step(state, src, p.dt)


@pytest.mark.parametrize("field", [0, 1])
def test_a_velocity_source_moves_only_the_solid(field):
    # vx / vz inject dt src / M_rho into v, the flux does not move, whatever the drag
    p, eq, load, out = _one_step(["vz"], field)
    assert eq.source_mode == "velocity"
    torch.testing.assert_close(out[field], p.dt * eq._inv_M * load, rtol=0, atol=0)
    torch.testing.assert_close(out[1 - field], torch.zeros_like(load), rtol=0, atol=0)
    for k in (3, 4):                                      # q_x, q_z
        assert not out[k].any()


@pytest.mark.parametrize("field", [3, 4])
def test_a_flux_source_moves_only_the_fluid(field):
    # qx / qz: every element copy of the flux at the node jumps by dt src / A(w m), v not
    p, eq, load, out = _one_step(["qz"], field)
    want = (p.dt * eq._inv_Mm * load)[eq.mesh.ibool]
    torch.testing.assert_close(out[field], want, rtol=0, atol=0)
    assert not out[7 - field].any() and not out[0].any() and not out[1].any()


def test_a_force_loads_both_equations():
    # fz is SPECFEM2D's poroelastic point force: the same load on the total momentum and on
    # the relative flow, i.e. on the solid and the fluid in proportion 1 - phi : phi
    from sweep.sem.prop_base import _Acq
    p, coef = _small(["fz"])
    eq = p._build_eq(coef)
    assert eq.source_mode == "load"
    w = _ricker(40, 5e-4, 12.0)
    p._set_nt(w)
    acq = _Acq(p, w, np.array([[12, 10]]), np.array([[[6, 6]]]), None)
    (prop,) = p._build_props(eq, None, acq, "warn")
    pats = prop._build_source_patterns(eq.mesh.n_global, "cpu", torch.float32)
    assert pats[1].any()
    torch.testing.assert_close(pats[4], pats[1], rtol=0, atol=0)
    for k in (0, 2, 3):
        assert not pats[k].any()


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
    # in the vmapped multi-shot batch as in the one-shot loop, and snapshotted on the grid
    # with the continuous p and v. The batch runs batched kernels, so it agrees to rounding
    # (torch 2.14 on CPU: 2e-6 of each field's peak), not bitwise.
    p, coef = _small(["sxx", "szz"], ["p", "sxx", "qz", "vz"])
    w = _ricker(120, 5e-4, 12.0)
    src = np.array([[12, 10], [8, 6]])
    rec = np.array([[[6, 7], [18, 14], [10, 3]]])
    with torch.no_grad():
        both = p(w, src, rec, models=coef)
        one = torch.cat([p(w, src[i:i + 1], rec, models=coef) for i in range(2)])
    peak = one.abs().amax((0, 1, 2))
    assert both.shape == (2, 120, 3, 4) and bool((peak > 0).all())
    torch.testing.assert_close(both / peak, one / peak, rtol=0, atol=1e-5)
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
          source_vector=(0.0, 1.0, 0.0, 0.0, 0.0))
    b = p(_ricker(40, 5e-4, 12.0), np.array([[12, 10]]), rec, models=coef)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_pml_is_designed_for_the_fast_p_wave():
    # not for the first model, which is a modulus here
    p, coef = _small()
    assert p._pml_speed(coef) == pytest.approx(biot_fast_p_velocity(coef))
    assert 3000.0 < p._pml_speed(coef) < 3500.0


# ------------------------------------------------------------------------------------------ #
# anisotropic permeability, viscous attenuation, parameter families, options
# ------------------------------------------------------------------------------------------ #
def _explosion(eq, models, rec, shape=(33, 33), src=(16, 16), nt=300, **kw):
    f64 = torch.float64
    prop = _prop(shape, 4.0, 2e-4, ["sxx", "szz"], ["vx", "vz", "p"], eq=eq, dtype=f64,
                 abcn=THIN_ABCN, **kw)
    with torch.no_grad():
        return prop(_ricker(nt, 2e-4, 30.0, f64), np.array([src]), np.array([rec]),
                    models=_fields(models, shape, f64))[0].numpy()


def test_an_isotropic_drag_tensor_is_the_scalar_drag():
    models = biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)
    b = models[7]
    rec = [[24, 8], [8, 26]]
    iso = _explosion(BiotVSSEM2DDeformed, models, rec)
    tensor = _explosion(BiotAnisoPermSEM2DDeformed, list(models[:7]) + [b, 0.0, b], rec)
    assert _rel(tensor, iso) < 1e-12, _rel(tensor, iso)


def test_the_drag_tensor_turns_with_the_medium():
    # mirroring about x = z swaps b_xx and b_zz, keeps b_xz, and swaps x and z of every
    # record: the off-diagonal coupling of the flux copies included
    from sweep.equations.biot_params import biot_drag_tensor

    models = list(biot_from_biot_constants(**DAI1, b=0.0, phi=0.3)[:7])
    b_xx, b_xz, b_zz = biot_drag_tensor(1e-3, 2e-9, -5e-10, 5e-10)
    a = _explosion(BiotAnisoPermSEM2DDeformed, models + [b_xx, b_xz, b_zz], [[24, 10]])
    b = _explosion(BiotAnisoPermSEM2DDeformed, models + [b_zz, b_xz, b_xx], [[10, 24]])
    assert _rel(b[:, 0, 1], a[:, 0, 0]) < 1e-10 and _rel(b[:, 0, 0], a[:, 0, 1]) < 1e-10
    assert _rel(b[:, 0, 2], a[:, 0, 2]) < 1e-10
    # ... and the tensor is felt: the isotropic mean drag gives another record
    mean = 0.5 * (b_xx + b_zz)
    c = _explosion(BiotAnisoPermSEM2DDeformed, models + [mean, 0.0, mean], [[24, 10]])
    assert _rel(c[:, 0, 2], a[:, 0, 2]) > 1e-2


def test_a_drag_tensor_must_be_positive_semidefinite():
    models = list(biot_from_biot_constants(**DAI1, b=0.0, phi=0.3)[:7])
    with pytest.raises(ValueError, match="positive semi-definite"):
        _explosion(BiotAnisoPermSEM2DDeformed, models + [1e5, 2e5, 1e5], [[24, 10]])


@pytest.mark.parametrize("q0", [1.0, 5.0])
def test_viscous_attenuation_matches_the_analytic_relaxation(q0):
    # SPECFEM2D's ATTENUATION_PORO_FLUID_PART: the drag a standard linear solid, Q0 at 30 Hz,
    # against the analytic line source with b (1 + i w tau_e) / (1 + i w tau_s). With the
    # slow wave critically damped (b = 7e4) it moves the pore pressure by 1.3% (Q0 = 1) and
    # 0.3% (Q0 = 5) and the solid velocity by less than 0.1% -- so its own part, num - plain
    # against ana - ana_plain, is checked on the pore pressure: 2.5% and 1.1% off
    _, num, ana = line_source_case(7e4, viscous_q=(q0, 30.0))
    _, plain, ana0 = line_source_case(7e4)
    for k, comp in enumerate(("vz", "p")):
        assert _rel(num[:, k], ana[:, k]) < 0.005, (comp, _rel(num[:, k], ana[:, k]))
    assert _rel(ana[:, 1], ana0[:, 1]) > 0.002
    err = _rel(num[:, 1] - plain[:, 1], ana[:, 1] - ana0[:, 1])
    assert err < 0.05, err


def test_equation_options_are_checked():
    p, coef = _small(equation_options={"viscous_q": (5.0, 30.0)})
    assert p._build_eq(coef).viscous_q == (5.0, 30.0)
    with pytest.raises(TypeError, match="not options of BiotVSSEM2DDeformed.build"):
        _small(equation_options={"viscous": (5.0, 30.0)})
    p, coef = _small(equation_options={"viscous_q": (0.0, 30.0)})
    with pytest.raises(ValueError, match="Q0 > 0"):
        p._build_eq(coef)


@pytest.mark.parametrize("family", ["primary", "density", "wavespeed"])
@pytest.mark.parametrize("tensor", [False, True])
def test_parameter_families_round_trip_and_chain_rule(family, tensor):
    from sweep.equations.biot_params import (BIOT_FAMILIES, biot_family_kernels,
                                             biot_from_family, biot_to_family)
    rocks = [biot_from_biot_constants(**DAI1, b=b, phi=phi) for b, phi in ((0.0, 0.3), (7e4, 0.25))]
    coef = [torch.tensor([float(r[i]) for r in rocks], dtype=torch.float64) for i in range(8)]
    if tensor:
        coef = coef[:7] + [coef[7], 0.3 * coef[7], 2.0 * coef[7]]
    y = biot_to_family(family, coef)
    for got, want in zip(biot_from_family(family, y), coef):
        assert float(((got - want).abs() / want.abs().clamp_min(1e-300)).max()) < 1e-12
    # a nonlinear misfit of the coefficients; its gradient taken to the family against
    # central differences in the family's own coordinates, one medium at a time
    torch.manual_seed(0)
    weights = [torch.rand(2, dtype=torch.float64) / (c.abs() + 1.0) for c in coef]

    def misfit(cs):
        return sum(((w * c) ** 2).sum() for w, c in zip(weights, cs))

    cf = [c.clone().requires_grad_() for c in coef]
    grads = torch.autograd.grad(misfit(cf), cf)
    kern = biot_family_kernels(family, coef, grads, relative=False)
    names = list(BIOT_FAMILIES[family])
    if tensor:
        i = names.index("b")
        names[i:i + 1] = ["b_xx", "b_xz", "b_zz"]
    assert list(kern) == names
    for j, name in enumerate(names):
        for q in range(2):
            h = 1e-6 * float(y[j][q].abs().clamp_min(1e-30))
            step = torch.zeros(2, dtype=torch.float64)
            step[q] = h
            up = [yy + (step if k == j else 0.0) for k, yy in enumerate(y)]
            dn = [yy - (step if k == j else 0.0) for k, yy in enumerate(y)]
            fd = float(misfit(biot_from_family(family, up))
                       - misfit(biot_from_family(family, dn))) / (2 * h)
            assert float(kern[name][q]) == pytest.approx(fd, rel=1e-5, abs=1e-300), (name, q)
    rel = biot_family_kernels(family, coef, grads)
    for j, name in enumerate(names):
        assert torch.allclose(rel[name], y[j] * kern[name])



# ------------------------------------------------------------------------------------------ #
# fluid and dry media, their interfaces
# ------------------------------------------------------------------------------------------ #
WATER = (2.25e9, 1000.0)                      # k_f, rho_f
SHALE = (3000.0, 1600.0, 2200.0)              # vp, vs, rho


def _layered(mesh, tops, media, dtype=torch.float64):
    """GLL-native coefficients: an element takes ``media[k]`` when its centre lies between
    the depths ``tops[k]`` and ``tops[k + 1]`` -- each element wholly one medium."""
    zc = mesh.z_e.mean((1, 2))
    k = torch.bucketize(zc, torch.tensor(tops[1:], dtype=zc.dtype, device=zc.device))
    E, n = mesh.n_elements, mesh.ngll
    return [torch.stack([torch.as_tensor(m[i], dtype=dtype) for m in media]).to(zc.device)[k]
            .view(E, 1, 1).expand(E, n, n).contiguous() for i in range(len(media[0]))]


def _three_media(mesh, b=0.0, tops=(0.0, 60.0, 140.0)):
    """Water over Dai's porous rock (drag ``b``) over a dry shale."""
    from sweep.equations.biot_params import biot_elastic, biot_fluid
    poro = biot_from_biot_constants(**DAI1, b=b, phi=0.3)
    return _layered(mesh, list(tops), [biot_fluid(*WATER), poro, biot_elastic(*SHALE)])


def _node(mesh, ez, ex, a=2, b=2):
    """The global node at GLL ``(a, b)`` of element row ``ez``, column ``ex``."""
    return int(mesh.ibool[ez * mesh.nex + ex, a, b])


def test_a_dry_solid_is_the_elastic_sem():
    # no pore fluid anywhere (M = 0): v and the stress obey ElasticVSSEM2DDeformed and the
    # pressure and the flux stay zero
    from sweep.equations.biot_params import biot_elastic
    shape, dh, dt, nt = (33, 41), 10.0, 5e-4, 300
    f64 = torch.float64
    kw = dict(abcn=0, free_surface=True, dtype=f64)
    rocks = [(2500.0, 1400.0, 2200.0), SHALE]
    biot = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz", "p", "qz"], **kw)
    elas = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz"], eq=ElasticVSSEM2DDeformed, **kw)
    src, rec = np.array([[20, 10]]), np.stack([np.arange(4, 37, 4), np.full(9, 22)], 1)[None]
    w = _ricker(nt, dt, 12.0, f64)
    with torch.no_grad():
        db = biot(w, src, rec, models=_layered(biot.mesh, [0.0, 150.0],
                                               [biot_elastic(*r) for r in rocks]))
        de = elas(w, src, rec, models=_layered(elas.mesh, [0.0, 150.0], rocks))
    assert not db[..., 2:].any()
    err = float((db[..., :2] - de).abs().max() / de.abs().max())
    assert err < 1e-11, err


def test_water_over_a_dry_solid_is_the_fluid_solid_sem():
    # AcousticElasticSEM2D (potential and displacement, the same FluidSolidInterface) and the
    # Biot SEM with a fluid over a dry solid are one discrete system: p = -chi'' and v = u',
    # a force in the solid, the free surface on the water, no C-PML
    from sweep.equations.biot_params import biot_elastic, biot_fluid
    from sweep.sem.equations import AcousticElasticSEM2D
    f64 = torch.float64
    shape, dh, dt, nt = (41, 61), 5.0, 5e-4, 400
    prop = _prop(shape, dh, dt, ["fz"], ["vz"], abcn=0, free_surface=True, dtype=f64)
    mesh = prop.mesh
    tops = [0.0, 60.0]
    eq_b = prop._build_eq(_layered(mesh, tops, [biot_fluid(*WATER), biot_elastic(*SHALE)]))
    vp, vs, rho = _layered(mesh, tops, [(1500.0, 0.0, 1000.0), SHALE])
    eq_a = AcousticElasticSEM2D.build(mesh, vp, vs, rho, dirichlet=("top",))
    load = torch.zeros(mesh.n_global, dtype=f64)
    load[_node(mesh, 5, 7, 1, 3)] = 1.0
    w = _ricker(nt, dt, 20.0, f64)
    rec_v = [_node(mesh, 3, 4, 0, 0), _node(mesh, 6, 11)]        # the seafloor, in the rock
    rec_p = [_node(mesh, 1, 9), _node(mesh, 2, 3, 4, 1)]         # the water, the seafloor
    sb, sa = eq_b.init_state(dt, "cpu", f64), eq_a.init_state(dt, "cpu", f64)
    zero = torch.zeros_like(load)
    v, p, u, chi = [], [], [], []
    for n in range(nt):
        sb = eq_b.step(sb, [zero, w[n] * load, zero, zero, zero], dt)
        sa = eq_a.step(sa, (zero, zero, w[n] * load), dt)
        v.append(sb[1][rec_v]); p.append(sb[2][rec_p])
        u.append(sa[3][rec_v]); chi.append(sa[0][rec_p])
    v, p, u, chi = (torch.stack(x) for x in (v, p, u, chi))
    u = torch.cat([torch.zeros_like(u[:1]), u])                     # u^0 ... u^nt
    chi = torch.cat([torch.zeros_like(chi[:1]), chi])
    v_a = (u[1:] - u[:-1]) / dt                                     # v^{n+1/2}
    p_a = -(chi[2:] - 2 * chi[1:-1] + chi[:-2]) / dt**2             # p^{n+1} = -chi''^{n+1}
    assert float(v.abs().max()) > 0 and float(p.abs().max()) > 0
    torch.testing.assert_close(v, v_a, rtol=0, atol=1e-9 * float(v_a.abs().max()))
    torch.testing.assert_close(p[:-1], p_a, rtol=0, atol=1e-9 * float(p_a.abs().max()))


def _green(eq, dt, nt, field, src_node, rec_field, rec_node, w):
    """``rec_field`` at ``rec_node`` after each of ``nt`` steps from rest, a unit load
    ``w[n]`` at ``src_node``: on the momentum (``field`` 0, 1), or a fluid volume on the
    pressure equation (``field`` 2): ``p`` and, as the flow does, the total stress by
    ``-alpha dp`` -- not the source ``p``, which leaves the total stress and so carries a
    stress source in a porous rock."""
    state = eq.init_state(dt, "cpu", torch.float64)
    zero = torch.zeros(eq.mesh.n_global, dtype=torch.float64)
    ibool = eq.mesh.ibool
    out = []
    for n in range(nt):
        src = [zero] * 5
        if field == 2:
            dp = zero.clone()
            dp[src_node] = dt * w[n] * eq._inv_Mp[src_node]
            state = list(state)
            state[2] = state[2] + dp
            for k in (5, 6):                                    # sigma_xx, sigma_zz
                state[k] = state[k] - eq._alpha * dp[ibool]
        else:
            src[field] = zero.index_fill(0, torch.tensor([src_node]), 1.0) * w[n]
        state = eq.step(tuple(state), src, dt)
        out.append(state[rec_field][rec_node])
    return torch.stack(out)


@pytest.mark.parametrize("b", [0.0, 7e5])
def test_the_interfaces_are_reciprocal(b):
    # the two interface integrals are exact transposes and the sealed interface is the
    # natural one, so the discrete Green's functions are symmetric across every interface,
    # with the drag too: pressure-pressure water <-> rock and seafloor <-> rock,
    # velocity-velocity rock <-> shale and seafloor (x) <-> shale (z)
    f64 = torch.float64
    shape, dh, dt, nt = (41, 49), 5.0, 2e-4, 500
    prop = _prop(shape, dh, dt, None, None, f0=40.0, abcn=0, free_surface=True, dtype=f64)
    mesh = prop.mesh
    eq = prop._build_eq(_three_media(mesh, b))
    w = _ricker(nt, dt, 40.0, f64)
    a, c, g = _node(mesh, 1, 4), _node(mesh, 4, 8, 1, 3), _node(mesh, 2, 6, 4, 2)
    d, e, f = _node(mesh, 5, 3), _node(mesh, 8, 7, 3, 1), _node(mesh, 3, 9, 0, 2)
    for fw, bw in (((2, a, 2, c), (2, c, 2, a)), ((2, g, 2, c), (2, c, 2, g)),
                   ((1, d, 1, e), (1, e, 1, d)), ((0, f, 1, e), (1, e, 0, f))):
        x, y = _green(eq, dt, nt, *fw, w), _green(eq, dt, nt, *bw, w)
        assert float(x.abs().max()) > 0
        torch.testing.assert_close(x, y, rtol=0, atol=1e-10 * float(x.abs().max()))


def test_an_undrained_porous_layer_on_a_dry_solid_is_two_elastic_layers():
    # b -> inf: the porous layer is its undrained elastic solid and the sealed interface asks
    # nothing more of it. Not to round-off, for the reason in
    # test_no_relative_flow_limit_is_the_undrained_elastic
    from sweep.equations.biot_params import biot_elastic
    shape, dh, dt, nt = (41, 49), 10.0, 5e-4, 500
    f64 = torch.float64
    kw = dict(abcn=0, free_surface=True, dtype=f64)
    biot = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz"], **kw)
    rock = dict(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, k_d=8e9, mu=6e9,
                phi=0.25, kappa=1e-13, tortuosity=2.0)
    poro = [float(c) for c in biot_from_rock(**rock)]
    poro[7] = 1e16
    coef = _layered(biot.mesh, [0.0, 200.0], [poro, biot_elastic(*SHALE)])
    h, mu, rho = poro[0], poro[1], poro[4]
    el = _layered(biot.mesh, [0.0, 200.0], [((h / rho) ** 0.5, (mu / rho) ** 0.5, rho), SHALE])
    src, rec = np.array([[24, 18]]), np.stack([np.arange(4, 46, 3), np.full(14, 30)], 1)[None]
    w = _ricker(nt, dt, 12.0, f64)
    with torch.no_grad():
        db = biot(w, src, rec, models=coef)
        de = _prop(shape, dh, dt, ["sxx", "szz"], ["vx", "vz"], eq=ElasticVSSEM2DDeformed, **kw)(
            w, src, rec, models=el)
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-2, err


@pytest.mark.parametrize("edit, match", [
    (lambda c, e: c[4].__setitem__((e, 0, 0), 1000.0), "on some of its nodes only"),
    (lambda c, e: [c[k].__setitem__(e, 0.0) for k in (3, 4)], "neither a solid frame"),
    (lambda c, e: c[3].__setitem__(e, -1.0), "M must be >= 0"),
    (lambda c, e: c[1].__setitem__(e, 1e9), r"fluid element \(rho = 0\) has nonzero \['mu'\]"),
    (lambda c, e: c[6].__setitem__(e, 0.0), "m must be > 0"),
])
def test_media_are_read_off_the_coefficients(edit, match):
    p = _prop((41, 49), 5.0, 2e-4, None, None, abcn=0, free_surface=True, dtype=torch.float64)
    coef = _three_media(p.mesh)
    edit(coef, 0)                                  # element 0 is water
    with pytest.raises(ValueError, match=match):
        p._build_eq(coef)


def test_a_dry_element_takes_no_fluid_coefficients():
    p = _prop((41, 49), 5.0, 2e-4, None, None, abcn=0, free_surface=True, dtype=torch.float64)
    coef = _three_media(p.mesh)
    coef[2][-1] = 1e9                              # C in the last (dry) element
    with pytest.raises(ValueError, match=r"dry elastic element \(M = 0\) has nonzero \['C'\]"):
        p._build_eq(coef)


def _patterns(names, src):
    from sweep.sem.prop_base import _Acq
    p = _prop((41, 49), 5.0, 2e-4, names, ["vz"], abcn=0, free_surface=True, dtype=torch.float64)
    eq = p._build_eq(_three_media(p.mesh))
    w = _ricker(40, 2e-4, 40.0, torch.float64)
    p._set_nt(w)
    (prop,) = p._build_props(eq, None, _Acq(p, w, np.array([src]), np.array([[[6, 6]]]), None),
                             "warn")
    return eq, prop._build_source_patterns(eq.mesh.n_global, "cpu", torch.float64)


@pytest.mark.parametrize("names, src", [(["sxx", "szz"], [21, 5]), (["vz"], [21, 5]),
                                        (["p"], [21, 33]), (["qz"], [21, 33])])
def test_a_source_needs_its_fields_in_its_medium(names, src):
    # grid row 5 is in the water, row 33 in the shale
    with pytest.raises(ValueError, match="none of which lives there"):
        _patterns(names, src)


def test_a_force_in_the_water_moves_the_water_only():
    # fz just above the seafloor: no load reaches the rock's nodes on the seafloor
    eq, pats = _patterns(["fz"], [21, 11])
    assert not pats[1].any() and pats[4].any()
    eq, pats = _patterns(["fz"], [21, 13])         # just below: the rock, both
    assert pats[1].any() and pats[4].any()


def test_a_stress_source_on_the_seafloor_loads_the_rock_alone():
    # on the water-rock edge the moment tensor takes the rock's one-sided derivative in full,
    # as on the free surface, not half of it
    from sweep.sem.propagator import _stamp_components, build_moment_stamp
    eq, pats = _patterns(["sxx", "szz"], [21, 12])
    stamp = build_moment_stamp(eq.mesh, 21 * 5.0, 12 * 5.0)
    rock = [s for s in stamp.parts if bool(eq._solid_e[s.elem_id])]
    assert len(stamp.parts) == 2 and len(rock) == 1
    comps = _stamp_components(rock[0], "moment_tensor", (1.0, 1.0, 0.0))
    g = eq.mesh.ibool[rock[0].elem_id].reshape(-1)
    for k in (0, 1):
        want = torch.zeros(eq.mesh.n_global, dtype=torch.float64).index_add(
            0, g, comps[k].reshape(-1).double())
        torch.testing.assert_close(pats[k], want)


@pytest.mark.parametrize("b", [0.0, 7e5])
def test_interfaces_through_the_pml_stay_stable(b):
    # the seafloor and the rock-shale interface cross the side bands with the plain
    # coupling: the records keep decaying (4 s runs: to 3% of the peak with the inviscid
    # slow wave, 0.1% with b = 7e5, never growing); here 1.2 s through five elements of
    # C-PML, the window maxima of the last 0.2 s against 0.4 - 0.6 s: 0.50 / 0.53 (b = 0),
    # 0.19 / 0.05 (ten elements: 0.48 / 0.53, 0.28 / 0.04)
    f64 = torch.float64
    shape, dh, dt, nt = (41, 49), 5.0, 2e-4, 6000
    prop = _prop(shape, dh, dt, ["p"], ["p", "vz"], f0=40.0, free_surface=True, dtype=f64,
                 abcn=5 * NO)
    w = _ricker(nt, dt, 40.0, f64)
    with torch.no_grad():
        out = prop(w, np.array([[24, 6]]), np.array([[[10, 6], [30, 20]]]),
                   models=_three_media(prop.mesh, b))[0]
    win = out.abs().reshape(6, 1000, 2, 2).amax(1)
    p_water, vz_rock = win[:, 0, 0], win[:, 1, 1]
    for x in (p_water, vz_rock):
        assert float(x[0]) > 0 and bool(torch.isfinite(x).all())
        assert float(x[-1]) < 0.75 * float(x[2]) and float(x[-1]) < 0.15 * float(x.max()), x


def test_gradient_through_the_three_media():
    # one coefficient of each medium; the coefficients a medium does not have get none, so
    # an update leaves the media as they are
    f64 = torch.float64
    shape, dh, dt, nt = (41, 49), 5.0, 2e-4, 500
    prop = _prop(shape, dh, dt, ["p"], ["p", "vx", "vz"], f0=40.0, free_surface=True, dtype=f64,
                 abcn=THIN_ABCN)
    coef = _three_media(prop.mesh, 7e5)
    src, rec = np.array([[24, 6]]), np.stack([np.arange(6, 44, 4), np.full(10, 20)], 1)[None]
    w = _ricker(nt, dt, 40.0, f64)
    obs = prop(w, src, rec, models=coef).detach()
    E, n = prop.mesh.n_elements, prop.mesh.ngll
    zc = prop.mesh.z_e.mean((1, 2))
    xc = prop._coords([24, 0])[0]                      # mid-model, the C-PML aside

    def J(models):
        return 0.5 * (prop(w, src, rec, models=models) - 1.05 * obs).square().sum()

    m0 = [c.clone().requires_grad_() for c in coef]
    J(m0).backward()
    water, dry = zc < 60, zc > 140
    for k in (0, 1, 2, 4, 5, 7):                          # H, mu, C, rho, rho_f, b
        assert not m0[k].grad[water].any(), k
    for k in (2, 3, 5, 6, 7):                             # C, M, rho_f, m, b
        assert not m0[k].grad[dry].any(), k
    for which, (z0, z1) in ((3, (0, 60)), (1, (60, 140)), (0, (140, 1e9))):   # M, mu, H
        sel = ((zc > z0) & (zc < z1)).to(f64).view(E, 1, 1).expand(E, n, n)
        bump = torch.exp(-((prop.mesh.x_e - xc) ** 2) / 800.0) * sel
        _fd_matches(J, coef, which, m0[which].grad, FD_STEP * coef[which] * bump)


def test_the_pml_speed_takes_every_medium():
    # Dai's fast P wave (3210 m/s) over the shale's 3000 and the water's 1500
    p = _prop((41, 49), 5.0, 2e-4, None, None, abcn=0, free_surface=True, dtype=torch.float64)
    coef = _three_media(p.mesh)
    fast = biot_fast_p_velocity(biot_from_biot_constants(**DAI1, b=0.0, phi=0.3))
    assert fast > SHALE[0] and p._pml_speed(coef) == pytest.approx(fast)
    zc = p.mesh.z_e.mean((1, 2))
    no_rock = (zc < 60) | (zc > 140)
    assert p._pml_speed([c[no_rock] for c in coef]) == pytest.approx(SHALE[0])
    assert p._pml_speed([c[zc < 60] for c in coef]) == pytest.approx((WATER[0] / WATER[1]) ** 0.5)

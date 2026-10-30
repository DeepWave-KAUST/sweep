"""BiotVSSEM3DDeformed (3-D velocity-stress Biot SEM): the dry and undrained limits against
ElasticVSSEM3DDeformed, the analytic point injection (CUDA), the flux copies and their drag,
the transposed flux and pressure operators on a curved mesh, the symmetry of the three axes,
gradient, sources and refusals.

Time levels as in the 2-D tests (test_biot_sem.py): a velocity sample ``n`` is
``v^{n+1/2}``, a pressure sample ``p^{n+1}``, and a pressure source's integrated wavelet leads
its rate by ``dt / 2``.
"""

import functools
import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from sweep.equations.biot import biot_from_biot_constants, biot_from_rock
from sweep.equations.biot_params import biot_elastic, biot_fluid
from sweep.propagator import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.sem import BiotVSSEM3DDeformed, ElasticVSSEM3DDeformed, PropSEM
from sweep.sem._assembly import phys_grads3, vector_divergence3
from sweep.sem.equations.biot_vs_deformed import viscous_relaxation_times
from sweep.sem.mesh.hex27 import build_topography_hex27
from test_biot import DAI1
from test_biot_sem import FD_STEP, _fd_matches, _gauss, _rel

NO = 4
F64 = torch.float64
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _prop(shape, dh, dt, src, rec, *, eq=BiotVSSEM3DDeformed, dtype=F64, f0=15.0, dev="cpu",
          **kw):
    kw.setdefault("eager_options", EagerOptions(use_compile=False))
    return PropSEM(eq, shape=shape, dh=dh, dt=dt, n_order=NO, f_dominant=f0, dev=dev,
                   impl="eager", dtype=dtype, source_type=src, receiver_type=rec, **kw)


def _ricker(nt, dt, f0, t0):
    a = (math.pi * f0 * (np.arange(nt) * dt - t0)) ** 2
    return torch.from_numpy((1 - 2 * a) * np.exp(-a))


def _layers(mesh, top, bottom, z=80.0):
    """GLL-native coefficients, one medium per element: ``bottom`` below ``z``, so that the
    equations compared see the same moduli (an interpolated grid model would mix the two
    media differently in ``H`` and in ``vp``)."""
    deep = (mesh.z_e.mean((1, 2, 3), keepdim=True) > z).expand(mesh.z_e.shape)
    return [torch.where(deep, torch.tensor(float(b), dtype=F64), torch.tensor(float(a), dtype=F64))
            for a, b in zip(top, bottom)]


def _gll(mesh, coef):
    return [torch.as_tensor(c, dtype=F64).expand(mesh.z_e.shape).contiguous() for c in coef]


# ------------------------------------------------------------------------------------------ #
# limits
# ------------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("source, elastic_source, hill", [
    (["fx"], ["vx"], False), (["vz"], ["vz"], False), (["fx"], ["vx"], True),
    (["sxx", "syy", "szz"], ["sxx", "syy", "szz"], True)])
def test_a_dry_solid_is_the_elastic_sem(source, elastic_source, hill):
    # no pore fluid (M = 0): v and the stress obey ElasticVSSEM3DDeformed, C-PML, free surface
    # (flat or a hill) and all, and the pressure and the flux stay zero; a force on both
    # equations (fx) and a velocity source (vz) are the elastic force, the moment tensor
    # the elastic one
    shape, dh, dt, nt = (17, 17, 17), 10.0, 5e-4, 160
    yy, xx = np.meshgrid(np.arange(17.0), np.arange(17.0), indexing="ij")
    rows = 1.0 + np.exp(-((yy - 8.0) ** 2 + (xx - 9.0) ** 2) / 16.0) if hill else None
    kw = dict(abcn=8, free_surface=True, topography=rows)
    rocks = [(2500.0, 1400.0, 2200.0), (3200.0, 1800.0, 2400.0)]
    dry = [[float(c) for c in biot_elastic(*(torch.tensor(v, dtype=F64) for v in r))]
           for r in rocks]
    biot = _prop(shape, dh, dt, source, ["vx", "vy", "vz", "p", "qz"], **kw)
    elas = _prop(shape, dh, dt, elastic_source, ["vx", "vy", "vz"], eq=ElasticVSSEM3DDeformed,
                 **kw)
    src = np.array([[8.6, 7.4, 6.3]])
    rec = np.array([[[4.0, 9.0, 10.0], [12.0, 9.0, 10.0], [8.0, 8.0, 3.0]]])
    w = _ricker(nt, dt, 15.0, 0.08)
    with torch.no_grad():
        db = biot(w, src, rec, models=_layers(biot.mesh, *dry))
        de = elas(w, src, rec, models=_layers(elas.mesh, *rocks))
    assert not db[..., 3:].any()
    err = float((db[..., :3] - de).abs().max() / de.abs().max())
    assert err < 1e-12, err


@pytest.mark.parametrize("source, elastic_source", [
    (["fx"], ["vx"]), (["fz"], ["vz"]), (["vz"], ["vz"])])
def test_no_relative_flow_limit_is_the_undrained_elastic(source, elastic_source):
    # b -> infinity freezes q: v and the total stress obey ElasticVSSEM3DDeformed in the
    # undrained medium. Not to round-off (see the 2-D test): the alpha^2 M part of the
    # undrained modulus passes through the continuous p, a mean of div v over the elements
    # at a node -- 1e-3 to 3e-3 on these 40 m elements, 6e-4 on 20 m ones with the source
    # off the element faces (on a face, where a force makes div v jump, it does not fall).
    shape, dh, dt, nt = (17, 17, 17), 10.0, 5e-4, 160
    kw = dict(abcn=8, free_surface=True)
    biot = _prop(shape, dh, dt, source, ["vx", "vy", "vz"], **kw)
    rocks = [dict(k_d=8e9, mu=6e9, phi=0.25), dict(k_d=14e9, mu=11e9, phi=0.15)]
    coef = _layers(biot.mesh, *[[float(c) for c in biot_from_rock(
        k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, kappa=1e-13,
        tortuosity=2.0, **r)] for r in rocks])
    coef[7] = torch.full_like(coef[7], 1e16)
    h, mu, rho = coef[0], coef[1], coef[4]
    src = np.array([[8.0, 7.0, 6.0]])
    rec = np.array([[[4.0, 9.0, 10.0], [8.0, 9.0, 10.0], [12.0, 9.0, 10.0], [8.0, 8.0, 3.0]]])
    w = _ricker(nt, dt, 15.0, 0.08)
    with torch.no_grad():
        db = biot(w, src, rec, models=coef)
        de = _prop(shape, dh, dt, elastic_source, ["vx", "vy", "vz"], eq=ElasticVSSEM3DDeformed,
                   **kw)(w, src, rec, models=[torch.sqrt(h / rho), torch.sqrt(mu / rho), rho])
    err = float((db - de).abs().max() / de.abs().max())
    assert err < 1e-2, err


# ------------------------------------------------------------------------------------------ #
# the analytic point injection
# ------------------------------------------------------------------------------------------ #
def analytic_point_source(models, t, x, y, z, s_time, force, b_of_w=None):
    """``(v (3, nt), p (nt,))`` at ``(x, y, z)`` from a point source in a homogeneous Biot
    medium, the 3-D :func:`test_biot_sem.analytic_line_source`: forces ``force = (total,
    relative) * grad Phi``, ``Phi = S delta`` (``S = s_time``), exp(+i w t) -- ``(1, 0)`` the
    explosion ``["sxx", "syy", "szz"]`` with the moment ``S``, ``(0, -1)`` the injection ``dp/dt
    = dS/dt delta``. The potentials ``Phi`` of ``u`` and ``w`` obey ``K lap Phi + w^2 R Phi =
    -F S delta``, ``R_22 = m - i b / w``; mode ``j`` of ``K^-1 R E = E diag(s^2)`` is ``psi_j =
    -a_j exp(-i k_j r) / (4 pi r)``, ``a = -(K E)^-1 F S``, ``k_j = w s_j``; ``u = grad Phi_u``
    and ``p = -C div u - M div w``."""
    h, mu, c, biot_m, rho, rho_f, m, b = (float(v) for v in models)
    dt, nt = t[1] - t[0], len(t)
    n = 16 * nt
    s_hat = np.fft.rfft(s_time(np.arange(n) * dt))
    w = 2 * math.pi * np.fft.rfftfreq(n, dt)
    r = math.sqrt(x * x + y * y + z * z)
    K = np.array([[h, c], [c, biot_m]], dtype=complex)
    v_hat = np.zeros((3, len(w)), complex)
    p_hat = np.zeros(len(w), complex)
    for i in np.nonzero(np.abs(s_hat) > 1e-14 * np.abs(s_hat).max())[0]:
        if w[i] == 0:
            continue
        bw = b if b_of_w is None else b_of_w(w[i])
        s2, E = np.linalg.eig(np.linalg.solve(
            K, np.array([[rho, rho_f], [rho_f, m - 1j * bw / w[i]]])))
        s = np.sqrt(s2)
        s = np.where(s.real < 0, -s, s)
        a = np.linalg.solve(K @ E, -np.asarray(force, dtype=complex) * s_hat[i])
        k = w[i] * s
        g = np.exp(-1j * k * r) / (4 * math.pi * r)
        u_r = np.sum(E[0] * a * (1j * k + 1.0 / r) * g)
        v_hat[:, i] = 1j * w[i] * u_r * np.array([x, y, z]) / r
        p_hat[i] = -np.sum((c * E[0] + biot_m * E[1]) * k ** 2 * a * g)
    return np.fft.irfft(v_hat, n)[:, :nt], np.fft.irfft(p_hat, n)[:nt]


@cuda
@pytest.mark.parametrize("b_dai, viscous_q", [(0.0, None), (7e3, None), (7e3, (5.0, 30.0))])
def test_point_injection_matches_analytic(b_dai, viscous_q):
    # Dai's medium, inviscid and with a drag that leaves the slow wave propagating at 30 Hz
    # but attenuated, also with the viscous attenuation. With more drag (Dai's b = 7e4) the
    # slow wave dies within a few metres of the source, and the solid's response to a pure
    # injection is its residue, which 16 m elements do not resolve: 17-41% there, 2-5% on 8 m
    # elements (the 2-D class alike: 10%, 1%). The receiver 55 m off in all three axes, the C-PML's
    # echoes reaching it at the end of the record; 20k elements, CUDA only.
    dh, dt, nt, f0, t0 = 4.0, 2e-4, 700, 30.0, 0.04
    shape, src, off = (61, 61, 61), (30, 30, 30), (10, 8, 5)
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=0.3)
    kw = {} if viscous_q is None else {"equation_options": {"viscous_q": viscous_q}}
    prop = _prop(shape, dh, dt, ["p"], ["vx", "vy", "vz", "p"], f0=f0, dev="cuda", abcn=24,
                 free_surface=False, **kw)
    gauss = _gauss(f0, t0)
    t = np.arange(nt) * dt
    with torch.no_grad():
        num = prop(torch.from_numpy(np.gradient(gauss(t), t)), np.array([src]),
                   np.array([[[s + o for s, o in zip(src, off)]]]),
                   models=[torch.full(shape, float(c), dtype=F64) for c in models])
    num = num[0, :, 0].cpu().numpy()
    b_of_w = None
    if viscous_q is not None:
        tau_e, tau_s = viscous_relaxation_times(*viscous_q)
        b = float(models[7])
        b_of_w = lambda w: b * (1 + 1j * w * tau_e) / (1 + 1j * w * tau_s)
    x, y, z = (o * dh for o in off)
    v, _ = analytic_point_source(models, t, x, y, z, lambda tt: gauss(tt + dt), (0.0, -1.0),
                                 b_of_w)
    _, p = analytic_point_source(models, t, x, y, z, lambda tt: gauss(tt + 1.5 * dt),
                                 (0.0, -1.0), b_of_w)
    for k, ref in enumerate((*v, p)):
        assert _rel(num[:, k], ref) < 0.01, (("vx", "vy", "vz", "p")[k], _rel(num[:, k], ref))


@cuda
@pytest.mark.parametrize("b_dai", [0.0, 7e4, 9e8])
def test_split_source_matches_analytic(b_dai):
    # the 2-D tests' split source in 3-D: the explosion on the total stress and qpart / phi
    # of it in the relative flow, from inviscid to the Darcy limit; the explosion drives the
    # solid, so the slow wave's unresolved residue near the source (see above) is a small
    # part of it. The fast P wave's echo from the C-PML is 2-3% of it, so the box is larger
    # than the injection's and the record ends as the echo arrives, after the slow wave:
    # 33k elements. (Alone, the injection's far field vanishes towards the Darcy limit: at
    # Dai's b = 9e8 it is 1e-6 of the explosion's, and the SEM's is the unresolved residue
    # of a diffusion a few centimetres long, 16 times that.)
    dh, dt, nt, f0, t0, qpart, phi = 4.0, 2e-4, 550, 30.0, 0.04, 0.1, 0.3
    shape, src, off = (81, 81, 81), (40, 40, 40), (10, 8, 5)
    models = biot_from_biot_constants(**DAI1, b=b_dai, phi=phi)
    coef = [torch.full(shape, float(c), dtype=F64) for c in models]
    gauss = _gauss(f0, t0)
    t = np.arange(nt) * dt
    rate = torch.from_numpy(np.gradient(gauss(t), t))
    rec = np.array([[[s + o for s, o in zip(src, off)]]])
    num = 0.0
    for names, scale in ((["sxx", "syy", "szz"], 1.0), (["p"], -qpart / phi)):
        prop = _prop(shape, dh, dt, names, ["vx", "vy", "vz", "p"], f0=f0, dev="cuda",
                     abcn=24, free_surface=False)
        with torch.no_grad():
            num = num + prop(rate * scale, np.array([src]), rec,
                             models=coef)[0, :, 0].cpu().numpy()
    x, y, z = (o * dh for o in off)
    force = (1.0, qpart / phi)
    v, _ = analytic_point_source(models, t, x, y, z, lambda tt: gauss(tt + dt), force)
    _, p = analytic_point_source(models, t, x, y, z, lambda tt: gauss(tt + 1.5 * dt), force)
    for k, ref in enumerate((*v, p)):
        assert _rel(num[:, k], ref) < 0.01, (("vx", "vy", "vz", "p")[k], _rel(num[:, k], ref))


# ------------------------------------------------------------------------------------------ #
# the flux copies, their drag, the operators
# ------------------------------------------------------------------------------------------ #
@functools.lru_cache(maxsize=None)
def _heterogeneous_eq():
    """An equation on a 3 x 3 x 3 box of 40 m elements (a corner node per element row) whose
    porosity and permeability change from element to element, dry in one, with viscous_q."""
    prop = _prop((13, 13, 13), 10.0, 5e-4, ["p"], ["vx"], abcn=0, free_surface=True)
    mesh = prop.mesh
    g = torch.Generator().manual_seed(3)
    e = mesh.n_elements
    phi = 0.1 + 0.25 * torch.rand(e, generator=g, dtype=F64)
    kappa = 10.0 ** (-13.0 + 4.0 * torch.rand(e, generator=g, dtype=F64))
    coef = [torch.as_tensor(c, dtype=F64).reshape(-1, 1, 1, 1).expand(mesh.z_e.shape).clone()
            for c in biot_from_rock(
        k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3, k_d=9e9, mu=7e9, phi=phi,
        kappa=kappa, tortuosity=2.0)]
    dry = [float(c) for c in biot_elastic(*(torch.tensor(v, dtype=F64)
                                          for v in (3000.0, 1700.0, 2300.0)))]
    for k, c in enumerate(dry):
        coef[k][5] = c
    eq = BiotVSSEM3DDeformed.build(mesh, *coef, dirichlet=("top",), viscous_q=(5.0, 15.0))
    return eq, coef


def test_the_flux_copies_are_every_element_node_once():
    eq, _ = _heterogeneous_eq()
    ibool = eq.mesh.ibool
    x = torch.randn(ibool.shape, dtype=F64, generator=torch.Generator().manual_seed(0))
    assert torch.equal(eq._from_blocks(eq._to_blocks(x), x), x)
    counts = torch.bincount(ibool.reshape(-1), minlength=eq.mesh.n_global)
    assert sorted(blk.k for blk in eq._blocks) == [1, 2, 4, 8]
    for blk, ids in zip(eq._blocks, eq._to_blocks(ibool.to(F64))):
        assert bool((counts[blk.nodes] == blk.k).all())
        assert torch.equal(ids, blk.nodes.to(F64)[:, None].expand_as(ids))   # a row: one node


def test_the_drag_operators_are_each_nodes_exponential():
    # at a sample of nodes of every group, E and Psi against the 2-D construction from the
    # element copies there, found in ibool order: a mixed-up copy would only show in a
    # medium that changes from element to element
    eq, (h, mu, c, bm, rho, rho_f, m, b) = _heterogeneous_eq()
    dt = 5e-4
    Es, Psis = eq.drag_operators(dt)
    flat = eq.mesh.ibool.reshape(-1)
    w = eq.mesh.w_jac.reshape(-1)
    fluid = (bm > 0).reshape(-1)
    tau_e, tau_s = eq._tau
    for blk, E, Psi in zip(eq._blocks, Es, Psis):
        for i in np.linspace(0, blk.nodes.numel() - 1, 5).astype(int):
            pos = torch.nonzero(flat == blk.nodes[i]).squeeze(1)
            wm = torch.where(fluid[pos], w[pos] * m.reshape(-1)[pos], 1.0).numpy()
            wr = (w[pos] * rho_f.reshape(-1)[pos]).numpy()
            wb = (w[pos] * b.reshape(-1)[pos] * tau_e / tau_s).numpy()
            A = np.diag(wm) - np.outer(wr, wr) / float(eq.mass_rho[blk.nodes[i]])
            k = blk.k
            X = expm(np.block([[-np.linalg.solve(A, np.diag(wb)), np.linalg.inv(A)],
                               [np.zeros((k, 2 * k))]]) * dt)
            assert np.allclose(E[i].numpy(), X[:k, :k], rtol=1e-10, atol=1e-12)
            assert np.allclose(Psi[i].numpy(), X[:k, k:], rtol=1e-10,
                               atol=1e-12 * abs(X[:k, k:]).max())


def test_the_flux_divergence_is_the_transpose_of_the_pressure_gradient():
    # <D^T W q, p> = sum w |J| q . grad p on a curved mesh, where the cross terms of the
    # inverse Jacobian do not vanish: the pressure equation's flux term and the flux's
    # pressure force are transposes, so the semi-discrete system keeps the Biot energy
    topo = lambda x, y: 40.0 + 30.0 * torch.sin(2 * math.pi * x / 300.0) * torch.cos(
        2 * math.pi * y / 300.0)
    mesh = build_topography_hex27(nex=3, ney=3, nez=3, x_max=300.0, y_max=300.0,
                                  z_bottom=300.0, z_top_fn=topo, n_order=NO, dtype=F64)
    g = torch.Generator().manual_seed(1)
    p = torch.randn(mesh.n_global, dtype=F64, generator=g)
    q = [torch.randn(mesh.ibool.shape, dtype=F64, generator=g) for _ in range(3)]
    div = vector_divergence3(mesh.deriv, mesh.ibool, mesh.inv_J, mesh.w_jac, q, p)
    grads = phys_grads3(mesh.deriv, mesh.ibool, mesh.inv_J, p)
    lhs = float((div * p).sum())
    rhs = float(sum((mesh.w_jac * qi * gi).sum() for qi, gi in zip(q, grads)))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_the_three_axes_are_alike():
    # a pore-pressure injection at the centre of a cube of C-PML on all six faces, in a
    # viscous medium: x, y and z are interchangeable, so are the records at the same offset
    # along each -- every component of the flux, stress and drag paths at once
    shape, dh, dt, nt = (13, 13, 13), 10.0, 5e-4, 120
    prop = _prop(shape, dh, dt, ["p"], ["vx", "vy", "vz", "p"], abcn=8, free_surface=False)
    coef = [torch.full(shape, float(c), dtype=F64)
            for c in biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)]
    rec = np.array([[[9.0, 6.0, 6.0], [6.0, 9.0, 6.0], [6.0, 6.0, 9.0]]])
    with torch.no_grad():
        d = prop(_ricker(nt, dt, 15.0, 0.06), np.array([[6.0, 6.0, 6.0]]), rec,
                 models=coef)[0]
    scale = float(d[..., :3].abs().max())
    for r in range(3):
        assert float((d[:, r, r] - d[:, 0, 0]).abs().max()) < 1e-12 * scale
        assert float((d[:, r, 3] - d[:, 0, 3]).abs().max()) < 1e-12 * float(d[..., 3].abs().max())
        for k in range(3):
            if k != r:
                assert float(d[:, r, k].abs().max()) < 1e-12 * scale


# ------------------------------------------------------------------------------------------ #
# gradient
# ------------------------------------------------------------------------------------------ #
@functools.lru_cache(maxsize=None)
def _gradient_case():
    shape, dh, dt, nt = (9, 9, 9), 10.0, 5e-4, 80
    rock = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                          k_d=9e9, mu=7e9, phi=0.2, kappa=1e-10, tortuosity=2.0)
    zz = torch.arange(shape[0], dtype=F64)[:, None, None].expand(shape)
    coef = [torch.full(shape, float(c), dtype=F64) * (1.0 + 0.02 * zz) for c in rock]
    prop = _prop(shape, dh, dt, ["p"], ["vx", "vz", "p"], abcn=4, free_surface=True,
                 equation_options={"viscous_q": (5.0, 15.0)})
    c_max = prop._pml_speed([prop._interp_to_gll(c) for c in coef])
    prop._pml_speed = lambda models: c_max
    src = np.array([[4.0, 4.0, 3.0]])
    rec = np.array([[[2.0, 3.0, 5.0], [6.0, 5.0, 5.0], [4.0, 4.0, 1.0]]])
    w = _ricker(nt, dt, 15.0, 0.03)
    obs = prop(w, src, rec, models=coef).detach()

    def J(models):
        return 0.5 * (prop(w, src, rec, models=models) - 1.05 * obs).square().sum()

    m0 = [c.clone().requires_grad_() for c in coef]
    return coef, J, torch.autograd.grad(J(m0), m0)


@pytest.mark.parametrize("which", [1, 3, 6, 7])   # mu, M, m, b
def test_gradient_matches_finite_difference(which):
    coef, J, grads = _gradient_case()
    bump = torch.zeros_like(coef[which])
    bump[1:-1, 1:-1, 1:-1] = 1.0                     # off the outermost cells (see 2-D)
    _fd_matches(J, coef, which, grads[which], FD_STEP * coef[which] * bump)


def test_checkpointed_gradient_is_the_full_one():
    # the drag operators are made in init_state, outside the checkpointed chunks
    shape, w = (9, 9, 9), _ricker(80, 5e-4, 15.0, 0.03)
    coef = [torch.full(shape, float(c), dtype=F64)
            for c in biot_from_biot_constants(**DAI1, b=7e4, phi=0.3)]
    src, rec = np.array([[4.0, 4.0, 3.0]]), np.array([[[2.0, 3.0, 5.0], [6.0, 5.0, 5.0]]])
    grads = []
    for memory in (Full(), Ckpt(chunks=4)):
        prop = _prop(shape, 10.0, 5e-4, ["p"], ["vx", "p"], abcn=4, free_surface=True,
                     memory=memory)
        m0 = [c.clone().requires_grad_() for c in coef]
        grads.append(torch.autograd.grad(prop(w, src, rec, models=m0).square().sum(), m0[7])[0])
    assert torch.allclose(grads[1], grads[0], rtol=1e-10, atol=1e-10 * float(grads[0].abs().max()))


# ------------------------------------------------------------------------------------------ #
# sources, receivers, refusals
# ------------------------------------------------------------------------------------------ #
def _small(src=None, rec=None, **kw):
    shape = (9, 9, 9)
    prop = _prop(shape, 10.0, 5e-4, src, rec, **kw)
    return prop, [torch.full(shape, float(c), dtype=F64)
                  for c in biot_from_biot_constants(**DAI1, b=3e3, phi=0.3)]


def test_defaults_are_an_explosion_and_the_solid_velocity():
    prop, _ = _small()
    assert prop.source_type == ["sxx", "syy", "szz"]
    assert prop.receiver_type == ["v_x", "v_y", "v_z"]


def test_two_shots_are_two_runs():
    prop, coef = _small(["p"], ["vx", "p"], abcn=4, free_surface=True)
    w = _ricker(60, 5e-4, 15.0, 0.02)
    src = np.array([[4.0, 4.0, 3.0], [3.0, 5.0, 4.0]])
    rec = np.array([[[2.0, 3.0, 5.0], [6.0, 5.0, 5.0]]] * 2)
    with torch.no_grad():
        both = prop(w, src, rec, models=coef)
        for i in range(2):
            one = prop(w, src[i:i + 1], rec[i:i + 1], models=coef)[0]
            assert torch.allclose(both[i], one, rtol=1e-12, atol=1e-12 * float(one.abs().max()))


def test_the_open_pore_surface_holds_no_pressure():
    prop, coef = _small(["p"], ["p"], free_surface=True)
    rec = np.array([[[2.0, 4.0, 0.0], [6.0, 3.0, 0.0], [4.0, 4.0, 4.0]]])
    with torch.no_grad():
        d = prop(_ricker(60, 5e-4, 15.0, 0.02), np.array([[4.0, 4.0, 3.0]]), rec, models=coef)
    assert not d[0, :, :2].any() and d[0, :, 2].abs().max() > 0


def test_a_fluid_element_is_refused():
    prop, _ = _small(["p"], ["vx"])
    coef = _gll(prop.mesh, [float(c) for c in biot_from_biot_constants(**DAI1, b=3e3, phi=0.3)])
    for k, c in enumerate(biot_fluid(2.25e9, 1000.0)):
        coef[k][0] = float(c)                            # element 0 is water
    with pytest.raises(NotImplementedError, match="fluid-solid interface is 2-D"):
        BiotVSSEM3DDeformed.build(prop.mesh, *coef)


def test_boundary_saving_is_refused():
    with pytest.raises(NotImplementedError, match="boundary saving"):
        _small(["p"], ["vx"], memory=BoundarySaving())

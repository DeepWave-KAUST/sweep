"""ViscoElasticVSSEM2DDeformed: the GSLS visco-elastic velocity-stress SEM.

Its rheology is the FD ViscoElastic's, and test/sem/specfem_compare/run_compare_visco.py
checks it against SPECFEM2D's; here: ``Q = inf`` is the elastic step bit for bit, a finite Q
attenuates, the checkpointed and vmapped rollouts give the full tape's records and
gradients, the gradients in ``vp``, ``Qp`` and ``Qs`` are the derivatives of the records,
and what it cannot do is refused.
"""
import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.sem import ElasticVSSEM2DDeformed, PropSEM, ViscoElasticVSSEM2DDeformed

SHAPE, NT, DT = (21, 41), 120, 1e-3
F64 = torch.float64


def _prop(equation=ViscoElasticVSSEM2DDeformed, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), **kw}
    return PropSEM(equation, shape=SHAPE, dh=10.0, dt=DT, n_order=4, abcn=8, free_surface=True,
                   f_dominant=15.0, dev="cpu", dtype=F64, impl="eager",
                   source_type=["sxx", "szz"], receiver_type=["vx", "vz"], **kw)


def _wavelet():
    t = np.arange(NT) * DT - 0.06
    a = (np.pi * 15.0 * t) ** 2
    return torch.from_numpy((1 - 2 * a) * np.exp(-a))


def _models(q=30.0, grad=False):
    z = torch.linspace(0.0, 1.0, SHAPE[0], dtype=F64)[:, None].expand(SHAPE)
    ms = [2200.0 + 300.0 * z, 1100.0 + 150.0 * z, torch.full(SHAPE, 2000.0, dtype=F64),
          torch.full(SHAPE, q, dtype=F64), torch.full(SHAPE, 0.8 * q, dtype=F64)]
    return [m.clone().requires_grad_(grad) for m in ms]


def _call(prop, models, shots=(0,)):
    src = np.array([[20.0 + 4 * b, 4.0] for b in shots])
    rx = np.arange(4.0, 37.0, 4.0)
    rec = np.stack([rx, np.full_like(rx, 2.0)], 1)[None].repeat(len(shots), 0)
    return prop(_wavelet(), src, rec, models=models)


def test_infinite_q_is_the_elastic_step_bit_for_bit():
    vp, vs, rho = _models()[:3]
    ref = _call(_prop(ElasticVSSEM2DDeformed), [vp, vs, rho])
    p = _prop()
    inf = torch.full((p.mesh.n_elements, p.mesh.ngll, p.mesh.ngll), float("inf"), dtype=F64)
    assert torch.equal(_call(p, [vp, vs, rho, inf, inf]), ref)


def test_a_finite_q_attenuates_more_with_distance():
    vp, vs, rho, qp, qs = _models(q=20.0)
    el = _call(_prop(ElasticVSSEM2DDeformed), [vp, vs, rho])[0]
    vi = _call(_prop(), [vp, vs, rho, qp, qs])[0]
    ratio = (vi[..., 1].norm(dim=0) / el[..., 1].norm(dim=0)).numpy()
    near, far = ratio[len(ratio) // 2 - 1:len(ratio) // 2 + 2].mean(), ratio[[0, -1]].mean()
    assert 0.5 < far < near < 1.0, ratio


@pytest.mark.parametrize("memory, nshots", [(Ckpt(chunks=8), 1), (Full(), 2), (Ckpt(chunks=8), 2)],
                         ids=["ckpt", "vmapped", "vmapped_ckpt"])
def test_other_rollouts_give_the_full_tape(memory, nshots):
    """Ckpt rebuilds the tape chunk by chunk, the memories in its state, and two shots
    run vmapped: the records and gradients of the full tape, shot by shot."""
    def run(mem, shots):
        ms = _models(grad=True)
        out = _call(_prop(memory=mem), ms, shots)
        out.square().sum().backward()
        return out.detach(), [m.grad for m in ms]

    out, grads = run(memory, tuple(range(nshots)))
    refs = [run(Full(), (b,)) for b in range(nshots)]
    ref_out = torch.cat([r[0] for r in refs])
    torch.testing.assert_close(out, ref_out, rtol=1e-11, atol=1e-11 * ref_out.abs().max())
    for i, g in enumerate(grads):
        ref = sum(r[1][i] for r in refs)
        torch.testing.assert_close(g, ref, rtol=1e-9, atol=1e-9 * ref.abs().max())


@pytest.mark.parametrize("which", [0, 3, 4], ids=["vp", "Qp", "Qs"])
def test_gradients_are_the_derivative_of_the_records(which):
    """A directional derivative against central differences, over three steps (the
    truncation error falls as h^2, round-off grows as 1/h: some step must agree)."""
    torch.manual_seed(0)
    ms = _models(grad=True)
    p = _prop()
    loss = lambda models: _call(p, models).square().sum()
    loss(ms).backward()
    g = ms[which].grad
    d = torch.randn(SHAPE, dtype=F64)
    d[:2] = d[-2:] = 0.0                       # off the outermost rows and columns
    d[:, :2] = d[:, -2:] = 0.0
    analytic = float((g * d).sum())
    scale = float(ms[which].detach().abs().mean())
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        plus = [m.detach().clone() for m in ms]
        minus = [m.detach().clone() for m in ms]
        plus[which] += h * scale * d
        minus[which] -= h * scale * d
        with torch.no_grad():
            fd = float(loss(plus) - loss(minus)) / (2 * h * scale)
        errs.append(abs(fd - analytic) / abs(analytic))
    assert min(errs) < 1e-7, errs


def test_what_it_cannot_do_is_refused():
    with pytest.raises(NotImplementedError, match="dissipative"):
        _prop(memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="impl='c'"):
        PropSEM(ViscoElasticVSSEM2DDeformed, shape=SHAPE, dh=10.0, dt=DT, n_order=4,
                dev="cpu", impl="c")
    vp, vs, rho, qp, qs = _models()
    with pytest.raises(ValueError, match="Bound Q from below"):
        _call(_prop(), [vp, vs, rho, torch.full(SHAPE, 1.1, dtype=F64), qs])


def test_specfem2d_conventions():
    # n_fit = 2 n_sls - 1: SPECFEM2D's classical least squares (lfit_zener at 2N - 1
    # log-spaced frequencies with the ends); velocities = "real_modulus": its
    # READ_VELOCITIES_AT_f0, Re M(f_ref) = rho v^2 (shift_velocities_from_f0)
    from sweep.equations.visco_elastic import gsls_relaxation_times
    f_ref, n_sls, q = 15.0, 3, 30.0
    band = (f_ref / 10.0, 10.0 * f_ref)
    full = lambda v: torch.full(SHAPE, v, dtype=F64)
    p = _prop(equation_options={"f_ref": f_ref, "f_band": band, "n_sls": n_sls,
                                "n_fit": 2 * n_sls - 1, "velocities": "real_modulus"})
    eq = p._build_eq([full(3000.0), full(1500.0), full(2400.0), full(q), full(q)])
    point = 1.0 / gsls_relaxation_times(n_sls, *band)
    x = 2.0 * np.pi * band[0] * (band[1] / band[0]) ** np.linspace(0.0, 1.0, 2 * n_sls - 1)
    afunc = x[:, None] * (point - x[:, None] / q) / (point ** 2 + x[:, None] ** 2)
    w = np.linalg.lstsq(afunc, np.full(len(x), 1.0 / q), rcond=None)[0]
    tp = np.array([float(t.flatten()[0]) for t in eq._p_tau]) / float(eq._lam2mu.flatten()[0])
    # _p_tau holds pi_R t_l; pi_U = pi_R (1 + sum t_l)
    t = tp * (1.0 + w.sum())
    assert np.allclose(t, w, rtol=1e-10)
    wt = 2.0 * np.pi * f_ref / point
    re_m = 1.0 + np.sum(w * wt ** 2 / (1.0 + wt ** 2))
    vp_u, vs_u = eq.unrelaxed_velocities()
    assert np.isclose(float(vp_u.flatten()[0]) ** 2, 3000.0 ** 2 * (1.0 + w.sum()) / re_m, rtol=1e-12)
    assert np.isclose(float(vs_u.flatten()[0]) ** 2, 1500.0 ** 2 * (1.0 + w.sum()) / re_m, rtol=1e-12)
    with pytest.raises(ValueError, match="velocities"):
        _prop(equation_options={"velocities": "group"})._build_eq(_models())

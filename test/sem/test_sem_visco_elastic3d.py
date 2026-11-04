"""ViscoElasticVSSEM3D(Deformed): the GSLS visco-elastic velocity-stress SEM in 3-D.

The 2-D class's rheology on the 3-D velocity-stress step: ``Q = inf`` is the elastic step bit
for bit (Hex8 and Hex27), a finite Q attenuates more with distance, a point force matches the
whole-space visco-elastic Green's function (the elastic one, near field included, with the
GSLS complex moduli), the checkpointed and vmapped rollouts give the full tape, the gradients
in ``vp``, ``Qp`` and ``Qs`` are the derivatives of the records, a flat Hex27 mesh gives the
Hex8 records, and what it cannot do is refused.
"""
import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.sem import (ElasticVSSEM3D, ElasticVSSEM3DDeformed, PropSEM, ViscoElasticVSSEM3D,
                       ViscoElasticVSSEM3DDeformed)

SHAPE, NT, DT = (9, 13, 17), 100, 1e-3
F64 = torch.float64


def _prop(equation=ViscoElasticVSSEM3D, shape=SHAPE, dt=DT, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 8,
          "free_surface": True, "dh": 10.0, "f_dominant": 15.0,
          "source_type": ["sxx", "syy", "szz"], "receiver_type": ["vx", "vy", "vz"], **kw}
    return PropSEM(equation, shape=shape, dt=dt, n_order=4, dev="cpu", dtype=F64,
                   impl="eager", **kw)


def _wavelet(nt=NT, dt=DT, f0=15.0, delay=0.06):
    a = (np.pi * f0 * (np.arange(nt) * dt - delay)) ** 2
    return torch.from_numpy((1 - 2 * a) * np.exp(-a))


def _models(q=30.0, grad=False):
    z = torch.linspace(0.0, 1.0, SHAPE[0], dtype=F64)[:, None, None].expand(SHAPE)
    ms = [2200.0 + 300.0 * z, 1100.0 + 150.0 * z, torch.full(SHAPE, 2000.0, dtype=F64),
          torch.full(SHAPE, q, dtype=F64), torch.full(SHAPE, 0.8 * q, dtype=F64)]
    return [m.clone().requires_grad_(grad) for m in ms]


def _call(prop, models, shots=(0,)):
    src = np.array([[8.0 + 2 * b, 6.0, 4.0] for b in shots])
    rx = np.arange(2.0, 15.0, 4.0)
    rec = np.stack([rx, np.full_like(rx, 6.0), np.full_like(rx, 2.0)], 1)[None]
    return prop(_wavelet(), src, rec.repeat(len(shots), 0), models=models)


@pytest.mark.parametrize("visco, elastic", [(ViscoElasticVSSEM3D, ElasticVSSEM3D),
                                            (ViscoElasticVSSEM3DDeformed, ElasticVSSEM3DDeformed)],
                         ids=["hex8", "hex27"])
def test_infinite_q_is_the_elastic_step_bit_for_bit(visco, elastic):
    vp, vs, rho = _models()[:3]
    ref = _call(_prop(elastic, equation_options={"band_split": False}), [vp, vs, rho])
    p = _prop(visco)
    g = p.mesh.ngll
    inf = torch.full((p.mesh.n_elements, g, g, g), float("inf"), dtype=F64)
    assert torch.equal(_call(p, [vp, vs, rho, inf, inf]), ref)


def test_a_finite_q_attenuates_more_with_distance():
    vp, vs, rho, qp, qs = _models(q=20.0)
    el = _call(_prop(ElasticVSSEM3D, equation_options={"band_split": False}), [vp, vs, rho])[0]
    vi = _call(_prop(), [vp, vs, rho, qp, qs])[0]
    ratio = (vi[..., 2].norm(dim=0) / el[..., 2].norm(dim=0)).numpy()
    assert 0.5 < ratio[0] < ratio[1] < 1.0 and 0.5 < ratio[-1] < ratio[-2] < 1.0, ratio


def test_a_flat_hex27_mesh_gives_the_hex8_records():
    ms = _models()
    ref = _call(_prop(), ms)
    out = _call(_prop(ViscoElasticVSSEM3DDeformed), ms)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12 * float(ref.abs().max()))


def _greens_vz(pi_w, mu_w, rho, x, w, dt, nf):
    """vz at ``x`` of a vertical point force ``w(t)`` in a whole space of moduli ``pi(w)``,
    ``mu(w)`` (on the rfft grid of ``nf``): Aki & Richards (4.23), near field included,
    sampled at ``(n + 1/2) dt`` like the velocity-stress records."""
    om = 2 * np.pi * np.fft.rfftfreq(nf, dt)
    r = np.linalg.norm(x)
    gzz = (x[2] / r) ** 2
    al, be = np.sqrt(pi_w / rho), np.sqrt(mu_w / rho)
    ta, tb = r / al, r / be
    with np.errstate(divide="ignore", invalid="ignore"):
        prim = lambda tau: np.exp(-1j * om * tau) * (1 + 1j * om * tau) / om ** 2
        near = prim(tb) - prim(ta)
    near[0] = 0.5 * (tb[0] ** 2 - ta[0] ** 2)
    uz = np.fft.rfft(w, nf) / (4 * np.pi * rho) * (
        (3 * gzz - 1) / r ** 3 * near + gzz / (al ** 2 * r) * np.exp(-1j * om * ta)
        - (gzz - 1) / (be ** 2 * r) * np.exp(-1j * om * tb))
    return np.fft.irfft(1j * om * uz * np.exp(0.5j * om * dt), nf)[:len(w)]


@pytest.mark.slow                  # ~1 min on 4 CPU threads
def test_a_point_force_matches_the_visco_elastic_greens_function():
    """A C-PML on all six sides; the moduli of the built equation. The record's error against
    the visco-elastic solution is the discretisation's, far below the attenuation's effect."""
    shape, dt, nt, f0, (vp, vs, rho, q_p, q_s) = (33, 33, 33), 1e-3, 450, 6.0, (
        2000.0, 1000.0, 2000.0, 25.0, 15.0)
    p = _prop(shape=shape, dt=dt, abcn=16, free_surface=False, f_dominant=f0,
              source_type=["vz"], receiver_type=["vz"])
    ms = [torch.full(shape, v, dtype=F64) for v in (vp, vs, rho, q_p, q_s)]
    w = _wavelet(nt, dt, f0, 0.2)
    src = np.array([[16.0, 16.0, 16.0]])
    offs = np.array([(9.0, 0.0, 0.0), (0.0, 0.0, 9.0), (5.0, 5.0, 5.0), (-4.0, 7.0, -6.0)])
    with torch.no_grad():
        out = p(w, src, (src + offs)[None], models=ms)[0, :, :, 0].numpy()
    eq = p._build_eq(ms)
    pi_U, mu_U = (float((eq.rho_e * v ** 2).flatten()[0]) for v in (eq.vp_e, eq.vs_e))
    tp = [float(c.flatten()[0]) for c in eq._p_tau]          # pi_R t_p,l
    tm = [float(c.flatten()[0]) for c in eq._m_tau]          # mu_R t_s,l
    nf = 8 * nt
    om = 2 * np.pi * np.fft.rfftfreq(nf, dt)
    modulus = lambda M_U, Mt: (M_U - sum(Mt)) + sum(
        c * 1j * om * s / (1 + 1j * om * s) for c, s in zip(Mt, eq._tau_sigma))
    pi_w, mu_w = modulus(pi_U, tp), modulus(mu_U, tm)
    el = lambda v: np.full(om.shape, rho * v * v, dtype=complex)
    for k, d in enumerate(offs):
        x = 10.0 * d                                         # dh = 10 m
        ana = _greens_vz(pi_w, mu_w, rho, x, w.numpy(), dt, nf)
        effect = np.linalg.norm(ana - _greens_vz(el(vp), el(vs), rho, x, w.numpy(), dt, nf))
        err = np.linalg.norm(out[:, k] - ana)
        assert err < 1e-2 * np.linalg.norm(ana) and err < 0.1 * effect, (k, err, effect)


@pytest.mark.parametrize("memory, nshots", [(Ckpt(chunks=8), 1), (Full(), 2)],
                         ids=["ckpt", "vmapped"])
def test_other_rollouts_give_the_full_tape(memory, nshots):
    """Ckpt rebuilds the tape chunk by chunk, the memories in its state, and two shots run
    vmapped: the records and gradients of the full tape, shot by shot."""
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
    for ax in range(3):                        # off the outermost layers
        d.narrow(ax, 0, 2).zero_()
        d.narrow(ax, SHAPE[ax] - 2, 2).zero_()
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


@pytest.mark.parametrize("equation", [ViscoElasticVSSEM3D, ViscoElasticVSSEM3DDeformed])
def test_what_it_cannot_do_is_refused(equation):
    with pytest.raises(NotImplementedError, match="dissipative"):
        _prop(equation, memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="impl='c'"):
        PropSEM(equation, shape=SHAPE, dh=10.0, dt=DT, n_order=4, dev="cpu", impl="c")
    vp, vs, rho, qp, qs = _models()
    with pytest.raises(ValueError, match="Bound Q from below"):
        _call(_prop(equation), [vp, vs, rho, torch.full(SHAPE, 1.1, dtype=F64), qs])

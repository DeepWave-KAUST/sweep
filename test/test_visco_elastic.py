"""ViscoElastic (GSLS, eager): elastic limit, constant-Q fit, analytic
attenuation / dispersion, and model gradients (vp, vs, rho, Qp, Qs).

The analytic check compares the two-receiver spectral ratio of a visco-elastic
run, normalised by the same ratio from an ``Elastic`` run (which cancels the
source spectrum and most of the grid dispersion), against the exact 2-D
homogeneous Green's function evaluated with the GSLS complex velocities.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ViscoElastic
from sweep.propagator.options import BoundarySaving
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _prop(eq, shape, **kw):
    kw.setdefault("abcn", 20)
    kw.setdefault("dh", 10.0)
    kw.setdefault("dt", 1e-3)
    if "memory" not in kw:
        kw.setdefault("use_ckpt", False)
    kw.setdefault("use_compile", False)
    return PropTorch(eq, shape=shape, impl="eager", **kw)


def _wavelet(nt, dt, f, delay):
    t = np.arange(nt) * dt - delay
    return torch.tensor((1e3 * ricker(t, f=f)).astype(np.float32), device=DEVICE)


# --------------------------------------------------------------- elastic limit
@pytest.mark.parametrize("free_surface", [False, True])
def test_q_inf_reduces_to_elastic(free_surface):
    """Qp = Qs = inf: memory variables stay zero and the unrelaxed moduli equal
    the elastic ones.  Bit-exact without a free surface; with one, only the
    algebraic form of the surface sigma_xx coefficient differs (rounding)."""
    NZ, NX = 48, 56
    wav = _wavelet(300, 1e-3, 12.0, 0.08)
    src = np.array([[NX // 2, 6]], np.int64)
    rx = np.arange(4, NX - 4, 3)
    rec = np.stack([rx, np.full_like(rx, 2)], -1)[None]
    zz = torch.linspace(0, 1, NZ, device=DEVICE)[:, None] * torch.ones(1, NX, device=DEVICE)
    vp, vs, rho = 2000 + 800 * zz, 1100 + 500 * zz, 1900 + 300 * zz
    inf = torch.full_like(vp, float("inf"))
    de = _prop(Elastic(4, DEVICE), (NZ, NX), free_surface=free_surface)(
        wav, src.copy(), rec.copy(), models=[vp, vs, rho])
    dv = _prop(ViscoElastic(4, DEVICE, f_ref=12.0), (NZ, NX), free_surface=free_surface)(
        wav, src.copy(), rec.copy(), models=[vp, vs, rho, inf, inf])
    if free_surface:
        assert float((de - dv).norm() / de.norm()) < 1e-5
    else:
        assert torch.equal(de, dv)


# ------------------------------------------------------------ constant-Q fit
def test_q_is_nearly_constant_over_band():
    """Per-mechanism least-squares fit: uniform accuracy from low to high Q."""
    eq = ViscoElastic(f_ref=15.0)
    fmin, fmax = eq.f_band
    f = np.geomspace(fmin, fmax, 50)
    for Q in (5.0, 10.0, 30.0, 100.0, 1000.0):
        assert np.all(eq.fit.tau_np(Q) > 0)
        q = eq.fit.q_of_frequency(Q, f)
        assert np.all(np.abs(q / Q - 1) < 0.07), (Q, q)


def test_tau_torch_matches_numpy_and_vanishes_at_inf():
    eq = ViscoElastic(f_ref=15.0)
    Q = torch.tensor([7.0, 30.0, 200.0, float("inf")], dtype=torch.float64)
    t = torch.stack(eq.fit.tau(Q), -1)
    for i, q in enumerate(Q[:3].tolist()):
        np.testing.assert_allclose(t[i].numpy(), eq.fit.tau_np(q), rtol=1e-8)
    assert torch.equal(t[3], torch.zeros(3, dtype=torch.float64))


def test_negative_relaxation_strength_refused():
    with pytest.raises(ValueError):
        ViscoElastic(f_band=(2.0, 30.0), n_sls=5)
    eq = ViscoElastic(f_ref=12.0)
    assert 1.0 < eq.q_min < 2.0
    ms = [torch.full((8, 8), v, dtype=torch.float32) for v in (2000.0, 1100.0, 2000.0, 30.0, 1.1)]
    with pytest.raises(ValueError):
        eq.prepare_models(ms)


def test_unrelaxed_velocities():
    """The CFL velocities: exactly vp / vs at Q = inf, faster for finite Q."""
    eq = ViscoElastic(f_ref=10.0)
    one = lambda v: torch.full((3, 4), float(v))
    vp_u, vs_u = eq.unrelaxed_velocities([one(2500), one(1400), one(2000), one("inf"), one("inf")])
    assert torch.allclose(vp_u, one(2500)) and torch.allclose(vs_u, one(1400))
    vp_u, vs_u = eq.unrelaxed_velocities([one(2500), one(1400), one(2000), one(30), one(20)])
    assert bool((vp_u > 2500).all()) and bool((vs_u > 1400).all())


def test_n_sls_configurable():
    eq = ViscoElastic(n_sls=4)
    assert len(eq.tau_sigma) == 4
    assert len(eq.wavefields) == 15 + 3 * 4


@pytest.mark.parametrize("n_sls", [1, 3, 4])
def test_slot_table_matches_cuda_layout(n_sls):
    """The declared bind order reproduces the hand-written counts and slab
    axes, and the memory variables sit where the eager field list has them."""
    eq = ViscoElastic(n_sls=n_sls)
    spec, t = eq.cuda_layout, eq.cuda_layout.slots
    assert (t.base_nvar, t.pml_nvar) == (spec.base_nvar, spec.pml_nvar) == (5 + 3 * n_sls, 10)
    assert t.pml_slot_axes == spec.pml_slot_axes
    assert spec.checkpoint_slot_axes == (None,) * spec.base_nvar + spec.pml_slot_axes
    assert [s.name for s in t.slots] == [n.replace("m_t", "m_s") for n in eq.wavefields]
    assert t.vel_idx == (0, 1) and t.pairs(adjoint=True) == () and t.u_blocks == ()
    assert not spec.stepped      # domain decomposition is refused by declaration


# ------------------------------------------ analytic attenuation + dispersion
def _hankel2(n, z):
    from scipy.special import hankel2
    return hankel2(n, z)


@pytest.mark.skipif(DEVICE != "cuda", reason="large homogeneous runs; GPU only")
@pytest.mark.parametrize("wave", ["P", "S"])
def test_spectral_ratio_matches_analytic(wave):
    pytest.importorskip("scipy")
    VP, VS, RHO, QP, QS, FREF = 2500.0, 1400.0, 2000.0, 40.0, 25.0, 15.0
    NZ, NX, DH, DT, NT = 160, 300, 5.0, 5e-4, 2000
    wav = _wavelet(NT, DT, FREF, 0.08)
    sx, sz, r1, r2 = 30, NZ // 2, 60, 180
    src = np.array([[sx, sz]], np.int64)
    rec = np.array([[[sx + r1, sz], [sx + r2, sz]]], np.int64)
    # explosion -> radial vx (pure P);  vertical force -> vz on the x axis (S + P near field)
    st, rt = (["sxx", "szz"], ["vx"]) if wave == "P" else (["vz"], ["vz"])
    full = lambda v: torch.full((NZ, NX), v, device=DEVICE)

    def run(eq, models):
        p = _prop(eq, (NZ, NX), abcn=40, dh=DH, dt=DT, source_type=st, receiver_type=rt)
        with torch.no_grad():
            return p(wav, src.copy(), rec.copy(), models=models)[0, :, :, 0].double().cpu().numpy()

    eqv = ViscoElastic(8, DEVICE, f_ref=FREF)
    de = run(Elastic(8, DEVICE), [full(VP), full(VS), full(RHO)])
    dv = run(eqv, [full(VP), full(VS), full(RHO), full(QP), full(QS)])
    nf = 8192
    fr = np.fft.rfftfreq(nf, DT)
    E, V = np.fft.rfft(de, nf, axis=0), np.fft.rfft(dv, nf, axis=0)
    meas = (V[:, 1] / V[:, 0]) / (E[:, 1] / E[:, 0])

    wr = 2 * np.pi * FREF

    def m(w, q):
        return eqv.fit.modulus_ratio(eqv.fit.tau_np(q), np.atleast_1d(w) / (2 * np.pi))

    def cvel(v, q, w):   # phase velocity = v at f_ref (e^{+iwt} convention)
        MR = RHO * v ** 2 * np.real(m(wr, q) ** -0.5)[0] ** 2
        return np.sqrt(MR * m(w, q) / RHO)

    def green(a, b, w, r):
        if wave == "P":
            return _hankel2(1, w / a * r)
        return (_hankel2(0, w / b * r) / b ** 2
                - (_hankel2(1, w / b * r) / b - _hankel2(1, w / a * r) / a) / (w * r))

    sel = (fr > 5.0) & (fr < 30.0)
    w = 2 * np.pi * fr[sel]
    ca, cb = cvel(VP, QP, w), cvel(VS, QS, w)
    th = (green(ca, cb, w, r2 * DH) / green(ca, cb, w, r1 * DH)) / \
         (green(VP, VS, w, r2 * DH) / green(VP, VS, w, r1 * DH))
    ms = meas[sel]
    # Measured (RTX 6000 Ada): P 2e-4 / 2e-4, S 1.7e-3 / 8e-4 (amp / phase).
    # A 5% shift of the stepping tau_sigma alone gives phase 4e-3 (P) / 1e-2 (S).
    assert np.max(np.abs(np.abs(ms) / np.abs(th) - 1)) < 4e-3
    assert np.max(np.abs(np.angle(ms / th))) < 2e-3


# --------------------------------------------------------------- gradients
def _grad_setup(free_surface):
    NZ, NX = 48, 56
    wav = _wavelet(400, 1e-3, 12.0, 0.08)
    src = np.array([[NX // 2, 6]], np.int64)
    rx = np.arange(4, NX - 4, 3)
    rec = np.stack([rx, np.full_like(rx, 2)], -1)[None]
    zz = torch.linspace(0, 1, NZ, device=DEVICE)[:, None] * torch.ones(1, NX, device=DEVICE)
    m0 = [2000 + 800 * zz, 1100 + 500 * zz, 1900 + 300 * zz, 20 + 30 * zz, 12 + 20 * zz]

    def prop(ckpt=False):
        return _prop(ViscoElastic(4, DEVICE, f_ref=12.0), (NZ, NX),
                     free_surface=free_surface, use_ckpt=ckpt)

    return NZ, NX, wav, src, rec, m0, prop


@pytest.mark.parametrize("free_surface", [False, True])
def test_gradient_matches_finite_difference(free_surface):
    """Directional derivative of <d, w> vs central FD.  Perturbations stay in
    the interior (the PML profile depends on the model non-differentiably)."""
    NZ, NX, wav, src, rec, m0, prop = _grad_setup(free_surface)
    g = torch.Generator().manual_seed(0)
    d0 = prop()(wav, src.copy(), rec.copy(), models=m0)
    wr = torch.randn(d0.shape, generator=g).to(DEVICE)
    J = lambda ms: (prop()(wav, src.copy(), rec.copy(), models=ms) * wr).sum()
    ms = [m.clone().requires_grad_(True) for m in m0]
    J(ms).backward()
    mask = torch.zeros(NZ, NX)
    mask[8:-8, 8:-8] = 1
    k = torch.ones(1, 1, 7, 7) / 49
    for i, scale in enumerate([50.0, 30.0, 40.0, 3.0, 2.0]):
        dm = torch.nn.functional.conv2d(torch.randn(1, 1, NZ, NX, generator=g), k, padding=3)[0, 0] * mask
        dm = (scale * dm / dm.abs().max()).to(DEVICE)
        eps = 1.0
        with torch.no_grad():
            mp = [m.clone() for m in m0]
            mm = [m.clone() for m in m0]
            mp[i] += eps * dm
            mm[i] -= eps * dm
            fd = float(J(mp) - J(mm)) / (2 * eps)
        ad = float((ms[i].grad * dm).sum())
        assert abs(fd - ad) < 2e-2 * abs(fd), (i, fd, ad)


def test_checkpoint_gradient_matches_full():
    NZ, NX, wav, src, rec, m0, prop = _grad_setup(True)
    grads = []
    for ckpt in (False, True):
        ms = [m.clone().requires_grad_(True) for m in m0]
        (prop(ckpt)(wav, src.copy(), rec.copy(), models=ms) ** 2).sum().backward()
        grads.append([m.grad for m in ms])
    for a, b in zip(*grads):
        assert float((a - b).norm() / a.norm()) < 1e-4


# ------------------------------------------------------------------ guards
def test_boundary_saving_refused():
    NZ, NX = 40, 48
    p = _prop(ViscoElastic(4, DEVICE), (NZ, NX), abcn=10, memory=BoundarySaving())
    ms = [torch.full((NZ, NX), v, device=DEVICE, requires_grad=True)
          for v in (2000.0, 1100.0, 2000.0, 30.0, 20.0)]
    with pytest.raises(NotImplementedError):
        p(torch.randn(50, device=DEVICE), np.array([[20, 5]]),
          np.array([[[10, 2], [30, 2]]]), models=ms).sum().backward()

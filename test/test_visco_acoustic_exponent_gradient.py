"""The frozen NCQ exponent's share of the Q gradient, on every backend.

ViscoAcoustic freezes the fractional exponent of its spectral terms at the
model average ``gbar = mean(arctan(1/Q) / pi)`` -- frozen in space, the paper's
freezing-unfreezing treatment.  ``gbar`` is still a function of Q, and every
backend dropped that path: eager and jax detached the exponent, impl='c' took
the fractional grids as data.  The Q gradient then missed
``dJ/dgbar * dgbar/dQ``: 40% against finite differences along a broad Q
perturbation (6x for a uniform one), while vp and omega were right.

Now eager and jax keep the exponent on the graph, and impl='c' carries it on a
fifth prepared model G (``gbar`` broadcast over the model): the backward
correlates the adjoint with the forward field through the exponent-
differentiated grids ``2 ln|k| D`` and autograd sums that map back to ``gbar``.

The check is the directional derivative along a broad interior Q bump (clear
of the outer ring, whose gradient EdgePadding crops) against a central
difference of the forward-only loss, in float32.  The reference frequency sits
above the source band (60 Hz for a 15 Hz Ricker): at the source frequency the
coefficient term (ln(vp/omega)) and the exponent term (ln|k|) nearly cancel,
which leaves a true derivative 200x smaller than either and asks the finite
difference for more than float32 has.  Here the old gradient had the wrong
sign (+2.3e-3 against -1.56e-3, eager); now all three agree with the
difference to ~2e-3.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.equations import ViscoAcoustic  # noqa: E402

NZ, NX, DH, DT, NT, ABCN, SO, F0 = 40, 48, 10.0, 1e-3, 300, 12, 4, 15.0
F_REF = 60.0      # the NCQ reference frequency, above the source band
H = 0.25          # finite-difference step along the unit-amplitude bump
REL_MAX = 1e-2    # before the fix: 2.5 (the wrong sign)


def _wavelet():
    t = np.arange(NT) * DT - 1.0 / F0
    a = (np.pi * F0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def _geometry():
    src = np.array([[NX // 2, 4]], np.int64)                       # (x, z)
    rx = np.arange(4, NX - 4, 2, dtype=np.int64)
    return src, np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]


def _models():
    """vp and Q both vary; Q low enough that attenuation moves the misfit."""
    ramp = np.linspace(0.0, 1.0, NZ, dtype=np.float64)[:, None] * np.ones((1, NX))
    vp = (2000.0 + 400.0 * ramp).astype(np.float32)
    q = (15.0 + 10.0 * ramp).astype(np.float32)
    om = np.full((NZ, NX), 2 * np.pi * F_REF, np.float32)
    box = np.zeros((NZ, NX))
    box[NZ // 3: 2 * NZ // 3, NX // 4: 3 * NX // 4] = 1.0
    q_true = (q + 4.0 * box).astype(np.float32)
    return vp, q, om, q_true


def _direction():
    """A broad positive bump: large mean, so gbar moves with it."""
    z, x = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    g = np.exp(-0.5 * (((z - NZ / 2) / (NZ / 5)) ** 2 + ((x - NX / 2) / (NX / 5)) ** 2))
    g[:3] = g[-3:] = 0.0
    g[:, :3] = g[:, -3:] = 0.0
    return g / g.max()


def _torch_loss(impl, device):
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    dev = torch.device(device)
    eq = ViscoAcoustic(spatial_order=SO, device=dev, backend="torch")
    kw = dict(shape=(NZ, NX), dh=DH, dt=DT, nt=NT, abcn=ABCN, dev=dev,
              source_type=["h1"], receiver_type=["h1"])
    if impl == "eager":
        prop = PropTorch(eq, impl="eager", use_ckpt=False,
                         eager_options=EagerOptions(use_compile=False), **kw)
    else:
        prop = PropTorch(eq, impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **kw)
    vp, q, om, q_true = _models()
    src, rec = _geometry()
    w = torch.tensor(_wavelet(), device=dev)
    t = lambda a, g=False: torch.tensor(a, device=dev, requires_grad=g)  # noqa: E731
    with torch.no_grad():
        obs = prop(w, src, rec, models=[t(vp), t(q_true), t(om)]).double()

    def loss(qq, grad=False):
        Q = t(qq.astype(np.float32), grad)
        J = 0.5 * (prop(w, src, rec, models=[t(vp), Q, t(om)]).double() - obs).pow(2).sum()
        if not grad:
            return float(J)
        J.backward()
        return float(J), Q.grad.double().cpu().numpy()

    return loss


def _jax_loss():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from sweep.propagator.jax import PropJax

    eq = ViscoAcoustic(SO, "cpu", "jax")
    prop = PropJax(eq, (NZ, NX), dh=DH, dt=DT, abcn=ABCN, nt=NT, use_ckpt=False)
    vp, q, om, q_true = _models()
    src, rec = _geometry()
    w = _wavelet()[None]
    srcj = src[None]

    def fwd(Q):
        o = prop(w, srcj, rec, models=[jnp.asarray(vp), Q, jnp.asarray(om)])
        return o[0] if isinstance(o, (tuple, list)) else o

    obs = np.asarray(fwd(jnp.asarray(q_true)), np.float64)

    def J(Q):
        return 0.5 * jnp.sum((fwd(Q) - obs) ** 2)

    def loss(qq, grad=False):
        Q = jnp.asarray(qq.astype(np.float32))
        if not grad:
            return float(J(Q))
        val, g = jax.value_and_grad(J)(Q)
        return float(val), np.asarray(g, np.float64)

    return loss


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


def _check(loss):
    _, q, _, _ = _models()
    d = _direction()
    _, g = loss(q.astype(np.float64), grad=True)
    adj = float(np.sum(g * d))
    fd = (loss(q + H * d) - loss(q - H * d)) / (2 * H)
    rel = abs(adj - fd) / abs(fd)
    assert rel < REL_MAX, f"Q directional derivative: adjoint {adj:+.6e} vs FD {fd:+.6e} (rel {rel:.2e})"


def test_eager_q_gradient_matches_finite_differences():
    _check(_torch_loss("eager", "cpu"))


def test_jax_q_gradient_matches_finite_differences():
    _check(_jax_loss())


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_c_q_gradient_matches_finite_differences():
    _check(_torch_loss("c", "cuda"))


# ---------------------------------------------------------------------------
# impl='c' plumbing, no device needed: G and the derivative grids appear
# exactly when the exponent needs a gradient.
# ---------------------------------------------------------------------------
def _prepared(requires_grad, phase=True, amp=True):
    eq = ViscoAcoustic(spatial_order=SO, device="cpu", backend="torch",
                       phase_shift=phase, amplitude_damping=amp)
    vp, q, om, _ = _models()
    pad = lambda a, g=False: torch.tensor(a[None, None], requires_grad=g)  # noqa: E731
    return eq, eq.prepare_models_c([pad(vp), pad(q, requires_grad), pad(om)])


def test_prepare_models_c_adds_g_only_when_q_needs_a_gradient():
    eq, prepared = _prepared(True)
    assert len(prepared) == 5
    G = prepared[4]
    assert G.shape == prepared[0].shape and G.requires_grad
    assert torch.equal(G, eq._gbar.expand(G.shape))
    assert len(_prepared(False)[1]) == 4
    with torch.no_grad():
        assert len(_prepared(True)[1]) == 4
    # neither spectral term: gbar does not enter the step
    assert len(_prepared(True, phase=False, amp=False)[1]) == 4


@pytest.mark.parametrize("phase,amp,n_base", [(True, True, 3), (True, False, 2), (False, True, 1)])
def test_c_eq_aux_appends_the_exponent_derivatives(phase, amp, n_base):
    from sweep.equations.base import init_wavenumbers

    eq, prepared = _prepared(True, phase, amp)
    shape = (NZ + 2 * (ABCN + SO // 2), NX + 2 * (ABCN + SO // 2))
    prop = SimpleNamespace(shape_cuda=shape, _grid_spacing=(DH, DH), dev=torch.device("cpu"))
    grids = eq.c_eq_aux(prop)
    n_prime = int(amp) + int(phase)
    assert len(grids) == n_base + n_prime
    k, _, _ = init_wavenumbers(shape, (DH, DH))
    pos = k > 0
    gbar = float(eq._gbar)
    expect = []
    if amp:
        expect.append(np.where(pos, 2 * np.log(np.where(pos, k, 1.0)) * np.power(k, 2 * gbar + 1, where=pos), 0.0))
    if phase:
        expect.append(np.where(pos, 2 * np.log(np.where(pos, k, 1.0)) * np.power(k, 2 * gbar + 2, where=pos), 0.0))
    for got, want in zip(grids[n_base:], expect):
        np.testing.assert_allclose(got.numpy(), want.astype(np.float32), rtol=1e-6, atol=1e-12)
    # and without the gradient, the composition the kernels always read
    eq._c_exp_grad = False
    assert len(eq.c_eq_aux(prop)) == n_base

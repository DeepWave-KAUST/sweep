"""Second derivatives: which paths have them, and a clear error from the ones that do not.

The uncompiled eager step is plain autograd, so ``create_graph=True`` gives a
gradient that can be differentiated again (a Hessian-vector product). The
compiled adjoint (impl='c'), eager boundary saving and the torch.compile'd
step compute their backward off the tape. They used to hand back a gradient
without a graph -- "element 0 of tensors does not require grad" later, or,
with a regulariser on the tape, the regulariser's Hessian alone -- and now
refuse create_graph=True with an error that names what works. So do the qP
equations, whose second derivative was off the tape (s_gradient='adjoint': the
qP Born pair returned a Hessian product 100 % off without a word), NaN
('exact') or not a Hessian ('frozen').
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding, requires_compile
from sweep.equations import Acoustic, AcousticTTI, AcousticVTI, AcousticVTIBorn
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator._torch_eager import _PropTorchEager
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
NT = 120
SRC = np.array([[8, 2], [19, 2]])
REC = np.repeat(np.array([[[ix, 2] for ix in range(2, 26, 2)]]), 2, axis=0)


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.04
    a = (np.pi * 25.0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))


def _models():
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    v_true = 1800.0 + 10.0 * z
    v_true[10:14, 10:18] += 150.0
    return v_true, 1800.0 + 10.0 * z


def _directions():
    g = torch.Generator().manual_seed(0)
    ring = torch.zeros(SHAPE)
    ring[2:-2, 2:-2] = 1.0     # off the outer cells, whose gradient is cropped (EdgePadding)
    return [torch.randn(SHAPE, generator=g) * ring for _ in range(2)]


def _prop(memory, impl="eager", dev="cpu", equation=None, **eager):
    kw = dict(shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, abcn=8, dev=dev, impl=impl,
              memory=memory)
    if impl == "eager":
        kw["eager_options"] = EagerOptions(**{"use_compile": False, **eager})
    return PropTorch(equation or Acoustic(spatial_order=4, device=dev), **kw)


def _misfit(prop, v, d_obs):
    rec = prop(_wavelet().to(v.device), SRC, REC, models=[v])
    return 0.5 * ((rec - d_obs) ** 2).sum()


def _observed(prop, v_true):
    with torch.no_grad():
        return prop(_wavelet().to(v_true.device), SRC, REC, models=[v_true])


@pytest.mark.parametrize("memory", [Full, Ckpt], ids=["full", "ckpt"])
def test_uncompiled_eager_hessian_vector_product_matches_difference_of_gradients(memory):
    prop = _prop(memory())
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)
    h1, h2 = _directions()

    v = v0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_misfit(prop, v, d_obs), v, create_graph=True)
    (hv1,) = torch.autograd.grad((g * h1).sum(), v, retain_graph=True)
    (hv2,) = torch.autograd.grad((g * h2).sum(), v, retain_graph=True)   # the graph is reused

    def grad_at(m):
        m = m.clone().requires_grad_(True)
        return torch.autograd.grad(_misfit(prop, m, d_obs), m)[0]

    eps = 1.0
    fd = (grad_at(v0 + eps * h1) - grad_at(v0 - eps * h1)) / (2 * eps)
    assert float((hv1 - fd).norm() / fd.norm()) < 1e-3
    a, b = float((h2 * hv1).sum()), float((h1 * hv2).sum())             # H is symmetric
    assert abs(a - b) <= 1e-4 * abs(a)


def test_eager_boundary_saving_refuses_create_graph():
    prop = _prop(BoundarySaving())
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)
    v = v0.clone().requires_grad_(True)
    # with a regulariser on the tape the gradient used to come back WITH a graph,
    # whose Hessian was the regulariser's alone
    loss = _misfit(prop, v, d_obs) + 1e-6 * (v ** 2).sum()
    with pytest.raises(RuntimeError, match="BoundarySaving.*not twice-differentiable"):
        torch.autograd.grad(loss, v, create_graph=True)


def test_eager_boundary_saving_first_derivative_unchanged():
    prop = _prop(BoundarySaving())
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)
    v = v0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_misfit(prop, v, d_obs), v)
    assert g.abs().max() > 0


def test_registered_gradient_imaging_condition_refuses_create_graph():
    prop = _prop(Full())
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd["h1"] * adj["h1"], mode="gradient")
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)
    v = v0.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="imaging condition.*not twice-differentiable"):
        torch.autograd.grad(_misfit(prop, v, d_obs), v, create_graph=True)


@requires_compile
def test_compiled_eager_refuses_create_graph_and_names_use_compile():
    prop = _prop(Full(), use_compile=True)
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)
    v = v0.clone().requires_grad_(True)
    loss = _misfit(prop, v, d_obs)
    if not prop.use_compile:
        pytest.skip("torch.compile cannot build the step here")
    with pytest.raises(RuntimeError, match="use_compile=False"):
        torch.autograd.grad(loss, v, create_graph=True)


@requires_compile
def test_compiled_eager_first_derivative_bit_exact_with_the_guard(monkeypatch):
    prop = _prop(Full(), use_compile=True)
    v_true, v0 = _models()
    d_obs = _observed(prop, v_true)

    def grad():
        v = v0.clone().requires_grad_(True)
        return torch.autograd.grad(_misfit(prop, v, d_obs), v)[0]

    guarded = grad()
    monkeypatch.setattr(_PropTorchEager, "_guard_create_graph",
                        lambda self, models: models)
    assert torch.equal(guarded, grad())


@requires_binding()
def test_impl_c_refuses_create_graph():
    prop = _prop(Full(), impl="c", dev="cuda")
    v_true, v0 = (m.cuda() for m in _models())
    d_obs = _observed(prop, v_true)
    v = v0.clone().requires_grad_(True)
    loss = _misfit(prop, v, d_obs) + 1e-6 * (v ** 2).sum()
    with pytest.raises(RuntimeError, match="impl='c'.*not twice-differentiable"):
        torch.autograd.grad(loss, v, create_graph=True)
    # and a plain gradient still runs
    v = v0.clone().requires_grad_(True)
    assert torch.autograd.grad(_misfit(prop, v, d_obs), v)[0].abs().max() > 0


QP = [AcousticVTI, AcousticTTI, AcousticVTIBorn]


def _qp(equation, s_gradient, requires_grad=True):
    eq = equation(spatial_order=4, device="cpu", s_gradient=s_gradient)
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    base = {"vp": 2000.0 + 8.0 * z, "epsilon": 0.1 + 0 * z, "delta": 0.05 + 0 * z,
            "theta": 0.3 + 0 * z, "mp": 0.02 + 0 * z}
    return _prop(Full(), equation=eq), [base[n].clone().requires_grad_(requires_grad) for n in eq.models]


@pytest.mark.parametrize("s_gradient", ["adjoint", "exact", "frozen"])
@pytest.mark.parametrize("equation", QP, ids=["vti", "tti", "vti_born"])
def test_qp_equations_refuse_a_second_derivative(equation, s_gradient):
    prop, models = _qp(equation, s_gradient)
    loss = 0.5 * (prop(_wavelet(), SRC, REC, models=models) ** 2).sum()
    if s_gradient == "adjoint":       # the gradient is off the tape: refused outright
        with pytest.raises(RuntimeError, match=r"(?s)Acoustic[VT]TI\w* gives first derivatives only.*not twice"):
            torch.autograd.grad(loss, models, create_graph=True)
    else:                             # a first-order graph, whose derivative is not a Hessian
        grads = torch.autograd.grad(loss, models, create_graph=True)
        with pytest.raises(RuntimeError, match="second derivative with respect to the models"):
            torch.autograd.grad(sum(g.sum() for g in grads), models)
    # first derivatives are untouched
    grads = torch.autograd.grad(0.5 * (prop(_wavelet(), SRC, REC, models=models) ** 2).sum(), models)
    assert all(torch.isfinite(g).all() for g in grads) and grads[0].abs().max() > 0


@pytest.mark.parametrize("s_gradient", ["exact", "frozen"])
def test_qp_double_vjp_jacobian_product_still_runs(s_gradient):
    """A Gauss-Newton solver takes J h by differentiating J^T u with respect
    to u, which never goes back through the models: it ran at the base, and
    still does."""
    prop, models = _qp(AcousticVTI, s_gradient)
    d = prop(_wavelet(), SRC, REC, models=models)
    u = torch.zeros_like(d, requires_grad=True)
    gs = torch.autograd.grad(d, models, grad_outputs=u, create_graph=True)
    h = [torch.ones_like(m) for m in models]
    (jh,) = torch.autograd.grad(sum((g * v).sum() for g, v in zip(gs, h)), u)
    assert torch.isfinite(jh).all() and jh.abs().max() > 0


@pytest.mark.parametrize("s_gradient", ["adjoint", "exact"])
def test_qp_wavelet_second_derivative_is_refused(s_gradient):
    prop, models = _qp(AcousticVTI, s_gradient, requires_grad=False)
    w = _wavelet().requires_grad_(True)
    loss = 0.5 * (prop(w, SRC, REC, models=models) ** 2).sum()
    if s_gradient == "adjoint":
        with pytest.raises(RuntimeError, match="not twice-differentiable"):
            torch.autograd.grad(loss, w, create_graph=True)
    else:
        (g,) = torch.autograd.grad(loss, w, create_graph=True)
        with pytest.raises(RuntimeError, match="second derivative"):
            torch.autograd.grad(g.sum(), w)


def test_refusals_leave_the_interpreter_exiting_normally():
    """A refusal raised in a node at the start of the backward left the rest
    of the engine's work queued, and the interpreter aborted at exit (SIGABRT)
    even when the script had caught the error.  They are raised where a
    backward ends."""
    import pathlib
    import subprocess
    import sys
    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    script = f"""
import sys, warnings
sys.path.insert(0, {src!r})
warnings.simplefilter("ignore")
import numpy as np, torch
from sweep.equations import Acoustic, AcousticVTI
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
t = np.arange(60, dtype=np.float32) * 1e-3 - 0.04
w = torch.tensor(((1 - 2 * (np.pi * 25 * t) ** 2) * np.exp(-(np.pi * 25 * t) ** 2)).astype(np.float32))
src, rec = np.array([[8, 3]]), np.array([[[ix, 2] for ix in range(2, 26, 2)]])
z = torch.arange(24, dtype=torch.float32)[:, None] * torch.ones(24, 28)
for eq, memory, models in [(Acoustic(spatial_order=4), BoundarySaving(), [2000 + 8 * z]),
                           (AcousticVTI(spatial_order=4), Full(), [2000 + 8 * z, 0.1 + 0 * z, 0.05 + 0 * z])]:
    prop = PropTorch(eq, shape=(24, 28), dh=10.0, dt=1e-3, nt=60, abcn=8, dev="cpu", impl="eager",
                     memory=memory, eager_options=EagerOptions(use_compile=False))
    ms = [m.clone().requires_grad_(True) for m in models]
    try:
        torch.autograd.grad((prop(w, src, rec, models=ms) ** 2).sum(), ms, create_graph=True)
        print("not refused")
    except RuntimeError:
        print("refused")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["refused", "refused"]
    assert "Fatal Python error" not in out.stderr

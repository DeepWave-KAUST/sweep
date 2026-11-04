"""Uncompiled eager stencils keep only their kernel for the backward.

A conv's autograd node saves its input, for the kernel's gradient; a stencil's
kernel is a constant, so the uncompiled operators run through
``sweep.operators.torch._StencilConv``, whose backward is the transposed conv
with the same kernel and keeps nothing else.  These tests pin: it equals the
plain conv bitwise (forward, gradient, forward-mode tangent) and to float
round-off (second derivative) on every kernel family; it saves no tensor; a
rollout keeps fewer fields per step; and whole uncompiled rollouts are
bitwise the plain conv's.
``STENCIL_BACKWARD_KEEPS_KERNEL_ONLY = False`` is the plain conv.
"""
import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

import sweep.operators.torch as OPS
import sweep.propagator._torch_eager as TE
from sweep.equations import (Acoustic, Acoustic3D, AcousticVRZ3D, AcousticVTI, Elastic, ElasticCurvilinear,
                             ViscoElastic)
from sweep.equations.elastic3d import Elastic as Elastic3D
from sweep.equations.utils import to_backend
from sweep.operators.general import StaggeredDerivative
from sweep.propagator.options import EagerOptions, Full
from sweep.propagator.torch import PropTorch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _deterministic_cudnn(monkeypatch):
    # cuDNN picks a non-deterministic conv backward by default: the conv
    # differs from itself at ~1e-7 then, which a bitwise check cannot tell
    # from a defect.
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)


def _operators(device, dtype):
    """name -> (u -> derivative(s), input shape), one per kernel family the
    eager equations build, through the public operators."""
    ac = Acoustic(spatial_order=8, device=device, backend="torch")
    ac.init_grad_kernels()
    lap = tuple(k.to(dtype) for k in ac.laplace_kernels)
    grad = {a: k.to(dtype) for a, k in ac.grad_kernels.items()}
    lap3 = tuple(k.to(dtype) for k in Acoustic3D(spatial_order=4, device=device, backend="torch").laplace_kernels)
    grad3 = {a: k.to(dtype) for a, k in AcousticVRZ3D(spatial_order=4, device=device, backend="torch").grad_kernels.items()}
    k1d = ac.kernel.to(dtype)
    s2, s3 = (2, 1, 15, 13), (2, 1, 9, 8, 7)
    ops = {
        "separable_d2_2d": (lambda u: torch.stack(OPS.separable_d2_2d(u, lap, 10.0, 12.0)), s2),
        "separable_d2_2d_1d_kernel": (lambda u: torch.stack(OPS.separable_d2_2d(u, k1d, 10.0, 12.0)), s2),
        "separable_d2_3d": (lambda u: torch.stack(OPS.separable_d2_3d(u, lap3, 10.0, 11.0, 12.0)), s3),
        "separable_d2_3d_1d_kernel": (lambda u: torch.stack(OPS.separable_d2_3d(u, k1d, 10.0, 11.0, 12.0)), s3),
        "laplacian_2d": (lambda u: OPS.laplacian_2d(u, lap, 10.0, 12.0), s2),
    }
    for ax in grad:
        ops[f"gradient_2d_axis{ax}"] = (lambda u, ax=ax: OPS.gradient(u, 10.0, ax, kernels=grad), s2)
    for ax in grad3:
        ops[f"gradient_3d_axis{ax}"] = (lambda u, ax=ax: OPS.gradient(u, (10.0, 11.0, 12.0), ax, kernels=grad3), s3)
    for nd in (2, 3):
        pd = StaggeredDerivative(8, device, "torch", ndim=nd)
        pd.to_backend(to_backend)
        apply = OPS.apply_kernels if nd == 2 else OPS.apply_kernels_3d
        for n in ("kxf", "kzb") + (("kyf",) if nd == 3 else ()):
            k = getattr(pd, n).to(dtype)
            ops[f"staggered{nd}d_{n}"] = (lambda u, k=k, apply=apply: apply(u, k), s2 if nd == 2 else s3)
    g = torch.Generator().manual_seed(1)
    bank2 = torch.randn(3, 1, 5, 3, generator=g, dtype=torch.float64).to(device, dtype)
    bank3 = torch.randn(3, 1, 3, 5, 3, generator=g, dtype=torch.float64).to(device, dtype)
    ops["kernel_bank_2d"] = (lambda u: OPS.apply_kernels(u, bank2), s2)
    ops["kernel_bank_3d"] = (lambda u: OPS.apply_kernels_3d(u, bank3), s3)
    return ops


OPERATOR_NAMES = sorted(_operators("cpu", torch.float64))


def _derivatives(fn, u0, r, v, t):
    """The output, the gradient of <out, r>, the second derivative d<grad, v>/dr
    and the forward-mode tangent along t."""
    u = u0.clone().requires_grad_(True)
    rr = r.clone().requires_grad_(True)
    out = fn(u)
    (gu,) = torch.autograd.grad((out * rr).sum(), u, create_graph=True)
    (gr,) = torch.autograd.grad((gu * v).sum(), rr)
    with fwAD.dual_level():
        tangent = fwAD.unpack_dual(fn(fwAD.make_dual(u0.clone(), t))).tangent
    return out.detach(), gu.detach(), gr, tangent


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("name", OPERATOR_NAMES)
def test_bitwise_the_plain_conv(name, dtype, device, monkeypatch):
    fn, shape = _operators(device, dtype)[name]
    g = torch.Generator().manual_seed(0)
    u0 = torch.randn(shape, generator=g, dtype=torch.float64).to(device, dtype)
    oshape = fn(u0).shape
    r, v, t = (torch.randn(s, generator=g, dtype=torch.float64).to(device, dtype) for s in (oshape, shape, shape))
    new = _derivatives(fn, u0, r, v, t)
    monkeypatch.setattr(OPS, "STENCIL_BACKWARD_KEEPS_KERNEL_ONLY", False)
    ref = _derivatives(fn, u0, r, v, t)
    for what, a, b in zip(("forward", "gradient", "second derivative", "tangent"), new, ref):
        if what == "second derivative":
            # The conv's double backward is other conv calls than the backward of
            # the transposed conv: bitwise on the machines measured, but on a
            # GitHub CPU runner (torch 2.15, float64, the kernel bank) they summed
            # in another order.  A wrong derivative is off by O(1).
            tol = (1e-13 if dtype == torch.float64 else 1e-5) * float(b.abs().max())
            assert float((a - b).abs().max()) <= tol, f"{name}: {what} differs from the plain conv"
        else:
            assert torch.equal(a, b), f"{name}: {what} differs from the plain conv"


class _SavedTensors:
    """Every tensor autograd saves inside the context, by storage."""

    def __init__(self):
        self.storages = {}

    def pack(self, t):
        s = t.untyped_storage()
        self.storages[s.data_ptr()] = s.nbytes()
        return t

    def __enter__(self):
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)

    @property
    def nbytes(self):
        return sum(self.storages.values())


def test_the_backward_saves_no_tensor(monkeypatch):
    k = Acoustic(spatial_order=4, device="cpu", backend="torch").laplace_kernels[1]
    u = torch.randn(1, 1, 12, 12, requires_grad=True)
    with _SavedTensors() as saved:
        out = OPS.apply_kernels(u, k)
    assert type(out.grad_fn).__name__ == "_StencilConvBackward"
    assert not saved.storages
    monkeypatch.setattr(OPS, "STENCIL_BACKWARD_KEEPS_KERNEL_ONLY", False)
    with _SavedTensors() as saved:
        out = OPS.apply_kernels(u, k)
    assert type(out.grad_fn).__name__ == "ConvolutionBackward0"
    assert u.untyped_storage().data_ptr() in saved.storages     # the input the change drops


def test_plain_conv_where_no_backward_runs_or_compiling(monkeypatch):
    k = Acoustic(spatial_order=4, device="cpu", backend="torch").laplace_kernels[1]
    u = torch.randn(1, 1, 12, 12, requires_grad=True)
    with torch.no_grad():
        assert OPS.apply_kernels(u, k).grad_fn is None
    assert OPS.apply_kernels(u.detach(), k).grad_fn is None
    # A differentiated kernel needs the conv's own backward.
    kg = k.clone().requires_grad_(True)
    assert type(OPS.apply_kernels(u, kg).grad_fn).__name__ == "ConvolutionBackward0"
    # Traced, an unregistered kernel (a clone carries no taps) keeps the conv
    # it always had: the compiled path is untouched.
    monkeypatch.setattr(OPS, "_is_compiling", lambda: True)
    assert type(OPS.apply_kernels(u, k.clone()).grad_fn).__name__ == "ConvolutionBackward0"


# ---- rollouts ------------------------------------------------------------------
def _prop(cls, shape, nt, memory="full", device="cpu", **kw):
    eq = cls(spatial_order=4, device=device, backend="torch")
    return PropTorch(eq, shape=shape, dh=10.0, dt=1e-3, nt=nt, dev=device, impl="eager", abcn=4, B=1,
                     memory=Full() if memory == "full" else None,
                     eager_options=EagerOptions(use_compile=False), **kw)


def _shoot(prop, cls_models, shape, nt, device="cpu"):
    t = np.arange(nt) * 1e-3 - 0.02
    a = (np.pi * 25.0 * t) ** 2
    wavelet = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=device, requires_grad=True)
    nx = shape[-1]
    if len(shape) == 2:
        src, rec = np.array([[nx // 2, 3]]), np.array([[[ix, 3] for ix in range(2, nx - 2, 3)]])
    else:
        src = np.array([[nx // 2, shape[1] // 2, 3]])
        rec = np.array([[[ix, shape[1] // 2, 3] for ix in range(2, nx - 2, 3)]])
    ramp = torch.linspace(0, 1, shape[0]).view(-1, *([1] * (len(shape) - 1)))
    models = [(v * (1 + 0.05 * ramp)).expand(shape).contiguous().to(device).requires_grad_(True)
              for v in cls_models]
    return prop(wavelet, src, rec, models=models), models + [wavelet]


VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "z": 5.0e6, "epsilon": 0.1, "delta": 0.05,
          "Qp": 80.0, "Qs": 60.0}
_CURV_HILL = 2.0 - 2.0 * np.exp(-((np.arange(24) - 12.0) / 5.0) ** 2)
ROLLOUTS = [
    pytest.param(Acoustic, (20, 24), {}, id="acoustic2d"),            # laplacian + CPML gradient
    pytest.param(Acoustic3D, (10, 9, 8), {}, id="acoustic3d"),
    pytest.param(AcousticVRZ3D, (10, 9, 8), {}, id="acoustic_vrz3d"),  # 3-D gradient kernels
    pytest.param(AcousticVTI, (20, 24), {}, id="acoustic_vti"),        # separable components
    pytest.param(Elastic, (20, 24), {}, id="elastic2d"),              # staggered apply_kernels
    pytest.param(Elastic, (20, 24), {"memory": "ckpt"}, id="elastic2d_ckpt"),
    pytest.param(Elastic3D, (10, 9, 8), {}, id="elastic3d"),
    pytest.param(ViscoElastic, (20, 24), {"free_surface": True}, id="visco_elastic_free_surface"),
    pytest.param(ElasticCurvilinear, (20, 24), {"free_surface": True, "topography": _CURV_HILL},
                 id="elastic_curvilinear"),                           # asymmetric operator
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls,shape,kw", ROLLOUTS)
def test_uncompiled_rollout_bitwise_the_plain_conv(cls, shape, kw, device, monkeypatch):
    kw = dict(kw)
    memory = kw.pop("memory", "full")
    nt = 30 if len(shape) == 2 else 16
    prop = _prop(cls, shape, nt, memory, device, **kw)
    values = [VALUES[n] for n in cls(spatial_order=4, device="cpu", backend="torch").models]

    def run():
        d, leaves = _shoot(prop, values, shape, nt, device)
        grads = torch.autograd.grad(d.square().sum(), leaves)
        return [d.detach()] + [g.detach() for g in grads]

    new = run()
    monkeypatch.setattr(OPS, "STENCIL_BACKWARD_KEEPS_KERNEL_ONLY", False)
    ref = run()
    for i, (a, b) in enumerate(zip(new, ref)):
        assert torch.equal(a, b), f"{'record' if i == 0 else f'gradient {i - 1}'} differs from the plain conv"


def test_second_derivative_through_a_rollout_bitwise_the_plain_conv(monkeypatch):
    shape, nt = (20, 24), 30
    prop = _prop(Acoustic, shape, nt)

    def hv():
        d, leaves = _shoot(prop, [VALUES["vp"]], shape, nt)
        (g,) = torch.autograd.grad(d.square().sum(), leaves[0], create_graph=True)
        h = torch.linspace(-1.0, 1.0, g.numel()).view_as(g)
        return torch.autograd.grad((g * h).sum(), leaves[0])[0]

    new = hv()
    monkeypatch.setattr(OPS, "STENCIL_BACKWARD_KEEPS_KERNEL_ONLY", False)
    assert torch.equal(new, hv())


def _saved_fields_per_step(cls, shape, values, device):
    """Wavefields kept for the backward per time step: the field-sized
    storages autograd saves in a 16-step and an 8-step rollout, differenced."""
    per = []
    for nt in (8, 16):
        prop = _prop(cls, shape, nt, device=device)
        with _SavedTensors() as saved:
            _shoot(prop, values, shape, nt, device)
        field = 4 * int(np.prod(prop._runtime_shape()))
        per.append(sum(b for b in saved.storages.values() if b >= field // 2))
    return (per[1] - per[0]) / 8 / field


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls,values,kept,plain", [
    # The two Laplacian components read one field; on CUDA Acoustic also takes
    # its two CPML gradients by conv (on the CPU they are torch.gradient, which
    # saves nothing).  Elastic: five staggered-derivative inputs.
    pytest.param(Acoustic, [2500.0], 2, {"cpu": 3, "cuda": 5}, id="acoustic2d"),
    pytest.param(Elastic, [2500.0, 1400.0, 2000.0], 8, {"cpu": 13, "cuda": 13}, id="elastic2d"),
])
def test_a_rollout_keeps_fewer_fields_per_step(cls, values, kept, plain, device, monkeypatch):
    # The stencils' share alone: the time-invariant coefficients made once per
    # rollout save more (test_eager_time_invariants.py).
    monkeypatch.setattr(TE, "HOIST_INVARIANTS", False)
    shape = (20, 24)
    assert _saved_fields_per_step(cls, shape, values, device) == kept
    monkeypatch.setattr(OPS, "STENCIL_BACKWARD_KEEPS_KERNEL_ONLY", False)
    assert _saved_fields_per_step(cls, shape, values, device) == plain[device]

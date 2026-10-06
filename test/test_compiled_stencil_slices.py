"""Under torch.compile the eager stencils run as shifted slices, not convs.

A conv is an opaque cuDNN call that Inductor cannot fuse with the pointwise
update around it; the same zero-padded cross-correlation written as a sum of
shifted slices fuses into the step kernel.  ``register_stencil`` records each
kernel's taps when the kernel is built and the operators switch to the slice
form only while Dynamo traces (uncompiled eager keeps the conv, where it is
faster).  These tests pin: the slice form equals the conv on every kernel
family sweep builds, a compiled step has no conv left in its graph, and the
compiled slice path agrees with the uncompiled conv path.
"""
import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import sweep.operators.torch as OPS
from sweep.equations import Acoustic, Acoustic3D, AcousticVRZ3D, Elastic, ElasticTTI
from sweep.equations.elastic3d import Elastic as Elastic3D
from sweep.equations.utils import to_backend
from sweep.operators.general import StaggeredDerivative
from sweep.operators.rsg import RSGDerivative
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _release_compiled_state():
    """Dynamo's cache keeps the compiled steps' propagators -- and their GPU
    buffers -- alive; release them so a later test that reads the absolute
    peak memory (test_eager_boundary_saving) does not count them."""
    yield
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _kernels():
    """Every conv-kernel family the eager equations build, on CPU."""
    ks = {}
    ac = Acoustic(spatial_order=8, device="cpu", backend="torch")
    ac.init_grad_kernels()                       # 9x9 square, one non-zero row/column
    ks["laplace_z"], ks["laplace_x"] = ac.laplace_kernels
    ks["grad_z"], ks["grad_x"] = ac.grad_kernels[-2], ac.grad_kernels[-1]
    ac3 = Acoustic3D(spatial_order=4, device="cpu", backend="torch")
    for name, k in zip(("z", "y", "x"), ac3.laplace_kernels):
        ks[f"laplace3d_{name}"] = k
    vrz = AcousticVRZ3D(spatial_order=4, device="cpu", backend="torch")
    for ax, k in vrz.grad_kernels.items():
        ks[f"vrz3d_grad{ax}"] = k
    for nd in (2, 3):
        pd = StaggeredDerivative(8, "cpu", "torch", ndim=nd)
        pd.to_backend(to_backend)
        names = ("kxf", "kxb", "kzf", "kzb") + (("kyf", "kyb") if nd == 3 else ())
        for n in names:
            ks[f"staggered{nd}d_{n}"] = getattr(pd, n)
    rsg = RSGDerivative(8, "cpu", "torch")
    rsg.to_backend(to_backend)
    for n in ("_kx_fwd", "_kx_bwd", "_kz_fwd", "_kz_bwd"):
        ks[f"rsg{n}"] = getattr(rsg, n)
    return ks


KERNELS = _kernels()


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_slices_equal_the_conv(name):
    k = KERNELS[name]
    taps = getattr(k, OPS._TAPS_ATTR, None)
    assert taps, f"{name} was built without registered taps"
    g = torch.Generator().manual_seed(0)
    shape = (2, 1, 23, 19) if k.ndim == 4 else (2, 1, 13, 11, 9)
    u = torch.randn(shape, generator=g, dtype=torch.float64)
    conv = F.conv2d if k.ndim == 4 else F.conv3d
    ref = conv(u, k.double(), padding=tuple(s // 2 for s in k.shape[2:]))
    torch.testing.assert_close(OPS._apply_taps(u, taps), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pads", [(2, 2), (0, 3), (1, 2, 3)])
def test_out_of_place_halo_zeroing_matches_in_place(pads):
    u = torch.randn((2, 1) + (12,) * len(pads))
    ref = (OPS._zero_halo if len(pads) == 2 else OPS._zero_halo_3d)(u.clone(), pads)
    torch.testing.assert_close(OPS._zero_halo_where(u, pads), ref, rtol=0, atol=0)


def test_kernels_the_slice_form_cannot_express_stay_unregistered():
    even = OPS.register_stencil(torch.ones(1, 1, 1, 4))
    multi_in = OPS.register_stencil(torch.ones(1, 2, 1, 5))
    assert not hasattr(even, OPS._TAPS_ATTR)
    assert not hasattr(multi_in, OPS._TAPS_ATTR)


# ---- compiled steps --------------------------------------------------------
EQUATIONS = [
    pytest.param(Acoustic, (40, 44), False, id="acoustic2d"),
    pytest.param(Acoustic3D, (16, 14, 12), False, id="acoustic3d"),
    pytest.param(AcousticVRZ3D, (16, 14, 12), False, id="acoustic_vrz3d"),
    pytest.param(Elastic, (40, 44), False, id="elastic2d"),
    # The free surface builds its near-surface order-2 operator lazily, on the
    # first (compiled) step; it must still come out registered.
    pytest.param(Elastic, (40, 44), True, id="elastic2d_free_surface"),
    pytest.param(Elastic3D, (16, 14, 12), False, id="elastic3d"),
    pytest.param(ElasticTTI, (40, 44), False, id="elastic_tti_rsg"),
]

MODEL_VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "z": 5.0e6, "vp0": 2500.0, "vs0": 1400.0,
                "epsilon": 0.1, "delta": 0.05, "gamma": 0.05, "theta": 0.3, "phi": 0.2}


def _run(cls, shape, free_surface, use_compile, backend=None):
    eq = cls(spatial_order=4, device=DEV, backend="torch")
    opts = EagerOptions(use_compile=use_compile, compile_backend=backend)
    solver = PropTorch(eq, shape=shape, dh=10.0, dt=1e-3, dev=DEV, impl="eager", abcn=8, nt=60,
                       eager_options=opts, free_surface=free_surface)
    wavelet = torch.zeros(60, device=DEV)
    wavelet[3] = 1.0
    src = np.array([[shape[-1] // 2, 4] if len(shape) == 2 else [shape[-1] // 2, shape[-2] // 2, 4]])
    if len(shape) == 2:
        rec = np.array([[[ix, 4] for ix in range(2, shape[-1] - 2, 3)]])
    else:
        rec = np.array([[[ix, shape[-2] // 2, 4] for ix in range(2, shape[-1] - 2, 3)]])
    # A gentle depth ramp: a homogeneous model hides operator defects.
    ramp = torch.linspace(0, 1, shape[0], device=DEV).view(-1, *([1] * (len(shape) - 1)))
    models = [(MODEL_VALUES[n] * (1 + 0.02 * ramp)).expand(shape).clone().requires_grad_(True)
              for n in eq.models]
    out = solver(wavelet, src, rec, models=models)
    out.square().sum().backward()
    return out.detach(), [m.grad for m in models]


def _conv_node_counter(counts):
    def backend(gm, example_inputs):
        counts.append(sum(1 for n in gm.graph.nodes
                          if n.op == "call_function" and "conv" in str(n.target)))
        return gm.forward
    return backend


@pytest.mark.parametrize("cls,shape,free_surface", EQUATIONS)
def test_compiled_step_has_no_conv(cls, shape, free_surface, monkeypatch):
    # The counter must be able to see a conv, or "zero" proves nothing.
    torch._dynamo.reset()
    monkeypatch.setattr(OPS, "SLICE_STENCILS_UNDER_COMPILE", False)
    before = []
    _run(cls, shape, free_surface, True, _conv_node_counter(before))
    assert sum(before) > 0, "conv counter saw nothing with the slice form off"

    torch._dynamo.reset()
    monkeypatch.setattr(OPS, "SLICE_STENCILS_UNDER_COMPILE", True)
    after = []
    _run(cls, shape, free_surface, True, _conv_node_counter(after))
    assert after and sum(after) == 0, f"conv left in the compiled step: {after}"


@pytest.mark.parametrize("cls,shape,free_surface", EQUATIONS)
def test_compiled_slices_match_uncompiled_convs(cls, shape, free_surface):
    torch._dynamo.reset()
    ref_out, ref_grads = _run(cls, shape, free_surface, False)
    out, grads = _run(cls, shape, free_surface, True, "aot_eager")
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-6 * ref_out.abs().max().item())
    for g, rg in zip(grads, ref_grads):
        # By hand: F.cosine_similarity clamps the norm product at 1e-8, which
        # zeroes the cosine of the (tiny, ~1e-13) elastic gradients.
        a, b = g.flatten().double(), rg.flatten().double()
        cos = (a @ b / (a.norm() * b.norm())).item()
        rel = ((a - b).norm() / b.norm()).item()
        assert cos > 1 - 1e-6 and rel < 1e-4, (cos, rel)

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
from sweep.equations import (Acoustic, Acoustic3D, AcousticVRZ3D, DASMu, Elastic, ElasticCurvilinear, ElasticTTI,
                             ElasticVRR, ViscoElastic)
from sweep.equations.elastic3d import Elastic as Elastic3D
from sweep.equations.utils import to_backend
from sweep.operators.general import StaggeredDerivative
from sweep.operators.rsg import RSGDerivative
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
from conftest import requires_compile

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def _release_compiled_state():
    """Dynamo's cache keeps the compiled steps' propagators -- and their GPU
    buffers -- alive; release them so a later test that reads the absolute
    peak memory (test_eager_boundary_saving) does not count them.  Taken by
    the tests that compile only: the reset and a full gc.collect() cost
    0.1-0.25 s a test, and the kernel-level tests build no compiled state."""
    yield
    if hasattr(torch, "_dynamo"):
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
    # Unregistered: the instance has no taps and reads the class-level None.
    assert getattr(even, OPS._TAPS_ATTR) is None
    assert getattr(multi_in, OPS._TAPS_ATTR) is None


# ---- compiled steps --------------------------------------------------------
# Each case is compiled once for three checks at once: no conv in the graph,
# one graph per rollout, and the record and gradients of the uncompiled conv
# path.  Compiling is most of this file's runtime.
_HILL = np.array([2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0, 1])
# A smooth hill for the boundary-fitted grid (surface rows, fractional).
_CURV_HILL = 2.0 - 2.0 * np.exp(-((np.arange(44) - 22.0) / 8.0) ** 2)

EQUATIONS = [
    pytest.param(Acoustic, (40, 44), {}, id="acoustic2d"),
    pytest.param(Acoustic3D, (16, 14, 12), {}, id="acoustic3d"),
    pytest.param(AcousticVRZ3D, (16, 14, 12), {}, id="acoustic_vrz3d"),
    pytest.param(Elastic, (40, 44), {}, id="elastic2d"),
    pytest.param(Elastic, (40, 44), {"free_surface": True}, id="elastic2d_free_surface"),
    # Small enough that the wave reaches the bottom and both sides in 60 steps.
    pytest.param(Elastic, (10, 14), {"free_surface": ["bottom", "left", "right"]},
                 id="elastic2d_per_edge_free_surface"),
    pytest.param(Elastic, (12, 16), {"topography": _HILL, "topo_method": "image"},
                 id="elastic2d_topography"),
    pytest.param(Elastic3D, (16, 14, 12), {}, id="elastic3d"),
    pytest.param(ElasticTTI, (40, 44), {}, id="elastic_tti_rsg"),
    # Their steps read lookup tables and a per-run cache that once broke the
    # graph on every torch.
    pytest.param(ElasticVRR, (40, 44), {}, id="elastic_vrr"),
    pytest.param(ViscoElastic, (40, 44), {"free_surface": True}, id="visco_elastic_free_surface"),
    pytest.param(DASMu, (40, 44), {"free_surface": True}, id="dasmu2d_free_surface"),
    # Its operator caches its kernels, and its free-surface closure is a
    # block of one-sided stencils; both once broke the graph at every
    # derivative.
    pytest.param(ElasticCurvilinear, (40, 44), {"free_surface": True, "topography": _CURV_HILL},
                 id="elastic_curvilinear"),
]
MODEL_VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "z": 5.0e6, "vp0": 2500.0, "vs0": 1400.0,
                "epsilon": 0.1, "delta": 0.05, "gamma": 0.05, "theta": 0.3, "phi": 0.2,
                "Rp_x": 0.01, "Rp_z": 0.01, "Rs_x": 0.01, "Rs_z": 0.01, "Qp": 80.0, "Qs": 60.0}


def _run(cls, shape, surface, use_compile, backend=None, fullgraph=False, cpml_strips=False):
    """Forward + backward."""
    eq = cls(spatial_order=4, device=DEV, backend="torch")
    opts = EagerOptions(use_compile=use_compile, compile_backend=backend, compile_fullgraph=fullgraph,
                        cpml_strips=cpml_strips)
    solver = PropTorch(eq, shape=shape, dh=10.0, dt=1e-3, dev=DEV, impl="eager", abcn=4, nt=60,
                       eager_options=opts, **surface)
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


def _counting_convs(counts, backend):
    """``backend``, after counting the conv nodes of each graph it is given."""
    from torch._dynamo.backends.registry import lookup_backend   # torch 2.0+, unlike torch._dynamo's
    compile_graph = lookup_backend(backend)

    def count(gm, example_inputs):
        counts.append(sum(1 for n in gm.graph.nodes
                          if n.op == "call_function" and "conv" in str(n.target)))
        return compile_graph(gm, example_inputs)
    return count


def _assert_matches(run, ref):
    (out, grads), (ref_out, ref_grads) = run, ref
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-6 * ref_out.abs().max().item())
    for g, rg in zip(grads, ref_grads):
        # By hand: F.cosine_similarity clamps the norm product at 1e-8, which
        # zeroes the cosine of the (tiny, ~1e-13) elastic gradients.
        a, b = g.flatten().double(), rg.flatten().double()
        cos = (a @ b / (a.norm() * b.norm())).item()
        rel = ((a - b).norm() / b.norm()).item()
        assert cos > 1 - 1e-6 and rel < 1e-4, (cos, rel)


@requires_compile
@pytest.mark.usefixtures("_release_compiled_state")
def test_the_conv_counter_sees_a_conv(monkeypatch):
    """With the slice form off the counter must find the convs, or the zero
    the tests below assert proves nothing."""
    torch._dynamo.reset()
    monkeypatch.setattr(OPS, "SLICE_STENCILS_UNDER_COMPILE", False)
    counts = []
    _run(Acoustic, (40, 44), {}, True, _counting_convs(counts, "eager"))
    assert sum(counts) > 0, "conv counter saw nothing with the slice form off"


@requires_compile
@pytest.mark.usefixtures("_release_compiled_state")
@pytest.mark.parametrize("cls,shape,surface", EQUATIONS)
def test_compiled_rollout_is_one_conv_free_graph_matching_uncompiled(cls, shape, surface):
    """A rollout compiles one graph (fullgraph=True raises on any break) with
    no conv left, and gives the uncompiled conv path's record and gradients
    through AOTAutograd.  A break splits the step's fusion; before torch 2.5
    the taps lookup and the free-surface switches broke it on every call, and
    the slices never ran.  A second graph was a step's inputs' requires_grad
    still changing: those first steps run uncompiled (_SettledSteps), which
    also builds the free surface's order-2 twin before the trace."""
    torch._dynamo.reset()
    counts = []
    run = _run(cls, shape, surface, True, _counting_convs(counts, "aot_eager"), fullgraph=True)
    assert len(counts) == 1, f"{len(counts)} graphs compiled for one rollout"
    assert counts == [0], f"conv left in the compiled step: {counts}"
    _assert_matches(run, _run(cls, shape, surface, False))


@requires_compile
@pytest.mark.usefixtures("_release_compiled_state")
@pytest.mark.parametrize("cls,shape,surface", [
    pytest.param(Elastic, (40, 44), {"free_surface": True}, id="elastic2d_free_surface"),
    pytest.param(ElasticCurvilinear, (40, 44), {"free_surface": True, "topography": _CURV_HILL},
                 id="elastic_curvilinear"),
    pytest.param(ElasticCurvilinear, (40, 44), {}, id="elastic_curvilinear_no_free_surface"),
])
def test_compiled_strips_rollout_is_one_conv_free_graph_matching_uncompiled(cls, shape, surface):
    """The same with the CPML memories on the absorbing strips: there the
    compiled step retakes each derivative from a window of its field."""
    torch._dynamo.reset()
    counts = []
    run = _run(cls, shape, surface, True, _counting_convs(counts, "aot_eager"), fullgraph=True, cpml_strips=True)
    assert len(counts) == 1, f"{len(counts)} graphs compiled for one rollout"
    assert counts == [0], f"conv left in the compiled step: {counts}"
    _assert_matches(run, _run(cls, shape, surface, False))


@requires_compile
@pytest.mark.usefixtures("_release_compiled_state")
def test_plain_warmup_prefix_changes_nothing(monkeypatch):
    """Before torch 2.5 a checkpointed rollout runs its first, compiling steps
    on the plain tape, because compiling inside the checkpoint breaks its
    recompute.  The record and gradients must not notice."""
    import sweep.propagator._torch_eager as TE
    if TE.CKPT_PLAIN_WARMUP:
        pytest.skip("this torch cannot compile inside the checkpoint at all; "
                    "test_compiled_rollout_is_one_conv_free_graph_matching_uncompiled covers the warm-up here")
    runs = []
    for warm in (False, True):
        torch._dynamo.reset()
        monkeypatch.setattr(TE, "CKPT_PLAIN_WARMUP", warm)
        runs.append(_run(Acoustic, (40, 44), {}, True, "aot_eager"))
    (out0, g0), (out1, g1) = runs
    assert torch.equal(out0, out1)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))

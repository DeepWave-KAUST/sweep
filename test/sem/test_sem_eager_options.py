"""``PropSEM(eager_options=...)``: the compile switch and settings, as on PropTorch.

The eager step is compiled by default with PropTorch's ``EagerOptions``; the
settings reach ``torch.compile``; the PropTorch options the SEM step has no
counterpart for are refused; and the call argument ``compile_step`` that the
switch used to be names its replacement.
"""
import numpy as np
import pytest
import torch

import sweep._compile as C
import sweep.propagator._torch_eager as TE
import sweep.sem.prop_base as PB
from sweep.propagator import BoundarySaving, Ckpt, EagerOptions
from sweep.sem import PropSEM, AcousticSEM2D

SHAPE, NT = (21, 33), 40


def _prop(**kw):
    return PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                   free_surface=True, f_dominant=15.0, dev="cpu", dtype=torch.float64,
                   impl="eager", **kw)


def _call(prop, nshots=1):
    w = torch.zeros(NT, dtype=torch.float64)
    w[3] = 1.0
    src = np.array([[10 + 10 * b, 2.0] for b in range(nshots)])
    rec = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]
    vp = torch.full(SHAPE, 2000.0, dtype=torch.float64, requires_grad=True)
    out = prop(w, src, rec, models=[vp])
    out.square().sum().backward()
    return out.detach(), vp.grad


@pytest.fixture
def compiles(monkeypatch):
    """torch.compile replaced by a recorder that returns the function itself."""
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(kwargs)
        return fn

    monkeypatch.setattr(C, "_FAILED", {})
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    monkeypatch.setattr(torch, "compile", fake_compile)
    return seen


def test_the_default_compiles_like_proptorch(monkeypatch, compiles):
    monkeypatch.setattr(PB, "_DEFAULT_EAGER_OPTIONS", None)       # undo the suite's uncompiled default
    p = _prop()
    assert p.eager_options == EagerOptions()
    assert p.eager_options.use_compile is True
    _call(p)
    assert len(compiles) == 1
    assert compiles[0] == {"mode": "default", "dynamic": False, "fullgraph": False}


@pytest.mark.parametrize("memory, nshots", [(None, 1), (BoundarySaving(), 1), (Ckpt(chunks=8), 1),
                                            (None, 2)],
                         ids=["loop", "boundary_saving", "ckpt", "vmapped"])
def test_the_settings_reach_torch_compile(compiles, memory, nshots):
    opts = EagerOptions(compile_mode="max-autotune-no-cudagraphs", compile_dynamic=True,
                        compile_fullgraph=True, compile_backend="aot_eager")
    kw = {} if memory is None else {"memory": memory}
    _call(_prop(eager_options=opts, **kw), nshots)
    assert compiles and all(c == {"mode": "max-autotune-no-cudagraphs", "dynamic": True,
                                  "fullgraph": True, "backend": "aot_eager"} for c in compiles)


@pytest.mark.parametrize("memory, nshots", [(None, 1), (BoundarySaving(), 1), (Ckpt(chunks=8), 1),
                                            (None, 2)],
                         ids=["loop", "boundary_saving", "ckpt", "vmapped"])
def test_use_compile_false_never_compiles(compiles, memory, nshots):
    kw = {} if memory is None else {"memory": memory}
    _call(_prop(eager_options=EagerOptions(use_compile=False), **kw), nshots)
    assert compiles == []


def test_a_dict_reads_like_the_dataclass():
    assert _prop(eager_options={"use_compile": False, "compile_mode": "reduce-overhead"}).eager_options \
        == EagerOptions(use_compile=False, compile_mode="reduce-overhead")
    with pytest.raises(TypeError, match="use_compiled"):
        _prop(eager_options={"use_compiled": False})


@pytest.mark.parametrize("field, match", [("cpml_strips", "band_split"),
                                          ("store_last_wavefield", "return_wavefield")])
def test_proptorch_only_options_are_refused(field, match):
    with pytest.raises(NotImplementedError, match=match):
        _prop(eager_options=EagerOptions(**{field: True}))


def test_the_old_switch_names_its_replacement():
    w = torch.zeros(NT, dtype=torch.float64)
    rec = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]
    vp = torch.full(SHAPE, 2000.0, dtype=torch.float64)
    with pytest.raises(TypeError, match=r"eager_options=EagerOptions\(use_compile=\.\.\.\)"):
        _prop()(w, np.array([[10, 2.0]]), rec, models=[vp], compile_step=True)
    for name in ("use_compile", "compile_step"):
        with pytest.raises(TypeError, match="eager_options=EagerOptions"):
            _prop(**{name: False})


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph compile modes")


@cuda
@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("memory", [BoundarySaving(), Ckpt(chunks=8)], ids=["boundary_saving", "ckpt"])
def test_cuda_graph_modes_are_refused_where_they_overwrite(mode, memory):
    """They replay the step with outputs in buffers the next replay reuses; boundary
    saving and Ckpt step again from kept states, and torch raised mid-run."""
    with pytest.raises(ValueError, match="compile_mode='default'"):
        PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8, free_surface=True,
                f_dominant=15.0, dev="cuda", impl="eager", memory=memory,
                eager_options=EagerOptions(compile_mode=mode))


@cuda
def test_reduce_overhead_runs_the_full_tape():
    def run(opts):
        p = PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                    free_surface=True, f_dominant=15.0, dev="cuda", impl="eager",
                    eager_options=opts)
        w = torch.zeros(NT, device="cuda")
        w[3] = 1.0
        rec = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]
        vp = torch.full(SHAPE, 2000.0, device="cuda", requires_grad=True)
        out = p(w, np.array([[16, 2.0]]), rec, models=[vp])
        (out / out.detach().abs().max()).square().sum().backward()
        return out.detach(), vp.grad
    torch._dynamo.reset()
    r0, g0 = run(EagerOptions(use_compile=False))
    r1, g1 = run(EagerOptions(compile_mode="reduce-overhead"))
    rel = lambda a, b: float((a - b).norm() / b.norm())
    assert rel(r1, r0) < 1e-5 and rel(g1, g0) < 1e-4

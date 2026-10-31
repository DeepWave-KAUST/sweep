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
from sweep.sem import PropSEM, AcousticSEM2D, ElasticVSSEM2DDeformed

from conftest import requires_compile

SHAPE, NT = (21, 33), 40


def _prop(**kw):
    return PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                   free_surface=True, f_dominant=15.0, dev="cpu", dtype=torch.float64,
                   impl="eager", **kw)


def _call(prop, nshots=1, grad=True):
    """One call in grad mode; ``grad`` makes the model require grad and runs the backward."""
    w = torch.zeros(NT, dtype=torch.float64)
    w[3] = 1.0
    src = np.array([[10 + 10 * b, 2.0] for b in range(nshots)])
    rec = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]
    vp = torch.full(SHAPE, 2000.0, dtype=torch.float64, requires_grad=grad)
    out = prop(w, src, rec, models=[vp])
    if grad:
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


def _announced(capsys):
    return [ln for ln in capsys.readouterr().err.splitlines()
            if ln.startswith("sweep: torch.compile is compiling")]


def test_each_new_compile_is_announced(compiles, capsys, monkeypatch):
    """Once per new grid size, shot count, dtype and gradient recording, before it
    compiles; the first line of the process says how long it takes and how to turn
    it off."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    _call(_prop(eager_options=EagerOptions()))
    first = _announced(capsys)
    assert len(first) == 1, first
    assert "the AcousticSEM2D step for a 21x33 grid in float64 (with gradients)" in first[0]
    assert "use_compile=False" in first[0]
    _call(_prop(eager_options=EagerOptions()))
    assert _announced(capsys) == [], "the same compile again: nothing new"
    _call(_prop(eager_options=EagerOptions()), nshots=2)
    two = _announced(capsys)
    assert len(two) == 1, two
    assert "the vmapped AcousticSEM2D step for 2 shots on a 21x33 grid in float64" in two[0]
    assert "use_compile=False" not in two[0]
    _call(_prop(eager_options=EagerOptions(use_compile=False)))
    assert _announced(capsys) == []


@pytest.mark.parametrize("nshots", [1, 2], ids=["loop", "vmapped"])
def test_gradients_are_named_only_when_autograd_records(compiles, capsys, monkeypatch, nshots):
    """The step reads the model's coefficients off the equation, not its arguments:
    with grad mode on and a model without requires_grad it compiles no backward, and
    the first step of a gradient call compiles one although its fields do not
    require grad yet."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    _call(_prop(eager_options=EagerOptions()), nshots, grad=False)
    plain = _announced(capsys)
    assert len(plain) == 1 and "with gradients" not in plain[0], plain
    _call(_prop(eager_options=EagerOptions()), nshots)
    grads = _announced(capsys)
    assert len(grads) == 1 and "in float64 (with gradients)" in grads[0], grads


def test_a_forward_that_records_nothing_is_one_compile(compiles, capsys, monkeypatch):
    """Without gradients the step compiles the one graph without a backward, in grad
    mode or not (Dynamo retraces it; Inductor's cache hits): one line."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    _call(_prop(eager_options=EagerOptions(), memory=Ckpt(chunks=8)), grad=False)   # under no_grad
    assert len(_announced(capsys)) == 1
    _call(_prop(eager_options=EagerOptions()), grad=False)    # the full tape, in grad mode
    assert _announced(capsys) == []


def test_a_full_tape_after_ckpt_announces_its_own_compile(compiles, capsys, monkeypatch):
    """Ckpt's backward compiles the step with its fields requiring grad; the first
    steps of the full tape compile it with only the coefficients requiring grad --
    other compiles, with a line of their own."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    _call(_prop(eager_options=EagerOptions(), memory=Ckpt(chunks=8)))
    assert any("(with gradients)" in ln for ln in _announced(capsys))
    _call(_prop(eager_options=EagerOptions()))            # the suite's default: the full tape
    full = _announced(capsys)
    assert len(full) == 1 and "(with gradients)" in full[0], full
    _call(_prop(eager_options=EagerOptions()))
    assert _announced(capsys) == []


def test_a_new_set_of_models_that_require_grad_is_announced(compiles, capsys, monkeypatch):
    """Dynamo specializes the step on which of the coefficients it reads off the
    equation require grad: inverting vp alone, then vp and vs, compiles twice."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())

    def fwi(*grads):
        p = PropSEM(ElasticVSSEM2DDeformed, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                    free_surface=True, f_dominant=15.0, dev="cpu", dtype=torch.float64,
                    source_type=["sxx", "szz"], receiver_type=["vx", "vz"], impl="eager",
                    eager_options=EagerOptions())
        w = torch.zeros(NT, dtype=torch.float64)
        w[3] = 1.0
        rec = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]
        ms = [torch.full(SHAPE, v, dtype=torch.float64, requires_grad=g)
              for v, g in zip((2000.0, 1100.0, 2000.0), grads)]
        p(w, np.array([[16, 2.0]]), rec, models=ms).square().sum().backward()
    fwi(True, False, False)
    first = _announced(capsys)
    assert len(first) == 1 and "(with gradients)" in first[0], first
    fwi(True, True, False)
    again = _announced(capsys)
    assert len(again) == 1 and "(with gradients)" in again[0], again
    fwi(True, True, False)
    assert _announced(capsys) == []


@requires_compile
@pytest.mark.parametrize("memory, nshots", [(None, 1), (None, 2), (Ckpt(chunks=8), 2)],
                         ids=["loop", "vmapped", "vmapped_ckpt"])
def test_a_gradient_rollout_compiles_one_graph(memory, nshots):
    """Dynamo guards on each input's requires_grad, and the fields pick it up over
    the first steps. Those steps run uncompiled (SettledSteps) -- compiled, each
    built a graph of its own -- so a rollout with gradients compiles one graph,
    and gives the uncompiled record and gradient; the checkpointed chunks'
    recompute takes the same steps' paths."""
    from torch._dynamo.backends.registry import lookup_backend
    aot_eager = lookup_backend("aot_eager")
    graphs = []

    def count(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    kw = {} if memory is None else {"memory": memory}
    torch._dynamo.reset()
    try:
        out, grad = _call(_prop(eager_options=EagerOptions(compile_backend=count), **kw), nshots)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1, f"{len(graphs)} graphs compiled for one rollout"
    ref_out, ref_grad = _call(_prop(eager_options=EagerOptions(use_compile=False), **kw), nshots)
    torch.testing.assert_close(out, ref_out, rtol=1e-12, atol=1e-12 * ref_out.abs().max().item())
    torch.testing.assert_close(grad, ref_grad, rtol=1e-12, atol=1e-12 * ref_grad.abs().max().item())

"""One compiled call of the eager step advances ``EagerOptions.steps_per_call`` steps.

A compiled call costs the host its guards, wrappers and launches whatever it
computes, which on a 2-D grid outlasts the step itself on a GPU.  So a call
runs ``k`` steps, the source injection and the receiver sampling inside.  The
calls start on multiples of ``k`` once the inputs' ``requires_grad`` settles;
the steps before and the ``nt % k`` after run alone, uncompiled, so no graph
is compiled for a remainder.  These tests pin that schedule, that every path
cut at multiples of ``k`` (full tape, checkpoint chunks, the plain warm-up)
takes it, and that the numbers are those of one step a call.

The bitwise checks run the traced code uncompiled too (``traced_everywhere``:
the slice stencils and the in-place source scatter that Dynamo sees), so a
compiled call and the steps around it execute the same aten ops; ``aot_eager``
then runs the graphs op by op.  The record is then bitwise whatever ``k`` is.
The gradient is not: backward sums a field's gradient contributions per graph,
so with ``k`` steps in one graph the sums associate differently.
"""
import numpy as np
import pytest
import torch

import sweep.operators.torch as OPS
import sweep.receivers.torch as REC
import sweep.sources.torch as SRC_MOD
from conftest import requires_compile
from sweep.equations import Acoustic, Elastic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator._torch_eager import _default_steps_per_call, _uses_cudagraphs
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
SRC = np.array([[10, 4], [18, 5]])
REC_XZ = np.repeat(np.array([[[ix, 3] for ix in range(2, 26, 3)]]), 2, axis=0)
NT = 41                       # 41 % 4 == 1 and 41 % 3 == 2: every k leaves a remainder


def _wavelet(nt=NT):
    t = np.arange(nt, dtype=np.float32) * 1e-3 - 0.012
    a = (np.pi * 30.0 * t) ** 2
    w = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    return np.stack([w, 0.7 * np.roll(w, 3)])                      # one per shot


def _prop(eq=Acoustic, memory=Full, nt=NT, **eager):
    kw = dict(source_type=["szz"], receiver_type=["vz", "vx"]) if eq is Elastic else {}
    return PropTorch(eq(spatial_order=4, device="cpu"), shape=SHAPE, dh=10.0, dt=1e-3,
                     nt=nt, abcn=6, dev="cpu", impl="eager", memory=memory(),
                     eager_options=EagerOptions(**eager), **kw)


def _models(eq, grad):
    ramp = torch.linspace(0, 1, SHAPE[0]).view(-1, 1)
    values = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0}
    return [(values[n] * (1 + 0.05 * ramp)).expand(SHAPE).clone().requires_grad_(grad)
            for n in eq(spatial_order=4, device="cpu").models]


def _run(prop, eq=Acoustic, grad=True, nt=NT, adj=False, **kw):
    """The record and, with ``grad``, the gradients of the models and the
    wavelet for a fixed random weighting of the record."""
    models = _models(eq, grad)
    wav = torch.tensor(_wavelet(nt), requires_grad=grad)
    with torch.set_grad_enabled(grad):
        out = prop(wav, SRC, REC_XZ, models=models, adj=adj, **kw)
        if not grad:
            return out, []
        record = out[0] if isinstance(out, tuple) else out
        weights = torch.randn(record.shape, generator=torch.Generator().manual_seed(0))
        (record * weights).sum().backward()
    return record.detach(), [m.grad for m in models] + [wav.grad]


@pytest.fixture
def traced_everywhere(monkeypatch):
    """Run the traced forms (slice stencils, in-place index_put scatter, tap
    by tap gathers) uncompiled too, so an uncompiled step runs the very ops of
    a compiled one."""
    for mod in (OPS, SRC_MOD, REC):
        monkeypatch.setattr(mod, "_is_compiling", lambda: True)
    assert OPS._is_compiling() and SRC_MOD._is_compiling() and REC._is_compiling()


def _counting(prop):
    """Count the steps each path takes: ``calls`` lists ``(compiled, steps)``
    per call, in order."""
    impl = prop._backend_impl
    calls = []
    for k in {1, impl._resolved_steps_per_call()}:
        impl._steps_fn(k)              # compile the real steps, not the counter below
    plain = impl._plain_steps_func
    steps_fn = impl._steps_fn

    def plain_counted(fields, models, dt, h, b, samples, *rest):
        calls.append((False, len(samples)))
        return plain(fields, models, dt, h, b, samples, *rest)

    def steps_fn_counted(k):
        fn = steps_fn(k)

        def counted(fields, models, dt, h, b, samples, *rest):
            calls.append((True, len(samples)))
            return fn(fields, models, dt, h, b, samples, *rest)
        return counted

    impl._plain_steps_func = plain_counted
    impl._steps_fn = steps_fn_counted
    return calls


def _forward_calls(calls, nt=NT):
    """The calls of the forward pass: up to the one that reaches step nt
    (the checkpoint's recomputes come after)."""
    done = 0
    for i, (_, n) in enumerate(calls):
        done += n
        if done == nt:
            return calls[:i + 1]
    raise AssertionError(f"the calls advance {done} steps, not {nt}")


# ---- options -------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -2, 2.0, True, "4"])
def test_steps_per_call_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="positive integer"):
        EagerOptions(steps_per_call=bad)


def test_more_than_one_step_a_call_needs_compile():
    with pytest.raises(ValueError, match="use_compile=True"):
        EagerOptions(use_compile=False, steps_per_call=4)
    assert EagerOptions(use_compile=False, steps_per_call=1).steps_per_call == 1
    assert EagerOptions(steps_per_call=np.int64(3)).steps_per_call == 3
    # The dict route skips the dataclass: the propagator checks it itself.
    with pytest.raises(ValueError, match="use_compile=True"):
        PropTorch(Acoustic(spatial_order=4, device="cpu"), shape=SHAPE, dh=10.0, dt=1e-3, nt=NT,
                  abcn=6, dev="cpu", impl="eager",
                  eager_options={"use_compile": False, "steps_per_call": 4})


def test_the_default_is_four_for_2d_gradients_on_a_gpu_and_one_elsewhere():
    assert _default_steps_per_call(2, "cuda", records=True) == 4
    assert _default_steps_per_call(2, torch.device("cuda", 1), records=True) == 4
    assert _default_steps_per_call(2, "cuda", records=False) == 1    # a forward without gradients
    assert _default_steps_per_call(3, "cuda:0", records=True) == 1
    assert _default_steps_per_call(2, "cpu", records=True) == 1
    impl = _prop(steps_per_call=None)._backend_impl
    assert impl._resolved_steps_per_call(records=True) == 1         # 2-D on the CPU
    assert _prop(use_compile=False)._backend_impl._resolved_steps_per_call() == 1
    # An explicit value applies with or without gradients.
    assert _prop(steps_per_call=3)._backend_impl._resolved_steps_per_call(records=False) == 3


def test_a_rollout_records_when_grad_mode_is_on_and_an_input_requires_grad():
    impl = _prop()._backend_impl
    m, w = torch.zeros(3, requires_grad=True), torch.zeros(4).unbind(0)
    assert impl._records([m], w)
    assert not impl._records([m.detach()], w)
    assert impl._records([m.detach()], torch.zeros(4, requires_grad=True).unbind(0))
    with torch.no_grad():
        assert not impl._records([m], w)


def test_boundary_saving_refuses_more_than_one_step_a_call():
    with pytest.raises(ValueError, match="one step a call"):
        _prop(memory=lambda: BoundarySaving(storage="gpu"), steps_per_call=4)
    _prop(memory=lambda: BoundarySaving(storage="gpu"))            # the default is fine


def test_only_cuda_graph_modes_mark_each_call():
    assert not _uses_cudagraphs("default", False)
    assert not _uses_cudagraphs(None, False)
    assert not _uses_cudagraphs("max-autotune-no-cudagraphs", False)
    assert _uses_cudagraphs("reduce-overhead", False)
    assert _uses_cudagraphs("max-autotune", False)


def test_the_compile_notice_names_the_steps_per_call(monkeypatch, capsys):
    import sweep._compile as C
    import sweep.propagator._torch_eager as TE
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    _run(_prop(steps_per_call=4), grad=False)
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if ln.startswith("sweep: torch.compile is compiling")]
    assert len(lines) == 1 and "4 Acoustic steps per call for 2 shots on a 24x28 grid" in lines[0], lines


# ---- the schedule ------------------------------------------------------------------

@requires_compile
@pytest.mark.parametrize("memory", [Full, lambda: Ckpt(chunks=8), lambda: Ckpt(chunks=6)],
                         ids=["full", "ckpt8", "ckpt6"])
@pytest.mark.parametrize("grad", [False, True], ids=["forward", "gradient"])
@pytest.mark.parametrize("k", [1, 3, 4])
def test_calls_start_on_multiples_of_k_once_the_inputs_settle(memory, grad, k):
    """Without gradients the inputs settle at step 1, with them at step 3
    (the second-order fields pick up requires_grad one at a time).  The
    compiled calls then start at the first multiple of k and stop before the
    last nt % k steps; a checkpoint chunk is rounded up to whole calls (6 ->
    8 for k = 4), so cutting the rollout changes nothing.  The checkpoint's
    recompute replays the forward's calls."""
    prop = _prop(memory=memory, use_compile=True, compile_backend="aot_eager", steps_per_call=k)
    calls = _counting(prop)
    _run(prop, grad=grad)
    settle = 3 if grad else 1
    first = -(-settle // k) * k
    last = NT - NT % k
    expected = [(False, 1)] * first + [(True, k)] * ((last - first) // k) + [(False, 1)] * (NT - last)
    forward = calls[:len(expected)]
    assert forward == expected
    if grad and memory is not Full:
        # backward recomputes every chunk in turn, last chunk first
        assert sorted(calls[len(expected):]) == sorted(expected)
    else:
        assert len(calls) == len(expected)


@requires_compile
def test_a_rollout_compiles_one_graph_and_a_new_nt_none(monkeypatch):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(sum(1 for n in gm.graph.nodes if "index_put" in str(n.target)))
        return gm.forward

    prop = _prop(use_compile=True, compile_backend=backend, steps_per_call=4)
    _run(prop)
    assert graphs == [4], "one graph, its four steps each scattering the source"
    _run(prop, nt=NT + 6)
    _run(prop, nt=NT - 1)
    assert graphs == [4], "a new nt compiled again"


# ---- the numbers ---------------------------------------------------------------------

def _assert_bitwise(a, b):
    assert torch.equal(a, b), float((a - b).abs().max())


def _assert_close(grads, ref, rel=2e-6):
    for g, r in zip(grads, ref):
        scale = float(r.abs().max())
        assert scale > 0
        assert float((g - r).abs().max()) <= rel * scale, (float((g - r).abs().max()) / scale)


@requires_compile
@pytest.mark.usefixtures("traced_everywhere")
@pytest.mark.parametrize("eq, strips", [(Acoustic, False), (Elastic, False), (Elastic, True)],
                         ids=["acoustic", "elastic", "elastic_strips"])
@pytest.mark.parametrize("adj", [False, True], ids=["forward", "adjoint"])
def test_k_steps_a_call_record_what_one_step_a_call_records(eq, strips, adj):
    ref, ref_grads = _run(_prop(eq, use_compile=True, compile_backend="aot_eager", steps_per_call=1,
                                cpml_strips=strips), eq, adj=adj)
    for memory in (Full, lambda: Ckpt(chunks=6)):
        for k in (3, 4):
            out, grads = _run(_prop(eq, memory, use_compile=True, compile_backend="aot_eager",
                                    steps_per_call=k, cpml_strips=strips), eq, adj=adj)
            _assert_bitwise(out, ref)
            _assert_close(grads, ref_grads)


@requires_compile
@pytest.mark.usefixtures("traced_everywhere")
@pytest.mark.parametrize("k", [3, 4])
def test_checkpointing_and_the_full_tape_agree_bitwise(k):
    """The same calls on the same steps: identical records and gradients."""
    full, full_grads = _run(_prop(memory=Full, use_compile=True, compile_backend="aot_eager", steps_per_call=k))
    ckpt, ckpt_grads = _run(_prop(memory=lambda: Ckpt(chunks=7), use_compile=True, compile_backend="aot_eager",
                                  steps_per_call=k))
    _assert_bitwise(ckpt, full)
    for g, r in zip(ckpt_grads, full_grads):
        _assert_bitwise(g, r)


@requires_compile
@pytest.mark.usefixtures("traced_everywhere")
def test_the_plain_warmup_takes_the_same_steps(monkeypatch):
    """Before torch 2.5 a checkpointed rollout runs up to its first compiled
    call on the plain tape; that call ends on a multiple of k, as the chunks
    do, so the steps compiled are the same and so are the numbers."""
    import sweep.propagator._torch_eager as TE
    runs = []
    for warm in (False, True):
        monkeypatch.setattr(TE, "CKPT_PLAIN_WARMUP", warm)
        prop = _prop(memory=lambda: Ckpt(chunks=8), use_compile=True, compile_backend="aot_eager",
                     steps_per_call=4)
        calls = _counting(prop)
        result = _run(prop)
        runs.append((result, _forward_calls(calls)))
    ((out0, g0), calls0), ((out1, g1), calls1) = runs
    assert calls0 == calls1
    _assert_bitwise(out1, out0)
    for g, r in zip(g1, g0):
        _assert_bitwise(g, r)


@requires_compile
@pytest.mark.parametrize("eq", [Acoustic, Elastic], ids=["acoustic", "elastic"])
def test_compiled_calls_match_the_uncompiled_step(eq):
    """Without the traced forms forced: the compiled slices against the
    uncompiled convs, as test_compiled_stencil_slices checks one step a call."""
    ref, ref_grads = _run(_prop(eq, use_compile=False), eq)
    out, grads = _run(_prop(eq, lambda: Ckpt(chunks=8), use_compile=True, compile_backend="aot_eager",
                            steps_per_call=4), eq)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    _assert_close(grads, ref_grads, rel=1e-4)


@requires_compile
@pytest.mark.usefixtures("traced_everywhere")
def test_return_wavefield_takes_one_step_a_call():
    """A snapshot is the state after its step: the call that returns the
    wavefield advances one step at a time, and records what k = 1 does."""
    prop4 = _prop(use_compile=True, compile_backend="aot_eager", steps_per_call=4)
    calls = _counting(prop4)
    (out4, snaps4), _ = _run(prop4, grad=False, return_wavefield=True, snapshot_interval=5)
    (out1, snaps1), _ = _run(_prop(use_compile=True, compile_backend="aot_eager", steps_per_call=1),
                             grad=False, return_wavefield=True, snapshot_interval=5)
    assert {n for _, n in calls} == {1} and any(c for c, _ in calls)
    _assert_bitwise(out4, out1)
    _assert_bitwise(snaps4, snaps1)
    out_plain, _ = _run(_prop(use_compile=True, compile_backend="aot_eager", steps_per_call=4), grad=False)
    _assert_bitwise(out_plain, out4)


# ---- on the GPU, through Inductor ------------------------------------------------------

def _gpu_run(eq, k, memory, nt=96):
    kw = dict(source_type=["szz"], receiver_type=["vz"]) if eq is Elastic else {}
    prop = PropTorch(eq(spatial_order=4, device="cuda"), shape=(64, 72), dh=10.0, dt=1e-3, nt=nt, abcn=10,
                     dev="cuda", impl="eager", memory=memory(), eager_options=EagerOptions(steps_per_call=k), **kw)
    g = torch.Generator().manual_seed(1)
    vp = (2400 + 400 * torch.rand((64, 72), generator=g)).cuda()
    models = [vp] + ([vp / 1.8, torch.full_like(vp, 2000.0)] if eq is Elastic else [])
    models = [m.clone().requires_grad_(True) for m in models]
    t = np.arange(nt) * 1e-3 - 0.05
    a = (np.pi * 25 * t) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")
    src = np.array([[36, 12]])
    rec = np.array([[[ix, 12] for ix in range(4, 68, 4)]])
    out = prop(wav, src, rec, models=models)
    out.square().sum().backward()
    return out.detach(), [m.grad for m in models]


@requires_compile
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor on a GPU")
@pytest.mark.parametrize("eq", [Acoustic, Elastic], ids=["acoustic", "elastic"])
@pytest.mark.parametrize("memory", [Full, lambda: Ckpt(chunks=24)], ids=["full", "ckpt"])
def test_inductor_k_steps_a_call_match_one_step_a_call(eq, memory):
    """Through Inductor the record and gradients move by round-off only: the
    fused calls also differ from one step a call in which ops fuse, and the
    remainder steps run uncompiled."""
    ref, ref_grads = _gpu_run(eq, 1, memory)
    out, grads = _gpu_run(eq, 4, memory)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    _assert_close(grads, ref_grads, rel=1e-5)


@requires_compile
@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA propagator")
def test_by_default_only_gradient_rollouts_take_four_steps_a_call():
    prop = PropTorch(Acoustic(spatial_order=4, device="cuda"), shape=(32, 36), dh=10.0, dt=1e-3, nt=20,
                     abcn=6, dev="cuda", impl="eager", memory=Full())
    impl = prop._backend_impl
    calls = _counting(prop)
    wav = torch.zeros(20, device="cuda")
    wav[2] = 1.0
    vp = torch.full((32, 36), 2000.0, device="cuda")
    src, rec = np.array([[18, 4]]), np.array([[[ix, 4] for ix in range(2, 34, 4)]])
    with torch.no_grad():
        prop(wav, src, rec, models=[vp])
    assert {n for _, n in calls} == {1}
    calls.clear()
    prop(wav, src, rec, models=[vp.clone().requires_grad_(True)]).sum().backward()
    assert (True, 4) in calls and impl._resolved_steps_per_call(records=True) == 4


@requires_compile
@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA propagator")
def test_the_default_mode_marks_no_cuda_graph_step(monkeypatch):
    """cudagraph_mark_step_begin is for the CUDA-graph modes; the default mode
    paid for it on every step."""
    marks = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(1))
    _gpu_run(Acoustic, 4, Full, nt=24)
    assert marks == []

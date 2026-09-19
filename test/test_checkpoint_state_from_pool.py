"""The checkpoint backward's replay state comes from the propagator.

Checkpoint modes replay each segment forward from a snapshot and image it in
reverse. The state that replay steps (the forward's struct: physical fields
plus the CPML memory it checkpoints), the bisection's scratch states, the
segment histories and the velocity carriers used to be allocated inside the
compiled backward -- per call, per segment, or per recursion level. Now:

* ``forward_wavefields`` carries the replay state sets, zeroed per backward
  call (``cuda_layout.checkpoint_state_nvar`` per set: the forward slot list
  without the psi double-buffer shadows; ``1 + depth(max_segment)`` sets for a
  driver that declares ``recursive_state_depth``);
* ``checkpoint_replay`` (persistent, next to the snapshots, never re-zeroed)
  carries the segment histories, ``checkpoint_replay_shapes(..., mode)``;
* the adjoint workspace pool carries the carriers, ``backward_workspace_shapes(mode)``.

Instrument: wrap the compiled entry point, fill every handed-over grid with a
sentinel right before the call, and require every grid written afterwards
(a driver that still allocated its own would leave every sentinel). Replay
buffers may be used at a prefix of rows on the last segment and the rim is
never written, so the criterion is "some cell changed" per tensor. The same
wrapper counts the caching allocator's calls across the backward and pins the
residual per equation and mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

SENTINEL = 12345.0

# (key, mode) -> allocations the compiled backward may still make of its own,
# measured after slices 13-14; binding prefix per key below.
RESIDUAL = {
    ("acoustic2d", "ckpt_chunk"): 0, ("acoustic2d", "ckpt_recursive"): 0,
    ("acoustic3d", "ckpt_chunk"): 0, ("acoustic3d", "ckpt_recursive"): 0,
    ("elastic2d", "ckpt_chunk"): 0, ("elastic2d", "ckpt_recursive"): 0,
    ("elastic3d", "ckpt_chunk"): 0, ("elastic3d", "ckpt_recursive"): 0,
    ("das_mu2d", "ckpt_chunk"): 0, ("das_mu2d", "ckpt_recursive"): 0,
    ("das_mu3d", "ckpt_chunk"): 0, ("das_mu3d", "ckpt_recursive"): 0,
    ("elastic_tti_sg2d", "ckpt_chunk"): 0,
    ("elastic_tti_sg3d", "ckpt_chunk"): 0,
    ("acoustic_vti_1st_2d", "ckpt_chunk"): 0,
    ("acoustic_vti_1st_3d", "ckpt_chunk"): 0,
    ("das2d", "ckpt_chunk"): 0, ("das2d", "ckpt_recursive"): 0,
    ("das3d", "ckpt_chunk"): 0, ("das3d", "ckpt_recursive"): 0,
    ("elastic_tti_2nd2d", "ckpt_chunk"): 0,
    # the VRZ checkpoint sweep still negates the adjoint source inside C++
    ("vrz2d", "ckpt_chunk"): 1, ("vrz2d", "ckpt_recursive"): 1,
    ("vrz3d", "ckpt_chunk"): 1, ("vrz3d", "ckpt_recursive"): 1,
    ("lsrtm2d", "ckpt_chunk"): 0, ("lsrtm2d", "ckpt_recursive"): 0,
    ("lsrtm3d", "ckpt_chunk"): 0, ("lsrtm3d", "ckpt_recursive"): 0,
}
BINDING = {"vrz2d": "acoustic_vrz2d", "vrz3d": "acoustic_vrz3d",
           "lsrtm2d": "acoustic_lsrtm2d", "lsrtm3d": "acoustic_lsrtm3d"}
FUNC = {"ckpt_chunk": "backward_ckpt_func", "ckpt_recursive": "backward_recursive_ckpt_func"}
PARAMS = [pytest.param(k, m, marks=requires_binding(f"{BINDING.get(k, k)}_forward"))
          for (k, m) in RESIDUAL]


def _build(key, mode, nt=40):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = nt
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", mode, scenario, shape, dev, ns,
                                Path("/tmp"), "ckptpool")
    models = suite.tensors_from_models(models_init, grad_flags, dev)
    return solver, wavelet, sources, receivers, models


def _written(t):
    return bool((t != SENTINEL).any())


class _Probe:
    def __init__(self, impl, name):
        self.impl, self.name, self.fn = impl, name, getattr(impl, name)
        self.calls = []
        setattr(impl, name, self)

    def __call__(self, params):
        state = list(params.forward_wavefields)
        replay = list(params.checkpoint_replay)
        for t in state + replay:
            t.fill_(SENTINEL)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = self.fn(params)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
        self.calls.append(dict(n_state=len(state), state_written=[_written(t) for t in state],
                               replay_written=[_written(t) for t in replay], allocs=allocs,
                               steps=params.checkpoint_steps))
        return out

    def restore(self):
        setattr(self.impl, self.name, self.fn)


@pytest.mark.parametrize("key, mode", PARAMS)
def test_checkpoint_backward_replays_into_the_handed_over_state(key, mode):
    solver, wavelet, sources, receivers, models = _build(key, mode)
    impl = getattr(solver, "_backend_impl", solver)
    func = FUNC[mode]
    if getattr(impl, func, None) is None:
        pytest.skip(f"{key} has no compiled {mode} backward")
    # warm-up: lazily created CUDA state is not what the allocation count measures
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    except (RuntimeError, NotImplementedError) as e:
        if "not yet imp" in str(e) or "NotImplemented" in type(e).__name__:
            pytest.skip(f"{key} refuses {mode}: {str(e).splitlines()[0][:80]}")
        raise
    for m in models:
        m.grad = None
    probe = _Probe(impl, func)
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        probe.restore()
    assert len(probe.calls) == 1, f"{key}/{mode}: {func} ran {len(probe.calls)} times"
    call = probe.calls[0]
    layout_mode = "recursive" if mode == "ckpt_recursive" else "ckpt"
    max_segment = (impl._longest_segment(call["steps"]) if layout_mode == "recursive"
                   else impl.ckpt_chunks)
    expected = len(impl._forward_state_shapes(1, layout_mode, max_segment))
    assert call["n_state"] == expected, (
        f"{key}/{mode}: {call['n_state']} replay state tensors handed over, expected {expected}")
    unwritten = [i for i, w in enumerate(call["state_written"]) if not w]
    assert not unwritten, (
        f"{key}/{mode}: replay state tensors {unwritten} still hold the sentinel after the "
        "backward: the driver did not bind them")
    unwritten = [i for i, w in enumerate(call["replay_written"]) if not w]
    assert not unwritten, (
        f"{key}/{mode}: checkpoint_replay buffers {unwritten} still hold the sentinel: "
        "the driver did not bind them")
    assert call["allocs"] <= RESIDUAL[(key, mode)], (
        f"{key}/{mode}: the compiled backward made {call['allocs']} allocations of its own, "
        f"expected at most {RESIDUAL[(key, mode)]}")
    assert any(m.grad is not None and torch.isfinite(m.grad).all() and float(m.grad.abs().max()) > 0
               for m in models)


def test_state_sets_follow_the_recursion_depth():
    """``recursive_state_depth`` hands one scratch set per bisection level,
    mirroring eq_driver.cuh recursive_checkpoint_scratch_depth."""
    from sweep.propagator._c import _recursive_scratch_depth
    assert [_recursive_scratch_depth(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17)] == [0, 1, 2, 2, 3, 3, 4, 4, 5]

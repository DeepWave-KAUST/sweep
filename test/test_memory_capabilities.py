"""What each backend can actually do, declared once and checked against reality.

The three strategies are the same on both impls -- `impl='eager'` does have
checkpointing (chunk mode is in fact its default) and does have boundary saving.
What differs is sub-options, and the important case was silent: eager accepted
`BoundarySaving(tail_steps=...)` and dropped it. `_apply_eager_memory` forwards
only storage and storage_dtype, and the eager driver has no truncation code at
all, so the run returned a full-length gradient while reporting a truncated
request.

Refusing an unsupported option is a nuisance. Accepting it and ignoring it is a
wrong answer that looks right.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from sweep.propagator.options import (
    _MEMORY_CAPABILITIES,
    BoundarySaving,
    Ckpt,
    Full,
    check_memory_supported,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build(impl, memory):
    from sweep.equations.acoustic import Acoustic
    from sweep.propagator.torch import PropTorch
    dev = torch.device("cuda")
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PropTorch(eq, backend="torch", impl=impl, shape=(48, 56), dev=dev,
                         dh=10.0, dt=1e-3, source_type=["p"], receiver_type=["p"],
                         abcn=20, nt=50, B=1, allow_growth=True, memory=memory)


# --------------------------------------------------------------------------- #
# the silent drop this table exists to stop
# --------------------------------------------------------------------------- #
def test_eager_refuses_tail_steps_instead_of_ignoring_it():
    with pytest.raises(NotImplementedError, match="tail_steps"):
        _build("eager", BoundarySaving(storage="gpu", tail_steps=20))


def test_the_compiled_path_still_takes_tail_steps():
    assert _build("c", BoundarySaving(storage="gpu", tail_steps=20)).memory_strategy == "boundary"


@pytest.mark.parametrize("memory,match", [
    (BoundarySaving(storage="disk", disk_dir="/tmp/x", transfer_interval=8), "storage='disk'"),
    (Ckpt(mode="recursive", count=4), "mode='recursive'"),
])
def test_eager_refuses_what_it_does_not_implement(memory, match):
    with pytest.raises(NotImplementedError, match=match):
        _build("eager", memory)


# --------------------------------------------------------------------------- #
# the table cannot drift from reality
# --------------------------------------------------------------------------- #
def test_all_three_strategies_work_on_both_impls():
    """Both impls implement all three. `eager` having no checkpointing is a
    common belief and it is not so -- chunk checkpointing is eager's DEFAULT."""
    for impl in ("c", "eager"):
        assert _build(impl, Full()).memory_strategy == "full"
        assert _build(impl, BoundarySaving(storage="gpu")).memory_strategy == "boundary"
        assert _build(impl, Ckpt(mode="chunk", chunks=10)).memory_strategy == "ckpt"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_every_declared_boundary_storage_really_builds(impl, tmp_path):
    """A capability claimed in the table but broken in the backend fails here,
    which is what stops the table becoming documentation."""
    for storage in _MEMORY_CAPABILITIES[impl]["boundary_storage"]:
        kw = dict(storage=storage)
        if storage == "disk":
            kw.update(disk_dir=str(tmp_path), transfer_interval=8)
        elif storage == "cpu":
            kw.update(transfer_interval=10)
        assert _build(impl, BoundarySaving(**kw)).memory_strategy == "boundary"


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_every_declared_ckpt_mode_really_builds(impl):
    for mode in _MEMORY_CAPABILITIES[impl]["ckpt_mode"]:
        kw = dict(mode="chunk", chunks=10) if mode == "chunk" else dict(mode="recursive", count=4)
        assert _build(impl, Ckpt(**kw)).memory_strategy == "ckpt"


# --------------------------------------------------------------------------- #
# the checker itself
# --------------------------------------------------------------------------- #
def test_the_check_is_a_no_op_for_an_unknown_backend_or_no_request():
    check_memory_supported("jax", BoundarySaving(tail_steps=5))   # not in the table
    check_memory_supported("eager", None)


# --------------------------------------------------------------------------- #
# per-equation gaps the per-impl table cannot see
# --------------------------------------------------------------------------- #
def test_recursive_ckpt_without_binding_is_refused_at_construction():
    """ElasticTTISG ships no recursive-ckpt C binding. The request used to be
    silently downgraded to chunk ckpt at the default interval, the requested
    count ignored -- a different memory/perf behaviour than asked for."""
    from sweep.equations import ElasticTTISG
    from sweep.propagator.torch import PropTorch
    dev = torch.device("cuda")
    eq = ElasticTTISG(spatial_order=4, device=dev, backend="torch")
    with pytest.raises(NotImplementedError, match="recursive-checkpoint"):
        PropTorch(eq, backend="torch", impl="c", shape=(48, 56), dev=dev,
                  dh=10.0, dt=1e-3, abcn=20, nt=50, B=1,
                  memory=Ckpt(mode="recursive", count=4))


def test_recursive_ckpt_with_binding_still_constructs():
    assert _build("c", Ckpt(mode="recursive", count=4)).memory_strategy == "ckpt"

"""Regression: the boundary-saving backward must not allocate a wavefield per call.

``<eq>/backward.cu`` hands ``last_two`` to ``EffectiveBoundarySaver::allocate``.
It used to pass ``{}``, so ``allocate_last_two`` took the self-allocating branch
and built a fresh ``{nvar, 2, B, 1, nz, ny, nx}`` FP32 buffer on EVERY call --
a buffer the backward never reads (its reverse seed comes from ``p.u_last_two``
directly).  Harmless for a monolithic backward (one call); ruinous under
DD/stepped, where the extension is entered once per time step.  On a production
615-tooth cascade that was ~382 MB per step and ~30k steps per iteration, and
on the staged path the buffer lands in HOST memory: 1760 s/iteration against
166 s once it binds ``p.u_last_two`` instead.

A value test cannot see this: the buffer is never read, so gradients were
always bit-exact.  What the defect changes is bytes allocated per call, so that
is what this pins.  ``allocated_bytes.all.allocated`` is a CUMULATIVE counter,
so the caching allocator reusing blocks does not hide it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_stepped_backward import (  # noqa: E402
    NT2D, NT3D, Harness, build, capture_backward, partitions_for,
    run_public_once,
)

if not torch.cuda.is_available():
    pytest.skip("boundary saving is impl='c' CUDA only", allow_module_level=True)


def _cum_alloc():
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocated_bytes.all.allocated"]


def _alloc_of(h, cuts):
    h.zero_state()
    a = _cum_alloc()
    h.replay_stepped(cuts)
    return _cum_alloc() - a


# ``bound`` is the measured fixed-tree baseline plus 1 x model_bytes -- the
# midpoint of the 2 x model_bytes the defect adds, so the pass and fail sides
# each keep a full model of margin.  Baselines measured on H100:
#   2D 3.02 x model   (bound 4.0, defect would be 5.02)
#   3D 2.00 x model   (bound 3.0, defect would be 4.00)
@pytest.mark.parametrize("ndim,nt,bound", [(2, NT2D, 4.0), (3, NT3D, 3.0)])
def test_stepped_backward_allocation_is_flat_in_segments(ndim, nt, bound):
    """Bytes allocated per backward call must not carry a whole wavefield.

    Replaying one reverse sweep as 2 segments and as ``nt`` segments is the
    same work; only the number of extension entries differs.  Extra bytes
    divided by extra calls is the per-call allocation.  The defect adds the
    ``last_two`` buffer -- 2 x model_bytes -- to every call, so it cannot fit
    under the bound below; the fixed path only allocates the small model-sized
    scratch the kernel genuinely needs.
    """
    # Order copied from run_case: capture has to happen before the forward
    # (the wrapper is read at forward time), and bs is the string "gpu", not a
    # dict.
    prop, wav, src, rec, models = build(ndim, bs="gpu", nt=nt)
    cap = capture_backward(prop)
    run_public_once(prop, wav, src, rec, models)
    h = Harness(cap, ndim, "bs")

    m = h.p.models[0]
    model_bytes = m.numel() * m.element_size()
    parts = partitions_for(nt)

    _alloc_of(h, parts["halves"])                      # warm every lazy buffer
    a = _alloc_of(h, parts["halves"])
    b = _alloc_of(h, parts["per_step"])
    n_extra = (len(parts["per_step"]) - 1) - (len(parts["halves"]) - 1)
    per_call = (b - a) / n_extra

    print(f"\n[{ndim}D nt={nt}] model {model_bytes/1e6:.2f} MB | per-call alloc "
          f"{per_call/1e6:.3f} MB = {per_call/model_bytes:.2f} x model "
          f"({n_extra} extra calls)")

    # The backward genuinely allocates per-call scratch (f_this, the CPML view,
    # the aux slabs).  The defect adds the last_two buffer on top -- EXACTLY
    # 2 x model_bytes more -- so the two paths sit either side of ``bound``.
    assert per_call < bound * model_bytes, (
        f"backward allocates {per_call/model_bytes:.2f} x model per call -- the "
        "last_two self-allocation is back (saver.cuh allocate_last_two)")

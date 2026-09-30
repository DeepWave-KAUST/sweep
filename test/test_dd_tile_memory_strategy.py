"""``ModelParallel`` must re-state the wrapped propagator's boundary strategy
using only the knobs that apply to the chosen storage.

The staging trio (transfer_interval / ring_buffers / pinned_memory) is
meaningless for ``storage='gpu'`` and ``BoundaryOptions.__post_init__`` rejects
it there. Re-emitting whatever was inherited therefore turned a configuration
the legacy dict route accepted into a construction error -- including the gpu
baseline of ``test/dd_session_bench.py`` at that file's own default, i.e. the
reference its bit-exactness gates compare against.

No GPU and no process group: the method reads five attributes off ``self``.
"""
import pytest

from sweep.parallel.dd_propagator import ModelParallel
from sweep.propagator.options import BoundarySaving


class _Inherited:
    """Stands in for a ModelParallel that inherited these knobs."""

    def __init__(self, storage, *, ti=64, ring=1, pinned=True, tail=0, dtype="fp32"):
        self._bstorage, self._bdtype = storage, dtype
        self._bti, self._bring, self._bpinned, self._btail = ti, ring, pinned, tail


def _strategy(storage, **kw):
    return ModelParallel._tile_memory_strategy(_Inherited(storage, **kw))


def test_gpu_storage_drops_the_staging_knobs():
    """The knobs that made this raise are simply not passed on."""
    s = _strategy("gpu")                      # inherits ti=64, ring=1, pinned=True
    assert isinstance(s, BoundarySaving) and s.storage == "gpu"
    assert s.transfer_interval is None and s.ring_buffers is None
    assert not s.pinned_memory


def test_cpu_storage_carries_all_three():
    s = _strategy("cpu", ti=8, ring=2, pinned=True)
    assert (s.transfer_interval, s.ring_buffers, s.pinned_memory) == (8, 2, True)


def test_disk_storage_carries_the_batching_but_not_pinned_memory():
    """BoundaryOptions rejects pinned_memory for disk as well."""
    s = _strategy("disk", ti=16, ring=2, pinned=True)
    assert (s.transfer_interval, s.ring_buffers) == (16, 2)
    assert not s.pinned_memory


@pytest.mark.parametrize("storage", ["gpu", "cpu", "disk"])
def test_tail_steps_survives_every_storage(storage):
    assert _strategy(storage, tail=120).tail_steps == 120
    assert _strategy(storage, tail=0).tail_steps is None

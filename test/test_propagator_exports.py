"""The current memory-strategy spellings are importable from ``sweep.propagator``.

The package exported only the deprecated ``MemoryOptions`` family, so the
spelling the docs recommend (``Full()`` / ``BoundarySaving(...)`` /
``Ckpt(...)``) needed the longer ``sweep.propagator.options`` path.
"""
import sweep.propagator as P
from sweep.propagator import options as O


def test_the_current_strategies_are_exported():
    for name in ("Full", "BoundarySaving", "Ckpt"):
        assert name in P.__all__
        assert getattr(P, name) is getattr(O, name)


def test_the_legacy_names_are_still_exported():
    for name in ("MemoryOptions", "BoundaryOptions", "CkptOptions", "CUDAOptions", "EagerOptions"):
        assert getattr(P, name) is getattr(O, name)

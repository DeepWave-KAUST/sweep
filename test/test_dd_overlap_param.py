"""``ModelParallel(prop, mesh, overlap=...)``: the comm/compute overlap of the
forward is a construction parameter, not an environment variable.

``SWEEP_DD_DISABLE_OVERLAP=1`` used to select the serial step-then-exchange
loop.  A switch between two schedules belongs in the config, where it is
recorded with the run and a misspelling is refused, so a set variable is an
error at construction, whatever its value, and a non-bool ``overlap`` is one
too.  Both are checked before any tile is built, so this needs no GPU and no
host core.  That the two schedules give the same record and gradient, and that
``overlap=False`` really takes the serial loop, is ``test_dd_two_rank_cpu.py``.
"""
import pytest

torch = pytest.importorskip("torch")

DEV = "cpu"


def _prop():
    from sweep.equations import Acoustic
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    return PropTorch(Acoustic(device=DEV, backend="torch"), impl="eager",
                     eager_options=EagerOptions(use_compile=False), use_ckpt=False,
                     dev=DEV, shape=(48, 56), abcn=16, dh=10.0, dt=8e-4,
                     free_surface=False, nt=50, B=1)


def _mesh():
    from sweep.parallel import MeshTopology

    return MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)


@pytest.mark.parametrize("value", ["1", "0", ""])   # "0" used to mean "keep the overlap"
def test_retired_env_is_refused(monkeypatch, value):
    from sweep.parallel import ModelParallel

    monkeypatch.setenv("SWEEP_DD_DISABLE_OVERLAP", value)
    with pytest.raises(ValueError, match=r"SWEEP_DD_DISABLE_OVERLAP is no longer read.*overlap=False"):
        ModelParallel(_prop(), _mesh())


@pytest.mark.parametrize("bad", ["0", 0, None])
def test_overlap_must_be_a_bool(monkeypatch, bad):
    from sweep.parallel import ModelParallel

    monkeypatch.delenv("SWEEP_DD_DISABLE_OVERLAP", raising=False)
    with pytest.raises(TypeError, match="overlap"):
        ModelParallel(_prop(), _mesh(), overlap=bad)

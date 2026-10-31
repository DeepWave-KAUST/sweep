"""Every new compile of the eager step is announced on stderr before it starts.

torch.compile compiles anew for a new grid size, a new shot count and the first call
that records gradients; each takes seconds to minutes and prints nothing, so each is
said once, and the first line of the process says how long it takes and how to turn
it off.
"""
import pytest
import torch

import sweep._compile as C
import sweep.propagator._torch_eager as TE
from sweep.equations import Acoustic, Elastic
from sweep.propagator import EagerOptions, Full, PropTorch

NT = 12


@pytest.fixture
def fake_compile(monkeypatch):
    """torch.compile that returns the step as it is: the announcements, not the compiler."""
    monkeypatch.setattr(C, "_ANNOUNCED", set())
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)


def _run(shape, nshots=1, grad=False, grad_mode=None):
    """One call; ``grad`` makes the model require grad and runs the backward,
    ``grad_mode`` sets grad mode (``grad`` when ``None``)."""
    p = PropTorch(Acoustic(), impl="eager", eager_options=EagerOptions(), memory=Full(),
                  shape=shape, dh=10.0, dt=1e-3, nt=NT, abcn=10, dev="cpu")
    w = torch.zeros(NT)
    w[2] = 1.0
    src = torch.tensor([[shape[1] // 2 + 2 * k, 5] for k in range(nshots)])
    rec = torch.tensor([[[5, 5], [shape[1] - 5, 5]]] * nshots)
    vp = torch.full(shape, 2000.0, requires_grad=grad)
    with torch.set_grad_enabled(grad if grad_mode is None else grad_mode):
        out = p(w, src, rec, models=[vp])
    if grad:
        out.square().sum().backward()


def _announced(capsys):
    return [ln for ln in capsys.readouterr().err.splitlines()
            if ln.startswith("sweep: torch.compile is compiling")]


def test_each_new_compile_is_announced_once(fake_compile, capsys):
    _run((30, 40))
    first = _announced(capsys)
    assert len(first) == 1, first
    assert "the Acoustic step for a 30x40 grid" in first[0] and "use_compile=False" in first[0]
    _run((30, 40))
    assert _announced(capsys) == [], "the same compile again: nothing new"
    _run((40, 50))
    grid = _announced(capsys)
    assert len(grid) == 1 and "the Acoustic step for a 40x50 grid" in grid[0]
    assert "use_compile=False" not in grid[0], "the explanation comes once per process"
    _run((30, 40), nshots=2)
    assert [ln for ln in _announced(capsys) if "2 shots on a 30x40 grid" in ln]
    _run((30, 40), grad=True)
    assert [ln for ln in _announced(capsys) if "a 30x40 grid (with gradients)" in ln]


def test_gradients_are_named_only_when_autograd_records(fake_compile, capsys):
    """Grad mode on and no model that requires grad compiles a graph without a
    backward; the first call with a model that does compiles again, and says so."""
    _run((30, 40), grad_mode=True)
    plain = _announced(capsys)
    assert len(plain) == 1 and "a 30x40 grid;" in plain[0], plain
    _run((30, 40), grad=True)
    grads = _announced(capsys)
    assert len(grads) == 1 and "a 30x40 grid (with gradients)" in grads[0], grads
    _run((30, 40), grad_mode=True)
    _run((30, 40))
    assert _announced(capsys) == [], "recording nothing, in grad mode or not, is one compile"


def test_a_new_set_of_models_that_require_grad_is_announced(fake_compile, capsys):
    """Dynamo specializes the step on which models require grad: inverting vp alone,
    then vp and vs, compiles twice, and each says so."""
    def fwi(*grads):
        p = PropTorch(Elastic(), impl="eager", eager_options=EagerOptions(), memory=Full(),
                      shape=(30, 40), dh=10.0, dt=1e-3, nt=NT, abcn=10, dev="cpu",
                      source_type=["sxx", "szz"], receiver_type=["vx", "vz"])
        w = torch.zeros(NT)
        w[2] = 1.0
        ms = [torch.full((30, 40), v, requires_grad=g) for v, g in zip((2000.0, 1100.0, 2000.0), grads)]
        p(w, torch.tensor([[20, 5]]), torch.tensor([[[5, 5], [35, 5]]]), models=ms).square().sum().backward()
    fwi(True, False, False)
    first = _announced(capsys)
    assert len(first) == 1 and "the Elastic step for a 30x40 grid (with gradients)" in first[0], first
    fwi(True, True, False)
    again = _announced(capsys)
    assert len(again) == 1 and "the Elastic step for a 30x40 grid (with gradients)" in again[0], again
    fwi(True, True, False)
    assert _announced(capsys) == []


def test_nothing_is_announced_uncompiled(fake_compile, capsys):
    p = PropTorch(Acoustic(), impl="eager", eager_options=EagerOptions(use_compile=False),
                  memory=Full(), shape=(30, 40), dh=10.0, dt=1e-3, nt=NT, abcn=10, dev="cpu")
    w = torch.zeros(NT)
    w[2] = 1.0
    with torch.no_grad():
        p(w, torch.tensor([[20, 5]]), torch.tensor([[[5, 5], [35, 5]]]),
          models=[torch.full((30, 40), 2000.0)])
    assert _announced(capsys) == []

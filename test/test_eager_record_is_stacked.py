"""The eager record must not be built by in-place slice writes -- under any
memory strategy.

Writing each step's gather into a live autograd tensor builds a chain of `nt`
`CopySlices` nodes, and every one of them allocates a full-record buffer and
copies the incoming gradient through it. The record's own backward cost is then
`2 * nt * |record|` -- 64 GB per shot at `nt=4000` with 500 receivers -- where
stacking once makes it `2 * |record|`. Boundary saving and chunk checkpointing
kept the in-place writes after the full-tape path was fixed.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import Ckpt
from sweep.propagator.torch import PropTorch

NZ, NX, NT = 24, 28, 40
MEMORY = {"full": Full, "ckpt": lambda: Ckpt(chunks=16), "bs": lambda: BoundarySaving(storage="gpu")}
# The node the record's graph ends in: one stack, or one cat of the chunks' stacks.
RECORD_NODE = {"full": "Stack", "ckpt": "Cat", "bs": "Stack"}


def _prop(memory):
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="eager", shape=(NZ, NX), dh=10.0, dt=1e-3, nt=NT,
                     abcn=6, B=1, dev="cpu", source_type=["h1"],
                     receiver_type=["h1"], memory=MEMORY[memory]())


def _shoot(memory="full", prop=None):
    prop = _prop(memory) if prop is None else prop
    src = np.array([[NX // 2, 4]], np.int64)
    rxx = np.arange(4, NX - 4, 3, np.int64)
    rec = np.stack([rxx, np.full(rxx.size, 4)], -1)[None]
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.006
    a = np.pi * 30.0 * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2)).astype(np.float32))
    vp = torch.full((NZ, NX), 2000.0, requires_grad=True)
    return prop(wav, src, rec, models=[vp]), vp


def _graph_node_names(root, limit=20000):
    seen, stack, names = set(), [root], []
    while stack and len(seen) < limit:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        for nxt, _ in getattr(node, "next_functions", ()):
            stack.append(nxt)
    return names


@pytest.mark.parametrize("memory", list(MEMORY))
def test_the_record_graph_has_no_copyslices_chain(memory):
    record, _ = _shoot(memory)
    names = _graph_node_names(record.grad_fn)
    copyslices = [n for n in names if "CopySlices" in n]
    assert not copyslices, (
        f"{len(copyslices)} CopySlices nodes on the record's graph; each one "
        "allocates and copies a full record during backward"
    )


@pytest.mark.parametrize("memory", list(MEMORY))
def test_the_record_is_a_stack(memory):
    record, _ = _shoot(memory)
    assert type(record.grad_fn).__name__.startswith(RECORD_NODE[memory])


@pytest.mark.parametrize("memory", list(MEMORY))
def test_the_gradient_still_flows(memory):
    record, vp = _shoot(memory)
    record.pow(2).mean().backward()
    assert vp.grad is not None and float(vp.grad.abs().max()) > 0


@pytest.mark.parametrize("memory", list(MEMORY))
def test_two_calls_do_not_alias(memory):
    """The stacked record is fresh, so it is returned without a defensive copy;
    a later forward of the same propagator must still not overwrite it."""
    prop = _prop(memory)
    first, _ = _shoot(prop=prop)
    baseline = first.detach().clone()
    second, _ = _shoot(prop=prop)
    second.detach().mul_(-3.0)
    assert torch.equal(first.detach(), baseline)

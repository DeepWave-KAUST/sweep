"""The eager record must not be built by in-place slice writes.

Writing each step's gather into a live autograd tensor builds a chain of `nt`
`CopySlices` nodes, and every one of them allocates a full-record buffer and
copies the incoming gradient through it. The record's own backward cost is then
`2 * nt * |record|` -- 64 GB per shot at `nt=4000` with 500 receivers -- where
stacking once makes it `2 * |record|`.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch

NZ, NX, NT = 24, 28, 40


def _shoot(**kw):
    prop = PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="eager", shape=(NZ, NX), dh=10.0, dt=1e-3, nt=NT,
                     abcn=6, B=1, dev="cpu", source_type=["h1"],
                     receiver_type=["h1"], **kw)
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


def test_the_record_graph_has_no_copyslices_chain():
    record, _ = _shoot(use_ckpt=False)
    names = _graph_node_names(record.grad_fn)
    copyslices = [n for n in names if "CopySlices" in n]
    assert not copyslices, (
        f"{len(copyslices)} CopySlices nodes on the record's graph; each one "
        "allocates and copies a full record during backward"
    )


def test_the_record_is_a_stack():
    record, _ = _shoot(use_ckpt=False)
    assert type(record.grad_fn).__name__.startswith("Stack")


def test_the_gradient_still_flows():
    record, vp = _shoot(use_ckpt=False)
    record.pow(2).mean().backward()
    assert vp.grad is not None and float(vp.grad.abs().max()) > 0


def test_two_calls_do_not_alias(monkeypatch):
    """The stacked record is fresh, so it is returned without a defensive copy;
    a later forward must still not overwrite it."""
    first, _ = _shoot(use_ckpt=False)
    baseline = first.detach().clone()
    _shoot(use_ckpt=False)
    assert torch.equal(first.detach(), baseline)

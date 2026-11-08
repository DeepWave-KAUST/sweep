"""Several forwards on one propagator before their backwards.

``u0 = s(...); u1 = s(...); (L0 + L1).backward()`` must give what each forward
followed by its own backward gives. On impl='c' it did not: the boundary
strips, ``last_two`` and the checkpoint snapshots live on the propagator, the
second forward rewrote them through raw pointers, and the first graph's
backward reconstructed shot 0 from shot 1's boundary -- with no error and no
warning on every boundary storage (gpu, cpu, int8). Only the paths where Python
happened to ``zero_()`` a tensor autograd had saved (checkpointing, disk
storage) raised, with "modified by an inplace operation", which names nothing.

The persistent storage is now leased to the forward whose backward will read it
(``_c._BufferLease``); a forward that finds it leased runs on storage of its
own. ``ModelParallel`` binds one set of buffers into its capture, so it refuses
the second forward instead.

A forward that saves nothing for a backward (``torch.no_grad()``) must not touch
that storage at all: on disk storage it zeroed the host staging and the
``last_two`` of the backward still pending, which then raised.

Backward ORDER is varied on purpose: the backward of the EARLIER forward is the
one that read the overwritten storage, whichever runs first.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from conftest import requires_binding, ricker

from sweep.equations import Acoustic, Acoustic3D, Elastic
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch

pytestmark = requires_binding("acoustic2d_forward")

DEV = torch.device("cuda:0") if torch.cuda.is_available() else None
# The defect is O(1) relative; the nondeterministic floors of the compiled
# backward (recursive checkpointing, int8 storage, 3-D atomics) reach ~1e-4.
TOL = 1e-3


def _setup(ndim):
    """Initial model + observed data from a model with a box anomaly, two shots
    far apart, so the two calls' boundary data differ everywhere."""
    if ndim == 2:
        shape, nt, abcn = (60, 80), 300, 12
        z = np.arange(shape[0], dtype=np.float32)[:, None] * np.ones((1, shape[1]), np.float32)
        init = 1800 + 8 * z
        true = init.copy()
        true[28:38, 30:50] += 250.0
        xs = np.arange(4, shape[1] - 4, 3)
        rec = np.stack([xs, np.full_like(xs, 4)], -1)[None]
        srcs = [np.array([[16, 5]]), np.array([[64, 5]])]
    else:
        shape, nt, abcn = (28, 24, 32), 200, 8
        z = np.arange(shape[0], dtype=np.float32)[:, None, None] * np.ones((1,) + shape[1:], np.float32)
        init = 1800 + 15 * z
        true = init.copy()
        true[12:18, 8:16, 10:22] += 250.0
        xs = np.arange(3, shape[2] - 3, 3)
        rec = np.stack([xs, np.full_like(xs, shape[1] // 2), np.full_like(xs, 3)], -1)[None]
        srcs = [np.array([[7, shape[1] // 2, 4]]), np.array([[25, shape[1] // 2, 4]])]
    return shape, nt, abcn, init, true, rec, srcs


class _Problem:
    def __init__(self, ndim=2, memory=None, elastic=False, **kw):
        shape, nt, abcn, init, true, rec, srcs = _setup(ndim)
        dt = 1e-3
        if elastic:
            eq = Elastic(spatial_order=4, device=DEV, backend="torch")
            vp0 = 1.2 * init
            self.init = [vp0, vp0 / 1.8, np.full(shape, 2000.0, np.float32)]
            self.true = [vp0 + (true - init), vp0 / 1.8, self.init[2]]
        else:
            eq = (Acoustic if ndim == 2 else Acoustic3D)(spatial_order=4, device=DEV, backend="torch")
            self.init, self.true = [init], [true]
        self.s = PropTorch(eq, backend="torch", impl="c", shape=shape, dh=10.0, dt=dt, nt=nt,
                           abcn=abcn, dev=DEV, memory=memory, **kw)
        assert self.s.impl == "c"
        self.wav = torch.tensor(ricker(nt, dt, fm=15.0, delay=0.07), device=DEV)
        self.rec, self.srcs = rec, srcs
        with torch.no_grad():
            models = self.true_models()
            self.obs = [self.s(self.wav, src, rec, models=models).clone() for src in srcs]

    def true_models(self):
        return [torch.tensor(m, device=DEV) for m in self.true]

    def leaves(self):
        return [torch.tensor(m, device=DEV, requires_grad=True) for m in self.init]

    def loss(self, models, i, **kw):
        return 0.5 * (self.s(self.wav, self.srcs[i], self.rec, models=models, **kw) - self.obs[i]).pow(2).sum()

    @staticmethod
    def grads(models):
        return [m.grad.detach().double().clone() for m in models]

    def per_call(self):
        """Forward -> backward, forward -> backward: always right."""
        models = self.leaves()
        self.loss(models, 0).backward()
        self.loss(models, 1).backward()
        return self.grads(models)


def _rel(got, ref):
    return max(float((g - r).norm() / r.norm()) for g, r in zip(got, ref))


MODES = ("full", "bs_gpu", "bs_gpu_int8", "bs_cpu", "bs_cpu_int8", "bs_disk", "bs_disk_int8",
         "ckpt_chunk", "ckpt_recursive")


def _memory(mode, tmp_path):
    return {
        "full": Full(),
        "bs_gpu": BoundarySaving(),
        "bs_gpu_int8": BoundarySaving(storage_dtype="int8"),
        "bs_cpu": BoundarySaving(storage="cpu"),
        "bs_cpu_int8": BoundarySaving(storage="cpu", storage_dtype="int8"),
        "bs_disk": BoundarySaving(storage="disk", disk_dir=str(tmp_path)),
        "bs_disk_int8": BoundarySaving(storage="disk", disk_dir=str(tmp_path), storage_dtype="int8"),
        "ckpt_chunk": Ckpt(mode="chunk", chunks=40),
        "ckpt_recursive": Ckpt(mode="recursive", count=5),
    }[mode]


@pytest.mark.parametrize("mode", MODES)
def test_c_two_forwards_then_backwards_match_per_call(mode, tmp_path):
    p = _Problem(2, _memory(mode, tmp_path))
    ref = p.per_call()
    assert all(float(r.abs().max()) > 0 for r in ref), "degenerate setup: zero reference gradient"

    models = p.leaves()
    (p.loss(models, 0) + p.loss(models, 1)).backward()
    summed = _rel(p.grads(models), ref)

    models = p.leaves()
    l0, l1 = p.loss(models, 0), p.loss(models, 1)
    l1.backward()
    l0.backward()                         # the earlier forward's backward runs last
    reversed_order = _rel(p.grads(models), ref)

    assert summed < TOL and reversed_order < TOL, (
        f"{mode}: two forwards then backward differ from forward->backward per call: "
        f"relL2 {summed:.3e} ((L0+L1).backward()), {reversed_order:.3e} (L1 then L0). "
        "A forward rewrote storage a pending backward still had to read.")


@pytest.mark.parametrize("case", ["acoustic3d_bs_gpu", "elastic2d_bs_gpu", "elastic2d_ckpt"])
def test_c_two_forwards_other_equations(case):
    if case == "acoustic3d_bs_gpu":
        p = _Problem(3, BoundarySaving())
    elif case == "elastic2d_bs_gpu":
        p = _Problem(2, BoundarySaving(), elastic=True)
    else:
        p = _Problem(2, Ckpt(mode="chunk", chunks=40), elastic=True)
    ref = p.per_call()
    models = p.leaves()
    (p.loss(models, 0) + p.loss(models, 1)).backward()
    err = _rel(p.grads(models), ref)
    assert err < TOL, f"{case}: two forwards then one backward, relL2 {err:.3e} against per call"


@pytest.mark.parametrize("mode", MODES)
def test_c_no_grad_forward_before_the_backward(mode, tmp_path):
    """A forward under no_grad between a gradient forward and its backward saves
    nothing for a backward, so it must leave the pending backward's storage
    alone -- and its own record must not depend on that pending state."""
    p = _Problem(2, _memory(mode, tmp_path))
    models = p.leaves()
    p.loss(models, 0).backward()
    ref = p.grads(models)

    models = p.leaves()
    l0 = p.loss(models, 0)
    with torch.no_grad():
        rec1 = p.s(p.wav, p.srcs[1], p.rec, models=p.true_models())
    l0.backward()
    err = _rel(p.grads(models), ref)
    assert err < TOL, (f"{mode}: a no_grad forward before the backward moved the gradient, "
                       f"relL2 {err:.3e}")
    assert torch.equal(rec1, p.obs[1]), "the no_grad record depends on the pending backward's state"


def test_c_full_forward_before_a_boundary_saving_backward():
    """A Full() gradient call zeroes the propagator's persistent forward state
    on every call. The GPU ``last_two`` -- the seed a boundary-saving backward
    reconstructs from -- must not live there, or a full-mode call while that
    backward is pending wipes it."""
    p = _Problem(2, BoundarySaving())
    ref = p.per_call()
    models = p.leaves()
    l0 = p.loss(models, 0)                                  # boundary saving
    l1 = p.loss(models, 1, use_boundary_saving=False)       # full, this call only
    (l0 + l1).backward()
    err = _rel(p.grads(models), ref)
    assert err < TOL, f"a full-mode forward wiped the pending boundary-saving backward: relL2 {err:.3e}"


def test_c_private_store_only_while_a_backward_is_pending(monkeypatch):
    """The fix costs nothing on the forward->backward path, and the lease ends
    with the backward -- or with the graph, when there is no backward."""
    p = _Problem(2, BoundarySaving())
    impl = p.s._backend_impl
    stores = []
    orig = impl._allocate_boundary_store
    monkeypatch.setattr(impl, "_allocate_boundary_store",
                        lambda *a, **k: stores.append(1) or orig(*a, **k))

    models = p.leaves()
    for i in (0, 1, 0):
        p.loss(models, i).backward()
    assert len(stores) == 1, "forward -> backward allocated more than the persistent store"
    persistent = impl.boundary_gpu_full

    l0, l1 = p.loss(models, 0), p.loss(models, 1)
    assert len(stores) == 2, "the second pending forward must run on a store of its own"
    (l0 + l1).backward()
    p.loss(models, 0).backward()
    assert len(stores) == 2, "the lease was not released by the backward"

    l0 = p.loss(models, 0)
    del l0                                # graph dropped without a backward
    p.loss(models, 1).backward()
    assert len(stores) == 2, "the lease was not released when the graph was freed"
    assert impl.boundary_gpu_full is persistent, "the persistent store was reallocated"

    with torch.no_grad():                 # saves nothing: no store, no lease
        p.s(p.wav, p.srcs[0], p.rec, models=p.true_models())
    l0 = p.loss(models, 0)
    assert len(stores) == 2, "a no_grad forward took the lease"
    l0.backward()


def test_c_failed_forward_does_not_hold_the_store(monkeypatch):
    """A forward that raises leaves no backward behind, so it must not leave the
    persistent store leased -- not even while its traceback, which holds the
    call's frames, is kept alive (a notebook keeps the last one). Otherwise
    every later call would pay for a private store on top of the idle
    persistent one: an OOM retried with a smaller batch could OOM again."""
    p = _Problem(2, BoundarySaving())
    impl = p.s._backend_impl
    models = p.leaves()
    p.loss(models, 0).backward()                 # allocates the persistent store
    stores = []
    orig = impl._allocate_boundary_store
    monkeypatch.setattr(impl, "_allocate_boundary_store",
                        lambda *a, **k: stores.append(1) or orig(*a, **k))
    forward_func = impl.forward_func

    def fail(params):
        raise RuntimeError("simulated failure inside the compiled forward")

    impl.forward_func = fail
    try:
        p.loss(models, 1)
    except RuntimeError as exc:
        kept = exc                               # keeps the traceback and its frames
    finally:
        impl.forward_func = forward_func
    assert "simulated failure" in str(kept)
    p.loss(models, 0).backward()
    assert not stores, "a failed forward left the persistent store leased"


def test_c_private_disk_store_is_removed_with_its_graph(tmp_path):
    p = _Problem(2, BoundarySaving(storage="disk", disk_dir=str(tmp_path)))

    def roots():
        return sorted(d for d in os.listdir(tmp_path) if d.startswith("sweep_boundary_"))

    models = p.leaves()
    p.loss(models, 0).backward()
    persistent = roots()
    assert len(persistent) == 1

    l0, l1 = p.loss(models, 0), p.loss(models, 1)
    assert len(roots()) == 2, "the pending second forward should write files of its own"
    l1.backward()
    assert roots() == persistent, "the private files outlived their backward"
    l0.backward()
    assert roots() == persistent

    l0, l1 = p.loss(models, 0), p.loss(models, 1)
    assert len(roots()) == 2
    del l1                                # graph dropped without a backward
    assert roots() == persistent, "the private files outlived their graph"
    l0.backward()


def _model_parallel():
    from sweep.parallel import MeshTopology, ModelParallel

    shape, nt, abcn, init, true, rec, srcs = _setup(2)
    dt = 1e-3
    prop = PropTorch(Acoustic(spatial_order=4, device=DEV, backend="torch"), backend="torch",
                     impl="c", shape=shape, dh=10.0, dt=dt, nt=nt, abcn=abcn, dev=DEV, B=1,
                     memory=BoundarySaving())
    ddp = ModelParallel(prop, MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0))
    wav = ricker(nt, dt, fm=15.0, delay=0.07)
    srcs = [s[None] for s in srcs]
    with torch.no_grad():
        obs = [ddp(wav, s, rec, models=[torch.tensor(true, device=DEV)]).clone() for s in srcs]

    def loss(vp, i):
        return 0.5 * (ddp(wav, srcs[i], rec, models=[vp]) - obs[i]).pow(2).sum()

    def leaf():
        return torch.tensor(init, device=DEV, requires_grad=True)

    return loss, leaf


def test_dd_refuses_a_forward_while_a_backward_is_pending():
    """ModelParallel keeps one ring for its last forward: a second forward --
    no_grad included, the capture froze boundary saving on -- gave the pending
    backward a wrong gradient with no error. It must refuse, and the instance
    must still work afterwards."""
    loss, leaf = _model_parallel()
    vp = leaf()
    loss(vp, 0).backward()
    ref = vp.grad.detach().clone()

    vp = leaf()
    l0 = loss(vp, 0)
    with pytest.raises(RuntimeError, match="has not run yet"):
        loss(vp, 1)
    with pytest.raises(RuntimeError, match="has not run yet"), torch.no_grad():
        loss(vp, 1)
    l0.backward()
    assert torch.equal(vp.grad, ref), "the refused calls disturbed the pending backward"

    vp = leaf()
    loss(vp, 1).backward()                # the lease ended with the backward
    l0 = loss(leaf(), 0)
    del l0                                # ... or with the graph
    vp = leaf()
    loss(vp, 0).backward()
    assert torch.equal(vp.grad, ref)

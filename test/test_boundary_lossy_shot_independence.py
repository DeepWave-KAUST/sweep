"""Lossy boundary storage (int8 / scaled fp16) must keep the shots independent.

The scaled store quantizes each saved boundary band in blocks of 256 cells, one
FP32 max-abs scale per block.  The blocks used to be cut out of a timestep's band
for the WHOLE batch, shot after shot, so the block holding the end of one shot
also held the start of the next: shot b's codes, and so its gradient, depended on
the other shots in the batch, most of all on a much louder one.  The blocks are
now cut per shot, and only the call's own shots are quantized.  Two consequences
are pinned here:

* shot 0's gradient run alone (B=1) equals its gradient inside a B=2 batch whose
  other shot is 1000x louder, bit for bit.  The loss reads shot 0's record only,
  so shot 1's adjoint source is exactly zero and so is its contribution to the
  model gradient: with fp32 storage the two are bit-identical (the fp32 case
  below), and lossy storage must not change that;
* a propagator that ran a larger batch first gives exactly what a fresh one gives
  on a smaller batch.  Its boundary buffers keep the larger (capacity) batch and
  are narrowed per call; 3-D used to stop with an error that blamed tangent_pad,
  and 2-D quantized the stale staging of the shots it no longer had.

Both need a grid whose per-shot band is not a multiple of 256 cells on some face
(on an aligned grid no block ever straddled two shots), which
``_assert_unaligned`` checks, and lossy storage that really differs from fp32
(``_assert_lossy``); otherwise the bitwise comparisons prove nothing.

The host (CPU) core runs the same savers with its own quantize loops, cut per
shot the same way; it is checked in 2-D with on-device storage (it refuses host
staging).

The staged copies run bounds-checked (``SWEEP_BOUNDARY_BOUNDS``): after a larger
batch they copy inside buffers narrowed from it, which the check used to measure
by their numel() and reject.

The eager backend lays its saved ring out per shot as well; its tests drive the
ring store directly, on the CPU.
"""

import os

import numpy as np
import pytest
import torch

# Read once into a static on the first staged copy, so set before any runs.
os.environ.setdefault("SWEEP_BOUNDARY_BOUNDS", "1")

from sweep.equations import Acoustic, Acoustic3D  # noqa: E402
from sweep.propagator._eager_boundary_saving import (  # noqa: E402
    ReconState,
    _put_ring,
    _ring_index_tuples,
    _ring_row,
    load_steps,
    restore_step,
    save_step,
    save_steps,
)
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.signal import ricker  # noqa: E402


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="impl='c' requires CUDA."
)


def _host_core():
    if os.environ.get("SWEEP_TEST_HOST_CORE", "1").lower() in ("0", "false", "no", "off"):
        return False
    from sweep.backend.c import hostcore
    return hostcore.available()


host_only = pytest.mark.skipif(not _host_core(), reason="no host (CPU) core at hand")


def _cases(storages):
    """(ndim, device, storage): impl='c' on CUDA with every storage, and on the
    host core in 2-D with on-device storage (``'gpu'``: the compute device's
    memory).  The host quantize loops do not depend on the dimension, and the 3-D
    cases take minutes on a CPU runner."""
    return ([pytest.param(n, "cuda", s, marks=cuda_only, id=f"{n}-cuda-{s}")
             for n in (2, 3) for s in storages]
            + [pytest.param(2, "cpu", "gpu", marks=host_only, id="2-cpu-gpu")])

_DT = 1.5e-3
_BLOCK = 256       # cells per FP32 scale (BOUNDARY_INT8_BLOCK)
_LOUD = 1000.0     # amplitude of the other shot in the batch


def _wavelet(nt, f=9.0):
    t = np.arange(nt, dtype=np.float32) * _DT
    return ricker(t - 0.12, f=f).astype(np.float32)


def _problem(ndim):
    """Two shots over a small acoustic model; neither grid is 256-aligned."""
    if ndim == 2:
        nz, nx = shape = (90, 110)
        z = np.arange(nz, dtype=np.float32)[:, None]
        vp0 = (1500.0 + 8.0 * z).repeat(nx, 1).astype(np.float32)
        vpt = vp0.copy()
        vpt[nz // 3:, nx // 3: 2 * nx // 3] += 300.0
        src = np.array([[int(nx * 0.3), 3], [int(nx * 0.7), 3]], dtype=np.int64)
        rx = np.linspace(2, nx - 3, 32).astype(np.int64)
        rec = np.stack([rx, np.full(rx.size, 4, dtype=np.int64)], axis=1)
        return dict(ndim=2, eq=Acoustic, shape=shape, dh=15.0, nt=500,
                    vp0=vp0, vpt=vpt, src=src, rec=rec)
    nz, ny, nx = shape = (26, 22, 30)
    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp0 = (1800.0 + 12.0 * z).repeat(ny, 1).repeat(nx, 2).astype(np.float32)
    vpt = vp0.copy()
    vpt[nz // 3:, :, :] += 300.0
    src = np.array([[int(nx * 0.3), ny // 2, 3], [int(nx * 0.7), ny // 2, 3]], dtype=np.int64)
    rx, ry = np.meshgrid(np.linspace(2, nx - 3, 5).astype(np.int64),
                         np.linspace(2, ny - 3, 5).astype(np.int64), indexing="ij")
    rec = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, 3, dtype=np.int64)], axis=1)
    return dict(ndim=3, eq=Acoustic3D, shape=shape, dh=20.0, nt=250,
                vp0=vp0, vpt=vpt, src=src, rec=rec)


def _solver(pb, dtype, storage="gpu", dev="cuda", **config):
    config = {"enabled": True, "storage": storage, "storage_dtype": dtype, **config}
    solver = PropTorch(pb["eq"](spatial_order=4, device=torch.device(dev), backend="torch"),
                       shape=pb["shape"], dh=pb["dh"], dt=_DT, impl="c", dev=dev,
                       boundary_saving_config=config)
    assert solver.impl == "c"
    return solver


_OBS = {}


def _observed_shot0(pb, dev):
    if (pb["ndim"], dev) not in _OBS:
        with torch.no_grad():
            d = _solver(pb, "fp32", dev=dev)(_wavelet(pb["nt"])[None], pb["src"][:1], pb["rec"][None],
                                             models=[torch.tensor(pb["vpt"], device=dev)])
        _OBS[pb["ndim"], dev] = d[0].detach().clone()
    return _OBS[pb["ndim"], dev]


def _shots(pb, batch, loud):
    """Wavelets, sources and receivers of the first ``batch`` shots; every shot
    after the first is ``loud`` times louder."""
    w = np.repeat(_wavelet(pb["nt"])[None], batch, axis=0)
    w[1:] *= loud
    return w, pb["src"][:batch], np.repeat(pb["rec"][None], batch, axis=0)


def _grad_shot0(solver, pb, batch, loud=1.0, dev="cuda"):
    """d/d(vp) of shot 0's misfit alone, run inside a batch of ``batch`` shots."""
    vp = torch.tensor(pb["vp0"], device=dev, requires_grad=True)
    d = solver(*_shots(pb, batch, loud), models=[vp])
    (d[0] - _observed_shot0(pb, dev)).pow(2).sum().backward()
    return vp.grad.detach().cpu().numpy()


def _forward_loud(solver, pb, dev="cuda"):
    """A B=2 forward with boundary saving and no backward: it leaves the last
    step's band of a loud shot 1 in the staging."""
    vp = torch.tensor(pb["vp0"], device=dev, requires_grad=True)
    solver(*_shots(pb, 2, _LOUD), models=[vp])


def _assert_unaligned(solver):
    """Some face's per-shot band is not whole blocks, so a block cut across the
    batch would hold two shots."""
    impl = solver._backend_impl
    lead = 2 if impl.ndim == 3 else 3          # (1, B, ...) in 3-D, (1, 1, B, ...) in 2-D
    cells = [int(np.prod(t.shape[lead:])) for t in impl.boundary_staging]
    assert cells and any(c % _BLOCK for c in cells), cells


def _assert_lossy(lossy, exact):
    assert np.abs(lossy).max() > 0
    assert not np.array_equal(lossy, exact), "the lossy storage stored nothing lossily"


def _rel(ref, got):
    return f"max|diff| / max|ref| = {np.abs(ref - got).max() / np.abs(ref).max():.3e}"


@pytest.mark.parametrize("dtype", ["fp32", "int8", "fp16"])
@pytest.mark.parametrize("ndim,dev,storage", _cases(["gpu", "cpu", "disk"]))
def test_shot_gradient_does_not_depend_on_the_batch(ndim, dtype, dev, storage):
    pb = _problem(ndim)
    solver = _solver(pb, dtype, storage, dev)
    alone = _grad_shot0(solver, pb, 1, dev=dev)
    if dtype != "fp32":
        _assert_unaligned(solver)
        _assert_lossy(alone, _grad_shot0(_solver(pb, "fp32", storage, dev), pb, 1, dev=dev))
    batched = _grad_shot0(_solver(pb, dtype, storage, dev), pb, 2, loud=_LOUD, dev=dev)
    assert np.array_equal(alone, batched), _rel(alone, batched)


@pytest.mark.parametrize("dtype", ["int8", "fp16"])
@pytest.mark.parametrize("ndim,dev,storage", _cases(["gpu", "cpu", "disk"]))
def test_smaller_batch_after_a_larger_one(ndim, dtype, dev, storage):
    """B=2, then B=1 on the same propagator, against a fresh propagator's B=1."""
    pb = _problem(ndim)
    solver = _solver(pb, dtype, storage, dev)
    _forward_loud(solver, pb, dev)
    after = _grad_shot0(solver, pb, 1, dev=dev)
    fresh = _grad_shot0(_solver(pb, dtype, storage, dev), pb, 1, dev=dev)
    assert np.array_equal(after, fresh), _rel(fresh, after)


@cuda_only
def test_smaller_batch_after_a_tail_steps_backward():
    """The same after an ordinary forward + backward with boundary tail
    truncation, whose last dequantize leaves a live band in the staging (the
    first saved step is not the silent step 0)."""
    pb = _problem(2)
    tail = pb["nt"] // 2
    solver = _solver(pb, "int8", tail_steps=tail)
    _grad_shot0(solver, pb, 2)
    after = _grad_shot0(solver, pb, 1)
    fresh = _grad_shot0(_solver(pb, "int8", tail_steps=tail), pb, 1)
    assert np.array_equal(after, fresh), _rel(fresh, after)


# ---- eager ring store --------------------------------------------------------

_EAGER_SHAPE = (37, 45)       # padded field; the ring is not whole blocks per shot
_EAGER_OFFSETS = (6, 6)
_EAGER_HALO = 2


def _eager_round_trip(fields, dtype):
    """Save one step of ``fields`` (each (B, 1, nz, nx)) through the eager ring
    store and restore it into zeroed fields."""
    shape = tuple(fields[0].shape)
    rings = _ring_index_tuples(shape, 2, _EAGER_OFFSETS, _EAGER_HALO)
    state = ReconState(1, rings, shape, len(fields), dtype, "cpu")
    save_step(state, 0, fields)
    return [restore_step(state, 0, k, torch.zeros_like(f)) for k, f in enumerate(fields)], rings


def _eager_fields(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn((batch, 1) + _EAGER_SHAPE, generator=g) for _ in range(2)]


@pytest.mark.parametrize("dtype", ["int8", "fp16"])
def test_eager_ring_store_keeps_shots_apart(dtype):
    alone = _eager_fields(1)
    loud = _eager_fields(1, seed=1)
    batch = [torch.cat([a, _LOUD * b]) for a, b in zip(alone, loud)]
    got_alone, rings = _eager_round_trip(alone, dtype)
    got_batch, _ = _eager_round_trip(batch, dtype)
    per_shot = sum(int(np.prod(alone[0][idx].shape)) for idx in rings)
    assert per_shot % _BLOCK, per_shot
    assert not torch.equal(got_alone[0][rings[0]], alone[0][rings[0]])
    for a, b in zip(got_alone, got_batch):
        assert torch.equal(a[0], b[0])


@pytest.mark.parametrize("dtype,rel", [("fp32", 0.0), ("bf16", 2.0 ** -8),
                                       ("fp16", 2.0 ** -10), ("int8", 1.0 / 127)])
def test_eager_ring_store_round_trip(dtype, rel):
    """Every shot's ring comes back to that shot, exactly for fp32 and otherwise
    within the storage precision of that shot's own amplitude (the shots are
    10x apart, so a misplaced or shared block would be far outside it); the
    cells off the ring stay untouched."""
    fields = [f * torch.tensor([1.0, 10.0, 100.0]).view(3, 1, 1, 1) for f in _eager_fields(3)]
    restored, rings = _eager_round_trip(fields, dtype)
    for f, r in zip(fields, restored):
        on_ring = torch.zeros_like(f, dtype=torch.bool)
        for idx in rings:
            on_ring[idx] = True
        assert not r[~on_ring].any()
        for b in range(f.shape[0]):
            peak = f[b][on_ring[b]].abs().max()
            err = (r[b] - f[b])[on_ring[b]].abs().max()
            assert err <= rel * peak, (b, float(err), float(peak))


def _eager_steps_round_trip(steps, dtype):
    """Save ``steps`` (a list of steps, each a list of fields (B, 1, nz, nx))
    the way the k-step rollout does (one ``_ring_row`` per step, ``save_steps``)
    and restore each step's fields from ``load_steps`` with ``_put_ring``."""
    shape = tuple(steps[0][0].shape)
    rings = _ring_index_tuples(shape, 2, _EAGER_OFFSETS, _EAGER_HALO)
    state = ReconState(len(steps), rings, shape, len(steps[0]), dtype, "cpu")
    save_steps(state, 0, torch.stack([_ring_row(fields, rings, state.pad) for fields in steps]))
    rows = load_steps(state, list(range(len(steps))))
    specs = tuple(zip(rings, state.ring_shapes, state.ring_sizes))
    return [[_put_ring(torch.zeros_like(f), rows[j], k, state.stride, specs)
             for k, f in enumerate(fields)] for j, fields in enumerate(steps)]


@pytest.mark.parametrize("dtype", ["int8", "fp16"])
def test_eager_ring_steps_keep_shots_apart(dtype):
    """The k-step path (``_BoundarySaveSteps``) cuts its blocks per shot too:
    shot 0 comes back as it does alone, next to a 1000x louder shot, and each
    step as the one-step store (``save_step`` / ``restore_step``) gives it."""
    alone = [_eager_fields(1, seed=s) for s in range(3)]
    batch = [[torch.cat([a, _LOUD * b]) for a, b in zip(fields, _eager_fields(1, seed=10 + j))]
             for j, fields in enumerate(alone)]
    got_alone = _eager_steps_round_trip(alone, dtype)
    got_batch = _eager_steps_round_trip(batch, dtype)
    for step_alone, step_batch, fields in zip(got_alone, got_batch, batch):
        one_step, _ = _eager_round_trip(fields, dtype)
        for a, b, c in zip(step_alone, step_batch, one_step):
            assert torch.equal(a[0], b[0])
            assert torch.equal(b, c)

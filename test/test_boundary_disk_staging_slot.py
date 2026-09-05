"""Staged (disk / cpu) boundary saving must reconstruct the SAME gradient as
gpu-direct -- bit-for-bit, on the synchronous paths.

Regression test for the PR #81 staging defect (848a100 + 76505cc):

1. ``prefetch_next_backward_chunk_if_needed`` issued the next chunk's H2D early
   whenever ``ring_buffers >= 2``, but ``backward_slot_for_chunk`` pins the
   synchronous DISK path to slot 0 regardless of ``ring_buffers`` -- and that
   path DEFAULTS to ring 3 (2-D) / 2 (3-D).  The early copy overwrote the slot
   the current chunk was still restoring from.  On the unfixed tree this file
   sees max|disk-gpu|/scale = 0.18 / 0.22 (Acoustic 2-D / 3-D, default knobs),
   0.22 / 0.52 with short chunks, and 0.8-5.0 on Elastic3D / DASMu3D.
2. ``copy_stream`` became non-blocking in 76505cc, but the synchronous-disk
   enqueue in ``prefetch_backward_chunk`` never got the matching
   ``cudaStreamWaitEvent(compute_ready_[slot])`` write-after-read fence.
   nvar>1 3-D (elastic3d, das_mu3d) lost that race deterministically:
   the multi-field H2D is large enough to still be landing while the
   restore kernels read, and #1 alone does not explain the 1.2-5.0x errors.
3. The two nvar>1 restore readers kept the slot-0 override for ALL non-async
   staging, so cpu staging with ``ring_buffers >= 2`` read slot 0 while the
   prefetch landed in slot k.  Latent on defaults (cpu ring = 1); with ring 2
   the unfixed tree is off by 1.4-2.4x on Elastic3D / DASMu3D.

Why bit-exact and not a tolerance: the compiled path is reproducible on one
GPU (run twice, identical bits), and gpu-direct / cpu-staged / sync-disk-staged
all execute the same restore kernels on the same numbers -- only WHERE the
boundary bytes waited differs.  A tolerance would have hidden #1 behind
"chunking noise"; it is corruption, and the assertion says so.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")

DEV = "cuda"


def _models_2d(nz, nx, elastic):
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :] += 300.0
    if not elastic:
        return [vp]
    return [vp, (vp / 1.73).astype(np.float32), np.full((nz, nx), 1900.0, np.float32)]


def _models_3d(nz, ny, nx, elastic):
    vp = np.full((nz, ny, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :, :] += 300.0
    if not elastic:
        return [vp]
    return [vp, (vp / 1.73).astype(np.float32), np.full((nz, ny, nx), 1900.0, np.float32)]


def _gradients(eqname, storage, disk_dir, *, transfer_interval=None, ring_buffers=None):
    """Gradients w.r.t. every model of one propagation, with the boundary held
    in ``storage``.  ``None`` knobs mean "whatever the resolver picks", which is
    exactly what a user who only writes ``storage='disk'`` gets."""
    import sweep.equations as E
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    cls = getattr(E, eqname)
    elastic = eqname.startswith(("Elastic", "DASMu"))
    nt, dt = 100, 0.0015
    if eqname.endswith("3D"):
        nz, ny, nx = 24, 20, 24
        shape = (nz, ny, nx)
        models = _models_3d(nz, ny, nx, elastic)
        src = np.array([[nx // 2, ny // 2 + 3, nz // 4]], dtype=np.int32)
        rx, ry = np.meshgrid(np.arange(2, nx - 2, 6), np.arange(2, ny - 2, 6))
        rx, ry = rx.ravel(), ry.ravel()
        rec = np.stack([rx, ry, np.full(rx.size, 2)], -1).astype(np.int32)[None]
        dh = (10.0, 10.0, 10.0)
    else:
        nz, nx = 40, 48
        shape = (nz, nx)
        models = _models_2d(nz, nx, elastic)
        src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
        rx = np.arange(2, nx - 2, 4, dtype=np.int32)
        rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
        dh = (10.0, 10.0)
    wav = torch.zeros(nt, device=DEV)
    wav[5] = 1.0

    kw = dict(storage=storage)
    if storage != "gpu":
        if transfer_interval is not None:
            kw["transfer_interval"] = transfer_interval
        if ring_buffers is not None:
            kw["ring_buffers"] = ring_buffers
    if storage == "disk":
        # the saver creates its files INSIDE disk_dir but does not create the
        # directory itself (tier B's build_cuda_options mkdirs it too)
        disk_dir.mkdir(parents=True, exist_ok=True)
        kw["disk_dir"] = str(disk_dir)
    co = CUDAOptions(memory=MemoryOptions(strategy="boundary", boundary=BoundaryOptions(**kw)))
    eq = cls(spatial_order=4, device=DEV, backend="torch")
    # no pml_type: let each equation pick its own (acoustic -> cpmlr, the
    # staggered family -> cpmls; forcing cpmlr on Elastic3D segfaults in the
    # forward on every tree, which is a test bug, not the defect under test)
    prop = PropTorch(eq, backend="torch", impl="c", cuda_options=co, shape=shape, dev=DEV,
                     dh=dh, dt=dt, abcn=15, nt=nt, B=1, allow_growth=True)
    ms = [torch.tensor(m, device=DEV, requires_grad=True) for m in models]
    prop(wav, src.copy(), rec.copy(), models=ms).pow(2).mean().backward()
    return [m.grad.detach().clone() for m in ms]


def _assert_bit_exact(ref, got, label):
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.isfinite(b).all(), f"{label}: grad{i} not finite"
        if torch.equal(a, b):
            continue
        scale = a.abs().max().item() + 1e-30
        rel = (a.double() - b.double()).abs().max().item() / scale
        pytest.fail(f"{label}: grad{i} differs from gpu-direct, max|d|/scale={rel:.2e} "
                    f"(same kernels, same numbers -- this is a staging slot/race bug, not noise)")


# (equation, why it is here)
_EQS = [
    ("Acoustic",   "2-D, nvar=1: the ring>=2 early-issue overwrite (#1) alone"),
    ("Acoustic3D", "3-D, nvar=1: #1 on the 3-D disk path"),
    ("Elastic3D",  "3-D, nvar>1: #1 plus the non-blocking-stream race (#2)"),
    ("DASMu3D",    "3-D, nvar>1, more fields: the other #2 survivor"),
]


@pytest.mark.parametrize("eqname", [e for e, _ in _EQS], ids=[e for e, _ in _EQS])
def test_disk_default_knobs_matches_gpu(eqname, tmp_path):
    """The config a user actually gets from ``storage='disk'`` alone: sync read,
    transfer_interval 32, ring 3 (2-D) / 2 (3-D)."""
    ref = _gradients(eqname, "gpu", tmp_path)
    got = _gradients(eqname, "disk", tmp_path / "disk")
    _assert_bit_exact(ref, got, f"{eqname} disk(defaults)")


@pytest.mark.parametrize("eqname", [e for e, _ in _EQS], ids=[e for e, _ in _EQS])
def test_disk_short_chunks_matches_gpu(eqname, tmp_path):
    """Short chunks so the reverse walk crosses many chunk boundaries (25 with
    nt=100), i.e. the early-issue / slot logic fires ~25 times per gradient."""
    ref = _gradients(eqname, "gpu", tmp_path)
    got = _gradients(eqname, "disk", tmp_path / "disk", transfer_interval=4, ring_buffers=2)
    _assert_bit_exact(ref, got, f"{eqname} disk(ti=4,ring=2)")


@pytest.mark.parametrize("eqname", ["Elastic3D", "DASMu3D"])
def test_cpu_ring2_nvar_gt1_matches_gpu(eqname, tmp_path):
    """Defect #3: host staging with ring_buffers>=2 on an nvar>1 equation -- the
    two multi-field restore readers must read the slot the prefetch wrote,
    not slot 0.  Latent on the default (cpu ring = 1), so ask for ring 2."""
    ref = _gradients(eqname, "gpu", tmp_path)
    got = _gradients(eqname, "cpu", tmp_path, transfer_interval=4, ring_buffers=2)
    _assert_bit_exact(ref, got, f"{eqname} cpu(ti=4,ring=2)")

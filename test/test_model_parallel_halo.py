"""Tests for :class:`sweep.parallel.HaloExchange`.

Single-process tests cover the degenerate "no neighbour" case where
HaloExchange must be a true no-op. Multi-process tests spawn 2 or 4 gloo
ranks on CPU and verify the actual halo-content exchange and its adjoint.

We avoid ``torch.autograd.gradcheck`` here because the natural finite-
difference setup requires perturbing the input on ONE rank while the other
ranks call forward with FIXED inputs, in lockstep with each gradcheck
iteration. That orchestration is fragile across PyTorch versions; an
explicit value-level check of forward and backward is equivalent and more
direct.
"""

from __future__ import annotations

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sweep.parallel import HaloExchange, ModelParallelMesh, exchange_halos


# ---------------------------------------------------------------------------
# Single-process tests (no torch.distributed init)
# ---------------------------------------------------------------------------

def _local_mesh(grid, world_size, rank):
    """Build a ModelParallelMesh without initialising dist (only valid for
    1-rank meshes where there are no neighbours)."""
    return ModelParallelMesh(grid=grid, shot_groups=1,
                             world_size=world_size, rank=rank)


def test_singleton_forward_is_noop_2d():
    mesh = _local_mesh((1, 1), 1, 0)
    w = torch.randn(1, 1, 8, 16)
    expected = w.clone()
    out = HaloExchange.apply(w, mesh, 2, ('x',))
    assert torch.equal(out, expected)
    assert out.data_ptr() == w.data_ptr()


def test_singleton_backward_passes_grad_through():
    """In the singleton case the halo is NEVER overwritten (no neighbours),
    so backward must pass the upstream grad through unchanged."""
    mesh = _local_mesh((1, 1), 1, 0)
    w = torch.randn(1, 1, 8, 16, requires_grad=True)
    out = HaloExchange.apply(w, mesh, 2, ('x',))
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert torch.equal(w.grad, grad_out)


def test_singleton_3d_two_axes_noop():
    mesh = _local_mesh((1, 1), 1, 0)
    w = torch.randn(1, 1, 4, 8, 10)
    expected = w.clone()
    out = HaloExchange.apply(w, mesh, 1, ('x', 'y'))
    assert torch.equal(out, expected)


def test_halo_zero_short_circuits():
    """halo=0 → both forward and backward do nothing, even when neighbours exist."""
    mesh = _local_mesh((1, 2), 2, 0)   # not a real 2-rank mesh but model_pg is None
    # The model_pg is None (because dist isn't init'd) but with halo=0 the function
    # short-circuits before touching the process group.
    w = torch.randn(1, 1, 4, 8, requires_grad=True)
    out = HaloExchange.apply(w, mesh, 0, ('x',))
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert out.data_ptr() == w.data_ptr()
    assert torch.equal(w.grad, grad_out)


# ---------------------------------------------------------------------------
# Multi-process spawn helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _test_backend() -> str:
    """Backend selected via ``HALO_TEST_BACKEND`` env var. Default ``gloo``."""
    return os.environ.get("HALO_TEST_BACKEND", "gloo")


def _test_device_kind() -> str:
    """``HALO_TEST_DEVICE`` env var; one of ``cpu`` (default) or ``cuda``."""
    return os.environ.get("HALO_TEST_DEVICE", "cpu")


def _worker_device(rank: int) -> torch.device:
    """Per-rank tensor device. ``cuda`` requires the worker to have called
    ``torch.cuda.set_device(rank)`` first (we do that in ``_init_pg``)."""
    if _test_device_kind() == "cuda":
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _world_size_supported(world_size: int) -> bool:
    """Whether the current env can host ``world_size`` test ranks. ``cuda``
    requires that many visible GPUs; ``cpu`` always can."""
    if _test_device_kind() == "cuda":
        return torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    return True


def _run_spawn(worker, world_size, *args, timeout: float = 120.0):
    """Spawn ``world_size`` processes and join with a hard timeout."""
    if not _world_size_supported(world_size):
        pytest.skip(
            f"Need {world_size} {_test_device_kind()} ranks, but env can't host them"
        )
    port = _find_free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=worker, args=(rank, world_size, port) + tuple(args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout)
    for p in procs:
        if p.is_alive():
            p.terminate()
            pytest.fail(f"worker pid={p.pid} did not finish in {timeout}s")
        if p.exitcode != 0:
            pytest.fail(f"worker pid={p.pid} exited with {p.exitcode}")


def _init_pg(rank, world_size, port):
    """Initialise the per-worker process group.

    Honors ``HALO_TEST_BACKEND`` (default ``gloo``). For ``nccl`` we also
    pin this worker to ``cuda:rank`` so the NCCL communicator binds cleanly.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    backend = _test_backend()
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 2-rank forward / backward (2-D, py=1, px=2)
# ---------------------------------------------------------------------------

def _worker_forward_2d_1x2(rank, world_size, port):
    try:
        _init_pg(rank, world_size, port)
        device = _worker_device(rank)
        mesh = ModelParallelMesh(grid=(1, 2), shot_groups=1)
        halo = 2
        Nz, Nx_loc = 4, 6
        total_x = Nx_loc + 2 * halo

        w = torch.full((1, 1, Nz, total_x), -1.0, device=device)
        positions = torch.arange(Nx_loc, device=device).float().view(1, 1, 1, Nx_loc)
        w[..., halo:-halo] = (rank + 1) * 100.0 + positions

        out = HaloExchange.apply(w, mesh, halo, ('x',))

        if rank == 0:
            assert (out[..., -halo:] ==
                    torch.tensor([200.0, 201.0], device=device)).all(), \
                out[..., -halo:]
            assert (out[..., :halo] == -1.0).all()
        else:
            assert (out[..., :halo] ==
                    torch.tensor([104.0, 105.0], device=device)).all(), \
                out[..., :halo]
            assert (out[..., -halo:] == -1.0).all()

        expected_interior = (rank + 1) * 100.0 + positions
        assert (out[..., halo:-halo] == expected_interior).all()
    finally:
        _cleanup_pg()


def test_halo_forward_2d_1x2():
    _run_spawn(_worker_forward_2d_1x2, 2)


def _worker_backward_2d_1x2(rank, world_size, port):
    try:
        _init_pg(rank, world_size, port)
        device = _worker_device(rank)
        mesh = ModelParallelMesh(grid=(1, 2), shot_groups=1)
        halo = 2
        Nz, Nx_loc = 4, 6
        total_x = Nx_loc + 2 * halo

        # NCCL does not support fp64 reductions on all GPUs; use float32 there.
        dtype = torch.float32 if _test_backend() == "nccl" else torch.float64
        w = torch.randn(1, 1, Nz, total_x, dtype=dtype, device=device,
                        requires_grad=True)
        out = HaloExchange.apply(w, mesh, halo, ('x',))
        # loss = out.sum() — every cell of `out` (interior + halo) gets unit grad.
        out.sum().backward()

        # Expected gradient w.r.t. w:
        # interior cells get 1 from own loss; cells adjacent to a neighbour also
        # receive that neighbour's halo gradient (= 1) accumulated in. Halo
        # cells facing a neighbour are zeroed (forward overwrote them).
        expected = torch.ones_like(w.detach())
        if mesh.neighbour_rank('x', +1) is not None:
            expected[..., -2 * halo:-halo] += 1.0     # interior_high gets +1 from rank+1
            expected[..., -halo:] = 0.0               # halo_high is overwritten, grad=0
        if mesh.neighbour_rank('x', -1) is not None:
            expected[..., halo:2 * halo] += 1.0       # interior_low gets +1 from rank-1
            expected[..., :halo] = 0.0                # halo_low overwritten

        assert torch.equal(w.grad, expected), (
            f"rank {rank} grad mismatch:\n got\n{w.grad}\n want\n{expected}"
        )
    finally:
        _cleanup_pg()


def test_halo_backward_2d_1x2():
    _run_spawn(_worker_backward_2d_1x2, 2)


# ---------------------------------------------------------------------------
# 4-rank 3-D (py=2, px=2): forward and backward exercise both x and y axes
# ---------------------------------------------------------------------------

def _worker_forward_3d_2x2(rank, world_size, port):
    try:
        _init_pg(rank, world_size, port)
        device = _worker_device(rank)
        mesh = ModelParallelMesh(grid=(2, 2), shot_groups=1)
        halo = 1
        Nz, Ny_loc, Nx_loc = 2, 3, 3
        total_y = Ny_loc + 2 * halo
        total_x = Nx_loc + 2 * halo

        w = torch.full((1, 1, Nz, total_y, total_x), -1.0, device=device)
        w[..., halo:-halo, halo:-halo] = (rank + 1) * 100.0
        out = HaloExchange.apply(w, mesh, halo, ('x', 'y'))

        # Expected neighbour tags for this rank
        tag_self = (rank + 1) * 100.0
        n_xm = mesh.neighbour_rank('x', -1)
        n_xp = mesh.neighbour_rank('x', +1)
        n_ym = mesh.neighbour_rank('y', -1)
        n_yp = mesh.neighbour_rank('y', +1)

        # Interior preserved
        assert torch.all(out[..., halo:-halo, halo:-halo] == tag_self)

        # x halos (along last axis), inspected in the interior y-strip so that
        # we test x exchange independently of y exchange.
        if n_xm is not None:
            assert torch.all(out[..., halo:-halo, :halo] == (n_xm + 1) * 100.0)
        else:
            assert torch.all(out[..., halo:-halo, :halo] == -1.0)
        if n_xp is not None:
            assert torch.all(out[..., halo:-halo, -halo:] == (n_xp + 1) * 100.0)
        else:
            assert torch.all(out[..., halo:-halo, -halo:] == -1.0)

        # y halos (along axis -2), inspected in the interior x-strip
        if n_ym is not None:
            assert torch.all(out[..., :halo, halo:-halo] == (n_ym + 1) * 100.0)
        else:
            assert torch.all(out[..., :halo, halo:-halo] == -1.0)
        if n_yp is not None:
            assert torch.all(out[..., -halo:, halo:-halo] == (n_yp + 1) * 100.0)
        else:
            assert torch.all(out[..., -halo:, halo:-halo] == -1.0)
    finally:
        _cleanup_pg()


def test_halo_forward_3d_2x2():
    _run_spawn(_worker_forward_3d_2x2, 4)


def _worker_backward_3d_2x2(rank, world_size, port):
    try:
        _init_pg(rank, world_size, port)
        device = _worker_device(rank)
        mesh = ModelParallelMesh(grid=(2, 2), shot_groups=1)
        halo = 1
        Nz, Ny_loc, Nx_loc = 2, 3, 3
        total_y = Ny_loc + 2 * halo
        total_x = Nx_loc + 2 * halo

        dtype = torch.float32 if _test_backend() == "nccl" else torch.float64
        w = torch.randn(1, 1, Nz, total_y, total_x,
                        dtype=dtype, device=device, requires_grad=True)
        out = HaloExchange.apply(w, mesh, halo, ('x', 'y'))
        out.sum().backward()

        expected = torch.ones_like(w.detach())
        # x axis adjustments (last dim)
        if mesh.neighbour_rank('x', +1) is not None:
            expected[..., -2 * halo:-halo] += 1.0
            expected[..., -halo:] = 0.0
        if mesh.neighbour_rank('x', -1) is not None:
            expected[..., halo:2 * halo] += 1.0
            expected[..., :halo] = 0.0
        # y axis adjustments (second-to-last dim)
        if mesh.neighbour_rank('y', +1) is not None:
            expected[..., -2 * halo:-halo, :] += 1.0
            expected[..., -halo:, :] = 0.0
        if mesh.neighbour_rank('y', -1) is not None:
            expected[..., halo:2 * halo, :] += 1.0
            expected[..., :halo, :] = 0.0

        assert torch.equal(w.grad, expected), (
            f"rank {rank}: grad mismatch\n{w.grad}\nvs\n{expected}"
        )
    finally:
        _cleanup_pg()


def test_halo_backward_3d_2x2():
    _run_spawn(_worker_backward_3d_2x2, 4)


# ---------------------------------------------------------------------------
# exchange_halos: multi-field convenience
# ---------------------------------------------------------------------------

def _worker_multi_field_2rank(rank, world_size, port):
    try:
        _init_pg(rank, world_size, port)
        device = _worker_device(rank)
        mesh = ModelParallelMesh(grid=(1, 2), shot_groups=1)
        halo = 1
        Nx_loc = 4
        total_x = Nx_loc + 2 * halo

        f0 = torch.full((1, 1, 2, total_x), -1.0, device=device)
        f0[..., halo:-halo] = (rank + 1)            # 1 or 2
        f1 = torch.full((1, 1, 2, total_x), -1.0, device=device)
        f1[..., halo:-halo] = (rank + 1) * 10       # 10 or 20

        outs = exchange_halos([f0, f1], mesh, halo, ('x',))

        # Each output's halo should reflect the OTHER rank's interior of that SAME field.
        if rank == 0:
            assert torch.all(outs[0][..., -halo:] == 2)   # rank 1's f0 = 2
            assert torch.all(outs[1][..., -halo:] == 20)  # rank 1's f1 = 20
            assert torch.all(outs[0][..., :halo] == -1)
            assert torch.all(outs[1][..., :halo] == -1)
        else:
            assert torch.all(outs[0][..., :halo] == 1)
            assert torch.all(outs[1][..., :halo] == 10)
    finally:
        _cleanup_pg()


def test_exchange_halos_multi_field():
    _run_spawn(_worker_multi_field_2rank, 2)

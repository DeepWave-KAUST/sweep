"""Halo exchange between tile neighbours for model-parallel wavefields.

A wavefield tile is laid out as ``(B, C, Nz, [Ny,] Nx_loc)`` with ``halo``
cells of padding on each side along the split axes. The TRUE tile data sits
in ``wavefield[..., halo : -halo]`` along each split axis; the cells outside
that interior are the halo strips that this module fills.

Naming convention along one split axis (length ``n = nx_loc + 2*halo``)::

    [   :halo]   halo_low      <- recv from -x neighbour's interior
    [halo:2h ]   interior_low  -> send to -x neighbour (becomes its halo_high)
    ...
    [-2h:-h  ]   interior_high -> send to +x neighbour (becomes its halo_low)
    [-h :    ]   halo_high     <- recv from +x neighbour's interior

Forward fills the halo regions in place. Backward is the formal adjoint:
the gradient sitting in our halo is what flowed FROM the neighbour during
forward, so we send it back and the neighbour accumulates it into its
interior strip. Our own halo gradient is then zeroed, because forward
overwrites the halo entirely — the input halo cells did not contribute to
any output value.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from sweep.parallel.mesh import ModelParallelMesh


_AXIS_DIMS = {"x": -1, "y": -2}


def _axis_dim(axis: str) -> int:
    """Tensor dim index for an axis name. ``x`` is always the trailing dim."""
    try:
        return _AXIS_DIMS[axis]
    except KeyError as e:
        raise ValueError(f"axis={axis!r} must be 'x' or 'y'") from e


def _slice_along_dim(
    t: torch.Tensor, dim: int, start: Optional[int], stop: Optional[int]
) -> torch.Tensor:
    """Return ``t[..., start:stop, ...]`` along the given ``dim``."""
    idx: List[slice] = [slice(None)] * t.ndim
    idx[dim] = slice(start, stop)
    return t[tuple(idx)]


def _strip(
    t: torch.Tensor,
    dim: int,
    start: Optional[int],
    stop: Optional[int],
    exclude: Sequence[Tuple[int, Optional[int], Optional[int]]] = (),
) -> torch.Tensor:
    """Slice ``t`` at ``[start:stop]`` along ``dim``, optionally narrowing other
    dims listed in ``exclude``.

    Each ``(other_dim, low_excl, high_excl)`` entry narrows ``other_dim`` to
    ``[low_excl : -high_excl]``; either side is skipped when its value is
    ``None`` (used when no later-axis neighbour writes that side, so the
    earlier axis still owns that halo edge).
    """
    idx: List[slice] = [slice(None)] * t.ndim
    idx[dim] = slice(start, stop)
    for d, low_excl, high_excl in exclude:
        stop_d = -high_excl if high_excl else None
        idx[d] = slice(low_excl, stop_d)
    return t[tuple(idx)]


class HaloExchange(torch.autograd.Function):
    """Exchange halo cells between neighbour ranks for one wavefield tensor.

    Parameters
    ----------
    wavefield : torch.Tensor
        Shape ``(B, C, Nz, [Ny,] Nx_loc)``. Halo strips are written in place
        on the requested axes; the rest of the tensor is untouched.
    mesh : ModelParallelMesh
        Source of neighbour ranks and the ``model_pg`` process group.
    halo : int
        Halo width per side per axis (typically ``equation.so // 2``).
    axes : tuple of {'x', 'y'}
        Split axes to exchange. 2-D: ``('x',)``; 3-D: ``('x', 'y')``.

    Returns
    -------
    wavefield : torch.Tensor
        The same tensor (halo regions updated in place).

    Notes
    -----
    Forward dispatches one fused ``dist.batch_isend_irecv`` per call covering
    all directions in ``axes`` (2-D: up to 2 messages; 3-D: up to 4). Backward
    runs the adjoint exchange (gradient ``halo → owner.interior += grad``)
    plus zeros our own halo gradient where a neighbour exists.
    """

    @staticmethod
    def forward(ctx, wavefield, mesh, halo, axes):
        ctx.mesh = mesh
        ctx.halo = int(halo)
        ctx.axes = tuple(axes)
        if ctx.halo > 0:
            _exchange_forward(wavefield, mesh, ctx.halo, ctx.axes)
        return wavefield

    @staticmethod
    def backward(ctx, grad_w):
        if ctx.halo > 0:
            # `grad_w` may arrive as a broadcast/expanded view (e.g. when the
            # downstream loss is `out.sum()`, autograd hands in a stride-0
            # ones-like tensor). In-place writes on such a view fail with
            # "more than one element refers to a single memory location";
            # materialise a real, writable buffer first.
            grad_w = grad_w.contiguous()
            _exchange_backward(grad_w, ctx.mesh, ctx.halo, ctx.axes)
        return grad_w, None, None, None


def _later_axes_exclude(
    axes: Tuple[str, ...],
    current_index: int,
    halo: int,
    mesh: ModelParallelMesh,
) -> List[Tuple[int, Optional[int], Optional[int]]]:
    """For the axis at position ``current_index`` in the exchange order,
    return slice exclusions along axes processed LATER, accounting for THIS
    rank's neighbour topology.

    An earlier axis must only narrow along a later axis's halo on the side
    where the later axis actually writes in forward — i.e. where the later
    axis has a neighbour. When the later axis has no neighbour on a side,
    that halo edge is never overwritten by the later axis and therefore
    stays "owned" by the earlier axis.

    Without this rank-aware narrowing the adjoint either over-counts
    (when the later axis doesn't actually overwrite a corner) or
    under-zeros (when the earlier axis erroneously skips a cell that no
    later axis writes).
    """
    excludes: List[Tuple[int, Optional[int], Optional[int]]] = []
    for later_axis in axes[current_index + 1:]:
        dim = _axis_dim(later_axis)
        has_low = mesh.neighbour_rank(later_axis, -1) is not None
        has_high = mesh.neighbour_rank(later_axis, +1) is not None
        if has_low or has_high:
            low_excl = halo if has_low else None
            high_excl = halo if has_high else None
            excludes.append((dim, low_excl, high_excl))
    return excludes


def _exchange_forward(
    wavefield: torch.Tensor,
    mesh: ModelParallelMesh,
    halo: int,
    axes: Tuple[str, ...],
) -> None:
    """In-place: fill halo strips with neighbour interior data.

    Earlier axes' sends/recvs are restricted to the interior of later axes,
    so each axis only writes the cells it actually "owns" (corners go to the
    last axis). Output is equivalent to the naive full-strip exchange where
    later axes overwrite earlier ones, but without the wasted bandwidth on
    corners.
    """
    p2p_ops: List[dist.P2POp] = []
    send_keepalive: List[torch.Tensor] = []
    deferred_copy: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, axis in enumerate(axes):
        dim = _axis_dim(axis)
        exclude = _later_axes_exclude(axes, i, halo, mesh)
        low_neighbour = mesh.neighbour_rank(axis, -1)
        high_neighbour = mesh.neighbour_rank(axis, +1)

        if low_neighbour is not None:
            send = _strip(wavefield, dim, halo, 2 * halo, exclude=exclude).contiguous()
            recv = torch.empty_like(send)
            send_keepalive.append(send)
            p2p_ops.append(dist.P2POp(dist.isend, send, low_neighbour, group=mesh.model_pg))
            p2p_ops.append(dist.P2POp(dist.irecv, recv, low_neighbour, group=mesh.model_pg))
            deferred_copy.append(
                (_strip(wavefield, dim, 0, halo, exclude=exclude), recv)
            )

        if high_neighbour is not None:
            send = _strip(wavefield, dim, -2 * halo, -halo, exclude=exclude).contiguous()
            recv = torch.empty_like(send)
            send_keepalive.append(send)
            p2p_ops.append(dist.P2POp(dist.isend, send, high_neighbour, group=mesh.model_pg))
            p2p_ops.append(dist.P2POp(dist.irecv, recv, high_neighbour, group=mesh.model_pg))
            deferred_copy.append(
                (_strip(wavefield, dim, -halo, None, exclude=exclude), recv)
            )

    if not p2p_ops:
        return

    reqs = dist.batch_isend_irecv(p2p_ops)
    for req in reqs:
        req.wait()

    for dest_view, src_buf in deferred_copy:
        dest_view.copy_(src_buf)


def _exchange_backward(
    grad_w: torch.Tensor,
    mesh: ModelParallelMesh,
    halo: int,
    axes: Tuple[str, ...],
) -> None:
    """In-place adjoint of :func:`_exchange_forward`.

    For each axis: send halo grads back to the owning neighbour (which will
    accumulate them into its interior strip), zero our own halo where the
    forward overwrote it, and accumulate the incoming neighbour-side halo
    grads into our own interior strip.

    Earlier axes operate on the same restricted strip as in forward (interior
    of later axes); the last axis owns the full strip including corners.
    """
    p2p_ops: List[dist.P2POp] = []
    send_keepalive: List[torch.Tensor] = []
    deferred_accum: List[Tuple[torch.Tensor, torch.Tensor]] = []
    halo_to_zero: List[torch.Tensor] = []

    for i, axis in enumerate(axes):
        dim = _axis_dim(axis)
        exclude = _later_axes_exclude(axes, i, halo, mesh)
        low_neighbour = mesh.neighbour_rank(axis, -1)
        high_neighbour = mesh.neighbour_rank(axis, +1)

        if low_neighbour is not None:
            send = _strip(grad_w, dim, 0, halo, exclude=exclude).contiguous()
            recv = torch.empty_like(send)
            send_keepalive.append(send)
            p2p_ops.append(dist.P2POp(dist.isend, send, low_neighbour, group=mesh.model_pg))
            p2p_ops.append(dist.P2POp(dist.irecv, recv, low_neighbour, group=mesh.model_pg))
            deferred_accum.append(
                (_strip(grad_w, dim, halo, 2 * halo, exclude=exclude), recv)
            )
            halo_to_zero.append(_strip(grad_w, dim, 0, halo, exclude=exclude))

        if high_neighbour is not None:
            send = _strip(grad_w, dim, -halo, None, exclude=exclude).contiguous()
            recv = torch.empty_like(send)
            send_keepalive.append(send)
            p2p_ops.append(dist.P2POp(dist.isend, send, high_neighbour, group=mesh.model_pg))
            p2p_ops.append(dist.P2POp(dist.irecv, recv, high_neighbour, group=mesh.model_pg))
            deferred_accum.append(
                (_strip(grad_w, dim, -2 * halo, -halo, exclude=exclude), recv)
            )
            halo_to_zero.append(_strip(grad_w, dim, -halo, None, exclude=exclude))

    if not p2p_ops:
        return

    reqs = dist.batch_isend_irecv(p2p_ops)
    for req in reqs:
        req.wait()

    # Zero halo BEFORE accumulating, so the accumulate writes the only non-zero
    # contribution to the interior gradient from this exchange.
    for halo_view in halo_to_zero:
        halo_view.zero_()
    for interior_view, src_buf in deferred_accum:
        interior_view.add_(src_buf)


def exchange_halos(
    wavefields: Sequence[torch.Tensor],
    mesh: ModelParallelMesh,
    halo: int,
    axes: Sequence[str],
) -> List[torch.Tensor]:
    """Exchange halos for each field in ``wavefields`` via separate calls.

    Convenience for multi-component equations. Each field calls
    ``HaloExchange.apply`` independently; messages are NOT fused across
    fields in this v1 (fusion is a follow-up optimisation when profiling
    flags per-step launch overhead).
    """
    return [HaloExchange.apply(w, mesh, halo, tuple(axes)) for w in wavefields]

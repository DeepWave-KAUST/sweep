"""Low-overhead forward-only halo exchanger for the per-step DD loop.

``exchange_halos`` (halo.py) costs ~415 µs/call on V100 (measured,
dd_cross_gpu_diag): ~330 µs is autograd.Function + per-call slice/exclude
computation + per-call ``contiguous()``/``empty_like`` allocations, ~85 µs
the bare batched P2P. The production forward/backward DD loops don't need
autograd through the exchange (gradients flow through the stepped C++
calls), so this class precomputes the slice views once, reuses pinned-down
staging buffers and P2POp lists, and per step only does:

    send-strip copy_ into reused buffer -> batch_isend_irecv -> wait
    -> recv buffer copy_ into halo strip

The wavefield tensor address must stay fixed across steps — true for the
stepped runners, which rotate ROLES over a fixed set of Python tensors.
Because the role rotates, construct one exchanger per (axis-set, tensor)
and look it up by the tensor's data_ptr (see :meth:`for_tensor`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .halo import _axis_dim, _strip
from .mesh import ModelParallelMesh


class FastHaloExchanger:
    """Reusable single-tensor halo exchange along ``axes`` (no autograd)."""

    def __init__(
        self,
        wavefield: torch.Tensor,
        mesh: ModelParallelMesh,
        halo: int,
        axes: Sequence[str],
    ) -> None:
        self._ops: List[dist.P2POp] = []
        self._send_views: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._recv_views: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for axis in axes:
            dim = _axis_dim(axis)
            low = mesh.neighbour_rank(axis, -1)
            high = mesh.neighbour_rank(axis, +1)
            if low is not None:
                sv = _strip(wavefield, dim, halo, 2 * halo)
                sbuf = sv.contiguous().clone()
                rv = _strip(wavefield, dim, 0, halo)
                rbuf = torch.empty_like(sbuf)
                self._send_views.append((sbuf, sv))
                self._recv_views.append((rv, rbuf))
                self._ops.append(dist.P2POp(dist.isend, sbuf, low, group=mesh.model_pg))
                self._ops.append(dist.P2POp(dist.irecv, rbuf, low, group=mesh.model_pg))
            if high is not None:
                sv = _strip(wavefield, dim, -2 * halo, -halo)
                sbuf = sv.contiguous().clone()
                rv = _strip(wavefield, dim, -halo, None)
                rbuf = torch.empty_like(sbuf)
                self._send_views.append((sbuf, sv))
                self._recv_views.append((rv, rbuf))
                self._ops.append(dist.P2POp(dist.isend, sbuf, high, group=mesh.model_pg))
                self._ops.append(dist.P2POp(dist.irecv, rbuf, high, group=mesh.model_pg))

        self._pending: Optional[list] = None

    def __call__(self) -> None:
        if not self._ops:
            return
        for sbuf, sview in self._send_views:
            sbuf.copy_(sview)
        for req in dist.batch_isend_irecv(self._ops):
            req.wait()
        for rview, rbuf in self._recv_views:
            rview.copy_(rbuf)

    def exchange_start(self) -> None:
        """Phase A of an overlapped exchange: copy the send strips and launch
        the batched P2P on the CURRENT stream (caller sets it to a comm stream),
        WITHOUT waiting. The recv-copy is deferred to :meth:`exchange_finish`
        so the caller can enqueue interior compute in between — that compute
        then runs concurrently with the in-flight P2P."""
        if not self._ops:
            return
        for sbuf, sview in self._send_views:
            sbuf.copy_(sview)
        self._pending = dist.batch_isend_irecv(self._ops)

    def exchange_finish(self) -> None:
        """Phase B: wait for the P2P to complete, then copy the received strips
        into the halo. ``req.wait()`` runs AFTER the interior compute has been
        enqueued, so the GPU already overlapped it; the recv-copy is correctly
        ordered after the transfer (no race)."""
        if not self._ops:
            return
        if self._pending is not None:
            for req in self._pending:
                req.wait()
            self._pending = None
        for rview, rbuf in self._recv_views:
            rview.copy_(rbuf)


class FastHaloSet:
    """Per-tensor exchanger cache for role-rotating wavefield lists.

    The stepped runners cycle u_now through 3 fixed tensors; ``exchange``
    builds (once) and reuses one :class:`FastHaloExchanger` per distinct
    tensor identity.
    """

    def __init__(self, mesh: ModelParallelMesh, halo: int, axes: Sequence[str]):
        self.mesh = mesh
        self.halo = halo
        self.axes = tuple(axes)
        self._cache: Dict[int, FastHaloExchanger] = {}

    def exchange(self, wavefield: torch.Tensor) -> None:
        key = wavefield.data_ptr()
        ex = self._cache.get(key)
        if ex is None:
            ex = FastHaloExchanger(wavefield, self.mesh, self.halo, self.axes)
            self._cache[key] = ex
        ex()

    def _get(self, wavefield: torch.Tensor) -> "FastHaloExchanger":
        key = wavefield.data_ptr()
        ex = self._cache.get(key)
        if ex is None:
            ex = FastHaloExchanger(wavefield, self.mesh, self.halo, self.axes)
            self._cache[key] = ex
        return ex

    def exchange_start(self, wavefield: torch.Tensor) -> None:
        """Phase A of an overlapped exchange (see FastHaloExchanger). Pair with
        :meth:`exchange_finish` on the SAME tensor, with interior compute in
        between, both on a comm stream."""
        self._get(wavefield).exchange_start()

    def exchange_finish(self, wavefield: torch.Tensor) -> None:
        self._get(wavefield).exchange_finish()

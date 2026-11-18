"""Low-overhead forward-only halo exchanger for the per-step DD loop.

``exchange_halos`` (halo.py) costs ~415 µs/call on V100 (measured with a
cross-GPU diagnostic): ~330 µs is autograd.Function + per-call slice/exclude
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

Transport
---------
``ModelParallel(prop, topo, halo=...)`` picks how the strips travel:

``nccl`` (default)
    ``torch.distributed.batch_isend_irecv`` on the tile process group, as
    above. Works on any backend and across nodes.
``ipc``
    CUDA peer-to-peer through torch symmetric memory: every rank's receive
    buffers live in one symmetric allocation per exchange batch, the sender
    copies its strip straight into the neighbour's buffer and raises a
    per-peer signal, the receiver waits on that signal on its stream and
    unpacks. No ``torch.distributed`` call per step (that call is 60-100 µs
    of CPU each), one copy per strip instead of two. Needs CUDA tiles, a
    torch with symmetric memory and every rank of the tile group on one node;
    anything else raises instead of silently using NCCL.

The strips carry the same bytes either way, so the two transports give
bit-identical results.
"""

from __future__ import annotations

import socket
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .halo import _axis_dim, _strip
from .mesh import ModelParallelMesh

#: Accepted values of ``ModelParallel(halo=...)``.
TRANSPORTS = ("nccl", "ipc")
_ALIGN = 128                      # byte alignment of every slot in an IPC buffer


def check_transport_name(transport) -> str:
    """``transport`` as one of :data:`TRANSPORTS`, or ``ValueError``."""
    if not isinstance(transport, str) or transport not in TRANSPORTS:
        raise ValueError(f"halo={transport!r}; expected one of {TRANSPORTS}")
    return transport


def _tile_group(mesh: ModelParallelMesh):
    return mesh.model_pg if mesh.model_pg is not None else dist.group.WORLD


def _check_transport(mesh: ModelParallelMesh, transport: str) -> None:
    """Every rank of the tile group must ask for the same transport, and
    ``ipc`` needs the whole group on one node. Runs (one small collective)
    each time a :class:`FastHaloSet` is built, for both transports, so a rank
    that differs is reported instead of deadlocking in the first exchange."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    pg = _tile_group(mesh)
    mine = (transport, socket.gethostname())
    allv: List = [None] * dist.get_world_size(pg)
    dist.all_gather_object(allv, mine, group=pg)
    if len({t for t, _ in allv}) != 1:
        raise RuntimeError(f"ranks of one DD tile group disagree on halo=: "
                           f"{[t for t, _ in allv]}")
    if transport == "ipc":
        hosts = sorted({h for _, h in allv})
        if len(hosts) != 1:
            raise RuntimeError(
                f"halo='ipc' needs every rank of a DD tile group on one node; this "
                f"group spans {hosts}. Use halo='nccl' for multi-node tiles.")


class _IpcBatch:
    """Symmetric-memory P2P exchange of every message of a list of exchangers.

    Layout: one uint8 symmetric buffer per batch, ``[parity][slot]`` where a
    slot is ``(exchanger j, side s)`` (s=0 receives from the low neighbour,
    s=1 from the high one). Slot capacities are the maximum over the group,
    so every rank computes the same offsets and a sender knows where its strip
    lands in the peer's buffer: my high-side send of exchanger j goes to the
    high neighbour's ``(j, 0)`` slot, and vice versa.

    Ordering. Step k writes parity ``k & 1``. Each side's put_signal follows
    its copies on its stream, each wait_signal precedes the unpack, so a
    sender reaching step k+2 has seen the receiver's step-(k+1) signal, which
    the receiver raised after unpacking step k: parity reuse never overwrites
    unread data. The signals are binary per (peer, parity) and reset by the
    wait, so nothing step-dependent is baked into a launch.

    Building is a collective over the tile group (all_gather + rendezvous):
    every rank builds its batches in the same order, which the DD driver
    already guarantees for the NCCL op lists.
    """

    def __init__(self, exchangers: Sequence["FastHaloExchanger"], mesh: ModelParallelMesh):
        import torch.distributed._symmetric_memory as symm_mem

        pg = _tile_group(mesh)
        dev = exchangers[0]._device
        if dev.type != "cuda":
            raise RuntimeError("halo='ipc' needs CUDA tiles; this DD run is on "
                               f"{dev}. Use halo='nccl'.")
        nslot = 2 * len(exchangers)
        need = torch.zeros(nslot, dtype=torch.int64)
        for j, ex in enumerate(exchangers):
            for _peer, side, _sv, rv in ex._msgs:
                need[2 * j + side] = rv.numel() * rv.element_size()
        world = dist.get_world_size(pg)
        rank = dist.get_rank(pg)
        allneed = torch.empty(world, nslot, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(allneed, need.to(dev), group=pg)
        allneed = allneed.cpu()
        cap = allneed.max(0).values.tolist()
        off, tot = [], 0
        for c in cap:
            off.append(tot)
            tot += -(-int(c) // _ALIGN) * _ALIGN
        tot = max(tot, _ALIGN)
        self._buf = symm_mem.empty(2 * tot, dtype=torch.uint8, device=dev)
        try:
            self._hdl = symm_mem.rendezvous(self._buf, pg.group_name)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "halo='ipc': symmetric-memory rendezvous failed (the GPUs of one DD "
                "tile group need CUDA peer access). Use halo='nccl'.") from e
        peer_bufs: Dict[int, torch.Tensor] = {}
        self._sends: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[], []]
        self._recvs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[], []]
        peers: List[int] = []
        for j, ex in enumerate(exchangers):
            for peer, side, sv, rv in ex._msgs:
                q = dist.get_group_rank(pg, peer)
                nb = rv.numel() * rv.element_size()
                if sv.numel() * sv.element_size() != int(allneed[q, 2 * j + 1 - side]):
                    raise RuntimeError(
                        f"halo strip size mismatch between tile ranks {rank} and {q} "
                        f"(exchanger {j}, side {side}): {sv.numel() * sv.element_size()} vs "
                        f"{int(allneed[q, 2 * j + 1 - side])} bytes")
                if q not in peer_bufs:
                    peer_bufs[q] = self._hdl.get_buffer(q, (2 * tot,), torch.uint8)
                    peers.append(q)
                for p in range(2):
                    o = p * tot + off[2 * j + 1 - side]
                    dst = peer_bufs[q][o:o + sv.numel() * sv.element_size()]
                    self._sends[p].append((dst.view(sv.dtype).view(sv.shape), sv))
                    o = p * tot + off[2 * j + side]
                    src = self._buf[o:o + nb]
                    self._recvs[p].append((rv, src.view(rv.dtype).view(rv.shape)))
        self._peers = peers
        self._parity = 0

    def start(self) -> None:
        p = self._parity
        for dst, sv in self._sends[p]:
            dst.copy_(sv)
        for q in self._peers:
            self._hdl.put_signal(q, channel=p)

    def finish(self) -> None:
        p = self._parity
        for q in self._peers:
            self._hdl.wait_signal(q, channel=p)
        for rv, src in self._recvs[p]:
            rv.copy_(src)
        self._parity = p ^ 1

    def __call__(self) -> None:
        self.start()
        self.finish()


class FastHaloExchanger:
    """Reusable single-tensor halo exchange along ``axes`` (no autograd)."""

    def __init__(
        self,
        wavefield: torch.Tensor,
        mesh: ModelParallelMesh,
        halo: int,
        axes: Sequence[str],
        transport: str = "nccl",
    ) -> None:
        self.transport = check_transport_name(transport)
        self._mesh = mesh
        self._device = wavefield.device
        self._ops: List[dist.P2POp] = []
        self._send_views: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._recv_views: List[Tuple[torch.Tensor, torch.Tensor]] = []
        #: (peer global rank, side 0=low/1=high, send view, recv view), in the
        #: order the NCCL op list uses.
        self._msgs: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
        self._ipc: Optional[_IpcBatch] = None

        for axis in axes:
            dim = _axis_dim(axis)
            for side, peer in ((0, mesh.neighbour_rank(axis, -1)),
                               (1, mesh.neighbour_rank(axis, +1))):
                if peer is None:
                    continue
                if side == 0:
                    sv = _strip(wavefield, dim, halo, 2 * halo)
                    rv = _strip(wavefield, dim, 0, halo)
                else:
                    sv = _strip(wavefield, dim, -2 * halo, -halo)
                    rv = _strip(wavefield, dim, -halo, None)
                self._msgs.append((peer, side, sv, rv))
                if transport != "nccl":
                    continue
                sbuf = sv.contiguous().clone()
                rbuf = torch.empty_like(sbuf)
                self._send_views.append((sbuf, sv))
                self._recv_views.append((rv, rbuf))
                self._ops.append(dist.P2POp(dist.isend, sbuf, peer, group=mesh.model_pg))
                self._ops.append(dist.P2POp(dist.irecv, rbuf, peer, group=mesh.model_pg))

        self._pending: Optional[list] = None

    def _ipc_batch(self) -> _IpcBatch:
        # Built on first use, on every rank (collective), before any early return.
        if self._ipc is None:
            self._ipc = _IpcBatch([self], self._mesh)
        return self._ipc

    def __call__(self) -> None:
        if self.transport == "ipc":
            self._ipc_batch()()
            return
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
        then runs concurrently with the in-flight P2P.

        With ``ipc`` this copies the strips into the neighbours' buffers and
        raises their signals; the wait moves to :meth:`exchange_finish`."""
        if self.transport == "ipc":
            self._ipc_batch().start()
            return
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
        if self.transport == "ipc":
            self._ipc_batch().finish()
            return
        if not self._ops:
            return
        if self._pending is not None:
            for req in self._pending:
                req.wait()
            self._pending = None
        for rview, rbuf in self._recv_views:
            rview.copy_(rbuf)


def _batch_transport(exchangers) -> str:
    ts = {ex.transport for ex in exchangers}
    if len(ts) > 1:
        raise ValueError(f"one halo batch mixes transports {sorted(ts)}")
    return ts.pop() if ts else "nccl"


class FastHaloGroup:
    """Exchange a FIXED list of wavefields in ONE batched isend/irecv.

    The elastic DD loop exchanges every velocity (then every stress) field
    after each half-step.  Running a separate ``batch_isend_irecv`` + ``wait``
    per field costs ``nphys`` (5 in 2-D, 9 in 3-D) synchronisation rounds per
    step; concatenating all fields of a group into a single P2P op list
    collapses that to ONE wait, cutting per-step comm latency (the bare P2P
    bandwidth is unchanged — this trades many small launches/waits for one).

    Fields must keep fixed addresses across steps — true for the elastic slots
    (no buffer-role rotation).  Both ranks build the op list in the same field
    order, so the per-field send/recv matching is exactly the per-field path's,
    just batched (bit-identical).
    """

    def __init__(self, fields, mesh: ModelParallelMesh, halo: int,
                 axes: Sequence[str], transport: str = "nccl") -> None:
        self._ex = [FastHaloExchanger(f, mesh, halo, axes, transport) for f in fields]
        self._ops = [op for ex in self._ex for op in ex._ops]
        self._mesh = mesh
        self.transport = transport
        self._ipc: Optional[_IpcBatch] = None

    def __call__(self) -> None:
        if self.transport == "ipc":
            if self._ipc is None:
                self._ipc = _IpcBatch(self._ex, self._mesh)
            self._ipc()
            return
        if not self._ops:
            return
        for ex in self._ex:
            for sbuf, sview in ex._send_views:
                sbuf.copy_(sview)
        for req in dist.batch_isend_irecv(self._ops):
            req.wait()
        for ex in self._ex:
            for rview, rbuf in ex._recv_views:
                rview.copy_(rbuf)


class FastHaloMultiGroup:
    """One batched isend/irecv across EVERY cut axis, not one per axis.

    :class:`FastHaloGroup` already collapses the FIELDS of one axis into a
    single P2P round, but the DD driver still issued one such round per cut
    axis, so a 2x2 mesh paid two waits per shipment.  These strips are
    latency-bound, not bandwidth-bound: on a large 3-D grid
    (2x2 tiles, halo 2) a six-field shipment of a few MiB ran at
    6.5 GiB/s on an NVLink that does 300+, i.e. the synchronisation
    rounds ARE the cost, not the bytes.  Concatenating both axes' op lists
    makes it one wait.

    Same buffers, same per-field pack/unpack order, same peers, same process
    group: only the GROUPING of the P2P ops changes, so no value can move.
    Every rank builds the list in the same (axis, field) order, which is what
    keeps the sends and receives matched.
    """

    def __init__(self, exchangers) -> None:
        self._ex = list(exchangers)
        self._ops = [op for ex in self._ex for op in ex._ops]
        self.transport = _batch_transport(self._ex)
        self._ipc: Optional[_IpcBatch] = None

    def __call__(self) -> None:
        if self.transport == "ipc":
            if self._ipc is None:
                self._ipc = _IpcBatch(self._ex, self._ex[0]._mesh)
            self._ipc()
            return
        if not self._ops:
            return
        for ex in self._ex:
            for sbuf, sview in ex._send_views:
                sbuf.copy_(sview)
        for req in dist.batch_isend_irecv(self._ops):
            req.wait()
        for ex in self._ex:
            for rview, rbuf in ex._recv_views:
                rview.copy_(rbuf)


class FastHaloSet:
    """Per-tensor exchanger cache for role-rotating wavefield lists.

    The stepped runners cycle u_now through 3 fixed tensors; ``exchange``
    builds (once) and reuses one :class:`FastHaloExchanger` per distinct
    tensor identity.  ``exchange_group`` batches a fixed list of fields
    (elastic velocity / stress) into one :class:`FastHaloGroup`.

    ``transport`` is ``ModelParallel``'s ``halo=``; building the set checks it
    across the tile group (see :func:`_check_transport`).
    """

    def __init__(self, mesh: ModelParallelMesh, halo: int, axes: Sequence[str],
                 transport: str = "nccl"):
        self.mesh = mesh
        self.halo = halo
        self.axes = tuple(axes)
        self.transport = check_transport_name(transport)
        _check_transport(mesh, self.transport)
        self._cache: Dict[int, FastHaloExchanger] = {}
        self._group_cache: Dict[tuple, FastHaloGroup] = {}

    def _get(self, wavefield: torch.Tensor) -> "FastHaloExchanger":
        key = wavefield.data_ptr()
        ex = self._cache.get(key)
        if ex is None:
            ex = FastHaloExchanger(wavefield, self.mesh, self.halo, self.axes,
                                   self.transport)
            self._cache[key] = ex
        return ex

    def exchange(self, wavefield: torch.Tensor) -> None:
        """Blocking exchange: build-or-reuse this tensor's exchanger and run it."""
        self._get(wavefield)()

    def exchange_group(self, fields: Sequence[torch.Tensor]) -> None:
        """Blocking exchange of several fixed-address fields in ONE batched P2P
        (elastic velocity / stress groups). Cached by the fields' data_ptrs."""
        self._group(fields)()

    def group_exchangers(self, fields: Sequence[torch.Tensor]):
        """This axis's per-field exchangers, for a caller batching ACROSS axes
        (:class:`FastHaloMultiGroup`). Same cache as :meth:`exchange_group`, so
        the buffers and the op order are the single-axis path's exactly."""
        return self._group(fields)._ex

    def _group(self, fields: Sequence[torch.Tensor]) -> "FastHaloGroup":
        key = tuple(f.data_ptr() for f in fields)
        grp = self._group_cache.get(key)
        if grp is None:
            grp = FastHaloGroup(list(fields), self.mesh, self.halo, self.axes,
                                self.transport)
            self._group_cache[key] = grp
        return grp

    def exchange_start(self, wavefield: torch.Tensor) -> None:
        """Phase A of an overlapped exchange (see FastHaloExchanger). Pair with
        :meth:`exchange_finish` on the SAME tensor, with interior compute in
        between, both on a comm stream."""
        self._get(wavefield).exchange_start()

    def exchange_finish(self, wavefield: torch.Tensor) -> None:
        self._get(wavefield).exchange_finish()

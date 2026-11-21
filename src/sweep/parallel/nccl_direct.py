"""NCCL point-to-point called directly through ctypes, for the eager halo
exchange (:mod:`sweep.parallel.eager_dd`).

A ``torch.distributed`` P2P call costs 60-350 us of CPU (the ops it builds,
the work objects, the wait); NCCL's own entry points cost a few.  On a 2-D
grid the eager step is host-bound, so that difference is a tenth of the
step.  The library is the one torch loaded (libnccl.so.2); one communicator per
process and device over the default group's ranks, made on first use (unique
id broadcast through torch.distributed) and kept.  Messages go on the CURRENT
stream, so they order with the step's kernels without events, and a CUDA graph
captured on that stream would hold them.
"""
from __future__ import annotations

import ctypes

import torch

_TYPES = {torch.float32: 7, torch.float64: 8}      # ncclFloat32, ncclFloat64


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_LIB = []
_COMMS = {}


def library():
    """The NCCL library torch loaded, typed; None if it cannot be opened."""
    if not _LIB:
        try:
            lib = ctypes.CDLL("libnccl.so.2")
        except OSError:
            _LIB.append(None)
            return None
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId,
                                         ctypes.c_int]
        for f in (lib.ncclSend, lib.ncclRecv):
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclGroupStart.argtypes = []
        lib.ncclGroupEnd.argtypes = []
        _LIB.append(lib)
    return _LIB[0]


def _check(rc):
    if rc != 0:
        raise RuntimeError(f"NCCL error {rc}: {library().ncclGetErrorString(rc).decode()}")


def communicator(device):
    """This process's communicator on ``device`` over the default process
    group's ranks (collective: every rank calls it the first time together)."""
    import torch.distributed as dist
    rank, world = dist.get_rank(), dist.get_world_size()
    key = (rank, world, device.index)
    comm = _COMMS.get(key)
    if comm is None:
        lib = library()
        uid = _UniqueId()
        if rank == 0:
            _check(lib.ncclGetUniqueId(ctypes.byref(uid)))
        t = torch.frombuffer(bytearray(bytes(uid)), dtype=torch.uint8).to(device)
        dist.broadcast(t, 0)
        uid = _UniqueId.from_buffer_copy(t.cpu().numpy().tobytes())
        comm = ctypes.c_void_p()
        with torch.cuda.device(device):
            _check(lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
        _COMMS[key] = comm
    return comm


class Messages:
    """``(peer, send, recv)`` triples of flat contiguous same-dtype CUDA
    tensors, exchanged on the current stream by :meth:`run`.  Built once per
    link; the buffers are static (the caller refills ``send`` and reads
    ``recv`` after the stream has run the messages)."""

    def __init__(self, triples):
        dev = triples[0][1].device
        dt = triples[0][1].dtype
        self.type = _TYPES[dt]
        self.comm = communicator(dev)
        self.lib = library()
        self.dev = dev
        self.ops = []
        for peer, s, r in triples:
            assert s.is_contiguous() and r.is_contiguous() and s.dtype == r.dtype == dt
            self.ops.append((int(peer), s.data_ptr(), r.data_ptr(), s.numel()))
        self.keep = triples

    def run(self):
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream)
        lib, comm, t = self.lib, self.comm, self.type
        _check(lib.ncclGroupStart())
        for q, sp, rp, n in self.ops:
            _check(lib.ncclSend(sp, n, t, q, comm, stream))
            _check(lib.ncclRecv(rp, n, t, q, comm, stream))
        _check(lib.ncclGroupEnd())

"""Checkpointed single-shot rollout of the SEM eager time loop (``memory=Ckpt(chunks=K)``).

The forward runs the loop in chunks of ``K`` steps without an autograd tape and keeps
one full-state checkpoint per chunk. The backward walks the chunks in reverse: it
restores a chunk's checkpoint, re-runs its ``K`` steps under autograd, back-propagates
them, and hands the state adjoint to the chunk before. Peak memory is ``nt/K`` states
plus one chunk's tape instead of the ``nt``-step tape, at the cost of one more forward.

On CUDA each chunk is captured once as a CUDA graph and replayed ``nt/K`` times: the
forward chunk (the source read off the wavelet at a device-side step counter, ``K``
calls of ``eq.step``, the receiver samples, the checkpoint) and the backward chunk
(restore, recompute, ``autograd.grad``, accumulate). The loop then does no host work
per kernel, which is what bounds the eager step at small and medium sizes (H100,
2-D elastic, 201x401: forward 0.45 -> 0.13 ms per step compiled, 1.46 -> 0.60 not).
The graphs are built per call: warm-up plus capture take 15-100 ms for ``K = 8``. A
step that cannot be captured (a host synchronisation inside it) runs the same chunks
without graphs, after one warning; so do the CPU and ``USE_CUDA_GRAPHS = False``.

An ``nt`` that ``K`` does not divide is padded with zero-wavelet steps whose records
are dropped; they come after every recorded step, so they change nothing.

The model reaches the loop as the equation's tensors that require grad, at any depth
of its attributes (Biot keeps its drag operators in a tuple, built by ``init_state``).
They are swapped for leaves on a copy of the equation, so each chunk's backward stops
at them, and their gradients leave through the autograd Function into the caller's
graph -- which is why ``init_state`` runs on the caller's equation first.
"""
from __future__ import annotations

import copy
import warnings

import torch

from ._assembly import scatter_point_sources

# Capture the chunks as CUDA graphs on CUDA devices (False: run them as they are).
USE_CUDA_GRAPHS = True
_CAPTURE_WARNED: set = set()


def _end_failed_capture():
    """Take the device's CUDA generator out of capture mode after a failed capture.

    ``CUDAGraph.capture_end`` ends the generators' capture only after
    ``cudaStreamEndCapture`` succeeds, so a capture that failed leaves them in it, and
    every later random draw on the device raises "Offset increment outside graph capture
    encountered unexpectedly" -- in the caller's code, long after the fallback. An empty
    capture begins and ends it again; the random stream is the one it was."""
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        pass


_WARMUP_STREAMS: dict = {}


def _warmup_stream(device):
    """The side stream the graphs are warmed up and captured on, one per device for the
    process.

    PyTorch keeps a cuBLAS workspace (8 MiB by default, 32 MiB on Hopper) for every
    (handle, stream) pair it has run cuBLAS on, until the process ends; a new stream per
    call would leave one behind per call -- two with a backward pass, which runs on the
    autograd thread's handle -- and an FWI loop would grow by that much every iteration.
    Capturing on the same stream as the warm-up keeps it to one set of workspaces."""
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in _WARMUP_STREAMS:
        _WARMUP_STREAMS[index] = torch.cuda.Stream(device=index)
    return _WARMUP_STREAMS[index]


def swap_grad_tensors(obj, leaves, originals, memo=None, active=None):
    """A copy of ``obj`` in which every tensor that requires grad -- at any depth of
    attributes, tuples, lists and dicts -- is a detached leaf (one per distinct
    tensor). Containers holding no such tensor are returned as they are, so the mesh
    and the constants stay shared. ``leaves`` / ``originals`` collect the pairs."""
    memo = {} if memo is None else memo
    active = set() if active is None else active
    if isinstance(obj, torch.Tensor):
        if not obj.requires_grad:
            return obj
        if id(obj) not in memo:
            leaf = obj.detach().clone().requires_grad_(True)
            memo[id(obj)] = leaf
            leaves.append(leaf)
            originals.append(obj)
        return memo[id(obj)]
    if id(obj) in active or isinstance(obj, (str, bytes, int, float, bool, complex,
                                             type(None), type, torch.device, torch.dtype)):
        return obj
    active.add(id(obj))
    try:
        rec = lambda x: swap_grad_tensors(x, leaves, originals, memo, active)
        if isinstance(obj, (tuple, list)):
            new = [rec(x) for x in obj]
            if all(a is b for a, b in zip(new, obj)):
                return obj
            return type(obj)(*new) if hasattr(obj, "_fields") else type(obj)(new)
        if isinstance(obj, dict):
            new = {k: rec(v) for k, v in obj.items()}
            return obj if all(new[k] is obj[k] for k in obj) else type(obj)(new)
        if hasattr(obj, "__dict__") and not callable(obj):
            changed = {}
            for k, v in vars(obj).items():
                v2 = rec(v)
                if v2 is not v:
                    changed[k] = v2
            if not changed:
                return obj
            c = copy.copy(obj)
            for k, v in changed.items():
                object.__setattr__(c, k, v)
            return c
        return obj
    finally:
        active.discard(id(obj))


class _Rollout:
    """The static buffers of one call and, on CUDA, its two captured chunks."""

    def __init__(self, eqs, step, leaves, source, wavelet, rec_GI, rec_W, recv, dt, nt,
                 chunk, need_grad):
        self.step, self.leaves, self.dt, self.need_grad = step, leaves, dt, need_grad
        self.what = type(eqs).__name__
        self.K = K = int(chunk)
        self.nc = -(-nt // K)
        self.nt, self.ntp = nt, self.nc * K
        self.recv, self.rec_GI, self.rec_W = tuple(recv), rec_GI, rec_W
        self.per_source = wavelet.dim() == 2
        self.source = source          # patterns, or (src_G, src_W, like) per source
        dev, dtype = rec_W.device, rec_W.dtype
        self.device = dev
        self.init = [s.detach().clone() for s in eqs.init_state(dt, dev, dtype)]
        self.state = [s.clone() for s in self.init]
        self.wav = wavelet.new_zeros(wavelet.shape[:-1] + (self.ntp,))
        self.seis = torch.zeros(self.ntp, rec_W.shape[0], len(recv), dtype=dtype, device=dev)
        self.t = torch.zeros((), dtype=torch.long, device=dev)
        self.c = torch.zeros((), dtype=torch.long, device=dev)
        self.arK = torch.arange(K, device=dev)
        if need_grad:
            self.ckpt = [s.new_zeros((self.nc,) + s.shape) for s in self.init]
            self.st_in = [s.clone().requires_grad_(True) for s in self.init]
            self.wk = self.wav.new_zeros(wavelet.shape[:-1] + (K,)).requires_grad_(True)
            self.cot = [torch.zeros_like(s) for s in self.init]
            self.gseis = torch.zeros_like(self.seis)
            self.g_leaves = [torch.zeros_like(l) for l in leaves]
            self.g_wav = torch.zeros_like(self.wav)
            self.used = [False] * len(leaves)
        self.g_f = self.g_b = None
        self._build_graphs()

    # ---- the two chunks -------------------------------------------------- #
    def _source_term(self, wk, j):
        if self.per_source:
            src_G, src_W, like = self.source
            return scatter_point_sources(src_G, src_W, wk[:, j], like)
        return [p * wk[j] for p in self.source]

    def _k_steps(self, st, wk):
        recs = []
        for j in range(self.K):
            st = self.step(st, self._source_term(wk, j), self.dt)
            recs.append(torch.stack([(st[i].reshape(-1)[self.rec_GI[i]] * self.rec_W).sum(1)
                                     for i in self.recv], -1))
        return st, torch.stack(recs)

    def _fwd_chunk(self):
        with torch.no_grad():
            if self.need_grad:
                for ck, s in zip(self.ckpt, self.state):
                    ck.index_copy_(0, self.c.view(1), s.unsqueeze(0))
            idx = self.t + self.arK
            st, recs = self._k_steps(tuple(self.state), self.wav.index_select(-1, idx))
            self.seis.index_copy_(0, idx, recs)
            mine = {s.data_ptr() for s in self.state}
            new = [n.clone() if n.data_ptr() in mine else n for n in st]   # pass-through fields
            for s, n in zip(self.state, new):
                s.copy_(n)
            self.t.add_(self.K)
            self.c.add_(1)

    def _bwd_chunk(self):
        with torch.no_grad():
            for si, ck in zip(self.st_in, self.ckpt):
                si.copy_(ck.index_select(0, self.c.view(1)).squeeze(0))
            idx = self.c * self.K + self.arK
            self.wk.copy_(self.wav.index_select(-1, idx))
            g_rec = self.gseis.index_select(0, idx)
        with torch.enable_grad():
            st, recs = self._k_steps(tuple(self.st_in), self.wk)
        outs, gos = [recs], [g_rec]
        for o, g in zip(st, self.cot):
            if o.requires_grad:
                outs.append(o)
                gos.append(g)
        ins = list(self.st_in) + list(self.leaves) + [self.wk]
        grads = torch.autograd.grad(outs, ins, grad_outputs=gos, allow_unused=True)
        ns, nl = len(self.st_in), len(self.leaves)
        with torch.no_grad():
            mine = {c.data_ptr() for c in self.cot}
            g_st = [None if g is None else (g.clone() if g.data_ptr() in mine else g)
                    for g in grads[:ns]]
            for cb, g in zip(self.cot, g_st):
                if g is None:
                    cb.zero_()
                else:
                    cb.copy_(g)
            for k, (acc, g) in enumerate(zip(self.g_leaves, grads[ns:ns + nl])):
                if g is not None:
                    acc.add_(g)
                    self.used[k] = True
            if grads[-1] is not None:
                self.g_wav.index_copy_(self.g_wav.dim() - 1, idx, grads[-1])
            self.c.sub_(1)

    # ---- capture --------------------------------------------------------- #
    def reset(self):
        with torch.no_grad():
            for s, i in zip(self.state, self.init):
                s.copy_(i)
            self.t.zero_()
            self.c.zero_()

    def _build_graphs(self):
        graphs = (USE_CUDA_GRAPHS and self.device.type == "cuda"
                  and torch.cuda.is_available())
        if not graphs:
            return
        with torch.cuda.device(self.device):
            side = _warmup_stream(self.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):        # warm-up: compiles, fills the autograd caches
                self.reset()
                self._fwd_chunk()
                if self.need_grad:
                    self.c.zero_()
                    self._bwd_chunk()
                    for g in self.g_leaves:
                        g.zero_()
            torch.cuda.current_stream().wait_stream(side)
            self.reset()
            before = torch.cuda.current_stream()
            try:
                g_f = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_f, stream=side):
                    self._fwd_chunk()
                g_b = None
                if self.need_grad:
                    g_b = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_b, stream=side):
                        self._bwd_chunk()
            except Exception as exc:          # a host sync in the step, or another capture error
                # torch.cuda.graph's __exit__ restores the caller's stream only after
                # capture_end succeeds: left on the capture stream, every later op of
                # the process would run unordered with work on the caller's stream
                torch.cuda.set_stream(before)
                torch.cuda.synchronize(self.device)
                _end_failed_capture()
                if self.what not in _CAPTURE_WARNED:
                    _CAPTURE_WARNED.add(self.what)
                    warnings.warn(
                        f"the {self.what} step could not be captured as a CUDA graph "
                        f"({type(exc).__name__}: {str(exc).strip().splitlines()[0][:200]}); "
                        f"memory=Ckpt runs its chunks without graphs, which is slower but "
                        f"gives the same result.", RuntimeWarning, stacklevel=4)
                self.reset()
                return
            self.g_f, self.g_b = g_f, g_b

    # ---- the two passes --------------------------------------------------- #
    def _replay(self, graph, chunk):
        if graph is None:
            for _ in range(self.nc):
                chunk()
            return
        with torch.cuda.device(self.device):
            for _ in range(self.nc):
                graph.replay()

    def run_forward(self, wavelet, leaf_values):
        with torch.no_grad():
            self.wav[..., :self.nt].copy_(wavelet)
            for l, v in zip(self.leaves, leaf_values):
                l.copy_(v)
            self.reset()
        self._replay(self.g_f, self._fwd_chunk)
        return self.seis[:self.nt].clone()

    def run_backward(self, gseis):
        with torch.no_grad():
            self.gseis.zero_()
            self.gseis[:self.nt].copy_(gseis)
            for b in (*self.cot, *self.g_leaves, self.g_wav):
                b.zero_()
            self.c.fill_(self.nc - 1)
        self._replay(self.g_b, self._bwd_chunk)

    def release(self):
        """Free the graphs' private pools and the checkpoints."""
        for g in (self.g_f, self.g_b):
            if g is not None:
                g.reset()
        self.g_f = self.g_b = None
        self.ckpt = self.state = self.init = None


class _CkptRolloutFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, R, wavelet, *originals):
        out = R.run_forward(wavelet, originals)
        ctx.R = R
        ctx.wav_grad = wavelet.requires_grad
        return out

    @staticmethod
    def backward(ctx, gseis):
        # create_graph=True runs this with grad on. The chunks' tapes are rebuilt
        # and dropped inside, so there is nothing to differentiate again -- and
        # once_differentiable would not say so: its error node hangs off fresh
        # leaves, which a second derivative w.r.t. the model never reaches.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "memory=Ckpt gives first derivatives only (it rebuilds each chunk's "
                "tape inside the backward pass); use memory=Full() for create_graph=True "
                "/ higher-order gradients.")
        R = ctx.R
        if R is None:
            raise RuntimeError(
                "memory=Ckpt's checkpointed rollout runs one backward pass per forward "
                "(it frees its checkpoints in it); a second one (retain_graph=True) "
                "needs memory=Full().")
        R.run_backward(gseis)
        g_wav = R.g_wav[..., :R.nt].clone() if ctx.wav_grad else None
        g_leaves = [g.clone() if used else None for g, used in zip(R.g_leaves, R.used)]
        R.release()
        ctx.R = None
        return (None, g_wav, *g_leaves)


def run_checkpointed(eq, step_of, source, wavelet, rec_GI, rec_W, recv, dt, nt, chunk):
    """The checkpointed rollout of one shot -> records ``(nt, nrec, ncomp)``.

    ``step_of(eq)`` returns the step function to run for an equation (its
    ``step``, or the compiled one); ``source`` is the list of source patterns of a
    shared wavelet ``(nt,)``, or ``(src_G, src_W, like)`` for a per-source one
    ``(P, nt)``; ``rec_GI`` maps each recorded field to its gather indices."""
    device, dtype = rec_W.device, rec_W.dtype
    eq.init_state(dt, device, dtype)       # model-derived operators join the caller's graph
    leaves, originals = [], []
    eqs = swap_grad_tensors(eq, leaves, originals)
    if eqs is eq:
        eqs = copy.copy(eq)
    need_grad = torch.is_grad_enabled() and (bool(leaves) or wavelet.requires_grad)
    if not need_grad:
        leaves, originals = [], []
    R = _Rollout(eqs, step_of(eqs), leaves, source, wavelet.detach(), rec_GI, rec_W, recv,
                 dt, nt, chunk, need_grad)
    if not need_grad:
        out = R.run_forward(wavelet.detach(), ())
        R.release()
        return out
    return _CkptRolloutFn.apply(R, wavelet, *originals)

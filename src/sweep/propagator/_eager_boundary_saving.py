"""Pure-PyTorch boundary-saving wavefield reconstruction for the eager path.

Clean, bug-fixed port of seistorch's boundary saving (``checkpoint*.py``) to
sweep's eager propagator.  See ``project_eager_boundary_saving_poc`` memory.

**Important framing**: the *adjoint* (gradient) is 100% autograd here — nothing
hand-written.  The only equation-specific piece is the *primal* reverse-time
reconstruction step, which is a forward-physics operation.  Two generic reverse
drivers, dispatched on the equation's time-discretisation, cover the library
without per-equation adjoint code:

* ``reverse='swap2nd'`` — 2nd-order-in-time schemes (``SecondOrderEquation``,
  e.g. ``Acoustic``).  ``u_{t+1} = 2u_t - u_{t-1} + dt^2 L(u_t)`` is symmetric in
  the two time levels, so the reverse step is just ``func`` with the levels
  swapped.  EXACT, no per-equation code.

* ``reverse='substep'`` — 1st-order leapfrog schemes (``FirstOrderEquation``,
  e.g. ``Acoustic1st``/``Elastic``).  The equation exposes its step as an ordered
  list of reversible sub-steps via ``interior_substeps()``; the exact inverse is
  those sub-steps composed in REVERSE order at ``-dt`` (with CPML zeroed, FS
  kept).  EXACT.  1st-order equations without ``interior_substeps`` are NOT
  supported for eager boundary saving (the propagator raises) — there is no
  approximate fallback.

Both zero the CPML memory variables during reconstruction (exact in the lossless
interior, where the model gradient lives; small near-PML error — same design as
the CUDA ``_bs`` path) and re-inject the saved boundary ring + the source.

Memory: ``O(nt * halo * perimeter)`` instead of ``O(nt * Ncells * nfields)``.
"""

import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import torch

# Ring geometry is backend-agnostic and shared with the JAX boundary-saving
# path; re-exported here so existing imports keep working.
from sweep.propagator._ring_geometry import (  # noqa: F401
    _interior_index_tuple,
    _ring_index_tuples,
    _ring_slice_shape,
)


# Low-precision storage for the saved boundary ring (compute stays FP32; only
# the stored values are compressed — same split as the CUDA `_bs` path, where
# `last_two`/the seed frame stay FP32 and only the per-step boundary band is
# down-cast / quantized).
_STORE_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_INT8_BLOCK = 256   # cells per FP32 scale (matches CUDA BOUNDARY_INT8_BLOCK)


def _quantize_int8(flat):
    """DeepWave-style symmetric per-block INT8 on a 1-D tensor: one FP32 scale
    per ``_INT8_BLOCK`` cells.  Returns (codes [int8, padded], scale [fp32])."""
    n = flat.numel()
    pad = (-n) % _INT8_BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, _INT8_BLOCK)
    scale = (blocks.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    codes = torch.round(blocks / scale).clamp_(-127, 127).to(torch.int8)
    return codes.reshape(-1), scale.squeeze(1)


def _dequantize_int8(codes, scale, n):
    blocks = codes.view(-1, _INT8_BLOCK).to(torch.float32) * scale.unsqueeze(1)
    return blocks.reshape(-1)[:n]


def _quantize_fp16(flat):
    """Per-block scaled FP16, mirroring the CUDA quantize_fp16_kernel: one FP32
    scale (= block max |val|) per ``_INT8_BLOCK`` cells, payload normalized to
    [-1, 1] before the fp16 cast.  A bare fp16 cast would flush everything
    below 2^-24 to zero — which wipes the velocity faces of elastic wavefields
    (values sit ~rho·vp below the stresses) and corrupts the gradient."""
    n = flat.numel()
    pad = (-n) % _INT8_BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, _INT8_BLOCK)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    codes = (blocks / scale).to(torch.float16)
    return codes.reshape(-1), scale.squeeze(1)


def _dequantize_fp16(codes, scale, n):
    blocks = codes.view(-1, _INT8_BLOCK).to(torch.float32) * scale.unsqueeze(1)
    return blocks.reshape(-1)[:n]


# ---- Ring storage ---------------------------------------------------------
# A step's saved ring -- the ring fields back to back -- is one row of a [T, n]
# buffer per part: the values (fp32/bf16), or the codes and the FP32 block
# scales (int8/fp16).  storage='gpu' keeps the rows on the compute device; 'cpu'
# keeps them in page-locked host memory and 'disk' in files, both copied on a
# side stream so the copies overlap the time steps, and fetched ahead of the
# reverse pass.


class _DeviceRows:
    """Rows in one place the copies block on: the compute device, pageable host
    memory, or a memory-mapped file."""

    def __init__(self, device, rows):
        self.device, self.rows = device, rows

    def save(self, j, parts):
        for row, part in zip(self.rows, parts):
            row[j].copy_(part)

    def load(self, j):
        return [row[j].to(self.device) for row in self.rows]


class _PinnedRows:
    """Rows in page-locked host memory, copied on a side stream: ``save``
    returns before its copy is done, and ``load`` starts copying the ``depth``
    steps the reverse pass reads next."""

    depth = 4

    def __init__(self, T, parts, device):
        self.device, self.ready = device, {}
        self.stream = torch.cuda.Stream(device)
        self.rows = [torch.empty((T, n), dtype=d, pin_memory=True) for n, d in parts]

    def host(self, j, write=False):
        return [row[j] for row in self.rows]

    def save(self, j, parts):
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            for row, part in zip(self.host(j, write=True), parts):
                row.copy_(part, non_blocking=True)
                part.record_stream(self.stream)
        self.saved(j)

    def load(self, j):
        for k in range(j, max(j - self.depth, -1), -1):
            if k not in self.ready:
                rows = self.host(k)
                with torch.cuda.stream(self.stream):
                    got = [row.to(self.device, non_blocking=True) for row in rows]
                    done = torch.cuda.Event()
                    done.record()
                self.ready[k] = got, done
                self.loaded(k, done)
        got, done = self.ready.pop(j)
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(done)
        for g in got:
            g.record_stream(stream)
        return got

    def saved(self, j):
        pass

    def loaded(self, j, done):
        pass


_STRIPES = 4    # files per part, written in parallel: a file takes one write at a time
_IO = None


def _io():
    """The background threads that write and read the 'disk' rows."""
    global _IO
    if _IO is None:
        _IO = ThreadPoolExecutor(_STRIPES, thread_name_prefix="sweep-bs-disk")
    return _IO


def _wait(futures):
    for f in futures:
        f.result()            # raises a failed write or read


class _DiskRows(_PinnedRows):
    """Rows in temporary files (already unlinked: they go with the rollout).
    Steps pass through page-locked slots of ``interval`` steps.  Once a slot
    is full, background threads write it out, a stripe of its rows to each
    file; in the reverse pass they read the chunks below the current one back
    into the slots already used."""

    def __init__(self, T, parts, device, directory=None, interval=32, slots=2):
        self.device, self.ready = device, {}
        self.stream = torch.cuda.Stream(device)
        self.T, self.K = T, interval
        self.depth = min(self.depth, interval)
        n = max(2, slots)
        self.slots = [[torch.empty((interval, m), dtype=d, pin_memory=True) for m, d in parts]
                      for _ in range(n)]
        self.bytes = [[t.view(torch.uint8).numpy() for t in slot] for slot in self.slots]
        self.files = [[tempfile.TemporaryFile(prefix="sweep_bs_", dir=directory) for _ in parts]
                      for _ in range(_STRIPES)]
        self.chunk = [None] * n               # the chunk each slot holds
        self.io = [[] for _ in range(n)]      # the slot's pending writes or reads
        self.free = [None] * n                # the slot's last copy to the device

    def host(self, j, write=False):
        c, r = divmod(j, self.K)
        s = c % len(self.slots)
        if self.chunk[s] != c:
            self._take(s, c, write)
        if not write:
            for p in range(c - 1, max(c - len(self.slots), -1), -1):
                if self.chunk[p % len(self.slots)] != p:
                    self._take(p % len(self.slots), p, False)
            _wait(self.io[s])
        return [part[r] for part in self.slots[s]]

    def _take(self, s, c, write):
        """Hand slot ``s`` to chunk ``c``: to be filled by the forward pass once
        its last writes are out, or read back from the files."""
        last, free = self.io[s], self.free[s]
        self.chunk[s] = c
        if write:
            _wait(last)
            self.io[s] = []
        else:
            self.io[s] = self._stripes(self._read, s, c, last, free)

    def saved(self, j):
        c, r = divmod(j, self.K)
        if r == self.K - 1 or j == self.T - 1:        # the slot is full: write it out
            done = torch.cuda.Event()
            done.record(self.stream)
            s = c % len(self.slots)
            self.io[s] = self._stripes(self._write, s, c, done)

    def loaded(self, j, done):
        self.free[j // self.K % len(self.slots)] = done

    def _stripes(self, fn, s, c, *args):
        """Run ``fn`` on each file's stripe of chunk ``c``'s rows."""
        rows = min(self.K, self.T - c * self.K)
        cut = [rows * k // _STRIPES for k in range(_STRIPES + 1)]
        return [_io().submit(fn, files, s, c * self.K, cut[k], cut[k + 1], *args)
                for k, files in enumerate(self.files) if cut[k + 1] > cut[k]]

    def _write(self, files, s, row0, a, b, done):
        done.synchronize()
        for f, part in zip(files, self.bytes[s]):
            _pwrite(f.fileno(), part[a:b], (row0 + a) * part.shape[1])

    def _read(self, files, s, row0, a, b, last, free):
        _wait(last)                # the slot's last writes (a failed one raises here)
        if free is not None:
            free.synchronize()     # the slot's copies to the device are done
        for f, part in zip(files, self.bytes[s]):
            _pread(f.fileno(), part[a:b], (row0 + a) * part.shape[1])


def _pwrite(fd, data, offset):
    view = memoryview(data).cast("B")
    while view:
        n = os.pwrite(fd, view, offset)
        view, offset = view[n:], offset + n


def _pread(fd, data, offset):
    view = memoryview(data).cast("B")
    while view:
        n = os.preadv(fd, [view], offset)
        if n == 0:
            raise EOFError("eager boundary saving: the ring file ends early")
        view, offset = view[n:], offset + n


def _mapped(T, n, dtype, directory):
    """A [T, n] host tensor kept in a temporary file (memory-mapped)."""
    fd, path = tempfile.mkstemp(prefix="sweep_bs_", dir=directory)
    try:
        os.ftruncate(fd, T * n * torch.empty((), dtype=dtype).element_size())
        return torch.from_file(path, shared=True, size=T * n, dtype=dtype).view(T, n)
    finally:
        os.close(fd)
        os.unlink(path)


def _ring_rows(storage, T, parts, device, pinned_memory=None, disk_dir=None,
               transfer_interval=None, ring_buffers=None):
    """The row store for ``storage`` ('gpu' | 'cpu' | 'disk'); ``parts`` lists
    each part's (row length, dtype)."""
    cuda = device.type == "cuda"
    if storage == "disk":
        if cuda:
            return _DiskRows(T, parts, device, disk_dir, transfer_interval or 32, ring_buffers or 2)
        return _DeviceRows(device, [_mapped(T, n, d, disk_dir) for n, d in parts])
    if storage == "cpu" and cuda and pinned_memory is not False:
        return _PinnedRows(T, parts, device)
    where = device if storage == "gpu" else torch.device("cpu")
    return _DeviceRows(device, [torch.empty((T, n), dtype=d, device=where) for n, d in parts])


def save_step(state, step, fields):
    """Save step ``step``'s ring of ``fields`` as one row, down-cast or
    quantized to ``state.store_dtype``."""
    runs = []
    for field in fields:
        runs += [field[idx].reshape(-1) for idx in state.rings]
        if state.pad is not None:
            runs.append(state.pad)
    flat = torch.cat(runs)
    if state.store_dtype == "int8":
        parts = _quantize_int8(flat)
    elif state.store_dtype == "fp16":
        parts = _quantize_fp16(flat)
    else:
        parts = (flat.to(_STORE_DTYPE[state.store_dtype]),)
    state.rows.save(step, parts)


def restore_step(state, step, pos, field):
    """Put field ``pos``'s saved ring at step ``step`` back into ``field``."""
    flat = state.decoded.get(step)
    if flat is None:
        parts = state.rows.load(step)
        if state.store_dtype == "int8":
            flat = _dequantize_int8(*parts, parts[0].numel())
        elif state.store_dtype == "fp16":
            flat = _dequantize_fp16(*parts, parts[0].numel())
        else:
            flat = parts[0]
        # The reverse pass walks down in time and reads this step and the one above.
        state.decoded = {k: v for k, v in state.decoded.items() if k == step + 1}
        state.decoded[step] = flat
    off = pos * state.stride
    for idx, shp, sz in zip(state.rings, state.ring_shapes, state.ring_sizes):
        field[idx] = flat[off:off + sz].view(shp).to(field.dtype)
        off += sz
    return field


def _interior_consistency_error(recomputed, frame, phys_indices, interior_idx):
    """Max over physical fields of the interior relative-L2 between the
    recomputed forward output ``func(reconstructed S_i)`` and the frame it must
    reproduce (``S_{i+1}``, with the source re-injected).

    This is exactly the property the autograd recompute relies on: the model
    gradient is the VJP of ``func`` linearised at the reconstructed state, so if
    that state re-runs forward to the SAME ``S_{i+1}`` across the lossless
    interior, the gradient is correct; if it does not, the reverse driver fails
    to invert the step and the gradient would be wrong.  Crucially this tolerates
    fields the reverse driver intentionally drops (e.g. swap2nd zeroes fields 2+)
    as long as ``func`` regenerates them — so it does NOT false-positive on
    multi-field 2nd-order equations like LSRTM, where a naive state round-trip
    would.

    Returns ``(error, assessed)``; ``assessed`` is False when every reference
    field norm is ~0 (a degenerate frame — nothing to check)."""
    worst = 0.0
    assessed = False
    for f in phys_indices:
        a = recomputed[f][interior_idx]
        b = frame[f][interior_idx]
        if not torch.isfinite(a).all():
            return float("inf"), True
        nb = b.norm().item()
        if nb < 1e-20:
            continue
        assessed = True
        worst = max(worst, (a - b).norm().item() / nb)
    return worst, assessed


class ReconState:
    """Per-rollout reconstruction state with PRE-ALLOCATED ring storage.

    Mirrors the CUDA boundary saver: the per-step boundary ring of every saved
    field lives in one big contiguous buffer per part, one row per step (see
    ``_ring_rows``), rather than a Python list of per-step tensors — so it
    avoids the caching allocator's 512-byte per-tensor rounding (which
    otherwise wastes memory on the many tiny per-step int8 scale tensors).
    ``device`` is the compute device: the zero placeholders and the
    reconstruction stay there wherever ``storage`` keeps the rows."""

    __slots__ = ("frame", "rings", "checked", "store_dtype", "ring_shapes",
                 "ring_sizes", "stride", "pad", "rows", "decoded", "zero")

    def __init__(self, nt, rings, shape, n_ring_fields, store_dtype, device,
                 storage="gpu", compute_dtype=torch.float32, field_shapes=None,
                 **storage_options):
        device = torch.device(device)
        self.frame = None
        # Reused read-only zero placeholders for the cpml / unpaired field slots in
        # backward, one per field -- one buffer per distinct shape (CPML strips
        # differ) for the whole rollout, not a per-step zeros_like. Matches the
        # wavefield (compute) dtype so it composes with the field ops.
        by_shape = {}
        self.zero = [by_shape.setdefault(tuple(s), torch.zeros(s, dtype=compute_dtype, device=device))
                     for s in (field_shapes or [shape])]
        self.rings = rings
        self.checked = False                         # one-time self-check latch
        self.store_dtype = store_dtype
        self.ring_shapes = [_ring_slice_shape(shape, idx) for idx in rings]
        self.ring_sizes = [math.prod(s) for s in self.ring_shapes]
        cells = sum(self.ring_sizes)
        # Scaled storage (int8, and fp16 sharing its layout -- see _quantize_fp16)
        # pads each field to whole blocks, so no block's scale spans two fields.
        scaled = store_dtype in ("int8", "fp16")
        self.stride = -(-cells // _INT8_BLOCK) * _INT8_BLOCK if scaled else cells
        self.pad = (torch.zeros(self.stride - cells, dtype=compute_dtype, device=device)
                    if self.stride > cells else None)
        n = n_ring_fields * self.stride
        if scaled:
            payload = torch.int8 if store_dtype == "int8" else torch.float16
            parts = [(n, payload), (n // _INT8_BLOCK, torch.float32)]
        else:
            parts = [(n, _STORE_DTYPE[store_dtype])]
        self.rows = _ring_rows(storage, nt + 1, parts, device, **storage_options)
        self.decoded = {}


# torch < 2.0 has no untyped_storage(); its storage() is the same buffer.
_storage = torch.Tensor.untyped_storage if hasattr(torch.Tensor, "untyped_storage") else torch.Tensor.storage


def _add_source(src, field, sample, inputs):
    """``field`` with the source ``sample`` added -- in place (a scatter onto a
    few cells) when ``field`` is a fresh step output, i.e. contiguous and
    sharing no storage with the step's ``inputs``; out of place otherwise."""
    ptr = _storage(field).data_ptr()
    if field.is_contiguous() and all(_storage(t).data_ptr() != ptr for t in inputs):
        return src.add_(field, sample)
    return src(field, sample)


# ---- Reverse-reconstruction drivers --------------------------------------
# Each driver reconstructs S_i (the step's input) from ``frame`` (= S_{i+1}) using
# forward physics ONLY — the gradient itself is autograd, in _BoundarySaveStep.
# Drivers are looked up by name (``cfg["reverse"]``) so a new time scheme can
# register one (``@register_reverse_driver``) without touching _BoundarySaveStep.
# A driver returns the reconstructed field list (saved ring restored, source
# re-injected, CPML zeroed in the lossless interior).
_REVERSE_DRIVERS = {}


def register_reverse_driver(name):
    """Register a reverse-reconstruction driver under ``name`` (the value used by
    ``enable_eager_boundary_saving(mode=name)`` / ``cfg['reverse']``)."""
    def _register(fn):
        _REVERSE_DRIVERS[name] = fn
        return fn
    return _register


@register_reverse_driver("swap2nd")
def _reconstruct_swap2nd(frame, cfg, st, step, time_index, zero):
    """2nd-order-in-time reverse step.  ``frame`` holds each physical field as a
    (now=u_{i+1}, prev=u_i) pair; swapping the two levels and running ``func`` ONCE
    yields u_{i-1} for EVERY pair at once — the coupled update evaluates each
    cross-field term (e.g. LSRTM's mp*background source for the scattered field) at
    the correct swapped levels, so multi-field equations reconstruct fully, not
    just field 0.  cpml + any unpaired field stay zeroed (lossless interior).  For
    a single-pair equation this is exactly the original swap2nd."""
    func, dt, h, models, nwf = cfg["func"], cfg["dt"], cfg["h"], cfg["models"], cfg["nwf"]
    pairs = cfg["pairs"]
    src_fields = set(cfg["source_indices"])
    src_op = cfg.get("src")
    swapped = list(zero)
    for now_i, prev_i in pairs:
        swapped[now_i] = frame[prev_i]               # u_i      -> now slot
        swapped[prev_i] = frame[now_i]               # u_{i+1}  -> prev slot
    out = list(func(swapped, models, dt, h, None))
    prev_step = max(step - 1, 0)
    S_i = list(zero)
    for pidx, (now_i, prev_i) in enumerate(pairs):
        u_i = frame[prev_i]
        u_im1 = out[now_i]                           # reconstructed previous level
        if now_i in src_fields and src_op is not None:
            u_im1 = _add_source(src_op, u_im1, cfg["wavelet"][time_index], swapped)
        S_i[now_i] = restore_step(st, step, pidx, u_i.clone())
        S_i[prev_i] = restore_step(st, prev_step, pidx, u_im1)
    return S_i


@register_reverse_driver("substep")
def _reconstruct_substep(frame, cfg, st, step, time_index, zero):
    """1st-order leapfrog reverse step.  Forward order was: step, then ADD source.
    Reverse: SUBTRACT source, zero CPML memory, then invert the leapfrog EXACTLY by
    composing the forward sub-steps (``cfg['substeps']``, built once / compiled if
    use_compile) in REVERSE order at -dt; finally restore the saved ring."""
    dt, h, models = cfg["dt"], cfg["h"], cfg["models"]
    cpml_idx, ring_fields = cfg["cpml_indices"], cfg["ring_fields"]
    frame_in = list(frame)
    src = cfg.get("src")
    if src is not None:
        wl = cfg["wavelet"][time_index]
        for sidx in cfg["source_indices"]:
            frame_in[sidx] = src(frame[sidx], -wl)   # out of place: frame is the saved state
    for c in cpml_idx:
        frame_in[c] = zero[c]
    state = frame_in
    for ss in reversed(cfg["substeps"]):
        state = ss(state, models, -dt, h)
    S_i = list(state)
    for c in cpml_idx:
        S_i[c] = zero[c]
    for j, f in enumerate(ring_fields):
        S_i[f] = restore_step(st, step, j, S_i[f])
    return S_i


class _BoundarySaveStep(torch.autograd.Function):
    """One time step ``S_out = equation.func(S_in, models)``, plus the source
    ``sample`` added to the source fields.

    Forward stores only the boundary ring of the physical fields (no full
    frame).  Backward reconstructs ``S_in`` by reverse-time marching from
    ``state.frame`` and re-runs the step under autograd for the gradients.
    """

    @staticmethod
    def forward(ctx, cfg, step, time_index, sample, *tensors):
        nwf, nm = cfg["nwf"], cfg["nm"]
        S_in = list(tensors[:nwf])
        models = list(tensors[nwf:nwf + nm])
        with torch.no_grad():
            S_out = list(cfg["func"](S_in, models, cfg["dt"], cfg["h"], None))
        save_step(cfg["state"], step + 1, [S_out[f] for f in cfg["ring_fields"]])
        # The ring is saved before the source goes in, as when the source was
        # an op after this Function; the reverse drivers re-inject it.  Inside,
        # it is a scatter onto the fresh outputs instead of a full-grid op.
        if sample is not None:
            for k in cfg["source_indices"]:
                S_out[k] = _add_source(cfg["src"], S_out[k], sample, S_in)
            ctx.sample_shape = sample.shape
        ctx.cfg = cfg
        ctx.step = step
        ctx.time_index = time_index
        return tuple(S_out)

    @staticmethod
    def backward(ctx, *grad_out):
        cfg = ctx.cfg
        if cfg is None:
            raise RuntimeError(
                "Eager boundary saving supports one backward pass per forward: the "
                "reconstruction has already consumed this rollout's boundary state.")
        # Drop the per-rollout state now.  Tensors cached by the propagator keep
        # this graph alive until its next forward, and the graph would otherwise
        # keep the boundary ring -- and, through cfg, the propagator itself --
        # resident with it.
        ctx.cfg = None
        st = cfg["state"]
        func, dt, h = cfg["func"], cfg["dt"], cfg["h"]
        nwf, nm = cfg["nwf"], cfg["nm"]
        i, time_index = ctx.step, ctx.time_index
        models = cfg["models"]
        frame = st.frame

        # Reverse-reconstruct S_i (the step input) from frame (= S_{i+1}) via the
        # registered driver — forward physics only; the gradient is autograd below.
        with torch.no_grad():
            S_i = _REVERSE_DRIVERS[cfg["reverse"]](frame, cfg, st, i, time_index, st.zero)
        st.frame = S_i

        # Re-run the (forward, +dt) step under autograd for input/model grads.
        # Only tensors that require grad may be passed to autograd.grad; models
        # that are held fixed (e.g. rho in an acoustic1st vp-only inversion) are
        # excluded and returned as None.
        S_i_rg = [t.detach().requires_grad_(True) for t in S_i]
        model_flags = [bool(m.requires_grad) for m in models]
        models_rg = [m.detach().requires_grad_(f) for m, f in zip(models, model_flags)]
        with torch.enable_grad():
            out_rg = func(S_i_rg, models_rg, dt, h, None)

        # One-time self-check (first backward step only): does the reverse driver
        # actually invert the step?  `out_rg` = func(reconstructed S_i) must
        # reproduce the frame we reconstructed FROM (S_{i+1}) across the lossless
        # interior, once the step's source is re-injected (the forward adds it
        # after `func`; `out_rg` is pre-source).  If not, the reconstructed state
        # is wrong and the gradient would be too — refuse loudly instead of
        # silently returning a bad gradient.  Reuses `out_rg` (already computed)
        # → ~free.  Forward-consistency (not state round-trip) is the right
        # criterion: it tolerates fields the driver drops but `func` regenerates.
        if cfg.get("self_check") and not st.checked:
            st.checked = True
            check = [o.detach() for o in out_rg]
            src_op = cfg.get("src")
            if src_op is not None:
                wl = cfg["wavelet"][time_index]
                for sidx in cfg["source_indices"]:
                    check[sidx] = src_op(check[sidx], wl)
            err, assessed = _interior_consistency_error(
                check, frame, cfg["phys_indices"], cfg["interior_idx"]
            )
            tol = cfg.get("check_tol", 1e-2)
            if assessed and not (err <= tol):
                raise RuntimeError(
                    f"Eager boundary saving: the '{cfg['reverse']}' reverse driver "
                    f"does not invert one step of {type(cfg['equation']).__name__} "
                    f"(interior forward-consistency rel-L2 = {err:.3g} > tol {tol:g}) "
                    "— the reconstructed wavefield would yield a WRONG gradient. "
                    "This equation is not supported by eager boundary saving; use "
                    "impl='c' or chunk checkpointing (use_ckpt=True). If you have "
                    "independently validated the gradient, skip this probe via "
                    "enable_eager_boundary_saving(self_check=False)."
                )

        diff_inputs = list(S_i_rg) + [m for m, f in zip(models_rg, model_flags) if f]
        grads = torch.autograd.grad(
            out_rg, diff_inputs, grad_outputs=grad_out,
            allow_unused=True, retain_graph=False,
        )
        grad_S = list(grads[:nwf])
        grad_M_iter = iter(grads[nwf:])
        grad_M = [next(grad_M_iter) if f else None for f in model_flags]
        # The source is additive: grad_S passes through it unchanged, and the
        # sample's gradient is grad_out read back at the source cells.
        grad_sample = None
        if ctx.needs_input_grad[3]:
            grad_sample = sum(cfg["src"].value_grad(grad_out[k], ctx.sample_shape)
                              for k in cfg["source_indices"])
        return tuple([None, None, None, grad_sample] + grad_S + grad_M)


# ---- Propagator integration -------------------------------------------
# Mixed into _PropTorchEager (resolved via MRO): wires the engine above
# (drivers, ReconState, _BoundarySaveStep) into the eager propagator —
# reverse-driver dispatch, per-rollout init, and the BS time loop.


class _EagerBoundarySavingMixin:
    """Eager boundary-saving methods for ``_PropTorchEager``."""

    def enable_eager_boundary_saving(self, flag=True, mode=None, self_check=True,
                                     check_tol=None, storage_dtype="fp32", storage="gpu",
                                     pinned_memory=None, disk_dir=None,
                                     transfer_interval=None, ring_buffers=None):
        """Route the eager rollout through pure-PyTorch boundary saving (see
        ``_eager_boundary_saving``) instead of the full autograd tape.  Mutually
        exclusive with ``use_ckpt``.  ``mode`` forces a reverse driver
        ('swap2nd' | 'substep'); None auto-dispatches.

        ``self_check`` (default True) runs a one-time interior forward-
        consistency probe on the first backward step: if the reverse driver does
        not invert the step (``func(reconstructed S_i)`` differs from ``S_{i+1}``
        in the lossless interior by rel-L2 > ``check_tol``), the propagator
        raises instead of returning a wrong gradient.  Set ``self_check=False``
        to skip the probe once an equation has been validated.

        ``storage_dtype`` ('fp32' | 'fp16' | 'bf16' | 'int8') compresses the
        per-step boundary ring only (compute and the seed frame stay FP32) —
        same split as the CUDA path.  ``storage`` ('gpu' | 'cpu' | 'disk')
        keeps the ring on device, in host RAM or in files — the ring is the
        dominant eager-BS GPU cost.  On a CUDA device, 'cpu' copies each step
        into page-locked memory on a side stream, overlapping the steps
        (``pinned_memory=False``: pageable memory, blocking copies); 'disk' goes
        through page-locked slots of ``transfer_interval`` steps (default 32;
        ``ring_buffers`` slots, default 2) that a background thread writes to
        temporary files in ``disk_dir`` (default: the system temp directory)
        and reads back ahead of the reverse pass.  ``check_tol=None`` picks a
        tolerance that clears the storage quantization floor while still
        catching a broken reverse (O(0.1-1) error)."""
        self._eager_bs = bool(flag)
        self._eager_bs_mode = mode
        self._eager_bs_self_check = bool(self_check)
        self._eager_bs_storage_dtype = storage_dtype
        self._eager_bs_storage = dict(storage=storage, pinned_memory=pinned_memory, disk_dir=disk_dir,
                                      transfer_interval=transfer_interval, ring_buffers=ring_buffers)
        if check_tol is None:
            check_tol = {"fp32": 1e-2, "fp16": 3e-2, "bf16": 6e-2, "int8": 1.5e-1}.get(storage_dtype, 1e-2)
        self._eager_bs_check_tol = float(check_tol)

    def _eager_bs_reverse_mode(self):
        """Pick (and validate) the exact reverse-reconstruction driver for eager
        boundary saving — dispatch rules live in the backend-shared
        ``_bs_dispatch`` module (same registry pattern as the JAX path).
        ``enable_eager_boundary_saving(mode=...)`` can force a mode."""
        from sweep.propagator._bs_dispatch import resolve_reverse_mode
        return resolve_reverse_mode(
            self.equation, _REVERSE_DRIVERS,
            forced=getattr(self, "_eager_bs_mode", None),
            free_surface=getattr(self, "free_surface", False),
            label="eager boundary saving",
            alternatives="impl='c' or chunk checkpointing (use_ckpt=True)",
        )

    def _init_eager_bs(self, wavefield, runtime_models, wavelet_steps, src, nt):
        """Set up the per-rollout state for eager (pure-PyTorch) boundary saving
        and return ``(state, base_cfg)``.

        Builds the ring geometry, resolves the reverse driver, pre-allocates the
        reconstruction storage (CUDA-style contiguous buffers) seeded with the
        initial frame, prepares the (optionally compiled) per-step physics
        callables, and assembles ``base_cfg`` — the part of the per-step config
        that does not change across time.  The caller passes the varying
        ``step``/``time_index`` to ``_BoundarySaveStep.apply``.  Reconstruction
        state is per-rollout (no class globals).  ``wavelet_steps`` is
        time-major: ``wavelet_steps[t]`` is step ``t``'s source sample.
        """
        from sweep.equations.base import SecondOrderEquation

        halo = self._runtime_fd_halo()
        offsets = tuple(self.abcn + halo for _ in range(self.ndim))
        shape = tuple(wavefield[0].shape)
        rings = _ring_index_tuples(shape, self.ndim, offsets, halo)
        interior_idx = _interior_index_tuple(shape, self.ndim, offsets)
        nwf = len(wavefield)
        nm = len(runtime_models)
        reverse_mode = self._eager_bs_reverse_mode()

        cpml_indices = [
            k for k, name in enumerate(self.wavefield_names)
            if getattr(self._wavefield_spec_index.get(name), "boundary_related", False)
        ]
        strips = self._strips
        if strips is not None:
            cpml_indices = [j for k in cpml_indices for j in strips.slots(k)]
        phys_indices = [k for k in range(nwf) if k not in cpml_indices]
        # 2nd-order wavefields are stored as consecutive (now, prev) physical
        # pairs.  swap2nd reconstructs EVERY pair — so a multi-field equation
        # (e.g. LSRTM: background h1/h2 + scattered sh1/sh2) recovers its full
        # physical state, not only field 0.  A single-field equation has one pair
        # (the original path).  ring_fields keys off the equation *type*, not the
        # reverse mode, so a forced mode cannot change which fields are stored.
        is_2nd = isinstance(self.equation, SecondOrderEquation)
        second_order_pairs = [
            (phys_indices[k], phys_indices[k + 1])
            for k in range(0, len(phys_indices) - 1, 2)
        ]
        ring_fields = [now for now, _ in second_order_pairs] if is_2nd else phys_indices

        # Pre-allocate the whole ring storage as big contiguous buffers (CUDA-
        # style), then write each step into a slice — no per-step allocation.
        # Seed step 0 with the initial frame.
        store_dtype = getattr(self, "_eager_bs_storage_dtype", "fp32")
        state = ReconState(nt, rings, shape, len(ring_fields), store_dtype,
                           wavefield[0].device,
                           compute_dtype=wavefield[0].dtype,
                           field_shapes=[tuple(w.shape) for w in wavefield],
                           **getattr(self, "_eager_bs_storage", {}))
        save_step(state, 0, [wavefield[f] for f in ring_fields])

        # Per-step physics callables, built ONCE and reused across every step + in
        # backward.  With use_compile=True the inner step is torch.compile'd; the
        # custom autograd.Function and the time loop stay eager (Dynamo treats the
        # Function as an opaque boundary), so only the per-step kernels are fused.
        bs_func = self._maybe_compile(self._grouped(self.equation.func))
        bs_substeps = (
            [self._maybe_compile(self._grouped(ss)) for ss in self.equation.interior_substeps()]
            if reverse_mode == "substep" else None
        )

        # Constant per-step config; step/time_index pass to apply() as args.
        base_cfg = {
            "equation": self.equation, "dt": self.dt, "h": self._equation_spacing,
            "nwf": nwf, "nm": nm, "nt": nt,
            "state": state, "models": runtime_models, "src": src,
            "source_indices": self.source_indices, "wavelet": wavelet_steps,
            "reverse": reverse_mode, "ring_fields": ring_fields,
            "cpml_indices": cpml_indices, "phys_indices": phys_indices,
            "pairs": second_order_pairs, "interior_idx": interior_idx,
            "self_check": getattr(self, "_eager_bs_self_check", True),
            "check_tol": getattr(self, "_eager_bs_check_tol", 1e-2),
            "func": bs_func, "substeps": bs_substeps,
        }
        return state, base_cfg

    def _grouped(self, fn):
        """``fn(wavefields, *args)`` on the flat eager state, its strip-held
        fields handed over grouped (CPML strips, ``StripLayout``)."""
        strips = self._strips
        if strips is None:
            return fn
        # A bound method is kept as function + object: Dynamo cannot guard on a
        # bound method held in a closure.
        call, obj = getattr(fn, "__func__", fn), getattr(fn, "__self__", None)
        if obj is None:
            return lambda wavefields, *args: strips.flatten(call(strips.group(wavefields), *args))
        return lambda wavefields, *args: strips.flatten(call(obj, strips.group(wavefields), *args))

    def _rollout_eager_bs(self, wavefield, runtime_models, wavelet_steps, src, rec,
                          receivers, nt, adj):
        """Time loop under eager (pure-PyTorch) boundary saving: each physics step
        is wrapped in a ``_BoundarySaveStep`` autograd.Function that stores only
        the boundary ring and adds the source; receiver sampling stays a normal
        op (its gather backward is value-free, so no full wavefield is
        retained).  Per-rollout setup lives in ``_init_eager_bs``; the loop passes
        the varying step/time straight to ``_BoundarySaveStep.apply``.  Seeds the
        reconstruction with the final frame and returns ``(wavefield, record)``,
        the record stacked (see ``_rollout_full``).
        """
        state, base_cfg = self._init_eager_bs(
            wavefield, runtime_models, wavelet_steps, src, nt
        )
        columns = []
        for i in range(nt):
            time = i if not adj else nt - i - 1
            wavefield = list(_BoundarySaveStep.apply(
                base_cfg, i, time, wavelet_steps[time], *wavefield, *runtime_models))
            columns.append(self._sample_receivers(rec, wavefield, receivers))
        # Seed the reconstruction with the final full frame (detached).
        state.frame = [w.detach() for w in wavefield]
        return wavefield, torch.stack(columns, dim=1)

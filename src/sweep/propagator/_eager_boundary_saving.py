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


def save_step(state, step, pos, field):
    """Consolidate this field's ring slices into one flat buffer and write it into
    the PRE-ALLOCATED per-rollout storage at ``[step, pos]`` (down-cast / int8
    quantized to ``state.store_dtype``).  No per-step tensor allocation — the
    persistent storage is a handful of big contiguous tensors (one ``buf`` for
    fp32/fp16/bf16, or ``codes``+``scale`` for int8), like the CUDA boundary
    saver — so it carries no per-tensor allocator rounding."""
    flat = torch.cat([field[idx].reshape(-1) for idx in state.rings])
    if state.store_dtype == "int8":
        codes, scale = _quantize_int8(flat)
        state.codes[step, pos].copy_(codes)
        state.scale[step, pos].copy_(scale)
    else:
        state.buf[step, pos].copy_(flat)             # copy_ casts fp32 -> store dtype


def restore_step(state, step, pos, field):
    """Read the saved ring buffer at ``[step, pos]``, up-cast / dequantize to the
    field's FP32 dtype, and split it back across ``field``'s ring slices."""
    if state.store_dtype == "int8":
        flat = _dequantize_int8(state.codes[step, pos], state.scale[step, pos], state.total_cells)
    else:
        flat = state.buf[step, pos]
    if flat.device != field.device:                  # offloaded ring (cpu) -> compute device
        flat = flat.to(field.device, non_blocking=True)
    off = 0
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
    field lives in ONE big contiguous buffer indexed by ``[step, ring_field]``
    (a single ``buf`` for fp32/fp16/bf16, or ``codes``+``scale`` for int8),
    rather than a Python list of per-step tensors — so it avoids the caching
    allocator's 512-byte per-tensor rounding (which otherwise wastes memory on
    the many tiny per-step int8 scale tensors)."""

    __slots__ = ("frame", "rings", "checked", "store_dtype", "ring_shapes",
                 "ring_sizes", "total_cells", "buf", "codes", "scale", "zero")

    def __init__(self, nt, rings, shape, n_ring_fields, store_dtype, device,
                 storage_device=None, compute_dtype=torch.float32):
        # ``device`` is the compute device (where the wavefields live); the ring
        # storage lives on ``storage_device`` instead (e.g. host RAM for the 'cpu'
        # offload — trading device memory for per-step H2D/D2H copies).  The zero
        # placeholder and reconstruction run on the compute device regardless.
        storage_device = device if storage_device is None else storage_device
        self.frame = None
        # Reused read-only zero placeholder for the cpml / unpaired field slots in
        # backward — one buffer for the whole rollout, not a per-step zeros_like.
        # Matches the wavefield (compute) dtype so it composes with the field ops.
        self.zero = torch.zeros(shape, dtype=compute_dtype, device=device)
        self.rings = rings
        self.checked = False                         # one-time self-check latch
        self.store_dtype = store_dtype
        self.ring_shapes = [_ring_slice_shape(shape, idx) for idx in rings]
        self.ring_sizes = [math.prod(s) for s in self.ring_shapes]
        self.total_cells = sum(self.ring_sizes)
        T, F = nt + 1, n_ring_fields
        self.buf = self.codes = self.scale = None
        if store_dtype == "int8":
            padded = ((self.total_cells + _INT8_BLOCK - 1) // _INT8_BLOCK) * _INT8_BLOCK
            self.codes = torch.zeros((T, F, padded), dtype=torch.int8, device=storage_device)
            self.scale = torch.zeros((T, F, padded // _INT8_BLOCK), dtype=torch.float32, device=storage_device)
        else:
            self.buf = torch.zeros((T, F, self.total_cells),
                                   dtype=_STORE_DTYPE[store_dtype], device=storage_device)


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
    swapped = [zero] * nwf
    for now_i, prev_i in pairs:
        swapped[now_i] = frame[prev_i]               # u_i      -> now slot
        swapped[prev_i] = frame[now_i]               # u_{i+1}  -> prev slot
    out = list(func(swapped, models, dt, h, None))
    prev_step = max(step - 1, 0)
    S_i = [zero] * nwf
    for pidx, (now_i, prev_i) in enumerate(pairs):
        u_i = frame[prev_i]
        u_im1 = out[now_i]                           # reconstructed previous level
        if now_i in src_fields and src_op is not None:
            u_im1 = src_op(u_im1, cfg["wavelet"][..., time_index])
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
        wl = cfg["wavelet"][..., time_index]
        for sidx in cfg["source_indices"]:
            frame_in[sidx] = src(frame[sidx].clone(), -wl)
    for c in cpml_idx:
        frame_in[c] = zero
    state = frame_in
    for ss in reversed(cfg["substeps"]):
        state = ss(state, models, -dt, h)
    S_i = list(state)
    for c in cpml_idx:
        S_i[c] = zero
    for j, f in enumerate(ring_fields):
        S_i[f] = restore_step(st, step, j, S_i[f])
    return S_i


class _BoundarySaveStep(torch.autograd.Function):
    """One time step ``S_out = equation.func(S_in, models)``.

    Forward stores only the boundary ring of the physical fields (no full
    frame).  Backward reconstructs ``S_in`` by reverse-time marching from
    ``state.frame`` and re-runs the step under autograd for the gradients.
    """

    @staticmethod
    def forward(ctx, cfg, step, time_index, *tensors):
        nwf, nm = cfg["nwf"], cfg["nm"]
        S_in = list(tensors[:nwf])
        models = list(tensors[nwf:nwf + nm])
        with torch.no_grad():
            S_out = list(cfg["func"](S_in, models, cfg["dt"], cfg["h"], None))
        st = cfg["state"]
        for pos, f in enumerate(cfg["ring_fields"]):
            save_step(st, step + 1, pos, S_out[f])
        ctx.cfg = cfg
        ctx.step = step
        ctx.time_index = time_index
        return tuple(S_out)

    @staticmethod
    def backward(ctx, *grad_out):
        cfg = ctx.cfg
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
                wl = cfg["wavelet"][..., time_index]
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
        return tuple([None, None, None] + grad_S + grad_M)


# ---- Propagator integration -------------------------------------------
# Mixed into _PropTorchEager (resolved via MRO): wires the engine above
# (drivers, ReconState, _BoundarySaveStep) into the eager propagator —
# reverse-driver dispatch, per-rollout init, and the BS time loop.


class _EagerBoundarySavingMixin:
    """Eager boundary-saving methods for ``_PropTorchEager``."""

    def enable_eager_boundary_saving(self, flag=True, mode=None, self_check=True,
                                     check_tol=None, storage_dtype="fp32", storage="gpu"):
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
        same split as the CUDA path.  ``storage`` ('gpu' | 'cpu') keeps the ring
        buffer on device or offloads it to host RAM — the ring is the dominant
        eager-BS GPU cost, so 'cpu' trades device memory for per-step H2D/D2H
        copies.  ``check_tol=None`` picks a tolerance that clears the storage
        quantization floor while still catching a broken reverse (O(0.1-1)
        error)."""
        self._eager_bs = bool(flag)
        self._eager_bs_mode = mode
        self._eager_bs_self_check = bool(self_check)
        self._eager_bs_storage_dtype = storage_dtype
        self._eager_bs_storage = storage
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

    def _init_eager_bs(self, wavefield, runtime_models, wavelet, src, nt):
        """Set up the per-rollout state for eager (pure-PyTorch) boundary saving
        and return ``(state, base_cfg, multi_receiver)``.

        Builds the ring geometry, resolves the reverse driver, pre-allocates the
        reconstruction storage (CUDA-style contiguous buffers) seeded with the
        initial frame, prepares the (optionally compiled) per-step physics
        callables, and assembles ``base_cfg`` — the part of the per-step config
        that does not change across time.  The caller passes the varying
        ``step``/``time_index`` to ``_BoundarySaveStep.apply``.  Reconstruction
        state is per-rollout (no class globals).
        """
        from sweep.equations.base import SecondOrderEquation

        multi_receiver = len(self.receiver_indices) > 1
        halo = self._runtime_fd_halo()
        offsets = tuple(self.abcn + halo for _ in range(self.ndim))
        shape = tuple(wavefield[0].shape)
        rings = _ring_index_tuples(shape, self.ndim, offsets, halo)
        interior_idx = _interior_index_tuple(shape, self.ndim, offsets)
        nwf = len(self.wavefield_names)
        nm = len(runtime_models)
        reverse_mode = self._eager_bs_reverse_mode()

        cpml_indices = [
            k for k, name in enumerate(self.wavefield_names)
            if getattr(self._wavefield_spec_index.get(name), "boundary_related", False)
        ]
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
        storage_loc = getattr(self, "_eager_bs_storage", "gpu")
        storage_device = wavefield[0].device if storage_loc == "gpu" else torch.device("cpu")
        state = ReconState(nt, rings, shape, len(ring_fields), store_dtype,
                           wavefield[0].device, storage_device,
                           compute_dtype=wavefield[0].dtype)
        for pos, f in enumerate(ring_fields):
            save_step(state, 0, pos, wavefield[f])

        # Per-step physics callables, built ONCE and reused across every step + in
        # backward.  With use_compile=True the inner step is torch.compile'd; the
        # custom autograd.Function and the time loop stay eager (Dynamo treats the
        # Function as an opaque boundary), so only the per-step kernels are fused.
        bs_func = self._maybe_compile(self.equation.func)
        bs_substeps = (
            [self._maybe_compile(ss) for ss in self.equation.interior_substeps()]
            if reverse_mode == "substep" else None
        )

        # Constant per-step config; step/time_index pass to apply() as args.
        base_cfg = {
            "equation": self.equation, "dt": self.dt, "h": self._equation_spacing,
            "nwf": nwf, "nm": nm, "nt": nt,
            "state": state, "models": runtime_models, "src": src,
            "source_indices": self.source_indices, "wavelet": wavelet,
            "reverse": reverse_mode, "ring_fields": ring_fields,
            "cpml_indices": cpml_indices, "phys_indices": phys_indices,
            "pairs": second_order_pairs, "interior_idx": interior_idx,
            "self_check": getattr(self, "_eager_bs_self_check", True),
            "check_tol": getattr(self, "_eager_bs_check_tol", 1e-2),
            "func": bs_func, "substeps": bs_substeps,
        }
        return state, base_cfg, multi_receiver

    def _rollout_eager_bs(self, wavefield, runtime_models, wavelet, src, rec,
                          record, receivers, nt, adj):
        """Time loop under eager (pure-PyTorch) boundary saving: each physics step
        is wrapped in a ``_BoundarySaveStep`` autograd.Function that stores only
        the boundary ring (source injection + receiver sampling stay normal ops —
        their gather/scatter backward is value-free, so no full wavefield is
        retained).  Per-rollout setup lives in ``_init_eager_bs``; the loop passes
        the varying step/time straight to ``_BoundarySaveStep.apply``.  Writes the
        record in place, seeds the reconstruction with the final frame, and
        returns the final wavefield list.
        """
        state, base_cfg, multi_receiver = self._init_eager_bs(
            wavefield, runtime_models, wavelet, src, nt
        )
        for i in range(nt):
            time = i if not adj else nt - i - 1
            wavefield = list(_BoundarySaveStep.apply(base_cfg, i, time, *wavefield, *runtime_models))
            for source_idx in self.source_indices:
                wavefield[source_idx] = src(wavefield[source_idx], wavelet[..., time])
            if multi_receiver:
                record[:, i, :, :] = rec.sample_fields([wavefield[idx] for idx in self.receiver_indices])
            else:
                receiver_idx = self.receiver_indices[0]
                record[:, i, :, 0] = rec(wavefield[receiver_idx]).view(*receivers.shape[:-1])
        # Seed the reconstruction with the final full frame (detached).
        state.frame = [w.detach() for w in wavefield]
        return wavefield

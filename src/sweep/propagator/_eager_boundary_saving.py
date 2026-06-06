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

import torch


def _ring_index_tuples(shape, ndim, offsets, halo):
    """Index tuples for the ``halo``-wide innermost-PML ring on both sides of
    every spatial axis (the cells the interior stencil reads)."""
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    rings = []
    for ai, ax in enumerate(spatial_axes):
        axis_len = shape[ax]
        off = offsets[ai]
        low = slice(off - halo, off)
        high = slice(axis_len - off, axis_len - off + halo)
        for sl in (low, high):
            idx = [slice(None)] * len(shape)
            idx[ax] = sl
            rings.append(tuple(idx))
    return rings


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


def save_ring(field, rings, store_dtype="fp32"):
    """Save the ring cells, optionally compressed.  ALL of this field's ring
    slices are concatenated into ONE flat buffer before storing (a single tensor
    for fp32/fp16/bf16, or a single codes+scale pair for int8) — this keeps the
    saved-state tensor count ~ndim*2x lower, which matters because the CUDA
    caching allocator rounds every allocation up to a 512-byte block: many tiny
    per-slice tensors (esp. the int8 FP32 scales) would otherwise waste ~512 B
    each.  Returns a tagged tuple carrying the per-slice shapes for restore."""
    slices = [field[idx] for idx in rings]
    shapes = [tuple(s.shape) for s in slices]
    sizes = [s.numel() for s in slices]
    flat = torch.cat([s.reshape(-1) for s in slices])    # contiguous copy (no alias)
    if store_dtype == "int8":
        codes, scale = _quantize_int8(flat)
        return ("int8", codes, scale, flat.numel(), shapes, sizes)
    if store_dtype in ("fp16", "bf16"):
        return ("cast", flat.to(_STORE_DTYPE[store_dtype]), shapes, sizes)
    return ("cast", flat, shapes, sizes)                 # fp32


def restore_ring(field, saved, rings):
    """Write a saved (consolidated) ring buffer back, up-casting / dequantizing
    to the field's (FP32) dtype and splitting it across the ring slices."""
    if saved[0] == "int8":
        _, codes, scale, n, shapes, sizes = saved
        flat = _dequantize_int8(codes, scale, n)
    else:
        _, flat, shapes, sizes = saved
    off = 0
    for idx, shp, sz in zip(rings, shapes, sizes):
        field[idx] = flat[off:off + sz].view(shp).to(field.dtype)
        off += sz
    return field


def _interior_index_tuple(shape, ndim, offsets):
    """Index tuple for the lossless deep interior — strictly inside the PML +
    stencil halo on every spatial axis (``[offset : N-offset]``), where the
    reverse reconstruction is exact (CPML is zeroed there and the restored ring
    sits outside it)."""
    idx = [slice(None)] * len(shape)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    for ai, ax in enumerate(spatial_axes):
        off = offsets[ai]
        idx[ax] = slice(off, shape[ax] - off)
    return tuple(idx)


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
    """Per-rollout reconstruction state.  Replaces seistorch's class globals."""

    __slots__ = ("frame", "cur_strip", "rings", "checked")

    def __init__(self, nt, rings):
        self.frame = None
        # cur_strip[k] = [save_ring(S_k[f]) for f in ring_fields], k = 0..nt
        self.cur_strip = [None] * (nt + 1)
        self.rings = rings
        # One-time self-check latch (first backward step probes the reverse).
        self.checked = False


class _BoundarySaveStep(torch.autograd.Function):
    """One time step ``S_out = equation.func(S_in, models)``.

    Forward stores only the boundary ring of the physical fields (no full
    frame).  Backward reconstructs ``S_in`` by reverse-time marching from
    ``state.frame`` and re-runs the step under autograd for the gradients.
    """

    @staticmethod
    def forward(ctx, cfg, *tensors):
        nwf, nm = cfg["nwf"], cfg["nm"]
        S_in = list(tensors[:nwf])
        models = list(tensors[nwf:nwf + nm])
        with torch.no_grad():
            S_out = list(cfg["func"](S_in, models, cfg["dt"], cfg["h"], None))
        st = cfg["state"]
        st.cur_strip[cfg["step"] + 1] = [
            save_ring(S_out[f], st.rings, cfg.get("store_dtype", "fp32"))
            for f in cfg["ring_fields"]
        ]
        ctx.cfg = cfg
        return tuple(S_out)

    @staticmethod
    def backward(ctx, *grad_out):
        cfg = ctx.cfg
        st = cfg["state"]
        func, dt, h = cfg["func"], cfg["dt"], cfg["h"]
        nwf, nm, i = cfg["nwf"], cfg["nm"], cfg["step"]
        models = cfg["models"]
        rings = st.rings
        ring_fields = cfg["ring_fields"]
        cpml_idx = cfg["cpml_indices"]
        frame = st.frame
        zero = torch.zeros_like(frame[0])

        def reinject_source(field):
            src = cfg.get("src")
            if src is None:
                return field
            return src(field, cfg["wavelet"][..., cfg["time_index"]])

        with torch.no_grad():
            if cfg["reverse"] == "swap2nd":
                # frame holds each 2nd-order field as (now=u_{i+1}, prev=u_i).
                # Reverse EVERY physical (now, prev) pair at once: swap each pair's
                # two levels and run `func` a SINGLE time.  The coupled update then
                # evaluates every cross-field term (e.g. LSRTM's mp*background
                # source for the scattered field) at the correct swapped levels,
                # so all pairs reconstruct together — not just field 0.  cpml +
                # any unpaired field stay zeroed (lossless interior).  For a
                # single-pair equation this is exactly the original swap2nd.
                pairs = cfg["pairs"]
                src_fields = set(cfg["source_indices"])
                swapped = [zero] * nwf
                for now_i, prev_i in pairs:
                    swapped[now_i] = frame[prev_i]    # u_i      -> now slot
                    swapped[prev_i] = frame[now_i]    # u_{i+1}  -> prev slot
                out = list(func(swapped, models, dt, h, None))
                prev_strip = st.cur_strip[i - 1] if i - 1 >= 0 else st.cur_strip[0]
                S_i = [zero] * nwf
                for pidx, (now_i, prev_i) in enumerate(pairs):
                    u_i = frame[prev_i]
                    u_im1 = out[now_i]                # reconstructed previous level
                    if now_i in src_fields:
                        u_im1 = reinject_source(u_im1)
                    S_i[now_i] = restore_ring(u_i.clone(), st.cur_strip[i][pidx], rings)
                    S_i[prev_i] = restore_ring(u_im1, prev_strip[pidx], rings)
            else:  # 1st-order leapfrog — 'substep' (exact, reuses forward sub-steps)
                # Forward order was: step, then ADD source to the source field.
                # Reverse: SUBTRACT source, zero CPML memory, then invert the
                # leapfrog EXACTLY by composing the forward sub-steps (built once,
                # compiled if use_compile) in REVERSE order at -dt.
                frame_in = list(frame)
                src = cfg.get("src")
                if src is not None:
                    wl = cfg["wavelet"][..., cfg["time_index"]]
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
                    S_i[f] = restore_ring(S_i[f], st.cur_strip[i][j], rings)

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
                wl = cfg["wavelet"][..., cfg["time_index"]]
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
        return tuple([None] + grad_S + grad_M)

"""Boundary-saving wavefield reconstruction for the JAX propagator.

Functional twin of ``_eager_boundary_saving`` (the pure-PyTorch path): the
forward ``lax.scan`` stores only the boundary ring of the physical fields each
step (stacked for free by the scan), and a ``jax.custom_vjp`` replaces the
stored-everything autodiff backward with a reverse-time scan that reconstructs
the forward trajectory step by step and applies the per-step VJP at the
reconstructed state.

**Important framing** (same as the eager path): the *adjoint* is 100% JAX
autodiff — ``jax.vjp`` of the unmodified forward step, nothing hand-written.
The only equation-specific piece is the *primal* reverse-time reconstruction,
covered by two generic drivers dispatched on the equation's time scheme:

* ``reverse='swap2nd'`` — 2nd-order-in-time schemes (``SecondOrderEquation``,
  e.g. ``Acoustic``).  ``u_{t+1} = 2u_t - u_{t-1} + dt^2 L(u_t)`` is symmetric
  in the two time levels, so the reverse step is ``func`` with the levels
  swapped.  EXACT, no per-equation code.

* ``reverse='substep'`` — 1st-order leapfrog schemes (``FirstOrderEquation``,
  e.g. ``Acoustic1st``/``Elastic``).  The equation exposes its step as an
  ordered list of reversible sub-steps via ``interior_substeps()``; the exact
  inverse is those sub-steps composed in REVERSE order at ``-dt``.  1st-order
  equations without ``interior_substeps`` are NOT supported (no approximate
  fallback).

Both zero the CPML memory variables during reconstruction (exact in the
lossless interior, where the model gradient lives; small near-PML error — same
design as the CUDA ``_bs`` and eager paths) and re-inject the saved boundary
ring + the source.  Ring geometry comes from the shared backend-agnostic
``_ring_geometry`` module, so eager and JAX save byte-identical strips.

Memory: ``O(nt * halo * perimeter)`` residuals instead of the ``O(nt * Ncells
* nfields)`` scan tape; compute is ~3x a forward step per reverse step (one
reconstruction + one linearisation + one VJP).
"""

import jax
import jax.numpy as jnp

from sweep.propagator._ring_geometry import _ring_index_tuples, _ring_layout

_STORE_DTYPE = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}


# ---- ring I/O (functional) -------------------------------------------------

def _extract_ring(field, rings):
    """Concatenate this field's ring slices into one flat vector."""
    return jnp.concatenate([field[idx].reshape(-1) for idx in rings])


def _write_ring(field, flat, rings, ring_shapes, ring_sizes):
    """Overwrite ``field``'s ring slices with the saved flat vector."""
    off = 0
    for idx, shp, sz in zip(rings, ring_shapes, ring_sizes):
        field = field.at[idx].set(flat[off:off + sz].reshape(shp).astype(field.dtype))
        off += sz
    return field


# ---- Reverse-reconstruction drivers ----------------------------------------
# Each driver reconstructs S_i (the step's input) from ``frame`` (= S_{i+1})
# using forward physics ONLY — the gradient itself is jax.vjp in the backward
# scan.  ``ring_t``/``ring_tm1`` are the [F, cells] storage rows for steps i
# and i-1 (already upcast to the compute dtype).  Drivers are pure functions of
# their inputs, so they trace once inside the reverse scan body.
_REVERSE_DRIVERS = {}


def register_reverse_driver(name):
    """Register a reverse-reconstruction driver under ``name`` (the value used
    by ``cfg['reverse']`` / a forced ``mode=``)."""
    def _register(fn):
        _REVERSE_DRIVERS[name] = fn
        return fn
    return _register


@register_reverse_driver("swap2nd")
def _reconstruct_swap2nd(frame, ring_t, ring_tm1, w_t, models, cfg):
    """2nd-order-in-time reverse step.  ``frame`` holds each physical field as
    a (now=u_{i+1}, prev=u_i) pair; swapping the two levels and running
    ``func`` ONCE yields u_{i-1} for EVERY pair at once (coupled cross-field
    terms evaluate at the correct swapped levels, so multi-field equations like
    LSRTM reconstruct fully).  cpml + any unpaired field stay zeroed (lossless
    interior); the saved ring is then re-imposed and the source re-injected."""
    func, dt, h, nwf = cfg["func"], cfg["dt"], cfg["h"], cfg["nwf"]
    rings, shapes, sizes = cfg["ring_layout"]
    src = cfg["src"]
    src_fields = cfg["source_index_set"]
    zero = jnp.zeros_like(frame[0])
    swapped = [zero] * nwf
    for now_i, prev_i in cfg["pairs"]:
        swapped[now_i] = frame[prev_i]               # u_i      -> now slot
        swapped[prev_i] = frame[now_i]               # u_{i+1}  -> prev slot
    out = func(tuple(swapped), models, dt, h, None)
    S_i = [zero] * nwf
    for pidx, (now_i, prev_i) in enumerate(cfg["pairs"]):
        u_i = frame[prev_i]
        u_im1 = out[now_i]                           # reconstructed previous level
        if now_i in src_fields and src is not None:
            u_im1 = src(u_im1, w_t)
        S_i[now_i] = _write_ring(u_i, ring_t[pidx], rings, shapes, sizes)
        S_i[prev_i] = _write_ring(u_im1, ring_tm1[pidx], rings, shapes, sizes)
    return tuple(S_i)


@register_reverse_driver("substep")
def _reconstruct_substep(frame, ring_t, ring_tm1, w_t, models, cfg):
    """1st-order leapfrog reverse step.  Forward order was: step, then ADD
    source.  Reverse: SUBTRACT source, zero CPML memory, invert the leapfrog
    EXACTLY by composing the forward sub-steps in REVERSE order at ``-dt``,
    re-zero CPML, then restore the saved ring."""
    dt, h = cfg["dt"], cfg["h"]
    rings, shapes, sizes = cfg["ring_layout"]
    cpml_idx, ring_fields = cfg["cpml_indices"], cfg["ring_fields"]
    src = cfg["src"]
    zero = jnp.zeros_like(frame[0])
    state = list(frame)
    if src is not None:
        for sidx in cfg["source_indices"]:
            state[sidx] = src(state[sidx], -w_t)
    for c in cpml_idx:
        state[c] = zero
    for ss in cfg["substeps"][::-1]:
        state = list(ss(state, models, -dt, h))
    for c in cpml_idx:
        state[c] = zero
    for j, f in enumerate(ring_fields):
        state[f] = _write_ring(state[f], ring_t[j], rings, shapes, sizes)
    return tuple(state)


# ---- driver resolution ------------------------------------------------------

def resolve_reverse_mode(equation, forced=None, free_surface=False):
    """Pick (and validate) the reverse driver for this equation — dispatch
    rules live in the backend-shared ``_bs_dispatch`` module."""
    from sweep.propagator._bs_dispatch import resolve_reverse_mode as _resolve
    return _resolve(
        equation, _REVERSE_DRIVERS, forced=forced, free_surface=free_surface,
        label="boundary saving",
        alternatives="chunk checkpointing (use_ckpt=True)",
    )


# ---- config -----------------------------------------------------------------

def build_config(*, equation, wavefield_specs, wavefield_names, dt, h, shape,
                 ndim, abcn, halo, nt, S0, src, rec, source_indices,
                 receiver_indices, batch_size, free_surface=False,
                 forced_mode=None, storage_dtype="fp32", scan_unroll=1):
    """Assemble the static per-rollout config consumed by ``make_propagate``.

    Mirrors ``PropTorch._init_eager_bs``: ring geometry from the spatial order
    + PML width, CPML fields from the ``boundary_related`` FieldSpec flag,
    2nd-order fields as consecutive (now, prev) physical pairs."""
    if storage_dtype not in _STORE_DTYPE:
        raise NotImplementedError(
            f"JAX boundary saving supports storage_dtype 'fp32'/'fp16'/'bf16' "
            f"(got {storage_dtype!r}); int8 is only on the torch paths."
        )
    from sweep.equations.base import SecondOrderEquation

    offsets = tuple(abcn + halo for _ in range(ndim))
    rings = _ring_index_tuples(shape, ndim, offsets, halo)
    _, ring_shapes, ring_sizes, _ = _ring_layout(shape, ndim, offsets, halo)
    nwf = len(wavefield_names)

    cpml_indices = [
        k for k, name in enumerate(wavefield_names)
        if getattr(wavefield_specs.get(name), "boundary_related", False)
    ]
    phys_indices = [k for k in range(nwf) if k not in cpml_indices]
    is_2nd = isinstance(equation, SecondOrderEquation)
    pairs = [
        (phys_indices[k], phys_indices[k + 1])
        for k in range(0, len(phys_indices) - 1, 2)
    ]
    ring_fields = [now for now, _ in pairs] if is_2nd else phys_indices

    reverse_mode = resolve_reverse_mode(equation, forced_mode, free_surface)
    substeps = list(equation.interior_substeps()) if reverse_mode == "substep" else None

    return {
        "func": equation.func, "substeps": substeps, "reverse": reverse_mode,
        "dt": dt, "h": h, "nwf": nwf, "nt": nt, "S0": tuple(S0),
        "src": src, "rec": rec,
        "source_indices": list(source_indices),
        "source_index_set": set(source_indices),
        "receiver_indices": list(receiver_indices),
        "batch_size": batch_size,
        "rings": rings, "ring_layout": (rings, ring_shapes, ring_sizes),
        "ring_fields": ring_fields, "pairs": pairs,
        "cpml_indices": cpml_indices, "phys_indices": phys_indices,
        "store_dtype": _STORE_DTYPE[storage_dtype],
        "scan_unroll": int(scan_unroll),
    }


# ---- the boundary-saving propagate ------------------------------------------

def _inject(S_list, w_t, cfg):
    """Add the source sample to the source fields (forward order: after func)."""
    src = cfg["src"]
    for sidx in cfg["source_indices"]:
        S_list[sidx] = src(S_list[sidx], w_t)
    return S_list


def _sample(S_list, cfg):
    """Receiver sampling — same gather/transpose as PropJax.forward_base, so
    rec_t matches the plain-autodiff path bit for bit."""
    rec = cfg["rec"]
    batch_size = cfg["batch_size"]
    wf_sel = [S_list[i] for i in cfg["receiver_indices"]]

    def one_channel(wf):
        y = rec(wf)
        return y.reshape(batch_size, -1, y.shape[-1])

    all_rec = jax.vmap(one_channel)(jnp.stack(wf_sel, axis=0))
    return jnp.transpose(all_rec, (1, 2, 0, 3))[..., 0]


def _extract_ring_row(S, cfg):
    """[F, total_cells] ring row of the to-be-saved fields, in storage dtype."""
    return jnp.stack(
        [_extract_ring(S[f], cfg["rings"]) for f in cfg["ring_fields"]]
    ).astype(cfg["store_dtype"])


def make_propagate(cfg):
    """Build ``propagate(models, wavelet) -> rec_seq [nt, B, nrec, ntype]``
    with a boundary-saving custom VJP.

    Forward: one ``lax.scan`` whose per-step output is (receiver row, ring
    row) — the scan stacks the rings into the ``[nt+1, F, cells]`` storage for
    free.  The ring is taken from the ``func`` output BEFORE source injection
    (same convention as the eager/CUDA savers; the drivers re-inject).

    Backward: a reverse-time ``lax.scan`` carrying ``(frame, adjoint state,
    model-grad accumulator)``.  Each step reconstructs ``S_t`` from
    ``frame = S_{t+1}`` via the reverse driver, then applies ``jax.vjp`` of the
    unmodified forward step at the reconstructed state.  Per-step wavelet
    cotangents come back through the scan's stacked outputs.
    """
    nt = cfg["nt"]
    dt, h = cfg["dt"], cfg["h"]
    func = cfg["func"]
    unroll = cfg.get("scan_unroll", 1)

    def step_with_rec(S, models, w_t):
        S_mid = func(S, models, dt, h, None)
        S1 = _inject(list(S_mid), w_t, cfg)
        return tuple(S1), _sample(S1, cfg)

    def fwd_scan(models, wavelet):
        def body(S, t):
            w_t = wavelet[..., t]
            S_mid = func(S, models, dt, h, None)
            ring_row = _extract_ring_row(S_mid, cfg)
            S1 = _inject(list(S_mid), w_t, cfg)
            return tuple(S1), (_sample(S1, cfg), ring_row)

        S_final, (rec_seq, ring_rows) = jax.lax.scan(
            body, cfg["S0"], jnp.arange(nt), unroll=unroll
        )
        return rec_seq, ring_rows, S_final

    @jax.custom_vjp
    def propagate(models, wavelet):
        rec_seq, _, _ = fwd_scan(models, wavelet)
        return rec_seq

    def propagate_fwd(models, wavelet):
        rec_seq, ring_rows, S_final = fwd_scan(models, wavelet)
        # Storage row s holds the ring of the PRE-SOURCE field entering step s
        # (row 0 = initial frame), matching the eager saver's [step] indexing.
        ring0 = _extract_ring_row(cfg["S0"], cfg)
        rings_full = jnp.concatenate([ring0[None], ring_rows], axis=0)
        return rec_seq, (rings_full, S_final, models, wavelet)

    def propagate_bwd(res, g_rec_seq):
        rings_full, S_final, models, wavelet = res
        driver = _REVERSE_DRIVERS[cfg["reverse"]]
        compute_dtype = S_final[0].dtype

        def body(carry, t):
            frame, S_bar, g_models = carry
            w_t = wavelet[..., t]
            ring_t = rings_full[t].astype(compute_dtype)
            ring_tm1 = rings_full[jnp.maximum(t - 1, 0)].astype(compute_dtype)
            S_t = driver(frame, ring_t, ring_tm1, w_t, models, cfg)
            _, vjp_fn = jax.vjp(step_with_rec, S_t, models, w_t)
            g_S, g_m, g_wt = vjp_fn((S_bar, g_rec_seq[t]))
            g_models = jax.tree.map(jnp.add, g_models, g_m)
            return (S_t, g_S, g_models), g_wt

        S_bar0 = jax.tree.map(jnp.zeros_like, S_final)
        g_models0 = jax.tree.map(jnp.zeros_like, models)
        (_, _, g_models), g_wt_seq = jax.lax.scan(
            body, (S_final, S_bar0, g_models0), jnp.arange(nt),
            reverse=True, unroll=unroll,
        )
        return g_models, jnp.moveaxis(g_wt_seq, 0, -1)

    propagate.defvjp(propagate_fwd, propagate_bwd)
    return propagate

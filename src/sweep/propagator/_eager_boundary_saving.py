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


def save_ring(field, rings):
    return [field[idx].clone() for idx in rings]


def restore_ring(field, ring_values, rings):
    for idx, val in zip(rings, ring_values):
        field[idx] = val
    return field


class ReconState:
    """Per-rollout reconstruction state.  Replaces seistorch's class globals."""

    __slots__ = ("frame", "cur_strip", "rings")

    def __init__(self, nt, rings):
        self.frame = None
        # cur_strip[k] = [save_ring(S_k[f]) for f in ring_fields], k = 0..nt
        self.cur_strip = [None] * (nt + 1)
        self.rings = rings


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
            save_ring(S_out[f], st.rings) for f in cfg["ring_fields"]
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
                # frame = S_{i+1} = [u_{i+1}, u_i, cpml...].  Reverse to S_i.
                u_ip1, u_i = frame[0], frame[1]
                swapped = [u_i, u_ip1] + [zero] * (nwf - 2)
                u_im1 = list(func(swapped, models, dt, h, None))[0]
                u_im1 = reinject_source(u_im1)        # + src_i
                u_i_c = restore_ring(u_i.clone(), st.cur_strip[i][0], rings)
                prev = st.cur_strip[i - 1][0] if i - 1 >= 0 else st.cur_strip[0][0]
                u_im1_c = restore_ring(u_im1, prev, rings)
                S_i = [u_i_c, u_im1_c] + [zero] * (nwf - 2)
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
        diff_inputs = list(S_i_rg) + [m for m, f in zip(models_rg, model_flags) if f]
        grads = torch.autograd.grad(
            out_rg, diff_inputs, grad_outputs=grad_out,
            allow_unused=True, retain_graph=False,
        )
        grad_S = list(grads[:nwf])
        grad_M_iter = iter(grads[nwf:])
        grad_M = [next(grad_M_iter) if f else None for f in model_flags]
        return tuple([None] + grad_S + grad_M)

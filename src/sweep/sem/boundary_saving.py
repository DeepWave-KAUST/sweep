"""Pure-PyTorch boundary-saving wavefield reconstruction for the SEM backend.

Cuts the backward-pass peak memory from ``O(nt · n_global)`` (full autograd
tape) to ``O(nt · |interface ring| + few frames)`` by *reconstructing* the
forward wavefield in reverse time during backward.

Class structure mirrors the FD eager path (``sweep.propagator
._eager_boundary_saving``):

* :class:`SEMReconState` — per-rollout state (``batch=B`` for the multi-shot
  march): a pre-allocated per-step
  interface-ring buffer plus ``frame``, the current reconstructed positional
  field tuple, marched backward through the reverse pass. Per-rollout (no
  globals) → multi-shot / re-entrant safe.
* :class:`_SEMBoundarySaveStep` — one time step as a custom ``autograd.Function``.
  Forward runs ``eq.step`` under ``no_grad`` and stores only the boundary ring.
  Backward reconstructs the step's input via ``eq.reverse_step``, restores the
  ring, then re-runs ``eq.bs_recompute`` under autograd; **PyTorch's engine
  threads the per-step adjoint** to the previous step and accumulates the model
  / source gradients.

The state is the equation's positional field tuple (see ``fields.py``); which
fields carry the propagated adjoint vs. are CPML memory (zeroed in reverse) is
read from ``field_specs`` (``boundary_related``). The equation owns the two
physics-specific pieces:

* ``eq.reverse_step(fields, src, dt) -> in_fields`` — exact primal reverse, CPML
  memory zeroed (swap2nd for 2nd-order, substep for velocity-stress).
* ``eq.bs_model_tensors()`` / ``eq.bs_recompute(fields, src, dt, leaves)`` — the
  model-dependent leaves to differentiate, and a forward recompute with them
  substituted (for the per-step VJP).
* ``eq.bs_ring_save`` (field indices) / ``eq.bs_ring_restore``
  (``(field_idx, save_slot, offset)``) — the interface-ring bookkeeping.

The non-reversible CPML band is zeroed and the one-element interior interface
ring restored, so the lossless interior reconstructs exactly (the PML-region
gradient is not reconstructed — same design as FD / SPECFEM; FWI masks it).
"""

from __future__ import annotations

import copy

import torch

from .._compile import compile_or_fallback
from ._assembly import scatter_point_sources


# Per-step source assembly shared by the BS forward / reverse / VJP. ``wav`` is
# the differentiable per-step source arg(s): a list of ``n_src`` field scalars in
# the shared-wavelet mode (one summed ``(n_global,)`` pattern per source field),
# or a single ``(P,)`` per-source amplitude slice in the FD source-encoding mode
# (``cfg["per_source"]`` — scatter ``Σ_k stamp_k·s_k(t)`` over the P sources).
def _assemble_src(cfg, wav):
    if cfg["per_source"]:
        return scatter_point_sources(cfg["src_G"], cfg["src_W"], wav[0], cfg["like"])
    return [p * w for p, w in zip(cfg["src_patterns"], wav)]


# --------------------------------------------------------------------------- #
# CPML node bookkeeping: absorbing nodes + the interior interface ring.
# --------------------------------------------------------------------------- #
def cpml_node_masks(eq) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cpml_zero_idx, interface_save_idx)`` Long index tensors over global
    nodes. Active = any of the CPML *input* coefficients ``b_{x,y,z}`` /
    ``db*_d*`` nonzero (exactly zero outside the band — NOT the decay coef ``a``,
    which is ~1 everywhere). Interface = inactive nodes sharing an element with
    an active one. Both empty when there is no CPML.

    Dimension-agnostic: the coefficient tensors are either global-node
    ``(n_global,)`` (acoustic) or element-local ``ibool``-shaped (elastic), in
    2-D ``(E, g, g)`` or 3-D ``(E, g, g, g)``; ``b_y`` / ``dby_dy`` are simply
    absent in 2-D. The SPECFEM-style *elastic* profile carries no ``b_*`` at all
    (its stretch lives in ``Sx*``/``zx*``/``l*``) and marks its band with the
    element-local bool ``node_in_pml`` — without it the elastic band would be
    reported empty and boundary saving would silently skip band restore.

    ``eq.bs_band_mode`` selects what the second tensor covers:

    * ``"full"`` (default) — active nodes plus their one-element rim. The
      boundary-saved gradient then equals the full-tape one everywhere.
    * ``"rim"`` — the rim only, i.e. an interface LINE instead of the band AREA.
      The saving grows with the mesh and with the C-PML thickness, because a rim
      is a perimeter and a band is an area: measured 7.7x fewer stored nodes at
      144x96 elements with a 6-element band and 11.8x at 288x192 with a
      10-element band.

      NOT a drop-in replacement. The gradient inside the absorbing band is not
      reproduced, and the band is an edge-replication of the model, so its
      (wrong) contribution folds back onto the outermost model pixels.
      Measured on a 49x65 model with a 6-element band, source at 15% of
      X, no free surface (4-sided band) and a 1500/4500 contrast — i.e. with
      real energy in the band — against the full tape:

      ===============  ================  ==================
      pixels cropped   cos vs full tape  max rel error
      ===============  ================  ==================
      0                0.83              0.37
      1                0.99              0.10
      2                0.999             0.17
      4                0.99999           2e-3
      8                1.0000000         < 1e-7
      ===============  ================  ==================

      So: exact from ~8 pixels (2 elements) in, badly wrong on the outermost
      ring. A gentle configuration hides this completely (the same table reads
      1e-4 everywhere when the source is centred and the record is short), so
      judge it on a case that actually drives energy into the band."""
    mesh = eq.mesh
    n_global = mesh.n_global
    device = mesh.ibool.device

    active = torch.zeros(n_global, dtype=torch.bool, device=device)
    cp = getattr(eq, "cpml", None)   # bare-only equations have no attr
    elem_shape = tuple(mesh.ibool.shape)
    if cp is not None:
        for name in ("b_x", "b_y", "b_z", "dbx_dx", "dby_dy", "dbz_dz",
                     "node_in_pml"):
            v = getattr(cp, name, None)
            if v is None:
                continue
            v = v.to(device)
            if tuple(v.shape) == (n_global,):              # global-node (acoustic)
                active |= v != 0
            elif tuple(v.shape) == elem_shape:             # element-local (elastic)
                active[mesh.ibool[v != 0]] = True
            else:
                raise ValueError(
                    f"unexpected CPML coefficient shape {tuple(v.shape)} for {name}")
    if not bool(active.any()):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    band, _ = band_from_active(active, mesh.ibool.reshape(mesh.n_elements, -1),
                               getattr(eq, "bs_band_mode", "full"))
    return (torch.nonzero(active, as_tuple=False).reshape(-1),
            torch.nonzero(band, as_tuple=False).reshape(-1))


def band_from_active(active, elem_nodes, mode="full"):
    """The C-PML ring's node mask, from the mask of nodes the profile damps.

    ``'full'`` is those nodes PLUS their one-element rim. The step is linear
    in the fields, so the adjoint propagation never needed the band values --
    but the MODEL gradients are bilinear (field x adjoint), and the band
    elements' contributions fold back into the interior edge pixels through
    the model-padding edge extension. Restoring the whole band exactly is what
    makes the boundary-saved gradient equal the full-tape one.

    ``'rim'`` is that rim ALONE: the interface LINE between the band and the
    interior, 7-12x fewer nodes, exact from a few model pixels in.

    Returns ``(band mask, which elements touch the active set)``. Four callers
    built this by hand with four copies of the refusal below -- the eager node
    masks, the DD node band, the DD elastic pin marker, and the coupled path's
    two domains.
    """
    has = active[elem_nodes].any(1)
    band = torch.zeros_like(active)
    band[elem_nodes[has].reshape(-1)] = True
    if mode == "rim":
        band &= ~active
        if active.any() and not band.any():
            raise ValueError(
                "bs_band_mode='rim' has nothing to store here: the C-PML band "
                f"covers every one of the {active.numel()} nodes, so there is "
                "no interface line between it and an interior. Enlarge the "
                "interior (more elements) or thin the band (a smaller "
                "abcn), or use bs_band_mode='full'.")
    elif mode != "full":
        raise ValueError(f"bs_band_mode must be 'full' or 'rim', got {mode!r}")
    return band, has


def cpml_band_element_mask(eq) -> torch.Tensor:
    """``(E,)`` bool: does this element touch the C-PML band?

    The index list (:func:`cpml_band_elements`) is what the ring saving wants;
    the MASK is what a kernel wants, because the element kernels launch one
    block per element and can then take an interior fast path with no warp
    divergence at all. Same computation, so the two cannot disagree.
    """
    mesh = eq.mesh
    device = mesh.ibool.device
    n_global = mesh.n_global
    cp = getattr(eq, "cpml", None)
    if cp is None:
        return torch.zeros(mesh.n_elements, dtype=torch.bool, device=device)
    active = torch.zeros(n_global, dtype=torch.bool, device=device)
    elem_shape = tuple(mesh.ibool.shape)
    for name in ("b_x", "b_y", "b_z", "dbx_dx", "dby_dy", "dbz_dz",
                 "node_in_pml"):
        v = getattr(cp, name, None)
        if v is None:
            continue
        v = v.to(device)
        if tuple(v.shape) == (n_global,):
            active |= v != 0
        elif tuple(v.shape) == elem_shape:
            active[mesh.ibool[v != 0]] = True
    return active[mesh.ibool.reshape(mesh.n_elements, -1)].any(dim=1)


def cpml_band_elements(eq) -> torch.Tensor:
    """Long tensor of element indices touching the CPML band (any active node).

    The elastic 2nd-order displacement form keeps its six CPML memory fields
    element-local ``(E, g, g, 2)``; they are exactly zero outside the band, so
    only these band ELEMENTS need ring-saving (``bs_elem_ring_save``). Empty
    when there is no CPML."""
    device = eq.mesh.ibool.device
    has = cpml_band_element_mask(eq)
    if not bool(has.any()):
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.nonzero(has, as_tuple=False).reshape(-1)


def band_support_idx(band_idx, coef):
    """The nodes of ``band_idx`` where the C-PML coefficient ``coef`` is
    non-zero: the ring index of the memory field that coefficient drives.

    ``psi_d`` / ``zeta_d`` are updated as ``b_d * (...) + a_d * psi_d`` from a
    zero start, so they are exactly ``0.0`` wherever ``b_d == 0``, for all
    time. Storing them on the union band kept those zeros: the union of the
    three axis slabs is ~2.4x any one slab at production sizes, so the ring
    was 2.2-2.5x larger than its information. Same ordering as ``band_idx``
    (a filter, not a re-sort), which is what lets a stored ring be compared
    against the old one position by position."""
    coef = coef.reshape(-1).to(band_idx.device)
    return band_idx[coef[band_idx] != 0]


def cpml_ring_masks(eq, save_idx):
    """``{field_index: ring index}`` for the fields whose ring can live on one
    axis's support -- declared by the equation as ``bs_ring_support =
    {field: coefficient name}``. Fields not declared (and equations that
    declare nothing) ring on ``save_idx`` as before. In rim mode ``save_idx``
    holds no absorbing node, so every declared field's ring is empty: exactly
    right, since the rim carries no C-PML memory."""
    out = {}
    sup = getattr(eq, "bs_ring_support", None) or {}
    cp = getattr(eq, "cpml", None)
    for f, name in sup.items():
        coef = getattr(cp, name, None) if cp is not None else None
        if coef is None or not save_idx.numel():
            continue
        out[f] = band_support_idx(save_idx, coef)
    return out


# --------------------------------------------------------------------------- #
# Per-rollout reconstruction state (positional).
# --------------------------------------------------------------------------- #
# The spellings ``bs_store_dtype`` accepts on the EAGER ring. The C ring has
# its own table (``prop_c._band_store``) because the two support different
# sets -- the C kernels ring fp32/bf16/int8, the eager one also fp16 -- but
# they must FAIL the same way, and they did not: the eager side passed the
# argument straight to ``torch.zeros``, so "bf16", which the C path accepts,
# came back as a TypeError from inside the ring allocator instead of a
# ValueError naming the option.
_STORE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16,
                 "fp16": torch.float16, "int8": torch.int8}


def _store_dtype(store):
    """``bs_store_dtype`` -> a torch dtype, or ``None`` for the compute dtype."""
    if store is None or isinstance(store, torch.dtype):
        return store
    if store in _STORE_DTYPES:
        return _STORE_DTYPES[store]
    raise ValueError(
        "bs_store_dtype must be None, a torch dtype, or one of "
        f"{sorted(_STORE_DTYPES)}; got {store!r}")


class SEMReconState:
    """Pre-allocated per-step ring buffer ``[nt+1, n_save, |ring|]`` + the current
    reconstructed frame (the equation's positional field tuple).

    ``frames`` holds optional **full-state checkpoints** ``{level: field tuple}``
    used to periodically re-anchor the reverse march. The acoustic and elastic
    reverse iterations are neutrally stable (measured ~×1.0001/step, error flat
    at 1.6e-6 from step 200 through step 600) and need no checkpoints; the
    coupled fluid–solid march accumulates ~2e-5 without them and ~4e-11 with,
    which is what they are for. See ``AcousticElasticSEM2D`` for the numbers and
    for what an earlier ~×1.15/step claim here turned out to be."""

    # no "_slot" / "batch": both were written and never read (the element
    # ring's "_eslot" twin IS read, which is what made them look load-bearing)
    __slots__ = ("save_fields", "save_idx", "idx", "cpml_zero_idx",
                 "ring", "scale", "_int8", "frame", "frames", "_ax",
                 "elem_fields", "_eslot", "elem_idx", "elem_ring")

    def __init__(self, nt, save_fields, save_idx, cpml_zero_idx, device, dtype,
                 store_dtype=None, elem_fields=(), elem_idx=None,
                 elem_trail=None, field_idx=None, batch=None):
        """``store_dtype`` controls the ring's storage precision (the dominant BS
        memory, ``O(nt · |interface|)``): ``None`` keeps the compute ``dtype``;
        ``torch.bfloat16`` / ``torch.float16`` halve it (cast on save, up-cast on
        restore); ``torch.int8`` quarters it via per-(step, field) symmetric
        quantization (a small fp32 ``scale`` table is kept alongside).

        ``field_idx`` maps a field index to ITS ring index (see
        :func:`cpml_ring_masks`); fields absent from it ring on ``save_idx``.
        The ring is therefore one ``(nt+1, |idx_f|)`` tensor per saved field
        rather than one ``(nt+1, n_save, |save_idx|)`` block.

        ``batch=B`` is the multi-shot rollout: every node field carries a
        leading shot axis, so the ring becomes ``(nt+1, B, |idx_f|)`` and the
        node axis moves from 0 to 1. Everything else is shared -- and that is
        the point of the parameter: this class and a ``_SEMReconStateBatched``
        twin used to sit side by side and had ALREADY drifted once (see
        :func:`_bs_setup`), the batched copy having no checkpoint frames and no
        element rings. Element rings ARE refused with ``batch`` -- they are a
        constructor argument, so they can be. ``frames`` is not: nothing passes
        it in, the single-shot rollout fills it after construction and the
        batched one simply never does, so there is nothing here to reject."""
        self.save_fields = list(save_fields)              # field indices to ring
        self.save_idx = save_idx
        self.cpml_zero_idx = cpml_zero_idx
        self._ax = 0 if batch is None else 1          # the node axis of a field
        if batch is not None and (elem_fields or elem_idx is not None):
            raise NotImplementedError(
                "the batched rollout has no element-local CPML memory ring; "
                "loop the shots through run_boundary_saved instead")
        field_idx = field_idx or {}
        self.idx = [field_idx.get(f, save_idx) for f in self.save_fields]
        n_save = len(self.save_fields)
        store_dtype = _store_dtype(store_dtype)
        self._int8 = store_dtype is torch.int8
        ring_dtype = store_dtype or dtype
        lead = () if batch is None else (batch,)
        self.ring = [torch.zeros(nt + 1, *lead, int(ix.numel()),
                                 dtype=ring_dtype, device=device)
                     for ix in self.idx]
        # the int8 scale is per (step, field) unbatched and per (step, field,
        # shot) batched: each shot's band has its own amplitude, and taking one
        # amax over the whole batch would quantise the quiet shots to nothing
        self.scale = (torch.zeros(nt + 1, n_save, *lead, dtype=dtype,
                                  device=device) if self._int8 else None)
        self.frame = None
        self.frames = {}
        # Element-local CPML memory ring (elastic 2nd-order): the band elements
        # of each memory field. Needed for long records — once the PML band is
        # strongly excited (surface waves) the zeroed memories corrupt the
        # adjoint entering the interior, not just the band gradient.
        self.elem_fields = list(elem_fields)
        self._eslot = {f: i for i, f in enumerate(self.elem_fields)}
        self.elem_idx = (elem_idx if elem_idx is not None
                         else torch.empty(0, dtype=torch.long, device=device))
        if self.elem_fields and self.elem_idx.numel():
            e_dtype = dtype if self._int8 else (store_dtype or dtype)
            self.elem_ring = torch.zeros(
                nt + 1, len(self.elem_fields), int(self.elem_idx.numel()),
                *elem_trail, dtype=e_dtype, device=device)
        else:
            self.elem_ring = None

    def _band(self, u, idx):
        """``u``'s rows at ``idx`` along the node axis: ``u[idx]`` unbatched,
        ``u[:, idx]`` batched."""
        return u[idx] if self._ax == 0 else u[:, idx]

    def save(self, step, fields):
        for slot, f in enumerate(self.save_fields):
            idx = self.idx[slot]
            if not idx.numel():
                continue
            band = self._band(fields[f], idx)            # (|idx_f|) or (B, |idx_f|)
            if self._int8:
                # amax over the NODE axis only, so each shot keeps its own
                # quantisation step. The amax over the field's own support
                # equals the one over the union band it used to be taken on:
                # the entries dropped are exactly zero, so the step is unchanged
                s = band.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-30
                self.ring[slot][step] = torch.round(band / s).to(torch.int8)
                self.scale[step, slot] = s.squeeze(-1)
            else:
                self.ring[slot][step] = band              # cast to ring dtype
        if self.elem_ring is not None:
            for f in self.elem_fields:
                self.elem_ring[step, self._eslot[f]] = fields[f][self.elem_idx]

    def restore(self, fields, step, restore_spec, elem_restore_spec=()):
        """``restore_spec`` = [(field_idx, save_slot, offset)]: zero the CPML band
        and copy the saved interface ring into those node fields. ``elem_restore_
        spec`` does the same for element-local memory fields on the band elements
        (they are exactly zero elsewhere, so zero the field then set the band)."""
        if not (self.cpml_zero_idx.numel() or self.save_idx.numel()
                or self.elem_ring is not None):
            return fields
        out = list(fields)
        for field_idx, slot, off in restore_spec:
            u = out[field_idx]
            if self.cpml_zero_idx.numel():
                u = u.index_fill(self._ax, self.cpml_zero_idx, 0.0)
            ix = self.idx[slot]
            if ix.numel():
                k = max(step + off, 0)
                band = self.ring[slot][k].to(u.dtype)
                if self._int8:
                    band = band * self.scale[k, slot].unsqueeze(-1)
                # nodes of the union band outside this field's support are
                # not written: they are absorbing nodes (zeroed just above)
                # or rim nodes, where the reverse step already put the exact
                # zero the old ring used to copy back
                u = u.index_copy(self._ax, ix, band)
            out[field_idx] = u
        if self.elem_ring is not None:
            for field_idx, slot, off in elem_restore_spec:
                idx = max(step + off, 0)
                u = torch.zeros_like(out[field_idx])   # memory is 0 off the band
                out[field_idx] = u.index_copy(
                    0, self.elem_idx, self.elem_ring[idx, slot].to(u.dtype))
        return tuple(out)


# --------------------------------------------------------------------------- #
# One boundary-saved time step (positional).
# --------------------------------------------------------------------------- #
class _SEMBoundarySaveStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cfg, *args):
        nm, nw = cfg["n_model"], cfg["n_wav"]
        wav = args[nm:nm + nw]
        fields = args[nm + nw:]                            # full positional tuple
        eq, dt = cfg["eq"], cfg["dt"]
        src = _assemble_src(cfg, wav)
        with torch.no_grad():
            out = eq.step(fields, src, dt)
        cfg["recon"].save(cfg["step"] + 1, out)
        ctx.cfg = cfg
        ctx.wav = tuple(w.detach() for w in wav)
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        cfg, eq, dt = ctx.cfg, ctx.cfg["eq"], ctx.cfg["dt"]
        recon, i = cfg["recon"], cfg["step"]
        gidx, need_wav = cfg["grad_idx"], cfg["need_wav"]
        model_names = cfg["model_names"]

        src_val = _assemble_src(cfg, ctx.wav)

        # Reconstruct the step's input frame (CPML zeroed), restore the rings.
        # A stored full-state checkpoint at this level overrides the
        # reconstruction outright (re-anchor for reverse-unstable equations).
        if i in recon.frames:
            in_fields = recon.frames[i]
        else:
            with torch.no_grad():
                in_fields = eq.reverse_step(recon.frame, src_val, dt)
            in_fields = recon.restore(in_fields, i, eq.bs_ring_restore,
                                      getattr(eq, "bs_elem_ring_restore", ()))
        recon.frame = tuple(in_fields)

        # Per-step VJP: re-run the forward with model + input + source leaves.
        leaves = {n: t.detach().requires_grad_(True)
                  for n, t in cfg["model_tensors"].items()}
        gset = set(gidx)
        fields_rg = tuple(f.detach().requires_grad_(j in gset)
                          for j, f in enumerate(in_fields))
        with torch.enable_grad():
            if need_wav:
                wav_rg = [w.detach().requires_grad_(True) for w in ctx.wav]
            else:
                wav_rg = list(ctx.wav)
            src_rg = _assemble_src(cfg, wav_rg)
            out = eq.bs_recompute(fields_rg, src_rg, dt, leaves)

        diff = ([leaves[n] for n in model_names]
                + [fields_rg[j] for j in gidx]
                + (wav_rg if need_wav else []))
        # Pure-passthrough outputs (e.g. CPML memory fields when the profile is
        # None: out[k] IS the detached non-grad input) carry no grad_fn;
        # autograd.grad rejects them, so feed it only the differentiable ones.
        keep = [k for k, o in enumerate(out) if o.requires_grad]
        grads = torch.autograd.grad([out[k] for k in keep], diff,
                                    grad_outputs=[grad_out[k] for k in keep],
                                    allow_unused=True, retain_graph=False)
        it = iter(grads)
        g_model = [next(it) for _ in model_names]
        g_field = {j: next(it) for j in gidx}
        g_wav = [next(it) for _ in range(cfg["n_wav"])] if need_wav else None

        grad_fields = [g_field.get(j, None) for j in range(len(in_fields))]
        wav_out = g_wav if need_wav else [None] * cfg["n_wav"]
        return (None, *g_model, *wav_out, *grad_fields)


# --------------------------------------------------------------------------- #
# Entry point (positional).
# --------------------------------------------------------------------------- #
def _bs_setup(eq, recv_idx):
    """The scaffolding both rollouts (single and vmap-batched) share.

    Pure reads: field roles, the adjoint-threading index set, the device/dtype
    reference and the C-PML masks. Extracted verbatim from the two functions
    below, which had already drifted once (frame checkpoints exist only in the
    single-shot one) -- shared scaffolding cannot drift.
    """
    if not hasattr(eq, "reverse_step"):
        raise NotImplementedError(
            f"{type(eq).__name__} has no reverse_step; boundary saving needs a "
            "reverse driver (swap2nd / substep).")
    specs = eq.field_specs
    # Fields whose adjoint is threaded step-to-step. ``boundary_related`` marks a
    # C-PML memory variable, which the reverse march ZEROES -- it does not mean
    # the field carries no gradient. Excluding those fields (the old default) is
    # harmless only while the absorbing band is quiet; once the wavefield reaches
    # it the dropped adjoint shows up in the INTERIOR gradient. Measured on
    # Marmousi with the acoustic potential form: cos 1.0000000 against the full
    # tape at 1.6 s, 0.9997 by 6.3 s. It is not round-off (fp64 gives the same
    # cosine) and not reverse-march drift (re-anchoring every 100 steps does not
    # move it). So the default now threads EVERY field, and an equation may opt
    # out through ``bs_grad_fields`` if it has a reason to.
    grad_idx = list(getattr(eq, "bs_grad_fields", None)
                    or list(range(len(specs))))
    if recv_idx is None:
        recv_idx = [i for i, s in enumerate(specs) if s.supports_receiver]
    n_src = sum(s.supports_source for s in specs)
    ref = getattr(eq, "mass", None)
    if ref is None:
        ref = eq.mass_rho
    return specs, grad_idx, recv_idx, n_src, ref.device, ref.dtype


def run_boundary_saved(eq, source_spec, wavelet, rec_G, rec_W, dt, nt,
                       *, compile_step=False, store_dtype=None,
                       frame_every=None, recv_idx=None):
    """Boundary-saved single-shot rollout for any SEM equation with BS hooks.

    ``frame_every=K`` additionally stores a **full-state checkpoint every K
    steps** and re-anchors the reverse march on them (memory: ``nt/K`` frames).
    Needed only by the coupled fluid–solid equation, whose reverse march
    accumulates ~2e-5 of the peak field without them and ~4e-11 with (the
    acoustic and elastic marches are flat and need none). Defaults to the
    equation's ``bs_frame_checkpoint_every`` class attribute when present.

    Two source modes, selected by ``wavelet`` dimensionality:

    * **shared** (``wavelet=(nt,)``) — ``source_spec`` is a list of
      ``(n_global,)`` unit patterns, one per source field; the term is
      ``pattern_f(x) · wavelet[n]`` (one time function for all sources).
    * **per-source** (``wavelet=(P, nt)``) — FD source-encoding: ``source_spec``
      is ``(src_G, src_W)`` from :meth:`SEMPropagator._build_per_source_patterns`
      (``P`` superposed point sources), and the term is
      ``Σ_k stamp_k(x) · wavelet[k, n]`` (a distinct time function per source —
      polarity / amplitude / time-shift encoding). The model gradient is the true
      cross-talk source-encoded gradient (one rollout, superposed wavefield).

    * ``rec_G`` / ``rec_W`` — receiver gather matrices, sampled on every receiver
      field.
    * ``store_dtype`` — interface-ring storage precision (the dominant BS memory):
      ``None`` (compute dtype), ``torch.bfloat16`` / ``torch.float16`` (½), or
      ``torch.int8`` (¼, per-step quantized). bf16 keeps the gradient within
      reconstruction noise; int8 trades a little accuracy for the smallest tape.
    * ``compile_step`` — wrap ``eq.step`` in :func:`torch.compile`. Because the
      reverse pass (``reverse_step``) and the per-step VJP (``bs_recompute``) both
      call ``self.step``, compiling that one method fuses the forward recompute,
      the reverse reconstruction *and* the adjoint recompute — ~2× faster on GPU
      with bit-identical gradients. (``eq`` is shallow-copied first so a cached
      caller equation is not mutated; the copy is kept alive by the per-step
      ``ctx`` until backward, so the compiled kernel is still in place then.)
    * returns ``{receiver_field_index: seis (nrec, nt)}``.
    """
    if compile_step:
        eq = copy.copy(eq)                          # isolate the step monkeypatch
    (specs, grad_idx, recv_idx, n_src,
     device, dtype) = _bs_setup(eq, recv_idx)
    if compile_step:
        eq.step = compile_or_fallback(eq.step, device,
                                      f"the {type(eq).__name__} step")

    cpml_zero_idx, save_idx = cpml_node_masks(eq)
    elem_fields = list(getattr(eq, "bs_elem_ring_save", ()) or ())
    elem_idx = elem_trail = None
    if elem_fields:
        elem_idx = cpml_band_elements(eq)
        elem_trail = tuple(eq.init_state(dt, device, dtype)[elem_fields[0]].shape[1:])
    recon = SEMReconState(nt, eq.bs_ring_save, save_idx, cpml_zero_idx,
                          device, dtype, store_dtype=store_dtype,
                          elem_fields=elem_fields, elem_idx=elem_idx,
                          elem_trail=elem_trail,
                          field_idx=cpml_ring_masks(eq, save_idx))

    fields = list(eq.init_state(dt, device, dtype))
    recon.save(0, fields)

    model_tensors = eq.bs_model_tensors()
    model_names = list(model_tensors.keys())
    need_wav = bool(wavelet.requires_grad)
    # Source assembly: shared one summed pattern per field × scalar/step; per-source
    # keeps the P sources separate and scatters Σ_k stamp_k·s_k(t) each step.
    per_source = wavelet.dim() == 2
    cfg = {
        "eq": eq, "dt": dt, "recon": recon,
        "grad_idx": grad_idx, "model_tensors": model_tensors,
        "model_names": model_names, "n_model": len(model_names),
        "need_wav": need_wav, "per_source": per_source,
        "n_wav": 1 if per_source else n_src,
    }
    if per_source:
        src_G, src_W = source_spec
        cfg["src_G"], cfg["src_W"] = src_G, src_W
        cfg["like"] = torch.zeros(eq.mesh.n_global, dtype=dtype, device=device)
    else:
        cfg["src_patterns"] = source_spec
    if frame_every is None:
        frame_every = getattr(eq, "bs_frame_checkpoint_every", None)
    seis = {j: [] for j in recv_idx}
    for i in range(nt):
        step_cfg = {**cfg, "step": i}
        wav_i = [wavelet[:, i]] if per_source else [wavelet[i]] * n_src
        fields = list(_SEMBoundarySaveStep.apply(
            step_cfg, *model_tensors.values(), *wav_i, *fields))
        # full-state checkpoint at level i+1 (the input of backward step i+1)
        if frame_every and (i + 1) % frame_every == 0:
            recon.frames[i + 1] = tuple(f.detach().clone() for f in fields)
        for j in recv_idx:
            seis[j].append((fields[j][rec_G] * rec_W).sum(-1))
    recon.frame = tuple(f.detach() for f in fields)
    return [torch.stack(seis[j], dim=-1) for j in recv_idx]


# --------------------------------------------------------------------------- #
# Batched (vmap over shots) boundary saving — one BS rollout for ALL shots.
# Mirrors the single-shot path above, but the positional fields carry a leading
# batch dim ``B`` and ``eq.step`` / ``reverse_step`` / ``bs_recompute`` are
# ``torch.vmap``-ed over it (the model leaves stay shared, so their gradient
# sums over shots). Bit-identical to looping the single-shot rollout; fills the
# GPU the same way the full-tape ``_run_batched`` path does.
# --------------------------------------------------------------------------- #
class _SEMBoundarySaveStepBatched(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cfg, *args):
        nm, ns = cfg["n_model"], cfg["n_src"]
        wav = args[nm:nm + ns]                             # each (B,)
        fields = args[nm + ns:]                            # each (B, n_global)
        src = [p * w[:, None] for p, w in zip(cfg["src_patterns"], wav)]
        with torch.no_grad():
            out = cfg["vstep"](fields, src)
        cfg["recon"].save(cfg["step"] + 1, out)
        ctx.cfg = cfg
        ctx.wav = tuple(w.detach() for w in wav)
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        cfg, eq, dt = ctx.cfg, ctx.cfg["eq"], ctx.cfg["dt"]
        recon, i = cfg["recon"], cfg["step"]
        gidx, need_wav = cfg["grad_idx"], cfg["need_wav"]
        model_names = cfg["model_names"]

        src_val = [p * w[:, None] for p, w in zip(cfg["src_patterns"], ctx.wav)]
        with torch.no_grad():
            in_fields = cfg["vreverse"](recon.frame, src_val)
        in_fields = recon.restore(in_fields, i, eq.bs_ring_restore)
        recon.frame = tuple(in_fields)

        leaves = {n: t.detach().requires_grad_(True)
                  for n, t in cfg["model_tensors"].items()}        # shared model
        gset = set(gidx)
        fields_rg = tuple(f.detach().requires_grad_(j in gset)
                          for j, f in enumerate(in_fields))
        with torch.enable_grad():
            if need_wav:
                wav_rg = [w.detach().requires_grad_(True) for w in ctx.wav]
            else:
                wav_rg = list(ctx.wav)
            src_rg = [p * w[:, None] for p, w in zip(cfg["src_patterns"], wav_rg)]
            vrec = torch.vmap(lambda f, s: eq.bs_recompute(f, s, dt, leaves),
                              in_dims=(0, 0))
            out = vrec(fields_rg, src_rg)

        diff = ([leaves[n] for n in model_names]
                + [fields_rg[j] for j in gidx]
                + (wav_rg if need_wav else []))
        grads = torch.autograd.grad(out, diff, grad_outputs=grad_out,
                                    allow_unused=True, retain_graph=False)
        it = iter(grads)
        g_model = [next(it) for _ in model_names]
        g_field = {j: next(it) for j in gidx}
        g_wav = [next(it) for _ in range(cfg["n_src"])] if need_wav else None

        grad_fields = [g_field.get(j, None) for j in range(len(in_fields))]
        wav_out = g_wav if need_wav else [None] * cfg["n_src"]
        return (None, *g_model, *wav_out, *grad_fields)


def run_boundary_saved_batched(eq, src_patterns, wavelet, rec_G, rec_W, dt, nt,
                               *, store_dtype=None, recv_idx=None):
    """Batched boundary-saved rollout: all shots in ONE vmapped time loop.

    * ``src_patterns`` — list (one per source field) of ``(B, n_global)`` unit
      patterns. ``wavelet`` — ``(B, nt)``. ``rec_G`` / ``rec_W`` — ``(B, nrec,
      ngll²)`` per-shot receiver gather. Returns a LIST of ``(B, nrec, nt)``
      in ``recv_idx`` order.

    Bit-identical to looping :func:`run_boundary_saved`; the model gradient sums
    over shots (the leaves are shared). ``compile_step`` is intentionally not
    offered here — ``compile(vmap)`` of the BS step is a separate lever.
    """
    if getattr(eq, "bs_elem_ring_save", None):
        raise NotImplementedError(
            f"{type(eq).__name__} needs an element-local CPML memory ring, which "
            "the batched BS rollout does not implement; loop the shots through "
            "run_boundary_saved instead.")
    (specs, grad_idx, recv_idx, n_src,
     device, dtype) = _bs_setup(eq, recv_idx)
    B = src_patterns[0].shape[0]

    cpml_zero_idx, save_idx = cpml_node_masks(eq)
    recon = SEMReconState(nt, eq.bs_ring_save, save_idx, cpml_zero_idx,
                          device, dtype, store_dtype=store_dtype,
                          field_idx=cpml_ring_masks(eq, save_idx), batch=B)

    st0 = eq.init_state(dt, device, dtype)
    fields = [f.unsqueeze(0).repeat(B, *([1] * f.dim())).contiguous() for f in st0]
    recon.save(0, fields)

    vstep = torch.vmap(lambda f, s: eq.step(f, s, dt), in_dims=(0, 0))
    vreverse = torch.vmap(lambda f, s: eq.reverse_step(f, s, dt), in_dims=(0, 0))

    model_tensors = eq.bs_model_tensors()
    model_names = list(model_tensors.keys())
    need_wav = bool(wavelet.requires_grad)
    ar = torch.arange(B, device=device)[:, None, None]
    cfg = {
        "eq": eq, "dt": dt, "recon": recon, "src_patterns": src_patterns,
        "grad_idx": grad_idx, "model_tensors": model_tensors,
        "model_names": model_names, "n_model": len(model_names), "n_src": n_src,
        "need_wav": need_wav, "vstep": vstep, "vreverse": vreverse,
    }
    seis = {j: [] for j in recv_idx}
    for i in range(nt):
        step_cfg = {**cfg, "step": i}
        wav_i = [wavelet[:, i]] * n_src                    # each (B,)
        fields = list(_SEMBoundarySaveStepBatched.apply(
            step_cfg, *model_tensors.values(), *wav_i, *fields))
        for j in recv_idx:
            seis[j].append((fields[j][ar, rec_G] * rec_W).sum(-1))   # (B, nrec)
    recon.frame = tuple(f.detach() for f in fields)
    return [torch.stack(seis[j], dim=-1) for j in recv_idx]          # (B,nrec,nt)

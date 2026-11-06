"""Custom-gradient / imaging-condition support for the eager propagator.

Split out of ``_PropTorchEager`` as a mixin: user-registered imaging conditions
(RTM / custom gradient kernels) and the hook-based adjoint-state forward.  These
methods use ``self.<core>`` and are mixed into ``_PropTorchEager`` (resolved via
MRO); they are grouped here only to keep the core propagator class focused.
"""
import torch
from sweep.sources.torch import SourceTorch
from sweep.receivers.torch import ReceiverTorch


class _GradBridgeFn(torch.autograd.Function):
    """Bridge between the registered model leaves and the hook-accumulated grads.

    The custom-gradient path runs the forward step-by-step under autograd
    (no manual adjoint).  PyTorch's own reverse pass delivers each saved
    ``wf[source_idx]_t`` a grad that physically *is* the adjoint wavefield
    ``ub_t``.  A backward hook on that tensor receives ``ub_t`` and calls
    the user-registered ``fn(uf_t, ub_t, models, dt, dh)``, accumulating
    into ``holder``.

    Only the models with a registration pass through this Function (as
    identity-then-detach); the others stay in autograd.  When the reverse
    pass finally reaches this node — *after* every per-step hook has fired,
    because they sit topologically between ``loss`` and the model inputs —
    we crop each leaf's runtime-padded accumulator and hand it back as the
    leaf's grad.
    """

    @staticmethod
    def forward(ctx, holder, keys, model_shapes, crop_fn, *model_leaves):
        ctx.holder = holder                       # {model_idx: tensor(shape_wf)}
        ctx.keys = keys                           # each leaf's model_idx
        ctx.model_shapes = model_shapes           # each leaf's shape
        ctx.crop_fn = crop_fn
        # Identity-detach: the propagator works on detached views so vp does
        # NOT propagate gradients through eq.func — we'll fill its grad
        # ourselves on backward.
        return tuple(m.detach() for m in model_leaves)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # grad_outputs (one per detached output) are whatever autograd would
        # have written to the detached tensors — we ignore them entirely and
        # substitute the hook-accumulated grads.
        return (None, None, None, None,
                *(ctx.crop_fn(ctx.holder[k], shape)
                  for k, shape in zip(ctx.keys, ctx.model_shapes)))


class _CustomGradientMixin:
    """Imaging-condition / custom-gradient methods for ``_PropTorchEager``."""

    def register_gradient(self, param_name, imaging_fn, mode="gradient",
                          imaging_type=None):
        """Register a user-defined imaging condition for a model parameter.

        Args:
            param_name: model name (e.g. ``"vp"``); must be in ``self.model_names``.
            imaging_fn: callable ``fn(forward, adjoint, models, dt, dh) -> tensor``
                returning the per-timestep contribution on the runtime-padded
                grid.

                **``forward`` and ``adjoint`` are always dicts** keyed by
                wavefield name:

                - ``forward[name]`` — the forward wavefield ``u_f(name, t)``,
                  i.e. a snapshot of ``wf[wavefield_names.index(name)]`` after
                  source injection at time t.
                - ``adjoint[name]`` — the adjoint wavefield ``u_b(name, t)``,
                  delivered by PyTorch autograd as ``∂loss/∂wf[name]_t``
                  (physically: the receiver residual back-propagated through
                  the wave equation to time t).

                Which keys are populated is controlled by ``imaging_type``.

            mode: ``"gradient"`` — the formula replaces the optimizer gradient
                (``param.grad`` after ``loss.backward()``) of a plain forward
                call that records it; it needs ``memory=Full()``, as it reads
                every step's forward wavefield off the full autograd tape.
                The other models and the wavelet keep autograd's gradient.
                ``"imaging"`` — the formula is only produced by :meth:`imaging`,
                leaving autograd gradients untouched.
            imaging_type: list of wavefield names to expose to ``fn``.
                Parallel in spirit to :attr:`source_type` / :attr:`receiver_type`.
                If ``None`` (default), the fn receives just
                ``{source_type[0]: tensor}`` — a one-key dict for the field
                that's source-injected.  For multi-field imaging conditions
                (e.g. PP/PS on elastic vx/vz), pass the explicit list.

        Examples:
            Standard acoustic cross-correlation (single-field default —
            wavefield name is ``'h1'`` for the bundled Acoustic equation)::

                prop.register_gradient('vp',
                    lambda fwd, adj, m, dt, dh: fwd['h1'] * adj['h1'])

            Elastic PP imaging (multi-field, divergence × divergence)::

                def pp_imaging(fwd, adj, m, dt, dh):
                    dvxf_dx, _ = grad_xz(fwd['vx'], dh)
                    _, dvzf_dz = grad_xz(fwd['vz'], dh)
                    dvxb_dx, _ = grad_xz(adj['vx'], dh)
                    _, dvzb_dz = grad_xz(adj['vz'], dh)
                    return (dvxf_dx + dvzf_dz) * (dvxb_dx + dvzb_dz)
                prop.register_gradient('vp', pp_imaging, mode='imaging',
                                       imaging_type=['vx', 'vz'])
        """
        if param_name not in self.model_names:
            raise ValueError(
                f"Unknown model '{param_name}'. Available: {list(self.model_names)}"
            )
        if mode not in ("gradient", "imaging"):
            raise ValueError(f"mode must be 'gradient' or 'imaging', got '{mode}'.")
        if not callable(imaging_fn):
            raise TypeError("imaging_fn must be callable.")
        if imaging_type is not None:
            imaging_type = list(imaging_type)
            unknown = [w for w in imaging_type if w not in self.wavefield_names]
            if unknown:
                raise ValueError(
                    f"Unknown wavefield(s) {unknown} in imaging_type. "
                    f"Available: {list(self.wavefield_names)}"
                )
            if len(imaging_type) == 0:
                raise ValueError("imaging_type list cannot be empty.")
        if mode == "gradient":
            self._refuse_custom_gradient_memory()
        self._custom_gradients[param_name] = (imaging_fn, mode, imaging_type)

    def clear_gradients(self):
        """Remove all registered custom gradient / imaging conditions."""
        self._custom_gradients = {}

    def _refuse_custom_gradient_memory(self):
        """A gradient-mode registration needs the full tape: the hooks read
        every step's forward wavefield off it. Chunk checkpointing ignored the
        registration and returned autograd's gradient; boundary saving
        honoured it but kept the whole tape, saving nothing."""
        if self.use_ckpt or getattr(self, "_eager_bs", False):
            memory = "Ckpt()" if self.use_ckpt else "BoundarySaving()"
            raise ValueError(
                "register_gradient(mode='gradient') needs memory=Full(): its "
                "hooks read every step's forward wavefield off the full autograd "
                f"tape, and this propagator runs memory={memory}. Build it with "
                "memory=Full(), or register mode='imaging' and call imaging(), "
                "which keeps its own tape whatever the strategy.")

    def _registered_recording(self, models):
        """The gradient-mode registrations this call records, as
        ``{model_index: (fn, imaging_type)}``: those whose model requires grad,
        with grad mode on. Any other call has no gradient for a registration
        to replace and takes the standard path."""
        registered = self._registered_indices("gradient")
        if not registered or not torch.is_grad_enabled():
            return {}
        models = list(self.parameters()) if models is None else list(models)
        return {i: reg for i, reg in registered.items()
                if torch.is_tensor(models[i]) and models[i].requires_grad}

    def _refuse_custom_gradient_call(self, adj, return_wavefield):
        """A call that records a registered model's gradient replaces it only
        on plain forward modelling under the full tape. The others below used
        to take the standard path and return autograd's gradient silently."""
        self._refuse_custom_gradient_memory()
        given = [name for name, on in (("adj=True", adj),
                                       ("return_wavefield=True", return_wavefield)) if on]
        if given:
            raise ValueError(
                f"{' and '.join(given)} on a call that records the gradient of a "
                "model with a register_gradient(mode='gradient') condition: the "
                "registration replaces the gradient of plain forward modelling "
                "only. Run this call under torch.no_grad() (or with the model "
                "detached), or clear_gradients() first.")

    def _registered_indices(self, mode):
        """Return {model_index: (fn, imaging_type)} for registrations of mode.

        ``imaging_type`` is the user-supplied list of wavefield names
        (``None`` ⇒ single-field shortcut, fn receives a tensor).
        """
        out = {}
        for name, (fn, m, imaging_type) in self._custom_gradients.items():
            if m == mode:
                out[self.model_names.index(name)] = (fn, imaging_type)
        return out

    def _hook_field_indices(self, registered_idx):
        """Union of wavefield indices that need a backward hook across all
        registrations.  ``imaging_type=None`` ⇒ hook the first source field; a
        list ⇒ hook each named field's index.
        """
        names = set()
        default_name = self.source_type[0]
        for _, (_fn, imaging_type) in registered_idx.items():
            if imaging_type is None:
                names.add(default_name)
            else:
                names.update(imaging_type)
        return [self.wavefield_names.index(n) for n in sorted(
            names, key=lambda n: self.wavefield_names.index(n))]

    def _run_with_hooks(self, wavelet, src_loc, rec_loc, source_encoding,
                        shape_wf, B, nrec, runtime_models,
                        registered_idx, holder):
        """Run the eager forward step-by-step *with* autograd enabled and
        register per-step backward hooks on every wavefield needed by any
        registered imaging condition.

        Hook synchronization: for multi-wavefield registrations
        (``imaging_type=['vx', 'vz', ...]``), we use
        ``torch.autograd.graph.register_multi_grad_hook(..., mode='all')`` so
        PyTorch waits until all targeted wavefields have received their
        gradient before calling the user fn — at that point we have the full
        ``{name: ub_t}`` dict.  For single-wavefield (``imaging_type=None``),
        we use a plain per-tensor hook (cheaper, no synchronization
        bookkeeping).

        ``holder[idx]`` (runtime-padded grid) accumulates each registered
        model's contribution; the cropped result is returned to autograd
        via :class:`_GradBridgeFn` (gradient mode) or read directly
        (imaging mode).

        ``wavelet`` is injected into every ``source_type`` field, as the
        standard path does. Something in the graph must require grad for the
        per-step fields to be in it: a bridged model (gradient mode) or an
        internal wavelet leaf (:meth:`imaging`).

        Returns ``record`` of shape (B, nt, nrec, 1) — autograd-aware.
        """
        eq = self.equation
        dt = self.dt
        h = self._equation_spacing
        # This path steps full-grid fields, one tensor per wavefield, so it finds
        # them by name: source_indices / receiver_indices are positions in the
        # eager state, which differ once CPML strips hold a field as two tensors.
        default_name = self.source_type[0]
        h1 = self.wavefield_names.index(default_name)
        rec_idx = [self.wavefield_names.index(n) for n in self.receiver_type]

        # Union of wavefield indices that need a hook + per-registration
        # field lists (kept in stable wavefield_names order for dict assembly).
        hook_field_idx = self._hook_field_indices(registered_idx)
        # Per-registration tuple: (idx, fn, wavefield_idx_list_or_None,
        # wavefield_name_list_or_None)
        regs = []
        for idx, (fn, imaging_type) in registered_idx.items():
            if imaging_type is None:
                regs.append((idx, fn, None, None))
            else:
                names = list(imaging_type)
                fld_idx = [self.wavefield_names.index(n) for n in names]
                regs.append((idx, fn, fld_idx, names))

        # Same checkerboard-suppression stencil as the core eager path, so a
        # custom imaging condition does not silently fall back to raw point
        # injection/sampling on the rotated staggered grid.
        sr_stencil = getattr(self.equation, "source_receiver_stencil", None)
        src = SourceTorch(src_loc, shape_wf, self.dev, source_encoding, adj=False,
                          spread_kernel=sr_stencil)
        rec = ReceiverTorch(rec_loc, gather_kernel=sr_stencil, shape=shape_wf)

        nt = wavelet.shape[-1]
        # One view per step, as the standard path takes them: a slice per
        # step would cost a whole-wavelet backward per step.
        wavelet_steps = wavelet.movedim(-1, 0).contiguous().unbind(0)

        wf = [torch.zeros(shape_wf, dtype=torch.float32, device=self.dev)
              for _ in self.wavefield_names]
        record_buf = []

        def _make_multi_callback(uf_snaps_by_name, hook_names):
            """Returns a callback for register_multi_grad_hook.  PyTorch
            passes a tuple of grads in the same order as the input tensors.

            After the callback fires (once per backward, when all targeted
            wavefields have received their grad), we clear the snapshot dict
            so PyTorch's hook registry can release its references to the
            ~``shape_wf``-sized tensors held in the closure.  Without this,
            the dict is reachable only via the hook (registered on a grad_fn
            that PyTorch's C++ side retains a reference to past backward),
            and Python GC cannot reclaim them — every shot leaks
            ``nt × per-step-snapshot`` of GPU memory.
            """
            def cb(grads):
                ub_by_name = {n: g for n, g in zip(hook_names, grads)
                              if g is not None}
                try:
                    if not ub_by_name:
                        return
                    for idx, fn, fld_idx, names in regs:
                        # Unified API: fn ALWAYS receives a dict
                        # {wavefield_name: tensor} for both forward and
                        # adjoint, even if only one field is requested.
                        # The keys are taken from `imaging_type=[...]`,
                        # or default to source_type[0] when imaging_type=None.
                        needed = names if names is not None else [default_name]
                        forward = {n: uf_snaps_by_name[n]
                                   for n in needed if n in uf_snaps_by_name}
                        adjoint = {n: ub_by_name[n]
                                   for n in needed if n in ub_by_name}
                        if len(adjoint) < len(needed):
                            continue
                        contrib = fn(forward, adjoint, runtime_models, dt, h)
                        holder[idx] = holder[idx] + contrib
                finally:
                    # Drop snapshot references so the underlying CUDA tensors
                    # can be freed once backward releases this hook.
                    uf_snaps_by_name.clear()
            return cb

        for i in range(nt):
            wf = list(eq.func(wf, runtime_models, dt, h, None))
            for k in self.source_indices:
                wf[k] = src(wf[k], wavelet_steps[i])

            # Hook the targeted wavefields IF they're already in autograd's
            # reverse graph at this step.  Early timesteps may have certain
            # fields still as zero leaves (e.g. vx/vz before any stress has
            # propagated) — those have no grad_fn and register_multi_grad_hook
            # would assert.  Skipping them is physically correct because
            # their ub_t is 0.
            hook_tensors = [wf[fi] for fi in hook_field_idx]
            if all(t.grad_fn is not None for t in hook_tensors):
                hook_names = [self.wavefield_names[fi] for fi in hook_field_idx]
                uf_snaps_by_name = {self.wavefield_names[fi]:
                                    wf[fi].detach().clone()
                                    for fi in hook_field_idx}
                torch.autograd.graph.register_multi_grad_hook(
                    hook_tensors,
                    _make_multi_callback(uf_snaps_by_name, hook_names),
                    mode='all',
                )

            # Record sampling: read every receiver_type wavefield so the
            # output shape matches the main forward (e.g. elastic's vx+vz),
            # which lets ``obs.view_as(record)`` work in imaging() with
            # multi-component observed data.
            if len(rec_idx) > 1:
                sampled = rec.sample_fields(
                    [wf[idx] for idx in rec_idx]
                )  # (B, nrec, len(receiver_type))
                record_buf.append(sampled)
            else:
                record_buf.append(
                    rec(wf[rec_idx[0]]).view(B, nrec, 1)
                )

        record = torch.stack(record_buf, dim=1)  # (B, nt, nrec, len(rec_type))
        return record

    def _custom_gradient_setup(self, wavelet, sources, receivers, batch_size, **kwargs):
        """Shared IO setup for the custom-gradient / imaging paths."""
        kwargs.setdefault("fd_pad", self._runtime_fd_pad())
        kwargs.setdefault("shape", self._runtime_shape())
        self.init_abc(**kwargs)
        shape_wf = (batch_size, 1) + self._runtime_shape()
        wavelet_t = self._as_device_tensor(wavelet, dtype=torch.float32)
        src_loc = self._as_device_tensor(sources, dtype=torch.long) + self.coord_offset
        rec_loc = self._as_device_tensor(receivers, dtype=torch.long) + self.coord_offset
        return wavelet_t, src_loc, rec_loc, shape_wf

    def _custom_gradient_forward(self, wavelet, sources, receivers, models,
                                 batch_size, source_encoding, registered_idx,
                                 **kwargs):
        """Plain forward whose backward uses the registered gradient-mode
        imaging condition(s) ``registered_idx`` (``_registered_recording``).

        Implementation: runs the forward under autograd with per-step hooks
        on ``wf[source_idx]``; PyTorch's reverse pass delivers each step's
        true ``ub_t``, and the hooks accumulate ``fn(uf_t, ub_t, …)`` into
        a buffer.  The registered model leaves pass through
        :class:`_GradBridgeFn`, which substitutes the cropped accumulator for
        ``vp.grad`` on backward; the other models and the wavelet stay in
        autograd and get its gradient.  No manually written adjoint.
        """
        if models is None:
            models = list(self.parameters())
        wavelet_t, src_loc, rec_loc, shape_wf = self._custom_gradient_setup(
            wavelet, sources, receivers, batch_size, **kwargs
        )
        nrec = rec_loc.shape[1]

        models_dev = [self._as_device_tensor(m, dtype=torch.float32)
                      for m in models]
        keys = sorted(registered_idx)

        # Runtime-padded grad accumulator.  Each entry is filled by the per-
        # step hooks during backward, then handed to _GradBridgeFn.backward.
        holder = {i: torch.zeros(shape_wf, dtype=torch.float32, device=self.dev)
                  for i in keys}

        # _GradBridgeFn outputs detached views of the registered leaves AND
        # wires the bridge into autograd: when backward reaches this node it
        # pulls from ``holder`` (which the per-step hooks have already
        # filled) and writes each leaf's grad.
        bridged = _GradBridgeFn.apply(
            holder, keys, [tuple(models_dev[i].shape) for i in keys],
            self._crop_to_model, *(models_dev[i] for i in keys),
        )
        for i, m in zip(keys, bridged):
            models_dev[i] = m
        _, runtime_models = self._pad_models(models_dev)

        return self._run_with_hooks(
            wavelet_t, src_loc, rec_loc, source_encoding, shape_wf,
            batch_size, nrec, runtime_models, registered_idx, holder,
        )

    def imaging(self, wavelet, sources, receivers, models=None,
                adjoint_source=None, observed=None):
        """Explicit RTM/imaging using the registered ``mode='imaging'``
        condition(s).  Does NOT touch any ``param.grad``.

        The adjoint source is, in priority order: ``adjoint_source`` if given;
        else ``record - observed`` (FWI residual) if ``observed`` is given;
        else the forward ``record`` itself (zero-lag migration).

        Returns ``{param_name: image_tensor}`` for every imaging-mode
        registration.

        Implementation: same autograd-with-hooks path as the gradient mode,
        but the reverse pass is triggered explicitly by ``record.backward
        (gradient=adj_src)`` so that no ``param.grad`` is written (the
        models enter detached, with no bridge).  It records its own graph
        whatever the caller's grad mode, so it runs under ``torch.no_grad()``
        too.
        """
        registered = {self.model_names.index(n): (fn, wavefields)
                      for n, (fn, m, wavefields) in self._custom_gradients.items()
                      if m == "imaging"}
        if not registered:
            raise ValueError(
                "imaging() called but no mode='imaging' gradient is registered."
            )
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "imaging() back-propagates through its own forward, which "
                "torch.inference_mode() forbids; call it under torch.no_grad() "
                "instead.")
        with torch.enable_grad():
            return self._imaging(registered, wavelet, sources, receivers, models,
                                 adjoint_source, observed)

    def _imaging(self, registered, wavelet, sources, receivers, models,
                 adjoint_source, observed):
        mode, B, _, _, source_encoding = self._normalize_io(wavelet, sources, receivers)
        sources = self._place_on_curvilinear(sources, "source")
        receivers = self._place_on_curvilinear(receivers, "receiver")
        if models is not None and not isinstance(models, (list, tuple)):
            models = list(models)
        if models is None:
            models = list(self.parameters())

        wavelet_t, src_loc, rec_loc, shape_wf = self._custom_gradient_setup(
            wavelet, sources, receivers, B
        )
        nrec = rec_loc.shape[1]

        # Detached models — autograd will not write any param.grad through
        # the propagator chain.
        models_dev = [self._as_device_tensor(m, dtype=torch.float32).detach()
                      for m in models]
        model_shapes = [tuple(m.shape) for m in models_dev]
        _, runtime_models = self._pad_models(models_dev)

        holder = {i: torch.zeros(shape_wf, dtype=torch.float32, device=self.dev)
                  for i in registered}
        # The models enter detached, so nothing would put the per-step fields
        # in the graph: inject an internal wavelet leaf instead, purely to
        # anchor it (its .grad is never read).
        wavelet_leaf = wavelet_t.detach().clone().requires_grad_(True)
        record = self._run_with_hooks(
            wavelet_leaf, src_loc, rec_loc, source_encoding, shape_wf,
            B, nrec, runtime_models, registered, holder,
        )

        if adjoint_source is not None:
            adj_src = self._as_device_tensor(adjoint_source, dtype=torch.float32)
        elif observed is not None:
            obs = self._as_device_tensor(observed, dtype=torch.float32)
            adj_src = record.detach() - obs.view_as(record)
        else:
            adj_src = record.detach()

        # Trigger PyTorch's reverse pass through ``record``; hooks fire and
        # fill ``holder``.  No leaf user-facing tensor receives a grad here
        # (models are detached; wavelet_leaf is an internal helper).
        record.backward(gradient=adj_src.view_as(record))

        return {self.model_names[i]: self._crop_to_model(holder[i], model_shapes[i])
                for i in registered}


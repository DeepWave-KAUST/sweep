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
    """Bridge between user model leaves and the hook-accumulated grads.

    The custom-gradient path runs the forward step-by-step under autograd
    (no manual adjoint).  PyTorch's own reverse pass delivers each saved
    ``wf[source_idx]_t`` a grad that physically *is* the adjoint wavefield
    ``ub_t``.  A backward hook on that tensor receives ``ub_t`` and calls
    the user-registered ``fn(uf_t, ub_t, models, dt, dh)``, accumulating
    into ``holder``.

    Model leaves don't participate in autograd through the propagator chain
    (they pass through this Function as identity-then-detach).  When the
    reverse pass finally reaches this node — *after* every per-step hook has
    fired, because they sit topologically between ``loss`` and the model
    inputs — we crop the runtime-padded accumulator and hand the result back
    as each leaf's grad.
    """

    @staticmethod
    def forward(ctx, holder, model_shapes, registered_idx, crop_fn,
                *model_leaves):
        ctx.holder = holder                       # {model_idx: tensor(shape_wf)}
        ctx.model_shapes = model_shapes
        ctx.registered_idx = registered_idx       # {model_idx: imaging_fn}
        ctx.crop_fn = crop_fn
        ctx.n_models = len(model_leaves)
        # Identity-detach: the propagator works on detached views so vp does
        # NOT propagate gradients through eq.func — we'll fill its grad
        # ourselves on backward.
        return tuple(m.detach() for m in model_leaves)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # grad_outputs (one per detached output) are whatever autograd would
        # have written to the detached tensors — we ignore them entirely and
        # substitute the hook-accumulated grads.
        out = [None] * ctx.n_models
        for i in ctx.registered_idx:
            buf = ctx.holder.get(i)
            if buf is None:
                continue
            out[i] = ctx.crop_fn(buf, ctx.model_shapes[i]).reshape(ctx.model_shapes[i])
        return (None, None, None, None, *out)


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
                (``param.grad`` after ``loss.backward()``);
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
        self._custom_gradients[param_name] = (imaging_fn, mode, imaging_type)

    def clear_gradients(self):
        """Remove all registered custom gradient / imaging conditions."""
        self._custom_gradients = {}

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
        registrations.  ``imaging_type=None`` ⇒ hook source_indices[0]; a
        list ⇒ hook each named field's index.
        """
        names = set()
        default_name = self.wavefield_names[self.source_indices[0]]
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

        Returns ``record`` of shape (B, nt, nrec, 1) — autograd-aware.
        """
        eq = self.equation
        dt = self.dt
        h = self._equation_spacing
        h1 = self.source_indices[0]
        default_name = self.wavefield_names[h1]

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
        rec = ReceiverTorch(rec_loc, gather_kernel=sr_stencil)

        nt = wavelet.shape[-1]
        # Source-injection needs at least one require_grad leaf for autograd
        # to retain the per-step ``wf[h1]`` tensors in the graph (model
        # leaves are detached by _GradBridgeFn). We clone the wavelet as an
        # internal leaf with requires_grad=True purely to anchor the graph —
        # its .grad is never read.
        wavelet_leaf = wavelet.detach().clone().requires_grad_(True)

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
            wf[h1] = src(wf[h1], wavelet_leaf[..., i])

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
            if len(self.receiver_indices) > 1:
                sampled = rec.sample_fields(
                    [wf[idx] for idx in self.receiver_indices]
                )  # (B, nrec, len(receiver_type))
                record_buf.append(sampled)
            else:
                record_buf.append(
                    rec(wf[self.receiver_indices[0]]).view(B, nrec, 1)
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
                                 batch_size, source_encoding, **kwargs):
        """Plain forward whose backward uses the registered gradient-mode
        imaging condition(s).

        Implementation: runs the forward under autograd with per-step hooks
        on ``wf[source_idx]``; PyTorch's reverse pass delivers each step's
        true ``ub_t``, and the hooks accumulate ``fn(uf_t, ub_t, …)`` into
        a buffer.  Model leaves pass through :class:`_GradBridgeFn` which
        substitutes the cropped accumulator for ``vp.grad`` on backward.
        No manually written adjoint.
        """
        if models is None:
            models = list(self.parameters())
        wavelet_t, src_loc, rec_loc, shape_wf = self._custom_gradient_setup(
            wavelet, sources, receivers, batch_size, **kwargs
        )
        nrec = rec_loc.shape[1]
        registered_idx = self._registered_indices("gradient")

        models_dev = [self._as_device_tensor(m, dtype=torch.float32)
                      for m in models]
        model_shapes = [tuple(m.shape) for m in models_dev]

        # Runtime-padded grad accumulator.  Each entry is filled by the per-
        # step hooks during backward, then handed to _GradBridgeFn.backward.
        holder = {i: torch.zeros(shape_wf, dtype=torch.float32, device=self.dev)
                  for i in registered_idx}

        # _GradBridgeFn outputs detached views of the leaves AND wires the
        # bridge into autograd: when backward reaches this node it pulls from
        # ``holder`` (which the per-step hooks have already filled) and
        # writes each leaf's grad.
        bridged = _GradBridgeFn.apply(
            holder, model_shapes, registered_idx, self._crop_to_model,
            *models_dev,
        )
        _, runtime_models = self._pad_models(list(bridged))

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
        models enter detached, with no bridge).
        """
        registered = {self.model_names.index(n): (fn, wavefields)
                      for n, (fn, m, wavefields) in self._custom_gradients.items()
                      if m == "imaging"}
        if not registered:
            raise ValueError(
                "imaging() called but no mode='imaging' gradient is registered."
            )
        mode, B, _, _, source_encoding = self._normalize_io(wavelet, sources, receivers)
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
        record = self._run_with_hooks(
            wavelet_t, src_loc, rec_loc, source_encoding, shape_wf,
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

        return {self.model_names[i]:
                self._crop_to_model(holder[i], model_shapes[i]).reshape(model_shapes[i])
                for i in registered}


import torch
from torch.utils.checkpoint import checkpoint as ckpt_torch

from sweep.propagator.base import PropBase
from sweep.propagator.options import EAGER_DEFAULTS
from sweep.receivers.torch import ReceiverTorch
from sweep.sources.torch import SourceTorch
from sweep.utils.torch import EdgePadding


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


class _PropTorchEager(PropBase, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        self.store_last_wavefield = kwargs.pop("store_last_wavefield", EAGER_DEFAULTS.store_last_wavefield)
        self.use_compile = kwargs.pop("use_compile", EAGER_DEFAULTS.use_compile)
        self.compile_backend = kwargs.pop("compile_backend", EAGER_DEFAULTS.compile_backend)
        self.compile_mode = kwargs.pop("compile_mode", EAGER_DEFAULTS.compile_mode)
        self.compile_dynamic = kwargs.pop("compile_dynamic", EAGER_DEFAULTS.compile_dynamic)
        self.compile_fullgraph = kwargs.pop("compile_fullgraph", EAGER_DEFAULTS.compile_fullgraph)
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

        self.register_buffer("dt", torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer("dh", torch.tensor(self._grid_spacing, device=self.dev, dtype=torch.float32))
        coord_offset = list(self._runtime_coord_offset())
        self.register_buffer("coord_offset", torch.tensor(coord_offset, device=self.dev, dtype=torch.long))
        self.source_indices = [self.wavefield_names.index(name) for name in self.source_type]
        self.receiver_indices = [self.wavefield_names.index(name) for name in self.receiver_type]
        self._equation_spacing = self._normalize_equation_spacing(self._grid_spacing)
        self._workspace_cache = {}
        # User-registered custom gradient / imaging conditions.
        #   {param_name: (imaging_fn, mode)}  with mode in {"gradient", "imaging"}
        self._custom_gradients = {}
        self.step_func = self._build_step_func()

    def _as_device_tensor(self, value, *, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.dev, dtype=dtype)
        return torch.as_tensor(value, device=self.dev, dtype=dtype)

    @staticmethod
    def _normalize_equation_spacing(spacing):
        spacing = tuple(float(value) for value in spacing)
        if not spacing:
            return 1.0
        if all(value == spacing[0] for value in spacing):
            return spacing[0]
        return spacing

    def _build_step_func(self):
        equation_step = self.equation.func
        num_wavefields = len(self.wavefield_names)

        def step_func(*flat_args):
            wavefields = flat_args[:num_wavefields]
            models = flat_args[num_wavefields:-3]
            dt, h, b = flat_args[-3:]
            return equation_step(wavefields, models, dt, h, b)

        return self._maybe_compile(step_func)

    def _maybe_compile(self, fn):
        """``torch.compile`` *fn* with the propagator's compile settings, after
        raising Dynamo's recompile limit.  Dynamo specializes the step on
        per-tensor metadata; each wavefield's ``requires_grad`` flips False->True
        the first time it absorbs a contribution from the (``requires_grad=True``)
        models, forcing a recompile.  Elastic/DAS have many wavefields, so the
        default cap of 8 is exhausted before Dynamo settles and it silently falls
        back to eager — bump the cap so specialization can finish (Acoustic's
        tighter step is unaffected).  Returns *fn* unchanged when compilation is
        off.  Shared by the full-tape step and the eager boundary-saving step."""
        if not self.use_compile or not hasattr(torch, "compile"):
            return fn
        dynamo_config = getattr(torch, "_dynamo", None)
        if dynamo_config is not None:
            cfg = getattr(dynamo_config, "config", None)
            if cfg is not None and hasattr(cfg, "recompile_limit") and cfg.recompile_limit < 32:
                cfg.recompile_limit = 32
        compile_kwargs = {
            "mode": self.compile_mode,
            "dynamic": self.compile_dynamic,
            "fullgraph": self.compile_fullgraph,
        }
        if self.compile_backend is not None:
            compile_kwargs["backend"] = self.compile_backend
        return torch.compile(fn, **compile_kwargs)

    def _mark_compile_step_begin(self):
        if self.use_compile and self.dev is not None and "cuda" in str(self.dev):
            compiler = getattr(torch, "compiler", None)
            if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
                compiler.cudagraph_mark_step_begin()

    def _compiled_step(self, wavefields, models, dt, h, b):
        self._mark_compile_step_begin()
        return self.step_func(*wavefields, *models, dt, h, b)

    def _prepare_runtime_models(self, models):
        prepare = getattr(self.equation, "prepare_models", None)
        if callable(prepare):
            return prepare(models)
        return models

    def _resolve_snapshot_times(self, nt, return_wavefield, snapshot_times, snapshot_interval):
        if not return_wavefield:
            return []
        if snapshot_times is not None and snapshot_interval is not None:
            raise ValueError("Use either snapshot_times or snapshot_interval, not both.")
        if snapshot_times is not None:
            times = sorted({int(t) for t in snapshot_times})
        elif snapshot_interval is not None:
            interval = int(snapshot_interval)
            if interval < 1:
                raise ValueError("snapshot_interval must be >= 1.")
            times = list(range(0, nt, interval))
        else:
            times = list(range(nt))
        for t in times:
            if t < 0 or t >= nt:
                raise ValueError(f"Snapshot time {t} is outside valid range [0, {nt - 1}].")
        return times

    def _get_cached_tensor(self, kind, shape, *, device, dtype):
        key = (kind, tuple(shape), str(device), str(dtype))
        cached = self._workspace_cache.get(key)
        if cached is None:
            cached = torch.zeros(shape, dtype=dtype, device=device)
            self._workspace_cache[key] = cached
        else:
            cached.detach_()
            cached.zero_()
        return cached

    def _run_chunk(self, wavefield, models, dt, h, b, wavelet, nt, start_t, chunk_size, src, rec, record_shape, adj=False):
        chunk_record = self._get_cached_tensor(
            "chunk_record",
            record_shape,
            device=self.dev,
            dtype=wavefield[0].dtype,
        )
        multi_receiver = len(self.receiver_indices) > 1
        for local_i in range(chunk_size):
            t = start_t + local_i
            if t >= nt:
                break
            wavefield = list(self._compiled_step(wavefield, models, dt, h, b))
            time = t if not adj else nt - t - 1
            for source_idx in self.source_indices:
                wavefield[source_idx] = src(wavefield[source_idx], wavelet[..., time])
            if multi_receiver:
                sampled = rec.sample_fields([wavefield[idx] for idx in self.receiver_indices])
                chunk_record[:, local_i, :, :] = sampled
            else:
                receiver_idx = self.receiver_indices[0]
                chunk_record[:, local_i, :, 0] = rec(wavefield[receiver_idx]).view(
                    record_shape[0], record_shape[2]
                )
        return tuple(wavefield), chunk_record

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), (
            f"Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}"
        )
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    # ------------------------------------------------------------------
    # Custom gradient / imaging-condition registration
    # ------------------------------------------------------------------
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

    def _pad_models(self, models):
        """EdgePad physical models to runtime grid + apply equation prep."""
        models_p = [
            EdgePadding.apply(self._as_device_tensor(m, dtype=torch.float32),
                              self._runtime_padding())
            for m in models
        ]
        return models_p, self._prepare_runtime_models(models_p)

    @staticmethod
    def _crop_to_model(runtime_grid, model_shape):
        """Center-crop a runtime-padded (…, nz_rt, nx_rt) tensor to the
        physical model shape."""
        rt_nz, rt_nx = runtime_grid.shape[-2], runtime_grid.shape[-1]
        nz, nx = model_shape[-2], model_shape[-1]
        pz = (rt_nz - nz) // 2
        px = (rt_nx - nx) // 2
        return runtime_grid[..., pz:pz + nz, px:px + nx]

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

        src = SourceTorch(src_loc, shape_wf, self.dev, source_encoding, adj=False)
        rec = ReceiverTorch(rec_loc)

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
        boundary saving, dispatching on the equation's time scheme.  Both drivers
        reuse the forward physics — there is no per-equation adjoint::

            2nd-order             -> 'swap2nd'  (reuses func with time levels swapped)
            1st-order + substeps  -> 'substep'  (reuses interior_substeps reversed)

        1st-order equations *without* ``interior_substeps`` are unsupported (no
        approximate fallback).  ``enable_eager_boundary_saving(mode=...)`` can
        force a mode.  Raises if the forced/derived mode is unavailable, or unsafe
        for the current BC (a free surface the substeps would silently drop).
        """
        from sweep.equations.base import SecondOrderEquation
        is_2nd = isinstance(self.equation, SecondOrderEquation)
        has_substeps = callable(getattr(self.equation, "interior_substeps", None))
        forced = getattr(self, "_eager_bs_mode", None)
        if forced is not None:
            from sweep.propagator._eager_boundary_saving import _REVERSE_DRIVERS
            if forced not in _REVERSE_DRIVERS:
                raise ValueError(
                    f"Unknown eager boundary-saving reverse mode {forced!r}; "
                    f"registered drivers: {sorted(_REVERSE_DRIVERS)}."
                )
            if forced == "substep" and not has_substeps:
                raise ValueError(
                    f"reverse mode 'substep' needs {type(self.equation).__name__}"
                    ".interior_substeps()."
                )
            reverse_mode = forced
        elif is_2nd:
            reverse_mode = "swap2nd"
        elif has_substeps:
            reverse_mode = "substep"
        else:
            raise NotImplementedError(
                "Eager boundary saving for the 1st-order equation "
                f"'{type(self.equation).__name__}' requires an "
                "interior_substeps() hook (the exact reverse driver). Add one "
                "(reusing the equation's step core), or use impl='c' / chunk "
                "checkpointing (use_ckpt=True)."
            )
        # The substep reconstruction carries the free-surface BC only if the
        # equation's interior_substeps do (declared via supports_bs_free_surface);
        # the ring geometry also assumes full PML.  swap2nd is always FS-safe
        # (reuses ``func``).  Refuse rather than return a wrong near-surface grad.
        if (
            reverse_mode == "substep"
            and getattr(self, "free_surface", False)
            and not getattr(self.equation, "supports_bs_free_surface", False)
        ):
            raise NotImplementedError(
                "Eager boundary saving with a free surface is not supported "
                f"for {type(self.equation).__name__} (its interior_substeps "
                "would drop the free-surface BC). Options: free_surface=False, "
                "impl='c', or chunk checkpointing (use_ckpt=True). Equations "
                "whose interior_substeps carry the free-surface BC (e.g. "
                "Elastic) set supports_bs_free_surface=True."
            )
        return reverse_mode

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
        from sweep.propagator._eager_boundary_saving import (
            ReconState, _interior_index_tuple, _ring_index_tuples, save_step,
        )
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

    def _rollout_ckpt(self, wavefield, models, wavelet, src, rec, record,
                      receivers, batch_size, nt, adj):
        """Time loop under chunk gradient checkpointing: the nt steps are split
        into ``ckpt_chunks``-sized chunks, each wrapped in ``ckpt_torch`` so its
        activations are recomputed in backward instead of stored.  Writes the
        receiver record in place and returns the final wavefield list.
        """
        chunk_size = int(max(1, self.ckpt_chunks))
        record_chunk_shape = (batch_size, chunk_size, receivers.shape[1], len(self.receiver_type))
        num_chunks = (nt + chunk_size - 1) // chunk_size
        num_wavefields = len(wavefield)
        num_models = len(models)

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size

            def checkpoint_chunk(*chunk_inputs, start_t=start_t):
                state = list(chunk_inputs[:num_wavefields])
                chunk_models = list(chunk_inputs[num_wavefields : num_wavefields + num_models])
                chunk_runtime_models = self._prepare_runtime_models(chunk_models)
                return self._run_chunk(
                    state,
                    chunk_runtime_models,
                    self.dt,
                    self._equation_spacing,
                    None,
                    wavelet,
                    nt,
                    start_t,
                    chunk_size,
                    src,
                    rec,
                    record_chunk_shape,
                    adj=adj,
                )

            wavefield, chunk_record = ckpt_torch(checkpoint_chunk, *wavefield, *models, use_reentrant=False)
            wavefield = list(wavefield)
            end_t = min(start_t + chunk_size, nt)
            record[:, start_t:end_t, :, :] = chunk_record[:, : end_t - start_t, :, :]
        return wavefield

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
        from sweep.propagator._eager_boundary_saving import _BoundarySaveStep
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

    def _rollout_full(self, wavefield, runtime_models, wavelet, src, rec, record,
                      receivers, nt, adj, return_wavefield, snapshots, snapshot_lookup):
        """Plain time loop recording the full autograd tape (the default path):
        PyTorch retains every step's activations and backward differentiates the
        whole graph.  This is the only path that supports ``return_wavefield``
        (wavefield snapshots).  Writes the record (and snapshots, if requested) in
        place and returns the final wavefield list.
        """
        multi_receiver = len(self.receiver_indices) > 1
        for i in range(nt):
            wavefield = list(self._compiled_step(wavefield, runtime_models, self.dt, self._equation_spacing, None))
            time = i if not adj else nt - i - 1
            for source_idx in self.source_indices:
                wavefield[source_idx] = src(wavefield[source_idx], wavelet[..., time])
            if return_wavefield and i in snapshot_lookup:
                snapshots[snapshot_lookup[i]] = torch.stack(
                    [self._crop_runtime_halo(w).detach().cpu() for w in wavefield],
                    0,
                )
            if multi_receiver:
                record[:, i, :, :] = rec.sample_fields([wavefield[idx] for idx in self.receiver_indices])
            else:
                receiver_idx = self.receiver_indices[0]
                record[:, i, :, 0] = rec(wavefield[receiver_idx]).view(*receivers.shape[:-1])
        return wavefield

    def forward(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        adj=False,
        return_wavefield=False,
        **kwargs,
    ):
        mode, batch_size, nsrc_per_shot, _, source_encoding = self._normalize_io(
            wavelet, sources, receivers
        )
        if models is not None and not isinstance(models, (list, tuple)):
            models = list(models)

        # Custom-gradient routing: when any model has a registered
        # gradient-mode imaging condition, replace the autograd path with the
        # explicit adjoint-state backward (only for plain forward modelling —
        # adj / return_wavefield / ckpt keep the standard path).
        if (self._registered_indices("gradient")
                and not adj and not return_wavefield and not self.use_ckpt):
            return self._custom_gradient_forward(
                wavelet, sources, receivers, models, batch_size, source_encoding, **kwargs
            )

        snapshot_times = kwargs.pop("snapshot_times", None)
        snapshot_interval = kwargs.pop("snapshot_interval", None)
        kwargs.setdefault("fd_pad", self._runtime_fd_pad())
        kwargs.setdefault("shape", self._runtime_shape())
        self.init_abc(**kwargs)

        nt = wavelet.shape[-1]
        snapshot_indices = self._resolve_snapshot_times(nt, return_wavefield, snapshot_times, snapshot_interval)
        snapshot_lookup = {t: i for i, t in enumerate(snapshot_indices)}
        self.snapshot_times_last = tuple(snapshot_indices)

        shape_wavefield = (batch_size, 1) + self._runtime_shape()
        wavelet = self._as_device_tensor(wavelet, dtype=torch.float32)
        sources = self._as_device_tensor(sources, dtype=torch.long) + self.coord_offset
        receivers = self._as_device_tensor(receivers, dtype=torch.long) + self.coord_offset

        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding, adj)
        rec = ReceiverTorch(receivers)

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshot_shape = (len(snapshot_indices), len(self.wavefield_names), batch_size, 1) + self.shape
            snapshots = self._get_cached_tensor(
                "snapshots",
                snapshot_shape,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        else:
            snapshots = None

        record = self._get_cached_tensor(
            "record",
            (batch_size, nt, receivers.shape[1], len(self.receiver_type)),
            device=self.dev,
            dtype=torch.float32,
        )

        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(self._as_device_tensor(para, dtype=torch.float32), self._runtime_padding()) for para in models]
        self.models_padded = models
        runtime_models = self._prepare_runtime_models(models)
        wavefield = [
            self._get_cached_tensor(f"wavefield:{index}", shape_wavefield, device=self.dev, dtype=torch.float32)
            for index, _ in enumerate(self.wavefield_names)
        ]
        if self.use_ckpt and self.ckpt_mode != "chunk":
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}' for PropTorch. Expected 'chunk'.")
        if self.use_ckpt and return_wavefield:
            raise ValueError("return_wavefield=True is not supported with chunk checkpointing in PropTorch yet.")

        if self.use_ckpt:
            wavefield = self._rollout_ckpt(
                wavefield, models, wavelet, src, rec, record, receivers, batch_size, nt, adj
            )
        elif getattr(self, "_eager_bs", False):
            if return_wavefield:
                raise ValueError("return_wavefield is not supported with eager boundary saving.")
            wavefield = self._rollout_eager_bs(
                wavefield, runtime_models, wavelet, src, rec, record, receivers, nt, adj
            )
        else:
            wavefield = self._rollout_full(
                wavefield, runtime_models, wavelet, src, rec, record, receivers, nt, adj,
                return_wavefield, snapshots, snapshot_lookup,
            )

        self.last_wavefields = tuple(wavefield) if self.store_last_wavefield else None
        # The eager backend reuses internal workspace buffers across calls.
        # Return clones so previous outputs do not alias buffers that a later
        # forward pass will overwrite.
        record_out = record.clone()
        if not has_aux:
            return record_out
        snapshots_out = snapshots.clone()
        return record_out, snapshots_out

    forward_base = forward

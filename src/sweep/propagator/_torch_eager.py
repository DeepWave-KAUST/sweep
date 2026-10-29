import types
import warnings
import weakref

import torch
from torch.utils.checkpoint import checkpoint as ckpt_torch


from sweep._compile import (SettledSteps as _SettledSteps, compile_notice,
                            compiler_failures as _compiler_failures, compiler_hint, failure_reason)
from sweep.propagator.base import PropBase
from sweep.propagator.options import EAGER_DEFAULTS
from sweep.equations._cpml import StripLayout
from sweep.receivers.torch import ReceiverTorch
from sweep.sources.torch import SourceTorch, scatter_source
from sweep.utils.torch import EdgePadding
from sweep.propagator._eager_boundary_saving import _EagerBoundarySavingMixin
from sweep.propagator._torch_eager_custom_grad import _CustomGradientMixin


_DYNAMO_LIMIT_WARNED = False
_NO_COMPILE_WARNED = False

# Compiling inside a checkpointed forward breaks its recompute before torch 2.5
# (see _rollout_ckpt).
CKPT_PLAIN_WARMUP = torch.__version__ < (2, 5)

# torch 2.2 keeps the table of compiled backends per thread.  When the autograd
# thread runs a compiled step (a checkpoint recompute, the boundary-saving
# backward), it checks every cache entry on the step's code object -- entries
# other propagators' torch.compile left there too -- and raises KeyError on a
# backend only the main thread knows.  There each propagator compiles its own
# copy of the code; that gives up the reuse of an earlier propagator's graphs,
# which other torch versions keep.
OWN_CODE_PER_COMPILE = (2, 2) <= torch.__version__ < (2, 3)

# torch 2.3's Inductor has two bugs, fixed in 2.4, that the step runs into.
# Its buffer reuse misses that a complex result's ``.real`` still points into
# the buffer and hands it to the next FFT's input: ViscoAcoustic's forward
# turns to NaN, on GPU as on CPU.  Turning reuse off cures that and costs the
# fused step no memory we could measure.  Its C++ backend also fuses a flat
# loop into a group of loops of another shape and fails an assertion, mostly
# while compiling the backward, where no fallback can catch it.  Nothing turns
# that off, so on CPU the step runs uncompiled.
INDUCTOR_23 = (2, 3) <= torch.__version__ < (2, 4)


def _with_own_code(fn):
    """``fn`` (or a bound method) with a fresh code object, for torch.compile
    caches that must not be shared (see OWN_CODE_PER_COMPILE)."""
    if isinstance(fn, types.MethodType):
        return types.MethodType(_with_own_code(fn.__func__), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        return fn
    own = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__,
                             fn.__defaults__, fn.__closure__)
    own.__kwdefaults__ = fn.__kwdefaults__
    own.__qualname__ = fn.__qualname__
    own.__dict__.update(fn.__dict__)
    return own


def _no_buffer_reuse(mode, dynamic):
    """Inductor options for ``mode`` with buffer reuse off (see INDUCTOR_23).
    torch.compile takes a mode or options, not both, so the mode is spelt out."""
    from torch._inductor import list_mode_options
    options = {} if mode in (None, "default") else dict(list_mode_options(mode, dynamic))
    options["allow_buffer_reuse"] = False
    return options


def _raise_dynamo_recompile_limit(target=32):
    """Raise Dynamo's recompile cap, whatever the installed torch calls it.

    The knob was renamed: ``cache_size_limit`` up to torch 2.5,
    ``recompile_limit`` from 2.6. torch is unpinned here, so keying on the new
    name alone made the bump a silent no-op for every user on an older torch --
    and the cap it was there to lift is exactly what makes Dynamo give up and
    fall back to eager on the many-wavefield equations. ``accumulated_``
    is the secondary cap on the old spelling and has to move with it.
    """
    global _DYNAMO_LIMIT_WARNED
    cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if cfg is None:
        return
    names = [n for n in ("recompile_limit", "cache_size_limit",
                         "accumulated_recompile_limit", "accumulated_cache_size_limit")
             if hasattr(cfg, n)]
    primary = [n for n in names if not n.startswith("accumulated_")]
    if not primary:
        if not _DYNAMO_LIMIT_WARNED:
            _DYNAMO_LIMIT_WARNED = True
            warnings.warn(
                f"torch {torch.__version__}'s Dynamo exposes neither "
                "recompile_limit nor cache_size_limit, so the recompile cap "
                "was left alone; torch.compile may fall back to eager on the "
                "many-wavefield equations.", RuntimeWarning, stacklevel=3)
        return
    for name in names:
        if getattr(cfg, name) < target:
            setattr(cfg, name, target)


class _PropTorchEager(
    PropBase, torch.nn.Module, _CustomGradientMixin, _EagerBoundarySavingMixin
):
    def __init__(self, *args, **kwargs):
        self.store_last_wavefield = kwargs.pop("store_last_wavefield", EAGER_DEFAULTS.store_last_wavefield)
        self.use_compile = kwargs.pop("use_compile", EAGER_DEFAULTS.use_compile)
        self.compile_backend = kwargs.pop("compile_backend", EAGER_DEFAULTS.compile_backend)
        self.compile_mode = kwargs.pop("compile_mode", EAGER_DEFAULTS.compile_mode)
        self.compile_dynamic = kwargs.pop("compile_dynamic", EAGER_DEFAULTS.compile_dynamic)
        self.compile_fullgraph = kwargs.pop("compile_fullgraph", EAGER_DEFAULTS.compile_fullgraph)
        self.cpml_strips = kwargs.pop("cpml_strips", EAGER_DEFAULTS.cpml_strips)
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

        self.register_buffer("dt", torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer("dh", torch.tensor(self._grid_spacing, device=self.dev, dtype=torch.float32))
        coord_offset = list(self._runtime_coord_offset())
        self.register_buffer("coord_offset", torch.tensor(coord_offset, device=self.dev, dtype=torch.long))
        self.source_indices = [self.wavefield_names.index(name) for name in self.source_type]
        self.receiver_indices = [self.wavefield_names.index(name) for name in self.receiver_type]
        # CPML strips: the eager state holds a strip-held field as its two strips.
        self._strips = StripLayout(self.equation) if self.cpml_strips else None
        if self._strips is not None and len(self._strips) == len(self.wavefield_names):
            self._strips = None                     # no field to strip
        if self._strips is not None:
            self.source_indices = [self._strips.slots(k)[0] for k in self.source_indices]
            self.receiver_indices = [self._strips.slots(k)[0] for k in self.receiver_indices]
        self._equation_spacing = self._normalize_equation_spacing(self._grid_spacing)
        self._workspace_cache = {}
        # User-registered custom gradient / imaging conditions.
        #   {param_name: (imaging_fn, mode)}  with mode in {"gradient", "imaging"}
        self._custom_gradients = {}
        self._plain_step_func = self._build_step_func()
        self.step_func = self._maybe_compile(self._plain_step_func)

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
        equation = self.equation
        strips = self._strips
        num_wavefields = len(self.wavefield_names) if strips is None else len(strips)
        source_indices = tuple(self.source_indices)

        def step_func(*flat_args, source=None):
            wavefields = flat_args[:num_wavefields]
            models = flat_args[num_wavefields:-3]
            dt, h, b = flat_args[-3:]
            # ``equation.func`` is looked up here, not bound outside: Dynamo
            # before torch 2.4 cannot call a bound method held in a closure.
            if strips is not None:
                wavefields = strips.group(wavefields)
            out = equation.func(wavefields, models, dt, h, b)
            if strips is not None:
                out = strips.flatten(out)
            if source is None:
                return out
            # Injected inside the step, the compiled graph scatters the source
            # straight into the field it just computed: no full-grid pass.
            out = list(out)
            for k in source_indices:
                out[k] = scatter_source(out[k], *source)
            return out

        return step_func

    def _maybe_compile(self, fn):
        """``torch.compile`` *fn* with the propagator's compile settings, after
        raising Dynamo's recompile limit.  Dynamo specializes the step on
        per-tensor metadata, ``requires_grad`` included, and the graphs of every
        propagator a process builds collect on the step's one code object; at
        the default cap of 8 it silently falls back to eager.  (A rollout runs
        its first steps, while the wavefields' ``requires_grad`` still changes,
        uncompiled: see ``_SettledSteps``.)  Returns *fn* unchanged when
        compilation is off.  Shared by the full-tape step and the eager
        boundary-saving step.

        The compiled step is wrapped so that a compiler that cannot build it
        (Inductor's CPU backend needs g++ >= 10, for one) costs speed, not the
        run: the step is a pure function of its arguments, so the failing call
        is repeated uncompiled, and so is every later one.  On torch 2.3,
        Inductor runs with buffer reuse off, and not at all on CPU
        (INDUCTOR_23)."""
        if not self.use_compile:
            return fn
        if not hasattr(torch, "compile"):
            self._no_compile(f"torch {torch.__version__} has no torch.compile (it needs torch >= 2.0)")
            return fn
        inductor = self.compile_backend in (None, "inductor")
        if INDUCTOR_23 and inductor and "cuda" not in str(self.dev):
            self._no_compile(f"torch {torch.__version__}'s Inductor miscompiles the step on CPU "
                             "(fixed in torch 2.4)")
            return fn
        _raise_dynamo_recompile_limit()
        compile_kwargs = {
            "mode": self.compile_mode,
            "dynamic": self.compile_dynamic,
            "fullgraph": self.compile_fullgraph,
        }
        if self.compile_backend is not None:
            compile_kwargs["backend"] = self.compile_backend
        if INDUCTOR_23 and inductor:
            compile_kwargs["options"] = _no_buffer_reuse(compile_kwargs.pop("mode"), self.compile_dynamic)
        compiled = torch.compile(_with_own_code(fn) if OWN_CODE_PER_COMPILE else fn, **compile_kwargs)
        # Weak: the step ends up inside autograd graphs (the eager boundary-
        # saving Function keeps it in its ctx) that tensors cached on this
        # propagator keep alive.  A strong reference closes a cycle through the
        # C++ graph that gc cannot see, and a dropped propagator would keep its
        # workspace and boundary ring on the device for good.
        owner = weakref.ref(self)
        # Compiling takes seconds to minutes and prints nothing: every new compile --
        # a new grid size or shot count (the wavefields' shape), a new set of models
        # that require grad -- is announced before it starts (sweep._compile). The
        # models are arguments of the step, and _SettledSteps keeps one set of flags
        # per rollout, so the notice repeats a line only for a new compile.
        grid = "x".join(str(n) for n in self._shape_phys)
        notice = compile_notice(f"the {type(self.equation).__name__} step",
                                lambda shape: f"{shape[0]} shots on a {grid} grid"
                                if shape and shape[0] > 1 else f"a {grid} grid")

        def step(*args, **kwargs):
            prop = owner()
            if prop is not None and not prop.use_compile:
                return fn(*args, **kwargs)
            notice(args, kwargs)
            try:
                return compiled(*args, **kwargs)
            except _compiler_failures() as exc:
                if prop is not None:
                    prop._disable_compile(exc)
                return fn(*args, **kwargs)

        return step

    def _no_compile(self, why):
        """This torch cannot compile the step (``why``): run uncompiled, and
        say so once per process (the step is built per propagator and per
        rollout)."""
        global _NO_COMPILE_WARNED
        self.use_compile = False
        if not _NO_COMPILE_WARNED:
            _NO_COMPILE_WARNED = True
            warnings.warn(
                f"{why}: the eager propagator runs uncompiled, which gives the same result but "
                "is several times slower (3-6x measured on GPU). Pass "
                "eager_options=EagerOptions(use_compile=False) to skip this notice.",
                RuntimeWarning, stacklevel=4)

    def _disable_compile(self, exc):
        """Run uncompiled from now on, and say so once."""
        self.use_compile = False
        hint = compiler_hint("cuda" if "cuda" in str(self.dev) else "cpu")
        warnings.warn(
            f"torch.compile could not build the {type(self.equation).__name__} step on "
            f"{self.dev} ({type(exc).__name__}: {failure_reason(exc)}); running it uncompiled, "
            f"which is slower but gives the same result.{hint} Pass "
            "eager_options=EagerOptions(use_compile=False) to skip the attempt.",
            RuntimeWarning, stacklevel=3)

    def _mark_compile_step_begin(self):
        if self.use_compile and self.dev is not None and "cuda" in str(self.dev):
            compiler = getattr(torch, "compiler", None)
            if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
                compiler.cudagraph_mark_step_begin()

    def _compiled_step(self, wavefields, models, dt, h, b, source=None, compiled=True):
        """One step; ``source`` (``SourceTorch.step_args``) is injected into
        the source fields inside it.  ``compiled=False`` runs it uncompiled
        (see ``_SettledSteps``)."""
        if not compiled:
            return self._plain_step_func(*wavefields, *models, dt, h, b, source=source)
        self._mark_compile_step_begin()
        return self.step_func(*wavefields, *models, dt, h, b, source=source)

    def _sample_receivers(self, rec, wavefield, receivers):
        """One record column, ``(B, nrec, len(receiver_type))``."""
        if len(self.receiver_indices) > 1:
            return rec.sample_fields([wavefield[idx] for idx in self.receiver_indices])
        return rec(wavefield[self.receiver_indices[0]]).view(*receivers.shape[:-1]).unsqueeze(-1)

    def _prepare_runtime_models(self, models):
        prepare = getattr(self.equation, "prepare_models", None)
        if callable(prepare):
            return prepare(models)
        return models

    def _runtime_batch_layout(self, models, batch_size):
        """Shape padded models to the eager batch layout, per-shot or shared.

        The eager wavefield is laid out as ``(B, 1, *spatial)`` (batch, singleton
        channel, spatial). A model must broadcast against that:

        * **shared** — ``m.ndim == self.ndim`` (e.g. ``(nz, nx)`` in 2-D): the
          model is left untouched so it broadcasts over both the batch and
          channel axes, exactly as before. The shared path is therefore
          bit-identical to the pre-batched-model behaviour.
        * **per-shot** — ``m.ndim == self.ndim + 1`` (e.g. ``(B, nz, nx)``): a
          singleton channel axis is inserted -> ``(B, 1, *spatial)`` so shot
          ``b`` multiplies wavefield ``b`` with model ``b`` (no cross-shot
          broadcast). Without the inserted axis the model's leading batch dim
          would collide with the wavefield's channel dim and broadcast to
          ``(B, B, *spatial)``. Only enabled for equations advertising
          ``supports_batched_models`` (currently the 2-D Acoustic and Elastic
          solvers); every other equation keeps raising rather than silently
          mis-broadcasting.
        """
        laid_out = []
        for m in models:
            if m.ndim == self.ndim:
                laid_out.append(m)
                continue
            if m.ndim == self.ndim + 1:
                if not (self.ndim == 2 and getattr(self.equation, "supports_batched_models", False)):
                    raise NotImplementedError(
                        "Per-shot batched velocity models (a leading batch dim) are "
                        "only supported by 2-D solvers whose kernels stride the "
                        "model per batch index (currently Acoustic and Elastic); "
                        f"got a {m.ndim}-D model for {type(self.equation).__name__} "
                        f"(ndim={self.ndim}). Pass a single shared {self.ndim}-D "
                        "model broadcast across the batch instead."
                    )
                if m.shape[0] != batch_size:
                    raise ValueError(
                        f"Per-shot model batch ({int(m.shape[0])}) must equal the "
                        f"shot batch size ({int(batch_size)}); provide exactly one "
                        "model per shot as (B, nz, nx)."
                    )
                laid_out.append(m[:, None, ...])
                continue
            raise ValueError(
                f"Model tensor has {m.ndim} dims; expected {self.ndim} for a shared "
                f"model or {self.ndim + 1} for a per-shot batched (B, ...) model."
            )
        return laid_out

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

    def _run_chunk(self, wavefield, models, dt, h, b, wavelet_steps, nt, start_t, chunk_size, src, rec, receivers,
                   settled, adj=False):
        columns = []
        for t in range(start_t, min(start_t + chunk_size, nt)):
            time = t if not adj else nt - t - 1
            wavefield = list(self._compiled_step(wavefield, models, dt, h, b,
                                                 source=src.step_args(wavelet_steps[time]),
                                                 compiled=settled.compiled(t, wavefield)))
            columns.append(self._sample_receivers(rec, wavefield, receivers))
        return tuple(wavefield), torch.stack(columns, dim=1)

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), (
            f"Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}"
        )
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    # ------------------------------------------------------------------
    # Custom gradient / imaging-condition registration
    # ------------------------------------------------------------------

    def _pad_models(self, models):
        """EdgePad physical models to runtime grid + apply equation prep."""
        models_p = self._curvilinear_models(
            [self._as_device_tensor(m, dtype=torch.float32) for m in models])
        models_p = [EdgePadding.apply(m, self._runtime_padding()) for m in models_p]
        return models_p, self._prepare_runtime_models(models_p)

    def _crop_to_model(self, runtime_grid, model_shape):
        """A runtime-grid (…, nz_rt, nx_rt) image as the gradient of the
        model: cropped as EdgePadding.backward crops (the pads are per face,
        so not centred: a free surface has none on top), then reshaped to the
        model and, on a boundary-fitted grid, taken back to the model grid."""
        crop = [slice(None)] * runtime_grid.dim()
        pad = self._runtime_padding()
        for i in range(len(pad) // 2):
            lo, hi = pad[2 * i], pad[2 * i + 1]
            crop[-(i + 1)] = slice(lo, -hi if hi > 0 else None)
        g = runtime_grid[tuple(crop)].reshape(model_shape)
        return self._curvilinear_model_grad(g)

    def _rollout_ckpt(self, wavefield, models, wavelet_steps, src, rec,
                      receivers, nt, adj):
        """Time loop under chunk gradient checkpointing: the nt steps are split
        into ``ckpt_chunks``-sized chunks, each wrapped in ``ckpt_torch`` so its
        activations are recomputed in backward instead of stored.  Returns
        ``(wavefield, record)``, the record concatenated from the chunks' own
        (see ``_rollout_full`` for why it is not written in place).
        """
        chunk_size = int(max(1, self.ckpt_chunks))
        num_wavefields = len(wavefield)
        num_models = len(models)

        settled = _SettledSteps()
        chunk_records = []
        start = 0
        if self.use_compile and CKPT_PLAIN_WARMUP:
            # Before torch 2.5, compiling inside a checkpointed forward saves
            # tensors its recompute does not, and backward raises a
            # CheckpointError.  Only the first compiled step compiles (see
            # _SettledSteps), so run the steps up to and including it on the
            # plain tape.
            runtime_models = self._prepare_runtime_models(models)
            while start < min(nt, 8) and settled.first is None:
                wavefield, column = self._run_chunk(
                    wavefield, runtime_models, self.dt, self._equation_spacing, None,
                    wavelet_steps, nt, start, 1, src, rec, receivers, settled, adj=adj)
                wavefield = list(wavefield)
                chunk_records.append(column)
                start += 1

        for start_t in range(start, nt, chunk_size):

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
                    wavelet_steps,
                    nt,
                    start_t,
                    chunk_size,
                    src,
                    rec,
                    receivers,
                    settled,
                    adj=adj,
                )

            wavefield, chunk_record = ckpt_torch(checkpoint_chunk, *wavefield, *models, use_reentrant=False)
            wavefield = list(wavefield)
            chunk_records.append(chunk_record)
        return wavefield, torch.cat(chunk_records, dim=1)

    def _rollout_full(self, wavefield, runtime_models, wavelet_steps, src, rec,
                      receivers, nt, adj, return_wavefield, snapshots, snapshot_lookup):
        """Plain time loop recording the full autograd tape (the default path):
        PyTorch retains every step's activations and backward differentiates the
        whole graph.  This is the only path that supports ``return_wavefield``
        (wavefield snapshots).  Returns ``(wavefield, record)``; the snapshots,
        which are detached host copies, are still written in place.
        """
        settled = _SettledSteps()
        columns = []
        for i in range(nt):
            time = i if not adj else nt - i - 1
            wavefield = list(self._compiled_step(wavefield, runtime_models, self.dt, self._equation_spacing, None,
                                                 source=src.step_args(wavelet_steps[time]),
                                                 compiled=settled.compiled(i, wavefield)))
            if return_wavefield and i in snapshot_lookup:
                snapshots[snapshot_lookup[i]] = torch.stack(
                    [self._crop_runtime_halo(w).detach().cpu() for w in wavefield],
                    0,
                )
            columns.append(self._sample_receivers(rec, wavefield, receivers))
        # One stack instead of nt in-place slice writes. Writing into a live
        # autograd tensor builds a chain of nt CopySlices nodes, and EACH of
        # them allocates a full-record buffer and copies the incoming gradient
        # through it, so the record alone cost O(nt^2) backward traffic:
        # 2*nt*|record|, which is 64 GB per shot at nt=4000 with 500 receivers.
        # Stacking makes it 2*|record|.
        return wavefield, torch.stack(columns, dim=1)

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
        # physical grid indices onto a boundary-fitted grid, which counts its
        # rows from the surface (identity on a Cartesian grid); both the
        # autograd and the custom-gradient paths below take the mapped ones
        sources = self._place_on_curvilinear(sources, "source")
        receivers = self._place_on_curvilinear(receivers, "receiver")
        # and its snapshots back onto the Cartesian grid, unless the caller
        # wants the grid's own rows (the surface on row 0 of every column)
        snapshot_grid = kwargs.pop("snapshot_grid", None)
        if snapshot_grid is not None and not return_wavefield:
            raise ValueError("snapshot_grid needs return_wavefield=True.")
        if snapshot_grid not in (None, "physical", "computational"):
            raise ValueError(
                f"snapshot_grid must be 'physical' or 'computational'; got {snapshot_grid!r}")
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
        if self._strips is not None:
            self._strips.size_strips(self.equation, kwargs["shape"], self.dev)

        nt = wavelet.shape[-1]
        snapshot_indices = self._resolve_snapshot_times(nt, return_wavefield, snapshot_times, snapshot_interval)
        snapshot_lookup = {t: i for i, t in enumerate(snapshot_indices)}
        self.snapshot_times_last = tuple(snapshot_indices)

        shape_wavefield = (batch_size, 1) + self._runtime_shape()
        wavelet = self._as_device_tensor(wavelet, dtype=torch.float32)
        # One contiguous view per step: the compiled step takes a step's sample
        # as an input, and a strided one would specialize the compiled code on
        # nt.  Unbinding once also gathers the wavelet's gradient in one stack,
        # where a slice per step cost a full-wavelet backward per step.
        wavelet_steps = wavelet.movedim(-1, 0).contiguous().unbind(0)
        sources = self._as_device_tensor(sources, dtype=torch.long) + self.coord_offset
        receivers = self._as_device_tensor(receivers, dtype=torch.long) + self.coord_offset

        # Equations whose collocated stencils have a checkerboard null space
        # (the rotated staggered grid) declare a small smoothing stencil that
        # is folded into the source injection and the receiver gather so
        # point sources/receivers neither excite nor sample the spurious
        # mode. None (the default) keeps plain single-cell behaviour.
        sr_stencil = getattr(self.equation, "source_receiver_stencil", None)
        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding, adj,
                          spread_kernel=sr_stencil)
        rec = ReceiverTorch(receivers, gather_kernel=sr_stencil, shape=shape_wavefield)

        has_aux = False
        if return_wavefield and self._strips is not None:
            raise NotImplementedError("return_wavefield with cpml_strips: the CPML fields are strips.")
        if return_wavefield:
            has_aux = True
            snapshot_shape = (len(snapshot_indices), len(self.wavefield_names), batch_size, 1) + self.shape
            # NOT through _get_cached_tensor: that cache has no eviction policy,
            # so the snapshot buffer -- whose size grows linearly in the number
            # of snapshot steps, and which defaults to EVERY step -- would be
            # pinned to this propagator for its whole lifetime, long after the
            # caller stopped holding the snapshots. It is also the one workspace
            # that is handed to the caller, so a fresh allocation is what lets
            # it be returned without the defensive clone below, halving the host
            # memory this path costs.
            snapshots = torch.zeros(snapshot_shape, device=torch.device("cpu"),
                                    dtype=torch.float32)
        else:
            snapshots = None

        models = models if models is not None else self.parameters()
        # physical-grid models onto a boundary-fitted grid's rows (identity on
        # a Cartesian grid), differentiably: the gradient is on the model grid
        models = self._curvilinear_models(
            [self._as_device_tensor(para, dtype=torch.float32) for para in models])
        models = [EdgePadding.apply(para, self._runtime_padding()) for para in models]
        self.models_padded = models
        # Insert a singleton channel axis for per-shot batched models so each
        # shot propagates in its own model; shared models pass through unchanged
        # (keeping the shared path bit-exact). Done after storing models_padded
        # so that attribute stays the raw padded model, matching the C backend.
        models = self._runtime_batch_layout(models, batch_size)
        runtime_models = self._prepare_runtime_models(models)
        shapes = ([shape_wavefield] * len(self.wavefield_names) if self._strips is None
                  else self._strips.shapes(shape_wavefield))
        wavefield = [
            self._get_cached_tensor(f"wavefield:{index}", shape, device=self.dev, dtype=torch.float32)
            for index, shape in enumerate(shapes)
        ]
        if self.use_ckpt and self.ckpt_mode != "chunk":
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}' for PropTorch. Expected 'chunk'.")
        if self.use_ckpt and return_wavefield:
            raise ValueError("return_wavefield=True is not supported with chunk checkpointing in PropTorch yet.")

        # Every rollout builds its record fresh (stacked, never a workspace
        # buffer), so it is returned without a defensive copy.
        if self.use_ckpt:
            wavefield, record = self._rollout_ckpt(
                wavefield, models, wavelet_steps, src, rec, receivers, nt, adj
            )
        elif getattr(self, "_eager_bs", False):
            if return_wavefield:
                raise ValueError("return_wavefield is not supported with eager boundary saving.")
            wavefield, record = self._rollout_eager_bs(
                wavefield, runtime_models, wavelet_steps, src, rec, receivers, nt, adj
            )
        else:
            wavefield, record = self._rollout_full(
                wavefield, runtime_models, wavelet_steps, src, rec, receivers, nt, adj,
                return_wavefield, snapshots, snapshot_lookup,
            )

        self.last_wavefields = tuple(wavefield) if self.store_last_wavefield else None
        if not has_aux:
            return record
        # ``snapshots`` is allocated per call above, so it aliases no workspace
        # and needs no defensive copy -- and may be mapped in place.
        if snapshot_grid != "computational":
            snapshots = self._curvilinear_snapshots(snapshots)
        return record, snapshots

    forward_base = forward

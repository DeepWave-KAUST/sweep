import torch
from torch.utils.checkpoint import checkpoint as ckpt_torch

from sweep.propagator.base import PropBase
from sweep.propagator.options import EAGER_DEFAULTS
from sweep.receivers.torch import ReceiverTorch
from sweep.sources.torch import SourceTorch
from sweep.utils.torch import EdgePadding


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

        if not self.use_compile or not hasattr(torch, "compile"):
            return step_func
        compile_kwargs = {
            "mode": self.compile_mode,
            "dynamic": self.compile_dynamic,
            "fullgraph": self.compile_fullgraph,
        }
        if self.compile_backend is not None:
            compile_kwargs["backend"] = self.compile_backend
        return torch.compile(step_func, **compile_kwargs)

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

    def forward(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        source_encoding=False,
        adj=False,
        return_wavefield=False,
        **kwargs,
    ):
        source_encoding = bool(source_encoding or self._auto_detect_source_encoding(wavelet, sources, receivers))
        if models is not None and not isinstance(models, (list, tuple)):
            models = list(models)

        snapshot_times = kwargs.pop("snapshot_times", None)
        snapshot_interval = kwargs.pop("snapshot_interval", None)
        kwargs.setdefault("fd_pad", self._runtime_fd_pad())
        kwargs.setdefault("shape", self._runtime_shape())
        self.init_abc(**kwargs)

        nt = wavelet.shape[-1]
        snapshot_indices = self._resolve_snapshot_times(nt, return_wavefield, snapshot_times, snapshot_interval)
        snapshot_lookup = {t: i for i, t in enumerate(snapshot_indices)}
        self.snapshot_times_last = tuple(snapshot_indices)
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
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
        else:
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

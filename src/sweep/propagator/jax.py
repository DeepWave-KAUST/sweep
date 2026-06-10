import jax
import jax.numpy as jnp
from sweep.propagator.base import PropBase
from sweep.sources.jax import SourceJax
from sweep.receivers.jax import ReceiverJax
from sweep.utils.jax import edge_pad

class PropJax(PropBase):

    def __init__(self, *args, memory=None, **kwargs):

        super().__init__(*args, **kwargs)
        self._bs_options = None
        if memory is not None:
            self._apply_memory_options(memory)

    def _apply_memory_options(self, memory):
        """Apply a ``MemoryOptions`` dataclass (the impl-agnostic memory API,
        same as PropTorch's ``memory=``): ``strategy='boundary'`` enables
        boundary-saving reconstruction, ``strategy='ckpt'`` enables chunked
        ``jax.checkpoint`` rematerialisation.  The two are mutually exclusive.
        """
        from sweep.propagator.options import MemoryOptions
        if not isinstance(memory, MemoryOptions):
            raise TypeError("memory= expects a MemoryOptions instance.")
        if memory.strategy == "boundary":
            boundary = memory.boundary
            storage = getattr(boundary, "storage", "gpu")
            if storage != "gpu":
                raise ValueError(
                    "JAX boundary saving keeps the ring storage on device "
                    "(BoundaryOptions.storage='gpu'); cpu/disk staging is "
                    "available on the torch paths."
                )
            self.use_ckpt = False
            self.enable_boundary_saving(
                True, storage_dtype=getattr(boundary, "storage_dtype", "fp32")
            )
        elif memory.strategy == "ckpt":
            ckpt = memory.ckpt
            mode = getattr(ckpt, "mode", "chunk") if ckpt is not None else "chunk"
            if mode != "chunk":
                raise ValueError("JAX checkpointing supports mode='chunk' only.")
            self._bs_options = None
            self.use_ckpt = True
            if ckpt is not None and getattr(ckpt, "chunks", None):
                self.ckpt_chunks = ckpt.chunks
        else:
            raise ValueError(
                f"Unsupported memory strategy {memory.strategy!r} for PropJax."
            )

    def enable_boundary_saving(self, flag=True, mode=None, storage_dtype="fp32"):
        """Route the rollout through boundary-saving reconstruction (see
        ``_jax_boundary_saving``) instead of differentiating through the scan
        tape.  Mirrors ``PropTorch.enable_eager_boundary_saving``.  ``mode``
        forces a reverse driver ('swap2nd' | 'substep'); None auto-dispatches
        on the equation's time scheme.  ``storage_dtype`` ('fp32' | 'fp16' |
        'bf16') compresses the saved ring only — compute stays FP32."""
        if flag and storage_dtype not in ("fp32", "fp16", "bf16"):
            raise NotImplementedError(
                "JAX boundary saving supports storage_dtype 'fp32'/'fp16'/'bf16' "
                f"(got {storage_dtype!r}); int8 is only on the torch paths."
            )
        self._bs_options = (
            {"mode": mode, "storage_dtype": storage_dtype} if flag else None
        )

    def _forward_bs(self, wavefields, models, wavelet, src, rec,
                    source_indices, receiver_indices, nt, batch_size):
        """Boundary-saving rollout: O(nt × ring) residual memory via a custom
        VJP whose backward reconstructs the forward wavefield in reverse time
        (``_jax_boundary_saving``)."""
        from sweep.propagator import _jax_boundary_saving as jbs

        cfg = jbs.build_config(
            equation=self.equation,
            wavefield_specs=self._wavefield_spec_index,
            wavefield_names=self.wavefield_names,
            dt=self._dt,
            h=jnp.asarray(self._grid_spacing, dtype=jnp.float32),
            shape=tuple(wavefields[0].shape),
            ndim=self.ndim,
            abcn=self.abcn,
            halo=self._runtime_fd_halo(),
            nt=nt,
            S0=wavefields,
            src=src,
            rec=rec,
            source_indices=source_indices,
            receiver_indices=receiver_indices,
            batch_size=batch_size,
            free_surface=getattr(self, "free_surface", False),
            forced_mode=self._bs_options.get("mode"),
            storage_dtype=self._bs_options.get("storage_dtype", "fp32"),
        )
        propagate = jbs.make_propagate(cfg)
        rec_seq = propagate(tuple(models), wavelet)    # [nt, B, nrec, ntype]
        return jnp.transpose(rec_seq, (1, 0, 2, 3))

    def pad(self, d, padding=None):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        if padding is None:
            padding = self._runtime_padding()
        else:
            padding = tuple(padding)
        padding = self._spatial_pad_pairs(padding)
        padding = (((0,0),)*(d.ndim-self.ndim)+padding)
        return edge_pad(d, padding)#jnp.pad(d, (padding_z, padding_x), mode='edge') DONOT USE jnp.pad
        # return jnp.pad(d, padding, mode='edge')

    def jaxpad(self, d, padding=None):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        if padding is None:
            padding = self._runtime_padding()
        else:
            padding = tuple(padding)
        padding = self._spatial_pad_pairs(padding)
        padding = (((0,0),)*(d.ndim-self.ndim)+padding)
        return jnp.pad(d, padding, mode='edge')
    
    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, jnp.array(data))

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
    
    def forward_base(self,
                     wavelet,
                     sources,
                     receivers,
                     models=None,
                     return_wavefield=False,
                     adj=False,
                     wave_equation=None,
                     aux_args=tuple(),
                     **kwargs,):
        """Forward pass of the wave equation.

        See :meth:`PropBase._normalize_io` for accepted ``wavelet`` /
        ``sources`` / ``receivers`` shapes (modes A1, A2, B).
        """
        fd_pad = self._runtime_fd_pad()
        snapshot_times = kwargs.pop("snapshot_times", None)
        snapshot_interval = kwargs.pop("snapshot_interval", None)
        kwargs.setdefault('fd_pad', fd_pad)
        kwargs.setdefault('shape', self._runtime_shape())
        self.init_abc(**kwargs)
        if getattr(self.equation, 'setup_pml', None):
            self.equation.setup_pml(self.pml_type)
        mode, batch_size, nsrc_per_shot, _, source_encoding = self._normalize_io(
            wavelet, sources, receivers
        )

        wavelet = jnp.array(wavelet, dtype=jnp.float32)
        wavelet = jnp.atleast_2d(wavelet)

        nt = wavelet.shape[-1]
        snapshot_indices = self._resolve_snapshot_times(
            nt,
            return_wavefield,
            snapshot_times,
            snapshot_interval,
        )
        self.snapshot_times_last = tuple(snapshot_indices)

        shape_wavefield = (batch_size, 1) + self._runtime_shape()

        sources = sources.copy()
        receivers = receivers.copy()

        sources = jnp.array(sources, dtype=jnp.int32)
        receivers = jnp.array(receivers, dtype=jnp.int32)

        coord_offset = jnp.asarray(self._runtime_coord_offset(), dtype=jnp.int32)
        sources = sources + coord_offset
        receivers = receivers + coord_offset

        src = SourceJax(sources, shape_wavefield, source_encoding, adj)
        rec = ReceiverJax(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, jnp.zeros(shape_wavefield, dtype=jnp.float32))

        self.models_padded = models

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = jnp.zeros(
                (len(snapshot_indices), len(self.wavefield_names), batch_size, 1) + self.shape,
                dtype=jnp.float32,
                device=jax.devices('cpu')[0],
            )
            snapshot_targets = jnp.asarray(snapshot_indices + [nt], dtype=jnp.int32)
        else:
            snapshots = None
            snapshot_targets = None

        step_args = (self._dt, jnp.asarray(self._grid_spacing, dtype=jnp.float32), None)

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        ridxs = jnp.asarray(receiver_idx_at, dtype=jnp.int32)  
        sidxs = jnp.asarray(source_idx_at, dtype=jnp.int32)

        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])

        if self._bs_options is not None:
            if adj:
                raise NotImplementedError(
                    "Boundary saving with adj=True is not supported on the JAX path."
                )
            if return_wavefield:
                raise ValueError(
                    "return_wavefield=True is not supported with boundary saving."
                )
            if self.use_ckpt:
                raise ValueError(
                    "Boundary saving and chunk checkpointing are mutually exclusive."
                )
            if wave_equation is not None or aux_args:
                raise NotImplementedError(
                    "Boundary saving supports the equation's default func only."
                )
            return self._forward_bs(
                wavefields, models, wavelet, src, rec,
                source_idx_at, receiver_idx_at, nt, batch_size,
            )

        chunk_size = int(max(1, self.ckpt_chunks))

        num_chunks = (nt + chunk_size - 1) // chunk_size

        wave_equation = getattr(self.equation, f'func') if wave_equation is None else wave_equation
        
        zero_rec = jnp.zeros(
            (batch_size, receivers.shape[1], len(self.receiver_type)),
            dtype=jnp.float32
        )
        num_snapshots = len(snapshot_indices)

        def step_fn_single(carry, it):
                
            wavefields, step_args, snapshots, snapshot_pos = carry
            dt, h, b = step_args

            time = it if not adj else nt - it - 1
            # Forward propagation
            wavefields = wave_equation(wavefields, models, dt, h, b, *aux_args)
            wavefields_arr = jnp.stack(wavefields, axis=0)
            # Add source
            wf_src = jnp.take(wavefields_arr, sidxs, axis=0)
            wf_src_new = jax.vmap(lambda w: src(w, wavelet[..., time]))(wf_src)
            wavefields_arr = wavefields_arr.at[sidxs].set(wf_src_new)
            # Save snapshots
            if snapshots is not None:
                should_save = jnp.logical_and(
                    snapshot_pos < num_snapshots,
                    snapshot_targets[snapshot_pos] == it,
                )

                def save_snapshot(state):
                    snap_buf, snap_idx = state
                    cropped = self._crop_runtime_halo(wavefields_arr)
                    snap_buf = snap_buf.at[snap_idx].set(cropped)
                    return snap_buf, snap_idx + 1

                snapshots, snapshot_pos = jax.lax.cond(
                    should_save,
                    save_snapshot,
                    lambda state: state,
                    (snapshots, snapshot_pos),
                )

            # Record receivers
            wf_sel = jnp.take(wavefields_arr, ridxs, axis=0) 
            def one_channel(wf):
                y = rec(wf)
                return y.reshape(batch_size, -1, y.shape[-1])
            all_rec = jax.vmap(one_channel)(wf_sel)
            rec_t = jnp.transpose(all_rec, (1, 2, 0, 3))[..., 0]

            return (tuple(wavefields_arr[i] for i in range(wavefields_arr.shape[0])), step_args, snapshots, snapshot_pos), rec_t


        def step_fn_single_with_skip(carry, it):

            def do_step(c):
                return step_fn_single(c, it)

            def skip_fn(c):
                return c, zero_rec
            
            carry, rec_t = jax.lax.cond(it < nt, do_step, skip_fn, carry)

            return carry, rec_t

        def run_chunk(carry, chunk_start):
            chunk_record = jnp.zeros(
                (chunk_size, batch_size, receivers.shape[1], len(self.receiver_type)),
                dtype=jnp.float32,
            )

            def body_fn(local_i, state):
                current_carry, current_record = state
                t = chunk_start + local_i
                next_carry, rec_t = step_fn_single_with_skip(current_carry, t)
                current_record = current_record.at[local_i].set(rec_t)
                return next_carry, current_record

            return jax.lax.fori_loop(
                0,
                chunk_size,
                body_fn,
                (carry, chunk_record),
            )

        checkpointed_run_chunk = jax.checkpoint(run_chunk)

        def chunked_step_fn(carry, chunk_idx):
            chunk_start = chunk_idx * chunk_size
            return checkpointed_run_chunk(carry, chunk_start)
        
        initial = (wavefields, step_args, snapshots, jnp.array(0, dtype=jnp.int32))
        step_fn = step_fn_single if not self.use_ckpt else chunked_step_fn
        num_steps = num_chunks if self.use_ckpt else nt
        (final), rec_seq = jax.lax.scan(step_fn, initial, jnp.arange(num_steps))

        n = num_steps * chunk_size if self.use_ckpt else nt
        rec_seq = rec_seq.reshape(n, batch_size, receivers.shape[1], len(self.receiver_type))
        rec = jnp.transpose(rec_seq, (1, 0, 2, 3))[:, :nt, :, :]

        return rec if not has_aux else (rec, final[2])
    
    def forward(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        return_wavefield=False,
        adj=False,
        wave_equation=None,
        aux_args=tuple(),
        **kwargs,
    ):
        return self.__call__(
            wavelet,
            sources,
            receivers,
            models=models,
            return_wavefield=return_wavefield,
            adj=adj,
            wave_equation=wave_equation,
            aux_args=aux_args,
            **kwargs,
        )

    def __call__(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        return_wavefield=False,
        adj=False,
        wave_equation=None,
        aux_args=tuple(),
        **kwargs,
    ):
        models = models if models is not None else self.parameters()
        models = [self.pad(para, self._runtime_padding()) for para in models]
        return self.forward_base(
            wavelet,
            sources,
            receivers,
            models=models,
            return_wavefield=return_wavefield,
            adj=adj,
            wave_equation=wave_equation,
            aux_args=aux_args,
            **kwargs,
        )
    
    def __call_forward__(self,*args, **kwargs):
        """ This function is useful when you want to compile forward modeling.
        """
        models = kwargs.pop("models", None)
        models = models if models is not None else self.parameters()
        models = [self.jaxpad(para, self._runtime_padding()) for para in models]
        return self.forward_base(*args, models=models, **kwargs)

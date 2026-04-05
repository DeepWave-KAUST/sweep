import jax
import jax.numpy as jnp
from sweep.propagator.base import PropBase
from sweep.sources.jax import SourceJax
from sweep.receivers.jax import ReceiverJax
from sweep.utils.jax import edge_pad

class PropJax(PropBase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def pad(self, d, padding=None):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        if padding is None:
            padding = self.padding
        padding = (self.padding_z,) + ((self.abcn,self.abcn), )* (self.ndim-1) 
        padding = (((0,0),)*(d.ndim-self.ndim)+padding)
        return edge_pad(d, padding)#jnp.pad(d, (padding_z, padding_x), mode='edge') DONOT USE jnp.pad
        # return jnp.pad(d, padding, mode='edge')

    def jaxpad(self, d, padding=None):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        if padding is None:
            padding = self.padding
        padding = (self.padding_z,) + ((self.abcn,self.abcn), )* (self.ndim-1) 
        padding = (((0,0),)*(d.ndim-self.ndim)+padding)
        return jnp.pad(d, padding, mode='edge')
    
    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, jnp.array(data))
    
    def forward_base(self, 
                     wavelet, 
                     sources, 
                     receivers, 
                     models=None,
                     source_encoding=False, 
                     return_wavefield=False, 
                     adj=False, 
                     wave_equation=None, 
                     aux_args=tuple(),
                     **kwargs,):
        """Forward pass of the wave equation

        Args:
            wavelet (jnp.array): Wavelet tensor (nt,)
            sources (jnp.array): Source coordinates (nshots, 2)
            receivers (jnp.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
            source_encoding (bool, optional): If True, the sources are encoded in the wavefield. Defaults to False.
            return_wavefield (bool, optional): If True, return the wavefields. Defaults to False.
            adj (bool, optional): If True, run the adjoint forward modeling. Defaults to False.
            wave_equation (callable, optional): The wave equation function to use. If None, use the equation defined in the class. Defaults to None.
            aux_args (tuple(list), optional): Auxiliary arguments for the wave equation function. Defaults to ().
        """
        fd_pad = [0, 0] * self.ndim
        kwargs.setdefault('fd_pad', fd_pad)
        self.init_abc(**kwargs)
        if getattr(self.equation, 'setup_pml', None):
            self.equation.setup_pml(self.pml_type)

        wavelet = jnp.array(wavelet, dtype=jnp.float32)
        wavelet = jnp.atleast_2d(wavelet)

        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        shape_wavefield = (batch_size, 1) + self.shape

        sources = sources.copy()
        receivers = receivers.copy()

        sources = jnp.array(sources, dtype=jnp.int32)
        receivers = jnp.array(receivers, dtype=jnp.int32)

        if self.free_surface:
            sources = sources.at[..., :-1].add(self.abcn)
            receivers = receivers.at[..., :-1].add(self.abcn)
        else:
            sources = sources.at[...].add(self.abcn)
            receivers = receivers.at[...].add(self.abcn)

        src = SourceJax(sources, shape_wavefield, source_encoding, adj)
        rec = ReceiverJax(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, jnp.zeros(shape_wavefield, dtype=jnp.float32))

        self.models_padded = models

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = jnp.zeros((nt, len(self.wavefield_names)) + shape_wavefield, dtype=jnp.float32, device=jax.devices('cpu')[0]) #
        else:
            snapshots = None

        fixargs = [self._dt, self._dh, None]

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        ridxs = jnp.asarray(receiver_idx_at, dtype=jnp.int32)  
        sidxs = jnp.asarray(source_idx_at, dtype=jnp.int32)

        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])

        chunk_size = self.ckpt_chunks

        num_chunks = (nt + chunk_size - 1) // chunk_size

        wave_equation = getattr(self.equation, f'func') if wave_equation is None else wave_equation
        
        zero_rec = jnp.zeros(
            (batch_size, receivers.shape[1], len(self.receiver_type)),
            dtype=jnp.float32
        )

        def step_fn_single(carry, it):
                
            wavefields, fixargs, snapshots = carry

            time = it if not adj else nt - it - 1
            # Forward propagation
            wavefields = wave_equation(*wavefields, *models, *fixargs, *aux_args)
            wavefields_arr = jnp.stack(wavefields, axis=0)
            # Add source
            wf_src = jnp.take(wavefields_arr, sidxs, axis=0)
            wf_src_new = jax.vmap(lambda w: src(w, wavelet[..., time]))(wf_src)
            wavefields_arr = wavefields_arr.at[sidxs].set(wf_src_new)
            # Save snapshots
            if snapshots is not None:
                snapshots = snapshots.at[it].set(jnp.stack(wavefields, 0))

            # Record receivers
            wf_sel = jnp.take(wavefields_arr, ridxs, axis=0) 
            def one_channel(wf):
                y = rec(wf)
                return y.reshape(batch_size, -1, y.shape[-1])
            all_rec = jax.vmap(one_channel)(wf_sel)
            rec_t = jnp.transpose(all_rec, (1, 2, 0, 3))[..., 0]

            return (tuple(wavefields_arr[i] for i in range(wavefields_arr.shape[0])), fixargs, snapshots), rec_t


        def step_fn_single_with_skip(carry, it):

            def do_step(c):
                return step_fn_single(c, it)

            def skip_fn(c):
                return c, zero_rec
            
            carry, rec_t = jax.lax.cond(it < nt, do_step, skip_fn, carry)

            return carry, rec_t

        def chunked_step_fn(carry, chunk_idx):

            def inner_step_fn(carry, it):
                t = chunk_idx * chunk_size + it
                return step_fn_single_with_skip(carry, t)
            return jax.checkpoint(lambda carry, idxs: 
                jax.lax.scan(inner_step_fn, carry, jnp.arange(chunk_size))
            )(carry, None)
        
        initial = (wavefields, tuple(fixargs), snapshots)
        step_fn = step_fn_single if not self.use_ckpt else chunked_step_fn
        num_steps = num_chunks if self.use_ckpt else nt
        (final), rec_seq = jax.lax.scan(step_fn, initial, jnp.arange(num_steps))

        n = num_steps * chunk_size if self.use_ckpt else nt
        rec_seq = rec_seq.reshape(n, batch_size, receivers.shape[1], len(self.receiver_type))
        rec = jnp.transpose(rec_seq, (1, 0, 2, 3))[:, :nt, :, :]

        return rec if not has_aux else (rec, final[-1])
    
    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        models = kwargs.pop("models", None)
        models = models if models is not None else self.parameters()
        models = [self.pad(para, self.padding) for para in models]
        return self.forward_base(*args, models=models, **kwargs)
    
    def __call_forward__(self,*args, **kwargs):
        """ This function is useful when you want to compile forward modeling.
        """
        models = kwargs.pop("models", None)
        models = models if models is not None else self.parameters()
        models = [self.jaxpad(para, self.padding) for para in models]
        return self.forward_base(*args, models=models, **kwargs)

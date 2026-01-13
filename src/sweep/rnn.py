
import jax, inspect
import jax.numpy as jnp
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint as ckpt_torch
from .sources import SourceTorch, SourceJax
from .receivers import ReceiverTorch, ReceiverJax
from .abc import abc_coefficients_2d, abc_coefficients_3d
from .cpml import set_pml_profiles
from sweep.equations.registry import CPML_SUPPORTED_EQUATIONS, FREE_SURFACE_SUPPORTED_EQUATIONS

from .utils import EdgePadding, edge_pad

class RNNBase:

    def __init__(self,
                 equation, 
                 shape, 
                 dev, 
                 source_type: list=[],
                 receiver_type: list=[],
                 abcn=50, 
                 free_surface=False, 
                 dh=10., 
                 dt=0.002, 
                 use_ckpt=True,
                 ckpt_chunks=100,
                 use_habc=False,
                 use_cpml=False,
                 **kwargs):
        """Base class for the RNN

        Args:
            equation (class): The wave equation class from sweep.equations
            shape (tupel or list): The shape of the model
            dev (torch.device): For pytorch, the dev should be torch.device, for jax, it will automatically detect the device
            source_type (list, optional): List of strings for the source type. Defaults to [].
            receiver_type (list, optional): List of strings for the receiver type. Defaults to [].
            abcn (int, optional): The number of layers of absorbing boundary conditions. Defaults to 50.
            free_surface (bool, optional): If the model has a free surface. Defaults to False.
            dh (float, optional): Grid spacing (meters). Defaults to 10..
            dt (float, optional): Time step (seconds). Defaults to 0.002.
            use_ckpt (bool, optional): Use checkpointing to save memory. Defaults to True.
            ckpt_chunks (int, optional): The number of time steps to chunk for checkpointing. Defaults to 50.
            use_habc (bool, optional): Use HABC instead of PML. Defaults to False.
        """
        
        self.equation = equation
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.shape = shape
        self.dev = dev
        self.abcn = abcn
        self.free_surface = free_surface
        self._dh = float(dh)
        self._dt = float(dt)
        self.use_ckpt = use_ckpt
        self.ckpt_chunks = ckpt_chunks
        self.ndim = len(shape)
        self.use_cpml = use_cpml

        self.abc_func = {2: abc_coefficients_2d, 3: abc_coefficients_3d}[self.ndim]

        if self.equation.__class__.__name__ not in FREE_SURFACE_SUPPORTED_EQUATIONS and self.free_surface:
            raise NotImplementedError(f'Free surface is not implemented for {self.equation.__class__.__name__} equation. Please set free_surface=False.')

        if self.equation.__class__.__name__ not in CPML_SUPPORTED_EQUATIONS and self.use_cpml:
            raise NotImplementedError(f'CPML is not implemented for {self.equation.__class__.__name__} equation. Please set use_cpml=False.')

        self.source_type = source_type
        self.receiver_type = receiver_type

        if self.free_surface:
            self.padding_z = (0, self.abcn)
            shape_z = self.shape[0] + self.abcn
        else:
            self.padding_z = (self.abcn, self.abcn)
            shape_z = self.shape[0] + 2*self.abcn

        self.padding = (self.abcn,) * 2*(self.ndim-1) + self.padding_z
        self.shape_nopad = tuple([w+2*self.equation.so for w in self.shape])
        self.shape = (shape_z,) + tuple(s+2*self.abcn for s in self.shape[1:])
        self.init_abc(**kwargs)

    def init_abc(self, **kwargs):
        # Coefficients for absorbing boundary conditions must be initialized after the shape is set

        # A bad ABC
        self.b = self.abc_func(self.shape, N=self.abcn, free_surface=self.free_surface)

        # CPML (best ABC performance)
        if self.use_cpml:
            self.b = set_pml_profiles(
                    pml_width=[self.abcn if not self.free_surface else 0] + (2**self.ndim-1) * [self.abcn],
                    accuracy=self.equation.so,
                    fd_pad=[self.equation.so//2 if not self.free_surface else 0] + (2**self.ndim-1) * [self.equation.so//2],
                    dt=self._dt,
                    grid_spacing=[self._dh]*self.ndim,
                    max_vel=kwargs.get('max_vel', 4500.0),
                    dtype=np.float32,
                    pml_freq=kwargs.get('pml_freq', 25.0),
                    shape=self.shape,
                )
            
        if getattr(self.equation, 'need_init', False):
            self.equation.init(self.shape, self.dev, self._dh)

    def crop(self, data):
        """Crop the data to the original shape

        Args:
            data (torch.Tensor or jnp.ndarray): The data to be cropped

        Returns:
            torch.Tensor or jnp.ndarray: The cropped data
        """
        if self.free_surface:
            return data[..., 0:-self.abcn, self.abcn:-self.abcn]
        else:
            s = slice(self.abcn, -self.abcn)
            return data[(...,) + (s,) * self.ndim]

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]
    
    @property
    def dh(self):
        """Grid spacing in meters"""
        return self._dh
    
    # @property
    # def dt(self):
    #     """Time step in seconds"""
    #     return self._dt
    
class RNNTorch(RNNBase, torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)
        self.setup_abc()

    def setup_abc(self, ):

        # Absorbing boundary conditions
        if isinstance(self.b, list):
            self.b = [torch.from_numpy(bi).to(self.dev)[None, ...] for bi in self.b]
        else:
            self.b = torch.from_numpy(self.b).to(self.dev)

        self.register_buffer('h', torch.tensor(self._dh, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(self._dt, dtype=torch.float32))

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False, **kwargs):
        """Forward pass of the wave equation

        Args:
            wavelet (np.array): Wavelet tensor (nt,)
            sources (np.array): Source coordinates (nshots, 2)
            receivers (np.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

        self.init_abc(**kwargs)
        self.setup_abc()

        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        shape_wavefield = (batch_size, 1) + self.shape
        sources = sources.copy()
        receivers = receivers.copy()

        if self.free_surface:
            sources[..., 0] += self.abcn
            receivers[..., 0] += self.abcn
        else:
            sources += self.abcn
            receivers += self.abcn

        wavelet = torch.from_numpy(wavelet).to(self.dev).float()
        sources = torch.from_numpy(sources).to(self.dev).long()
        receivers = torch.from_numpy(receivers).to(self.dev).long()

        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding, adj)
        rec = ReceiverTorch(receivers)

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = torch.zeros((nt, len(self.wavefield_names)) + shape_wavefield, dtype=torch.float32, device=torch.device('cpu'))
        else:
            snapshots = None

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, torch.zeros(shape_wavefield, device=self.dev))

        record = torch.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=torch.float32, device=self.dev)

        # Get the model parameters
        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, self.padding) for para in models]
        self.models_padded = models
        fixargs = models+[self.dt, self.h, self.b]
    
        for i in range(nt):

            wavefield = [getattr(self, name) for name in self.wavefield_names]

            # Time step forward
            if self.use_ckpt:
                wavefield = ckpt_torch(self.equation.func, *wavefield, *fixargs, use_reentrant=False)
            else:
                wavefield = self.equation.func(*wavefield, *fixargs)

            if return_wavefield:
                snapshots[i] = torch.stack([w.detach().cpu() for w in wavefield], 0)

            # Exchange wavefields
            for name, data in zip(self.wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.source_type:
                time = i if not adj else nt - i - 1
                setattr(self, source_type, src(getattr(self, source_type), wavelet[..., time]))

            # Record wavefields
            for ic, receiver_type in enumerate(self.receiver_type):
                record[:, i, :, ic] = rec(getattr(self, receiver_type)).view(*receivers.shape[:-1])

        if not has_aux:
            return record
        else:
            return record, snapshots

    forward_base = forward
    
class RNNJax(RNNBase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.setup_abc()

    def setup_abc(self,):
        
        # Absorbing boundary conditions
        if isinstance(self.b, list):
            self.b = [jnp.array(bi)[None, ...] for bi in self.b]
        else:
            self.b = jnp.array(self.b)

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
        self.init_abc(**kwargs)
        self.setup_abc()

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

        fixargs = [self._dt, self._dh, self.b]

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
    
class RNNCUDA(RNNBase):

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
        return np.pad(d, padding, mode='edge')
    
    def forward(self, 
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
        
        nt = wavelet.shape[-1]
        # import conv2d_cpp

        sources = sources.copy()
        receivers = receivers.copy()

        if self.free_surface:
            sources[..., 0] += self.abcn
            receivers[..., 0] += self.abcn
        else:
            sources += self.abcn
            receivers += self.abcn

        models = [self.pad(para, self.padding) for para in models]

        record = wave_equation.forward(models[0], self.equation.kernel, self.b, wavelet, sources, receivers, None, None, nt, self._dt, self._dh)

        # print(kernel.shape)
        # exit()
        # record = conv2d_cpp.forward(models[0], kernel, self.b, wavelet, sources, receivers, nt, self.dt, self._dh)

        return record
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class RNN:

    def __init__(self, equation, *args, **kwargs):
        backend = getattr(equation, 'backend', 'jax').lower()

        backend_func = {'torch': RNNTorch, 
                        'jax': RNNJax, 
                        'cuda': RNNCUDA}

        if backend in backend_func.keys():
            self._impl = backend_func[backend](equation, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        impl_doc = getattr(self._impl.forward_base, "__doc__", None)
        type(self).__call__.__doc__ = impl_doc

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __setattr__(self, name, value):
        if name == '_impl':
            super().__setattr__(name, value)
        else:
            setattr(self._impl, name, value)

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

RNNJax.__init__.__doc__ = RNNBase.__init__.__doc__
RNNJax.__call__.__doc__ = RNNJax.forward_base.__doc__
RNNTorch.__init__.__doc__ = RNNBase.__init__.__doc__

RNN.__init__.__doc__ = RNNBase.__init__.__doc__

__all__ = ["RNN", "RNNJax"]

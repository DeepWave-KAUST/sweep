
import jax, inspect
# a=jax.numpy.array([2.0])
import jax.numpy as jnp
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint as ckpt_torch
from .sources import SourceTorch, SourceJax
from .receivers import ReceiverTorch, ReceiverJax
from .abc import abc_coefficients_2d, abc_coefficients_3d, habc_coefficients_2d
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
                 ckpt_chunks=50,
                 use_habc=False,
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
        self.use_habc = use_habc

        self.abc_func = {2: abc_coefficients_2d, 3: abc_coefficients_3d}[self.ndim]

        if self.equation.__class__.__name__ not in ['AcousticVRZ', 'Acoustic', 'AcousticLSRTM', 'AEC', 'AECLSRTM', 'Acoustic1st'] and self.free_surface:
            raise NotImplementedError(f'Free surface is not implemented for {self.equation.__class__.__name__} equation. Please set free_surface=False.')

        self.source_type = source_type
        self.receiver_type = receiver_type

        if self.free_surface:
            self.padding_z = (0, self.abcn)
            shape_z = self.shape[0] + self.abcn
        else:
            self.padding_z = (self.abcn, self.abcn)
            shape_z = self.shape[0] + 2*self.abcn

        self.padding = (self.abcn,) * 2*(self.ndim-1) + self.padding_z
        self.shape = (shape_z,) + tuple(s+2*self.abcn for s in self.shape[1:])

        # Coefficients for absorbing boundary conditions must be initialized after the shape is set
        self.b = self.abc_func(self.shape, N=self.abcn, free_surface=self.free_surface)
        ######### HABC
        if self.use_habc:
            self.b = habc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)

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
            return data[..., self.abcn:-self.abcn, self.abcn:-self.abcn]

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
        self.b = torch.from_numpy(self.b).to(self.dev)

        self.register_buffer('h', torch.tensor(self._dh, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(self._dt, dtype=torch.float32))

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False):
        """Forward pass of the wave equation

        Args:
            wavelet (np.array): Wavelet tensor (nt,)
            sources (np.array): Source coordinates (nshots, 2)
            receivers (np.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

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
                wavefield = ckpt_torch(self.equation.func, *wavefield, *fixargs)
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
    
class RNNJax(RNNBase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.setup_abc()

    def setup_abc(self, ):

        # Absorbing boundary conditions
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

        ############# For HABC
        if getattr(self.equation, 'init_habc', False) and self.ndim==2:
            self.equation.init_habc(self.shape, self.abcn, self.free_surface, batchsize=batch_size, use_habc=self.use_habc)
        #############

        record = jnp.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=jnp.float32)

        self.models_padded = models

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = jnp.zeros((nt, len(self.wavefield_names)) + shape_wavefield, dtype=jnp.float32) #, device=jax.devices('cpu')[0]
        else:
            snapshots = None

        fixargs = [self._dt, self._dh, self.b]

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        
        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])

        chunk_size = self.ckpt_chunks

        num_chunks = (nt + chunk_size - 1) // chunk_size

        post_fix = '_jax3d' if self.ndim == 3 else ''
        wave_equation = getattr(self.equation, f'func{post_fix}') if wave_equation is None else wave_equation
        
        def step_fn_single(carry, it):

            def do_step(carry):
                
                wavefields, fixargs, snapshots, _rec = carry

                time = it if not adj else nt - it
                # Forward propagation
                wavefields = wave_equation(*wavefields, *models, *fixargs, *aux_args)
                wavefields = list(wavefields)

                # Add source
                for sidx in source_idx_at:
                    wavefields[sidx] = src(wavefields[sidx], wavelet[..., time])
                wavefields = tuple(wavefields)

                # Save snapshots
                if snapshots is not None:
                    snapshots = snapshots.at[it].set(jnp.stack(wavefields, 0))

                # Record receivers
                for channel, ridx in enumerate(receiver_idx_at):
                    rec_this_step = jnp.array(jnp.split(rec(wavefields[ridx]), batch_size))
                    _rec = _rec.at[:, it, :, channel:channel+1].set(rec_this_step)

                return (wavefields, fixargs, snapshots, _rec)
        
            def skip_fn(carry):
                return carry
            
            carry = jax.lax.cond(it < nt, do_step, skip_fn, carry)

            return carry, None
        

        def chunked_step_fn(carry, chunk_idx):

            def inner_step_fn(carry, it):
                t = chunk_idx * chunk_size + it
                return step_fn_single(carry, t)
            return jax.checkpoint(lambda carry, idxs: 
                jax.lax.scan(inner_step_fn, carry, jnp.arange(chunk_size))
            )(carry, None)
        
        initial = (wavefields, tuple(fixargs), snapshots, record)
        step_fn = step_fn_single if not self.use_ckpt else chunked_step_fn
        num_steps = num_chunks if self.use_ckpt else nt
        (final), _ = jax.lax.scan(step_fn, initial, jnp.arange(num_steps))
        rec = final[-1]

        return rec if not has_aux else (rec, final[-2])
    
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

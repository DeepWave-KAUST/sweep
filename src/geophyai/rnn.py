
import jax, inspect
a=jax.numpy.array([2.0])
import jax.numpy as jnp
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint as ckpt_torch
from .sources import SourceTorch, SourceJax
from .receivers import ReceiverTorch, ReceiverJax
from .abc import abc_coefficients_2d
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
                 **kwargs):
        """Base class for the RNN

        Args:
            equation (class): The wave equation class from geophyai.equations
            shape (tupel or list): The shape of the model
            dev (torch.device): For pytorch, the dev should be torch.device, for jax, it will automatically detect the device
            source_type (list, optional): List of strings for the source type. Defaults to [].
            receiver_type (list, optional): List of strings for the receiver type. Defaults to [].
            abcn (int, optional): The number of layers of absorbing boundary conditions. Defaults to 50.
            free_surface (bool, optional): If the model has a free surface. Defaults to False.
            dh (float, optional): Grid spacing (meters). Defaults to 10..
            dt (float, optional): Time step (seconds). Defaults to 0.002.
            use_ckpt (bool, optional): Use checkpointing to save memory. Defaults to True.
        """
        
        self.equation = equation
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.shape = shape
        self.dev = dev
        self.abcn = abcn
        self.free_surface = free_surface
        self._dh = dh
        self._dt = dt
        self.use_ckpt = use_ckpt

        if self.equation.__class__.__name__ not in ['Acoustic', 'AcousticLSRTM'] and self.free_surface:
            raise NotImplementedError(f'Free surface is not implemented for {self.equation.__class__.__name__} equation. Please set free_surface=False.')

        self.source_type = source_type
        self.receiver_type = receiver_type

        if self.free_surface:
            self.padding = (self.abcn, self.abcn, 0, self.abcn) # left, right. top, bottom, refer to torch.nn.functional.pad
            self.shape = (self.shape[0]+self.abcn, self.shape[1]+2*self.abcn)
        else:
            self.padding = (self.abcn, )*4 # left, right. top, bottom, refer to  torch.nn.functional.pad
            self.shape = (self.shape[0]+2*self.abcn, self.shape[1]+2*self.abcn)

        if getattr(self.equation, 'need_init', False):
            self.equation.init(self.shape, self.dev, self._dh)
        self.b = abc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)

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

    def forward(self, wavelet, sources, receivers, models=None, sill=None, rill=None, source_encoding=False):
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

        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding=source_encoding)
        rec = ReceiverTorch(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, torch.zeros(shape_wavefield, device=self.dev))

        record = torch.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=torch.float32, device=self.dev)

        # Get the model parameters
        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, self.padding) for para in models]
        
        fixargs = models+[self.dt, self.h, self.b]

        # def save_grad(grad):
        #     rill[:] += torch.sum(grad[..., self.abcn:-self.abcn, self.abcn:-self.abcn]**2, 0).squeeze()

        for i in range(nt):

            wavefield = [getattr(self, name) for name in self.wavefield_names]

            # if wavefield[0].requires_grad and rill is not None:
                # wavefield[0].register_hook(save_grad)
            # if sill is not None:
                # sill[:] += torch.sum(wavefield[0][..., self.abcn:-self.abcn, self.abcn:-self.abcn].detach()**2, 0).squeeze()

            # Time step forward
            if self.use_ckpt:
                wavefield = ckpt_torch(self.equation.func, *wavefield, *fixargs)
            else:
                wavefield = self.equation.func(*wavefield, *fixargs)

            # if i % 100 == 0:
                # np.save(f'data/wavefield_{i:04d}.npy', np.stack([w.detach().cpu().numpy() for w in wavefield]))

            # Exchange wavefields
            for name, data in zip(self.wavefield_names, wavefield):
                setattr(self, name, data)

            # Add source
            for source_type in self.source_type:
                setattr(self, source_type, src(getattr(self, source_type), wavelet[..., i]))

            # Record wavefields
            for ic, receiver_type in enumerate(self.receiver_type):
                record[:, i, :, ic] = rec(getattr(self, receiver_type)).view(*receivers.shape[:-1])

        return record
    
class RNNJax(RNNBase):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.setup_abc()

    def setup_abc(self, ):

        # Absorbing boundary conditions
        self.b = abc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)
        self.b = jnp.array(self.b)

    def pad(self, d, padding=None):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        if padding is None:
            padding = self.padding
        padding_z = (padding[2], padding[3])
        padding_x = (padding[0], padding[1])
        padding = (padding_z, padding_x)
        if d.ndim == 4: padding = (((0,0),)*2+padding) # Model split case, the input velocity is 4D (batch, 1, nz, nx)
        return edge_pad(d, padding)#jnp.pad(d, (padding_z, padding_x), mode='edge') DONOT USE jnp.pad
    
    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, jnp.array(data))

    def forward(self, wavelet, sources, receivers, models=None, sill=None, rill=None, source_encoding=False, return_wavefield=False, adj=False):
        """Forward pass of the wave equation

        Args:
            wavelet (jnp.array): Wavelet tensor (nt,)
            sources (jnp.array): Source coordinates (nshots, 2)
            receivers (jnp.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

        wavelet = jnp.array(wavelet, dtype=jnp.float32)
        
        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        shape_wavefield = (batch_size, 1) + self.shape

        sources = sources.copy()
        receivers = receivers.copy()

        sources = jnp.array(sources, dtype=jnp.int32)
        receivers = jnp.array(receivers, dtype=jnp.int32)

        if self.free_surface:
            sources = sources.at[..., 0].add(self.abcn)
            receivers = receivers.at[..., 0].add(self.abcn)
        else:
            sources = sources.at[...].add(self.abcn)
            receivers = receivers.at[...].add(self.abcn)

        src = SourceJax(sources, shape_wavefield, source_encoding, adj)
        rec = ReceiverJax(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, jnp.zeros(shape_wavefield, dtype=jnp.float32))

        record = jnp.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=jnp.float32)

        # Get the model parameters
        models = models if models is not None else self.parameters()

        models = [self.pad(para, self.padding) for para in models]

        self.models_padded = models

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = jnp.zeros((nt, len(self.wavefield_names)) + shape_wavefield, dtype=jnp.float32, device=jax.devices('cpu')[0])
        else:
            snapshots = None

        fixargs = [self._dt, self._dh, self.b]

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        def step_fn(carry, it):

            wavefields, fixargs, snapshots, _rec  = carry

            # Forward
            wavefields = self.equation.func_jax(*wavefields, *models, *fixargs)

            # Apply source
            wavefields = list(wavefields)

            for sidx in source_idx_at:
                time = it if not adj else nt - it
                wavefields[sidx] = src(wavefields[sidx], wavelet[..., time])
            wavefields = tuple(wavefields)

            # Snapshots
            if snapshots is not None:
                snapshots = snapshots.at[it].set(jnp.stack(wavefields, 0))

            # Measure probe(s)
            for channel, ridx in enumerate(receiver_idx_at):
                rec_this_step = jnp.array(jnp.split(rec(wavefields[ridx]), batch_size))
                _rec = _rec.at[:, it, :, channel:channel+1].set(rec_this_step)
            
            return (wavefields, fixargs, snapshots, _rec), None
        
        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])

        step_fn = jax.checkpoint(step_fn) if self.use_ckpt else step_fn
        # step_fn = step_fn
        initial = (wavefields, tuple(fixargs), snapshots, record)
        (final), _ = jax.lax.scan(step_fn, initial, jnp.arange(nt))
        rec = final[-1]
        if not has_aux:
            return rec
        else:
            return rec, final[-2]
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
RNNJax.__init__.__doc__ = RNNBase.__init__.__doc__
RNNJax.__call__.__doc__ = RNNJax.forward.__doc__
RNNTorch.__init__.__doc__ = RNNBase.__init__.__doc__

RNN = RNNTorch
RNN.__init__.__doc__ = RNNBase.__init__.__doc__

__all__ = ["RNN", "RNNJax"]
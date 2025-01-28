
import jax, torch, inspect
import jax.numpy as jnp
import numpy as np
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

        self.source_type = source_type
        self.receiver_type = receiver_type

        if self.free_surface:
            self.padding = (self.abcn, self.abcn, 0, self.abcn) # left, right. top, bottom, refer to torch.nn.functional.pad
            self.shape = (self.shape[0]+self.abcn, self.shape[1]+2*self.abcn)
        else:
            self.padding = (self.abcn, )*4 # left, right. top, bottom, refer to  torch.nn.functional.pad
            self.shape = (self.shape[0]+2*self.abcn, self.shape[1]+2*self.abcn)
        self.b = abc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]
    
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

        def save_grad(grad):
            rill[:] += torch.sum(grad[..., self.abcn:-self.abcn, self.abcn:-self.abcn]**2, 0).squeeze()

        for i in range(nt):

            wavefield = [getattr(self, name) for name in self.wavefield_names]

            if wavefield[0].requires_grad and rill is not None:
                wavefield[0].register_hook(save_grad)
            if sill is not None:
                sill[:] += torch.sum(wavefield[0][..., self.abcn:-self.abcn, self.abcn:-self.abcn].detach()**2, 0).squeeze()

            # Time step forward
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

    def pad(self, d, padding):
        """Padding the model parameters

        Args:
            padding (list): 4 elements list for padding the model parameters
        """
        padding_z = (padding[0], padding[1])
        padding_x = (padding[2], padding[3])
        return edge_pad(d, (padding_z, padding_x))#jnp.pad(d, (padding_z, padding_x), mode='edge')
    
    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, jnp.array(data))

    def forward(self, wavelet, sources, receivers, models=None, sill=None, rill=None, source_encoding=False):
        """Forward pass of the wave equation

        Args:
            wavelet (jnp.array): Wavelet tensor (nt,)
            sources (jnp.array): Source coordinates (nshots, 2)
            receivers (jnp.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        shape_wavefield = (batch_size, 1) + self.shape
        sources = sources.copy()
        receivers = receivers.copy()

        if self.free_surface:
            sources = sources.at[..., 0] + self.abcn
            receivers = receivers.at[..., 0] + self.abcn
        else:
            sources = sources + self.abcn
            receivers = receivers + self.abcn

        sources = jnp.array(sources, dtype=jnp.int32)
        receivers = jnp.array(receivers, dtype=jnp.int32)

        src = SourceJax(sources, shape_wavefield, source_encoding=source_encoding)
        rec = ReceiverJax(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, jnp.zeros(shape_wavefield, dtype=jnp.float32))

        record = jnp.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=jnp.float32)

        # Get the model parameters
        models = models if models is not None else self.parameters()
        models = [self.pad(para, self.padding) for para in models]
        
        fixargs = models+[self._dt, self._dh, self.b]

        source_idx_at = []
        receiver_idx_at = []

        for source_type in self.source_type:
            source_idx_at.append(self.wavefield_names.index(source_type))

        for receiver_type in self.receiver_type:
            receiver_idx_at.append(self.wavefield_names.index(receiver_type))

        def step_fn(carry, it):

            wavefields, fixargs, _rec = carry

            # Forward
            wavefields = self.equation.func_jax(*wavefields, *fixargs)

            # Apply source
            wavefields = list(wavefields)
            for sidx in source_idx_at:
                wavefields[sidx] = src(wavefields[sidx], wavelet[..., it])
            wavefields = tuple(wavefields)

            # Measure probe(s)
            for channel, ridx in enumerate(receiver_idx_at):
                rec_this_step = jnp.array(jnp.split(rec(wavefields[ridx]), batch_size))
                _rec = _rec.at[:, it, :, channel:channel+1].set(rec_this_step)
            
            return (wavefields, fixargs, _rec), None
        
        wavefields = tuple([getattr(self, name) for name in self.wavefield_names])

        initial = (wavefields, tuple(fixargs), record)
        (final), _ = jax.lax.scan(step_fn, initial, jnp.arange(nt))
        rec = final[-1]

        return rec
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
RNNJax.__init__.__doc__ = RNNBase.__init__.__doc__
RNNTorch.__init__.__doc__ = RNNBase.__init__.__doc__

RNN = RNNTorch
RNN.__init__.__doc__ = RNNBase.__init__.__doc__

__all__ = ["RNN", "RNNJax"]
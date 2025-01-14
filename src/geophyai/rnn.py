
import torch
import numpy as np
from .sources import Source
from .receivers import Receiver
from .abc import abc_coefficients_2d
from .utils import EdgePadding

class RNN(torch.nn.Module):
    
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

        super(RNN, self).__init__()

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

        self.setup_abc()

    def setup_abc(self, ):
        if self.free_surface:
            self.padding = (self.abcn, self.abcn, 0, self.abcn) # left, right. top, bottom, refer to torch.nn.functional.pad
            self.shape = (self.shape[0]+self.abcn, self.shape[1]+2*self.abcn)
        else:
            self.padding = (self.abcn, )*4 # left, right. top, bottom, refer to  torch.nn.functional.pad
            self.shape = (self.shape[0]+2*self.abcn, self.shape[1]+2*self.abcn)

        # Absorbing boundary conditions
        self.b = abc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)
        self.b = torch.from_numpy(self.b).to(self.dev)

        self.register_buffer('h', torch.tensor(self._dh, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(self._dt, dtype=torch.float32))

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]

    def forward(self, wavelet, sources, receivers, sill=None, rill=None, source_encoding=False, models=None):
        """Forward pass of the wave equation

        Args:
            wavelet (torch.Tensor): Wavelet tensor (nt,)
            sources (torch.Tensor): Source coordinates (nshots, 2)
            receivers (torch.Tensor): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

        nt = wavelet.shape[0]
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

        sources = torch.from_numpy(sources).to(self.dev).long()
        receivers = torch.from_numpy(receivers).to(self.dev).long()

        src = Source(sources, shape_wavefield, self.dev, source_encoding=source_encoding)
        rec = Receiver(receivers)

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
                setattr(self, source_type, src(getattr(self, source_type), wavelet[i]))

            # Record wavefields
            for ic, receiver_type in enumerate(self.receiver_type):
                record[:, i, :, ic] = rec(getattr(self, receiver_type)).view(*receivers.shape[:-1])

        return record
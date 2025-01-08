
import torch
import numpy as np
from .sources import Source
from .receivers import Receiver
from .abc import abc_coefficients_2d
from .utils import EdgePadding

class RNN(torch.nn.Module):
    
    def __init__(self, equation, shape, dev, abcn=50, free_surface=False):

        super(RNN, self).__init__()

        self.equation = equation
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.shape = shape
        self.dev = dev
        self.abcn = abcn
        self.free_surface = free_surface

        self.setup_abc()

    def setup_abc(self, ):
        if self.free_surface:
            self.padding = (0, )+(self.abcn, )*3
            self.shape = (self.shape[0]+self.abcn, self.shape[1]+2*self.abcn)
        else:
            self.padding = (self.abcn, )*4
            self.shape = (self.shape[0]+2*self.abcn, self.shape[1]+2*self.abcn)

        # Absorbing boundary conditions
        self.b = abc_coefficients_2d(self.shape, N=self.abcn, free_surface=self.free_surface)
        self.b = torch.from_numpy(self.b).to(self.dev)

    def auxillary(self, ):
        self.register_buffer('h', torch.tensor(self.geom['h'], dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(self.geom['dt'], dtype=torch.float32))
        self.source_type = self.geom['source_type']
        self.receiver_type = self.geom['receiver_type']

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]

    def forward(self, wavelet, sources, receivers, sill, rill):
        """Forward pass of the wave equation

        Args:
            wavelet (torch.Tensor): Wavelet tensor (nt,)
            sources (torch.Tensor): Source coordinates (nshots, 2)
            receivers (torch.Tensor): Receiver coordinates (nshots, nreceivers, 2)
        """
        self.auxillary()

        nt = wavelet.shape[0]
        nshots = sources.shape[0]
        shape_wavefield = (nshots, 1) + self.shape

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

        src = Source(sources, shape_wavefield, self.dev)
        rec = Receiver(receivers)

        # Memory allocation for wavefields
        for name in self.wavefield_names:
            setattr(self, name, torch.zeros(shape_wavefield, device=self.dev))

        record = torch.zeros((sources.shape[0], nt, receivers.shape[1], len(self.receiver_type)), dtype=torch.float32, device=self.dev)
        fixargs = [EdgePadding.apply(para, self.padding) for para in self.parameters()] +[self.dt, self.h, self.b]

        for i in range(nt):

            wavefield = [getattr(self, name) for name in self.wavefield_names]

            # Time step forward
            wavefield = self.equation.func(*wavefield, *fixargs)

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
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_torch
from sweep.propagator.base import PropBase
from sweep.sources.torch import SourceTorch
from sweep.receivers.torch import ReceiverTorch
from sweep.utils.torch import EdgePadding


class PropTorch(PropBase, torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._dh, device=self.dev, dtype=torch.float32))

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


        # Extract adjoint wavefields
        # self.adjoint_wavefields = torch.zeros((nt, 5, *shape_wavefield), device=self.dev, dtype=torch.float32)
        # def hook_it(index, t):
        #     def _hook(grad):
        #         self.adjoint_wavefields[t, index] = grad.detach().clone()
        #     return _hook

        record = torch.zeros((batch_size, nt, receivers.shape[1], len(self.receiver_type)), dtype=torch.float32, device=self.dev)

        # Get the model parameters
        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, self.padding) for para in models]
        self.models_padded = models
        fixargs = models+[self.dt, self.dh, None]
    
        for i in range(nt):

            wavefield = [getattr(self, name) for name in self.wavefield_names]

            # # register hook for adjoint wavefield extraction
            # for w, name in zip(wavefield, self.wavefield_names[:5]):
            #     if w.requires_grad:
            #         w.register_hook(hook_it(self.wavefield_names.index(name), i))

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
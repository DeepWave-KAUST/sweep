import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_torch
from sweep.propagator.base import PropBase
from sweep.sources.torch import SourceTorch
from sweep.receivers.torch import ReceiverTorch
from sweep.utils.torch import EdgePadding


class PropTorch(PropBase, torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        self.use_compile = kwargs.pop('use_compile', False)
        self.compile_backend = kwargs.pop('compile_backend', None)
        self.compile_mode = kwargs.pop('compile_mode', 'default')
        self.compile_dynamic = kwargs.pop('compile_dynamic', False)
        self.compile_fullgraph = kwargs.pop('compile_fullgraph', False)
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._dh, device=self.dev, dtype=torch.float32))
        coord_offset = [self.abcn] * self.ndim
        if self.free_surface:
            coord_offset[-1] = 0
        self.register_buffer('coord_offset', torch.tensor(coord_offset, device=self.dev, dtype=torch.long))
        self.source_indices = [self.wavefield_names.index(name) for name in self.source_type]
        self.receiver_indices = [self.wavefield_names.index(name) for name in self.receiver_type]
        self.step_func = self._build_step_func()

    def _as_device_tensor(self, value, *, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.dev, dtype=dtype)
        return torch.as_tensor(value, device=self.dev, dtype=dtype)

    def _build_step_func(self):
        step_func = self.equation.func
        if not self.use_compile or not hasattr(torch, 'compile'):
            return step_func
        # Compile the single-step update instead of the full forward loop to keep
        # geometry setup and optional snapshot logic outside the graph.
        compile_kwargs = {
            'mode': self.compile_mode,
            'dynamic': self.compile_dynamic,
            'fullgraph': self.compile_fullgraph,
        }
        if self.compile_backend is not None:
            compile_kwargs['backend'] = self.compile_backend
        return torch.compile(step_func, **compile_kwargs)

    def _mark_compile_step_begin(self):
        if self.use_compile and self.dev is not None and 'cuda' in str(self.dev):
            compiler = getattr(torch, 'compiler', None)
            if compiler is not None and hasattr(compiler, 'cudagraph_mark_step_begin'):
                compiler.cudagraph_mark_step_begin()

    def _compiled_step(self, wavefield, fixargs):
        self._mark_compile_step_begin()
        return self.step_func(*wavefield, *fixargs)

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
        fd_pad = [0,0]*self.ndim
        kwargs.setdefault('fd_pad', fd_pad)
        self.init_abc(**kwargs)

        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        shape_wavefield = (batch_size, 1) + self.shape
        wavelet = self._as_device_tensor(wavelet, dtype=torch.float32)
        sources = self._as_device_tensor(sources, dtype=torch.long) + self.coord_offset
        receivers = self._as_device_tensor(receivers, dtype=torch.long) + self.coord_offset

        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding, adj)
        rec = ReceiverTorch(receivers)

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshots = torch.zeros((nt, len(self.wavefield_names)) + shape_wavefield, dtype=torch.float32, device=torch.device('cpu'))
        else:
            snapshots = None

        # # Extract adjoint wavefields
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
        wavefield = [torch.zeros(shape_wavefield, device=self.dev) for _ in self.wavefield_names]
        # import numpy as np
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ay', 'by', 'ayh', 'byh', 'ax', 'bx', 'axh', 'bxh']):
        #     np.save(f'{name}.npy', b.detach().cpu().numpy())
        for i in range(nt):

            # # register hook for adjoint wavefield extraction
            # for w, name in zip(wavefield, self.wavefield_names[:5]):
            #     if w.requires_grad:
            #         w.register_hook(hook_it(self.wavefield_names.index(name), i))

            # Time step forward
            if self.use_ckpt:
                wavefield = list(ckpt_torch(self.step_func, *wavefield, *fixargs, use_reentrant=False))
            else:
                wavefield = list(self._compiled_step(wavefield, fixargs))

            if return_wavefield:
                snapshots[i] = torch.stack([w.detach().cpu() for w in wavefield], 0)

            # Add source
            for source_idx in self.source_indices:
                time = i if not adj else nt - i - 1
                wavefield[source_idx] = src(wavefield[source_idx], wavelet[..., time])

            # Record wavefields
            for ic, receiver_idx in enumerate(self.receiver_indices):
                record[:, i, :, ic] = rec(wavefield[receiver_idx]).view(*receivers.shape[:-1])

        for name, data in zip(self.wavefield_names, wavefield):
            setattr(self, name, data)

        if not has_aux:
            return record
        else:
            return record, snapshots

    forward_base = forward

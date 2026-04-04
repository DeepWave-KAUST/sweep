import torch
import torch.nn.functional as F

import numpy as np
import sweep._C as _C
from sweep.memory.torch import Allocator
from sweep.memory.shape import Layout
from sweep.propagator.base import PropBase
from sweep.utils.torch import EdgePadding
from sweep.scalars import fd_coefficients

class Warpper(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        forward_func,
        backward_func,
        backward_bs_func,
        wavelet,             # (B, nsrc, nt)
        sources_loc,        # (B, nsrc, 2)
        receivers_loc,      # (B, nrec, 2)
        source_field_indices,
        receiver_field_indices,
        coes_list,          
        M: int,
        abcn: int,
        spacing: list,       # list of 2 floats for dx and dz
        dt: float,
        pml_vals: list,     # list of 6 tensors for PML profiles
        use_boundary_saving: bool=False,
        use_pinned_memory: bool=False,
        free_surface: bool=False,
        transfer_interval: int=1,
        boundary_on_cpu: bool=False,
        forward_wavefields: tuple=(),
        adjoint_wavefields: tuple=(),
        last_two: torch.Tensor=None,
        boundary_cpu: tuple=(),
        boundary_gpu: tuple=(),
        *models             # list of (B, nz, nx) tensors
    ):
        """
        Forward modeling: vp -> synthetic seismograms
        """
        
        lap_coes, grad_coes = coes_list
        nt = wavelet.shape[-1]
        spacing = [s.item() for s in spacing]
        dt = float(dt.item())
        
        # only forward modeling, no need to save wavefield for backward
        save_all_wavefields = False
        if any(m.requires_grad for m in models): 
            save_all_wavefields = True
        if any(m.requires_grad for m in models) and use_boundary_saving:
            save_all_wavefields = False
        if not any(m.requires_grad for m in models):
            save_all_wavefields = False
            use_boundary_saving = False
        
        params = _C.ForwardInput()
        params.wavefields = forward_wavefields
        params.last_two = last_two
        params.boundary_cpu = [b.zero_() for b in boundary_cpu] if boundary_on_cpu else []
        params.boundary_gpu = [b.zero_() for b in boundary_gpu] if use_boundary_saving else []
        params.transfer_interval = transfer_interval
        params.models = [m.contiguous() for m in models]
        params.source = wavelet.contiguous()
        params.lap_coes = lap_coes.contiguous()
        params.grad_coes = grad_coes.contiguous()
        params.M = M
        params.abcn = abcn
        params.sources_loc = sources_loc.contiguous()
        params.receivers_loc = receivers_loc.contiguous()
        params.source_field_indices = source_field_indices.contiguous()
        params.receiver_field_indices = receiver_field_indices.contiguous()
        params.pml_vals = [p.contiguous() for p in pml_vals]
        params.save_all_wavefields = save_all_wavefields
        params.use_boundary_saving = use_boundary_saving
        params.boundary_on_cpu = boundary_on_cpu
        params.use_pinned_memory = use_pinned_memory
        params.free_surface = free_surface
        params.nt = nt
        params.dt = dt
        params.spacing = spacing

        # -------- CUDA forward --------
        (u_allt, last, syn) = forward_func(params)
        if any([save_all_wavefields, use_boundary_saving]):
            
            ctx.save_for_backward(
                u_allt,
                last,
                sources_loc,
                receivers_loc,
                source_field_indices,
                receiver_field_indices,
                lap_coes, grad_coes,
            )
            ctx.transfer_interval = transfer_interval
            ctx.models = models
            ctx.boundary_on_cpu = boundary_on_cpu
            ctx.boundary_cpu = boundary_cpu if boundary_on_cpu else ()
            ctx.boundary_gpu = boundary_gpu if use_boundary_saving else ()
            ctx.pml_vals = pml_vals
            ctx.abcn = abcn
            ctx.M = M
            ctx.nt = nt
            ctx.spacing = spacing
            ctx.dt = dt
            ctx.free_surface = free_surface
            ctx.use_boundary_saving = use_boundary_saving
            ctx.use_pinned_memory = use_pinned_memory
            ctx.backward_func = backward_func
            ctx.backward_bs_func = backward_bs_func
            ctx.forward_source = wavelet

            ctx.adjoint_wavefields = adjoint_wavefields

        return syn
    
    @staticmethod
    def backward(ctx, adjoint_source):

        # -------- unpack --------
        (
            u_allt,
            last,
            forward_sources_loc,
            adjoint_sources_loc,
            source_field_indices,
            receiver_field_indices,
            lap_coes, grad_coes,
        ) = ctx.saved_tensors

        abcn = ctx.abcn
        M  = ctx.M
        nt = ctx.nt
        dt = ctx.dt

        params = _C.BackwardInput()
        # common
        params.transfer_interval = ctx.transfer_interval
        params.adjoint_wavefields = [a.zero_() for a in ctx.adjoint_wavefields]
        params.models = [m.contiguous() for m in ctx.models]
        params.adjoint_source = adjoint_source.contiguous()
        params.lap_coes = lap_coes.contiguous()
        params.grad_coes = grad_coes.contiguous()
        params.M = M
        params.abcn = abcn
        params.adjoint_sources_loc = adjoint_sources_loc.contiguous()
        params.source_field_indices = source_field_indices.contiguous()
        params.receiver_field_indices = receiver_field_indices.contiguous()
        params.pml_vals = [p.contiguous() for p in ctx.pml_vals]
        params.nt = nt
        params.dt = dt
        params.spacing = ctx.spacing
        params.free_surface = ctx.free_surface
        params.boundary_on_cpu = ctx.boundary_on_cpu
        params.use_pinned_memory = ctx.use_pinned_memory

        if not ctx.use_boundary_saving:
            params.u_forward = u_allt.contiguous()
            gradients = ctx.backward_func(params)
        else:
            params.boundary_cpu = list(ctx.boundary_cpu) if ctx.boundary_on_cpu else []
            params.boundary_gpu = list(ctx.boundary_gpu) if ctx.use_boundary_saving else []
            params.u_last_two = last.contiguous()
            params.forward_source = ctx.forward_source.contiguous()
            params.forward_sources_loc = forward_sources_loc.contiguous()
            gradients = ctx.backward_bs_func(params)

        del ctx.backward_func, ctx.backward_bs_func
        del ctx.pml_vals, ctx.forward_source
        del ctx.models
        # print(gradients[0].shape)
        return (
            None, None, None, # functions
            None,      # wavelet
            None,      # sources_loc
            None,      # receivers_loc
            None,      # source_field_indices
            None,      # receiver_field_indices
            None,      # fd_coeff
            None,      # M
            None,      # abcn
            None,      # spacing
            None,      # dt
            None,      # pml_vals
            None,      # use_boundary_saving
            None,      # use_pinned_memory
            None,      # free_surface
            None,      # transfer_interval
            None,      # boundary_on_cpu
            None,      # forward wavefields
            None,      # adjoint wavefields
            None,      # last_two
            None,      # boundary cpu
            None,      # boundary gpu
            *gradients[-1] # models
        )

class PropCUDA(PropBase, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)
        
        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._dh, device=self.dev, dtype=torch.float32))

        self.forward_func, self.backward_func, self.backward_bs_func = self.equation._C()

        # Initilize memory for wavefields
        self.forward_allocator = Allocator(self.dev)
        self.boundary_cpu_allocator = Allocator('cpu')
        self.boundary_gpu_allocator = Allocator(self.dev)

        total_wavefields = self.equation.base_nvar + self.equation.pml_nvar
        self.forward_wavefields = self.forward_allocator.zeros(total_wavefields * [[self.B, 1, *self.shape_cuda]])
        self.adjoint_wavefields = self.forward_allocator.zeros(total_wavefields * [[self.B, 1, *self.shape_cuda]])
        self.boundary_cpu = ()
        self.boundary_gpu = ()
        self.boundary_gpu_full = ()
        self.last_two = torch.empty(0, device=self.dev)
        self._boundary_cache_mode = None
        self._boundary_cache_interval = None
        self._boundary_cache_pinned = None
        self._boundary_cache_nt = None

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), f'Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}'
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    def _default_field_types(self, kinds, is_source):
        if kinds:
            return kinds

        if self.equation.__class__.__name__ == 'Elastic':
            if self.ndim == 2:
                return ['sxx', 'szz'] if is_source else ['vx', 'vz']
            return ['sxx', 'syy', 'szz'] if is_source else ['vx', 'vy', 'vz']

        return [self.wavefield_names[0]]

    def _field_indices_tensor(self, kinds, is_source):
        resolved = self._default_field_types(kinds, is_source)
        missing = [name for name in resolved if name not in self.wavefield_names]
        if missing:
            role = 'source_type' if is_source else 'receiver_type'
            raise ValueError(f'Invalid {role} entries {missing}; available wavefields are {self.wavefield_names}')
        indices = [self.wavefield_names.index(name) for name in resolved]
        return torch.tensor(indices, dtype=torch.int32, device=self.dev)

    def _ensure_boundary_buffers(self, boundary_on_cpu, transfer_interval, use_pinned_memory):
        if (
            self._boundary_cache_mode == boundary_on_cpu
            and self._boundary_cache_interval == transfer_interval
            and self._boundary_cache_pinned == use_pinned_memory
            and self._boundary_cache_nt == self.nt
        ):
            return

        layout = Layout(
            self.shape_cuda,
            self.equation.base_nvar,
            self.nt,
            self.abcn,
            self.equation.so // 2,
            self.B,
            transfer_interval,
            self.free_surface,
            self.equation.so // 2 + 1,
        )

        self.boundary_cpu_allocator = Allocator('cpu')
        self.boundary_gpu_allocator = Allocator(self.dev)
        last_two_storage_nvar = getattr(self.equation, "last_two_storage_nvar", self.equation.base_nvar)
        last_two_shape = [last_two_storage_nvar, self.equation.last_two_nvar, self.B, 1, *self.shape_cuda]

        if boundary_on_cpu:
            self.boundary_cpu = self.boundary_cpu_allocator.zeros(
                layout.cpu_shapes,
                dtype=torch.float32,
                dev='cpu',
                pin_memory=use_pinned_memory,
            )
            self.boundary_gpu = self.boundary_gpu_allocator.zeros(layout.gpu_shapes)
            self.boundary_gpu_full = ()
            self.last_two = self.boundary_cpu_allocator.zeros(
                [last_two_shape],
                dtype=torch.float32,
                dev='cpu',
                pin_memory=use_pinned_memory,
            )[0]
        else:
            self.boundary_cpu = ()
            self.boundary_gpu = ()
            self.boundary_gpu_full = self.boundary_gpu_allocator.zeros(layout.gpu_full_shapes)
            self.last_two = self.forward_allocator.zeros([last_two_shape])[0]

        self._boundary_cache_mode = boundary_on_cpu
        self._boundary_cache_interval = transfer_interval
        self._boundary_cache_pinned = use_pinned_memory
        self._boundary_cache_nt = self.nt

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False, use_boundary_saving=None, boundary_saving_config=None, **kwargs):
        """Forward pass of the wave equation

        Args:
            wavelet (np.array): Wavelet tensor (nt,)
            sources (np.array): Source coordinates (nshots, 2)
            receivers (np.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """

        legacy_override = {}
        if "transfer_interval" in kwargs:
            legacy_override["transfer_interval"] = kwargs.pop("transfer_interval")
        if "boundary_on_cpu" in kwargs:
            legacy_override["storage"] = "cpu" if kwargs.pop("boundary_on_cpu") else "gpu"
        if "use_pinned_memory" in kwargs:
            legacy_override["pinned_memory"] = kwargs.pop("use_pinned_memory")
        if boundary_saving_config is None and legacy_override:
            boundary_saving_config = legacy_override
        elif legacy_override:
            boundary_saving_config = {**legacy_override, **boundary_saving_config}

        boundary_cfg = self.resolve_boundary_saving_config(
            override=boundary_saving_config,
            use_boundary_saving=use_boundary_saving,
        )
        use_boundary_saving = boundary_cfg["enabled"]
        boundary_on_cpu = boundary_cfg["storage"] == "cpu"
        transfer_interval = boundary_cfg["transfer_interval"]
        use_pinned_memory = boundary_cfg["pinned_memory"]

        self.nt = wavelet.shape[-1]

        # Set zeros
        self.forward_allocator.zero_()
        if boundary_on_cpu:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_on_cpu, transfer_interval, use_pinned_memory)
            self.boundary_cpu_allocator.zero_()
            self.boundary_gpu_allocator.zero_()
        else:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_on_cpu, transfer_interval, use_pinned_memory)
                for t in self.boundary_gpu_full:
                    t.zero_()

        M = self.equation.so // 2

        pml_padding = M
        padding = [p+M for p in self.padding]
        base_shift = M + self.abcn

        shape_for_pml = [p+2*M for p in self.shape]

        kwargs['shape'] = shape_for_pml
        self.init_abc(**kwargs)

        nt = self.nt
        nshots = sources.shape[0]
        
        batch_size = 1 if source_encoding else nshots
        sources = sources.copy()
        receivers = receivers.copy()

        if self.free_surface:
            sources[..., 0] += base_shift
            receivers[..., 0] += base_shift

            if self.ndim == 3:
                sources[..., 1] += base_shift
                receivers[..., 1] += base_shift

            # For cuda implementation, we pad in z-direction for free surface with width M
            sources[..., -1] += M
            receivers[..., -1] += M
        else:
            sources += base_shift
            receivers += base_shift

        # Batch the wavelet, sources and receivers
        wavelet = torch.from_numpy(wavelet).to(self.dev).float()[None, None, :].repeat(batch_size, 1, 1)  # (B, 1, nt)
        sources = torch.from_numpy(sources).to(self.dev).int()[:, None, :]
        receivers = torch.from_numpy(receivers).to(self.dev).int()
        source_field_indices = self._field_indices_tensor(self.source_type, is_source=True)
        receiver_field_indices = self._field_indices_tensor(self.receiver_type, is_source=False)
        # Get the model parameters

        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, padding) for para in models]
        self.models_padded = models
        # self.equation.b = pad_pml_vals(self.equation.b, pml_padding)
        
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ay', 'by', 'ayh', 'byh', 'ax', 'bx', 'axh', 'bxh']):
        # for b, name in zip(self.equation.b, ['az', 'bz', 'dbzdz', 'ax', 'bx', 'dbxdx']):
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ax', 'bx', 'axh', 'bxh']):
        #     np.save(f'{name}.npy', b.detach().cpu().numpy())

        lap_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        grad_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        lap_coes[1:] = torch.from_numpy(fd_coefficients(2, 2*M)).to(self.dev).float()
        lap_coes[0] = torch.sum(lap_coes[1:]) * 2
        grad_coes[1:] = torch.from_numpy(fd_coefficients(1, 2*M)).to(self.dev).float()
        grad_coes[0] = 0.

        models = [m[None, None, ...].repeat(batch_size, *([1]*(m.ndim+1))) for m in self.models_padded]

        spacing = [self.dh] * self.ndim

        syn = Warpper.apply(
                self.forward_func, 
                self.backward_func, 
                self.backward_bs_func,
                wavelet,
                sources,
                receivers,
                source_field_indices,
                receiver_field_indices,
                (lap_coes, grad_coes),
                M,
                self.abcn,
                spacing,
                self.dt,
                self.equation.b,
                use_boundary_saving,
                use_pinned_memory,
                self.free_surface,
                transfer_interval,
                boundary_on_cpu,
                self.forward_wavefields,
                self.adjoint_wavefields,
                self.last_two,
                self.boundary_cpu,
                self.boundary_gpu if boundary_on_cpu else self.boundary_gpu_full,
                *models,
            )
        
        return syn

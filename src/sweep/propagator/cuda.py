import torch
import torch.nn.functional as F

import numpy as np
import sweep._C as _C
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
        free_surface: bool=False,
        transfer_interval: int=1,
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
        params.free_surface = free_surface
        params.nt = nt
        params.dt = dt
        params.spacing = spacing

        # -------- CUDA forward --------
        (u_allt, boundary_vals, last, syn) = forward_func(params)

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
            # ctx.boundary_vals = [b.detach().cpu() for b in boundary_vals]
            ctx.boundary_vals = boundary_vals
            ctx.pml_vals = pml_vals
            ctx.abcn = abcn
            ctx.M = M
            ctx.nt = nt
            ctx.spacing = spacing
            ctx.dt = dt
            ctx.free_surface = free_surface
            ctx.use_boundary_saving = use_boundary_saving
            ctx.backward_func = backward_func
            ctx.backward_bs_func = backward_bs_func
            ctx.forward_source = wavelet

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

        if not ctx.use_boundary_saving:
            params.u_forward = u_allt.contiguous()
            gradients = ctx.backward_func(params)
        else:
            params.u_boundary = [b.contiguous() for b in ctx.boundary_vals]
            params.u_last_two = last.contiguous()
            params.forward_source = ctx.forward_source.contiguous()
            params.forward_sources_loc = forward_sources_loc.contiguous()
            gradients = ctx.backward_bs_func(params)

        del ctx.backward_func, ctx.backward_bs_func
        del ctx.boundary_vals, ctx.pml_vals, ctx.forward_source
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
            None,      # free_surface
            None,      # transfer_interval
            *gradients[-1] # models
        )

def pad_along_non_singleton_dim(tensor, pad_width):
    """
    Pad tensor along the dimension whose size != 1.
    Only one such dimension is assumed.
    """
    shape = tensor.shape
    ndim = tensor.dim()

    non_singleton_dims = [i for i, s in enumerate(shape) if s != 1]

    if len(non_singleton_dims) != 1:
        raise ValueError(
            f"Expect exactly one non-singleton dimension, got {shape}"
        )

    dim = non_singleton_dims[0]

    pad = []
    for d in reversed(range(ndim)):
        if d == dim:
            pad.extend([pad_width, pad_width])
        else:
            pad.extend([0, 0])

    return F.pad(tensor, pad)


def pad_pml_vals(pml_vals, pad_width):
    return [
        pad_along_non_singleton_dim(t, pad_width)
        for t in pml_vals
    ]

class PropCUDA(PropBase, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._dh, device=self.dev, dtype=torch.float32))

        self.forward_func, self.backward_func, self.backward_bs_func = self.equation._C()

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

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False, use_boundary_saving=False, transfer_interval=1, **kwargs):
        """Forward pass of the wave equation

        Args:
            wavelet (np.array): Wavelet tensor (nt,)
            sources (np.array): Source coordinates (nshots, 2)
            receivers (np.array): Receiver coordinates (nshots, nreceivers, 2)
            models (list): List of model parameters (Must be torch.Tensor)
        """
        M = self.equation.so // 2

        pml_padding = M
        padding = [p+M for p in self.padding]
        base_shift = M + self.abcn

        shape_for_pml = [p+2*M for p in self.shape]

        kwargs['shape'] = shape_for_pml
        self.init_abc(**kwargs)

        nt = wavelet.shape[-1]
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
                self.free_surface,
                transfer_interval,
                *models,
            )
        
        return syn

import torch
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
        coes_list,          
        M: int,
        abcn: int,
        spacing: list,       # list of 2 floats for dx and dz
        dt: float,
        pml_vals: list,     # list of 6 tensors for PML profiles
        use_boundary_saving: bool=False,
        free_surface: bool=False,
        *models             # list of (B, nz, nx) tensors
    ):
        """
        Forward modeling: vp -> synthetic seismograms
        """

        lap_coes, grad_coes = coes_list
        nt = wavelet.shape[-1]

        # only forward modeling, no need to save wavefield for backward
        save_all_wavefield = False
        if any(m.requires_grad for m in models): 
            save_all_wavefield = True
        if any(m.requires_grad for m in models) and use_boundary_saving:
            save_all_wavefield = False

        # -------- CUDA forward --------
        (u_allt, boundary_vals, last, syn) = forward_func(
            [m.contiguous() for m in models],
            wavelet.contiguous(),
            lap_coes.contiguous(),  # u0, not used --- IGNORE ---
            grad_coes.contiguous(),
            M,
            abcn,
            sources_loc.contiguous(),
            receivers_loc.contiguous(),
            [p.contiguous() for p in pml_vals],
            save_all_wavefield,
            use_boundary_saving,
            free_surface,
            nt,
            dt,
            spacing
        )

        ctx.save_for_backward(
            u_allt,
            last,
            sources_loc,
            receivers_loc,
            lap_coes, grad_coes,
        )
        # np.save('u_last.npy', last.detach().cpu().numpy())
        ctx.models = models
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
        ctx.wavelet = wavelet

        return syn        
    
    @staticmethod
    def backward(ctx, adjoint_source):
        
        # -------- unpack --------
        (
            u_allt,
            last,
            forward_sources_loc,
            adjoint_sources_loc,
            lap_coes, grad_coes,
        ) = ctx.saved_tensors

        abcn = ctx.abcn
        M  = ctx.M
        nt = ctx.nt
        dt = ctx.dt

        # -------- CUDA adjoint --------

        if not ctx.use_boundary_saving:
            gradients = ctx.backward_func(
                u_allt.contiguous(),
                [m.contiguous() for m in ctx.models],
                adjoint_source.contiguous(),
                lap_coes.contiguous(), 
                grad_coes.contiguous(),
                M,
                abcn,
                adjoint_sources_loc.contiguous(),
                [p.contiguous() for p in ctx.pml_vals],
                nt,
                dt,
                ctx.spacing
            )
        else:
            gradients = ctx.backward_bs_func(
                [b.contiguous() for b in ctx.boundary_vals],
                last.contiguous(),
                [m.contiguous() for m in ctx.models],
                adjoint_source.contiguous(),
                ctx.wavelet.contiguous(),
                lap_coes.contiguous(),
                grad_coes.contiguous(),
                M,
                abcn,
                adjoint_sources_loc.contiguous(),
                forward_sources_loc.contiguous(),
                [p.contiguous() for p in ctx.pml_vals],
                nt,
                dt,
                ctx.spacing,
                ctx.free_surface
            )
            np.save('/data/tmp/u_forward.npy', gradients[0].detach().cpu().numpy())
            np.save('/data/tmp/u_adoint.npy', gradients[1].detach().cpu().numpy())
            gradients = gradients[2:]
            # print([g.shape for g in gradients])
        return (
            None, None, None, # functions
            None,      # wavelet
            None,      # sources_loc
            None,      # receivers_loc
            None,      # fd_coeff
            None,      # M
            None,      # abcn
            None,      # spacing
            None,      # dt
            None,      # pml_vals
            None,      # use_boundary_saving
            None,      # free_surface
            *gradients # models
        )

import torch
import torch.nn.functional as F

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

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False, use_boundary_saving=False,**kwargs):
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
        # Get the model parameters

        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, padding) for para in models]
        self.models_padded = models
        self.equation.b = pad_pml_vals(self.equation.b, pml_padding)
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ax', 'bx', 'axh', 'bxh']):
        #     np.save(f'{name}.npy', b.detach().cpu().numpy())

        lap_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        grad_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        lap_coes[1:] = torch.from_numpy(fd_coefficients(2, 2*M)).to(self.dev).float()
        lap_coes[0] = torch.sum(lap_coes[1:]) * 2
        grad_coes[1:] = torch.from_numpy(fd_coefficients(1, 2*M)).to(self.dev).float()
        grad_coes[0] = 0.

        models = [m[None, None, ...].repeat(batch_size, *([1]*(m.ndim+1))) for m in self.models_padded]

        spacing = [self.dh.item()] * self.ndim
        
        syn = Warpper.apply(
                self.forward_func, 
                self.backward_func, 
                self.backward_bs_func,
                wavelet,
                sources,
                receivers,
                (lap_coes, grad_coes),
                M,
                self.abcn,
                spacing,
                self.dt.item(),
                self.equation.b,
                use_boundary_saving,
                self.free_surface,
                *models,
            )
        
        return syn
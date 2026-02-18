import torch
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
        vp,                 # (B, nz, nx)
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
    ):
        """
        Forward modeling: vp -> synthetic seismograms
        """

        lap_coes, grad_coes = coes_list
        nt = wavelet.shape[-1]

        # only forward modeling, no need to save wavefield for backward
        save_all_wavefield = False
        if vp.requires_grad: 
            save_all_wavefield = True
        if vp.requires_grad and use_boundary_saving:
            save_all_wavefield = False

        # -------- CUDA forward --------
        (u_allt, boundary_vals, last, syn) = forward_func(
            vp.contiguous(),
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
            nt,
            dt,
            spacing
        )

        ctx.save_for_backward(
            vp,
            u_allt,
            last,
            receivers_loc,
            lap_coes, grad_coes,
        )

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

        return syn        
    
    @staticmethod
    def backward(ctx, adjoint_source):
        
        # -------- unpack --------
        (
            vp,
            u_allt,
            last,
            receivers_loc,
            lap_coes, grad_coes,
        ) = ctx.saved_tensors

        abcn = ctx.abcn
        M  = ctx.M
        nt = ctx.nt
        dt = ctx.dt

        # -------- CUDA adjoint --------

        if not ctx.use_boundary_saving:
            (grad_vp, ) = ctx.backward_func(
                u_allt.contiguous(),
                vp.contiguous(),
                adjoint_source.contiguous(),
                lap_coes.contiguous(), 
                grad_coes.contiguous(),
                M,
                abcn,
                receivers_loc,
                [p.contiguous() for p in ctx.pml_vals],
                nt,
                dt,
                ctx.spacing
            )
        else:
            (grad_vp, ) = ctx.backward_bs_func(
                [b.contiguous() for b in ctx.boundary_vals],
                last.contiguous(),
                vp.contiguous(),
                adjoint_source.contiguous(),
                lap_coes.contiguous(),
                grad_coes.contiguous(),
                M,
                abcn,
                receivers_loc.contiguous(),
                [p.contiguous() for p in ctx.pml_vals],
                nt,
                dt,
                ctx.spacing,
                ctx.free_surface
            )

        return (
            None, None, None,
            grad_vp,   # vp
            None,      # source
            None,      # sources_loc
            None,      # receivers_loc
            None,      # fd_coeff
            None,      # M
            None,      # abcn
            None,      # nt
            None, None, None,  # dx, dz, dt
            None,      # pml_vals
            None,      # use_boundary_saving
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

        padding = [p+M for p in self.padding]

        self.init_abc(**kwargs)

        nt = wavelet.shape[-1]
        nshots = sources.shape[0]

        batch_size = 1 if source_encoding else nshots
        sources = sources.copy()
        receivers = receivers.copy()

        if self.free_surface:
            sources[..., 0] += (M+self.abcn)
            receivers[..., 0] += (M+self.abcn)

            if self.ndim == 3:
                sources[..., 1] += (M+self.abcn)
                receivers[..., 1] += (M+self.abcn)

            # For cuda implementation, we pad in z-direction for free surface with width M
            sources[..., -1] += M
            receivers[..., -1] += M
        else:
            sources += (M+self.abcn)
            receivers += (M+self.abcn)

        # Batch the wavelet, sources and receivers
        wavelet = torch.from_numpy(wavelet).to(self.dev).float()[None, None, :].repeat(batch_size, 1, 1)  # (B, 1, nt)
        sources = torch.from_numpy(sources).to(self.dev).int()[:, None, :]
        receivers = torch.from_numpy(receivers).to(self.dev).int()
        # Get the model parameters

        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, padding) for para in models]
        self.models_padded = models
        self.equation.b = pad_pml_vals(self.equation.b, M)

        lap_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        grad_coes = torch.zeros(M+1, dtype=torch.float32, device=self.dev)
        lap_coes[1:] = torch.from_numpy(fd_coefficients(2, 2*M)).to(self.dev).float()
        lap_coes[0] = torch.sum(lap_coes[1:]) * 2
        grad_coes[1:] = torch.from_numpy(fd_coefficients(1, 2*M)).to(self.dev).float()
        grad_coes[0] = 0.

        vp = self.models_padded[0][None, None, ...]
        vp = vp.repeat(batch_size, *([1]*(vp.ndim-1)))  # vp

        spacing = [self.dh.item()] * self.ndim

        syn = Warpper.apply(
                self.forward_func, 
                self.backward_func, 
                self.backward_bs_func,
                vp,  # vp
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
            )
        
        return syn
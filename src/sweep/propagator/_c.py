import os
import shutil
import tempfile

import torch
import torch.nn.functional as F

import numpy as np
from sweep.memory.torch import Allocator
from sweep.memory.shape import Layout
from sweep.propagator.base import PropBase
from sweep.equations._edges import is_top_only_or_none
from sweep.utils.torch import EdgePadding
from sweep.scalars import fd_coefficients, staggered_grid_coes
from sweep.equations.base import FirstOrderEquation


def _get_C():
    import torch  # Ensure PyTorch is loaded before importing the compiled extension.
    import sweep._C as _C

    return _C


# ---------------------------------------------------------------------------
# Record layout helpers
# ---------------------------------------------------------------------------
# The CUDA kernels write the receiver record with time innermost so each
# timestep writes contiguous receiver values. This gives two raw layouts:
#
#   * single-channel solvers : ``syn.shape == (B, nrec, nt)``
#   * multi-channel solvers  : ``syn.shape == (nfield, B, nrec, nt)``
#
# Both differ from the canonical layout shared with the eager backend and
# with ``sweep_loss``:
#
#   * canonical             : ``(B, nt, nrec, nfield)``
#
# To keep the user-facing API consistent across backends we permute the
# CUDA output to canonical inside the Warpper / RTM wrappers, and permute
# the canonical-shaped autograd / RTM gradient back to the raw CUDA layout
# before handing it to the C++ adjoint-source kernels.

def _cuda_record_to_canonical(syn: torch.Tensor) -> torch.Tensor:
    """Permute a CUDA record tensor to ``(B, nt, nrec, nfield)``.

    Accepts ``(B, nrec, nt)`` (single-channel) or
    ``(nfield, B, nrec, nt)`` (multi-channel).
    """
    if syn.ndim == 3:
        return syn.permute(0, 2, 1).unsqueeze(-1).contiguous()
    if syn.ndim == 4:
        return syn.permute(1, 3, 2, 0).contiguous()
    raise ValueError(
        f"Unexpected CUDA record ndim={syn.ndim}, shape={tuple(syn.shape)}; "
        "expected (B, nrec, nt) or (nfield, B, nrec, nt)."
    )


def _canonical_to_cuda_record(grad: torch.Tensor, cuda_ndim: int) -> torch.Tensor:
    """Inverse of :func:`_cuda_record_to_canonical`.

    Given canonical ``(B, nt, nrec, nfield)`` (or a 3-D fallback variant),
    return the raw CUDA layout. ``cuda_ndim`` is the ndim of the original
    CUDA output (3 = single-channel, 4 = multi-channel).
    """
    if cuda_ndim == 3:
        if grad.ndim == 4:
            if grad.shape[-1] != 1:
                raise ValueError(
                    "Single-channel CUDA backward expects canonical "
                    f"grad with last dim = 1, got shape {tuple(grad.shape)}."
                )
            grad = grad.squeeze(-1)
        if grad.ndim != 3:
            raise ValueError(
                f"Single-channel CUDA backward needs 3-D grad, got {tuple(grad.shape)}."
            )
        # (B, nt, nrec) -> (B, nrec, nt)
        return grad.permute(0, 2, 1).contiguous()
    if cuda_ndim == 4:
        if grad.ndim != 4:
            raise ValueError(
                f"Multi-channel CUDA backward needs 4-D grad, got {tuple(grad.shape)}."
            )
        # (B, nt, nrec, nfield) -> (nfield, B, nrec, nt)
        return grad.permute(3, 0, 2, 1).contiguous()
    raise ValueError(f"Unexpected cuda_ndim={cuda_ndim}; expected 3 or 4.")


class Warpper(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        forward_func,
        backward_func,
        backward_bs_func,
        backward_ckpt_func,
        backward_recursive_ckpt_func,
        wavelet,             # (B, nsrc, nt)
        sources_loc,        # (B, nsrc, 2)
        receivers_loc,      # (B, nrec, 2)
        source_field_indices,
        receiver_field_indices,
        coes_list,          
        M: int,
        abcn: int,
        spacing: list,       # list of floats for grid spacing
        dt: float,
        pml_vals: list,     # list of 6 tensors for PML profiles
        use_checkpoint: bool=False,
        checkpoint_interval: int=1,
        use_recursive_checkpoint: bool=False,
        checkpoint_count: int=0,
        checkpoint_steps: torch.Tensor=None,
        checkpoint_on_cpu: bool=False,
        use_boundary_saving: bool=False,
        use_pinned_memory: bool=False,
        free_surface: bool=False,
        transfer_interval: int=1,
        boundary_ring_buffers: int=1,
        boundary_on_cpu: bool=False,
        boundary_on_disk: bool=False,
        boundary_disk_async_read: bool=False,
        forward_wavefields: tuple=(),
        adjoint_wavefields: tuple=(),
        adjoint_workspace: tuple=(),
        checkpoint_buffers: tuple=(),
        last_two: torch.Tensor=None,
        boundary_cpu: tuple=(),
        boundary_gpu: tuple=(),
        boundary_disk_files: tuple=(),
        source_illumination_buffer: torch.Tensor=None,
        receiver_illumination_buffer: torch.Tensor=None,
        illumination_padding: tuple=(),
        adcig_buffer: torch.Tensor=None,      # (nlag, nz, nx[, ny]) model-shaped
        adcig_max_lag: int=0,
        topo_rows_param: torch.Tensor=None,   # runtime padded surface row per col
        has_topo_param: bool=False,
        topo_category_param: torch.Tensor=None,  # runtime padded APM category int32
        use_apm_param: bool=False,
        fs_faces: int=-1,   # per-edge free-surface bitmask (-1 => legacy z-min)
        cut_face_mask: int=0,  # DD cut faces (0 => single domain)
        *models             # list of (B, nz, nx) tensors
    ):
        """
        Forward modeling: vp -> synthetic seismograms
        """
        
        lap_coes, grad_coes = coes_list
        nt = wavelet.shape[-1]
        spacing = [float(s) for s in spacing]
        dt = float(dt)
        
        # only forward modeling, no need to save wavefield for backward
        requires_model_grad = any(m.requires_grad for m in models)
        requires_wavelet_grad = wavelet.requires_grad
        requires_backward = bool(requires_model_grad or requires_wavelet_grad)
        use_recursive_checkpoint = bool(use_recursive_checkpoint and backward_recursive_ckpt_func is not None)
        use_checkpoint = bool(use_checkpoint and (backward_ckpt_func is not None or use_recursive_checkpoint))
        save_all_wavefields = requires_backward
        if requires_backward and (use_boundary_saving or use_checkpoint):
            save_all_wavefields = False
        if not requires_backward:
            save_all_wavefields = False
            use_boundary_saving = False
            use_checkpoint = False
            use_recursive_checkpoint = False

        _C = _get_C()
        params = _C.ForwardInput()
        params.wavefields = forward_wavefields
        params.last_two = last_two
        if boundary_on_disk:
            params.boundary_cpu = [b.zero_() for b in boundary_cpu]
            params.boundary_gpu = [b.zero_() for b in boundary_gpu] if use_boundary_saving else []
        elif boundary_on_cpu:
            params.boundary_cpu = list(boundary_cpu)
            params.boundary_gpu = list(boundary_gpu) if use_boundary_saving else []
        else:
            params.boundary_cpu = []
            params.boundary_gpu = [b.zero_() for b in boundary_gpu] if use_boundary_saving else []
        params.boundary_disk_files = list(boundary_disk_files) if boundary_on_disk else []
        params.checkpoints = [c.zero_() for c in checkpoint_buffers] if use_checkpoint else []
        params.transfer_interval = transfer_interval
        params.boundary_ring_buffers = boundary_ring_buffers
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
        params.use_checkpoint = use_checkpoint
        params.use_recursive_checkpoint = use_recursive_checkpoint
        params.checkpoint_on_cpu = checkpoint_on_cpu
        params.boundary_on_cpu = boundary_on_cpu
        params.boundary_on_disk = boundary_on_disk
        params.boundary_disk_async_read = boundary_disk_async_read
        params.use_pinned_memory = use_pinned_memory
        params.free_surface = free_surface
        params.fs_faces = fs_faces
        # DD cut faces MUST be told to C too, not just to the Python Layout:
        # ``Layout(cut_mask=...)`` drops a cut face's boundary buffer to numel 0
        # (gpu_full_shapes), and only ``ctx.cut_*`` stops boundary_kernel2d from
        # writing there. Setting one without the other writes through a null
        # data_ptr. Both sides read the same ``PropBase._dd_cut_mask``.
        params.cut_face_mask = cut_face_mask
        # Topography plumbing (image method) — empty tensor + has_topo=False
        # for flat. topo_rows_param is passed in via the autograd Function
        # call site (see Warpper.apply below).
        if has_topo_param:
            params.topo_rows = topo_rows_param.to(torch.int32).contiguous()
            params.has_topo = True
        else:
            params.topo_rows = torch.empty(0, dtype=torch.int32,
                                            device=wavelet.device)
            params.has_topo = False
        # APM (Cao & Chen 2018) plumbing.  ``params.models`` is already
        # extended with the precomputed effective moduli by the caller
        # (positions 6..10); we just attach the category and flag here.
        if use_apm_param:
            params.topo_category = topo_category_param.to(torch.int32).contiguous()
            params.use_apm = True
        else:
            params.topo_category = torch.empty(0, dtype=torch.int32,
                                                device=wavelet.device)
            params.use_apm = False
        params.nt = nt
        params.dt = dt
        params.spacing = spacing
        params.checkpoint_interval = checkpoint_interval
        params.checkpoint_count = checkpoint_count
        params.checkpoint_steps = checkpoint_steps if checkpoint_steps is not None else torch.empty(0, dtype=torch.int32)

        # -------- CUDA forward --------
        (u_allt, last, syn) = forward_func(params)

        # Permute to canonical (B, nt, nrec, nfield) for the user-facing
        # output; remember the raw CUDA ndim so backward can invert it.
        cuda_record_ndim = int(syn.ndim)
        ctx.cuda_record_ndim = cuda_record_ndim
        syn = _cuda_record_to_canonical(syn)
        if any([save_all_wavefields, use_boundary_saving, use_checkpoint]):
            
            ctx.save_for_backward(
                u_allt,
                last,
                sources_loc,
                receivers_loc,
                source_field_indices,
                receiver_field_indices,
                lap_coes, grad_coes,
                checkpoint_steps if checkpoint_steps is not None else torch.empty(0, dtype=torch.int32),
                *checkpoint_buffers,
            )
            ctx.transfer_interval = transfer_interval
            ctx.boundary_ring_buffers = boundary_ring_buffers
            ctx.checkpoint_interval = checkpoint_interval
            ctx.checkpoint_count = checkpoint_count
            ctx.models = models
            ctx.boundary_on_cpu = boundary_on_cpu
            ctx.boundary_on_disk = boundary_on_disk
            ctx.boundary_disk_async_read = boundary_disk_async_read
            ctx.boundary_cpu = boundary_cpu if boundary_on_cpu else ()
            ctx.boundary_gpu = boundary_gpu if use_boundary_saving else ()
            ctx.boundary_disk_files = tuple(boundary_disk_files) if boundary_on_disk else ()
            ctx.pml_vals = pml_vals
            ctx.abcn = abcn
            ctx.M = M
            ctx.nt = nt
            ctx.spacing = spacing
            ctx.dt = dt
            ctx.free_surface = free_surface
            ctx.fs_faces = fs_faces
            ctx.cut_face_mask = cut_face_mask
            # Topography (image method): preserve runtime row-index tensor so
            # the autograd backward can plumb it without referencing ``self``.
            ctx.topo_rows_param = topo_rows_param
            ctx.has_topo_param  = has_topo_param
            ctx.topo_category_param = topo_category_param
            ctx.use_apm_param   = use_apm_param
            ctx.use_boundary_saving = use_boundary_saving
            ctx.use_checkpoint = use_checkpoint
            ctx.use_recursive_checkpoint = use_recursive_checkpoint
            ctx.checkpoint_on_cpu = checkpoint_on_cpu
            ctx.use_pinned_memory = use_pinned_memory
            ctx.backward_func = backward_func
            ctx.backward_bs_func = backward_bs_func
            ctx.backward_ckpt_func = backward_ckpt_func
            ctx.backward_recursive_ckpt_func = backward_recursive_ckpt_func
            ctx.forward_source = wavelet
            # save_all binds the propagator's persistent buffers; the other
            # modes (BS/ckpt) pass per-call transient scratch that must NOT
            # outlive the forward — and their backwards expect an empty list
            # here (the ckpt recompute allocates its own legacy 7/9-slot
            # state paired with the u-only swap()).
            ctx.forward_wavefields = forward_wavefields if save_all_wavefields else ()
            ctx.adjoint_wavefields = adjoint_wavefields
            ctx.adjoint_workspace = adjoint_workspace
            ctx.source_illumination_buffer = source_illumination_buffer
            ctx.receiver_illumination_buffer = receiver_illumination_buffer
            ctx.illumination_padding = tuple(illumination_padding)
            ctx.adcig_buffer = adcig_buffer
            ctx.adcig_max_lag = int(adcig_max_lag)

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
            checkpoint_steps,
            *checkpoint_tensors,
        ) = ctx.saved_tensors

        abcn = ctx.abcn
        M  = ctx.M
        nt = ctx.nt
        dt = ctx.dt
        fs_faces = getattr(ctx, "fs_faces", -1)
        cut_face_mask = getattr(ctx, "cut_face_mask", 0)

        _C = _get_C()
        params = _C.BackwardInput()
        # common
        params.transfer_interval = ctx.transfer_interval
        params.boundary_ring_buffers = ctx.boundary_ring_buffers
        params.checkpoint_interval = ctx.checkpoint_interval
        params.checkpoint_count = ctx.checkpoint_count
        # Compute source/receiver illumination only if the caller requested it
        # (solver.compute_illumination=True allocates a real, non-empty buffer in
        # forward).  It is a ~1/3-of-backward extra grid pass; vp grad unaffected.
        def _wants_illum(b):
            return isinstance(b, torch.Tensor) and b.numel() > 0
        params.compute_illumination = (
            _wants_illum(getattr(ctx, "source_illumination_buffer", None))
            or _wants_illum(getattr(ctx, "receiver_illumination_buffer", None))
        )
        # Space-lag ADCIG: a real, non-empty buffer allocated in forward is the
        # ON signal (mirrors illumination).  ``adcig_max_lag`` sizes the lag axis.
        params.compute_adcig = _wants_illum(getattr(ctx, "adcig_buffer", None))
        params.adcig_max_lag = int(getattr(ctx, "adcig_max_lag", 0))
        if params.compute_adcig and not ctx.use_boundary_saving:
            raise RuntimeError(
                "compute_adcig=True currently requires boundary-saving mode. The "
                "full/checkpoint forward stores vp^2*Lap(u) for the vp gradient, "
                "not the raw pressure the space-lag ADCIG imaging condition needs. "
                "Enable boundary saving via boundary_saving_config={'enabled': True} "
                "or memory=MemoryOptions(strategy='boundary')."
            )
        params.adjoint_wavefields = [a.zero_() for a in ctx.adjoint_wavefields]
        params.adjoint_workspace = list(ctx.adjoint_workspace)
        params.models = [m.contiguous() for m in ctx.models]
        # ``adjoint_source`` arrives in the canonical (B, nt, nrec, nfield)
        # layout that ``forward`` returned; permute it back to the raw CUDA
        # layout the C++ adjoint-source kernels expect.
        cuda_record_ndim = getattr(ctx, "cuda_record_ndim", None)
        if cuda_record_ndim is None:
            params.adjoint_source = adjoint_source.contiguous()
        else:
            params.adjoint_source = _canonical_to_cuda_record(
                adjoint_source, cuda_record_ndim
            )
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
        params.fs_faces = fs_faces
        params.cut_face_mask = cut_face_mask   # see the forward path
        # Topography plumbing (image method) — mirrors forward path.
        # ``ctx`` carries the runtime row tensor saved at forward time;
        # ``self`` doesn't exist here (Warpper.backward is a staticmethod).
        topo_rows_rt = getattr(ctx, "topo_rows_param", None)
        has_topo     = bool(getattr(ctx, "has_topo_param", False))
        if has_topo and topo_rows_rt is not None:
            params.topo_rows = topo_rows_rt.to(torch.int32).contiguous()
            params.has_topo = True
        else:
            params.topo_rows = torch.empty(
                0, dtype=torch.int32, device=adjoint_source.device,
            )
            params.has_topo = False
        # APM CUDA backward: forward saved the category + flag in ctx.
        topo_cat_rt = getattr(ctx, "topo_category_param", None)
        use_apm     = bool(getattr(ctx, "use_apm_param", False))
        if use_apm and topo_cat_rt is not None:
            params.topo_category = topo_cat_rt.to(torch.int32).contiguous()
            params.use_apm = True
        else:
            params.topo_category = torch.empty(0, dtype=torch.int32,
                                                device=adjoint_source.device)
            params.use_apm = False
        params.boundary_on_cpu = ctx.boundary_on_cpu
        params.boundary_on_disk = ctx.boundary_on_disk
        params.boundary_disk_async_read = ctx.boundary_disk_async_read
        params.use_pinned_memory = ctx.use_pinned_memory
        params.checkpoint_on_cpu = ctx.checkpoint_on_cpu

        if ctx.use_checkpoint:
            params.checkpoints = list(checkpoint_tensors)
            params.checkpoint_steps = checkpoint_steps.contiguous()
            params.forward_source = ctx.forward_source.contiguous()
            params.forward_sources_loc = forward_sources_loc.contiguous()
            params.forward_wavefields = [f.zero_() for f in ctx.forward_wavefields]
            if ctx.use_recursive_checkpoint:
                gradients = ctx.backward_recursive_ckpt_func(params)
            else:
                gradients = ctx.backward_ckpt_func(params)
        elif not ctx.use_boundary_saving:
            params.u_forward = u_allt.contiguous()
            params.forward_source = ctx.forward_source.contiguous()
            params.forward_sources_loc = forward_sources_loc.contiguous()
            gradients = ctx.backward_func(params)
        else:
            params.boundary_cpu = list(ctx.boundary_cpu) if ctx.boundary_on_cpu else []
            params.boundary_gpu = list(ctx.boundary_gpu) if ctx.use_boundary_saving else []
            params.boundary_disk_files = list(ctx.boundary_disk_files) if ctx.boundary_on_disk else []
            params.u_last_two = last.contiguous()
            params.forward_source = ctx.forward_source.contiguous()
            params.forward_sources_loc = forward_sources_loc.contiguous()
            gradients = ctx.backward_bs_func(params)

        returned_grads = gradients[1] if len(gradients) >= 2 else gradients[-1]
        if len(gradients) >= 4:
            source_illumination, receiver_illumination = gradients[2], gradients[3]
            source_buffer = getattr(ctx, "source_illumination_buffer", None)
            receiver_buffer = getattr(ctx, "receiver_illumination_buffer", None)

            def fit_illumination_to_model(illumination, target):
                while illumination.dim() > target.dim():
                    illumination = illumination.sum(dim=0)

                slices = [slice(None)] * illumination.dim()
                pad = getattr(ctx, "illumination_padding", ())
                pad_pairs = min(len(pad) // 2, illumination.dim(), target.dim())
                for i in range(pad_pairs):
                    left = int(pad[2 * i])
                    right = int(pad[2 * i + 1])
                    dim = -(i + 1)
                    end = -right if right > 0 else None
                    slices[dim] = slice(left, end)

                illumination = illumination[tuple(slices)]
                if illumination.shape != target.shape:
                    trim = []
                    for size, target_size in zip(illumination.shape, target.shape):
                        trim.append(slice(0, target_size))
                    illumination = illumination[tuple(trim)]
                return illumination

            try:
                if (
                    isinstance(source_buffer, torch.Tensor)
                    and isinstance(source_illumination, torch.Tensor)
                ):
                    source_buffer.copy_(fit_illumination_to_model(source_illumination, source_buffer))
            except RuntimeError:
                pass
            try:
                if (
                    isinstance(receiver_buffer, torch.Tensor)
                    and isinstance(receiver_illumination, torch.Tensor)
                ):
                    receiver_buffer.copy_(fit_illumination_to_model(receiver_illumination, receiver_buffer))
            except RuntimeError:
                pass

        # Space-lag ADCIG cube (gradients[4]): (nlag, N, C, nz, nx[, ny]) on the
        # runtime-padded grid.  Sum over the batch (N, C) — keeping the leading
        # lag axis — then crop the PML/halo padding, and copy into the
        # model-shaped buffer the user reads back as ``solver.adcig``.
        adcig_buffer = getattr(ctx, "adcig_buffer", None)
        if len(gradients) >= 5 and _wants_illum(adcig_buffer):
            adcig_returned = gradients[4]
            if isinstance(adcig_returned, torch.Tensor) and adcig_returned.numel() > 0:
                def fit_adcig_to_model(adcig, target):
                    # collapse batch dims (dim 1 == N, then C); keep lag at dim 0
                    while adcig.dim() > target.dim():
                        adcig = adcig.sum(dim=1)
                    slices = [slice(None)] * adcig.dim()
                    pad = getattr(ctx, "illumination_padding", ())
                    # never crop the leading lag axis (dim 0)
                    pad_pairs = min(len(pad) // 2, adcig.dim() - 1)
                    for i in range(pad_pairs):
                        left = int(pad[2 * i])
                        right = int(pad[2 * i + 1])
                        end = -right if right > 0 else None
                        slices[-(i + 1)] = slice(left, end)
                    adcig = adcig[tuple(slices)]
                    if adcig.shape != target.shape:
                        adcig = adcig[tuple(slice(0, s) for s in target.shape)]
                    return adcig
                try:
                    adcig_buffer.copy_(fit_adcig_to_model(adcig_returned, adcig_buffer))
                except RuntimeError:
                    pass

        wavelet_grad = None
        model_grads = returned_grads
        if len(returned_grads) == len(ctx.models) + 1:
            wavelet_grad = returned_grads[0]
            model_grads = returned_grads[1:]

        del ctx.backward_func, ctx.backward_bs_func, ctx.backward_ckpt_func, ctx.backward_recursive_ckpt_func
        del ctx.pml_vals, ctx.forward_source
        del ctx.forward_wavefields
        del ctx.adjoint_workspace
        del ctx.source_illumination_buffer, ctx.receiver_illumination_buffer
        del ctx.illumination_padding
        del ctx.adcig_buffer, ctx.adcig_max_lag
        del ctx.models
        return (
            None, None, None, None, None, # functions
            wavelet_grad,
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
            None,      # use_checkpoint
            None,      # checkpoint_interval
            None,      # use_recursive_checkpoint
            None,      # checkpoint_count
            None,      # checkpoint_steps
            None,      # checkpoint_on_cpu
            None,      # use_boundary_saving
            None,      # use_pinned_memory
            None,      # free_surface
            None,      # transfer_interval
            None,      # boundary_ring_buffers
            None,      # boundary_on_cpu
            None,      # boundary_on_disk
            None,      # boundary_disk_async_read
            None,      # forward wavefields
            None,      # adjoint wavefields
            None,      # adjoint workspace
            None,      # checkpoint buffers
            None,      # last_two
            None,      # boundary cpu
            None,      # boundary gpu
            None,      # boundary disk files
            None,      # source illumination buffer
            None,      # receiver illumination buffer
            None,      # illumination padding
            None,      # adcig buffer
            None,      # adcig_max_lag
            None,      # topo_rows_param
            None,      # has_topo_param
            None,      # topo_category_param
            None,      # use_apm_param
            None,      # fs_faces
            None,      # cut_face_mask
            *model_grads # models
        )

class _CompiledPropagator(PropBase, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        # Pre-set the attributes ``__del__`` may touch so a raise-during-init
        # (e.g. the topography guard below) doesn't trip a second exception.
        self._boundary_disk_root = None
        self._boundary_disk_files = ()
        super().__init__(*args, **kwargs)

        # impl='c' does not support per-edge PML *thickness* (a non-scalar abcn):
        # the C side derives every non-free-surface face's pad from the single
        # scalar abcn, so a per-edge abcn list would be silently wrong.  (A
        # per-edge free_surface with a scalar abcn is fine.)
        if not isinstance(self._abcn_arg, int) or isinstance(self._abcn_arg, bool):
            raise NotImplementedError(
                "per-edge PML thickness (abcn as a list) is not supported on "
                "impl='c' yet; pass a scalar abcn (a per-edge free_surface is "
                "fine) or use impl='eager'."
            )
        # Per-edge free surface on impl='c' is staged separately from eager: it
        # needs the migrated CUDA kernels and (for now) runs on CUDA only.
        if not is_top_only_or_none(self.fs_faces):
            if not getattr(self.equation, "supports_per_edge_free_surface_c", False):
                raise NotImplementedError(
                    f"per-edge free surface on impl='c' is not implemented for "
                    f"{type(self.equation).__name__} yet; use impl='eager'."
                )
            if (getattr(self.equation, "supports_per_edge_free_surface_c_z_only", False)
                    and (self.fs_faces[2] or self.fs_faces[3])):
                raise NotImplementedError(
                    f"per-edge free surface on impl='c' for "
                    f"{type(self.equation).__name__} currently supports only the z "
                    "faces (top/bottom); x faces (left/right) need impl='eager'."
                )
            if 'cuda' not in str(self.dev):
                raise NotImplementedError(
                    "per-edge free surface on impl='c' currently requires CUDA; "
                    "use impl='eager' on CPU."
                )
            # All three CUDA backward memory modes — full, checkpointing, and
            # boundary saving — are gradient-consistent for every per-edge free
            # surface (cos=1.0 vs eager, all four faces + z∩x corners).  The bs
            # reverse reconstruction re-runs the (now per-edge-correct) forward
            # kernels, so no separate handling is needed.

        # Topography is supported on CUDA via two paths:
        #   * topo_method='image' — per-column staircase (vacuum for Acoustic,
        #     Robertsson 1996 image method for Elastic).  Always available.
        #   * topo_method='apm'   — Cao & Chen 2018 parameter-modified path.
        #     CUDA forward only; backward currently falls back to eager.
        # Plumb the appropriate equation._C() function set below.

        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._grid_spacing, device=self.dev, dtype=torch.float32))

        funcs = self.equation._C()
        self.forward_func = funcs[0]
        self.backward_func = funcs[1]
        self.backward_bs_func = funcs[2]
        self.backward_ckpt_func = funcs[3] if len(funcs) > 3 else None
        self.backward_recursive_ckpt_func = funcs[4] if len(funcs) > 4 else None
        self.rtm_func = self.equation._C_rtm() if hasattr(self.equation, "_C_rtm") else None

        # APM CUDA path — only attached when the equation supports it AND the
        # user selected ``topo_method='apm'``.  Forward only: the APM backward
        # kernels are attached so the dispatch stays uniform, but they return a
        # wrong rho gradient at body-force source cells, so
        # ``_guard_apm_backward`` refuses any backward through this path and
        # sends gradients to eager autograd (see that method for the numbers).
        if (getattr(self, "_topo_method", None) == "apm"
                and hasattr(self.equation, "_C_apm")):
            apm_funcs = self.equation._C_apm()
            self.forward_func = apm_funcs[0]
            self.backward_func = apm_funcs[1]
            self.backward_bs_func = apm_funcs[2]
            # APM has no checkpoint backward implementation; disable
            # checkpoint dispatch so backward routes to full / bs only.
            self.backward_ckpt_func = None
            self.backward_recursive_ckpt_func = None
            self.use_ckpt = False

        # Initialize reusable runtime buffers lazily so they always match the
        # current batch size used by the CUDA kernels.
        self.forward_allocator = Allocator(self.dev)
        self.adjoint_allocator = Allocator(self.dev)
        self.boundary_cpu_allocator = Allocator('cpu')
        self.boundary_gpu_allocator = Allocator(self.dev)
        self.forward_wavefields = ()
        self.adjoint_wavefields = ()
        self.workspace_allocator = Allocator(self.dev)
        self.adjoint_workspace = ()
        self.boundary_cpu = ()
        self.boundary_gpu = ()
        self.boundary_gpu_full = ()
        self.checkpoint_allocator = Allocator(self.dev)
        self.checkpoints = ()
        self.last_two = torch.empty(0, device=self.dev)
        self._buffer_capacity_batch = None
        self._boundary_cache_mode = None
        self._boundary_cache_interval = None
        self._boundary_cache_ring_buffers = None
        self._boundary_cache_pinned = None
        self._boundary_cache_disk_dir = None
        self._boundary_cache_nt = None
        self._boundary_cache_batch = None
        self._boundary_disk_root = None
        self._boundary_disk_files = ()
        self._checkpoint_cache_interval = None
        self._checkpoint_cache_count = None
        self._checkpoint_cache_nt = None
        self._checkpoint_cache_batch = None
        self._checkpoint_cache_storage = None
        self._checkpoint_cache_pinned = None
        self._workspace_cache_batch = None
        self._workspace_cache_nt = None
        self.source_illumination = None
        self.receiver_illumination = None
        # Source/receiver illumination (RTM image) is a ~1/3-of-backward extra
        # grid pass.  Default OFF for speed; set ``solver.compute_illumination =
        # True`` to compute it (then read it back from ``solver.source_illumination``
        # / ``solver.receiver_illumination`` after backward).
        self.compute_illumination = False
        # Space-lag ADCIG (angle-domain common-image gathers via the horizontal
        # subsurface-offset extended imaging condition).  Default OFF; set
        # ``solver.compute_adcig = True`` and ``solver.adcig_max_lag = L`` (lag in
        # cells), then read the ``(2L+1, nz, nx[, ny])`` cube from
        # ``solver.adcig`` after backward.  Like illumination it is an extra
        # per-timestep grid pass and only runs when requested.
        self.compute_adcig = False
        self.adcig_max_lag = 0
        self.adcig = None

    def _guard_apm_backward(self, requires_backward):
        """The CUDA APM backward returns a WRONG rho gradient.

        ``topo_method='apm'`` (Cao & Chen 2018 parameter-modified surface) is a
        CUDA **forward** path; its backward was always meant to be taken through
        eager autograd — see the note on the ``_C_apm()`` dispatch above — but
        nothing enforced that, so it quietly handed back a gradient instead.

        What it hands back, finite-difference arbitrated (elastic 2-D, source
        ``['vz']``, d(loss)/d(rho) at the source cell z=8 x=28):

            FD (truth)  +2.606052e-05
            eager       +2.606044e-05   rel 3.2e-06   correct
            impl='c'    +2.243012e-06   rel 9.1e-01   off by 11.6x

        The error is confined to that one cell — masking it drops the whole-field
        rel_l2 from 8.9e-01 to 2.4e-04, and single-cell FD at the four
        neighbours confirms impl='c' is right there — and it needs a body-force
        source: rho cos against eager is 0.54 for ``['vz']``, 0.29 for
        ``['vx']``, but 1.0000000 for the pure-stress loadings.  It does not
        need a hill (flat-zero APM topography reproduces it), it fires in all
        four backward modes (full, boundary saving, both checkpoint flavours),
        and it reproduces bit-identically on dev, so it is not a regression.
        The signature is a missing body-force rho injection correction in the
        APM rho backward — the non-APM path got exactly that fix in 46172fd.

        vp and vs happen to agree with eager to ~1e-3 here, but that path has
        never been validated against the truth either, so the whole APM backward
        is gated rather than just the rho slot.
        """
        if not requires_backward or getattr(self, "_topo_method", None) != "apm":
            return
        raise NotImplementedError(
            "topo_method='apm' has no gradient on impl='c': the CUDA APM path is "
            "forward-only, and its backward returns a wrong rho gradient at "
            "body-force source cells (off by ~11x, finite-difference arbitrated; "
            "all backward modes, with or without a hill). Compute gradients "
            "through impl='eager', which matches finite differences to 3e-6 under "
            "APM:\n"
            "  * impl='eager' for the whole run, or\n"
            "  * keep impl='c' for forward-only modelling (that IS supported and "
            "is unaffected) and run the gradient pass on impl='eager', or\n"
            "  * topo_method='image' if you want the gradient on impl='c' — the "
            "image-method topography backward is gradient-consistent with eager "
            "(cos 1.0000000, rel 1e-6) in the full and checkpoint modes."
        )

    def _guard_boundary_saving_topography(self, active):
        """Boundary saving + ``topography=`` produces a WRONG gradient on impl='c'.

        The boundary-saving backward rebuilds the forward wavefield by running
        the propagator backwards in time from the saved edge strips.  With a
        per-column surface that reverse pass does not reproduce the surface
        treatment the forward applied, so the reconstructed wavefield — and with
        it the imaging condition — drifts away from the true one.  The forward
        record is unaffected (it matches the full path to ~5e-7), which is why
        this stays invisible until you look at the gradient:

            elastic 2-D, Gaussian hill, source vz, d(loss)/d(vp,vs,rho)
                full / chunk ckpt / recursive ckpt   cos 1.0000000  rel 1e-6
                boundary saving                      cos -0.028     rel 11

        It is not a near-miss and not a tolerance question — the gradient points
        the wrong way.  It scales with how far the surface sits from row 0 (a
        one-cell-high hill already gives cos 0.52) and hits any surface that is
        not identically zero, including a CONSTANT non-zero topography.  Acoustic
        is affected too (cos 0.9935, rel 0.12 on the same hill), so this is a
        property of the reconstruction, not of the elastic kernels.  Unaffected:
        ``free_surface=True`` with no topography, both checkpoint modes, the full
        path, and eager boundary saving (``impl='eager'`` +
        ``MemoryOptions(strategy='boundary')``), which all agree to ~1e-6.

        Fail loud rather than hand back a wrong gradient.
        """
        if not active:
            return
        # ``topography=`` was actually given: the image path stores the runtime
        # surface rows, the APM path stores per-cell categories instead.  A flat
        # ``free_surface=True`` also resolves ``_topo_method='image'`` but leaves
        # ``_topo_rows_runtime`` None — that configuration is fine and must NOT
        # trip this guard.
        has_topography = (getattr(self, "_topo_rows_runtime", None) is not None
                          or getattr(self, "_topo_method", None) == "apm")
        if not has_topography:
            return
        raise NotImplementedError(
            "boundary saving cannot be combined with topography= on impl='c': the "
            "reverse reconstruction does not reproduce the surface treatment, so "
            "the gradient comes out wrong (cosine against the full/checkpoint path "
            "drops to ~0 for a 5-cell hill). The forward record is unaffected, "
            "which is why this is otherwise silent.\n"
            "NOTE: boundary saving is impl='c''s DEFAULT memory strategy, so you "
            "can hit this without having asked for it. Fixes:\n"
            "  * cuda_options=CUDAOptions(memory=MemoryOptions(strategy='ckpt', "
            "ckpt=CkptOptions(mode='chunk', chunks=N)))  — checkpointing, "
            "gradient-consistent under topography\n"
            "  * boundary_saving_config={'enabled': False}  — full wavefield, "
            "also gradient-consistent, but stores every step\n"
            "  * impl='eager' with MemoryOptions(strategy='boundary')  — the eager "
            "reconstruction is correct under topography\n"
            "A flat free surface (free_surface=True, no topography=) is unaffected."
        )

    def _cuda_spacing(self):
        # PropBase stores spacing in model-axis order: (dz, dx) or (dz, dy, dx).
        # CUDA kernels expect Cartesian order: [dx, dz] or [dx, dy, dz].
        return list(reversed(self._grid_spacing))

    def _build_fd_coefficients(self, M):
        lap_coes = torch.zeros(M + 1, dtype=torch.float32, device=self.dev)

        lap_coes[1:] = torch.from_numpy(fd_coefficients(2, 2 * M)).to(self.dev).float()
        lap_coes[0] = torch.sum(lap_coes[1:]) * 2

        # First-order CUDA equations use staggered-grid forward/backward
        # derivatives (`sgradient`), so their first-derivative coefficients
        # must match the staggered stencil rather than the centered gradient
        # coefficients used by second-order equations.
        if isinstance(self.equation, FirstOrderEquation):
            # Runtime staggered kernels read coeff[0..M-1] directly.
            grad_coes = torch.from_numpy(staggered_grid_coes(M)).to(self.dev).float()
        else:
            # Runtime centered-gradient kernels read coeff[1..M], leaving
            # coeff[0] unused for consistency with the Laplacian layout.
            grad_coes = torch.zeros(M + 1, dtype=torch.float32, device=self.dev)
            grad_coes[1:] = torch.from_numpy(fd_coefficients(1, 2 * M)).to(self.dev).float()
            grad_coes[0] = 0.0
        return lap_coes, grad_coes

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

    def _cuda_layout(self):
        layout = getattr(self.equation, "cuda_layout", None)
        if layout is not None:
            return layout

        raise AttributeError(
            f"{type(self.equation).__name__} must define `cuda_layout` to run with PropTorch impl='c'."
        )

    def _remove_boundary_disk_cache(self):
        # Bypass nn.Module.__setattr__ for these book-keeping attrs: they are
        # plain state, not Parameters/submodules. __setattr__ runs
        # isinstance(value, Parameter), which on interpreter shutdown fails with
        # TypeError once torch.Tensor has been torn down to None -- and __del__
        # calls this method, so the failure surfaces as "Exception ignored in
        # __del__". Writing straight to __dict__ avoids that path entirely.
        root = self.__dict__.get("_boundary_disk_root")
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
        self.__dict__["_boundary_disk_root"] = None
        self.__dict__["_boundary_disk_files"] = ()

    def _allocate_boundary_disk_files(self, shapes, disk_dir, element_size=4):
        root = tempfile.mkdtemp(prefix="sweep_boundary_", dir=disk_dir)
        files = []
        for idx, shape in enumerate(shapes):
            path = os.path.join(root, f"boundary_{idx}.bin")
            numel = int(torch.Size(shape).numel())
            with open(path, "wb") as handle:
                handle.truncate(numel * element_size)
            files.append(path)
        self._boundary_disk_root = root
        self._boundary_disk_files = tuple(files)

    def _int8_scale_shapes(self, main_shapes, block=256):
        """Per-face FP32 scale-buffer shapes for INT8 boundary storage.

        Mirror each uint8 main shape but collapse the spatial dims of one
        timestep slot to ``ceil(cells_per_step / block)`` blocks, preserving
        the outer (time*field, batch[, ...]) axes so ``_slice_boundary_buffers``
        narrows the same batch axis as the main tensors.  Used for both the
        gpu-direct (full-nt) buffers and the staged (ring) buffers, so the
        C++ saver sees a consistent per-step block count on either path.
        """
        scale_shapes = []
        for s in main_shapes:
            if self.ndim == 3:
                outer = (s[0], s[1])
                spatial = s[2:]
            else:
                outer = (s[0], s[1], s[2])
                spatial = s[3:]
            cells_per_step = 1
            for d in spatial:
                cells_per_step *= d
            n_blocks = (cells_per_step + block - 1) // block
            scale_shapes.append(tuple(outer) + (n_blocks,))
        return tuple(scale_shapes)

    def _allocate_boundary_disk_files_int8(self, main_shapes, scale_shapes, disk_dir,
                                           main_element_size=1):
        """Disk files for staged scaled storage (int8/fp16): N payload main
        files (es=1 for uint8, 2 for fp16) followed by N FP32 per-block scale
        files (es=4), full-nt sized.  The order (main..., scale...) matches
        the concatenation the C++ saver expects in ``boundary_disk_files``."""
        root = tempfile.mkdtemp(prefix="sweep_boundary_", dir=disk_dir)
        files = []
        for idx, shape in enumerate(main_shapes):
            path = os.path.join(root, f"boundary_{idx}.bin")
            numel = int(torch.Size(shape).numel())
            with open(path, "wb") as handle:
                handle.truncate(numel * main_element_size)
            files.append(path)
        for idx, shape in enumerate(scale_shapes):
            path = os.path.join(root, f"boundary_scale_{idx}.bin")
            numel = int(torch.Size(shape).numel())
            with open(path, "wb") as handle:
                handle.truncate(numel * 4)  # fp32 scale
            files.append(path)
        self._boundary_disk_root = root
        self._boundary_disk_files = tuple(files)

    def _ensure_boundary_buffers(self, boundary_storage, transfer_interval, use_pinned_memory, disk_dir=None, ring_buffers=1, boundary_dtype=None):
        boundary_on_cpu = boundary_storage in {"cpu", "disk"}
        staging_pinned = use_pinned_memory or boundary_storage == "disk"
        staging_interval = transfer_interval * ring_buffers
        # Precedence: explicit Python kwarg (from BoundaryOptions /
        # boundary_saving_config dict) wins over the SWEEP_BOUNDARY_DTYPE env
        # var (the documented global override), which wins over the legacy
        # SWEEP_FP16_BOUNDARY=1 gate, which wins over the ``fp32`` default.
        # This only decides what dtype to ALLOCATE the boundary buffers in; we
        # deliberately do NOT write the choice back into os.environ.  The C++
        # saver derives the storage dtype from the buffer tensors it is handed
        # (see csrc/.../boundary/saver.cuh), so a per-instance storage_dtype
        # cannot leak into later default-config propagators in the same process.
        if boundary_dtype is None:
            _env_dtype = os.environ.get('SWEEP_BOUNDARY_DTYPE', '').strip().lower()
            if _env_dtype in ('fp32', 'fp16', 'bf16', 'int8'):
                boundary_dtype = _env_dtype
            elif os.environ.get('SWEEP_FP16_BOUNDARY', '') in ('1', 'true', 'yes', 'on'):
                boundary_dtype = 'fp16'
            else:
                boundary_dtype = 'fp32'
        if boundary_dtype not in ('fp32', 'fp16', 'bf16', 'int8'):
            raise ValueError(f"boundary_dtype must be 'fp32'/'fp16'/'bf16'/'int8', got {boundary_dtype!r}")
        # All storage locations (gpu/cpu/disk) support every storage_dtype
        # (fp32/fp16/bf16/int8).  Staged int8 uses a uint8 main + FP32 per-block
        # scale ring (and, for disk, parallel uint8 + scale files); see the
        # int8 allocation branch below and saver.cuh / runtime.cuh.
        if (
            self._boundary_cache_batch == self.B
            and
            self._boundary_cache_mode == boundary_storage
            and self._boundary_cache_interval == transfer_interval
            and self._boundary_cache_ring_buffers == ring_buffers
            and self._boundary_cache_pinned == staging_pinned
            and self._boundary_cache_disk_dir == disk_dir
            and self._boundary_cache_nt == self.nt
            and getattr(self, '_boundary_cache_dtype', None) == boundary_dtype
        ):
            return

        cuda_layout = self._cuda_layout()
        layout = Layout(
            self.shape_cuda,
            cuda_layout.resolved_boundary_save_nvar(),
            self.nt,
            self.abcn,
            self.equation.so // 2,
            self.B,
            staging_interval if boundary_on_cpu else transfer_interval,
            self._image_method_active,
            self.equation.so // 2 + 1,
            tangent_pad=cuda_layout.boundary_tangent_pad,
            pad=self.pad,
            cut_mask=getattr(self, "_dd_cut_mask", 0),
        )

        self.boundary_cpu_allocator = Allocator('cpu')
        self.boundary_gpu_allocator = Allocator(self.dev)
        self._remove_boundary_disk_cache()
        last_two_shape = [
            cuda_layout.resolved_last_two_storage_nvar(),
            cuda_layout.last_two_nvar,
            self.B,
            1,
            *self.shape_cuda,
        ]

        # Storage dtype for the boundary buffers (fp32 default).  int8 is
        # handled separately (gpu-only) below; ``last_two`` always stays fp32.
        _bdry_dtype = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }.get(boundary_dtype, torch.float32)

        if boundary_on_cpu and boundary_dtype in ('int8', 'fp16'):
            # Staged scaled storage (int8 / fp16): a persistent payload main
            # (uint8 or fp16) + FP32 per-block scale on the cpu side
            # (boundary_cpu = main..., scale...) AND a payload main ring +
            # FP32 scale ring on the gpu (boundary_gpu = main..., scale...),
            # both concatenated main-then-scale to mirror the gpu-direct
            # layout the saver already splits.  For storage='disk' the cpu
            # side is staging-sized and the full-nt data lives in payload +
            # FP32 files.  fp16 shares int8's two-pass flow because a bare
            # fp16 cast flushes values below 2^-24 to zero, which wipes the
            # velocity faces of elastic wavefields (see quantize_fp16_kernel).
            _BLOCK = 256
            main_dtype = torch.uint8 if boundary_dtype == 'int8' else torch.float16
            cpu_main_shapes = (layout.gpu_shapes if boundary_storage == "disk"
                               else layout.cpu_shapes)
            ring_main_shapes = layout.gpu_shapes
            cpu_scale_shapes = self._int8_scale_shapes(cpu_main_shapes, _BLOCK)
            ring_scale_shapes = self._int8_scale_shapes(ring_main_shapes, _BLOCK)
            if boundary_storage == "disk":
                self._allocate_boundary_disk_files_int8(
                    layout.cpu_shapes,
                    self._int8_scale_shapes(layout.cpu_shapes, _BLOCK),
                    disk_dir,
                    main_element_size=torch.empty((), dtype=main_dtype).element_size())
            cpu_main = self.boundary_cpu_allocator.zeros(
                cpu_main_shapes, dtype=main_dtype, dev='cpu', pin_memory=staging_pinned)
            cpu_scale = self.boundary_cpu_allocator.zeros(
                cpu_scale_shapes, dtype=torch.float32, dev='cpu', pin_memory=staging_pinned)
            self.boundary_cpu = tuple(cpu_main) + tuple(cpu_scale)
            ring_main = self.boundary_gpu_allocator.zeros(
                ring_main_shapes, dtype=main_dtype)
            ring_scale = self.boundary_gpu_allocator.zeros(
                ring_scale_shapes, dtype=torch.float32)
            self.boundary_gpu = tuple(ring_main) + tuple(ring_scale)
            self.boundary_gpu_full = ()
            self.last_two = self.boundary_cpu_allocator.zeros(
                [last_two_shape],
                dtype=torch.float32,
                dev='cpu',
                pin_memory=staging_pinned,
            )[0]
        elif boundary_on_cpu:
            if boundary_storage == "disk":
                self._allocate_boundary_disk_files(
                    layout.cpu_shapes, disk_dir,
                    element_size=torch.empty((), dtype=_bdry_dtype).element_size())
                self.boundary_cpu = self.boundary_cpu_allocator.zeros(
                    layout.gpu_shapes,
                    dtype=_bdry_dtype,
                    dev='cpu',
                    pin_memory=staging_pinned,
                )
            else:
                self.boundary_cpu = self.boundary_cpu_allocator.zeros(
                    layout.cpu_shapes,
                    dtype=_bdry_dtype,
                    dev='cpu',
                    pin_memory=staging_pinned,
                )
            # Staging ring must match the persistent buffer dtype.
            self.boundary_gpu = self.boundary_gpu_allocator.zeros(
                layout.gpu_shapes, dtype=_bdry_dtype)
            self.boundary_gpu_full = ()
            self.last_two = self.boundary_cpu_allocator.zeros(
                [last_two_shape],
                dtype=torch.float32,
                dev='cpu',
                pin_memory=staging_pinned,
            )[0]
        else:
            self.boundary_cpu = ()
            self.boundary_gpu = ()
            # Boundary storage dtype: FP32 (default) / FP16 / BF16 cast at
            # the storage boundary inside boundary_kernel*_{fp16,bf16}, or
            # INT8 with per-block symmetric quantization (DeepWave-style,
            # see boundarysaver.cu: quantize_int8_kernel).  last_two stays
            # FP32 — it's a wavefield snapshot used to bootstrap backward.
            if boundary_dtype in ('int8', 'fp16'):
                # Scaled storage (int8 / fp16): Python owns a persistent
                # payload main (uint8 or fp16) + FP32 scale tensors so they
                # survive across forward/backward calls (CUDA-side saver is
                # recreated each call and would otherwise lose the data).
                # Layout: list of N main payload tensors followed by N scale
                # FP32 tensors, where N = 4 (2D) or 6 (3D).  Saver detects
                # this pattern via dtype and binds main + scale + allocates
                # FP32 staging internally.  fp16 shares int8's two-pass flow
                # because a bare fp16 cast flushes values below 2^-24 to
                # zero, wiping the velocity faces of elastic wavefields.
                # Scale shapes preserve the batch dimension so that
                # ``_slice_boundary_buffers`` narrows the same axis as the
                # main tensors.  Spatial dims collapse to n_blocks (shared
                # with the staged path via ``_int8_scale_shapes``).
                #   3D main  (nvar*nt, B, W, Ny_b, Nx_b)
                #     scale  (nvar*nt, B, n_blocks_per_step)
                #   2D main  (nvar, nt, B, W, Nx_b)
                #     scale  (nvar, nt, B, n_blocks_per_step)
                main_dtype = torch.uint8 if boundary_dtype == 'int8' else torch.float16
                main_shapes = layout.gpu_full_shapes
                main_tensors = self.boundary_gpu_allocator.zeros(
                    main_shapes, dtype=main_dtype)
                scale_tensors = self.boundary_gpu_allocator.zeros(
                    self._int8_scale_shapes(main_shapes), dtype=torch.float32)
                self.boundary_gpu_full = tuple(main_tensors) + tuple(scale_tensors)
            else:
                self.boundary_gpu_full = self.boundary_gpu_allocator.zeros(
                    layout.gpu_full_shapes, dtype=_bdry_dtype)
            self.last_two = self.forward_allocator.zeros([last_two_shape])[0]

        self._boundary_cache_dtype = boundary_dtype
        self._boundary_cache_mode = boundary_storage
        self._boundary_cache_interval = transfer_interval
        self._boundary_cache_ring_buffers = ring_buffers
        self._boundary_cache_pinned = staging_pinned
        self._boundary_cache_disk_dir = disk_dir
        self._boundary_cache_nt = self.nt
        self._boundary_cache_batch = self.B

    def __del__(self):
        self._remove_boundary_disk_cache()

    def _aux_slab_len(self, caxis):
        """Slab length of one CPML aux axis (caxis: 0=z, 1=y, 2=x, C order).

        Mirrors SolverContext::aux_slab_formula exactly: per non-cut side
        ``pad + 3*M + 1`` (band + widest adjoint write band + stencil tap
        reach + staggered half cell), a DD cut side carries nothing, an
        all-cut axis keeps one dummy column for clamped reads, and when the
        two slabs meet the axis degenerates to full coverage.  The C++ side
        recomputes this from the same inputs and TORCH_CHECKs the bound
        tensors, so drift between the two is loud.
        """
        M = self.equation.so // 2
        dims = {0: 0, 1: 1, 2: self.ndim - 1}
        n = self.shape_cuda[dims[caxis]]
        pad_i = {0: 0, 1: 2, 2: (2 if self.ndim == 2 else 4)}[caxis]
        cm = getattr(self, "_dd_cut_mask", 0) or 0
        cut_lo, cut_hi = {0: (4, 8), 1: (16, 32), 2: (1, 2)}[caxis]
        lo = 0 if cm & cut_lo else self.pad[pad_i] + 3 * M + 1
        hi = 0 if cm & cut_hi else self.pad[pad_i + 1] + 3 * M + 1
        if lo + hi == 0:
            lo = 1
        if lo + hi >= n:
            return n
        return lo + hi

    def _aux_slab_shape(self, axis_char, lead):
        """[B, 1, ...] shape of one slab-allocated aux slot (lead = [B, 1])."""
        caxis = {"z": 0, "y": 1, "x": 2}[axis_char]
        w = self._aux_slab_len(caxis)
        sp = list(self.shape_cuda)
        sp[{0: 0, 1: 1, 2: self.ndim - 1}[caxis]] = w
        return lead + sp

    def _forward_wavefield_shapes(self):
        """Per-slot forward wavefield shapes: physical slots on the full grid,
        CPML aux slots as per-axis slabs when the equation declares
        ``pml_slot_axes`` (kernels adapt per bound tensor either way)."""
        cuda_layout = self._cuda_layout()
        base = [[self.B, 1, *self.shape_cuda]] * cuda_layout.base_nvar
        axes = getattr(cuda_layout, "pml_slot_axes", None)
        if not axes:
            return base + [[self.B, 1, *self.shape_cuda]] * cuda_layout.pml_nvar
        assert len(axes) == cuda_layout.pml_nvar, "pml_slot_axes/pml_nvar mismatch"
        return base + [self._aux_slab_shape(a, [self.B, 1]) for a in axes]

    def _ensure_wavefield_buffers(self, batch_size, persist_forward_state=False, need_adjoint=True):
        # persist_forward_state: keep the forward propagation-state buffers
        # (u_prev/now/next + psi/zeta, shape [B,1,*spatial]) as persistent,
        # reused-across-calls tensors. True only in save_all (full) mode,
        # whose backward replays from these. BS/ckpt/no-grad use per-call
        # transient state instead (_transient_forward_wavefields) — they are
        # NOT gated here. This is orthogonal to save_all_wavefields, which
        # gates the big (nt,B,*spatial) u_allt history array, not this state.
        current_capacity = self._buffer_capacity_batch
        if (
            current_capacity is not None
            and batch_size <= current_capacity
            and (not persist_forward_state or self.forward_wavefields)
            and (not need_adjoint or self.adjoint_wavefields)
        ):
            return

        if current_capacity is not None and batch_size > current_capacity and not self.allow_growth:
            raise ValueError(
                f"Input batch size {batch_size} exceeds preallocated CUDA buffer capacity {current_capacity}. "
                "Increase B when constructing PropTorch(..., impl='c') or set allow_growth=True."
            )

        target_capacity = batch_size
        if current_capacity is None:
            target_capacity = max(self.B, batch_size)

        self.B = target_capacity
        cuda_layout = self._cuda_layout()
        total_wavefields = cuda_layout.base_nvar + cuda_layout.pml_nvar
        wavefield_shapes = self._forward_wavefield_shapes()
        # The adjoint may need extra double-buffer tensors (fused single-kernel
        # adjoint double-buffers zeta); the forward never does.  Acoustic's
        # fused adjoint stencil-taps psi/zeta, so its aux stays FULL-domain
        # (adjoint_pml_slab=False); elastic memory variables are own-cell only
        # and reuse the forward slab shapes.
        adjoint_wavefields_n = total_wavefields + int(getattr(cuda_layout, "adjoint_extra_nvar", 0))
        if getattr(cuda_layout, "adjoint_pml_slab", False):
            extra = adjoint_wavefields_n - total_wavefields
            adjoint_wavefield_shapes = list(wavefield_shapes) + \
                [[self.B, 1, *self.shape_cuda]] * extra
        else:
            adjoint_wavefield_shapes = adjoint_wavefields_n * [[self.B, 1, *self.shape_cuda]]
        if batch_size > (current_capacity or 0):
            self.forward_allocator = Allocator(self.dev)
            self.adjoint_allocator = Allocator(self.dev)
            self.forward_wavefields = ()
            self.adjoint_wavefields = ()

        if persist_forward_state and not self.forward_wavefields:
            self.forward_wavefields = self.forward_allocator.zeros(wavefield_shapes)
        if need_adjoint and not self.adjoint_wavefields:
            self.adjoint_wavefields = self.adjoint_allocator.zeros(adjoint_wavefield_shapes)
        self._buffer_capacity_batch = target_capacity
        self._boundary_cache_batch = None
        self._boundary_cache_ring_buffers = None
        self._checkpoint_cache_batch = None

    def _slice_wavefield_buffers(self, batch_size):
        return tuple(t[:batch_size] for t in self.forward_wavefields), tuple(t[:batch_size] for t in self.adjoint_wavefields)

    def _transient_forward_wavefields(self, batch_size):
        """Per-call zeroed forward wavefields for the non-save_all modes
        (boundary saving / checkpoint / no-grad forward).

        All wavefield state is allocated Python-side so ``cuda_layout``
        stays the single layout authority — the C++-internal ``allocate()``
        once missed the psi double-buffer slots and silently re-enabled the
        in-place psi RAW race.  Deliberately transient (not the persistent
        Allocator): propagation scratch with one-call lifetime, exactly like
        the C++ scratch it replaces, so boundary-saving backward peak memory
        is unchanged.
        """
        return tuple(
            torch.zeros([batch_size, 1, *shape[2:]], device=self.dev)
            for shape in self._forward_wavefield_shapes()
        )

    def _ensure_adjoint_workspace_buffers(self, batch_size):
        cuda_layout = self._cuda_layout()
        workspace_nvar = int(cuda_layout.backward_workspace_nvar)
        custom_shapes_fn = cuda_layout.backward_workspace_shapes
        has_custom_shapes = callable(custom_shapes_fn)
        if workspace_nvar <= 0 and not has_custom_shapes:
            self.adjoint_workspace = ()
            self._workspace_cache_batch = batch_size
            self._workspace_cache_nt = self.nt
            return

        if (
            self._workspace_cache_batch is not None
            and batch_size <= self._workspace_cache_batch
            and self._workspace_cache_nt == self.nt
        ):
            return

        self.workspace_allocator = Allocator(self.dev)
        if has_custom_shapes:
            workspace_shapes = custom_shapes_fn(self.B, self.nt, self.shape_cuda)
        else:
            workspace_shapes = workspace_nvar * [[self.B, 1, *self.shape_cuda]]
        self.adjoint_workspace = self.workspace_allocator.zeros(workspace_shapes)
        self._workspace_cache_batch = self.B
        self._workspace_cache_nt = self.nt

    def _slice_adjoint_workspace_buffers(self, batch_size):
        if not self.adjoint_workspace:
            return ()
        return tuple(t[:batch_size] for t in self.adjoint_workspace)

    def _slice_last_two(self, batch_size):
        return self.last_two[:, :, :batch_size]

    def _slice_boundary_buffers(self, tensors, batch_size):
        if not tensors:
            return ()

        batch_dim = 1 if self.ndim == 3 else 2
        return tuple(t.narrow(batch_dim, 0, batch_size) for t in tensors)

    def _ensure_checkpoint_buffers(self, checkpoint_interval=None, checkpoint_count=None, batch_size=None):
        if not self.use_ckpt or (self.backward_ckpt_func is None and self.backward_recursive_ckpt_func is None):
            return

        if checkpoint_count is not None:
            n_checkpoints = int(checkpoint_count)
        else:
            n_checkpoints = max(1, (self.nt + checkpoint_interval - 1) // checkpoint_interval)

        # Allocate the checkpoint buffer at the *active* batch size for this
        # call, not at the cached ``self.B`` capacity.  The batch dimension
        # is dim 1 of the buffer, so ``t[:, :active]`` would be a
        # non-contiguous view when ``active < B`` — and the CUDA checkpoint
        # kernels require contiguous storage.  Allocating at the exact
        # active size lets ``_slice_checkpoint_buffers`` return the tensor
        # as-is (always contiguous).
        active_batch = int(batch_size) if batch_size is not None else int(self.B)

        checkpoint_storage = self.ckpt_storage
        checkpoint_pinned = self.ckpt_pinned_memory if checkpoint_storage == "cpu" else False

        if (
            self._checkpoint_cache_batch == active_batch
            and self._checkpoint_cache_interval == checkpoint_interval
            and self._checkpoint_cache_count == n_checkpoints
            and self._checkpoint_cache_nt == self.nt
            and self._checkpoint_cache_storage == checkpoint_storage
            and self._checkpoint_cache_pinned == checkpoint_pinned
        ):
            return

        cuda_layout = self._cuda_layout()
        num_checkpoint_tensors = int(cuda_layout.resolved_checkpoint_nvar())
        ckpt_axes = getattr(cuda_layout, "checkpoint_slot_axes", None)
        if ckpt_axes:
            # Per-slot shapes: physical slots on the full grid, CPML aux
            # slots as per-axis slabs (mirrors the C++ checkpoint_tensors()
            # slot order for this equation).
            assert len(ckpt_axes) == num_checkpoint_tensors, \
                "checkpoint_slot_axes/checkpoint_nvar mismatch"
            checkpoint_shapes = [
                [n_checkpoints, active_batch, 1, *self.shape_cuda] if a is None
                else self._aux_slab_shape(a, [n_checkpoints, active_batch, 1])
                for a in ckpt_axes
            ]
        else:
            checkpoint_shapes = (
                num_checkpoint_tensors
                * [[n_checkpoints, active_batch, 1, *self.shape_cuda]]
            )
        checkpoint_device = "cpu" if checkpoint_storage == "cpu" else self.dev
        self.checkpoint_allocator = Allocator(checkpoint_device)
        self.checkpoints = tuple(
            self.checkpoint_allocator.zeros(
                checkpoint_shapes,
                dtype=torch.float32,
                dev=checkpoint_device,
                pin_memory=checkpoint_pinned,
            )
        )
        self._checkpoint_cache_batch = active_batch
        self._checkpoint_cache_interval = checkpoint_interval
        self._checkpoint_cache_count = n_checkpoints
        self._checkpoint_cache_nt = self.nt
        self._checkpoint_cache_storage = checkpoint_storage
        self._checkpoint_cache_pinned = checkpoint_pinned

    def _slice_checkpoint_buffers(self, batch_size):
        # Checkpoints are allocated at the active batch size in
        # ``_ensure_checkpoint_buffers``, so no slicing is required.  Kept
        # as a stable callsite so the existing forward()/rtm() flow doesn't
        # need restructuring.
        if not self.checkpoints:
            return ()
        return tuple(self.checkpoints)

    def _model_to_cuda_batch(self, m, batch_size):
        """Shape a padded model tensor to the CUDA batch layout ``(B, 1, *spatial)``.

        Two input forms are accepted:

        * **shared** — ``m.ndim == self.ndim`` (e.g. ``(nz, nx)`` in 2-D): the
          model is broadcast across the shot batch by repeating it
          ``batch_size`` times.  PyTorch autograd therefore *sums* the per-shot
          CUDA model gradient back into one shared-model gradient — the
          historical ``impl='c'`` behaviour, preserved bit-for-bit.
        * **per-shot** — ``m.ndim == self.ndim + 1`` (e.g. ``(B, nz, nx)`` in
          2-D): shot ``b`` propagates in ``m[b]``.  A singleton channel axis is
          inserted with **no** repeat, so the compiled kernels' per-batch model
          stride reads each shot's own model and autograd keeps the gradient
          per-shot ``(B, *spatial)``.  Only enabled for equations that advertise
          ``supports_batched_models`` (currently the 2-D Acoustic and Elastic
          solvers); every other equation keeps erroring on a batched model
          rather than silently mis-striding it.
        """
        if m.ndim == self.ndim:
            # Shared model — broadcast across the batch.  Kept identical to the
            # original expression so the shared-model path stays bit-exact.
            return m[None, None, ...].repeat(batch_size, *([1] * (m.ndim + 1)))
        if m.ndim == self.ndim + 1:
            if not (self.ndim == 2 and getattr(self.equation, "supports_batched_models", False)):
                raise NotImplementedError(
                    "Per-shot batched velocity models (a leading batch dim) are "
                    "only supported by 2-D impl='c' solvers whose kernels stride "
                    "the model per batch index (currently Acoustic and Elastic); "
                    f"got a {m.ndim}-D model for {type(self.equation).__name__} "
                    f"(ndim={self.ndim}). Pass a single shared {self.ndim}-D "
                    "model broadcast across the batch instead."
                )
            if m.shape[0] != batch_size:
                raise ValueError(
                    f"Per-shot model batch ({int(m.shape[0])}) must equal the "
                    f"shot batch size ({int(batch_size)}); provide exactly one "
                    "model per shot as (B, nz, nx)."
                )
            # Per-shot model: (B, *spatial) -> (B, 1, *spatial).  No repeat, so
            # the CUDA per-batch gradient flows back per-shot.
            return m[:, None, ...]
        raise ValueError(
            f"Model tensor has {m.ndim} dims; expected {self.ndim} for a shared "
            f"model or {self.ndim + 1} for a per-shot batched (B, ...) model."
        )

    @torch._dynamo.disable
    def forward(self, wavelet, sources, receivers, models=None, adj=False, return_wavefield=False, use_boundary_saving=None, boundary_saving_config=None, **kwargs):
        """Forward pass of the wave equation.

        Accepted input shapes (see :meth:`PropBase._normalize_io`):

        - ``wavelet=(nt,)`` + ``sources=(nshots, ndim)`` — shared wavelet, one
          point source per shot.
        - ``wavelet=(nshots, nt)`` + ``sources=(nshots, ndim)`` — per-shot
          wavelet, one point source per shot.
        - ``wavelet=(nt,)`` or ``(nsrc, nt)`` + ``sources=(1, nsrc, ndim)`` —
          source encoding (one super-shot, ``nsrc`` superposed point sources).

        ``receivers`` must always be ``(B, nrec, ndim)``; pre-broadcast a
        shared receiver array to per-shot form.

        Args:
            wavelet: Source wavelet (numpy or torch).
            sources: Source coordinates.
            receivers: Receiver coordinates.
            models: List of model parameters (must be ``torch.Tensor``).
        """

        legacy_override = {}
        if "transfer_interval" in kwargs:
            legacy_override["transfer_interval"] = kwargs.pop("transfer_interval")
        if "boundary_on_cpu" in kwargs:
            legacy_override["storage"] = "cpu" if kwargs.pop("boundary_on_cpu") else "gpu"
        if "use_pinned_memory" in kwargs:
            legacy_override["pinned_memory"] = kwargs.pop("use_pinned_memory")
        if "boundary_ring_buffers" in kwargs:
            legacy_override["ring_buffers"] = kwargs.pop("boundary_ring_buffers")
        if "boundary_disk_async_read" in kwargs:
            legacy_override["disk_async_read"] = kwargs.pop("boundary_disk_async_read")
        if boundary_saving_config is None and legacy_override:
            boundary_saving_config = legacy_override
        elif legacy_override:
            boundary_saving_config = {**legacy_override, **boundary_saving_config}

        mode, batch_size, nsrc_per_shot, nrec, source_encoding = self._normalize_io(
            wavelet, sources, receivers
        )

        boundary_cfg = self.resolve_boundary_saving_config(
            override=boundary_saving_config,
            use_boundary_saving=use_boundary_saving,
        )
        use_boundary_saving = boundary_cfg["enabled"]
        boundary_storage = boundary_cfg["storage"]
        boundary_on_cpu = boundary_storage in {"cpu", "disk"}
        boundary_on_disk = boundary_storage == "disk"
        transfer_interval = boundary_cfg["transfer_interval"]
        boundary_ring_buffers = boundary_cfg["ring_buffers"]
        boundary_disk_async_read = boundary_cfg["disk_async_read"]
        use_pinned_memory = boundary_cfg["pinned_memory"]
        boundary_disk_dir = boundary_cfg.get("disk_dir")

        self.nt = wavelet.shape[-1]
        if self.use_ckpt and self.ckpt_mode not in {"chunk", "recursive"}:
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}'. Expected 'chunk' or 'recursive'.")
        checkpoint_steps = torch.empty(0, dtype=torch.int32)

        # Set zeros
        M = self.equation.so // 2

        pml_padding = M
        padding = [p+M for p in self.padding]
        base_shift = M + self.abcn

        shape_for_pml = [p+2*M for p in self.shape]

        kwargs['shape'] = shape_for_pml
        self.init_abc(**kwargs)

        nt = self.nt
        sources = sources.copy()
        receivers = receivers.copy()

        # Shift physical (x,[y,]z) coords into the padded runtime grid by each
        # axis' LOW-side pad + M.  Per-edge aware (free-surface faces have 0 pad,
        # so e.g. a top free surface shifts z by only M, a left free surface x by
        # only M), and DD-aware for free: ``self.pad`` already carries the cut
        # faces, so a cut face shifts by only M and coords land in the right
        # runtime cell on a compact-padded tile.  For the top-only / no-FS
        # single-domain defaults this reproduces the old ``base_shift`` (x/y) +
        # ``M`` (z) behaviour bit-for-bit.
        coord_offset = self._runtime_coord_offset()   # (x, [y,] z) order
        for _i in range(self.ndim):
            sources[..., _i] += coord_offset[_i]
            receivers[..., _i] += coord_offset[_i]

        # Canonicalize wavelet/sources to (B, nsrc_per_shot, nt) / (B, nsrc_per_shot, ndim).
        # `mode` was validated by _normalize_io above.
        if isinstance(wavelet, torch.Tensor):
            wavelet = wavelet.to(self.dev, dtype=torch.float32)
        else:
            wavelet = torch.from_numpy(wavelet).to(self.dev).float()
        if mode == 'A1':
            wavelet = wavelet[None, None, :].repeat(batch_size, 1, 1)
        elif mode == 'A2':
            wavelet = wavelet[:, None, :]
        else:  # mode == 'B'
            if wavelet.ndim == 1:
                wavelet = wavelet[None, None, :].repeat(1, nsrc_per_shot, 1)
            else:  # (nsrc, nt)
                wavelet = wavelet[None, :, :]

        sources = torch.from_numpy(sources).to(self.dev).int()
        if mode in ('A1', 'A2'):
            sources = sources[:, None, :]
        # mode == 'B': already (1, nsrc, ndim)
        receivers = torch.from_numpy(receivers).to(self.dev).int()
        source_field_indices = self._field_indices_tensor(self.source_type, is_source=True)
        receiver_field_indices = self._field_indices_tensor(self.receiver_type, is_source=False)
        # Get the model parameters

        models = list(models if models is not None else self.parameters())
        unpadded_models = models
        models = [EdgePadding.apply(para, padding) for para in models]
        self.models_padded = models
        # self.equation.b = pad_pml_vals(self.equation.b, pml_padding)
        
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ay', 'by', 'ayh', 'byh', 'ax', 'bx', 'axh', 'bxh']):
        # for b, name in zip(self.equation.b, ['az', 'bz', 'dbzdz', 'ax', 'bx', 'dbxdx']):
        # for b, name in zip(self.equation.b, ['az', 'bz', 'azh', 'bzh', 'ax', 'bx', 'axh', 'bxh']):
        #     np.save(f'{name}.npy', b.detach().cpu().numpy())

        lap_coes, grad_coes = self._build_fd_coefficients(M)

        models = [self._model_to_cuda_batch(m, batch_size) for m in self.models_padded]
        if getattr(self.equation, "prepare_models_for_c", False):
            prepare = getattr(self.equation, "prepare_models", None)
            if not callable(prepare):
                raise AttributeError(
                    f"{type(self.equation).__name__} sets prepare_models_for_c=True "
                    "but does not define prepare_models()."
                )
            models = list(prepare(models))
        requires_model_grad = any(m.requires_grad for m in models)
        requires_wavelet_grad = wavelet.requires_grad
        requires_backward = bool(requires_model_grad or requires_wavelet_grad)
        if requires_backward and self.compute_illumination:
            self.source_illumination = torch.zeros_like(unpadded_models[0])
            self.receiver_illumination = torch.zeros_like(unpadded_models[0])
        else:
            self.source_illumination = None
            self.receiver_illumination = None
        if requires_backward and self.compute_adcig:
            nlag = 2 * int(self.adcig_max_lag) + 1
            self.adcig = torch.zeros(
                (nlag, *unpadded_models[0].shape),
                dtype=unpadded_models[0].dtype,
                device=unpadded_models[0].device,
            )
        else:
            self.adcig = None
        use_checkpoint = bool(self.use_ckpt and requires_backward)
        if not requires_backward:
            use_boundary_saving = False
        # APM first: it is the more fundamental limitation of the two, so its
        # message is the useful one when both would fire.
        self._guard_apm_backward(requires_backward)
        self._guard_boundary_saving_topography(use_boundary_saving and requires_backward)
        use_recursive_checkpoint = bool(
            use_checkpoint and self.ckpt_mode == "recursive" and self.backward_recursive_ckpt_func is not None
        )
        checkpoint_on_cpu = bool(use_checkpoint and self.ckpt_storage == "cpu")
        save_all_wavefields = bool(requires_backward and not use_boundary_saving and not use_checkpoint)
        self._ensure_wavefield_buffers(batch_size, persist_forward_state=save_all_wavefields, need_adjoint=requires_backward)
        self._ensure_adjoint_workspace_buffers(batch_size)
        if use_checkpoint:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()), batch_size=batch_size)
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks, batch_size=batch_size)
        forward_wavefields, adjoint_wavefields = self._slice_wavefield_buffers(batch_size)
        if not forward_wavefields:
            forward_wavefields = self._transient_forward_wavefields(batch_size)
        adjoint_workspace = self._slice_adjoint_workspace_buffers(batch_size)
        checkpoint_buffers = self._slice_checkpoint_buffers(batch_size) if use_checkpoint else ()

        if self.forward_wavefields:
            self.forward_allocator.zero_()
        if self.adjoint_wavefields:
            self.adjoint_allocator.zero_()
        if adjoint_workspace:
            self.workspace_allocator.zero_()
        if use_checkpoint and checkpoint_buffers:
            self.checkpoint_allocator.zero_()
        if boundary_on_cpu:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers, boundary_dtype=boundary_cfg.get('storage_dtype'))
            if boundary_on_disk:
                self.boundary_cpu_allocator.zero_()
                self.boundary_gpu_allocator.zero_()
            boundary_cpu = self._slice_boundary_buffers(self.boundary_cpu, batch_size)
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu, batch_size)
        else:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers, boundary_dtype=boundary_cfg.get('storage_dtype'))
                for t in self.boundary_gpu_full:
                    t.zero_()
            boundary_cpu = ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu_full, batch_size)
        last_two = self._slice_last_two(batch_size) if use_boundary_saving else self.last_two

        spacing = self._cuda_spacing()

        # Topography / APM plumbing.  Image method uses _topo_rows_runtime
        # only; APM additionally precomputes effective moduli + category
        # from the runtime-padded air mask and prepends them to ``models``.
        topo_rows_runtime = getattr(self.equation, "_topo_rows_runtime", None)
        topo_rows_arg = topo_rows_runtime
        has_topo_arg = topo_rows_runtime is not None
        topo_cat_arg = None
        use_apm_arg = False
        models_arg = models
        if getattr(self, "_topo_method", None) == "apm":
            # models[0..2] = vp, vs, rho already padded to runtime by EdgePadding.
            vp_r, vs_r, rho_r = models[0], models[1], models[2]
            lam_r = rho_r * (vp_r ** 2 - 2 * vs_r ** 2)
            mu_r  = rho_r * (vs_r ** 2)
            lam_2mu_r = lam_r + 2 * mu_r
            air_mask_rt = self.equation._apm_air_mask_runtime

            if self.ndim == 2:
                from sweep.equations._topography import (
                    classify_topography, precompute_apm_moduli,
                )
                cat_np = classify_topography(air_mask_rt)
                cat_t = torch.from_numpy(cat_np).to(device=vp_r.device, dtype=torch.int32)
                lam_eff, mu_eff, mu_xz, rho_x, rho_z = precompute_apm_moduli(
                    lam_r, mu_r, rho_r, cat_np,
                )
                # 2-D APM model layout (11 tensors):
                #   vp, vs, rho, lam, mu, lam_2mu,
                #   lam_eff, mu_eff, mu_xz, rho_x, rho_z
                models_arg = (
                    vp_r, vs_r, rho_r, lam_r, mu_r, lam_2mu_r,
                    lam_eff, mu_eff, mu_xz, rho_x, rho_z,
                )
            else:  # 3-D
                from sweep.equations._topography import (
                    classify_topography_3d, precompute_apm_moduli_3d,
                )
                cat_np = classify_topography_3d(air_mask_rt)
                cat_t = torch.from_numpy(cat_np).to(device=vp_r.device, dtype=torch.int32)
                (alpha_xx, alpha_yy, alpha_zz,
                 lam_xx_yy, lam_xx_zz,
                 lam_yy_xx, lam_yy_zz,
                 lam_zz_xx, lam_zz_yy,
                 mu_xy, mu_xz, mu_yz,
                 inv_rho_x, inv_rho_y, inv_rho_z) = precompute_apm_moduli_3d(
                    lam_r, mu_r, rho_r, cat_np,
                )
                # 3-D APM model layout (21 tensors):
                #   vp, vs, rho, lam, mu, lam_2mu,
                #   alpha_xx, alpha_yy, alpha_zz,
                #   lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz,
                #   lam_zz_xx, lam_zz_yy,
                #   mu_xy, mu_xz, mu_yz,
                #   inv_rho_x, inv_rho_y, inv_rho_z
                models_arg = (
                    vp_r, vs_r, rho_r, lam_r, mu_r, lam_2mu_r,
                    alpha_xx, alpha_yy, alpha_zz,
                    lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz,
                    lam_zz_xx, lam_zz_yy,
                    mu_xy, mu_xz, mu_yz,
                    inv_rho_x, inv_rho_y, inv_rho_z,
                )
            topo_cat_arg = cat_t
            use_apm_arg = True

        # CPML profiles are built on ``equation.device`` (default 'cpu'); pin
        # them to the propagator's compute device so an equation created
        # without ``device=`` (e.g. ``Acoustic()`` + ``PropTorch(dev='cuda')``)
        # doesn't feed host pointers to the CUDA kernel -> illegal address.
        # ``.to`` is a no-op when the tensors already live on ``self.dev``.
        pml_vals = [b.to(self.dev) for b in self.equation.b]
        syn = Warpper.apply(
                self.forward_func,
                self.backward_func,
                self.backward_bs_func,
                self.backward_ckpt_func,
                self.backward_recursive_ckpt_func,
                wavelet,
                sources,
                receivers,
                source_field_indices,
                receiver_field_indices,
                (lap_coes, grad_coes),
                M,
                self.abcn,
                spacing,
                self._dt,
                pml_vals,
                use_checkpoint,
                self.ckpt_chunks,
                use_recursive_checkpoint,
                int(checkpoint_steps.numel()),
                checkpoint_steps,
                checkpoint_on_cpu,
                use_boundary_saving,
                use_pinned_memory,
                self._image_method_active,
                transfer_interval,
                boundary_ring_buffers,
                boundary_on_cpu,
                boundary_on_disk,
                boundary_disk_async_read,
                forward_wavefields,
                adjoint_wavefields,
                adjoint_workspace,
                checkpoint_buffers,
                last_two,
                boundary_cpu,
                boundary_gpu,
                self._boundary_disk_files if boundary_on_disk else (),
                self.source_illumination if self.source_illumination is not None else torch.empty(0, device=self.dev),
                self.receiver_illumination if self.receiver_illumination is not None else torch.empty(0, device=self.dev),
                tuple(padding),
                self.adcig if self.adcig is not None else torch.empty(0, device=self.dev),
                int(self.adcig_max_lag),
                # Topography plumbing — both image-method (1-D row) and
                # APM (per-cell category + extended models) are routed
                # through ``Warpper.forward`` via these positional args.
                topo_rows_arg,
                has_topo_arg,
                topo_cat_arg,
                use_apm_arg,
                self._fs_faces_c,
                getattr(self, "_dd_cut_mask", 0),
                *models_arg,
            )
        
        return syn

    def rtm(self, wavelet, sources, receivers, adjoint_source, models=None, **kwargs):
        if self.rtm_func is None:
            raise NotImplementedError(f"RTM is not implemented for {self.equation.__class__.__name__}.")

        use_boundary_saving = kwargs.pop("use_boundary_saving", None)
        boundary_saving_config = kwargs.pop("boundary_saving_config", None)
        mode, batch_size, nsrc_per_shot, nrec, _ = self._normalize_io(
            wavelet, sources, receivers
        )
        if mode == 'B':
            raise NotImplementedError(
                "RTM does not support source encoding inputs "
                "(sources with shape (1, nsrc, ndim)); use naive multi-shot."
            )
        self.nt = wavelet.shape[-1]
        self._ensure_wavefield_buffers(batch_size)
        self._ensure_adjoint_workspace_buffers(batch_size)

        legacy_override = {}
        if "transfer_interval" in kwargs:
            legacy_override["transfer_interval"] = kwargs.pop("transfer_interval")
        if "boundary_on_cpu" in kwargs:
            legacy_override["storage"] = "cpu" if kwargs.pop("boundary_on_cpu") else "gpu"
        if "use_pinned_memory" in kwargs:
            legacy_override["pinned_memory"] = kwargs.pop("use_pinned_memory")
        if "boundary_ring_buffers" in kwargs:
            legacy_override["ring_buffers"] = kwargs.pop("boundary_ring_buffers")
        if "boundary_disk_async_read" in kwargs:
            legacy_override["disk_async_read"] = kwargs.pop("boundary_disk_async_read")
        if boundary_saving_config is None and legacy_override:
            boundary_saving_config = legacy_override
        elif legacy_override:
            boundary_saving_config = {**legacy_override, **boundary_saving_config}
        boundary_cfg = self.resolve_boundary_saving_config(
            override=boundary_saving_config,
            use_boundary_saving=use_boundary_saving,
        )
        use_boundary_saving = boundary_cfg["enabled"]
        boundary_storage = boundary_cfg["storage"]
        boundary_on_cpu = boundary_storage in {"cpu", "disk"}
        boundary_on_disk = boundary_storage == "disk"
        transfer_interval = boundary_cfg["transfer_interval"]
        boundary_ring_buffers = boundary_cfg["ring_buffers"]
        boundary_disk_async_read = boundary_cfg["disk_async_read"]
        use_pinned_memory = boundary_cfg["pinned_memory"]
        boundary_disk_dir = boundary_cfg.get("disk_dir")

        # 2-D RTM only supports the full-wavefield path today.  Silently
        # force it on so the new impl='c' default (boundary saving / GPU,
        # see _normalize_cuda_memory_kwargs in torch.py) doesn't trip up
        # RTM users with a NotImplementedError.  Use a local ``use_ckpt``
        # shadow rather than mutating ``self.use_ckpt`` so other
        # forward()/rtm() calls on this solver keep their configured
        # strategy.
        use_ckpt = self.use_ckpt
        if self.ndim == 2:
            use_boundary_saving = False
            use_ckpt = False
        # RTM runs the same adjoint machinery as the gradient, so it inherits
        # both limitations: the APM backward and, under topography, the
        # boundary-saving reverse reconstruction.
        self._guard_apm_backward(True)
        self._guard_boundary_saving_topography(use_boundary_saving)

        use_recursive_checkpoint = bool(use_ckpt and self.ckpt_mode == "recursive" and self.backward_recursive_ckpt_func is not None)
        checkpoint_on_cpu = bool(use_ckpt and self.ckpt_storage == "cpu")
        if use_ckpt and self.ckpt_mode not in {"chunk", "recursive"}:
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}'. Expected 'chunk' or 'recursive'.")
        checkpoint_steps = torch.empty(0, dtype=torch.int32)
        if use_ckpt:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()), batch_size=batch_size)
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks, batch_size=batch_size)

        self.forward_allocator.zero_()
        checkpoint_buffers = self._slice_checkpoint_buffers(batch_size) if use_ckpt else ()
        if use_ckpt and checkpoint_buffers:
            self.checkpoint_allocator.zero_()
        if use_boundary_saving:
            self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers, boundary_dtype=boundary_cfg.get('storage_dtype'))
        if boundary_on_cpu:
            if boundary_on_disk and use_boundary_saving:
                self.boundary_cpu_allocator.zero_()
                self.boundary_gpu_allocator.zero_()
            boundary_cpu = self._slice_boundary_buffers(self.boundary_cpu, batch_size) if use_boundary_saving else ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu, batch_size) if use_boundary_saving else ()
        else:
            if use_boundary_saving:
                for t in self.boundary_gpu_full:
                    t.zero_()
            boundary_cpu = ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu_full, batch_size) if use_boundary_saving else ()
        last_two = self._slice_last_two(batch_size) if use_boundary_saving else self.last_two
        forward_wavefields, adjoint_wavefields = self._slice_wavefield_buffers(batch_size)
        save_all_wavefields = not use_boundary_saving and not use_ckpt

        M = self.equation.so // 2
        padding = [p + M for p in self.padding]
        base_shift = M + self.abcn
        shape_for_pml = [p + 2 * M for p in self.shape]
        kwargs["shape"] = shape_for_pml
        self.init_abc(**kwargs)

        sources = sources.copy()
        receivers = receivers.copy()
        # Per-edge (and cut-aware) coord shift, see the forward path: each axis'
        # low-side pad + M.
        coord_offset = self._runtime_coord_offset()   # (x, [y,] z) order
        for _i in range(self.ndim):
            sources[..., _i] += coord_offset[_i]
            receivers[..., _i] += coord_offset[_i]

        if isinstance(wavelet, torch.Tensor):
            wavelet_t = wavelet.to(self.dev, dtype=torch.float32)
        else:
            wavelet_t = torch.from_numpy(wavelet).to(self.dev).float()
        if mode == 'A1':
            wavelet_t = wavelet_t[None, None, :].repeat(batch_size, 1, 1)
        else:  # mode == 'A2'
            wavelet_t = wavelet_t[:, None, :]
        sources_t = torch.from_numpy(sources).to(self.dev).int()[:, None, :]
        receivers_t = torch.from_numpy(receivers).to(self.dev).int()
        adjoint_source_t = torch.as_tensor(adjoint_source, device=self.dev, dtype=torch.float32)
        if adjoint_source_t.ndim == 4:
            if adjoint_source_t.shape[-1] != 1:
                raise ValueError(
                    "PropTorch impl='c' RTM currently expects a single receiver channel; "
                    f"got adjoint_source shape {tuple(adjoint_source_t.shape)}"
                )
            adjoint_source_t = adjoint_source_t[..., 0]
        if adjoint_source_t.ndim == 3:
            # Convert recorded layout (B, nt, nrec) to source injection layout (B, nrec, nt).
            if adjoint_source_t.shape[1] == self.nt:
                adjoint_source_t = adjoint_source_t.transpose(1, 2)
        else:
            raise ValueError(
                "PropTorch impl='c' RTM expects adjoint_source with shape (B, nt, nrec[, 1]) "
                f"or (B, nrec, nt), got {tuple(adjoint_source_t.shape)}"
            )
        adjoint_source_t = adjoint_source_t.contiguous()

        models = models if models is not None else self.parameters()
        models = [EdgePadding.apply(para, padding) for para in models]
        self.models_padded = models

        lap_coes, grad_coes = self._build_fd_coefficients(M)

        models = [m[None, None, ...].repeat(batch_size, *([1] * (m.ndim + 1))) for m in self.models_padded]
        requires_model_grad = any(m.requires_grad for m in models)
        save_all_wavefields = bool(self.ndim == 2 or (requires_model_grad and not use_boundary_saving and not use_ckpt))
        self._ensure_wavefield_buffers(batch_size, persist_forward_state=save_all_wavefields, need_adjoint=requires_model_grad)
        self._ensure_adjoint_workspace_buffers(batch_size)
        if use_ckpt:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()), batch_size=batch_size)
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks, batch_size=batch_size)
        forward_wavefields, adjoint_wavefields = self._slice_wavefield_buffers(batch_size)
        adjoint_workspace = self._slice_adjoint_workspace_buffers(batch_size)
        checkpoint_buffers = self._slice_checkpoint_buffers(batch_size) if use_ckpt else ()

        if self.forward_wavefields:
            self.forward_allocator.zero_()
        if self.adjoint_wavefields:
            self.adjoint_allocator.zero_()
        if adjoint_workspace:
            self.workspace_allocator.zero_()
        if use_ckpt and checkpoint_buffers:
            self.checkpoint_allocator.zero_()
        if boundary_on_cpu:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers, boundary_dtype=boundary_cfg.get('storage_dtype'))
            if boundary_on_disk:
                self.boundary_cpu_allocator.zero_()
                self.boundary_gpu_allocator.zero_()
            boundary_cpu = self._slice_boundary_buffers(self.boundary_cpu, batch_size)
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu, batch_size)
        else:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers, boundary_dtype=boundary_cfg.get('storage_dtype'))
                for t in self.boundary_gpu_full:
                    t.zero_()
            boundary_cpu = ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu_full, batch_size)
        last_two = self._slice_last_two(batch_size) if use_boundary_saving else self.last_two
        spacing = self._cuda_spacing()

        _C = _get_C()
        fwd = _C.ForwardInput()
        fwd.wavefields = (
            list(forward_wavefields) if save_all_wavefields
            else list(self._transient_forward_wavefields(batch_size))
        )
        fwd.last_two = last_two
        if boundary_on_disk and use_boundary_saving:
            fwd.boundary_cpu = [b.zero_() for b in boundary_cpu]
            fwd.boundary_gpu = [b.zero_() for b in boundary_gpu]
        elif boundary_on_cpu and use_boundary_saving:
            fwd.boundary_cpu = list(boundary_cpu)
            fwd.boundary_gpu = list(boundary_gpu)
        else:
            fwd.boundary_cpu = []
            fwd.boundary_gpu = [b.zero_() for b in boundary_gpu] if use_boundary_saving else []
        fwd.boundary_disk_files = list(self._boundary_disk_files) if boundary_on_disk and use_boundary_saving else []
        fwd.checkpoints = [c.zero_() for c in checkpoint_buffers] if use_ckpt else []
        fwd.checkpoint_steps = checkpoint_steps if use_ckpt else torch.empty(0, dtype=torch.int32)
        fwd.models = [m.contiguous() for m in models]
        fwd.source = wavelet_t.contiguous()
        fwd.lap_coes = lap_coes.contiguous()
        fwd.grad_coes = grad_coes.contiguous()
        fwd.M = M
        fwd.abcn = self.abcn
        fwd.sources_loc = sources_t.contiguous()
        fwd.receivers_loc = receivers_t.contiguous()
        fwd.source_field_indices = self._field_indices_tensor(self.source_type, is_source=True)
        fwd.receiver_field_indices = self._field_indices_tensor(self.receiver_type, is_source=False)
        fwd.pml_vals = [p.to(self.dev).contiguous() for p in self.equation.b]
        fwd.save_all_wavefields = save_all_wavefields
        fwd.use_boundary_saving = use_boundary_saving
        fwd.use_checkpoint = use_ckpt
        fwd.use_recursive_checkpoint = use_recursive_checkpoint
        fwd.checkpoint_on_cpu = checkpoint_on_cpu
        fwd.boundary_on_cpu = boundary_on_cpu
        fwd.boundary_on_disk = boundary_on_disk
        fwd.boundary_disk_async_read = boundary_disk_async_read
        fwd.use_pinned_memory = use_pinned_memory
        fwd.free_surface = self._image_method_active
        fwd.fs_faces = self._fs_faces_c
        fwd.cut_face_mask = getattr(self, "_dd_cut_mask", 0)
        # Topography plumbing (image method).  Empty + has_topo=False for flat.
        topo_rows_rt = getattr(self.equation, "_topo_rows_runtime", None)
        if topo_rows_rt is not None:
            fwd.topo_rows = topo_rows_rt.to(torch.int32).contiguous()
            fwd.has_topo = True
        else:
            fwd.topo_rows = torch.empty(0, dtype=torch.int32, device=self.dev)
            fwd.has_topo = False
        # APM not supported on rtm path yet — empty stubs.
        fwd.topo_category = torch.empty(0, dtype=torch.int32, device=self.dev)
        fwd.use_apm = False
        fwd.nt = self.nt
        fwd.dt = self._dt
        fwd.spacing = spacing
        fwd.transfer_interval = transfer_interval
        fwd.boundary_ring_buffers = boundary_ring_buffers
        fwd.checkpoint_interval = self.ckpt_chunks
        fwd.checkpoint_count = int(checkpoint_steps.numel()) if use_recursive_checkpoint else self.ckpt_num

        u_forward, u_last_two, syn = self.forward_func(fwd)
        fwd.wavefields = []  # release per-call scratch before the backward half
        # Permute to canonical (B, nt, nrec, nfield) for the user-facing
        # output; RTM is currently single-channel acoustic-only so this
        # always lands as (B, nt, nrec, 1).
        syn = _cuda_record_to_canonical(syn)

        _C = _get_C()
        bwd = _C.BackwardInput()
        bwd.u_forward = u_forward.contiguous() if save_all_wavefields else torch.empty(0, device=self.dev)
        bwd.u_boundary = []
        bwd.u_last_two = u_last_two.contiguous() if use_boundary_saving else torch.empty(0, device=self.dev)
        bwd.checkpoints = list(checkpoint_buffers) if use_ckpt else []
        bwd.checkpoint_steps = checkpoint_steps.contiguous() if use_ckpt else torch.empty(0, dtype=torch.int32)
        bwd.adjoint_wavefields = [a.zero_() for a in adjoint_wavefields]
        bwd.forward_wavefields = []
        bwd.adjoint_workspace = list(adjoint_workspace)
        bwd.boundary_cpu = list(boundary_cpu) if boundary_on_cpu and use_boundary_saving else []
        bwd.boundary_gpu = list(boundary_gpu) if use_boundary_saving else []
        bwd.boundary_disk_files = list(self._boundary_disk_files) if boundary_on_disk and use_boundary_saving else []
        bwd.models = [m.contiguous() for m in models]
        bwd.adjoint_source = adjoint_source_t
        bwd.forward_source = wavelet_t.contiguous()
        bwd.lap_coes = lap_coes.contiguous()
        bwd.grad_coes = grad_coes.contiguous()
        bwd.M = M
        bwd.abcn = self.abcn
        bwd.adjoint_sources_loc = receivers_t.contiguous()
        bwd.forward_sources_loc = sources_t.contiguous()
        bwd.source_field_indices = self._field_indices_tensor(self.source_type, is_source=True)
        bwd.receiver_field_indices = self._field_indices_tensor(self.receiver_type, is_source=False)
        bwd.pml_vals = [p.to(self.dev).contiguous() for p in self.equation.b]
        bwd.nt = self.nt
        bwd.dt = self._dt
        bwd.spacing = spacing
        bwd.free_surface = self._image_method_active
        bwd.fs_faces = self._fs_faces_c
        bwd.cut_face_mask = getattr(self, "_dd_cut_mask", 0)
        topo_rows_rt = getattr(self.equation, "_topo_rows_runtime", None)
        if topo_rows_rt is not None:
            bwd.topo_rows = topo_rows_rt.to(torch.int32).contiguous()
            bwd.has_topo = True
        else:
            bwd.topo_rows = torch.empty(0, dtype=torch.int32, device=self.dev)
            bwd.has_topo = False
        # APM not supported on rtm path yet — empty stubs.
        bwd.topo_category = torch.empty(0, dtype=torch.int32, device=self.dev)
        bwd.use_apm = False
        bwd.boundary_on_cpu = boundary_on_cpu
        bwd.boundary_on_disk = boundary_on_disk
        bwd.boundary_disk_async_read = boundary_disk_async_read
        bwd.use_pinned_memory = use_pinned_memory
        bwd.checkpoint_on_cpu = checkpoint_on_cpu
        bwd.transfer_interval = transfer_interval
        bwd.boundary_ring_buffers = boundary_ring_buffers
        bwd.checkpoint_interval = self.ckpt_chunks
        bwd.checkpoint_count = int(checkpoint_steps.numel()) if use_recursive_checkpoint else self.ckpt_num

        image, source_illumination, receiver_illumination, adcig = self.rtm_func(bwd)
        if isinstance(adcig, torch.Tensor) and adcig.numel() > 0:
            self.adcig = adcig
        return syn, image, source_illumination, receiver_illumination

    def _build_recursive_checkpoint_steps(self, nt, checkpoint_count):
        checkpoint_count = int(max(0, checkpoint_count))
        if checkpoint_count == 0 or nt <= 1:
            return torch.empty(0, dtype=torch.int32)

        steps = np.linspace(1, nt - 1, num=checkpoint_count + 2, dtype=np.int32)[1:-1]
        steps = np.unique(steps)
        return torch.from_numpy(steps.astype(np.int32, copy=False))

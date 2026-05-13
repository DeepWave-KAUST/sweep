import os
import shutil
import tempfile

import torch
import torch.nn.functional as F

import numpy as np
from sweep.memory.torch import Allocator
from sweep.memory.shape import Layout
from sweep.propagator.base import PropBase
from sweep.utils.torch import EdgePadding
from sweep.scalars import fd_coefficients, staggered_grid_coes
from sweep.equations.base import FirstOrderEquation


def _get_C():
    import torch  # Ensure PyTorch is loaded before importing the compiled extension.
    import sweep._C as _C

    return _C

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
        params.nt = nt
        params.dt = dt
        params.spacing = spacing
        params.checkpoint_interval = checkpoint_interval
        params.checkpoint_count = checkpoint_count
        params.checkpoint_steps = checkpoint_steps if checkpoint_steps is not None else torch.empty(0, dtype=torch.int32)

        # -------- CUDA forward --------
        (u_allt, last, syn) = forward_func(params)
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
            ctx.forward_wavefields = forward_wavefields
            ctx.adjoint_wavefields = adjoint_wavefields
            ctx.adjoint_workspace = adjoint_workspace
            ctx.source_illumination_buffer = source_illumination_buffer
            ctx.receiver_illumination_buffer = receiver_illumination_buffer
            ctx.illumination_padding = tuple(illumination_padding)

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

        _C = _get_C()
        params = _C.BackwardInput()
        # common
        params.transfer_interval = ctx.transfer_interval
        params.boundary_ring_buffers = ctx.boundary_ring_buffers
        params.checkpoint_interval = ctx.checkpoint_interval
        params.checkpoint_count = ctx.checkpoint_count
        params.adjoint_wavefields = [a.zero_() for a in ctx.adjoint_wavefields]
        params.adjoint_workspace = list(ctx.adjoint_workspace)
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
            *model_grads # models
        )

class _CompiledPropagator(PropBase, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        kwargs.setdefault("use_ckpt", False)
        super().__init__(*args, **kwargs)
        
        self.register_buffer('dt', torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer('dh', torch.tensor(self._grid_spacing, device=self.dev, dtype=torch.float32))

        funcs = self.equation._C()
        self.forward_func = funcs[0]
        self.backward_func = funcs[1]
        self.backward_bs_func = funcs[2]
        self.backward_ckpt_func = funcs[3] if len(funcs) > 3 else None
        self.backward_recursive_ckpt_func = funcs[4] if len(funcs) > 4 else None
        self.rtm_func = self.equation._C_rtm() if hasattr(self.equation, "_C_rtm") else None

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
        if self._boundary_disk_root is not None:
            shutil.rmtree(self._boundary_disk_root, ignore_errors=True)
        self._boundary_disk_root = None
        self._boundary_disk_files = ()

    def _allocate_boundary_disk_files(self, shapes, disk_dir):
        root = tempfile.mkdtemp(prefix="sweep_boundary_", dir=disk_dir)
        files = []
        for idx, shape in enumerate(shapes):
            path = os.path.join(root, f"boundary_{idx}.bin")
            numel = int(torch.Size(shape).numel())
            with open(path, "wb") as handle:
                handle.truncate(numel * torch.empty((), dtype=torch.float32).element_size())
            files.append(path)
        self._boundary_disk_root = root
        self._boundary_disk_files = tuple(files)

    def _ensure_boundary_buffers(self, boundary_storage, transfer_interval, use_pinned_memory, disk_dir=None, ring_buffers=1):
        boundary_on_cpu = boundary_storage in {"cpu", "disk"}
        staging_pinned = use_pinned_memory or boundary_storage == "disk"
        staging_interval = transfer_interval * ring_buffers
        if (
            self._boundary_cache_batch == self.B
            and
            self._boundary_cache_mode == boundary_storage
            and self._boundary_cache_interval == transfer_interval
            and self._boundary_cache_ring_buffers == ring_buffers
            and self._boundary_cache_pinned == staging_pinned
            and self._boundary_cache_disk_dir == disk_dir
            and self._boundary_cache_nt == self.nt
        ):
            return

        cuda_layout = self._cuda_layout()
        layout = Layout(
            self.shape_cuda,
            cuda_layout.base_nvar,
            self.nt,
            self.abcn,
            self.equation.so // 2,
            self.B,
            staging_interval if boundary_on_cpu else transfer_interval,
            self.free_surface,
            self.equation.so // 2 + 1,
            tangent_pad=cuda_layout.boundary_tangent_pad,
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

        if boundary_on_cpu:
            if boundary_storage == "disk":
                self._allocate_boundary_disk_files(layout.cpu_shapes, disk_dir)
                self.boundary_cpu = self.boundary_cpu_allocator.zeros(
                    layout.gpu_shapes,
                    dtype=torch.float32,
                    dev='cpu',
                    pin_memory=staging_pinned,
                )
            else:
                self.boundary_cpu = self.boundary_cpu_allocator.zeros(
                    layout.cpu_shapes,
                    dtype=torch.float32,
                    dev='cpu',
                    pin_memory=staging_pinned,
                )
            self.boundary_gpu = self.boundary_gpu_allocator.zeros(layout.gpu_shapes)
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
            self.boundary_gpu_full = self.boundary_gpu_allocator.zeros(layout.gpu_full_shapes)
            self.last_two = self.forward_allocator.zeros([last_two_shape])[0]

        self._boundary_cache_mode = boundary_storage
        self._boundary_cache_interval = transfer_interval
        self._boundary_cache_ring_buffers = ring_buffers
        self._boundary_cache_pinned = staging_pinned
        self._boundary_cache_disk_dir = disk_dir
        self._boundary_cache_nt = self.nt
        self._boundary_cache_batch = self.B

    def __del__(self):
        self._remove_boundary_disk_cache()

    def _ensure_wavefield_buffers(self, batch_size, need_forward=False, need_adjoint=True):
        current_capacity = self._buffer_capacity_batch
        if (
            current_capacity is not None
            and batch_size <= current_capacity
            and (not need_forward or self.forward_wavefields)
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
        wavefield_shapes = total_wavefields * [[self.B, 1, *self.shape_cuda]]
        if batch_size > (current_capacity or 0):
            self.forward_allocator = Allocator(self.dev)
            self.adjoint_allocator = Allocator(self.dev)
            self.forward_wavefields = ()
            self.adjoint_wavefields = ()

        if need_forward and not self.forward_wavefields:
            self.forward_wavefields = self.forward_allocator.zeros(wavefield_shapes)
        if need_adjoint and not self.adjoint_wavefields:
            self.adjoint_wavefields = self.adjoint_allocator.zeros(wavefield_shapes)
        self._buffer_capacity_batch = target_capacity
        self._boundary_cache_batch = None
        self._boundary_cache_ring_buffers = None
        self._checkpoint_cache_batch = None

    def _slice_wavefield_buffers(self, batch_size):
        return tuple(t[:batch_size] for t in self.forward_wavefields), tuple(t[:batch_size] for t in self.adjoint_wavefields)

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

    def _ensure_checkpoint_buffers(self, checkpoint_interval=None, checkpoint_count=None):
        if not self.use_ckpt or (self.backward_ckpt_func is None and self.backward_recursive_ckpt_func is None):
            return

        if checkpoint_count is not None:
            n_checkpoints = int(checkpoint_count)
        else:
            n_checkpoints = max(1, (self.nt + checkpoint_interval - 1) // checkpoint_interval)

        checkpoint_storage = self.ckpt_storage
        checkpoint_pinned = self.ckpt_pinned_memory if checkpoint_storage == "cpu" else False

        if (
            self._checkpoint_cache_batch == self.B
            and self._checkpoint_cache_interval == checkpoint_interval
            and self._checkpoint_cache_count == n_checkpoints
            and self._checkpoint_cache_nt == self.nt
            and self._checkpoint_cache_storage == checkpoint_storage
            and self._checkpoint_cache_pinned == checkpoint_pinned
        ):
            return

        checkpoint_shape = [n_checkpoints, self.B, 1, *self.shape_cuda]
        cuda_layout = self._cuda_layout()
        num_checkpoint_tensors = int(cuda_layout.resolved_checkpoint_nvar())
        checkpoint_device = "cpu" if checkpoint_storage == "cpu" else self.dev
        self.checkpoint_allocator = Allocator(checkpoint_device)
        self.checkpoints = tuple(
            self.checkpoint_allocator.zeros(
                [checkpoint_shape] * num_checkpoint_tensors,
                dtype=torch.float32,
                dev=checkpoint_device,
                pin_memory=checkpoint_pinned,
            )
        )
        self._checkpoint_cache_batch = self.B
        self._checkpoint_cache_interval = checkpoint_interval
        self._checkpoint_cache_count = n_checkpoints
        self._checkpoint_cache_nt = self.nt
        self._checkpoint_cache_storage = checkpoint_storage
        self._checkpoint_cache_pinned = checkpoint_pinned

    def _slice_checkpoint_buffers(self, batch_size):
        if not self.checkpoints:
            return ()
        return tuple(t[:, :batch_size] for t in self.checkpoints)

    @torch._dynamo.disable
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
        if "boundary_ring_buffers" in kwargs:
            legacy_override["ring_buffers"] = kwargs.pop("boundary_ring_buffers")
        if "boundary_disk_async_read" in kwargs:
            legacy_override["disk_async_read"] = kwargs.pop("boundary_disk_async_read")
        if boundary_saving_config is None and legacy_override:
            boundary_saving_config = legacy_override
        elif legacy_override:
            boundary_saving_config = {**legacy_override, **boundary_saving_config}

        source_encoding = bool(source_encoding or self._auto_detect_source_encoding(wavelet, sources, receivers))

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
        nshots = sources.shape[0]
        batch_size = 1 if source_encoding else nshots
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
        if isinstance(wavelet, torch.Tensor):
            wavelet = wavelet.to(self.dev, dtype=torch.float32)
        else:
            wavelet = torch.from_numpy(wavelet).to(self.dev).float()
        if wavelet.ndim == 1:
            wavelet = wavelet[None, None, :].repeat(batch_size, 1, 1)
        elif wavelet.ndim == 2:
            if wavelet.shape[0] != batch_size:
                raise ValueError(
                    f"Expected wavelet batch dimension {batch_size}, got {wavelet.shape[0]}"
                )
            wavelet = wavelet[:, None, :]
        elif wavelet.ndim == 3:
            if wavelet.shape[0] != batch_size:
                raise ValueError(
                    f"Expected wavelet batch dimension {batch_size}, got {wavelet.shape[0]}"
                )
        else:
            raise ValueError(
                f"wavelet must have shape (nt,), (B, nt), or (B, nsrc, nt), got {tuple(wavelet.shape)}"
            )

        sources = torch.from_numpy(sources).to(self.dev).int()
        if sources.ndim == 2:
            if sources.shape[0] != batch_size:
                raise ValueError(
                    f"Expected sources batch dimension {batch_size}, got {sources.shape[0]}"
                )
            sources = sources[:, None, :]
        elif sources.ndim != 3:
            raise ValueError(
                f"sources must have shape (B, dim) or (B, nsrc, dim), got {tuple(sources.shape)}"
            )
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

        models = [m[None, None, ...].repeat(batch_size, *([1]*(m.ndim+1))) for m in self.models_padded]
        requires_model_grad = any(m.requires_grad for m in models)
        requires_wavelet_grad = wavelet.requires_grad
        requires_backward = bool(requires_model_grad or requires_wavelet_grad)
        if requires_backward:
            self.source_illumination = torch.zeros_like(unpadded_models[0])
            self.receiver_illumination = torch.zeros_like(unpadded_models[0])
        else:
            self.source_illumination = None
            self.receiver_illumination = None
        use_checkpoint = bool(self.use_ckpt and requires_backward)
        if not requires_backward:
            use_boundary_saving = False
        use_recursive_checkpoint = bool(
            use_checkpoint and self.ckpt_mode == "recursive" and self.backward_recursive_ckpt_func is not None
        )
        checkpoint_on_cpu = bool(use_checkpoint and self.ckpt_storage == "cpu")
        save_all_wavefields = bool(requires_backward and not use_boundary_saving and not use_checkpoint)
        self._ensure_wavefield_buffers(batch_size, need_forward=save_all_wavefields, need_adjoint=requires_backward)
        self._ensure_adjoint_workspace_buffers(batch_size)
        if use_checkpoint:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()))
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks)
        forward_wavefields, adjoint_wavefields = self._slice_wavefield_buffers(batch_size)
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
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers)
            if boundary_on_disk:
                self.boundary_cpu_allocator.zero_()
                self.boundary_gpu_allocator.zero_()
            boundary_cpu = self._slice_boundary_buffers(self.boundary_cpu, batch_size)
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu, batch_size)
        else:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers)
                for t in self.boundary_gpu_full:
                    t.zero_()
            boundary_cpu = ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu_full, batch_size)
        last_two = self._slice_last_two(batch_size) if use_boundary_saving else self.last_two

        spacing = self._cuda_spacing()

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
                self.equation.b,
                use_checkpoint,
                self.ckpt_chunks,
                use_recursive_checkpoint,
                int(checkpoint_steps.numel()),
                checkpoint_steps,
                checkpoint_on_cpu,
                use_boundary_saving,
                use_pinned_memory,
                self.free_surface,
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
                *models,
            )
        
        return syn

    def rtm(self, wavelet, sources, receivers, adjoint_source, models=None, **kwargs):
        if self.rtm_func is None:
            raise NotImplementedError(f"RTM is not implemented for {self.equation.__class__.__name__}.")

        use_boundary_saving = kwargs.pop("use_boundary_saving", None)
        boundary_saving_config = kwargs.pop("boundary_saving_config", None)
        self.nt = wavelet.shape[-1]
        nshots = sources.shape[0]
        batch_size = nshots
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

        if self.ndim == 2 and (use_boundary_saving or self.use_ckpt):
            raise NotImplementedError(
                "PropTorch impl='c' RTM for 2D acoustic currently supports full-wavefield mode only; "
                "disable boundary saving and checkpointing."
            )

        use_recursive_checkpoint = bool(self.use_ckpt and self.ckpt_mode == "recursive" and self.backward_recursive_ckpt_func is not None)
        checkpoint_on_cpu = bool(self.use_ckpt and self.ckpt_storage == "cpu")
        if self.use_ckpt and self.ckpt_mode not in {"chunk", "recursive"}:
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}'. Expected 'chunk' or 'recursive'.")
        checkpoint_steps = torch.empty(0, dtype=torch.int32)
        if self.use_ckpt:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()))
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks)

        self.forward_allocator.zero_()
        checkpoint_buffers = self._slice_checkpoint_buffers(batch_size) if self.use_ckpt else ()
        if self.use_ckpt and checkpoint_buffers:
            self.checkpoint_allocator.zero_()
        if use_boundary_saving:
            self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers)
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
        save_all_wavefields = not use_boundary_saving and not self.use_ckpt

        M = self.equation.so // 2
        padding = [p + M for p in self.padding]
        base_shift = M + self.abcn
        shape_for_pml = [p + 2 * M for p in self.shape]
        kwargs["shape"] = shape_for_pml
        self.init_abc(**kwargs)

        sources = sources.copy()
        receivers = receivers.copy()
        if self.free_surface:
            sources[..., 0] += base_shift
            receivers[..., 0] += base_shift
            if self.ndim == 3:
                sources[..., 1] += base_shift
                receivers[..., 1] += base_shift
            sources[..., -1] += M
            receivers[..., -1] += M
        else:
            sources += base_shift
            receivers += base_shift

        wavelet_t = torch.from_numpy(wavelet).to(self.dev).float()[None, None, :].repeat(batch_size, 1, 1)
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
        save_all_wavefields = bool(self.ndim == 2 or (requires_model_grad and not use_boundary_saving and not self.use_ckpt))
        self._ensure_wavefield_buffers(batch_size, need_forward=save_all_wavefields, need_adjoint=requires_model_grad)
        self._ensure_adjoint_workspace_buffers(batch_size)
        if self.use_ckpt:
            if use_recursive_checkpoint:
                checkpoint_steps = self._build_recursive_checkpoint_steps(self.nt, self.ckpt_num)
                self._ensure_checkpoint_buffers(checkpoint_count=int(checkpoint_steps.numel()))
            elif self.backward_ckpt_func is not None:
                self._ensure_checkpoint_buffers(checkpoint_interval=self.ckpt_chunks)
        forward_wavefields, adjoint_wavefields = self._slice_wavefield_buffers(batch_size)
        adjoint_workspace = self._slice_adjoint_workspace_buffers(batch_size)
        checkpoint_buffers = self._slice_checkpoint_buffers(batch_size) if self.use_ckpt else ()

        if self.forward_wavefields:
            self.forward_allocator.zero_()
        if self.adjoint_wavefields:
            self.adjoint_allocator.zero_()
        if adjoint_workspace:
            self.workspace_allocator.zero_()
        if self.use_ckpt and checkpoint_buffers:
            self.checkpoint_allocator.zero_()
        if boundary_on_cpu:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers)
            if boundary_on_disk:
                self.boundary_cpu_allocator.zero_()
                self.boundary_gpu_allocator.zero_()
            boundary_cpu = self._slice_boundary_buffers(self.boundary_cpu, batch_size)
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu, batch_size)
        else:
            if use_boundary_saving:
                self._ensure_boundary_buffers(boundary_storage, transfer_interval, use_pinned_memory, boundary_disk_dir, boundary_ring_buffers)
                for t in self.boundary_gpu_full:
                    t.zero_()
            boundary_cpu = ()
            boundary_gpu = self._slice_boundary_buffers(self.boundary_gpu_full, batch_size)
        last_two = self._slice_last_two(batch_size) if use_boundary_saving else self.last_two
        spacing = self._cuda_spacing()

        _C = _get_C()
        fwd = _C.ForwardInput()
        fwd.wavefields = list(forward_wavefields) if save_all_wavefields else []
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
        fwd.checkpoints = [c.zero_() for c in checkpoint_buffers] if self.use_ckpt else []
        fwd.checkpoint_steps = checkpoint_steps if self.use_ckpt else torch.empty(0, dtype=torch.int32)
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
        fwd.pml_vals = [p.contiguous() for p in self.equation.b]
        fwd.save_all_wavefields = save_all_wavefields
        fwd.use_boundary_saving = use_boundary_saving
        fwd.use_checkpoint = self.use_ckpt
        fwd.use_recursive_checkpoint = use_recursive_checkpoint
        fwd.checkpoint_on_cpu = checkpoint_on_cpu
        fwd.boundary_on_cpu = boundary_on_cpu
        fwd.boundary_on_disk = boundary_on_disk
        fwd.boundary_disk_async_read = boundary_disk_async_read
        fwd.use_pinned_memory = use_pinned_memory
        fwd.free_surface = self.free_surface
        fwd.nt = self.nt
        fwd.dt = self._dt
        fwd.spacing = spacing
        fwd.transfer_interval = transfer_interval
        fwd.boundary_ring_buffers = boundary_ring_buffers
        fwd.checkpoint_interval = self.ckpt_chunks
        fwd.checkpoint_count = int(checkpoint_steps.numel()) if use_recursive_checkpoint else self.ckpt_num

        u_forward, u_last_two, syn = self.forward_func(fwd)

        _C = _get_C()
        bwd = _C.BackwardInput()
        bwd.u_forward = u_forward.contiguous() if save_all_wavefields else torch.empty(0, device=self.dev)
        bwd.u_boundary = []
        bwd.u_last_two = u_last_two.contiguous() if use_boundary_saving else torch.empty(0, device=self.dev)
        bwd.checkpoints = list(checkpoint_buffers) if self.use_ckpt else []
        bwd.checkpoint_steps = checkpoint_steps.contiguous() if self.use_ckpt else torch.empty(0, dtype=torch.int32)
        bwd.adjoint_wavefields = [a.zero_() for a in adjoint_wavefields]
        bwd.forward_wavefields = []
        bwd.adjoint_workspace = []
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
        bwd.pml_vals = [p.contiguous() for p in self.equation.b]
        bwd.nt = self.nt
        bwd.dt = self._dt
        bwd.spacing = spacing
        bwd.free_surface = self.free_surface
        bwd.boundary_on_cpu = boundary_on_cpu
        bwd.boundary_on_disk = boundary_on_disk
        bwd.boundary_disk_async_read = boundary_disk_async_read
        bwd.use_pinned_memory = use_pinned_memory
        bwd.checkpoint_on_cpu = checkpoint_on_cpu
        bwd.transfer_interval = transfer_interval
        bwd.boundary_ring_buffers = boundary_ring_buffers
        bwd.checkpoint_interval = self.ckpt_chunks
        bwd.checkpoint_count = int(checkpoint_steps.numel()) if use_recursive_checkpoint else self.ckpt_num

        image, source_illumination, receiver_illumination = self.rtm_func(bwd)
        return syn, image, source_illumination, receiver_illumination

    def _build_recursive_checkpoint_steps(self, nt, checkpoint_count):
        checkpoint_count = int(max(0, checkpoint_count))
        if checkpoint_count == 0 or nt <= 1:
            return torch.empty(0, dtype=torch.int32)

        steps = np.linspace(1, nt - 1, num=checkpoint_count + 2, dtype=np.int32)[1:-1]
        steps = np.unique(steps)
        return torch.from_numpy(steps.astype(np.int32, copy=False))

"""One object for the compiled call's non-differentiable arguments.

``Wrapper`` is a ``torch.autograd.Function``, and autograd's contract is
positional: ``backward`` must return exactly one gradient per ``forward`` input,
in order. That contract had been paid literally -- 51 positional parameters, and
a hand-maintained wall of 48 ``None``s in ``backward`` that only stays correct if
every future edit inserts its ``None`` in the matching slot.

Only two of those inputs are differentiable: ``wavelet`` and the trailing
``*models``. Both STAY positional, because autograd can only see a tensor it
receives as an input -- a tensor hidden inside this object would silently get no
gradient. Everything else is configuration, buffers, or bound C functions, all of
which already returned ``None``; moving them behind one argument changes nothing
about what autograd computes and collapses the signature to
``forward(ctx, p, wavelet, *models)`` and the return to
``(None, wavelet_grad, *model_grads)``.

The object is READ-ONLY to ``forward``. Five parameters are refined inside the
body (``spacing``, ``dt``, and the three memory-strategy flags, which are turned
off when no gradient is required); those are bound to locals at the top, so the
refinement is visible to the reader and never written back to a caller's object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CompiledCallParams:
    """Everything ``Wrapper.forward`` needs that autograd does not differentiate.

    Field order is irrelevant to correctness here -- that is the entire point of
    the change. Adding one means adding a field, not also counting ``None``s.
    """
    forward_func: Any
    backward_func: Any
    backward_bs_func: Any
    backward_ckpt_func: Any
    backward_recursive_ckpt_func: Any
    sources_loc: Any   # (B, nsrc, 2)
    receivers_loc: Any   # (B, nrec, 2)
    source_field_indices: Any
    receiver_field_indices: Any
    coes_list: Any
    M: int
    abcn: int
    spacing: list   # list of floats for grid spacing
    dt: float
    pml_vals: list   # list of 6 tensors for PML profiles
    use_checkpoint: bool = False
    checkpoint_interval: int = 1
    use_recursive_checkpoint: bool = False
    checkpoint_count: int = 0
    checkpoint_steps: torch.Tensor = None
    checkpoint_on_cpu: bool = False
    use_boundary_saving: bool = False
    use_pinned_memory: bool = False
    free_surface: bool = False
    transfer_interval: int = 1
    boundary_ring_buffers: int = 1
    boundary_on_cpu: bool = False
    boundary_on_disk: bool = False
    boundary_disk_async_read: bool = False
    boundary_tail_steps: int = 0
    forward_wavefields: tuple = ()
    adjoint_wavefields: tuple = ()
    adjoint_workspace: tuple = ()
    checkpoint_buffers: tuple = ()
    last_two: torch.Tensor = None
    boundary_cpu: tuple = ()
    boundary_gpu: tuple = ()
    boundary_disk_files: tuple = ()
    source_illumination_buffer: torch.Tensor = None
    receiver_illumination_buffer: torch.Tensor = None
    illumination_padding: tuple = ()
    adcig_buffer: torch.Tensor = None   # (nlag, nz, nx[, ny]) model-shaped
    adcig_max_lag: int = 0
    topo_rows_param: torch.Tensor = None   # runtime padded surface row per col
    has_topo_param: bool = False
    topo_category_param: torch.Tensor = None   # runtime padded APM category int32
    use_apm_param: bool = False
    fs_faces: int = -1   # per-edge free-surface bitmask (-1 => legacy z-min)
    cut_face_mask: int = 0   # DD cut faces (0 => single domain)
    # Equation-specific aux tensors, opaque to the wrapper (e.g. the
    # visco-acoustic |k| grid).  Constants -- no grad flows through them.
    eq_aux: tuple = ()

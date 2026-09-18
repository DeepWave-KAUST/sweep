from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                       # avoids a slot_table <-> cuda_layout cycle
    from .slot_table import SlotTable


@dataclass(frozen=True)
class CUDALayoutSpec:
    """CUDA runtime metadata for compiled propagator buffer layout."""

    base_nvar: int
    pml_nvar: int
    last_two_nvar: int
    last_two_storage_nvar: int | None = None
    checkpoint_nvar: int | None = None
    backward_workspace_nvar: int = 0
    # ``fn(B, nt, shape_cuda, mode) -> list of shapes`` for equations whose
    # backward scratch is not simply N padded grids -- ``mode`` is the memory
    # mode the propagator runs ("full", "bs", "ckpt", "recursive"), so a
    # driver whose modes keep different scratch alive gets a pool sized for
    # the mode in use, never the union.
    backward_workspace_shapes: Callable | None = None
    # Per-call scratch the compiled FORWARD takes from the propagator
    # (``ForwardInput.forward_workspace``): one padded grid per shot each,
    # allocated for the duration of one call, never re-zeroed -- a driver
    # zeroes what it needs. 0 = the forward allocates nothing of its own.
    forward_workspace_nvar: int = 0
    # Model-shaped slots the propagator hands the compiled forward AND backward
    # as ``derived_models``: one per coefficient the driver derives from the
    # bound models (Lame parameters, VTI stiffness, 1/z -- the slot order is
    # the enum in csrc/cuda/common/derived_models.h). Allocated uninitialised
    # per call (the driver's kernel writes every cell) and dropped with the
    # call. An int, or ``fn(mode) -> int`` with ``mode`` = "forward" or the
    # backward's memory mode ("full"/"bs"/"ckpt"/"recursive") for a driver
    # that derives only in some modes, so no unused slot is ever allocated.
    # 0 = the driver derives nothing.
    derived_model_nvar: int | Callable = 0
    # ``fn(B, nrec, nfield, nt) -> tuple``: the record the compiled forward
    # writes, in the driver's own layout (``record_single`` for the
    # ``(B, nrec, nt)`` acoustic family, ``record_multi`` for the
    # ``(nfield, B, nrec, nt)`` staggered family); the propagator allocates it
    # per call and binds it as ``record_out``. None = the forward allocates.
    record_shape: Callable | None = None
    # ``fn(B, nt, shape_cuda, max_segment, mode) -> list of shapes``: the replay
    # buffers a checkpoint-mode backward keeps (the recomputed forward of one
    # segment, ``max_segment`` steps long -- the chunk length, or the longest
    # recursive segment; ``mode`` is "ckpt" or "recursive", for a driver whose
    # two modes keep different histories). Allocated with the checkpoint
    # snapshots, never re-zeroed: a driver writes every row it reads. None =
    # the compiled backward allocates its own.
    checkpoint_replay_shapes: Callable | None = None
    # The replay STATE a checkpoint-mode backward steps (the forward's state
    # struct: its ``base_nvar`` physical fields and the CPML memory it
    # checkpoints, in bind order -- the forward slot list without the psi
    # double-buffer shadows), handed over per backward call as
    # ``forward_wavefields``. None = derive that count from the forward slots;
    # 0 = the driver keeps its replay state elsewhere (LSRTM: workspace slots).
    checkpoint_state_nvar: int | None = None
    # The recursive-checkpoint backward bisects each segment and keeps one
    # scratch state set per recursion level (the acoustic skeleton): the
    # propagator then hands ``1 + depth(max_segment)`` state sets instead of one.
    recursive_state_depth: bool = False
    # Shape of the full-mode forward history (``u_allt``) the propagator
    # allocates per call and hands the compiled forward as
    # ``ForwardInput.u_allt_out``: ``fn(B, nt, shape_cuda) -> tuple``, in the
    # driver's own layout (see ``history_fields`` / ``history_plain``). None =
    # the compiled forward allocates its own (only equations without a
    # compiled full mode are left there).
    save_all_shape: Callable | None = None
    # Extra wavefield buffers allocated for the ADJOINT only (not the forward).
    # The fused single-kernel adjoint double-buffers zeta (the forward already
    # double-buffers psi via pml_nvar): adjoint gets base+pml+adjoint_extra
    # tensors, forward stays base+pml.  0 = equation has no fused-adjoint path.
    adjoint_extra_nvar: int = 0
    boundary_tangent_pad: int = 0
    boundary_save_nvar: int | None = None
    # Padded per-shot grids the boundary-saving backward is handed as its
    # reconstruction state (``BackwardInput.forward_wavefields``: the forward's
    # physical fields stepped backwards from ``u_last_two``, plus any carrier
    # the imaging reads), allocated zeroed per backward call. None = derive it
    # from ``slots.recon`` (the DD path's source of the same count); 0 = the
    # driver has no boundary-saving reconstruction. See ``reconstruction_nvar``.
    bs_reconstruction_nvar: int | None = None
    # CPML aux strip (slab) storage.  ``pml_slot_axes`` tags each of the
    # pml_nvar FORWARD slots with its differencing axis ('x'/'y'/'z') in the
    # C++ bind order; the runtime then allocates those slots as per-axis
    # slabs (band + stencil reach) instead of full grids.  The kernels adapt
    # per bound tensor, so ``None`` (default) keeps full-domain allocation.
    # ``checkpoint_slot_axes`` does the same for the checkpoint snapshot
    # slots (``None`` entries are physical, full-grid slots).  The ADJOINT
    # aux stays full-domain unless ``adjoint_pml_slab`` is set (acoustic's
    # fused adjoint stencil-taps psi/zeta and is kept on the legacy layout;
    # elastic memory variables are own-cell only and can opt in).
    pml_slot_axes: tuple | None = None
    checkpoint_slot_axes: tuple | None = None
    adjoint_pml_slab: bool = False
    # The compiled backward_bs honours nt_saved < nt (truncated boundary
    # backward for steady-state / frequency-selection FWI).  Declared here so
    # the driver refuses tail_steps by declaration instead of by class name.
    supports_boundary_tail_steps: bool = False
    # Declarative CUDA wavefield bind order (see ``slot_table.py``).  Optional:
    # equations without one keep the legacy path, where every count above is
    # declared by hand and the matching index tuples live in
    # ``propagator/_stepped.py`` and ``parallel/dd_propagator.py``.  Where it IS
    # declared, ``test_slot_table_consistency.py`` requires the derived values to
    # equal the hand-written ones, so the two cannot drift while both exist.
    slots: "SlotTable | None" = None
    # ---- Domain decomposition ------------------------------------------
    # The compiled forward AND backward_bs honour the stepped it_begin/it_end
    # range (propagation state persists across Python-driven segments) -- the
    # prerequisite for domain decomposition.  Declared here so ModelParallel
    # admits by declaration instead of by the class-name whitelist it used to
    # keep.  An equation without it would not raise under DD -- it would run
    # the full record on every stepped call and return zeros -- so the flag
    # must only be set once the kernels' drivers are actually stepped (the
    # shared template drivers in csrc/cuda/common are).
    stepped: bool = False

    @property
    def reconstruction_nvar(self) -> int:
        """Reconstruction grids the boundary-saving backward binds: the explicit
        ``bs_reconstruction_nvar``, else the slot table's ``recon`` list."""
        if self.bs_reconstruction_nvar is not None:
            return int(self.bs_reconstruction_nvar)
        return int(self.slots.nrecon) if self.slots is not None else 0
    # The compiled backward implements the NUMBERED backward phases its DD
    # schedule drives (the elastic physics split, the VRZ coupling exchange).
    # The plain acoustic schedule phases nothing and ignores this flag.  The
    # 2-D AcousticVRZ is the live gap: stepped, but its backward lacks the
    # coupling-exchange phases the 3-D sibling implements.
    dd_backward_phases: bool = False
    # Some equations' model gradient is a spatial DIVERGENCE of an intermediate
    # coupling field rather than a pointwise product, so at a cut seam it needs
    # the neighbour's coupling values -- the DD backward has to build the field,
    # exchange it, and only then take the divergence.  Variable-density VRZ is
    # the case in the tree (c/e = lambda*vp*grad(p)); plain acoustic's
    # u_tt*lambda needs no such exchange.
    #
    # Declared here rather than sniffed from the class name, which is what the
    # DD driver used to do ("vrz" in type(equation).__name__.lower()) and which
    # silently misses a subclass named anything else.
    # Output-binding facts about the compiled backward, which the DD driver used
    # to infer from the equation family:
    #   grads_out.size() == models.size() + 1, slot 0 = grad_wavelet
    #     (acoustic2d/backward.cu:101, acoustic3d:125, acoustic_vrz3d:329)
    #   vs grads_out.size() == models.size()
    #     (elastic2d/backward.cu:306, elastic3d:528)
    grads_out_has_wavelet: bool = False
    # illum_out.size() == 2 (acoustic2d/backward.cu:104, acoustic3d:128) vs
    # illum_out.empty() (elastic2d/backward.cu:310, elastic3d:532).
    #
    # NOTE acoustic_vrz3d declares 2 to reproduce today's behaviour, not because
    # it needs them: its backward.cu contains no reference to illum_out at all,
    # so the driver has been allocating two model-sized buffers that nothing
    # reads. Dropping it to 0 is a real (numerically inert) saving, but it is a
    # separate change with its own gate rather than a free rider here.
    illum_nvar: int = 0
    dd_coupling_nvar: int = 0
    # Model-only adjoint coefficients the fused adjoint reads across the cut
    # (constant within a backward, so built and exchanged once before the
    # reverse loop rather than per step).
    dd_adjoint_coeff_nvar: int = 0

    def resolved_last_two_storage_nvar(self) -> int:
        return self.base_nvar if self.last_two_storage_nvar is None else self.last_two_storage_nvar

    def resolved_checkpoint_nvar(self) -> int:
        default = self.base_nvar + self.pml_nvar
        return default if self.checkpoint_nvar is None else self.checkpoint_nvar

    def resolved_boundary_save_nvar(self) -> int:
        """Number of distinct fields the boundary saver writes per timestep.

        Defaults to ``base_nvar`` for backward compatibility, but second-order
        time-stepping schemes (acoustic, acoustic_vrz, acoustic_lsrtm) only
        write the primary pressure field to the boundary buffer — they should
        set ``boundary_save_nvar=1`` to avoid over-allocating the GPU-side
        boundary tensors by a factor of ``base_nvar``.
        """
        return self.base_nvar if self.boundary_save_nvar is None else self.boundary_save_nvar


def history_fields(n: int):
    """``save_all_shape`` for the ``(nt, n, B, *grid)`` history layout: ``n``
    physical fields per time step, batch inside."""
    def shape(B, nt, grid):
        return (nt, n, B, *grid)
    return shape


def history_plain():
    """``save_all_shape`` for the acoustic-family ``(nt, B, *grid)`` layout: one
    field per step and no field axis."""
    def shape(B, nt, grid):
        return (nt, B, *grid)
    return shape


def record_single():
    """``record_shape`` for the ``(B, nrec, nt)`` record of a single-field driver."""
    def shape(B, nrec, nfield, nt):
        return (B, nrec, nt)
    return shape


def record_multi():
    """``record_shape`` for the ``(nfield, B, nrec, nt)`` record of a multi-field
    driver (one receiver field per leading index, even when there is one)."""
    def shape(B, nrec, nfield, nt):
        return (nfield, B, nrec, nt)
    return shape

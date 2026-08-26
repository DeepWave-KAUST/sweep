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
    backward_workspace_shapes: Callable | None = None
    # Extra wavefield buffers allocated for the ADJOINT only (not the forward).
    # The fused single-kernel adjoint double-buffers zeta (the forward already
    # double-buffers psi via pml_nvar): adjoint gets base+pml+adjoint_extra
    # tensors, forward stays base+pml.  0 = equation has no fused-adjoint path.
    adjoint_extra_nvar: int = 0
    boundary_tangent_pad: int = 0
    boundary_save_nvar: int | None = None
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
    # Declarative CUDA wavefield bind order (see ``slot_table.py``).  Optional:
    # equations without one keep the legacy path, where every count above is
    # declared by hand and the matching index tuples live in
    # ``propagator/_stepped.py`` and ``parallel/dd_propagator.py``.  Where it IS
    # declared, ``test_slot_table_consistency.py`` requires the derived values to
    # equal the hand-written ones, so the two cannot drift while both exist.
    slots: "SlotTable | None" = None
    # ---- Domain decomposition ------------------------------------------
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

"""Acoustic 2-D second-order wave equation on a boundary-fitted
curvilinear grid (irregular free-surface topography).

Companion to ``Acoustic`` for the case where the surface is non-flat.
The PDE on the physical (X, Z) domain is mapped onto a rectangular
computational (ξ, η) domain via the simple linear stretch
``Z = z_top(ξ) + D(ξ)·η``; see :class:`sweep.utils.curvilinear.CurvilinearGrid`
for the metric derivation.

Strategy:
  - Reuse the existing ``separable_d2_2d`` and ``gradient`` primitives on
    the computational grid — they are direction-independent FD ops and
    don't care that ``η`` is dimensionless.
  - Combine ``p_ξξ`` / ``p_ηη`` / mixed ``p_ξη`` / first-order ``p_η``
    with the precomputed metric coefficients to build the physical
    Laplacian.
  - Split the result into a ξ-half and an η-half that feed the existing
    :func:`step_cpml` CPML wrapper.
  - Free-surface BC at the **flat** computational top row η=0 is just
    the existing ``zero_top_halo_fields`` (no per-column logic; the
    metrics carry the geometry).

This eliminates the staircase corner diffraction / late-time
exponential instability of Stage 1.
"""

from __future__ import annotations

from .acoustic import step_cpml
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields


class AcousticCurvilinear(SecondOrderEquation):
    """Curvilinear-grid Acoustic 2-D for irregular topography.

    Same field / source / receiver layout as :class:`Acoustic`. The
    propagator must build a :class:`CurvilinearGrid` from the user's
    ``topography`` and attach its padded metric tensors to this
    equation via ``set_curvilinear_metrics`` before the first step.

    
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",),
                  description="Acoustic P-wave velocity model.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"),
                  description="Primary acoustic pressure-like wavefield.",
                  supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",),
                  description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the ξ-derivative term.",
                  internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the η-derivative term.",
                  internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the ξ-direction update.",
                  internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the η-direction update.",
                  internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        use_fast_grad_kernels = backend == "torch" and "cuda" in str(device)
        super().__init__(
            spatial_order, device, backend, dim=dim, other_kernels=use_fast_grad_kernels
        )
        super().init_laplace(ltype="1dsep", backend=backend)
        self.grad_kernels = None
        if use_fast_grad_kernels:
            self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}
        # Filled by Propagator at init time once topography is processed.
        self._curv_alpha = None
        self._curv_metric_pηη = None
        self._curv_metric_pη = None
        # Spacing in the η axis (dimensionless [0, 1]) — also filled by
        # the propagator. ξ spacing == user dh (passed via the standard
        # propagator ``h`` arg).
        self._curv_d_eta = None
        # Stable marker the propagator can use to detect the curvilinear path.
        self.is_curvilinear = True

    def set_curvilinear_metrics(self, alpha, metric_pηη, metric_pη, d_eta):
        """Attach precomputed metric tensors (padded to runtime shape)."""
        self._curv_alpha = alpha
        self._curv_metric_pηη = metric_pηη
        self._curv_metric_pη = metric_pη
        self._curv_d_eta = float(d_eta)

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        (vp,) = models
        if self._curv_alpha is None or self._curv_d_eta is None:
            raise RuntimeError(
                "AcousticCurvilinear was stepped before set_curvilinear_metrics() "
                "was called — the Propagator must attach metrics during init."
            )

        # ξ spacing == user-supplied physical dh. η spacing is the
        # dimensionless cell size on the computational grid. Override
        # the propagator's ``h`` tuple here so the FD operators get the
        # right scaling on each axis.
        hx = h if (isinstance(h, (int, float))) else h[-1]
        hz = self._curv_d_eta

        # Base derivatives on the computational (ξ, η) grid.
        lap_pηη, lap_pξξ = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        p_η = self.gradient(u_now, hz, -2, kernels=self.grad_kernels)
        # Mixed partial p_ξη = ∂(p_η)/∂ξ.
        p_ξη = self.gradient(p_η, hx, -1, kernels=self.grad_kernels)

        # Curvilinear Laplacian, split half-and-half between the
        # x-direction and z-direction PML wrappers so each gets a piece
        # of the cross term:
        #   ξ-PML half : p_ξξ + α p_ξη
        #   η-PML half : (α² + β²) p_ηη + α p_ξη + (α_ξ + α α_η) p_η
        # Sum = p_ξξ + 2α p_ξη + (α²+β²) p_ηη + (α_ξ + α α_η) p_η
        # = Δ_phys p.
        alpha = self._curv_alpha
        lap_x_curv = lap_pξξ + alpha * p_ξη
        lap_z_curv = (
            self._curv_metric_pηη * lap_pηη
            + alpha * p_ξη
            + self._curv_metric_pη * p_η
        )

        out = step_cpml(
            *wavefields, vp, dt, h, b,
            lap_x_curv, lap_z_curv,
            self.b, self.gradient, self.grad_kernels,
        )
        if getattr(self, "free_surface", False):
            # On the flat computational top row η=0, the BC is the usual
            # ``p = 0`` Dirichlet — the same flat-surface treatment used
            # by ``Acoustic``. No per-column logic needed; the metrics
            # carry the topography.
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

    def _C(self):
        raise NotImplementedError(
            "AcousticCurvilinear does not have a CUDA kernel; use impl='eager'."
        )

    @property
    def cuda_layout(self):
        # Required by the propagator base class even when impl='c' is
        # rejected; copy Acoustic's layout for interface compatibility.
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=4,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            boundary_save_nvar=1,
        )

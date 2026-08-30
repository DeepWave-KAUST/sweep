from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import zero_above_topo


def step_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_z, 
        pml,
        grad_op,
        grad_kernels=None,
        ):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.
    
    # Use fixed stencil convolutions when available; this is much cheaper than torch.gradient.
    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)
    
    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + grad_op(az*psiz, h, -2, kernels=grad_kernels)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + grad_op(ax*psix, h, -1, kernels=grad_kernels)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, zetax, zetaz

class Acoustic(SecondOrderEquation):
    """Second-order 2-D acoustic wave equation with CPML auxiliary fields.

    The default scalar-wave solver: a pressure-like field ``h1`` is driven by
    ``vp**2 · Laplace(u)`` and absorbed at every side by a split-step CPML
    formulation (``cpmlr`` by default). This is the most common forward and
    inversion equation in the codebase.

    
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic P-wave velocity model.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    # Eager ``func`` honours a per-edge free surface (any subset of the 4 faces,
    # each pressure-release) via ``self.fs_faces``; see ``_apply_free_surface``.
    supports_per_edge_free_surface = True
    # The 2-D Acoustic CUDA kernels honour ``fs_faces`` per edge via the
    # SolverContext per-face pad (region tests + set_boundary_zeros).  The CPU-C
    # kernels are not migrated yet — the propagator guards per-edge impl='c' to
    # CUDA (see _CompiledPropagator.__init__).
    supports_per_edge_free_surface_c = True

    # The 2-D Acoustic CUDA kernels (forward / adjoint / gradient) all stride
    # the ``vp`` buffer per batch index ``b`` (``vp + b*nz*nx``) and write the
    # model gradient into a per-shot ``zeros_like(vp)`` buffer, so a per-shot
    # batched model ``(B, nz, nx)`` is supported directly by ``impl='c'``:
    # shot ``b`` propagates in ``vp[b]`` and its gradient is kept per-shot.
    supports_batched_models = True

    def __init__(self, spatial_order=4, device='cpu', backend='torch', dim=2):
        """Build the acoustic equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer
                (typical values ``2, 4, 6, 8, 10, …``). Higher orders cut
                grid dispersion at the cost of more compute per step and a
                wider PML halo.
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/acoustic2d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static kernels (Laplace /
                gradient coefficients). Use ``'cuda'`` / a ``torch.device``
                when running on GPU so the propagator can follow without
                a host↔device copy. Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                When you later want ``impl='c'``, leave this on ``'torch'``
                — the compiled CUDA kernels are dispatched through the
                Torch binding. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class; use
                :class:`Acoustic3D` for 3-D. Defaults to 2.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_separable_laplace()
        if backend == 'torch' and 'cuda' in str(device):
            super().init_grad_kernels()

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        (vp,) = models
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        out = step_cpml(*wavefields, vp, dt, h, b, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)
        if getattr(self, "free_surface", False):
            topo_rows = getattr(self, "_topo_rows_runtime", None)
            if topo_rows is not None:
                # Irregular surface: zero ALL air cells per column.
                # In the flat-degenerate case (topo_rows == halo*ones) this
                # matches ``zero_top_halo_fields`` bit-for-bit.
                out = tuple(zero_above_topo(field, topo_rows, axis=-2) for field in out)
            else:
                out = self._apply_free_surface(out)
        return out

    def _C(self, ):
        # CUDA IMPLEMENTATION
        import torch
        from sweep._C import (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )
        return (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        import torch
        from sweep._C import acoustic2d_rtm

        return acoustic2d_rtm

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            # psix, psiz, zetax, zetaz (4) + psixn, psizn (2) for the race-free
            # forward psi double-buffer: the forward reads psi at neighbours then
            # writes the next psi to a SEPARATE buffer (no in-place RAW race).
            # bind() detects the larger count (3 u + 6 pml = 9) and swap_pml()
            # rotates psix<->psixn each step.
            pml_nvar=6,
            # Fused single-kernel adjoint also double-buffers zeta: the ADJOINT
            # wavefield gets +2 (zetaxn, zetazn) -> 11 tensors; forward stays 9.
            adjoint_extra_nvar=2,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            # forward slots psix,psiz,zetax,zetaz,psixn,psizn live in per-axis
            # slabs; checkpoints carry u_prev,u_now + the same aux slabs.
            pml_slot_axes=("x", "z", "x", "z", "x", "z"),
            checkpoint_slot_axes=(None, None, "x", "z", "x", "z"),
            boundary_save_nvar=1,
            backward_workspace_nvar=1,
        )

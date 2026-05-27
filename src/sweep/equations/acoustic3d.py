from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields
from ._free_surface import zero_above_topo


def step_cpml(
        u_now, u_pre, psix, psiy, psiz, zetax, zetay, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_y, lap_z, 
        pml,
        grad_op
        ):

    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = pml

    w_sum = 0.

    dudz = grad_op(u_now, h, -3)
    dudy = grad_op(u_now, h, -2)
    dudx = grad_op(u_now, h, -1)

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + grad_op(az*psiz, h, -3)
    w_sum += (1+bz) * tmpz + az * zetaz

    psizn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # Y direction
    tmpy = ((1+by)*lap_y + dbydy * dudy) + grad_op(ay*psiy, h, -2)
    w_sum += (1+by) * tmpy + ay * zetay
    psiyn = by * dudy + ay * psiy
    zetay = by * tmpy + ay * zetay

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + grad_op(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, psizn, zetax, zetay, zetaz

class Acoustic3D(SecondOrderEquation):
    """Second-order 3-D acoustic wave equation with CPML auxiliary fields.

    Three-dimensional generalisation of :class:`Acoustic`: a pressure-like
    field ``h1`` is driven by ``vp**2 · Laplace(u)`` (assembled from
    separable 1-D Laplacians in z / y / x) and absorbed on every face by a
    split-step CPML formulation (``cpmlr`` by default).

    !!! info "Models (constructor input order)"

        - ``vp`` (m/s): 3D acoustic P-wave velocity model.

    !!! info "Wavefields"

        - ``h1`` (aliases: ``pressure``, ``p``): Primary 3D acoustic pressure-like wavefield; default source and receiver.
        - ``h2`` (aliases: ``pressure_prev``): Previous-step pressure-like wavefield (internal).
        - ``psix``: CPML memory variable for the x-derivative term (internal).
        - ``psiy``: CPML memory variable for the y-derivative term (internal).
        - ``psiz``: CPML memory variable for the z-derivative term (internal).
        - ``zetax``: CPML auxiliary wavefield for the x-direction update (internal).
        - ``zetay``: CPML auxiliary wavefield for the y-direction update (internal).
        - ``zetaz``: CPML auxiliary wavefield for the z-direction update (internal).

    !!! info "Defaults"

        - ``source_type``: ``['h1']``
        - ``receiver_type``: ``['h1']``
        - ``pml_type``: ``'cpmlr'``
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="3D acoustic P-wave velocity model.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary 3D acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiy", description="CPML memory variable for the y-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetay", description="CPML auxiliary wavefield for the y-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=3):
        """Build the 3-D acoustic equation operator.

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
                ``src/sweep/csrc/cuda/equations/acoustic3d/forward.cu``)
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
            dim: Stored dimensionality. Always ``3`` for this class; use
                :class:`Acoustic` for 2-D. Defaults to 3.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        self._wavefields = []
        super().init_laplace(ltype='3dsep', backend=backend)

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return [spec.name for spec in self.FIELD_SPECS]

    @property
    def field_specs(self):
        return list(self.FIELD_SPECS)

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        (vp,) = models
        hz, hy, hx = self._spacings_3d(h)
        lap_z, lap_y, lap_x = self.laplace3d_sep(u_now, self.laplace_kernels, hz, hy, hx)
        out = step_cpml(*wavefields, vp, dt, h, b, lap_x, lap_y, lap_z, self.b, self.gradient)
        if getattr(self, "free_surface", False):
            topo_rows = getattr(self, "_topo_rows_runtime", None)
            if topo_rows is not None:
                # Irregular surface: zero ALL air cells per (iy, ix) column.
                # ``topo_rows`` is 2-D shape (ny, nx) on the runtime grid.
                out = tuple(zero_above_topo(field, topo_rows, axis=-3) for field in out)
            else:
                out = zero_top_halo_fields(out, self.so // 2, axis=-3)
        return out

    def _C(self, ):
        import torch
        from sweep._C import (
            acoustic3d_forward,
            acoustic3d_backward,
            acoustic3d_backward_bs,
            acoustic3d_backward_ckpt,
            acoustic3d_backward_recursive_ckpt,
        )
        return (
            acoustic3d_forward,
            acoustic3d_backward,
            acoustic3d_backward_bs,
            acoustic3d_backward_ckpt,
            acoustic3d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        import torch
        from sweep._C import acoustic3d_rtm

        return acoustic3d_rtm

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=6,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
            boundary_save_nvar=1,
            backward_workspace_nvar=1,
        )

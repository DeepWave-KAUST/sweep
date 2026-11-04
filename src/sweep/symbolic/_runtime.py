"""Base class of the equations ``sweep.symbolic`` generates.

It holds what does not depend on the PDE: the operator set-up, the default
source / receiver fields and the free surface. Its set-up mirrors
:class:`~sweep.equations.Acoustic` (2-D) and :class:`~sweep.equations.Acoustic3D`
(3-D), first-derivative kernels included: the fixed stencil on a 2-D CUDA
device, ``torch.gradient`` otherwise.
"""
from __future__ import annotations

from sweep.equations._free_surface import zero_above_topo
from sweep.equations.base import SecondOrderEquation
from sweep.equations.utils import zero_top_halo_fields


class SymbolicSecondOrderEquation(SecondOrderEquation):
    NDIM = 2
    # "zero" (pressure release on every field) or None (no free surface).
    SYMBOLIC_FREE_SURFACE = None
    SYMBOLIC_PML = "cpml"
    DEFAULT_SOURCE_FIELDS = ()
    DEFAULT_RECEIVER_FIELDS = ()

    _fs_active = False
    # Appended by the propagator to its refusal of a topography on a PDE built
    # without a free surface (``supports_image_topography`` is False there).
    image_topography_hint = "Build the PDE with free_surface='zero' to run it on a topography."

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the equation operator.

        Args:
            spatial_order: FD accuracy order of the second derivatives; an even
                integer. Defaults to 4.
            device: device of the static stencils. Defaults to ``'cpu'``.
            backend: ``'torch'`` only. Defaults to ``'torch'``.
        """
        if backend != "torch":
            raise NotImplementedError(
                f"{type(self).__name__}: backend={backend!r}; generated equations run on 'torch' only")
        super().__init__(spatial_order, device, backend, dim=self.NDIM)
        self.init_separable_laplace()
        if self.NDIM == 2 and "cuda" in str(device):
            self.init_grad_kernels()

    @property
    def default_source_fields(self):
        return list(self.DEFAULT_SOURCE_FIELDS)

    @property
    def default_receiver_fields(self):
        return list(self.DEFAULT_RECEIVER_FIELDS)

    # The propagator writes ``equation.free_surface`` once it has resolved the
    # boundary. A PDE built without a free surface refuses it here: its step has
    # no surface to apply, and running on would model an absorbing top.
    @property
    def free_surface(self):
        return self._fs_active

    @free_surface.setter
    def free_surface(self, value):
        if value and self.SYMBOLIC_FREE_SURFACE is None:
            raise NotImplementedError(
                f"{type(self).__name__} was built without a free surface; build the PDE "
                "with free_surface='zero' to run it with free_surface=True or a topography")
        self._fs_active = bool(value)

    def _free_surface_out(self, out):
        if not self.free_surface:
            return out
        topo_rows = getattr(self, "_topo_rows_runtime", None)
        if topo_rows is not None:
            return tuple(zero_above_topo(field, topo_rows, axis=-self.NDIM) for field in out)
        if self.NDIM == 2:
            return self._apply_free_surface(out)
        return zero_top_halo_fields(out, self.so // 2, axis=-3)

"""Base classes of the equations ``sweep.symbolic`` generates.

They hold what does not depend on the PDE: the operator set-up, the default
source / receiver fields and the free surface. The second-order set-up mirrors
:class:`~sweep.equations.Acoustic` (2-D) and :class:`~sweep.equations.Acoustic3D`
(3-D), first-derivative kernels included: the fixed stencil of the FD order in
2-D on every device, ``torch.gradient`` in 3-D. The first-order one is
:class:`~sweep.equations.base.FirstOrderEquation`'s staggered-grid operator, as
:class:`~sweep.equations.Acoustic1st` and :class:`~sweep.equations.Elastic` use,
plus, for a dissipative term, the second-order equations' separable second
derivatives.
"""
from __future__ import annotations

from sweep.equations._free_surface import zero_above_topo
from sweep.equations.base import FirstOrderEquation, SecondOrderEquation
from sweep.equations.utils import zero_top_halo_fields
from sweep.operators.derivative import Derivative
from sweep.operators.factory import LaplaceGradientOps
from sweep.scalars import generate_convolution_kernel


def _torch_only(cls, backend):
    if backend != "torch":
        raise NotImplementedError(
            f"{cls.__name__}: backend={backend!r}; generated equations run on 'torch' only")


def _refuse_lossy_boundary_saving(cls, lossy):
    """Eager boundary saving rebuilds the wavefield by running the step
    backwards in time, which a dissipative or damping term does not allow."""
    if lossy:
        raise NotImplementedError(
            f"{cls.__name__} cannot use boundary saving: it rebuilds the wavefield by "
            f"stepping backwards in time, which the dissipative or damping term"
            f"{'s' if len(lossy) > 1 else ''} {', '.join(lossy)} do"
            f"{'' if len(lossy) > 1 else 'es'} not allow: stepped backwards, "
            "a diffusion or a damping amplifies every round-off error instead of damping it, and "
            "the gradient would be wrong. Use memory=Full() or memory=Ckpt().")
    return True


def _high_order_stencils(eq, device):
    """``{(order, stagger): Derivative}`` for the derivatives of order 3 and
    higher the step takes, at the equation's FD accuracy."""
    return {(n, st): Derivative(n, eq.so, eq.NDIM, st, device=device) for n, st in eq.HIGH_ORDER_STENCILS}


class SymbolicSecondOrderEquation(SecondOrderEquation):
    NDIM = 2
    # "zero" (pressure release on every field) or None (no free surface).
    SYMBOLIC_FREE_SURFACE = None
    SYMBOLIC_PML = "cpml"
    DEFAULT_SOURCE_FIELDS = ()
    DEFAULT_RECEIVER_FIELDS = ()
    # ((order, stagger), ...) of the derivatives of order 3 and higher.
    HIGH_ORDER_STENCILS = ()
    # The damping terms (``b*p.dt``) as text; empty without one.
    DAMPING_TERMS = ()

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
        _torch_only(type(self), backend)
        super().__init__(spatial_order, device, backend, dim=self.NDIM)
        self.init_separable_laplace()
        if self.NDIM == 2:
            self.init_grad_kernels()
        self._dn = _high_order_stencils(self, device)

    # See SymbolicFirstOrderEquation.supports_boundary_saving_eager.
    @property
    def supports_boundary_saving_eager(self):
        return _refuse_lossy_boundary_saving(type(self), self.DAMPING_TERMS)

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


class SymbolicFirstOrderEquation(FirstOrderEquation):
    NDIM = 2
    SYMBOLIC_PML = "cpml"
    DEFAULT_SOURCE_FIELDS = ()
    DEFAULT_RECEIVER_FIELDS = ()
    # The dissipative (viscous, diffusive) terms of the PDE as text,
    # ("nu*laplace(vx)", ...); empty for a lossless system.
    DISSIPATIVE_TERMS = ()
    # The damping terms (``-r/tau_s*r``: a field times a coefficient in its own
    # equation) as text; empty without one.
    DAMPING_TERMS = ()
    HIGH_ORDER_STENCILS = ()
    default_pml_type = "cpmls"
    # Appended by the propagator to its refusal of a topography.
    image_topography_hint = "A first-order PDE takes no free surface yet."

    # A dissipative term's second derivatives: the separable stencil of the
    # second-order equations.
    init_separable_laplace = SecondOrderEquation.init_separable_laplace
    _prepare_separable_laplace_kernels = SecondOrderEquation._prepare_separable_laplace_kernels

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first
                derivatives and of the second derivatives of the dissipative
                terms; an even integer. Defaults to 4.
            device: device of the static stencils. Defaults to ``'cpu'``.
            backend: ``'torch'`` only. Defaults to ``'torch'``.
        """
        _torch_only(type(self), backend)
        super().__init__(spatial_order, device, backend, ndim=self.NDIM)
        if self.DISSIPATIVE_TERMS:
            LaplaceGradientOps.__init__(self, backend=backend)
            self.kf = generate_convolution_kernel
            self.init_separable_laplace()
        self._dn = _high_order_stencils(self, device)

    # Eager boundary saving rebuilds the wavefield by running the step
    # backwards in time. A property rather than the flag the shipped
    # dissipative equations set to False: the propagator reads it when boundary
    # saving is requested, and this refusal names the terms (the flag's shared
    # refusal speaks of round-off and offers impl='c', which a generated class
    # does not have).
    @property
    def supports_boundary_saving_eager(self):
        return _refuse_lossy_boundary_saving(type(self), self.DISSIPATIVE_TERMS + self.DAMPING_TERMS)

    @property
    def default_source_fields(self):
        return list(self.DEFAULT_SOURCE_FIELDS)

    @property
    def default_receiver_fields(self):
        return list(self.DEFAULT_RECEIVER_FIELDS)

    # The propagator writes ``equation.free_surface`` once it has resolved the
    # boundary; the step applies none, so a surface is refused rather than run
    # as an absorbing top.
    @property
    def free_surface(self):
        return False

    @free_surface.setter
    def free_surface(self, value):
        if value:
            raise NotImplementedError(
                f"{type(self).__name__} is a first-order PDE, which takes no free surface yet; "
                "run it with free_surface=False")

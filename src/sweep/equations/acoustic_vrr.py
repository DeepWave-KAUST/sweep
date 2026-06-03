from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields


def step_cpml(u_now, u_pre, psix, psiz, zetax, zetaz,
              vp, rx, rz, dt, h, b,
              lap_x, lap_z,
              pml, grad_op, grad_kernels=None):
    """Single split-step CPML update of the 2-D VRR acoustic equation.

    The VRR propagator is the scalar second-order PDE

        ∂²p/∂t² = vp²·∇²p + (vp·∇vp − 2·vp²·r)·∇p ,

    i.e. ``vp²·∇²p`` plus a first-derivative coupling whose coefficient is
    ``cx = vp·∂vp/∂x − 2·vp²·rx`` (and ``cz`` analogously). This shares the
    ``vp²·∇²p + (coef)·∇p`` shape of the VRZ form, so the CPML reconstruction
    of the Laplacian (``w_sum``) and of the first derivatives
    (``dpd{x,z}_cpml``) is identical to :func:`acoustic_vrz.step_cpml`; only
    the first-derivative coefficients differ.
    """
    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.

    dpdx = grad_op(u_now, h, axis=-1, kernels=grad_kernels)
    dpdz = grad_op(u_now, h, axis=-2, kernels=grad_kernels)
    dvpdx = grad_op(vp, h, axis=-1, kernels=grad_kernels)
    dvpdz = grad_op(vp, h, axis=-2, kernels=grad_kernels)

    # Z direction
    tmpz = ((1 + bz) * lap_z + dbzdz * dpdz) + grad_op(az * psiz, h, axis=-2, kernels=grad_kernels)
    w_sum += (1 + bz) * tmpz + az * zetaz

    psizn = bz * dpdz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1 + bx) * lap_x + dbxdx * dpdx) + grad_op(ax * psix, h, axis=-1, kernels=grad_kernels)
    w_sum += (1 + bx) * tmpx + ax * zetax

    psixn = bx * dpdx + ax * psix
    zetax = bx * tmpx + ax * zetax

    dpdx_cpml = dpdx + psixn
    dpdz_cpml = dpdz + psizn

    cx = vp * dvpdx - 2 * vp**2 * rx
    cz = vp * dvpdz - 2 * vp**2 * rz

    u_next = 2 * u_now - u_pre + dt**2 * (
        vp**2 * w_sum + cx * dpdx_cpml + cz * dpdz_cpml
    )

    return u_next, u_now, psixn, psizn, zetax, zetaz


class AcousticVRR(SecondOrderEquation):
    """Second-order 2-D acoustic wave equation in variable-density VRR form.

    Pressure-only scalar acoustics with density coupling carried by two
    auxiliary first-derivative parameters ``rx``, ``rz`` (rather than the
    impedance-like ``z`` of :class:`AcousticVRZ`). The propagator is a single
    second-order PDE in ``h1``,

        ∂²p/∂t² = vp²·∇²p + vp·(∇vp·∇p) − 2·vp²·(r·∇p) ,

    so it refracts at sharp contrasts without needing a staggered velocity
    field. Absorbing boundaries via split-step CPML (``cpmlr``).

    Reference: 10.3997/2214-4609.202010332.

    !!! info "Models (constructor input order)"

        - ``vp`` (m/s): Acoustic velocity model.
        - ``rx``: Auxiliary horizontal parameter used by the VRR formulation.
        - ``rz``: Auxiliary vertical parameter used by the VRR formulation.

    !!! info "Wavefields"

        - ``h1`` (aliases: ``pressure``, ``p``): Primary VRR acoustic pressure-like wavefield; default source and receiver.
        - ``h2`` (aliases: ``pressure_prev``): Previous-step VRR acoustic pressure-like wavefield (internal).
        - ``psix``: CPML memory variable for the x-derivative term (internal).
        - ``psiz``: CPML memory variable for the z-derivative term (internal).
        - ``zetax``: CPML auxiliary wavefield for the x-direction update (internal).
        - ``zetaz``: CPML auxiliary wavefield for the z-direction update (internal).

    !!! info "Defaults"

        - ``source_type``: ``['h1']``
        - ``receiver_type``: ``['h1']``
        - ``pml_type``: ``'cpmlr'``
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic velocity model.", unit="m/s"),
        ModelSpec("rx", description="Auxiliary horizontal parameter used by the VRR formulation."),
        ModelSpec("rz", description="Auxiliary vertical parameter used by the VRR formulation."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary VRR acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step VRR acoustic pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device='cpu', backend='torch', dim=2):
        """Build the 2-D VRR acoustic equation operator.

        Args:
            spatial_order (int, optional): The order of the Taylor expansion
                (must be even) for the spatial Laplacian and the auxiliary
                first-derivative kernels used by the ``∇vp·∇p`` /
                ``r·∇p`` terms. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs. Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class. Defaults to 2.
        """
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
        super().init_grad_kernels()

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]


    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        out = step_cpml(*wavefields, *models, dt, h, b, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

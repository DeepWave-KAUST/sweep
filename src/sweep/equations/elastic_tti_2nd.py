"""2D displacement-based (second-order) elastic TTI wave equation.

Implements the forward/inverse formulation of Oh, Shin, Alkhalifah & Min
(2020), *Multistage elastic full-waveform inversion for tilted transverse
isotropic media*, Geophys. J. Int. 223, 57-76,
[10.1093/gji/ggaa295](https://doi.org/10.1093/gji/ggaa295):

- second-order displacement equations (their eqs 11-12) for ``(ux, uz)``
  in a 2-D P-SV TTI medium with six stiffness entries
  ``C~11, C~33, C~13, C~55, C~15, C~35``;
- the 2-D Bond rotation about y (their eq. 10, Macbeth 2002);
- the hierarchical VTI parametrization (their eqs 18-19):
  ``(vh, vs, rho, epsilon, eta, theta)`` with the HORIZONTAL P velocity
  ``vh`` and the anellipticity ``eta`` instead of (vp0, delta).

The paper solves the equations with a frequency-domain FEM; here they are
time-stepped with a leapfrog scheme, and every spatial term is a nested
forward/backward staggered first-derivative pair,
``D_b [ C D_f u ]`` — a conservative discretisation whose stiffness
operator is exactly self-adjoint (``K = -D^T C D``), checkerboard-free,
and CPML-absorbed per first derivative (8 memory variables, ``cpmls``).
"""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec

TTI2ND_STIFFNESS_KEYS = ("C11", "C33", "C13", "C55", "C15", "C35")


def step(
    ux, uz, ux_pre, uz_pre,
    m_gxux, m_gzux, m_gxuz, m_gzuz,
    m_sxxx, m_sxzz, m_sxzx, m_szzz,
    rho, C11, C33, C13, C55, C15, C35,
    dt, h, b, pd,
    pml=None,
):
    """One displacement leapfrog step of Oh et al. (2020) eqs 11-12.

    Inner forward derivatives build the three stresses from the full
    stiffness row; outer backward derivatives take their divergence.
    Both derivative layers carry their own CPML recursive-convolution
    memory (half-node profiles inside, integer-node outside).
    """

    del h
    pml = pml if pml is not None else ()
    if len(pml) != 8:
        raise ValueError("ElasticTTI2nd requires pml_type='cpmls', which provides eight staggered CPML profiles.")
    az, bz, azh, bzh, ax, bx, axh, bxh = pml

    gxux = pd.x_forward(ux)
    gzux = pd.z_forward(ux)
    gxuz = pd.x_forward(uz)
    gzuz = pd.z_forward(uz)

    m_gxux = axh * m_gxux + bxh * gxux
    gxux = gxux + m_gxux
    m_gzux = azh * m_gzux + bzh * gzux
    gzux = gzux + m_gzux
    m_gxuz = axh * m_gxuz + bxh * gxuz
    gxuz = gxuz + m_gxuz
    m_gzuz = azh * m_gzuz + bzh * gzuz
    gzuz = gzuz + m_gzuz

    exz = gzux + gxuz
    sxx = C11 * gxux + C13 * gzuz + C15 * exz
    szz = C13 * gxux + C33 * gzuz + C35 * exz
    sxz = C15 * gxux + C35 * gzuz + C55 * exz

    dsxx_dx = pd.x_backward(sxx)
    dsxz_dz = pd.z_backward(sxz)
    dsxz_dx = pd.x_backward(sxz)
    dszz_dz = pd.z_backward(szz)

    m_sxxx = ax * m_sxxx + bx * dsxx_dx
    dsxx_dx = dsxx_dx + m_sxxx
    m_sxzz = az * m_sxzz + bz * dsxz_dz
    dsxz_dz = dsxz_dz + m_sxzz
    m_sxzx = ax * m_sxzx + bx * dsxz_dx
    dsxz_dx = dsxz_dx + m_sxzx
    m_szzz = az * m_szzz + bz * dszz_dz
    dszz_dz = dszz_dz + m_szzz

    scale = dt * dt / rho
    ux_next = 2.0 * ux - ux_pre + scale * (dsxx_dx + dsxz_dz)
    uz_next = 2.0 * uz - uz_pre + scale * (dsxz_dx + dszz_dz)

    return (
        ux_next, uz_next, ux, uz,
        m_gxux, m_gzux, m_gxuz, m_gzuz,
        m_sxxx, m_sxzz, m_sxzx, m_szzz,
    )


class ElasticTTI2nd(FirstOrderEquation):
    """Displacement-based 2-D elastic TTI wave equation (Oh et al. 2020).

    Second-order-in-time P-SV formulation ``rho u_tt = div(C~ : grad u)``
    (GJI 223, eqs 11-12) with the hierarchical VTI parametrization of
    their eq. (19): horizontal P velocity ``vh``, vertical S velocity
    ``vs``, density, Thomsen ``epsilon``, anellipticity ``eta`` and the
    tilt ``theta`` — all six differentiable. ``prepare_models`` builds
    the four VTI constants and Bond-rotates them (their eq. 10) into the
    six TTI entries consumed by the update, so autograd chains the
    stiffness gradients back to the physical parameters.

    Differences from :class:`ElasticTTI`/:class:`ElasticTTISG` (first-order
    velocity-stress, 2-D-3C, 8 parameters): two displacement components
    only (no SH — no ``gamma``/``phi``), ``vh``/``eta`` instead of
    ``vp0``/``delta``, and a leapfrog second-order time stepping whose
    stiffness operator is discretely self-adjoint.


    """

    # The isotropic image-method free surface is WRONG for an anisotropic
    # medium (see EquationBase.supports_free_surface); fail loud instead.
    supports_free_surface = False

    prepare_models_for_c = True
    default_pml_type = "cpmls"

    MODEL_SPECS = (
        ModelSpec("vh", description="Horizontal P-wave velocity (VTI frame).", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Vertical S-wave velocity.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density.", unit="kg/m^3"),
        ModelSpec("epsilon", description="Thomsen epsilon."),
        ModelSpec("eta", description="Anellipticity eta = (epsilon - delta) / (1 + 2 delta)."),
        ModelSpec("theta", description="Tilt angle of the TI symmetry axis.", unit="rad"),
    )

    FIELD_SPECS = (
        FieldSpec("ux", aliases=("displacement_x",), description="Horizontal displacement.", supports_source=True, supports_receiver=True),
        FieldSpec("uz", aliases=("displacement_z",), description="Vertical displacement.", supports_source=True, supports_receiver=True),
        FieldSpec("ux_pre", description="Previous-step horizontal displacement.", internal=True),
        FieldSpec("uz_pre", description="Previous-step vertical displacement.", internal=True),
        FieldSpec("m_gxux", description="CPML memory for dux/dx.", internal=True, boundary_related=True),
        FieldSpec("m_gzux", description="CPML memory for dux/dz.", internal=True, boundary_related=True),
        FieldSpec("m_gxuz", description="CPML memory for duz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_gzuz", description="CPML memory for duz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxxx", description="CPML memory for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxzz", description="CPML memory for dsxz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxzx", description="CPML memory for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_szzz", description="CPML memory for dszz/dz.", internal=True, boundary_related=True),
    )

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the displacement-based 2-D elastic TTI operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/elastic_tti_2nd2d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. When you later want ``impl='c'``, leave this
                on ``'torch'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend, ndim=2)

    @property
    def default_source_fields(self):
        return ["uz"]

    @property
    def default_receiver_fields(self):
        return ["ux", "uz"]

    def prepare_models(self, models):
        """(vh, vs, rho, epsilon, eta, theta) -> [rho] + 6 rotated stiffnesses.

        VTI constants from Oh et al. (2020) eq. (19) — note C13 uses the
        NMO velocity ``vh^2/(1+2 eta)`` — then the closed-form 2-D Bond
        rotation of their eq. (10). Pure smooth tensor ops: autograd
        differentiates every physical parameter, including theta (no
        VTI-fallback masking needed — the polynomials are exact at
        theta = 0).
        """
        vh, vs, rho, epsilon, eta, theta = models
        vh2 = vh * vh
        vs2 = vs * vs
        one_p2e = 1.0 + 2.0 * epsilon
        one_p2n = 1.0 + 2.0 * eta

        C11_0 = rho * vh2
        C33_0 = rho * vh2 / one_p2e
        C55_0 = rho * vs2
        radicand = (vh2 / one_p2e - vs2) * (vh2 / one_p2n - vs2)
        if hasattr(radicand, "clamp_min"):
            radicand = radicand.clamp_min(0.0)
        else:
            import jax.numpy as jnp

            radicand = jnp.maximum(radicand, 0.0)
        C13_0 = rho * (radicand ** 0.5 - vs2)

        if hasattr(theta, "cos"):
            ct, st = theta.cos(), theta.sin()
        else:
            import jax.numpy as jnp

            ct, st = jnp.cos(theta), jnp.sin(theta)
        ct2 = ct * ct
        st2 = st * st
        s2c2 = st2 * ct2
        ct4 = ct2 * ct2
        st4 = st2 * st2

        mix = 2.0 * C13_0 + 4.0 * C55_0
        C11t = C11_0 * ct4 + C33_0 * st4 + mix * s2c2
        C33t = C11_0 * st4 + C33_0 * ct4 + mix * s2c2
        C13t = (C11_0 + C33_0 - 4.0 * C55_0) * s2c2 + C13_0 * (st4 + ct4)
        C55t = (C11_0 + C33_0 - 2.0 * C13_0) * s2c2 + C55_0 * (ct2 - st2) ** 2
        A = C13_0 - C11_0 + 2.0 * C55_0
        Bc = C13_0 - C33_0 + 2.0 * C55_0
        C15t = A * ct2 * ct * st - Bc * ct * st2 * st
        C35t = A * st2 * st * ct - Bc * st * ct2 * ct

        return [rho, C11t, C33t, C13t, C55t, C15t, C35t]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == len(self.MODEL_SPECS):
            models = self.prepare_models(models)
        elif len(models) != 1 + len(TTI2ND_STIFFNESS_KEYS):
            raise ValueError(
                "ElasticTTI2nd.func expected 6 raw models or "
                f"{1 + len(TTI2ND_STIFFNESS_KEYS)} prepared models, got {len(models)}."
            )
        return step(
            *wavefields,
            *models,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            **kwargs,
        )

    def _C(self):
        import sweep._C as _C

        return (
            _C.elastic_tti_2nd2d_forward,
            _C.elastic_tti_2nd2d_backward,
            _C.elastic_tti_2nd2d_backward_bs,
            _C.elastic_tti_2nd2d_backward_ckpt,
            None,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            # 6 displacement buffers (ux, uz) x (now, pre, next): the CUDA
            # leapfrog rotates a race-free triple buffer, mirroring the
            # acoustic2d second-order layout.
            base_nvar=6,
            pml_nvar=8,
            # bs reconstruction needs the last TWO time levels of both
            # displacement components: storage axis = (ux, uz), level
            # axis = (now, pre).
            last_two_nvar=2,
            last_two_storage_nvar=2,
            boundary_save_nvar=2,
            backward_workspace_nvar=8,
        )

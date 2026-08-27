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

Absorbing boundary: the plain CPML that works for the velocity-stress
solvers is **unstable** here past a few seconds. That is a documented
limitation of CPML for the second-order displacement system, not a defect
of this discretisation — Li & Bou Matar (2010, *J. Acoust. Soc. Am.* 127,
1318-1327), who derive exactly this CPML, report the same growth and note it
occurs "even in the case of an isotropic medium"; the root cause is the
Bécache, Fauqueux & Joly (2003) condition, which Oh et al. themselves cite
(they sidestep it with parameter bounds, and their frequency-domain FEM
cannot show late-time growth at all). Measured here on a uniform TTI medium:
the record envelope passes the direct-arrival peak at 9.6 s and overflows to
NaN at 19.5 s, growing at +34 dB/s from the PML corners.

The fix is the multiaxial PML (M-PML) of Meza-Fajardo & Papageorgiou, in the
form of Li & Bou Matar's eq. (23): mix a fraction ``mpml_ratio`` of each
damping profile into the other axis. The update in :func:`step` is untouched
— only the profiles change. At ``mpml_ratio=0.05`` the boundary artefact
goes from -56.3 dB to -53.2 dB against a domain no reflection can reach,
and the envelope decays for as long as float64 can resolve it.
"""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec

TTI2ND_STIFFNESS_KEYS = ("C11", "C33", "C13", "C55", "C15", "C35")

#: Default multiaxial-PML mixing ratio.  Measured stability threshold on a
#: uniform TTI medium is between 0.02 (blows up at 23.3 s) and 0.05 (stable);
#: Li & Bou Matar use 0.25 for a much harder orthotropic medium.  Each step up
#: costs absorption: 0.05 -> -53.2 dB, 0.10 -> -49.9 dB, 0.25 -> -44.5 dB,
#: against -56.3 dB for the (unstable) plain CPML.
DEFAULT_MPML_RATIO = 0.05


def _cpml_decompose(a, b, dt):
    """Recover ``(sigma, alpha)`` from a CPML ``(a, b)`` profile pair.

    ``setup_pml`` builds ``a = exp(-(sigma + alpha) dt)`` and
    ``b = sigma / (sigma + alpha) * (a - 1)``, then zeroes ``a`` outside the
    layer.  Outside, ``sigma = 0`` and the CFS ``alpha`` sits at its maximum,
    which is what the multiaxial mix needs when the other axis pulls damping
    into a region this axis does not damp on its own.
    """
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    inside = a > 0
    a_safe = np.where(inside, a, 1.0)
    sum_sa = -np.log(a_safe) / dt                       # sigma + alpha
    den = a_safe - 1.0
    den = np.where(np.abs(den) < 1e-12, -1e-12, den)
    sigma = np.where(inside, b * sum_sa / den, 0.0)
    smax = max(float(sigma.max()), 1e-30)
    frac = np.sqrt(np.clip(sigma / smax, 0.0, None))    # sigma ~ frac**2
    alpha_in = np.where(inside, sum_sa - sigma, 0.0)
    near = inside & (frac < 0.5)
    alpha0 = float((alpha_in[near] / (1.0 - frac[near])).max()) if near.any() \
        else float(alpha_in.max())
    return sigma, alpha0 * (1.0 - frac)


def _cpml_recombine(sigma, alpha, dt, dtype):
    import numpy as np

    sum_sa = sigma + alpha
    sum_sa = np.where(np.abs(sum_sa) < 1e-9, 1e-9, sum_sa)
    a = np.exp(-sum_sa * dt)
    b = (sigma / sum_sa) * (a - 1.0)
    off = sigma <= 0
    return (np.where(off, 0.0, a).astype(dtype, copy=False),
            np.where(off, 0.0, b).astype(dtype, copy=False))


def mpml_profiles(profiles, dt, ratio):
    """Turn the eight per-axis CPML profiles into multiaxial (2-D) ones.

    Li & Bou Matar (2010) eq. (23), after Meza-Fajardo & Papageorgiou::

        sigma_z(x, z) = sigma_zz(z) + ratio * sigma_xx(x)
        sigma_x(x, z) = sigma_xx(x) + ratio * sigma_zz(z)

    Every CPML equation stays as it was; only the profiles change, so
    :func:`step` needs no modification. The per-axis inputs broadcast
    (``(1, nz, 1)`` and ``(1, 1, nx)``), so the sums come out ``(1, nz, nx)``.

    The result is deliberately no longer perfectly matched — that is the
    trade the multiaxial layer makes for stability, and it is why ``ratio``
    should be the smallest value that holds.
    """
    import numpy as np

    az, bz, azh, bzh, ax, bx, axh, bxh = [np.asarray(t) for t in profiles]
    dtype = az.dtype
    if ratio <= 0:
        # Still hand back FULL 2-D fields: the compiled kernel indexes the
        # profiles as ``iz * nx + ix`` unconditionally, so there is one code
        # path instead of two.  Broadcasting the unmixed profiles reproduces
        # the plain CPML bit for bit.
        zeros = np.zeros(np.broadcast_shapes(az.shape, ax.shape), dtype=dtype)
        return [np.ascontiguousarray(t + zeros, dtype=dtype)
                for t in (az, bz, azh, bzh, ax, bx, axh, bxh)]
    out = []
    for (a_z, b_z), (a_x, b_x) in (((az, bz), (ax, bx)), ((azh, bzh), (axh, bxh))):
        sz, alz = _cpml_decompose(a_z, b_z, dt)
        sx, alx = _cpml_decompose(a_x, b_x, dt)
        A_z, B_z = _cpml_recombine(sz + ratio * sx, alz + np.zeros_like(sx), dt, dtype)
        A_x, B_x = _cpml_recombine(sx + ratio * sz, alx + np.zeros_like(sz), dt, dtype)
        out.append((A_z, B_z, A_x, B_x))
    (Az, Bz, Ax, Bx), (Azh, Bzh, Axh, Bxh) = out
    return [np.ascontiguousarray(t, dtype=dtype)
            for t in (Az, Bz, Azh, Bzh, Ax, Bx, Axh, Bxh)]


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

    def __init__(self, spatial_order=4, device="cpu", backend="torch",
                 mpml_ratio=DEFAULT_MPML_RATIO):
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
            mpml_ratio: Multiaxial-PML mixing ratio (Li & Bou Matar 2010
                eq. 23). Plain CPML is unstable for this second-order
                displacement system — see the module docstring — and this is
                the knob that fixes it: each damping profile receives this
                fraction of the other axis'. ``0`` restores the plain CPML
                and with it the late-time growth (envelope past the direct
                arrival at 9.6 s, NaN at 19.5 s on a uniform TTI medium).
                Larger values are safer but absorb worse: measured boundary
                artefact against a domain no reflection can reach is
                -56.3 dB at 0 (unstable), -53.2 dB at 0.05, -49.9 dB at 0.10,
                -44.5 dB at 0.25. Defaults to
                :data:`DEFAULT_MPML_RATIO` (0.05).
        """
        super().__init__(spatial_order, device, backend, ndim=2)
        if mpml_ratio < 0:
            raise ValueError(f"mpml_ratio must be >= 0, got {mpml_ratio!r}.")
        self.mpml_ratio = float(mpml_ratio)

    def init_abc(self, type="cpml", **kwargs):
        """Build the CPML profiles, then mix them multiaxially.

        The mix has to happen here rather than in ``step``: that is exactly
        the property of the multiaxial layer that makes it cheap — the update
        never learns about it.  The profiles come back as full 2-D fields even
        at ``mpml_ratio=0`` so the compiled kernel has a single index path.
        """
        super().init_abc(type=type, **kwargs)
        from .utils import to_backend

        import numpy as np

        profiles = [np.asarray(t.detach().cpu() if hasattr(t, "detach") else t)
                    for t in self.b]
        mixed = mpml_profiles(profiles, float(kwargs["dt"]), self.mpml_ratio)
        self.b = mixed if self.backend == "jax" else to_backend(
            mixed, self.backend, self.device)

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

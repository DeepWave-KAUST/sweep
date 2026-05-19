"""First-order acoustic VTI wave equations on a standard staggered grid.

Implements the Duveneck et al. (2008) first-order velocity-stress system for
acoustic VTI media in 2-D (``AcousticVTI1st``) and 3-D (``AcousticVTI1st3D``).
This is an independent sibling to the second-order pseudo-acoustic
``AcousticVTI`` class; the two coexist with different trade-offs:

* ``AcousticVTI1st`` — variable-density, free-surface, 4 wavefields, 4 models.
* ``AcousticVTI`` — memory-light constant-density RTM, 2nd-order scalar form.

References
----------
Duveneck, E., Milcik, P., Bakker, P. M., Perkins, C. (2008),
"Acoustic VTI wave equations and their application for anisotropic
reverse-time migration", SEG Las Vegas 2008 Annual Meeting, pp. 2186–2190.
DOI: 10.1190/1.3059320

Alkhalifah, T. (1998), "Acoustic approximations for processing in TI media",
Geophysics 63, 623–631. (V_S = 0 approximation.)

Thomsen, L. (1986), "Weak elastic anisotropy", Geophysics 51, 1954–1966.
(ε, δ parameterisation.)
"""

import warnings

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
# Free-surface helpers from `_free_surface` are intentionally NOT imported:
# the VTI free-surface BC requires a Mittet/Robertsson-style anisotropic
# treatment that the naive image method does not provide.  Until that lands,
# `free_surface=True` raises NotImplementedError on construction.


# ---------------------------------------------------------------------------
# 2-D step function
# ---------------------------------------------------------------------------

def step_vti_2d(
    vx, vz, sH, sV,
    m_sHx, m_sVz, m_vxx, m_vzz,
    c11, c33, c13, inv_rho,
    dt, h, b, pd,
    pml=None,
    free_surface=False,
):
    """One leapfrog time step of the 2-D acoustic VTI system.

    Implements Duveneck et al. (2008) Eqs (1)–(2) on a standard staggered
    grid.  Absorbing boundaries via CPML (cpmls format).

    Equations of motion (2-D):
      ∂v_x/∂t = (1/ρ) ∂σ_H/∂x
      ∂v_z/∂t = (1/ρ) ∂σ_V/∂z

    Constitutive equations (Duveneck 2008 Eq 2):
      ∂σ_H/∂t = c11 ∂v_x/∂x + c13 ∂v_z/∂z
      ∂σ_V/∂t = c13 ∂v_x/∂x + c33 ∂v_z/∂z

    with cached stiffness coefficients:
      c11 = ρ V_P² (1 + 2ε),   c33 = ρ V_P²,   c13 = ρ V_P² √(1 + 2δ)

    References
    ----------
    Duveneck et al. (2008) DOI: 10.1190/1.3059320
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    if free_surface:
        # The class constructor raises NotImplementedError to prevent this.
        # The branch remains only as a stub for when a proper Mittet/
        # Robertsson-style anisotropic FS is implemented (TODO_free_surface).
        raise NotImplementedError(
            "Free surface for AcousticVTI1st is not yet implemented."
        )

    # --- velocity update ---
    # dsH/dx at (i+½, j) for vx
    dsH_dx = pd.x_forward(sH)
    # dsV/dz at (i, j+½) for vz
    dsV_dz = pd.z_forward(sV)

    m_sHx = axh * m_sHx + bxh * dsH_dx
    dsH_dx = dsH_dx + m_sHx

    m_sVz = azh * m_sVz + bzh * dsV_dz
    dsV_dz = dsV_dz + m_sVz

    vx = vx + dt * inv_rho * dsH_dx
    vz = vz + dt * inv_rho * dsV_dz

    # --- stress update ---
    # dvx/dx at cell centre (i, j)
    dvx_dx = pd.x_backward(vx)
    # dvz/dz at cell centre (i, j)
    dvz_dz = pd.z_backward(vz)

    m_vxx = ax * m_vxx + bx * dvx_dx
    dvx_dx = dvx_dx + m_vxx

    m_vzz = az * m_vzz + bz * dvz_dz
    dvz_dz = dvz_dz + m_vzz

    sH = sH + dt * (c11 * dvx_dx + c13 * dvz_dz)
    sV = sV + dt * (c13 * dvx_dx + c33 * dvz_dz)

    return vx, vz, sH, sV, m_sHx, m_sVz, m_vxx, m_vzz


# ---------------------------------------------------------------------------
# 3-D step function
# ---------------------------------------------------------------------------

def step_vti_3d(
    vx, vy, vz, sH, sV,
    m_sHx, m_sHy, m_sVz, m_vxx, m_vyy, m_vzz,
    c11, c33, c13, inv_rho,
    dt, h, b, pd,
    pml=None,
    free_surface=False,
):
    """One leapfrog time step of the 3-D acoustic VTI system.

    Extends the 2-D Duveneck et al. (2008) system to 3-D via the isotropy in
    the horizontal plane: σ_H = σ_11 = σ_22.

    Equations of motion (3-D):
      ∂v_x/∂t = (1/ρ) ∂σ_H/∂x
      ∂v_y/∂t = (1/ρ) ∂σ_H/∂y
      ∂v_z/∂t = (1/ρ) ∂σ_V/∂z

    Constitutive equations:
      ∂σ_H/∂t = c11 (∂v_x/∂x + ∂v_y/∂y) + c13 ∂v_z/∂z
      ∂σ_V/∂t = c13 (∂v_x/∂x + ∂v_y/∂y) + c33 ∂v_z/∂z

    References
    ----------
    Duveneck et al. (2008) DOI: 10.1190/1.3059320
    """
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml
    if free_surface:
        raise NotImplementedError(
            "Free surface for AcousticVTI1st3D is not yet implemented."
        )

    # --- velocity update ---
    dsH_dx = pd.x_forward(sH)
    dsH_dy = pd.y_forward(sH)
    dsV_dz = pd.z_forward(sV)

    m_sHx = axh * m_sHx + bxh * dsH_dx
    dsH_dx = dsH_dx + m_sHx

    m_sHy = ayh * m_sHy + byh * dsH_dy
    dsH_dy = dsH_dy + m_sHy

    m_sVz = azh * m_sVz + bzh * dsV_dz
    dsV_dz = dsV_dz + m_sVz

    vx = vx + dt * inv_rho * dsH_dx
    vy = vy + dt * inv_rho * dsH_dy
    vz = vz + dt * inv_rho * dsV_dz

    # --- stress update ---
    dvx_dx = pd.x_backward(vx)
    dvy_dy = pd.y_backward(vy)
    dvz_dz = pd.z_backward(vz)

    m_vxx = ax * m_vxx + bx * dvx_dx
    dvx_dx = dvx_dx + m_vxx

    m_vyy = ay * m_vyy + by * dvy_dy
    dvy_dy = dvy_dy + m_vyy

    m_vzz = az * m_vzz + bz * dvz_dz
    dvz_dz = dvz_dz + m_vzz

    div_h = dvx_dx + dvy_dy

    sH = sH + dt * (c11 * div_h + c13 * dvz_dz)
    sV = sV + dt * (c13 * div_h + c33 * dvz_dz)

    return vx, vy, vz, sH, sV, m_sHx, m_sHy, m_sVz, m_vxx, m_vyy, m_vzz


# ---------------------------------------------------------------------------
# Helper: compute stiffness cache (backend-agnostic)
# ---------------------------------------------------------------------------

def _build_stiffness(vp, epsilon, delta, rho):
    """Return (c11, c33, c13, inv_rho) from Thomsen parameters.

    c11 = ρ V_P² (1 + 2ε)
    c33 = ρ V_P²
    c13 = ρ V_P² √(1 + 2δ)       [exact V_S = 0 form; NOT linearised]

    Reference: Duveneck et al. (2008) Eq (1).
    """
    vp2 = vp ** 2
    rho_vp2 = rho * vp2
    c11 = rho_vp2 * (1.0 + 2.0 * epsilon)
    c33 = rho_vp2
    c13 = rho_vp2 * (1.0 + 2.0 * delta) ** 0.5
    inv_rho = 1.0 / rho
    return c11, c33, c13, inv_rho


# ---------------------------------------------------------------------------
# 2-D equation class
# ---------------------------------------------------------------------------

class AcousticVTI1st(FirstOrderEquation):
    """First-order acoustic VTI wave equation on a standard staggered grid (2-D).

    Implements the Duveneck et al. (2008) velocity-stress formulation with
    4 wavefields (v_x, v_z, σ_H, σ_V) and 4 model parameters (V_P, ε, δ, ρ).
    Variable density and CPML absorbing boundaries are supported natively.

    The V_S = 0 approximation (Alkhalifah 1998) sets all shear stiffness
    coefficients to zero, yielding c_12 = c_11 and σ_11 = σ_22 ≡ σ_H so
    only two independent normal stresses exist.

    Parameters
    ----------
    spatial_order : int, optional
        Finite-difference spatial order.  Default 4.
    device : str, optional
        Torch/JAX device.  Default 'cpu'.
    backend : str, optional
        'torch' or 'jax'.  Default 'torch'.
    free_surface : bool, optional
        Enable a free-surface boundary at the top.  **Currently NOT
        SUPPORTED for the VTI system.**  A naive isotropic-style image
        method is *physically incorrect* for VTI because the surface
        constraint σ_V = 0 couples ∂v_z/∂z to ∂v_x/∂x via the off-diagonal
        stiffness c_13 (Robertsson 1996; Mittet 2002).  Passing
        ``free_surface=True`` will raise ``NotImplementedError``.  Use
        absorbing PML on all four sides for now.  See ``TODO_free_surface``
        on this class.

    Notes
    -----
    Stability requires ε ≥ δ everywhere (Grechka, Zhang, Rector 2004,
    Geophysics 69, 576).  ``prepare_models`` emits a warning when violated.

    **This class does NOT have a CUDA binding.**  A pure-Python (PyTorch /
    JAX) time loop is used.  CUDA support is a planned future extension.

    References
    ----------
    Duveneck et al. (2008) DOI: 10.1190/1.3059320
    Alkhalifah (1998) Geophysics 63, 623–631.
    Thomsen (1986) Geophysics 51, 1954–1966.
    Robertsson (1996) Geophysics 61, 1921–1934.  (Proper anisotropic FS.)
    Mittet (2002) Geophysics 67, 1616–1623.  (Staggered-grid extension.)
    """

    # TODO_free_surface: Implement a Mittet/Robertsson-style anisotropic
    # free-surface BC for VTI.  The constraint σ_V = 0 at the surface forces
    # ∂v_z/∂z = -(c13/c33) ∂v_x/∂x, so the σ_H update at the surface row
    # uses the "effective" stiffness (c11 - c13²/c33) on ∂v_x/∂x.  The naive
    # image method (top_free_surface_derivative with odd=True) used by the
    # existing ElasticTTI / ElasticTTISG classes is approximate and may
    # produce spurious surface waves in strong-anisotropy regimes.  Until
    # this is verified against the Robertsson 1996 / Mittet 2002 analytic
    # benchmarks, free_surface=True is disabled here.

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Vertical P-wave velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon anisotropy parameter (ε ≥ δ required).", unit=""),
        ModelSpec("delta", description="Thomsen delta anisotropy parameter.", unit=""),
        ModelSpec("rho", aliases=("density",), description="Density.", unit="kg/m^3"),
    )

    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in x.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in z.", supports_receiver=True),
        FieldSpec("sH", aliases=("stress_h", "sigma_H"), description="Horizontal normal stress σ_11 = σ_22.", supports_source=True, supports_receiver=True),
        FieldSpec("sV", aliases=("stress_v", "sigma_V"), description="Vertical normal stress σ_33.", supports_source=True, supports_receiver=True),
        FieldSpec("m_sHx", description="CPML memory variable for ∂σ_H/∂x.", internal=True, boundary_related=True),
        FieldSpec("m_sVz", description="CPML memory variable for ∂σ_V/∂z.", internal=True, boundary_related=True),
        FieldSpec("m_vxx", description="CPML memory variable for ∂v_x/∂x.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for ∂v_z/∂z.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", free_surface=False):
        super().__init__(spatial_order, device, backend)
        if free_surface:
            raise NotImplementedError(
                "AcousticVTI1st does not yet support free_surface=True. "
                "The isotropic-style image method is physically incorrect for "
                "VTI (Robertsson 1996; Mittet 2002): the σ_V=0 constraint "
                "couples ∂v_z/∂z to ∂v_x/∂x via c_13, requiring a modified "
                "effective stiffness (c11 - c13²/c33) for the σ_H surface "
                "update. Use absorbing PML on all four sides for now. "
                "See TODO_free_surface on this class."
            )
        self.free_surface = False

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
        return ["sH", "sV"]

    @property
    def default_receiver_fields(self):
        return ["vz"]

    def prepare_models(self, models):
        """Pre-compute stiffness coefficients from Thomsen parameters.

        Warns (via ``warnings.warn``) when ε < δ anywhere, since this
        violates the stability condition identified by Grechka, Zhang, Rector
        (2004), Geophysics 69, 576.

        Cached outputs: c11 = ρV_P²(1+2ε), c33 = ρV_P², c13 = ρV_P²√(1+2δ),
        inv_rho = 1/ρ.

        References
        ----------
        Thomsen (1986) Geophysics 51, 1954–1966.  (ε, δ parameterisation.)
        Grechka et al. (2004) Geophysics 69, 576.  (ε ≥ δ stability.)
        """
        vp, epsilon, delta, rho = models
        # Stability warning
        diff = epsilon - delta
        try:
            import torch
            if isinstance(diff, torch.Tensor):
                min_diff = float(diff.min().item())
            else:
                import numpy as _np
                min_diff = float(_np.asarray(diff).min())
        except Exception:
            import numpy as _np
            min_diff = float(_np.asarray(diff).min())

        if min_diff < -1e-6:
            warnings.warn(
                f"AcousticVTI1st: min(ε − δ) = {min_diff:.3e} < 0. "
                "The acoustic VTI system is unstable when ε < δ "
                "(Grechka, Zhang, Rector 2004, Geophysics 69, 576). "
                "The time loop will NOT be terminated, but expect instability.",
                stacklevel=2,
            )

        c11, c33, c13, inv_rho = _build_stiffness(vp, epsilon, delta, rho)
        return [vp, epsilon, delta, rho, c11, c33, c13, inv_rho]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 8:
            _vp, _eps, _dlt, _rho, c11, c33, c13, inv_rho = models
        elif len(models) == 4:
            vp, epsilon, delta, rho = models
            c11, c33, c13, inv_rho = _build_stiffness(vp, epsilon, delta, rho)
        else:
            raise ValueError(f"AcousticVTI1st.func: expected 4 or 8 models, got {len(models)}")
        return step_vti_2d(
            *wavefields,
            c11, c33, c13, inv_rho,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )

    @classmethod
    def recommended_dt(cls, vp_max, epsilon_max, h, cfl=0.5):
        """CFL-stable time step for the 2-D fourth-order scheme.

        Δt ≤ CFL · h / (V_P_max · √(1 + 2 ε_max))

        Parameters
        ----------
        vp_max : float
            Maximum vertical P-wave velocity in the model (m/s).
        epsilon_max : float
            Maximum Thomsen ε in the model.
        h : float
            Grid spacing (m).
        cfl : float, optional
            CFL number.  Default 0.5 (safe for 4th-order spatial scheme).
        """
        return cfl * h / (vp_max * (1.0 + 2.0 * epsilon_max) ** 0.5)

    def _C(self):
        """Expose the compiled CUDA forward + Phase-1 backward bindings.

        * forward          — full CUDA forward (verified against eager,
                             ~62× speedup on 401×401).
        * backward         — Phase 1 (full mode): uses saved forward
                             wavefield; interior gradients correct, PML-band
                             gradients approximate.  Mask the PML in your
                             gradient if it matters.
        * backward_bs / _ckpt / _recursive_ckpt — Phase 2+, currently raise
                             TORCH_CHECK.  Use ``use_ckpt=False`` and
                             ``save_all_wavefields=True`` for full mode.
        """
        import sweep._C as _C
        return (
            _C.acoustic_vti_1st_2d_forward,
            _C.acoustic_vti_1st_2d_backward,
            _C.acoustic_vti_1st_2d_backward_bs,
            _C.acoustic_vti_1st_2d_backward_ckpt,
            _C.acoustic_vti_1st_2d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        """CUDA buffer layout.

        base_nvar = 4   (vx, vz, sH, sV)
        pml_nvar  = 4   (m_sHx, m_sVz, m_vxx, m_vzz)
        last_two_storage_nvar = 4  (snapshot of vx, vz, sH, sV)
        backward_workspace_nvar = 0  (Phase 1 backward doesn't need extra
                                       adjoint scratch — the 8 standard
                                       adjoint-wavefield tensors are passed
                                       via ``adjoint_wavefields``).
        """
        return CUDALayoutSpec(
            base_nvar=4,
            pml_nvar=4,
            last_two_nvar=1,
            last_two_storage_nvar=4,
            backward_workspace_nvar=0,
        )


# ---------------------------------------------------------------------------
# 3-D equation class
# ---------------------------------------------------------------------------

class AcousticVTI1st3D(FirstOrderEquation):
    """First-order acoustic VTI wave equation on a standard staggered grid (3-D).

    Extends ``AcousticVTI1st`` to 3-D with 5 wavefields (v_x, v_y, v_z,
    σ_H, σ_V) and the same 4 model parameters.  The horizontal isotropy of
    VTI media means σ_11 = σ_22 ≡ σ_H so only two distinct normal stresses
    are tracked.

    See ``AcousticVTI1st`` for full documentation.

    References
    ----------
    Duveneck et al. (2008) DOI: 10.1190/1.3059320
    """

    MODEL_SPECS = AcousticVTI1st.MODEL_SPECS

    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in x.", supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in y.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in z.", supports_receiver=True),
        FieldSpec("sH", aliases=("stress_h", "sigma_H"), description="Horizontal normal stress σ_11 = σ_22.", supports_source=True, supports_receiver=True),
        FieldSpec("sV", aliases=("stress_v", "sigma_V"), description="Vertical normal stress σ_33.", supports_source=True, supports_receiver=True),
        FieldSpec("m_sHx", description="CPML memory variable for ∂σ_H/∂x.", internal=True, boundary_related=True),
        FieldSpec("m_sHy", description="CPML memory variable for ∂σ_H/∂y.", internal=True, boundary_related=True),
        FieldSpec("m_sVz", description="CPML memory variable for ∂σ_V/∂z.", internal=True, boundary_related=True),
        FieldSpec("m_vxx", description="CPML memory variable for ∂v_x/∂x.", internal=True, boundary_related=True),
        FieldSpec("m_vyy", description="CPML memory variable for ∂v_y/∂y.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for ∂v_z/∂z.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", free_surface=False):
        super().__init__(spatial_order, device, backend, ndim=3)
        if free_surface:
            raise NotImplementedError(
                "AcousticVTI1st3D does not yet support free_surface=True. "
                "Same reason as the 2-D case — see AcousticVTI1st docstring "
                "(TODO_free_surface). Use absorbing PML on all six sides."
            )
        self.free_surface = False

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
        return ["sH", "sV"]

    @property
    def default_receiver_fields(self):
        return ["vz"]

    def prepare_models(self, models):
        """Pre-compute stiffness coefficients (identical logic to 2-D).

        See ``AcousticVTI1st.prepare_models`` for details.
        """
        vp, epsilon, delta, rho = models
        diff = epsilon - delta
        try:
            import torch
            if isinstance(diff, torch.Tensor):
                min_diff = float(diff.min().item())
            else:
                import numpy as _np
                min_diff = float(_np.asarray(diff).min())
        except Exception:
            import numpy as _np
            min_diff = float(_np.asarray(diff).min())

        if min_diff < -1e-6:
            warnings.warn(
                f"AcousticVTI1st3D: min(ε − δ) = {min_diff:.3e} < 0. "
                "Instability likely (Grechka et al. 2004).",
                stacklevel=2,
            )

        c11, c33, c13, inv_rho = _build_stiffness(vp, epsilon, delta, rho)
        return [vp, epsilon, delta, rho, c11, c33, c13, inv_rho]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 8:
            _vp, _eps, _dlt, _rho, c11, c33, c13, inv_rho = models
        elif len(models) == 4:
            vp, epsilon, delta, rho = models
            c11, c33, c13, inv_rho = _build_stiffness(vp, epsilon, delta, rho)
        else:
            raise ValueError(f"AcousticVTI1st3D.func: expected 4 or 8 models, got {len(models)}")
        return step_vti_3d(
            *wavefields,
            c11, c33, c13, inv_rho,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )

    @classmethod
    def recommended_dt(cls, vp_max, epsilon_max, h, cfl=0.5):
        """CFL-stable time step for the 3-D fourth-order scheme.

        Same formula as the 2-D case (the fast horizontal P velocity governs).
        """
        return cfl * h / (vp_max * (1.0 + 2.0 * epsilon_max) ** 0.5)

    @property
    def cuda_layout(self):
        return None

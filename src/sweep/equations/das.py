from __future__ import annotations

from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec


def _is_torch_tensor(value):
    return type(value).__module__.startswith("torch")


def _zeros_like(value):
    return value * 0


def _with_cpml(derivative, memory, a, b):
    memory = a * memory + b * derivative
    return derivative + memory, memory


def _dxx_cpml(u, pd, m_xf, m_xb, ax, bx, axh, bxh):
    ux = pd.x_forward(u)
    ux, m_xf = _with_cpml(ux, m_xf, axh, bxh)
    uxx = pd.x_backward(ux)
    uxx, m_xb = _with_cpml(uxx, m_xb, ax, bx)
    return uxx, m_xf, m_xb


def _dyy_cpml(u, pd, m_yf, m_yb, ay, by, ayh, byh):
    uy = pd.y_forward(u)
    uy, m_yf = _with_cpml(uy, m_yf, ayh, byh)
    uyy = pd.y_backward(uy)
    uyy, m_yb = _with_cpml(uyy, m_yb, ay, by)
    return uyy, m_yf, m_yb


def _dzz_cpml(u, pd, m_zf, m_zb, az, bz, azh, bzh):
    uz = pd.z_forward(u)
    uz, m_zf = _with_cpml(uz, m_zf, azh, bzh)
    uzz = pd.z_backward(uz)
    uzz, m_zb = _with_cpml(uzz, m_zb, az, bz)
    return uzz, m_zf, m_zb


def _gauge_cells(gauge_cells=None, gauge_length=None, spacing=1.0):
    if gauge_cells is not None:
        cells = int(gauge_cells)
    elif gauge_length is not None:
        cells = int(round(float(gauge_length) / float(spacing)))
    else:
        cells = 1
    return max(cells, 1)


def gauge_average(values, gauge_cells=None, *, gauge_length=None, spacing=1.0, axis=-1):
    """Average DAS strain-rate over a finite gauge length.

    This implements the finite-difference analogue of Eqs. (15)-(16) in the
    Zhao et al. DAS simulation paper: a centered moving average along the
    selected fiber axis. ``values`` may be a torch tensor or a NumPy array.
    """

    cells = _gauge_cells(gauge_cells, gauge_length, spacing)
    if cells <= 1:
        return values

    ndim = values.ndim
    axis = axis if axis >= 0 else ndim + axis
    left = cells // 2
    right = cells - 1 - left

    if _is_torch_tensor(values):
        import torch
        import torch.nn.functional as F

        moved = torch.movedim(values, axis, -1)
        shape = moved.shape
        flat = moved.reshape(-1, 1, shape[-1])
        padded = F.pad(flat, (left, right), mode="replicate")
        averaged = F.avg_pool1d(padded, kernel_size=cells, stride=1)
        averaged = averaged.reshape(shape)
        return torch.movedim(averaged, -1, axis)

    import numpy as np

    pad_width = [(0, 0)] * ndim
    pad_width[axis] = (left, right)
    padded = np.pad(values, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, cells, axis=axis)
    return windows.mean(axis=-1)


def helical_das_response(
    exx,
    ezz,
    eyy=None,
    *,
    angle=35.3,
    core_axis="x",
    gauge_cells=None,
    gauge_length=None,
    spacing=1.0,
    gauge_axis=-1,
):
    """Project normal strain-rate components to a regular helical DAS response.

    The paper neglects shear strain-rate for symmetrical periodic helices. For
    a 35.3 degree winding, the normal-component sensitivity ratio is 1:1:1.
    For 54.7 degrees, the component along the winding core has weight 4 and the
    two transverse components have weight 1. In 2D, pass ``eyy=None``.
    """

    if eyy is None:
        eyy = _zeros_like(exx)

    rounded = round(float(angle), 1)
    if abs(rounded - 35.3) < 1e-3:
        out = exx + eyy + ezz
    elif abs(rounded - 54.7) < 1e-3:
        core_axis = str(core_axis).lower()
        if core_axis == "x":
            out = 4 * exx + eyy + ezz
        elif core_axis == "y":
            out = exx + 4 * eyy + ezz
        elif core_axis == "z":
            out = exx + eyy + 4 * ezz
        else:
            raise ValueError("core_axis must be one of 'x', 'y', or 'z'.")
    else:
        raise ValueError("Only the paper's 35.3 and 54.7 degree helical windings are supported.")

    return gauge_average(out, gauge_cells, gauge_length=gauge_length, spacing=spacing, axis=gauge_axis)


def step_das_2d(
    exx,
    ezz,
    sxx,
    szz,
    txx,
    tzz,
    m_sxx_xf,
    m_sxx_xb,
    m_szz_zf,
    m_szz_zb,
    m_txx_zf,
    m_txx_zb,
    m_tzz_xf,
    m_tzz_xb,
    das35,
    das54x,
    das54z,
    vp,
    vs,
    rho,
    lame_lambda,
    lame_mu,
    dt,
    h,
    b,
    pd,
    pml=None,
):
    """One 2D stress and normal-strain-rate elastic step.

    This is the 2D, y-invariant reduction of Eq. (9) in Zhao et al. The dynamic
    variables are normal strain-rates ``exx``/``ezz``, normal stresses
    ``sxx``/``szz``, and auxiliary variables ``txx``/``tzz``.
    """

    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    dxx_sxx, m_sxx_xf, m_sxx_xb = _dxx_cpml(sxx, pd, m_sxx_xf, m_sxx_xb, ax, bx, axh, bxh)
    dzz_szz, m_szz_zf, m_szz_zb = _dzz_cpml(szz, pd, m_szz_zf, m_szz_zb, az, bz, azh, bzh)
    dzz_txx, m_txx_zf, m_txx_zb = _dzz_cpml(txx, pd, m_txx_zf, m_txx_zb, az, bz, azh, bzh)
    dxx_tzz, m_tzz_xf, m_tzz_xb = _dxx_cpml(tzz, pd, m_tzz_xf, m_tzz_xb, ax, bx, axh, bxh)

    shear_xz = dzz_txx + dxx_tzz

    exx = exx + dt / rho * (dxx_sxx + shear_xz)
    ezz = ezz + dt / rho * (dzz_szz + shear_xz)

    sxx = sxx + dt * ((lame_lambda + 2 * lame_mu) * exx + lame_lambda * ezz)
    szz = szz + dt * ((lame_lambda + 2 * lame_mu) * ezz + lame_lambda * exx)
    txx = txx + dt * lame_mu * exx
    tzz = tzz + dt * lame_mu * ezz

    das35 = helical_das_response(exx, ezz, angle=35.3)
    das54x = helical_das_response(exx, ezz, angle=54.7, core_axis="x")
    das54z = helical_das_response(exx, ezz, angle=54.7, core_axis="z")

    return (
        exx,
        ezz,
        sxx,
        szz,
        txx,
        tzz,
        m_sxx_xf,
        m_sxx_xb,
        m_szz_zf,
        m_szz_zb,
        m_txx_zf,
        m_txx_zb,
        m_tzz_xf,
        m_tzz_xb,
        das35,
        das54x,
        das54z,
    )


def step_das_3d(
    exx,
    eyy,
    ezz,
    sxx,
    syy,
    szz,
    txx,
    tyy,
    tzz,
    m_sxx_xf,
    m_sxx_xb,
    m_syy_yf,
    m_syy_yb,
    m_szz_zf,
    m_szz_zb,
    m_txx_yf,
    m_txx_yb,
    m_txx_zf,
    m_txx_zb,
    m_tyy_xf,
    m_tyy_xb,
    m_tyy_zf,
    m_tyy_zb,
    m_tzz_xf,
    m_tzz_xb,
    m_tzz_yf,
    m_tzz_yb,
    das35,
    das54x,
    das54y,
    das54z,
    vp,
    vs,
    rho,
    lame_lambda,
    lame_mu,
    dt,
    h,
    b,
    pd,
    pml=None,
):
    """One 3D stress and normal-strain-rate elastic step from Eq. (9)."""

    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml

    dxx_sxx, m_sxx_xf, m_sxx_xb = _dxx_cpml(sxx, pd, m_sxx_xf, m_sxx_xb, ax, bx, axh, bxh)
    dyy_syy, m_syy_yf, m_syy_yb = _dyy_cpml(syy, pd, m_syy_yf, m_syy_yb, ay, by, ayh, byh)
    dzz_szz, m_szz_zf, m_szz_zb = _dzz_cpml(szz, pd, m_szz_zf, m_szz_zb, az, bz, azh, bzh)
    dyy_txx, m_txx_yf, m_txx_yb = _dyy_cpml(txx, pd, m_txx_yf, m_txx_yb, ay, by, ayh, byh)
    dzz_txx, m_txx_zf, m_txx_zb = _dzz_cpml(txx, pd, m_txx_zf, m_txx_zb, az, bz, azh, bzh)
    dxx_tyy, m_tyy_xf, m_tyy_xb = _dxx_cpml(tyy, pd, m_tyy_xf, m_tyy_xb, ax, bx, axh, bxh)
    dzz_tyy, m_tyy_zf, m_tyy_zb = _dzz_cpml(tyy, pd, m_tyy_zf, m_tyy_zb, az, bz, azh, bzh)
    dxx_tzz, m_tzz_xf, m_tzz_xb = _dxx_cpml(tzz, pd, m_tzz_xf, m_tzz_xb, ax, bx, axh, bxh)
    dyy_tzz, m_tzz_yf, m_tzz_yb = _dyy_cpml(tzz, pd, m_tzz_yf, m_tzz_yb, ay, by, ayh, byh)

    exx = exx + dt / rho * (dxx_sxx + dyy_txx + dxx_tyy + dzz_txx + dxx_tzz)
    eyy = eyy + dt / rho * (dyy_syy + dyy_txx + dxx_tyy + dzz_tyy + dyy_tzz)
    ezz = ezz + dt / rho * (dzz_szz + dzz_txx + dxx_tzz + dzz_tyy + dyy_tzz)

    div_e = exx + eyy + ezz
    sxx = sxx + dt * (lame_lambda * div_e + 2 * lame_mu * exx)
    syy = syy + dt * (lame_lambda * div_e + 2 * lame_mu * eyy)
    szz = szz + dt * (lame_lambda * div_e + 2 * lame_mu * ezz)
    txx = txx + dt * lame_mu * exx
    tyy = tyy + dt * lame_mu * eyy
    tzz = tzz + dt * lame_mu * ezz

    das35 = helical_das_response(exx, ezz, eyy, angle=35.3)
    das54x = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="x")
    das54y = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="y")
    das54z = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="z")

    return (
        exx,
        eyy,
        ezz,
        sxx,
        syy,
        szz,
        txx,
        tyy,
        tzz,
        m_sxx_xf,
        m_sxx_xb,
        m_syy_yf,
        m_syy_yb,
        m_szz_zf,
        m_szz_zb,
        m_txx_yf,
        m_txx_yb,
        m_txx_zf,
        m_txx_zb,
        m_tyy_xf,
        m_tyy_xb,
        m_tyy_zf,
        m_tyy_zb,
        m_tzz_xf,
        m_tzz_xb,
        m_tzz_yf,
        m_tzz_yb,
        das35,
        das54x,
        das54y,
        das54z,
    )


class DASElastic(FirstOrderEquation):
    """2D stress and normal-strain-rate elastic DAS equation.

    Parameter order: ``vp, vs, rho``. Receiver fields ``exx`` and ``ezz`` are
    the straight-fiber normal strain-rate components. ``das35`` and ``das54x``
    / ``das54z`` are regular helical-fiber projections from Eqs. (13)-(14).
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("exx", aliases=("strain_rate_xx", "epsilon_dot_xx"), description="Normal strain-rate in the x direction.", supports_receiver=True),
        FieldSpec("ezz", aliases=("strain_rate_zz", "epsilon_dot_zz"), description="Normal strain-rate in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("txx", aliases=("tau_xx",), description="Auxiliary stress-like variable tau_xx.", internal=True),
        FieldSpec("tzz", aliases=("tau_zz",), description="Auxiliary stress-like variable tau_zz.", internal=True),
        FieldSpec("m_sxx_xf", description="CPML memory variable for forward d(sxx)/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxx_xb", description="CPML memory variable for backward d2(sxx)/dx2.", internal=True, boundary_related=True),
        FieldSpec("m_szz_zf", description="CPML memory variable for forward d(szz)/dz.", internal=True, boundary_related=True),
        FieldSpec("m_szz_zb", description="CPML memory variable for backward d2(szz)/dz2.", internal=True, boundary_related=True),
        FieldSpec("m_txx_zf", description="CPML memory variable for forward d(txx)/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txx_zb", description="CPML memory variable for backward d2(txx)/dz2.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_xf", description="CPML memory variable for forward d(tzz)/dx.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_xb", description="CPML memory variable for backward d2(tzz)/dx2.", internal=True, boundary_related=True),
        FieldSpec("das35", aliases=("helical35",), description="35.3 degree helical-fiber axial strain-rate.", supports_receiver=True),
        FieldSpec("das54x", aliases=("helical54", "helical54x"), description="54.7 degree helical-fiber axial strain-rate for x-oriented core.", supports_receiver=True),
        FieldSpec("das54z", aliases=("helical54z",), description="54.7 degree helical-fiber axial strain-rate for z-oriented core.", supports_receiver=True),
    )

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        super().__init__(spatial_order, device, backend, ndim=2)

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
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx", "ezz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]

    def func(self, *args, **kwargs):
        if len(args) == 25:
            return step_das_2d(*args, pd=self.pd, pml=self.b, **kwargs)
        if len(args) == 23:
            wavefields = args[:17]
            vp, vs, rho = args[17:20]
            dt, h, b = args[20:23]
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
            return step_das_2d(*wavefields, vp, vs, rho, lame_lambda, lame_mu, dt, h, b, pd=self.pd, pml=self.b, **kwargs)
        raise ValueError(f"DASElastic.func expected 23 or 25 positional args, got {len(args)}")


class DASElastic3D(FirstOrderEquation):
    """3D stress and normal-strain-rate elastic DAS equation from Eq. (9)."""

    MODEL_SPECS = DASElastic.MODEL_SPECS
    FIELD_SPECS = (
        FieldSpec("exx", aliases=("strain_rate_xx", "epsilon_dot_xx"), description="Normal strain-rate in the x direction.", supports_receiver=True),
        FieldSpec("eyy", aliases=("strain_rate_yy", "epsilon_dot_yy"), description="Normal strain-rate in the y direction.", supports_receiver=True),
        FieldSpec("ezz", aliases=("strain_rate_zz", "epsilon_dot_zz"), description="Normal strain-rate in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("syy", aliases=("stress_yy",), description="Normal stress in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("txx", aliases=("tau_xx",), description="Auxiliary stress-like variable tau_xx.", internal=True),
        FieldSpec("tyy", aliases=("tau_yy",), description="Auxiliary stress-like variable tau_yy.", internal=True),
        FieldSpec("tzz", aliases=("tau_zz",), description="Auxiliary stress-like variable tau_zz.", internal=True),
        FieldSpec("m_sxx_xf", description="CPML memory variable for forward d(sxx)/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxx_xb", description="CPML memory variable for backward d2(sxx)/dx2.", internal=True, boundary_related=True),
        FieldSpec("m_syy_yf", description="CPML memory variable for forward d(syy)/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syy_yb", description="CPML memory variable for backward d2(syy)/dy2.", internal=True, boundary_related=True),
        FieldSpec("m_szz_zf", description="CPML memory variable for forward d(szz)/dz.", internal=True, boundary_related=True),
        FieldSpec("m_szz_zb", description="CPML memory variable for backward d2(szz)/dz2.", internal=True, boundary_related=True),
        FieldSpec("m_txx_yf", description="CPML memory variable for forward d(txx)/dy.", internal=True, boundary_related=True),
        FieldSpec("m_txx_yb", description="CPML memory variable for backward d2(txx)/dy2.", internal=True, boundary_related=True),
        FieldSpec("m_txx_zf", description="CPML memory variable for forward d(txx)/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txx_zb", description="CPML memory variable for backward d2(txx)/dz2.", internal=True, boundary_related=True),
        FieldSpec("m_tyy_xf", description="CPML memory variable for forward d(tyy)/dx.", internal=True, boundary_related=True),
        FieldSpec("m_tyy_xb", description="CPML memory variable for backward d2(tyy)/dx2.", internal=True, boundary_related=True),
        FieldSpec("m_tyy_zf", description="CPML memory variable for forward d(tyy)/dz.", internal=True, boundary_related=True),
        FieldSpec("m_tyy_zb", description="CPML memory variable for backward d2(tyy)/dz2.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_xf", description="CPML memory variable for forward d(tzz)/dx.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_xb", description="CPML memory variable for backward d2(tzz)/dx2.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_yf", description="CPML memory variable for forward d(tzz)/dy.", internal=True, boundary_related=True),
        FieldSpec("m_tzz_yb", description="CPML memory variable for backward d2(tzz)/dy2.", internal=True, boundary_related=True),
        FieldSpec("das35", aliases=("helical35",), description="35.3 degree helical-fiber axial strain-rate.", supports_receiver=True),
        FieldSpec("das54x", aliases=("helical54", "helical54x"), description="54.7 degree helical-fiber axial strain-rate for x-oriented core.", supports_receiver=True),
        FieldSpec("das54y", aliases=("helical54y",), description="54.7 degree helical-fiber axial strain-rate for y-oriented core.", supports_receiver=True),
        FieldSpec("das54z", aliases=("helical54z",), description="54.7 degree helical-fiber axial strain-rate for z-oriented core.", supports_receiver=True),
    )

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        super().__init__(spatial_order, device, backend, ndim=3)

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
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx", "eyy", "ezz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]

    def func(self, *args, **kwargs):
        if len(args) == 39:
            return step_das_3d(*args, pd=self.pd, pml=self.b, **kwargs)
        if len(args) == 37:
            wavefields = args[:31]
            vp, vs, rho = args[31:34]
            dt, h, b = args[34:37]
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
            return step_das_3d(*wavefields, vp, vs, rho, lame_lambda, lame_mu, dt, h, b, pd=self.pd, pml=self.b, **kwargs)
        raise ValueError(f"DASElastic3D.func expected 37 or 39 positional args, got {len(args)}")


DAS = DASElastic
DAS3D = DASElastic3D

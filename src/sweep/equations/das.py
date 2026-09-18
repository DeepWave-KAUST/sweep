from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ._free_surface import zero_top_row
from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec, history_fields, record_multi

from .fields import FieldSpec, ModelSpec
from ._registry import register_equation
from .elastic3d import step as _elastic3d_step
from ._elastic_step_core import elastic_stress_substep, elastic_velocity_substep


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



_ZHAO_DAS_METHODS = {
    "zhao",
}
_MU_DAS_METHODS = {
    "mu",
}

_MU_STRAIN_RATE_FIELDS = {
    "exx_t": "exx",
    "eyy_t": "eyy",
    "ezz_t": "ezz",
    "exy_t": "exy",
    "exz_t": "exz",
    "eyz_t": "eyz",
}
_MU_DAS_RATE_FIELDS = {
    "das35_t",
    "das54x_t",
    "das54y_t",
    "das54z_t",
}


def _normalize_das_method(method: str) -> str:
    method = str(method).lower()
    if method in _ZHAO_DAS_METHODS:
        return "zhao"
    if method in _MU_DAS_METHODS:
        return "mu"
    raise ValueError("DAS method must be 'zhao' or 'mu'.")


def _append_unique(fields: list[str], name: str) -> None:
    if name not in fields:
        fields.append(name)


def _receiver_fields_from_specs(equation) -> set[str]:
    return {spec.name for spec in equation.field_specs if spec.supports_receiver}


def _mu_solver_receiver_type(equation, receiver_type: Sequence[str]) -> tuple[str, ...]:
    physical_fields = _receiver_fields_from_specs(equation)
    solver_fields: list[str] = []
    unsupported: list[str] = []

    for field in receiver_type:
        if field in physical_fields:
            _append_unique(solver_fields, field)
        elif field in _MU_STRAIN_RATE_FIELDS:
            _append_unique(solver_fields, _MU_STRAIN_RATE_FIELDS[field])
        elif field in _MU_DAS_RATE_FIELDS:
            _append_unique(solver_fields, "exx")
            _append_unique(solver_fields, "ezz")
            if "eyy" in physical_fields:
                _append_unique(solver_fields, "eyy")
        else:
            unsupported.append(field)

    if unsupported:
        raise ValueError(f"Unsupported Mu DAS receiver fields: {unsupported}.")
    return tuple(solver_fields)


def _standardize_das_record_layout(record: torch.Tensor, *, nreceivers: int, receiver_type: Sequence[str]) -> torch.Tensor:
    """Return DAS records as ``(batch, nt, nrec, nfield)``."""

    if not isinstance(record, torch.Tensor):
        raise TypeError("record must be a torch.Tensor.")
    if record.ndim != 4:
        raise ValueError(f"record must be 4D, got shape {tuple(record.shape)}.")

    nfields = len(receiver_type)
    if record.shape[-1] == nfields and record.shape[-2] == nreceivers:
        return record
    if record.shape[0] == nfields and record.shape[2] == nreceivers:
        return record.permute(1, 3, 2, 0)

    raise ValueError(
        "Could not identify DAS record layout. Expected eager layout "
        f"(batch, nt, {nreceivers}, {nfields}) or c layout "
        f"({nfields}, batch, {nreceivers}, nt), got {tuple(record.shape)}."
    )


class DAS(torch.nn.Module):
    """Unified Torch DAS (distributed acoustic sensing) modeling interface.

    Facade ``nn.Module`` that wires one of the underlying DAS equations
    to a :class:`PropTorch` propagator and post-processes records into
    a uniform ``(batch, nt, nrec, nfield)`` layout. Two methods:

    * ``method="zhao"`` — runs the Zhao (2022) stress / normal-strain-rate
      equation (:class:`DASZhao` / :class:`DASZhao3D`); receivers are
      strain-rates already, no time differentiation needed.
    * ``method="mu"`` — runs the Mu (Mu & Hung 2022) velocity-stress
      equation augmented with integrated strain (:class:`DASMu` /
      :class:`DASMu3D`); strain-rate (``exx_t`` etc.) and helical DAS
      receivers (``das35_t``, ``das54x_t``, …) are derived by numerical
      time differentiation after the solve, which requires ``dt`` to
      be passed through ``propagator_kwargs``.

    Reference: Zhao Y. et al. 2022 (DAS strain-rate equation); Mu &
    Hung 2022 (velocity-stress-strain coupling).

    Note:
        This class used to be called ``DASModeler``; the shorter name
        mirrors the DAS field convention and matches the
        :class:`AcousticAniso` facade for anisotropic acoustic
        equations. ``DASModeler`` is kept as a back-compat alias at
        the bottom of this module.
    """

    def __init__(
        self,
        *,
        shape,
        method: str = "zhao",
        ndim: int | None = None,
        spatial_order: int = 4,
        source_type: Sequence[str] | None = None,
        receiver_type: Sequence[str] | None = None,
        gauge_cells: int | None = None,
        gauge_length: float | None = None,
        gauge_axis: int = -1,
        backend: str = "torch",
        impl: str | None = None,
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        **propagator_kwargs,
    ):
        """Build the DAS facade.

        Args:
            shape: Spatial model shape ``(nz, nx)`` for 2-D or
                ``(nz, ny, nx)`` for 3-D.
            method: DAS formulation. One of ``'zhao'`` (stress /
                normal-strain-rate, :class:`DASZhao` / :class:`DASZhao3D`)
                or ``'mu'`` (velocity-stress + integrated strain,
                :class:`DASMu` / :class:`DASMu3D`). Defaults to
                ``'zhao'``.
            ndim: Spatial dimensionality (2 or 3). Inferred from
                ``len(shape)`` when ``None``. Defaults to ``None``.
            spatial_order: FD accuracy order — e.g. ``spatial_order=4`` is
                fourth-order accurate. Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Even integer. Defaults to 4.
            source_type: Names of fields injected by the source.
                Defaults to the underlying equation's
                ``default_source_fields``.
            receiver_type: Names of receiver fields to extract. Mu can
                emit strain-rate (``exx_t``, ``ezz_t``, …) and helical
                DAS-rate channels (``das35_t``, ``das54x_t``,
                ``das54y_t``, ``das54z_t``); these require ``dt``.
                Defaults to the underlying equation's
                ``default_receiver_fields``.
            gauge_cells: Gauge length in number of cells along the
                fibre axis for helical-fibre averaging. Mutually
                exclusive with ``gauge_length``. Defaults to ``None``
                (no averaging).
            gauge_length: Gauge length in metres along the fibre axis;
                converted to cells using the propagator's ``dh``.
                Defaults to ``None``.
            gauge_axis: Axis along which the gauge moving-average is
                applied. Defaults to ``-1``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``.
            impl: Propagator implementation. ``'eager'`` for the
                PyTorch / JAX time loop, ``'c'`` for the compiled CUDA
                kernel. Defaults to ``None`` (propagator decides).
            backend_options: Forwarded to :class:`PropTorch` as
                backend-level options.
            eager_options: Forwarded to :class:`PropTorch` for the
                eager path (e.g. checkpointing).
            cuda_options: Forwarded to :class:`PropTorch` for the
                compiled CUDA path.
            **propagator_kwargs: Extra propagator kwargs (``dh``,
                ``dt``, ``abcn``, ``dev``, …).
        """
        super().__init__()
        self.shape = tuple(shape)
        self.ndim = int(len(self.shape) if ndim is None else ndim)
        if self.ndim not in (2, 3):
            raise ValueError("DAS only supports ndim=2 or ndim=3.")

        self.method = _normalize_das_method(method)
        self.spatial_order = int(spatial_order)
        self.gauge_cells = gauge_cells
        self.gauge_length = gauge_length
        self.gauge_axis = int(gauge_axis)
        self.dh = propagator_kwargs.setdefault("dh", 10.0)
        self.dt = propagator_kwargs.get("dt")

        equation_device = propagator_kwargs.get("dev") or "cpu"
        if self.method == "mu":
            equation_cls = DASMu if self.ndim == 2 else DASMu3D
        else:
            equation_cls = DASZhao if self.ndim == 2 else DASZhao3D
        equation = equation_cls(spatial_order=self.spatial_order, device=equation_device, backend="torch")
        self.receiver_type = tuple(equation.default_receiver_fields if receiver_type is None else receiver_type)
        if self.method == "mu":
            solver_receiver_type = _mu_solver_receiver_type(equation, self.receiver_type)
            if set(self.receiver_type) - set(solver_receiver_type) and self.dt is None:
                raise ValueError("Mu strain-rate and DAS-rate receiver fields require dt.")
        else:
            solver_receiver_type = self.receiver_type
        source_type = list(equation.default_source_fields if source_type is None else source_type)

        self.source_type = tuple(source_type)
        self.solver_receiver_type = tuple(solver_receiver_type)

        from sweep.propagator.torch import PropTorch

        self.solver = PropTorch(
            equation,
            shape=self.shape,
            source_type=list(self.source_type),
            receiver_type=list(self.solver_receiver_type),
            backend=backend,
            impl=impl,
            backend_options=backend_options,
            eager_options=eager_options,
            cuda_options=cuda_options,
            **propagator_kwargs,
        )

    @property
    def channels(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.receiver_type)}

    def forward(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        adj=False,
        return_wavefield=False,
        **kwargs,
    ):
        # (no source_encoding parameter: the solver derives the encoding mode
        # from the wavelet/sources shapes; the old parameter was forwarded to a
        # forward() that never read it.)
        if return_wavefield:
            raise NotImplementedError("DAS facade returns records only. Use PropTorch directly for wavefields.")

        record = self.solver(
            wavelet,
            sources=sources,
            receivers=receivers,
            models=models,
            adj=adj,
            return_wavefield=False,
            **kwargs,
        )
        nreceivers = int(np.asarray(receivers).shape[-2])
        record = _standardize_das_record_layout(
            record,
            nreceivers=nreceivers,
            receiver_type=self.solver_receiver_type,
        )
        if self.method == "mu" and self.solver_receiver_type != self.receiver_type:
            record = self._postprocess_mu_record(record)
        return record

    def _mu_time_derivative(self, values: torch.Tensor) -> torch.Tensor:
        if self.dt is None:
            raise ValueError("Mu strain-rate receiver fields require dt.")
        dt = float(self.dt)
        first = values[:, :1] / dt
        rest = (values[:, 1:] - values[:, :-1]) / dt
        return torch.cat([first, rest], dim=1)

    def _postprocess_mu_record(self, record: torch.Tensor) -> torch.Tensor:
        source_index = {name: index for index, name in enumerate(self.solver_receiver_type)}
        values = {name: record[..., index] for name, index in source_index.items()}

        needs_das_rates = any(field in self.receiver_type for field in _MU_DAS_RATE_FIELDS)
        for rate_name, strain_name in _MU_STRAIN_RATE_FIELDS.items():
            if rate_name in self.receiver_type or (needs_das_rates and rate_name in {"exx_t", "eyy_t", "ezz_t"}):
                if strain_name in values:
                    values[rate_name] = self._mu_time_derivative(values[strain_name])

        if needs_das_rates:
            exx_t = values.get("exx_t")
            eyy_t = values.get("eyy_t")
            ezz_t = values.get("ezz_t")
            if exx_t is None or ezz_t is None:
                raise ValueError("Mu DAS-rate receiver fields require exx/ezz strain records.")
            values["das35_t"] = helical_das_response(
                exx_t,
                ezz_t,
                eyy_t,
                angle=35.3,
                gauge_cells=self.gauge_cells,
                gauge_length=self.gauge_length,
                spacing=self.dh,
                gauge_axis=self.gauge_axis,
            )
            values["das54x_t"] = helical_das_response(
                exx_t,
                ezz_t,
                eyy_t,
                angle=54.7,
                core_axis="x",
                gauge_cells=self.gauge_cells,
                gauge_length=self.gauge_length,
                spacing=self.dh,
                gauge_axis=self.gauge_axis,
            )
            values["das54z_t"] = helical_das_response(
                exx_t,
                ezz_t,
                eyy_t,
                angle=54.7,
                core_axis="z",
                gauge_cells=self.gauge_cells,
                gauge_length=self.gauge_length,
                spacing=self.dh,
                gauge_axis=self.gauge_axis,
            )
            values["das54y_t"] = helical_das_response(
                exx_t,
                ezz_t,
                eyy_t,
                angle=54.7,
                core_axis="y",
                gauge_cells=self.gauge_cells,
                gauge_length=self.gauge_length,
                spacing=self.dh,
                gauge_axis=self.gauge_axis,
            )

        return torch.stack([values[field] for field in self.receiver_type], dim=-1)


def step_das_mu_2d(
    vx,
    vz,
    sxx,
    szz,
    sxz,
    exx,
    ezz,
    exz,
    m_vxx,
    m_vxz,
    m_vzx,
    m_vzz,
    m_txxx,
    m_txxz,
    m_tzzx,
    m_tzzz,
    m_txzx,
    m_txzz,
    vp,
    vs,
    rho,
    lame_lambda,
    lame_mu,
    lame_lambda_2mu,
    dt,
    h,
    b,
    pd,
    pml=None,
    free_surface=False,
):
    """One Mu velocity-stress-strain DAS step in 2D.

    This IS the shared elastic pair (:func:`elastic_velocity_substep` +
    :func:`elastic_stress_substep`) plus three strain integrations
    (``exx_t = vx_x``, ``ezz_t = vz_z``, ``exz_t = 0.5 (vx_z + vz_x)``) -- the
    bodies were maintained as literal copies apart from variable names and the
    staggered-parameter generality (DASMu uses the plain ``rho`` and
    ``lame_mu`` where elastic passes staggered averages, so those are forwarded
    as-is and the arithmetic is unchanged). The post-step ``szz`` zeroing is the
    caller's job by the sub-step contract, and happens here as before.
    """
    (vx, vz, sxx, szz, sxz,
     m_vxx, m_vxz, m_vzx, m_vzz,
     m_txxx, m_txxz, m_tzzx, m_tzzz,
     m_txzx, m_txzz) = elastic_velocity_substep(
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
        lame_lambda=lame_lambda, lame_mu=lame_mu, mu_xz=lame_mu,
        rho_x=rho, rho_z=rho,
        dt=dt, h=h, b=b, pd=pd, pml=pml,
        free_surface=free_surface,
        lame_lambda_2mu=lame_lambda_2mu,
    )
    *state, (vx_x, vz_z, vx_z, vz_x) = elastic_stress_substep(
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
        lame_lambda=lame_lambda, lame_mu=lame_mu, mu_xz=lame_mu,
        rho_x=rho, rho_z=rho,
        dt=dt, h=h, b=b, pd=pd, pml=pml,
        free_surface=free_surface,
        lame_lambda_2mu=lame_lambda_2mu,
        return_gradients=True,
    )
    (vx, vz, sxx, szz, sxz,
     m_vxx, m_vxz, m_vzx, m_vzz,
     m_txxx, m_txxz, m_tzzx, m_tzzz,
     m_txzx, m_txzz) = state

    exx = exx + dt * vx_x
    ezz = ezz + dt * vz_z
    exz = exz + 0.5 * dt * (vx_z + vz_x)

    if free_surface:
        szz = zero_top_row(szz, pd.coes.shape[0], axis=-2)

    return (
        vx,
        vz,
        sxx,
        szz,
        sxz,
        exx,
        ezz,
        exz,
        m_vxx,
        m_vxz,
        m_vzx,
        m_vzz,
        m_txxx,
        m_txxz,
        m_tzzx,
        m_tzzz,
        m_txzx,
        m_txzz,
    )


def step_das_mu_3d(
    vx,
    vy,
    vz,
    sxx,
    syy,
    szz,
    sxy,
    sxz,
    syz,
    exx,
    eyy,
    ezz,
    exy,
    exz,
    eyz,
    m_vxx,
    m_vxy,
    m_vxz,
    m_vyx,
    m_vyy,
    m_vyz,
    m_vzx,
    m_vzy,
    m_vzz,
    m_sxxx,
    m_szzz,
    m_sxyx,
    m_sxyy,
    m_sxzx,
    m_sxzz,
    m_syyy,
    m_syzy,
    m_syzz,
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
    free_surface=False,
):
    """One Mu velocity-stress-strain DAS step in 3D.

    This IS :func:`elastic3d.step` plus six strain integrations -- the two
    bodies were maintained as ~130-line literal copies, verified statement-for-
    statement identical before being merged (with ``topo_rows=None`` the
    elastic free-surface helper reduces by construction to the very functions
    this copy called). ``return_gradients=True`` hands back the CPML-corrected
    velocity gradients the stress update consumed, which are exactly what the
    strain-rate state integrates.
    """
    *state, (dvx_dx, dvy_dy, dvz_dz, dvx_dy, dvy_dx,
             dvx_dz, dvz_dx, dvy_dz, dvz_dy) = _elastic3d_step(
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
        m_vxx, m_vxy, m_vxz,
        m_vyx, m_vyy, m_vyz,
        m_vzx, m_vzy, m_vzz,
        m_sxxx, m_szzz,
        m_sxyx, m_sxyy,
        m_sxzx, m_sxzz,
        m_syyy,
        m_syzy, m_syzz,
        vp, vs, rho,
        lame_lambda, lame_mu,
        dt, h, b, pd,
        pml=pml,
        free_surface=free_surface,
        return_gradients=True,
    )
    (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
     m_vxx, m_vxy, m_vxz,
     m_vyx, m_vyy, m_vyz,
     m_vzx, m_vzy, m_vzz,
     m_sxxx, m_szzz,
     m_sxyx, m_sxyy,
     m_sxzx, m_sxzz,
     m_syyy,
     m_syzy, m_syzz) = state

    exx = exx + dt * dvx_dx
    eyy = eyy + dt * dvy_dy
    ezz = ezz + dt * dvz_dz
    exy = exy + 0.5 * dt * (dvx_dy + dvy_dx)
    exz = exz + 0.5 * dt * (dvx_dz + dvz_dx)
    eyz = eyz + 0.5 * dt * (dvy_dz + dvz_dy)

    return (
        vx,
        vy,
        vz,
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        exx,
        eyy,
        ezz,
        exy,
        exz,
        eyz,
        m_vxx,
        m_vxy,
        m_vxz,
        m_vyx,
        m_vyy,
        m_vyz,
        m_vzx,
        m_vzy,
        m_vzz,
        m_sxxx,
        m_szzz,
        m_sxyx,
        m_sxyy,
        m_sxzx,
        m_sxzz,
        m_syyy,
        m_syzy,
        m_syzz,
    )


def step_das_zhao_2d(
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
    das35_t,
    das54x_t,
    das54z_t,
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

    das35_t = helical_das_response(exx, ezz, angle=35.3)
    das54x_t = helical_das_response(exx, ezz, angle=54.7, core_axis="x")
    das54z_t = helical_das_response(exx, ezz, angle=54.7, core_axis="z")

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
        das35_t,
        das54x_t,
        das54z_t,
    )


def step_das_zhao_3d(
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
    das35_t,
    das54x_t,
    das54y_t,
    das54z_t,
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

    das35_t = helical_das_response(exx, ezz, eyy, angle=35.3)
    das54x_t = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="x")
    das54y_t = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="y")
    das54z_t = helical_das_response(exx, ezz, eyy, angle=54.7, core_axis="z")

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
        das35_t,
        das54x_t,
        das54y_t,
        das54z_t,
    )


def _das_derived_model_nvar(mode):
    """mu and lambda (common/derived_models.h LameSlot) wherever the DAS drivers
    step the elastic forward: the forward itself, the boundary-saving backward's
    reconstruction and the checkpoint replays. The full-mode backward reads the
    stored strain history and derives nothing, so it gets no slots."""
    return 0 if mode == "full" else 2


def _das2d_adjoint_workspace(B, nt, shape, mode):
    """The compiled 2-D DAS backward's scratch (das2d/backward.cu WorkspaceSlot),
    one padded grid per shot each: a read-only zero strain and eight adjoint
    derivative grids in every mode; boundary saving also keeps the two current
    strains and four derivative temporaries alive through its adjoint loop, so
    that mode -- and only that mode -- gets six more.
    """
    n = 15 if mode == "bs" else 9
    return n * [[B, 1, *shape]]


@register_equation(aliases=('DASElastic',))
class DASZhao(FirstOrderEquation):
    """First-order 2-D stress / normal-strain-rate DAS equation (Zhao 2022).

    The dynamic variables are normal strain-rates ``exx_t``, ``ezz_t``,
    normal stresses ``sxx``, ``szz``, and auxiliary stress-like
    components ``txx``, ``tzz``. The system is the 2-D, y-invariant
    reduction of Eq. (9) of the Zhao et al. paper; helical-fibre
    strain-rate projections (``das35_t``, ``das54x_t``, ``das54z_t``)
    are computed inline from ``exx_t`` and ``ezz_t``. Staggered-grid
    CPML (``cpmls``, 8 profiles).

    Reference: Zhao Y. et al. 2022, *DAS modelling using a stress and
    normal-strain-rate elastic formulation*.

    
    """

    C_NAME = "das2d"

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("exx_t", description="Normal strain-rate in the x direction.", supports_receiver=True),
        FieldSpec("ezz_t", description="Normal strain-rate in the z direction.", supports_receiver=True),
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
        FieldSpec("das35_t", description="35.3 degree helical-fiber axial strain-rate.", supports_receiver=True),
        FieldSpec("das54x_t", description="54.7 degree helical-fiber axial strain-rate for x-oriented core.", supports_receiver=True),
        FieldSpec("das54z_t", description="54.7 degree helical-fiber axial strain-rate for z-oriented core.", supports_receiver=True),
    )

    default_pml_type = "cpmls"  # staggered-grid CPML: step_das_zhao_2d unpacks 8 profiles

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 2-D Zhao DAS equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator (forward / backward FD pairs along x and z) — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be
                an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/das2d/forward.cu``)
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
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx_t", "ezz_t"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 5:
            vp, vs, rho, lame_lambda, lame_mu = models
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
        else:
            raise ValueError(f"DASZhao.func expected 3 or 5 models, got {len(models)}")
        return step_das_zhao_2d(*wavefields, vp, vs, rho, lame_lambda, lame_mu, dt, h, b, pd=self.pd, pml=self.b, **kwargs)

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            # The checkpoint modes recompute the whole strain history (nt steps);
            # the buffer is the same shape as the full-mode u_allt.
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: [(nt, 2, B, *grid)],
            save_all_shape=history_fields(2),
            base_nvar=9,
            pml_nvar=8,
            last_two_nvar=1,
            last_two_storage_nvar=9,
            backward_workspace_shapes=_das2d_adjoint_workspace,
            # The four per-step derivative scratch grids the compiled forward
            # used to allocate itself (das2d/forward.cu ForwardWorkspaceSlot).
            forward_workspace_nvar=4,
            bs_reconstruction_nvar=9,   # exx, ezz, sxx, szz, txx, tzz + das35, das54x, das54z
            derived_model_nvar=_das_derived_model_nvar,
        )


@register_equation(aliases=('DASElastic3D',))
class DASZhao3D(FirstOrderEquation):
    """First-order 3-D stress / normal-strain-rate DAS equation (Zhao 2022).

    Three-dimensional generalisation of :class:`DASZhao` from Eq. (9)
    of the Zhao et al. paper. The dynamic variables are normal
    strain-rates ``exx_t``, ``eyy_t``, ``ezz_t``, normal stresses
    ``sxx``, ``syy``, ``szz``, and auxiliary stress-like components
    ``txx``, ``tyy``, ``tzz``. Helical-fibre strain-rate projections
    are computed inline using all three normal strain-rates.
    Staggered-grid CPML (``cpmls``, 12 profiles).

    Reference: Zhao Y. et al. 2022.

    
    """

    C_NAME = "das3d"

    # The compiled 3-D backward has NO boundary-saving path.  The forward never
    # writes boundary strips (it returns an empty ``last_two``), and
    # ``backward_bs`` -- like ``backward_ckpt`` and
    # ``backward_recursive_ckpt`` -- is three lines that re-run the whole
    # forward and allocate the complete ``{nt, 3, B, nz, ny, nx}`` strain
    # history, which is exactly what full storage holds.  Meanwhile the Python
    # side still allocated a 13-field boundary ring and a 13-grid ``last_two``
    # that nothing writes and nothing reads.  Asking for boundary saving here
    # therefore cost strictly MORE memory than 'full', and it was the implicit
    # impl='c' default, so every plain 3-D DAS gradient paid it.  Declaring the
    # capability makes that default resolve to 'full' and an explicit request
    # raise.  This is "not implemented", not "impossible": das2d writes real
    # strips and DASMu / DASMu3D get them from the shared staggered skeleton.
    supports_boundary_saving_c = False

    MODEL_SPECS = DASZhao.MODEL_SPECS
    FIELD_SPECS = (
        FieldSpec("exx_t", description="Normal strain-rate in the x direction.", supports_receiver=True),
        FieldSpec("eyy_t", description="Normal strain-rate in the y direction.", supports_receiver=True),
        FieldSpec("ezz_t", description="Normal strain-rate in the z direction.", supports_receiver=True),
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
        FieldSpec("das35_t", description="35.3 degree helical-fiber axial strain-rate.", supports_receiver=True),
        FieldSpec("das54x_t", description="54.7 degree helical-fiber axial strain-rate for x-oriented core.", supports_receiver=True),
        FieldSpec("das54y_t", description="54.7 degree helical-fiber axial strain-rate for y-oriented core.", supports_receiver=True),
        FieldSpec("das54z_t", description="54.7 degree helical-fiber axial strain-rate for z-oriented core.", supports_receiver=True),
    )

    default_pml_type = "cpmls"  # staggered-grid CPML: step_das_zhao_3d unpacks 12 profiles

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 3-D Zhao DAS equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding).
                Must be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/das3d/forward.cu``)
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
        super().__init__(spatial_order, device, backend, ndim=3)

    @property
    def default_source_fields(self):
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx_t", "eyy_t", "ezz_t"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 5:
            vp, vs, rho, lame_lambda, lame_mu = models
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
        else:
            raise ValueError(f"DASZhao3D.func expected 3 or 5 models, got {len(models)}")
        return step_das_zhao_3d(*wavefields, vp, vs, rho, lame_lambda, lame_mu, dt, h, b, pd=self.pd, pml=self.b, **kwargs)

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: [(nt, 3, B, *grid)],
            save_all_shape=history_fields(3),
            base_nvar=13,
            pml_nvar=18,
            last_two_nvar=1,
            last_two_storage_nvar=13,
            # One read-only zero strain, nine adjoint derivative grids and nine
            # gradient-projection grids of the compiled backward (das3d/backward.cu
            # WorkspaceSlot); the checkpoint replay aliases the derivative grids.
            backward_workspace_nvar=19,
            # The nine per-step derivative scratch grids the compiled forward
            # used to allocate itself (das3d/forward.cu ForwardWorkspaceSlot).
            forward_workspace_nvar=9,
            derived_model_nvar=_das_derived_model_nvar,
        )


@register_equation()
class DASMu(FirstOrderEquation):
    """First-order 2-D velocity-stress-strain DAS equation (Mu).

    Standard 2-D elastic velocity-stress system augmented with three
    integrated strain fields ``exx``, ``ezz``, ``exz`` that track the
    velocity-gradient sources: ``exx_t = d(vx)/dx``,
    ``ezz_t = d(vz)/dz``, ``exz_t = 0.5 · (d(vx)/dz + d(vz)/dx)``.
    The :class:`DAS` facade differentiates the integrated strains in
    time to produce strain-rate (``exx_t``, …) and helical DAS-rate
    (``das35_t``, ``das54x_t``, ``das54z_t``) receivers — which
    requires ``dt`` to be passed through the propagator. Staggered-grid
    CPML (``cpmls``, 8 profiles).

    Reference: Mu & Hung 2022, *Velocity-stress-strain coupling for DAS
    forward modelling*.

    
    """

    C_NAME = "das_mu2d"

    MODEL_SPECS = DASZhao.MODEL_SPECS
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx", "sigma_xx"), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz", "sigma_zz"), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "sigma_xz", "sigma_zx"), description="Shear stress component.", supports_source=True, supports_receiver=True),
        FieldSpec("exx", description="Integrated normal strain in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("ezz", description="Integrated normal strain in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("exz", description="Integrated shear strain component.", supports_source=True, supports_receiver=True),
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txxz", description="Reserved elastic auxiliary field.", internal=True, boundary_related=True),
        FieldSpec("m_tzzx", description="Reserved elastic auxiliary field.", internal=True, boundary_related=True),
        FieldSpec("m_tzzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"  # staggered-grid CPML: step_das_mu_2d unpacks 8 profiles

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 2-D Mu DAS equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding).
                Must be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/das_mu2d/forward.cu``)
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
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx", "ezz", "exz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu, lame_lambda + 2 * lame_mu]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 6:
            vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu = models
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
            lame_lambda_2mu = lame_lambda + 2 * lame_mu
        else:
            raise ValueError(f"DASMu.func expected 3 or 6 models, got {len(models)}")
        return step_das_mu_2d(
            *wavefields,
            vp,
            vs,
            rho,
            lame_lambda,
            lame_mu,
            lame_lambda_2mu,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            save_all_shape=history_fields(2),   # vx, vz
            base_nvar=8,
            pml_nvar=10,
            last_two_nvar=1,
            last_two_storage_nvar=8,
            # The compiled backward's scratch (das_mu2d/driver_traits.cuh
            # WS_CARRIERS), one padded grid per shot each: the 8 elastic
            # adjoint grids [qxx, qzz, qxz, qzx, pxx, pzz, pxz, pzx] in every
            # mode; the checkpoint modes add the velocity carriers -- v(t) at
            # slots 8, 9 (vx, vz) and v(t+1) at 10, 11.
            backward_workspace_shapes=lambda B, nt, shape, mode: [[B, 1, *shape]] * (
                8 + (4 if mode in ("ckpt", "recursive") else 0)),
            # ckpt: the per-segment vx, vz histories, one row per replayed step
            # plus the segment start (das_mu2d/driver_traits.cuh seg_buffers);
            # the recursive mode replays per step and keeps none.
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: (
                [(seg + 1, B, 1, *grid)] * 2 if mode == "ckpt" else []),
            bs_reconstruction_nvar=7,   # vx, vz, sxx, szz, sxz + fvx_prev, fvz_prev (the DAS strains are dead in the bs reverse loop)
            derived_model_nvar=2,   # mu, lambda (common/derived_models.h LameSlot)
        )


@register_equation()
class DASMu3D(FirstOrderEquation):
    """First-order 3-D velocity-stress-strain DAS equation (Mu).

    Three-dimensional generalisation of :class:`DASMu`. Standard 3-D
    elastic velocity-stress system (nine physical fields) augmented
    with six integrated strain fields tracking the velocity gradients.
    The :class:`DAS` facade differentiates the integrated strains in
    time to produce strain-rate and helical DAS-rate receivers — which
    requires ``dt`` to be passed through the propagator. Staggered-grid
    CPML (``cpmls``, 12 profiles).

    Reference: Mu & Hung 2022.

    
    """

    C_NAME = "das_mu3d"

    MODEL_SPECS = DASZhao.MODEL_SPECS
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx", "sigma_xx"), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("syy", aliases=("stress_yy", "sigma_yy"), description="Normal stress in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz", "sigma_zz"), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxy", aliases=("stress_xy", "sigma_xy", "sigma_yx"), description="Shear stress xy component.", supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "sigma_xz", "sigma_zx"), description="Shear stress xz component.", supports_source=True, supports_receiver=True),
        FieldSpec("syz", aliases=("stress_yz", "sigma_yz", "sigma_zy"), description="Shear stress yz component.", supports_source=True, supports_receiver=True),
        FieldSpec("exx", description="Integrated normal strain in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("eyy", description="Integrated normal strain in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("ezz", description="Integrated normal strain in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("exy", description="Integrated shear strain xy component.", supports_source=True, supports_receiver=True),
        FieldSpec("exz", description="Integrated shear strain xz component.", supports_source=True, supports_receiver=True),
        FieldSpec("eyz", description="Integrated shear strain yz component.", supports_source=True, supports_receiver=True),
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxy", description="CPML memory variable for dvx/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vyx", description="CPML memory variable for dvy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vyy", description="CPML memory variable for dvy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vyz", description="CPML memory variable for dvy/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzy", description="CPML memory variable for dvz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_szzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxyx", description="CPML memory variable for dsxy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxyy", description="CPML memory variable for dsxy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_sxzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_syyy", description="CPML memory variable for dsyy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzy", description="CPML memory variable for dsyz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzz", description="CPML memory variable for dsyz/dz.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"  # staggered-grid CPML: step_das_mu_3d unpacks 12 profiles

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 3-D Mu DAS equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding).
                Must be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/das_mu3d/forward.cu``)
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
        super().__init__(spatial_order, device, backend, ndim=3)

    @property
    def default_source_fields(self):
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["exx", "eyy", "ezz", "exy", "exz", "eyz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 5:
            vp, vs, rho, lame_lambda, lame_mu = models
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
        else:
            raise ValueError(f"DASMu3D.func expected 3 or 5 models, got {len(models)}")
        return step_das_mu_3d(
            *wavefields,
            vp,
            vs,
            rho,
            lame_lambda,
            lame_mu,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            save_all_shape=history_fields(3),   # vx, vy, vz
            base_nvar=15,
            pml_nvar=18,
            last_two_nvar=1,
            last_two_storage_nvar=15,
            # The compiled backward's scratch (das_mu3d/driver_traits.cuh
            # WS_CARRIERS), one padded grid per shot each: the 18 elastic
            # adjoint grids (9 q** + 9 p**) in every mode; the checkpoint
            # modes add the velocity carriers -- v(t) at slots 18, 19, 20
            # (vx, vy, vz) and v(t+1) at 21, 22, 23.
            backward_workspace_shapes=lambda B, nt, shape, mode: [[B, 1, *shape]] * (
                18 + (6 if mode in ("ckpt", "recursive") else 0)),
            # ckpt: the per-segment vx, vy, vz histories, one row per replayed
            # step plus the segment start (das_mu3d/driver_traits.cuh
            # seg_buffers); the recursive mode replays per step and keeps none.
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: (
                [(seg + 1, B, 1, *grid)] * 3 if mode == "ckpt" else []),
            bs_reconstruction_nvar=12,  # 9 elastic fields + fvx/fvy/fvz_prev (the DAS strains are dead in the bs reverse loop)
            derived_model_nvar=2,   # mu, lambda (common/derived_models.h LameSlot)
        )


DASElastic   = DASZhao
DASElastic3D = DASZhao3D

# Back-compat: the unified facade used to be named ``DASModeler``.
DASModeler = DAS

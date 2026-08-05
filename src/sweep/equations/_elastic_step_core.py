"""Shared FD core for 2-D first-order velocity-stress elastic equations.

The Virieux (1986) staggered-grid stencil + CPML accumulation + leapfrog
state update is identical between

* :mod:`sweep.equations.elastic`     — bulk moduli, image-method FS,
* :mod:`sweep.equations.elastic_apm` — APM-modified moduli, no image FS,

so we factor it out here.  Callers are responsible for **all post-step
boundary-condition enforcement** (image-method top-row zeroing, APM
traction zeroing, AIR-cell wavefield zeroing).

The function takes "effective" constitutive arrays so both call sites can
share one signature:

================== ==========================================
For ``Elastic``    For ``ElasticAPM``
================== ==========================================
rho_x = rho        rho_x = rho_x_eff      (½ ρ at H/VL/VR…)
rho_z = rho        rho_z = rho_z_eff
lame_lambda        lame_lambda_eff        (0 at H, …)
lame_mu            lame_mu_eff            (α/2 at H, …)
mu_xz = lame_mu    mu_xz = mu_xz_node     (harmonic at σ_xz node)
================== ==========================================

For ``Elastic`` the caller simply aliases ``rho_x = rho_z = rho`` and
``mu_xz = lame_mu`` — these are zero-copy tensor reference assignments,
no extra memory.
"""

from __future__ import annotations

import os as _os

from ._free_surface import (
    top_free_surface_derivative,
    top_free_surface_derivative_topo,
    overwrite_top_row,
    overwrite_at_topo,
    overwrite_surface_row,
    get_o2_pd as _get_o2_pd,
    fs_deriv as _fs_deriv,
    fs_deriv_edges as _fs_deriv_edges,
    sides_from_fs_faces_2d as _sides_from_fs_faces_2d,
    near_surface_o2_count as _near_surface_o2_count,
)


def _fs_sides(fs_faces, free_surface):
    """Resolve (z_sides, x_sides) lists of 'low'/'high' from either the per-edge
    ``fs_faces`` tuple or the legacy top-only ``free_surface`` bool."""
    if fs_faces is not None:
        return _sides_from_fs_faces_2d(fs_faces)
    return (["low"] if free_surface else []), []


# Opposite staggered operator (used by the high-side flip identity in
# ``fs_deriv_edges``): flipping the axis swaps forward <-> backward differences.
_OP_SWAP = {
    "z_forward": "z_backward", "z_backward": "z_forward",
    "x_forward": "x_backward", "x_backward": "x_forward",
}


def _fsd(pd, pd2, name, field, odd, axis, sides, halo, n_o2):
    """FS derivative of ``field`` using ``pd.<name>`` (e.g. ``'z_forward'``),
    honouring every surface in ``sides`` with near-surface order-2 blending from
    ``pd2``.  Supplies the opposite operator for the high-side flip identity."""
    swap = _OP_SWAP[name]
    o2 = getattr(pd2, name) if pd2 is not None else None
    o2_sw = getattr(pd2, swap) if pd2 is not None else None
    return _fs_deriv_edges(
        field, getattr(pd, name), getattr(pd, swap), o2, o2_sw,
        halo, odd, axis, sides, n_o2,
    )


def elastic_velocity_substep(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, m_txzz,
    *,
    lame_lambda, lame_mu, mu_xz,
    rho_x, rho_z,
    dt, h, b, pd, pml,
    free_surface=False,
    topo_rows=None,
    fs_faces=None,
    lame_lambda_2mu=None,
):
    """Velocity sub-step: update ``(vx, vz)`` from the stress gradients (a pure
    shear — stresses are read, not written).  This is the first half of
    :func:`elastic_step_core`; factored out so the eager boundary-saving reverse
    driver can REUSE it (composed in reverse order at ``-dt`` with ``pml`` zeroed)
    instead of duplicating the physics.  Returns all 15 fields with ``vx``/``vz``
    and the four stress-derivative CPML memories updated."""
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    has_topo = free_surface and topo_rows is not None
    z_sides, x_sides = _fs_sides(fs_faces, free_surface)
    # Near-surface order reduction (flat FS only): order-2 image derivative in
    # the near-surface band tames the high-order FS-∩-CPML-corner instability.
    _n_o2 = _near_surface_o2_count(top_halo, _os.environ.get("SWEEP_FS_NEARSURF_O2", "1")) \
        if ((z_sides or x_sides) and not has_topo) else 0
    _pd2 = _get_o2_pd(pd) if _n_o2 else None

    # ---- Stress gradients ------------------------------------------------
    txx_x = pd.x_forward(sxx)
    if has_topo:
        txz_z = top_free_surface_derivative_topo(
            sxz, pd.z_backward, top_halo, True, axis=-2, iz_surf=topo_rows
        )
        tzz_z = top_free_surface_derivative_topo(
            szz, pd.z_forward, top_halo, True, axis=-2, iz_surf=topo_rows
        )
    elif z_sides:
        txz_z = _fsd(pd, _pd2, "z_backward", sxz, True, -2, z_sides, top_halo, _n_o2)
        tzz_z = _fsd(pd, _pd2, "z_forward", szz, True, -2, z_sides, top_halo, _n_o2)
    else:
        txz_z = pd.z_backward(sxz)
        tzz_z = pd.z_forward(szz)
    txz_x = pd.x_backward(sxz)
    # x-normal free surface: mirror the x-derivatives of sxx (odd) and sxz (odd)
    # — same operators as the plain x-derivatives above, plus the image mirror.
    if x_sides:
        txx_x = _fsd(pd, _pd2, "x_forward", sxx, True, -1, x_sides, top_halo, _n_o2)
        txz_x = _fsd(pd, _pd2, "x_backward", sxz, True, -1, x_sides, top_halo, _n_o2)

    # ---- CPML accumulation + velocity update -----------------------------
    m_tzzz = azh * m_tzzz + bzh * tzz_z
    tzz_z = tzz_z + m_tzzz
    m_txzx = ax * m_txzx + bx * txz_x
    txz_x = txz_x + m_txzx
    vz = vz + dt / rho_z * (tzz_z + txz_x)

    m_txzz = az * m_txzz + bz * txz_z
    txz_z = txz_z + m_txzz
    m_txxx = axh * m_txxx + bxh * txx_x
    txx_x = txx_x + m_txxx
    vx = vx + dt / rho_x * (txx_x + txz_z)

    return (
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
    )


def elastic_stress_substep(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, m_txzz,
    *,
    lame_lambda, lame_mu, mu_xz,
    rho_x, rho_z,
    dt, h, b, pd, pml,
    free_surface=False,
    topo_rows=None,
    fs_faces=None,
    lame_lambda_2mu=None,
):
    """Stress sub-step: update ``(sxx, szz, sxz)`` from the velocity gradients (a
    pure shear — velocities are read, not written).  Second half of
    :func:`elastic_step_core`; reused by the boundary-saving reverse driver.
    Returns all 15 fields with the stresses and the four velocity-derivative CPML
    memories updated.  Post-step free-surface stress zeroing is left to the
    caller (``Elastic.func`` / the reverse closure), matching the historic
    layout."""
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    has_topo = free_surface and topo_rows is not None
    z_sides, x_sides = _fs_sides(fs_faces, free_surface)
    lame_lambda_2mu = (lame_lambda + 2 * lame_mu) if lame_lambda_2mu is None else lame_lambda_2mu
    _n_o2 = _near_surface_o2_count(top_halo, _os.environ.get("SWEEP_FS_NEARSURF_O2", "1")) \
        if ((z_sides or x_sides) and not has_topo) else 0
    _pd2 = _get_o2_pd(pd) if _n_o2 else None

    # ---- Velocity gradients ----------------------------------------------
    vx_x = pd.x_backward(vx)
    if has_topo:
        vz_z = top_free_surface_derivative_topo(
            vz, pd.z_backward, top_halo, True, axis=-2, iz_surf=topo_rows
        )
        vx_z = top_free_surface_derivative_topo(
            vx, pd.z_forward, top_halo, False, axis=-2, iz_surf=topo_rows
        )
    elif z_sides:
        vz_z = _fsd(pd, _pd2, "z_backward", vz, True, -2, z_sides, top_halo, _n_o2)
        vx_z = _fsd(pd, _pd2, "z_forward", vx, False, -2, z_sides, top_halo, _n_o2)
    else:
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
    vz_x = pd.x_forward(vz)
    # x-normal free surface: mirror the x-derivatives of vx (normal, odd) and vz
    # (tangential, even) — same operators as the plain x-derivatives above.
    if x_sides:
        vx_x = _fsd(pd, _pd2, "x_backward", vx, True, -1, x_sides, top_halo, _n_o2)
        vz_x = _fsd(pd, _pd2, "x_forward", vz, False, -1, x_sides, top_halo, _n_o2)

    # ---- CPML accumulation + stress update -------------------------------
    m_vzz = az * m_vzz + bz * vz_z
    vz_z = vz_z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_x
    vx_x = vx_x + m_vxx

    sxx_pre_fs = sxx
    szz_pre_fs = szz
    szz = szz + dt * (lame_lambda_2mu * vz_z + lame_lambda * vx_x)
    sxx = sxx + dt * (lame_lambda_2mu * vx_x + lame_lambda * vz_z)
    _mod_sxx = _os.environ.get("SWEEP_FS_MOD_SXX", "1") == "1"
    if _mod_sxx and (has_topo or z_sides):
        # Robertsson free-surface fix at a z-normal surface: surface-row sigma_xx
        # uses the modified coefficient 4 mu (lam+mu)/(lam+2mu) * d vx/dx  (from
        # sigma_zz=0).  Flat top row, per-column topography, or z-bottom row (the
        # image-method mirror alone leaves the surface sigma_xx ~mu/lambda too big).
        _coef = 4.0 * lame_mu * (lame_lambda + lame_mu) / lame_lambda_2mu
        _sxx_surf = sxx_pre_fs + dt * _coef * vx_x
        if has_topo:
            sxx = overwrite_at_topo(sxx, _sxx_surf, topo_rows, axis=-2)
        else:
            for side in z_sides:
                sxx = overwrite_surface_row(sxx, _sxx_surf, top_halo, axis=-2, side=side)
    if _mod_sxx and x_sides:
        # x-normal analogue (x<->z swap): at sigma_xx=0 the surface sigma_zz uses
        # the modified coefficient times d vz/dz.
        _coef_x = 4.0 * lame_mu * (lame_lambda + lame_mu) / lame_lambda_2mu
        _szz_surf = szz_pre_fs + dt * _coef_x * vz_z
        for side in x_sides:
            szz = overwrite_surface_row(szz, _szz_surf, top_halo, axis=-1, side=side)

    m_vxz = azh * m_vxz + bzh * vx_z
    vx_z = vx_z + m_vxz
    m_vzx = axh * m_vzx + bxh * vz_x
    vz_x = vz_x + m_vzx
    sxz = sxz + dt * mu_xz * (vx_z + vz_x)

    return (
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
    )


def elastic_step_core(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, m_txzz,
    *,
    lame_lambda, lame_mu, mu_xz,
    rho_x, rho_z,
    dt, h, b, pd, pml,
    free_surface=False,
    topo_rows=None,
    fs_faces=None,
    lame_lambda_2mu=None,
):
    """One leapfrog step on the 2-D first-order velocity-stress elastic
    equations: ``stress ∘ velocity``.  Pure stencil + CPML + state update; NO
    post-step BC enforcement (callers do that).

    Composed from :func:`elastic_velocity_substep` then
    :func:`elastic_stress_substep` so the forward step and the boundary-saving
    reverse share one implementation of the physics (the reverse composes the
    same two sub-steps in the opposite order at ``-dt`` with ``pml`` zeroed).

    Parameters are unchanged from the historic monolithic version: ``pml`` is the
    8-tuple ``(az, bz, azh, bzh, ax, bx, axh, bxh)``; ``free_surface``/
    ``topo_rows`` select the image-method top derivatives; ``lame_lambda_2mu`` is
    an optional precomputed ``λ + 2 μ``.
    """
    kw = dict(
        lame_lambda=lame_lambda, lame_mu=lame_mu, mu_xz=mu_xz,
        rho_x=rho_x, rho_z=rho_z, dt=dt, h=h, b=b, pd=pd, pml=pml,
        free_surface=free_surface, topo_rows=topo_rows, fs_faces=fs_faces,
        lame_lambda_2mu=lame_lambda_2mu,
    )
    state = elastic_velocity_substep(
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz, **kw,
    )
    return elastic_stress_substep(*state, **kw)

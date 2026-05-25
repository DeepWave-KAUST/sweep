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

from ._free_surface import (
    top_free_surface_derivative,
    top_free_surface_derivative_topo,
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
    lame_lambda_2mu=None,
):
    """One leapfrog step on the 2-D first-order velocity-stress elastic
    equations.  Pure stencil + CPML + state update; NO post-step BC
    enforcement (callers do that).

    Parameters
    ----------
    vx, vz, sxx, szz, sxz
        Five wavefields at time level n.
    m_vxx ... m_txzz
        Ten CPML memory variables.
    lame_lambda, lame_mu, mu_xz
        Effective Lamé parameters.  ``mu_xz`` is μ at the σ_xz staggered
        node — equal to ``lame_mu`` in the bulk case, harmonic-averaged
        for APM.
    rho_x, rho_z
        Effective densities at the v_x and v_z staggered nodes
        respectively.  Equal in the bulk case, modified per
        Cao & Chen 2018 in APM.
    dt, h, b
        Time step, grid spacing, batch size (latter passed through for
        API parity with the rest of SWEEP).
    pd
        Partial-derivative module (provides ``x_forward``, ``z_forward``,
        ``x_backward``, ``z_backward`` and a ``coes`` attribute holding
        the stencil half-width).
    pml
        Tuple of 8 CPML profiles ``(az, bz, azh, bzh, ax, bx, axh, bxh)``.
    free_surface, topo_rows
        If ``free_surface=True``, replace the top z-derivatives with the
        image-method derivatives.  ``topo_rows`` selects the per-column
        staircase variant; ``None`` selects the flat-top variant.
    lame_lambda_2mu
        Optional precomputed ``λ + 2 μ``.  If ``None``, computed locally.
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    lame_lambda_2mu = (lame_lambda + 2 * lame_mu) if lame_lambda_2mu is None else lame_lambda_2mu

    # Topo path picks per-column ``top_free_surface_derivative_topo`` (image-
    # method mirror with column-dependent surface row). When ``topo_rows`` is
    # None we fall back to the flat ``top_free_surface_derivative`` used by
    # the historic free-surface=True path.  When free_surface is False the
    # plain ``pd.z_*`` derivatives apply at every row.
    has_topo = free_surface and topo_rows is not None

    # ---- Stress gradients ------------------------------------------------
    txx_x = pd.x_forward(sxx)
    if free_surface:
        if has_topo:
            txz_z = top_free_surface_derivative_topo(
                sxz, pd.z_backward, top_halo, True, axis=-2, iz_surf=topo_rows
            )
            tzz_z = top_free_surface_derivative_topo(
                szz, pd.z_forward, top_halo, True, axis=-2, iz_surf=topo_rows
            )
        else:
            txz_z = top_free_surface_derivative(
                sxz, pd.z_backward, top_halo, odd=True, axis=-2
            )
            tzz_z = top_free_surface_derivative(
                szz, pd.z_forward, top_halo, odd=True, axis=-2
            )
    else:
        txz_z = pd.z_backward(sxz)
        tzz_z = pd.z_forward(szz)
    txz_x = pd.x_backward(sxz)

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

    # ---- Velocity gradients ----------------------------------------------
    vx_x = pd.x_backward(vx)
    if free_surface:
        if has_topo:
            vz_z = top_free_surface_derivative_topo(
                vz, pd.z_backward, top_halo, True, axis=-2, iz_surf=topo_rows
            )
            vx_z = top_free_surface_derivative_topo(
                vx, pd.z_forward, top_halo, False, axis=-2, iz_surf=topo_rows
            )
        else:
            vz_z = top_free_surface_derivative(
                vz, pd.z_backward, top_halo, odd=True, axis=-2
            )
            vx_z = top_free_surface_derivative(
                vx, pd.z_forward, top_halo, odd=False, axis=-2
            )
    else:
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
    vz_x = pd.x_forward(vz)

    # ---- CPML accumulation + stress update -------------------------------
    m_vzz = az * m_vzz + bz * vz_z
    vz_z = vz_z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_x
    vx_x = vx_x + m_vxx

    szz = szz + dt * (lame_lambda_2mu * vz_z + lame_lambda * vx_x)
    sxx = sxx + dt * (lame_lambda_2mu * vx_x + lame_lambda * vz_z)

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

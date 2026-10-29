"""1st-order velocity-stress **VTI** elastic SEM on a deformed Quad9 mesh.

A minimal anisotropic extension of :class:`ElasticVSSEM2DDeformed`: it overrides
only the constitutive seam (``_stress_increment``) with the VTI Voigt
contraction and inherits the entire velocity-stress time loop, CPML, free
surface, source/receiver handling, autograd gradient and boundary-saving
machinery unchanged. This is the "share the physics, keep the operator"
factoring — ``∇v`` (with its CPML stretching) is computed by the base class;
only ``σ = C:∇v`` differs.

Constitutive (in-plane P-SV, see :func:`sweep.sem.constitutive.thomsen_to_vti_cij`):

    σ_xx += dt (c11 ∂x v_x + c13 ∂z v_z)
    σ_zz += dt (c13 ∂x v_x + c33 ∂z v_z)
    σ_xz += dt  c55 (∂z v_x + ∂x v_z)

``epsilon = delta = 0`` reduces bit-for-bit to the isotropic equation.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..fields import check_element_arrays

from ..constitutive import thomsen_to_vti_cij
from ..cpml.elastic_vs import SimpleElasticCPMLProfile
from ..mesh.quad9 import Quad9Mesh
from .elastic_vs_deformed import ElasticVSSEM2DDeformed


class ElasticVTISEM2DDeformed(ElasticVSSEM2DDeformed):
    """VTI velocity-stress elastic SEM (vertical symmetry axis)."""

    # Boundary saving: the model leaves are the Voigt cij (not the isotropic
    # Lamé params), so the per-step VJP substitutes / differentiates these.
    bs_model_attrs = ("_c11", "_c13", "_c33", "_c55")

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        epsilon_e: torch.Tensor,
        delta_e: torch.Tensor,
        *,
        free_top: bool = False,
        cpml: Optional[SimpleElasticCPMLProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticVTISEM2DDeformed":
        """Build a VTI equation. ``vp_e`` / ``vs_e`` are the *vertical* P / S
        velocities; ``epsilon_e`` / ``delta_e`` are the Thomsen parameters
        (all ``(E, ngll, ngll)`` GLL-node tensors)."""
        check_element_arrays(mesh, epsilon_e=epsilon_e, delta_e=delta_e)
        # Reuse the isotropic build for mass, geometry, free-surface tangents
        # and the CPML wiring, then recast to VTI and attach the cij.
        eq = ElasticVSSEM2DDeformed.build(
            mesh, vp_e, vs_e, rho_e, free_top=free_top, cpml=cpml, band_split=band_split)
        eq.__class__ = cls
        c11, c13, c33, c55 = thomsen_to_vti_cij(
            vp_e, vs_e, epsilon_e, delta_e, rho_e)
        eq._c11, eq._c13, eq._c33, eq._c55 = c11, c13, c33, c55
        eq.epsilon_e, eq.delta_e = epsilon_e, delta_e
        return eq

    def _stress_increment(self, d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz, dt, rows=None):
        pick = (lambda t: t) if rows is None else (lambda t: t[rows])
        c11, c13, c33, c55 = pick(self._c11), pick(self._c13), pick(self._c33), pick(self._c55)
        inc_xx = dt * (c11 * d_vx_dx + c13 * d_vz_dz)
        inc_zz = dt * (c13 * d_vx_dx + c33 * d_vz_dz)
        inc_xz = dt * c55 * (d_vx_dz + d_vz_dx)
        return inc_xx, inc_zz, inc_xz

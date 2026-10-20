"""1st-order velocity-stress elastic SEM on a deformed Hex27 mesh.

The 3-D analogue of :class:`~sweep.sem.equations.elastic_vs_deformed.ElasticVSSEM2DDeformed`,
and the only equation in the package that offers **3-D elastic + topography +
an absorbing boundary** at once. Before it, the choice was:

* :class:`ElasticSEM3DDeformed` -- topography, but models no C-PML at all;
* :class:`ElasticVSSEM3D` -- a C-PML, but only on an axis-aligned Hex8 mesh.

It is a subclass, not a copy. Everything that is physics rather than geometry
-- the velocity-stress time loop, the constitutive update, the C-PML memory
recursion, ``field_specs``, the boundary-saving hooks -- is inherited unchanged
from :class:`ElasticVSSEM3D`. Exactly three things are overridden, and all
three are the same substitution: an axis-aligned mesh lets a physical
derivative be ONE reference derivative times a constant (``2/dx_e``), while a
deformed mesh needs the full chain rule

    d/dx_d = sum_m inv_J[..., m, d] * d/dxi_m

with all nine components of the per-GLL inverse Jacobian. That is exactly what
the 2-D deformed velocity-stress equation already does with a 2x2 ``inv_J``.

The C-PML profile is reused as-is: ``build_simple_elastic_cpml_3d`` builds its
damping from NODE COORDINATES (``mesh.x_e``/``y_e``/``z_e``), not from element
indices or an axis-alignment assumption, so it is meaningful on a curved mesh
without modification.

On a FLAT Hex27 this reduces to :class:`ElasticVSSEM3D` on the equivalent Hex8
mesh -- same global node set, agreement to float64 round-off, NOT bit-for-bit
(the zero cross terms still enter the sums and re-associate them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..fields import check_element_arrays

from .._assembly import phys_grads3, stress_divergence3
from ..cpml.elastic_vs3d import SimpleElasticCPML3DProfile
from ..mesh.hex27 import Hex27Mesh
from .elastic_vs3d import ElasticVSSEM3D


@dataclass
class ElasticVSSEM3DDeformed(ElasticVSSEM3D):
    """Velocity-stress elastic on Hex27, with the per-GLL Jacobian chain rule."""

    # Geometry the flat parent does not carry: the per-GLL inverse Jacobian and
    # the weighted Jacobian determinant. The parent's _wHx/_wHy/_wHz and
    # _inv_half_d* are the axis-aligned shortcuts these replace; they are still
    # set (the parent is a dataclass) but unused on this path.
    _inv_J: Optional[torch.Tensor] = None      # (E, ngll, ngll, ngll, 3, 3)
    _w_jac: Optional[torch.Tensor] = None      # w_a w_b w_c * |J|

    @classmethod
    def build(
        cls,
        mesh: Hex27Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
    ) -> "ElasticVSSEM3DDeformed":
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype
        w = mesh.w_ref
        w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]

        mu = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        # |J| varies per GLL node here, so the mass comes from w_jac rather
        # than from a scalar jacobian_det.
        w_jac = mesh.w_jac
        m_e = w_jac * rho_e
        mass_rho = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass_rho.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass_rho=mass_rho, _inv_M=1.0 / mass_rho, cpml=cpml,
            _D=mesh.deriv, _lam=lam, _mu=mu, _lam2mu=lam2mu, _w3=w3,
            # the axis-aligned shortcuts are unused on this path; keep the
            # dataclass fields well-formed rather than leaving them None
            _wHx=w_jac, _wHy=w_jac, _wHz=w_jac,
            _inv_half_dx=1.0, _inv_half_dy=1.0, _inv_half_dz=1.0,
            _inv_J=mesh.inv_J, _w_jac=w_jac,
        )

    # -- the two geometry-dependent operators, and nothing else ----------- #
    def velocity_gradients(self, vx_g, vy_g, vz_g):
        """The nine ``d v_l / d x_k``, in the parent's fixed order."""
        g = phys_grads3(self._D, self.mesh.ibool, self._inv_J, vx_g, vy_g, vz_g)
        vx_x, vx_y, vx_z, vy_x, vy_y, vy_z, vz_x, vz_y, vz_z = g
        return (vx_x, vy_y, vz_z,        # gvx_x, gvy_y, gvz_z
                vy_x, vx_y, vz_x,        # gvy_x, gvx_y, gvz_x
                vx_z, vz_y, vy_z)        # gvx_z, gvz_y, gvy_z

    def assemble_stress_divergence(self, sxx_x, sxy_y, sxz_z,
                                   sxy_x, syy_y, syz_z,
                                   sxz_x, syz_y, szz_z):
        like = sxx_x.new_zeros(self.mesh.n_global)
        return stress_divergence3(
            self._D, self.mesh.ibool, self._inv_J, self._w_jac,
            (sxx_x, sxy_y, sxz_z, sxy_x, syy_y, syz_z, sxz_x, syz_y, szz_z),
            like)

"""1st-order velocity-stress VTI and TTI elastic SEM in 3-D.

:class:`ElasticVSSEM3D` (Hex8) and :class:`ElasticVSSEM3DDeformed` (Hex27, a topography) with a
transversely isotropic medium. As in 2-D (:mod:`.elastic_vti_vs_deformed`) only the
constitutive seam ``_stress_increment`` changes: the velocity gradients (C-PML-stretched by the
parent's step) go through the Voigt contraction

    dσ_I = dt · Σ_J c_IJ e_J,    e = (∂x v_x, ∂y v_y, ∂z v_z, ∂z v_y + ∂y v_z,
                                       ∂z v_x + ∂x v_z, ∂y v_x + ∂x v_y)

(``I, J`` = ``xx, yy, zz, yz, xz, xy``), and the time loop, the C-PML, the free surface, the
sources and receivers, the gradients and boundary saving are the parent's.

* :class:`ElasticVTISEM3D` / :class:`ElasticVTISEM3DDeformed`: models ``vp, vs, rho, epsilon,
  delta, gamma`` -- the vertical velocities and Thomsen's parameters, nine stiffnesses.
* :class:`ElasticTTISEM3D` / :class:`ElasticTTISEM3DDeformed`: ``theta`` and ``phi`` added, the
  symmetry axis tilted by ``theta`` from the vertical toward the azimuth ``phi`` (radians), all
  21 stiffnesses -- the parameters and the Bond rotation of the FD ``ElasticTTISG3D``
  (:func:`sweep.sem.constitutive.thomsen_to_cij3d`), which at ``phi = 0`` is the 2-D
  ``ElasticTTISEM2DDeformed``'s tilt.

The explicit step's stability limit is set by the fastest velocity, ``vp sqrt(1 + 2 epsilon)``
for a positive ``epsilon``: take ``dt`` from that. ``impl='c'`` runs the velocity-stress CUDA
kernels with these stiffnesses (``aniso`` = 9 or 21, :mod:`sweep.sem.prop_c`).
"""

from __future__ import annotations

from typing import ClassVar, Optional

import torch

from ..constitutive import thomsen_to_cij3d
from ..cpml.elastic_vs3d import SimpleElasticCPML3DProfile
from ..fields import check_element_arrays
from ..mesh.hex27 import Hex27Mesh
from ..mesh.hex8 import Hex8Mesh
from .elastic_vs3d import ElasticVSSEM3D
from .elastic_vs3d_deformed import ElasticVSSEM3DDeformed

# the 21 Voigt stiffnesses, in the order of the FD ElasticTTISG3D's STIFFNESS_KEYS_3D
_CIJ = ("_c11", "_c12", "_c13", "_c14", "_c15", "_c16", "_c22", "_c23", "_c24", "_c25", "_c26",
        "_c33", "_c34", "_c35", "_c36", "_c44", "_c45", "_c46", "_c55", "_c56", "_c66")
_VTI = ("_c11", "_c12", "_c13", "_c22", "_c23", "_c33", "_c44", "_c55", "_c66")
# no isotropic bulk modulus, so no pressure receiver; strain and rotation as the parent's
_RECEIVERS = {k: v for k, v in ElasticVSSEM3D.spatial_receiver_types.items() if k != "p"}


def _strain_rates(g):
    """The Voigt strain rates ``e1 .. e6`` of the nine velocity gradients (the order of
    :meth:`ElasticVSSEM3D.velocity_gradients`)."""
    vx_x, vy_y, vz_z, vy_x, vx_y, vz_x, vx_z, vz_y, vy_z = g
    return vx_x, vy_y, vz_z, vz_y + vy_z, vz_x + vx_z, vy_x + vx_y


class ElasticTTISEM3D(ElasticVSSEM3D):
    """TTI velocity-stress elastic SEM on Hex8: the 21 Voigt stiffnesses of a tilted
    transversely isotropic medium. See the module docstring."""

    # the elastic class whose build gives the geometry, the mass and the C-PML
    _elastic: ClassVar[type] = ElasticVSSEM3D
    # boundary saving: the model leaves are the stiffnesses, not the Lame parameters
    bs_model_attrs: ClassVar[tuple] = _CIJ
    spatial_receiver_types: ClassVar[dict] = _RECEIVERS

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        epsilon_e: torch.Tensor,
        delta_e: torch.Tensor,
        gamma_e: torch.Tensor,
        theta_e: torch.Tensor,
        phi_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticTTISEM3D":
        """``vp_e``, ``vs_e``: the velocities along the symmetry axis; ``epsilon_e``,
        ``delta_e``, ``gamma_e``: Thomsen's parameters; ``theta_e``, ``phi_e``: the axis's tilt
        and azimuth (radians). All ``(E, ngll, ngll, ngll)``."""
        check_element_arrays(mesh, theta_e=theta_e, phi_e=phi_e)
        return cls._build(mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, theta_e, phi_e,
                          cpml=cpml, band_split=band_split)

    @classmethod
    def _build(cls, mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, theta_e, phi_e, *,
               cpml, band_split):
        check_element_arrays(mesh, epsilon_e=epsilon_e, delta_e=delta_e, gamma_e=gamma_e)
        # the elastic build for the mass, the geometry and the C-PML; its Lame set is unused
        eq = cls._elastic.build(mesh, vp_e, vs_e, rho_e, cpml=cpml, band_split=band_split)
        eq.__class__ = cls
        for name, c in zip(_CIJ, thomsen_to_cij3d(vp_e, vs_e, epsilon_e, delta_e, gamma_e,
                                                  rho_e, theta_e, phi_e)):
            setattr(eq, name, c)
        eq.epsilon_e, eq.delta_e, eq.gamma_e = epsilon_e, delta_e, gamma_e
        eq.theta_e, eq.phi_e = theta_e, phi_e
        return eq

    def _stress_increment(self, g, dt, rows=None):
        """``dσ = dt C e`` with the 21 stiffnesses, returned as ``xx, yy, zz, xy, xz, yz``."""
        e1, e2, e3, e4, e5, e6 = _strain_rates(g)
        pick = (lambda t: t) if rows is None else (lambda t: t[rows])
        (c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, c34, c35, c36,
         c44, c45, c46, c55, c56, c66) = (pick(getattr(self, a)) for a in _CIJ)
        inc_xx = dt * (c11 * e1 + c12 * e2 + c13 * e3 + c14 * e4 + c15 * e5 + c16 * e6)
        inc_yy = dt * (c12 * e1 + c22 * e2 + c23 * e3 + c24 * e4 + c25 * e5 + c26 * e6)
        inc_zz = dt * (c13 * e1 + c23 * e2 + c33 * e3 + c34 * e4 + c35 * e5 + c36 * e6)
        inc_yz = dt * (c14 * e1 + c24 * e2 + c34 * e3 + c44 * e4 + c45 * e5 + c46 * e6)
        inc_xz = dt * (c15 * e1 + c25 * e2 + c35 * e3 + c45 * e4 + c55 * e5 + c56 * e6)
        inc_xy = dt * (c16 * e1 + c26 * e2 + c36 * e3 + c46 * e4 + c56 * e5 + c66 * e6)
        return inc_xx, inc_yy, inc_zz, inc_xy, inc_xz, inc_yz


class ElasticVTISEM3D(ElasticTTISEM3D):
    """VTI velocity-stress elastic SEM on Hex8: nine Voigt stiffnesses of a vertical
    symmetry axis. See the module docstring."""

    bs_model_attrs: ClassVar[tuple] = _VTI

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        epsilon_e: torch.Tensor,
        delta_e: torch.Tensor,
        gamma_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticVTISEM3D":
        """``vp_e``, ``vs_e``: the vertical velocities; ``epsilon_e``, ``delta_e``,
        ``gamma_e``: Thomsen's parameters. All ``(E, ngll, ngll, ngll)``."""
        return cls._build(mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, cpml=cpml,
                          band_split=band_split)

    @classmethod
    def _build(cls, mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, *, cpml, band_split):
        check_element_arrays(mesh, epsilon_e=epsilon_e, delta_e=delta_e, gamma_e=gamma_e)
        eq = cls._elastic.build(mesh, vp_e, vs_e, rho_e, cpml=cpml, band_split=band_split)
        eq.__class__ = cls
        for name, c in zip(_CIJ, thomsen_to_cij3d(vp_e, vs_e, epsilon_e, delta_e, gamma_e,
                                                  rho_e)):
            if name in _VTI:
                setattr(eq, name, c)
        eq.epsilon_e, eq.delta_e, eq.gamma_e = epsilon_e, delta_e, gamma_e
        return eq

    def _stress_increment(self, g, dt, rows=None):
        """``dσ = dt C e`` with the nine VTI stiffnesses."""
        e1, e2, e3, e4, e5, e6 = _strain_rates(g)
        pick = (lambda t: t) if rows is None else (lambda t: t[rows])
        c11, c12, c13, c22, c23, c33, c44, c55, c66 = (pick(getattr(self, a)) for a in _VTI)
        # the TTI step's sums with its zero terms dropped, so a TTI medium with no tilt
        # (exactly these stiffnesses) is this step bit for bit
        return (dt * (c11 * e1 + c12 * e2 + c13 * e3),
                dt * (c12 * e1 + c22 * e2 + c23 * e3),
                dt * (c13 * e1 + c23 * e2 + c33 * e3),
                dt * (c66 * e6), dt * (c55 * e5), dt * (c44 * e4))


class ElasticTTISEM3DDeformed(ElasticTTISEM3D, ElasticVSSEM3DDeformed):
    """:class:`ElasticTTISEM3D` on a deformed Hex27 mesh (a topography): its stiffnesses on the
    geometry of :class:`ElasticVSSEM3DDeformed`."""

    _elastic: ClassVar[type] = ElasticVSSEM3DDeformed

    @classmethod
    def build(
        cls,
        mesh: Hex27Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        epsilon_e: torch.Tensor,
        delta_e: torch.Tensor,
        gamma_e: torch.Tensor,
        theta_e: torch.Tensor,
        phi_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticTTISEM3DDeformed":
        check_element_arrays(mesh, theta_e=theta_e, phi_e=phi_e)
        return cls._build(mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, theta_e, phi_e,
                          cpml=cpml, band_split=band_split)


class ElasticVTISEM3DDeformed(ElasticVTISEM3D, ElasticVSSEM3DDeformed):
    """:class:`ElasticVTISEM3D` on a deformed Hex27 mesh (a topography)."""

    _elastic: ClassVar[type] = ElasticVSSEM3DDeformed

    @classmethod
    def build(
        cls,
        mesh: Hex27Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        epsilon_e: torch.Tensor,
        delta_e: torch.Tensor,
        gamma_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticVTISEM3DDeformed":
        return cls._build(mesh, vp_e, vs_e, rho_e, epsilon_e, delta_e, gamma_e, cpml=cpml,
                          band_split=band_split)

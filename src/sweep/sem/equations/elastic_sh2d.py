"""SH (antiplane) elastic SEM on Quad9: SPECFEM2D's ``P_SV = .false.``.

The out-of-plane displacement ``u_y`` of an elastic medium invariant along y obeys

    rho u_y'' = d/dx (mu du_y/dx) + d/dz (mu du_y/dz) + f_y,

the scalar wave equation of :class:`AcousticPotentialSEM2D` with its mass weight ``1 / kappa``
read as ``rho`` and its stiffness weight ``1 / rho`` as ``mu``: this class is that one built
with ``vp = vs`` and the density ``1 / (rho vs^2)``, so it shares its operators, its C-PML
(double convolution) and its boundary-saving hooks. The surface is traction-free, the weak
form's natural condition, so it has no Dirichlet nodes: the sides PropSEM leaves without an
absorbing layer are free. Models ``vs, rho``; the source is a force along y (``u_y``), and the
receivers record ``u_y`` or its velocity ``vy``. Eager.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Optional

from ..cpml.acoustic import CPMLProfile
from ..fields import FieldSpec
from ..mesh.domains import DomainMask
from ..mesh.quad9 import Quad9Mesh
from .acoustic_potential2d import AcousticPotentialSEM2D


@dataclass
class ElasticSHSEM2DDeformed(AcousticPotentialSEM2D):
    """SH elastic SEM (``u_y``, traction-free surface) on a Quad9 mesh."""

    takes_density: ClassVar[bool] = False
    field_specs: ClassVar[list] = [
        FieldSpec("u_y", supports_source=True, supports_receiver=True,
                  seis_key="seis_uy", snap_key="snap_uy"),
        FieldSpec("u_y_prev", internal=True),
        FieldSpec("psi_x", internal=True, boundary_related=True),
        FieldSpec("psi_z", internal=True, boundary_related=True),
        FieldSpec("zeta_x", internal=True, boundary_related=True),
        FieldSpec("zeta_z", internal=True, boundary_related=True),
    ]
    derived_receiver_types: ClassVar[dict] = {"vy": ("u_y", "velocity")}
    # a force along y, recorded as u_y: none of the potential form's PropSEM conventions
    # (the wavelet as the pressure source, the pressure by default, the fluid's particle motion)
    spatial_receiver_types: ClassVar[dict] = {}
    propsem_pressure_wavelet: ClassVar[bool] = False
    propsem_default_receiver_type: ClassVar[Optional[tuple]] = None

    @classmethod
    def build(cls, mesh: Quad9Mesh, vs_e, rho_e, *,
              dirichlet: tuple[str, ...] = (),
              domain: Optional[DomainMask] = None,
              cpml: Optional[CPMLProfile] = None) -> "ElasticSHSEM2DDeformed":
        """Build the equation from the shear velocity ``vs_e`` and the density ``rho_e``
        (``(E, ngll, ngll)``). ``dirichlet`` is accepted for PropSEM's call and ignored: an SH
        surface is traction-free."""
        base = AcousticPotentialSEM2D.build(mesh, vs_e, 1.0 / (rho_e * vs_e * vs_e),
                                            dirichlet=(), domain=domain, cpml=cpml)
        eq = cls(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        return eq

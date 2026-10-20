"""Spectral element method (SEM) backend.

A SPECFEM-style continuous-Galerkin SEM on quad meshes with
Gauss-Lobatto-Legendre (GLL) quadrature. Lives next to
``sweep.equations`` (the dense-Cartesian FD path); the two backends
do not share base classes.

Layout
------
``basis``      GLL nodal basis + Q2 biquadratic shape functions.
``mesh``       Quad4 (axis-aligned) and Quad9 (deformable) meshes.
``cpml``       CPML absorbing-profile builders (acoustic / elastic /
               elastic velocity-stress).
``equations``  Weak-form step kernels (acoustic + elastic, 2nd- and
               1st-order velocity-stress).
``propagator`` One generic time-stepping driver (``SEMPropagator``)
               that runs every equation above.

All public names are re-exported here, so ``from sweep.sem import X``
keeps working regardless of the internal module layout.

See ``docs/user-guide/sem.md`` for the user guide.
"""

from .basis import gll_nodes_weights, lagrange_derivative_matrix
from .mesh import (
    Quad4Mesh,
    build_quad4_mesh,
    assemble_mass_vector,
    boundary_element_edges,
    Quad9Mesh,
    build_axis_aligned_quad9,
    build_topography_quad9,
    quad9_from_control_nodes,
    Hex8Mesh,
    build_hex8_mesh,
    boundary_element_faces,
    Hex27Mesh,
    build_axis_aligned_hex27,
    build_topography_hex27,
    hex27_from_control_nodes,
)
from .cpml import (
    CPMLProfile,
    build_cpml,
    CPML3DProfile,
    build_cpml3d,
    ElasticCPMLProfile,
    build_elastic_cpml,
    SimpleElasticCPMLProfile,
    build_simple_elastic_cpml,
    SimpleElasticCPML3DProfile,
    build_simple_elastic_cpml_3d,
)
# Equations are DISCOVERED by sweep.sem.equations, so a new equation module is
# exported without editing a list here. Three equations used to be missing from
# the hand-kept list and were unreachable as ``sweep.sem.X``.
from . import equations as _equations
from .equations import *                                    # noqa: F401,F403
globals().update(_equations.equation_classes())
from .constitutive import thomsen_to_vti_cij
from .propagator import (SEMPropagator, PointStamp, PointStamp3D, EdgeStamp,
                         build_point_stamp, build_moment_stamp)
from .prop_base import PropSEM
from .boundary_saving import run_boundary_saved
from .discretization import suggest_discretization, SEMDiscretization

__all__ = [
    "gll_nodes_weights",
    "lagrange_derivative_matrix",
    "Quad4Mesh",
    "build_quad4_mesh",
    "assemble_mass_vector",
    "boundary_element_edges",
    "Quad9Mesh",
    "build_axis_aligned_quad9",
    "build_topography_quad9",
    "quad9_from_control_nodes",
    "Hex8Mesh",
    "build_hex8_mesh",
    "boundary_element_faces",
    "Hex27Mesh",
    "build_axis_aligned_hex27",
    "build_topography_hex27",
    "hex27_from_control_nodes",
    "CPMLProfile",
    "build_cpml",
    "CPML3DProfile",
    "build_cpml3d",
    "ElasticCPMLProfile",
    "build_elastic_cpml",
    "SimpleElasticCPMLProfile",
    "build_simple_elastic_cpml",
    "SimpleElasticCPML3DProfile",
    "build_simple_elastic_cpml_3d",
    *_equations.__all__,
    "SEMPropagator",
    "PointStamp",
    "PointStamp3D",
    "EdgeStamp",
    "build_point_stamp",
    "build_moment_stamp",
    "PropSEM",
    "run_boundary_saved",
    "suggest_discretization",
    "SEMDiscretization",
]

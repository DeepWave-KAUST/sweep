"""SEM meshes.

``quad4`` Axis-aligned Quad4 mesh (constant Jacobian per element).
``quad9`` Biquadratic Quad9 mesh (per-GLL Jacobian; supports topography).
``hex8``  Axis-aligned Hex8 mesh (3-D; constant Jacobian per element).

``boundary_element_edges`` re-exported here is the Quad4 variant (matching
the historical top-level export); the Quad9 variant lives at
``sweep.sem.mesh.quad9.boundary_element_edges``. The 3-D face sets are
``sweep.sem.mesh.hex8.boundary_element_faces``.
"""

from .quad4 import (
    Quad4Mesh,
    build_quad4_mesh,
    assemble_mass_vector,
    boundary_element_edges,
)
from .quad9 import (
    Quad9Mesh,
    build_axis_aligned_quad9,
    build_topography_quad9,
    quad9_from_control_nodes,
)
from .hex8 import (
    Hex8Mesh,
    Hex8Layout,
    build_hex8_mesh,
    hex8_layout,
    hex8_ibool,
    boundary_element_faces,
)
from .hex27 import (
    Hex27Mesh,
    build_axis_aligned_hex27,
    build_topography_hex27,
    hex27_from_control_nodes,
)

__all__ = [
    "Quad4Mesh",
    "build_quad4_mesh",
    "assemble_mass_vector",
    "boundary_element_edges",
    "Quad9Mesh",
    "build_axis_aligned_quad9",
    "build_topography_quad9",
    "quad9_from_control_nodes",
    "Hex8Mesh",
    "Hex8Layout",
    "build_hex8_mesh",
    "hex8_layout",
    "hex8_ibool",
    "boundary_element_faces",
    "Hex27Mesh",
    "build_axis_aligned_hex27",
    "build_topography_hex27",
    "hex27_from_control_nodes",
]

# Builders per mesh family, keyed by ``sweep.sem.fields.mesh_kind``. Lets a
# propagator pick the mesh from the equation's own signature instead of a chain
# of ``is``/``in`` tests it has to keep in step with the equation list.
MESH_BUILDERS = {
    "quad4": (build_quad4_mesh, None),
    "quad9": (build_axis_aligned_quad9, build_topography_quad9),
    "hex8":  (build_hex8_mesh, None),
    "hex27": (build_axis_aligned_hex27, build_topography_hex27),
}

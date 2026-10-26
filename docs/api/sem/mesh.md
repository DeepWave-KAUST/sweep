# Meshes, basis and C-PML

`PropSEM` builds all of this from `shape`, `n_order`, `topography` and `abcn`.
The builders are public for driving [`SEMPropagator`](low_level.md) directly:
a mesh of your own, a C-PML profile tuned by hand, or an equation `PropSEM`
does not drive.

## GLL basis

::: sweep.sem.gll_nodes_weights

::: sweep.sem.lagrange_derivative_matrix

## 2-D meshes

`Quad4Mesh` is the flat, axis-aligned mesh with a constant Jacobian; `Quad9Mesh`
is the curved one (biquadratic elements) that follows a topography.

::: sweep.sem.Quad4Mesh

::: sweep.sem.build_quad4_mesh

::: sweep.sem.Quad9Mesh

::: sweep.sem.build_axis_aligned_quad9

::: sweep.sem.build_topography_quad9

::: sweep.sem.quad9_from_control_nodes

::: sweep.sem.boundary_element_edges

## 3-D meshes

`Hex8Mesh` and `Hex27Mesh` are the 3-D counterparts of `Quad4Mesh` and
`Quad9Mesh`.

::: sweep.sem.Hex8Mesh

::: sweep.sem.build_hex8_mesh

::: sweep.sem.Hex27Mesh

::: sweep.sem.build_axis_aligned_hex27

::: sweep.sem.build_topography_hex27

::: sweep.sem.hex27_from_control_nodes

::: sweep.sem.boundary_element_faces

## Mass matrix

::: sweep.sem.assemble_mass_vector

## C-PML profiles

One profile per equation family; the equation's `build` takes it as `cpml`
(the coupled fluid-solid form takes an acoustic and an elastic one).

::: sweep.sem.CPMLProfile

::: sweep.sem.build_cpml

::: sweep.sem.CPML3DProfile

::: sweep.sem.build_cpml3d

::: sweep.sem.ElasticCPMLProfile

::: sweep.sem.build_elastic_cpml

::: sweep.sem.SimpleElasticCPMLProfile

::: sweep.sem.build_simple_elastic_cpml

::: sweep.sem.SimpleElasticCPML3DProfile

::: sweep.sem.build_simple_elastic_cpml_3d

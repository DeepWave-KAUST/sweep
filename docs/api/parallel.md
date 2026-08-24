# Parallel (domain decomposition)

Model-parallel / domain-decomposition entry points. Usage guide:
[Domain decomposition](../user-guide/parallel.md); hands-on notebooks
[25](../notebooks/25_domain_decomposition.ipynb) and
[26](../notebooks/26_dd_overthrust_3d.ipynb).

## Rank grid

::: sweep.parallel.MeshTopology

## The DD propagator wrapper

<!-- full module path: the top-level re-export is lazy (module __getattr__),
     which griffe's static collection cannot see -->
::: sweep.parallel.dd_propagator.ModelParallel

## Mesh padding helpers

::: sweep.parallel.pad_to_mesh

::: sweep.parallel.unpad_from_mesh

## Building blocks

`ModelParallel` composes these; they are public so a caller can drive a mesh
by hand (a custom halo pattern, a static partition computed ahead of time)
without reimplementing the arithmetic.

::: sweep.parallel.mesh.ModelParallelMesh

::: sweep.parallel._topology.balanced_grid

::: sweep.parallel.routing.partition_global_coords

::: sweep.parallel.pml.build_rank_pml_widths

::: sweep.parallel.halo.HaloExchange

::: sweep.parallel.halo.exchange_halos

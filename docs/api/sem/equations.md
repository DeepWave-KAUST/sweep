# SEM equations

The equation classes of `sweep.sem.equations`, all importable as
`sweep.sem.<Class>`. A class is not constructed by hand: it is passed to
[`PropSEM`](prop_sem.md), which builds the mesh, interpolates the models onto
its nodes and calls the class's `build`:

```python
from sweep.sem import PropSEM, AcousticPotentialSEM2D

solver = PropSEM(AcousticPotentialSEM2D, shape=(141, 681), dh=25.0, dt=0.002)
seis = solver(wavelet, sources, receivers, models=[vp])
```

`build(mesh, <models>, *, ...)` is that call. Its positional arguments after
the mesh are the models as element arrays (`vp_e` is `vp` at the GLL nodes of
every element), in the order `models=` takes them, except `rho` on
`AcousticPotentialSEM2D`, which `PropSEM` fills from `density=`. Its keywords
are what `PropSEM` sets up from its own options, such as the free surface and
the C-PML profile. The mesh family, the models, the CUDA kernels and the
boundary-saving coverage of every class are in the
[SEM family table](../../user-guide/equations.md#sem-family-spectral-elements);
`sweep show <Class>` prints the fields and the source and receiver types.

## Registry

Equations are discovered, not listed: every class with `field_specs` in a
module of `sweep.sem.equations` is one.

::: sweep.sem.equations.equation_classes
    options:
      heading_level: 3

::: sweep.sem.equations.summary
    options:
      heading_level: 3

::: sweep.sem.equations.describe
    options:
      heading_level: 3

## Acoustic

The potential form (SPECFEM's) is the FWI workhorse: shot batching, per-shot
models and batched boundary saving on `impl='c'` are there. Its `build` takes
a density model; `PropSEM` gives it the constant `density=`. The other four
take `vp` alone, one per mesh family.

::: sweep.sem.AcousticPotentialSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Elastic, displacement form

Second-order in time (SPECFEM's Newmark scheme); the source and receiver fields
are `u_x`, `u_z` (and `u_y`). `ElasticSEM2D`, `ElasticSEM3D` and
`ElasticSEM3DDeformed` model no C-PML: `PropSEM` refuses an `abcn` for them.

::: sweep.sem.ElasticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Elastic, velocity-stress form

First-order in time; the source and receiver fields are `v_x`, `v_z` (and
`v_y`). The stresses are element-local, so they take no receivers.

::: sweep.sem.ElasticVSSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVTISEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVSSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVSSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Coupled fluid-solid

An acoustic potential in the water over an elastic solid, with a C-PML on
each; records `chi`, `u_x`, `u_z` and the derived `p`, `vx`, `vz`.

::: sweep.sem.AcousticElasticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

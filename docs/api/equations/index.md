# Equations

This section documents the equation classes in `sweep.equations`.

## Overview

An equation defines:

- the physical model parameters required by the solver
- the wavefields carried during propagation
- the user-facing source and receiver field choices
- the numerical update rule used by the propagator
- whether a compiled PyTorch extension binding exists

## Shared Equation Interface

Every equation exposes:

- `models`: ordered model names required by `solver(..., models=[...])`
- `wavefields`: full internal wavefield list, including auxiliary and CPML fields

Most user-facing equations now also expose structured metadata through
`FieldSpec` and `ModelSpec`, plus helper methods on the base `WaveEquation`.

Field metadata helpers:

- `available_fields()`
- `available_fields(role="source")`
- `available_fields(role="receiver")`
- `describe_field(name)`
- `default_source_fields`
- `default_receiver_fields`

Model metadata helpers:

- `available_models()`
- `describe_model(name)`

`available_fields()` is the recommended way to discover what users should pass
into `source_type` and `receiver_type`. By default it filters out internal and
boundary-related fields such as CPML memory variables.

`available_models()` is the recommended way to discover the expected model
order for `solver(..., models=[...])`.

## Metadata Types

- `FieldSpec`
  Describes one wavefield entry. It carries the canonical name, aliases,
  description, source/receiver support flags, and whether the field is internal
  or boundary-related.
- `ModelSpec`
  Describes one model parameter. It carries the canonical name, aliases,
  description, units, and whether the parameter is required.

## Ordering Semantics

Field and model metadata are not just documentation.

- `FieldSpec` order is semantically significant.
  The propagators map source and receiver names to positional wavefield
  indices, and equation step functions are expected to return tensors in the
  same order.
- `ModelSpec` order defines the required order of `solver(..., models=[...])`.
  If an equation lists `["vp", "vs", "rho"]`, then the runtime model list must
  follow the same order.

For equations that support the compiled extension backend, runtime buffer metadata is
now grouped under:

- `cuda_layout`

This replaces the older pattern of scattering extension-specific scalar properties
through the equation class.

!!! note

    In equation constructors, the `backend` argument refers to the tensor and
    operator backend, not the propagator implementation. For the compiled `c`
    implementation, the equation `backend` should still normally be `"torch"`,
    not `"cuda"`.

## API Tabs

=== "Acoustic"

    ```python
    class Acoustic(
        spatial_order=4,
        device="cpu",
        backend="torch",
        dim=2,
    )
    ```

    Second-order 2D acoustic equation with CPML-compatible auxiliary fields.

    See [Acoustic](acoustic.md) for parameter meanings.

=== "Acoustic3D"

    ```python
    class Acoustic3D(
        spatial_order=4,
        device="cpu",
        backend="torch",
        dim=3,
    )
    ```

    Second-order 3D acoustic equation.

    See [Acoustic3D](acoustic3d.md) for parameter meanings.

=== "Elastic"

    ```python
    class Elastic(
        spatial_order=4,
        device="cpu",
        backend="torch",
    )
    ```

    First-order 2D elastic velocity-stress equation.

    See [Elastic](elastic.md) for parameter meanings.

=== "Elastic3D"

    ```python
    class Elastic3D(
        spatial_order=4,
        device="cpu",
        backend="torch",
    )
    ```

    First-order 3D elastic velocity-stress equation.

    See [Elastic3D](elastic3d.md) for parameter meanings.

=== "ElasticTTI"

    ```python
    class ElasticTTI(
        spatial_order=4,
        device="cpu",
        backend="torch",
    )
    ```

    First-order TTI elastic equation (non-staggered grid, eager-only).

    See [ElasticTTI](elastic_tti.md) for parameter meanings.

=== "ElasticTTISG"

    ```python
    class ElasticTTISG(
        spatial_order=4,
        device="cpu",
        backend="torch",
    )
    ```

    First-order TTI elastic equation on a rotated staggered grid; supports the
    compiled C++ / CUDA binding.

    See [ElasticTTISG](elastic_tti_sg.md) for parameter meanings; the 3-D
    variant is [ElasticTTISG3D](elastic_tti_sg3d.md) (axis-aligned SG, 21
    Bond-rotated stiffness entries, full/bs/ckpt CUDA backward).

=== "DAS family"

    ```python
    from sweep.equations import (
        DAS,                            # unified facade (was DASModeler)
        DASElastic, DASElastic3D,
        DASMu, DASMu3D,
        DASZhao, DASZhao3D,
    )
    ```

    Strain-rate / helical-fiber receiver equations for DAS modelling. The
    six raw equation variants share `['vp', 'vs', 'rho']` and ship the
    compiled binding. `DAS` is the unified facade — pass `method="zhao"`
    or `method="mu"` to pick a variant.

    See [DAS family](das.md) for the comparison table.

## Equation Pages

The following pages document constructor parameters, required models,
wavefields, and backend / binding behavior:

**Acoustic family**

- [Acoustic](acoustic.md)
- [Acoustic3D](acoustic3d.md)
- [Acoustic1st](acoustic1st.md)
- [AcousticVRZ](acoustic_vrz.md)
- [AcousticLSRTM](acoustic_lsrtm.md)
- [AcousticLSRTM3D](acoustic_lsrtm3d.md)

**Elastic family**

- [Elastic](elastic.md)
- [Elastic3D](elastic3d.md)
- [ElasticTTI](elastic_tti.md)
- [ElasticTTISG](elastic_tti_sg.md)
- [ElasticTTISG3D](elastic_tti_sg3d.md)
- [ElasticTTI2nd](elastic_tti_2nd.md)

**DAS family**

- [DAS family overview](das.md) — covers `DAS` (the unified facade, formerly
  named `DASModeler`), plus the raw equation classes `DASElastic` /
  `DASElastic3D`, `DASMu` / `DASMu3D`, `DASZhao` / `DASZhao3D`.

**Anisotropic acoustic family**

The raw equation classes are:

| Class | Symmetry | Dim | Reference |
| --- | --- | --- | --- |
| `AcousticVTI` | VTI | 2D | Liang K. et al. 2022 (2nd-order pseudo-acoustic) |
| `AcousticTTI` | TTI | 2D | Liang K. et al. 2022 (TTI extension) |
| `AcousticTariq` | VTI / TTI | 2D | Alkhalifah 2000 (qP η-formulation) |
| `AcousticVTI1st` / `AcousticVTI1st3D` | VTI | 2D / 3D | Duveneck et al. 2008 (1st-order velocity-stress) |

`AcousticAniso` is a thin **factory** that dispatches by `(method, symmetry,
ndim)` and returns one of the above instances; it does **not** wrap a solver.
The user constructs `PropTorch` separately:

```python
from sweep.equations import AcousticAniso
from sweep.propagator.torch import PropTorch

eq = AcousticAniso(method="duveneck", symmetry="vti", ndim=2)
prop = PropTorch(eq, shape=(nz, nx), source_type=["sH", "sV"],
                 receiver_type=["vz"], dh=10.0, dt=1e-3, abcn=30)
record = prop(wavelet, sources, receivers, models=[vp, eps, delta, rho])
```

Author-named aliases also exist for the raw classes: `AcousticVTILiang`,
`AcousticTTILiang`, `AcousticVTIAlkhalifah`, `AcousticTTIAlkhalifah`,
`AcousticVTIDuveneck`, `AcousticVTIDuveneck3D`.  Use `sweep show <ClassName>`
from the CLI to inspect any of them; the [CLI page](../../user-guide/cli.md)
has a full output example.

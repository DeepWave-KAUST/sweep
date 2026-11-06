# Equations

This section documents the equation classes in `sweep.equations`.

## Overview

An equation defines:

- the physical model parameters required by the solver
- the wavefields carried during propagation
- the user-facing source and receiver field choices
- the numerical update rule used by the propagator
- whether it has CUDA-core bindings (`impl='c'`: `C_NAME` declared, `supports_torch_binding()` True)

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

For equations with CUDA-core bindings, runtime buffer metadata is
now grouped under:

- `cuda_layout`

This replaces the older pattern of scattering core-specific scalar properties
through the equation class.

!!! note

    In equation constructors, the `backend` argument refers to the tensor and
    operator backend, not the propagator implementation. For the compiled `c`
    implementation, the equation `backend` should still normally be `"torch"`,
    not `"cuda"`.

## Absorbing boundaries

Outgoing waves are absorbed in a layer of `abcn` cells (default 50) that the
propagator adds outside the model on every face that is not a free surface.
The propagator argument `pml_type` selects the formulation. Every equation
accepts only its `default_pml_type`, except `Acoustic1st`, which also accepts
`spml`.

Both convolutional PML (CPML) forms replace a spatial derivative inside a
layer normal to $x$ by the stretched derivative

$$
\frac{\partial}{\partial \tilde{x}} = \frac{1}{s_x} \frac{\partial}{\partial x},
\qquad
s_x = 1 + \frac{d_x}{\alpha_x + i\omega},
$$

with no real stretching factor ($\kappa_x = 1$). In the time domain $1/s_x$
is a convolution, which is updated recursively at every time step with the
coefficients

$$
a_x = e^{-(d_x + \alpha_x)\Delta t},
\qquad
b_x = \frac{d_x}{d_x + \alpha_x} \left( a_x - 1 \right).
$$

The same rules apply along $y$ and $z$.

### Damping profiles

`cpmlr` and `cpmls` share the profiles

$$
d_x = d_0 \left( \frac{\ell}{L} \right)^{2},
\qquad
d_0 = -\frac{3\, v_{\max} \ln R}{2 L_{\max}},
\qquad
\alpha_x = \pi f_{\mathrm{pml}} \left( 1 - \frac{\ell}{L} \right),
$$

that is, a power $n = 2$ in $d_0 = -(n+1)\, v_{\max} \ln R / (2L_{\max})$,
with $R = 10^{-3}$. Here $\ell/L \in [0, 1]$ is the distance into the layer
on that side, measured in cells from its inner edge and divided by the
layer's width; $L_{\max}$ is the physical thickness of the thicker of the
two layers along the axis. $v_{\max}$ and $f_{\mathrm{pml}}$ are the keyword
arguments `max_vel` (default 4500 m/s) and `pml_freq` (default 25 Hz) of the
propagator call; $v_{\max}$ is not taken from the model. Where $\ell = 0$,
which includes the whole interior, $a_x = b_x = 0$, so the memory variables
stay zero and the derivatives are unchanged.

### `cpmlr`: second-order equations

The second-order equations (`Acoustic`, `Acoustic3D`, `AcousticBorn`,
`AcousticVRZ`, `ViscoAcoustic` and others) replace each term
$\partial_{xx} u$ of the Laplacian by
$\partial_{\tilde{x}} ( \partial_{\tilde{x}} u )$, following Pasalic and
McGarry (2010). The two convolutions are carried by two memory variables per
axis, `psi*` and `zeta*`:

$$
\psi_x^{n} = a_x \psi_x^{n-1} + b_x\, \partial_x u^{n},
\qquad
\zeta_x^{n} = a_x \zeta_x^{n-1}
 + b_x \left( \partial_{xx} u^{n} + \partial_x \psi_x^{n} \right),
$$

$$
\partial_{\tilde{x}} \left( \partial_{\tilde{x}} u \right)
 \approx \partial_{xx} u^{n} + \partial_x \psi_x^{n} + \zeta_x^{n}.
$$

The step evaluates $\partial_x \psi_x^{n}$ as
$\partial_x (a_x \psi_x^{n-1}) + (\partial_x b_x)\, \partial_x u^{n}
+ b_x\, \partial_{xx} u^{n}$, so each axis carries three profiles,
`a, b, db`, where `db` is the central-difference derivative of $b_x$ along
the axis. The first derivatives $\partial_x u$ and $\partial_x(a_x\psi_x)$
are central differences; the second derivative $\partial_{xx} u$ is the
equation's own stencil of order `spatial_order`.

### `cpmls`: staggered-grid first-order equations

The first-order equations on staggered grids (`Acoustic1st`, `Elastic`,
`Elastic3D` and others) follow Komatitsch and Martin (2007). Each spatial
derivative in the step that is taken inside a layer has its own memory
variable:

$$
\psi^{n} = a_x \psi^{n-1} + b_x\, \partial_x u,
\qquad
\partial_{\tilde{x}} u = \partial_x u + \psi^{n},
$$

where $\partial_x u$ is the staggered derivative at the time level used by
that update. Derivatives that fall on integer nodes use the profiles `a, b`,
and derivatives that fall half a cell forward use `ah, bh`, the same
formulas evaluated at $x + \Delta x/2$. Each axis therefore carries four
profiles, `a, b, ah, bh`.

### `spml`: split-field PML (`Acoustic1st` only)

With `pml_type="spml"`, `Acoustic1st` splits the pressure into
$p = p_x + p_z$ and damps each pressure component and each particle
velocity with the profile of its own axis, without the frequency shift
$\alpha$:

$$
\left( 1 + \tfrac{1}{2} d_x \Delta t \right) q^{\mathrm{new}}
 = \left( 1 - \tfrac{1}{2} d_x \Delta t \right) q^{\mathrm{old}}
 + \Delta t\, F_x,
$$

where $q$ is $v_x$ or $p_x$ and $F_x$ is the undamped right-hand side of
its update ($z$ likewise). The profile is
$d_x = -3\, v_{\max} \ln R\, (\ell / L)^2 / (2L)$ with $R = 10^{-5}$,
$\ell$ the physical distance into the layer (one cell at its inner edge)
and $L$ the thickness of the thicker of the two layers along the axis.
`pml_freq` has no effect in this mode.

### References

- Komatitsch, D., and R. Martin, 2007, An unsplit convolutional perfectly
  matched layer improved at grazing incidence for the seismic wave equation:
  Geophysics, 72(5), SM155–SM167,
  [doi:10.1190/1.2757586](https://doi.org/10.1190/1.2757586).
- Pasalic, D., and R. McGarry, 2010, Convolutional perfectly matched layer for
  isotropic and anisotropic acoustic wave equations: SEG Technical Program
  Expanded Abstracts 2010, 2925–2929,
  [doi:10.1190/1.3513453](https://doi.org/10.1190/1.3513453).

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
        spatial_order=8,
        device="cpu",
        backend="torch",
        checkerboard_smoothing=True,
    )
    ```

    First-order 2-D three-component TTI elastic equation on a rotated
    staggered grid (RSG); eager-only.

    See [ElasticTTI](elastic_tti.md) for parameter meanings.

=== "ElasticTTISG"

    ```python
    class ElasticTTISG(
        spatial_order=8,
        device="cpu",
        backend="torch",
    )
    ```

    First-order 2-D three-component TTI elastic equation on an axis-aligned
    staggered grid; supports the CUDA core (`impl='c'`).

    See [ElasticTTISG](elastic_tti_sg.md) for parameter meanings; the 3-D
    variant is [ElasticTTISG3D](elastic_tti_sg3d.md) (axis-aligned SG, 21
    Bond-rotated stiffness entries, full/bs/ckpt CUDA backward).

=== "DAS family"

    ```python
    from sweep.equations import (
        DAS,                            # unified facade (was DASModeler)
        DASElastic, DASElastic3D,       # aliases of DASZhao / DASZhao3D
        DASMu, DASMu3D,
        DASZhao, DASZhao3D,
    )
    ```

    Strain-rate / helical-fiber receiver equations for DAS modelling. The
    four raw equation classes (`DASZhao` / `DASZhao3D`, `DASMu` / `DASMu3D`)
    share `['vp', 'vs', 'rho']` and have CUDA-core bindings; `DASElastic` /
    `DASElastic3D` are aliases of `DASZhao` / `DASZhao3D`. `DAS` is the
    unified facade — pass `method="zhao"` or `method="mu"` to pick a variant.

    See [DAS family](das.md) for the API and the
    [DAS table](../../user-guide/equations.md#distributed-acoustic-sensing-das-family)
    in the user guide for a comparison.

## Equation Pages

The following pages document constructor parameters, required models,
wavefields, and backend / binding behavior:

**Acoustic family**

- [Acoustic](acoustic.md)
- [Acoustic3D](acoustic3d.md)
- [Acoustic1st](acoustic1st.md)
- [AcousticVRZ / AcousticVRZ3D](acoustic_vrz.md)
- [AcousticBorn](acoustic_born.md)
- [AcousticBorn3D](acoustic_born3d.md)
- [ViscoAcoustic](visco_acoustic.md)

**Elastic family**

- [Elastic](elastic.md)
- [Elastic3D](elastic3d.md)
- [ElasticBorn](elastic_born.md) — Born modelling (background + scattered
  field), the elastic LSRTM / RWI operator
- [ViscoElastic](visco_elastic.md)
- [ElasticTTI](elastic_tti.md)
- [ElasticTTISG](elastic_tti_sg.md)
- [ElasticTTISG3D](elastic_tti_sg3d.md)
- [ElasticTTI2nd](elastic_tti_2nd.md)

**Poroelastic**

- [Biot](biot.md) — low-frequency Biot poroelastic waves (solid velocity, Darcy
  flux, total stress, pore pressure); eager only

**DAS family**

- [DAS family overview](das.md) — covers `DAS` (the unified facade, formerly
  named `DASModeler`), plus the raw equation classes `DASZhao` /
  `DASZhao3D` and `DASMu` / `DASMu3D` (`DASElastic` / `DASElastic3D` are
  aliases of `DASZhao` / `DASZhao3D`).

**Other equations**

- [Other equations](others.md) — curvilinear-grid topography
  (`AcousticCurvilinear`, `ElasticCurvilinear`; eager-only), vector
  reflectivity (`AcousticVRR`, `ElasticVRR`), and the anisotropic acoustic
  classes below with the `AcousticAniso` factory.

**Anisotropic acoustic family**

The raw equation classes (API on [Other equations](others.md)) are:

| Class | Symmetry | Dim | Reference |
| --- | --- | --- | --- |
| `AcousticVTI` | VTI | 2D | Liang K. et al. 2023 (2nd-order pure qP) |
| `AcousticTTI` | TTI | 2D | Liang K. et al. 2023 (TTI extension) |
| `AcousticTariq` | VTI | 2D | Alkhalifah 2000 (qP η-formulation) |
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
`AcousticTTILiang`, `AcousticVTIAlkhalifah`, `AcousticVTIDuveneck`,
`AcousticVTIDuveneck3D`.  Use `sweep show <ClassName>`
from the CLI to inspect any of them; the [CLI page](../../user-guide/cli.md)
has a full output example.

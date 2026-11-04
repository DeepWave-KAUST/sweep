# Symbolic PDEs

`sweep.symbolic` turns a wave equation written as a PDE into an eager equation
class. You write the equation; sweep writes the time step, the CPML terms and
the field tables, and the class runs through `PropTorch(impl="eager")` like any
shipped equation, with gradients from autograd, `torch.compile`, eager boundary
saving and strip-held CPML memory.

```python
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Eq, Field, Model, laplace

vp = Model("vp", unit="m/s")
p = Field("p", source=True, receiver=True)

MyAcoustic = PDE(
    [Eq(p.dt2, vp**2 * laplace(p))],
    name="MyAcoustic",
    free_surface="zero",
).build(ndim=2)

solver = PropTorch(MyAcoustic(spatial_order=8, device="cuda"), shape=(nz, nx),
                   dh=10.0, dt=1e-3, nt=nt, dev="cuda", impl="eager")
record = solver(wavelet, sources, receivers, models=[vp_model])
```

This PDE restates [`Acoustic`](../api/equations/acoustic.md), and the generated
class reproduces it bit for bit, record and gradient, in 2-D and (with
`build(ndim=3)`) against `Acoustic3D`: with and without a free surface, per
edge, on a topography, with `EagerOptions(cpml_strips=True)`, compiled or not,
with `Full()` or `BoundarySaving()`.

## Writing a PDE

| Object | Meaning |
| --- | --- |
| `Model("vp", unit=, aliases=, description=)` | a model parameter; the propagator's `models=` takes them in declaration order (or the order given by `PDE(models=...)`) |
| `Field("p", source=, receiver=, aliases=, description=)` | a wavefield advanced by the PDE; its name is what `source_type` / `receiver_type` take |
| `p.dt2` | second time derivative |
| `p.dz2`, `p.dy2`, `p.dx2` | second derivative along one axis |
| `laplace(p)` | sum of the second derivatives over every axis of the build |
| `Eq(lhs, rhs)` | one equation; each field has exactly one |

Coefficients are sympy expressions of models and numbers, with `sqrt`, `exp`,
`log`, the trigonometric and hyperbolic functions and `Abs` available
(`sympy.sqrt(1 + 2 * delta)`). The left-hand side may carry a factor
(`Eq(m * p.dt2, laplace(p))`); it is divided out.

A system couples any number of fields. A pseudo-acoustic VTI system, for
example:

```python
import sympy

eps, delta = Model("eps"), Model("delta")
q = Field("q", receiver=True)

VTI = PDE([
    Eq(p.dt2, vp**2 * (1 + 2 * eps) * p.dx2 + vp**2 * sympy.sqrt(1 + 2 * delta) * q.dz2),
    Eq(q.dt2, vp**2 * sympy.sqrt(1 + 2 * delta) * p.dx2 + vp**2 * q.dz2),
], name="VTI", free_surface="zero").build(ndim=2)
```

### What is supported

- Second-order-in-time fields; the right-hand side linear in the fields, each
  term a model-only coefficient times a second derivative along one axis or
  times a field.
- 2-D and 3-D, any even `spatial_order`.
- `pml="cpml"` (default): C-PML on every second derivative, the update the
  shipped second-order equations use (`cpmlr`); `pml="none"` drops the
  absorbing terms.
- `free_surface="zero"`: pressure release, every field zeroed above the
  surface; flat, per edge (2-D), or an image-method topography.
  `free_surface=None` (default) builds a class that refuses the propagator's
  `free_surface=True` and `topography=`.

Everything else raises when the `PDE` is constructed, with the reason: first
spatial derivatives, first time derivatives (first-order systems, damping),
terms nonlinear in the fields, terms without a field (sources are injected by
the propagator), symbols that are not models, a field used without its own
equation. A `y` derivative raises at `build(ndim=2)`.

## The generated class

`build(ndim)` writes the class as Python source and executes it.
`PDE.source(ndim)` returns that source and `PDE.export(path, ndim)` saves it as
a module, which imports and runs on its own: read it to see exactly what is
computed, or commit it as the starting point of a hand-written equation.

The step is written the way the shipped equations write theirs: the same
operators (`separable_d2_2d/3d`, `gradient`, `cpml_axis_update`), in the same
order (every Laplacian, then every first derivative, then the CPML update of
each axis followed by its in-place sum). Autograd accumulates gradients and
Inductor fuses in the order the graph was built, so another order of the same
operations would move an uncompiled gradient by about 1e-7.

| Generated | From |
| --- | --- |
| `MODEL_SPECS` | the models the equations use |
| `FIELD_SPECS` | each field and its previous time level (`p`, `p_prev`), then the CPML memory fields of each second derivative taken (`psix`, `zetax`, ...; suffixed with the field name in a system) |
| `default_source_fields` / `default_receiver_fields` | the first field with `source=True` / `receiver=True` |
| `func` | the time step |

`build(ndim, register=True)` also registers the class under `name`, so
`get_equation(name)` and `sweep list equations` find it. A rebuild under the
same name (a notebook cell run again) replaces the registration; a name a
shipped equation holds raises.

The class runs on `backend="torch"` with `impl="eager"`. It has no compiled
`impl="c"` kernels; an exported module can be given them by hand as described
in [Extending](extending.md#part-2-adding-a-compiled-cuda-kernel-implc).

## Accuracy

| Check | Result |
| --- | --- |
| `p.dt2 = vp**2 laplace(p)` vs `Acoustic` / `Acoustic3D` | bit for bit, record and gradient, 54 configurations on CPU and CUDA |
| Coupled VTI system, finite differences (canonical 48 x 56 case, fp32) | error falls as h² down to the fp32 floor, ~1e-3 |
| Coupled VTI system, `BoundarySaving` vs `Full()` | 2e-5 to 9e-5 (`Acoustic` on the same case: 1.4e-5) |
| `torch.compile(fullgraph=True)` | one graph per step |

On CUDA, an uncompiled run is deterministic only with
`torch.backends.cudnn.deterministic = True`; otherwise cuDNN chooses
non-deterministic backward algorithms and the gradient moves by about 3e-7
from run to run, for the shipped equations as well.

## Memory

A restated `Acoustic` / `Acoustic3D` needs the same CUDA memory as the shipped
class, in 2-D and 3-D: the same memory held for the backward pass and, to within
0.1 MB, the same peak, compiled or not, with `Full()`, checkpointing and
boundary saving.

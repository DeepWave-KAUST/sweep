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
| `Coefficient("mu", rho * vs**2)` | a coefficient derived from the models, computed once per run before the time loop |
| `Field("p", source=, receiver=, aliases=, description=)` | a second-order wavefield; its name is what `source_type` / `receiver_type` take |
| `Field("vx", time_order=1, stagger="x", ...)` | a field of a first-order system, half a cell after the node along x (see below) |
| `p.dt2`, `vx.dt` | second / first time derivative |
| `p.dz2`, `p.dy2`, `p.dx2` | second derivative along one axis (second-order fields) |
| `laplace(p)` | sum of the second derivatives over every axis of the build |
| `vx.dz`, `vx.dy`, `vx.dx` | first derivative along one axis (first-order systems) |
| `Eq(lhs, rhs)` | one equation; each field has exactly one |

Coefficients are sympy expressions of models, coefficients and numbers, with
`sqrt`, `exp`, `log`, the trigonometric and hyperbolic functions and `Abs`
available (`sympy.sqrt(1 + 2 * delta)`). The left-hand side may carry a factor
(`Eq(m * p.dt2, laplace(p))`); it is divided out. A coefficient written inline
is computed at every step; a `Coefficient` once per run, in the class's
`prepare_models`, which the propagator calls before the time loop.

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

## First-order systems

A system of first-order fields runs on a staggered grid. Each field declares
where it sits: `stagger=None` on the integer nodes, `"x"` half a cell after the
node along x, `"-x"` half a cell before it, `("z", "x")` along both. A
derivative is then a staggered difference from the field's position to the
position of the field whose equation reads it: forward when that field sits
half a cell after, backward when half a cell before. The two fields must sit
alike along the other axes.

The 2-D velocity-stress system of Virieux (1986), with the Lamé parameters
computed once per run:

```python
from sweep.symbolic import Coefficient

vp, vs, rho = Model("vp", unit="m/s"), Model("vs", unit="m/s"), Model("rho", unit="kg/m^3")
lam = Coefficient("lam", rho * (vp**2 - 2 * vs**2))
mu = Coefficient("mu", rho * vs**2)
lam2mu = Coefficient("lam2mu", lam + 2 * mu)

vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
vz = Field("vz", time_order=1, stagger="z", source=True, receiver=True)
sxx = Field("sxx", time_order=1, source=True, receiver=True)
szz = Field("szz", time_order=1, source=True, receiver=True)
sxz = Field("sxz", time_order=1, stagger=("z", "x"), source=True)

MyElastic = PDE([
    Eq(vz.dt, (szz.dz + sxz.dx) / rho),
    Eq(vx.dt, (sxx.dx + sxz.dz) / rho),
    Eq(szz.dt, lam2mu * vz.dz + lam * vx.dx),
    Eq(sxx.dt, lam2mu * vx.dx + lam * vz.dz),
    Eq(sxz.dt, mu * (vx.dz + vz.dx)),
], name="MyElastic").build(ndim=2)
```

This restates [`Elastic`](../api/equations/elastic.md) bit for bit. The
pressure-velocity system of [`Acoustic1st`](../api/equations/acoustic1st.md)
places its velocities half a cell before the pressure node:

```python
p = Field("p", time_order=1, source=True, receiver=True)
vx = Field("vx", time_order=1, stagger="-x", receiver=True)
vz = Field("vz", time_order=1, stagger="-z", receiver=True)

MyAcoustic1st = PDE([
    Eq(vz.dt, -p.dz / rho),
    Eq(vx.dt, -p.dx / rho),
    Eq(p.dt, -rho * vp**2 * (vx.dx + vz.dz)),
], name="MyAcoustic1st").build(ndim=2)
```

The fields are updated in the order the equations are written, and a later
equation reads the fields earlier ones updated (a leapfrog). Runs of equations
that read none of the fields the run updates form the sub-steps of the step
(here the velocities, then the stresses). The class also returns them, without
the CPML terms, as `interior_substeps()`: eager boundary saving composes them
in reverse order at `-dt` to step back in time exactly. Every derivative gets a
C-PML memory field (`cpmls`), shared by the equations that read it; the
half-node damping profiles serve a derivative taken at a staggered position.

### What is supported

- Second-order-in-time fields; the right-hand side linear in the fields, each
  term a model-only coefficient times a second derivative along one axis or
  times a field.
- First-order systems; each right-hand side a model-only coefficient times
  first derivatives of the other fields, or a sum of such terms.
- 2-D and 3-D, any even `spatial_order`.
- `pml="cpml"` (default): C-PML on every spatial derivative, the updates the
  shipped equations use (`cpmlr` for second-order fields, `cpmls` for
  first-order systems); `pml="none"` drops the absorbing terms.
- `free_surface="zero"` (second-order fields): pressure release, every field
  zeroed above the surface; flat, per edge (2-D), or an image-method
  topography. `free_surface=None` (default) builds a class that refuses the
  propagator's `free_surface=True` and `topography=`.

Everything else raises when the `PDE` is constructed, with the reason: first
derivatives in a second-order equation, a system mixing first- and
second-order fields, damping terms (`f.dt` next to `f.dt2`), terms nonlinear in
the fields, terms without a field (sources are injected by the propagator),
symbols that are neither models nor coefficients, a field used without its own
equation; in a first-order system also a free surface, a field without a
derivative on a right-hand side, a derivative between fields at the same
position (collocated) or a whole cell apart, and a derivative read before and
after its field is updated. A `y` derivative or a `y` stagger raises at
`build(ndim=2)`.

## The generated class

`build(ndim)` writes the class as Python source and executes it.
`PDE.source(ndim)` returns that source and `PDE.export(path, ndim)` saves it as
a module, which imports and runs on its own: read it to see exactly what is
computed, or commit it as the starting point of a hand-written equation.

The step is written the way the shipped equations write theirs: the same
operators and the same order. For second-order fields, `separable_d2_2d/3d`,
`gradient` and `cpml_axis_update`: every Laplacian, then every first
derivative, then the CPML update of each axis followed by its in-place sum.
For a first-order system, the staggered differences and `cpml_update`: per
sub-step, its derivatives, then per equation the CPML corrections of its
derivatives (along z, y, x) and its update. Autograd accumulates gradients in
the order the graph was built, so another order of the same operations would
move an uncompiled gradient by about 1e-7.

| Generated | From |
| --- | --- |
| `MODEL_SPECS` | the models the equations use (coefficients included) |
| `FIELD_SPECS` | second order: each field and its previous time level (`p`, `p_prev`), then the CPML memory fields of each second derivative taken (`psix`, `zetax`, ...; suffixed with the field name in a system). First order: each field, then one CPML memory field per derivative taken (`m_vx_z` for `vx.dz`) |
| `default_source_fields` / `default_receiver_fields` | the first field with `source=True` / `receiver=True` |
| `prepare_models` | the coefficients, if any |
| `func` | the time step |
| `interior_substeps` | first order: the sub-steps without CPML, for eager boundary saving |

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
| Restated `Acoustic1st` and 2-D `Elastic`, uncompiled | bit for bit, record and gradient: spatial orders 4 and 8, with and without `cpml_strips`, `Full()` and `BoundarySaving()` |
| The same, compiled | `Acoustic1st` bit for bit; `Elastic` record bit for bit, gradients 8e-8 to 2.3e-7 |
| Restated 3-D `Elastic` | record bit for bit, gradients 5e-8 to 8.4e-7 |
| Elliptic anisotropic first-order acoustics, finite differences (canonical case, fp32) | error falls as h² down to the fp32 floor, 6e-5 to 2e-3 |
| `torch.compile(fullgraph=True)` | one graph per step |

The restated 3-D `Elastic` takes its derivatives and CPML corrections in an
order of its own, which the generator does not follow (it follows the 2-D
equations), hence its gradients. Compiled, Inductor orders the backward by the
operand order of each sum, which sympy does not keep as written (a sum is held
in a canonical order), hence the compiled `Elastic` gradients; written in the
hand-written operand order, the same step compiles to the same bits.

On CUDA, an uncompiled run is deterministic only with
`torch.backends.cudnn.deterministic = True`; otherwise cuDNN chooses
non-deterministic backward algorithms and the gradient moves by about 3e-7
from run to run, for the shipped equations as well.

## Memory

A restated `Acoustic` / `Acoustic3D` needs the same CUDA memory as the shipped
class, in 2-D and 3-D: the same memory held for the backward pass and, to within
0.1 MB, the same peak, compiled or not, with `Full()`, checkpointing and
boundary saving.

So do the restated first-order equations, to within one wavefield (1 MB in the
2-D runs measured, 200 x 400 with two shots; 3.5 MB in the 3-D ones), compiled
or not, with `Full()`, checkpointing and boundary saving: `Acoustic1st`, the
2-D `Elastic` (whose generated class also drops the two memory fields the
shipped one carries unused) and the 3-D `Elastic`. The generated step frees a
sub-step's derivatives at its end, and computes a sum of derivatives that
several equations read once per sub-step (the 3-D divergence), as the shipped
steps do: written out in every equation, autograd would keep a copy of the sum
per equation, 7 % more memory in a `Full()` 3-D run.

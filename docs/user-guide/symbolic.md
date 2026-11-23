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
with `Full()`, `Ckpt()` or `BoundarySaving()`. Like the shipped `Acoustic` and
`Elastic`, a 2-D generated class takes a batch of shots with a model per shot,
`(B, nz, nx)`, as well as one model shared by the batch.

## Writing a PDE

| Object | Meaning |
| --- | --- |
| `Model("vp", unit=, aliases=, description=)` | a model parameter; the propagator's `models=` takes them in declaration order (or the order given by `PDE(models=...)`) |
| `Coefficient("mu", rho * vs**2)` | a coefficient derived from the models, computed once per run before the time loop |
| `Field("p", source=, receiver=, aliases=, description=)` | a second-order wavefield; its name is what `source_type` / `receiver_type` take |
| `Field("vx", time_order=1, stagger="x", ...)` | a field of a first-order system, half a cell after the node along x (see below) |
| `p.dt2`, `vx.dt` | second / first time derivative |
| `p.dz2`, `p.dy2`, `p.dx2` | second derivative along one axis (second-order fields; in a first-order system, of the field itself: a dissipative term) |
| `laplace(p)` | sum of the second derivatives over every axis of the build |
| `vx.dz`, `vx.dy`, `vx.dx` | first derivative along one axis (first-order systems) |
| `Eq(lhs, rhs)` | one equation; each field has exactly one |

Coefficients are sympy expressions of models, coefficients and numbers, with
`sqrt`, `exp`, `log`, the trigonometric and hyperbolic functions and `Abs`
available (`sympy.sqrt(1 + 2 * delta)`). The left-hand side may carry a factor
(`Eq(m * p.dt2, laplace(p))`); it is divided out. A `Coefficient` is computed
once per run, in the class's `prepare_models`, which the propagator calls
before the time loop. A coefficient written inline is computed in the step,
together with `dt` where the step multiplies by it (`vp**2 * dt**2`,
`dt/rho`): once per rollout in an uncompiled run (once per segment when
checkpointing), as the shipped equations compute theirs, and at every step in
a compiled run.

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

This restates [`Elastic`](../api/equations/elastic.md): uncompiled, the record
and the `vp` and `vs` gradients bit for bit, the `rho` gradient to about 1e-8,
because `Elastic` computes `dt / rho` twice, for `vz` and for `vx`, and the
generated step once for both. The
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

### Dissipative terms

A first-order equation may also carry second derivatives of its own field, a
model-only coefficient times `laplace(f)`, `f.dz2`, `f.dx2` ...: viscosity,
diffusion. Heat conduction is one such term, and the unsteady Stokes equations
in Chorin's artificial-compressibility form add one to each velocity of the
pressure-velocity system:

```python
kappa = Model("kappa", unit="m^2/s")
T = Field("T", time_order=1, source=True, receiver=True)
Heat = PDE([Eq(T.dt, kappa * laplace(T))], name="Heat").build(ndim=2)

rho, nu, c = Model("rho", unit="kg/m^3"), Model("nu", unit="m^2/s"), Model("c", unit="m/s")
vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
vz = Field("vz", time_order=1, stagger="z", receiver=True)
p = Field("p", time_order=1, receiver=True)

Stokes = PDE([
    Eq(vx.dt, -p.dx / rho + nu * laplace(vx)),
    Eq(vz.dt, -p.dz / rho + nu * laplace(vz)),
    Eq(p.dt, -rho * c**2 * (vx.dx + vz.dz)),
], name="Stokes").build(ndim=2)
```

The second derivatives are taken with the second-order fields' stencil
(`separable_d2_2d/3d`), on the field's own position, from its value at the
start of the step (the one update that changes it): the term steps forward in
time (explicit Euler). It gets no CPML correction; in the absorbing layer the
field diffuses on, and beyond the padded grid it is taken as zero. A second
derivative of another field raises: it would not land on the position of the
field it updates.

The explicit step is stable only for

```text
dt <= 2 / (S * sum over the axes of coef / h_axis**2)
   =  2 h^2 / (ndim * S * nu)          (equal spacings, one coefficient nu)
```

with `coef` the largest coefficient of the term along each axis and `S` the
largest value of the stencil's symbol, reached at the Nyquist wavenumber:
`S = 4` at spatial order 2, which gives `dt <= h^2 / (2 ndim nu)`, `16/3` at
order 4, `6.04` at order 6 and `6.50` at order 8, so the bound is 0.75, 0.66
and 0.62 times `h^2 / (2 ndim nu)`. Measured on the heat equation in 2-D, at
orders 2, 4 and 8: 0.98 times the bound decays, 1.02 times grows to 1e30 within
2000 steps. The wave terms keep their own CFL condition. Forward Euler is first
order in time: an error that falls as `dt`.

A dissipative term cannot be run backwards: stepped back in time, a diffusion
amplifies every round-off error instead of damping it. Eager boundary saving
rebuilds the wavefield exactly that way, so the class has no
`interior_substeps` and refuses `BoundarySaving()`, naming the terms
(`DISSIPATIVE_TERMS`), instead of returning a wrong gradient. `Full()` and
`Ckpt()` run as usual.

### Higher derivatives

A derivative of order 3 or higher along one axis is written `u.dx3`, `u.dz4`,
`u.deriv("x", 4)` or as a chain, `u.dx2.dx2`. It is taken with
`sweep.operators.Derivative` (Fornberg weights at the class's `spatial_order`)
where its result lands on the field it updates:

- **Second-order equations take even orders of any field.** The stencil is
  centred on the shared grid, like the second derivatives. A plate's flexural
  waves, for example:

```python
kappa = Model("kappa")
u = Field("u", source=True, receiver=True)
Flexural = PDE([Eq(u.dt2, -kappa * (u.dx4 + u.dz4))], name="Flexural").build(ndim=2)
```

- **First-order systems take odd orders of the other fields.** They are
  staggered differences, like the first derivatives:
  `Eq(p.dt, -rho * c**2 * (vx.dx - beta * vx.dx3))` adds a dispersive correction
  to long waves.
- **A first-order field takes even orders of itself as dissipative terms.** For
  example `-beta * T.dx4`, a hyperdiffusion. Like `nu * laplace(v)`, such a
  term is stepped with forward Euler, and its class refuses boundary saving.

None of these derivatives gets a CPML correction. In the absorbing layer the
term runs unstretched, so a wave that a fourth derivative carries reaches the
edge of the padded grid and reflects there. Make the grid large enough, or
keep the term small next to the absorbed ones.

The explicit step is stable only for a time step below a bound set by `S_n`,
the largest value of the order-`n` stencil, reached at the Nyquist wavenumber.
For a fourth derivative, `S_4` is 16 at spatial order 2, 26.7 at order 4, 34.1
at order 6 and 39.7 at order 8. With equal spacings `h` and one coefficient on
every axis:

```text
u_tt = -kappa * sum(u.d<ax>4)   (leapfrog):      dt <= 2 h^2 / sqrt(ndim * S_4 * kappa)
T_t  = -beta  * sum(T.d<ax>4)   (forward Euler): dt <= 2 h^4 / (ndim * S_4 * beta)
```

At order 8 in 2-D, `kappa = 1` and `h = 1`, the first bound is a time step of
0.224. A fourth derivative along x alone has `ndim = 1`.

Two kinds of term are refused. An odd derivative in a second-order equation is
one. An odd derivative of a first-order field in its own equation is the
other, such as the `u_xxx` of the Airy or KdV equation. Both would need a
centred odd stencil, which returns zero for the sawtooth `(-1)**i` and so lets
odd and even points evolve apart, and forward Euler is unstable for such a
term. Mixed derivatives (`u.dx.dz`) are not supported yet.

### What is supported

- Second-order-in-time fields; the right-hand side linear in the fields, each
  term a model-only coefficient times a second derivative along one axis or
  times a field.
- First-order systems; each right-hand side a model-only coefficient times
  first derivatives of the other fields, or a sum of such terms, plus
  dissipative terms: a model-only coefficient times second derivatives of the
  field itself, explicit in time, without CPML, and no boundary saving.
- Derivatives of order 3 and higher along one axis (`u.dx4`, `v.dx3`), without
  CPML: even orders in second-order equations; in first-order systems odd
  orders of the other fields and even orders of the field itself (dissipative).
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
equation, mixed derivatives, odd derivatives of order 3 or higher in a
second-order equation; in a first-order system also a free surface, a field without a
derivative on a right-hand side (damping and coupling terms), a second
derivative of another field, an odd derivative of order 3 or higher of the field
itself, a derivative between fields at the same position
(collocated) or a whole cell apart, and a derivative read before and after its
field is updated. A `y` derivative or a `y` stagger raises at
`build(ndim=2)`.

## The system as LaTeX

`PDE.latex()` returns the system as an `aligned` environment, one equation
per line in the order written, followed by the definition of each
`Coefficient` (`coefficients=False` leaves them out); a notebook displays a
`PDE` or an `Eq` this way. The usual seismic names print as their symbols,
`vp` as `v_p`, `rho` as `\rho`, `lam` and `mu` as the Lamé letters, `vx` as
`v_x`, `sxz` as `\sigma_{xz}`, a Greek name (`epsilon`, `delta`, `kappa`) as
the letter; `symbols=` overrides or extends that, by name or by the declared
object:

```python
print(Acoustic.latex())
# \begin{aligned}
# \frac{\partial^{2} p}{\partial t^{2}} &= v_{p}^{2} \nabla^{2} p
# \end{aligned}

MyElastic.latex(symbols={"lam2mu": "M"})   # the P-wave modulus, M = λ + 2μ
```

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
derivatives (along z, y, x), the second derivatives of its dissipative terms
(`separable_d2_2d/3d`) and its update. Autograd accumulates gradients in
the order the graph was built, so another order of the same operations would
move an uncompiled gradient by about 1e-7. The products of models and `dt`
go through `once` (`sweep.equations._invariants`), keyed by their text, so an
uncompiled rollout computes each once, as the shipped equations do.

| Generated | From |
| --- | --- |
| `MODEL_SPECS` | the models the equations use (coefficients included) |
| `FIELD_SPECS` | second order: each field and its previous time level (`p`, `p_prev`), then the CPML memory fields of each second derivative taken (`psix`, `zetax`, ...; suffixed with the field name in a system). First order: each field, then one CPML memory field per derivative taken (`m_vx_z` for `vx.dz`) |
| `default_source_fields` / `default_receiver_fields` | the first field with `source=True` / `receiver=True` |
| `prepare_models` | the coefficients, if any |
| `func` | the time step |
| `interior_substeps` | first order: the sub-steps without CPML, for eager boundary saving; not for a system with dissipative terms |
| `DISSIPATIVE_TERMS` | first order: the dissipative terms (`"nu*laplace(vx)"`, ...), which make the class refuse boundary saving |

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
| `p.dt2 = vp**2 laplace(p)` vs `Acoustic` / `Acoustic3D` | bit for bit, record and gradient: 15 configurations uncompiled on CPU and on CUDA; `Full()`, `Ckpt()` and `BoundarySaving()`, compiled or not, on CUDA |
| Coupled VTI system, finite differences (canonical 48 x 56 case, fp32) | error falls as h² down to the fp32 floor, ~1e-3 |
| Coupled VTI system, `BoundarySaving` vs `Full()` | 2e-5 to 9e-5 (`Acoustic` on the same case: 1.4e-5) |
| Restated `Acoustic1st` and 2-D `Elastic`, uncompiled | bit for bit, record and gradient: spatial orders 4 and 8, with and without `cpml_strips`, `Full()`, `Ckpt()` and `BoundarySaving()`; but the `Elastic` `rho` gradient with `Full()` and `Ckpt()`, 1.1e-8 to 1.4e-8 (see above) |
| The same, compiled | `Acoustic1st` bit for bit with `Full()` and `Ckpt()`, gradients to 6e-8 with `BoundarySaving()`; `Elastic` record bit for bit, gradients 3e-8 to 9.6e-8 |
| Restated 3-D `Elastic` | record bit for bit; gradients 1.8e-8 to 3.7e-7 with `Full()`, `Ckpt()` and uncompiled `BoundarySaving()`, compiled or not; compiled with `BoundarySaving()` 7.6e-5 to 2.9e-4, the size of either class's own boundary-saving error (1.2e-4 to 3e-4 from its `Full()` gradient, the shipped class's too) |
| Elliptic anisotropic first-order acoustics, finite differences (canonical case, fp32) | error falls as h² down to the fp32 floor, 6e-5 to 2e-3 |
| Per-shot batched models, `(B, nz, nx)`, two shots, uncompiled | restated `Acoustic` and `Elastic` as the hand-written class: record bit for bit, gradients bit for bit but the `Elastic` `rho` gradient (1.3e-8); each shot vs its own single-shot run: record bit for bit, gradients 2e-9 to 5e-7 (the batched backward accumulates in another order) |
| Heat conduction, an impulse vs the heat kernel (8th order, `kappa dt / h^2 = 0.05`) | 2-D 7.7e-4 at t = 40 h²/κ, falling as dt (orders 4 and 8); 3-D 2.4e-3 at t = 20 h²/κ; the heat conserved to 5e-6 |
| Unsteady Stokes, a Gaussian vortex vs Lamb-Oseen (8th order, `nu dt / h^2 = 0.05`) | vorticity 3.6e-4 by the time the core area has quadrupled, half that at dt / 2; circulation to 1e-6; p stays at 1e-5 of rho v_max² |
| Stokes and heat gradients, finite differences (fp32) | `nu` 3.3e-5, `c` 3.2e-5, `kappa` 5.3e-5; `rho` 2.5e-4, its weak data sensitivity at the fp32 floor |
| Stokes and heat, `cpml_strips` and `Ckpt()` vs `Full()` | `cpml_strips` bit for bit, record and gradient; `Ckpt()` record bit for bit, gradients to 2.5e-7 (each checkpointed segment computes the step's products of models and `dt` anew, as for the shipped equations); compiled 1.5e-7 to 1.1e-6 |
| Flexural waves `u_tt = -u_xxxx`, a Gaussian at rest vs its Fourier solution (8th order, fp64) | 8.8e-5 at dt 0.2, 1.9e-5 at dt 0.1 (second order in time) |
| Hyperdiffusion `T_t = -T_xxxx`, the same (8th order, fp64) | 2.2e-5 at dt 0.02, 1.1e-5 at dt 0.01 (first order); grows without bound just above `2 h^4 / S_4` |
| Flexural gradient, finite differences (fp32) | `kappa` 3.6e-4, at the fp32 floor of the finite difference |
| Flexural and dispersive long waves, `Ckpt()` and `BoundarySaving()` vs `Full()` | records bit for bit; gradients 2.5e-7 to 7.3e-7 (`Ckpt()`), 4.3e-6 to 8.6e-6 (`BoundarySaving()`) |
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

Peak CUDA memory of a forward and backward pass, measured on an A100 (one
shot; 200 x 400 in 2-D, 72³ in 3-D; `abcn=12`), compiled or not, with
`Full()`, checkpointing and boundary saving:

- A restated `Acoustic` / `Acoustic3D` peaks at the shipped class's memory, to
  the byte in 2-D and to within 0.5 MB (0.14 of a wavefield) in 3-D.
- A restated `Acoustic1st` or 2-D `Elastic` peaks at no more than the shipped
  one: `Acoustic1st` up to one wavefield less, the 2-D `Elastic` 2 to 6 less
  (its generated class also drops the two memory fields the shipped one
  carries unused).
- The restated 3-D `Elastic` peaks within half a wavefield of the shipped one
  with `Full()` and uncompiled checkpointing. With strip-held CPML memory, the
  3-D default, it needs 4 wavefields more under boundary saving and 4 to 8 more
  under compiled checkpointing; with `cpml_strips=False`, 9 less and the same.

The generated step frees a sub-step's derivatives at its end, and computes a
sum of derivatives that several equations read once per sub-step (the 3-D
divergence), as the shipped steps do: written out in every equation, autograd
would keep a copy of the sum per equation, 7 % more memory in a `Full()` 3-D
run.

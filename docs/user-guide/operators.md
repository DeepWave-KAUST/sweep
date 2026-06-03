# Operators

When you write a new equation's `func()` you reach for SWEEP's
spatial-derivative operators — Laplacians, partial derivatives, staggered
finite differences. This page lays out the three operator families, when to
pick each, and how to call them.

You usually do **not** instantiate operators yourself — the base classes in
`sweep.equations.base` build them for you and expose them as instance
attributes. Read this page if you are about to write a new `func()` and want
to know what is already on `self`.

## Decision tree

| Your equation's structure | Path | What you call inside `func()` |
| --- | --- | --- |
| 2-D / 3-D second-order acoustic (`u_tt = vp² · Laplacian(u) + ...`) | Inherit `SecondOrderEquation` | `self.laplacian_2d / laplacian_3d` (isotropic) or `self.separable_d2_2d / separable_d2_3d` (anisotropic) |
| First-order velocity-stress (`∂v/∂t = ∂σ/∂x ...`) | Inherit `FirstOrderEquation` | `self.pd.x_forward / x_backward / y_* / z_*` |
| Rotated staggered-grid elastic TTI | Explicitly construct `RSGDerivative` | `self.rsg.lx_fwd / lx_bwd / lz_fwd / lz_bwd` |

The three classes are all importable from `sweep.operators`:

```python
from sweep.operators import LaplaceGradientOps, StaggeredDerivative, RSGDerivative
```

but the base classes wire them up automatically — you typically only `import`
them if you are building a custom operator stack outside an `Equation`
subclass.

## Path 1 — `SecondOrderEquation`

The `LaplaceGradientOps` bundle, mixed into `SecondOrderEquation`, binds five
backend-dispatched functions as instance attributes:

| Attribute | Returns | Used by |
| --- | --- | --- |
| `self.separable_d2_2d(u, kernels, hz, hx)` | `(d²u/∂z², d²u/∂x²)` tuple | anisotropic equations needing per-axis components |
| `self.separable_d2_3d(u, kernels, hz, hy, hx)` | `(d²u/∂z², d²u/∂y², d²u/∂x²)` tuple | 3-D anisotropic equations |
| `self.laplacian_2d(u, kernels, hz, hx)` | scalar `∇²u` (= sum of components) | isotropic equations |
| `self.laplacian_3d(u, kernels, hz, hy, hx)` | scalar `∇²u` (= sum of components) | 3-D isotropic equations |
| `self.gradient(u, h, axis, kernels=None)` | first-order `∂u/∂xᵢ` along one axis | CPML wiring, RTM imaging conditions |

The Laplacian helpers consume `self.laplace_kernels`, populated by a single
init call:

```python
class MyScalar(SecondOrderEquation):
    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_separable_laplace()   # ← ndim auto-dispatched
```

`init_separable_laplace()` is zero-argument: it picks 2-D vs 3-D from
`self.ndim` and builds the kernel stack on `self.backend / self.device`.

### Recipe — isotropic 2-D acoustic

```python
def func(self, wavefields, models, dt, h, b, **kwargs):
    u_now, u_pre = wavefields
    (vp,) = models
    hz, hx = self._spacings_2d(h)
    lap = self.laplacian_2d(u_now, self.laplace_kernels, hz, hx)
    u_next = 2 * u_now - u_pre + (vp * dt) ** 2 * lap
    return u_next, u_now
```

### Recipe — anisotropic (per-axis components)

`AcousticVTI` style: the wave equation couples `d²u/∂z²` and `d²u/∂x²`
with different coefficients, so you need the components separately:

```python
def func(self, wavefields, models, dt, h, b, **kwargs):
    u_now, u_pre = wavefields
    vp, epsilon, delta = models
    hz, hx = self._spacings_2d(h)
    d2z, d2x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
    vp2 = vp * vp
    u_next = 2 * u_now - u_pre + dt * dt * (
        vp2 * (1 + 2 * epsilon) * d2x          # horizontal stiffer
        + vp2 * (1 + 2 * delta) * d2z          # vertical
    )
    return u_next, u_now
```

### Recipe — gradient (CPML auxiliary fields)

```python
dudz = self.gradient(u_now, h, axis=-2)
dudx = self.gradient(u_now, h, axis=-1)
```

`gradient()` has two modes. The default (`kernels=None`) falls back to
`torch.gradient`. Passing the optional `kernels=` dict routes through a
fixed-stencil convolution that is ~5–10× faster on CUDA:

```python
# In __init__:
use_fast_grad_kernels = backend == "torch" and "cuda" in str(device)
super().__init__(spatial_order, device, backend, dim=dim,
                 other_kernels=use_fast_grad_kernels)
super().init_separable_laplace()
self.grad_kernels = None
if use_fast_grad_kernels:
    self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

# In func() / step_cpml:
dudz = self.gradient(u_now, h, axis=-2, kernels=self.grad_kernels)
```

`Acoustic.step_cpml` is the canonical reference.

## Path 2 — `FirstOrderEquation`

`FirstOrderEquation.__init__` constructs `self.pd = StaggeredDerivative(...)`
for you. It exposes 6 (2-D) or 12 (3-D) methods on a Mac-grid stencil:

| Method | Returns | Stencil |
| --- | --- | --- |
| `self.pd.x_forward(u, h=None)` | `∂u/∂x` evaluated at face / center | forward staggered |
| `self.pd.x_backward(u, h=None)` | `∂u/∂x` evaluated at the opposite half-cell | backward staggered |
| `self.pd.z_forward(u, h=None)` | same, z-axis | — |
| `self.pd.z_backward(u, h=None)` | — | — |
| `self.pd.y_forward / y_backward` | only present when `ndim=3` | — |

`h=None` makes the call divide by `self._spacing[axis]`, which you set once:

```python
self.pd.set_spacing((dz, dx))     # or (dz, dy, dx) in 3-D
```

`set_spacing` accepts a scalar (isotropic) or a tuple matching `ndim`.

### Recipe — first-order velocity-stress (acoustic)

`Acoustic1st` writes one half-step of `(v, p)` in eager mode like this:

```python
def func(self, wavefields, models, dt, h, b, **kwargs):
    vx, vz, p = wavefields
    vp, rho = models
    # Pressure update from velocity divergence:
    dvx_dx = self.pd.x_backward(vx)
    dvz_dz = self.pd.z_backward(vz)
    p_next = p - dt * (vp * vp * rho) * (dvx_dx + dvz_dz)
    # Velocity update from pressure gradient:
    dp_dx = self.pd.x_forward(p_next)
    dp_dz = self.pd.z_forward(p_next)
    vx_next = vx - dt / rho * dp_dx
    vz_next = vz - dt / rho * dp_dz
    return vx_next, vz_next, p_next
```

Forward and backward sweeps are paired by the **staggering convention** —
applying the same direction to both `v` and `p` would offset the result by
half a cell.

## Path 3 — `RSGDerivative`

For elastic TTI with anisotropy axes rotated relative to the grid axes, the
rotated staggered grid keeps the discretisation stable. `RSGDerivative` lives
in `sweep.operators` and is **not** built automatically — you construct it
yourself, mirroring `ElasticTTI`:

```python
from sweep.operators import RSGDerivative

class MyTTI(FirstOrderEquation):
    def __init__(self, spatial_order=8, device="cpu", backend="torch"):
        super().__init__(spatial_order, device, backend, ndim=2)
        self.rsg = RSGDerivative(spatial_order, device, backend, ndim=2)
        self.rsg.to_backend(to_backend)
        self.rsg.set_spacing((dz, dx))   # call once after building
```

The operator exposes four `(forward, backward) × (x, z)` methods:

| Method | Returns |
| --- | --- |
| `self.rsg.lx_fwd(u)` | rotated-staggered `∂u/∂x`, forward shift |
| `self.rsg.lx_bwd(u)` | rotated-staggered `∂u/∂x`, backward shift |
| `self.rsg.lz_fwd(u)` | rotated-staggered `∂u/∂z`, forward shift |
| `self.rsg.lz_bwd(u)` | rotated-staggered `∂u/∂z`, backward shift |

Only 2-D is supported today; `RSGDerivative(ndim=3)` raises.
`ElasticTTI.func` is the canonical reference.

## Spacing helpers on the base class

Every equation `func()` receives an `h` argument that may be a scalar
(isotropic spacing) or a length-`ndim` array. The base classes provide two
unpacking helpers so you do not have to branch:

```python
hz, hx = self._spacings_2d(h)           # works for scalar or (hz, hx)
hz, hy, hx = self._spacings_3d(h)       # works for scalar or (hz, hy, hx)
```

These read `self.ndim`, so the same code runs on the 2-D and 3-D variants of
an equation.

## Backend dispatch

`LaplaceGradientOps` and `StaggeredDerivative` import the right backend module
(`sweep.operators.torch` or `sweep.operators.jax`) at construction time, so
calling `self.laplacian_2d(...)` from a `backend="jax"` equation transparently
runs the JAX-jitted version. You only need to think about backends when you
explicitly write low-level helpers — every equation in this repo just lets
the base class pick.

## See also

- [Operators API reference](../api/operators/index.md) — full signatures for
  `LaplaceGradientOps`, `StaggeredDerivative`, `RSGDerivative`.
- [Extending: Adding a New Equation](extending.md) — full lifecycle of a new
  equation (`FIELD_SPECS`, registration, `_C()` hook, CUDA path).
- [Add a new equation notebook](../notebooks/18_extending_add_new_equation.ipynb)
  — runnable walkthrough of a 2-D scalar equation using
  `self.laplacian_2d` end-to-end.
- [`Acoustic` source](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/acoustic.py)
  — canonical example of `LaplaceGradientOps` + `gradient` + CPML wiring.
- [`Acoustic1st` source](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/acoustic1st.py)
  — canonical example of `StaggeredDerivative` (`self.pd`).
- [`ElasticTTI` source](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/elastic_tti.py)
  — canonical example of `RSGDerivative` (`self.rsg`).

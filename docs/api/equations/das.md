# DAS family

Distributed Acoustic Sensing equations — strain / strain-rate forward operators
for fiber-optic cable measurements. There are two formulations, each in 2D and
3D: Zhao (`DASZhao` / `DASZhao3D`, stress / normal-strain-rate) and Mu
(`DASMu` / `DASMu3D`, velocity-stress plus integrated strain). The `DAS`
facade runs either one (`method="zhao"` or `method="mu"`; 2D or 3D from
`shape`). `DASElastic` / `DASElastic3D` are aliases of `DASZhao` /
`DASZhao3D`, and `DASModeler` is the former name of `DAS`.

Both formulations solve the isotropic elastic wave equation with the Lamé
parameters $\lambda = \rho (v_p^2 - 2 v_s^2)$ and $\mu = \rho v_s^2$ computed
from the models `vp`, `vs` and `rho`. They differ in the dynamic variables, in
the quantity they record, and in the boundary conditions they support.

## Zhao formulation: stress and normal strain rate

`DASZhao` and `DASZhao3D` advance the normal strain rates
$\dot\varepsilon_{ii} = \partial v_i / \partial x_i$ (fields `exx_t`, `eyy_t`,
`ezz_t`), the normal stresses $\sigma_{ii}$ (`sxx`, `syy`, `szz`) and the
auxiliary variables $\tau_{ii}$ (`txx`, `tyy`, `tzz`), following Eq. 9 of
Zhao et al. (2022) as cited in the source. For each $i \in \{x, y, z\}$
($\{x, z\}$ in 2-D), with no summation over $i$,

$$
\begin{aligned}
\rho \frac{\partial \dot\varepsilon_{ii}}{\partial t}
 &= \frac{\partial^2 \sigma_{ii}}{\partial x_i^2}
  + \sum_{j \neq i} \left( \frac{\partial^2 \tau_{ii}}{\partial x_j^2}
                         + \frac{\partial^2 \tau_{jj}}{\partial x_i^2} \right),
\\
\frac{\partial \sigma_{ii}}{\partial t}
 &= \lambda \sum_{k} \dot\varepsilon_{kk} + 2\mu\, \dot\varepsilon_{ii},
\\
\frac{\partial \tau_{ii}}{\partial t}
 &= \mu\, \dot\varepsilon_{ii}.
\end{aligned}
$$

In 2-D the two strain-rate equations share the same shear term
$\partial_z^2 \tau_{xx} + \partial_x^2 \tau_{zz}$. The system is the
derivative $\partial_i$ of the velocity–stress momentum equation for $v_i$,
with each shear term $\partial_i \partial_j \sigma_{ij}$ replaced by
$\partial_j^2 \tau_{ii} + \partial_i^2 \tau_{jj}$. This is exact when $\rho$
and $\mu$ are constant; in a heterogeneous model the spatial derivatives of
$1/\rho$ and $\mu$ are neglected. The velocities and the shear stresses are
not computed.

All variables share the integer grid nodes. Each second derivative is a
forward staggered difference followed by a backward one, both of order
`spatial_order` (default 4) and each with its own C-PML memory variable. Time
stepping is leapfrog: the strain rates first, then $\sigma_{ii}$ and
$\tau_{ii}$ from the new strain rates. The default source fields are the
normal stresses (an explosive source); the default receiver fields are the
strain rates `exx_t`, `ezz_t` (and `eyy_t` in 3-D).

**Recorded quantity.** The fields `exx_t`, `eyy_t`, `ezz_t` record the normal
strain rates at a point. The fields `das35_t`, `das54x_t`, `das54y_t` (3-D
only) and `das54z_t` are evaluated at every step as the axial strain rate of a
helically wound fibre, with the shear strain rates neglected:

$$
\dot\varepsilon_{35} = \dot\varepsilon_{xx} + \dot\varepsilon_{yy}
 + \dot\varepsilon_{zz},
\qquad
\dot\varepsilon_{54,x} = 4\dot\varepsilon_{xx} + \dot\varepsilon_{yy}
 + \dot\varepsilon_{zz},
$$

for winding angles of 35.3° and 54.7° (core along $x$; the $y$ and $z$ cores
put the weight 4 on their own component), with $\dot\varepsilon_{yy} = 0$ in
2-D. They are point values: the `DAS` facade does not apply the gauge-length
average to the Zhao records.

## Mu formulation: velocity–stress with integrated strain

`DASMu` and `DASMu3D` run the velocity–stress system of
[Elastic](elastic.md) and [Elastic3D](elastic3d.md) unchanged and integrate
the strain from the velocity derivatives used in the stress update
(Mu and Hung, 2022, as cited in the source):

$$
\frac{\partial \varepsilon_{ii}}{\partial t}
 = \frac{\partial v_i}{\partial x_i},
\qquad
\frac{\partial \varepsilon_{ij}}{\partial t}
 = \frac{1}{2} \left( \frac{\partial v_i}{\partial x_j}
                    + \frac{\partial v_j}{\partial x_i} \right),
\quad i \neq j.
$$

The strains `exx`, `ezz`, `exz` (2-D) or `exx`, `eyy`, `ezz`, `exy`, `exz`,
`eyz` (3-D) sit on the nodes of the matching stress component and are updated
with $\varepsilon^{n+1} = \varepsilon^{n} + \Delta t\, \partial v^{n+1/2}$,
using the C-PML-corrected derivatives inside the absorbing layer.

**Recorded quantity.** The equation records the strain itself; its default
receiver fields are all strain components. The `DAS` facade with
`method="mu"` also returns strain rates and helical-fibre responses, computed
from the recorded strain after the solve. A strain rate such as `exx_t` is the
backward difference

$$
\dot\varepsilon^{\,n} = \frac{\varepsilon^{n} - \varepsilon^{n-1}}{\Delta t},
\qquad
\dot\varepsilon^{\,0} = \frac{\varepsilon^{0}}{\Delta t},
$$

which needs `dt` in the propagator arguments. The helical channels `das35_t`
and `das54x_t`, `das54y_t`, `das54z_t` combine the normal strain rates with
the weights of the Zhao formulation. They are then averaged along the
receiver axis (`gauge_axis`, default the last) with a centred moving average
over `gauge_cells` cells, or `gauge_length`/`dh` rounded, with the end values
repeated at the edges; without either argument no average is applied.

## Boundary conditions

All four equations use `pml_type="cpmls"` by default, the staggered-grid C-PML
with one memory variable per difference; see
[Absorbing boundaries](index.md#absorbing-boundaries).

`DASMu` and `DASMu3D` support a free surface on the top face only
(`free_surface=True`), with the image method and surface $\sigma_{xx}$
($\sigma_{xx}$, $\sigma_{yy}$ in 3-D) correction of [Elastic](elastic.md) and
[Elastic3D](elastic3d.md). `DASZhao` and `DASZhao3D` implement no
free-surface condition, although `free_surface=True` is accepted.

## Facade

::: sweep.equations.DAS

## 2-D

::: sweep.equations.DASZhao

::: sweep.equations.DASMu

## 3-D

::: sweep.equations.DASZhao3D

::: sweep.equations.DASMu3D

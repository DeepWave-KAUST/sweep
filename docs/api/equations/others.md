# Other equations

Exported equation classes without a page of their own; see the
[equation index](index.md) for the families. Compiled `impl='c'` kernels exist
for `ElasticVRR`, `AcousticVTI1st`, `AcousticVTI1st3D`, `AcousticVTI`,
`AcousticTTI` and the Born pair `AcousticVTIBorn` / `AcousticTTIBorn`; every
other class on this page runs on the eager backend only.

## Curvilinear-grid topography

`AcousticCurvilinear` and `ElasticCurvilinear` are **eager-only**: they have no
CUDA kernel, so `impl='auto'` runs them eager and an explicit `impl='c'` falls
back to eager with a `UserWarning`.

### Coordinate mapping

Both classes map the physical domain $(x, z)$, whose top boundary is the
surface $z = h(x)$, onto a rectangular computational domain $(\xi, \eta)$ by
the vertical linear stretch

$$
x = \xi,
\qquad
z = h(\xi) + D(\xi)\,\eta,
\qquad
D(\xi) = z_{\mathrm{b}} - h(\xi),
\qquad
0 \le \eta \le 1,
$$

where $h$ is the propagator argument `topography` (row indices) times the grid
spacing, smoothed with a Gaussian kernel when it is integer-valued, and
$z_{\mathrm{b}}$ is the depth of the last physical row. The computational top
row $\eta = 0$ therefore follows the topography, and a computational row
$\Delta\eta = 1/(n_z - 1)$ spans $D(\xi)/(n_z - 1)$ metres. The inverse
metrics are

$$
\frac{\partial \eta}{\partial x} = \alpha(\xi, \eta)
 = -\frac{h'(\xi)\,(1 - \eta)}{D(\xi)},
\qquad
\frac{\partial \eta}{\partial z} = \beta(\xi) = \frac{1}{D(\xi)},
$$

so that $\partial_x = \partial_\xi + \alpha\,\partial_\eta$ and
$\partial_z = \beta\,\partial_\eta$; the Jacobian is $J = D$. The metric
coefficients are continued into the absorbing layers by replicating the
nearest physical value. Boundary-conforming grids of this kind are used by
Tessmer and Kosloff (1994) and Hestholm and Ruud (1998), the references cited
in `sweep.utils.curvilinear`.

### `AcousticCurvilinear`

The class solves the constant-density equation of `Acoustic`,
$\partial_t^2 p = v_p^2 \nabla^2 p$, with $p$ the wavefield `h1` and $v_p$ the
model `vp`. Under the mapping the Laplacian becomes

$$
\nabla^2 p = p_{\xi\xi} + 2\alpha\, p_{\xi\eta}
 + \left( \alpha^2 + \beta^2 \right) p_{\eta\eta}
 + \left( \alpha_\xi + \alpha\, \alpha_\eta \right) p_\eta ,
$$

where $\alpha_\xi$ and $\alpha_\eta$ are precomputed by central differences.
The second derivatives use the central stencils of order `spatial_order`
(default 4) on the computational grid, and the mixed derivative is formed as
$\partial_\xi (\partial_\eta p)$. The time update is that of `Acoustic`. The
operator is split into a $\xi$ part, $p_{\xi\xi} + \alpha p_{\xi\eta}$, and an
$\eta$ part holding the remaining terms; these replace $\partial_{xx} p$ and
$\partial_{zz} p$ in the CPML of `Acoustic` (default `pml_type="cpmlr"`; see
[Absorbing boundaries](index.md#absorbing-boundaries)). The $z$ derivatives
in the CPML terms are taken in physical depth, $\partial_Z = \beta\,\partial_\eta$,
since one computational row is $D\,\Delta\eta$ metres rather than the grid
spacing. With a free surface the pressure-release condition of `Acoustic` is
applied at the computational top row, which coincides with the topography.

### `ElasticCurvilinear`

The class solves the isotropic velocity-stress system in the conservative
form of Bartana, Xu and Rickett, on a collocated grid:

$$
\rho J \frac{\partial v_i}{\partial t}
 = \frac{\partial}{\partial \xi} \left( J \sigma_{ix} \right)
 + \frac{\partial}{\partial \eta}
   \left( J\alpha\, \sigma_{ix} + \sigma_{iz} \right),
\qquad
\frac{\partial \sigma_{ij}}{\partial t}
 = \lambda\, \delta_{ij}\, \partial_k v_k
 + \mu \left( \partial_i v_j + \partial_j v_i \right),
$$

where $i, j \in \{x, z\}$, the physical derivatives in the constitutive law
are formed with $\partial_x = \partial_\xi + \alpha\partial_\eta$ and
$\partial_z = \beta\partial_\eta$, and $J\beta = 1$ has been used in the
$\eta$-flux. The wavefields are `vx`, `vz`, `sxx`, `szz`, `sxz`; the models
`vp`, `vs`, `rho` give $\lambda = \rho (v_p^2 - 2 v_s^2)$ and
$\mu = \rho v_s^2$. All five fields share the same grid points. Spatial
derivatives use asymmetric first-derivative operators of half-width
$L$ = `spatial_order`/2 (default `spatial_order=16`) on the velocities and
their negative transposes on the stress fluxes, which makes the semi-discrete
operator self-adjoint. The product $J\alpha$ is evaluated as
$(1 - \eta)\,\partial_\xi J$ with the scheme's own operator, so that a uniform
stress produces no discrete force.

The top row of the computational grid is the free surface. Its stress rate
is not taken from the constitutive law but from the traction-free condition
in the frame of the surface tangent
$\mathbf{t} = (1, h') / \sqrt{1 + h'^2}$:

$$
\frac{\partial \boldsymbol{\sigma}}{\partial t}
 = \frac{4\mu (\lambda + \mu)}{\lambda + 2\mu}\,
   \dot{\varepsilon}_{tt}\, \mathbf{t}\mathbf{t}^{\mathsf{T}},
$$

where $\dot{\varepsilon}_{tt}$ is the tangential strain rate, computed from
derivatives along the surface only and corrected for surface curvature. The
traction on the surface row therefore stays zero, and all fields above it are
held at zero. By default (`surface_closure="mirror"`) the $\eta$-derivative of
the stress flux at the surface is closed by an odd reflection about the
surface row, and a Kreiss-Oliger filter (`surface_damping`) is applied in a
band below the surface to remove a high-wavenumber stress mode that the
closure supports. Absorption uses the default `pml_type="cpmls"`, with CPML
memory variables on the $\xi$- and $\eta$-derivatives (see
[Absorbing boundaries](index.md#absorbing-boundaries)).

::: sweep.equations.AcousticCurvilinear

::: sweep.equations.ElasticCurvilinear

## Vector reflectivity (VRR)

Both classes carry the density through the vector reflectivity of an
impedance $Z$,

$$
\mathbf{R} = \frac{1}{2Z} \nabla Z = \frac{1}{2} \nabla \ln Z ,
$$

instead of the density itself. The reflectivity components are model inputs;
for consistent models they are obtained from $(v_p, v_s, \rho)$ with
`sweep.equations.elastic_vrr.compute_vector_reflectivity`, which evaluates
this definition with $Z_P = \rho v_p$ and $Z_S = \rho v_s$ by central
differences.

### `AcousticVRR`

`AcousticVRR` solves the scalar second-order equation

$$
\frac{\partial^2 p}{\partial t^2}
 = v_p^2 \nabla^2 p
 + \left( v_p \nabla v_p - 2 v_p^2\, \mathbf{r} \right) \cdot \nabla p ,
$$

where $p$ is the wavefield `h1`, $v_p$ the model `vp` and
$\mathbf{r} = (r_x, r_z)$ the models `rx`, `rz`. With
$\mathbf{r} = \frac{1}{2}\nabla \ln(\rho v_p)$ the equation is identical to the
variable-density acoustic equation
$\partial_t^2 p = \rho v_p^2 \nabla \cdot (\rho^{-1} \nabla p)$, since
$-\nabla \ln \rho = -2\mathbf{r} + \nabla \ln v_p$; the code treats `rx` and
`rz` as independent models. Time and the Laplacian are discretised as in
`Acoustic`; $\nabla v_p$ and $\nabla p$ use central first-derivative stencils
of order `spatial_order` (default 4), and $\nabla v_p$ is recomputed at every
step. In the absorbing layer the Laplacian and the first derivatives of $p$
both receive the CPML corrections of the default `pml_type="cpmlr"` (see
[Absorbing boundaries](index.md#absorbing-boundaries)). With
`free_surface=True` the selected faces are pressure-release, $p = 0$, as in
`Acoustic`. The code cites Whitmore et al. (2020).

### `ElasticVRR`

`ElasticVRR` implements the first-order formulation of Soares and Sacchi
(2025). The state variables are the particle momentum $p_i = \rho v_i$ (`px`,
`pz`) and the Cartesian stress (`sxx`, `szz`, `sxz`), so that the equation of
motion contains no density:

$$
\frac{\partial p_i}{\partial t} = \frac{\partial \sigma_{ij}}{\partial x_j} .
$$

For each wave type $\alpha \in \{P, S\}$ the code forms

$$
\Gamma^{\alpha}_{ij}
 = V_\alpha \left( \partial_i V_\alpha \right) p_j
 - 2 V_\alpha^2 R_{\alpha,i}\, p_j
 + V_\alpha^2\, \partial_i p_j ,
$$

and updates the stresses as

$$
\frac{\partial \sigma_{xx}}{\partial t}
 = \Gamma^{P}_{xx} + \Gamma^{P}_{zz} - 2\Gamma^{S}_{zz},
\qquad
\frac{\partial \sigma_{zz}}{\partial t}
 = \Gamma^{P}_{xx} + \Gamma^{P}_{zz} - 2\Gamma^{S}_{xx},
\qquad
\frac{\partial \sigma_{xz}}{\partial t}
 = \Gamma^{S}_{xz} + \Gamma^{S}_{zx} .
$$

Here $V_P$ and $V_S$ are the models `vp` and `vs`, and
$\mathbf{R}_P = (R_{P,x}, R_{P,z})$ and $\mathbf{R}_S$ are the models
`Rp_x`, `Rp_z`, `Rs_x`, `Rs_z`, with $\mathbf{R}_\alpha = \frac{1}{2}\nabla
\ln(\rho V_\alpha)$. For consistent models the system is the isotropic
velocity-stress system with $\lambda = \rho(V_P^2 - 2V_S^2)$ and
$\mu = \rho V_S^2$. The fields sit on the staggered grid of `Elastic`, and the
momentum is updated before the stress. The gradients $\partial_i V_\alpha$ are
central differences of order `spatial_order` (order 4 above 8), computed once
per run, and model values are multiplied with the staggered momentum without
interpolation. Absorption uses the default `pml_type="cpmls"` (see
[Absorbing boundaries](index.md#absorbing-boundaries)). With
`free_surface=True` the top row is treated as in `Elastic`: $\sigma_{zz}$ and
$\sigma_{xz}$ are set to zero, and $\sigma_{xx}$ is advanced with
$4 V_S^2 (V_P^2 - V_S^2) / V_P^2\, \partial_x p_x$, which is
$4\mu(\lambda + \mu)/(\lambda + 2\mu)\, \partial_x v_x$ written in momentum
with $\partial_x \rho$ neglected.

::: sweep.equations.AcousticVRR

::: sweep.equations.ElasticVRR

## Anisotropic acoustic family

`AcousticAniso` is a factory: it returns an instance of one of the classes
below. The author-named aliases (`AcousticVTILiang`, `AcousticTTILiang`,
`AcousticVTIAlkhalifah`, `AcousticVTIDuveneck`, `AcousticVTIDuveneck3D`) and
the `AcousticVTIDefault` / `AcousticVTIDefault3D` routing names are these same
classes. `method="duveneck"` returns
`AcousticVTI1st` (`ndim=2`) or `AcousticVTI1st3D` (`ndim=3`, VTI only),
`method="liang"` returns `AcousticVTI` or `AcousticTTI` (2-D), and
`method="alkhalifah"` returns `AcousticTariq` (2-D, VTI only;
`symmetry="tti"` raises `NotImplementedError`).

::: sweep.equations.AcousticAniso

The pure-qP pair and its Born equations: the derivation, the three
`s_gradient` modes and the compiled backend are in
[Pure-qP VTI / TTI](../../user-guide/qp_born.md). `AcousticVTI` and
`AcousticTTI` solve the decoupled pure-qP equation obtained with the acoustic
approximation of Liang et al. (2023), in which the qSV phase velocity is set to
zero in every direction; the operator $S_n$ follows Xu and Zhou (2014), as
cited in the code.

::: sweep.equations.AcousticVTI

::: sweep.equations.AcousticTTI

::: sweep.equations.AcousticVTIBorn

::: sweep.equations.AcousticTTIBorn

### `AcousticTariq`

`AcousticTariq` solves the acoustic VTI equation of Alkhalifah (2000), written
as two second-order equations in the wavefield $p$ (`h1`) and an auxiliary
field $F$ (`f1`) with $p = \partial_t^2 F$:

$$
\frac{\partial^2 p}{\partial t^2}
 = (1 + 2\eta)\, v^2 \frac{\partial^2 p}{\partial x^2}
 + v_v^2 \frac{\partial^2 p}{\partial z^2}
 - 2\eta\, v^2 v_v^2 \frac{\partial^4 F}{\partial x^2 \partial z^2},
\qquad
\frac{\partial^2 F}{\partial t^2} = p .
$$

Here $v$ is the model `v`, the NMO velocity $v_{p0}\sqrt{1 + 2\delta}$;
$v_v$ is the model `vv`, the vertical P-wave velocity; and $\eta$ is the anellipticity `eta`. The horizontal
velocity is $v\sqrt{1 + 2\eta}$. Both fields are advanced with the
second-order central difference in time, $F$ with $p^n$ on the right-hand
side. The second derivatives use central stencils of order `spatial_order`
(default 4), and the mixed term applies the $x$ and $z$ stencils in sequence.
The class has no tilt-angle model and is VTI only.
The default `pml_type="cpmlr"` stretches the $\partial_{xx} p$ and
$\partial_{zz} p$ terms as in `Acoustic`; the mixed term receives no CPML
correction (see [Absorbing boundaries](index.md#absorbing-boundaries)). The
class has no free surface and rejects `free_surface=True`.

::: sweep.equations.AcousticTariq

### `AcousticVTI1st` and `AcousticVTI1st3D`

These classes solve the first-order acoustic VTI system of Duveneck et al.
(2008), which follows from the elastic VTI equations with the shear velocity
on the symmetry axis set to zero (Alkhalifah, 1998). In 2-D the state is the
particle velocity (`vx`, `vz`) and the two normal stresses
$\sigma_H = \sigma_{xx} = \sigma_{yy}$ (`sH`) and $\sigma_V = \sigma_{zz}$
(`sV`):

$$
\rho \frac{\partial v_x}{\partial t} = \frac{\partial \sigma_H}{\partial x},
\qquad
\rho \frac{\partial v_z}{\partial t} = \frac{\partial \sigma_V}{\partial z},
$$

$$
\frac{\partial \sigma_H}{\partial t}
 = c_{11} \frac{\partial v_x}{\partial x}
 + c_{13} \frac{\partial v_z}{\partial z},
\qquad
\frac{\partial \sigma_V}{\partial t}
 = c_{13} \frac{\partial v_x}{\partial x}
 + c_{33} \frac{\partial v_z}{\partial z},
$$

with

$$
c_{11} = \rho v_p^2 (1 + 2\varepsilon),
\qquad
c_{33} = \rho v_p^2,
\qquad
c_{13} = \rho v_p^2 \sqrt{1 + 2\delta},
$$

where $v_p$ is the vertical P-wave velocity `vp`, $\varepsilon$ and $\delta$
are the Thomsen (1986) parameters `epsilon` and `delta`, and $\rho$ is `rho`.
The square root in $c_{13}$ is kept, not linearised. In 3-D, $\sigma_H$ also
drives $\rho\,\partial_t v_y = \partial_y \sigma_H$, and $\partial_x v_x$ is
replaced by $\partial_x v_x + \partial_y v_y$ in both stress equations. The
fields sit on a standard staggered grid with first-derivative stencils of
order `spatial_order` (default 4), and the velocities are updated before the
stresses. The system is unstable where $\varepsilon < \delta$ (Grechka et al.,
2004); `prepare_models` warns in that case but does not stop the run.
Absorption uses the default `pml_type="cpmls"` (see
[Absorbing boundaries](index.md#absorbing-boundaries)). No free surface is
implemented, and `free_surface=True` raises `NotImplementedError`.

::: sweep.equations.AcousticVTI1st

::: sweep.equations.AcousticVTI1st3D

## References

- Alkhalifah, T., 1998, Acoustic approximations for processing in
  transversely isotropic media: Geophysics, 63(2), 623–631,
  [doi:10.1190/1.1444361](https://doi.org/10.1190/1.1444361).
- Alkhalifah, T., 2000, An acoustic wave equation for anisotropic media:
  Geophysics, 65(4), 1239–1250,
  [doi:10.1190/1.1444815](https://doi.org/10.1190/1.1444815).
- Bartana, A., J. Xu, and J. Rickett, Solving the elastodynamic equations in
  curvilinear coordinates using asymmetric operators (cited in
  `elastic_curvilinear.py` without publication details).
- Duveneck, E., P. Milcik, P. M. Bakker, and C. Perkins, 2008, Acoustic VTI
  wave equations and their application for anisotropic reverse-time
  migration: SEG Technical Program Expanded Abstracts 2008, 2186–2190,
  [doi:10.1190/1.3059320](https://doi.org/10.1190/1.3059320).
- Grechka, V., L. Zhang, and J. W. Rector III, 2004, Shear waves in acoustic
  anisotropic media: Geophysics, 69(2), 576–582,
  [doi:10.1190/1.1707077](https://doi.org/10.1190/1.1707077).
- Hestholm, S., and B. Ruud, 1998, 3-D finite-difference elastic wave
  modeling including surface topography: Geophysics, 63(2), 613–622,
  [doi:10.1190/1.1444360](https://doi.org/10.1190/1.1444360).
- Liang, K., D. Cao, S. Sun, and X. Yin, 2023, Decoupled wave equation and
  forward modeling of qP wave in VTI media with the new acoustic
  approximation: Geophysics, 88(1), WA335–WA344,
  [doi:10.1190/geo2022-0292.1](https://doi.org/10.1190/geo2022-0292.1).
- Soares, A. S. Q., and M. D. Sacchi, 2025, Vector-reflectivity elastic wave
  equation with first-order formulation: Fifth International Meeting for
  Applied Geoscience & Energy, Expanded Abstracts, 1942–1946,
  [doi:10.1190/image2025-4316471.1](https://doi.org/10.1190/image2025-4316471.1).
- Tessmer, E., and D. Kosloff, 1994, 3-D elastic modeling with surface
  topography by a Chebychev spectral method: Geophysics, 59(3), 464–473,
  [doi:10.1190/1.1443608](https://doi.org/10.1190/1.1443608).
- Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954–1966,
  [doi:10.1190/1.1442051](https://doi.org/10.1190/1.1442051).
- Whitmore, N. D., J. Ramos-Martinez, Y. Yang, and A. Valenciano, 2020, Full
  wavefield modeling with vector reflectivity: EAGE 2020 Annual Conference &
  Exhibition Online, Extended Abstracts, 1–5,
  [doi:10.3997/2214-4609.202010332](https://doi.org/10.3997/2214-4609.202010332).
- Xu, S., and H. Zhou, 2014, Accurate simulations of pure quasi-P-waves in
  complex anisotropic media: Geophysics, 79(6), T341–T348,
  [doi:10.1190/geo2014-0242.1](https://doi.org/10.1190/geo2014-0242.1).

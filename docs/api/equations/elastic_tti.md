# ElasticTTI

## Governing equations

`ElasticTTI` solves the first-order velocity-stress equations of a tilted
transversely isotropic (TTI) solid in two dimensions with three displacement
components: the medium and the wavefield vary in $x$ and $z$ ($z$ positive
downward) and are invariant in $y$ ($\partial_y = 0$). With the particle
velocities $v_x, v_y, v_z$ (`vx`, `vy`, `vz`), the stresses $\sigma_{xx}$,
$\sigma_{zz}$, $\sigma_{yz}$, $\sigma_{xz}$, $\sigma_{xy}$ (`sxx`, `szz`,
`syz`, `sxz`, `sxy`) and the density $\rho$ (`rho`),

$$
\rho \frac{\partial v_x}{\partial t}
 = \frac{\partial \sigma_{xx}}{\partial x}
 + \frac{\partial \sigma_{xz}}{\partial z},
\qquad
\rho \frac{\partial v_y}{\partial t}
 = \frac{\partial \sigma_{xy}}{\partial x}
 + \frac{\partial \sigma_{yz}}{\partial z},
\qquad
\rho \frac{\partial v_z}{\partial t}
 = \frac{\partial \sigma_{xz}}{\partial x}
 + \frac{\partial \sigma_{zz}}{\partial z},
$$

$$
\frac{\partial \sigma_I}{\partial t}
 = C_{I1}\, \dot e_1 + C_{I3}\, \dot e_3 + C_{I4}\, \dot e_4
 + C_{I5}\, \dot e_5 + C_{I6}\, \dot e_6,
\qquad I \in \{1, 3, 4, 5, 6\},
$$

in Voigt notation ($1 = xx$, $2 = yy$, $3 = zz$, $4 = yz$, $5 = xz$,
$6 = xy$), with the strain rates

$$
\dot e_1 = \frac{\partial v_x}{\partial x}, \quad
\dot e_3 = \frac{\partial v_z}{\partial z}, \quad
\dot e_4 = \frac{\partial v_y}{\partial z}, \quad
\dot e_5 = \frac{\partial v_x}{\partial z} + \frac{\partial v_z}{\partial x},
\quad
\dot e_6 = \frac{\partial v_y}{\partial x}.
$$

Because $\dot e_2 = \partial_y v_y = 0$ and $\sigma_{yy}$ does not enter the
momentum equations, $\sigma_{yy}$ is not carried and only the 15 stiffness
entries $C_{11}, C_{13}, \ldots, C_{66}$ that multiply
$\dot e_1, \dot e_3, \ldots, \dot e_6$ are used.

**Stiffness from the model parameters.** The VTI stiffness is built from the
vertical velocities $v_{p0}$ (`vp0`), $v_{s0}$ (`vs0`), $\rho$ and the Thomsen
parameters $\varepsilon$, $\delta$, $\gamma$ (`epsilon`, `delta`, `gamma`):

$$
C_{33}^0 = \rho v_{p0}^2, \quad
C_{44}^0 = C_{55}^0 = \rho v_{s0}^2, \quad
C_{11}^0 = C_{22}^0 = (1 + 2\varepsilon)\, C_{33}^0, \quad
C_{66}^0 = (1 + 2\gamma)\, C_{44}^0,
$$

$$
C_{12}^0 = C_{11}^0 - 2C_{66}^0, \qquad
C_{13}^0 = C_{23}^0 = \sqrt{\max\!\left[0,\;
 (C_{33}^0 - C_{44}^0)^2 + 2\delta\, C_{33}^0 (C_{33}^0 - C_{44}^0)\right]}
 - C_{44}^0 ,
$$

which is Thomsen's exact definition of $\delta$ solved for $C_{13}$. The
tensor is then rotated, $C_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C^0_{pqrs}$,
with

$$
R = \begin{pmatrix}
\cos\phi \cos\theta & -\sin\phi & \cos\phi \sin\theta \\
\sin\phi \cos\theta & \cos\phi & \sin\phi \sin\theta \\
-\sin\theta & 0 & \cos\theta
\end{pmatrix},
$$

so the symmetry axis points along
$(\sin\theta \cos\phi,\ \sin\theta \sin\phi,\ \cos\theta)$: the tilt $\theta$
(`theta`) is measured from the $z$ axis and the azimuth $\phi$ (`phi`) from
the $x$ axis toward $y$. The azimuth is used in 2-D: for $\phi = 0$ the
entries $C_{14}, C_{16}, C_{34}, C_{36}, C_{45}, C_{56}$ vanish and the P-SV
fields (`vx`, `vz`, `sxx`, `szz`, `sxz`) decouple from the SH fields (`vy`,
`syz`, `sxy`); for $\phi \neq 0$ the two systems are coupled. Where
$|\theta| < 10^{-7}$ the unrotated VTI entries are used directly, so the
gradient with respect to `theta` and `phi` is zero at those points.

## Discretisation

Space is discretised on a rotated staggered grid: all three velocities share
the grid nodes and all five stresses share the cell centres, so no quantity is
interpolated. The $x$ and $z$ derivatives are formed from the two diagonal
differences,

$$
\frac{\partial u}{\partial x} \approx \frac{1}{2\Delta x}
\sum_{m=1}^{M} c_m \left[ u_{++} - u_{--} + u_{+-} - u_{-+} \right],
\qquad
\frac{\partial u}{\partial z} \approx \frac{1}{2\Delta z}
\sum_{m=1}^{M} c_m \left[ u_{++} - u_{--} - u_{+-} + u_{-+} \right],
$$

where
$u_{\pm\pm} = u\big(x \pm (m - \tfrac12)\Delta x,\; z \pm (m - \tfrac12)\Delta z\big)$,
$c_m$ are the staggered-grid coefficients and $M$ = `spatial_order` / 2
(default `spatial_order=8`). Time stepping is leapfrog: the velocities are advanced from the stresses, then the stresses
from the new velocities. The rotated stencils leave the checkerboard mode
$(-1)^{i+j}$ undamped and unpropagated; with `checkerboard_smoothing=True`
(default) point sources and receivers are spread over a 3 × 3 binomial stencil
whose response to that mode is zero. There is no compiled `impl='c'` path.

## Boundary conditions

The absorbing boundary is a CPML (see
[Absorbing boundaries](index.md#absorbing-boundaries)); the class requires the
default `pml_type="cpmlr"` but uses only its $a$ and $b$ profiles, with one
memory variable per first derivative, $\psi \leftarrow a\psi + b\,\partial u$
and $\partial u \leftarrow \partial u + \psi$. A free surface is not
implemented: `free_surface=True` is refused at construction, because the
isotropic image method would not give a stress-free surface for an
anisotropic medium.

## References

- Saenger, E. H., N. Gold, and S. A. Shapiro, 2000, Modeling the propagation
  of elastic waves using a modified finite-difference grid: Wave Motion,
  31(1), 77–92,
  [doi:10.1016/s0165-2125(99)00023-2](https://doi.org/10.1016/s0165-2125(99)00023-2).
- Saenger, E. H., and T. Bohlen, 2004, Finite-difference modeling of
  viscoelastic and anisotropic wave propagation using the rotated staggered
  grid: Geophysics, 69(2), 583–591,
  [doi:10.1190/1.1707078](https://doi.org/10.1190/1.1707078).
- Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954–1966,
  [doi:10.1190/1.1442051](https://doi.org/10.1190/1.1442051).

::: sweep.equations.ElasticTTI

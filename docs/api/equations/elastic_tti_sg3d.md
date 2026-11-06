# ElasticTTISG3D

## Governing equations

`ElasticTTISG3D` solves the three-dimensional first-order velocity-stress
equations of a tilted transversely isotropic solid. With the particle
velocities $v_i$ (`vx`, `vy`, `vz`), the stresses $\sigma_{ij}$ (`sxx`, `syy`,
`szz`, `sxy`, `sxz`, `syz`) and the density $\rho$ (`rho`),

$$
\rho \frac{\partial v_i}{\partial t} = \sum_j \frac{\partial \sigma_{ij}}{\partial x_j},
\qquad
\frac{\partial \sigma_I}{\partial t} = \sum_{J=1}^{6} C_{IJ}\, \dot e_J,
$$

in Voigt notation ($1 = xx$, $2 = yy$, $3 = zz$, $4 = yz$, $5 = xz$,
$6 = xy$), with $\dot e_1 = \partial_x v_x$, $\dot e_2 = \partial_y v_y$,
$\dot e_3 = \partial_z v_z$ and the engineering shear strain rates
$\dot e_4 = \partial_z v_y + \partial_y v_z$,
$\dot e_5 = \partial_z v_x + \partial_x v_z$,
$\dot e_6 = \partial_y v_x + \partial_x v_y$. All 21 independent entries of
$C_{IJ}$ are used.

**Stiffness from the model parameters.** The VTI stiffness is built from
`vp0`, `vs0`, `rho` and the Thomsen parameters `epsilon`, `delta`, `gamma`,

$$
C_{33}^0 = \rho v_{p0}^2, \quad
C_{44}^0 = C_{55}^0 = \rho v_{s0}^2, \quad
C_{11}^0 = C_{22}^0 = (1 + 2\varepsilon)\, C_{33}^0, \quad
C_{66}^0 = (1 + 2\gamma)\, C_{44}^0, \quad
C_{12}^0 = C_{11}^0 - 2C_{66}^0,
$$

$$
C_{13}^0 = C_{23}^0 = \sqrt{\max\!\left[0,\;
 (C_{33}^0 - C_{44}^0)^2 + 2\delta\, C_{33}^0 (C_{33}^0 - C_{44}^0)\right]}
 - C_{44}^0 ,
$$

and rotated with the Bond matrix $\mathbf{M}(R)$ of the rotation

$$
R = \begin{pmatrix}
\cos\phi \cos\theta & -\sin\phi & \cos\phi \sin\theta \\
\sin\phi \cos\theta & \cos\phi & \sin\phi \sin\theta \\
-\sin\theta & 0 & \cos\theta
\end{pmatrix},
\qquad
\mathbf{C} = \mathbf{M}\, \mathbf{C}^0\, \mathbf{M}^{\mathsf T},
$$

which is the same as rotating the fourth-order tensor,
$C_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C^0_{pqrs}$. The symmetry axis points
along $(\sin\theta \cos\phi,\ \sin\theta \sin\phi,\ \cos\theta)$: the tilt
$\theta$ (`theta`) is measured from the $z$ axis ($z$ positive downward) and
the azimuth $\phi$ (`phi`) from the $x$ axis toward $y$. Where
$|\theta| < 10^{-7}$ the unrotated VTI entries are used directly, so the
gradient with respect to `theta` and `phi` is zero at those points. With
$\varepsilon = \delta = \gamma = \theta = \phi = 0$ the update is numerically
identical to the isotropic [`Elastic3D`](elastic3d.md).

## Discretisation

Derivatives are standard staggered-grid differences of order `spatial_order`
(default 4), on the Virieux grid shared with `Elastic3D`: relative to the
node $(i, j, k)$ of the normal stresses, `vx`, `vy`, `vz` sit half a cell
along $x$, $y$, $z$, and `sxy`, `sxz`, `syz` at the centres of the $xy$, $xz$,
$yz$ cell faces. Each strain rate is computed where its staggered derivative
falls ($\dot e_1, \dot e_2, \dot e_3$ with the normal stresses,
$\dot e_4, \dot e_5, \dot e_6$ with `syz`, `sxz`, `sxy`), and the stress update
uses all six at that location without interpolation. A coupling
$C_{IJ}\dot e_J$ whose strain rate lives at another node than $\sigma_I$ is
therefore applied with a half-cell offset along one or two axes. For an
untilted medium all non-zero couplings are co-located; the offsets enter only
through the tilt-induced entries. The density and stiffness arrays are read at
the array index of each velocity or stress, without averaging. Time stepping
is leapfrog (velocities, then stresses).

## Boundary conditions

The default `pml_type="cpmls"` applies a CPML to every first derivative, with
profiles evaluated on the integer and half-integer nodes (see
[Absorbing boundaries](index.md#absorbing-boundaries)). A free surface is not
implemented: `free_surface=True` is refused at construction, because the
isotropic image method would not give a stress-free surface for an
anisotropic medium.

## References

- Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954–1966,
  [doi:10.1190/1.1442051](https://doi.org/10.1190/1.1442051).
- Virieux, J., 1986, P-SV wave propagation in heterogeneous media:
  Velocity-stress finite-difference method: Geophysics, 51(4), 889–901,
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

::: sweep.equations.ElasticTTISG3D

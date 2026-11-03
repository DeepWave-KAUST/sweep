# ElasticTTISG

## Governing equations

`ElasticTTISG` solves the same equations as
[`ElasticTTI`](elastic_tti.md): the first-order velocity-stress system of a
tilted transversely isotropic solid, invariant in $y$, with the velocities
`vx`, `vy`, `vz`, the stresses `sxx`, `szz`, `syz`, `sxz`, `sxy` and the
density `rho`,

$$
\rho\, \partial_t v_x = \partial_x \sigma_{xx} + \partial_z \sigma_{xz},
\qquad
\rho\, \partial_t v_y = \partial_x \sigma_{xy} + \partial_z \sigma_{yz},
\qquad
\rho\, \partial_t v_z = \partial_x \sigma_{xz} + \partial_z \sigma_{zz},
$$

$$
\partial_t \sigma_I = C_{I1}\, \dot e_1 + C_{I3}\, \dot e_3 + C_{I4}\, \dot e_4
 + C_{I5}\, \dot e_5 + C_{I6}\, \dot e_6,
\qquad I \in \{1, 3, 4, 5, 6\},
$$

with $\dot e_1 = \partial_x v_x$, $\dot e_3 = \partial_z v_z$,
$\dot e_4 = \partial_z v_y$, $\dot e_5 = \partial_z v_x + \partial_x v_z$ and
$\dot e_6 = \partial_x v_y$. The eight models `vp0`, `vs0`, `rho`, `epsilon`,
`delta`, `gamma`, `theta`, `phi` are mapped to the 15 stiffness entries
exactly as in `ElasticTTI` (Thomsen VTI stiffness, then the rotation by tilt
$\theta$ from $z$ and azimuth $\phi$ from $x$ toward $y$; `phi` couples the
P-SV and SH fields in 2-D).

## Discretisation

Derivatives are standard staggered-grid differences of order `spatial_order`
(default 8) along $x$ and $z$. Relative to the node $(i, j)$ of `sxx` and
`szz`, the fields sit at

| Field | Position |
| --- | --- |
| `sxx`, `szz`, `vy` | $(i, j)$ |
| `vx`, `sxy` | $(i + \tfrac12, j)$ |
| `vz`, `syz` | $(i, j + \tfrac12)$ |
| `sxz` | $(i + \tfrac12, j + \tfrac12)$ |

Each strain rate is computed where its staggered derivative falls: $\dot e_1$
and $\dot e_3$ at $(i, j)$, $\dot e_6$ at $(i + \tfrac12, j)$, $\dot e_4$ at
$(i, j + \tfrac12)$ and $\dot e_5$ at $(i + \tfrac12, j + \tfrac12)$. The
stress update uses every strain rate at that location without interpolation,
so a coupling $C_{IJ}\dot e_J$ whose strain rate lives at another node than
$\sigma_I$ is applied with a half-cell (or diagonal half-cell) offset. For an
untilted medium ($\theta = 0$) all non-zero couplings are co-located and the
scheme is the standard staggered grid; the offsets enter only through the
tilt-induced entries. The density and stiffness arrays are likewise read at
the array index of each velocity or stress, without averaging. Time stepping
is leapfrog (velocities, then stresses). Point sources and receivers are not
smoothed, since the axis-aligned stencils have no checkerboard null space.

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

::: sweep.equations.ElasticTTISG

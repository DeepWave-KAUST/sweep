# ElasticTTI2nd

## Governing equations

`ElasticTTI2nd` solves the second-order displacement equations of a tilted
transversely isotropic solid in the $x$-$z$ plane (P-SV only), following
Oh et al. (2020, eqs 11–12). With the displacements $u_x$, $u_z$ (`ux`, `uz`)
and the density $\rho$ (`rho`),

$$
\rho \frac{\partial^2 u_x}{\partial t^2}
 = \frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xz}}{\partial z},
\qquad
\rho \frac{\partial^2 u_z}{\partial t^2}
 = \frac{\partial \sigma_{xz}}{\partial x} + \frac{\partial \sigma_{zz}}{\partial z},
$$

$$
\begin{pmatrix} \sigma_{xx} \\ \sigma_{zz} \\ \sigma_{xz} \end{pmatrix}
=
\begin{pmatrix}
\tilde C_{11} & \tilde C_{13} & \tilde C_{15} \\
\tilde C_{13} & \tilde C_{33} & \tilde C_{35} \\
\tilde C_{15} & \tilde C_{35} & \tilde C_{55}
\end{pmatrix}
\begin{pmatrix}
\partial_x u_x \\ \partial_z u_z \\ \partial_z u_x + \partial_x u_z
\end{pmatrix}.
$$

The stresses are not carried as wavefields; they are formed inside each step.

**Stiffness from the model parameters.** The VTI constants follow the
parametrization of Oh et al. (2020, eq. 19), with the horizontal P velocity
$v_h$ (`vh`), the vertical S velocity $v_s$ (`vs`), $\rho$, Thomsen's
$\varepsilon$ (`epsilon`) and the anellipticity
$\eta = (\varepsilon - \delta)/(1 + 2\delta)$ (`eta`):

$$
C_{11} = \rho v_h^2, \quad
C_{33} = \frac{\rho v_h^2}{1 + 2\varepsilon}, \quad
C_{55} = \rho v_s^2, \quad
C_{13} = \rho \left[ \sqrt{\max\!\left(0,\;
 \Big(\tfrac{v_h^2}{1 + 2\varepsilon} - v_s^2\Big)
 \Big(\tfrac{v_h^2}{1 + 2\eta} - v_s^2\Big)\right)} - v_s^2 \right].
$$

Here $v_h^2/(1+2\varepsilon)$ is the vertical P velocity squared and
$v_h^2/(1+2\eta)$ the NMO velocity squared, so $C_{13}$ agrees with Thomsen's
$\delta$. The tilt $\theta$ (`theta`) rotates the symmetry axis from $z$
toward $x$, to $(\sin\theta, \cos\theta)$, through the 2-D Bond rotation
(Oh et al. 2020, eq. 10). With $c = \cos\theta$, $s = \sin\theta$,

$$
\begin{aligned}
\tilde C_{11} &= C_{11} c^4 + C_{33} s^4 + (2C_{13} + 4C_{55})\, s^2 c^2, \\
\tilde C_{33} &= C_{11} s^4 + C_{33} c^4 + (2C_{13} + 4C_{55})\, s^2 c^2, \\
\tilde C_{13} &= (C_{11} + C_{33} - 4C_{55})\, s^2 c^2 + C_{13} (s^4 + c^4), \\
\tilde C_{55} &= (C_{11} + C_{33} - 2C_{13})\, s^2 c^2 + C_{55} (c^2 - s^2)^2, \\
\tilde C_{15} &= A\, c^3 s - B\, c\, s^3, \qquad
\tilde C_{35} = A\, s^3 c - B\, s\, c^3,
\end{aligned}
$$

with $A = C_{13} - C_{11} + 2C_{55}$ and $B = C_{13} - C_{33} + 2C_{55}$.
These are the $\phi = 0$ entries of the rotation used by
[`ElasticTTI`](elastic_tti.md). Unlike there, $\theta = 0$ is not
special-cased, so the gradient with respect to `theta` is also defined in
untilted regions. There is no $\gamma$ or $\phi$: the SH system is not
modelled.

## Discretisation

Both displacement components live on the same grid nodes. Each term
$\partial_a (\tilde C\, \partial_b u)$ is evaluated as
$D_a^{-} \big[\tilde C\, D_b^{+} u\big]$, a forward staggered difference
$D^{+}$ followed by a backward one $D^{-}$, both of order `spatial_order`
(default 4) and without interpolation. The discrete stiffness operator is then
symmetric, $K = -D^{\mathsf T} \tilde C D$, and has no checkerboard null space;
the mixed terms ($a \neq b$) carry a diagonal half-cell offset. Time stepping
is the second-order central difference,

$$
u^{n+1} = 2u^{n} - u^{n-1}
 + \frac{\Delta t^2}{\rho}\, \big( D^{-} \cdot \sigma(u^{n}) \big).
$$

Oh et al. (2020) solve the same equations with a frequency-domain finite
element method; the time-domain scheme here is not theirs.

## Boundary conditions

The default `pml_type="cpmls"` applies a CPML to every first derivative: the
inner forward differences use the half-node profiles, the outer backward
differences the integer-node profiles (see
[Absorbing boundaries](index.md#absorbing-boundaries)). For this second-order
displacement system the plain CPML grows at late times, which Li and
Bou Matar (2010) report for the same CPML and which follows from the
stability condition of Bécache et al. (2003). The profiles are therefore made
multiaxial (Meza-Fajardo and Papageorgiou, 2008, in the form of Li and
Bou Matar, 2010, eq. 23):

$$
d_x(x, z) = d_x(x) + r\, d_z(z),
\qquad
d_z(x, z) = d_z(z) + r\, d_x(x),
$$

where $d$ is the damping profile, $r$ is the constructor argument
`mpml_ratio` (default 0.05) and the frequency-shift $\alpha$ of each axis is
kept. `mpml_ratio=0` restores the plain CPML. A free surface is not
implemented: `free_surface=True` is refused at construction, because the
isotropic image method would not give a stress-free surface for an
anisotropic medium.

## References

- Bécache, E., S. Fauqueux, and P. Joly, 2003, Stability of perfectly matched
  layers, group velocities and anisotropic waves: Journal of Computational
  Physics, 188(2), 399–433,
  [doi:10.1016/s0021-9991(03)00184-0](https://doi.org/10.1016/s0021-9991(03)00184-0).
- Li, Y. F., and O. Bou Matar, 2010, Convolutional perfectly matched layer
  for elastic second-order wave equation: The Journal of the Acoustical
  Society of America, 127(3), 1318–1327,
  [doi:10.1121/1.3290999](https://doi.org/10.1121/1.3290999).
- Meza-Fajardo, K. C., and A. S. Papageorgiou, 2008, A nonconvolutional,
  split-field, perfectly matched layer for wave propagation in isotropic and
  anisotropic elastic media: Stability analysis: Bulletin of the
  Seismological Society of America, 98(4), 1811–1836,
  [doi:10.1785/0120070223](https://doi.org/10.1785/0120070223).
- Oh, J.-W., Y. Shin, T. Alkhalifah, and D.-J. Min, 2020, Multistage elastic
  full-waveform inversion for tilted transverse isotropic media: Geophysical
  Journal International, 223(1), 57–76,
  [doi:10.1093/gji/ggaa295](https://doi.org/10.1093/gji/ggaa295).

::: sweep.equations.ElasticTTI2nd

# Biot

## Governing equations

`Biot` solves the two-dimensional low-frequency Biot equations of a
fluid-saturated porous solid (Biot, 1962) in velocity-stress form. The
unknowns are the solid particle velocity $\mathbf{v}$ (`vx`, `vz`), the Darcy
flux $\mathbf{q} = \phi_0 (\mathbf{v}_f - \mathbf{v})$ (`qx`, `qz`), i.e. the
fluid velocity relative to the solid per unit bulk area with porosity
$\phi_0$, the total stress $\boldsymbol{\sigma}$ (`sxx`, `szz`, `sxz`) and the
pore pressure $p$ (`p`):

$$
\begin{aligned}
\rho\, \partial_t \mathbf{v} + \rho_f\, \partial_t \mathbf{q}
 &= \nabla \cdot \boldsymbol{\sigma}, \\
\rho_f\, \partial_t \mathbf{v} + m\, \partial_t \mathbf{q} + b\, \mathbf{q}
 &= -\nabla p, \\
\partial_t \boldsymbol{\sigma}
 &= (H - 2\mu)(\nabla \cdot \mathbf{v})\, \mathbf{I}
 + \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf T} \right)
 + C\, (\nabla \cdot \mathbf{q})\, \mathbf{I}, \\
\partial_t p &= -C\, \nabla \cdot \mathbf{v} - M\, \nabla \cdot \mathbf{q}.
\end{aligned}
$$

The eight models are the undrained P-wave modulus $H = \lambda_u + 2\mu$
(`H`), the frame shear modulus $\mu$ (`mu`), the coupling modulus
$C = \alpha M$ (`C`, Biot coefficient $\alpha$ times Biot modulus), the Biot
modulus $M$ (`M`), the bulk density
$\rho = (1 - \phi_0)\rho_s + \phi_0 \rho_f$ (`rho`), the fluid density
$\rho_f$ (`rho_f`), the effective fluid inertia $m = T \rho_f / \phi_0$ (`m`,
tortuosity $T$) and the viscous coupling $b = \eta / \kappa$ (`b`, fluid
viscosity over permeability). `biot_from_rock` builds them from rock and
fluid properties with Gassmann's $\alpha = 1 - K_d/K_s$ and
$1/M = (\alpha - \phi_0)/K_s + \phi_0/K_f$. Because the fluid unknown is the
relative flux, the equations carry no porosity-gradient terms in
heterogeneous media.

## Discretisation

Space is discretised on the staggered grid of [`Elastic`](elastic.md)
(Virieux, 1986), with standard staggered differences of order
`spatial_order` (default 4): `qx` shares the node of `vx`, `qz` that of `vz`,
and `p` that of `sxx` and `szz`. The material arrays are used at every node
without averaging. Time stepping is leapfrog. The viscous term is stiff, with
relaxation rate

$$
\gamma = \frac{b}{\tilde m}, \qquad \tilde m = m - \frac{\rho_f^2}{\rho},
$$

typically $10^5$–$10^6\ \mathrm{s^{-1}}$. Eliminating $\partial_t \mathbf{v}$
gives $\partial_t \mathbf{q} = -\gamma \mathbf{q} + \mathbf{G}$ with
$\mathbf{G} = -\big(\nabla p + (\rho_f/\rho)\, \nabla \cdot \boldsymbol{\sigma}\big)/\tilde m$.
With the stresses and pressure held at time level $n$ over a step, this is
integrated exactly,

$$
\begin{aligned}
\mathbf{q}^{n+1/2} &= e^{-\gamma \Delta t}\, \mathbf{q}^{n-1/2}
 + \frac{1 - e^{-\gamma \Delta t}}{\gamma}\, \mathbf{G}^{n}, \\
\mathbf{v}^{n+1/2} &= \mathbf{v}^{n-1/2}
 + \frac{1}{\rho} \left[ \Delta t\, \nabla \cdot \boldsymbol{\sigma}^{n}
 - \rho_f \left( \mathbf{q}^{n+1/2} - \mathbf{q}^{n-1/2} \right) \right],
\end{aligned}
$$

after which $\boldsymbol{\sigma}$ and $p$ are advanced with the new
$\mathbf{v}$ and $\mathbf{q}$. The step is therefore stable for any
permeability, with $\Delta t$ limited by the inviscid fast P wave
(`biot_fast_p_velocity`). In the limit $b \to \infty$ the flux vanishes and
the scheme reduces to `Elastic` with $v_p = \sqrt{H/\rho}$ and
$v_s = \sqrt{\mu/\rho}$. The equation runs on the eager torch path only.

## Boundary conditions

The default `pml_type="cpmls"` applies a CPML to every first derivative (see
[Absorbing boundaries](index.md#absorbing-boundaries)). With
`free_surface=True` the top face is an open-pore surface: zero traction of
the total stress and $p = 0$ on the `szz` row, imposed with the image method
of `Elastic` ($p$ mirrored like $\sigma_{zz}$, $q_z$ evenly). On that row
$\sigma_{xx}$ is updated with $\partial_z v_z$ eliminated through
$\sigma_{zz} = 0$ and $\partial_z q_z$ extrapolated linearly from the two rows
below,

$$
\partial_t \sigma_{xx} = \frac{4\mu(\lambda_u + \mu)}{H}\, \partial_x v_x
 + \frac{2\mu C}{H} \left( \partial_x q_x + \partial_z q_z \right).
$$

Only a flat top free surface is implemented; other faces and topography are
refused.

## References

- Biot, M. A., 1962, Mechanics of deformation and acoustic propagation in
  porous media: Journal of Applied Physics, 33(4), 1482–1498,
  [doi:10.1063/1.1728759](https://doi.org/10.1063/1.1728759).
- Carcione, J. M., and G. Quiroga-Goode, 1995, Some aspects of the physics
  and numerical modeling of Biot compressional waves: Journal of
  Computational Acoustics, 3(4), 261–280,
  [doi:10.1142/s0218396x95000136](https://doi.org/10.1142/s0218396x95000136).
- Dai, N., A. Vafidis, and E. R. Kanasewich, 1995, Wave propagation in
  heterogeneous, porous media: A velocity-stress, finite-difference method:
  Geophysics, 60(2), 327–340,
  [doi:10.1190/1.1443769](https://doi.org/10.1190/1.1443769).
- Robertsson, J. O. A., 1996, A numerical free-surface condition for
  elastic/viscoelastic finite-difference modeling in the presence of
  topography: Geophysics, 61(6), 1921–1934,
  [doi:10.1190/1.1444107](https://doi.org/10.1190/1.1444107).
- Virieux, J., 1986, P-SV wave propagation in heterogeneous media:
  Velocity-stress finite-difference method: Geophysics, 51(4), 889–901,
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

::: sweep.equations.Biot

::: sweep.equations.biot.biot_from_rock

::: sweep.equations.biot.biot_from_biot_constants

::: sweep.equations.biot.biot_wave_velocities

::: sweep.equations.biot.biot_fast_p_velocity

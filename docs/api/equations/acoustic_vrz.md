# AcousticVRZ / AcousticVRZ3D

## Governing equation

`AcousticVRZ` (2-D) and `AcousticVRZ3D` (3-D) solve the variable-density
acoustic wave equation for the pressure alone,

$$
\frac{\partial^2 p}{\partial t^2}
 = \kappa\, \nabla \cdot \left( b\, \nabla p \right)
 = \kappa \left( b\, \nabla^2 p + \nabla b \cdot \nabla p \right),
\qquad
\kappa = Z v_p,
\qquad
b = \frac{v_p}{Z},
$$

where $p$ is the pressure-like wavefield `h1` (the source and receiver
field), $v_p$ is the model `vp` and $Z$ is the model `z`. With
$Z = \rho v_p$, the acoustic impedance, $\kappa = \rho v_p^2$ is the bulk
modulus and $b = 1/\rho$ the buoyancy, so the models (`vp`, `z`) are an
impedance parameterisation of the variable-density equation. Time is
discretised with the second-order central difference. $\nabla^2 p$ is
assembled from second-derivative stencils of order `spatial_order`
(default 4), and $\nabla p$ and $\nabla b$ from first-derivative central
differences of the same order. $\nabla b$ is formed with the product rule
from the two models,
$\nabla b = (\nabla v_p)/Z + v_p \nabla (1/Z)$, not by differencing
$v_p/Z$. After each update the wavelet sample is added to $p^{n+1}$ at the
source cell, without scaling.

## Boundary conditions

The default `pml_type="cpmlr"` replaces $\nabla^2 p$ by its stretched form
and $\nabla p$ in the $\nabla b \cdot \nabla p$ term by
$\partial_x p + \psi_x$ (likewise $y$, $z$), with the memory fields
`psix`, `zetax`, `psiz`, `zetaz` (and `psiy`, `zetay` in 3-D); see
[Absorbing boundaries](index.md#absorbing-boundaries). With
`free_surface=True` the top face is pressure-release, $p = 0$.

## References

- Whitmore, N. D., J. Ramos-Martinez, Y. Yang, and A. Valenciano, 2020,
  Full wavefield modeling with vector reflectivity: EAGE 2020 Annual
  Conference & Exhibition Online, 1–5,
  [doi:10.3997/2214-4609.202010332](https://doi.org/10.3997/2214-4609.202010332).

::: sweep.equations.AcousticVRZ

::: sweep.equations.AcousticVRZ3D

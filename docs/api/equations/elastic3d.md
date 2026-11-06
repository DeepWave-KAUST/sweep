# Elastic3D

## Governing equations

`Elastic3D` solves the three-dimensional isotropic elastic wave equation in
velocity–stress form (Virieux, 1986):

$$
\begin{aligned}
\rho \frac{\partial v_i}{\partial t}
 &= \sum_{j} \frac{\partial \sigma_{ij}}{\partial x_j},
\\
\frac{\partial \sigma_{ii}}{\partial t}
 &= \lambda\, \nabla \cdot \mathbf{v}
  + 2\mu \frac{\partial v_i}{\partial x_i},
\\
\frac{\partial \sigma_{ij}}{\partial t}
 &= \mu \left( \frac{\partial v_i}{\partial x_j}
             + \frac{\partial v_j}{\partial x_i} \right),
 \qquad i \neq j,
\end{aligned}
$$

with $i, j \in \{x, y, z\}$ and no summation over repeated $i$ in the second
line. The particle velocities $v_x$, $v_y$, $v_z$ are the fields `vx`, `vy`,
`vz`, and the stresses $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$,
$\sigma_{xy}$, $\sigma_{xz}$, $\sigma_{yz}$ are `sxx`, `syy`, `szz`, `sxy`,
`sxz`, `syz`. The Lamé parameters are computed from the models `vp`, `vs` and
`rho` as

$$
\lambda = \rho \left( v_p^2 - 2 v_s^2 \right),
\qquad
\mu = \rho\, v_s^2 .
$$

The grid is the three-dimensional analogue of [Elastic](elastic.md): the
normal stresses sit at the integer nodes, each velocity component is shifted
half a cell along its own axis, and each shear stress $\sigma_{ij}$ is shifted
half a cell along both $i$ and $j$. The first derivatives are staggered
differences of order `spatial_order` (default 4), and the time stepping is
leapfrog (velocities first, then stresses). $\rho$, $\lambda$ and $\mu$ are
not averaged onto the staggered positions. After each step the wavelet sample
is added, unscaled, to every source field at the source cell; the default
source fields `sxx`, `syy` and `szz` form an explosive source. The default
receiver fields are `vx`, `vy` and `vz`; the shear stresses can be source but
not receiver fields.

## Boundary conditions

The default `pml_type="cpmls"` is the staggered-grid C-PML, with one memory
variable per first derivative (`m_vxx`, ..., `m_syzz`); see
[Absorbing boundaries](index.md#absorbing-boundaries).

The free surface is available on the top face only (`free_surface=True`). The
traction-free condition $\sigma_{zz} = \sigma_{xz} = \sigma_{yz} = 0$ is
imposed on the plane of the first model row by the image method: $\sigma_{zz}$,
$v_x$ and $v_y$ are mirrored about that plane, and $\sigma_{xz}$,
$\sigma_{yz}$ and $v_z$ about the plane half a cell above it
(Kristek et al., 2002); the stresses and $v_z$ are odd, $v_x$ and $v_y$ even.
After each step $\sigma_{zz}$ is set to zero on the surface row, and the
surface values of $\sigma_{xx}$ and $\sigma_{yy}$ are updated with the moduli
that follow from $\sigma_{zz} = 0$ (Robertsson, 1996):

$$
\frac{\partial \sigma_{xx}}{\partial t}
 = \frac{4\mu (\lambda + \mu)}{\lambda + 2\mu}
   \frac{\partial v_x}{\partial x}
 + \frac{2\lambda\mu}{\lambda + 2\mu}
   \frac{\partial v_y}{\partial y},
$$

and likewise for $\sigma_{yy}$ with $x$ and $y$ exchanged. With `topography`
(`impl='eager'` only), the surface follows a staircase of per-column surface
rows.

## References

- Kristek, J., P. Moczo, and R. J. Archuleta, 2002, Efficient methods to
  simulate planar free surface in the 3D 4th-order staggered-grid
  finite-difference schemes: Studia Geophysica et Geodaetica, 46(2), 355–381,
  [doi:10.1023/a:1019866422821](https://doi.org/10.1023/a:1019866422821).
- Robertsson, J. O. A., 1996, A numerical free-surface condition for
  elastic/viscoelastic finite-difference modeling in the presence of
  topography: Geophysics, 61(6), 1921–1934,
  [doi:10.1190/1.1444107](https://doi.org/10.1190/1.1444107).
- Virieux, J., 1986, P-SV wave propagation in heterogeneous media:
  Velocity-stress finite-difference method: Geophysics, 51(4), 889–901,
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

::: sweep.equations.Elastic3D

# Elastic

## Governing equations

`Elastic` solves the two-dimensional isotropic elastic wave equation in
velocity–stress form (Virieux, 1986):

$$
\begin{aligned}
\rho \frac{\partial v_x}{\partial t}
 &= \frac{\partial \sigma_{xx}}{\partial x}
  + \frac{\partial \sigma_{xz}}{\partial z},
&
\rho \frac{\partial v_z}{\partial t}
 &= \frac{\partial \sigma_{xz}}{\partial x}
  + \frac{\partial \sigma_{zz}}{\partial z},
\\
\frac{\partial \sigma_{xx}}{\partial t}
 &= (\lambda + 2\mu) \frac{\partial v_x}{\partial x}
  + \lambda \frac{\partial v_z}{\partial z},
&
\frac{\partial \sigma_{zz}}{\partial t}
 &= \lambda \frac{\partial v_x}{\partial x}
  + (\lambda + 2\mu) \frac{\partial v_z}{\partial z},
\\
\frac{\partial \sigma_{xz}}{\partial t}
 &= \mu \left( \frac{\partial v_x}{\partial z}
             + \frac{\partial v_z}{\partial x} \right),
\end{aligned}
$$

where $v_x$, $v_z$ are the particle velocities `vx`, `vz` and $\sigma_{xx}$,
$\sigma_{zz}$, $\sigma_{xz}$ the stresses `sxx`, `szz`, `sxz`. The Lamé
parameters are computed from the models `vp`, `vs` and `rho` as

$$
\lambda = \rho \left( v_p^2 - 2 v_s^2 \right),
\qquad
\mu = \rho\, v_s^2 .
$$

The fields are stored on the standard staggered grid: $\sigma_{xx}$ and
$\sigma_{zz}$ at the integer nodes $(x, z)$, $v_x$ at $(x + h/2, z)$, $v_z$ at
$(x, z + h/2)$ and $\sigma_{xz}$ at $(x + h/2, z + h/2)$. The first derivatives
are staggered differences of order `spatial_order` (default 4). Time stepping
is leapfrog: the velocities are advanced from the stresses at step $n$, then
the stresses from the new velocities. The model values are not averaged onto
the staggered positions; $v_x$, $v_z$ and $\sigma_{xz}$ use $\rho$ and $\mu$ of
the cell with the same index. After each step the wavelet sample is added,
unscaled, to every source field at the source cell; the default source fields
`sxx` and `szz` form an explosive source. The default receiver fields are `vx`
and `vz`; `sxz` can be a source but not a receiver field.

## Boundary conditions

The default `pml_type="cpmls"` is the staggered-grid C-PML: each of the eight
first derivatives in the equations above is replaced inside the layer by its
stretched form with one memory variable (`m_vxx`, `m_vxz`, `m_vzx`, `m_vzz`,
`m_txxx`, `m_tzzz`, `m_txzx`, `m_txzz`), using the profiles of the integer or
the half-integer nodes as the derivative requires; see
[Absorbing boundaries](index.md#absorbing-boundaries).

With `free_surface`, the traction-free condition
$\sigma_{zz} = \sigma_{xz} = 0$ is imposed on the plane of the first model row
(the $\sigma_{zz}$ nodes) by the image method. Above the surface $\sigma_{zz}$
and $v_x$ are mirrored about that plane, and $\sigma_{xz}$ and $v_z$ about the
plane half a cell above it (Kristek et al., 2002); $\sigma_{zz}$,
$\sigma_{xz}$ and $v_z$ are odd and $v_x$ is even. After each step
$\sigma_{zz}$ is set to zero on the surface row. The surface value of
$\sigma_{xx}$ is updated with the modulus that follows from $\sigma_{zz} = 0$,

$$
\frac{\partial \sigma_{xx}}{\partial t}
 = \frac{4\mu (\lambda + \mu)}{\lambda + 2\mu}
   \frac{\partial v_x}{\partial x}
$$

(Robertsson, 1996). On a flat surface, the derivatives normal to the surface
are second-order accurate in the $M$ = `spatial_order`/2 rows next to it,
the surface row included. Any subset of the four faces can be a free surface
(for example `free_surface=["top", "left"]`). On an $x$ face the roles of $x$
and $z$ are exchanged, and on a bottom or right face the surface row of
$\sigma_{xz}$ is also set to zero. With `topography`, the top surface follows
a staircase of per-column surface rows. A PML that runs between two faces of the other axis
with no PML, one of them free, is refused: the solid between them is a
plate, whose backward-propagating Lamb modes the PML amplifies (Loh et al.,
2009).

## References

- Kristek, J., P. Moczo, and R. J. Archuleta, 2002, Efficient methods to
  simulate planar free surface in the 3D 4th-order staggered-grid
  finite-difference schemes: Studia Geophysica et Geodaetica, 46(2), 355–381,
  [doi:10.1023/a:1019866422821](https://doi.org/10.1023/a:1019866422821).
- Loh, P.-R., A. F. Oskooi, M. Ibanescu, M. Skorobogatiy, and S. G. Johnson,
  2009, Fundamental relation between phase and group velocity, and
  application to the failure of perfectly matched layers in backward-wave
  structures: Physical Review E, 79(6), 065601,
  [doi:10.1103/PhysRevE.79.065601](https://doi.org/10.1103/PhysRevE.79.065601).
- Robertsson, J. O. A., 1996, A numerical free-surface condition for
  elastic/viscoelastic finite-difference modeling in the presence of
  topography: Geophysics, 61(6), 1921–1934,
  [doi:10.1190/1.1444107](https://doi.org/10.1190/1.1444107).
- Virieux, J., 1986, P-SV wave propagation in heterogeneous media:
  Velocity-stress finite-difference method: Geophysics, 51(4), 889–901,
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

::: sweep.equations.Elastic

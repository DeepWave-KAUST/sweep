# Acoustic1st

## Governing equation

`Acoustic1st` solves the two-dimensional variable-density acoustic equations
in first-order pressure–velocity form,

$$
\rho \frac{\partial v_x}{\partial t} = -\frac{\partial p}{\partial x},
\qquad
\rho \frac{\partial v_z}{\partial t} = -\frac{\partial p}{\partial z},
\qquad
\frac{\partial p}{\partial t}
 = -\rho v_p^2 \left( \frac{\partial v_x}{\partial x}
                    + \frac{\partial v_z}{\partial z} \right),
$$

where $p$ is the pressure `p` (the default source and receiver field),
$v_x$ and $v_z$ are the particle velocities `vx` and `vz`, $v_p$ is the
model `vp` and $\rho$ the model `rho`; $\rho v_p^2$ is the bulk modulus.
The fields lie on a staggered grid: $p$ on the integer nodes and $v_x$, $v_z$
half a cell away along their own axis. The first derivatives are staggered
differences of order `spatial_order` (default 4). Time is discretised with a
staggered leapfrog: the velocities are updated from $p^{n}$, then the
pressure from the new velocities. After each step the wavelet sample is added
to $p$ at the source cell, without scaling.

## Boundary conditions

The default `pml_type="cpmls"` gives each of the four derivatives its own
memory field: `psix`, `psiz` for $\partial_x p$ and $\partial_z p$, and
`phix`, `phiz` for $\partial_x v_x$ and $\partial_z v_z$; see
[Absorbing boundaries](index.md#absorbing-boundaries). The alternative
`pml_type="spml"` is the split-field PML described there; its fields are
`px`, `pz`, `vx`, `vz` with $p = p_x + p_z$, the default source field is
`px` and the default receiver field is `vx`. In this mode the step solves
$\rho\, \partial_t \mathbf{v} = +\nabla p$ and
$\partial_t p = +\rho v_p^2 \nabla \cdot \mathbf{v}$, so the pressure obeys
the same wave equation but the velocities have the opposite sign to those of
`cpmls`.

The step applies no free-surface condition. `free_surface=True` is accepted,
but it only removes the absorbing layer from the top face, which then
becomes the edge of the grid; it is not a calibrated pressure-release
surface.

::: sweep.equations.Acoustic1st

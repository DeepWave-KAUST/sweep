# Acoustic3D

## Governing equation

`Acoustic3D` solves the three-dimensional constant-density acoustic wave
equation

$$
\frac{\partial^2 p}{\partial t^2}
 = v_p^2 \left( \frac{\partial^2 p}{\partial x^2}
              + \frac{\partial^2 p}{\partial y^2}
              + \frac{\partial^2 p}{\partial z^2} \right),
$$

where $p$ is the pressure-like wavefield `h1` (the source and receiver field)
and $v_p$ is the model `vp`. As in [Acoustic](acoustic.md), time is
discretised with the second-order central difference and each second
derivative with a central difference of order `spatial_order` (default 4):

$$
p^{n+1} = 2p^{n} - p^{n-1}
 + \Delta t^2\, v_p^2 \left( \delta_{xx} + \delta_{yy} + \delta_{zz} \right)
   p^{n}.
$$

After each update the wavelet sample is added to $p^{n+1}$ at the source
cell, without the factor $v_p^2 \Delta t^2$.

## Boundary conditions

The default `pml_type="cpmlr"` replaces each term $\partial_{xx} p$ of the
Laplacian inside the absorbing layers by
$\partial_{\tilde{x}} ( \partial_{\tilde{x}} p )$, with the memory fields
`psix`, `zetax`, `psiy`, `zetay` and `psiz`, `zetaz`; see
[Absorbing boundaries](index.md#absorbing-boundaries). With
`free_surface=True` the top face is pressure-release, $p = 0$; the other
faces are always absorbing.

::: sweep.equations.Acoustic3D

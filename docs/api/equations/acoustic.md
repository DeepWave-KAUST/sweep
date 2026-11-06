# Acoustic

## Governing equation

`Acoustic` solves the two-dimensional constant-density acoustic wave equation

$$
\frac{\partial^2 p}{\partial t^2}
 = v_p^2 \left( \frac{\partial^2 p}{\partial x^2}
              + \frac{\partial^2 p}{\partial z^2} \right),
$$

where $p$ is the pressure-like wavefield `h1` (the source and receiver field)
and $v_p$ is the model `vp`. Time is discretised with the second-order central
difference and the second derivatives with central differences of order
`spatial_order` (default 4):

$$
p^{n+1} = 2p^{n} - p^{n-1}
 + \Delta t^2\, v_p^2 \left( \delta_{xx} + \delta_{zz} \right) p^{n}.
$$

After each update the wavelet sample $s^n$ is added to $p^{n+1}$ at the source
cell. It is not multiplied by $v_p^2 \Delta t^2$, so the recorded amplitude
depends on $\Delta t$ and the grid spacing.

## Boundary conditions

The default `pml_type="cpmlr"` replaces each term $\partial_{xx} p$ of the
Laplacian inside the absorbing layers by the stretched form
$\partial_{\tilde{x}} ( \partial_{\tilde{x}} p )$, carried by the memory
fields `psix`, `zetax` ($x$) and `psiz`, `zetaz` ($z$); the profiles and the
recursion are given under
[Absorbing boundaries](index.md#absorbing-boundaries). With
`free_surface=True` the selected faces are pressure-release, $p = 0$.

::: sweep.equations.Acoustic

# ViscoAcoustic

## Governing equation

`ViscoAcoustic` solves the two-dimensional nearly constant-Q acoustic wave
equation with decoupled fractional Laplacians of Zhu and Harris (2014,
eqs. 10 and 11),

$$
\frac{\partial^2 p}{\partial t^2}
 = -c^2 \hat{\eta}\, \left( -\nabla^2 \right)^{\bar{\gamma} + 1} p
   - c^2 \hat{\tau}\, \left( -\nabla^2 \right)^{\bar{\gamma} + 1/2}
     \frac{\partial p}{\partial t},
$$

$$
\gamma = \frac{1}{\pi} \arctan \frac{1}{Q},
\qquad
c = v_p \cos \frac{\pi \gamma}{2},
\qquad
\hat{\eta} = \cos(\pi \gamma) \left( \frac{v_p}{\omega_0} \right)^{2\gamma},
\qquad
\hat{\tau} = \frac{\sin(\pi \gamma)}{v_p}
             \left( \frac{v_p}{\omega_0} \right)^{2\gamma},
$$

where $p$ is the pressure-like wavefield `h1` (the source and receiver
field), $v_p$ is the model `vp`, $Q$ the model `Q` and $\omega_0$ the model
`omega`, the reference angular frequency in rad/s. The first term gives the
power-law velocity dispersion and the second the amplitude loss. The
coefficients $c$, $\hat{\eta}$ and $\hat{\tau}$ vary in space with the local
$\gamma$, but the exponent of the fractional operators is frozen at the mean
$\bar{\gamma}$ of $\gamma$ over the model (one mean per shot for per-shot
models); its derivative with respect to $Q$ is not propagated. At
$\gamma = 0$ the equation reduces to the acoustic equation of
[Acoustic](acoustic.md).

The fractional operators are applied in the wavenumber domain,
$(-\nabla^2)^{s} \leftrightarrow |\mathbf{k}|^{2s}$, with FFTs over the whole
padded grid (absorbing layers included), and the space-dependent
coefficients multiply the result after the inverse transform. The step is
that of [Acoustic](acoustic.md) at the velocity $c$, plus two spectral
corrections:

$$
p^{n+1} = 2p^{n} - p^{n-1}
 + \Delta t^2 c^2 \tilde{L} p^{n}
 + \Delta t^2 c^2 \left( \mathcal{F}^{-1}\!\left[ |\mathbf{k}|^2 \hat{p}^{n} \right]
   - \hat{\eta}\, \mathcal{F}^{-1}\!\left[ |\mathbf{k}|^{2\bar{\gamma}+2}
     \hat{p}^{n} \right] \right)
 - \Delta t^2 c^2 \hat{\tau}\, \mathcal{F}^{-1}\!\left[
   |\mathbf{k}|^{2\bar{\gamma}+1}\,
   \mathcal{F}\!\left[ \frac{p^{n} - p^{n-1}}{\Delta t} \right] \right],
$$

where $\tilde{L}$ is the finite-difference Laplacian of order
`spatial_order` (with the CPML terms inside the absorbing layers) and
$\hat{p}^{n} = \mathcal{F}[p^{n}]$. The term
$\mathcal{F}^{-1}[|\mathbf{k}|^2 \hat{p}^{n}] = -\nabla^2 p^{n}$ cancels the
Laplacian in $\tilde{L}$ to the accuracy of the stencil, and the term after it
puts the fractional dispersion operator in its place; the pair vanishes
identically at $\gamma = 0$. The time derivative in the loss term is the
backward difference. The wavenumber grid is built on the padded runtime grid
from both spacings, $k_z$ with `dz` and $k_x$ with `dx`. After the spectral terms the `spatial_order // 2` halo cells on all
four edges are set to zero. The wavelet sample is added to $p^{n+1}$ at the
source cell, as in [Acoustic](acoustic.md).

The constructor switches act on the two terms separately.
`phase_shift=False` drops the dispersion term and its spectral correction and
uses $c = v_p$; `amplitude_damping=False` drops the loss term. With both off
the step is that of `Acoustic`.

## Boundary conditions

The default `pml_type="cpmlr"` acts on the finite-difference Laplacian
$\tilde{L}$ only, with the memory fields `psix`, `zetax`, `psiz`, `zetaz`;
see [Absorbing boundaries](index.md#absorbing-boundaries). With
`free_surface=True` the selected faces are pressure-release, $p = 0$.

## References

- Zhu, T., and J. M. Harris, 2014, Modeling acoustic wave propagation in
  heterogeneous attenuating media using decoupled fractional Laplacians:
  Geophysics, 79(3), T105–T116,
  [doi:10.1190/geo2013-0245.1](https://doi.org/10.1190/geo2013-0245.1).

::: sweep.equations.ViscoAcoustic

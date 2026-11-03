# ViscoElastic

## Governing equations

`ViscoElastic` solves the two-dimensional visco-elastic wave equation for a
generalised standard linear solid: $L$ standard linear solids (`n_sls`,
default 3) in parallel, with stress relaxation times $\tau_{\sigma l}$ fixed
by a frequency band. The velocity equations are those of
[Elastic](elastic.md),

$$
\rho \frac{\partial v_x}{\partial t}
 = \frac{\partial \sigma_{xx}}{\partial x}
 + \frac{\partial \sigma_{xz}}{\partial z},
\qquad
\rho \frac{\partial v_z}{\partial t}
 = \frac{\partial \sigma_{xz}}{\partial x}
 + \frac{\partial \sigma_{zz}}{\partial z},
$$

and the stresses carry one memory variable per component and mechanism
(Robertsson et al., 1994). With $\theta = \partial_x v_x + \partial_z v_z$,

$$
\begin{aligned}
\frac{\partial \sigma_{xx}}{\partial t}
 &= \pi_U\, \theta - 2\mu_U \frac{\partial v_z}{\partial z}
  + \sum_{l=1}^{L} r_{xx,l},
\\
\frac{\partial \sigma_{zz}}{\partial t}
 &= \pi_U\, \theta - 2\mu_U \frac{\partial v_x}{\partial x}
  + \sum_{l=1}^{L} r_{zz,l},
\\
\frac{\partial \sigma_{xz}}{\partial t}
 &= \mu_U \left( \frac{\partial v_x}{\partial z}
               + \frac{\partial v_z}{\partial x} \right)
  + \sum_{l=1}^{L} r_{xz,l},
\\
\frac{\partial r_{xx,l}}{\partial t}
 &= -\frac{1}{\tau_{\sigma l}} \left( r_{xx,l}
  + \pi_R \tau^{p}_{l}\, \theta
  - 2\mu_R \tau^{s}_{l} \frac{\partial v_z}{\partial z} \right),
\\
\frac{\partial r_{zz,l}}{\partial t}
 &= -\frac{1}{\tau_{\sigma l}} \left( r_{zz,l}
  + \pi_R \tau^{p}_{l}\, \theta
  - 2\mu_R \tau^{s}_{l} \frac{\partial v_x}{\partial x} \right),
\\
\frac{\partial r_{xz,l}}{\partial t}
 &= -\frac{1}{\tau_{\sigma l}} \left( r_{xz,l}
  + \mu_R \tau^{s}_{l} \left( \frac{\partial v_x}{\partial z}
                            + \frac{\partial v_z}{\partial x} \right) \right),
\end{aligned}
$$

where $r_{xx,l}$, $r_{zz,l}$, $r_{xz,l}$ are the fields `r_xx_l`, `r_zz_l`,
`r_xz_l` ($l$ counted from 0 in the field names), $\pi_R$ and $\mu_R$ are the
relaxed P-wave and shear moduli, and the unrelaxed moduli are

$$
\pi_U = \pi_R \Big( 1 + \sum_{l} \tau^{p}_{l} \Big),
\qquad
\mu_U = \mu_R \Big( 1 + \sum_{l} \tau^{s}_{l} \Big).
$$

Each modulus $M$ ($\pi$ with strengths $\tau^{p}_{l}$, $\mu$ with
$\tau^{s}_{l}$) is therefore

$$
M(\omega) = M_R \left( 1 + \sum_{l=1}^{L} \tau_l
 \frac{i\omega\tau_{\sigma l}}{1 + i\omega\tau_{\sigma l}} \right).
$$

The relaxation times are $\tau_{\sigma l} = 1/(2\pi f_l)$ with $f_l$
log-spaced over `f_band` $= (f_{\min}, f_{\max})$; the default band is
$(f_{\mathrm{ref}}/\sqrt{12},\ f_{\mathrm{ref}}\sqrt{12})$ and the default
reference frequency `f_ref` is 10 Hz. The strengths $\tau^{p}_{l}$ and
$\tau^{s}_{l}$ follow from the models `Qp` and `Qs` by the least-squares
constant-$Q$ fit of Emmerich and Korn (1987): with
$a_l = (\omega\tau_{\sigma l})^2 / (1 + (\omega\tau_{\sigma l})^2)$ and
$b_l = \omega\tau_{\sigma l} / (1 + (\omega\tau_{\sigma l})^2)$, the condition
$\operatorname{Re} M / \operatorname{Im} M = Q$ reads

$$
\sum_{l=1}^{L} \tau_l \left( Q\, b_l(\omega) - a_l(\omega) \right) = 1,
$$

which is solved in the least-squares sense at 200 log-spaced frequencies in
the band (see `GSLSFit` below). The solution is a rational function of $1/Q$,
so `Qp` and `Qs` are differentiable models.

The models `vp` and `vs` are phase velocities at `f_ref`. The relaxed moduli
are set so that $\operatorname{Re}\sqrt{\rho / M(\omega_{\mathrm{ref}})}
= 1/v$:

$$
\pi_R = \rho\, v_p^2\, \frac{\cos^2(\phi_p/2)}{|m_p|},
\qquad
\mu_R = \rho\, v_s^2\, \frac{\cos^2(\phi_s/2)}{|m_s|},
\qquad
m = \frac{M(\omega_{\mathrm{ref}})}{M_R} = |m|\, e^{i\phi},
$$

with $m_p$ evaluated with $\tau^{p}_{l}$ and $m_s$ with $\tau^{s}_{l}$. With
`Qp = Qs = inf` all strengths vanish and the equation reduces exactly to
[Elastic](elastic.md).

The grid, stencil and leapfrog are those of [Elastic](elastic.md). The memory
variables are stored at integer time levels and advanced with the
trapezoidal rule,

$$
r_l^{n+1}
 = \frac{1 - \Delta t / 2\tau_{\sigma l}}{1 + \Delta t / 2\tau_{\sigma l}}\,
   r_l^{n}
 - \frac{\Delta t / \tau_{\sigma l}}{1 + \Delta t / 2\tau_{\sigma l}}\, F_l,
$$

where $F_l$ is the bracketed source term above evaluated with the velocities
at $n + 1/2$; the stress update adds $\tfrac{\Delta t}{2} \sum_l
(r_l^{n+1} + r_l^{n})$. The explicit update runs at the unrelaxed velocities
$\sqrt{\pi_U/\rho}$ and $\sqrt{\mu_U/\rho}$, which exceed `vp` and `vs` for
finite $Q$, so $\Delta t$ must satisfy the stability condition for these
velocities (`unrelaxed_velocities`). $Q$ must exceed `q_min`, below which the
fit gives a negative strength.

## Boundary conditions

The default `pml_type="cpmls"` is the staggered-grid C-PML of
[Elastic](elastic.md); see
[Absorbing boundaries](index.md#absorbing-boundaries). The C-PML acts on the
velocity derivatives that drive both the stresses and the memory variables.

A flat free surface is supported on any subset of the four faces; topography
is not. The image method of [Elastic](elastic.md) is used, but instead of the
elastic surface modulus the normal strain rate on the surface row
($\partial_z v_z$ on a $z$ face, $\partial_x v_x$ on an $x$ face) is solved
from the condition that the normal stress stays zero through the full update,
memory variables included. In the elastic limit this gives the modulus
$4\mu(\lambda + \mu)/(\lambda + 2\mu)$ used by [Elastic](elastic.md).
As for [Elastic](elastic.md), a PML between two unabsorbed faces of the other
axis, one of them free, is refused; attenuation does not stop the PML
amplifying the plate's backward Lamb modes.

## References

- Emmerich, H., and M. Korn, 1987, Incorporation of attenuation into
  time-domain computations of seismic wave fields: Geophysics, 52(9),
  1252–1264, [doi:10.1190/1.1442386](https://doi.org/10.1190/1.1442386).
- Robertsson, J. O. A., J. O. Blanch, and W. W. Symes, 1994, Viscoelastic
  finite-difference modeling: Geophysics, 59(9), 1444–1456,
  [doi:10.1190/1.1443701](https://doi.org/10.1190/1.1443701).

::: sweep.equations.ViscoElastic

::: sweep.equations.visco_elastic.GSLSFit

::: sweep.equations.gsls_nonlinear.GSLSNonlinearFit

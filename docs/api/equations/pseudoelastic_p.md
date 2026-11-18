# PseudoelasticP

## Governing equation

`PseudoelasticP` solves the pseudoelastic pure P-mode wave equation of
Sun and Alkhalifah (2021): the two-dimensional isotropic elastic equations
in velocity–stress form, with the particle velocity replaced by its
curl-free part. With the normalised gradient
$\tilde\nabla = \nabla / \sqrt{-\Delta}$ the first-order form reads

$$
\frac{\partial P}{\partial t}
 = \tilde\nabla \cdot \left[ \frac{1}{\rho}
   \left( \nabla \cdot \boldsymbol{\tau} + \mathbf{f} \right) \right],
\qquad
\frac{\partial \boldsymbol{\tau}}{\partial t}
 = \mathbf{C} : \nabla \left( \tilde\nabla P \right),
$$

where $\boldsymbol{\tau}$ is the stress tensor (`sxx`, `szz`, `sxz`),
$\mathbf{C}$ the isotropic stiffness built from the models `vp`, `vs`, `rho`
($\lambda = \rho(v_p^2 - 2 v_s^2)$, $\mu = \rho v_s^2$) and $P$ the scalar
P-wave potential rate. The velocity that drives the stresses is
$\mathbf{v} = \tilde\nabla P = \tilde\nabla \tilde\nabla \cdot \mathbf{v}$:
the irrotational projection of the elastic velocity. The equation therefore
propagates no S wave and no P–S or S–P conversion, and its grid is set by
the P wavelength alone (`vs` does not constrain the spacing). The PP
reflections keep the elastic amplitude variation with angle, because `vs`
enters the stress update through $\mu$: at an interface with a jump in
$\mu$ only, the acoustic equation reflects nothing while this one reflects
like `Elastic` at oblique incidence. Since no energy leaves to converted
waves, the PP reflection comes out stronger than the elastic one by about
the relative shear-modulus jump squared (5 % at a 6 % jump in $v_s$, 25 %
at a 50 % jump); the radiation patterns agree to first order.

Compared with a first-order (Born) elastic correction of the acoustic
equation, the shear modulus acts here through the full stress update, so
large contrasts and post-critical reflections are handled without a
small-perturbation assumption.

## Implementation

The step is `Elastic`'s staggered-grid leapfrog — same stencils of order
`spatial_order`, CPML, source and receiver fields, image-method free
surface — with the projection inserted between the velocity and the stress
sub-steps. The projection is spectral: two 2-D FFTs each way over the whole
padded grid. It is the orthogonal projector onto the range of the discrete
staggered gradient, built from the exact symbol of the propagator's stencil
(the FFT of its impulse response), so a discrete FD gradient field is left
unchanged; with `vs = 0` the record equals that of `Elastic` with `vs = 0`
to round-off.

The wavefields and models are those of `Elastic`: `(vx, vz, sxx, szz,
sxz)` plus the ten CPML memories, models `(vp, vs, rho)`. `vx` and `vz`
hold the projected (P) velocity. The default source is the explosion
`['sxx', 'szz']` and the default receivers `['vx', 'vz']`; a vertical force
`['vz']` is also accepted — only its P part radiates.

## Boundary conditions

The CPML of `Elastic` is applied in the two sub-steps; the projection also
acts on the absorbing layer. With `free_surface=True` the grid is mirrored
about the surface row inside the projection (`vx` odd, `vz` even), which
keeps the normal stresses zero on the surface; this is exact for a fluid top
(`vs = 0` along the surface row, as in marine data) and is refused
otherwise. Only a flat free surface on the top face is supported; per-edge
free surfaces and topography are refused.

## Limits

- Eager only (`impl='eager'`); there is no compiled path and no 3-D
  version. `torch.compile` works on the step.
- The projection reads the whole grid, so boundary saving is refused; use
  the default chunk checkpointing or `memory=Full()`. Domain decomposition
  is not available.
- No S waves: data windows dominated by converted waves (e.g. the vertical
  geophone of ocean-bottom nodes at late times) are not modelled.

Reference: B. Sun and T. Alkhalifah, 2021, Pseudoelastic pure P-mode wave
equation, *Geophysics* 86(6), T469–T485,
[10.1190/geo2021-0084.1](https://doi.org/10.1190/geo2021-0084.1).

::: sweep.equations.PseudoelasticP

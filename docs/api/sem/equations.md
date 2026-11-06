# SEM equations

The equation classes of `sweep.sem.equations`, all importable as
`sweep.sem.<Class>`. A class is not constructed by hand: it is passed to
[`PropSEM`](prop_sem.md), which builds the mesh, interpolates the models onto
its nodes and calls the class's `build`:

```python
from sweep.sem import PropSEM, AcousticPotentialSEM2D

solver = PropSEM(AcousticPotentialSEM2D, shape=(141, 681), dh=25.0, dt=0.002)
seis = solver(wavelet, sources, receivers, models=[vp])
```

`build(mesh, <models>, *, ...)` is that call. Its positional arguments after
the mesh are the models as element arrays (`vp_e` is `vp` at the GLL nodes of
every element), in the order `models=` takes them, except `rho` on
`AcousticPotentialSEM2D`, which `PropSEM` fills from `density=`. Its keywords
are what `PropSEM` sets up from its own options, such as the free surface and
the C-PML profile. The mesh family, the models, the CUDA kernels and the
boundary-saving coverage of every class are in the
[SEM family table](../../user-guide/equations.md#sem-family-spectral-elements);
`sweep show <Class>` prints the fields and the source and receiver types.

## Governing equations

Each subsection below gives the continuous equations of a group of classes,
ties the symbols to the models (`build` arguments, in `models=` order) and to
the field names, and states the boundary conditions. The coordinates are $x$
(right), $y$ and $z$ (down); $\phi_i$ is the GLL basis function of global
node $i$.

### Discretisation

An element spans `n_order` cells of the model grid in each direction. Inside
it the fields are Lagrange polynomials of degree $N$ = `n_order` (default 4)
on the $N+1$ Gauss–Lobatto–Legendre (GLL) points per direction, and the models
are interpolated from the grid to those points. The integrals of the weak form
are evaluated by GLL quadrature on the same points, so the mass matrix
$\mathbf{M}$ is diagonal and is inverted node by node. The `Quad4Mesh` and
`Hex8Mesh` classes (`AcousticSEM2D`, `AcousticSEM3D`, `ElasticSEM2D`,
`ElasticSEM3D`, `ElasticVSSEM3D`) have straight, axis-aligned elements; the
`Quad9Mesh` and `Hex27Mesh` classes (all others) map each element with a 9- or
27-node quadratic geometry that follows a `topography`, and evaluate the
spatial derivatives with the inverse Jacobian at every GLL point (Komatitsch
and Vilotte, 1998; Komatitsch and Tromp, 1999).

The second-order classes (scalar acoustic, potential, elastic displacement and
fluid–solid) advance the semi-discrete system
$\mathbf{M}\ddot{\mathbf{u}} + \mathbf{K}\mathbf{u} = \mathbf{f}$ with the
explicit central difference

$$
\mathbf{u}^{n+1} = 2\mathbf{u}^{n} - \mathbf{u}^{n-1}
 + \Delta t^2\, \mathbf{M}^{-1} \left( \mathbf{f}^{n}
 - \mathbf{K}\mathbf{u}^{n} \right),
$$

which is SPECFEM's Newmark scheme ($\beta = 0$, $\gamma = 1/2$) written in
three-level form. The velocity–stress and Biot classes use a staggered
leapfrog: velocity (and Darcy flux) at $n + 1/2$, stress (and pore pressure)
at $n + 1$. A point source enters $\mathbf{f}$ as a weak-form load, the
wavelet times the Lagrange basis evaluated at the source position, and a
receiver interpolates the nodal field with the same basis.

### Constant-density acoustic

`AcousticSEM2D`, `AcousticSEM2DDeformed`, `AcousticSEM3D` and
`AcousticSEM3DDeformed` solve

$$
\frac{\partial^2 u}{\partial t^2} = v_p^2\, \nabla^2 u + s,
$$

where $u$ is the field `u` (the source and receiver field; the receiver name
`p` is an alias of it, not a transform) and $v_p$ is the model `vp`. The mass
$M_{ij} = \int_\Omega \phi_i \phi_j \, d\Omega$ and the stiffness
$K_{ij} = \int_\Omega \nabla\phi_i \cdot \nabla\phi_j \, d\Omega$ contain no
velocity; $v_p^2$ multiplies the assembled term at each global node,

$$
\ddot{\mathbf{u}} = -\mathbf{V}^2 \mathbf{M}^{-1} \mathbf{K}\mathbf{u}
 + \mathbf{M}^{-1} \mathbf{f},
$$

with $\mathbf{V}^2$ the diagonal of nodal $v_p^2$. The scheme therefore
discretises $v_p^2 \nabla^2 u$, the operator of the finite-difference
[`Acoustic`](../equations/acoustic.md), and not
$\nabla \cdot (v_p^2 \nabla u)$, which would keep $v_p^2\, \partial u /
\partial n$ instead of $\partial u / \partial n$ continuous at a velocity
contrast and reverse the sign of the reflection coefficients. The source is
not multiplied by $v_p^2$. A `free_surface` side is pressure-release,
$u = 0$, imposed by zeroing its nodes after every step; a side with neither a
free surface nor an absorbing layer carries the natural condition
$\partial u / \partial n = 0$.

### Potential-form acoustic

`AcousticPotentialSEM2D` solves the variable-density acoustic equation in
SPECFEM's potential normalisation,

$$
\frac{1}{\kappa} \frac{\partial^2 \chi}{\partial t^2}
 = \nabla \cdot \left( \frac{1}{\rho} \nabla \chi \right) + s,
\qquad \kappa = \rho v_p^2,
$$

where $\chi$ is the field `chi`. The displacement is
$\mathbf{u} = \rho^{-1} \nabla \chi$ and the pressure is
$p = -\partial^2 \chi / \partial t^2$. The weak form

$$
\int_\Omega \frac{1}{\kappa}\, \phi_i\, \ddot{\chi} \, d\Omega
 + \int_\Omega \frac{1}{\rho}\, \nabla\phi_i \cdot \nabla\chi \, d\Omega
 = \int_\Omega \phi_i\, s \, d\Omega
$$

gives a $\kappa^{-1}$-weighted diagonal mass and a stiffness with $\rho^{-1}$
inside the integral. Under this normalisation the fluid–solid coupling terms
carry no explicit density (see [Fluid–solid coupling](#fluidsolid-coupling)).
`build` takes `vp_e` and `rho_e`; `PropSEM` fills `rho_e` with the constant
`density=` (default 1000), so `models=[vp]`. For constant $\rho$ the equation
reduces to $\ddot{\chi} = v_p^2 \nabla^2 \chi$. The recorded field is `chi`;
the receiver `p` is $-\partial^2 \chi / \partial t^2$, formed from the
recorded trace by the centred second difference. A `free_surface` side is
$\chi = 0$, which is $p = 0$; a side with neither a free surface nor an
absorbing layer is rigid, $\mathbf{u} \cdot \mathbf{n} = 0$, the natural
condition of the weak form.

### Elastic displacement

`ElasticSEM2D`, `ElasticSEM2DDeformed`, `ElasticSEM3D` and
`ElasticSEM3DDeformed` solve

$$
\rho \frac{\partial^2 \mathbf{u}}{\partial t^2}
 = \nabla \cdot \boldsymbol{\sigma} + \mathbf{f},
\qquad
\boldsymbol{\sigma} = \lambda (\nabla \cdot \mathbf{u})\, \mathbf{I}
 + \mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right),
$$

with $\lambda = \rho (v_p^2 - 2 v_s^2)$ and $\mu = \rho v_s^2$ from the models
`vp`, `vs`, `rho`. The displacement components are the fields `u_x`, `u_z`
(and `u_y`); the stress is formed inside the stiffness and not stored. In the
weak form

$$
\int_\Omega \rho\, \mathbf{w} \cdot \ddot{\mathbf{u}} \, d\Omega
 + \int_\Omega \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{w})
 \, d\Omega
 = \int_\Omega \mathbf{w} \cdot \mathbf{f} \, d\Omega
 + \oint_\Gamma \mathbf{w} \cdot (\boldsymbol{\sigma}\mathbf{n}) \, d\Gamma
$$

the boundary integral is dropped, so every side without an absorbing layer,
including a free surface and a `topography`, is traction-free
($\boldsymbol{\sigma}\mathbf{n} = 0$) as a natural condition; no image or
projection is used. The receivers `vx`, `vy`, `vz` are
$\partial \mathbf{u} / \partial t$, formed from the recorded traces by the
centred first difference. Of the four, only `ElasticSEM2DDeformed` has a
C-PML.

### Elastic velocity–stress and VTI

`ElasticVSSEM2DDeformed`, `ElasticVSSEM3D` and `ElasticVSSEM3DDeformed` solve
the first-order system

$$
\rho \frac{\partial \mathbf{v}}{\partial t}
 = \nabla \cdot \boldsymbol{\sigma} + \mathbf{f},
\qquad
\frac{\partial \boldsymbol{\sigma}}{\partial t}
 = \lambda (\nabla \cdot \mathbf{v})\, \mathbf{I}
 + \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}} \right),
$$

with $\lambda$, $\mu$ and the models as in the displacement form. The
velocity components `v_x`, `v_z` (and `v_y`) are continuous nodal fields and
are recorded directly (`vx`, `vy`, `vz` are aliases). The stress components
(`sigma_xx`, `sigma_zz`, `sigma_xz` in 2-D; six in 3-D) are element-local:
each element keeps its own value at a shared node, so a material contrast on
an element edge is represented on both sides. The momentum equation is taken
in weak form with the $\rho$-weighted diagonal mass; the stress update uses
the velocity gradients at the GLL points. The free surface is the natural
traction-free condition. `ElasticVSSEM2DDeformed` also accepts
`free_top=True` (through `equation_options`), which sets
$\sigma_{nn} = \sigma_{nt} = 0$ on the top row of GLL points, in the local
tangent–normal frame, after every stress update.

`ElasticVTISEM2DDeformed` replaces the isotropic stress rate by that of a
transversely isotropic medium with a vertical symmetry axis (P-SV),

$$
\begin{aligned}
\partial_t \sigma_{xx} &= c_{11}\, \partial_x v_x + c_{13}\, \partial_z v_z, \\
\partial_t \sigma_{zz} &= c_{13}\, \partial_x v_x + c_{33}\, \partial_z v_z, \\
\partial_t \sigma_{xz} &= c_{55} \left( \partial_z v_x + \partial_x v_z \right),
\end{aligned}
$$

with the stiffnesses computed from the models `vp`, `vs` (vertical P and S
velocities), `rho`, `epsilon` and `delta` (Thomsen, 1986):

$$
c_{33} = \rho v_p^2, \quad
c_{55} = \rho v_s^2, \quad
c_{11} = c_{33} (1 + 2\epsilon), \quad
c_{13} = \sqrt{2\delta\, c_{33} (c_{33} - c_{55}) + (c_{33} - c_{55})^2}
 - c_{55}.
$$

A negative radicand is clamped to zero. With $\epsilon = \delta = 0$ the
class reduces to `ElasticVSSEM2DDeformed`.

`ElasticTTISEM2DDeformed` tilts the symmetry axis by the model `theta`
(radians) in the $(x, z)$ plane. The rotated stiffnesses add the couplings
$c_{15}$, $c_{35}$ of the normal stresses to the shear strain rate:

$$
\begin{aligned}
\partial_t \sigma_{xx} &= c_{11}\, \partial_x v_x + c_{13}\, \partial_z v_z
 + c_{15} \left( \partial_z v_x + \partial_x v_z \right), \\
\partial_t \sigma_{zz} &= c_{13}\, \partial_x v_x + c_{33}\, \partial_z v_z
 + c_{35} \left( \partial_z v_x + \partial_x v_z \right), \\
\partial_t \sigma_{xz} &= c_{15}\, \partial_x v_x + c_{35}\, \partial_z v_z
 + c_{55} \left( \partial_z v_x + \partial_x v_z \right),
\end{aligned}
$$

and `theta = 0` gives `ElasticVTISEM2DDeformed`.

In 3-D, `ElasticVTISEM3D` and `ElasticVTISEM3DDeformed` (models `vp`, `vs`,
`rho`, `epsilon`, `delta`, `gamma`) and `ElasticTTISEM3D` and
`ElasticTTISEM3DDeformed` (`theta` and `phi` added) replace the isotropic
stress rate of `ElasticVSSEM3D` by

$$
\partial_t \sigma_I = \sum_J c_{IJ}\, e_J,
\qquad
\mathbf{e} = \left( \partial_x v_x,\ \partial_y v_y,\ \partial_z v_z,\
 \partial_z v_y + \partial_y v_z,\ \partial_z v_x + \partial_x v_z,\
 \partial_y v_x + \partial_x v_y \right),
$$

with the Voigt order $I, J$ = $xx, yy, zz, yz, xz, xy$. The VTI stiffnesses
are those of the 2-D class with $c_{22} = c_{11}$, $c_{23} = c_{13}$,
$c_{44} = c_{55}$, $c_{66} = c_{55} (1 + 2\gamma)$ and
$c_{12} = c_{11} - 2 c_{66}$. The TTI classes rotate them (Bond
transformation) by the tilt `theta` from the vertical toward the azimuth
`phi` (radians), as the finite-difference
[`ElasticTTISG3D`](../equations/elastic_tti_sg3d.md) does, which fills all 21
stiffnesses; at `phi = 0` the tilt is that of the 2-D class. The time step is
limited by the fastest velocity, $v_p \sqrt{1 + 2\epsilon}$ for
$\epsilon > 0$.

### SH waves

`ElasticSHSEM2DDeformed` solves the antiplane (SH) equation for the
out-of-plane displacement $u_y$ (`u_y`) of a medium invariant along $y$,

$$
\rho \frac{\partial^2 u_y}{\partial t^2}
 = \frac{\partial}{\partial x} \left( \mu \frac{\partial u_y}{\partial x} \right)
 + \frac{\partial}{\partial z} \left( \mu \frac{\partial u_y}{\partial z} \right)
 + f_y,
\qquad \mu = \rho v_s^2,
$$

with the models `vs` and `rho`. This is the potential-form acoustic equation
with $1/\kappa$ read as $\rho$ and $1/\rho$ as $\mu$, and the class is
`AcousticPotentialSEM2D` built with those weights, with its operators and its
C-PML. Every side without an absorbing layer is traction-free,
$\mu\, \partial u_y / \partial n = 0$, the natural condition. The source is a
force along $y$; the receivers record `u_y` or its velocity `vy`.

### Fluid–solid coupling

`AcousticElasticSEM2D` solves the potential-form acoustic equation in the
fluid and the displacement-form elastic equation in the solid on one Quad9
mesh, with the models `vp`, `vs`, `rho`. An element is fluid if `vs` is zero
on all its GLL nodes and solid if it is positive on all of them; a mixed
element is refused. On the interface $\Gamma$, with $\mathbf{n}$ the unit
normal pointing out of the fluid, the normal displacement and the traction
are continuous:

$$
\frac{1}{\rho} \frac{\partial \chi}{\partial n} = \mathbf{u} \cdot \mathbf{n},
\qquad
\boldsymbol{\sigma} \mathbf{n} = -p\, \mathbf{n},
\qquad
p = -\frac{\partial^2 \chi}{\partial t^2};
$$

the tangential displacement is free. These enter the two weak forms as the
surface terms

$$
S^{\mathrm{ac}}_i
 = \oint_\Gamma (\mathbf{u} \cdot \mathbf{n})\, \phi_i \, d\Gamma,
\qquad
\mathbf{S}^{\mathrm{el}}_i = \oint_\Gamma p\, \mathbf{n}\, \phi_i \, d\Gamma,
$$

added to the fluid and the solid right-hand sides. Both use the 1-D GLL
quadrature and the normal of the fluid element, so that one is the transpose
of the other (Komatitsch et al., 2000). Within a step, $\ddot{\chi}$ is
computed first, with $S^{\mathrm{ac}}$ from the current $\mathbf{u}$, and the
solid then reads $p = -\ddot{\chi}$ at the same time level. A `free_surface`
side of the fluid is $\chi = 0$, which also takes precedence where the
interface meets the surface. The recorded fields are `chi`, `u_x`, `u_z`; `p`,
`vx`, `vz` are derived as in the uncoupled classes.

### Biot poroelasticity

`BiotVSSEM2DDeformed` solves the low-frequency Biot (1962) equations of the
finite-difference [`Biot`](../equations/biot.md), with solid velocity
$\mathbf{v}$ (`v_x`, `v_z`), Darcy flux $\mathbf{q}$ (`q_x`, `q_z`), total
stress $\boldsymbol{\sigma}$ (`sigma_xx`, `sigma_zz`, `sigma_xz`) and pore
pressure $p$ (`p`):

$$
\rho\, \partial_t \mathbf{v} + \rho_f\, \partial_t \mathbf{q}
 = \nabla \cdot \boldsymbol{\sigma},
\qquad
\rho_f\, \partial_t \mathbf{v} + m\, \partial_t \mathbf{q} + b\, \mathbf{q}
 = -\nabla p,
$$

$$
\partial_t \boldsymbol{\sigma}
 = (H - 2\mu) (\nabla \cdot \mathbf{v})\, \mathbf{I}
 + \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}} \right)
 + C (\nabla \cdot \mathbf{q})\, \mathbf{I},
\qquad
\partial_t p = -C\, \nabla \cdot \mathbf{v} - M\, \nabla \cdot \mathbf{q}.
$$

The models are the eight coefficients `[H, mu, C, M, rho, rho_f, m, b]` of
`sweep.equations.biot.biot_from_rock`, where $b = \eta / \kappa$ is the
fluid viscosity over the permeability.
`BiotAnisoPermSEM2DDeformed` replaces $b\,\mathbf{q}$ by $\mathbf{B}\mathbf{q}$
with the symmetric tensor
$\mathbf{B} = \begin{pmatrix} b_{xx} & b_{xz} \\ b_{xz} & b_{zz} \end{pmatrix}$
($x$ right, $z$ down), the models ending in `b_xx`, `b_xz`, `b_zz`.
`BiotVSSEM3DDeformed` solves the same equations in 3-D on a Hex27 mesh, with
`v_y`, `q_y` and six stress components added and an isotropic drag $b$.

The differences from the finite-difference class are in the discretisation.
$\mathbf{v}$ and $p$ are continuous nodal fields, $\boldsymbol{\sigma}$ and
$\mathbf{q}$ element-local. The pressure equation is taken in weak form, with
the total stress split as $\boldsymbol{\sigma} = \boldsymbol{\sigma}' - \alpha
p\, \mathbf{I}$, $\alpha = C / M$, and
$\partial_t \boldsymbol{\sigma}' = \lambda_d (\nabla \cdot \mathbf{v})\,
\mathbf{I} + \mu (\nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}})$,
$\lambda_d = H - 2\mu - C^2 / M$, which is the constitutive law above. At each
node the solid momentum and the element copies of $\mathbf{q}$ are coupled
through $\rho_f$; with the forces held fixed over the step, the drag term is
integrated exactly (one matrix exponential per node), so the time step is set
by the inviscid fast P wave for any permeability. The option
`viscous_q=(Q0, f0)` makes the drag a standard linear solid,
$b\, (1 + i\omega\tau_e) / (1 + i\omega\tau_s)$ with
$\tau_{e,s} = (\sqrt{Q_0^2 + 1} \pm 1) / (2\pi f_0 Q_0)$. An element with
`rho = 0` is a fluid ($\mathbf{q}$ is then the fluid velocity) and one with
`M = 0` a dry elastic solid. A `free_surface` side is open-pore:
traction-free with $p = 0$ on its nodes. A side with neither a free surface
nor an absorbing layer is traction-free and closed to flow
($\mathbf{q} \cdot \mathbf{n} = 0$). The receivers default to `vx`, `vz`; `p`,
`qx`, `qz`, `sxx`, `szz` and `sxz` are also available. Sources, interfaces
and validation are in the
[user guide](../../user-guide/sem.md#poroelastic-waves-biot).

### Attenuation

The visco classes add a generalized standard linear solid: $L$ = `n_sls`
mechanisms with relaxation times $\tau_{\sigma,l}$ log-spaced over `f_band`
and strengths $\tau_l(Q)$ fitted for a constant $Q$ there, as for the
finite-difference [`ViscoElastic`](../equations/visco_elastic.md) (see that
page for the fit). $Q = \infty$ gives the elastic step.

`ViscoElasticVSSEM2DDeformed`, `ViscoElasticVSSEM3D` and
`ViscoElasticVSSEM3DDeformed` (models `vp`, `vs`, `rho`, `Qp`, `Qs`) take the
rheology of the finite-difference class. With $\theta = \nabla \cdot
\mathbf{v}$, the unrelaxed and relaxed P moduli $\pi_U$, $\pi_R$, shear
moduli $\mu_U$, $\mu_R$, and $\tau^p_l = \tau_l(Q_p)$,
$\tau^s_l = \tau_l(Q_s)$, the 2-D normal stress is

$$
\partial_t \sigma_{xx}
 = \pi_U \theta - 2 \mu_U\, \partial_z v_z + \sum_l r_{xx,l},
\qquad
\partial_t r_{xx,l}
 = -\frac{1}{\tau_{\sigma,l}} \left( r_{xx,l} + \pi_R \tau^p_l \theta
 - 2 \mu_R \tau^s_l\, \partial_z v_z \right),
$$

and the other components follow in the same way (in 3-D, with
$\partial_y v_y + \partial_z v_z$ in place of $\partial_z v_z$), with
$\pi_U = \pi_R (1 + \sum_l \tau^p_l)$ and $\mu_U = \mu_R (1 + \sum_l
\tau^s_l)$. The memories $r$ are element-local, like the stress, and advance by
the trapezoidal rule. `vp` and `vs` are the phase velocities at `f_ref`.

`ViscoElasticSEM2DDeformed` (displacement form, models `vp`, `vs`, `rho`,
`Qkappa`, `Qmu`) is SPECFEM2D's attenuation: the bulk modulus
$\kappa = \lambda + \mu$ and the shear modulus relax through memories of the
volumetric strain, of $\partial_x u_x - \theta / 2$ and of the shear strain,
each updated as
$e_l^n = A_l^2 e_l^{n-1} + B_l (\varepsilon^n + A_l \varepsilon^{n-1})$ with
$A_l = e^{-\Delta t / (2 \tau_{\sigma,l})}$. Its defaults are SPECFEM2D's: the
band `f_ref / 10` to `10 f_ref` and `vp`, `vs` read as unrelaxed velocities.
`ViscoAcousticPotentialSEM2D` (models `vp`, `rho`, `Qkappa`) relaxes the bulk
modulus of the potential form with the same recursion on the node field
$\mathbf{M}^{-1}(-\mathbf{K}\chi)$. On these two classes `fit="nonlinear"`
fits the relaxation times together with the strengths for each $Q$
(SPECFEM2D's `USE_SOLVOPT`).

`ViscoBiotVSSEM2DDeformed` applies the velocity–stress GSLS to the drained
frame of `BiotVSSEM2DDeformed`: `Qp` attenuates $\pi_d = H - C^2/M$ and `Qs`
attenuates $\mu$, while the fluid's share $C^2/M$ of $H$ does not relax.
Validation against SPECFEM2D is in the
[user guide](../../user-guide/sem.md#visco-elastic-attenuation).

### Absorbing boundary

`PropSEM` adds `abcn` grid cells (whole elements; default ten elements) of
convolutional PML (C-PML) outside the model on every side without a free
surface, with the model edge-extended into it. All classes that have a C-PML
use the damping and frequency-shift profiles

$$
d(\ell) = -\frac{(n_p + 1)\, v_{\max} \ln R}{2L}
 \left( \frac{\ell}{L} \right)^{n_p},
\qquad
\alpha(\ell) = \pi f_0 \left( 1 - \frac{\ell}{L} \right),
$$

where $\ell$ is the distance into a layer of thickness $L$ = `abcn`
$\times$ `dh`, $n_p = 2$, $R = 10^{-3}$ (Collino and Tsogka, 2001),
$f_0$ is the `PropSEM` argument `f_dominant` (default 10 Hz) and $v_{\max}$
is the largest `vp` of the model (for the Biot classes, the fast P velocity of
the coefficients). The real part of the stretching function is one (no
coordinate scaling). Three schemes are used.

- The acoustic classes (with `ViscoAcousticPotentialSEM2D` and
  `ElasticSHSEM2DDeformed`) and the fluid of `AcousticElasticSEM2D` use the
  double-convolution form `cpmlr` of the finite-difference second-order
  equations (see [Absorbing boundaries](../equations/index.md#absorbing-boundaries);
  Komatitsch and Martin, 2007; Pasalic and McGarry, 2010): each term
  $\partial_{xx}$ of the Laplacian becomes
  $\partial_{\tilde{x}} ( \partial_{\tilde{x}} \cdot )$, with the memory
  variables `psi_x`, `zeta_x`, `psi_z`, `zeta_z` (and `psi_y`, `zeta_y` in
  3-D) on the global nodes and the coefficients
  $a = e^{-(d + \alpha)\Delta t}$, $b = d\, (a - 1) / (d + \alpha)$. In the
  potential form the memory acts on the flux $\rho^{-1} \nabla \chi$.
- The velocity–stress and Biot classes, like the finite-difference `cpmls`,
  replace every spatial derivative $\partial_x$ by $\partial_x + \psi_x$
  with the single recursion
  $\psi_x^{n} = a_x \psi_x^{n-1} + b_x\, \partial_x(\cdot)^{n}$ and the same
  $a$, $b$: on the velocity gradients in the stress update and on the
  assembled stress divergence, split by derivative direction, in the velocity
  update (Biot adds $\nabla \cdot \mathbf{q}$ and $\nabla p$). The outer edge
  of the layer is rigid, $\mathbf{v} = 0$. `PropSEM` refuses a layer that
  runs between two free faces of another axis: the solid between them is a
  plate, whose backward-propagating Lamb modes the layer amplifies.
- `ElasticSEM2DDeformed`, `ViscoElasticSEM2DDeformed` and the solid of
  `AcousticElasticSEM2D` use a port of
  SPECFEM2D's C-PML for the second-order displacement form (Xie et al., 2014):
  the $x$- and $z$-derivatives of $\mathbf{u}$ are convolved with separate
  coefficient sets, each with two memory variables, so that the stress
  divergence is not symmetric in the layer, and a damping term with two
  further memory variables is subtracted from the acceleration. The outer
  edge of the layer is rigid, $\mathbf{u} = 0$.

`ElasticSEM2D`, `ElasticSEM3D` and `ElasticSEM3DDeformed` have no C-PML:
`PropSEM` refuses a nonzero `abcn` for them, and every side is then
traction-free. For 3-D elastic modelling with an absorbing boundary use
`ElasticVSSEM3D` or `ElasticVSSEM3DDeformed`.

## References

- Biot, M. A., 1962, Mechanics of deformation and acoustic propagation in
  porous media: Journal of Applied Physics, 33(4), 1482–1498,
  [doi:10.1063/1.1728759](https://doi.org/10.1063/1.1728759).
- Collino, F., and C. Tsogka, 2001, Application of the perfectly matched
  absorbing layer model to the linear elastodynamic problem in anisotropic
  heterogeneous media: Geophysics, 66(1), 294–307,
  [doi:10.1190/1.1444908](https://doi.org/10.1190/1.1444908).
- Komatitsch, D., D. Barnes, and J. Tromp, 2000, Wave propagation near a
  fluid-solid interface: A spectral-element approach: Geophysics, 65(2),
  623–631, [doi:10.1190/1.1444758](https://doi.org/10.1190/1.1444758).
- Komatitsch, D., and R. Martin, 2007, An unsplit convolutional perfectly
  matched layer improved at grazing incidence for the seismic wave equation:
  Geophysics, 72(5), SM155–SM167,
  [doi:10.1190/1.2757586](https://doi.org/10.1190/1.2757586).
- Komatitsch, D., and J. Tromp, 1999, Introduction to the spectral element
  method for three-dimensional seismic wave propagation: Geophysical Journal
  International, 139(3), 806–822,
  [doi:10.1046/j.1365-246x.1999.00967.x](https://doi.org/10.1046/j.1365-246x.1999.00967.x).
- Komatitsch, D., and J.-P. Vilotte, 1998, The spectral element method: An
  efficient tool to simulate the seismic response of 2D and 3D geological
  structures: Bulletin of the Seismological Society of America, 88(2),
  368–392, [doi:10.1785/BSSA0880020368](https://doi.org/10.1785/BSSA0880020368).
- Pasalic, D., and R. McGarry, 2010, Convolutional perfectly matched layer for
  isotropic and anisotropic acoustic wave equations: SEG Technical Program
  Expanded Abstracts 2010, 2925–2929,
  [doi:10.1190/1.3513453](https://doi.org/10.1190/1.3513453).
- Thomsen, L., 1986, Weak elastic anisotropy: Geophysics, 51(10), 1954–1966,
  [doi:10.1190/1.1442051](https://doi.org/10.1190/1.1442051).
- Xie, Z., D. Komatitsch, R. Martin, and R. Matzen, 2014, Improved forward
  wave propagation and adjoint-based sensitivity kernel calculations using a
  numerically stable finite-element PML: Geophysical Journal International,
  198(3), 1714–1747, [doi:10.1093/gji/ggu219](https://doi.org/10.1093/gji/ggu219).

## Registry

Equations are discovered, not listed: every class with `field_specs` in a
module of `sweep.sem.equations` is one.

::: sweep.sem.equations.equation_classes
    options:
      heading_level: 3

::: sweep.sem.equations.summary
    options:
      heading_level: 3

::: sweep.sem.equations.describe
    options:
      heading_level: 3

## Acoustic

The potential form (SPECFEM's) is the FWI workhorse: shot batching, per-shot
models and batched boundary saving on `impl='c'` are there. Its `build` takes
a density model; `PropSEM` gives it the constant `density=`. The other four
take `vp` alone, one per mesh family. Under `PropSEM` the wavelet is the
pressure source on the potential form too: its `p` has the pressure form's
waveform for the same wavelet, and is what it records by default.

::: sweep.sem.AcousticPotentialSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.AcousticSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Elastic, displacement form

Second-order in time (SPECFEM's Newmark scheme); the source and receiver fields
are `u_x`, `u_z` (and `u_y`). `ElasticSEM2D`, `ElasticSEM3D` and
`ElasticSEM3DDeformed` model no C-PML: `PropSEM` refuses an `abcn` for them.

::: sweep.sem.ElasticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Elastic, velocity-stress form

First-order in time; the source and receiver fields are `v_x`, `v_z` (and
`v_y`). The stresses are element-local, so they take no receivers.

::: sweep.sem.ElasticVSSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVTISEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticTTISEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVTISEM3D
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ElasticVTISEM3DDeformed
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ElasticTTISEM3D
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ElasticTTISEM3DDeformed
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ElasticSHSEM2DDeformed
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ViscoElasticVSSEM2DDeformed
    options:
      heading_level: 3
      members: [build, unrelaxed_velocities]

::: sweep.sem.ViscoElasticVSSEM3D
    options:
      heading_level: 3
      members: [build, unrelaxed_velocities]

::: sweep.sem.ViscoElasticVSSEM3DDeformed
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ViscoElasticSEM2DDeformed
    options:
      heading_level: 3
      members: [build]

::: sweep.sem.ViscoAcousticPotentialSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVSSEM3D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ElasticVSSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Coupled fluid-solid

An acoustic potential in the water over an elastic solid, with a C-PML on
each; records `chi`, `u_x`, `u_z` and the derived `p`, `vx`, `vz` (PropSEM's
default: `p`, `vx`, `vz`).

::: sweep.sem.AcousticElasticSEM2D
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

## Poroelastic (Biot)

The FD `sweep.equations.Biot` on the spectral-element mesh: solid velocity
and pore pressure on the continuous nodes, Darcy flux and total stress in the
elements, with the same eight coefficients `[H, mu, C, M, rho, rho_f, m, b]`
and the same source and receiver names. See
[Poroelastic waves (Biot)](../../user-guide/sem.md#poroelastic-waves-biot).

::: sweep.sem.BiotVSSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.BiotAnisoPermSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.ViscoBiotVSSEM2DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

::: sweep.sem.BiotVSSEM3DDeformed
    options:
      heading_level: 3
      members: [build]
      inherited_members: true
      show_if_no_docstring: true

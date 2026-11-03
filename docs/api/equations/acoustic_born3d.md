# AcousticBorn3D

## Governing equation

`AcousticBorn3D` is the three-dimensional form of
[AcousticBorn](acoustic_born.md#governing-equation): the background field
$p_0$ (`h1`, the source field) and the scattered field $\delta p$ (`sh1`,
the receiver field) obey

$$
\frac{\partial^2 p_0}{\partial t^2} = v_p^2\, \nabla^2 p_0,
\qquad
\frac{\partial^2 \delta p}{\partial t^2}
 = v_p^2\, \nabla^2 \delta p + m_p\, v_p^2\, \nabla^2 p_0,
$$

with $\nabla^2 = \partial_{xx} + \partial_{yy} + \partial_{zz}$, $v_p$ the
background model `vp` and $m_p = \delta(v_p^2)/v_p^2$ the model `mp`. The
discretisation is that of [Acoustic3D](acoustic3d.md), and the wavelet sample
is added to $p_0^{n+1}$ only.

## Boundary conditions

The default `pml_type="cpmlr"` is applied to both fields with separate memory
fields (`psi*`, `zeta*` for $p_0$; `spsi*`, `szeta*` for $\delta p$, on the
$x$, $y$ and $z$ axes); see
[Absorbing boundaries](index.md#absorbing-boundaries). With
`free_surface=True` the top face is pressure-release for both fields.

::: sweep.equations.AcousticBorn3D

Term III of the velocity gradient splits out the same way as in 2-D
(`SWEEP_LSRTM_SPLIT_III=1`, `sweep.propagator.last_grad_split_iii()`): see
[AcousticBorn](acoustic_born.md#term-iii-on-its-own).

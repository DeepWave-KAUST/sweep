# ElasticBorn

## Governing equations

`ElasticBorn` splits the model into a background $(v_p, v_s, \rho)$, the
models `vp`, `vs`, `rho`, and a perturbation
$(\delta v_p, \delta v_s, \delta\rho)$, the models `dvp`, `dvs`, `drho`. The
perturbations are absolute and carry the units of the background. The
background field $(v_x, v_z, \sigma_{xx}, \sigma_{zz}, \sigma_{xz})$ obeys the
[Elastic](elastic.md) equations in the background model. The scattered field
$(\delta v_x, \delta v_z, \delta\sigma_{xx}, \delta\sigma_{zz},
\delta\sigma_{xz})$, the fields `svx`, `svz`, `ssxx`, `sszz`, `ssxz`, obeys
the same equations with the perturbation acting on the background field as
the source (Li et al., 2019, here in velocity–stress form):

$$
\begin{aligned}
\rho \frac{\partial \delta v_x}{\partial t}
 &= \frac{\partial \delta\sigma_{xx}}{\partial x}
  + \frac{\partial \delta\sigma_{xz}}{\partial z}
  - \delta\rho \frac{\partial v_x}{\partial t},
\\
\rho \frac{\partial \delta v_z}{\partial t}
 &= \frac{\partial \delta\sigma_{xz}}{\partial x}
  + \frac{\partial \delta\sigma_{zz}}{\partial z}
  - \delta\rho \frac{\partial v_z}{\partial t},
\\
\frac{\partial \delta\sigma_{xx}}{\partial t}
 &= (\lambda + 2\mu) \frac{\partial \delta v_x}{\partial x}
  + \lambda \frac{\partial \delta v_z}{\partial z}
  + (\delta\lambda + 2\delta\mu) \frac{\partial v_x}{\partial x}
  + \delta\lambda \frac{\partial v_z}{\partial z},
\\
\frac{\partial \delta\sigma_{zz}}{\partial t}
 &= \lambda \frac{\partial \delta v_x}{\partial x}
  + (\lambda + 2\mu) \frac{\partial \delta v_z}{\partial z}
  + \delta\lambda \frac{\partial v_x}{\partial x}
  + (\delta\lambda + 2\delta\mu) \frac{\partial v_z}{\partial z},
\\
\frac{\partial \delta\sigma_{xz}}{\partial t}
 &= \mu \left( \frac{\partial \delta v_x}{\partial z}
             + \frac{\partial \delta v_z}{\partial x} \right)
  + \delta\mu \left( \frac{\partial v_x}{\partial z}
                   + \frac{\partial v_z}{\partial x} \right).
\end{aligned}
$$

The background Lamé parameters are $\lambda = \rho (v_p^2 - 2 v_s^2)$ and
$\mu = \rho v_s^2$, and their perturbations are the linearisation of these
relations at the background:

$$
\delta\lambda = \delta\rho \left( v_p^2 - 2 v_s^2 \right)
 + 2\rho \left( v_p\, \delta v_p - 2 v_s\, \delta v_s \right),
\qquad
\delta\mu = \delta\rho\, v_s^2 + 2 \rho\, v_s\, \delta v_s .
$$

Both fields use the staggered grid, stencil and leapfrog of
[Elastic](elastic.md), and each carries its own ten C-PML memory variables.
The density term is discretised as
$-(\delta\rho/\rho)\,(v_x^{n+1/2} - v_x^{n-1/2})$ in the velocity update, and
the stress source uses the same C-PML-corrected background velocity
derivatives as the background stress update. The scattered step is therefore
the exact derivative of the discrete Elastic step, and the scattered record
equals the directional derivative of the Elastic record along
$(\delta v_p, \delta v_s, \delta\rho)$.

The wavelet is injected into the background fields only (default `sxx` and
`szz`). The default receiver fields are the scattered velocities `svx` and
`svz`; the background fields `vx`, `vz`, `sxx`, `szz` and the scattered
stresses `ssxx`, `sszz` can also be recorded.

## Boundary conditions

The default `pml_type="cpmls"` applies the staggered-grid C-PML of
[Elastic](elastic.md) to both fields; see
[Absorbing boundaries](index.md#absorbing-boundaries). With `free_surface`
(per edge in 2-D, or with `topography`) both fields satisfy the image-method
conditions of [Elastic](elastic.md). On the surface row the update of
$\delta\sigma_{xx}$ is the linearisation of the modified surface update,

$$
\frac{\partial \delta\sigma_{xx}}{\partial t}
 = c\, \frac{\partial \delta v_x}{\partial x}
 + \delta c\, \frac{\partial v_x}{\partial x},
\qquad
c = \frac{4\mu (\lambda + \mu)}{\lambda + 2\mu},
\qquad
\delta c = \frac{4\mu^2\, \delta\lambda
 + 4\left(\lambda^2 + 2\lambda\mu + 2\mu^2\right) \delta\mu}
 {(\lambda + 2\mu)^2},
$$

and on an $x$ face the same holds for $\delta\sigma_{zz}$ with $x$ and $z$
exchanged.

## References

- Li, Y., Q. Guo, Z. Li, and T. Alkhalifah, 2019, Elastic reflection waveform
  inversion with variable density: Geophysics, 84(4), R553–R567,
  [doi:10.1190/geo2017-0722.1](https://doi.org/10.1190/geo2017-0722.1).

::: sweep.equations.ElasticBorn

## Gradients

| gradient of | `impl='eager'` | `impl='c'` |
|---|---|---|
| `dvp`, `dvs`, `drho` (elastic LSRTM) | autograd | every memory mode (`Full`, `BoundarySaving`, `Ckpt`) |
| `vp`, `vs`, `rho` (reflection waveform inversion) | autograd | refused |
| wavelet | autograd | refused |

On `impl='c'` the backward is the [Elastic](elastic.md) one: the adjoint of the
scattered field obeys the background elastic equation, and its imaging
condition against the background field is exactly the gradient of the
perturbation. A background receiver (`vx`, ..., recorded next to the scattered
`svx`, ...) does not depend on the perturbation, so its residual adds nothing
to these gradients.

```python
from sweep.equations import ElasticBorn
from sweep.propagator.torch import PropTorch

eq = ElasticBorn(spatial_order=4, device="cuda")
prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, abcn=40, free_surface=True)
dvp, dvs, drho = (torch.zeros_like(vp).requires_grad_() for _ in range(3))
d_sc = prop(wavelet, sources, receivers, models=[vp, vs, rho, dvp, dvs, drho])
loss = ((d_sc - d_obs) ** 2).sum()
loss.backward()   # dvp.grad, dvs.grad, drho.grad
```

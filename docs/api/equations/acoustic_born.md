# AcousticBorn

## Governing equation

`AcousticBorn` propagates two coupled constant-density acoustic fields, a
background field $p_0$ and a scattered field $\delta p$:

$$
\frac{\partial^2 p_0}{\partial t^2} = v_p^2\, \nabla^2 p_0,
\qquad
\frac{\partial^2 \delta p}{\partial t^2}
 = v_p^2\, \nabla^2 \delta p + m_p\, v_p^2\, \nabla^2 p_0,
$$

where $p_0$ is the wavefield `h1` (the source field), $\delta p$ is
`sh1` (the receiver field), $v_p$ is the background model `vp` and
$m_p$ is the model `mp`. The second equation is the first-order (Born)
perturbation of the first for $v_p^2 \to v_p^2 (1 + m_p)$, so $m_p$ is the
relative perturbation of the squared velocity, $m_p = \delta(v_p^2)/v_p^2$.
The scattering source $m_p v_p^2 \nabla^2 p_0$ is evaluated from $p_0^{n}$
in the same step, with the same discrete Laplacian that advances $p_0$.
Both fields use the time and space discretisation of
[Acoustic](acoustic.md). The wavelet sample is added to $p_0^{n+1}$ at the
source cell after each step; it does not enter $\delta p$ directly.

## Boundary conditions

The default `pml_type="cpmlr"` is applied to both fields with separate memory
fields, `psix`, `zetax`, `psiz`, `zetaz` for $p_0$ and `spsix`, `szetax`,
`spsiz`, `szetaz` for $\delta p$; see
[Absorbing boundaries](index.md#absorbing-boundaries). Inside the absorbing
layers the scattering source uses the stretched Laplacian of $p_0$. With
`free_surface=True` the top face is pressure-release for both fields.

## References

- Wu, Z., and T. Alkhalifah, 2015, Simultaneous inversion of the background
  velocity and the perturbation in full-waveform inversion: Geophysics,
  80(6), R317–R329,
  [doi:10.1190/geo2014-0365.1](https://doi.org/10.1190/geo2014-0365.1).
  The `impl='c'` gradient of `vp` and the term-III split below follow their
  decomposition.

::: sweep.equations.AcousticBorn

## Term III on its own

Wu & Alkhalifah (2015, eq. 18-20) weight the image-point term III of the RWI
velocity gradient separately: `(II+IV) + beta*III`. Set
`SWEEP_LSRTM_SPLIT_III=1` and the `impl='c'` full or boundary-saving backward
of `AcousticBorn` / `AcousticBorn3D` puts II+IV in `vp.grad` and keeps III
aside, on the same grid:

```python
import os
from sweep.propagator import last_grad_split_iii

os.environ["SWEEP_LSRTM_SPLIT_III"] = "1"   # read by every backward
loss.backward()
grad_vp = vp.grad + beta * last_grad_split_iii()
```

Without the switch `vp.grad` holds II+III+IV. The checkpointing modes return
no vp gradient, split or not. `last_grad_split_iii()` returns `None` until a
split backward has run, and afterwards the III of the most recent one.
[Notebook 32](../../notebooks/32_rwi_acoustic_vs_lsrtm_gradient.ipynb) shows
the split on the paper's Figure 1 setup.

::: sweep.propagator.last_grad_split_iii
    options:
      heading_level: 3

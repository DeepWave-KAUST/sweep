# ElasticBorn

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

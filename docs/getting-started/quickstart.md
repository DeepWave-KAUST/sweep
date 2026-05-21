# Quick Start

The smallest useful SWEEP workflow is **two objects, four arguments**:

```python
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch

equation = Acoustic()
solver   = PropTorch(equation, shape=(96, 128), dh=10.0, dt=0.002)
```

Every other knob — PML width, PML type, free surface, checkpointing, source/receiver
field selection — has a sensible default. You only touch them when you need to.

This page walks through running that solver end-to-end (forward + gradient) on a
two-layer model, then points to the runnable example that does the same thing in
a notebook.

> Prefer to skim a notebook?
> [`examples/notebooks/00_hello_forward.ipynb`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/notebooks/00_hello_forward.ipynb)
> is the executable version of this quick start.

## 1. Imports and a velocity model

```python
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

shape = (96, 128)              # (nz, nx) interior grid; PML is added automatically
dh, dt, nt = 10.0, 0.002, 600  # m, s, samples
freq, delay = 10.0, 0.12       # Ricker dominant frequency, time shift (s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vp_np = np.full(shape, 1500.0, dtype=np.float32)
vp_np[shape[0] // 2:, :] = 2500.0
```

## 2. Build the equation and the solver

```python
equation = Acoustic(device=device)
solver   = PropTorch(equation, shape=shape, dh=dh, dt=dt)
```

That is the whole solver. The defaults you are silently accepting:

| argument        | default      | when to change                              |
| --------------- | ------------ | ------------------------------------------- |
| `spatial_order` | `4`          | bump to `8` for stricter dispersion control |
| `abcn`          | `50`         | PML width in grid points                    |
| `pml_type`      | `'cpmlr'`    | `'cpmls'` for stricter equations            |
| `free_surface`  | `False`      | `True` for land/free-surface acquisitions   |
| `use_ckpt`      | `True`       | turn off for tiny problems with lots of RAM |
| `source_type`   | inherited    | inject into a non-default field             |
| `receiver_type` | inherited    | sample a non-default field                  |
| `impl`          | `'eager'`    | `'c'` for the compiled CUDA backend         |

See the [Propagators user guide](../user-guide/propagators.md) for the full list.

## 3. Wavelet, sources, receivers

```python
t = np.arange(nt, dtype=np.float32) * dt
wavelet = ricker(t - delay, f=freq)

sources   = np.array([[shape[1] // 2, 2]], dtype=np.int64)       # [nshots, ndim] in (x, z)
rec_x     = np.linspace(0, shape[1] - 1, 64, dtype=np.int64)
receivers = np.stack([rec_x, np.full_like(rec_x, 4)], axis=1)[None]  # [nshots, nrec, ndim]
```

Shapes:

- `wavelet`: `(nt,)`
- `sources`: `(nshots, ndim)`
- `receivers`: `(nshots, nreceivers, ndim)`

## 4. Forward modeling

```python
vp = torch.from_numpy(vp_np).to(device).requires_grad_(True)
obs = solver(wavelet, sources, receivers, models=[vp])
```

`obs` has shape `(nshots, nt, nreceivers, len(receiver_type))`. For acoustic with
the default receiver field that trailing dimension is `1`.

## 5. Backward propagation

```python
loss = obs.pow(2).sum()
loss.backward()

grad = vp.grad.detach().cpu().numpy()
```

`grad` is the gradient of the scalar loss with respect to `vp` — the building
block for FWI, LSRTM, and source-encoded inversion.

## Optional — inspect the equation before instantiating

Sometimes you want to know which fields and models an equation exposes before
you start writing code:

```python
print([f.name for f in Acoustic.available_fields(role="source")])
print([f.name for f in Acoustic.available_fields(role="receiver")])
print([m.name for m in Acoustic.available_models()])
print(Acoustic.describe_field("pressure"))
```

These are class methods and work without a device or shape.

## Full minimal example

```python
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

shape, dh, dt, nt = (96, 128), 10.0, 0.002, 600
freq, delay = 10.0, 0.12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vp_np = np.full(shape, 1500.0, dtype=np.float32)
vp_np[shape[0] // 2:, :] = 2500.0

equation = Acoustic(device=device)
solver   = PropTorch(equation, shape=shape, dh=dh, dt=dt)

t = np.arange(nt, dtype=np.float32) * dt
wavelet = ricker(t - delay, f=freq)
sources   = np.array([[shape[1] // 2, 2]], dtype=np.int64)
rec_x     = np.linspace(0, shape[1] - 1, 64, dtype=np.int64)
receivers = np.stack([rec_x, np.full_like(rec_x, 4)], axis=1)[None]

vp = torch.from_numpy(vp_np).to(device).requires_grad_(True)
obs = solver(wavelet, sources, receivers, models=[vp])
loss = obs.pow(2).sum()
loss.backward()

print("obs shape :", tuple(obs.shape))
print("loss      :", float(loss.detach().cpu()))
print("grad shape:", tuple(vp.grad.shape))
```

## Next steps

- [Installation](installation.md) — pick the right backend for your hardware.
- [Backends](../user-guide/backends.md) — Torch eager vs. compiled CUDA vs. JAX.
- [Examples](../examples/index.md) — runnable notebooks and scripts for FWI,
  LSRTM, wavefields, anisotropic media, multi-GPU, and memory savings.

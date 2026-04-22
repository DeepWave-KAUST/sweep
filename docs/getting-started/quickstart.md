# Quick Start

This page walks through the smallest useful SWEEP workflow step by step.

The goal is simple:

1. create a tiny 2D velocity model
2. build one acoustic solver
3. run forward modeling
4. run backward propagation and get a gradient

This quick start uses the simplest path:

- equation: `Acoustic`
- propagator: `PropTorch`
- backend: `eager`

If you want CUDA binding or JAX after this, see the User Guide and Examples.

## Step 1. Import the minimal pieces

```python
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker
```

These are the only SWEEP objects you need for the smallest Torch example:

- `Acoustic`: defines the wave equation
- `PropTorch`: runs the equation
- `ricker`: builds a source wavelet

## Step 2. Define a small model and basic settings

```python
nt = 750
dt = 0.002
dh = 10.0
delay = 0.1
fm = 8.0
shape = (100, 100)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vp_np = np.full(shape, 1500.0, dtype=np.float32)
vp_np[50:, :] = 2000.0
```

This creates:

- a `100 x 100` 2D velocity model
- a two-layer Earth model
- a Torch device that uses GPU if available

## Step 3. Build the equation

```python
equation = Acoustic(
    spatial_order=8,
    device=dev,
    backend="torch",
)
```

At this stage you are only defining the physics. Nothing has run yet.

## Step 4. Query available source and receiver fields

You can ask the equation class which user-facing fields are valid for source
injection and receiver sampling before you even instantiate it:

```python
print([field.name for field in Acoustic.available_fields()])
print([field.name for field in Acoustic.available_fields(role="source")])
print([field.name for field in Acoustic.available_fields(role="receiver")])
print(Acoustic.describe_field("pressure"))
```

After instantiation, the same queries also work on the object:

```python
print([field.name for field in equation.available_fields()])
print(equation.describe_field("p"))
```

For this acoustic example, the main user-facing field is the pressure-like
field `h1`, which also accepts aliases such as `pressure` and `p`.

`available_fields()` filters out internal boundary variables by default, so you
do not need to look through CPML helper fields when choosing
`source_type` or `receiver_type`.

## Step 5. Build the solver

```python
solver = PropTorch(
    equation,
    shape=shape,
    dev=dev,
    dh=dh,
    dt=dt,
    source_type=["h1"],
    receiver_type=["h1"],
    abcn=20,
    free_surface=False,
    pml_type="cpmlr",
    backend="eager",
)
```

The most important arguments are:

- `shape`: model size
- `dh`: grid spacing
- `dt`: time step
- `source_type`: which wavefield gets the source injection
- `receiver_type`: which wavefield gets sampled at receivers

For a first example, you can treat `source_type=["h1"]` and
`receiver_type=["h1"]` as the standard acoustic setup.

## Step 6. Create the wavelet

```python
t = np.arange(nt, dtype=np.float32) * dt
wave = ricker(t - delay, f=fm).astype(np.float32)
```

`wave` has shape `(nt,)`.

## Step 7. Define one source and one receiver line

```python
sources = np.array([[50, 2]], dtype=np.int32)
receivers = np.array([[[ix, 2] for ix in range(10, 90)]], dtype=np.int32)
```

The expected shapes are:

- `sources`: `(nshots, ndim)`
- `receivers`: `(nshots, nreceivers, ndim)`

Here that means:

- one shot
- a 2D model, so coordinates are `(x, z)` in grid indices
- one receiver line for that shot

## Step 8. Prepare the model tensor

```python
vp = torch.from_numpy(vp_np).to(dev).requires_grad_(True)
```

`models=[vp]` is how you pass the model list required by `Acoustic`.

## Step 9. Run forward modeling

```python
obs = solver(wave, sources, receivers, models=[vp])
```

`obs` is the simulated receiver data.

For this example, the output shape is approximately:

```python
(nshots, nt, nreceivers, 1)
```

depending on the backend and equation details.

## Step 10. Run backward propagation and get a gradient

```python
loss = obs.pow(2).sum()
loss.backward()

grad = vp.grad.detach().cpu().numpy()
```

Now `grad` is the gradient of the scalar loss with respect to the velocity
model.

## Step 11. Full minimal example

```python
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

nt = 750
dt = 0.002
dh = 10.0
delay = 0.1
fm = 8.0
shape = (100, 100)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vp_np = np.full(shape, 1500.0, dtype=np.float32)
vp_np[50:, :] = 2000.0

equation = Acoustic(
    spatial_order=8,
    device=dev,
    backend="torch",
)

solver = PropTorch(
    equation,
    shape=shape,
    dev=dev,
    dh=dh,
    dt=dt,
    source_type=["h1"],
    receiver_type=["h1"],
    abcn=20,
    free_surface=False,
    pml_type="cpmlr",
    backend="eager",
)

t = np.arange(nt, dtype=np.float32) * dt
wave = ricker(t - delay, f=fm).astype(np.float32)

sources = np.array([[50, 2]], dtype=np.int32)
receivers = np.array([[[ix, 2] for ix in range(10, 90)]], dtype=np.int32)

vp = torch.from_numpy(vp_np).to(dev).requires_grad_(True)

obs = solver(wave, sources, receivers, models=[vp])

loss = obs.pow(2).sum()
loss.backward()

print("obs shape:", tuple(obs.shape))
print("loss:", float(loss.detach().cpu()))
print("grad shape:", tuple(vp.grad.shape))
```

## Step 12. What to remember

Every SWEEP run follows the same pattern:

1. choose an equation
2. build a propagator
3. create `wave`, `sources`, `receivers`, and `models`
4. call `solver(...)`

The minimal call is always conceptually:

```python
obs = solver(wave, sources, receivers, models=[...])
```

## Next steps

- For installation choices, see [Installation](installation.md).
- For backend differences, see [Backends](../user-guide/backends.md).
- For complete runnable scripts, see [Examples](../examples/index.md).

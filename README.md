# SWEEP
Seismic Wave Equation Exploration Platform (SWEEP) is a Python package for seismic wave-equation modeling, migration, and inversion.

Recent interface updates:

- lazy imports are supported, so you only need to install the backend you plan to use
- `PropTorch` is now the main Torch-family entry point, with `backend="eager"` or `backend="cuda"`
- backend-specific options are grouped through `EagerOptions`, `CUDAOptions`, `MemoryOptions`, `BoundaryOptions`, and `CkptOptions`
- examples are reorganized by task family under `examples/`, including new `wavefields/` and `reducingmemory/` groups

## Installation

Install SWEEP from the repository root. If you have not downloaded the source code yet, clone the repository first and change into the project directory:

```bash
git clone <repository-url>
cd geophyai
```

### Pytorch/Jax only
This is the simplest way to install SWEEP, but it may not be the most efficient way to use it. If you only want to use sweep with PyTorch or JAX APIs, you can install the package with the following command:
```bash
pip install .
```

### Pytorch Bindings
Faster, recommended, but may take more time to install due to the need to compile CUDA kernels. If you have a compatible NVIDIA GPU and CUDA toolkit installed, you can install the PyTorch bindings with CUDA support using the following command:
```bash
SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
```

## Usage
The following example shows how to compute the gradient of a toy model with respect to the velocity model.
```python
import torch
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import EagerOptions
from sweep.equations import Acoustic
from sweep.signal import ricker
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
nt = 1500
dt = 0.002
dh = 10
delay = 0.1
fm = 5
spatial_order = 8
shape = (100,100)

# Device
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create a 2-layer model
true_model = np.ones(shape, dtype=np.float32)*1500
true_model[50:, :] = 2000

# Create a model
model = PropTorch(Acoustic(spatial_order=spatial_order, device=dev, backend='torch'), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            pml_type='cpmlr',
            free_surface=False,
            backend="eager",
            eager_options=EagerOptions(use_compile=False))
            
# Set the model parameters (Pytorch)
vp = torch.from_numpy(true_model).to(dev).requires_grad_(True)
# Create a wavelet
t = np.arange(0, int(nt//2)*dt, dt)
wave = ricker(t-delay, f=fm)

# Acquicition geometry
sources = np.array([[1, 1]]) # in grid, shape=(nshots, 2)
receivers = np.array([[[99, 1]]]) # in grid, shape=(nshots, nreceivers, 2)

# Forward modeling
# Backward propagation (Pytorch)
obs = model.forward(wave, sources, receivers, models=[vp])
obs.pow(2).sum().backward()
# Show the results
fig, axes=plt.subplots(1,3, figsize=(12,3))

axes[0].imshow(true_model, cmap='seismic', aspect='auto')
axes[0].set_title('True model')
axes[1].plot(obs.detach().cpu().numpy().squeeze(), label='Observed data')
grad = vp.grad.detach().cpu().numpy() # Pytorch
vmin,vmax=np.percentile(grad, [1,99])
axes[1].set_title('Observed data')
axes[2].imshow(grad, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[2].set_title('Gradient of vp')
fig.tight_layout()
fig.savefig('grad_vp.png', dpi=300, bbox_inches='tight')
plt.close()
```
The ground truth model, observed data, and the gradient of the velocity model are shown below.
![grad_vp](figures/grad_vp.png)

# Examples
Examples are organized by task family under `examples/`:

- `FWI/`
- `LSRTM/`
- `wavefields/`
- `reducingmemory/`
- `multi-gpu/`

See [examples/README.md](examples/README.md) for the current layout.

## License
This project is licensed under the MIT License.

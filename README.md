# SWEEP

[![Docs](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://deepwave-kaust.github.io/sweep/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![JAX](https://img.shields.io/badge/JAX-supported-9cf)](https://github.com/google/jax)
[![GitHub stars](https://img.shields.io/github/stars/DeepWave-KAUST/sweep?style=social)](https://github.com/DeepWave-KAUST/sweep/stargazers)

English | [中文](README.zh-CN.md)

**Seismic Wave Equation Exploration Platform (SWEEP)** is a Python framework for differentiable seismic wave-equation modeling, migration, and full-waveform inversion.

> One API, many physics: switch between PyTorch / JAX backends, eager / compiled CUDA implementations, and 20+ wave equations including acoustic, elastic, anisotropic (TTI / VTI), and DAS.

Documentation: <https://deepwave-kaust.github.io/sweep/>

Recent interface updates:

- lazy imports are supported, so you only need to install the backend you plan to use
- `backend` now names the array/programming backend (`torch` or `jax`); Torch execution chooses `impl="eager"` for PyTorch operators or `impl="c"` for the compiled C++/CUDA extension
- backend-specific options are grouped through `EagerOptions`, `CUDAOptions`, `MemoryOptions`, `BoundaryOptions`, and `CkptOptions`
- examples are reorganized by task family under `examples/`, including new `wavefields/` and `reducingmemory/` groups

## Installation

Install SWEEP from the repository root. If you have not downloaded the source code yet, clone the repository first and change into the project directory:

```bash
git clone git@github.com:DeepWave-KAUST/sweep.git
cd sweep
```

### PyTorch/JAX only
This is the simplest way to install SWEEP, but it may not be the most efficient way to use it. If you only want to use sweep with PyTorch or JAX APIs, you can install the package with the following command:
```bash
pip install .
```

### C++/CUDA bindings
Faster, recommended, but may take more time to install because it compiles the C++ CPU kernels and CUDA kernels. If you have PyTorch and a compatible CUDA toolkit installed, build the compiled extension with:

```bash
SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
```

If CUDA architectures cannot be detected automatically during installation, set them explicitly before running the install command. For example:

```bash
# V100
export TORCH_CUDA_ARCH_LIST="7.0"

# A100
export TORCH_CUDA_ARCH_LIST="8.0"

# Build one wheel that supports both V100 and A100
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
```

If the CUDA toolkit you want is not the default one on the system, also set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

`SWEEP_CUDA_ARCH_LIST` can be used as a package-specific alias for `TORCH_CUDA_ARCH_LIST`. If neither variable is set, the build defaults to V100-compatible `7.0` to avoid PyTorch's empty architecture auto-detection on CPU-only build nodes.

## Backends, Implementations, and Devices

Use `backend` for the array/programming backend, `impl` for the implementation under that backend, and `device` for CPU/CUDA placement.

| Backend | Impl | Device | Implementation |
| --- | --- | --- | --- |
| `backend="torch"` | `impl="eager"` | CPU or CUDA | PyTorch eager operators and PyTorch autograd |
| `backend="torch"` | `impl="c"` | CPU | Torch C++ extension with compiled CPU kernels |
| `backend="torch"` | `impl="c"` | CUDA | Torch C++ extension with compiled CUDA kernels |
| `backend="jax"` | equation-specific | CPU or CUDA | JAX implementation |

In the FWI example scripts:

```bash
# PyTorch eager on GPU
python examples/FWI/2d/acoustic/torch/fwi_marmousi_gpu.py --impl eager

# compiled CUDA kernels
python examples/FWI/2d/acoustic/torch/fwi_marmousi_gpu.py --impl c

# compiled C++ CPU kernels
python examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py --impl c

# PyTorch eager CPU kernels
python examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py --impl eager

# compiled C++ CPU kernels with MPI shot parallelism
mpirun -np 4 python examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py \
    --impl c --mpi --mpi-forward-batchsize 4

# PyTorch eager CPU with MPI shot parallelism
mpirun -np 4 python examples/FWI/2d/acoustic/torch/fwi_marmousi_cpu_mpi.py \
    --impl eager --mpi --mpi-forward-batchsize 4
```

The acoustic Marmousi Torch example is split into a GPU entry point and a CPU/MPI entry point. Other updated examples still accept older backend aliases such as `--backend pytorch`, `--backend eager`, `--backend c`, `--backend cuda`, and `--backend cpu` for compatibility.

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
            backend="torch",
            impl="eager",
            eager_options=EagerOptions(use_compile=False))
            
# Set the model parameters (Pytorch)
vp = torch.from_numpy(true_model).to(dev).requires_grad_(True)
# Create a wavelet
t = np.arange(0, int(nt//2)*dt, dt)
wave = ricker(t-delay, f=fm)

# Acquisition geometry
sources = np.array([[[1, 1]]], dtype=np.int32) # in grid, shape=(nshots, nsources, 2)
receivers = np.array([[[99, 1]]], dtype=np.int32) # in grid, shape=(nshots, nreceivers, 2)

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

## License
This project is licensed under the MIT License.

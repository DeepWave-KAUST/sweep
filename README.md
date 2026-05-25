<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo/sweep-icon-dark.svg">
    <img src="docs/assets/logo/sweep-icon-light.svg" alt="SWEEP" width="180">
  </picture>
</p>

<h1 align="center">SWEEP</h1>

<p align="center">
  <a href="https://deepwave-kaust.github.io/sweep/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white"></a>
</p>

<p align="center">English | <a href="README.zh-CN.md">中文</a></p>

**Seismic Wave Equation Exploration Platform** — a differentiable framework for seismic wave-equation modeling, migration, and full-waveform inversion. One API, 20+ equations (acoustic / elastic / VTI / TTI / DAS), PyTorch and JAX backends, eager and compiled CUDA paths.

📖 **Documentation**: <https://deepwave-kaust.github.io/sweep/>

## Install

```bash
# Python-only (PyTorch / JAX through the pure-Python path)
pip install .

# With the compiled C++ / CUDA extension (recommended for production)
SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation
```

If the build can't auto-detect your GPU, set `TORCH_CUDA_ARCH_LIST` (e.g. `"7.0"` for V100, `"8.0"` for A100, `"8.9"` for RTX 6000 Ada) before the second command. Full install notes — including the GPU-only fast build and multi-CUDA setups — are in [the docs](https://deepwave-kaust.github.io/sweep/getting-started/installation/).

## Hello SWEEP

One shot, one receiver, one `.backward()` — read off the velocity-model gradient for a single trace:

```python
import numpy as np
import torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

shape = (96, 128)
dh, dt, nt = 10.0, 0.002, 800
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vp_true = np.full(shape, 1500.0, dtype=np.float32)
vp_true[shape[0] // 2:, :] = 2500.0
vp_init = np.full(shape, 1500.0, dtype=np.float32)

solver = PropTorch(Acoustic(device=device), shape=shape, dh=dh, dt=dt,
                   dev=device, pml_type="cpmlr", use_ckpt=False)

t = np.arange(nt) * dt
wavelet = ricker(t - 0.14, f=10.0).astype(np.float32)
sources   = np.array([[shape[1] // 4, shape[0] // 2]], dtype=np.int64)
receivers = np.array([[[3 * shape[1] // 4, shape[0] // 2]]], dtype=np.int64)

with torch.no_grad():
    obs = solver(wavelet, sources, receivers, models=[torch.tensor(vp_true, device=device)])

vp_t = torch.tensor(vp_init, device=device, requires_grad=True)
pred = solver(wavelet, sources, receivers, models=[vp_t])
(0.5 * (pred - obs).pow(2).sum()).backward()

print("vp gradient shape:", tuple(vp_t.grad.shape))
```

Swap `Acoustic` for `Elastic`, `AcousticVTI`, `ElasticTTI`, ... — the surrounding code is unchanged.

## Notebooks & examples

- **Hello SWEEP** — forward / backward / 5-line FWI loop: [`examples/notebooks/00_hello_fwi.ipynb`](examples/notebooks/00_hello_fwi.ipynb)
- **FWI on Marmousi** (acoustic / elastic / multiscale): see [`examples/notebooks/01_*`–`03_*`](examples/notebooks/)
- **Wavefields, DAS, anisotropic, RTM**: [`examples/notebooks/04_*`–`08_*`](examples/notebooks/)
- **Production scripts** (multi-GPU, MPI shot parallelism, multi-shot batching): under [`examples/`](examples/)

## Citing

```bibtex
@misc{wang2026sweep,
  title  = {{SWEEP} ({S}eismic {W}ave {E}quation {E}xploration {P}latform):
            A Unified Solver Framework for Differentiable Wave Physics},
  author = {Wang, Shaowen and Alkhalifah, Tariq},
  year   = {2026},
  eprint = {2604.14189},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2604.14189},
}
```

## License

MIT — see [LICENSE](LICENSE).

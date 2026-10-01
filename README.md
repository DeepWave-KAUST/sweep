<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo/sweep-icon-dark.svg">
    <img src="docs/assets/logo/sweep-icon-light.svg" alt="SWEEP" width="180">
  </picture>
</p>

<h1 align="center">SWEEP</h1>

<p align="center">
  <a href="https://sweepx.deepwave.group/solver/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white"></a>
</p>

<p align="center">English | <a href="README.zh-CN.md">中文</a></p>

**Seismic Wave Equation Exploration Platform** — a differentiable framework for seismic wave-equation modeling, migration, and full-waveform inversion. One API, 20+ equations (acoustic / elastic / VTI / TTI / DAS), PyTorch and JAX backends, eager and compiled CUDA paths.

📖 **Documentation**: <https://sweepx.deepwave.group/solver/>

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # match your driver, see below
pip install sweepx
```

Install a torch built for your driver first: PyPI's default torch is CUDA 13 and needs
driver >= 580, so on an older driver (or a V100) use `cu126` as above. The
[install selector](https://sweepx.deepwave.group/solver/getting-started/installation/)
gives the exact command for your GPU and driver.

The wheel ships prebuilt CUDA cores, so the compiled backend (`impl='c'`) works right
after install with any PyTorch version: no nvcc, no compiler, no build step. It needs
an NVIDIA GPU and driver; the core for your torch's CUDA version is picked
automatically (CUDA 12: V100 and newer; CUDA 13: T4 and newer, driver >= 580). The
eager PyTorch and JAX backends are pure Python.

**From source**: a clone has no prebuilt core, so the CUDA core is compiled locally
once (with an nvcc matching your torch's CUDA version) and cached:

```bash
pip install .
python -m sweep.build   # optional: compile it now instead of on the first impl='c' call
```

<sub>`sweepx` (Python >= 3.10) is the PyPI name and also installs the `sweep-agent` companion;
you `import sweep`. `pip install sweep-solver` installs the solver alone (Python >= 3.9).
GPU coverage, custom cores and developer builds are covered in
[Building the CUDA core](https://sweepx.deepwave.group/solver/dev/building/).</sub>

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
                   device=device, use_ckpt=False)

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

- **Hello SWEEP** — forward / backward / 5-line FWI loop: [`docs/notebooks/00_hello_fwi.ipynb`](docs/notebooks/00_hello_fwi.ipynb)
- **FWI on Marmousi** (acoustic / elastic / multiscale): see [`docs/notebooks/01_*`–`03_*`](docs/notebooks/)
- **Wavefields, DAS, anisotropic, RTM**: [`docs/notebooks/04_*`–`08_*`](docs/notebooks/)
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

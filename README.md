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

**From PyPI** — one wheel, any PyTorch version, any Python 3:

```bash
pip install sweepx
python -c "import sweep; sweep.precompile()"   # build the CUDA backend now (one-time ~3–5 min)
```

`sweepx` ships the C++/CUDA *sources*; the compiled backend (`impl='c'`) is compiled
against **your** torch — only for your GPU's architecture, then cached in
`~/.cache/torch_extensions`. The `precompile()` line does it up front; drop it and it
happens automatically on first use of `impl='c'`. No torch/CUDA version lock-in. Needs
a CUDA GPU + `nvcc >= 12.4` (a system install, your cluster's `module load cuda`, or
`conda install -c nvidia cuda-toolkit`); the pure-Python **eager** / **JAX** backends
work without nvcc.

**From source** (a clone):

```bash
# pure-Python (PyTorch / JAX eager path); impl='c' JIT-compiles on first use
pip install .

# prebuild the C++/CUDA extension now — skips the first-use compile (needs nvcc)
SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation
```

If the prebuild can't auto-detect your GPU, set `TORCH_CUDA_ARCH_LIST` (e.g. `"7.0"`
V100, `"8.0"` A100, `"8.9"` RTX 6000 Ada) before the second command.

<sub>`sweepx` is the PyPI distribution name; you `import sweep` (the `scikit-learn` → `import sklearn`
pattern, because the bare name `sweep` is taken on PyPI). `pip install sweep-solver` is equivalent.
Full install notes are in [the docs](https://sweepx.deepwave.group/solver/getting-started/installation/).</sub>

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

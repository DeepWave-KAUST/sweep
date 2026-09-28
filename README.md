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
python -c "import sweep; sweep.precompile()"   # optional: check that a CUDA core is in place (a no-op with the shipped one)
```

The wheel carries a **prebuilt CUDA core** per CUDA major — `sweep/lib/cu12/` and
`sweep/lib/cu13/`, each a `libsweep_core.so` fat binary — and loads the one your
torch's CUDA major names. cu12 covers V100 and newer through H100/H200 (sm_70–sm_90
SASS) and Blackwell through its sm_90 PTX; cu13 covers T4/RTX 20 and newer
(sm_75–sm_90 SASS) with Blackwell native (sm_100, sm_120) — no V100, since nvcc 13
cannot emit sm_70, so a V100 needs a torch built for CUDA 12 (a local build cannot
help: nvcc 13 cannot target it either) — and needs driver >= 580, exactly what torch
cu130 needs. The
compiled backend (`impl='c'`) drives the core through a pure-Python `ctypes` layer
(`sweep._capi`) that fills the core's C structs straight from each tensor's
`data_ptr()`, shape, strides and dtype — so after `pip install` **nothing compiles**:
no nvcc, no C++ compiler, no CUDA headers (the `ninja` dependency only runs for a
local core build). No torch C++ ABI is involved,
which is why the same wheel works with any torch version. `nvcc >= 12.4` is needed
only when no shipped core fits — a torch built for a CUDA major with no core (neither
12 nor 13), or a GPU outside the shipped archs *and* older than the shipped PTX (e.g.
Pascal sm_6x; PTX is what makes newer cards fit) — or when installing from an
sdist/clone, which carry no core. Even then only the CUDA core is compiled, locally
for your card (2–5 min, at `python -m sweep.build` or on first use) and reused by
later runs without nvcc; there is never a torch shim to build.
`SWEEP_CORE=<path/to/libsweep_core.so>` points at a custom core. The core links
cuFFT at run time; torch's CUDA wheels bring it, and `pip install "sweepx[cuda12]"`
(or `"sweepx[cuda13]"`, matching your core) pulls it if yours did not. The
pure-Python **eager** / **JAX** backends need none of this.

**From source** (a clone):

```bash
# pure-Python (PyTorch / JAX eager path); a clone has no shipped core, so the
# first use of impl='c' builds the CUDA core locally (nvcc)
pip install .

# prebuild the C++/CUDA extension now — skips the first-use core build (a clone has
# no shipped core, so this and the first-use build both need nvcc)
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

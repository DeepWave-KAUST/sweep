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

<p align="center"><a href="README.md">English</a> | 中文</p>

**Seismic Wave Equation Exploration Platform** —— 一个可微分的地震波方程正演、偏移与全波形反演框架。一套 API,20+ 种波动方程(声波 / 弹性 / VTI / TTI / DAS),支持 PyTorch 与 JAX 后端,eager 与编译 CUDA 两条实现路径。

📖 **文档**: <https://sweepx.deepwave.group/solver/>

## 安装

**从 PyPI 安装** —— 一个 wheel,通吃任意 PyTorch 版本、任意 Python 3:

```bash
pip install sweepx
python -c "import sweep; sweep.precompile()"   # 可选:现在就预热 torch shim(~1 分钟,无需 nvcc)
```

wheel 自带面向 CUDA 12 的**预编译 CUDA core**(`sweep/lib/cu12/libsweep_core.so`,sm_70–sm_90 + PTX 的 fat binary;以后可以在同一个 wheel 里再搭上别的 CUDA tag)。编译版后端(`impl='c'`)只剩一层薄 torch shim 要对着**你自己的** torch 编 —— 纯 C++,约一分钟,不需要 nvcc,之后缓存在 `~/.cache/torch_extensions`。这一步只要一个 C++ 编译器和 CUDA runtime 头文件;torch 的 pip cu12 wheel 自带这些头文件(`nvidia-cuda-runtime-cu12` 和 `nvidia-cuda-nvcc-cu12`),toolkit 的 include 目录也行,所以只要这套 pip CUDA runtime 在,就不需要 toolkit。上面 `precompile()` 那行让它**现在就完成**;去掉它则在**第一次使用 `impl='c'` 时**自动编。**不锁 torch/CUDA 版本**。只有在没有合适的预编 core 时才需要 `nvcc >= 12.4` —— 你的 torch 是另一个 CUDA 大版本,或 GPU 既不在自带的架构列表里**又**比自带的 PTX 更老(例如 Pascal sm_6x;PTX 只能让更新的卡适配)—— 这时 core 会在本地按你的卡编一遍(2–5 分钟);从 sdist / 克隆安装同样需要 nvcc。`SWEEP_CORE=<path/to/libsweep_core.so>` 可指定自定义 core。如果你的 torch wheel 没带 cuFFT 或 CUDA runtime 头文件,`pip install "sweepx[cuda12]"` 补上。纯 Python 的 **eager** / **JAX** 后端这些都不需要。

**从源码安装**(克隆仓库):

```bash
# 纯 Python(PyTorch / JAX eager 路径);impl='c' 首次使用时 JIT 编译
pip install .

# 现在就预编译 C++/CUDA 扩展 —— 跳过首次使用时的编译(克隆里没有预编 core,
# 这一步和首次使用时的编译都需要 nvcc)
SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation
```

如果预编译无法自动检测 GPU 架构,在第二条命令前先设置 `TORCH_CUDA_ARCH_LIST`(如 V100 用 `"7.0"`、A100 用 `"8.0"`、RTX 6000 Ada 用 `"8.9"`)。

<sub>`sweepx` 是 PyPI 发行名,导入用 `import sweep`(类似 `scikit-learn` → `import sklearn`,因为裸名 `sweep` 在 PyPI 已被占)。`pip install sweep-solver` 等价。完整说明见[文档](https://sweepx.deepwave.group/solver/getting-started/installation/)。</sub>

## Hello SWEEP

一炮、一道、一次 `.backward()` —— 就能读出单道对应的速度模型梯度:

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

把 `Acoustic` 换成 `Elastic`、`AcousticVTI`、`ElasticTTI`…… —— 周围的代码完全不变。

## Notebooks 与示例

- **Hello SWEEP** —— forward / backward / 5 行 FWI 循环: [`docs/notebooks/00_hello_fwi.ipynb`](docs/notebooks/00_hello_fwi.ipynb)
- **Marmousi 上的 FWI**(声波 / 弹性 / 多尺度): 见 [`docs/notebooks/01_*`–`03_*`](docs/notebooks/)
- **波场、DAS、各向异性、RTM**: [`docs/notebooks/04_*`–`08_*`](docs/notebooks/)
- **生产脚本**(多 GPU、MPI 炮并行、多炮 batching): 在 [`examples/`](examples/) 目录下

## 引用

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

## 许可证

MIT —— 见 [LICENSE](LICENSE)。

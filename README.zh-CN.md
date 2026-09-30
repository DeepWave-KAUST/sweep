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

```bash
pip install sweepx
```

wheel 自带预编译的 CUDA core,装完就能用编译版后端(`impl='c'`),任意 PyTorch 版本都行:不需要 nvcc、不需要编译器,也没有编译步骤。只需要 NVIDIA GPU 和驱动;按你 torch 的 CUDA 版本自动选 core(CUDA 12:V100 及更新的卡;CUDA 13:T4 及更新的卡,驱动 >= 580)。eager(PyTorch)和 JAX 后端是纯 Python。

**从源码安装**:克隆里没有预编 core,CUDA core 会在本地编一次(需要和 torch CUDA 版本匹配的 nvcc)并缓存:

```bash
pip install .
python -m sweep.build   # 可选:现在就编,而不是等第一次调用 impl='c'
```

<sub>`sweepx`(Python >= 3.10)是 PyPI 发行名,会顺带装上 `sweep-agent`;导入用 `import sweep`。只装求解器用 `pip install sweep-solver`(Python >= 3.9)。GPU 覆盖范围、自定义 core、开发者构建等细节见[安装文档](https://sweepx.deepwave.group/solver/getting-started/installation/)。</sub>

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
                   device=device, pml_type="cpmlr", use_ckpt=False)

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

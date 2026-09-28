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
python -c "import sweep; sweep.precompile()"   # 可选:确认 CUDA core 已就位(有自带 core 时什么都不做)
```

wheel 按 CUDA 大版本各自带一个**预编译 CUDA core** —— `sweep/lib/cu12/` 与 `sweep/lib/cu13/`(各是一个 `libsweep_core.so` fat binary)—— 加载时按你 torch 的 CUDA 大版本选。cu12 覆盖 V100 及更新的卡直到 H100/H200(sm_70–sm_90 SASS),Blackwell 靠 sm_90 PTX;cu13 覆盖 T4/RTX 20 及更新的卡(sm_75–sm_90 SASS),Blackwell 原生(sm_100、sm_120)—— 不含 V100,因为 nvcc 13 编不出 sm_70,所以 V100 要用面向 CUDA 12 的 torch(本地编译也救不了:nvcc 13 同样编不出 sm_70)—— 且需要驱动 >= 580,和 torch cu130 的要求一样。编译版后端(`impl='c'`)通过一层纯 Python 的 `ctypes` 层(`sweep._capi`)驱动这个 core:直接用每个张量的 `data_ptr()`、shape、stride 和 dtype 填 core 的 C 结构体 —— 所以 `pip install` 之后**什么都不用编译**:不需要 nvcc、不需要 C++ 编译器、不需要 CUDA 头文件(依赖里的 `ninja` 只在本地编 core 时才会跑)。全程不经过 torch 的 C++ ABI,因此同一个 wheel 任意 torch 版本都能用。只有在没有合适的预编 core 时才需要 `nvcc >= 12.4` —— 你的 torch 是没带 core 的 CUDA 大版本(cu12、cu13 之外),或 GPU 既不在自带的架构列表里**又**比自带的 PTX 更老(例如 Pascal sm_6x;PTX 只能让更新的卡适配)—— 或者从 sdist / 克隆安装(两者都不带 core)。即便这时,也只编 CUDA core 本身,在本地按你的卡编一遍(2–5 分钟,`python -m sweep.build` 时或首次使用时),之后的运行直接复用,不再需要 nvcc;从来不需要编 torch shim。`SWEEP_CORE=<path/to/libsweep_core.so>` 可指定自定义 core。core 运行时链接 cuFFT;torch 的 CUDA wheel 自带它,没带的话 `pip install "sweepx[cuda12]"`(或 `"sweepx[cuda13]"`,按你的 core 选)补上。纯 Python 的 **eager** / **JAX** 后端这些都不需要。

**从源码安装**(克隆仓库):

```bash
# 纯 Python(PyTorch / JAX eager 路径);克隆里没有预编 core,所以
# 首次使用 impl='c' 时会在本地编 CUDA core(需要 nvcc)
pip install .

# 现在就预编译 C++/CUDA 扩展 —— 跳过首次使用时的 core 编译(克隆里没有预编 core,
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

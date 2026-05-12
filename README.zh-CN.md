# SWEEP

[English](README.md) | 中文

Seismic Wave Equation Exploration Platform (SWEEP) 是一个用于地震波方程正演、偏移和反演的 Python 软件包。

文档：https://deepwave-kaust.github.io/sweep/

近期接口更新：

- 支持惰性导入，因此只需要安装计划使用的后端
- `PropTorch` 是 Torch 系列接口的主要入口，`backend="eager"` 使用 PyTorch 算子，`backend="cuda"` 使用编译后的 C++/CUDA 扩展
- 后端相关选项通过 `EagerOptions`、`CUDAOptions`、`MemoryOptions`、`BoundaryOptions` 和 `CkptOptions` 组织
- 示例按任务类型整理在 `examples/` 下，包括新的 `wavefields/` 和 `reducingmemory/` 示例组

## 安装

请在 SWEEP 仓库根目录安装。如果还没有下载源码，先克隆仓库并进入项目目录：

```bash
git clone git@github.com:DeepWave-KAUST/sweep.git
cd sweep
```

### 仅使用 PyTorch/JAX

这是最简单的安装方式，但不一定是最高效的运行方式。如果只需要使用 PyTorch 或 JAX API，可以运行：

```bash
pip install .
```

### C++/CUDA 绑定

这种方式更快，也更推荐，但安装时间会更长，因为需要编译 C++ CPU kernel 和 CUDA kernel。如果已经有 PyTorch 和兼容的 CUDA toolkit，可以用下面的命令构建编译扩展：

```bash
SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
```

如果安装时无法自动检测 CUDA 架构，可以在安装前显式设置。例如：

```bash
# V100
export TORCH_CUDA_ARCH_LIST="7.0"

# A100
export TORCH_CUDA_ARCH_LIST="8.0"

# 构建同时支持 V100 和 A100 的 wheel
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
```

如果系统默认 CUDA toolkit 不是你想使用的版本，也可以设置 `CUDA_HOME`：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

`SWEEP_CUDA_ARCH_LIST` 可以作为 `TORCH_CUDA_ARCH_LIST` 的包内别名使用。如果两个变量都没有设置，构建会默认使用兼容 V100 的 `7.0`，避免 PyTorch 在无法自动检测架构时出现空架构列表错误。

## 后端和设备

对 `PropTorch` 来说，公开后端如下：

| `PropTorch` 后端 | Tensor 设备 | 实现 |
| --- | --- | --- |
| `backend="eager"` | CPU 或 CUDA | PyTorch eager 算子和 PyTorch autograd |
| `backend="cuda"` | CPU | 编译后的 C++ CPU kernel |
| `backend="cuda"` | CUDA | 编译后的 CUDA kernel |

在 FWI 示例脚本中，CLI 会分开指定实现和设备：

```bash
# PyTorch eager，默认有 GPU 时使用 GPU
python examples/FWI/2d/acoustic/torch/fwi_marmousi.py --backend pytorch --device auto

# 编译后的 CUDA kernel
python examples/FWI/2d/acoustic/torch/fwi_marmousi.py --backend c --device cuda

# 编译后的 C++ CPU kernel
python examples/FWI/2d/acoustic/torch/fwi_marmousi.py --backend c --device cpu

# 使用 MPI 做炮并行的 C++ CPU kernel
mpirun -np 4 python examples/FWI/2d/acoustic/torch/fwi_marmousi.py \
    --backend c --device cpu --mpi --mpi-forward-batchsize 4
```

`--device auto` 会在 PyTorch 报告 CUDA 可用时使用 CUDA，否则回退到 CPU。

## 使用示例

下面的例子展示如何计算一个简单模型相对于速度模型的梯度。

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

# Set the model parameters (PyTorch)
vp = torch.from_numpy(true_model).to(dev).requires_grad_(True)
# Create a wavelet
t = np.arange(0, int(nt//2)*dt, dt)
wave = ricker(t-delay, f=fm)

# Acquisition geometry
sources = np.array([[[1, 1]]], dtype=np.int32) # in grid, shape=(nshots, nsources, 2)
receivers = np.array([[[99, 1]]], dtype=np.int32) # in grid, shape=(nshots, nreceivers, 2)

# Forward modeling
# Backward propagation (PyTorch)
obs = model.forward(wave, sources, receivers, models=[vp])
obs.pow(2).sum().backward()
# Show the results
fig, axes=plt.subplots(1,3, figsize=(12,3))

axes[0].imshow(true_model, cmap='seismic', aspect='auto')
axes[0].set_title('True model')
axes[1].plot(obs.detach().cpu().numpy().squeeze(), label='Observed data')
grad = vp.grad.detach().cpu().numpy() # PyTorch
vmin,vmax=np.percentile(grad, [1,99])
axes[1].set_title('Observed data')
axes[2].imshow(grad, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[2].set_title('Gradient of vp')
fig.tight_layout()
fig.savefig('grad_vp.png', dpi=300, bbox_inches='tight')
plt.close()
```

下图展示真实模型、观测数据以及速度模型梯度。

![grad_vp](figures/grad_vp.png)

# 示例

示例按任务类型组织在 `examples/` 下：

- `FWI/`
- `LSRTM/`
- `wavefields/`
- `reducingmemory/`
- `multi-gpu/`

## 许可证

本项目使用 MIT License。

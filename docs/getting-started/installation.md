# Installation

This page explains how to install SWEEP for different use cases.

## Option 1: Base Installation

Use this when you only need the Pytorch/Jax package interface.

```bash
pip install .
```

## Option 2: PyTorch CUDA Binding Build

Choose this option when you want to build the PyTorch CUDA binding (`sweep._C`).

The compiled binding provides CUDA-based faster propagation and boundary saving support (for saving memory) for the PyTorch backend.

```bash
SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
```

It's currently only support the following wave equations:
- 2D/3D acoustic wave equations
- 2D/3D elastic wave equations

## Requirements

- Python 3.8+
- A working [PyTorch](https://pytorch.org/get-started/locally/) or [JAX](https://docs.jax.dev/en/latest/installation.html) environment depending on your backend
- CUDA toolkit and compatible NVIDIA drivers if building CUDA bindings

## Verify the Installation

```bash
sweep list equations
```

You can also check backend availability in Python:

```python
import sweep

print(sweep.backend.torch.is_available())
print(sweep.backend.jax.is_available())
print(sweep.backend.torch.cuda.is_available())
```

## Notes

- The base package supports lazy imports
- You do not need to install both JAX and PyTorch unless you plan to use both
- If you want to compile and use the PyTorch binding, use Option 2 instead of the base installation
- CUDA source files are needed for source builds, but not for normal runtime imports

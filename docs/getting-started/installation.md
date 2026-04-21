# Installation

This page explains how to install SWEEP depending on whether your working
environment is based on JAX, plain PyTorch, or PyTorch with the compiled CUDA
binding.

## Get the Source Code

Install from the project root directory. If you have not downloaded the source
code yet, clone the repository first and change into the repository root:

```bash
git clone https://github.com/DeepWave-KAUST/sweep
cd sweep
```

## Install by Backend

=== "PyTorch + CUDA Binding"

    Use this path when you want the compiled CUDA backend in addition to the
    regular PyTorch interface.

    1. Install a compatible PyTorch + CUDA environment first.
    2. Make sure your CUDA toolkit and NVIDIA driver are available for builds.
    3. Build and install SWEEP with the CUDA extra:

    ```bash
    SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
    ```

    Notes:

    - This build produces the compiled extension module `sweep._C`.
    - After installation, you can use the CUDA backend through:

    ```python
    from sweep.propagator.torch import PropTorch

    solver = PropTorch(..., backend="cuda")
    ```

    - The lower-level CUDA-specific class `sweep.propagator.cuda.PropCUDA`
      remains available as well.
    - The compiled binding currently supports:
      - 2D/3D acoustic equations
      - 2D/3D elastic equations

=== "PyTorch"

    Use this path when your environment is PyTorch-first, but you only need
    the eager Torch backend and do not want to build the compiled binding.

    1. Install a working PyTorch environment first.
    2. Install SWEEP from the repository root:

    ```bash
    pip install .
    ```

    Notes:

    - This path gives you the Torch-family Python interface, including
      `PropTorch(..., backend="eager")`.
    - You can still use checkpointing and `torch.compile` through
      `EagerOptions`.

=== "JAX"

    Use this path when your environment is JAX-first and you do not need the
    PyTorch CUDA binding.

    1. Install a working JAX environment first.
    2. Install SWEEP from the repository root:

    ```bash
    pip install .
    ```

    Notes:

    - SWEEP supports lazy imports, so you do not need to install PyTorch just
      to use the JAX path.
    - This path gives you the Python package interface and `PropJax`.

## Requirements

- Python 3.8+
- A working [PyTorch](https://pytorch.org/get-started/locally/) or
  [JAX](https://docs.jax.dev/en/latest/installation.html) environment depending
  on your backend
- CUDA toolkit and compatible NVIDIA drivers if building CUDA bindings

## Verify the Installation

From the shell:

```bash
sweep list equations
```

From Python:

```python
import sweep

print(sweep.backend.torch.is_available())
print(sweep.backend.jax.is_available())
print(sweep.backend.torch.cuda.is_available())
```

If you built the CUDA binding, you can also verify that the extension is
importable:

```python
import importlib.util

print(importlib.util.find_spec("sweep._C") is not None)
```

## Notes

- Lazy imports mean you do not need to install both JAX and PyTorch unless you
  plan to use both.
- If you want the compiled Torch CUDA binding, use the `PyTorch + CUDA Binding`
  path rather than the base install.
- CUDA source files are needed for source builds, but not for normal runtime
  imports after installation.

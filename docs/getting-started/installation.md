# Installation

This page explains how to install SWEEP depending on whether your working
environment is based on JAX, plain PyTorch, or PyTorch with the compiled
C++/CUDA Torch extension binding.

## Get the Source Code

Install from the project root directory. If you have not downloaded the source
code yet, clone the repository first and change into the repository root:

```bash
git clone https://github.com/DeepWave-KAUST/sweep
cd sweep
```

## Install by Backend and Binding

=== "PyTorch + Extension Binding"

    Use this path when you want compiled C++/CUDA kernels in addition to the
    regular PyTorch interface.

    1. Install a compatible PyTorch + CUDA environment first.
    2. Make sure your CUDA toolkit and NVIDIA driver are available for builds.
    3. Build and install SWEEP with the CUDA extra:

    ```bash
    SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
    ```

    Notes:

    - This build produces the compiled extension module `sweep._C`.
    - After installation, you can use the compiled extension implementation through:

    ```python
    from sweep.propagator.torch import PropTorch

    solver = PropTorch(..., backend="torch", impl="c")
    ```

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
      `PropTorch(..., backend="torch", impl="eager")`.
    - You can still use checkpointing and `torch.compile` through
      `EagerOptions`.

=== "JAX"

    Use this path when your environment is JAX-first and you do not need the
    PyTorch extension binding.

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

- Python 3.9+
- A working [PyTorch](https://pytorch.org/get-started/locally/) or
  [JAX](https://docs.jax.dev/en/latest/installation.html) environment depending
  on your backend
- CUDA toolkit and compatible NVIDIA drivers if building the CUDA side of the extension binding

## Verify the Installation

From the shell:

```bash
sweep list equations
sweep show Acoustic
```

From Python, the simplest one-liner is:

```python
import sweep

# True when PyTorch + CUDA + the compiled sweep._C binding are all importable.
print(sweep.is_torch_binding_available())
```

For finer-grained diagnostics:

```python
import sweep

print(sweep.backend.torch.is_available())          # PyTorch importable
print(sweep.backend.torch.cuda.is_available())     # PyTorch sees a CUDA device
print(sweep.backend.torch.binding.is_available())  # sweep._C extension importable
print(sweep.backend.torch.binding.diagnostics())   # dict with the same details
print(sweep.backend.jax.is_available())            # JAX importable
```

## Notes

- Lazy imports mean you do not need to install both JAX and PyTorch unless you
  plan to use both.
- If you want the compiled Torch extension binding, use the `PyTorch + Extension Binding`
  path rather than the base install.
- CUDA source files are needed for source builds, but not for normal runtime
  imports after installation.

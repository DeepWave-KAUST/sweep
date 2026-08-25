# Installation

## From PyPI (recommended)

One wheel, any PyTorch version, any Python 3:

```bash
pip install sweepx
python -c "import sweep; sweep.precompile()"   # build the CUDA backend now (one-time ~3–5 min)
```

`sweepx` ships the C++/CUDA *sources*; the compiled backend (`impl='c'`) is compiled
against **your** torch — only for your GPU's architecture, then cached in
`~/.cache/torch_extensions`, so there is no torch/CUDA version lock-in. The
`precompile()` line does it up front; drop it and the compile happens automatically
on first use of `impl='c'`. This needs a CUDA GPU and a CUDA toolkit with
`nvcc >= 12.4` (a system install, `module load cuda`, or
`conda install -c nvidia cuda-toolkit`). The pure-Python eager/JAX backends work
without nvcc.

!!! note
    `sweepx` is the PyPI distribution name; you `import sweep` — the
    `scikit-learn` → `import sklearn` pattern, because the bare name `sweep` is
    already taken on PyPI. `pip install sweep-solver` is equivalent.

The rest of this page covers installing **from a clone** — for development, or to
pre-build the compiled extension and skip the one-time first-use compile.

## Get the Source Code

Install from the project root directory. If you have not downloaded the source
code yet, clone the repository first and change into the repository root:

```bash
git clone https://github.com/DeepWave-KAUST/sweep
cd sweep
```

## Install by Backend and Binding

=== "PyTorch + Extension Binding"

    Use this from a clone to **pre-build** the compiled `sweep._C` now — the same
    kernels the PyPI `sweepx` wheel builds on first use, but ahead of time so there
    is no first-use compile wait. (A prebuilt `_C` extension takes precedence over
    the JIT loader automatically.)

    1. Install a compatible PyTorch + CUDA environment first.
    2. Make sure `nvcc >= 12.4` and your NVIDIA driver are available for builds.
    3. Build and install SWEEP with the CUDA extra:

    ```bash
    SWEEP_BUILD_CUDA=1 pip install -v .[cuda] --no-build-isolation
    ```

    Notes:

    - This build produces the compiled extension module `sweep._C`.
    - After installation, `PropTorch` auto-detects the binding by default:

    ```python
    from sweep.propagator.torch import PropTorch

    solver = PropTorch(...)              # impl='auto' → 'c' when available
    solver = PropTorch(..., impl="c")    # explicit; warns + falls back if missing
    solver = PropTorch(..., impl="eager")  # force pure-PyTorch
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
- For the compiled `impl='c'` backend: a CUDA GPU and a CUDA toolkit with
  `nvcc >= 12.4` (12.0–12.3 ship a broken `<cuda/std>` bf16 header; set
  `SWEEP_JIT_ALLOW_OLD_CUDA=1` to try one anyway), plus compatible NVIDIA
  drivers — used by both the PyPI JIT first-use compile and a source prebuild

## Verify the Installation

From the shell:

```bash
sweep list equations
sweep show Acoustic
```

From Python, the simplest one-liner is:

```python
import sweep

# True when sweep._C is already compiled on disk, OR PyTorch + a CUDA GPU +
# nvcc are present so it can be JIT-compiled on first use (this check itself
# does NOT trigger the compile).
print(sweep.is_torch_binding_available())
```

For finer-grained diagnostics:

```python
import sweep

print(sweep.backend.torch.is_available())            # PyTorch importable
print(sweep.backend.torch.cuda.is_available())       # PyTorch sees a CUDA device
print(sweep.backend.torch.binding.is_available())    # backend usable (pre-built, or torch + GPU + nvcc>=12.4)
print(sweep.backend.torch.binding.is_compiled())     # backend already built (pre-built/compiled/cached)
print(sweep.backend.torch.binding.diagnostics())     # {'usable', 'reason', 'cuda_home', 'already_compiled', 'prebuilt'}
print(sweep.backend.jax.is_available())              # JAX importable
```

To build the compiled backend up front **and confirm it succeeds**, run:

```bash
python -c "import sweep; sweep.precompile()"   # exits 0 on success; raises a clear error if nvcc/GPU is missing
```

Afterwards `sweep.backend.torch.binding.is_compiled()` returns `True`.

## Notes

- Lazy imports mean you do not need to install both JAX and PyTorch unless you
  plan to use both.
- If you want the compiled Torch extension binding, use the `PyTorch + Extension Binding`
  path rather than the base install.
- CUDA source files are needed for source builds, but not for normal runtime
  imports after installation.

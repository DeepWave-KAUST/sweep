# Installation

## From PyPI (recommended)

One wheel, any PyTorch version, any Python 3:

```bash
pip install sweepx
python -c "import sweep; sweep.precompile()"   # optional: warm up the torch shim now (~1 min, no nvcc)
```

The wheel carries a **prebuilt CUDA core** for CUDA 12 (`sweep/lib/cu12/libsweep_core.so`,
a fat binary for sm_70–sm_90 plus PTX for forward compatibility; more CUDA tags can
ride along in the same wheel later). The compiled backend (`impl='c'`) is that core
plus a thin torch shim; only the shim is compiled against **your** torch — plain
C++, about a minute, no nvcc — then cached in `~/.cache/torch_extensions`, so there
is no torch/CUDA version lock-in. The shim compile needs a C++ compiler and the CUDA
runtime headers: torch's pip cu12 wheels bring them (`nvidia-cuda-runtime-cu12`), or
a toolkit's include dir serves — so no nvcc, and no toolkit as long as the pip CUDA
runtime is present. The `precompile()` line does it up front; drop it and the
compile happens automatically on first use of `impl='c'`.

`nvcc >= 12.4` is needed only when no shipped core fits your process — your torch was
built for a different CUDA major than the shipped core, or your GPU is outside the
shipped archs *and* older than the shipped PTX (e.g. Pascal sm_6x; PTX is what makes
newer cards fit). Then the core is built locally for your card (2–5 min, the same
local build a clone uses) and cached next to the shim. Installing from an sdist or a
clone always takes that path, because neither carries a core.

- `SWEEP_CORE=<path/to/libsweep_core.so>` points at a custom core (its `core.json`
  sidecar should sit beside it; without one the core is accepted unchecked).
- The core links cuFFT dynamically. Torch's pip cu12 wheels normally bring it
  (`nvidia-cufft-cu12`) together with the CUDA runtime headers
  (`nvidia-cuda-runtime-cu12`); if yours did not, `pip install "sweepx[cuda12]"`
  adds both.
- The pure-Python eager/JAX backends need neither the core nor nvcc.

!!! note
    `sweepx` is the PyPI distribution name; you `import sweep` — the
    `scikit-learn` → `import sklearn` pattern, because the bare name `sweep` is
    already taken on PyPI. `pip install sweep-solver` is equivalent.

The rest of this page covers installing **from a clone** — for development, or to
pre-build the compiled extension and skip the one-time first-use compile — and
building a shippable core.

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
    kernels the PyPI `sweepx` wheel ships as a prebuilt core, but built here from
    source (a clone has no core, so nvcc is required) and ahead of time so there is
    no first-use compile wait. (A prebuilt `_C` extension takes precedence over the
    JIT loader automatically.)

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
- For the compiled `impl='c'` backend: a CUDA GPU with compatible NVIDIA drivers,
  a C++ compiler for the torch shim, and the CUDA runtime headers — torch's pip
  cu12 wheels bring them (`nvidia-cuda-runtime-cu12`), or a toolkit's include dir
  does. From the PyPI wheel that is all: the CUDA core is prebuilt, so no nvcc and
  no toolkit as long as the pip CUDA runtime is present. A CUDA toolkit with
  `nvcc >= 12.4` (12.0–12.3 ship a broken `<cuda/std>` bf16 header; set
  `SWEEP_JIT_ALLOW_OLD_CUDA=1` to try one anyway) is needed only when no shipped
  core fits (a torch built for another CUDA major, or a GPU outside the shipped
  archs and older than the shipped PTX) or for a source build

## Verify the Installation

From the shell:

```bash
sweep list equations
sweep show Acoustic
```

From Python, the simplest one-liner is:

```python
import sweep

# True when sweep._C is already compiled on disk, OR PyTorch + a CUDA GPU are
# present and either a shipped core fits or nvcc can build one, so the backend
# can be JIT-compiled on first use (this check itself does NOT trigger the compile).
print(sweep.is_torch_binding_available())
```

For finer-grained diagnostics:

```python
import sweep

print(sweep.backend.torch.is_available())            # PyTorch importable
print(sweep.backend.torch.cuda.is_available())       # PyTorch sees a CUDA device
print(sweep.backend.torch.binding.is_available())    # backend usable (pre-built, or torch + GPU + shipped core / nvcc>=12.4)
print(sweep.backend.torch.binding.is_compiled())     # backend already built (pre-built/compiled/cached)
print(sweep.backend.torch.binding.diagnostics())     # {'usable', 'reason', 'cuda_home', 'already_compiled', 'prebuilt', 'shipped_core'}
print(sweep.backend.jax.is_available())              # JAX importable
```

To see which core `impl='c'` would use — the shipped one, `SWEEP_CORE`, or none
(and why, in which case the local nvcc build runs):

```python
from sweep._jit import shipped_core_info

print(shipped_core_info())   # {'path': ..., 'reason': ..., 'tag': 'cu12'}
```

To build the compiled backend up front **and confirm it succeeds**, run:

```bash
python -c "import sweep; sweep.precompile()"   # exits 0 on success; raises a clear error if the GPU (or nvcc, when needed) is missing
```

Afterwards `sweep.backend.torch.binding.is_compiled()` returns `True`.

### Building where there is no GPU (CI, or a CPU allocation on a cluster)

Compiling needs a compiler and a target architecture — not a card (and, with a
shipped core that fits, neither nvcc nor `TORCH_CUDA_ARCH_LIST`: only the shim is
compiled, and the arch list only names the target of a local core build). Name the
arch and build ahead of time, then let the GPU run pick the cache up:

```bash
TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/scratch/ext python -m sweep.build --no-gpu-required
```

Point the GPU job at the same `TORCH_EXTENSIONS_DIR` and it starts without
compiling. This matters on a shared cluster: without it, every build has to sit
inside a GPU allocation to run a compiler that never touches the GPU, and the
queue for a GPU partition is usually much longer than for CPU.

`--no-gpu-required` only relaxes the *build*. `sweep.is_torch_binding_available()`
still reports `False` on a machine with no device — you cannot run `impl='c'`
there, only produce the `.so`.

### Building a shippable core (release machines)

The core the wheel carries is a fat binary built once, with no GPU in sight:

```bash
python -m sweep.build --core                        # defaults: archs 7.0;7.5;8.0;8.6;8.9;9.0+PTX
python -m sweep.build --core --archs "8.0;9.0+PTX" --out /path/to/lib/cu12 --cuda-home /usr/local/cuda-12.9
```

It stages the sources, runs the core's own ninja graph with `nvcc`, and writes
`libsweep_core.so` plus a `core.json` sidecar (ABI version, CUDA release, archs,
PTX, sha256, flags). `--out` defaults to inside the installed `sweep` package,
`sweep/lib/<tag>` — `src/sweep/lib/<tag>` in a clone, `cu12` for a CUDA 12 toolkit. A wheel built from
a tree that contains such a core becomes a `manylinux` platform wheel that ships
it; without one the wheel stays `py3-none-any` and users compile from source as
before.

Three rules for a core that is going to PyPI:

- **Build the core, then `python -m build --wheel` from the same tree.**
  `src/sweep/lib/` is git-ignored and pruned from the sdist (`MANIFEST.in`), so a
  bare `python -m build` packs the wheel from the pruned sdist and ships no core.
  The release script sets `SWEEP_REQUIRE_CORE=1`, which makes `setup.py` refuse a
  tree with no `src/sweep/lib/*/libsweep_core.so` instead of quietly producing a
  core-less wheel.
- **Build on the oldest glibc host you support.** The wheel's `manylinux_X_Y` tag
  is the build host's glibc, and pip rejects the wheel on anything older. The core
  links the host's `libstdc++` dynamically as well, so the same host sets the
  `libstdc++` floor: a user machine with an older one fails at import with a
  `GLIBCXX_...` version error.
- **Mind the size.** The default arch list (six SASS targets + `sm_90` PTX) gives a 134 MB `.so`
  and a 38 MB wheel; PyPI's per-file limit is 100 MB. Add SASS entries sparingly and
  keep exactly one `+PTX` entry, the newest arch: each embedded PTX is another copy
  of every kernel, and a card newer than every SASS entry only ever needs the
  newest one.

## Notes

- Lazy imports mean you do not need to install both JAX and PyTorch unless you
  plan to use both.
- If you want the compiled Torch extension binding, use the `PyTorch + Extension Binding`
  path rather than the base install.
- CUDA source files are needed for source builds, but not for normal runtime
  imports after installation.

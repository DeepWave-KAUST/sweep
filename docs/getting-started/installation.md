# Installation

## From PyPI (recommended)

One wheel, any PyTorch version, any Python 3:

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
compiled backend (`impl='c'`) is that core
plus a pure-Python `ctypes` layer (`sweep._capi`) that fills the core's C structs
straight from each tensor's `data_ptr()`, shape, strides and dtype. So after
`pip install` **nothing compiles**: no nvcc, no C++ compiler, no CUDA headers; the
`ninja` dependency is only run when a local core is built. No torch C++ ABI is
involved, which is why the same wheel works with any torch version — there is no
torch/CUDA version lock-in.

`nvcc >= 12.4` is needed only when no shipped core fits your process — your torch was
built for a CUDA major with no shipped core (neither 12 nor 13), or your GPU is outside the
shipped archs *and* older than the shipped PTX (e.g. Pascal sm_6x; PTX is what makes
newer cards fit). Then the core is built locally for your card (2–5 min, the same
local build a clone uses) at `python -m sweep.build` or on the first use of
`impl='c'`, and cached under `TORCH_EXTENSIONS_DIR` (default
`~/.cache/torch_extensions`). Installing from an sdist or a clone always takes that
path, because neither carries a core. Even then only the CUDA core is compiled:
there is never a torch shim to build.

- `sweep.precompile()` just makes sure a core is there and loads it — a no-op with
  a shipped core that fits, the local core build otherwise. Optional: the first
  `impl='c'` call does the same. A local core built once is reused by later runs
  without nvcc: a source stamp (`sources.sha256`, the csrc tree plus the nvcc
  flags) beside it says whether it is current.
- `SWEEP_CORE=<path/to/libsweep_core.so>` points at a custom core. Its `core.json`
  sidecar, when it sits beside it, gets the same fit check as the shipped one: the
  core's ABI version against the shim's, its CUDA major against torch's, and the
  visible GPU against its archs (an exact or same-major-lower-minor SASS entry, or
  a PTX entry the driver can JIT forward — which needs a driver at least as new as
  the toolkit that emitted it). Without a sidecar only a CUDA build of torch is
  required and none of that is checked.
- The core links cuFFT dynamically, at run time. Torch's pip CUDA wheels bring it
  (`nvidia-cufft-cu12` with a cu12 torch; the unsuffixed `nvidia-cufft` 12.x with a
  cu13 one — the core's rpath reaches both pip layouts, `nvidia/cufft/lib` and
  `nvidia/cu13/lib`). If yours did not (conda torch, a CPU wheel plus a system
  driver), `pip install "sweepx[cuda12]"` or `"sweepx[cuda13]"` adds the one your
  core needs.
- The pure-Python eager/JAX backends need neither the core nor nvcc.

!!! note
    `sweepx` is the PyPI distribution name; you `import sweep` — the
    `scikit-learn` → `import sklearn` pattern, because the bare name `sweep` is
    already taken on PyPI. `pip install sweep-solver` is equivalent.

The rest of this page covers installing **from a clone** — for development, or to
build the CUDA core ahead of time and skip the one-time first-use core build — and
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
    no first-use core build. (A prebuilt `_C` extension takes precedence over the
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

### Developer path: the compiled shim (`SWEEP_JIT_FULL=1`)

`impl='c'` normally reaches the core through the pure-Python `ctypes` layer and
compiles nothing. `SWEEP_JIT_FULL=1` switches to the developer path instead: the
old pybind torch shim (`module.cpp`, compiled against **your** torch through
`torch.utils.cpp_extension`) plus the CPU engine (`csrc/cpu`), built by the JIT
loader on first use and cached under `TORCH_EXTENSIONS_DIR`. That path needs
torch's C++ headers, a C++ compiler and `nvcc`, and the result is tied to the
torch it was built against. Use it to work on the binding or the CPU engine, or to
A/B the two shims; `sweep.backend.torch.binding.diagnostics()["shim"]` says which
one is loaded (`"ctypes"` or `"pybind"`).

## Requirements

- Python 3.9+
- A working [PyTorch](https://pytorch.org/get-started/locally/) or
  [JAX](https://docs.jax.dev/en/latest/installation.html) environment depending
  on your backend
- For the compiled `impl='c'` backend: a CUDA GPU with compatible NVIDIA drivers.
  From the PyPI wheel that is all: the CUDA core is prebuilt and the binding is
  pure Python, so nothing compiles — no nvcc, no C++ compiler, no CUDA headers, no
  toolkit. A CUDA toolkit with `nvcc >= 12.4` (12.0–12.3 ship a broken
  `<cuda/std>` bf16 header; set `SWEEP_JIT_ALLOW_OLD_CUDA=1` to try one anyway) is
  needed only when no shipped core fits (a torch built for a CUDA major with no
  shipped core, or
  a GPU outside the shipped archs and older than the shipped PTX), for a source
  build, or for the `SWEEP_JIT_FULL=1` developer path — and outside that path only
  the core is compiled

## Verify the Installation

From the shell:

```bash
sweep list equations
sweep show Acoustic
```

From Python, the simplest one-liner is:

```python
import sweep

# True when a prebuilt sweep._C extension is on disk, OR PyTorch + a CUDA GPU are
# present and either a core is at hand (the shipped one, SWEEP_CORE, a cached
# local build) or nvcc can build one. This check itself builds nothing.
print(sweep.is_torch_binding_available())
```

For finer-grained diagnostics:

```python
import sweep

print(sweep.backend.torch.is_available())            # PyTorch importable
print(sweep.backend.torch.cuda.is_available())       # PyTorch sees a CUDA device
print(sweep.backend.torch.binding.is_available())    # backend usable (pre-built, or torch + GPU + shipped core / nvcc>=12.4)
print(sweep.backend.torch.binding.is_compiled())     # a core (or pre-built extension) is already in place: first impl='c' use is instant
print(sweep.backend.torch.binding.diagnostics())     # {'usable', 'reason', 'cuda_home', 'already_compiled', 'prebuilt', 'shipped_core', 'shim'}
                                                     # 'shim' is 'ctypes' (default) or 'pybind' (SWEEP_JIT_FULL=1)
print(sweep.backend.jax.is_available())              # JAX importable
```

To see which core `impl='c'` would use — the shipped one, `SWEEP_CORE`, or none
(and why, in which case the local nvcc build runs):

```python
from sweep._jit import shipped_core_info

print(shipped_core_info())   # {'path': ..., 'reason': ..., 'tag': 'cu12'}; the tag is your torch's CUDA major, cu12 or cu13
```

To make sure a core is in place up front **and confirm it** (optional), run:

```bash
python -c "import sweep; sweep.precompile()"   # exits 0 on success (a no-op with a shipped core); raises a clear error if the GPU (or nvcc, when the core must be built) is missing
```

Afterwards `sweep.backend.torch.binding.is_compiled()` returns `True`.

### Building where there is no GPU (CI, or a CPU allocation on a cluster)

This only matters when the core has to be built locally — with a shipped core that
fits there is nothing to build and `python -m sweep.build` returns at once. A local
core build needs nvcc and a target architecture — not a card. Name the arch and
build ahead of time, then let the GPU run pick the cache up:

```bash
TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/scratch/ext python -m sweep.build --no-gpu-required
```

Point the GPU job at the same `TORCH_EXTENSIONS_DIR` and it starts without
compiling — and without nvcc on that node: the build leaves a source stamp
(`sources.sha256`) beside the core, and a core whose stamp matches the tree and
the target is reused as it is. This matters on a shared cluster: without it, every build has to sit
inside a GPU allocation to run a compiler that never touches the GPU, and the
queue for a GPU partition is usually much longer than for CPU.

`--no-gpu-required` only relaxes the *build*. `sweep.is_torch_binding_available()`
still reports `False` on a machine with no device — you cannot run `impl='c'`
there, only produce the `.so`.

### Building a shippable core (release machines)

The wheel carries one core per CUDA major, each a fat binary built once, with no
GPU in sight — one command per nvcc:

```bash
python -m sweep.build --core --cuda-home /usr/local/cuda-12.9   # -> src/sweep/lib/cu12, archs 7.0;7.5;8.0;8.6;8.9;9.0+PTX
python -m sweep.build --core --cuda-home /usr/local/cuda-13.0   # -> src/sweep/lib/cu13, archs 7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX
python -m sweep.build --core --archs "8.0;9.0+PTX" --out /path/to/lib/cu12 --cuda-home /usr/local/cuda-12.9   # a custom list
```

Each run stages the sources, runs the core's own ninja graph with that `nvcc`, and
writes `libsweep_core.so` plus a `core.json` sidecar (ABI version, CUDA release,
archs, PTX, sha256, flags). `--out` defaults to inside the installed `sweep` package,
`sweep/lib/<tag>` — `src/sweep/lib/<tag>` in a clone — where `<tag>` is the nvcc's
CUDA major: `cu12` for a CUDA 12 toolkit, `cu13` for CUDA 13. The loader picks the
tag torch's CUDA major names, so both cores ride in the same wheel. Without
`--archs` each toolkit gets its recommended list above; they differ because nvcc 13
dropped offline compilation for compute capability < 7.5 (an `sm_70` entry under
nvcc 13 is refused up front), so the cu13 core has no V100 and adds Blackwell
(sm_100, sm_120) natively, which the cu12 core reaches only through its sm_90 PTX.
A wheel built from a tree that contains such a core becomes a `manylinux` platform
wheel that ships it; without one the wheel stays `py3-none-any` and users build the
core locally (nvcc) as before.

Three rules for a core that is going to PyPI:

- **Build both cores, then `python -m build --wheel` from the same tree.**
  `src/sweep/lib/` is git-ignored and pruned from the sdist (`MANIFEST.in`), so a
  bare `python -m build` packs the wheel from the pruned sdist and ships no core.
  The release script sets `SWEEP_REQUIRE_CORE=cu12,cu13`, which makes `setup.py`
  refuse a tree missing either `src/sweep/lib/<tag>/libsweep_core.so`, naming the
  missing tag, instead of quietly producing a wheel with one core or none
  (`SWEEP_REQUIRE_CORE=1` asks only for at least one).
- **Build on the oldest glibc host you support.** The wheel's `manylinux_X_Y` tag
  is the build host's glibc, and pip rejects the wheel on anything older. The core
  links the host's `libstdc++` dynamically as well, so the same host sets the
  `libstdc++` floor: a user machine with an older one fails at import with a
  `GLIBCXX_...` version error.
- **Mind the size.** The cu12 list (six SASS targets + `sm_90` PTX) gives a 126 MB
  `.so` and a 37 MB wheel; the cu13 core adds a second `.so` of similar size to the
  same wheel. PyPI's per-file limit is 100 MB. Add SASS entries sparingly and keep
  exactly one `+PTX` entry per core, the newest arch: each embedded PTX is another
  copy of every kernel, and a card newer than every SASS entry only ever needs the
  newest one.

## Notes

- Lazy imports mean you do not need to install both JAX and PyTorch unless you
  plan to use both.
- From a clone, use the `PyTorch + Extension Binding` path if you want the compiled
  backend built ahead of time; from the PyPI wheel the base install already has it.
- CUDA source files are needed for source builds, but not for normal runtime
  imports after installation.

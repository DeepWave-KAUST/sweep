# Building the CUDA core

Details behind [Installation](../getting-started/installation.md). Users of the PyPI wheel rarely need them.

## Which core is loaded

The wheel ships a **prebuilt CUDA core**, so the compiled backend (`impl='c'`)
works right after install, with any PyTorch version: nothing compiles, no nvcc,
no C++ compiler. The core matching your torch's CUDA major is loaded:

| Core | GPUs |
|---|---|
| `cu12` | V100 (sm_70) through H100/H200; Blackwell through PTX |
| `cu13` | T4 / RTX 20 (sm_75) through Blackwell (native); driver >= 580, as torch cu130 needs |

A V100 needs a torch built for CUDA 12 (nvcc 13 cannot target sm_70). The eager
PyTorch and JAX backends are pure Python and need no core.

The core links cuFFT at run time. Torch's pip CUDA wheels bring it; if yours did
not (conda torch, a CPU wheel plus a system driver), add it with
`pip install "sweep-solver[cuda12]"` (or `[cuda13]`, matching your core).

## When nvcc is needed

Only when no shipped core fits (a torch built for a CUDA major with no core, or a
GPU older than the shipped architectures, e.g. Pascal) or when installing from
source. Then the core alone is built locally, once: 2–5 min, on the first
`impl='c'` call or with `python -m sweep.build`. It is cached under
`TORCH_EXTENSIONS_DIR/sweep_C/core/` (default `~/.cache/torch_extensions/...`) and
reused by later runs without nvcc. Use an nvcc of your torch's CUDA major:
>= 12.4 for CUDA 12 (>= 12.8 to target Blackwell; `SWEEP_JIT_ALLOW_OLD_CUDA=1`
tries 12.0–12.3), or any CUDA 13 nvcc.

## Installing from source

A clone carries no prebuilt core; building one needs only nvcc (no torch headers,
no C++ compiler). `PropTorch(...)` picks `impl='c'` when it can run and falls back
to `"eager"` otherwise; which equations have a compiled implementation is in the
`impl="c"` column of [Equations](../user-guide/equations.md). The JAX backend
(`PropJax`) needs a working
[JAX install](https://docs.jax.dev/en/latest/installation.html); SWEEP imports it
lazily, only when used.

## Building where there is no GPU

A local core build needs nvcc and a target architecture, not a card. On a CPU node
(CI, or a CPU allocation on a cluster), name the arch and a cache directory:

```bash
TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/scratch/ext python -m sweep.build --no-gpu-required
```

Point the GPU job at the same `TORCH_EXTENSIONS_DIR` and it starts without
compiling, and without nvcc: a source stamp beside the core says it matches the
tree and the target.

## Diagnostics

```bash
sweep list equations
```

```python
import sweep
print(sweep.is_torch_binding_available())          # can impl='c' run here? builds nothing
print(sweep.backend.torch.binding.diagnostics())   # why or why not, which core, which shim
```

`sweep.precompile()` loads the core up front (building it first when none fits)
and raises a clear error if the GPU, or nvcc when needed, is missing. It is
optional: the first `impl='c'` call does the same.

## Advanced

- **Custom core**: `SWEEP_CORE=<path/to/libsweep_core.so>`. A `core.json` beside
  it gets the same fit check as a shipped core (ABI, CUDA major, architectures).
- **Developer builds of the pybind shim**: same kernels and results as the default
  ctypes layer, but tied to the torch they are built against.
    - `SWEEP_JIT_FULL=1` builds it with the JIT loader on first use (torch's C++
      headers, a C++ compiler and the CUDA runtime headers).
    - `SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation` builds it
      ahead of time from a clone, as a `sweep._C` extension that takes precedence
      over the ctypes layer (nvcc, a C++ compiler and torch's headers; set
      `TORCH_CUDA_ARCH_LIST` to your card and have `ninja` on `PATH`).
    - `sweep.backend.torch.binding.diagnostics()["shim"]` reads `"pybind"` for
      either.
- **Release cores**: how the wheel's cores are built is in
  [Release cores](release_cores.md).

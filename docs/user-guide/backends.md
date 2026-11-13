# Backends

SWEEP supports two main user-facing backend families:

- `torch`
- `jax`

Within the Torch family, `PropTorch` now selects the actual implementation:

- `impl=None` *(default — equivalent to `impl="auto"`)*: on a CUDA device,
  probe `sweep.is_torch_binding_available()` and pick `"c"` when PyTorch, a
  visible CUDA GPU and a CUDA core (the wheel's prebuilt one, a cached local
  build, or an nvcc to build one) are present and the equation has compiled
  kernels; on a CPU device, pick `"c"` when the host core comes up (below);
  otherwise transparently fall back to `"eager"`. A plain `import sweep._C`
  always succeeds and proves nothing.
- `impl="eager"`: pure-PyTorch implementation (no build step required).
- `impl="c"`: the compiled implementation — on a CUDA device the prebuilt
  `libsweep_core.so` the wheel ships (`sweep/lib/cu12/` or `cu13/`, picked by
  your torch's CUDA major), on a CPU device the **host core** (the same sources
  built for the CPU, `sweep/lib/cpu/` or a local `g++` build), both driven by
  the pure-Python ctypes layer `sweep.backend.c`; nothing compiles after
  `pip install` when a shipped core fits. If no core fits and none can be built,
  or the equation has no compiled kernels, `PropTorch` falls back to `"eager"`
  with a `UserWarning` so the slowdown is visible.

The wheel ships two cores: `cu12` (sm_70–sm_90 SASS + sm_90 PTX) and `cu13`
(sm_75–sm_120 SASS + sm_120 PTX; no V100, driver >= 580). The loader picks
`lib/cu<torch CUDA major>/` and checks its `core.json` (ABI 2, CUDA major, archs
and PTX gated on the driver version). A local core build happens only when no
shipped core fits — a torch of another CUDA major (e.g. cu11), a GPU older than
sm_70 on cu12, an sdist/clone install — on the first `impl="c"` use, or ahead
of time with `python -m sweep.build`. It needs an nvcc of torch's CUDA major
(for CUDA 12: `>= 12.4`, `>= 12.8` for Blackwell targets). A GPU older than
sm_75 under a cu13 torch is refused rather than built for; use a cu12 torch
there. See [Building the CUDA core](../dev/building.md).

## User-Facing Backend Families

- `torch`: PyTorch-based propagation and differentiation
- `jax`: JAX-based propagation and differentiation

`cuda` is not a separate top-level backend alongside `torch` and `jax`. It is a
device choice. `impl="c"` follows the device of the propagator: CUDA tensors
run the CUDA core, CPU tensors run the host core — one compiled code base, two
builds (see [The host (CPU) core](../dev/host_core.md)). On a CPU device
`impl=None` (`"auto"`) takes the host core when the equation is built into it
and you passed no eager options (`eager_options=`, `use_compile=`, ...) and no
host staging (`storage='cpu'`/`'disk'`). The core is the wheel's `lib/cpu` one
(from 0.3.7) or `SWEEP_CORE_CPU=<path>`; a source install builds it once with
g++ (>= 9, with OpenMP) on the first such construction — about 1–3 minutes,
announced on stderr, then cached — as it builds the CUDA core with nvcc. With
no compiler that builds it, `"auto"` is `"eager"`, quietly; a build that fails
warns once and the process runs eager. The host core is faster than eager on
the CPU in nearly every case measured (1.2–18x at 8–28 threads; at 4 threads
compiled eager still wins a 3-D forward). An explicit `impl="c"` resolves the
core the same way, falls back to eager with a warning that says why when there
is none or the equation is not built into it, and raises at its first call
when the build fails. `python -m sweep.build --cpu` builds the core ahead of
time with the compiler's output shown. With the host core
a CPU run gets what a CUDA run gets from `impl="c"`: the compiled numerics
(as close to eager as the CUDA core is, ~1e-5 relative on acoustic2d),
boundary saving by default, no source-wavelet gradient for the elastic, DAS
and VTI 1st-order equations, and no double backward; pass `impl="eager"` for
those. `ViscoAcoustic` (cuFFT) is CUDA-only,
and `ElasticBorn`, the qP equations and the SEM kernels are not in the host
core yet. Boundary saving and checkpointing work on the host core with
on-device storage; staging to `'cpu'`/`'disk'` and pinned memory are refused
there, since the compute device already is host RAM.

Typical Torch-family usage:

```python
from sweep.propagator.torch import PropTorch

solver_eager = PropTorch(..., backend="torch", impl="eager")
solver_c = PropTorch(..., backend="torch", impl="c")
```

## Compiled CUDA core (`sweep._C`)

`sweep._C` is a lazy entry point: importing it is free, and the first attribute
access loads a core through ctypes (`sweep.backend.c` — `loader.py`,
`adapt.py`, `entries.py`, `runners.py`, the generated `abi.py`; `jit.py`
decides which CUDA core, `hostcore.py` which host core). Each entry picks the
core by the device of the tensors it is given, so `sweep._C.acoustic2d_forward`
serves CUDA and CPU models alike. The cores cover the acoustic (2-D/3-D), VRZ,
Born, VTI 1st-order, elastic (2-D/3-D), elastic TTI (SG 2-D/3-D, 2nd), elastic
VRR, DAS and visco-acoustic propagators, and the CUDA core also elastic Born,
qP (VTI/TTI) and SEM — `sweep list equations` is the authoritative list. The
host core leaves out `ViscoAcoustic` (cuFFT) and, not ported yet, `ElasticBorn`,
`AcousticVTI` / `AcousticTTI` with their Born twins and the SEM kernels
(`sweep.backend.c.hostcore.HOST_UNPORTED`).

You can inspect backend capability from Python:

```python
import sweep

sweep.backend.torch.is_available()
sweep.backend.jax.is_available()
sweep.backend.torch.cuda.is_available()
sweep.backend.torch.binding.is_available()
sweep.backend.torch.binding.diagnostics()
```

`sweep.backend.torch.cuda.is_available()` only answers whether PyTorch can see CUDA.
`sweep.backend.torch.binding.is_available()` answers whether `impl="c"` is
usable on CUDA here: PyTorch present, a CUDA GPU visible, and a CUDA core at
hand (the shipped one for your torch's CUDA major, `SWEEP_CORE`, a cached local
build, or an nvcc of torch's CUDA major to build one). It loads and compiles
nothing. The host core's answer is `diagnostics()["host_core"]`
(`sweep.backend.c.hostcore.info()`): a shipped `lib/cpu` core, a cached build,
or a C++ compiler to build one (and why each compiler was turned down), and
which equations are left out.

Example diagnostics output:

```python
{
    "usable": True,
    "reason": "ok",
    "shim": "ctypes",            # "pybind" under SWEEP_JIT_FULL=1 or with a prebuilt extension
    "cuda_home": "/usr/local/cuda-12.9",   # None is fine: a shipped core needs no nvcc
    "already_compiled": False,   # True once the core is loaded (sweep.precompile())
    "prebuilt": False,           # a SWEEP_BUILD_CUDA=1 ahead-of-time extension on disk
    "shipped_core": {"path": ".../sweep/lib/cu12/libsweep_core.so",
                     "reason": "ok", "tag": "cu12", "available": ["cu12", "cu13"]},
    "host_core": {"shipped": None,
                  "shipped_reason": "the install ships no host (CPU) core (no lib/cpu/libsweep_core.so)",
                  "cached": ".../sweep_C/core_cpu/libsweep_core.so",
                  "can_build": True, "build_reason": "ok (g++ 11)",
                  "unported": ["visco_acoustic2d"], "available": True},
}
```

## Equation-Level Binding Support

Not every equation exposes the compiled binding path.

Use the CLI to inspect support — the listing is generated live from the
installed package, so it always reflects the current environment:

```bash
sweep list equations
```

See [CLI · `sweep list equations`](cli.md#sweep-list-equations) for the full
table. Each row tells you:

- **Torch Binding** — whether the equation's source declares compiled-extension support.
- **Binding Ready** — whether `impl="c"` can run *right now* (`sweep.is_torch_binding_available()`: PyTorch, a visible CUDA GPU, and a fitting or buildable CUDA core).

## Choosing Between Torch and JAX

Choose `torch` when:

- you want the Torch ecosystem and autograd workflow
- you want to use `PropTorch(..., backend="torch", impl="eager")`
- you want to use the compiled C++/CUDA path through `PropTorch(..., backend="torch", impl="c")`

Choose `jax` when:

- you want JAX transformations and device placement
- you plan to use `PropJax`

## Extension Kernels in the Torch Family

Use the `c` implementation inside the Torch family when:

- a CUDA core is available (`sweep.backend.torch.binding.is_available()`)
- your equation supports the binding
- you want hand-written CUDA kernels or memory modes such as boundary
  saving, disk-backed boundary storage, or `c` checkpointing

`c` memory modes are equation-specific. Full-wavefield storage is available
across the `c`-backed solvers, and boundary saving across all of them except
`ViscoAcoustic` and `DASZhao3D` (their `c` default is full storage);
checkpoint modes are available where the equation exposes the corresponding
backward implementation. The user-facing entry point is:

```python
PropTorch(..., backend="torch", impl="c")
```

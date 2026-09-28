# Extending: Adding a New Equation

SWEEP treats equations as plug-in units: the propagator owns the time loop,
source/receiver wiring, PML, memory strategies, and (for `impl="c"`) the call
into the CUDA core. An equation class only has to describe **what fields it propagates**
and **how one time step advances them**.

The guide is in two parts you can read independently:

- **[Part 1](#part-1-python-only-eager-equation)** — the Python-only path.
  One file, no compiler, runs through `PropTorch(impl="eager")` or `PropJax`.
  This is enough for prototyping, JAX-only workflows, and anything that does
  not need hand-written CUDA kernels.
- **[Part 2](#part-2-adding-a-compiled-cuda-kernel-implc)** — what to add on
  top of Part 1 to enable `impl="c"`. Strictly additive: you do not rewrite
  any Part 1 code, just hand the propagator a compiled forward / backward.

For a **runnable walkthrough** that builds a toy `MyScalar` from Part 1 end
to end, see the
[Add a new equation notebook](../notebooks/18_extending_add_new_equation.ipynb).

## Registration is two lines

Both parts share the same registration mechanism: a decorator on the class,
and an import that makes the decorator run.

```python
# src/sweep/equations/my_scalar.py
from ._registry import register_equation

@register_equation()                     # 1. claim the public name
class MyScalar(SecondOrderEquation):
    ...
```

```python
# src/sweep/equations/__init__.py
from . import my_scalar                  # 2. import the module so it executes
```

`@register_equation()` takes the class name by default; pass `name=` to export
it under a different one, `aliases=(...)` for extra names that resolve to the
same class, and `method=` to tag a non finite-difference discretization
(`"sem"`). A name may only be claimed once, so a clash raises at import rather
than silently shadowing.

After both lines, `MyScalar` appears in:

- `sweep list equations` (CLI)
- `sweep.equations.equation_classes()` and `get_equation("MyScalar")`
- `sweep.equations.torch_binding_supported_equations()` — only if the class
  declares `C_NAME` (Part 2), which installs the `_C()` hook (`base.py`
  `__init_subclass__`)

Importing the class without the decorator binds a name in the
`sweep.equations` namespace and registers nothing: `equation_classes()` will
not list it, `get_equation` raises `KeyError`, and the CLI does not show it. Everything else below describes what
goes **inside** `my_scalar.py` (Part 1) and the CUDA directory (Part 2).

## Part 1 — Python-only (eager) equation

### When this is enough

Pick the Python-only path when any of the following is true:

- you are prototyping a new physics and want to iterate fast in pure Python
- you target `backend="jax"` and run through `PropJax`
- the model sizes you care about fit comfortably in full-wavefield memory,
  so you do not need boundary saving / disk-backed checkpoint
- you are happy with `torch.compile` (default) for speed and do not need
  hand-tuned CUDA kernels

You can ship the Python version, run real experiments with it, and add the
CUDA path (Part 2) later without touching the Python side.

### What you write

A new eager equation declares **four** things on the class. The
`wavefields` / `models` / `field_specs` / `model_specs` properties,
`init_abc`, PML profiles, and separable Laplace kernels are all inherited
from the base classes and derived from the spec tables automatically.

| Attribute / method | Purpose | Read by |
| --- | --- | --- |
| `MODEL_SPECS` (class attr) | ordered model tensor list with units / aliases | model validation, `available_models`; derives `models` / `model_specs` |
| `FIELD_SPECS` (class attr) | ordered wavefield list with source / receiver flags | source injection, receiver sampling, defaults; derives `wavefields` / `field_specs` |
| `default_pml_type` (class attr) | `"cpmlr"`, `"cpmls"`, or `"spml"` | PML profile initialisation |
| `func(wavefields, models, dt, h, b, **kwargs)` | one time step | `PropTorch` eager loop, `PropJax` step |

The order of `FIELD_SPECS` is semantically significant: `func` must return a
tuple in the same order, and receiver/source name → buffer index resolution
walks the same list.

### A minimal example

A 2-D scalar wave equation with one velocity model and a single pressure-like
field. Drop this into `src/sweep/equations/my_scalar.py`:

```python
from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec


class MyScalar(SecondOrderEquation):
    """Toy second-order scalar wave equation: u_tt = vp**2 * Laplace(u)."""

    MODEL_SPECS = (
        ModelSpec("vp", description="P-wave velocity.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec(
            "h1",
            description="Primary scalar wavefield.",
            supports_source=True,
            supports_receiver=True,
        ),
        FieldSpec(
            "h2",
            description="Previous-step scalar wavefield.",
            internal=True,
        ),
    )
    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        super().__init__(spatial_order, device, backend, dim=dim)
        self.init_separable_laplace()

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now, u_pre = wavefields
        (vp,) = models
        lap = self.laplacian(u_now, h)
        u_next = 2 * u_now - u_pre + (vp * dt) ** 2 * lap
        return u_next, u_now
```

This example deliberately ignores the PML coefficient buffer `b` so the time
step stays one line — the simulation simply has to stop before the wave
reaches the boundary. For a production-ready PML-coupled version, mirror
[`Acoustic.step_cpml`](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/acoustic.py)
and add the `psix / psiz / zetax / zetaz` CPML auxiliary fields to
`FIELD_SPECS`.

After your class is in place, add the decorator and the module import shown
under [Registration is two lines](#registration-is-two-lines). The
[Add a new equation notebook](../notebooks/18_extending_add_new_equation.ipynb)
runs exactly this class end-to-end against `PropTorch` and plots the
propagating P-wave ring.

### Verification (Part 1)

Before opening a PR:

1. `sweep list equations` shows your class.
2. `test/test_installation_smoke.py` still passes.
3. A small `solver_gradient_mode_suite`-style script compares your eager
   gradient on the canonical grid (`nz=48, nx=56, dh=10.0, dt=0.0015,
   nt=120`, Ricker `freq=10 Hz, delay=0.06 s`) against a finite-difference
   reference.
4. `mkdocs build` passes after you add a row to the summary table in
   [Equations](equations.md).

If all four pass, the Python-only equation is shippable.

## Part 2 — Adding a compiled CUDA kernel (`impl="c"`)

### When you need this

Add Part 2 on top of Part 1 when any of the following is true:

- you run FWI on production-size models where full-wavefield checkpointing
  exhausts GPU memory and you need
  [boundary saving / disk-backed boundary storage](propagators.md#memory-saving-features)
- you need `CkptOptions(mode="chunk")` or `CkptOptions(mode="recursive")`
  memory modes (only the C path exposes these)
- you want hand-written CUDA kernels for the forward / adjoint inner loop
  instead of relying on `torch.compile`

Part 2 is **strictly additive**. The Python `MyScalar` class from Part 1
keeps working as the eager fallback; the C path only adds new hooks.

### What you add to Part 1

Three additions on top of Part 1:

1. **Python side** — two class attributes: `C_NAME = "my_scalar"` (the
   entry-name prefix; `base.py` then installs `_C()` returning
   `my_scalar_forward / _backward / _backward_bs / _backward_ckpt /
   _backward_recursive_ckpt` from `sweep._C`; set
   `C_HAS_RECURSIVE_CKPT = False` if you skip the last one) and a
   `cuda_layout` property describing the buffer shapes.
2. **C++ / CUDA side** — a new equation directory under
   `src/sweep/csrc/cuda/equations/<my_scalar>/` following the existing
   convention (`<my_scalar>.h`, `driver_traits.cuh`, `kernels.cuh`,
   `kernels.cu`, `forward.cu`, `backward.cu`).
3. **Glue** — five entries in the core's C table: add `SWEEP_ENTRY_MY_SCALAR_*`
   ids to the `SweepEntry` enum in `src/sweep/csrc/core/capi.h` (and bump
   `SWEEP_ENTRY_COUNT`), and the matching rows in `ENTRY_NAMES` / `ENTRY_KINDS`
   / the `dispatch()` switch of `src/sweep/csrc/cuda/common/capi.cu`, plus one
   `#include` there. The ctypes layer (`sweep.backend.c`) reads that table from
   the loaded core by name, so nothing on the Python side lists your functions.
   Only if you also work on the `SWEEP_JIT_FULL=1` pybind shim do you add the
   five `m.def(...)` lines to `src/sweep/csrc/bindings/module.cpp`.

The core build globs `cuda/common/**/*.cu` and `cuda/equations/**/*.cu`
(`sweep/backend/c/jit.py` `_sources`), so you do **not** edit
`pyproject.toml`, `build_config.py` or `setup_cuda.py`. If you add a field to
`ForwardInput` / `BackwardInput` (`shared/wavetypes.h`) you must rerun
`utils/gen_input_core.py`, which regenerates `core/input_core.h`,
`cuda/common/adapt_inputs.h`, `cuda/common/layout.cu` and the ctypes mirror
`sweep/backend/c/abi.py`; the loader's ABI guard refuses a core whose struct
layout differs from the mirror.

### Python side (incremental)

Add `C_NAME` and `cuda_layout` to the same class you wrote in Part 1:

```python
class MyScalar(SecondOrderEquation):
    # ... Part 1 body (MODEL_SPECS / FIELD_SPECS / default_pml_type / func) ...

    C_NAME = "my_scalar"              # -> sweep._C.my_scalar_forward, _backward, _backward_bs, _backward_ckpt, _backward_recursive_ckpt
    # C_HAS_RECURSIVE_CKPT = False    # if you do not ship *_backward_recursive_ckpt

    @property
    def cuda_layout(self):
        from .cuda_layout import CUDALayoutSpec
        return CUDALayoutSpec(
            base_nvar=2,           # h1, h2
            pml_nvar=4,            # psix, psiz, zetax, zetaz (CPML aux)
            last_two_nvar=2,       # second-order time stencil keeps 2 history fields
            last_two_storage_nvar=1,  # only one needs to be stored
            checkpoint_nvar=6,
            boundary_save_nvar=1,
            backward_workspace_nvar=0,
        )
```

Set `C_HAS_RECURSIVE_CKPT = False` to skip the recursive-checkpoint entry
(`PropTorch` then refuses `Ckpt(mode="recursive")` for your equation); the other
four entries must exist in the core. A hand-written `_C()` still works if you
need something unusual (see `acoustic_curvilinear.py`).

#### `cuda_layout` fields

The propagator allocates GPU buffers entirely from `CUDALayoutSpec`. The
fields are read in `src/sweep/propagator/_c.py`:

| Field | Meaning | Default |
| --- | --- | --- |
| `base_nvar` | non-PML wavefield tensors per timestep | required |
| `pml_nvar` | CPML / SPML auxiliary tensors per timestep | required |
| `last_two_nvar` | size of the rolling "last two snapshots" buffer | required |
| `last_two_storage_nvar` | tensors actually written into that buffer | `base_nvar` |
| `checkpoint_nvar` | tensors saved per checkpoint | `base_nvar + pml_nvar` |
| `boundary_save_nvar` | distinct fields the boundary saver writes per step | `base_nvar` (set to `1` for 2nd-order scalar equations) |
| `backward_workspace_nvar` | adjoint workspace tensors | `0` |
| `backward_workspace_shapes` | optional shape callback for the workspace | `None` |
| `boundary_tangent_pad` | extra tangential cells for staggered grids | `0` |
| `adjoint_extra_nvar` | extra adjoint-only tensors for a fused adjoint | `0` |
| `pml_slot_axes` | per-slot differencing axis (`'x'`/`'y'`/`'z'`) of the `pml_nvar` **forward** aux slots, in C++ bind order — tagged slots are allocated as per-axis slabs (PML band + stencil reach) instead of full grids | `None` (full grids) |
| `checkpoint_slot_axes` | the same tagging for the checkpoint snapshot slots; `None` entries stay physical full-grid slots | `None` |
| `adjoint_pml_slab` | also slab the **adjoint** aux. Only safe when the adjoint touches its memory variables own-cell (elastic); a fused adjoint that stencil-taps psi/zeta must stay full-domain (acoustic) | `False` |
| `stepped` | the compiled forward / `backward_bs` honour the `it_begin` / `it_end` segment range (the shared template drivers do) — required for domain decomposition | `False` |
| `dd_backward_phases` | the compiled backward implements the numbered phases its DD schedule drives (elastic physics split, VRZ coupling exchange) | `False` |

The full field list (`forward_workspace_nvar` / `_shapes`, `derived_model_nvar`,
`record_shape`, `checkpoint_replay_shapes`, `checkpoint_state_nvar`,
`save_all_shape`, `bs_reconstruction_nvar`, `supports_boundary_tail_steps`,
`slots`, ...) is documented on the `CUDALayoutSpec` dataclass in
`src/sweep/equations/cuda_layout.py`.

`pml_slot_axes`, `checkpoint_slot_axes` and `adjoint_pml_slab` are opt-in: leave them unset and your aux buffers are
full-domain, which always works. Tagging them cuts CPML aux memory to the
bands, and is what `Acoustic`/`Acoustic3D`/`Elastic`/`Elastic3D` do — the
kernels adapt per bound tensor, so a mis-tagged slot shows up as a wrong
read, not as a silent full-grid fallback.

If the buffer counts are wrong, the propagator either over-allocates GPU
memory or reads uninitialised data — there is no second line of defence, so
double-check these against your CUDA kernels' actual reads/writes.

### C++ / CUDA side

Create `src/sweep/csrc/cuda/equations/my_scalar/` and follow the existing
convention (`<dir>/<dir>.h` plus `driver_traits.cuh`, `kernels.cuh`,
`kernels.cu`, `forward.cu`, `backward.cu`). Sizes below are typical orders of
magnitude; the 19 existing equations under `src/sweep/csrc/cuda/equations/`
(e.g. `acoustic2d/`) are direct references.

| File | Role | Typical size |
| --- | --- | --- |
| `my_scalar.h` | the `*_core` entry declarations (forward / backward variants, runner factories) | ~30 lines |
| `driver_traits.cuh` | the equation's traits for the shared template drivers (`common/eq_driver.cuh` / `sg_driver.cuh`) | varies |
| `kernels.cuh` / `kernels.cu` | per-step CUDA kernels (state update, gradient / imaging) | 300–1200 lines |
| `forward.cu` | forward entry: initialisation, receiver write-out | 200–300 lines |
| `backward.cu` | adjoint entries, parameter-gradient accumulation, RTM path | 800–1500 lines |

The core build (`sweep/backend/c/jit.py` `_sources`) globs
`cuda/common/**/*.cu` + `cuda/equations/**/*.cu`; the same list feeds
`python -m sweep.build --core`.

#### Reusable infrastructure

The headers under `src/sweep/csrc/cuda/common/` and `cuda/operators/` cover
roughly 70–80 % of the boilerplate for a new CUDA equation. Reach for these
before re-inventing:

- `common/context.h` — `SolverContext` (grid sizes, dt, PML widths, FD half-stencil)
- `operators/laplace.cuh`, `gradient.cuh`, `staggered.cuh` — templated FD operators
- `common/eq_driver.cuh` / `sg_driver.cuh` — the shared template drivers (second-order / staggered families) that make an equation `stepped`
- `common/boundarysaver.cuh` / `.cu`, `common/boundary/` — ring-buffer boundary save / load (GPU / CPU / disk) and the session runtime
- `common/checkpoint_runtime.cuh` — checkpoint allocation and replay
- equation-family CPML / wavefield structs: `common/acoustic.h`, `common/elastic.h`

What you write per equation is the state-update kernel and the
parameter-gradient kernel — the rest is reuse.

### Glue: register in the C API table

```cpp
// src/sweep/csrc/core/capi.h  (enum SweepEntry) -- append after the last equation
SWEEP_ENTRY_MY_SCALAR_FORWARD = 94,
SWEEP_ENTRY_MY_SCALAR_BACKWARD = 95,
SWEEP_ENTRY_MY_SCALAR_BACKWARD_BS = 96,
SWEEP_ENTRY_MY_SCALAR_BACKWARD_CKPT = 97,
SWEEP_ENTRY_MY_SCALAR_BACKWARD_RECURSIVE_CKPT = 98,
SWEEP_ENTRY_COUNT = 99

// src/sweep/csrc/cuda/common/capi.cu
#include "../equations/my_scalar/my_scalar.h"
// ENTRY_NAMES: "my_scalar_forward", "my_scalar_backward", "my_scalar_backward_bs",
//              "my_scalar_backward_ckpt", "my_scalar_backward_recursive_ckpt"
// ENTRY_KINDS: 0, 1, 1, 1, 1
// dispatch(): one case per entry, e.g.
case 94: *static_cast<ForwardOutputCore*>(out) = my_scalar::forward_core(*static_cast<const ForwardInputCore*>(in)); return;
case 95: *static_cast<BackwardOutputCore*>(out) = my_scalar::backward_core(*static_cast<const BackwardInputCore*>(in)); return;
// ... _backward_bs, _backward_ckpt, _backward_recursive_ckpt likewise
```

The ctypes layer (`sweep.backend.c`) enumerates the table through
`sweep_entry_count()` / `sweep_entry_name()` at load, so the Python names need
no further wiring. `bindings/module.cpp` (the pybind shim) mirrors the same
table and is only compiled under `SWEEP_JIT_FULL=1`; keep it in step by hand if
you use that path (the generator that produced it is not in the repository).

### Build + load

Rebuild the CUDA core from this tree (incremental; needs an nvcc >= 12.4 of your
torch's CUDA major; no GPU needed):

```bash
python -m sweep.build --core --cuda-home /usr/local/cuda-12.9   # -> src/sweep/lib/cu12/ (+ core.json)
# or, with no core under src/sweep/lib/, the first impl='c' call / `python -m sweep.build`
# builds and caches one under TORCH_EXTENSIONS_DIR
# or point at a core built elsewhere:
SWEEP_CORE=/path/to/libsweep_core.so python ...
```

A shipped core under `src/sweep/lib/<tag>/` wins over the local build, so after
editing csrc either rerun `--core` or remove that directory. `SWEEP_JIT_FULL=1`
is needed only to rebuild the pybind shim / CPU engine.

Then verify the binding is wired up:

```python
from sweep.equations import supports_torch_binding, MyScalar
assert supports_torch_binding(MyScalar)
```

### Verification (Part 2)

On top of Part 1's checks:

1. `sweep list equations` shows your class with **Torch Binding ✓** in the
   table.
2. `sweep.backend.torch.binding.is_available()` returns `True`, and
   `from sweep._C import my_scalar_forward` succeeds (the first attribute
   access loads the core and its entry table — a missing entry fails here by
   name).
3. `test/solver_gradient_mode_suite.py`-style test compares your eager
   gradient (Part 1) against the compiled full-wavefield, boundary saving,
   and (if implemented) checkpoint modes on the canonical grid
   (`nz=48, nx=56, dh=10.0, dt=0.0015, nt=120`, Ricker
   `freq=10 Hz, delay=0.06 s`). Acceptance thresholds: `rel_l2 < 1.5`,
   `cosine_similarity > 0.8` per mode.
4. `mkdocs build` still passes.

## Equation capability flags

`EquationBase` carries a small set of class attributes the propagator reads
to decide what your equation is allowed to do. Override them on your class
when the corresponding capability is implemented (or, for
``supports_free_surface``, when it is *not* valid):

| Flag | Default | Meaning |
|---|---|---|
| `supports_free_surface` | `True` | Whether a free surface is physically valid for this equation at all. **Set `False` for anisotropic media** — the isotropic image method does not satisfy the anisotropic stress-free condition, so the propagator raises `NotImplementedError` on any `free_surface=` request (see `AcousticVTI1st`, `ElasticTTI`). |
| `supports_per_edge_free_surface` | `False` | Opt-in: the eager `func` honours `fs_faces` (a free surface on any subset of the faces, each with its own PML pad) rather than only the top-only bool. `Acoustic` and `Elastic` (2-D) opt in. |
| `supports_per_edge_free_surface_c` | `False` | Same, for the compiled `impl='c'` CUDA path — set `True` only once the CUDA forward **and** adjoint honour `fs_faces`. |
| `supports_apm` | `False` | Parameter-modified (APM) irregular topography path — see below. |
| `supports_batched_models` | `False` | The CUDA kernels accept per-shot batched models `(B, *spatial)` in addition to a single shared model. |
| `default_pml_type` | `"cpmlr"` | PML formulation used when the propagator is not given one explicitly. |

## Out of scope here

A few features require touching the propagator base or `_c.py` rather than
just the equation:

- **Irregular free-surface topography (APM / curvilinear).** Set
  `supports_apm = True` and implement `_C_apm()` returning `(forward,
  backward, backward_bs)`. See `ElasticAPM` and `ElasticCurvilinear` for
  references, plus the `topography=` and `free_surface=` plumbing in
  `src/sweep/propagator/base.py`.
- **RTM imaging.** No equation in the tree implements a `_C_rtm()` hook today;
  the core's C API reserves an `rtm` entry kind for it (`csrc/core/capi.h`,
  kind 2). This is a propagator-level addition.
- **A new memory mode** (beyond full-wavefield, boundary saving, and the
  two checkpoint modes). This is a propagator-level change, not an
  equation-level one.

These are platform extensions, not new equations, and intentionally sit
outside this guide.

## See also

- [Add a new equation notebook](../notebooks/18_extending_add_new_equation.ipynb)
  — runnable walkthrough for Part 1 (builds and runs `MyScalar` end-to-end).
- [Equations](equations.md) — user-facing summary of every shipped equation.
- [Propagators](propagators.md) — how `PropTorch` / `PropJax` consume the
  equation interface; details of the memory-saving features that motivate
  Part 2.
- [Backends](backends.md) — `impl="eager"` vs `impl="c"` choice.
- [`Acoustic` source](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/acoustic.py)
  — the smallest end-to-end reference, Part 1 + Part 2 in one file.

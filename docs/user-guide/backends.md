# Backends

SWEEP supports two main user-facing backend families:

- `torch`
- `jax`

Within the Torch family, `PropTorch` now selects the actual implementation:

- `impl=None` *(default — equivalent to `impl="auto"`)*: probe `sweep._C`;
  pick `"c"` when the compiled binding is importable, otherwise transparently
  fall back to `"eager"`.
- `impl="eager"`: pure-PyTorch implementation (no build step required).
- `impl="c"`: compiled C++/CUDA implementation through the Torch extension.
  If the binding is missing, `PropTorch` falls back to `"eager"` with a
  `UserWarning` so the slowdown is visible.

## User-Facing Backend Families

- `torch`: PyTorch-based propagation and differentiation
- `jax`: JAX-based propagation and differentiation

`cuda` is not a separate top-level backend alongside `torch` and `jax`. It is a
device choice. The compiled Torch extension can run C++ CPU kernels or CUDA
kernels depending on the tensor device.

Typical Torch-family usage:

```python
from sweep.propagator.torch import PropTorch

solver_eager = PropTorch(..., backend="torch", impl="eager")
solver_c = PropTorch(..., backend="torch", impl="c")
```

## Torch Extension Binding

Some equations provide compiled PyTorch extension kernels through `sweep._C`.

The extension binding currently covers the main 2D and 3D acoustic, VRZ,
LSRTM, elastic, and DAS propagators used by the Torch workflow. CPU tensors use
the C++ CPU kernels; CUDA tensors use the CUDA kernels.

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
`sweep.backend.torch.binding.is_available()` checks whether the compiled
`sweep._C` extension is importable.

Example diagnostics output:

```python
{
    "binding_importable": True,
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
- **Binding Ready** — whether that binding is loadable *right now* (i.e. `sweep._C` imports cleanly).

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

- the compiled binding is installed
- your equation supports the binding
- you want hand-written CPU/CUDA kernels or memory modes such as boundary
  saving, disk-backed boundary storage, or `c` checkpointing

`c` memory modes are equation-specific. Full-wavefield and
boundary-saving modes are available across the `c`-backed solvers;
checkpoint modes are available where the equation exposes the corresponding
backward implementation. The user-facing entry point is:

```python
PropTorch(..., backend="torch", impl="c")
```

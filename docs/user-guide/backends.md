# Backends

SWEEP supports two main user-facing backend families:

- `torch`
- `jax`

Within the Torch family, `PropTorch` now selects the actual execution mode:

- `backend="eager"`: Python/Torch implementation
- `backend="cuda"`: compiled CUDA implementation through the Torch binding

## User-Facing Backend Families

- `torch`: PyTorch-based propagation and differentiation
- `jax`: JAX-based propagation and differentiation

`cuda` is no longer treated here as a separate top-level backend alongside
`torch` and `jax`. Instead, it is the accelerated execution mode inside the
Torch family.

Typical Torch-family usage:

```python
from sweep.propagator.torch import PropTorch

solver_eager = PropTorch(..., backend="eager")
solver_cuda = PropTorch(..., backend="cuda")
```

## CUDA Binding

Some equations provide a compiled PyTorch CUDA binding through `sweep._C`.

The CUDA binding currently covers the main 2D and 3D acoustic, VRZ, LSRTM, and
elastic propagators used by the Torch workflow.

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

Use the CLI to inspect support:

```bash
sweep list equations
```

Example output:

```text
Available equations:

  Equation       Models               Torch Binding  Binding Ready
  -------------  -------------------  -------------  -------------
  Acoustic       ['vp']               yes            yes
  Acoustic1st    ['vp', 'rho']        no             no
  Acoustic3D     ['vp']               yes            yes
  AcousticLSRTM  ['vp', 'mp']         yes            yes
  AcousticLSRTM3D ['vp', 'mp']        yes            yes
  AcousticVRZ    ['vp', 'z']          yes            yes
  AcousticVRZ3D  ['vp', 'z']          yes            yes
  Elastic        ['vp', 'vs', 'rho']  yes            yes
  Elastic3D      ['vp', 'vs', 'rho']  yes            yes
```

The table distinguishes:

- Whether an equation supports torch binding
- Whether the current environment can actually use it

## Choosing Between Torch and JAX

Choose `torch` when:

- you want the Torch ecosystem and autograd workflow
- you want to use `PropTorch(..., backend="eager")`
- you want to use the compiled CUDA path through `PropTorch(..., backend="cuda")`

Choose `jax` when:

- you want JAX transformations and device placement
- you plan to use `PropJax`

## CUDA in the Torch Family

Use the CUDA execution mode inside the Torch family when:

- the compiled binding is installed
- your equation supports the binding
- you want CUDA-specific features such as boundary saving, disk-backed boundary
  storage, or CUDA checkpointing

CUDA memory modes are equation-specific. Full-wavefield and boundary-saving
modes are available across the CUDA-backed solvers; checkpoint modes are
available where the equation exposes the corresponding backward implementation.

The lower-level class `sweep.propagator.cuda.PropCUDA` still exists, but for
most user-facing workflows the recommended entry point is:

```python
PropTorch(..., backend="cuda")
```

# Backends

SWEEP supports multiple execution paths.

## Python Backends

- `torch`: PyTorch-based propagation and differentiation
- `jax`: JAX-based propagation and differentiation
- `cuda`: PyTorch-based CUDA binding for propagation and differentiation

## CUDA Binding

Some equations provide a compiled PyTorch CUDA binding through `sweep._C`.

At the moment, PyTorch CUDA binding support is limited to 2D/3D acoustic and 2D/3D elastic wave equations.

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
  AcousticLSRTM  ['vp', 'mp']         no             no
  AcousticVRZ    ['vp', 'z']          yes            yes
  Elastic        ['vp', 'vs', 'rho']  yes            yes
  Elastic3D      ['vp', 'vs', 'rho']  yes            yes
```

The table distinguishes:

- Whether an equation supports torch binding
- Whether the current environment can actually use it

## Notes To Expand

- Backend-specific dependencies
- CPU vs GPU behavior
- When to use PyTorch bindings instead of pure Python paths

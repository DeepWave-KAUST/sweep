# Backends

SWEEP supports multiple execution paths.

## Python Backends

- `torch`: PyTorch-based propagation and differentiation
- `jax`: JAX-based propagation and differentiation

## CUDA Binding

Some equations provide a compiled PyTorch CUDA binding through `sweep._C`.

At the moment, PyTorch CUDA binding support is limited to 2D/3D acoustic and 2D/3D elastic wave equations.

You can inspect backend capability from Python:

```python
import sweep

sweep.backend.torch.is_available()
sweep.backend.jax.is_available()
sweep.backend.torch.cuda.is_available()
```

## Equation-Level Binding Support

Not every equation exposes the compiled binding path.

Use the CLI to inspect support:

```bash
sweep list equations
```

The table distinguishes:

- Whether an equation supports torch binding
- Whether the current environment can actually use it

## Notes To Expand

- Backend-specific dependencies
- CPU vs GPU behavior
- When to use PyTorch bindings instead of pure Python paths

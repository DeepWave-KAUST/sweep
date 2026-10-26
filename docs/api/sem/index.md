# Spectral element (SEM)

`sweep.sem` is the spectral-element solver: a SPECFEM-style
continuous-Galerkin SEM on quad (2-D) and hex (3-D) meshes with
Gauss-Lobatto-Legendre quadrature, next to the finite-difference
`sweep.equations`. The two do not share base classes, so the SEM equations are
not `WaveEquation` subclasses and run with `PropSEM`, not `PropTorch`. Usage
is in the [SEM propagator](../../user-guide/sem.md) guide; one forward and one
gradient end to end is
[Hello · SEM](../../notebooks/43_sem_forward_backward.ipynb).

| Page | What is on it |
| --- | --- |
| [PropSEM](prop_sem.md) | The solver, with `PropTorch`'s constructor and call, and `suggest_discretization` for the mesh and time step |
| [SEM equations](equations.md) | The equation classes passed to `PropSEM`, and the registry helpers |
| [Meshes, basis and C-PML](mesh.md) | The GLL basis, the Quad4 / Quad9 / Hex8 / Hex27 meshes and their builders, the C-PML profiles |
| [SEMPropagator](low_level.md) | The low-level time loop under `PropSEM`, the point and moment stamps, boundary saving |
| [Parallel (DD)](parallel.md) | Splitting one model across GPUs on `impl='c'` |

Every SEM equation, with its mesh family, models and CUDA support, is listed by
the CLI next to the FD ones; `sweep show` describes one:

```bash
sweep list equations
sweep show AcousticPotentialSEM2D
```

See [CLI](../../user-guide/cli.md).

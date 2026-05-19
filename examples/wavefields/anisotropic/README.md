# Anisotropic wavefield examples

This directory contains compact forward-modeling examples for anisotropic
wave equations.

## Scripts

- `anisotropic_wavefields.py`: acoustic qP examples for Tariq, VTI, and TTI.
- `elastic_tti_wavefields.py`: representative 2D three-component elastic TTI
  experiments on rotated staggered grid (RSG) and standard staggered grid (SG).

### `AcousticVTI1st` (Duveneck 2008, first-order velocity-stress) examples

| Script | What it shows |
|--|--|
| `duveneck_vti_wavefield.py` | Reproduces Duveneck (2008) Fig. 1 — homogeneous VTI snapshot at t = 0.6 s with the diamond-shaped V_S = 0 S-wave artefact clearly visible on all four panels (-σ_V, -σ_H, v_x, v_z). |
| `duveneck_vti_shear_suppression.py` | Reproduces Duveneck (2008) Fig. 2 — same setup but with the δ → ε source-region taper (`smooth_delta_to_epsilon_disk`, r=8 cells). Prints the diamond/P-front amplitude-ratio drop (~140× in the canonical run). |
| `duveneck_vti_pml_absorption.py` | Two-case CPML absorption test: (A) isotropic medium gives ~5.7×10⁵× residual drop (machine-precision absorption); (B) VTI medium with shear-artefact taper still gets ~85× absorption. Regression for the PML coefficients. |
| `duveneck_vti_backward_gradients.py` | eager-autograd vs CUDA backward gradient comparison on the canonical `solver_gradient_mode_suite` 2-D grid (nz=48, nx=56, nt=120, dh=10 m, dt=1.5 ms, abcn=30, Ricker f=10 Hz). Saves two PNGs to `outputs/`: a 4×3 full-field grid (`vti_gradients_eager_vs_cuda.png`) and a 1-D cross-section (`vti_gradients_xsection.png`). |

#### Backends and memory modes

`AcousticVTI1st` is available through both the eager Python (PyTorch
autograd) path and the compiled CUDA extension. Pick the path via `impl=`:

| backend | impl | device | notes |
|--|--|--|--|
| `torch` | `eager` | CPU or CUDA | PyTorch step + autograd. Reference. |
| `torch` | `c` | CUDA only | Compiled kernel; ~60× faster forward on a 401×401 grid (RTX 6000 Ada). 2-D only. |
| `jax` | — | CPU or CUDA | Equivalent JAX step function (no compiled kernel). |

The CUDA path supports three backward memory strategies via
`PropTorch(memory_options=...)`:

- **full** (default `None`): saves every forward wavefield; cleanest, biggest memory.
- **boundary saving** (`MemoryOptions(strategy="boundary", boundary=BoundaryOptions(...))`): forward saves PML-band slabs only; backward re-propagates the interior in reverse time.
- **chunk checkpoint** (`MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=N))`): forward saves a checkpoint every `chunks` steps; backward replays each chunk and runs adjoint over it.

The `recursive` checkpoint mode is currently a `TORCH_CHECK` stub.

Run any script with source-tree imports:

```bash
PYTHONPATH=/path/to/sweep/src \
  python examples/wavefields/anisotropic/duveneck_vti_wavefield.py
```

Outputs land in `outputs/`.

## Elastic TTI examples

Run both representative experiments:

```bash
python elastic_tti_wavefields.py --device cuda
```

Run only the rotation experiment:

```bash
python elastic_tti_wavefields.py --experiment rotation --device cuda
```

Run only the free-surface comparison:

```bash
python elastic_tti_wavefields.py --experiment free-surface --device cuda
```

For a quick smoke run:

```bash
python elastic_tti_wavefields.py --quick --device cuda
```

The script writes compact figures and a metrics text file into `outputs/`.

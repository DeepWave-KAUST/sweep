# Curvilinear-Grid Free-Surface Topography — Stage 3

**Branch**: `feat/topography-curvilinear` (from `feat/topography-acoustic2d`)
**Owner**: wangs0j
**Date**: 2026-05-22
**Status**: design

---

## 0. Goal

Replace the staircase-vacuum (acoustic) / staircase-image-method (elastic)
free-surface handling with a **boundary-fitted curvilinear coordinate
transform**. The physical (X, Z) domain with irregular top boundary maps
onto a rectangular computational (ξ, η) domain where η = 0 is the surface.

This eliminates the two known limitations of Stage 1+2:

  1. Acoustic staircase corner diffraction noise above ~30° slopes
  2. Elastic staircase image-method late-time exponential instability

Both equations need:

  - A new equation class (`AcousticCurvilinear`, `ElasticCurvilinear`)
    that applies metric chain-rule terms inside `step()`
  - A precomputed metric tensor (α, β, plus first derivatives) attached
    to the propagator
  - No change to FD operator primitives, source/receiver mechanics, or
    CPML on the side/bottom (top stays free-surface, no PML needed)

## 1. Coordinate transform

The user supplies an integer-row topography ``topo[ix]`` (same API as
Stage 1+2). We map it to:

```
X = ξ
Z = z_top(ξ) + (z_bottom − z_top(ξ)) · η,  η ∈ [0, 1]
```

where ``z_top(ξ) = topo[ix] · dh``. Both ξ and η are sampled on the same
``(nz_phys, nx_phys)`` grid the user already uses; only the *physical*
spacing in z varies per column (cells are taller in mountainous regions,
shorter in valley regions — standard boundary-fitted grid).

Inverse metrics (the only things that enter the PDE):

```
∂ξ/∂X = 1
∂ξ/∂Z = 0
∂η/∂X ≡ α(ξ, η) = −h'(ξ)·(1 − η) / D(ξ)
∂η/∂Z ≡ β(ξ)    = 1 / D(ξ)
where  h(ξ) = z_top,  D(ξ) = z_bottom − z_top
```

Chain rule:

```
∂f/∂X = f_ξ + α · f_η          ∂f/∂Z = β · f_η
```

## 2. Acoustic 2-nd order PDE in (ξ, η)

```
∂²p/∂t² = vp² · Δ_phys p
Δ_phys p = p_ξξ + 2α p_ξη + (α² + β²) p_ηη + (α_ξ + α α_η) p_η
```

Operators we need on the (ξ, η) computational grid:

  - p_ξ, p_η (first derivatives — reuse `pd.x_central` / `pd.z_central`)
  - p_ξξ, p_ηη (separable laplacian — reuse `laplace1d_sep`)
  - **p_ξη — mixed partial** (compute as ``pd.x_central(p_η)``)

Precomputed metric fields stored on the propagator:

| Field | Shape | Definition |
|---|---|---|
| `alpha`        | (nz, nx) | α(ξ, η) |
| `beta`         | (nx,)    | β(ξ) (η-independent) |
| `alpha_xi`     | (nz, nx) | ∂α/∂ξ |
| `alpha_eta`    | (nz, nx) | ∂α/∂η |
| `metric_pηη`   | (nz, nx) | α² + β² |
| `metric_pη`    | (nz, nx) | α_ξ + α · α_η |

(`beta` broadcasts in z; the others are per-cell.)

Free-surface BC: ``p = 0`` at η = 0, enforced exactly by the existing
flat `zero_top_halo_fields` (no topo-aware logic). The metrics handle
the geometry; the BC is on a flat row.

## 3. Elastic 1-st order PDE in (ξ, η)

Each ``∂/∂X`` and ``∂/∂Z`` in the existing `step()` is replaced:

```
v_t   = (1/ρ) · ∇·σ        with ∂σ/∂X = (σ)_ξ + α (σ)_η
                                ∂σ/∂Z = β (σ)_η
σ_t   = stiffness · ∇v      symmetric — same chain rule
```

Free-surface BC: σ_zz = σ_xz = 0 at η = 0, enforced by the existing flat
``top_free_surface_derivative`` (image-method mirror on a flat row) +
``zero_top_row``. **No more staircase corners → no instability.**

For staggered metrics: in the first cut we evaluate all metric fields
at cell centers and use them for every derivative regardless of
staggered offset. The accuracy hit is O(dh) at the surface (acceptable
for the demo). A future refinement could interpolate metrics to
half-grid positions.

## 4. Architecture

### New files

| File | Purpose |
|---|---|
| `src/sweep/utils/curvilinear.py` | `CurvilinearGrid` utility: takes physical topography + grid size + dh, returns the six metric tensors |
| `src/sweep/equations/acoustic_curvilinear.py` | `AcousticCurvilinear` equation class |
| `src/sweep/equations/elastic_curvilinear.py` | `ElasticCurvilinear` equation class |
| `test/test_acoustic_curvilinear.py` | Unit + integration tests |
| `test/test_elastic_curvilinear.py` | Unit + integration tests |
| `examples/wavefields/topography/acoustic2d_curvilinear_demo.py` | QC images |
| `examples/wavefields/topography/elastic2d_curvilinear_demo.py` | QC images |

### Modified files (minimal)

- `src/sweep/propagator/base.py`: in `_process_topography`, when the
  equation is a curvilinear variant, build the `CurvilinearGrid` and
  mirror its tensors onto the equation. Skip the staircase
  `_topo_rows_runtime` setup.
- `src/sweep/equations/__init__.py`: export the new classes.

### User API

Curvilinear is opt-in by choosing the curvilinear equation class:

```python
from sweep.equations import AcousticCurvilinear
eq = AcousticCurvilinear(spatial_order=4, device='cuda', backend='torch')
prop = PropTorch(
    eq,
    shape=(nz, nx),
    topography=topo,          # SAME API as Stage 1+2
    free_surface=True,
    abcn=30, dh=10, dt=1e-3,
    impl='eager',
)
syn = prop(wavelet, sources, receivers, models=[vp])
```

`topography=None` with a curvilinear equation degenerates to a flat-grid
identity transform (metrics α = α_ξ = α_η = 0, β = 1/D) — the simulation
must agree with `Acoustic(...)` flat case bit-for-bit (regression
guard).

## 5. Test matrix

For each equation (acoustic, elastic):

1. **Flat topo (topo = 0) ⇔ standard equation** — receiver record matches
   to 1e-5 (acoustic) / 1e-4 (elastic).
2. **Flat topo (topo = K · ones) ⇔ shifted standard equation** —
   translation invariance, like Stage 1+2.
3. **Gentle Gaussian hill** — short forward pass, check finite + no NaN.
4. **Long-time stability** — same hill, NT = 4000 (~3 s), `|v|max` stays
   bounded over time (no exponential growth) — this is the Stage 3 win.
5. **Gradient consistency** — autograd through curvilinear path, vp
   gradient finite + non-trivial.

## 6. Milestones (this branch)

| MS | Deliverable | Done when |
|---|---|---|
| M0 | This document, branch ready | now |
| M1 | `CurvilinearGrid` utility + unit tests for metric values | hand-computed test passes |
| M2 | `AcousticCurvilinear` + flat-degenerate test passes | record matches Acoustic flat |
| M3 | Long-time hill stability test passes | NT=4000 stays bounded |
| M4 | AcousticCurvilinear demo + QC images | model.png, record.png, wavefield.gif |
| M5 | `ElasticCurvilinear` + flat-degenerate test passes | record matches Elastic flat |
| M6 | Long-time elastic hill stability test passes | NT=4000 stays bounded; Rayleigh visible |
| M7 | ElasticCurvilinear demo + QC images | model.png, record.png, wavefield.gif |
| M8 | PR-ready commit | reviewer-ready diff against `feat/topography-acoustic2d` |

## 7. Known limitations of Stage 3 (out of scope)

  - **CUDA path** — eager torch only, like Stage 1+2. `impl='c'` raises
    `NotImplementedError`. Stage 4 would add CUDA support.
  - **3D** — 2-D only. 3-D adds one more inverse metric and the mixed
    partials proliferate.
  - **Other equation families** (VTI, TTI, DAS, viscoelastic) — not
    addressed. Once the framework is in place, each is its own
    integration task (chain-rule each derivative call).
  - **Smooth-grid generation beyond linear stretch** — the simple
    ``Z = z_top + D·η`` mapping is good enough for moderate topography.
    For very rough surfaces or near-vertical features, a transfinite
    interpolation or elliptic-grid generation would give better grid
    quality.
  - **Metric staggering for elastic** — first cut uses cell-centred
    metrics for all staggered derivatives. Half-grid metric
    interpolation is a future refinement.

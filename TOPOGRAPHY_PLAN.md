# Free-Surface Topography — Acoustic2d Python-side MVP

**Branch**: `feat/topography-acoustic2d` (from `dev`)
**Owner**: wangs0j
**Date**: 2026-05-22
**Status**: design / pre-implementation

---

## 0. Goal

Add support for **irregular (undulating) free-surface topography** to
`sweep.equations.Acoustic` (2nd-order pressure formulation, 2D) on the
**Python eager backend only** (`impl='python'` / torch). Existing flat
free-surface behavior is preserved bit-for-bit when the new `topography`
argument is `None`.

This is Stage 1 of a staged effort. CUDA, 3D, elastic, and VTI/TTI are
explicitly out of scope here; see §7.

## 1. Public API

Single new keyword argument on `Propagator`:

```python
prop = Propagator(
    equation=Acoustic(spatial_order=4),
    shape=(nz, nx),
    free_surface=True,
    topography=topo_rows,   # NEW: (nx,) int → padded-grid row index, or None
    dh=10.0, dt=0.0015,
    ...
)
```

- `topography=None` (default) → existing flat free-surface code path, unchanged.
- `topography` is a 1-D array of length `nx_phys`, integer dtype, giving the
  surface row index in **physical-grid** coordinates (0 = topmost row,
  growing downward). Values must satisfy `0 <= topo < nz_phys - so//2`.
- Internal storage: `self._topo_rows: torch.LongTensor` on the same device
  as the equation; a derived `self._topo_rows_padded` shifted to padded-grid
  coords (no shift in z because `padding_z = (0, abcn)` with `free_surface=True`).

Float-elevation input (meters) and subgrid placement are deferred to Stage 1.5.

## 2. File-by-file changes

All changes are Python; no CUDA, no C++.

### Modified

| File | Change | Approx LOC |
|---|---|---|
| `src/sweep/propagator/options.py` | `PROP_DEFAULTS` gains `topography=None` | +2 |
| `src/sweep/propagator/base.py` | `__init__` accepts/validates `topography`; stores `self._topo_rows_padded`; new `_validate_topography()` helper | +30 |
| `src/sweep/equations/_free_surface.py` | Add `extend_top_free_surface_topo(u, halo, odd, axis, iz_surf)` and `zero_above_topo(u, iz_surf, axis)`; keep existing helpers untouched | +50 |
| `src/sweep/equations/acoustic.py` | In `step()`: if `self.topography is not None`, route the z-derivative through `extend_top_free_surface_topo` and replace `zero_top_halo_fields` with `zero_above_topo` | +25 |
| `src/sweep/equations/utils.py` | Tiny re-export / shim if needed | +5 |
| `src/sweep/propagator/_c.py` | Guard: `topography is not None` → `NotImplementedError("topography requires impl='python'; CUDA path is Stage 2")` | +5 |
| `src/sweep/propagator/_torch_eager.py` | Pass `prop._topo_rows_padded` into the equation context if equation doesn't already pull it via `self.topography` | +5 |

### New tests (under `sweep/test/`)

| File | Purpose |
|---|---|
| `topography_acoustic2d_degenerate_test.py` | `topography = halo*ones(nx)` must match `topography=None` to `1e-6` |
| `topography_acoustic2d_hill_test.py` | Gaussian hill; verify (a) pressure stays ~0 above surface, (b) primary reflection arrival time ≈ analytic, (c) free-surface reflection phase polarity correct |
| `topography_acoustic2d_gradient_test.py` | FWI gradient consistency under canonical config (see §4.4) |

## 3. Algorithm

### 3.1 Per-column image-method mirror

Given padded field `u` of shape `(..., nz, nx)` and surface row `iz_surf[ix]`,
build a virtual extended field where air cells (`z < iz_surf[ix]`) are
filled with the reflection of the interior across the local surface:

```python
def extend_top_free_surface_topo(u, halo, odd, axis, iz_surf):
    """
    u:        (..., nz, nx), padded-grid tensor (torch)
    iz_surf:  (nx,) LongTensor — surface row per column in padded coords
    halo:     stencil half-width (= equation.so // 2)
    odd:      True for z-anti-symmetric fields (vz on staggered; not used by
              2nd-order acoustic but kept for API symmetry with the flat helper)
    axis:     z-axis index, e.g. -2 for 2D (..., nz, nx)
    """
    nz = u.shape[axis]
    # Build broadcast-able z and surf tensors
    z = torch.arange(nz, device=u.device).view(
        *([1] * (u.ndim + axis if axis < 0 else axis)), nz,
        *([1] * (-axis - 1 if axis < 0 else u.ndim - axis - 1))
    )
    surf = iz_surf.view(
        *([1] * (u.ndim + axis if axis < 0 else axis)), 1,
        *iz_surf.shape  # shape (nx,) -> trailing
    )
    above = z < surf
    mirror_z = torch.where(above, 2 * surf - z, z).clamp(0, nz - 1)
    mirror_z = mirror_z.expand_as(u)
    out = u.gather(axis, mirror_z)
    if odd:
        out = torch.where(above.expand_as(u), -out, out)
    return out
```

For 2nd-order acoustic the field is pressure (even parity → `odd=False`).
The mirror is then differentiated by the existing FD operator:

```python
dpdz = top_free_surface_derivative(p, pd.z_laplace, halo, odd=False, axis=-2)
# becomes →
ext = extend_top_free_surface_topo(p, halo, odd=False, axis=-2, iz_surf=self._topo_rows_padded)
dpdz = pd.z_laplace(ext)
```

(The existing `top_free_surface_derivative` is one-liner sugar; the new helper
matches its semantics.)

### 3.2 Air-region masking

After each time step, zero pressure in cells **strictly above** the surface
in each column. This prevents accumulation of any residual from numerical
boundary errors:

```python
def zero_above_topo(u, iz_surf, axis):
    nz = u.shape[axis]
    z = torch.arange(nz, device=u.device).view(...)  # same broadcast as above
    mask = z < iz_surf.view(...)
    return u.masked_fill(mask.expand_as(u), 0.0)
```

The existing `zero_top_halo_fields` (clears a single row) is **replaced**
by `zero_above_topo` when `topography is not None`, keeping flat behavior
when degenerate.

### 3.3 Index conventions and PML interaction

- `free_surface=True` already sets `padding_z = (0, abcn)`, so physical row
  `iz_phys=0` maps to padded row `iz_pad=0`. No additional shift needed for
  the topo array; `self._topo_rows_padded = self._topo_rows + _runtime_fd_halo`
  (only the runtime FD halo, if any, is added).
- Bottom and side PML are untouched. `iz_surf < nz_phys - so//2` is enforced
  so the FD stencil at the surface never touches the bottom-PML or side
  ghost zones beyond what the flat case already does.
- The mirror is applied **per column independently**; no smoothing across
  columns. Staircase artifacts are expected for steep slopes (see §6).

### 3.4 Source / receiver placement

Out of scope for this stage. Users specify `(iz, ix)` indices as today; for
on-surface sources they should pass `iz = topo_rows[ix_src]`. A helper
`sweep.utils.surface_index(x_idx, topo_rows)` is a Stage 1.5 add-on.

## 4. Tests and acceptance

### 4.1 M1 — unit tests for `_free_surface.extend_top_free_surface_topo`

Pure numpy/torch tests, no propagator:
- `iz_surf = halo * ones(nx)` against existing `extend_top_free_surface` → `allclose 1e-7`
- Hand-computed reflection for a 3×5 toy tensor with `iz_surf = [1, 1, 2, 2, 1]`
- `odd=True` flag negates correctly

### 4.2 M2 — Propagator degenerate test

Canonical 2D config (mirror of `solver_gradient_mode_suite.py` defaults so
results compare across runs):

```
nz=48, nx=56, dh=10.0 m, dt=0.0015 s, nt=120
spatial_order=4, abcn=30
source at (ix=nx//2, iz=nz//4)   # interior
receivers: horizontal line at iz=2, x-stride 6
Ricker: freq=10 Hz, delay=0.06 s
vp: depth-ramp 1800→2400 m/s, +180 m/s box anomaly
```

Run forward with `topography=None` and `topography = halo*ones(nx)` (in
padded coords). Wavefield snapshots and receiver traces must match to
`atol=1e-6, rtol=1e-6`. **This is the primary regression guard.**

### 4.3 M3 — Hill reflection smoketest

Gaussian hill, amplitude 10 grid rows, half-width 15 grid points,
centered at `nx//2`. Source at the hill crest (one row below surface).

Assertions:
- After full time march, `p` in air cells (above topo) has L∞ < `1e-10`
- Receiver array placed below the surface picks up an arrival at the
  expected two-way time for normal incidence ≈ `2 * z_bot / vp_avg`
  within ±2 samples
- The first reflection is positive (free surface = phase 0)
- (Visual, manual) wavefield snapshot at mid-time shows clean reflection,
  no obvious vacuum-cell ringing

A reference image lives at `sweep/test/figures/topography_acoustic2d_hill.png`
(committed once first run looks right).

### 4.4 M4 — Gradient consistency

Canonical config (§4.2) + hill of amplitude 5 rows; FWI loss
`L = 0.5 * ||p_rec - p_obs||²` over a depth-ramp vs ramp+box vp pair.

```
grad_eager = torch.autograd.grad(L, vp_param)
grad_ref   = grad_eager  # serves as own reference; just check it runs and is finite
```

For a richer check, compare against the flat-surface gradient on the same
canonical config with `iz_surf = halo*ones(nx)` and confirm cosine similarity
to the no-topo eager gradient `> 0.99` (it should be identical up to fp32
rounding). Threshold inheritance from existing suite: `rel_l2 < 1.5`,
`cosine_similarity > 0.8` if we later compare against an analytic reference.

### 4.5 Test execution policy

Per `feedback_verify_before_commit.md`: every milestone's test must be
**actually executed** (not just type-checked / dry-run) before that
milestone is marked done. Use `ifwitorch` conda env on KW60443.

## 5. Milestones

| MS | Deliverable | Done when | Approx effort |
|---|---|---|---|
| **M0** | This document + branch | — | done |
| **M1** | `extend_top_free_surface_topo` + `zero_above_topo` + unit tests | unit tests green | 0.5 day |
| **M2** | `Propagator(topography=...)` plumbing + `Acoustic.step` integration + degenerate test | degenerate test passes 1e-6 | 1 day |
| **M3** | Hill smoketest + a runnable `examples/wavefields/topography/acoustic2d_hill.py` | reflection time within ±2 samples, air cells clean | 1 day |
| **M4** | Gradient consistency test | rel_l2 < 1.5 vs flat case | 0.5 day |
| **M5** | User-facing docstring updates + `docs/user-guide/topography.md` stub + PR opened against `dev` | reviewer-ready PR | 0.5 day |

User sign-off required at the end of each milestone before starting the
next (per `feedback_no_unilateral_git_publish.md` and the geophyai-core
scope rule).

## 6. Risks / known limits

1. **Staircase artifacts for steep slopes.** Per-column mirror is an
   image method on a rectangular grid; slopes > ~30° will scatter
   energy off the steps. Documented limitation, fix is Stage 3
   (curvilinear). Hill smoketest uses ≤30° to stay in the valid regime.
2. **Quantization error at source/receiver.** Surface sources placed
   at integer rows can sit 0–`dh/2` away from the physical surface,
   causing a small static delay. Acceptable for MVP; Stage 1.5 helper
   will document this.
3. **Air-cell ringing.** If air-mask is applied **after** the FD step
   on the mirrored field, the mirror already provides the correct
   ghost; the mask only suppresses any residue. Order in `step()`:
   `(1) extend → (2) FD update → (3) mask` is the safe sequence.
4. **CFL and PML.** No change — the surface only adjusts where the BC is
   imposed, not the time step. But `iz_surf` must stay clear of the
   bottom/side PML zones; `_validate_topography()` enforces this.
5. **Backward / autograd.** `torch.gather`, `torch.where`, and
   `masked_fill` are all differentiable. Boundary-saving and CUDA-replay
   paths are NOT enabled in MVP; checkpointing under eager autograd is
   the only differentiation path tested in M4.
6. **Equation `field_specs` parity.** The 2nd-order acoustic pressure is
   even-parity; `odd=False` everywhere. When we extend to elastic
   (Stage 2+) we'll need to thread `odd` through per-field.

## 7. Out of scope (explicitly)

- CUDA / `impl='c'` topography support (Stage 2; will need
  `SolverContext.topo_rows` and per-column kernel logic).
- 3D acoustic topography (Stage 1b — straightforward extension of the
  same gather pattern with `(ny, nx)` surface; punted to keep MVP small).
- Elastic / VTI / TTI topography (Stage 2 + Robertsson(1996) local
  staggered-grid corrections).
- Curvilinear / coordinate-transform formulations (Stage 3).
- Float-elevation input with subgrid quantization (Stage 1.5).
- Automatic surface-snap for source/receiver coords (Stage 1.5).
- `sweep-viz` integration / topo overlay on wavefield plots (Stage 1.5).
- Boundary-saving / disk-checkpoint support for topo-FS forward
  (re-enable once Python MVP is stable).

## 8. Open questions to resolve before M1

- [ ] Confirm `topography` accepts `LongTensor` only, or also `np.ndarray`
  / Python list (auto-convert). Default: accept both, normalize to
  `LongTensor` on equation device.
- [ ] Confirm validation behavior: hard-raise vs warn-and-clip when
  `topo_rows` are out of bounds. Default: hard-raise (`ValueError`).
- [ ] Where to expose `_topo_rows_padded` to the equation — through
  `self.equation._topo_rows = ...` mirror (matches how `free_surface` /
  `abcn` are propagated in `base.py:160`), or via an explicit argument
  to `step()`. Default: mirror onto the equation, like `free_surface`.

These are all minor; will be settled while writing M1.

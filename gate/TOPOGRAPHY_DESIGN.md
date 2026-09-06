# Extracting the topography cluster from `PropBase`

Target: `/home/wangs0j/sweep-local/dd-refactor/src/sweep/propagator/base.py:380-809` (430 lines, five methods) → `/home/wangs0j/sweep-local/dd-refactor/src/sweep/core/topography.py`.

A design for this already exists on the dead branch — `git show refactor/python-core:src/sweep/core/topography.py` — and it is basically right. I reuse its function boundaries below but the mutation-boundary reasoning, the risk list and the migration order are re-derived against today's `dev`, which has grown per-edge free surfaces and DD since that branch was cut.

---

## 1. The split

Five methods, and they are not one cluster but three with different shapes:

| method | line | what it actually is |
|---|---|---|
| `_resolve_topo_method` | `base.py:380` | pure policy over four flags. No side effects at all. |
| `_process_topography` | `base.py:464` | orchestration + the ownership reset. **Stays.** |
| `_canonicalise_topography` | `base.py:561` | pure validation + derivation. |
| `_populate_image_method_topography` | `base.py:635` | pure build + 4 assignments |
| `_populate_apm_topography` | `base.py:695` | pure build + 3 assignments |
| `_attach_curvilinear_metrics` | `base.py:752` | pure build + 6 assignments (one via a setter) |

### Moves to `core/topography.py`

```python
def resolve_topo_method(*, topography, topo_method, free_surface,
                        supports_apm, is_curvilinear, equation_name):
    """-> (method, free_surface_physical, image_method_active)"""

def physical_extent(shape, *, abcn, ndim, image_method_active):
    """Padded runtime shape -> physical extent, z asymmetric under image method.
    -> (nz,nx) for 2-D, (nz,ny,nx) for 3-D"""

def canonicalise_topography(topo_input, *phys_extent):
    """-> (topo_row_phys int64, air_mask_phys float32)"""

def build_image_method_topo_rows(topo_row_phys, *, abcn, halo, device):
    """-> int32 runtime rows, synchronized if on CUDA"""

def build_apm_air_mask(air_mask_phys, *, abcn, halo, device):
    """-> float32 runtime mask, replicate-padded on every face"""

def build_curvilinear_metrics(topography, *, shape, ndim, abcn, halo,
                              grid_spacing, image_method_active, device):
    """-> (padded_metrics_dict, CurvilinearGrid)"""
```

Note `resolve_topo_method` needs `equation_name` only for the error string at `base.py:453` (asserted by prefix in `test/test_topography_acoustic2d.py:205`, so the interpolated half is untested but must be preserved anyway). Note also that `is_curvilinear` is an **instance** attribute (`acoustic_curvilinear.py:85`, `elastic_curvilinear.py:264`) with no class-level default, unlike `supports_apm` which does have one (`equations/base.py:61`) — so `getattr(..., False)` at `base.py:422` is load-bearing for every non-curvilinear equation and the binding must keep supplying the default, not the raw attribute.

### Stays in `PropBase` — exactly these assignments

```
base.py:487-491   reset:  self.topography, self._topo_rows_runtime,
                          equation.topography, equation._topo_rows_runtime,
                          equation._apm_air_mask_runtime  <- all to None
base.py:540-541   curvilinear-with-topo: self.topography, equation.topography
base.py:690-693   image:  self.topography, self._topo_rows_runtime,
                          equation.topography, equation._topo_rows_runtime
base.py:748-750   apm:    self.topography, equation.topography,
                          equation._apm_air_mask_runtime
base.py:793-808   curv:   equation.set_curvilinear_metrics(...), _curv_beta,
                          _curv_alpha_xi, _curv_alpha_eta, _curv_h_prime,
                          self._curvilinear_grid
```

Plus all of `_process_topography`'s control flow: the `ndim in (2,3)` guard (`base.py:500-505`), the `torch.as_tensor` (`base.py:510`), the method dispatch (`base.py:549-559`), and the flat-curvilinear early return (`base.py:493-498`).

### Binding sketch

```python
# base.py
from sweep.core import geometry, topography as _topo, validation

    def _resolve_topo_method(self, *, topography, topo_method, free_surface):
        return _topo.resolve_topo_method(
            topography=topography, topo_method=topo_method,
            free_surface=free_surface,
            supports_apm=bool(getattr(self.equation, 'supports_apm', False)),
            is_curvilinear=bool(getattr(self.equation, 'is_curvilinear', False)),
            equation_name=type(self.equation).__name__,
        )

    def _populate_image_method_topography(self, topo_row_phys):
        topo_runtime = _topo.build_image_method_topo_rows(
            topo_row_phys,
            abcn=self.abcn,
            halo=self.equation.so // 2,
            device=getattr(self.equation, "device", None) or self.dev,
        )
        self.topography = topo_row_phys
        self._topo_rows_runtime = topo_runtime
        self.equation.topography = topo_row_phys
        self.equation._topo_rows_runtime = topo_runtime
```

`base.py` 1053 → roughly 780. The `import torch` / `import torch.nn.functional as F` stay **function-local** inside `core/topography.py` (as they are today at `base.py:509, 583, 653, 711`): `base.py` has no top-level torch import (lines 1-14 are `Sequence`, `inspect`, `numpy`, sweep) and `from sweep.core import ... topography` at module head would drag torch into import time for anyone importing a propagator.

---

## 2. The mutation boundary

**Recommendation: keep writing (thin binding). Do not route the installs through the equation.**

The argument that the equation is a specification is the right long-term aim and is false today. Every eager kernel reads runtime topo state off `self` at call time:

- `equations/acoustic.py:138` — `topo_rows = getattr(self, "_topo_rows_runtime", None)` inside `func`
- `equations/elastic.py:173` — `air_mask_rt = getattr(self, "_apm_air_mask_runtime", None)` dispatches the *entire step* to the APM branch
- `equations/elastic.py:186`, `elastic.py:260` (`interior_substeps`, i.e. the boundary-saving reverse driver), `elastic3d.py:453/463`, `acoustic3d.py:126`, `visco_acoustic.py:185`
- and the compiled path does the same off `self.equation`: `_c.py:1688`, `_c.py:1700`, `_c.py:2050`, `_c.py:2106`

There is no other channel. `func(wavefields, models, dt, h, b, **kwargs)` has no parameter for a surface, and `interior_substeps` is called by the BS driver with no propagator in scope. Turning the equation into a real spec means changing `func`'s signature across ~30 equation classes plus both BS drivers plus `_c.py` — a different refactor, and one that cannot be landed under a bit-exact gate in one commit. So the equation is, today, a spec *plus* a per-propagator runtime binding site, and this extraction should not pretend otherwise.

Three further reasons not to invert the direction now:

**(a) The reset is an ownership claim, not bookkeeping.** `base.py:487-491` clears all three equation attributes on *every* construction, topo or not. That is what stops `Prop(eq, topography=hill)` followed by `Prop(eq)` from leaving the second propagator silently running on the first's surface. If you replace "reset, then maybe set" with "return a `TopoRuntime` and install it on the topo path", the reset has nowhere natural to live and the staleness bug is reintroduced. Nothing in `bitgate` or `ddgate` builds two propagators on one equation, so that regression ships green.

**(b) The "equation installs itself" pattern already exists here and is already broken.** `set_curvilinear_metrics` takes four of the seven metrics; `_curv_beta` is set by direct assignment from outside because the acoustic setter's signature doesn't accept it — the equation itself documents this at `elastic_curvilinear.py:272-273`. Extending that half-done pattern to topography would double the inconsistency.

**(c) The value-return does buy something, but at the value level only.** The right compromise: the moved functions return plain values; the *installer* stays in `PropBase`, and the five scattered write sites (`540-541`, `690-693`, `748-750`, `793-808`) collapse into one `_install_topography(self, *, rows=None, air_mask=None, phys=None, metrics=None, grid=None)`. That is a real gain — the complete mutation set becomes readable in one screen instead of inferred from three methods — and it is the *only* part of the extraction that can change presence or ordering, which is why it belongs in its own commit (step 6 below), not folded into any of the pure moves.

Two facts worth writing into the module docstring while you are there, because they are surprising and currently undocumented:

- `self.topography` and `self.equation.topography` (`487`, `489`, `540-541`, `690-692`, `748-749`) have **zero readers** in `src/`, `test/`, `gate/` — they are write-only, and worse, the same name holds different things per path: 1-D physical rows on image/curvilinear, the 2-D/3-D **air mask** on APM (`base.py:748`). Keep them (a user script may read `prop.topography`); do not "fix" the inconsistency inside a bit-exact step.
- `equation._curv_alpha_xi` / `_curv_alpha_eta` (`base.py:802-803`) are also write-only across the whole tree. Dead, deletable, separate commit.

---

## 3. Bit-exactness risks, concretely

**Dtype**

- `base.py:663-676` builds the image rows as `float32` → replicate-pad → `.to(torch.int32)`. The int32 is not cosmetic: `_c.py` reads `data_ptr<int>()` and the comment at `base.py:659-661` records why a cast *there* is unsafe. "Simplifying" to pad int64 directly (numpy `np.pad` accepts ints, `F.pad(mode="replicate")` does not) turns a wrong-dtype into a pointer reinterpretation — silent memory corruption, not a wrong number.
- `base.py:604, 630`: air mask is `(iz < rows).to(torch.float32)`. It is consumed by `classify_topography` (`_c.py:1704` → `equations/_topography.py:126`, which branches on `hasattr(mask,"device")`) and by `F.pad(mode="replicate")` (`base.py:715`), which rejects bool. Returning bool from the moved `canonicalise_topography` and casting at the call site is the obvious "cleanup" and it breaks both.
- `base.py:510` `torch.as_tensor(topography)` then `.to(torch.long)` at `597`/`624`. If the moved function is given the raw user array and does `np.asarray` instead, a float topography truncates differently. Keep `torch.as_tensor` at `base.py:510`, in `PropBase`.

**Device**

- `base.py:677, 741, 766`: `device = getattr(self.equation, "device", None) or self.dev`. This is a **truthiness** `or`, not `is not None`. `torch.device` has no `__bool__` so it is truthy and the two agree today, but `""` or `0` would not. Preserve the operator; do not tidy it.
- `base.py:603, 629`: `iz = torch.arange(nz_phys, device=topo_row_phys.device)` — the air mask is born on the *input's* device, i.e. CPU for a numpy topography even on a CUDA run. So after `base.py:748` `self.topography` is a CPU tensor while `equation._apm_air_mask_runtime` is on GPU. Do not "helpfully" move `air_mask_phys` to `device` before returning it.
- `base.py:687-688`: `torch.cuda.synchronize(topo_runtime.device)` after the H2D copy. This is the single highest-risk line in the cluster. The comment records ~30% non-determinism in the CUDA forward without it, and dropping it as "defensive" produces a **flaky** gate, not a red one — a bit-exact comparator that passes on the run you happen to do. Also do not reorder `.to(int32)` / `.to(device)` around it, since the sync is predicated on `topo_runtime.device.type`.
- `base.py:772`: `CurvilinearGrid(..., device="cpu")` with the move deferred to after padding (`base.py:787-791`). The metrics are `np.gradient` in float64 on CPU (`utils/curvilinear.py:152-153`). Building on the target device changes both precision and reduction order. The literal `"cpu"` must survive.

**Axis order**

- `base.py:670-674` (image, 3-D): a 4-tuple `F.pad` on a `(1,1,ny,nx)` view is `(W_l, W_r, H_l, H_r)` = x then y — **reversed** relative to the array's `(ny, nx)`.
- `base.py:722-731` (APM, 3-D): a 6-tuple on `(1,1,nz,ny,nx)` is `(x, x, y, y, z, z)` — reversed relative to `(nz, ny, nx)`.
  In both cases all six widths are `pad_each`, so a transposition is **invisible** unless `ny != nx`. Any golden test written for this must use a non-square 3-D grid.
- `base.py:585` `_canonicalise_topography` dispatches 2-D vs 3-D on `len(phys_extent)`. If the signature is rewritten to keywords (`nz=, ny=, nx=`), a `ny=None` slipping through picks the wrong branch silently. Keep the `*phys_extent` star-arg, or pass an explicit `ndim`.
- `base.py:772` `float(self._grid_spacing[-1])` is **dx** (last axis), not dz. It coincides with `self._dh` (`base.py:245`) only for scalar `dh`. Pass `grid_spacing` and index inside the moved function; do not substitute `self._dh`.

**Rounding**

Only two: `.to(torch.long)` truncation on a possibly-float topography (`597`, `624`), and `.to(torch.int32)` after a float32 replicate pad (`667`, `675`). Replicate copies values so there is no interpolation and both are exact for realistic row counts. The risk is not the arithmetic, it is that a `np.pad`/`mode="edge"` substitution makes the int dtype look free — see the dtype item above.

**Presence (`hasattr` / `getattr`)**

- The reset at `base.py:487-491` makes all three equation attributes *exist* (as `None`) after any construction. Every reader uses `getattr(..., None)` — `acoustic.py:138`, `elastic.py:173/186/260`, `elastic3d.py:453/463`, `acoustic3d.py:126`, `visco_acoustic.py:185`, `_c.py:1688/2050/2106` — so absence would read the same. The one bare access is `_c.py:1700` `self.equation._apm_air_mask_runtime`, guarded by `_topo_method == 'apm'` so always set in practice. **The real presence risk is staleness, not absence** — see §2(a).
- `self._topo_rows_runtime` (propagator, not equation — `base.py:488, 691`) is read by the boundary-saving guard at `_c.py:804`. When consolidating the two writes it is easy to keep only the equation copy; that turns the "BS + topography on impl='c' gives a wrong gradient" refusal (`_c.py:795-815`) into a silent wrong gradient.
- `self._curvilinear_grid` (`base.py:808`) has zero readers in `src/` or `test/`; it is only reachable as `prop._curvilinear_grid.physical_to_computational(...)` from a notebook. Dropping it is invisible to every gate.
- `prop._topo_method` and `prop.equation._apm_air_mask_runtime` are asserted directly by `test/test_topography_acoustic2d.py:216` and `test/test_elastic_apm.py:120-124`. Those names are pinned.

**`abcn` vs `pad` — the one that will actually bite**

Every extent computation in the cluster uses the scalar `self.abcn`: `base.py:517-527` (phys extent), `664` and `719` (`pad_each = self.abcn + halo`), `764-766` and `782-784` (curvilinear). Under a per-edge pad or a DD cut face, `self.abcn` is **not** the pad on that face. The per-edge case is refused at `base.py:199-202`; the DD case is **not** — `build_rank_pml_widths` shrinks `self.pad` on cut faces at `base.py:344-347` while `self.abcn` stays at the full width (set once at `base.py:178`). Passing `self.pad` into the moved functions "because it's more correct" changes DD-with-topography output. Pass `abcn`, verbatim, and put a comment on the parameter saying so.

---

## 4. Migration order, with honest gate coverage

The gates: `gate/bitgate.py` tiers A (27 cfgs) / B (full matrix) / C (every equation once), and `gate/ddgate.py` world=1 and world=2. Every step below assumes the house procedure: baseline saved on unmodified code, `--verify-reproducible` first, `--self-test` to confirm the comparator can go red.

**Step 0 — extend the gates. Do this before touching `base.py`.**

Without it, steps 2-5 are unguarded. Concretely:

- `bitgate.py` `SCENARIOS` come from `test/solver_gradient_mode_suite.py:230-242`: `interior`, `fd_edge`, `free_surface`, `free_surface_all4`. **All four are flat models.** `topography=` is never passed by any bitgate config.
- `ddgate.py`'s `DDCfg` carries only `free_surface: bool` (`gate/ddgate.py:56-57`) and builds `PropTorch(... free_surface=cfg.free_surface ...)` at `gate/ddgate.py:130-136`. Same: flat only.

So add a `topo_hill` / `topo_stairs` scenario (reuse the generators at `test/topo_gradient_mode_suite.py:100-121`) and a matching `DDCfg(topography=...)`. **Two traps when you do:**

1. The obvious config `Cfg("elastic2d", "c", "bs_gpu", "topo_hill", "canon")` **raises** — `_c.py:795-815` refuses BS+topography, and `_c.py:753` refuses any APM backward. `bitgate._run_matrix` records a raise as `{"error": ...}`, and an error compares equal to an error, so that config is a green that tests nothing. Topo configs on `impl='c'` must use `full` or `ckpt_chunk`.
2. The 3-D topo configs must use a grid with `ny != nx` or the 3-D pad-tuple transposition (§3) cannot be caught. `bitgate`'s canon 3-D grid is 24×20×24, which is fine — it just has to actually be given a topography.

**Step 1 — `resolve_topo_method` → pure.**
Gate: tier A + tier C + DD world=1/2. These genuinely cover it, because the `free_surface=True/False` no-topo branches (`base.py:435-447`) are what every flat config takes. Not covered by the gates: the topo branches (`base.py:449-464`) and the curvilinear early return (`base.py:422-425`) — those rest on `test/test_topography_acoustic2d.py:209-228`, `test/test_elastic_apm.py:120`, `test/test_acoustic_curvilinear.py`. Since this step is pure flag logic with no tensors, pytest is adequate here.

**Step 2 — `physical_extent` + `canonicalise_topography` → pure.** Moved together: the validation messages at `base.py:594-602/615-627` are phrased in terms of `nz_phys`/`nx_phys`, so one golden test covers both.
Gate: **zero bit-exact coverage without step 0.** Author a golden test first that pins `(topo_row_phys, air_mask_phys)` byte-for-byte — `torch.equal` plus dtype plus device — for 2-D hill and a **non-square** 3-D ridge.

**Step 3 — `build_image_method_topo_rows` → pure.**
Gate: step 0's topo scenarios on `full`/`ckpt`, tiers A and C, DD world=1 and world=2 (DD does have a topography path — `parallel/dd_spec.py:264` documents phase 2's air-clear pre-pass — it has simply never been gated). Additionally: a repeat-construction determinism check (build 20×, assert identical `int32` rows and identical forward record) specifically to protect `base.py:687-688`. A one-shot comparison cannot see a missing synchronize.

**Step 4 — `build_apm_air_mask` → pure.**
Gate: step 0's topo scenario on **eager** only for gradients (`_c.py:753` refuses APM backward on `impl='c'`; eager APM gradients are exercised by `test/test_elastic_apm.py:294`), plus `impl='c'` forward for the record. `elastic2d`/`elastic3d` only.

**Step 5 — `build_curvilinear_metrics` → returns `(padded, grid)`.**
Gate: **none exists and none can be cheaply added.** Curvilinear equations are eager-only (`AcousticCurvilinear._C()` raises), are absent from `bitgate.ALL_SOLVERS` (`gate/bitgate.py:130-135`) and from `ddgate.DD_EQUATIONS` (`gate/ddgate.py:87`). The only coverage is `test/test_acoustic_curvilinear.py` and `test/test_elastic_curvilinear.py` — four tests each, tolerance/stability assertions, not before/after. And note the flat case (`base.py:496-498`) is on the path of *every* curvilinear construction, not just topographic ones. Either add `acoustic_curvilinear` to `bitgate` as an eager-only solver, or accept that this step is guarded by hand-written golden metric tensors, and say so in the commit message.

**Step 6 — consolidate the writes into one `_install_topography`.**
Alone, last, and gated by re-running everything above, because it is the only step that can change attribute presence or write ordering. Add one test the gates cannot express: build two propagators on the same equation instance, the first with a hill and the second without, and assert the second's `equation._topo_rows_runtime is None`.

### What the gates do not cover, stated plainly

- No topography path of any kind has a bit-exact gate today. `bitgate` and `ddgate` both run flat models exclusively.
- `topo_gradient_mode_suite.py` and `topo3d_gradient_mode_suite.py` are the only topo-wide coverage, and they are **not** before/after gates: they compare eager against `c` at `cosine > 0.8` / `rel_l2 < 1.5` (`test/topo_gradient_mode_suite.py:16-18`), they are scripts rather than pytest (0 `def test_`), and a refactor that moves eager and `c` together is invisible to them by construction.
- APM on `impl='c'` is forward-only by design (`_c.py:753`), so no gate can ever cover its gradient.
- Curvilinear is in neither gate and is not reachable from `impl='c'` at all.
- BS + topography on `impl='c'` is a refusal, not a path, so the interaction between this cluster and boundary saving is gated only in the negative (`test/test_boundary_saving_topography_guard.py`).

---

## 5. What should not move, and why

1. **Every assignment.** `base.py:487-491, 540-541, 690-693, 748-750, 793-808`. Argued in §2. The propagator owns the state; `core/` only computes it. This is also what makes the moved functions testable without constructing a propagator, which is the whole point of `core/geometry.py` and `core/validation.py`.

2. **`_process_topography`'s orchestration** (`base.py:464-559`). The reset, the `ndim in (2,3)` guard, the `torch.as_tensor`, the dispatch on `self._topo_method`, and the flat-curvilinear early return. This method's correctness depends on a *construction-order* invariant — `base.py:374-375` records that it must run after `self.shape` has been PML-padded — and construction order is propagator business, not a pure computation. (`physical_extent` is the one piece I do move out of it, because the `- abcn` vs `- 2*abcn` asymmetry at `base.py:515-519` is exactly the kind of thing that deserves a name and a docstring.)

3. **The `_resolve_topo_method` call site and the fs_faces fold-back** (`base.py:216-221` and `base.py:231-234`). The topo→`fs_faces`→`pad` coupling — topography implicitly turning the top face into a free surface and re-running `normalize_pad` — is boundary-layout logic that belongs next to the rest of the per-edge boundary handling, not in a topography module.

4. **The function-local torch imports.** `core/topography.py` keeps `import torch` inside each function, as `base.py:509, 583, 653, 711` do today. `base.py` has no top-level torch import; a module-level one in `core/topography.py` would be pulled in by the `from sweep.core import ...` at `base.py:5`.

5. **`src/sweep/equations/_topography.py` (748 lines).** Do **not** merge it into `core/topography.py` despite the near-identical name. That file is the APM *discretisation* — Cao & Chen 2018 cell taxonomy and modulus modification — consumed by both the eager equation (`elastic.py:173`) and the compiled path (`_c.py:1704`). `core/topography.py` is grid bookkeeping: extents, padding, dtype, device. Different layer, different consumers. The confusable names are worth an explicit cross-reference in both module docstrings; renaming either one is a separate commit.

6. **`_c.py` and the equation-side reads.** Untouched by this work. If any of them need to change, the extraction has gone wrong.
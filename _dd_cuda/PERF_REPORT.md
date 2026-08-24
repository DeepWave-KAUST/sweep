# DD performance — cut-aware pad + comm/compute overlap (8×V100, ibex)

Two committed optimizations on `perf/dd-cut-aware-pad`:

* `1e0c9cc` **cut-aware compact pad** — cut faces allocate only the M stencil
  halo (not abcn+M). Per-card memory cut **19–59 %** (scales with tile
  thinness + #cut-axes). Bonus: cut tiles also *compute* faster (no cut-side
  PML work) — the px8 "no-comm" floor (1.174 ms) is BELOW the px1 full-pad
  step (1.356 ms).
* `a7b7bb4` **comm/compute overlap** (acoustic fwd) — phase-1 cut strips →
  async halo exchange on a comm stream (start = copy-send+P2P, no wait;
  finish = wait+copy-recv) running while phase-2 interior computes. Bit-exact
  vs serial; **source-safe** (serial fallback when a source sits in a cut
  strip, since the source is injected in phase 2).

## Scaling (acoustic 3D, so4, abcn20, V100, per_step ms)

WEAK (fixed tile/GPU, problem grows):

| tile/GPU | px1 | px2 | px4 | px8 | 8-GPU eff |
|----------|-----|-----|-----|-----|-----------|
| 256³     |1.356|1.396|1.397|1.419| 7.65× (95.6%) |
| 320³     |2.578|  –  |  –  |2.584| **7.98× (99.8%)** |

STRONG (fixed global 256×256×1024, split):

| GPUs | 1 | 2 | 4 | 8 |
|------|---|---|---|---|
| ms   |4.964|2.712|1.396|0.845|
| speedup|1×|1.83×|3.56×|**5.87×**|

## Conclusions

* **Weak scaling reaches 8× on 8 GPUs** (7.98× / 99.8 % at 320³; 7.65 % at
  256³). The user's "8卡8倍" target is met for realistic tile sizes — bigger
  tiles → closer to ideal as the halo becomes a smaller fraction.
* **Strong scaling tops out at ~5.9× (8 GPUs).** This is structural, not a
  missing optimization:
  * the per-step halo comm (~0.2 ms) is **copy + req.wait bound**, not
    P2P-bound (the NCCL transfer itself is ~40 µs). It is 4 strided staging
    copies + a CPU-blocking `work.wait()`; overlap hides only part of it.
  * the **compute-only** strong floor at px8 is already just 7.36× (thin
    128-wide tiles drop FD-kernel efficiency), so even perfect comm hiding
    could not reach 8×.
* NCCL env tuning (`NCCL_P2P_LEVEL=NVL`, `NCCL_ALGO=Ring`) did not help
  (default best). Larger tiles are the lever for weak-scaling efficiency.

Remaining levers are deep/fragile (custom fused strided-halo copy kernels, a
GPU-event P2P sync to drop the CPU `work.wait`, or splitting z) — not worth the
robustness cost for the residual strong-scaling gap.

## Comm profile + why K-step doesn't help (loop round 2)

Per-step halo exchange breakdown (256³ x-halo, 2 ranks, ms):

| copy-send | copy-recv | P2P+wait | Python/API | full |
|-----------|-----------|----------|------------|------|
| 0.025 | 0.051 | 0.068 | 0.054 | 0.198 |

No single dominant component — copies, P2P+wait, and Python overhead are each
~1/3. Only the P2P (~0.04 ms, NVLink) can truly overlap compute; the strided
staging copies use SMs (compete with phase-2) and `work.wait`/Python are CPU,
so the SPECFEM overlap recovers only a fraction.

**K-step exchange (wider halo, exchange every K steps) is net-negative here.**
It amortizes the per-exchange overhead (~0.064 ms/step saved at K=4) but forces
each step to redundantly compute the K·M halo region: 2(K−1)M extra x-cells =
12/nxp_eff of the tile. On the thin strong-scaling tiles (nxp=128) that is
~9.4 % extra compute > the 7.2 % comm saved → slower; on fat weak tiles it is
roughly neutral. So the textbook latency-hiding lever does not apply.

**Conclusion (round 2): the *x-cut* clean optimizations are maximized.** Strong
scaling ~5.9× is the practical limit *for x-cut* (comm is irreducibly ~0.2 ms
split 3-ways and only partly hideable; the x-cut compute-only floor is 7.36×).
Weak scaling already meets the 8× goal. — But round 3 found the x-cut *axis
choice itself* was the limit; see below.

## Round 3: copy-engine (rejected) + decomposition-axis (the real win)

**Copy-engine halo staging — TRIED, REJECTED (net-negative).** Hypothesis: move
the strided halo staging copies off the SMs onto the GPU copy engine
(`cudaMemcpy2DAsync`) so they overlap the stencil. Implemented + validated
**bit-exact** (dd_api_check px8 acoustic/elastic 2D/3D all PASS with the engine
genuinely on). But it is **slower** everywhere: the isolated strided copy went
0.074→0.094 ms (+27 %, DMA launch latency dominates these tiny D2D strided
copies), and every end-to-end config regressed (strong px8 overlap 0.841→0.908,
+compute-stream 0.829→0.885). Reverted. (Reusable finding: cudaMemcpy2DAsync D2D
is the wrong tool for small strided halo strips on V100.)

**Decomposition axis — the actual lever.** The 7.36× compute floor was blamed on
launch-amortization; it is really the **x-cut tile SHAPE**. A 1-D x-cut shrinks
the *contiguous* x-dimension (px8: Nx/8) — worst for the FD kernel and the halo
copy. Cutting more axes (a) saves more cut-aware PML (more cut faces → less PML
work) and (b) gives a squarer tile. Compute-floor sweep, equal 8.39 M cells/tile
(global 256²×1024 / 8), none-mode on 8× V100:

| decomposition | tile (Nz,Ny,Nx) | x_contig | per_step | peak_mem |
|---------------|-----------------|----------|----------|----------|
| x-cut px8 py1 | (256,256,128)   | 128      | 0.666 ms | 0.80 GB |
| **bal px4 py2** | (256,128,256) | 256      | **0.608 ms** | **0.74 GB** |
| bal px2 py4   | (256, 64,512)   | 512      | 0.683 ms | 0.82 GB |
| y-cut px1 py8 | (256, 32,1024)  | 1024     | 1.031 ms | 1.03 GB |

Not monotonic — a **balanced** tile wins; y-cut (fat x, thin y) is *worst*
(refutes "fat contiguous x is better"). End-to-end via the production
`ModelParallel` (correct corner halo; forward, 8× V100):

| global (Nz,Ny,Nx) | x-cut px8 | balanced px4 py2 | speedup | mem |
|-------------------|-----------|------------------|---------|-----|
| 256 × 256 × 1024  | 0.893 ms (6.55×) | **0.814 ms (7.18×)** | +9 %  | 2.69→2.56 GB |
| 384 × 384 × 384   | 1.026 ms (4.67×) | **0.716 ms (6.69×)** | +43 % | 2.84→2.29 GB |
| 512 × 512 × 512   | 1.890 ms        | **1.439 ms**         | +31 % | 5.66→4.71 GB |
| 256 × 512 × 1024  | 1.673 ms        | **1.417 ms**         | +15 % | 4.94→4.54 GB |

**The balanced 2-D decomposition is up to ~1.5× faster and ~18 % lighter for
strong scaling, generalising across shapes — biggest for cubic globals where the
x-cut tile is thinnest.** Shipped as `sweep.parallel.balanced_grid(world, shape)`
(returns the recommended `(py, px)`; pure arithmetic, additive — does not change
any default). Default caps `py<=2` (a conservative load-balance choice that
already captures +9–43 %). `py>=3` (the cubic optimum, e.g. 384³ px2py4 =
0.688 ms / 6.96×) is opt-in by raising `max_py` (e.g. `max_py=world`) and is
**validated** (8× V100 bit-exact).

**py>=3 boundary-save crash — FIXED (`83a70df`, ibex bit-exact confirmed).** An
earlier `CUDA error: invalid configuration argument` for `py>=3` was **not** a
missing y-kernel guard: `ModelParallel._capture` ran a public fwd/bwd with
`cut_face_mask=0` on a thin cut-aware tile, so the kernel sized a cut face as
full PML → negative boundary count → invalid launch. Fix is pure-Python
(`_capture` fwrap/bwrap set the mask; **no CUDA rebuild**); `Boundary3D::front_back`
is dead code. See lesson_dd_capture_cut_mask.

**Updated bottom line:** weak 8× met; **strong scaling improves from ~5.9× to
7.0–7.2× simply by choosing a balanced grid** (≈8× compute is reachable because
cutting more axes is super-linear via cut-aware PML savings). Use
`balanced_grid()` instead of a 1-D x-cut. Bench tools: `dd_ddp_timing.py`
(production-path per-decomposition timing), `dd_axis_strong.sbatch`,
`dd_axis_generalize.sbatch`.

## Round 4: per-step / per-forward driver optimizations (2026-06-15)

A pass over the DD driver targeting the per-step Python overhead (×nt), the
per-forward redundancy, and readability/extensibility. All bit-exact — ibex
`dd_api_check` acoustic/elastic 2D/3D (+ free surface) PASS, the 3D 2×2 corner
stays PASS_TOL (≤1e-5) as before; per-change gate sbatches in `_dd_cuda/`.

* **per-step caching** (`89154ca`) — the stepped runners rebuilt
  `list(wavefields)` every step and `ModelParallel._halo_view` rebuilt the halo
  crop slice every exchange. The bound order depends only on `(k%3, k%2)` (≤6
  distinct lists) and the crop slice is loop-invariant, so both are cached.
  Bit-exact (same persistent tensors; only roles rotate).
* **`forward(models=None)`** (`622100a`) — an FWI epoch fires many shots through
  one model, yet forward re-padded the runtime model and ran the NCCL model-halo
  collective every shot. `models=None` reuses the buffers a prior forward set →
  one model-halo per epoch, not per shot. Explicit on purpose (no version-
  guessing that could silently run on a stale model).
* **elastic halo aggregation** (`9db162a`) — the elastic loop fired a separate
  `batch_isend_irecv`+wait per field (nphys = 5 in 2-D, 9 in 3-D) each step.
  `FastHaloGroup` concatenates each field-group (velocity / stress fwd;
  adjoint+recon bwd) into ONE batched P2P → 2 waits/step. **Measured
  1.28–1.30× elastic-2D forward** (2× V100, nphys=5, tiles 64²/128², nt300;
  `dd_agg_bench.py`); larger for 3-D (nphys=9 → 2). Bit-exact.
  *True comm/compute overlap stays acoustic-only:* the elastic kernel's
  `step_phase` is a velocity/stress field-group split, not the strip/interior
  spatial split overlap needs (phase 2 reads all velocity halos), so hiding the
  exchange behind interior compute would require a C++ kernel sub-split.
* **shot-parallel gradient** (`789fe7d`) — `gradient()` all_reduces the per-tile
  gradient across the shot process group when `shot_groups>1` (the FWI gradient
  is a sum over shots), enabling combined shot+model parallelism without a B>1
  rewrite. No-op for `shot_groups==1`. (`dd_shotpar_check.py`, world=4 = 2×2.)
* **autograd-transparent forward** — DD now behaves like single-domain
  `PropTorch`: if a model tensor `requires_grad`, the record carries a grad_fn
  and a plain `loss.backward()` populates each model's `.grad` (per-tile leaf →
  tile grad; replicated global leaf → global grad, `all_reduce` to assemble) via
  a `_DDForward(autograd.Function)` whose backward runs the DD adjoint. The
  explicit `gradient()` method is **removed** (dev-stage, no back-compat) — an
  arbitrary adjoint goes through `record.backward(gradient=adjoint)`.
* **API / readability** — `balanced_grid(max_py=…)` replaces the misleading
  `allow_y_thin` bool (**removed**, no deprecated alias); `FastHaloSet.exchange`
  deduped via `_get` (`dcfb3a9`); `forward()` split into `_prepare_call` + per-
  family loop helpers — 33 lines, was ~120 (`4ace00d`).
* **stale-test fix** (`f8711cd`) — `test_dd_tiles_3d` / `test_dd_elastic_tiles_3d`
  predated the cut-aware pad (symmetric `PAD` offsets, missing/mis-offset
  model-halo fill) and asserted against the wrong tile region → spurious gross
  failures; migrated to per-tile `prop.padding` offsets. Production path was
  always correct.

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

**Conclusion: the clean optimizations are maximized.** Strong scaling ~5.9× is
the practical limit (comm is irreducibly ~0.2 ms split 3-ways and only partly
hideable; the compute-only floor is 7.36× anyway). Weak scaling already meets
the 8× goal. Further would need fragile low-level work (copy-engine 2-D strided
DMA, exposing the NCCL event for a GPU-side P2P sync) with modest, capped return.

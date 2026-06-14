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

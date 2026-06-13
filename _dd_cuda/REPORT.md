# DD (impl='c') — running report

Branch `feat/cuda-domain-decomp`, worktree `/home/wangs0j/sweep-local/sweep-dd-cuda`.
Plan: `_dd_cuda/PLAN.md`. All work uncommitted pending approval (see §commits).

## Status

| Milestone | State | Evidence |
|---|---|---|
| M1 stepped forward API | **DONE** | `test/test_stepped_forward.py` 24/24, bit-exact k∈{1,7,40}, PML/BS/FS/ckpt |
| M2 two-tile DD forward (1 GPU) | **DONE** | `test/test_dd_two_tile.py` 5/5 bit-exact vs single domain |
| M3 NCCL on ibex V100 | running | jobs 47446652 (build, a100) → 47446653 (v100:4) |
| M4c 3-D DD gradient | **DONE** | `test_dd_backward_two_tile_3d.py` 1/1 — acoustic3d two-tile gradient **bitwise** (grad_vp/grad_wavelet/illums; 12/15-slot rotations, x-cut). The "3-D public-vs-replay ulp quirk" seen here is root-caused & FIXED (see §public-vs-replay below): public forward was running the legacy racy in-place psi branch; now public ≡ replay bitwise in 3-D too |
| M4b NCCL multi-GPU backward (end-to-end) | **DONE** | `test/dd_nccl_backward_check.py` (job 47470000): 2-rank AND 4-rank V100 — grad_vp / grad_wavelet / illuminations all **bitwise** vs single-domain reference |
| M4 stepped backward + DD grad | **DONE** | `test_stepped_backward.py` 9/9 bitwise (any partition == monolithic, bs+full, 2D+3D); `test_dd_backward_two_tile.py` 2/2 — two-tile DD gradient **bitwise == single domain** (grad_vp/grad_wavelet/illums, Design B: cut faces skip strip-restore + per-step λ and recon u_now halo exchange, cut_face_mask plumbed to kernels). Regression: 101 pytest + suite 120/120 |
| M5 acoustic3d 2-axis | 1-GPU part **DONE** | `test/test_dd_tiles_3d.py` 3/3 bit-exact: x-split, y-split, 2×2 grid |
| M6 efficiency loop (HARD) | data gathering | see below |
| E1 elastic2d/3d stepped + DD forward (incl. FS) | **DONE** | `test_stepped_forward_elastic.py` + `test_dd_elastic_two_tile.py` 4/4 + `test_dd_elastic_tiles_3d.py` 4/4 — all bitwise vs single domain, free surface on/off. Key designs: (a) HALF-STEP exchange protocol — `step_phase` physics split (1=velocity kernel, 2=stress+tail), exchange v slots after phase 1 and s slots after phase 2, M wide each: owned stress cells always read exchanged velocities, so transverse CPML memory (`m_*z/y`) divergence in halo cells never reaches owned cells (single-exchange 2M variants fail at 1e-5..1e-2); (b) cut-aware PML-band predicates mirrored into elastic kernels (zero-coefficient PML branch is NOT bitwise equal to the interior branch); (c) static model halo from the globally padded model (staggered updates read neighbour material). Elastic gradient suite 48/48, acoustic regression 69 pytest green |

## Key facts established

- **Race-free forward applied** (working-tree apply of grad-align `ee13307`):
  the dev-branch acoustic CPML psi RAW race made even two identical runs
  differ (~1e-4) — bit-exact DD validation impossible without it. After:
  gradient suite 120/120, all M1/M2 bitwise asserts pass (2D AND 3D).
- Forward wavefield layouts (psi double-buffer): 2D 9 slots, 3D 12 slots.
  Rotation between stepped calls: u-slots left-rotate by k%3; (psi, psin)
  pairs swap when k odd. `src/sweep/propagator/_stepped.py`.
- Symmetric model padding kept in DD (interior pad = abcn+M with zero PML
  coefficients): mathematically equivalent, halo overwrite blocks any
  influence from the stale pad interior. Bit-exactness confirms.
- **Per-step overhead of stepped(k=1) vs monolithic (KW60443, RTX 6000 Ada):**
  - 2D 2048², nt=200: 32 µs/step, overhead 0.2 µs/step (ratio 1.006)
  - 3D 256³, nt=60: 792 µs/step, overhead 0.3 µs/step (ratio 1.000)
  - 2D 512² (launch-bound, not a DD target): 7 µs/step, overhead 4.2 µs/step
  → Python round-trip is hidden by async kernel queueing; the efficiency
  question is **only** about the halo exchange.

## Efficiency requirement (user, hard)

Weak scaling primary: N ranks × (single-GPU-sized tile each) must finish in
≈ the single-GPU time t (eff ≥ ~90–95%). Strong scaling reported too.
Bench: `test/dd_nccl_bench.py` (production loop, no per-step host sync).

### V100 findings (job 47453801, 4× V100-SXM2-32GB, 2026-06-11)

- **Cross-GPU determinism: same-model V100s do NOT reproduce bitwise.**
  Identical single-domain problem on 4 GPUs: rank1≡rank0 bitwise, ranks
  2/3 differ by ~1.2e-6 (ulp). ⇒ cross-GPU DD acceptance = rel ≤ 1e-5
  (graded PASS_TOL); single-GPU manual harness stays bitwise. NCCL checks:
  2-rank PASS (bit), 4-rank PASS_TOL — **no DD bug**.
- **Exchange micro-bench (per call):** `exchange_halos` 416 µs = ~330 µs
  Python/autograd wrapper + 84 µs bare batched P2P (incl. construction)
  + 29 µs staging. → `FastHaloSet` (fast_halo.py): preallocated buffers,
  prebuilt P2POps, no autograd. Round-2 job measures it.
- **Weak scaling round 1 (std exchanger, zero tuning):**
  2D 4096²/GPU: 0.536/0.572/0.592 ms/step @ px=1/2/4 → eff 93.7%/90.5%.
  3D 320³/GPU: 2.578/2.801/2.960 ms/step → eff 92.0%/87.1%.
  Strong 2D 4096×8192: 1→4 GPUs 2.85× (71%).
- ibex build gotcha: rsync preserves mtimes older than cached build/*.o
  (clock skew) → ninja silently skips recompiling → stale .so ran the v4
  job's pytest (11 "failures" = missing new bindings, not numerics).
  Fixed: `rm -rf build` in the build sbatch.

### ODR kernel-name collision (root cause of ALL "nondeterminism", fixed)

`acoustic_lsrtm2d/kernels.cuh` defined global `acoustic2nd*` template
kernels colliding with acoustic2d's — CUDA picks a winner per process.
Explains: the "cross-GPU ulp differences" (two bit-groups of ranks) and the
flaky DD-backward test. (The old "3D nondeterminism residue" turned out to
be a SEPARATE bug — see §public-vs-replay below.)
Fix: renamed to `acoustic2d_single*` (lsrtm3d convention). After fix +
overlap (round 5, job 47456935): **CROSS_GPU_DETERMINISM: BITWISE; all
NCCL checks PASS at bit level (2D small/large, 3D).** origin/dev carries
the same latent collision — separate fix chip spawned.

### Public-vs-replay forward quirk (3-D ulp ring/grad diffs) — root-caused & FIXED (2026-06-12)

**Root cause (one paragraph).** Every public Warpper forward that does not
save all wavefields (i.e. all boundary-saving and all no-grad runs —
`_ensure_wavefield_buffers(need_forward=False)` leaves `forward_wavefields`
empty) reaches the CUDA forward with an empty `ForwardInput.wavefields`, so
`AcousticWavefieldTensor::allocate()` runs — and it never allocated the psi
double-buffers, leaving `double_buffer_psi=false`. The stencil kernel then
takes the legacy branch `(f.psixn ? f.psixn : f.psix)[idx] = ...`, writing
psi IN PLACE in the same launch that neighbour-reads psi via
`gradient<>(f.psi*, ...)` — exactly the intra-launch RAW race the psi
double-buffer (grad-align `ee13307`, dev PR #41) was introduced to remove.
The stepped replay binds 12 tensors → `bind()` → race-free read-old/write-new
branch. So the quirk was NOT a benign inherent difference: the replay side
was correct and deterministic; the PUBLIC side was racy.

Evidence (`_dd_cuda/quirk_public_vs_replay.py`, pre-fix, canonical
24×20×32 / abcn=10 / so=4 / nt=60 / BS-gpu): replay-12 vs replay-12 bitwise
EQUAL (race-free path deterministic); replay-9 (legacy branch, all other
inputs identical) vs itself DIFFERS; public vs public (fresh prop) DIFFERS —
a live race, which also eliminates the other suspects (record_out binding,
Warpper-mutated ForwardInput fields, topo_rows/coes) by construction. First
divergence signature: u_now+zetaz at 3 cells in the z-PML while ALL psi
fields are still bitwise clean — psi's own update is a per-cell recurrence
(deterministic); what races is the *neighbour read* dpsi/dz feeding tmpz →
u_next/zetaz; psi then inherits the diff a step later through dudz. Diff
cells appear only in pml[z]/pml[y], never pml[x]: x-neighbour loads are
warp-lockstep with the store in the same instruction stream, y/z neighbours
cross warps → real race window. 2-D at 48×56 never trips it (public ≡
replay-7 ≡ replay-9 bitwise) — why the 2-D sanity assert always passed.

**Fix** (this worktree, uncommitted): `allocate()` gained an opt-in
`double_buffer_psi_` parameter; the 8 forward-time-loop call sites
(acoustic2d/3d, vrz2d/3d, lsrtm2d/3d bg+sc — all already pair with
`swap_pml()`) pass true. Backward ckpt/recon `allocate()` sites deliberately
stay legacy: they pair with the u-only `swap()`, so a psi*n write there
would be lost. Post-fix verification: public ≡ replay-12 ≡ replay-empty
bitwise, public deterministic across fresh props (E1/E3/E5/E6 EQUAL);
stepped+DD suites 44 passed (incl. `test_stepped_forward` whose
public==replay sanity is now asserted in 3-D too, previously 2-D-only);
gradient suite 6 eqs × {full, bs_gpu} 12/12 pass.

**Residuals.** (a) origin/dev had the identical gap (same `allocate()`, same
`need_forward=False`) → all public 3-D acoustic-family BS/no-grad forwards
on dev were ulp-nondeterministic run-to-run; **fixed on standalone branch
`fix/alloc-psi-double-buffer` (worktree `sweep-fix-psi-dbuf`, off dev
`9fb2823`, unpushed pending approval) — TWO commits: `9c9b3a4` C++
allocate() opt-in psi*n (hardening), `b020dde` the design fix per the
maintainer rule "all wavefield buffers are python-allocated": BS/ckpt/
no-grad forwards (+3-D rtm) pass per-call transient python zeros from
cuda_layout, making the C++ fallback unreachable from public paths
(pristine-dev probe DIFFER → EQUAL incl. no-grad; acoustic 24/24 incl.
ckpt modes; elastic/das/vti 8/8; bs_gpu peaks unchanged)**. When THIS
worktree rebases onto a dev that contains those commits, drop the
duplicate allocate() hunks here.
(b) THIRD instance of the race class, found+fixed
2026-06-12: the vrz ADJOINT kernels (PML zone propagates the adjoint with
forward-CPML) neighbour-read psi and wrote it in place UNCONDITIONALLY (no
psixn branch, u-only swap) → vrz3d backward stayed ulp-nondeterministic
even after the forward fix. Fixed on `fix/vrz-bs-recon-state` (`b3a865f`,
worktree `sweep-vrz-recon`, stacked on the psi branch): conditional psi*n
write + swap_pml() + dbuf adjoint allocate fallbacks; same commit makes the
vrz3d BS recon u-only (use_pml false — NOPML recon never touches psi; 6
dead fields dropped, bs_gpu peak 111→96MB) and adds vrz2d's missing recon
bind branch (python-bindable recon = prereq for DD stepped backward on
vrz). vrz2d pre/post BITWISE equal; vrz3d public×2 now BITWISE; suite 8/8.
(c) Full-library audit done (2026-06-12,
`sweep-vrz-recon/_probe/race_audit.py` — per-kernel write∩neighbour-read
over every __global__/__device__ in all 16 eq dirs, fwd AND bwd): only the
6 acoustic-family eqs ever had the pattern (all conditional-dbuf now); the
other 10 eqs' kernels are clean (VTI adjoint split-fix holds; no hand-rolled
stencils). All 16 forward.cu have bind branches → the python transient
wavefields engage everywhere. Last LIVE instance found empirically:
**lsrtm3d backward ADJOINT — λ keeps forward-CPML in the PML band via the
shared conditional helper (dpsi neighbour reads), but all four backward
variants bind the adjoint as slice(adjoint_wavefields, 0, 9) → 9-slot
legacy, no psi*n → conditional write degrades to in-place → race (u-only
swap). grad_mp nondeterministic (46/15360 cells, max 1.1e-16;
`sweep-vrz-recon/_probe/lsrtm_determinism.py`; lsrtm2d stable). **FIXED on
`fix/lsrtm-adjoint-psi-dbuf` `909c9a2` (stacked, unpushed): slices 0,9→
0,12 (2-D begin+7→+9), swap()→swap_pml(), dbuf fallbacks — no kernel
change; lsrtm2d pre/post BITWISE, lsrtm3d post-fix public×2 BITWISE,
suite 8/8 all modes.** Audit blind spot: "conditional = safe when bound"
must be checked against each call site's BINDING WIDTH.
CKPT RECOMPUTE (all acoustic eqs) remains legacy-racy by design,
tolerance-covered (lsrtm's `*_SINGLE` full-PML replays are ckpt-only; its
BS recon is SINGLE_NOPML + use_pml=false — clean).

### Overlap (SPECFEM iphase) — implemented, round-5 numbers

Phase-split stepped step (ForwardInput.step_phase: 1 = cut-strip stencil
only; 2 = interior + source/record/swap tail) + dual-stream driver
(NCCL on a comm stream concurrent with the interior kernel). Single-GPU
4-tile overlap harness: bitwise PASS.

Weak scaling (job 47456935; baseline = same-job px=1, slow-card mix noise
±4% applies):
- 3D 320³/GPU: px=4 serial 84.2% → **overlap 96.1%**; px=2 **96.1%**
- 2D 4096²/GPU: px=2 overlap **93.1%**; px=4 serial 87.0% → overlap 88.3%
  (2D exchange is latency-bound ~52 µs/step; phase-split overhead eats the
  overlap gain; remaining gap ≈ NCCL small-message latency + card-bin mix)
- Strong scaling 2D 4096×8192: 1→2 GPUs 89%, 1→4 72% (surface/volume).

### FINAL numbers (round 6, job 47465508 — same-job none denominator,
### two identical rounds, cv ≤ 0.002)

px=4 weak scaling, denominator = none mode (4 GPUs computing, zero comm
= slowest-card step time with the exact same card mix):

| case | none | serial | overlap | efficiency (overlap) |
|---|---|---|---|---|
| 2D 4096²/GPU | 0.537 ms/step | 0.593 (90.6%) | 0.590 | **91.0%** |
| 3D 320³/GPU | 2.582 ms/step | 2.957 (87.3%) | 2.676 | **96.5%** |

**User's hard requirement (weak-scaling eff ≥ ~90%) is MET in both
dimensions.** Remaining 2D gap (~53 µs/step) ≈ NCCL small-message latency
(~39 µs, physical) + phase-split launch overhead; 3D residue ~94 µs/step
(staging tail not fully hidden). M6 CLOSED.

### Spatial-order sweep (round 7, job 47466513) — conclusion is
### order-robust; order-8 NCCL checks PASS bitwise (2D + 3D)

px=4 weak scaling, none denominator, best of serial/overlap:

| so (halo M) | 2D 4096²/GPU | 3D 320³/GPU (overlap) |
|---|---|---|
| 2 (1) | 90.6% | 95.3% |
| 4 (2) | 90.7% | 97.3% |
| 6 (3) | 90.4% | 98.1% |
| 8 (4) | 90.6% | 97.1% |

2D is latency-bound at every order (54–61 µs/step, halo width barely
matters; serial suffices — overlap even loses 3% at so=8). 3D gains
with order (kernel cost grows faster than halo bytes; overlap hides the
larger exchange). Practical guidance: serial exchanger for 2D, overlap
for 3D.

### ELASTIC DD multi-GPU (round 11, job 47485365, 8× V100-SXM2-32GB, 2026-06-13)

Elastic uses the **half-step exchange protocol** (split by physics, not
region): phase 1 = velocity, exchange v halo; phase 2 = stress + tail,
exchange s halo. Per step it exchanges 5 fields in 2-D, 9 in 3-D, and both
exchanges sit on the critical path — **no overlap** (would need
cut-strip-only elastic kernels). Despite 5–9× the acoustic comm volume,
efficiency is HIGHER than acoustic because each elastic step is far heavier
(≈6 ms/step 2-D), so NCCL latency/bytes are well amortised.

**Correctness — ALL 12 NCCL checks BITWISE PASS** (no PASS_TOL, no FAIL):

| check | ranks | result |
|---|---|---|
| forward 2D (FS off) | 2,4,8 | PASS (bit) |
| forward 2D (FS on) | 4 | PASS (bit) |
| forward 3D (FS off) | 2,4 | PASS (bit) |
| forward 3D (FS on) | 2 | PASS (bit) |
| backward 2D (FS off) | 2,4 | PASS (bit) |
| backward 2D (FS on) | 2 | PASS (bit) |
| backward 3D (FS off/on) | 2 | PASS (bit) |

Elastic DD forward record AND backward grad_vp/vs/rho are bit-exact vs the
single domain on real multi-GPU, free surface included.

**Weak scaling** (per-GPU tile fixed; none/full same-job denominator):

| config | px=2 | px=4 | px=8 |
|---|---|---|---|
| 2D 4096²/GPU so4 ab20 | 97.4% | 96.4% | 96.2% |
| 3D 256³/GPU so4 ab20 | 93.2% | 91.2% | 91.2% |

px=8 robustness sweeps: 2D so2/so8 96.2/96.4%, ab10/ab40 96.2/96.4%,
nt100/1600 96.4/96.3%, FS 96.3%; 3D so2/so8 91.9/90.9%, ab10/ab40
89.4/92.9%, nt50/400 91.1/91.2%, FS 91.0%. Tile-size: 2D 2048² 79.8% →
8192² 98.8%; 3D 192³ 89.0% → 320³ 92.8% (bigger tile → higher eff). The
≥90% HARD bar is met for the production tile sizes (2D ≥96%, 3D ≥91%).
`none(px8) ≈ baseline(px1)` (5.96 vs 5.95 ms 2-D; 18.07 vs 18.8 ms 3-D),
confirming the none denominator is a valid single-GPU proxy.

**Strong scaling** (fixed global model split across GPUs):

| global | 1-GPU | px=2 | px=4 | px=8 |
|---|---|---|---|---|
| 2D 4096×16384 | 23.0 ms | 1.94× (97%) | 3.72× (93%) | 6.79× (85%) |
| 3D 256²×1024 | 59.5 ms | 1.82× (91%) | 3.00× (75%) | 4.72× (59%) |

3D 8-GPU degrades (thin 256²×128 tile → comm-bound), same pattern as
acoustic. ALL M1–M6 + E1–E3 complete.

## Proposed commits (awaiting approval)

1. stepped forward API (wavetypes/module/acoustic2d/3d forward, ckpt cursor,
   `_stepped.py`, `test_stepped_forward.py`)
2. apply of grad-align ee13307 (race-free CPML fwd + fused adjoint) — NOTE:
   duplicates the unpushed `investigate/eager-c-grad-align` branch; decide
   merge order with user (either land that branch's PR first and rebase, or
   land this copy and drop it there).
3. two-tile DD harness + NCCL check/bench scripts

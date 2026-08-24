# Plan: Domain decomposition for sweep impl='c' (multi-GPU FD CUDA)

Branch `feat/cuda-domain-decomp` (worktree `/home/wangs0j/sweep-local/sweep-dd-cuda`),
based on `feat/model-parallel` (a4d6392), which is 99 commits behind `origin/dev`
(→ rebase is step 0, see M1).

Goal: run a single shot whose spatial grid does not fit on one GPU by splitting
the grid across N GPUs with per-step halo exchange, for the `impl='c'` CUDA path —
forward AND gradient (autograd) — while keeping the single-GPU path untouched.

---

## 1. What we learned from SPECFEM (and what transfers)

Research notes from SPECFEM3D / SPECFEM3D_GLOBE sources (verified subroutine
names; see section refs):

| SPECFEM mechanism | Where | FD/sweep translation |
|---|---|---|
| Static structured decomposition (GLOBE: `NPROC_XI × NPROC_ETA` slices, no graph partitioner) | `src/meshfem3D/create_MPI_interfaces.f90` | Regular-grid FD needs no SCOTCH/METIS: split along grid axes; `MeshTopology` (already in `feat/model-parallel`) is exactly this |
| One interface = one neighbor rank = one flat buffer = one MPI message | `nibool_interfaces_ext_mesh`, `ibool_interfaces_ext_mesh` | Per-neighbor halo slabs; index lists degenerate to regular strided slices |
| Two-phase overlap: compute **outer** elements → post `isend/irecv` → compute **inner** elements → wait + assemble | `compute_forces_viscoelastic_calling_routine.F90` (`iphase=1/2`), `assemble_MPI_vector_async_send/recv` | Phase-2 optimization: split stencil launch into boundary strip (width M) + interior, exchange during interior compute. NOT needed for v1 correctness |
| GPU: pack kernel → async D2H on `copy_stream` → MPI on host → async H2D → unpack kernel; 2 streams + pinned buffers | `src/gpu/assemble_MPI_vector_cuda.cu`, `prepare_boundary_accel_on_device.cu` | Single node: NCCL does D2D directly, no host staging. Pack kernel only needed for non-contiguous (x/y-cut) slabs — torch `.contiguous()` + NCCL covers v1 |
| CUDA-aware MPI: same `isend/irecv` code on raw device pointers | GLOBE `assemble_MPI_vector_gpu.c`, `WITH_CUDA_AWARE_MPI` | NCCL via `torch.distributed` gives this for free |
| SEM **sums** interface contributions (partial forces, atomicAdd, ordering issues → `_w_ord` variant) | `assemble_boundary_accel_on_device.cu` | **FD copies ghost layers — one-way overwrite. No atomics, no ordering concern, bitwise reproducibility is free.** Halo width = so/2 (not 1 node layer) |
| Absorbing boundaries only on *physical* faces; partition cuts are never absorbing | Stacey faces from physical boundary surfaces only | Already implemented: `build_rank_pml_widths` zeroes PML on neighbor-facing sides |
| Source/receiver owned by exactly one rank (min-distance tie-break); only owner injects/records | `locate_source.F90`, `islice_selected_source` | Already implemented: `partition_global_coords`; ownership tie-break at cut planes must be deterministic (half-open intervals) |
| Adjoint runs the **identical** decomposition/exchange on `b_accel` | `FORWARD_OR_ADJOINT==3`, same `iphase` loop | Our backward needs per-step halo exchange for BOTH the adjoint field and the boundary-saving reconstructed forward field |
| Overlap economics: outer/inner ratio must be small; ≤3% MPI wait at 27.5% outer (Komatitsch et al. JCP 2010) | 192-GPU weak scaling | Fast FD kernels shrink the overlap window → don't make per-GPU tiles too small; report outer fraction in benchmarks |
| Load balance: weight by cell cost; PML weighting was a TODO even in SPECFEM | `part_decompose_mesh.F90` LOAD constants | PML cells cost more; with edge ranks owning all PML, interior ranks idle. v1: ignore (document); later: shift cut positions |

Key structural difference to exploit: FD halo exchange is *copy* semantics —
after exchange, every cell's stencil inputs are bit-identical to the
single-domain run, so **DD forward should be bit-exact vs single GPU**. That is
the acceptance bar (SEM could never have this).

## 2. Current state of sweep

### impl='c' architecture (from code survey of `src/sweep/csrc/`)

- **Whole nt-loop lives in C++**: `csrc/cuda/equations/acoustic2d/forward.cu`
  (`acoustic2d::forward`, loop ~L131–200), one Python→C++ call per propagation
  (`propagator/_c.py` `Warpper.forward/backward`). **No per-step hook** — this
  is the gap to close.
- Per step (acoustic2d): air-clear (topo) → fused stencil+CPML kernel
  (`acoustic2nd<Order>`) → boundary-saving strips → `add_source` →
  `record_kernel` → pointer-rotate `swap_pml()` (u: 3-cycle, psi/zeta double
  buffers: 2-cycle). All state (u_prev/now/next, psi/zeta(+n), boundary rings,
  checkpoints) lives in **Python-allocated tensors bound by reference** —
  cross-call persistence of a stepped API is naturally available.
- Layout 2D `(B, nz, nx)`, 3D `(B, nz, ny, nx)`, x fastest-varying. An x-cut
  halo is a strided column block (needs `.contiguous()` pack); a z-cut slab
  would be contiguous, but we keep **z unsplit** (free surface, image method,
  shallow sources) — consistent with `parallel/pml.py` v1 ("z never split").
- CPML: coefficient vectors per axis from `equation.init_abc` (per-rank widths
  already wired via `PropBase.model_parallel`); psi/zeta memory tensors are
  full-grid-shaped. At an interior cut, PML width is 0 → psi/zeta stay ≡0
  there; the stencil kernel's `in_pml` test is position-based
  (`ix < abcn+halo …`) — with per-side widths it must consult the *rank-local*
  widths (C++ `SolverContext` currently assumes symmetric `abcn`; needs
  per-side fields). **Constraint to enforce: cut planes must be ≥ halo away
  from any global PML band** (true for any sane partition since PML hugs the
  global edge).
- Backward: `backward_bs` reconstructs forward from boundary rings + last-two
  snapshots, fused adjoint kernel, same C++ nt-loop → same stepping treatment
  needed.

### `feat/model-parallel` inventory (reuse as-is)

- `parallel/_topology.py` `MeshTopology` — rank-grid arithmetic, no dist dep.
- `parallel/mesh.py` `ModelParallelMesh` — topology + `model_pg`/`shot_pg`
  process groups (shot-parallel × model-parallel 2-D process grid).
- `parallel/halo.py` `HaloExchange` (autograd.Function) + `exchange_halos` —
  fused `dist.batch_isend_irecv`, forward fills halos in place, backward does
  the adjoint (halo grad → neighbor interior +=, own halo grad zeroed).
- `parallel/pml.py` `build_rank_pml_widths` — zero PML on neighbor-facing
  sides; wired into `PropBase.init_abc` (cache key includes rank coord).
- `parallel/routing.py` `partition_global_coords` — global→tile coords.
- Tests: `test_model_parallel_{topology,halo,pml}.py` (~900 lines, single-proc
  + gloo).

What's missing = this project: the impl='c' C++ loop cannot yield control
between steps, so `HaloExchange` has nowhere to run.

## 3. Design

### D1 — Orchestration: chunked C++ stepping API + Python-side NCCL (v1)

Options considered:

- **(A) Stepped C++ API**: parameterize the C++ time loop as
  `forward(..., it_begin, it_end)`; Python loops over steps, calling
  `exchange_halos` between calls. Reuses the entire `feat/model-parallel`
  stack including its autograd backward. Per-step cost ≈ one extension call
  (~20 µs) + NCCL messages — negligible against ms-scale stencil kernels on
  DD-sized grids.
- **(B) In-C++ comm**: ncclSend/Recv or P2P memcpy inside the C++ loop, with
  outer/inner overlap (SPECFEM-style). Lowest overhead, but C++ must learn
  topology/process groups (extracting ncclComm from a torch ProcessGroup is
  fragile), and forward/backward/bs/ckpt variants all need it.

**Decision: (A) for v1**, profile, and only then consider (B) as a targeted
perf phase (M6). DD's target workload (large 3-D grids) amortizes per-step
Python overhead; correctness and gradient support come first.

Stepping-API details:

- Add `it_begin/it_end` to `ForwardInput` (default `0/nt` → existing behavior
  and bindings unchanged; single-GPU path stays bit-identical).
- Buffer-role rotation across calls: `swap_pml()` rotates C++-side pointers
  only; with per-call re-binding from the Python tensor list, Python must
  re-order the wavefield list between calls by `(steps % 3)` for u and
  `(steps % 2)` for psi/zeta double buffers (or C++ returns the rotation
  offset). With per-step calls (k=1) this is a fixed permutation. Verify with
  the M1 equivalence test.
- `record`/source indexing already uses absolute `it` — no change.

### D2 — Split axes: x (2-D), x+y (3-D); z never split

Matches `parallel/` v1. Halo slabs along x are strided → `HaloExchange`
already handles pack/unpack via tensor slicing; a dedicated pack kernel is an
M6 optimization, not a correctness need.

### D3 — What gets exchanged, and how wide

- Per step, **only the u-wavefield halos**, width `M = so // 2`, on split axes.
- psi/zeta need no exchange: at interior cuts PML width is 0, so their values
  are identically zero within any cut-adjacent band (enforced by the
  cut-vs-PML distance constraint above). Assert this constraint at setup.
- Exchange point in the step: after `swap_pml()` (i.e. on the new `u_now`)
  and before the next stencil — equivalently, between two stepped calls.

### D4 — Backward/gradient (M4)

Two sub-fields need halos during the backward sweep:

1. the adjoint field (handled by `HaloExchange.backward` automatically if we
   drive backward through autograd, or by an explicit forward-direction
   exchange on the adjoint tensors if we drive `backward_bs` manually), and
2. the boundary-saving **reconstructed forward field** — reconstruction is
   time-reversed forward propagation, which at a cut needs neighbor data.

Options: (i) exchange reconstruction halos every backward step (2 exchanges
per step, no extra memory) vs (ii) save interface strips to the boundary ring
during forward (extra memory `nt · M · interface_area`, no extra backward
comm). **Decision: (i)** — mirrors SPECFEM's adjoint (`b_accel` runs the same
assembly), keeps the boundary-saving format untouched, and comm is already
paid for the adjoint exchange in the same step. Revisit only if profiling
shows backward comm-bound.

Per-tile boundary saving itself is unchanged: each rank saves strips of its
*own* faces; on neighbor-facing faces the "boundary" is reconstructed via (i)
rather than from the ring. Concretely: `save_width` strips are only needed on
physical faces; interior-cut faces read neighbors. Implementation detail to
settle in M4: the existing `backward_bs` C++ reconstructs using all 4/6 faces
— for cut faces we feed the halo-exchanged values instead of ring reads.

### D5 — Process model & dev-time testing on 1 GPU

- Production: `torchrun` multi-process, NCCL, one rank per GPU, via
  `ModelParallelMesh` (also gives `shot_pg` for shot-parallel × model-parallel
  later).
- **Dev harness (critical, KW60443 has exactly 1 GPU; NCCL forbids two ranks
  on one device):** a single-process "manual two-tile" driver that builds two
  propagators (tiles) on the same GPU and copies halos with plain tensor ops
  between stepped calls — numerically identical to the NCCL path. All
  correctness work happens here; NCCL/torchrun runs are validated on ibex
  multi-GPU nodes (V100/A100) at milestone gates.

## 4. Milestones

**M1 — Rebase + stepped forward API (acoustic2d), single-GPU equivalence**
- Rebase `feat/cuda-domain-decomp` onto `origin/dev` (99 behind; grad-align
  rebased 75 with 0 conflicts, expect similar).
- `it_begin/it_end` in `ForwardInput` + loop bounds in `acoustic2d/forward.cu`;
  Python `Warpper`-level helper that runs nt steps as nt (or k-sized) chunks
  with buffer-role permutation.
- Accept: chunked run (k=1 and k=7) **bit-exact** vs single-call run, with and
  without PML, with boundary saving on. pytest in `test/`.

**M2 — Two-tile DD forward, acoustic2d, single GPU**
- Tile-local model slicing (pad each tile with M ghost cells on cut sides;
  per-side PML widths through existing `model_parallel` wiring; C++
  `SolverContext` per-side pml widths for the `in_pml` test).
- Manual two-tile harness: per-step halo copy between stepped calls.
- Accept: 2-tile (x-cut) forward record + final wavefield **bit-exact** (fallback
  tolerance rel ≤1e-6 if a fused-multiply boundary effect appears — investigate
  before accepting) vs single-domain, canonical 48×56 case + one PML-heavy case
  + free-surface case; source/receiver on both tiles and on the cut line.

**M3 — NCCL path on real multi-GPU**
- Glue `ModelParallelMesh` + `exchange_halos` + stepped API into a
  `DDPropagator` (thin orchestrator in `sweep/parallel/`, NOT in the solver
  core hot path).
- Run on ibex multi-GPU node via `torchrun --nproc-per-node=2..4`.
- Accept: NCCL 2/4-rank == manual harness == single domain (same tolerance as
  M2); a smoke perf number (% step time in exchange).

**M4 — Gradient: stepped backward + DD adjoint (acoustic2d)**
- `it_begin/it_end` for `backward_bs` (+ plain `backward`); per-step adjoint
  halo exchange + reconstruction halo exchange (D4-i); cut-face reconstruction
  feeding.
- Accept: canonical gradient case (48×56, suite defaults), 2-tile vs
  single-domain `impl='c'`: grad rel_l2 < 1e-5, cosine > 0.99999 (target:
  near-bit); gradient-mode suite green on the DD config.

**M5 — acoustic3d + 2-axis (x,y) decomposition**
- Same stepping + halo on `(B, nz, ny, nx)`; 2-D rank grid; corner/edge
  ordering already defined by `halo.py`'s sequential-axis exclude logic.
- Accept: 3-D bit-exact forward (2×1, 2×2 rank grids) + gradient ≥ M4 bar;
  demo: a grid >1 GPU's memory runs on 4 GPUs (the actual point of DD) with
  per-GPU peak memory ≈ 1/N + halo overhead measured.

**M6 — Efficiency loop (HARD requirement, user 2026-06-11)**
- Requirement: decomposition must not hurt efficiency. Operationalized:
  - **Weak scaling (primary, user's wording "分到多卡计算时间不变")**: take a
    model that one GPU computes in time t; give each of N GPUs a tile of that
    same size (global model N× larger) → wall time stays ≈ t.
    Bar: efficiency t/t_N ≥ ~90–95%.
  - **Strong scaling (also reported)**: same model on N GPUs → ≈ t/N.
- **Iterate (loop) until the bar is met or shown structurally unreachable**:
  measure → identify overhead (per-step Python round-trip, exchange latency,
  pack cost, no overlap) → apply next lever → re-measure. Levers in order:
  per-step overhead trim / k>1 fusing where legal, comm/compute overlap
  (boundary-strip kernel first, NCCL exchange concurrent with interior kernel
  — SPECFEM iphase pattern), pack kernels, in-C++ NCCL/P2P (D1-B).
- Hardware: ibex V100 nodes (user: no local multi-GPU; request via SLURM).
- Benchmark discipline: idle GPUs only (nvidia-smi guard), ≥1.5 s warm-up,
  REPEATS≥7 + CV sniff, sweep size/order/nt so conclusions generalize.

Out of scope for now: other equations (elastic2d/3d, VTI, DAS — the stepping
API generalizes mechanically once acoustic2d/3d land), multi-node (NCCL
across nodes works untested; not a v1 target), z-axis splitting, shot×model
2-D process grid end-to-end (plumbing exists in `mesh.py`).

## 5. Validation matrix (recurring)

| Check | Config | Bar |
|---|---|---|
| Stepped == monolithic | 1 GPU, k∈{1,7}, PML on/off, BS on | bit-exact |
| DD forward == single | 2/4 tiles, canonical + PML-heavy + free-surface, src/rec straddling cuts | bit-exact (≤1e-6 fallback) |
| DD gradient == single | canonical gradient case | rel_l2 <1e-5, cos >0.99999 |
| NCCL == manual harness | ibex 2–4 GPUs | bit-exact |
| Single-GPU path untouched | full existing pytest + gradient suite (54) | no regressions |

## 6. Dev environment notes

- Build: `SWEEP_BUILD_CUDA=1 pip install -e .` with `TORCH_CUDA_ARCH_LIST=8.9`
  (KW60443) / `7.0;8.0` (ibex V100/A100); ninja on PATH; verify
  `is_torch_binding_available()`. Run with `PYTHONPATH=src` in this worktree.
- KW60443: 1× RTX 6000 Ada → all M1/M2/M4 dev here (manual harness).
  ibex multi-GPU for M3/M5/M6. KW60443 is the only git host.
- `ifwitorch` conda env.

## 7. Risks / open questions

1. **Rebase distance** (99 commits): `csrc/` has been active (race-free CPML,
   fused adjoint). Do the rebase FIRST, before touching C++.
2. `in_pml` position test in kernels assumes symmetric `abcn`; converting
   `SolverContext` to per-side widths touches every equation's kernels'
   assumptions — restrict edits to acoustic2d/3d, leave others on the
   symmetric path (compile-time identical when widths are symmetric).
3. Buffer-role rotation across stepped calls is the most likely source of
   subtle wrongness — M1's bit-exact gate exists precisely for this.
4. Per-step Python overhead on small tiles: accepted for v1 (DD targets big
   grids); measure in M3 and report, don't pre-optimize.
5. Boundary-saving + DD interaction (cut-face reconstruction feed) is the
   least-explored corner; if it turns ugly, fallback for M4 is
   `save_all_wavefields=True` first (memory-rich validation), then BS.
6. geophyai solver-core edits happen freely in this worktree, but the branch
   targets a PR eventually → user sign-off on this plan before
   implementation, and again at PR time (per collaboration rules).

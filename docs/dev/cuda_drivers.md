# CUDA driver skeletons (`eq_driver.cuh` / `sg_driver.cuh`)

For developers who need to read or extend the `impl="c"` time loop. The user-facing
"add an equation" walkthrough is [Extending](../user-guide/extending.md); this page
covers only the driver layer under `src/sweep/csrc/cuda/` — its structure, its hook
ordering, and the gates that verify it.

## 1. Architecture

**Two skeletons, one traits struct per equation, a thin entry layer.** Every equation
directory used to carry a hand-copied forward driver (~300 lines) and a four-mode
backward driver (~1000 lines), 60–80% of it line-for-line identical. Cross-cutting
capabilities — stepped ranges, phase splits, boundary-tail truncation — existed only
in whichever copies happened to implement them. Now:

| Layer | File | Contents |
|---|---|---|
| Skeleton (acoustic family) | `common/eq_driver.cuh` | `template <class Eq>`: `generic_forward` / `generic_backward` (full) / `generic_backward_bs` / `generic_backward_ckpt` / `generic_backward_recursive_ckpt`, plus `GenericForwardRunner` / `GenericBackwardBsRunner`. Second-order displacement form: `u_prev/u_now/u_next` buffer rotation, single-field sources and receivers, boundary saving stores one field, backward produces `grad_wavelet` and illumination. Members: acoustic2d, acoustic3d, acoustic_vrz2d. |
| Skeleton (staggered family) | `common/sg_driver.cuh` | The five `sg_generic_*` entries plus `SgForwardRunner` / `SgBackwardBsRunner`. First-order velocity–stress form: fields update in place (no rotation), each step is a velocity substep followed by a stress substep, sources and receivers loop over field indices, boundary saving stores a list of fields per step, `last_two` is a snapshot of the final field set, and backward has neither `grad_wavelet` nor illumination. Members: elastic2d/3d, das_mu2d/3d, elastic_tti_sg2d/3d, elastic_vr2d. |
| Per-equation traits | `equations/<eq>/driver_traits.cuh` | One `struct Driver`: constants, type aliases, and all-static composite launch hooks, laid out in five sections — [1] identity, [2] forward, [3] full, [4] bs, [5] ckpt+recursive — and within a section in skeleton call order. Reading the traits top to bottom is close to reading the execution flow. |
| Thin entries | `equations/<eq>/forward.cu`, `backward.cu` | One line per entry: `return eqdrv::generic_forward<Driver>(in);` and so on, plus the `forward_runner` / `backward_bs_runner` factories. Exceptions: the APM entries of elastic2d/3d (`apm_forward` / `apm_backward*`) are still hand-written in the same file, and acoustic_vrz2d's chunk and recursive checkpoint backwards stay hand-written because they scan linear segments rather than bisecting the way the acoustic ones do. |

`eq_driver.cuh` is a line-for-line transcription of acoustic2d's hand-written driver
and `sg_driver.cuh` of elastic2d's; not one line of physics kernel changed. The hook
granularity is deliberately **coarse** — one composite operation per step, not one
hook per kernel — because family members differ in the **order** of operations within
a step (2-D acoustic images after source injection and the swap, 3-D images before
injection, VRZ injects before the restore). That order is load-bearing to the bit, so
the differences live in the equation's hooks and the skeleton carries no per-equation
branch. The skeleton owns only what is genuinely shared: input validation,
stepped/phase bookkeeping, buffer binding with the legacy fallback allocation,
boundary and checkpoint runtime orchestration, the time loop, and output packing.
Still hand-written: das2d/3d (a third shape — derivative-buffer form),
elastic_tti_2nd2d, acoustic_lsrtm2d/3d, acoustic_vrz3d, acoustic_vti_1st_2d/3d.

### Persistent runners

`GenericForwardRunner` / `GenericBackwardBsRunner` (acoustic) and `SgForwardRunner` /
`SgBackwardBsRunner` (staggered) implement `IForwardRunner` / `IBackwardRunner` from
`shared/wavetypes.h`. **The constructor runs the entire prologue once** — validation,
binding, `SolverContext`, CPML, boundary saver and runtime, checkpoint runtime,
`State`, workspace; declaration order is construction order and destruction is its
reverse, matching the stack unwinding of the hand-written functions — and
`run(it_begin, it_end, step_phase)` / `run(bw_it_begin, bw_it_end, step_phase)` is
**only the time loop**. Actions specific to the first segment
(`seed_reconstruction` / `zero_adjoint_if_first_segment_bs` / `seed_recon` / the
initial prefetch) stay inside `run()`, gated on whether this range is the first
segment, exactly as in the hand-written versions.

A monolithic entry is just "construct, then `run` once" — `generic_forward<Eq>` is
`GenericForwardRunner<Eq>(in).run(in.it_begin, in.it_end, in.step_phase)` — so the
bit-exactness gate hammers the runner path itself. The reuse contract, enforced in
C++ on the second `run`: gpu-direct boundary storage and no checkpointing, the only
two modes whose cross-call state lives entirely in Python-bound buffers. The
motivation is that DD paid the ~1–2 ms host prologue on every step (30–100× the
launch floor), and a CUDA graph cannot fix host logic; after the move to runners,
end-to-end DD is 1.8–4.4× on elastic 2-D, ~1.5× on Elastic3D and 1.05–1.8× on
acoustic. On the default path `sweep.backend.c.runners` wraps the core's runner C
API (`core/capi.h`, `core/runner.h`) in `ForwardRunner` / `BackwardRunner` and
exports the factories as `{C_NAME}_forward_runner` / `{C_NAME}_backward_bs_runner`
for the equations listed in `sweep.backend.c.RUNNER_EQUATIONS` (`module.cpp` does
the same through `py::class_` for the `SWEEP_JIT_FULL=1` shim); on the Python side
`WaveEquation._compiled_runner_factories()` resolves the same convention and falls
back to the per-call stepped path when a factory is missing.

### Stepped ranges and DD

* **forward** advances `[it_begin, it_end)` (`it_end < 0` means `nt`). A continuation
  with `it_begin > 0` must bind the Python-side `wavefields`, `record_out`,
  `u_allt_out` (when `save_all_wavefields`) and `boundary_gpu` (under boundary
  saving) — otherwise the internal `allocate()` silently zeroes the propagation
  state. `save_last_state` runs only on the final segment, `it_end == nt`. Acoustic
  boundary-tail truncation (`boundary_tail_steps = K`) shifts `bs_it0` using the
  global `it`, so it composes transparently with segmentation.
* **backward** runs in reverse from `bw_it_begin` (exclusive upper end, `< 0` means
  `nt`) down to `bw_it_end` (inclusive lower end). When stepped it requires
  `adjoint_wavefields` (`ADJ_WF_COUNT` of them), `grads_out` and `illum_out` to be
  bound, and in bs mode also `forward_wavefields` (the `RECON_WF_COUNT`
  reconstruction list). DD (`cut_face_mask != 0`) **supports `backward_bs` only**:
  the full path calls `set_cut_mask(0)`, and both checkpoint modes refuse stepped,
  phased and cut inputs. Acoustic DD supports gpu-direct or cpu storage (not disk);
  staggered DD supports gpu-direct only.
* Scratch is declared, not allocated: `forward_workspace_nvar` and
  `backward_workspace_nvar` (or `backward_workspace_shapes`) in `cuda_layout` say how
  many padded grids per shot the propagator hands the driver as `forward_workspace`
  (transient, one call) and `adjoint_workspace` (persistent, zeroed before every
  gradient-bearing forward). A driver takes each slot with `pool_or_zeros`
  (`common/cudautils.h`), names the slots in an enum, checks the declared count at
  entry, and allocates nothing of its own; an unbound pool still falls back to a
  fresh zero tensor per slot.
  Gradient outputs follow the same rule: the propagator allocates `grads_out`
  (zeroed; `grads_out_has_wavelet` decides whether slot 0 is `grad_wavelet`) on the
  monolithic path as well as the stepped one, and every driver binds them
  (`bind_grads`, or slot by slot with `pool_or_zeros`) rather than allocating.
  The full-mode history is the same story: `cuda_layout.save_all_shape(B, nt, grid)`
  declares the driver's own `u_allt` layout, the propagator allocates it per
  gradient-bearing full-mode call and binds it as `u_allt_out`, and the driver
  takes it through `bound_or_zeros` (shape-checked).
  The record is the same: `cuda_layout.record_shape(B, nrec, nfield, nt)` declares the
  driver's layout (`record_single` / `record_multi`), the propagator allocates it per
  call and binds it as `record_out` on the monolithic path as well as the stepped one.
  Derived model coefficients (Lame parameters, VTI stiffness, 1/z) are the same:
  `cuda_layout.derived_model_nvar` (an int, or `fn(mode)` for a driver that derives
  only in some modes -- the DAS full backward reads the stored strain history and
  gets no slot) declares how many model-shaped slots the propagator hands a
  forward or backward as `derived_models` (`torch.empty`: the driver writes every
  cell), and one fused kernel per family
  (`common/derived_models.h`, slots named by `LameSlot` / `VtiSlot` / `VrzSlot`)
  fills them from the bound models, reading each model once -- the torch
  expressions that used to do this inside the call (and their temporaries) are
  gone. The kernels use `__f*_rn` intrinsics in the torch expressions' own
  association, so the coefficients are bit-identical under `--use_fast_math`.
  The boundary-saving reconstruction is the same: `cuda_layout.reconstruction_nvar`
  (an explicit `bs_reconstruction_nvar`, or the slot table's `recon` list -- the DD
  path's source of the same count) says how many zeroed padded grids the propagator
  hands the bs backward as `forward_wavefields` (the physical fields it steps
  backwards from `u_last_two`, plus the carriers the imaging reads; no CPML memory,
  the reverse loop injects boundaries instead). A driver takes the list through
  `wavefields_bound` (count + per-slot geometry check, `common/cudautils.h`), binds
  the physical fields with `use_pml=false` (or a `bind_physical`/`bind_recon` of its
  own struct that leaves the memory members undefined -- `view()` then hands
  `ptr_or_null` for them) and the carriers from the tail of the list, and allocates
  only when nothing was bound.
  The checkpoint modes take the same route: `forward_wavefields` holds the replay
  STATE sets (one set = the forward slot list without the psi double-buffer
  shadows, `cuda_layout.checkpoint_state_nvar` when declared; `1 + depth(max_segment)`
  sets for a bisecting driver that declares `recursive_state_depth`), taken with
  `wavefield_set(list, k, n, what)` and bound with the struct's full bind; the
  segment histories come from `checkpoint_replay` (`checkpoint_replay_shapes(B, nt,
  shape, max_segment, mode)`, bound once per call and narrowed per segment with
  `pool_rows`); the velocity carriers and other per-call scratch come from the
  adjoint workspace pool (`backward_workspace_shapes(..., mode)`). A driver whose state
  set is not the forward slot list declares `checkpoint_state_nvar` explicitly (LSRTM: 7 / 9).
  Scratch that is not a padded grid (complex spectra as float32 `[B, 1, *grid, 2]`
  slots viewed with `torch::view_as_complex`, a cuFFT work area as a flat slot) is
  declared by shape: `forward_workspace_shapes(B, shape)` (uninitialised, transient) and
  `backward_workspace_shapes`; the visco driver runs its FFTs on the plan ATen itself
  would build (`at::native::detail::CuFFTConfig`, work area from the pool), so the
  spectra are bit-identical to `at::fft_fft2` and nothing is allocated per step.
  Illumination accumulators come as `illum_out` (`cuda_layout.illum_nvar`, bound by the
  monolithic backward when illumination was asked for) and are allocated in C++ only for
  a caller that binds nothing.
  The boundary saver's last tensors went too: a scaled boundary store (`storage_dtype`
  `int8` or `fp16`, on gpu, cpu or disk alike) quantizes through a one-timestep FP32 band
  per face, and those bands now come as `boundary_staging` (`Layout.staging_shapes`, the
  persistent face shape with the time axes collapsed to one slot), allocated in the
  propagator's boundary GPU allocator beside `boundary_gpu` and bound by forward and
  backward; `fp32`/`bf16` storage never stages and gets no bytes. They are zeroed once at
  allocation rather than per call: every cell the band kernel writes is overwritten before
  `launch_quantize_*` reduces over it, and the cells it never writes -- the tangential pad
  of a `tangent_pad > 0` layout, a DD cut face -- must read 0 (they enter the per-block
  max) and stay 0, since the only other writer maps a cell quantized from 0 back to 0.
  The last per-call tensors went the same way: un-injecting a source in a reverse
  reconstruction and injecting a stress residual use `add_source_signed` /
  `add_source_3d_signed` (`common/common.cu`: the sample's sign bit flipped, exact) instead
  of a negated copy of the source, and the staggered full-mode backward's read-only zero
  velocity (v(nt) for the last reverse step's imaging) is one pool slot declared for the
  full mode only (`SgCarrierSlots::FULL_ZERO`, the index the checkpoint modes give their
  first velocity carrier -- the pools are per mode, so the slot never aliases).
* On the Python side, `stepped=True` in `equations/cuda_layout.py` declares that both
  forward and `backward_bs` honour ranges. Only a migrated equation may set it: an
  equation that does not honour ranges will not raise, it will run the whole record
  on every stepped call and return zeros. `dd_backward_phases=True` declares that the
  backward implements numbered phases. `ModelParallel` admits on those two flags; the
  schedule itself is declared in `parallel/dd_spec.py` and interpreted by
  `parallel/dd_propagator.py`.

**Two different phase splits** (`step_phase`):

| | Acoustic family (`eq_driver`) | Staggered family (`sg_driver`) |
|---|---|---|
| forward | **Spatial strip split.** Phase 1 is the M-wide physical edge strips adjacent to the cut face only (`cut_face_mask` bit0/bit1 = x_lo/x_hi; v1 supports x cuts only), with no boundary saving, source, receiver, swap or checkpoint; phase 2 is the strict complement plus the whole tail. No grid point may run twice — a double-buffered CPML psi write would be advanced twice. Requires `it_end == it_begin + 1`, `cut_face_mask != 0`, and a tile at least 2M wide. Purpose: overlap the halo exchange with phase 2's compute (`ACOUSTIC_FWD_OVERLAP`). | **Physical split.** Phase 1 is the whole-grid velocity substep; phase 2 is the whole-grid stress substep plus the source/checkpoint/boundary/receiver tail. DD exchanges v between the two phases and s after phase 2, so the stress columns next to the cut read the exchanged velocities rather than locally recomputed ones. No cut precondition, and legal at `world_size == 1`. |
| backward | **No phases**: `check_stepped_backward` refuses `step_phase != 0` loudly. One call per reverse step, then `(lambda, recon u)` are exchanged; the floor is 0 (the adjoint-only tail at `it == 0` still contributes `grad_wavelet`, `BS_HAS_IT0_ADJOINT_TAIL`). | **bs phases 3 → 1 → 2** (`sg_check_stepped_backward` allows 0–3; phased is restricted to `backward_bs` and to single-step ranges). 3 is injection only (`fix_rho_grad_at_sources` / `inject_residuals` / `uninject_forward_source`, and the first segment's `seed_recon` belongs to this phase); 1 is `bs_stress_half` (stress NOPML reconstruction, strip restore, imaging, receiver-rho, stress adjoint half); 2 is `bs_velocity_half` (velocity adjoint half, carrier capture, velocity NOPML reconstruction, restore, prefetch). Monolithic `step_phase = 0` does the injection at the top of the loop and executes exactly the same operator sequence. The floor is 1. |

VRZ: the 3-D sibling's backward has a further four-phase coupling exchange (the
gradient is the divergence of a coupled field, so the cut needs neighbour values) and
is still hand-written. acoustic_vrz2d's kernels are not ranged
(`launch_step_range` refuses a sub-range loudly) and its backward has no phases, so it
is stepped but refused by DD.

**`cut_face_mask`**: `SolverContext::set_cut_mask` defines bits 0..5 as x_lo, x_hi,
z_lo, z_hi, y_lo, y_hi, and each equation restricts the legal bits with
`CUT_MASK_BITS` (0xF for 2-D, 0x3F for acoustic3d, 0x33 for elastic3d — x/y only —
and 0x0 for equations without DD). Once set, `phys_x0()/phys_x1()` and friends make
the physical boundary on the cut side a stencil halo (M) instead of pad+M, which
affects the boundary strip restore (skipped on the cut face), the rim zeroing at seed
time, the NOPML exclusion band, the `pure_interior` predicate of the fused adjoint,
the `in_pml` predicate of the adjoint prepare kernels, and the `wxl/wxh/wzl/wzh` of
the band and strip kernels (zero on the cut side). Note that the free-surface bit
mask in the same `SolverContext` uses the opposite axis order (bit0 = z_lo);
`test_cut_face_mask.py` pins that difference.

## 2. Hook timing map (verbatim from the two skeleton file headers)

`common/eq_driver.cuh`:

```text
// Shared per-equation driver skeleton.
//
// Every equation directory used to hand-copy a ~300-line forward driver and a
// ~1000-line four-mode backward driver; they were 60-80% line-identical, and
// cross-cutting abilities (the stepped it_begin/it_end range that domain
// decomposition needs, phase-split launches, boundary-tail truncation) existed
// only in the copies that happened to have them.  This header owns that
// skeleton ONCE, as ``template <class Eq>`` drivers; an equation supplies a
// traits struct (constants + composite launch hooks) and 1-line entry points.
//
// Hook granularity is deliberately COARSE — one hook per in-step compound
// operation, not per kernel.  The three families disagree on the in-step
// ORDER (e.g. 2-D acoustic images the boundary-saving gradient after the
// forward source injection and swap, 3-D acoustic before the injection, VRZ
// injects before the restore), and that order is bit-load-bearing.  The
// template owns what is genuinely identical: input validation, stepped/phase
// bookkeeping, buffer binding with legacy fallback allocation, Boundary- and
// Checkpoint-runtime orchestration, the time loops, and output packing.
//
// Bit-exactness contract: this skeleton is a line-faithful transcription of
// acoustic2d's drivers (the reference, gated by bitgate tiers A/B/C/T and
// ddgate).  Physics kernels are not touched by the migration.  Where another
// equation's copy disagreed with acoustic2d in loop structure, the difference
// lives in that equation's hooks, never in a per-equation branch here.
//
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_forward / (backward:
// check_stepped + validate_backward + bind_backward_outputs / alloc_grads +
// rtm gate), bind_or_alloc_* wavefields, alloc_cpml, setup_ctx,
// init_aux_slabs, make_state, make_bwd_workspace.
//
// generic_forward — per it in [it_begin, it_end):
//   launch_step_range          the whole per-range stencil step (air-clear
//                              prepass included); DD phase 1 = the cut-side
//                              M-wide strips, phase 2 = strict complement,
//                              unphased = (0, nx)
//   save_boundary_fwd          BS strips (when use_boundary_saving)
//   inject_source_fwd          source injection
//   record                     receiver sampling
//   rotate_buffers             u_pre/u_now buffer-role rotation
//   capture_allt               deferred u_allt snapshot (only 3-D uses it)
//   <checkpoint save>          shared runtime, not a hook
//   after the loop: save_last_state (final u pair for backward_bs)
//
// generic_backward (full storage) — per reverse it:
//   adjoint_step               adjoint stencil; with HAS_FUSED_FULL_IMG the
//                              imaging of u_forward[it+1] fuses into it
//   inject_adjoint_source      residual injection
//   rotate_adjoint_buffers     adjoint buffer-role rotation
//   accumulate_source_grad     grad_wavelet sampling
//   image_step                 standalone imaging / RTM+illumination taps
//                              (skipped when fused, except for RTM)
//   after the loop (fused only): one trailing image_step at it == 0.
//
// generic_backward_bs — per reverse it, floor max(max(it_lo, 1), bs_stop):
//   adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
//   accumulate_source_grad     same four as full mode
//   bs_recon_step              reconstruction (un-inject, NOPML reverse,
//                              strip restore) + gradient imaging, in the
//                              equation's exact order
//   bs_rtm_tap                 RTM / illumination tap
//   before the loop (first segment): seed_reconstruction from u_last_two;
//   after the loop (BS_HAS_IT0_ADJOINT_TAIL): the four adjoint hooks once at it == 0.
//
// generic_backward_ckpt — per chunk: replay then reverse:
//   replay:  replay_step / inject_source_fwd / rotate_recon_buffers
//   reverse: adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
//            accumulate_source_grad / image_step
//
// generic_backward_recursive_ckpt — bisection over each ckpt segment; a
//   leaf runs one replay triple, then the reverse-five of ckpt mode with
//   the imaging fed from the leaf's scratch u.
// ---------------------------------------------------------------------------
```

`common/sg_driver.cuh`:

```text
// Shared driver skeleton for the STAGGERED (elastic-family) equations.
//
// Sibling of eq_driver.cuh (the second-order acoustic-family skeleton), same
// philosophy: the control flow every hand-written copy shared lives here once,
// per-equation physics stays in composite traits hooks, and cross-cutting
// abilities (the stepped it_begin/it_end range, the physics phase-split, the
// segmented backward) become properties of the skeleton instead of of whichever
// copies happened to implement them.
//
// The staggered shape differs from the acoustic one in ways that are
// bit-load-bearing, which is why it is a second template rather than more
// hooks on the first: fields update in place (no buffer-role rotation), each
// step is a velocity substep then a stress substep (the phase-split is a
// PHYSICS split, not a spatial strip split), sources/receivers are per-field
// index loops, boundary saving stores a field LIST per step, last_two is a
// final-state field snapshot, and the backward computes no grad_wavelet and no
// illumination.  Reference transcription: elastic2d (gated by bitgate tiers
// A/B/C/T and ddgate).  das2d/das3d (derivative-buffer shape) and
// elastic_tti_2nd2d (second-order displacement — acoustic-shaped) are NOT this
// family.
//
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_backward (backward only),
// parse_models, setup_ctx, bind_or_alloc_* wavefields, init_aux_slabs,
// alloc_cpml, bind_grads / alloc_grads, make_workspace, make_state,
// adjoint_source_signs.
//
// sg_generic_forward — per it in [it_begin, it_end):
//   velocity_substep           v: t -> t+1/2           (DD step_phase 1)
//   stress_substep             s: t -> t+1, u_allt[it] (DD step_phase 2 from here)
//   inject_source              per source field
//   <checkpoint save>          shared runtime, not a hook
//   save_boundary_fields       BS strips (when use_boundary_saving)
//   record_field               per receiver field
//   after the loop: save_last_state (final 5-field snapshot for backward_bs)
//
// sg_generic_backward (full storage) — per reverse it:
//   fix_rho_grad_at_sources    body-force rho correction (pre-residual)
//   inject_residuals           signed residuals into the adjoint fields
//   vel_ptrs_from_u_forward    v(it) / v(it+1) pointers from u_forward
//   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends
//   it  > 0: full_mode_step    imaging + receiver-rho + adjoint step, in
//                              the equation's exact fused order
//
// sg_generic_backward_bs — per reverse it, floor max(it_lo, 1):
//   fix_rho_grad_at_sources / inject_residuals / uninject_forward_source  [inject_step]
//   bs_stress_half             stress recon (NOPML) + strip restore +
//                              imaging + receiver-rho + stress-adjoint half
//   bs_velocity_half           velocity-adjoint half + carrier capture +
//                              velocity recon (NOPML) + strip restore + prefetch
//   before the loop (first segment): seed_recon from u_last_two.
//   (DD runs step_phase 3 = injections, then 1, then 2 — same op order.)
//
// sg_generic_backward_ckpt — per chunk (sg_backward_segment):
//   replay:  velocity_substep / stress_substep / save_seg_velocities /
//            inject_forward_sources
//   reverse: fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_seg /
//            image_standalone / fix_rho_grad_at_receivers /
//            (it > 0) plain_adjoint_step
//   after each chunk: export_seg_next_v hands v(start+1) to the older chunk.
//
// sg_generic_backward_recursive_ckpt — per reverse it:
//   fix_rho_grad_at_sources / inject_residuals
//   sg_replay_forward_to_time: velocity/stress substeps + capture_velocities
//                              (IMAGING_USES_NEXT_V eqs also capture v at it+1)
//   vel_ptrs_from_carriers / image_standalone / fix_rho_grad_at_receivers /
//   (it > 0) plain_adjoint_step
// ---------------------------------------------------------------------------
```

## 3. Hook glossary

Mode abbreviations: F = forward, B = backward (full), BS = backward_bs,
CK = backward_ckpt, RC = backward_recursive_ckpt, all = all five entries. The tables
below list only the hooks the skeleton calls; private helpers factored out inside a
traits struct to avoid duplication — the staggered family's
`stress_adjoint_prepare/apply` and `velocity_adjoint_half`, EVR's
`momentum_adjoint_half` — are not hooks and the skeleton does not know them.

### Acoustic family (`eq_driver.cuh`)

Constants: `NDIM`, `NAME`, `CKPT_NVAR`, `BS_NVAR` (how many fields the saver stores),
`BS_LAST_TWO_NVAR`, `TANGENT_PAD` (tangential strip pad = TANGENT_PAD×M; 1 for VRZ),
`CUT_MASK_BITS`/`CUT_MASK_DESC`, `ADJ_WF_COUNT`, `RECON_WF_COUNT`,
`HAS_FUSED_FULL_IMG` (mode B folds the lagged imaging into the adjoint kernel),
`ADCIG_IN_FULL_MODES`, `BS_HAS_IT0_ADJOINT_TAIL` (whether the BS loop is followed by
the four adjoint hooks at `it == 0`).
Types: `Wavefield`, `CPML`, `State`, `BwdWorkspace`, `BsScratch`.

| Hook | Purpose | Modes |
|---|---|---|
| `make_state(p, d, ctx, launch, src_cfg, rec_cfg)` | Model pointers, operator parameter blocks and launch configuration, built once outside the loop | all |
| `make_bwd_workspace(p, state, ctx, adjoint)` | Adjoint scratch (empty for acoustic; VRZ zeroes the adjoint state here and builds the C0/Cx/Cz coefficients) | B, BS, CK, RC |
| `make_bs_scratch(p, vp)` | Per-step BS scratch (the 3-D NOPML output field `f_this`) | BS |
| `validate_forward(p)` / `validate_backward(p, need_recon)` | The equation's own entry validation, hand-written text preserved | F / B, BS |
| `setup_ctx(ctx, p)` | `SolverContext` extras for the family: topography rows, per-edge free-surface faces, APM flags | all |
| `init_aux_slabs(ctx, wf)` | Install the CPML aux strip (slab) geometry | all |
| `alloc_cpml(cpml, p)` | Allocate the CPML profile tensors | all |
| `allt_shape(d, nt)` | Shape of the `u_allt` / ckpt chunk buffer | F, CK |
| `save_width(abcn, M)` | Boundary strip width | F, BS |
| `bind_or_alloc_forward` / `_adjoint` / `_recon` / `_recon_ckpt`, `bind_or_alloc_recursive_scratch` | Bind the Python wavefield lists, allocating internally when empty (ckpt shapes follow the checkpoint slot layout) | F / B,BS,CK,RC / BS / CK,RC / RC |
| `bind_backward_outputs(p, grads, illum, want_adcig)` | Bind or allocate `grads` (slot 0 is `grad_wavelet`) and the illumination outputs; VRZ implements its own | B, BS |
| `alloc_grads(p, grads)` / `pack_outputs(out, grads, illum)` | Internal gradient allocation for ckpt / packing the `BackwardOutput` | CK, RC / B, BS, CK, RC |
| `rtm_out_full` / `rtm_out_bs` | Whether RTM/illumination/ADCIG are on (returns a pointer or nullptr) | B, CK, RC / BS |
| `fused_grad_ptr(grads)` / `u_forward_ptr(p, it)` | Gradient target for fused imaging / pointer to the full-storage forward field at step it | B (fused) / B, CK |
| `launch_step_range(state, ctx, xb, xe, view, save_all, u_thist, cpml)` | The whole stencil step over x ∈ [xb, xe) (air-clear prepass included); the skeleton passes the phase strip range | F |
| `save_boundary_fwd(rt, state, ctx, view, it_shifted, nt_shifted, bs, w)` | Store the boundary strips (in tail-truncation-shifted coordinates) | F |
| `inject_source_fwd(state, ctx, view, p, it, nsrc)` | Source injection; overloaded for `ForwardInput` and `BackwardInput` (the latter for ckpt replay) | F, CK, RC |
| `record(state, ctx, view, record, p, it, nrec)` | Receiver sampling | F |
| `rotate_buffers(wf)` | Forward buffer-role rotation (`swap_pml`: u and psi double buffers) | F |
| `capture_allt(u_allt, wf, it)` | Tensor-copy history capture after the swap (VRZ stores 5 fields; empty for acoustic) | F |
| `save_last_state(saver, wf)` | After the final segment, store `u_prev/u_now` into `last_two` | F |
| `adjoint_step(state, ctx, adj_view, cpml, ws, img_fwd, grad_out)` | Fused adjoint stencil; when `img_fwd/grad_out` are non-null it also does the lagged imaging | B, BS, CK, RC |
| `inject_adjoint_source(state, ctx, adj_view, p, it, nsrc, ws)` | Residual injection (VRZ injects the negated residual) | B, BS, CK, RC |
| `rotate_adjoint_buffers(wf)` | Adjoint buffer rotation (`swap_aux`, or `swap_pml` for VRZ) | B, BS, CK, RC |
| `accumulate_source_grad(state, ctx, adjoint, p, grads, it, nsrc)` | Sample `grad_wavelet` (empty for VRZ) | B, BS, CK, RC |
| `image_step(state, ctx, fwd_ptr, adjoint, grads*, rtm_out, ws)` | Standalone imaging plus RTM/illumination; `grads == nullptr` means the imaging is already fused | B, CK, RC |
| `seed_reconstruction(state, ctx, forward, p)` | First segment: seed the reconstruction fields from `u_last_two` and zero the absorbing rim (except on cut faces) | BS |
| `bs_recon_step(state, ctx, forward, adjoint, rt, bs, w, cpml, p, grads, rtm, ws, scratch, it, bs_it0)` | One reverse reconstruction step (NOPML, restore, imaging, source injection, swap) — the order here is this equation's bit-level order | BS |
| `bs_rtm_tap(state, ctx, forward, adjoint, illum, compute_illum)` | RTM/illumination/ADCIG sampling after the prefetch | BS |
| `replay_step(state, ctx, view, cpml, save_all, u_this)` / `rotate_recon_buffers(wf)` | Whole-domain forward step for ckpt replay / the swap after a replay | CK, RC |

### Staggered family (`sg_driver.cuh`)

Constants: `NDIM`, `NAME`, `CKPT_NVAR`, `CKPT_COUNT_MSG`, `CKPT_RECURSIVE_COUNT_MSG`,
`BS_NVAR`, `CUT_MASK_BITS`/`CUT_MASK_DESC`, `ADJ_WF_COUNT`, `RECON_WF_COUNT`,
`RECON_LIST_DESC`, `N_VEL` (number of velocity components), `IMAGING_USES_NEXT_V`
(whether imaging consumes a v(t+1) carrier; when false, recursive replay breaks
immediately after the target step and allocates no cross-segment carrier).
Types: `Wavefield`, `WfView`, `CPML`, `Models`, `State`, `Workspace`, `VelPtrs`,
`ReconCarriers`.

| Hook | Purpose | Modes |
|---|---|---|
| `parse_models(p)` | Bind the models from `p.models` and fill the derived coefficients (lambda/mu and so on) into the propagator's `derived_models` slots through `derived::lame` / `derived::vti_stiffness` / `derived::reciprocal` (`common/derived_models.h`); the struct holds them alive | all |
| `make_state(p, d, models, launch, src_cfg, rec_cfg)` / `make_workspace(p, vp)` | Parameter pack built outside the loop / adjoint workspace (`init_adjoint_workspace` or internal scratch) | all / B, BS, CK, RC |
| `validate_forward(p)` / `validate_backward(p, "full"\|"bs"\|"ckpt"\|"ckpt_recursive")` | Entry validation, hand-written text preserved per mode, run before the stepped checks | F / B, BS, CK, RC |
| `setup_ctx` / `init_aux_slabs` / `alloc_cpml` / `allt_shape` | As in the acoustic family (sg's `save_width` is fixed at `M + 1`, so there is no hook) | all |
| `field_ptr(wf, idx)` / `view(wf)` | Pointer by field index (for the source/receiver loops) / get a `WfView` | all |
| `bind_or_alloc_forward` / `_adjoint` / `_recon` (returns `ReconCarriers`) / `_recon_ckpt`, `bind_or_alloc_recursive_scratch`, `check_ckpt_aux_layout` | Bind or allocate each wavefield set; recon also carries v(t+1); ckpt validates that the aux layout agrees | F / B,BS,CK,RC / BS / CK,RC / CK,RC / CK |
| `zero_adjoint_if_first_segment(adjoint, first_segment)` / `zero_adjoint_if_first_segment_bs(...)` | Zero the adjoint state on the first segment (needed by the 3-D members; empty in 2-D) | B / BS |
| `bind_grads(p, grads)` / `alloc_grads(vp, grads)` | Bind `grads_out` (stepped) or allocate; the element count is the number of models | B, BS, CK / RC |
| `adjoint_source_signs(p, receiver_fields)` | The sign each receiver field's residual is injected with (stress receivers -1, velocity +1; EVR all +1): `add_source_signed` flips the sample's sign bit, no negated copy of the residual is built | B, BS, CK, RC |
| `velocity_substep(state, wf, cpml, solver)` / `stress_substep(state, wf, cpml, solver, u_this)` | The two half-step kernels (also used by ckpt and recursive replay) | F, CK, RC |
| `inject_source(state, solver, field, source, loc, it, nsrc)` | Single-field source injection (the skeleton loops over `source_field_indices`) | F |
| `save_boundary_fields(rt, state, solver, wf, it, nt, bs, w)` / `record_field(...)` | Store the BS field list / sample one receiver field | F |
| `save_last_state(saver, wf)` | Snapshot every field's final state into `last_two` | F |
| `fix_rho_grad_at_sources(state, solver, adj_view, p, src_fields, it, grads)` | Rho-gradient correction at body-force source cells; must run **before** this step's residual injection | B, BS, CK, RC |
| `inject_residuals(state, solver, adj_view, p, rec_fields, signed, it, nsrc)` | Inject the signed residuals into the adjoint fields (EVR additionally zeroes the adjoint stress surface rows at the end) | B, BS, CK, RC |
| `vel_ptrs_from_u_forward(p, it, zero_v)` / `vel_ptrs_from_seg(seg, now, next, next_seg_v)` / `vel_ptrs_from_carriers(cur_v, next_v)` | v(it)/v(it+1) pointers from three sources: full storage / ckpt segment buffers / recursive carriers | B / CK / RC |
| `image_standalone(state, solver, adj_view, vptrs, grads)` | Standalone gradient kernel (plus EVR's chain-rule kernel) | B (it==0), CK, RC |
| `fix_rho_grad_at_receivers(state, solver, grads, vptrs, p, rec_fields, it, nsrc)` | Undo the contamination the just-injected residual causes in the rho imaging (at velocity receiver cells) | B, BS, CK, RC |
| `full_mode_step(state, solver, adjoint, ws, cpml, vptrs, grads, p, rec_fields, it, nsrc)` | One full-mode step: imaging, receiver-rho and the adjoint step, in the equation's own fused order | B (it>0) |
| `plain_adjoint_step(state, solver, adjoint, ws, cpml)` | The four-kernel adjoint step with no imaging arguments | CK, RC (it>0) |
| `seed_recon(forward, p)` | First segment: seed the reconstruction fields from `u_last_two` | BS |
| `uninject_forward_source(state, solver, for_view, p, src_fields, neg_src, it, nsrc)` | Un-inject the source from the reconstruction fields (-source) | BS |
| `bs_stress_half(...)` / `bs_velocity_half(...)` | See the timing map; DD phases 1 and 2 call one each | BS |
| `seg_buffers(p, vp, max_rows)` / `save_seg_velocities(seg, fwd, slot)` / `export_seg_next_v(prev, seg)` | Per-segment velocity buffers for ckpt: allocate / capture per step / hand v(start+1) to the earlier segment | CK |
| `inject_forward_sources(state, solver, for_view, p, src_fields, it)` | Forward source injection during replay (`BackwardInput` field names) | CK, RC |
| `capture_velocities(v, forward)` | Recursive replay captures v(it) at the target step (and v(it+1) when `IMAGING_USES_NEXT_V`) | RC |

## 4. How each equation differs from the reference

The references: **acoustic2d** is what `eq_driver.cuh` was transcribed from and
**elastic2d** is what `sg_driver.cuh` was transcribed from, so their traits are the
zero-difference baseline for their family. The rest is drawn from the file headers of
each `driver_traits.cuh`.

| Equation | Difference from the reference |
|---|---|
| acoustic2d | The acoustic family reference (the baseline itself); its file header spells out what "same as acoustic2d" means term by term — constants, `State`, the behaviour of each hook, and the bs order NOPML → restore → band imaging → injection → swap. |
| acoustic3d | No `ctx.set_per_edge` (per-edge free surfaces are 2-D only); the fused adjoint carries triple double-buffering of psi and zeta (15 adjoint tensors, `adjoint_extra_nvar=3`); the BS reverse step images **before** the forward source injection (2-D images after injection and the swap), and its NOPML kernel writes a per-step scratch field (`BsScratch.f_this`); ADCIG is offered by `backward_bs` only (the quantity full/ckpt imaging correlates is vp²·Lap(u), not the raw pressure) and there is no seed rim zeroing; the old hand-written `backward_bs` passed nullptr lap/grad coefficient pointers in its `SolverContext` while the skeleton always passes real ones — the bs path never dereferences them, so this is inert to the bit. |
| acoustic_vrz2d | Models are `[vp, z]` with `inv_z` derived; `TANGENT_PAD=1` (strips sit M inside the pad, offset −M); `ADJ_WF_COUNT=9` (the adjoint rotates through `swap_pml`, with no zeta double buffer); no `grad_wavelet` (`accumulate_source_grad` is an empty hook and `grads_out` slot 0 is unused) and no RTM/illumination/ADCIG (both gates return nullptr, `ADCIG_IN_FULL_MODES=false`); `HAS_FUSED_FULL_IMG=false` (a standalone `CALCULATE_GRAD_VRZ2D_AUTO` per step); `BS_HAS_IT0_ADJOINT_TAIL=false` (the bs floor is it==1); `BwdWorkspace` holds the negated residual, the one-shot `BUILD_VRZ_ADJOINT_COEFFS` C0/Cx/Cz and the split-gradient scratch, and `make_bwd_workspace` zeroes the adjoint state on the way through; adjoint injection uses the **negated** residual; `u_allt` stores 5 fields (u, psix, psiz, zetax, zetaz) written by `capture_allt` as a tensor copy, with the in-kernel `u_this` path off; `save_width` is always M+1; no `setup_ctx` and no aux slabs; the BS order is NOPML → source injection → restore → swap → image on the post-swap `u_now`; the seed additionally zeroes `u_next` and its rim zeroing carries no cut mask; `launch_step_range` refuses a sub-range (no phase split); the chunk and recursive ckpt backwards keep their hand-written linear-segment scan (the recursive entry simply forwards to the chunk one). |
| elastic2d | The staggered family reference (the baseline itself); its file header likewise spells out what "same as elastic2d" means. The APM entries stay hand-written. |
| elastic3d | 9 physical fields / 36 wavefield tensors / an 18-tensor adjoint workspace, and three velocity carriers (`N_VEL=3`); DD cuts on x/y only (mask 0x33), which the forward validates; the bound and snapshot lists backfill a missing `m_syzx` memory field (a historical layout quirk); the full backward zeroes the adjoint state on the first segment only (2-D relies on Python-zeroed buffers); reconstruction binding accepts a 12-tensor list (9 fields + 3 carriers) or, leniently, any complete list that carries its own carriers. The APM entries stay hand-written. |
| das_mu2d | The velocity substep is elastic2d's own kernel, reached through the wavefield's `elastic_view()` adapter; the stress substep is a custom stress+strain kernel (the strain integration happens inside it), so each step's view is a **pair** (the das view and the elastic view). The CPML memory variables stay whole-domain: an identity aux slab **must** be installed before any kernel launch (the aux-slab contention incident). An 8-field BS list (5 elastic + 3 strain; only the 5 elastic ones are restored — strain is recorded only), 18 wavefield tensors, an 8-field `last_two` that also reads the old 5-field format leniently. The full backward has **no** gradient fusion: standalone imaging → receiver-rho correction → then the adjoint step. Checkpointing snapshots the whole-domain state (`allocate`, not `allocate_from_snapshots`) with no aux layout check. No DD cut support (`CUT_MASK_BITS=0`: the borrowed kernels are not cut-aware). New from the skeleton (dormant for the old callers): stepped ranges, Python-bound record/wavefield/gradient buffers, the physical phase split, and loud stepped/phase validation; the old `backward_bs`'s dead `f_this` scratch allocation is gone. |
| das_mu3d | Structurally the 3-D das_mu2d (family differences above). Its own 3-D differences: 15 physical fields (9 elastic + 6 strain) / 33 wavefield tensors / an 18-tensor adjoint workspace, three velocity carriers (`N_VEL=3`); BS stores all 15 fields but restores only the 9 elastic ones (strain is recorded only and, unlike 2-D, is never seeded from `last_two` — the hand-written seed copies 9 fields); reconstruction wavefield binding and allocation carry **no** CPML memory tensors (`use_pml=false`; 2-D keeps them); the full backward zeroes the adjoint state after binding (2-D relies on Python zeroing), mapped onto the first-segment `zero_adjoint_if_first_segment`; full/ckpt imaging uses the shared `LAUNCH_CALCULATE_GRAD_3DELASTIC_BS` over a pure velocity view (2-D has a dedicated `_NOBS` kernel). |
| elastic_tti_sg2d | The model set is rho plus 15 stiffness tensors (16 gradients); the kernels take a `StiffnessPointer` rebuilt on demand from `p.models`/grads. Three velocity components on a 2-D grid (TTI couples vy), `N_VEL=3`, and the signed adjoint sources use the 3-D field layout. The adjoint workspace is six plain scratch tensors, taken from `p.adjoint_workspace` when bound (`ElasticTTISG.cuda_layout.backward_workspace_shapes`: 6 per shot in 2-D) and allocated internally otherwise (`eqdrv::pool_or_zeros`). `u_allt` stores all 8 physical fields rather than velocities only. BS reconstruction binds the propagator's 11-grid list (8 physical fields + 3 velocity carriers, `bs_reconstruction_nvar=11`), allocating only when nothing was bound. Per-mode entry validation keeps its hand-written text (`validate_backward`). No recursive checkpointing: the forward refuses it, `backward.cu` does not instantiate the recursive driver, and the recursive-only hooks (`capture_velocities`, `vel_ptrs_from_carriers`, `CKPT_RECURSIVE_COUNT_MSG`) are deliberately absent. No DD cut support and no aux slabs (the CPML memory lives in the equation's own wavefield tensors). |
| elastic_tti_sg3d | Against its 2-D sibling: the model set is rho plus 21 stiffness tensors (22 gradients) and 12 PML profiles; the wavefields and workspace are the **shared** elastic types (`ElasticWavefieldTensor`, 36 tensors; `ElasticAdjointWorkspaceTensor` via `init_adjoint_workspace`, bound from `p.adjoint_workspace` -- 18 per shot, like elastic3d -- when the propagator declares it) — and unlike elastic3d there is no `m_syzx` backfill; `u_allt` stores the three velocities only (2-D stores all 8 fields); **both** the full and the BS backward zero the adjoint state after binding (`zero_adjoint_if_first_segment` and `zero_adjoint_if_first_segment_bs`; 2-D zeroes in neither); the forward refuses `free_surface` outright (an anisotropic medium refuses the image method); the velocity kernels take only `model.rho` while the stress kernels take the full `StiffnessPointer`. Same as 2-D: reconstruction wavefields are always allocated internally, no recursive checkpointing (the recursive-only hooks are deliberately absent), no DD cut support, no aux slabs. |
| elastic_vr2d | Six primary models `{vp, vs, Rp_x, Rp_z, Rs_x, Rs_z}`, six gradients, and no rho — so every rho hook (`fix_rho_grad_at_sources`, `fix_rho_grad_at_receivers`) is empty and the imaging has no v(t+1) term (`IMAGING_USES_NEXT_V=false`: recursive replay breaks at the target step and allocates no cross-segment velocity carrier). The wavefields reuse `ElasticWavefieldTensor` (the vx/vz slots hold the momenta px/pz), and the 15-tensor binding, the checkpoint layout and the 5-field BS list all match elastic2d. Every backward mode zeroes the adjoint stress surface rows immediately after residual injection (the adjoint of the forward free-surface BC); that kernel sits at the end of `inject_residuals`. Every mode follows its gradient kernel with a chain-rule kernel (`LAUNCH_EVR_GRAD_CHAIN_APPLY`) — both live in `image_standalone`. The 14-slot adjoint workspace pool is split between the adjoint-step half (slots 0–9, `Workspace`) and the imaging half (slots 10–13 plus a zero-momentum buffer, hung off `State` for `image_standalone`). `backward_bs` never binds Python reconstruction wavefields (it always allocates, with no carriers), and the ckpt/recursive states use a plain full-shape `allocate` rather than a snapshot-driven aux layout. |

## 5. Worked example: the kernel launch sequence of one `backward_bs` reverse step

### elastic2d (`sg_generic_backward_bs`, monolithic `step_phase = 0`; the DD phase split 3 → 1 → 2 produces exactly the same sequence)

`for it = it_hi-1 … max(it_lo, 1)` (before the loop on the first segment:
`seed_recon` = 5 `copy_` calls, no kernels):

* `inject_step(it)` (DD phase 3)
    1. `add_body_force_rho_grad_correction` — once per source field belonging to vx/vz (stress sources are skipped) [`fix_rho_grad_at_sources`]
    2. `add_source` (signed residual → adjoint field) — once per receiver field [`inject_residuals`]
    3. `add_source` (`-forward_source` → reconstruction field) — once per source field [`uninject_forward_source`]
* `bs_stress_half` (DD phase 1)
    1. `elastic_stress_kernel_nopml<order>` — reverse stress reconstruction (NOPML)
    2. `boundary_kernel2d` (or `_compact` / `_bf16` / the dequantised int8 variant, depending on the storage dtype) × 3 — restore sxx, szz, sxz (field 2 waits for the chunk first) [`restore_backward_2d_field`]
    3. `elastic_stress_adjoint_prepare<order>` — with the imaging pointers: the vp/vs/rho gradients fuse in here (reading `for_view.v* = v(it)` and the carriers `fv*_prev = v(it+1)`) [helper `stress_adjoint_prepare`]
    4. `sub_receiver_rho_grad_correction` — once per velocity receiver field (stress receivers have no rho term) [`fix_rho_grad_at_receivers`]
    5. `elastic_stress_adjoint_apply<order>` [helper `stress_adjoint_apply`]
* `bs_velocity_half` (DD phase 2)
    1. `elastic_velocity_adjoint_prepare<order>` [helper `velocity_adjoint_half`]
    2. `elastic_velocity_adjoint_apply<order>` [same]
    3. `elastic_capture_strips_2d` — copy v(it) on the restore strips into the carriers first (skipped when `n_strip == 0`)
    4. `elastic_velocity_kernel_nopml<order>` — reverse velocity reconstruction; the kernel writes the loaded value into `fvx_prev/fvz_prev` before its read-modify-write
    5. `boundary_kernel2d` (same variants) × 2 — restore vx, vz (vz marks the chunk done) [`restore_backward_2d_field`]
    6. `prefetch_next_backward_chunk_if_needed` — host side; no kernel under gpu-direct

### acoustic2d (`generic_backward_bs`)

`for it = it_hi-1 … max(max(it_lo, 1), bs_stop)` (before the loop on the first
segment: `seed_reconstruction` = 2 `copy_` calls + `set_boundary_zeros` × 2):

1. `acoustic2nd_adjoint_fused<order>` — the fused adjoint (no imaging pointers in bs mode) [`adjoint_step`]
2. `add_source` (residual → `adj.u_next`) [`inject_adjoint_source`]
3. host: `adjoint.swap_aux()` — u + psi + zeta double-buffer rotation [`rotate_adjoint_buffers`]
4. `accumulate_source_grad_2d` [`accumulate_source_grad`]
5. `acoustic2nd_nopml<order>` — reverse reconstruction, with the vp gradient imaging fused in (for every cell the restore does not overwrite) [`bs_recon_step`]
6. `boundary_kernel2d` (or `_compact` / `_bf16` / a dequantising variant) — restore `u_next` [`restore_backward_2d`]
7. `calculate_grad_utt_band` — imaging for the restore strips only (skipped when `n_strip == 0`)
8. `add_source` (`forward_source` → `recon.u_next`)
9. host: `forward.swap()`
10. host: `prefetch_next_backward_chunk_if_needed`
11. `accumulate_rtm_image_2d` — only under `compute_illumination` [`bs_rtm_tap`]
12. `accumulate_adcig_2d` — only when ADCIG was requested

After the loop (`BS_HAS_IT0_ADJOINT_TAIL`, `it_lo == 0` and no tail truncation):
steps 1–4 run once more at `it = 0`.

## 6. Checklist for adding an equation

1. **Pick a family**: second-order displacement form with buffer rotation →
   `eq_driver.cuh`; first-order velocity–stress with in-place updates →
   `sg_driver.cuh`; neither (das2d/3d's derivative-buffer form, for instance) → a
   hand-written driver.
2. **kernels.cuh / kernels.cu**: the whole-step stencil, the NOPML reverse step, the
   adjoint step, the imaging kernels. To use the acoustic phase split the kernels
   must honour ranged launches via `ctx.x_base/x_limit`; to support DD they must use
   the cut-aware `in_pml` / `phys_*()` predicates (the shared P2 helpers — see
   `gate/in_pml_equiv.cpp`).
3. **driver_traits.cuh**: copy the reference (acoustic2d or elastic2d), keep the five
   sections [1]–[5] and the call order inside each; fill in the constants; put only
   launches in the hooks, remembering that the order inside a composite hook is
   load-bearing to the bit; switch off capabilities you do not need with empty hooks
   or constants (`HAS_*`, `IMAGING_USES_NEXT_V`, `CUT_MASK_BITS = 0`); and record in
   the file header how this equation differs from the reference (that header is the
   source for section 4).
4. **forward.cu / backward.cu / `<eq>.h`**: five one-line entries plus the
   `forward_runner` / `backward_bs_runner` factories; do not instantiate the
   recursive driver if you do not offer that mode (templates instantiate lazily, so
   the hooks may be absent).
5. **C API table**: add the five `{C_NAME}_*` ids to `SweepEntry` in `core/capi.h`
   (bump `SWEEP_ENTRY_COUNT`), the matching rows to `ENTRY_NAMES` / `ENTRY_KINDS`
   and the `dispatch()` cases in `cuda/common/capi.cu`; for a stepped equation also
   the two runner-factory `case`s there and its name in
   `sweep.backend.c.RUNNER_EQUATIONS`. `bindings/module.cpp` mirrors the table for
   the `SWEEP_JIT_FULL=1` shim only.
6. **Python**: `C_NAME` and `cuda_layout` (`base_nvar`, `pml_nvar`, `last_two_nvar`,
   `checkpoint_nvar`, `adjoint_extra_nvar`, `boundary_tangent_pad`, `slots`,
   `grads_out_has_wavelet`, …). Set `stepped=True` once migrated, and
   `dd_backward_phases=True` once the backward implements numbered phases. Without
   recursive checkpointing set `C_HAS_RECURSIVE_CKPT = False`. For DD, also pick or
   declare a schedule in `parallel/dd_spec.py`.
7. **Rebuild the core**: `python -m sweep.build` (or the next `impl='c'` use)
   re-stages only the files whose SHA-256 changed and runs the core's ninja graph
   incrementally; no `rm -rf` is needed (under `SWEEP_JIT_FULL=1` the pybind shim is
   rebuilt the same way). Make sure no wheel-style core sits under
   `src/sweep/lib/<cuN>/` and `SWEEP_CORE` is unset: a fitting shipped core is loaded
   as it is and `csrc/` edits are silently ignored
   (`sweep.backend.torch.binding.diagnostics()['shipped_core']` tells).
8. **Put it under the gate**: add it to `test/solver_gradient_mode_suite.py::SOLVERS`
   and `gate/bitgate.py::ALL_SOLVERS`, and record a baseline before migrating.

## 7. Verification

**Principle**: the bit-exactness criterion is `torch.equal`, and
`--verify-reproducible` must show it is attainable before it is used; every
configuration runs in its own subprocess; `gate/run_gate.sh` is the only approved way
to run (never through a pipe, a missing verdict line is a failure, and
`ran == PASS+FAIL+MISSING+NEW` is checked against truncation). Environment:
`. gate/env.sh` pins `PY`, `PYTHONPATH=worktree/src` and a dedicated
`TORCH_EXTENSIONS_DIR` (the local core's cache); the gate runs on the default ctypes
path, and a leaked `SWEEP_JIT_FULL` switches it to the pybind shim and produces
screens of false red. Acceptance for each migration step:
tiers A/C/T/dd1 bit-exact green plus no new pytest failures, with tier B added when a
family is finished. After touching `csrc/`, rebuild the core (`python -m sweep.build`) and check
that `libsweep_core.so`'s mtime is later than the edit and earlier than the gate
log.

| Tool | Coverage | Usage |
|---|---|---|
| `gate/bitgate.py` tier **A** (27 configurations, ~2 min) | acoustic2d/3d, elastic2d/3d, vrz2d, lsrtm2d: eager+c × full/bs_gpu/bs_cpu/bs_gpu_int8/ckpt_chunk/ckpt_recursive × interior/free_surface/free_surface_all4 × canonical/physical grids. Run after every commit. | `gate/run_gate.sh A base_A.pt` |
| tier **C** (30, ~2 min) | One eager-full and one c-bs_gpu run for every equation in `ALL_SOLVERS` (shallow but complete) | `gate/run_gate.sh C base_C.pt` |
| tier **B** (~165, ~12 min) | Every equation × 7 c-side memory modes + physical grid + free surface; run when a stage is finished. `gate/noise_floors.json` records the gradient tolerance for the configurations already measured as non-deterministic (DAS, 3-D ckpt, int8); the record and the loss stay strict always. The baseline holds one expected error-text entry (`elastic_tti_sg2d\|c\|ckpt_recursive`). | `gate/run_gate.sh B base_B.pt`; for a subset, `$PY gate/bitgate.py --tier B --only elastic --compare gate/base_B.pt` |
| tier **T** (10) | Topography (hill/stairs, image method and APM); every other tier is flat ground | `gate/run_gate.sh T base_T.pt` |
| `--verify-reproducible` / `--self-test` / `--measure-noise` | Same code twice, bit for bit; a 1-ULP perturbation must go red; repeated runs measure the floor | `$PY gate/bitgate.py --tier A --verify-reproducible` |
| `gate/ddgate.py` world=1 | Single-tile `ModelParallel`: capture, lazy adjoint promotion, per-shot geometry rebinding, both families' step loops, buffer-role rotation, the persistent runner path; 12 configurations (Acoustic/3D, AcousticVRZ3D, Elastic/3D × fs, plus two bodyforce cases) | `gate/run_gate.sh dd1 base_dd1.pt` |
| `gate/ddgate.py` world≥2 | Real tiles, NCCL halo exchange, `cut_face_mask` — the only rung that can catch a wrong send-field list. On ibex: `torchrun --nproc-per-node=2 gate/ddgate.py --ranks 2 …`, with the baseline re-recorded from dev inside the same job | `dd_reverify.sbatch` |
| `gate/evr_ab.py` | `ElasticVRR` (elastic_vr2d) is not in the suite, so A/B/C/T can all be green without testing it: 4 backward modes × free surface, comparing the record plus 6 gradients — 56 tensors — bit for bit | `$PY gate/evr_ab.py --out new.pt --compare gate/evr_base.pt` |
| `gate/check_equations_api.py` | Freezes the public surface of `sweep.equations` (name count, registered equation count, alias identity) | Run when touching `equations/` |
| pytest | the suite exits 0 (`test_import_does_not_pull_optional_deps` runs in a fresh interpreter). Driver-related: `test_stepped_forward{,_elastic}.py`, `test_stepped_backward{,_elastic}.py`, `test_dd_*two_tile*.py`, `test_dd_tiles_3d.py`, `test_cut_face_mask.py`, `test_slot_table_consistency.py`, `test_dd_supported_equations.py`, `test_boundary_tail_truncation.py`; C-vs-eager gradient consistency lives in `test/solver_gradient_mode_suite.py` | `$PY -m pytest test/` (the default ctypes path, what the wheel ships); a second run under `SWEEP_JIT_FULL=1` covers the pybind shim and the CPU engine |

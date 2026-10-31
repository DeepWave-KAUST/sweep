# gate/

The bit-exactness gate: what has to stay identical while the solver is
refactored, and the tooling that checks it.

Run it through `run_gate.sh`, never by calling `bitgate.py` through a pipe --
a gate piped into `head` is SIGPIPEd part-way through and the pipeline's exit
status comes from `head`, so a truncated run looks successful. That happened
once here (tier A ran 20 of 27 configs and the wrapper still printed DONE),
which is why "produced no verdict line" is treated as a failure.

```bash
. gate/env.sh                                   # machine-specific paths
gate/run_gate.sh A base_A.pt C base_C.pt        # tier/baseline pairs
gate/run_gate.sh dd1 base_dd1.pt                # dd1/dd2 go through ddgate
```

## The gate is not the whole story

`gate/verify_all.sh` runs everything that protects the branch, because the gate
has two **structural** blind spots that other instruments exist to cover:

| blind spot | why it exists | what covers it |
|---|---|---|
| compares C against C | it cannot see a defect the full and boundary-saving paths **share** — the shape of the `acoustic_vti_1st` operator-adjoint bug | the eager leg of `test/backend_gradient_matrix.py` |
| `ALL_SOLVERS` omits `elastic_vr2d` | the solver suite has no ElasticVRR entry | `gate/evr_ab.py`, 56 tensors |
| `visco2d` joined `ALL_SOLVERS` only on 2026-09-17 | its CUDA backend had pytest coverage only: tier B stayed 187/187 green while every visco backward raised | `test/test_visco_acoustic_cuda.py`, and now tiers B/C |
| `born2d/3d` started from a zero reflectivity until 2026-09-17 | the Born record was identically 0, so no record comparison could see a forward change (the gradient, the adjoint of the noise data, was non-zero) | `make_models` now starts from a 0.04 box; those baselines were re-recorded from `6b0d8306` |
| no tier sets `compute_illumination` | the illumination path was never gated at all | `test/test_illumination_pin.py` |
| tier T's receivers sat in the air until 2026-10-05 | the suite puts receivers at a fixed depth (`spatial_order // 2`), above the hill and the stairs for every 2-D receiver and 40 of 49 3-D ones: six of the ten records were identically 0 (and the eager gradients with them) -- two zero records compare equal | `run_one` now moves a topography config's sources and receivers `_SURFACE_CLEARANCE` rows below their column's surface; a successful entry whose record, or every gradient, is identically 0 now FAILS the comparison and is refused by `--save`; tier T re-recorded |
| noise floors can be a race | until 2026-10-06 the 3-D `bs_gpu_int8` keys reproduced themselves only to 1e-5..3e-3 (the compact restore wrote an edge/corner cell from whichever band's thread ran last; int8 quantizes each face with its own scales), and the 3-D / visco2d `ckpt_*` keys to 1e-9..2e-6 (the checkpoint replay updated psi in place); their floors (x `NOISE_HEADROOM` 5, up to 1.7e-2 on lsrtm3d) could have hidden a real regression | `boundary_kernel3d_compact_owned` and the replay psi double-buffer; the 15 keys re-measured (4 passes: bitwise) and re-recorded in `base_B.pt` (old copy: `base_B.pre-owned.pt`), their floors null; fp16 storage only via `test/test_host_core.py::TestLossyBoundary3DRestore` and `test_compact_matches_scan_acoustic3d_lossy`.  Before accepting a floor, ask whether it is a race |
| compares each impl against itself | a c-vs-eager disagreement that predates the baseline is invisible -- e.g. elastic2d image topography, c and eager 26% apart in the record two rows below the surface | the eager legs of the pytest suites; nothing yet for near-surface elastic topography |

```bash
. gate/env.sh && gate/verify_all.sh            # everything (~30 min)
. gate/env.sh && gate/verify_all.sh --quick    # skips tier B and the full matrix
```

It prints, at the end, what **nothing** covers. Keep that list honest: an
uncovered path that nobody has written down reads as a covered one.

## What is tracked, and what is not

Tracked: the tooling (`bitgate.py`, `ddgate.py`, `check_equations_api.py`,
`evr_ab.py`, `srcrec_*.py`, the runners, the design notes) plus the two inputs
it reads, `golden_equations_api.json` and `noise_floors.json`.

Not tracked (see `.gitignore`): the `.pt` baselines and every run transcript.
A baseline is large, is only meaningful for the GPU model that produced it, and
is rewritten by `--save`.

**Because the baselines are not in version control, copy one before you edit
it.** They are the only record of what the tree produced before a change, and
`git checkout` cannot bring one back.

## Changing what the gate covers

Configurations an equation refuses are covered on purpose: the baseline stores
the refusal's error text and compares that (tier B has 9 such entries). Do NOT
filter refused configurations out of the tier generators -- that removes the
coverage that says the refusal has not silently changed. When a configuration's
expected behaviour changes by design, re-baseline THAT KEY, print its before
and after, and leave the rest of the baseline alone.

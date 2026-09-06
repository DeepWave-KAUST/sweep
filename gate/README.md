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
| also omits `elastic_tti_sg3d`, `elastic_tti_2nd2d`, `visco_acoustic2d` | never added | the matrix for the first two; `test/test_visco_acoustic_cuda.py` for visco |
| no tier sets `compute_illumination` | the illumination path was never gated at all | `test/test_illumination_pin.py` |

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

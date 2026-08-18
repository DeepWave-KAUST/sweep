# Torch Multi-GPU Examples

This directory contains Torch distributed examples launched with `torchrun`.
They cover the two independent ways of using more than one GPU, which solve
different problems and compose with each other:

| | what is split | each rank holds | use it when |
|---|---|---|---|
| **shot parallel** (`fwi_marmousi_dist.py`) | the shot list | the whole model | the model fits on one GPU and you have many shots |
| **model parallel / DD** (`dd_*.py`) | the model, into tiles | one tile, all shots | the model does **not** fit on one GPU |

`MeshTopology(py, px, shot_groups=...)` does both at once.

## Shot parallel: 2D acoustic FWI on Marmousi

Script:

- `fwi_marmousi_dist.py`

Prepare the Marmousi `.npy` files first:

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
python3 examples/models/marmousi/extract_model_segy.py
python3 examples/models/marmousi/convert_segy_to_npy.py
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 \
  --target-dh 12.5 \
  --radii 16,16 \
  --passes 3
```

Run on two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/multi-gpu/torch/fwi_marmousi_dist.py --backend torch --impl c --device cuda
```

The script splits each global shot batch across ranks, sums model gradients
with `torch.distributed.all_reduce`, and applies the same optimizer step on
every rank. Rank 0 writes figures to `multi_gpu_acoustic_fwi_cuda/`.

## Model parallel: domain decomposition

`ModelParallel(prop, mesh)` wraps an ordinary propagator and runs it with the
model cut into tiles. The call signature does not change and the forward stays
autograd-transparent, so the only DD-specific lines in an FWI loop are two
`all_reduce` calls — one for the global misfit, one to assemble the global
gradient.

Two rules keep the split invisible to everything else:

- **Keep the optimisation variable on the physical grid** and apply
  `sweep.parallel.pad_to_mesh` inside the loss closure. DD needs uniform tiles
  (`Nx % px == 0`) and never pads implicitly; padding the *stored* model
  instead would move the pad into your optimiser state and, with a
  reparameterised model, silently rewrite the whole velocity field.
- **Reduce a detached misfit.** `all_reduce` is invisible to autograd, so
  reducing a tensor that is still on the graph and then calling `.backward()`
  on it gives the local gradient under a global-looking number. That happens
  to be right for a plain sum, and is wrong the moment anything nonlinear
  follows the reduction.

### 2D — Marmousi, with a single-GPU cross-check

- `dd_fwi_marmousi_2d.py`

A full inversion (Adam, 60 iterations, 20 shots), written so the same script
runs decomposed or not. Marmousi at downsample 8 is 1701 cells wide, which is
not divisible by 4, so `--pad-px` is decoupled from `--px`: both sides can be
put on the same padded grid, which is what makes the comparison meaningful.

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/multi-gpu/torch/dd_fwi_marmousi_2d.py --px 4 --tag dd4 --outdir out
python3 examples/multi-gpu/torch/dd_fwi_marmousi_2d.py \
  --px 1 --pad-px 4 --tag single --outdir out --check dd4
```

The first command writes this — Marmousi at 10 m, 20 shots, 5 Hz, 60 Adam
iterations, the model split four ways (dashed lines are the cuts):

![Marmousi 2-D DD FWI](../../../docs/figures/examples/dd_fwi_marmousi_2d_dd4.png)

The layering, the faulted zone near 10 km and the fast wedge at 2.5–3 km all
come back; below ~3 km the shots do not illuminate and the starting model
survives untouched. Nothing marks the tile boundaries — a halo bug shows up
there first, as a step or a ringing streak across the cut.

The second command adds the comparison:

![single vs dd4](../../../docs/figures/examples/dd_fwi_marmousi_2d_single_vs_dd4.png)

The middle panel is `single - dd4` on a fixed ±0.001 m/s scale, so an exactly
zero difference stays blank instead of being stretched into noise; after 60
iterations `max |dvp|` is 0. The right panel is the per-iteration misfit gap:
half the iterations agree to the last bit and the rest sit within 1–2 fp64
ulp. That residue is the reduction order, not the physics — under DD each rank
sums only its own receivers before the `all_reduce`. The models stay
bit-identical because the *gradient* drives Adam, and the gradient is exact.

`--check` also prints those numbers, so you do not need the figure to see
them. A third run with `--pad-px 1 --obs-pad-px 4` inverts the same observed
data on the unpadded grid, which separates what the pad itself costs from
anything DD does: three extra columns out of 1701 leave the inversion quality
alone (RMSE 309.92 vs 309.89 m/s) but move the final model by up to 185 m/s
locally. Change the tile count and you change the pad, so pin `--pad-px` if
you need runs to be comparable.

Each run writes `dd_fwi_marmousi_2d_<tag>.png` plus `hist_<tag>.npz` and
`vp_final_<tag>.npy`.

#### How big does it have to be before DD is also faster?

The run above is deliberately small so it finishes in minutes, and at that
size DD *loses*: 48.6 s per iteration on four V100s against 11.5 s on one.
Nothing is wrong — each tile is 351×426, its kernels take tens of
microseconds, and one halo exchange per time step costs more than that. The
only lever is the tile size. One forward+backward on V100s, steady state:

| `--downsample` | grid | 1 GPU | 4 GPUs | speedup |
|---|---|---|---|---|
| 8 | 0.60 M | 145 µs/step | 502 | **0.29×** |
| 4 | 2.38 M | 450 µs/step | 527 | **0.85×** |
| 2 | 9.53 M | 1604 µs/step | 620 | **2.59×** |
| 1 | 38.10 M | 6175 µs/step | 1859 | **3.32×** |

The break-even is between 2.4 M and 9.5 M cells. The per-step cost does not
depend on the record length (measured from 600 to 64000 steps, unchanged), so
these extrapolate to any `--seconds`. The single-GPU column is not linear
either: at 0.60 M one card reaches 4.1 Gcell/s against 6.1 Gcell/s at 38 M, so
a small model wastes one GPU as well as four.

Memory *is* linear in the record length, because the boundary ring is. At
8.0 s (32000 steps at `--downsample 2`, 64000 at `--downsample 1`):

| `--downsample` | ring | 1 GPU | 4 GPUs |
|---|---|---|---|
| 2 | fp32 | 8.35 GiB | 2.65 GiB |
| 1 | fp32 | **out of memory** | 10.57 GiB |
| 1 | bf16 | 21.62 GiB | 7.14 GiB |
| 1 | int8 | 15.85 GiB | 5.45 GiB |

So full resolution at 8 s needs the ring compressed to fit one card at all —
with fp32 it wants 1.6 GiB more than a 32 GB V100 has:

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/multi-gpu/torch/dd_fwi_marmousi_2d.py --px 4 --downsample 1 \
  --seconds 8 --nshot 1 --iters 2 --boundary-dtype bf16 --tag big4 --outdir out
python3 examples/multi-gpu/torch/dd_fwi_marmousi_2d.py --px 1 --pad-px 4 \
  --downsample 1 --seconds 8 --nshot 1 --iters 2 --boundary-dtype bf16 \
  --tag big1 --outdir out
```

`--seconds` rather than `--nt`: dt is derived from the CFL number, so halving
`--downsample` halves both dh and dt for you and the record stays 8 s.

Two things to know before you time this yourself:

- **Never time DD's first call.** It pays a one-time setup — the capture probe
  forward, the ring allocation, the NCCL bring-up. At full resolution that is
  about 98 s, which charged to a 64000-step run looks like a 1.8× slowdown and
  is easy to misread as a bad interconnect or an nt effect. Single-GPU runs
  have no such cost, so only the DD numbers move. Time the *second* call.
- **A lossy ring is not bit-exact across tile counts.** bf16 and int8 quantise
  per block, and a tile's blocks are not the whole domain's, so `--check` will
  report a small difference. Keep `--boundary-dtype fp32` for the parity
  recipe above and use the lossy modes only when memory forces it.

### 3D — Overthrust, one model update

- `dd_fwi_overthrust_update.py`

One complete FWI iteration on a 3-D model split 2×2: observed data, multi-shot
gradient, preconditioning, a backtracking line search, and a re-evaluated
misfit that says whether the update actually helped.

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/multi-gpu/torch/dd_fwi_overthrust_update.py --py 2 --px 2
```

It also shows the forward-only path: generating observed data under
`torch.no_grad()` keeps the DD capture free of adjoint wavefields and of the
boundary ring, and the capture is promoted automatically at the first
`.backward()`.

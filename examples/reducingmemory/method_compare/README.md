# Memory Method Compare

This directory focuses on memory-saving methods that preserve the standard
shot-based objective while changing how wavefields are stored or recomputed.

Main methods:

- `PyTorch eager + compile`
  - Uses `torch.compile` on the eager propagator.
  - Mainly targets execution speed, but can also shift memory behavior depending on the graph.
  - Most useful as a baseline against plain eager and eager+ckpt.

- `PyTorch eager + ckpt`
  - Uses `torch.utils.checkpoint` inside the eager propagator.
  - Reduces activation memory by recomputing chunks during backward.
  - Main tuning parameter: `ckpt_chunks`.

- `PyTorch eager + compile + ckpt`
  - Combines graph compilation with eager-side checkpoint replay.
  - Useful when you want to see whether compile recovers some of the checkpoint overhead.

- `CUDA boundary saving`
  - Stores only boundary wavefields plus the last states, then reconstructs the interior during backward.
  - Main tuning parameters:
    - `BoundaryOptions.storage`: `gpu` or `cpu`
    - `BoundaryOptions.transfer_interval`
    - `BoundaryOptions.pinned_memory`
  - `storage="gpu"` keeps replay data on device and is usually faster.
  - `storage="cpu"` saves more device memory, but transfer cost becomes important.

- `CUDA ckpt`
  - Stores selected full states and replays between checkpoints.
  - Two modes:
    - `CkptOptions(mode="chunk", chunks=...)`
    - `CkptOptions(mode="recursive", count=...)`
  - `chunk` is easier to reason about.
  - `recursive` is useful when you want to cap the checkpoint budget directly.

General guidance:

- If you need the simplest eager-side memory reduction, start with PyTorch checkpointing.
- If you are already on the CUDA backend and want the best memory/runtime tradeoff, test CUDA boundary saving first.
- If boundary saving is still too memory hungry, try CUDA checkpointing.
- CPU boundary saving is the most aggressive for GPU memory reduction, but often costs the most wall time.

Benchmark script:

- [common_benchmark.py](./common_benchmark.py)
  - Shared benchmark loop, method table, summary plotting, and gradient plotting for both 2D and 3D tests.
- [acoustic2d_memory_benchmark.py](./acoustic2d_memory_benchmark.py)
  - 2D acoustic benchmark setup: geometry, model, and 2D gradient panel extraction.
- [acoustic3d_memory_benchmark.py](./acoustic3d_memory_benchmark.py)
  - 3D acoustic benchmark setup: geometry, model, and mid-`z` gradient slice extraction.

It compares:

- eager full
- eager compile full
- eager ckpt with different chunk sizes
- eager compile + ckpt
- cuda full
- cuda boundary saving on GPU
- cuda boundary saving on CPU with different transfer settings
- cuda ckpt chunk
- cuda ckpt recursive

The 3D benchmark uses the same family of methods, but on a smaller 3D acoustic setup and
plots an energetic `y`-slice gradient cross-section for each method.

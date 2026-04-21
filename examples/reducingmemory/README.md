# Reducing Memory

Memory-reduction examples live here. The main methods in this repo fall into two groups:

- `source encoding`
  - Reduces the number of effective shots per iteration by mixing multiple shots into one encoded batch.
  - Changes the optimization objective into a stochastic one.
  - Best viewed as an algorithmic· approximation.

- `wavefield-storage / recomputation methods`
  - Keep the standard shot-based objective, but reduce memory by saving less state or replaying more work.
  - This is where PyTorch checkpointing and CUDA memory-saving methods belong.

Directory guide:

- [source_encoding/README.md](./source_encoding/README.md)
  - Source encoding examples and tradeoffs.
- [method_compare/README.md](./method_compare/README.md)
  - PyTorch checkpointing vs CUDA boundary saving / CUDA checkpointing.
- [method_compare/common_benchmark.py](./method_compare/common_benchmark.py)
  - Shared benchmark utilities used by the 2D and 3D comparison scripts.
- [source_encoding/torch/source_encoding_fwi.py](./source_encoding/torch/source_encoding_fwi.py)
  - Torch/CUDA source-encoding FWI example.
- [source_encoding/jax/source_encoding_fwi.py](./source_encoding/jax/source_encoding_fwi.py)
  - JAX source-encoding FWI example.
- [method_compare/acoustic2d_memory_benchmark.py](./method_compare/acoustic2d_memory_benchmark.py)
  - Benchmark `eager`, `eager+ckpt`, `cuda+boundary`, and `cuda+ckpt`.
- [method_compare/acoustic3d_memory_benchmark.py](./method_compare/acoustic3d_memory_benchmark.py)
  - 3D acoustic benchmark for the same memory-saving methods.

## Methods

### 1. Source Encoding

Use when:

- you have many shots,
- exact per-shot gradients are too expensive,
- some stochasticity is acceptable.

Main tuning knobs:

- encoded batch size,
- polarity design,
- time-shift range.

Typical tradeoff:

- much lower per-iteration memory and cost,
- noisier gradients,
- often more iterations to converge.

### 2. PyTorch Checkpointing

Implemented in the eager propagator through chunked replay during backward.

Use when:

- you want to stay on the eager PyTorch backend,
- you need lower memory without changing the optimization objective,
- you accept extra recomputation.

Main tuning knob:

- `ckpt_chunks`
  - smaller chunks: lower peak memory, more recomputation overhead
  - larger chunks: higher peak memory, less recomputation overhead

Typical tradeoff:

- moderate-to-strong memory reduction,
- noticeable runtime increase,
- simple and robust.

### 3. PyTorch Compile

Implemented through `EagerOptions(use_compile=True, ...)` on the eager backend.

Use when:

- you want to compare plain eager against compiled eager,
- you care about execution speed as much as memory,
- your workload shape is stable enough for compile warmup to pay off.

Typical tradeoff:

- potentially lower steady-state runtime,
- extra warmup/compile cost up front,
- memory behavior may improve or worsen depending on graph fusion and captured buffers.

### 4. CUDA Boundary Saving

Stores boundary wavefields and reconstructs the interior during backward.

Use when:

- you are already using the CUDA backend,
- you want a better memory/runtime tradeoff than full checkpoint replay.

Main tuning knobs:

- `BoundaryOptions.storage`
  - `gpu`: faster, uses more GPU memory
  - `cpu`: lowest GPU memory, adds transfer overhead
- `BoundaryOptions.transfer_interval`
  - relevant only for `storage="cpu"`
  - larger interval reduces transfer frequency but increases staging usage
- `BoundaryOptions.pinned_memory`
  - relevant only for `storage="cpu"`
  - can improve transfer throughput

Typical tradeoff:

- often better runtime than checkpointing at similar memory levels,
- more backend-specific tuning,
- CPU staging mode can become PCIe-bandwidth bound.

### 5. CUDA Checkpointing

Stores selected states and replays segments during backward.

Two modes:

- `CkptOptions(mode="chunk", chunks=...)`
- `CkptOptions(mode="recursive", count=...)`

Chunk mode:

- easier to reason about,
- chunk size directly controls replay granularity.

Recursive mode:

- better when you want to cap checkpoint count directly,
- more indirect tuning because replay schedule depends on `count`.

Typical tradeoff:

- strong memory reduction,
- generally slower than CUDA boundary saving,
- still useful when boundary saving is not enough or not ideal for the workload.

## What To Compare

When comparing `PyTorch ckpt`, `PyTorch compile`, and CUDA methods, focus on:

- peak GPU memory,
- end-to-end forward+backward time,
- parameter sensitivity:
  - `ckpt_chunks`
  - `BoundaryOptions.storage`
  - `BoundaryOptions.transfer_interval`
  - `BoundaryOptions.pinned_memory`
  - `CkptOptions.mode`
  - `CkptOptions.chunks`
  - `CkptOptions.count`

Practical expectations:

- `eager full` is usually the fastest baseline, but memory hungry.
- `eager compile` may reduce steady-state runtime once warmup is amortized.
- `eager ckpt` reduces memory, but runtime rises due to replay.
- `eager compile + ckpt` is worth testing when you want to see whether compile offsets some checkpoint overhead.
- `cuda boundary (gpu)` is often the best balance on CUDA.
- `cuda boundary (cpu)` can save the most device memory, but may pay heavily in transfer time.
- `cuda ckpt chunk` is easier to tune than recursive mode.
- `cuda ckpt recursive` is useful when you want to bound checkpoint budget directly.

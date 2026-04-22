# Reducing Memory

Source directory:

- `examples/reducingmemory/`

This example family focuses on memory-reduction techniques for inversion and
wave propagation workloads.

## Main Method Groups

- `source encoding`: mixes multiple shots into one encoded batch and changes
  the optimization objective into a stochastic one
- `wavefield storage / recomputation`: keeps the standard shot-based objective
  but reduces memory by saving less state or replaying more work

## Example Layout

- `source_encoding/`: source-encoding examples and tradeoffs
- `method_compare/`: PyTorch checkpointing, Torch compile, CUDA boundary
  saving, and CUDA checkpointing benchmarks
- `acoustic/torch/vrz_forward_compare.py`: acoustic VRZ forward comparison

## Main Methods

### Source Encoding

Use when:

- the number of shots is large
- exact per-shot gradients are too expensive
- some stochasticity is acceptable

Typical tradeoff:

- much lower per-iteration memory and cost
- noisier gradients
- often more iterations to converge

### PyTorch Checkpointing

Implemented in the eager propagator through chunked replay during backward.

Use when:

- you want to stay on the eager PyTorch backend
- you need lower memory without changing the optimization objective
- you accept extra recomputation

Main tuning knob:

- `ckpt_chunks`

### PyTorch Compile

Implemented through `EagerOptions(use_compile=True, ...)` on the eager backend.

Use when:

- you want to compare plain eager against compiled eager
- you care about execution speed as much as memory
- your workload shape is stable enough for compile warmup to pay off

### CUDA Boundary Saving

Stores boundary wavefields and reconstructs the interior during backward.

Main tuning knobs:

- `BoundaryOptions.storage`
- `BoundaryOptions.transfer_interval`
- `BoundaryOptions.pinned_memory`

### CUDA Checkpointing

Stores selected states and replays segments during backward.

Supported modes:

- `CkptOptions(mode="chunk", chunks=...)`
- `CkptOptions(mode="recursive", count=...)`

## What to Compare

When comparing these methods, focus on:

- peak GPU memory
- end-to-end forward and backward time
- sensitivity to `ckpt_chunks`
- sensitivity to `BoundaryOptions.storage`
- sensitivity to `BoundaryOptions.transfer_interval`
- sensitivity to `BoundaryOptions.pinned_memory`
- sensitivity to `CkptOptions.mode`, `chunks`, and `count`

## Related Pages

- [Source Encoding Overview](source_encoding_overview.md)
- [Memory Method Compare](memory_method_compare.md)

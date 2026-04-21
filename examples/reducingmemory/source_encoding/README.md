# Source Encoding

Source encoding reduces memory and runtime by mixing multiple shots into a
single encoded super-shot during each optimization step.

This changes the optimization objective, so it is not a drop-in replacement
for exact shot-by-shot gradients. In practice it is most useful when:

- the number of shots is large,
- GPU memory limits the encoded batch size,
- some gradient noise is acceptable in exchange for lower cost per iteration.

Examples:

- [./torch/source_encoding_fwi.py](./torch/source_encoding_fwi.py)
- [./jax/source_encoding_fwi.py](./jax/source_encoding_fwi.py)

What it saves:

- fewer simultaneous shots in memory,
- fewer forward/adjoint solves per iteration,
- smaller effective observed/synthetic shot gathers per update.

Tradeoffs:

- the gradient is stochastic,
- convergence may need more iterations,
- encoding design matters: polarity, time shift, and batch size all affect variance.

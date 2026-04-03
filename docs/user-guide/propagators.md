# Propagators

This page should explain how propagators connect equations, geometry, and model parameters.

## Main Propagator APIs

- `sweep.propagator.torch.PropTorch`
- `sweep.propagator.jax.PropJax`
- `sweep.propagator.cuda.PropCUDA`

## Suggested Topics

- Constructor arguments
- Required geometry formats
- `source_type` and `receiver_type`
- PML configuration
- Forward and backward workflows

## TODO

- Add a minimal example for each propagator
- Document expected tensor and array shapes
- Explain differences between backend implementations

# SWEEP

Seismic Wave Equation Exploration Platform (SWEEP) is a Python package for seismic wave-equation modeling, propagation, and inversion.

## What SWEEP Provides

- Various wave equations (acoustic, elastic, visco-acoustic, etc.) in a unified framework
- PyTorch and JAX backends
- Optional PyTorch C++/CUDA extension bindings for faster propagation and memory-saving modes
- Tools for forward modeling, FWI, and research prototyping
- A modular structure for extending equations and operators

## Documentation Map

- Start with [Installation](getting-started/installation.md) to set up the package
- Continue with [Quick Start](getting-started/quickstart.md) for a minimal example
- See [Backends](user-guide/backends.md) for JAX, PyTorch, and extension binding notes
- Use [CLI](user-guide/cli.md) to inspect supported equations from the terminal
- Browse [Examples](examples/index.md) for runnable scripts in the repository

## Suggested Writing Plan

- Fill in installation and environment requirements first
- Add one minimal PyTorch example and one JAX example
- Document equation classes and required model parameters
- Add backend capability notes and performance guidance
- Link each example page to the corresponding file in `examples/`

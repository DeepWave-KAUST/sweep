# Contributing to SWEEP

Thanks for your interest in improving SWEEP. This guide covers the minimum
you need to know to set up a development environment and submit a change.

## Development install

Clone the repository and install in editable mode. The compiled CUDA
extension is optional for most documentation and Python-level changes:

```bash
git clone https://github.com/DeepWave-KAUST/sweep.git
# or, if you have an SSH key registered with GitHub:
# git clone git@github.com:DeepWave-KAUST/sweep.git
cd sweep
pip install -e .
```

For the compiled C++ / CUDA extension (needed when working on the
`impl="c"` path):

```bash
SWEEP_BUILD_CUDA=1 pip install -v -e ".[cuda]" --no-build-isolation
```

See [Installation](docs/getting-started/installation.md) for
CUDA-architecture configuration if the build cannot detect your GPU
automatically.

## Branch convention

- Active development happens on `dev`. Do not commit directly to `dev`.
- Feature work and refactors live on `feat/<slug>` branches cut from `dev`.
- Bug fixes live on `fix/<slug>` branches cut from `dev`.
- Pull requests target `dev`.

## Pull-request checklist

Before opening a PR, please confirm:

1. The change builds and the relevant tests pass locally.
2. Documentation under `docs/` is updated if user-facing behavior changed.
3. New public Python API has NumPy-style docstrings consistent with the
   existing modules under `src/sweep/`.
4. The PR description explains *why* the change is needed; reviewers should
   not have to reconstruct intent from the diff.

## Adding a new equation

SWEEP discovers equation classes by reflecting over `sweep.equations`'s
namespace — there is no manual registry to update. A new equation is one
Python file under `src/sweep/equations/` (eager) plus, optionally, a CUDA
equation directory under `src/sweep/csrc/cuda/equations/` and a five-line
entry in `src/sweep/csrc/bindings/module.cpp` (`impl="c"`).

Two entry points:

- The [Extending guide](docs/user-guide/extending.md) — reference: interface
  contract, `CUDALayoutSpec` field table, C++ registration pattern,
  out-of-scope boundary.
- The [Add a new equation notebook](examples/notebooks/18_extending_add_new_equation.ipynb)
  — runnable walkthrough: builds a toy `MyScalar` from scratch, runs it
  end-to-end against `PropTorch`, plots the result.

## Reporting issues

Use the GitHub issue tracker at
<https://github.com/DeepWave-KAUST/sweep/issues>. A good bug report includes:

- Your platform (OS, Python version, PyTorch / JAX version, CUDA version)
- The smallest snippet that reproduces the problem
- The full traceback or unexpected output
- What you expected to happen

For usage questions rather than bugs, please open a GitHub Discussion.

## Code style

- Python: PEP 8, NumPy-style docstrings, type hints where they aid clarity.
- C++ / CUDA: follow the conventions already in `src/sweep/csrc/`.

## Documentation changes

The documentation lives under `docs/` and is built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Preview
locally before submitting documentation PRs:

```bash
pip install mkdocs-material
mkdocs serve  # http://127.0.0.1:8000
```

Run a strict build to catch broken links and warnings:

```bash
mkdocs build --strict
```

## License

By contributing you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).

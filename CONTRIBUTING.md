# Contributing to SWEEP

Thanks for your interest in improving SWEEP. This guide covers the minimum
you need to know to set up a development environment and submit a change.

## Development install

Clone the repository and install in editable mode. A clone carries no prebuilt
CUDA core (`src/sweep/lib/` is git-ignored), so `impl='c'` needs a one-time local
core build (below); the eager / JAX backends and documentation work need nothing
compiled:

```bash
git clone https://github.com/DeepWave-KAUST/sweep.git
# or, if you have an SSH key registered with GitHub:
# git clone git@github.com:DeepWave-KAUST/sweep.git
cd sweep
pip install -e .
```

For `impl='c'` from a clone, build the CUDA core once. It needs an nvcc >= 12.4 on
`PATH` or `CUDA_HOME` (of your torch's CUDA major; >= 12.8 for Blackwell targets,
or a CUDA 13 nvcc; `ninja` comes with the package) and is cached under
`TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions/...`) with a source
stamp, so an edit under `src/sweep/csrc/` triggers an incremental rebuild on the
next run:

```bash
python -m sweep.build                    # or just use impl='c': the core builds itself on first use
SWEEP_JIT_FULL=1 python -m sweep.build   # developer path: also the pybind shim (torch headers, C++ compiler)
```

Do not keep a wheel-style core under `src/sweep/lib/<cuN>/` (what
`utils/build_cores_manylinux.sh` writes) while editing kernels: a fitting core
there is loaded as it is and `csrc/` edits are ignored until it is rebuilt.
`python -c "from sweep.backend.torch import binding; print(binding.diagnostics())"`
shows which core and which shim (`ctypes` / `pybind`) a process uses.

See [Building the CUDA core](docs/dev/building.md) for building the core
on a machine with no visible GPU (`TORCH_CUDA_ARCH_LIST` names the target).

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

An equation registers itself: decorate the class with `@register_equation()`
(`src/sweep/equations/_registry.py`) and import its module from
`src/sweep/equations/__init__.py`; nothing reflects over the namespace. A new
equation is one Python file under `src/sweep/equations/` (eager) plus,
optionally for `impl="c"`, a CUDA equation directory under
`src/sweep/csrc/cuda/equations/`, `C_NAME = "<prefix>"` on the class (the base
derives `_C()` from it) with a `cuda_layout`, and five `<prefix>_*` entries in
the core's C API table (`SweepEntry` in `src/sweep/csrc/core/capi.h`;
`ENTRY_NAMES` / `ENTRY_KINDS` / `dispatch()` in
`src/sweep/csrc/cuda/common/capi.cu`). `src/sweep/csrc/bindings/module.cpp`
mirrors that table and is compiled only on the `SWEEP_JIT_FULL=1` developer path.

Two entry points:

- The [Extending guide](docs/user-guide/extending.md) — reference: interface
  contract, `CUDALayoutSpec` field table, C++ registration pattern,
  out-of-scope boundary.
- The [Add a new equation notebook](docs/notebooks/18_extending_add_new_equation.ipynb)
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

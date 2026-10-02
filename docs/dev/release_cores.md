# Release cores

How the CUDA cores that ship in the PyPI wheel are built. Users never need this:
it is for whoever cuts a release (see [Installation](../getting-started/installation.md)
for installing).

The wheel carries one core per CUDA major, each a fat binary built once, with no
GPU in sight — one command per nvcc:

```bash
python -m sweep.build --core --cuda-home /usr/local/cuda-12.9   # -> src/sweep/lib/cu12, archs 7.0;7.5;8.0;8.6;8.9;9.0+PTX
python -m sweep.build --core --cuda-home /usr/local/cuda-13.0   # -> src/sweep/lib/cu13, archs 7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX
python -m sweep.build --core --archs "8.0;9.0+PTX" --out /path/to/lib/cu12 --cuda-home /usr/local/cuda-12.9   # a custom list
```

Each run stages the sources, runs the core's own ninja graph with that `nvcc`, and
writes `libsweep_core.so` plus a `core.json` sidecar (ABI version, CUDA release,
archs, PTX, sha256, flags). `--out` defaults to inside the installed `sweep` package,
`sweep/lib/<tag>` — `src/sweep/lib/<tag>` in a clone — where `<tag>` is the nvcc's
CUDA major: `cu12` for a CUDA 12 toolkit, `cu13` for CUDA 13. The loader picks the
tag torch's CUDA major names, so both cores ride in the same wheel. Without
`--archs` each toolkit gets its recommended list above; they differ because nvcc 13
dropped offline compilation for compute capability < 7.5 (an `sm_70` entry under
nvcc 13 is refused up front), so the cu13 core has no V100 and adds Blackwell
(sm_100, sm_120) natively, which the cu12 core reaches only through its sm_90 PTX.
A wheel built from a tree that contains such a core becomes a `manylinux` platform
wheel that ships it; without one the wheel stays `py3-none-any` and users build the
core locally (nvcc) as before.

Three rules for a core that is going to PyPI:

- **Build both cores, then `python -m build --wheel` from the same tree.**
  `src/sweep/lib/` is git-ignored and pruned from the sdist (`MANIFEST.in`), so a
  bare `python -m build` packs the wheel from the pruned sdist and ships no core.
  The release script sets `SWEEP_REQUIRE_CORE=cu12,cu13`, which makes `setup.py`
  refuse a tree missing either `src/sweep/lib/<tag>/libsweep_core.so`, naming the
  missing tag, instead of quietly producing a wheel with one core or none
  (`SWEEP_REQUIRE_CORE=1` asks only for at least one).
- **Build inside the manylinux container, not on a developer box.**
  `utils/build_cores_manylinux.sh` runs both `--core` builds and the wheel build in
  PyTorch's `manylinux2_28-builder` images (glibc 2.28, a 2018-era `libstdc++`), so
  the shipped `.so` needs only `GLIBC_2.17` / `GLIBCXX_3.4.22` and the wheel is
  tagged `manylinux_2_28`, exactly like torch's own wheels. A core linked on a
  developer box binds to that box's `libstdc++`: one built on an updated Ubuntu
  20.04 required `GLIBCXX_3.4.30` (GCC 12) and failed to load on a stock Debian 11
  or RHEL 9 with a `GLIBCXX_...` version error -- conda environments hide this,
  because they carry their own `libstdc++`.
- **Mind the size.** The cu12 list (six SASS targets + `sm_90` PTX) gives a 71 MB
  `.so`, the cu13 list (seven SASS targets + `sm_120` PTX) an 86 MB one; compressed
  together they make a ~42 MB wheel. PyPI's per-file limit is 100 MB. Every kernel is
  compiled once (`kernels.cu` and its launch tables, see `cuda_drivers.md`); a kernel
  named in a header would be compiled again into every includer. Add SASS entries sparingly and keep
  exactly one `+PTX` entry per core, the newest arch: each embedded PTX is another
  copy of every kernel, and a card newer than every SASS entry only ever needs the
  newest one.

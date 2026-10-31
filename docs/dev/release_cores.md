# Release cores

How the cores that ship in the PyPI wheel are built -- one CUDA core per CUDA
major and, from 0.3.7, the host (CPU) core. Users never need this: it is for
whoever cuts a release (see [Installation](../getting-started/installation.md)
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

- **Build all three cores, then `python -m build --wheel` from the same tree.**
  `src/sweep/lib/` is git-ignored and pruned from the sdist (`MANIFEST.in`), so a
  bare `python -m build` packs the wheel from the pruned sdist and ships no core.
  The release script sets `SWEEP_REQUIRE_CORE=cu12,cu13,cpu`, which makes
  `setup.py` refuse a tree missing any `src/sweep/lib/<tag>/libsweep_core.so`,
  naming the missing tag, instead of quietly producing a wheel with fewer cores
  (`SWEEP_REQUIRE_CORE=1` asks only for at least one).
- **Build inside the manylinux container, not on a developer box.**
  `utils/build_cores_manylinux.sh` runs the three core builds and the wheel build in
  PyTorch's `manylinux2_28-builder` images (glibc 2.28, a 2018-era `libstdc++`), so
  the shipped `.so` needs only `GLIBC_2.17` / `GLIBCXX_3.4.22` and the wheel is
  tagged `manylinux_2_28`, exactly like torch's own wheels. A core linked on a
  developer box binds to that box's `libstdc++`: one built on an updated Ubuntu
  20.04 required `GLIBCXX_3.4.30` (GCC 12) and failed to load on a stock Debian 11
  or RHEL 9 with a `GLIBCXX_...` version error -- conda environments hide this,
  because they carry their own `libstdc++`.
- **Mind the size.** The cu12 list (six SASS targets + `sm_90` PTX) gives a 102 MB
  `.so`, the cu13 list (seven SASS targets + `sm_120` PTX) a 123 MB one; compressed
  together they make a ~62 MB wheel, and the host core adds 23 MB (7 MB compressed),
  ~69 MB. PyPI's per-file limit is 100 MB. Every kernel is compiled once
  (`kernels.cu` and its launch tables, see `cuda_drivers.md`); a kernel named in a
  header would be compiled again into every includer, and `test/test_core_capi.py`
  fails when an arch of the core at hand lists a kernel twice. Add SASS entries
  sparingly and keep exactly one `+PTX` entry per core, the newest arch: each
  embedded PTX is another copy of every kernel, and a card newer than every SASS
  entry only ever needs the newest one.

## Host (CPU) core

The third core, `src/sweep/lib/cpu/`, is the same sources built by g++ instead
of nvcc ([The host (CPU) core](host_core.md)), shipped from 0.3.7:

```bash
python -m sweep.build --cpu --core     # -> src/sweep/lib/cpu/libsweep_core.so + core.json
```

It needs `g++ >= 9` (or `clang++ >= 10`) with OpenMP and `ninja`, no toolkit,
and the same manylinux rule as the CUDA cores: the release script builds it in
the cu12 builder image, so the libstdc++ floor holds. Baseline x86-64 ISA, no
`-march`: the shipped core must run on any x86-64.

It is what `impl='auto'` runs on a CPU device without building: a wheel without
`lib/cpu` sends each CPU user with g++ through a one-time 1-3 min build on
the first CPU construction, and keeps one without a compiler on eager, 1.2-18x slower at
8-28 threads ([host core](host_core.md), section 4).

The core reaches OpenMP through `libgomp.so.1`, and inside a torch process
that resolves to the copy torch already loaded (`torch/lib/libgomp.so.1` in
the pip wheels) -- one runtime, which is why `torch.set_num_threads` sets the
core's thread count too.  So the core must not need a `GOMP_*` / `OMP_*`
symbol version newer than torch's copy defines (`GOMP_5.0` / `OMP_5.0` in
torch 2.9's); the 0.3.7 core needs `GOMP_4.0` / `OMP_1.0` (`#pragma omp
parallel`, the `omp_get_*` queries), and the release script refuses one that
needs more than 5.0.

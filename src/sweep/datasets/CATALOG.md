# `sweep.datasets` — benchmark model catalog

Unified registry of benchmark velocity models: tiny **embedded** demo blobs
(ship in the wheel, no network) plus **downloadable** full-size public
benchmarks (fetched on demand, cached, license printed on first load).

```python
from sweep import datasets
datasets.catalog()                         # print the whole table (kind/license/verified)
datasets.available()                       # [(name, variant), ...] machine-readable
prob = datasets.load("overthrust", "3d-acoustic")
vp = prob["vp"]
```

**`variant` is optional.** With a sole variant it's inferred
(`load("bp-2004")`); a multi-variant name resolves to its `default=True`
entry (`load("marmousi")` → the embedded `2d-demo`, `load("overthrust")` →
`2d-demo`). A name with several variants and no default would raise and list
the choices rather than guess.

Optional deps: `pip install sweep-solver[datasets]` adds the HTTP client
(requests/tqdm) for downloads. Parsing (SEG-Y / raw grids) is numpy-only.

Cache dir: `$SWEEP_DATASETS_CACHE` → `$XDG_CACHE_HOME/sweep-datasets` →
`~/.cache/sweep-datasets`.

## CLI

```bash
sweep datasets list                     # table of all datasets (or: sweep-datasets list)
sweep datasets list marmousi            # one family
sweep datasets info bp-2004             # one entry's metadata
sweep datasets download overthrust 3d-acoustic          # fetch to cache, report shape
sweep datasets download bp-2004 --downsample 2 4        # per-axis downsample
sweep datasets where                    # print the cache directory
```

## Catalog

| name | variant | kind | grid (loaded) | vp range (m/s) | license | re-host? | verified |
|---|---|---|---|---|---|---|---|
| `marmousi` | `2d-demo` | embedded | 281×1361 @ 12.5 m | 1028–4700 | SEG open (demo) | ✅ | ✅ |
| `overthrust` | `2d-demo` | embedded | 187×801 @ 25 m | — | CC-BY src (demo) | ✅ | ✅ |
| `marmousi` | `2d-acoustic` | download | 2801×13601 @ 1.25 m (native) | 1028–4700 | SEG open, cite | ➖ | ✅ |
| `overthrust` | `3d-acoustic` | download | 187×801×801 @ 25 m | 2179–6000 | **CC-BY-4.0** | ✅ | ✅ |
| `seg-eage-salt` | `3d-acoustic` | download | 210×676×676 (nz,ny,nx) @ 20 m | 1500–4482 | **CC-BY-4.0** | ✅ | ✅ |
| `marmousi2` | `2d-elastic` | download | 2801×13601 @ 1.25 m (vp/vs/rho) | 1028–4700 | open academic, cite | ➖ | ✅ |
| `bp-2004` | `2d-acoustic` | download | 1911×5395, dz6.25/dx12.5 | 1429–4790 | free+attr, AS-IS | ➖ | ✅ |
| `bp-2007-tti` | `2d-tti` | download | 1801×12596 @ 6.25 m (vp/ε/δ/θ) | 1492–4554 | free+attr, courtesy BP | ➖ | ✅ |
| `hess-vti` | `2d-vti` | download | 1500×3617 @ ~7.62 m (vp/ε/δ) | 1524–4511¹ | free+attr, Hess AS-IS | ➖ | ✅¹ |

**re-host?** ✅ = license permits re-hosting the bytes; ➖ = fetch from official
source only (do not vendor/re-host).

**verified** ✅ = byte-verified against a real download (KW60443, 2026-07):
shape, physical value range, and slice image confirmed for every entry.

¹ **hess-vti**: the SEG-Y ships vp in **ft/s** (5000–14800) and the grid in
feet; the loader converts vp→m/s (×0.3048) and uses dh≈7.62 m (25 ft, the
widely-used value — confirm against your reference if dh matters).

Verification notes (resolved 2026-07-06):
- **overthrust 3d**: official SEG/EAGE 3-D Modeling Series CD (CD1 →
  `3D-Velocity-Grid/overthrust.vites`), raw **big-endian IEEE float32**; the
  `.vites.h` sidecar gives n1=801 n2=801 n3=187, d=25 m, m/s (n1 fastest → read
  as C-order `(187,801,801)`). File size 479 917 548 = 801·801·187·4 confirms
  the grid; values 2179–6000 m/s. ✅
- **seg-eage-salt**: `tar.gz → SALTF.ZIP → Saltf@@` raw big-endian `>f4`,
  95 964 960 = 676·676·210, **x-fastest** (like `.vites`) → read as C-order
  `(nz,ny,nx) = (210,676,676)`; range 1500–4482. (Axis order confirmed by
  slicing — a depth section shows the salt dome; the earlier `(676,676,210)`
  scrambled it.) ✅
- **marmousi2**: each field is a nested `.segy.tar.gz` inside the outer tarball
  (`MODEL_{P-WAVE,S-WAVE,DENSITY}_VELOCITY_1.25m.segy.tar.gz`); IBM-float SEG-Y,
  read as `(nx,nz)` traces then **transposed** to `(nz,nx)`. ✅
- **bp-2004 / bp-2007 / hess**: IBM-float SEG-Y; the reader returns
  `(n_traces=nx, n_samples=nz)` and `read_model` **transposes** to `(nz,nx)`
  (a flat reshape would scramble the grid — that bug was caught and fixed here).
  bp-2007 members are `.sgy` (not `.segy`). ✅

## Deliberately excluded

| model | why excluded |
|---|---|
| **Sigsbee2A / 2B** | Gated behind a signed **Data Release Agreement**; original host (`delphi.tudelft.nl`) is dead (NXDOMAIN). Not freely redistributable. Get it from the SEG software repo under their terms. |
| **SEAM Phase I** | License genuinely ambiguous across sources (**CC-BY-4.0 vs CC-BY-NC-SA-4.0**); official pages Cloudflare-gated so no verifiable direct URL. Do not bundle until the license inside the data package is confirmed. |
| **OpenFWI** | Datasets are **CC BY-NC-SA 4.0** (non-commercial), incompatible with a permissive package; Google-Drive hosted (per-subset ids), so the download path can't be auto-verified. Use the upstream `openfwi-lanl` tooling directly. |

## Adding a model

Register an `Entry` (see `_benchmarks.py`):

```python
from sweep.datasets import Entry, register
register(Entry(
    name="foo", variant="2d-acoustic", loader=_load_foo,
    description="...", citation="...", license="...",
    kind="download", redistributable=False, tags=["2d", "acoustic"],
))
```

Loaders return a dict with at least `vp` and `dh`; most add `dt`, `nt`,
`geometry`, and (for multi-parameter models) `vs`/`rho`/`epsilon`/`delta`/etc.

## Downsampling

Every download loader takes `downsample`, either a single int (applied to all
axes) or a per-axis list/tuple; `dh` is scaled per axis to match. Multi-field
models (marmousi2, bp-2007-tti, hess-vti) decimate every field by the same
factor. Factors must be `>= 1` and the sequence length must equal the model
dimensionality (else `ValueError`).

```python
datasets.load("bp-2004", downsample=2)          # (956, 2698), dh (12.5, 25.0)
datasets.load("bp-2004", downsample=(2, 4))     # (956, 1349), per-axis, dh (12.5, 50.0)
datasets.load("seg-eage-salt", downsample=[1, 1, 3])  # decimate depth only -> (676, 676, 70)
```

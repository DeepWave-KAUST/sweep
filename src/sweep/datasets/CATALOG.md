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
(requests/tqdm) for downloads, and h5py for the HDF5 copy that Overthrust 3-D
falls back to. Parsing everything else (SEG-Y / raw grids) is numpy-only.

Cache dir: `$SWEEP_DATASETS_CACHE` → `$XDG_CACHE_HOME/sweep-datasets` →
`~/.cache/sweep-datasets`.

**Sources.** Each downloaded file is pinned by the sha256 of the original
bytes (and the md5 its host publishes, where there is one); its URLs are tried
in order and a download that does not match is discarded; a cached file gets
the same check on every load and is fetched again if it fails. If no source
answers, the error names the cache path to put the original file at. The SEG
open-data S3 bucket (`open.source.geoscience`) that served every file stopped serving
anonymous downloads in 2026. The originals of five models are re-hosted on
Zenodo, one record each, listed first, and in the Hugging Face dataset
`shaowinw/seismic-benchmark-models` (one folder per record), listed second; the
bucket stays last:

| model | record |
|---|---|
| Overthrust 3-D (`Overthrust_3D_CD1.tar.gz`) | doi:10.5281/zenodo.23469292 |
| SEG/EAGE Salt 3-D (`Salt_Model_3D.tar.gz`) | doi:10.5281/zenodo.23469308 |
| Marmousi2 (`elastic-marmousi-model.tar.gz`, used by `marmousi` and `marmousi2`) | doi:10.5281/zenodo.23469300 |
| BP 2004 (`vel_z6.25m_x12.5m_exact.segy.gz` with its README files) | doi:10.5281/zenodo.23469290 |
| BP 2007 TTI (`ModelParams.tar.gz` with its terms of use) | doi:10.5281/zenodo.23475636 |

`overthrust:3d-acoustic` falls back to the HDF5 copy of the model
(doi:10.5281/zenodo.4252588) if no source serves the CD. `hess-vti` lists only
the bucket: it loads from an existing cache, or from the original files placed
at the path the error message names.

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
| `marmousi` | `2d-demo` | embedded | 281×1361 @ 12.5 m | 1028–4700 | CC-BY src (demo) | ✅ | ✅ |
| `overthrust` | `2d-demo` | embedded | 187×801 @ 25 m | — | CC-BY src (demo) | ✅ | ✅ |
| `marmousi` | `2d-acoustic` | download | 2801×13601 @ 1.25 m (native) | 1028–4700 | **CC-BY-4.0** | ✅ | ✅ |
| `overthrust` | `3d-acoustic` | download | 187×801×801 @ 25 m | 2179–6000 | **CC-BY-4.0** | ✅ | ✅ |
| `seg-eage-salt` | `3d-acoustic` | download | 210×676×676 (nz,ny,nx) @ 20 m | 1500–4482 | **CC-BY-4.0** | ✅ | ✅ |
| `marmousi2` | `2d-elastic` | download | 2801×13601 @ 1.25 m (vp/vs/rho) | 1028–4700 | **CC-BY-4.0** | ✅ | ✅ |
| `bp-2004` | `2d-acoustic` | download | 1911×5395, dz6.25/dx12.5 | 1429–4790 | free+attr, AS-IS | ➖ | ✅ |
| `bp-2007-tti` | `2d-tti` | download | 1801×12596 @ 6.25 m (vp/ε/δ/θ) | 1492–4554 | free+attr, courtesy BP | ➖ | ✅ |
| `hess-vti` | `2d-vti` | download | 1500×3617 @ ~7.62 m (vp/ε/δ) | 1524–4511¹ | free+attr, Hess AS-IS | ➖ | ✅¹ |

**re-host?** ✅ = CC-BY-4.0: anyone may re-host the files, with attribution;
➖ = the files may be passed on only with their original documentation and terms
(the project's BP records carry them; `hess-vti` is not re-hosted).

**verified** ✅ = byte-verified against a real download (2026-07):
shape, physical value range, and slice image confirmed for every entry.

¹ **hess-vti**: the SEG-Y ships vp in **ft/s** (5000–14800) and the grid in
feet; the loader converts vp→m/s (×0.3048) and uses dh≈7.62 m (25 ft, the
widely-used value — confirm against your reference if dh matters).

Verification notes (resolved 2026-07-06):
- **overthrust 3d**: official SEG/EAGE 3-D Modeling Series CD (CD1 →
  `3D-Velocity-Grid/overthrust.vites`), raw **big-endian IEEE float32**; the
  `.vites.h` sidecar gives n1=801 n2=801 n3=187, d=25 m, m/s (n1 fastest → read
  as C-order `(187,801,801)`). File size 479 917 548 = 801·801·187·4 confirms
  the grid; values 2179–6000 m/s. ✅ The fallback when no source serves the CD
  is the Zenodo HDF5 copy: dataset `m` is squared slowness (s²/km²), `(207,801,801)` on
  axes (z, x, y) with 20 rows of 1500 m/s water on top; `1000/sqrt(m[20:])`
  in float64, transposed to (z, y, x) and rounded to float32, equals the CD1
  grid in every cell. ✅
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

# Datasets

`sweep.datasets` gives one-line access to benchmark velocity models for FWI /
LSRTM / RTM examples and smoke tests. Two flavours share a single registry:

- **Embedded demos** — tiny base85-encoded blobs (Marmousi, Overthrust-2D) that
  ship *inside the wheel*, so notebooks and quick-starts run with no downloads
  and no extra dependencies.
- **Downloadable benchmarks** — full-size public models fetched on first use
  and cached locally, each carrying its own license and citation. Every file is
  checked against the sha256 of the original before it is used.

Implementation: `src/sweep/datasets/`.

!!! warning "The SEG open-data bucket is closed"
    All download entries used to come from the SEG open-data S3 bucket
    (`open.source.geoscience`), which stopped serving anonymous downloads in
    2026. The original files of five models are re-hosted on Zenodo and are
    downloaded from there:
    Overthrust ([10.5281/zenodo.23469292](https://doi.org/10.5281/zenodo.23469292)),
    SEG/EAGE Salt ([10.5281/zenodo.23469308](https://doi.org/10.5281/zenodo.23469308)),
    Marmousi2 ([10.5281/zenodo.23469300](https://doi.org/10.5281/zenodo.23469300), used by
    `marmousi` and `marmousi2`), BP 2004
    ([10.5281/zenodo.23469290](https://doi.org/10.5281/zenodo.23469290)) and BP 2007 TTI
    ([10.5281/zenodo.23475636](https://doi.org/10.5281/zenodo.23475636)). The same files are also in
    the Hugging Face dataset
    [shaowinw/seismic-benchmark-models](https://huggingface.co/datasets/shaowinw/seismic-benchmark-models),
    which is tried when Zenodo does not answer. Each file can also be fetched by hand, e.g.
    `wget https://zenodo.org/records/23469292/files/Overthrust_3D_CD1.tar.gz`.
    `hess-vti` has no public copy: it loads only from an existing cache, or
    from the original files placed at the cache path that the error message
    names. `chevron-fwi-2014` is on Hugging Face only (folder
    `chevron_fwi_2014/`); without it, its files are extracted from the bucket's
    original archive as the Internet Archive captured it (3.8 GB).

## Quick start

```python
from sweep import datasets

datasets.catalog()                       # print the table below
prob = datasets.load("marmousi")         # embedded demo (offline)
vp = prob["vp"]                          # numpy float32 array
dh = prob["dh"]                          # grid spacing per axis

full = datasets.load("overthrust", "3d-acoustic")   # downloads on first call
```

`load(name, variant=None, **kwargs)` returns a `dict`. Its keys:

- **Always** — `vp` (ndarray), `dh` (per-axis spacing tuple), and the metadata
  `name`, `variant`, `citation`, `license`.
- **Download models** — also `dt`, `nt` (suggested time step / step count).
- **Multi-parameter models** — extra field arrays: `marmousi2` adds `vs`, `rho`;
  `bp-2007-tti` adds `epsilon`, `delta`, `theta`; `hess-vti` adds `epsilon`, `delta`.
- **Model-specific** — `units` (`hess-vti`, demos), `presets` (embedded demos),
  `geometry` (a sources/receivers dict on `marmousi:2d-acoustic`).

The minimum you can rely on for any model is `vp` and `dh`. The one data
set, `chevron-fwi-2014`, has no `vp` (see below).

!!! note "`variant` is optional"
    With a single variant it is inferred (`load("bp-2004")`). A multi-variant
    name resolves to its canonical default — `load("marmousi")` and
    `load("overthrust")` return the embedded `2d-demo`. If a name has several
    variants and no default, omitting `variant` raises and lists the choices
    rather than guessing.

## Catalog

| name | kind | grid | license |
|---|---|---|---|
| `marmousi` | embedded | 281×1361 @ 12.5 m | CC-BY src (demo) |
| `overthrust` | embedded | 187×801 @ 25 m | CC-BY src (demo) |
| `marmousi` | download | 2801×13601 @ 1.25 m | **CC-BY-4.0** |
| `overthrust` | download | 187×801×801 @ 25 m | **CC-BY-4.0** |
| `seg-eage-salt` | download | 210×676×676 @ 20 m | **CC-BY-4.0** |
| `marmousi2` | download | 2801×13601 @ 1.25 m | **CC-BY-4.0** |
| `bp-2004` | download | 1911×5395 | free + attribution |
| `bp-2007-tti` | download | 1801×12596 @ 6.25 m | courtesy of BP |
| `hess-vti` | download | 1500×3617 @ ~7.62 m | Hess "AS IS" |
| `chevron-fwi-2014` | download (data) | 1600 shots; `vp_init` 1200×3820, dz 5 / dx 12.5 m | Chevron DLA |

Multi-parameter models return every field: `marmousi2` → `vp`/`vs`/`rho`,
`bp-2007-tti` → `vp`/`epsilon`/`delta`/`theta`, `hess-vti` → `vp`/`epsilon`/`delta`.
Every download entry was byte-verified against a real fetch (grid shape, value
range, and slice image confirmed).

## The Chevron 2014 FWI data set

`chevron-fwi-2014` is the synthetic Chevron distributed for the SEG 2014 FWI
workshop: pressure shot gathers computed in a 2-D isotropic elastic marine
model, with free-surface multiples. Chevron never released that model, so the
entry returns what was distributed instead of `vp`:

| key | content |
|---|---|
| `data` | `(nshots, 2001, 321)` float32 gathers, 4 ms, 8 s |
| `vp_init` | smooth starting vp, `(1200, 3820)`, `dh = (5.0, 12.5)`, x from 0 m |
| `wavelet`, `wavelet_dt` | far-field source wavelet without ghost, sampled at 1/1500 s |
| `well_log` | `{"x": 39375.0, "depth": ..., "vp": ...}`, 1000–2450 m every 5 m |
| `geometry` | `source_x` `(nshots,)`, `receiver_x` `(nshots, 321)` in metres, `source_depth` = `receiver_depth` = 15 m, `dt`, `nt` |

Shot `i` is at x = 1000 + 25 i m with its receivers at offsets 0–8000 m on the
+x side, so the receivers of the last shots reach x = 48975 m, beyond the
starting model (47737.5 m): extend the model before modelling those shots.
The water velocity is 1510 m/s.

```python
d = datasets.load("chevron-fwi-2014")                       # 3.8 GB download, 4.1 GB of gathers
d = datasets.load("chevron-fwi-2014", shots=slice(0, 1600, 100))   # 16 shots
d = datasets.load("chevron-fwi-2014", fields=("vp_init", "wavelet"))  # 19 MB, no gathers
```

`fields` picks any of `vp_init`, `data`, `wavelet`, `well_log` (default all);
the 3.8 GB file is fetched only when `data` is asked for. `shots` (0-based
indices: int, sequence or slice) selects the gathers and the geometry. Each
gather's trace headers are checked against this geometry when it is read.
The entry takes no `downsample`.

## Downsampling

Every model download loader takes `downsample` — a single int (all axes) or a
per-axis list/tuple. Multi-field models decimate all fields by the same factor.

`load(...)` returns a `dict`; read the array from `["vp"]` (and `["vs"]` /
`["rho"]` / ... for multi-parameter models) and the grid spacing from `["dh"]`:

```python
prob = datasets.load("bp-2004")     # native
vp = prob["vp"]                     # ndarray, shape (1911, 5395)
dh = prob["dh"]                     # (6.25, 12.5) = (dz, dx) in metres
```

!!! note "`dh` is scaled with `downsample`"
    Decimating by a factor makes each remaining cell span that many original
    cells, so `dh` is **multiplied by the same per-axis factor**. The physical
    extent (`dh × n_cells`) is preserved — the model just gets coarser, not
    smaller.

```python
datasets.load("bp-2004")["dh"]                       # (6.25, 12.5)
datasets.load("bp-2004", downsample=2)["dh"]         # (12.5, 25.0)   ← ×2 both axes
datasets.load("bp-2004", downsample=(2, 4))["dh"]    # (12.5, 50.0)   ← dz×2, dx×4
datasets.load("bp-2004", downsample=(2, 4))["vp"].shape        # (956, 1349)
datasets.load("seg-eage-salt", downsample=[3, 1, 1])["dh"]     # (60.0, 20.0, 20.0)  ← depth only (axes nz, ny, nx)
```

## Cache and dependencies

Downloads are cached under `$SWEEP_DATASETS_CACHE` →
`$XDG_CACHE_HOME/sweep-datasets` → `~/.cache/sweep-datasets`. A cached model is
reused on the next `load` (no re-download), after its file is checked against
the sha256 again; a file that no longer matches is downloaded again. Caches
written by earlier versions of sweep are reused the same way.

If no source serves a file, `load` raises `OSError` with every URL it tried and
the cache path the file belongs at: put the original file there (it must match
the sha256 in the message) and call `load` again.

Parsing (SEG-Y and raw grids) is **numpy-only** — no extra parser dependency.
Downloading needs an HTTP client, and the HDF5 copy that `overthrust:3d-acoustic`
falls back to, when no source serves the SEG/EAGE CD, is read with h5py:

```bash
pip install sweep-solver[datasets]     # adds requests + tqdm for downloads, h5py for the Overthrust 3-D fallback
```

Embedded demos need nothing beyond numpy.

## CLI

```bash
sweep datasets list                       # the catalog table (or: sweep-datasets list)
sweep datasets list marmousi              # one family
sweep datasets info bp-2004               # one entry's full metadata
sweep datasets download overthrust 3d-acoustic         # pre-fetch, report shape
sweep datasets download bp-2004 --downsample 2 4       # per-axis downsample
sweep datasets where                      # print the cache directory
```

`download` prints the error and exits with status 1, without a traceback, when
no source serves a file or an optional dependency is missing.

## Back-compatible helpers

The original embedded-model API is preserved, so existing notebooks keep
working unchanged:

```python
from sweep.datasets import load_marmousi, MARMOUSI_DH

vp = load_marmousi("vp_true")        # bare ndarray; presets: {vp,vs,rho}_{true,smooth,linear}
vs = load_marmousi("vs_smooth")
print(MARMOUSI_DH)                   # 12.5
```

## Licensing

Each entry prints its citation and license on first load, and carries a
`redistributable` flag: `True` for the CC-BY-4.0 models, whose files anyone may
re-host with attribution. The BP 2004 and BP 2007 terms ("AS IS", free download
with attribution) allow passing the files on only together with their original
documentation and terms, which the project's Zenodo records and the Hugging Face
copy include; loading these models prints a note saying so. `hess-vti` is not
re-hosted. The Chevron Data License Agreement of `chevron-fwi-2014` has the
same conditions and asks that publications using the data acknowledge Chevron;
its Hugging Face folder carries the agreement and the original README.

!!! warning "Intentionally not bundled"
    - **Sigsbee2A/2B** — gated behind a signed Data Release Agreement; original
      host is dead.
    - **SEAM Phase I** — license ambiguous across sources (CC-BY vs
      CC-BY-NC-SA) and no verifiable direct URL.
    - **OpenFWI** — datasets are CC BY-NC-SA 4.0 (non-commercial), incompatible
      with a permissive package; use the upstream `openfwi-lanl` tooling.

## Registering your own model

```python
from sweep.datasets import Entry, register

def _load_foo():
    return {"vp": my_array, "dh": (10.0, 10.0), "dt": 0.001, "nt": 2000}

register(Entry(
    name="foo", variant="2d-acoustic", loader=_load_foo,
    description="...", citation="...", license="...",
    kind="download", redistributable=False, tags=["2d", "acoustic"],
))
```

See also [CLI](cli.md) and the [Quick Start](../getting-started/quickstart.md).

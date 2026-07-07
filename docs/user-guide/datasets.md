# Datasets

`sweep.datasets` gives one-line access to benchmark velocity models for FWI /
LSRTM / RTM examples and smoke tests. Two flavours share a single registry:

- **Embedded demos** — tiny base85-encoded blobs (Marmousi, Overthrust-2D) that
  ship *inside the wheel*, so notebooks and quick-starts run with no downloads
  and no extra dependencies.
- **Downloadable benchmarks** — full-size public models fetched from their
  official open-data source on first use and cached locally, each carrying its
  own license and citation.

Implementation: `src/sweep/datasets/`.

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

The minimum you can rely on for any model is `vp` and `dh`.

!!! note "`variant` is optional"
    With a single variant it is inferred (`load("bp-2004")`). A multi-variant
    name resolves to its canonical default — `load("marmousi")` and
    `load("overthrust")` return the embedded `2d-demo`. If a name has several
    variants and no default, omitting `variant` raises and lists the choices
    rather than guessing.

## Catalog

| name | kind | grid | license |
|---|---|---|---|
| `marmousi` | embedded | 281×1361 @ 12.5 m | SEG open (demo) |
| `overthrust` | embedded | 187×801 @ 25 m | CC-BY src (demo) |
| `marmousi` | download | 2801×13601 @ 1.25 m | SEG open |
| `overthrust` | download | 187×801×801 @ 25 m | **CC-BY-4.0** |
| `seg-eage-salt` | download | 210×676×676 @ 20 m | **CC-BY-4.0** |
| `marmousi2` | download | 2801×13601 @ 1.25 m | open academic |
| `bp-2004` | download | 1911×5395 | free + attribution |
| `bp-2007-tti` | download | 1801×12596 @ 6.25 m | courtesy of BP |
| `hess-vti` | download | 1500×3617 @ ~7.62 m | Hess "AS IS" |

Multi-parameter models return every field: `marmousi2` → `vp`/`vs`/`rho`,
`bp-2007-tti` → `vp`/`epsilon`/`delta`/`theta`, `hess-vti` → `vp`/`epsilon`/`delta`.
Every download entry was byte-verified against a real fetch (grid shape, value
range, and slice image confirmed).

## Downsampling

Every download loader takes `downsample` — a single int (all axes) or a
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
datasets.load("seg-eage-salt", downsample=[1, 1, 3])["dh"]     # (20, 20, 60)  depth only
```

## Cache and dependencies

Downloads are cached under `$SWEEP_DATASETS_CACHE` →
`$XDG_CACHE_HOME/sweep-datasets` → `~/.cache/sweep-datasets`. A cached model is
reused on the next `load` (no re-download).

Parsing (SEG-Y and raw grids) is **numpy-only** — no extra parser dependency.
Downloading needs an HTTP client:

```bash
pip install sweep-solver[datasets]     # adds requests + tqdm for downloads
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
`redistributable` flag (`True` only when the license permits re-hosting the
bytes, e.g. CC-BY-4.0). For "free download + attribution / AS-IS" models the
loader fetches from the official source and never re-hosts the data.

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

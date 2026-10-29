# CLI and datasets

Contents: 1. `sweep` CLI · 2. `sweep.datasets` in Python · 3. Catalog · 4. Cache, downloads,
licences

## 1. `sweep` CLI

Introspection only (no modelling runs; YAML-driven FWI/LSRTM runs are the separate `sweepx`
toolkit's `sweep-tasks run`).

```bash
sweep list equations | head -n 12        # FD classes (PropTorch), then SEM classes (PropSEM)
sweep show Elastic                       # wavefields, model order, compiled-binding status
sweep show AcousticPotentialSEM2D        # SEM: mesh, C-PML, source/receiver types, models=
sweep show Acoustic3d || echo "exit=$?"  # unknown name: exit 1 + "Did you mean: Acoustic3D, ..."
sweep datasets list | head -n 15         # dataset catalog
sweep datasets where                     # cache directory
```

- `Torch Binding` column: the class has CUDA kernels. `Binding Ready`: `impl='c'` can run here
  now (GPU visible, fitting core). A yes/no mismatch: check
  `sweep.backend.torch.binding.diagnostics()["reason"]` or `python scripts/check_env.py`.
- `sweep show X > /dev/null` exits 0 only for a real equation: usable as an availability probe.
  The facades `DAS` and `AcousticAniso` are not listed and `show` refuses them.
- `Wavefields` in `show` is the full internal state (CPML memories included); valid
  `source_type`/`receiver_type` names come from `eq.available_source_fields()` /
  `eq.available_receiver_fields()`.

Downloads (large; not run here):

<!-- skill-test: skip -->
```bash
sweep datasets info bp-2004
sweep datasets download overthrust 3d-acoustic
sweep datasets download bp-2004 --downsample 2 4      # per-axis (dz, dx) decimation
```

## 2. `sweep.datasets` in Python

`load(name, variant=None, **kwargs)` returns a dict: always `vp` (float32 ndarray, depth first),
`dh` (per-axis spacing, model order), `name`, `variant`, `citation`, `license`; downloads also
give suggested `dt`, `nt`; multi-parameter models add fields (`marmousi2`: `vs`, `rho`;
`bp-2007-tti`: `epsilon`, `delta`, `theta`; `hess-vti`: `epsilon`, `delta`).

```python
import numpy as np, torch
from sweep import datasets
from sweep.datasets import load_marmousi, MARMOUSI_DH

prob = datasets.load("marmousi")              # embedded 2d-demo, offline, no download
vp, dh = prob["vp"], prob["dh"]               # (281, 1361) float32, (12.5, 12.5)
print(prob["variant"], vp.shape, dh, prob["presets"][:3])
vp_smooth = load_marmousi("vp_smooth")        # presets: {vp,vs,rho}_{true,smooth,linear}
print(vp_smooth.shape, MARMOUSI_DH)
print(datasets.available())                   # (name, variant) pairs

# typical: decimate the demo for a quick test, keep dh consistent
vp_small = vp[::4, ::4]
dh_small = tuple(4 * d for d in dh)
vp_t = torch.from_numpy(np.ascontiguousarray(vp_small))
assert vp_t.shape == (71, 341) and dh_small == (50.0, 50.0)
```

- `variant` is optional when a name has one variant or a default (`"marmousi"` and
  `"overthrust"` default to the embedded `2d-demo`); otherwise omitting it raises and lists the
  choices.
- `downsample=` (int or per-axis tuple) on download entries decimates every field and
  **multiplies `dh`** by the same factors (physical extent preserved).
- Feed `vp` as `torch.from_numpy(np.ascontiguousarray(vp)).to(device)` and `dh=prob["dh"]` (it is
  already in `(dz, dx)` order). Check stability for the model's `vp.max()` and dispersion for
  `vp.min()` (`scripts/stability_check.py`).

## 3. Catalog

| name | variant | kind | grid | licence |
|---|---|---|---|---|
| `marmousi` | `2d-demo` | embedded | 281 x 1361 @ 12.5 m | CC-BY source (demo) |
| `overthrust` | `2d-demo` | embedded | 187 x 801 @ 25 m | CC-BY source (demo) |
| `marmousi` | `2d-acoustic` | download | 2801 x 13601 @ 1.25 m | CC-BY-4.0 |
| `marmousi2` | `2d-elastic` | download | 2801 x 13601 @ 1.25 m | CC-BY-4.0 |
| `overthrust` | `3d-acoustic` | download | 187 x 801 x 801 @ 25 m | CC-BY-4.0 |
| `seg-eage-salt` | `3d-acoustic` | download | 210 x 676 x 676 @ 20 m | CC-BY-4.0 |
| `bp-2004` | `2d-acoustic` | download | 1911 x 5395 | free + attribution |
| `bp-2007-tti` | `2d-tti` | download | 1801 x 12596 @ 6.25 m | courtesy of BP |
| `hess-vti` | `2d-vti` | cache / manual only | 1500 x 3617 | Hess "AS IS" |
| `chevron-fwi-2014` | `2d-elastic-data` | download (shot data, 3.8 GB) | 1600 shots; `vp_init` only, no true model | Chevron DLA |

`chevron-fwi-2014` returns `data` `(nshots, 2001, 321)`, `vp_init`, `wavelet`, `well_log` and
`geometry` instead of `vp`; `fields=` and `shots=` limit what is read. Sigsbee, SEAM and
OpenFWI are intentionally not bundled (licences).

## 4. Cache, downloads, licences

- Cache: `$SWEEP_DATASETS_CACHE`, else `$XDG_CACHE_HOME/sweep-datasets`, else
  `~/.cache/sweep-datasets`. Every file is checked against its sha256 on each load.
- Downloads need `pip install "sweep-solver[datasets]"` (requests, tqdm, h5py). Sources: Zenodo
  records, then the Hugging Face dataset `shaowinw/seismic-benchmark-models`; the old SEG S3
  bucket no longer serves files. If nothing answers, `load` raises `OSError` listing every URL
  tried and the cache path to place the file at by hand.
- Each entry prints its citation and licence on first load; cite them in publications.
- Own models: `datasets.register(datasets.Entry(...))` (see `api-signatures.md`).

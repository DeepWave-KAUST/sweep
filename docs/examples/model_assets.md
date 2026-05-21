# Model Helper Scripts

> :material-github: **Source on GitHub** &mdash; [`examples/models/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/models) (clone, run, modify)

## Zero-download path (recommended for getting started)

Both Marmousi and the 2D Overthrust slice are **embedded inside the `sweep`
package** as fp16+zlib compressed presets — no external download is needed for
the 2D acoustic / elastic examples:

```python
from sweep.datasets import (
    load_marmousi, MARMOUSI_DH,
    load_overthrust_2d, OVERTHRUST_2D_DH,
)

# Marmousi (~100 KB embedded)
vp_true   = load_marmousi("true")          # (141, 681) float32, m/s
vp_smooth = load_marmousi("smooth")
vp_linear = load_marmousi("linear")
dh_marm   = MARMOUSI_DH                    # 25.0 m

# Overthrust 2D middle-y slice (~125 KB embedded)
vp_true_o   = load_overthrust_2d("true")   # (187, 801) float32, m/s
vp_smooth_o = load_overthrust_2d("smooth")
dh_over     = OVERTHRUST_2D_DH             # 25.0 m
```

Quantization error is < 2 m/s (< 0.05% relative), invisible for FWI demos.

For Overthrust 3D, the prepared NumPy arrays are too large (~460 MB each) to
ship inside the repo. Use the GitHub Release fetcher instead:

```bash
python examples/models/overthrust/download_release_asset.py
# → examples/models/overthrust/{true_3d,smooth_3d}.npy
```

## Slow path (regenerate from original SEG/EAGE sources)

The original archives (Marmousi tarball ~150 MB, Overthrust archive ~150 MB)
and intermediate SEG-Y / vites files are not committed. Use the helper scripts
below to recreate them from the public hosts when you need the full resolution
or non-default preparation.

The generated model files under `examples/models/` are intentionally ignored by
git (except the embedded Marmousi presets). The scripts in that directory are
tracked; downloaded archives, extracted SEG-Y files, converted NumPy arrays,
and preview figures are local artifacts.

Model preparation steps live with the examples that require those files:

- Marmousi acoustic FWI: [`01_fwi_acoustic_marmousi.ipynb`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/notebooks/01_fwi_acoustic_marmousi.ipynb)
- Marmousi elastic FWI: [`02_fwi_elastic_marmousi.ipynb`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/notebooks/02_fwi_elastic_marmousi.ipynb)
- Marmousi acoustic LSRTM: [2D Acoustic LSRTM on Marmousi with Torch](acoustic_lsrtm_torch.md)
- Marmousi source-encoded acoustic FWI: [2D Acoustic FWI with Source Encoding on Marmousi with Torch](acoustic_fwi_encoding_torch.md)
- Overthrust 3D acoustic FWI: [3D Acoustic FWI on Overthrust with Torch](acoustic_fwi_3d_torch.md)
- Overthrust 2D elastic FWI: [2D Elastic FWI on Overthrust with Torch](elastic_fwi_torch_overthrust.md)
- Overthrust 3D acoustic LSRTM: [3D Acoustic LSRTM on Overthrust with Torch](acoustic_lsrtm_3d_torch.md)

## Marmousi Scripts

Folder:

- [`examples/models/marmousi/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/models/marmousi)

Tracked helper scripts:

- `download_marmousi.py`
- `extract_model_segy.py`
- `convert_segy_to_npy.py`
- `prepare_fwi_models.py`
- `plot_models.py`
- `clean_generated.py`

Typical generated files:

- [`examples/models/marmousi/npy/vp_1p25m.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/npy/vp_1p25m.npy)
- [`examples/models/marmousi/npy/vs_1p25m.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/npy/vs_1p25m.npy)
- [`examples/models/marmousi/npy/rho_1p25m.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/npy/rho_1p25m.npy)
- [`examples/models/marmousi/true.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/true.npy)
- [`examples/models/marmousi/smooth.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/smooth.npy)
- [`examples/models/marmousi/linear.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/linear.npy)

## Overthrust Scripts

Folder:

- [`examples/models/overthrust/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/models/overthrust)

Tracked helper scripts:

- `download_release_asset.py` &mdash; **fast path**: pull prepared `true_3d.npy` / `smooth_3d.npy` from the SWEEP GitHub Release
- `download_3d_overthrust.py` &mdash; slow path: fetch the original SEG/EAGE archive
- `convert_3d_overthrust_vites_to_npy.py`
- `make_smooth_model.py`
- `extract_2d_slice.py`
- `plot_true_smooth.py`
- `clean_generated.py`

Typical generated files:

- [`examples/models/overthrust/true_3d.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/overthrust/true_3d.npy)
- [`examples/models/overthrust/smooth_3d.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/overthrust/smooth_3d.npy)
- [`examples/models/overthrust/true_2d.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/overthrust/true_2d.npy)
- [`examples/models/overthrust/smooth_2d.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/overthrust/smooth_2d.npy)

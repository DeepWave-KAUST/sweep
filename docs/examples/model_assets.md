# Model Helper Scripts

The generated model files under `examples/models/` are intentionally ignored by
git. The scripts in that directory are tracked; downloaded archives, extracted
SEG-Y files, converted NumPy arrays, and preview figures are local artifacts.

Model preparation steps live with the examples that require those files:

- Marmousi acoustic FWI: [2D Acoustic FWI on Marmousi with Torch](acoustic_fwi_torch.md)
- Marmousi acoustic FWI with JAX: [2D Acoustic FWI on Marmousi with JAX](acoustic_fwi_jax.md)
- Marmousi elastic FWI: [2D Elastic FWI on Marmousi with Torch](elastic_fwi_torch_marmousi.md)
- Marmousi acoustic LSRTM: [2D Acoustic LSRTM on Marmousi with Torch](acoustic_lsrtm_torch.md)
- Marmousi source-encoded acoustic FWI: [2D Acoustic FWI with Source Encoding on Marmousi with Torch](acoustic_fwi_encoding_torch.md)
- Overthrust 3D acoustic FWI: [3D Acoustic FWI on Overthrust with Torch](acoustic_fwi_3d_torch.md)
- Overthrust 2D elastic FWI: [2D Elastic FWI on Overthrust with Torch](elastic_fwi_torch_overthrust.md)
- Overthrust 3D acoustic LSRTM: [3D Acoustic LSRTM on Overthrust with Torch](acoustic_lsrtm_3d_torch.md)

## Marmousi Scripts

Folder:

- `examples/models/marmousi/`

Tracked helper scripts:

- `download_marmousi.py`
- `extract_model_segy.py`
- `convert_segy_to_npy.py`
- `prepare_fwi_models.py`
- `plot_models.py`
- `clean_generated.py`

Typical generated files:

- `examples/models/marmousi/npy/vp_1p25m.npy`
- `examples/models/marmousi/npy/vs_1p25m.npy`
- `examples/models/marmousi/npy/rho_1p25m.npy`
- `examples/models/marmousi/true.npy`
- `examples/models/marmousi/smooth.npy`
- `examples/models/marmousi/linear.npy`

## Overthrust Scripts

Folder:

- `examples/models/overthrust/`

Tracked helper scripts:

- `download_3d_overthrust.py`
- `convert_3d_overthrust_vites_to_npy.py`
- `make_smooth_model.py`
- `extract_2d_slice.py`
- `plot_true_smooth.py`
- `clean_generated.py`

Typical generated files:

- `examples/models/overthrust/true_3d.npy`
- `examples/models/overthrust/smooth_3d.npy`
- `examples/models/overthrust/true_2d.npy`
- `examples/models/overthrust/smooth_2d.npy`

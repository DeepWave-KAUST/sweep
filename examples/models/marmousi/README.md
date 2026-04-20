# Marmousi Model Assets

This folder stores the scripts and derived NumPy models used by the Marmousi-based FWI examples.

Official Marmousi source:

- `https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz`

Key files inside the official archive:

- `elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz`
- `elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz`
- `elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.tar.gz`

The current acoustic FWI examples expect:

- `true.npy`
- `smooth.npy`
- `linear.npy`

Usage:

1. Download and extract the official Marmousi archive.

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
```

2. Extract the three model SEG-Y files into `examples/models/marmousi/segy/`.

```bash
python3 examples/models/marmousi/extract_model_segy.py
```

3. Convert the Marmousi SEG-Y files into NumPy arrays at the original `1.25 m` spacing.

```bash
python3 examples/models/marmousi/convert_segy_to_npy.py
```

4. Build the downsampled FWI-ready acoustic models expected by the current examples.

```bash
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 \
  --target-dh 25.0 \
  --radii 8,8 \
  --passes 3
```

5. Plot the prepared `true/smooth/linear` models for a quick sanity check.

```bash
python3 examples/models/marmousi/plot_models.py
```

6. Remove downloaded and generated artifacts when you want to reset this folder.

```bash
python3 examples/models/marmousi/clean_generated.py
```

Notes:

- `convert_segy_to_npy.py` requires `segyio`.
- `prepare_fwi_models.py` uses the P-wave velocity grid to generate the acoustic FWI inputs.
- The default preparation step downsamples the original `1.25 m` grid to `25 m` so it matches the current example configuration.
- `true.npy`, `smooth.npy`, and `linear.npy` are written directly into this folder because the FWI example configs already point here.

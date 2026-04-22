# Model Assets

This page summarizes how the example models under `examples/models/` are
obtained and how the NumPy files used by the FWI examples are generated.

## Marmousi

Folder:

- `examples/models/marmousi/`

Key scripts:

- `download_marmousi.py`: download the official Elastic Marmousi archive
- `extract_model_segy.py`: extract the model SEG-Y files from the archive
- `convert_segy_to_npy.py`: convert SEG-Y grids to `.npy`
- `prepare_fwi_models.py`: build FWI-ready `true/smooth/linear` acoustic models
- `plot_models.py`: save a quick comparison figure
- `clean_generated.py`: remove downloaded and generated artifacts

Official source archive:

- `https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz`

Important files inside the archive:

- `elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz`
- `elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz`
- `elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.tar.gz`

### Step 1. Download and extract the official archive

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
```

### Step 2. Extract the SEG-Y model files

```bash
python3 examples/models/marmousi/extract_model_segy.py
```

### Step 3. Convert SEG-Y to NumPy

```bash
python3 examples/models/marmousi/convert_segy_to_npy.py
```

This creates:

- `examples/models/marmousi/npy/vp_1p25m.npy`
- `examples/models/marmousi/npy/vs_1p25m.npy`
- `examples/models/marmousi/npy/rho_1p25m.npy`

### Step 4. Build the FWI-ready acoustic models

```bash
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 \
  --target-dh 25.0 \
  --radii 8,8 \
  --passes 3
```

This generates:

- `examples/models/marmousi/true.npy`
- `examples/models/marmousi/smooth.npy`
- `examples/models/marmousi/linear.npy`

Notes:

- `prepare_fwi_models.py` uses the P-wave velocity grid to generate the acoustic
  FWI inputs.
- the default workflow downsamples the original `1.25 m` grid to `25 m` so it
  matches the current example configuration.
- `convert_segy_to_npy.py` requires `segyio`

### Step 5. Plot the prepared models

```bash
python3 examples/models/marmousi/plot_models.py
```

This saves:

- `examples/models/marmousi/true_smooth_linear.png`

Preview:

`true_smooth_linear.png`: a three-panel preview of the generated Marmousi
acoustic inputs, showing the true model, the smoothed initial model, and the
linear initial model.

![Marmousi true, smooth, and linear models](../figures/examples/marmousi_true_smooth_linear.png)

### Step 6. Clean generated artifacts

```bash
python3 examples/models/marmousi/clean_generated.py
```

## Overthrust

Folder:

- `examples/models/overthrust/`

Key scripts:

- `download_3d_overthrust.py`: download the official Overthrust archive
- `convert_3d_overthrust_vites_to_npy.py`: convert the official binary grid to
  `true_3d.npy`
- `make_smooth_model.py`: create smoothed 2D or 3D initial models
- `extract_2d_slice.py`: extract a 2D slice from the 3D volume
- `plot_true_smooth.py`: save preview figures
- `clean_generated.py`: remove downloaded and generated artifacts

Official source archive:

- `https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Overthrust_3D_CD1.tar.gz`

Important files inside the archive:

- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites`
- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites.h`
- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites.vo`

### Step 1. Download and extract the official archive

```bash
python3 examples/models/overthrust/download_3d_overthrust.py --extract
```

### Step 2. Convert the 3D binary model to NumPy

```bash
python3 examples/models/overthrust/convert_3d_overthrust_vites_to_npy.py
```

This creates:

- `examples/models/overthrust/true_3d.npy`

The converter reads the official binary using:

```python
vel = np.fromfile("overthrust.vites", dtype=">f4")
vel = vel.reshape(187, 801, 801, order="C")
```

### Step 3. Build a smoothed 3D initial model

```bash
python3 examples/models/overthrust/make_smooth_model.py \
  --input examples/models/overthrust/true_3d.npy \
  --output examples/models/overthrust/smooth_3d.npy \
  --radii 6,6,6 \
  --passes 3
```

This creates:

- `examples/models/overthrust/smooth_3d.npy`

### Step 4. Extract 2D models for 2D examples

True model:

```bash
python3 examples/models/overthrust/extract_2d_slice.py \
  --input examples/models/overthrust/true_3d.npy \
  --output examples/models/overthrust/true_2d.npy \
  --axis y
```

Smooth model:

```bash
python3 examples/models/overthrust/extract_2d_slice.py \
  --input examples/models/overthrust/smooth_3d.npy \
  --output examples/models/overthrust/smooth_2d.npy \
  --axis y
```

This creates:

- `examples/models/overthrust/true_2d.npy`
- `examples/models/overthrust/smooth_2d.npy`

Notes:

- `extract_2d_slice.py` assumes the volume order is `(z, y, x)`
- `make_smooth_model.py` supports both 2D and 3D `.npy` inputs

### Step 5. Plot the generated models

```bash
python3 examples/models/overthrust/plot_true_smooth.py
```

This saves:

- `examples/models/overthrust/true_smooth_3d_slices.png`
- `examples/models/overthrust/true_smooth_2d.png`

### Step 6. Clean generated artifacts

```bash
python3 examples/models/overthrust/clean_generated.py
```

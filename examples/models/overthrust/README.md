# 3D Overthrust Model Assets

This folder stores the 2D and 3D overthrust assets used by the examples.

Current local files:

- `true_2d.npy`
- `smooth_2d.npy`
- `preview.png`

Official 3D model source:

- `https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Overthrust_3D_CD1.tar.gz`
- SEG wiki landing page: `https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models`

Key files inside the official archive:

- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites`
- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites.h`
- `Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites.vo`

The converter script uses the direct loading pattern:

```python
vel = np.fromfile("overthrust.vites", dtype=">f4")
vel = vel.reshape(187, 801, 801, order="C")
```

Usage:

1. Download and extract the official Overthrust 3D archive.

```bash
python3 examples/models/overthrust/download_3d_overthrust.py --extract
```

2. Convert the official `overthrust.vites` binary velocity grid into `true_3d.npy`.

```bash
python3 examples/models/overthrust/convert_3d_overthrust_vites_to_npy.py
```

3. Build a smoothed 3D initial model from the 3D true model.

```bash
python3 examples/models/overthrust/make_smooth_model.py \
  --input examples/models/overthrust/true_3d.npy \
  --output examples/models/overthrust/smooth_3d.npy \
  --radii 6,6,6 \
  --passes 3
```

4. Extract the middle `y` slice from the 3D true model to create the 2D true model.

```bash
python3 examples/models/overthrust/extract_2d_slice.py \
  --input examples/models/overthrust/true_3d.npy \
  --output examples/models/overthrust/true_2d.npy \
  --axis y
```

5. Extract the same `y` slice from the 3D smooth model to create the 2D smooth initial model.

```bash
python3 examples/models/overthrust/extract_2d_slice.py \
  --input examples/models/overthrust/smooth_3d.npy \
  --output examples/models/overthrust/smooth_2d.npy \
  --axis y
```

6. Plot the 3D true/smooth middle slices and the 2D true/smooth comparison.

```bash
python3 examples/models/overthrust/plot_true_smooth.py
```

7. Remove downloaded and generated artifacts when you want to reset this folder.

```bash
python3 examples/models/overthrust/clean_generated.py
```

Notes:

- The download script fetches the official `Overthrust_3D_CD1.tar.gz` archive and can extract it in place.
- The converter reads the extracted `overthrust.vites` binary grid and writes `true_3d.npy`.
- `make_smooth_model.py` creates a 2D or 3D smoothed initial model using edge-padded box smoothing.
- `extract_2d_slice.py` extracts a 2D slice from a 3D model; it assumes the array order is `(z, y, x)`.
- `plot_true_smooth.py` saves one figure for 3D true/smooth middle slices and one for 2D true/smooth models.
- The converter follows the verified read pattern `dtype=">f4"` with shape `(187, 801, 801)`.

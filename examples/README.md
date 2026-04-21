# Examples Layout

`examples/` has been reorganized by task family:

- `FWI/`
- `LSRTM/`
- `jointmigrationinversion/`
- `wavefields/`
- `reducingmemory/`
- `multi-gpu/`

Rules used for the main inversion examples:

- `FWI/` and `LSRTM/` are split into `2d/` and `3d/`
- each `2d/` or `3d/` folder is split into `acoustic/` and `elastic/`
- each physics folder is split into `jax/` and `torch/`

Shared configs live in `examples/_shared/`.
Historical scratch scripts live in `examples/_legacy/`.
All model assets are centralized in `examples/models/`.
The 3D overthrust download and conversion helpers live in `examples/models/overthrust/`.

Current populated examples include:

- `FWI/2d/acoustic/jax/fwi_marmousi.py`
- `FWI/2d/acoustic/torch/fwi_marmousi.py`
- `FWI/3d/acoustic/jax/fwi_overthrust.py`
- `FWI/3d/acoustic/torch/fwi_overthrust.py`
- `FWI/2d/elastic/jax/fwi_marmousi.py`
- `FWI/2d/elastic/jax/fwi_overthrust.py`
- `FWI/2d/elastic/torch/fwi_overthrust.py`
- `LSRTM/2d/acoustic/jax/lsrtm.py`
- `wavefields/elastic_free_surface_view_torch.py`
- `reducingmemory/acoustic/jax/source_encoding_fwi.py`
- `reducingmemory/acoustic/torch/source_encoding_fwi.py`
- `reducingmemory/acoustic/torch/vrz_forward_compare.py`

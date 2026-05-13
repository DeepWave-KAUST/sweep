# Anisotropic wavefield examples

This directory contains compact forward-modeling examples for anisotropic
wave equations.

## Scripts

- `anisotropic_wavefields.py`: acoustic qP examples for Tariq, VTI, and TTI.
- `elastic_tti_wavefields.py`: representative 2D three-component elastic TTI
  experiments on rotated staggered grid (RSG) and standard staggered grid (SG).

## Elastic TTI examples

Run both representative experiments:

```bash
python elastic_tti_wavefields.py --device cuda
```

Run only the rotation experiment:

```bash
python elastic_tti_wavefields.py --experiment rotation --device cuda
```

Run only the free-surface comparison:

```bash
python elastic_tti_wavefields.py --experiment free-surface --device cuda
```

For a quick smoke run:

```bash
python elastic_tti_wavefields.py --quick --device cuda
```

The script writes compact figures and a metrics text file into `outputs/`.

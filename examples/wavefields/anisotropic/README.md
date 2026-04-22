# Anisotropic Wavefields

This example runs an isotropic acoustic baseline plus the qP anisotropic equations with a centered source in a square physical domain and saves:

- `acoustic_tariq_snapshots.png`: wavefield snapshots for Acoustic and Tariq
- `acoustic_tariq_records.png`: receiver records for Acoustic and Tariq
- `vti_snapshots.png`: wavefield snapshots for `VTI-A/B/C`
- `vti_records.png`: receiver records for `VTI-A/B/C`
- `tti_snapshots.png`: wavefield snapshots for `TTI-A/B/C`
- `tti_records.png`: receiver records for `TTI-A/B/C`
- `boundary_metrics.txt`: simple edge-to-interior amplitude ratios at each snapshot time

Common simulation parameters:

- Physical size: `(Lz, Lx) = (960 m, 960 m)`
- Grid spacing: `(dz, dx) = (5 m, 15 m)`
- Time step: `dt = 0.001 s`
- Number of time steps: `nt = 1100`
- Absorbing boundary width: `abcn = 20`
- PML type: `cpmlr`
- Dominant frequency: `15 Hz`
- Spatial order: `8`
- Snapshot times: `220, 320, 460`
- Source location: center of the physical domain, `(x, z) = (480 m, 480 m)`
- Receiver line: `z = 90 m`, `x = 90 m .. 870 m`, `33` receivers

Model parameters used in the comparisons:

| Model | `vp` (m/s) | `vv` (m/s) | `v` (m/s) | `epsilon` | `delta` | `eta` | `theta` (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Acoustic baseline | 2500 | - | - | - | - | - | - |
| Tariq | - | 2300 | 2100 | - | - | 0.15 | - |
| VTI-A | 2500 | - | - | 0.3 | 0.3 | - | - |
| VTI-B | 2500 | - | - | 0.3 | 0.1 | - | - |
| VTI-C | 2500 | - | - | 0.1 | 0.3 | - | - |
| TTI-A | 2500 | - | - | 0.3 | 0.3 | - | 20 |
| TTI-B | 2500 | - | - | 0.3 | 0.1 | - | 20 |
| TTI-C | 2500 | - | - | 0.1 | 0.3 | - | 20 |

Recorded fields:

- Acoustic, VTI, TTI: source on `h1`, receivers on `h1`
- Tariq: source on `h1`, receivers on `f1`

Run it with:

```bash
cd /home/wangs0j/repo/sweep/examples/wavefields/anisotropic
python anisotropic_wavefields.py
```

If you want the repo sources instead of an installed package:

```bash
PYTHONPATH=../../../src python anisotropic_wavefields.py
```

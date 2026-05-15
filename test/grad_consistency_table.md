# Gradient consistency (eager autograd vs C-impl with fast-path)

Thresholds: `cosine ≥ 0.8`, `rel_l2 ≤ 1.5`. **bold** cosines flag failures.

| solver | scenario | model | rel_l2 | cosine | status |
|---|---|---|---|---|---|
| acoustic2d | interior | vp | 4.637e-01 | 1.0000 | ✓ |
| acoustic2d | fd_edge | vp | 3.665e-01 | 0.9977 | ✓ |
| acoustic2d | free_surface | vp | 4.513e-01 | 1.0000 | ✓ |
| acoustic3d | interior | vp | 4.066e-02 | 1.0000 | ✓ |
| acoustic3d | fd_edge | vp | 3.822e-01 | 0.9973 | ✓ |
| acoustic3d | free_surface | vp | 9.756e-02 | 1.0000 | ✓ |
| vrz2d | interior | vp | 4.049e-01 | 0.9309 | ✓ |
| vrz2d | interior | z | 4.555e-01 | 0.9293 | ✓ |
| vrz2d | fd_edge | vp | 2.637e-01 | 0.9768 | ✓ |
| vrz2d | fd_edge | z | 8.137e-01 | 0.8796 | ✓ |
| vrz2d | free_surface | vp | 2.271e-01 | 0.9756 | ✓ |
| vrz2d | free_surface | z | 1.392e+00 | 0.8343 | ✓ |
| vrz3d | interior | vp | 2.887e-01 | 0.9838 | ✓ |
| vrz3d | interior | z | 1.508e+00 | **0.8617** | ✗ |
| vrz3d | fd_edge | vp | 3.727e-01 | 0.9799 | ✓ |
| vrz3d | fd_edge | z | 1.077e+00 | 0.8707 | ✓ |
| vrz3d | free_surface | vp | 2.955e-01 | 0.9817 | ✓ |
| vrz3d | free_surface | z | 1.455e+00 | 0.8604 | ✓ |
| lsrtm2d | interior | mp | 8.667e-01 | 1.0000 | ✓ |
| lsrtm2d | fd_edge | mp | 8.647e-01 | 1.0000 | ✓ |
| lsrtm2d | free_surface | mp | 8.628e-01 | 1.0000 | ✓ |
| lsrtm3d | interior | mp | 8.704e-01 | 1.0000 | ✓ |
| lsrtm3d | fd_edge | mp | 8.672e-01 | 1.0000 | ✓ |
| lsrtm3d | free_surface | mp | 8.628e-01 | 1.0000 | ✓ |
| das2d | interior | rho | 2.235e-05 | 1.0000 | ✓ |
| das2d | interior | vp | 4.077e-05 | 1.0000 | ✓ |
| das2d | interior | vs | 1.767e-05 | 1.0000 | ✓ |
| das3d | interior | rho | 2.293e-05 | 1.0000 | ✓ |
| das3d | interior | vp | 2.608e-05 | 1.0000 | ✓ |
| das3d | interior | vs | 2.263e-04 | 1.0000 | ✓ |
| das_mu2d | interior | rho | 3.153e-02 | 0.9995 | ✓ |
| das_mu2d | interior | vp | 5.009e-06 | 1.0000 | ✓ |
| das_mu2d | interior | vs | 6.489e-06 | 1.0000 | ✓ |
| das_mu2d | fd_edge | rho | 3.476e-02 | 0.9994 | ✓ |
| das_mu2d | fd_edge | vp | 1.078e-03 | 1.0000 | ✓ |
| das_mu2d | fd_edge | vs | 5.045e-04 | 1.0000 | ✓ |
| das_mu2d | free_surface | rho | 2.885e-02 | 0.9996 | ✓ |
| das_mu2d | free_surface | vp | 1.398e-05 | 1.0000 | ✓ |
| das_mu2d | free_surface | vs | 2.196e-05 | 1.0000 | ✓ |
| das_mu3d | interior | rho | 2.026e-01 | 0.9801 | ✓ |
| das_mu3d | interior | vp | 1.202e-05 | 1.0000 | ✓ |
| das_mu3d | interior | vs | 3.952e-06 | 1.0000 | ✓ |
| das_mu3d | fd_edge | rho | 2.726e-02 | 0.9996 | ✓ |
| das_mu3d | fd_edge | vp | 2.380e-04 | 1.0000 | ✓ |
| das_mu3d | fd_edge | vs | 1.126e-04 | 1.0000 | ✓ |
| das_mu3d | free_surface | rho | 3.751e-02 | 0.9993 | ✓ |
| das_mu3d | free_surface | vp | 3.214e-06 | 1.0000 | ✓ |
| das_mu3d | free_surface | vs | 2.719e-06 | 1.0000 | ✓ |
| elastic2d | interior | rho | 3.153e-02 | 0.9995 | ✓ |
| elastic2d | interior | vp | 5.110e-06 | 1.0000 | ✓ |
| elastic2d | interior | vs | 6.348e-06 | 1.0000 | ✓ |
| elastic2d | fd_edge | rho | 3.476e-02 | 0.9994 | ✓ |
| elastic2d | fd_edge | vp | 1.077e-03 | 1.0000 | ✓ |
| elastic2d | fd_edge | vs | 5.042e-04 | 1.0000 | ✓ |
| elastic2d | free_surface | rho | 2.885e-02 | 0.9996 | ✓ |
| elastic2d | free_surface | vp | 1.386e-05 | 1.0000 | ✓ |
| elastic2d | free_surface | vs | 2.194e-05 | 1.0000 | ✓ |
| elastic3d | interior | rho | 2.026e-01 | 0.9801 | ✓ |
| elastic3d | interior | vp | 1.359e-05 | 1.0000 | ✓ |
| elastic3d | interior | vs | 4.379e-06 | 1.0000 | ✓ |
| elastic3d | fd_edge | rho | 2.725e-02 | 0.9996 | ✓ |
| elastic3d | fd_edge | vp | 1.043e-04 | 1.0000 | ✓ |
| elastic3d | fd_edge | vs | 5.766e-05 | 1.0000 | ✓ |
| elastic3d | free_surface | rho | 3.751e-02 | 0.9993 | ✓ |
| elastic3d | free_surface | vp | 2.607e-06 | 1.0000 | ✓ |
| elastic3d | free_surface | vs | 2.805e-06 | 1.0000 | ✓ |
| elastic_tti_sg2d | interior | rho | 2.439e-02 | 0.9997 | ✓ |
| elastic_tti_sg2d | interior | vp0 | 1.683e-05 | 1.0000 | ✓ |
| elastic_tti_sg2d | interior | vs0 | 1.181e-05 | 1.0000 | ✓ |
| elastic_tti_sg2d | fd_edge | rho | 2.924e-02 | 0.9996 | ✓ |
| elastic_tti_sg2d | fd_edge | vp0 | 1.771e-03 | 1.0000 | ✓ |
| elastic_tti_sg2d | fd_edge | vs0 | 5.391e-04 | 1.0000 | ✓ |
| elastic_tti_sg2d | free_surface | rho | 2.584e-02 | 0.9997 | ✓ |
| elastic_tti_sg2d | free_surface | vp0 | 1.449e-02 | 0.9999 | ✓ |
| elastic_tti_sg2d | free_surface | vs0 | 3.260e-03 | 1.0000 | ✓ |

## Failures (1)
- **vrz3d/interior/z**: cos=0.8617, rel_l2=1.508e+00


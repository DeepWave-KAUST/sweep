# CUDA FD Orders

These scripts compare PropCUDA wavefields under different spatial orders on a
uniform 2D model.

- `acoustic_fd_orders.py`: acoustic wavefield comparison for orders 2, 6, 10, 14
- `elastic_fd_orders.py`: elastic `vx` and `vz` comparison for orders 2, 6, 10, 14 using an isotropic stress source

The source is placed at the center of the model and the outputs are written to
the local `outputs/` directory.

Run:

```bash
python acoustic_fd_orders.py
python elastic_fd_orders.py
```

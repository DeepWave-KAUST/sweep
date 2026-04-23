from __future__ import annotations

import numpy as np


def receiver_distance_axis(receivers, dh):
    coords = np.asarray(receivers, dtype=np.float32)
    if coords.ndim == 3:
        coords = coords[0]
    if coords.ndim != 2:
        raise ValueError(f"Expected receivers with shape (nrec, dim) or (nshot, nrec, dim), got {coords.shape}")
    offsets = (coords - coords[0]) * float(dh)
    return np.linalg.norm(offsets, axis=-1)


def gather_extent(nt, dt, receivers, dh):
    distance = receiver_distance_axis(receivers, dh)
    tmax = float(max(nt - 1, 0) * float(dt))
    return [float(distance.min()), float(distance.max()), tmax, 0.0]


def plot_loss_curve(ax, losses, ylabel):
    iterations = np.arange(1, len(losses) + 1, dtype=np.int32)
    ax.plot(iterations, losses, color="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)

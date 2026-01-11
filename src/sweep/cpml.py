"""
The Original Code is part of Deepwave (https://github.com/ar4/deepwave/tree/master),
which implements CPML in second-order acoustic wave equation.
For mathematical details, please refer to: 
10.1190/1.3513453 <Convolutional perfectly matched layer for isotropic and anisotropic acoustic wave equations>.
"""
import numpy as np
import math
from typing import List, Optional, Tuple

def setup_pml(
    pml_width: List[int],
    pml_start: List[float],
    max_pml: float,
    dt: float,
    n: int,
    max_vel: float,
    dtype: np.dtype,
    pml_freq: float,
    start: float = 0.0,
    r_val: float = 0.001,
    n_power: int = 2,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a and b profiles for C-PML (NumPy version)."""
    alpha0 = math.pi * pml_freq

    if max_pml == 0:
        a = np.zeros(n, dtype=dtype)
        b = np.zeros(n, dtype=dtype)
        return a, b

    sigma0 = -(1 + n_power) * max_vel * math.log(r_val) / (2 * max_pml)

    # x in "grid cells"
    x = (np.arange(n, dtype=dtype) + dtype(start))

    if pml_width[0] == 0:
        pml_frac0 = np.zeros_like(x)
    else:
        pml_frac0 = (dtype(pml_start[0]) - x) / dtype(pml_width[0])

    if pml_width[1] == 0:
        pml_frac1 = np.zeros_like(x)
    else:
        pml_frac1 = (x - dtype(pml_start[1])) / dtype(pml_width[1])

    pml_frac = np.maximum(pml_frac0, pml_frac1)
    pml_frac = np.clip(pml_frac, 0.0, 1.0)

    sigma = dtype(sigma0) * (pml_frac ** n_power)
    alpha = dtype(alpha0) * (dtype(1.0) - pml_frac)
    sigmaalpha = sigma + alpha

    # avoid div-by-zero (just in case)
    sigmaalpha_safe = np.where(np.abs(sigmaalpha) < eps, dtype(eps), sigmaalpha)

    a = np.exp(-sigmaalpha_safe * abs(dt)).astype(dtype, copy=False)
    b = (sigma / sigmaalpha_safe) * (a - dtype(1.0))

    # keep same behavior as torch: set a=0 where pml_frac==0
    a = a.copy()
    a[pml_frac == 0] = dtype(0.0)

    return a, b.astype(dtype, copy=False)


def diffx1(a: np.ndarray, accuracy: int, rdx: float) -> np.ndarray:
    """First derivative in the last axis (NumPy version), with zero-padding like torch.nn.functional.pad."""
    a = np.asarray(a)
    rdx = float(rdx)

    if accuracy == 2:
        core = (0.5 * (a[..., 2:] - a[..., :-2])) * rdx
        return np.pad(core, pad_width=[(0, 0)] * (a.ndim - 1) + [(1, 1)], mode="constant")

    if accuracy == 4:
        core = (
            (8.0 / 12.0) * (a[..., 3:-1] - a[..., 1:-3])
            + (-1.0 / 12.0) * (a[..., 4:] - a[..., :-4])
        ) * rdx
        return np.pad(core, pad_width=[(0, 0)] * (a.ndim - 1) + [(2, 2)], mode="constant")

    if accuracy == 6:
        core = (
            (3.0 / 4.0) * (a[..., 4:-2] - a[..., 2:-4])
            + (-3.0 / 20.0) * (a[..., 5:-1] - a[..., 1:-5])
            + (1.0 / 60.0) * (a[..., 6:] - a[..., :-6])
        ) * rdx
        return np.pad(core, pad_width=[(0, 0)] * (a.ndim - 1) + [(3, 3)], mode="constant")

    # default: accuracy == 8 (same as your torch code)
    core = (
        (4.0 / 5.0) * (a[..., 5:-3] - a[..., 3:-5])
        + (-1.0 / 5.0) * (a[..., 6:-2] - a[..., 2:-6])
        + (4.0 / 105.0) * (a[..., 7:-1] - a[..., 1:-7])
        + (-1.0 / 280.0) * (a[..., 8:] - a[..., :-8])
    ) * rdx
    return np.pad(core, pad_width=[(0, 0)] * (a.ndim - 1) + [(4, 4)], mode="constant")


def set_pml_profiles(
    pml_width: List[int],
    accuracy: int,
    fd_pad: List[int],
    dt: float,
    grid_spacing: List[float],
    max_vel: float,
    dtype: np.dtype,
    pml_freq: float,
    shape: Tuple[int, ...],
) -> List[np.ndarray]:
    """Sets up PML profiles for a regular grid (NumPy version).

    Returns list like: [a_dim0, b_dim0, db_dim0, a_dim1, b_dim1, db_dim1, ...]
    where each entry is broadcastable to (1, *shape) if you treat leading axis as batch/channel.
    """
    ndim = len(shape)

    pml_start: List[List[float]] = [
        [
            float(fd_pad[dim * 2] + pml_width[dim * 2]),
            float(shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1]),
        ]
        for dim in range(ndim)
    ]

    physical_widths: List[float] = []
    for dim in range(ndim):
        physical_widths.append(pml_width[dim * 2] * grid_spacing[dim])
        physical_widths.append(pml_width[dim * 2 + 1] * grid_spacing[dim])
    max_pml = max(physical_widths) if physical_widths else 0.0

    pml_profiles: List[np.ndarray] = []

    for dim in range(ndim):
        a, b = setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            pml_start[dim],
            max_pml,
            dt,
            shape[dim],
            max_vel,
            dtype,
            pml_freq,
        )

        db = diffx1(b, accuracy, 1.0 / grid_spacing[dim])

        # Make them broadcast like torch's a[tuple(s)] where s has length ndim+1
        # torch code: s = [None]*(ndim+1); s[1+dim] = slice(None)
        # => output shape: (1, ..., shape[dim], ...), total ndim+1 dims
        s: List[Optional[object]] = [None] * (ndim + 1)
        s[1 + dim] = slice(None)

        a_b = a[tuple(s)]
        b_b = b[tuple(s)]
        db_b = db[tuple(s)]

        pml_profiles.extend([a_b, b_b, db_b])

    return pml_profiles
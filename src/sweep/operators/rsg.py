"""Rotated staggered-grid derivative operators.

This module implements the fused RSG ``Lx`` and ``Lz`` operators from thesis
equations 3-11 and 3-12. The forward variants map integer-node fields to
cell-centered samples; the backward variants map cell-centered fields back to
integer-node samples. Both variants use odd-sized kernels, with the forward
kernel shifted by one grid index relative to the backward kernel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sweep.scalars import staggered_grid_coes


def _build_Lx_kernels(half_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Lx_fwd, Lx_bwd)`` kernels before spacing scaling.

    The index formulas in the RSG spec are written as ``(x, z)`` offsets.
    SWEEP tensors are stored as ``(..., z, x)``, so the arrays are transposed
    before returning. With that layout, ``lx_*`` differentiates along the last
    tensor axis and ``lz_*`` along the penultimate axis.
    """

    coes = staggered_grid_coes(half_order)
    size = 2 * half_order + 1
    center = half_order
    lx_bwd = np.zeros((size, size), dtype=np.float32)
    lx_fwd = np.zeros((size, size), dtype=np.float32)

    for m, coef in enumerate(coes, start=1):
        # Backward: input at cell centers, output at integer nodes.
        lx_bwd[center + m - 1, center + m - 1] += coef
        lx_bwd[center + m - 1, center - m] += coef
        lx_bwd[center - m, center - m] -= coef
        lx_bwd[center - m, center + m - 1] -= coef

        # Forward: input at integer nodes, output at cell centers.
        lx_fwd[center + m, center + m] += coef
        lx_fwd[center + m, center - m + 1] += coef
        lx_fwd[center - m + 1, center - m + 1] -= coef
        lx_fwd[center - m + 1, center + m] -= coef

    return lx_fwd.T.copy(), lx_bwd.T.copy()


def _build_Lz_kernels(half_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Lz_fwd, Lz_bwd)`` kernels before spacing scaling."""

    coes = staggered_grid_coes(half_order)
    size = 2 * half_order + 1
    center = half_order
    lz_bwd = np.zeros((size, size), dtype=np.float32)
    lz_fwd = np.zeros((size, size), dtype=np.float32)

    for m, coef in enumerate(coes, start=1):
        # Backward: NE - NW diagonal difference.
        lz_bwd[center + m - 1, center + m - 1] += coef
        lz_bwd[center + m - 1, center - m] -= coef
        lz_bwd[center - m, center - m] -= coef
        lz_bwd[center - m, center + m - 1] += coef

        # Forward: same pattern shifted by (+1, +1).
        lz_fwd[center + m, center + m] += coef
        lz_fwd[center + m, center - m + 1] -= coef
        lz_fwd[center - m + 1, center - m + 1] -= coef
        lz_fwd[center - m + 1, center + m] += coef

    return lz_fwd.T.copy(), lz_bwd.T.copy()


class RSGDerivative:
    """2D rotated staggered-grid fused derivative operator."""

    def __init__(self, spatial_order: int = 4, device: str = "cpu", backend: str = "torch", ndim: int = 2):
        if ndim != 2:
            raise ValueError("RSGDerivative currently supports 2D only.")
        if spatial_order % 2 != 0:
            raise ValueError("spatial_order must be even.")

        self.so = int(spatial_order)
        self.L = self.so // 2
        self.coes = staggered_grid_coes(self.L)
        self.backend = backend
        self.device = device
        self.ndim = ndim

        lx_fwd, lx_bwd = _build_Lx_kernels(self.L)
        lz_fwd, lz_bwd = _build_Lz_kernels(self.L)
        self._kx_fwd = lx_fwd[None, None, ...]
        self._kx_bwd = lx_bwd[None, None, ...]
        self._kz_fwd = lz_fwd[None, None, ...]
        self._kz_bwd = lz_bwd[None, None, ...]

        self._inv_2dx = 0.5
        self._inv_2dz = 0.5

    def set_spacing(self, spacing):
        if np.isscalar(spacing):
            dz = dx = float(spacing)
        else:
            values = tuple(float(v) for v in spacing)
            if len(values) != 2:
                raise ValueError(f"RSGDerivative expects 2D spacing, got {len(values)} values.")
            dz, dx = values
        self._inv_2dx = 1.0 / (2.0 * dx)
        self._inv_2dz = 1.0 / (2.0 * dz)

    def to_backend(self, to: Callable, *args, **kwargs):
        self._kx_fwd = to(self._kx_fwd, self.backend, self.device)
        self._kx_bwd = to(self._kx_bwd, self.backend, self.device)
        self._kz_fwd = to(self._kz_fwd, self.backend, self.device)
        self._kz_bwd = to(self._kz_bwd, self.backend, self.device)

    @staticmethod
    def _to_nchw_torch(u):
        if u.ndim == 2:
            return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
        if u.ndim == 3:
            return u.unsqueeze(1), lambda x: x.squeeze(1)
        if u.ndim == 4:
            return u, lambda x: x
        raise ValueError(f"Expected 2D/3D/4D input, got shape {tuple(u.shape)}.")

    @staticmethod
    def _to_nchw_jax(u):
        if u.ndim == 2:
            return u[None, None, ...], lambda x: x[0, 0]
        if u.ndim == 3:
            return u[:, None, ...], lambda x: x[:, 0]
        if u.ndim == 4:
            return u, lambda x: x
        raise ValueError(f"Expected 2D/3D/4D input, got shape {u.shape}.")

    def _conv2d_torch(self, u, kernel):
        import torch.nn.functional as F

        u_nchw, restore = self._to_nchw_torch(u)
        out = F.conv2d(u_nchw, kernel, padding=self.L)
        return restore(out)

    def _conv2d_jax(self, u, kernel):
        from jax import lax

        u_nchw, restore = self._to_nchw_jax(u)
        out = lax.conv_general_dilated(
            u_nchw,
            kernel,
            window_strides=(1, 1),
            padding=[(self.L, self.L), (self.L, self.L)],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return restore(out)

    def _conv2d(self, u, kernel):
        if self.backend == "torch":
            return self._conv2d_torch(u, kernel)
        if self.backend == "jax":
            return self._conv2d_jax(u, kernel)
        raise ValueError(f"Unsupported backend {self.backend!r}.")

    def lx_fwd(self, u):
        return self._conv2d(u, self._kx_fwd) * self._inv_2dx

    def lx_bwd(self, u):
        return self._conv2d(u, self._kx_bwd) * self._inv_2dx

    def lz_fwd(self, u):
        return self._conv2d(u, self._kz_fwd) * self._inv_2dz

    def lz_bwd(self, u):
        return self._conv2d(u, self._kz_bwd) * self._inv_2dz

    # Backward-compatible aliases used by older tests/examples.
    def dx_fwd(self, u):
        return self.lx_fwd(u)

    def dx_bwd(self, u):
        return self.lx_bwd(u)

    def dz_fwd(self, u):
        return self.lz_fwd(u)

    def dz_bwd(self, u):
        return self.lz_bwd(u)

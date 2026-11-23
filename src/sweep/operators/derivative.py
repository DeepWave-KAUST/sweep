"""Finite-difference derivatives of any order along one axis (torch).

:class:`Derivative` is the general form of the two derivatives the equations
already use: the second derivative of ``separable_d2_2d/3d`` (order 2, centred)
and the staggered first derivative of :class:`~sweep.operators.StaggeredDerivative`
(order 1, ``stagger="forward"/"backward"``).  Any order, any even accuracy, built
from Fornberg weights (:func:`sweep.scalars.fd_stencil`).

Where the result lands: a centred stencil returns the derivative on the input's
own grid points.  An odd-order derivative may instead be staggered: it moves the
result half a cell along the axis (``"forward"``: to ``i + 1/2``), which keeps the
stencil compact and leaves no null mode.  A centred odd-order stencil on a single
grid (``Derivative(1)``, ``Derivative(3)``) spans every other point next to the
centre and returns zero for the sawtooth ``(-1)**i``: it suits a field whose
derivative has to land on its own points (``u * u.dx``, ``u.dxxx`` in the
equation of ``u``), with that odd-even decoupling to watch.

Mixed derivatives compose: ``d.dx(dz(u))``.  The operator is linear, with zeros
taken beyond the grid, like the other stencils; no absorbing layer is applied.
Under ``torch.compile`` it runs as fused shifted slices, uncompiled as a conv
whose backward keeps only the kernel (``sweep.operators.torch``).
"""
from __future__ import annotations

import torch

from sweep.scalars import fd_stencil

_AXES = {2: ("z", "x"), 3: ("z", "y", "x")}


class Derivative:
    """The ``order``-th derivative along one axis, ``O(h**accuracy)`` accurate.

    Args:
        order: Derivative order, ``>= 1``.
        accuracy: Even accuracy order of the stencil (the equations'
            ``spatial_order``).  The stencil has ``accuracy + order - 1``
            points for even orders and staggered ones, ``accuracy + order``
            for centred odd orders.
        ndim: 2 (axes ``z, x``) or 3 (``z, y, x``).
        stagger: ``None`` (centred), or ``"forward"`` / ``"backward"`` for an
            odd order: the result half a cell after / before each point.
        device, dtype: Of the kernels (float32 by default); ``to()`` moves them.

    Call it as ``d(u, "x", h)`` or ``d.dx(u, h)``, ``u`` of shape
    ``(B, 1, nz, nx)`` (2-D) or ``(B, 1, nz, ny, nx)`` (3-D).  ``h`` is the
    spacing along that axis (a float or a 0-d tensor); without it the result
    is per grid spacing.
    """

    def __init__(self, order: int, accuracy: int = 4, ndim: int = 2, stagger=None,
                 device="cpu", dtype=torch.float32):
        if ndim not in _AXES:
            raise ValueError(f"ndim must be 2 or 3, got {ndim!r}")
        self.order, self.accuracy, self.ndim, self.stagger = int(order), int(accuracy), ndim, stagger
        self.shifts, self.weights = fd_stencil(order, accuracy, stagger)
        self.halo = int(abs(self.shifts).max())
        taps = torch.zeros(2 * self.halo + 1, dtype=torch.float64)
        taps[torch.as_tensor(self.shifts + self.halo)] = torch.as_tensor(self.weights)
        self._taps = taps
        self.kernels = {}
        self.to(device, dtype)

    def to(self, device=None, dtype=None):
        """Move the kernels; returns ``self``."""
        from sweep.operators.torch import register_stencil

        ref = next(iter(self.kernels.values()), None)
        device = device if device is not None else (ref.device if ref is not None else "cpu")
        dtype = dtype if dtype is not None else (ref.dtype if ref is not None else torch.float32)
        axes = _AXES[self.ndim]
        k1d = self._taps.to(device=device, dtype=dtype)
        self.kernels = {}
        for i, ax in enumerate(axes):
            shape = [1, 1] + [1] * self.ndim
            shape[2 + i] = k1d.numel()
            self.kernels[ax] = register_stencil(k1d.view(shape).contiguous())
        return self

    def __call__(self, u, axis, h=None):
        from sweep.operators.torch import apply_kernels, apply_kernels_3d

        if axis not in self.kernels:
            raise ValueError(f"axis must be one of {tuple(self.kernels)} in {self.ndim}-D, got {axis!r}")
        if u.ndim != self.ndim + 2:
            raise ValueError(f"expected u of shape (B, 1, {'nz, ny, nx' if self.ndim == 3 else 'nz, nx'}), "
                             f"got {tuple(u.shape)}")
        out = (apply_kernels_3d if self.ndim == 3 else apply_kernels)(u, self.kernels[axis])
        if h is None:
            return out
        return out / h ** self.order if self.order > 1 else out / h

    def dz(self, u, h=None):
        return self(u, "z", h)

    def dy(self, u, h=None):
        return self(u, "y", h)

    def dx(self, u, h=None):
        return self(u, "x", h)

    def __repr__(self):
        s = "" if self.stagger is None else f", stagger={self.stagger!r}"
        return f"Derivative(order={self.order}, accuracy={self.accuracy}, ndim={self.ndim}{s})"

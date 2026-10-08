import math

import numpy as np
import torch
from sweep.operators.torch import _is_compiling
from sweep.sources.base import SourceBase


def _point_values(wavelet, npoints, gather, weight):
    values = wavelet.reshape(-1)
    values = values.expand(npoints) if gather is None else values.index_select(0, gather)
    return values if weight is None else values * weight


def scatter_source(field, wavelet, index, cells, gather=None, weight=None):
    """``field`` plus ``weight * wavelet[gather]`` scattered onto its cells
    (out of place): ``index`` flat, ``cells`` the same as per-axis indices.

    A pure function of its tensor arguments, so the eager propagator fuses it
    into its ``torch.compile``'d step.  ``SourceTorch.step_args`` builds the
    trailing arguments.
    """
    values = _point_values(wavelet, index.shape[0], gather, weight)
    if _is_compiling():
        # Indexing the field's own axes keeps its layout.  The flat form hands
        # back a contiguous view, and when the step's layout differs only in a
        # size-1 stride Inductor still pays a full-grid copy to produce it.
        return field.index_put(cells, values, accumulate=True)
    # Uncompiled, index_put(accumulate=True) sorts its indices, ~0.25 ms a
    # call; index_add does not.
    return field.reshape(-1).index_add(0, index, values).view_as(field)


class SourceTorch(SourceBase, torch.nn.Module):
    def __init__(self, coords, shape, dev, source_encoding=False, adj=False,
                 spread_kernel=None):
        """Source class for the wave equation

        A step adds the wavelet sample to a handful of cells of the
        ``(B, 1, *spatial)`` wavefield, so injection is a scatter onto those
        cells -- not a whole-grid ``mask * wavelet`` pass, which cost two
        full-grid kernels per step for a few non-zero cells.

        Args:
            coords (torch.Tensor): Source coordinates, ``(B, ndim)`` (one
                source per shot) or ``(B, nsrc, ndim)``, in ``(x, [y,] z)``
                order.
            shape (tuple): Wavefield shape ``(B, 1, *spatial)``.
            source_encoding (bool): Encoded super-shot: every source goes
                into batch 0.
            adj (bool): Adjoint modelling. The propagator reverses the time
                index; the injection itself is the same.
            spread_kernel (array-like, optional): Odd square 2-D stencil (e.g.
                a 3x3 binomial) the injection is spread over. Equations whose
                collocated stencils have a checkerboard null space (the
                rotated staggered grid) declare it via
                ``equation.source_receiver_stencil`` so a point source does
                not excite the spurious mode. ``None`` keeps the plain
                single-cell injection. Spread cells off the grid are dropped,
                as a zero-padded convolution drops them.

        A step's wavelet sample has 1 value (shared by every source), ``B``
        (one per shot) or ``B * nsrc`` (one per source).
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        self.se = source_encoding
        self.adj = adj
        self.coords = coords

        # The cell lists are data-dependent (spread cells off the grid are
        # dropped), so they are built on the host, once per forward call.
        c = torch.as_tensor(coords).detach().to("cpu", torch.long)
        if c.ndim == 2:
            c = c.unsqueeze(1)
        if c.ndim != 3:
            raise ValueError(f"source coords must be (B, ndim) or (B, nsrc, ndim); got {tuple(c.shape)}")
        batch, nsrc, ndim = c.shape
        spatial = tuple(int(n) for n in shape[2:])
        if ndim != len(spatial):
            raise ValueError(f"source coords have {ndim} components for a {len(spatial)}-D wavefield")
        if not source_encoding and batch != shape[0]:
            raise ValueError(f"source coords have {batch} shot(s) for a wavefield batch of {shape[0]}")
        self.batch, self.nsrc = batch, nsrc
        cells = torch.flip(c, [-1]).reshape(-1, ndim).numpy()          # (B*nsrc, ndim), (z, [y,] x)
        upper = np.array(spatial)
        outside = ((cells < 0) | (cells >= upper)).any(axis=1)
        if outside.any():
            raise ValueError(
                f"source at {tuple(int(v) for v in cells[outside][0][::-1])} lies outside the "
                f"{spatial[::-1]} (x, [y,] z) runtime grid")

        offsets, weights = np.zeros((1, ndim), np.int64), None
        if spread_kernel is not None:
            if ndim != 2:
                raise NotImplementedError("source spread_kernel is only supported for 2-D wavefields")
            k = np.asarray(torch.as_tensor(spread_kernel).detach().cpu(), dtype=np.float32)
            if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2 == 0:
                raise ValueError(f"source spread_kernel must be an odd square 2-D stencil; got {k.shape}")
            half = k.shape[0] // 2
            # A delta cross-correlated with k (what conv2d computes) puts
            # k[half - dz, half - dx] at offset (dz, dx): the kernel, flipped.
            taps = [((dz, dx), k[half - dz, half - dx])
                    for dz in range(-half, half + 1) for dx in range(-half, half + 1)
                    if k[half - dz, half - dx] != 0.0]
            offsets = np.array([o for o, _ in taps], np.int64)
            weights = np.array([w for _, w in taps], np.float32)

        # One point per (source, offset); ``slot`` is its source's position in
        # the flattened (B, nsrc) layout.
        pts = cells[:, None, :] + offsets[None, :, :]                   # (S, K, ndim)
        on_grid = ((pts >= 0) & (pts < upper)).all(axis=-1)
        slot = np.broadcast_to(np.arange(len(cells))[:, None], on_grid.shape)[on_grid]
        pts = pts[on_grid]
        shot = np.zeros_like(slot) if source_encoding else slot // nsrc
        strides = np.cumprod((1,) + spatial[:0:-1])[::-1]               # row-major cell strides
        self.index = torch.as_tensor(shot * int(np.prod(spatial)) + pts @ strides, device=dev)
        self.cells = tuple(torch.as_tensor(np.ascontiguousarray(a), device=dev)
                           for a in (shot, np.zeros_like(shot), *pts.T))
        self.weight = None
        if weights is not None:
            self.weight = torch.as_tensor(np.broadcast_to(weights, on_grid.shape)[on_grid], device=dev)

        # Which value of a step's wavelet sample feeds each point, per sample
        # size; None when it is the sample itself, in order, or a scalar.
        def gather(g):
            return None if np.array_equal(g, np.arange(len(g))) else torch.as_tensor(g, device=dev)
        self._gathers = {batch * nsrc: gather(slot), batch: gather(slot // nsrc)}
        self._gathers[1] = None

    def _gather(self, n):
        if n not in self._gathers:
            raise ValueError(
                f"Wavelet time slice has {n} values, expected 1, {self.batch}, or "
                f"{self.batch * self.nsrc} for {self.batch} batch(es) and {self.nsrc} source(s).")
        return self._gathers[n]

    def step_args(self, wavelet):
        """``(wavelet, index, cells, gather, weight)``: the arguments of
        ``scatter_source`` for one step's wavelet sample."""
        return wavelet, self.index, self.cells, self._gather(wavelet.numel()), self.weight

    def forward(self, wavefield, wavelet):
        """``wavefield`` with this step's sources added (out of place)."""
        return scatter_source(wavefield, *self.step_args(wavelet))

    @torch.no_grad()
    def add_(self, wavefield, wavelet):
        """Add this step's sources to a contiguous ``wavefield`` in place,
        outside autograd -- the boundary-saving forward and reconstruction,
        whose fields are fresh -- skipping the copy ``forward`` makes."""
        values = _point_values(wavelet, self.index.shape[0], self._gather(wavelet.numel()), self.weight)
        wavefield.view(-1).index_add_(0, self.index, values)
        return wavefield

    def value_grad(self, grad_field, shape):
        """The adjoint of the injection with respect to the wavelet sample:
        the gradient of a sample of ``shape`` given that of the field it was
        added to."""
        n = math.prod(shape)
        gather = self._gather(n)
        g = grad_field.reshape(-1).index_select(0, self.index)
        if self.weight is not None:
            g = g * self.weight
        if gather is not None:
            g = g.new_zeros(n).index_add_(0, gather, g)
        elif n == 1:
            g = g.sum()
        return g.reshape(shape)

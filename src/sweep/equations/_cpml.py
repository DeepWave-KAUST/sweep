"""CPML memory-variable updates, on the full grid or on the absorbing strips only.

Every CPML equation runs one of two updates, written out once here.

:func:`cpml_update`, staggered grid (``cpmls``), one memory field per derivative::

    m = a * m + b * d
    d = d + m

:func:`cpml_axis_update`, second order (``cpmlr``), two memory fields per axis::

    tmp   = ((1 + b) * lap_axis + dbd * du) + grad_op(a * psi, h, axis, kernels=kernels)
    w_sum += (1 + b) * tmp + a * zeta
    psi_n =  b * du  + a * psi
    zeta  =  b * tmp + a * zeta

**The expression trees are reproduced exactly, not tidied.** The acceptance
criterion for gathering the equations' copies here was bit-exact output, and
floating-point addition is not associative -- ``(1 + b) * lap + dbd * du`` and
``dbd * du + (1 + b) * lap`` are different numbers. The caller still owns the
accumulation (``w_sum += contrib``) so that the order in which axes are summed
stays where it was.

**Strips.** Away from the absorbing layers the profiles are zero, the memory
fields stay zero and both updates hand the derivative back unchanged, so a
memory field only needs to cover the absorbing strips. With
``EagerOptions(cpml_strips=True)`` the eager propagator keeps every field whose
spec names a ``cpml_axis`` as a :class:`Strips` -- its two strips along that
axis, laid out by :class:`StripLayout` -- and both updates then run on the
strips only. The step code is the same in either layout. Its ``src`` argument
says how the derivative was taken, so that the compiled step can retake the
derivative on the strips from a window of the field: the full-grid derivative is
then read by the interior only and stays fused into the step's main kernel.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from sweep.operators.torch import _compiled_taps, _resolve_spacing_for_axis


# ---- the two updates ---------------------------------------------------------

def cpml_update(d, m, a, b, src=None):
    """Staggered-grid CPML on the derivative ``d``: returns ``(d + m', m')`` with
    ``m' = a * m + b * d``.

    ``src = (field, kernel, h)``, meaning ``d = apply_kernels(field, kernel) / h``,
    is used only when ``m`` is :class:`Strips`.
    """
    if not isinstance(m, Strips):
        m = a * m + b * d
        return d + m, m
    if not (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)) and a == 0 and b == 0:
        # Scalar zero profiles (a lossless interior sub-step): nothing absorbs.
        return d, Strips(torch.zeros_like(m.lo), torch.zeros_like(m.hi), *m[2:])

    def side(start, width, keep, lo, m_s):
        d_s = _strip_derivative(d, src, m.axis, start, width)
        m_s = a.narrow(m.axis, start, width) * m_s + b.narrow(m.axis, start, width) * d_s
        return d_s + m_s, m_s

    return _on_strips(d, (m,), side)


def cpml_axis_update(lap_axis, du, psi, zeta, a, b, dbd, h, axis, grad_op,
                     kernels=None, src=None):
    """One axis's second-order CPML contribution and its two memory-variable updates.

    Returns ``(contrib, psi_next, zeta_next)``. The caller adds ``contrib`` to
    its own accumulator; ``psi_next`` goes to a SEPARATE buffer from ``psi``
    (the forward reads neighbours' psi before writing, so an in-place update
    would be a read-after-write race). ``kernels=None`` is what
    ``operators.torch.gradient`` already defaults to.

    ``src = (u, laplace_kernels, spacings)`` -- ``lap_axis`` and ``du`` are
    derivatives of ``u``; kernels and spacings per axis, in axis order -- is used
    only when ``psi`` and ``zeta`` are :class:`Strips`.
    """
    if not isinstance(psi, Strips):
        g = grad_op(a * psi, h, axis, kernels=kernels)
        return _axis_terms(lap_axis, du, g, a, b, dbd, psi, zeta)
    kernel = None if kernels is None else kernels[axis]
    r = 1 if kernel is None else _halo(kernel)[axis]          # the gradient's reach
    lap_taps = None                                            # compiling: retake lap and du from u
    if src is not None and isinstance(src[1], tuple):
        u, laplace_kernels, spacings = src
        lap_taps = _compiled_taps(laplace_kernels[axis])
        lap_taps = lap_taps if lap_taps is not None and _along(lap_taps, axis) else None

    def side(start, width, keep, lo, psi_s, zeta_s):
        a_s, b_s, db_s = (p.narrow(axis, start, width) for p in (a, b, dbd))
        if lap_taps is None:
            lap_s, du_s = lap_axis.narrow(axis, start, width), du.narrow(axis, start, width)
        else:
            # Both from one window of u, so the backward reaches u once; the
            # full-grid derivatives are then read by the interior only.
            m = max(r, _reach(lap_taps, axis))
            win = _window(u, axis, start, width, m)
            lap_s = _taps(win, lap_taps, axis, m, width) / (spacings[axis] * spacings[axis])
            du_s = _strip_gradient(win, m, width, h, axis, grad_op, kernels, keep, lo)
        g = _strip_gradient(_window(a_s * psi_s, axis, 0, width, r), r, width, h, axis, grad_op, kernels, keep, lo)
        return _axis_terms(lap_s, du_s, g, a_s, b_s, db_s, psi_s, zeta_s)

    return _on_strips(lap_axis, (psi, zeta), side)


def _axis_terms(lap_axis, du, g, a, b, dbd, psi, zeta):
    tmp = ((1 + b) * lap_axis + dbd * du) + g
    contrib = (1 + b) * tmp + a * zeta
    psi_next = b * du + a * psi
    zeta_next = b * tmp + a * zeta
    return contrib, psi_next, zeta_next


# ---- strip-held memory fields ------------------------------------------------

class Strips(NamedTuple):
    """A memory field kept as its two strips along ``axis``: ``lo`` holds the
    first ``widths[0]`` cells of the axis, ``hi`` the last ``widths[1]`` (either
    may be empty). ``keep[s]`` holds strip ``s``'s grid-edge masks for the
    gradient (second order only): the cells it keeps, keyed by its kernel's
    half-widths, or ``"edge"`` for ``torch.gradient`` (all but its one-sided
    cell). Built once, outside the step: a mask built inside the compiled step
    is saved for its backward, a kernel each."""
    lo: torch.Tensor
    hi: torch.Tensor
    axis: int
    widths: tuple
    keep: tuple


_AXES = {2: {"z": -2, "x": -1}, 3: {"z": -3, "y": -2, "x": -1}}


class StripLayout:
    """The eager state of an equation with strip-held memory fields: one tensor
    per wavefield, two (low strip, high strip) for a field whose spec names a
    ``cpml_axis``. The eager propagator builds it, sizes it with
    :meth:`size_strips` once the CPML profiles exist, and hands the step its
    fields through :meth:`group` / :meth:`flatten`."""

    def __init__(self, equation):
        names = _AXES[equation.ndim]
        specs = {s.name: s for s in equation.field_specs}
        self.ndim = equation.ndim
        self.axes = [names.get(getattr(specs.get(f), "cpml_axis", ""), None) for f in equation.wavefields]
        self.strips = {}                    # axis -> (widths, keep)
        self._masks = {}                    # built once per strip and gradient

    def __len__(self):
        return sum(1 if ax is None else 2 for ax in self.axes)

    def slots(self, field):
        """Positions of wavefield ``field`` in the eager state."""
        j = sum(1 if ax is None else 2 for ax in self.axes[:field])
        return [j] if self.axes[field] is None else [j, j + 1]

    def size_strips(self, equation, shape, device):
        """Size each axis's strips: the cells from either end where any of the
        axis's CPML profiles (``equation.b``) is non-zero. Second-order profiles
        (``a, b, db`` per axis) also take the cells that ``grad(a psi)`` reaches
        past a non-zero ``a``, and get the gradient's grid-edge masks; staggered
        ones (``a, b, ah, bh``) need neither."""
        profiles = equation.b
        per_axis = len(profiles) // self.ndim
        second_order = per_axis == 3
        grad_kernels = getattr(equation, "grad_kernels", None) or {}
        for i, axis in enumerate(range(-self.ndim, 0)):
            live = list((torch.stack([p.detach().reshape(-1) for p in profiles[per_axis * i:per_axis * (i + 1)]]) != 0).cpu())
            if second_order:
                live[0] = _dilate(live[0], equation.so // 2)
            widths = _extent(torch.stack(live).any(0))
            n, keep = shape[axis], []
            for start, width in ((0, widths[0]), (n - widths[1], widths[1])):
                masks = {}
                if second_order and width:
                    edge = [0] * self.ndim
                    edge[axis] = 1
                    masks["edge"] = self._mask(shape, axis, start, width, tuple(edge), device)
                    if axis in grad_kernels:
                        halo = _halo(grad_kernels[axis])
                        masks[halo] = self._mask(shape, axis, start, width, halo, device)
                keep.append(masks)
            self.strips[axis] = (widths, tuple(keep))

    def _mask(self, *key):
        if key not in self._masks:
            self._masks[key] = _strip_mask(*key)
        return self._masks[key]

    def shapes(self, shape):
        """Each eager tensor's shape, for wavefields of ``shape``."""
        out = []
        for ax in self.axes:
            if ax is None:
                out.append(tuple(shape))
                continue
            for w in self.strips[ax][0]:
                s = list(shape)
                s[ax] = w
                out.append(tuple(s))
        return out

    def group(self, flat):
        """The eager state as the step takes it: strip-held fields as :class:`Strips`."""
        out, j = [], 0
        for ax in self.axes:
            if ax is None:
                out.append(flat[j])
                j += 1
            else:
                out.append(Strips(flat[j], flat[j + 1], ax, *self.strips[ax]))
                j += 2
        return out

    @staticmethod
    def flatten(fields):
        out = []
        for f in fields:
            out.extend((f.lo, f.hi) if isinstance(f, Strips) else (f,))
        return out


def _dilate(mask, r):
    out = mask.clone()
    for s in range(1, r + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _extent(live):
    """``(lo, hi)``: cells from each end of the axis up to its last live cell."""
    n = live.numel()
    idx = torch.arange(n)
    lo_cells, hi_cells = idx[live & (idx < n // 2)], idx[live & (idx >= n // 2)]
    return (int(lo_cells.max()) + 1 if lo_cells.numel() else 0,
            n - int(hi_cells.min()) if hi_cells.numel() else 0)


def _halo(kernel):
    """The cells a kernel gradient zeroes at each grid edge, per spatial dim."""
    return tuple(k // 2 for k in kernel.shape[2:])


def _strip_mask(shape, axis, start, width, halo, device):
    """The cells of the strip ``[start, start + width)`` along ``axis`` that lie at
    least ``halo[d]`` cells from the grid edges along every spatial dim ``d``."""
    nd = len(shape)
    keep = None
    for d, p in enumerate(halo):
        if p <= 0:
            continue
        along = d == nd + axis
        n = width if along else shape[d]
        idx = torch.arange(n, device=device) + (start if along else 0)
        m = ((idx >= p) & (idx < shape[d] - p)).view([n if e == d else 1 for e in range(nd)])
        keep = m if keep is None else keep & m
    return keep


# ---- one update on the strips ------------------------------------------------

def _on_strips(d, mems, side):
    """Run ``side(start, width, keep, lo, *strips) -> (part, *new strips)`` on both
    strips of the memory fields ``mems`` (all along one axis). Returns ``d`` with
    the parts in its strips, then the updated memory fields."""
    first = mems[0]
    axis, widths = first.axis, first.widths
    n = d.shape[axis]
    parts, new = [], [[m.lo, m.hi] for m in mems]
    for k, (start, width) in enumerate(((0, widths[0]), (n - widths[1], widths[1]))):
        if width:
            part, *updated = side(start, width, first.keep[k], k == 0, *(m[k] for m in mems))
            parts.append(part)
            for new_m, u in zip(new, updated):
                new_m[k] = u
    out = _assemble(d, parts, axis, widths) if parts else d
    # Strips(...), not _replace: Dynamo cannot rebuild a NamedTuple made by
    # _replace in a frame resumed after a graph break.
    return (out, *(Strips(lo, hi, *m[2:]) for m, (lo, hi) in zip(mems, new)))


def _assemble(d, parts, axis, widths):
    """``d`` with its strips replaced by ``parts`` (low strip first): each part
    zero-padded to the axis, one ``where`` -- no cut or concatenation."""
    n = d.shape[axis]
    lo, hi = widths
    strips = None
    if lo:
        strips = _pad_axis(parts[0], axis, 0, n - lo)
    if hi:
        pad_hi = _pad_axis(parts[-1], axis, n - hi, 0)
        strips = pad_hi if strips is None else strips + pad_hi
    idx = torch.arange(n, device=d.device)
    shape = [1] * d.ndim
    shape[axis] = n
    return torch.where(((idx >= lo) & (idx < n - hi)).view(shape), d, strips)


def _strip_derivative(d, src, axis, start, width):
    """``d`` on the cells ``[start, start + width)`` along ``axis``. Compiling, it
    is retaken from a window of the field (``src = (field, kernel, scale)``,
    ``d = apply_kernels(field, kernel) / scale``), so that the full-grid ``d`` is
    read by the interior only; otherwise it is cut from ``d``."""
    if src is not None:
        field, kernel, scale = src
        taps = _compiled_taps(kernel)
        if taps is not None and _along(taps, axis):
            m = _reach(taps, axis)
            out = _taps(_window(field, axis, start, width, m), taps, axis, m, width)
            return out if scale is None else out / scale
    return d.narrow(axis, start, width)


def _strip_gradient(win, m, width, h, axis, grad_op, kernels, keep, lo):
    """``grad_op(f, h, axis, kernels=kernels)`` on a strip at the grid edge (the
    low one if ``lo``), from ``win``: the strip's ``width`` cells of ``f`` with
    ``m`` cells of margin either side, zero past the grid. The grid-edge rules --
    the halo the kernel gradient zeroes, ``torch.gradient``'s one-sided cell --
    come from the strip's masks ``keep``."""
    h_axis = _resolve_spacing_for_axis(h, axis, win.ndim)
    kernel = None if kernels is None else kernels[axis]
    if kernel is None:
        # torch.gradient written out (its cat would cost kernels when compiled):
        # ((f[i+1] - f[i-1]) / h) / 2, one-sided at the grid edge.
        ahead, here, behind = (win.narrow(axis, m + k, width) for k in (1, 0, -1))
        central = ((ahead - behind) / h_axis) / 2
        one_sided = ((ahead - here) if lo else (here - behind)) / h_axis
        return torch.where(keep["edge"], central, one_sided)
    taps = _compiled_taps(kernel)
    if taps is not None and _along(taps, axis):
        out = _taps(win, taps, axis, m, width) / h_axis
    else:
        out = grad_op(win, h, axis, kernels=kernels).narrow(axis, m, width)
    return torch.where(keep[_halo(kernel)], out, torch.zeros((), dtype=out.dtype, device=out.device))


def _window(f, axis, start, width, m):
    """Cells ``[start - m, start + width + m)`` of ``f`` along ``axis``, zero past
    the ends of ``f``."""
    n = f.shape[axis]
    first, last = max(start - m, 0), min(start + width + m, n)
    if last - first < n:                            # (a no-op narrow would cost a kernel in the backward)
        f = f.narrow(axis, first, last - first)
    return _pad_axis(f, axis, first - (start - m), start + width + m - last)


def _taps(win, taps, axis, m, width):
    """The stencil ``taps`` on the ``width`` cells of ``win`` past its margin
    ``m``, summed in ``_apply_taps``'s order."""
    d = len(taps[0][0]) + axis
    out = None
    for off, c in taps:
        term = win.narrow(axis, m + off[d], width) * c
        out = term if out is None else out + term
    return out


def _reach(taps, axis):
    d = len(taps[0][0]) + axis
    return max(abs(o[d]) for o, _ in taps)


def _along(taps, axis):
    d = len(taps[0][0]) + axis
    return all(o[e] == 0 for o, _ in taps for e in range(len(o)) if e != d)


def _pad_axis(x, axis, before, after):
    """Zero-pad ``x`` along ``axis`` (a negative index)."""
    spec = [0, 0] * (-axis)
    spec[-2], spec[-1] = before, after
    return F.pad(x, spec)

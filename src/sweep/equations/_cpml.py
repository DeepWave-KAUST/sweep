"""The one CPML axis update every second-order acoustic equation re-derives.

Nine equation files carried 27 copies of the same four lines, one per axis:

    tmp   = ((1 + b) * lap_axis + dbd * du) + grad_op(a * psi, h, axis)
    w_sum += (1 + b) * tmp + a * zeta
    psi_n =  b * du  + a * psi
    zeta  =  b * tmp + a * zeta

Transcribing that correctly is a prerequisite for adding an equation, and a
single misplaced ``1 +`` produces a wave that looks plausible and absorbs wrong.

**The expression tree is reproduced exactly, not tidied.** Every parenthesis and
every operand order is the one the copies used, because the acceptance criterion
for this refactor is bit-exact output and floating-point addition is not
associative -- ``(1 + b) * lap + dbd * du`` and ``dbd * du + (1 + b) * lap`` are
different numbers. The caller still owns the accumulation (``w_sum += contrib``)
so that the ORDER in which axes are summed stays where it was.

``kernels=None`` is what ``operators.torch.gradient`` already defaults to, so
passing it explicitly is identical to the copies that omitted it.
"""
from __future__ import annotations


def cpml_axis_update(lap_axis, du, psi, zeta, a, b, dbd, h, axis, grad_op,
                     kernels=None):
    """One axis's CPML contribution and its two memory-variable updates.

    Returns ``(contrib, psi_next, zeta_next)``. The caller adds ``contrib`` to
    its own accumulator; ``psi_next`` goes to a SEPARATE buffer from ``psi``
    (the forward reads neighbours' psi before writing, so an in-place update
    would be a read-after-write race).
    """
    tmp = ((1 + b) * lap_axis + dbd * du) + grad_op(a * psi, h, axis, kernels=kernels)
    contrib = (1 + b) * tmp + a * zeta
    psi_next = b * du + a * psi
    zeta_next = b * tmp + a * zeta
    return contrib, psi_next, zeta_next

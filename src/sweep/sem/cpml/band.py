"""Which elements a C-PML profile reaches: the split the eager steps run on.

Outside the absorbing band every C-PML coefficient of an element-local profile is
an exact no-op (a stretch of 1, a recursion or accelerating coefficient of 0), so
the memory variables there stay exactly zero and the C-PML terms vanish. The
eager steps therefore run the bare stiffness on the interior elements and the
full C-PML formulation on the band elements only, and keep the memory variables
on the band rows (the ``band`` / ``band2`` field layouts) -- the same split the
CUDA element kernels make with one branch per element. On a 801x1601 grid 90% of
the elements are interior.

The split costs a second element pipeline per step -- more kernels, and a gather
and a scatter of the band rows -- and pays where the C-PML work it saves is large
next to that: on large meshes, however the step runs. Measured on 2-D elastic
(``ElasticSEM2DDeformed``), forward | forward+backward: uncompiled on an A100,
1.4x | 1.5x slower at 201x401 (4.9k interior elements), 1.1x | 1.4x slower at
401x801 (19.7k), 2.6x | 2.0x faster at 801x1601 (79k); compiled, 0.95x | 0.85x at
201x401 (RTX 6000 Ada) and 2.2x | 1.3x at 801x1601 (A100). Replayed as CUDA graphs
at 201x401 it gains 1.25x | 1.1x on this equation and loses a little on the
velocity-stress forms; on a 49^3 velocity-stress mesh, three quarters band, the
compiled split step is 1.36x slower forward+backward (A100). So ``split=None``
splits the meshes with at least ``SPLIT_MIN_INTERIOR`` interior elements, and an
unsplit equation keeps every element on "the band" (``interior`` empty): its step
is the full-mesh formulation it had before.
"""
from __future__ import annotations

import torch


def active_elements(coefficients, n_elements) -> torch.Tensor:
    """``(E,)`` bool: elements where any coefficient differs from its no-op value.

    ``coefficients`` yields ``(tensor, no_op)`` pairs, each tensor element-local
    (leading axis ``E``). Testing the values themselves, rather than a "node in
    the PML" flag, is what makes the split exact by construction: an element left
    out has exactly no-op coefficients."""
    act = None
    for t, no_op in coefficients:
        a = (t != no_op).reshape(n_elements, -1).any(1)
        act = a if act is None else act | a
    if act is None:
        raise ValueError("no C-PML coefficients to split on")
    return act


SPLIT_MIN_INTERIOR = 40000


def element_split(active, split=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(band, interior)``: the element ids of ``active`` and of the rest, both
    ascending (``band`` is the row order of the band-layout fields). Unsplit --
    ``split=False``, or ``None`` with fewer than ``SPLIT_MIN_INTERIOR`` interior
    elements -- every element is on the band and the interior is empty."""
    interior = torch.nonzero(~active, as_tuple=False).reshape(-1)
    if split is None:
        split = interior.numel() >= SPLIT_MIN_INTERIOR
    if not split:
        return (torch.arange(active.numel(), device=active.device),
                interior.new_empty(0))
    return torch.nonzero(active, as_tuple=False).reshape(-1), interior

"""Global node values from their per-element copies, deterministically.

A shared node has one copy per element that holds it. On a conforming mesh
every copy carries the same coordinate up to rounding (each element
evaluates its own map at the same physical point), so any one of them is
the node's value. The deformed mesh builders used to AVERAGE the copies
with ``index_add_``, whose scatter order on CUDA differs from run to run:
the average moved by an ulp, and with it the C-PML band edge -- two builds
of the same topography mesh disagreed on which nodes are in the layer
(measured: ``a_x`` rel 2.8e-2 on the flipped nodes, the x ring 17442 vs
17444 entries). Taking the LOWEST-numbered copy instead is a choice no
atomic order can change: a min is the same in every order.
"""

from __future__ import annotations

import torch


def first_copy_slot(ibool: torch.Tensor, n_global: int) -> torch.Tensor:
    """``(n_global,)`` flat element-node slot of each node's lowest copy."""
    idx = ibool.reshape(-1)
    slot = torch.full((n_global,), idx.numel(), dtype=torch.long, device=idx.device)
    return slot.scatter_reduce_(0, idx, torch.arange(idx.numel(), device=idx.device),
                                "amin")


def node_values(slot: torch.Tensor, *per_element: torch.Tensor):
    """Each ``(E, ngll, ...)`` per-element array read at the chosen copies."""
    return tuple(t.reshape(-1)[slot] for t in per_element)

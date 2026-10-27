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

Material fields read at nodes (``c²``, ``κ``) need one copy for a different
reason: their GRADIENT. ``node[ibool] = per_element`` has a backward that
hands the node's gradient to every copy, so the model gradient came out
multiplied by the node's share count (2-D: up to 4, 3-D: up to 8).
Gathering one fixed copy gives that copy the whole node gradient and the
others none. The HIGHEST copy is what ``index_put_`` kept on one CPU thread
(its last writer). Where the copies differ -- an element-wise material jump
-- the old scatter's choice moved with the thread count and from run to run
on CUDA (measured on a 200x150 quad4 mesh: 2411 / 5631 nodes differ at 4 / 8
threads, ~113k on CUDA); the gather is the same everywhere.
"""

from __future__ import annotations

import torch


def first_copy_slot(ibool: torch.Tensor, n_global: int) -> torch.Tensor:
    """``(n_global,)`` flat element-node slot of each node's lowest copy."""
    idx = ibool.reshape(-1)
    slot = torch.full((n_global,), idx.numel(), dtype=torch.long, device=idx.device)
    return slot.scatter_reduce_(0, idx, torch.arange(idx.numel(), device=idx.device),
                                "amin")


def last_copy_slot(ibool: torch.Tensor, n_global: int,
                   select: torch.Tensor | None = None) -> torch.Tensor:
    """``(n_global,)`` flat element-node slot of each node's highest copy,
    among the copies where the flat ``select`` mask is true; ``-1`` where a
    node has none."""
    idx = ibool.reshape(-1)
    arange = torch.arange(idx.numel(), device=idx.device)
    slot = torch.full((n_global,), -1, dtype=torch.long, device=idx.device)
    if select is not None:
        idx, arange = idx[select], arange[select]
    return slot.scatter_reduce_(0, idx, arange, "amax")


def node_field(ibool: torch.Tensor, n_global: int, per_element: torch.Tensor,
               select: torch.Tensor | None = None) -> torch.Tensor:
    """A per-element material field read at the nodes, one copy per node
    (see the module note): differentiable, with each node's gradient landing
    on exactly one copy. Nodes with no selected copy are 0."""
    slot = last_copy_slot(ibool, n_global, select)
    vals = per_element.reshape(-1)[slot.clamp(min=0)]
    if select is None:
        return vals
    return torch.where(slot >= 0, vals, torch.zeros_like(vals))


def node_values(slot: torch.Tensor, *per_element: torch.Tensor):
    """Each ``(E, ngll, ...)`` per-element array read at the chosen copies."""
    return tuple(t.reshape(-1)[slot] for t in per_element)

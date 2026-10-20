"""Per-element physical-domain flags (acoustic vs elastic) on one shared mesh.

SPECFEM2D does **not** split the mesh at a fluid–solid interface: it allocates
both the fluid potential and the solid displacement over the *entire* mesh with
one continuous global numbering (``nglob_acoustic = nglob_elastic = nglob``,
``prepare_wavefields.f90``), flags each *element* as acoustic or elastic, and
assembles each operator only over its own elements. At the interface the two
sides literally share the global node number. Nothing is "welded", because the
two domains carry **different unknowns** — the fluid's only DOF is the scalar
potential χ, so there is no fluid tangential DOF to over-constrain, and
tangential continuity is simply never imposed (= free slip, the correct
fluid–solid condition).

Our ``ibool`` is exactly that data model, so this module only has to supply:

* the per-element flags (``is_acoustic`` / ``is_elastic``, mutually exclusive
  and complete — SPECFEM validates the same two properties twice), and
* the derived node membership, plus
* :func:`masked_inv_mass`, which kills the **dead DOFs** (χ inside the solid,
  **u** inside the fluid). SPECFEM gives those a dummy unit mass
  (``invert_mass_matrix.F90``: ``where(rmass_inverse <= 0) rmass_inverse = 1``);
  since our masked assembly already makes their residual identically zero, we
  instead set ``inv_M = 0`` there, which pins them to exactly 0 forever.

Note the ``torch.where`` is **static** (keyed on the mask, not on the value of
the mass). A ``clamp(min=eps)`` or a value-dependent branch would leak a
gradient into dead DOFs and quietly pollute an FWI kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DomainMask:
    """Which elements are fluid, which are solid, and the derived node sets.

    ``node_in_acoustic`` / ``node_in_elastic`` are True for a node touched by at
    least one element of that domain, so **interface nodes are in both** — which
    is the point: they carry a live χ *and* a live **u**.
    """
    is_acoustic: torch.Tensor        # (E,) bool
    is_elastic: torch.Tensor         # (E,) bool
    node_in_acoustic: torch.Tensor   # (n_global,) bool
    node_in_elastic: torch.Tensor    # (n_global,) bool

    @classmethod
    def from_element_flags(cls, mesh, is_acoustic) -> "DomainMask":
        dev = mesh.ibool.device
        ac = torch.as_tensor(is_acoustic, dtype=torch.bool, device=dev).reshape(-1)
        if ac.numel() != mesh.n_elements:
            raise ValueError(
                f"is_acoustic has {ac.numel()} entries, mesh has "
                f"{mesh.n_elements} elements")
        el = ~ac                       # exclusive + complete by construction
        n = mesh.n_global
        nia = torch.zeros(n, dtype=torch.bool, device=dev)
        nie = torch.zeros(n, dtype=torch.bool, device=dev)
        nia[mesh.ibool[ac].reshape(-1)] = True
        nie[mesh.ibool[el].reshape(-1)] = True
        if not bool((nia | nie).all()):
            raise ValueError("some global node belongs to no element")
        return cls(is_acoustic=ac, is_elastic=el,
                   node_in_acoustic=nia, node_in_elastic=nie)

    @classmethod
    def from_vs(cls, mesh, vs_e) -> "DomainMask":
        """SPECFEM's rule: an element is acoustic **iff** its Vs is zero
        (``read_materials.f90``). Vs must be uniformly 0 or uniformly > 0 within
        an element — a mixed element is a meshing error, not something to
        average away."""
        zero = vs_e.flatten(1) == 0
        ac, any_zero = zero.all(dim=1), zero.any(dim=1)
        if bool((any_zero & ~ac).any()):
            bad = int((any_zero & ~ac).nonzero()[0])
            raise ValueError(
                f"element {bad} mixes vs==0 and vs>0 nodes; an element must be "
                "wholly acoustic or wholly elastic")
        return cls.from_element_flags(mesh, ac)

    # ---- helpers ------------------------------------------------------- #
    def elem_weight(self, which: str, like: torch.Tensor) -> torch.Tensor:
        """``(E, 1, 1)`` 0/1 weight to multiply an element-local integrand by
        **before** ``assemble`` — the whole of "domain masking"."""
        m = self.is_acoustic if which == "acoustic" else self.is_elastic
        return m.to(like.dtype).reshape(-1, 1, 1)

    @property
    def n_acoustic(self) -> int:
        return int(self.is_acoustic.sum())

    @property
    def n_elastic(self) -> int:
        return int(self.is_elastic.sum())


def masked_inv_mass(mass: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """``1/mass`` on ``alive`` nodes, exactly ``0`` on dead ones.

    The branch is **static** (on ``alive``, never on the mass value), so no
    gradient can leak into a dead DOF. The inner ``where`` only exists to keep
    ``1/0`` out of the graph.
    """
    safe = torch.where(alive, mass, torch.ones_like(mass))
    return torch.where(alive, 1.0 / safe, torch.zeros_like(mass))

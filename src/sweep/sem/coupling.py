"""Fluid–solid (acoustic–elastic) interface operators.

Two continuous conditions hold on Γ — and only two, because the fluid has no
tangential DOF (free slip):

1. normal displacement continuity   ``(1/ρ)∇χ · n = u_s · n``   (Neumann data for the fluid)
2. traction continuity              ``σ · n_s = −p n_s``        (Neumann data for the solid)

With ``n`` taken **outward from the fluid on both sides** (so ``n_s = −n``) and
``p = −χ̈``, the two surface terms both enter with a literal ``+``:

    S^ac_i = ∮_Γ (u_s · n) φ_i dΓ            → added to the fluid residual
    S^el_i = ∮_Γ p n φ_i dΓ                  → added to the solid residual

The sign asymmetry is carried entirely by ``p = −χ̈`` and by the shared
fluid-outward normal — two negatives that cancel. Anyone "fixing" one of them
breaks it.

Quadrature is **1-D GLL along the edge** (``w_1D · jac1D``), never the 2-D
weight product, and is evaluated on the **acoustic** element's metric so the two
sides use literally the same weights and normals. That makes
:meth:`FluidSolidInterface.apply_ac` and :meth:`~FluidSolidInterface.apply_el`
**exact discrete transposes** of each other — the property that makes the FWI
kernel exact, and which autograd then differentiates for free (so, unlike
SPECFEM, we never hand-code an adjoint coupling).

Interface detection exploits our structured ``(nez, nex)`` element grid: a
fluid–solid face is just a domain change between neighbours, and the two sides'
GLL rows pair up identically and in the same order. SPECFEM needs a whole
``ivalue``/``ivalue_inverse`` reversed-traversal machinery here only because its
mesher hands over unstructured CCW-numbered element pairs; we do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mesh.edges import edge_metric, orient_normal

# side of the FLUID element facing the given structured-grid neighbour
_TOWARD_DOWN, _TOWARD_UP = "bottom", "top"
_TOWARD_RIGHT, _TOWARD_LEFT = "right", "left"


@dataclass(frozen=True)
class FluidSolidInterface:
    """Per-GLL-node interface quadrature: nodes, weights and fluid-outward normal.

    Every field is ``(n_face, ngll)``. ``n_face`` counts *faces*, not elements —
    a fluid element can contribute several faces.
    """
    node_idx: torch.Tensor      # global node index of each interface GLL point
    weight: torch.Tensor        # w_1D · jac1D  (acoustic element's metric)
    nx: torch.Tensor            # unit normal, OUTWARD FROM THE FLUID
    nz: torch.Tensor
    n_global: int

    # ------------------------------------------------------------------ #
    @classmethod
    def detect(cls, mesh, domain) -> "FluidSolidInterface":
        """Find every fluid–solid face by scanning structured-grid neighbours."""
        nex, nez = mesh.nex, mesh.nez
        dev = mesh.ibool.device
        ids = torch.arange(mesh.n_elements, device=dev).reshape(nez, nex)
        ac = domain.is_acoustic.reshape(nez, nex)

        pairs = []      # (fluid_elem, solid_elem, side_of_fluid)
        # vertical neighbours: a above b
        a, b = ids[:-1, :].reshape(-1), ids[1:, :].reshape(-1)
        aa, ab = ac[:-1, :].reshape(-1), ac[1:, :].reshape(-1)
        ch = aa != ab
        if bool(ch.any()):
            pairs.append((a[ch & aa], b[ch & aa], _TOWARD_DOWN))   # fluid above
            pairs.append((b[ch & ab], a[ch & ab], _TOWARD_UP))     # fluid below
        # horizontal neighbours: a left of b
        a, b = ids[:, :-1].reshape(-1), ids[:, 1:].reshape(-1)
        aa, ab = ac[:, :-1].reshape(-1), ac[:, 1:].reshape(-1)
        ch = aa != ab
        if bool(ch.any()):
            pairs.append((a[ch & aa], b[ch & aa], _TOWARD_RIGHT))  # fluid on left
            pairs.append((b[ch & ab], a[ch & ab], _TOWARD_LEFT))   # fluid on right

        idx, w, nx, nz = [], [], [], []
        for fluid, solid, side in pairs:
            if fluid.numel() == 0:
                continue
            em = orient_normal(edge_metric(mesh, fluid, side), mesh, fluid, solid)
            # the paired solid face must see the SAME global nodes in the SAME
            # order — free on a structured grid, but assert it rather than trust it
            other = {"bottom": "top", "top": "bottom",
                     "right": "left", "left": "right"}[side]
            em_s = edge_metric(mesh, solid, other)
            if not torch.equal(em.node_idx, em_s.node_idx):
                raise ValueError("fluid/solid faces do not pair node-for-node")
            # dtype-aware conformity guard: the two sides compute jac1d through
            # different inv_J entries, so they agree only to ~1e3 ulp (fp32
            # ~1e-4); genuine non-conformity differs at O(1). A fixed 1e-10
            # rtol falsely rejects valid fp32 meshes.
            rtol = max(1e-10, 1e3 * float(torch.finfo(em.jac1d.dtype).eps))
            if not torch.allclose(em.jac1d, em_s.jac1d, rtol=rtol, atol=0):
                raise ValueError("fluid/solid faces disagree on jac1d (non-conforming)")
            idx.append(em.node_idx); w.append(em.weight)
            nx.append(em.nx); nz.append(em.nz)

        if not idx:
            ng = mesh.ngll
            empty = torch.zeros((0, ng), dtype=torch.long, device=dev)
            z = torch.zeros((0, ng), dtype=mesh.x_e.dtype, device=dev)
            return cls(empty, z, z, z, mesh.n_global)
        return cls(torch.cat(idx), torch.cat(w), torch.cat(nx), torch.cat(nz),
                   mesh.n_global)

    # ------------------------------------------------------------------ #
    @property
    def n_face(self) -> int:
        return int(self.node_idx.shape[0])

    def _scatter(self, vals, like):
        return torch.zeros_like(like).index_add(
            0, self.node_idx.reshape(-1), vals.reshape(-1))

    def apply_ac(self, u_x, u_z, like):
        """``S^ac_i = ∮_Γ (u_s · n) φ_i dΓ`` → ``(n_global,)``, for the fluid."""
        g = self.node_idx
        un = self.nx * u_x[g] + self.nz * u_z[g]
        return self._scatter(self.weight * un, like)

    def apply_el(self, p, like):
        """``S^el_i = ∮_Γ p n φ_i dΓ`` → ``(S_x, S_z)``, for the solid.

        Exact discrete transpose of :meth:`apply_ac` (same weights, same nodes,
        same normal).
        """
        pw = self.weight * p[self.node_idx]
        return self._scatter(pw * self.nx, like), self._scatter(pw * self.nz, like)


def assert_no_corner_pml_on_interface(ifc: "FluidSolidInterface", ac_profile):
    """SPECFEM2D's restriction, enforced at BUILD time instead of its run-time
    stop: a fluid–solid interface node may sit in an X-only or Z-only absorbing
    band (the plain coupling term is stable there; the cost is absorption
    quality at Γ), but **never** in a corner region where both directions are
    stretched — that configuration is unsupported even by SPECFEM
    (compute_coupling_acoustic_el.f90 makes it a fatal error).

    The region type is read off the acoustic profile's per-node damping
    (``b_x`` / ``b_z`` are nonzero exactly where that direction is stretched).
    """
    if ifc.n_face == 0:
        return
    g = ifc.node_idx.reshape(-1)
    corner = (ac_profile.b_x[g] != 0) & (ac_profile.b_z[g] != 0)
    if bool(corner.any()):
        raise ValueError(
            f"{int(corner.sum())} fluid-solid interface node(s) fall inside a "
            "CORNER absorbing region (both x and z stretched). This is "
            "unsupported (SPECFEM2D makes it a fatal error too): shrink the "
            "PML bands, move the interface, or drop the offending side.")

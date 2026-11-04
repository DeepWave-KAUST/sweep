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
from typing import Optional

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



@dataclass(frozen=True)
class InterfaceCPML:
    """The interface terms' stretching on the points of Γ inside the absorbing bands.

    The plain terms of :class:`FluidSolidInterface` are exact only outside the
    bands. Inside, each domain's C-PML carries its own stretching into Γ:

    * the solid's equation is SPECFEM2D's, scaled by ``s_x s_z``
      (:mod:`~sweep.sem.cpml.elastic`), so integrating its stress divergence
      by parts leaves ``s_z σ_x·n_x + s_x σ_z·n_z`` on Γ. With the traction
      condition ``σ ñ = −p ñ`` in the stretched frame (``ñ ∝ (s_z n_x,
      s_x n_z)``) the load becomes ``(s_z⋆p n_x, s_x⋆p n_z)``;
    * the fluid's is the unscaled double-convolution form
      (:mod:`~sweep.sem.cpml.acoustic`), whose flux enters inside the outer
      ``1/s`` of each axis, so its load becomes
      ``(1/s_x)⋆u_x n_x + (1/s_z)⋆u_z n_z``.

    On a horizontal sea floor through a side band that is ``s_x⋆p`` on the
    solid and the plain term on the fluid. SPECFEM2D scales its fluid as well
    and applies ``s_x`` (``s_z`` in a bottom band) to both loads; its solid
    side drops the convolution in a bottom band.

    With ``p = −χ̈``, ``s⋆p = p − (σ χ̇ − ασ χ + α²σ (e^{−αt}⋆χ))`` and
    ``(1/s)⋆u = u − σ (e^{−(α+σ)t}⋆u)``; ``χ̇`` is SPECFEM's corrected
    Newmark velocity ``(χ − χ_prev)/Δt + Δt/2 χ̈`` (the one the solid's own
    mass term centres its damping on), and the convolutions are SPECFEM's
    second-order recursions. Each side takes its own profile at the node:
    the solid's ``d``, ``α`` for the solid load, the fluid's for the fluid
    load. The four memories live on ``nodes`` (unique, so autograd sees one
    write per node); the loads go out per point.
    """
    nodes: torch.Tensor     # (n_u,) global node of each band interface node
    pt: torch.Tensor        # (n_p,) position in ``nodes`` of each band point
    pt_g: torch.Tensor      # (n_p,) global node of each band point
    pt_flat: torch.Tensor   # (n_p,) the point's index in the flat ifc arrays
    w_nx: torch.Tensor      # (n_p,) weight * n_x, weight * n_z
    w_nz: torch.Tensor
    solid: tuple            # per axis (x, z): (A, B, C, D, c0, c1, c2) on nodes
    fluid: tuple            # per axis (x, z): (sigma, c0, c1, c2) on nodes

    @classmethod
    def build(cls, ifc: "FluidSolidInterface", mesh, domain, cpml_ac, cpml_el,
              dt: float) -> "Optional[InterfaceCPML]":
        """``None`` when no interface point sits in a band (the plain terms
        are then the whole coupling)."""
        from .cpml.elastic import _compute_coef_convolution as conv
        if ifc.n_face == 0:
            return None
        if cpml_ac.sigma_x is None or cpml_el.d_x is None:
            raise ValueError(
                "the fluid-solid coupling needs the C-PML profiles' per-axis damping "
                "(sigma_x/alpha_x on the acoustic profile, d_x/alpha_x on the elastic "
                "one); build them with build_cpml / build_elastic_cpml")
        g_all = ifc.node_idx.reshape(-1)
        solid = domain.is_elastic
        ib = mesh.ibool[solid].reshape(-1)
        on_nodes = lambda t: torch.zeros(mesh.n_global, dtype=t.dtype, device=t.device
                                         ).index_copy(0, ib, t[solid].reshape(-1))
        el = [on_nodes(t) for t in (cpml_el.d_x, cpml_el.alpha_x, cpml_el.d_z, cpml_el.alpha_z)]
        ac = [cpml_ac.sigma_x, cpml_ac.alpha_x, cpml_ac.sigma_z, cpml_ac.alpha_z]
        live = (el[0] != 0) | (el[2] != 0) | (ac[0] != 0) | (ac[2] != 0)
        keep = torch.nonzero(live[g_all]).reshape(-1)
        if keep.numel() == 0:
            return None
        nodes, pt = torch.unique(g_all[keep], return_inverse=True)
        el = [t[nodes] for t in el]
        ac = [t[nodes] for t in ac]

        def stretch(d, a):          # s⋆p - p = -(A χ̈ + B χ + C χ_prev + D r)
            return (0.5 * dt * d, d / dt - a * d, -d / dt, a * a * d) + conv(a, dt)

        def shrink(s, a):           # (1/s)⋆u - u = -s r, r = e^{-(α+σ)t}⋆u
            return (s,) + conv(a + s, dt)

        w = ifc.weight.reshape(-1)[keep]
        return cls(nodes=nodes, pt=pt, pt_g=g_all[keep], pt_flat=keep,
                   w_nx=w * ifc.nx.reshape(-1)[keep], w_nz=w * ifc.nz.reshape(-1)[keep],
                   solid=(stretch(el[0], el[1]), stretch(el[2], el[3])),
                   fluid=(shrink(ac[0], ac[1]), shrink(ac[2], ac[3])))

    def table(self) -> torch.Tensor:
        """``(n_p, 24)``: each band point's row for the CUDA kernels
        (``sem2d_ifc_pml_ac`` in ``sem2d/kernels.cuh`` gives the layout):
        ``w n_x, w n_z``, the solid's x and z stretch ``(A, B, C, D, c0, c1,
        c2)``, the fluid's x and z ``(s, c0, c1, c2)``."""
        cols = [self.w_nx, self.w_nz]
        for part in self.solid + self.fluid:
            cols += [c[self.pt] for c in part]
        return torch.stack(cols, 1)

    @staticmethod
    def _recur(r, c0, c1, c2, now, prev, nodes):
        return c0 * r[nodes] + c1 * now[nodes] + c2 * prev[nodes]

    def solid_load(self, chi_dd, chi, chi_prev, r_x, r_z):
        """The solid load's stretching ``(Δf_x, Δf_z)`` and the two memories at
        this level, ``e^{−α_x t}⋆χ`` and ``e^{−α_z t}⋆χ`` on ``nodes``."""
        n = self.nodes
        d = []
        for (A, B, C, D, c0, c1, c2), r in zip(self.solid, (r_x, r_z)):
            rn = self._recur(r, c0, c1, c2, chi, chi_prev, n)
            d.append((-(A * chi_dd[n] + B * chi[n] + C * chi_prev[n] + D * rn), rn))
        (dp_x, rx), (dp_z, rz) = d
        f_x = torch.zeros_like(chi).index_add(0, self.pt_g, self.w_nx * dp_z[self.pt])
        f_z = torch.zeros_like(chi).index_add(0, self.pt_g, self.w_nz * dp_x[self.pt])
        return f_x, f_z, r_x.index_copy(0, n, rx), r_z.index_copy(0, n, rz)

    def fluid_load(self, u_x, u_z, u_x_prev, u_z_prev, q_x, q_z):
        """The fluid load's stretching ``Δs_ac`` and the two memories at this
        level, ``e^{−(α+σ)t}⋆u`` per axis on ``nodes``."""
        n = self.nodes
        (sx, *cx), (sz, *cz) = self.fluid
        qx = self._recur(q_x, *cx, u_x, u_x_prev, n)
        qz = self._recur(q_z, *cz, u_z, u_z_prev, n)
        un = self.w_nx * (-sx * qx)[self.pt] + self.w_nz * (-sz * qz)[self.pt]
        s = torch.zeros_like(u_x).index_add(0, self.pt_g, un)
        return s, q_x.index_copy(0, n, qx), q_z.index_copy(0, n, qz)


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


# --------------------------------------------------------------------------- #
# 3-D: Hex27 faces
# --------------------------------------------------------------------------- #
def _hex_face(mesh, elems, m, s):
    """``(node_idx, weight, normal)`` of the face ``ξ_m = s`` (``m``: 0 ξ / x, 1 η / y,
    2 γ / z; ``s = ±1``) of the Hex27 elements ``elems``, each ``(n_e, ngll**2)`` (the normal
    ``(n_e, ngll**2, 3)``). Nanson: the area element is ``|J| |∇ξ_m| dξ_a dξ_b`` and the
    outward normal ``s ∇ξ_m / |∇ξ_m|`` (``|J| > 0``), so the weight is ``w_a w_b |J| |∇ξ_m|``."""
    ng = mesh.ngll
    k = 0 if s < 0 else ng - 1
    face = [slice(None)] * 3
    face[2 - m] = k                       # ibool / metric axes are (γ, η, ξ)
    sel = (elems,) + tuple(face)
    node = mesh.ibool[sel].reshape(len(elems), -1)
    grad = mesh.inv_J[sel + (m,)].reshape(len(elems), -1, 3)       # ∂ξ_m/∂x_d
    norm = torch.linalg.vector_norm(grad, dim=-1)
    w2 = (mesh.w_ref[:, None] * mesh.w_ref[None, :]).reshape(1, -1)
    weight = w2 * mesh.det_J[sel].reshape(len(elems), -1) * norm
    return node, weight, s * grad / norm[..., None]


@dataclass(frozen=True)
class FluidSolidInterface3D:
    """:class:`FluidSolidInterface` on a Hex27 mesh: per GLL node of every fluid-solid face
    (``(n_face, ngll**2)``) the global node, the surface quadrature weight and the unit normal
    outward from the fluid, on the fluid element's metric, so that :meth:`apply_ac` and
    :meth:`apply_el` are exact transposes."""
    node_idx: torch.Tensor
    weight: torch.Tensor
    nx: torch.Tensor
    ny: torch.Tensor
    nz: torch.Tensor
    n_global: int

    @classmethod
    def detect(cls, mesh, fluid) -> "FluidSolidInterface3D":
        """Every face between a fluid element (``fluid``, ``(E,)`` bool) and a solid one,
        from the structured ``(nez, ney, nex)`` element grid; the two sides must see the
        same global nodes in the same order."""
        dev = mesh.ibool.device
        ids = torch.arange(mesh.n_elements, device=dev).reshape(mesh.nez, mesh.ney, mesh.nex)
        fl = fluid.reshape(mesh.nez, mesh.ney, mesh.nex)
        cen = torch.stack([c.flatten(1).mean(1) for c in (mesh.x_e, mesh.y_e, mesh.z_e)], -1)
        idx, w, n = [], [], []
        for axis in range(3):                              # grid axis (z, y, x)
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            a, b = ids[tuple(lo)].reshape(-1), ids[tuple(hi)].reshape(-1)  # b past a
            fa, fb = fl[tuple(lo)].reshape(-1), fl[tuple(hi)].reshape(-1)
            ch = fa != fb
            for f, sol, s in ((a[ch & fa], b[ch & fa], 1), (b[ch & fb], a[ch & fb], -1)):
                if f.numel() == 0:
                    continue
                node, weight, normal = _hex_face(mesh, f, 2 - axis, s)
                if not torch.equal(node, _hex_face(mesh, sol, 2 - axis, -s)[0]):
                    raise ValueError("fluid/solid faces do not pair node-for-node")
                toward = ((cen[sol] - cen[f])[:, None, :] * normal).sum(-1)
                if bool((toward <= 0).any()):
                    raise ValueError("a fluid-solid face normal does not point from the "
                                     "fluid toward the solid (inverted element?)")
                idx.append(node)
                w.append(weight)
                n.append(normal)
        if not idx:
            empty = torch.zeros((0, mesh.ngll ** 2), dtype=torch.long, device=dev)
            z = torch.zeros((0, mesh.ngll ** 2), dtype=mesh.x_e.dtype, device=dev)
            return cls(empty, z, z, z, z, mesh.n_global)
        n = torch.cat(n)
        return cls(torch.cat(idx), torch.cat(w), n[..., 0], n[..., 1], n[..., 2],
                   mesh.n_global)

    @property
    def n_face(self) -> int:
        return int(self.node_idx.shape[0])

    def _scatter(self, vals, like):
        return torch.zeros_like(like).index_add(
            0, self.node_idx.reshape(-1), vals.reshape(-1))

    def apply_ac(self, u_x, u_y, u_z, like):
        """``∮_Γ (u_s · n) φ_i dΓ`` -> ``(n_global,)``, for the fluid."""
        g = self.node_idx
        return self._scatter(self.weight * (self.nx * u_x[g] + self.ny * u_y[g]
                                            + self.nz * u_z[g]), like)

    def apply_el(self, p, like):
        """``∮_Γ p n φ_i dΓ`` -> ``(S_x, S_y, S_z)``, for the solid: the exact transpose of
        :meth:`apply_ac`."""
        pw = self.weight * p[self.node_idx]
        return tuple(self._scatter(pw * c, like) for c in (self.nx, self.ny, self.nz))


def assert_no_normal_pml_corner_on_interface(ifc: "FluidSolidInterface3D", b_x, b_y, b_z):
    """:func:`assert_no_corner_pml_on_interface` in 3-D, on the nodal damping ``b_x``,
    ``b_y``, ``b_z``. The plain coupling term is the boundary integral of the derivative
    along the interface's normal, so a band that stretches only other directions leaves it
    exact -- a sea floor crosses the x-y corner columns of the C-PML that way -- and a band
    of the normal direction alone is SPECFEM2D's single band. Refused, as SPECFEM2D's corner:
    a node where the normal's direction and another one are both stretched."""
    if ifc.n_face == 0:
        return
    g = ifc.node_idx.reshape(-1)
    normal = torch.stack([ifc.nx, ifc.ny, ifc.nz], -1).reshape(-1, 3).abs().argmax(-1)
    st = torch.stack([b_x[g] != 0, b_y[g] != 0, b_z[g] != 0], -1)
    bad = st.gather(1, normal[:, None])[:, 0] & (st.sum(-1) >= 2)
    if bool(bad.any()):
        raise ValueError(
            f"{int(bad.sum())} fluid-solid interface node(s) fall where the C-PML stretches "
            "the interface's normal direction and another one (an edge or corner region of "
            "the absorbing layer across the interface), which the plain coupling does not "
            "support: shrink the PML bands, move the interface, or drop the offending side.")

"""Shared element-local ⇆ global operators for SEM equations.

The equations are independent dataclasses (``SEMEquation`` is only a
``typing.Protocol`` — no shared base), so these are free functions that take the
mesh's ``ibool`` / metric explicitly and compose without inheritance. Two
patterns recurred verbatim across every equation module:

* :func:`assemble` — scatter element-local residuals to global GLL nodes
  (``Σ_e`` over shared nodes), out-of-place so it stays ``torch.vmap``-safe.
* :func:`phys_grads` — element-local physical ∂/∂x, ∂/∂z of one or more global
  fields via the deformed ``inv_J`` chain rule, in one batched gather + two
  batched ξ/η einsums.
"""
from __future__ import annotations

import torch


def assemble(r_e: torch.Tensor, ibool: torch.Tensor, like: torch.Tensor):
    """Assemble element-local ``r_e`` to global nodes.

    Dimension-agnostic: ``ibool`` is the per-element node-index tensor
    (``(E, ngll, ngll)`` in 2D, ``(E, ngll, ngll, ngll)`` in 3D). A single
    field has ``r_e.shape == ibool.shape`` → returns one ``(n_global,)`` field;
    a ``K``-stack has ``r_e.shape == ibool.shape + (K,)`` → returns a tuple of
    ``K`` ``(n_global,)`` fields.

    The scatter is the out-of-place ``index_add`` on a target derived from
    ``like`` (a ``(n_global,)`` field), which keeps it ``torch.vmap``-safe (the
    multi-shot batched path) — an in-place ``index_add_`` on a fresh
    ``zeros(n_global, …)`` breaks vmap.
    """
    idx = ibool.reshape(-1)
    if r_e.dim() == ibool.dim():
        return torch.zeros_like(like).index_add(0, idx, r_e.reshape(-1))
    k = r_e.shape[-1]
    out = torch.zeros_like(like).unsqueeze(-1).repeat(1, k)
    return out.index_add(0, idx, r_e.reshape(-1, k)).unbind(-1)


def assemble_each(fields, ibool, like):
    """Assemble several element-local fields to global nodes, one single-field
    ``index_add`` each — the unstacked counterpart of ``assemble(stack(fields))``.

    On the GPU at 3-D sizes this is **~1.8-2.2× faster and ~20% less peak memory**
    than stacking the ``K`` fields on a trailing axis and doing one ``K``-wide
    scatter: the stack materialises a big ``(E, ngll**d, K)`` copy (e.g. 9×125×E
    ≈ 500 MB) and the ``K``-wide ``index_add`` is far slower than ``K`` narrow
    ones — the same "narrow beats fat" effect seen in the reference-derivative
    einsums. Out-of-place ``index_add`` keeps each scatter ``torch.vmap``-safe.

    Returns a tuple of ``len(fields)`` global ``(n_global,)`` fields.
    """
    idx = ibool.reshape(-1)
    return tuple(torch.zeros_like(like).index_add(0, idx, F.reshape(-1))
                 for F in fields)


def scatter_point_sources(src_G, src_W, wv, like):
    """Per-time-step superposed point-source term ``Σ_k stamp_k(x) · s_k(t)``.

    The FD mode-B (source-encoding) source: ``P`` point sources sharing one
    wavefield, each with its own per-step amplitude ``s_k(t)`` (so polarity /
    amplitude / time-shift can differ per source). Mirrors the FD
    ``SourceTorch._add_indexed_sources`` ``index_put_(accumulate=True)``, but
    with the SEM Lagrange stamp (``ngll**dim`` nodes per source) instead of a
    single grid cell.

    * ``src_G`` — ``(P, m)`` long, the global node indices of each source's
      element (``m = ngll**dim``).
    * ``src_W`` — list (one per source *field*, in ``field_specs`` order) of
      ``(P, m)`` Lagrange stamp weights for that field component of each source.
    * ``wv`` — ``(P,)`` per-source amplitude ``s_k(t)`` for this step (may carry
      autograd, e.g. the per-source wavelet gradient).
    * ``like`` — a ``(n_global,)`` field giving the target shape/dtype/device.

    Returns a list (per source field) of ``(n_global,)`` source vectors. The
    out-of-place ``index_add`` keeps it autograd- and ``torch.vmap``-safe.
    """
    gflat = src_G.reshape(-1)
    return [torch.zeros_like(like).index_add(0, gflat, (W * wv[:, None]).reshape(-1))
            for W in src_W]


def phys_grads(D, ibool, ixx, ixz, iex, iez, *fields):
    """Element-local physical derivatives ``(∂/∂x, ∂/∂z)`` of each global field.

    One batched gather + two batched ξ/η reference-derivative einsums, then the
    inv_J chain rule ``∂/∂x = ixx ∂/∂ξ + iex ∂/∂η`` (and z analogously). Returns
    a flat list ``[df0_dx, df0_dz, df1_dx, df1_dz, …]`` (two entries per field).
    On a single field it uses the unbatched einsum (bit-identical to the old
    per-field code); on ≥2 fields it stacks them on a trailing axis (the fused
    pattern — fewer kernel launches).
    """
    if len(fields) == 1:
        f_e = fields[0][ibool]
        d_dxi  = torch.einsum("bg,eag->eab", D, f_e)
        d_deta = torch.einsum("ag,egb->eab", D, f_e)
        return [ixx * d_dxi + iex * d_deta, ixz * d_dxi + iez * d_deta]
    f_e = torch.stack(fields, dim=-1)[ibool]                  # (E,ngll,ngll,K)
    d_dxi  = torch.einsum("bg,eagk->eabk", D, f_e)
    d_deta = torch.einsum("ag,egbk->eabk", D, f_e)
    d_dx = ixx.unsqueeze(-1) * d_dxi + iex.unsqueeze(-1) * d_deta
    d_dz = ixz.unsqueeze(-1) * d_dxi + iez.unsqueeze(-1) * d_deta
    out = []
    for k in range(len(fields)):
        out += [d_dx[..., k], d_dz[..., k]]
    return out


def phys_grads3(D, ibool, inv_J, *fields):
    """3-D twin of :func:`phys_grads`: ``(∂/∂x, ∂/∂y, ∂/∂z)`` per global field.

    ``inv_J`` is ``(E, ngll, ngll, ngll, 3, 3)`` with ``inv_J[..., m, d] =
    ∂ξ_m/∂x_d`` (the Hex27 convention, see :mod:`sweep.sem.basis.hex27`), so the
    chain rule is ``∂/∂x_d = Σ_m inv_J[..., m, d] · ∂/∂ξ_m`` -- three terms per
    physical derivative instead of the single constant scaling an axis-aligned
    mesh allows.

    Returns a flat list ``[df0_dx, df0_dy, df0_dz, df1_dx, …]``.
    """
    f_e = torch.stack(fields, dim=-1)[ibool]        # (E,ngll,ngll,ngll,K)
    d_ref = (torch.einsum("xX,ezyXk->ezyxk", D, f_e),      # ∂/∂ξ  (x-axis)
             torch.einsum("yY,ezYxk->ezyxk", D, f_e),      # ∂/∂η  (y-axis)
             torch.einsum("zZ,eZyxk->ezyxk", D, f_e))      # ∂/∂γ  (z-axis)
    out = []
    for k in range(len(fields)):
        for d in range(3):
            acc = None
            for m in range(3):
                t = inv_J[..., m, d].unsqueeze(-1) * d_ref[m]
                acc = t if acc is None else acc + t
            out.append(acc[..., k])
    return out


def stress_divergence3(D, ibool, inv_J, w_jac, sig, like):
    """3-D weak-form stress divergence on a deformed mesh.

    ``R_i = ∫ ∂φ/∂x_j · σ_ij dV``. The chain rule ``∂φ/∂x_j = Σ_m inv_J[m,j]
    ∂φ/∂ξ_m`` is folded into a per-reference-direction flux

        F_i^m = w|J| · Σ_j σ_ij · inv_J[..., m, j]

    after which each flux is contracted with ``D^T`` along its own reference
    axis. That is the transpose of :func:`phys_grads3`, which is what makes the
    assembled operator symmetric -- getting the transpose wrong yields an
    operator that still runs and still looks like a wavefield.

    ``sig`` is the 9-tuple ``(s_xx_x, s_xy_y, s_xz_z, s_xy_x, s_yy_y, s_yz_z,
    s_xz_x, s_yz_y, s_zz_z)`` -- one entry per (equation component i, stress
    axis j), so a C-PML caller can pass a differently stretched field for each.
    """
    (sxx_x, sxy_y, sxz_z, sxy_x, syy_y, syz_z, sxz_x, syz_y, szz_z) = sig
    rows = ((sxx_x, sxy_y, sxz_z), (sxy_x, syy_y, syz_z), (sxz_x, syz_y, szz_z))
    R = []
    for row in rows:                       # equation component i = x, y, z
        acc = None
        for m, ein in enumerate(("Xx,ezyX->ezyx", "Yy,ezYx->ezyx",
                                 "Zz,eZyx->ezyx")):
            F = w_jac * sum(row[j] * inv_J[..., m, j] for j in range(3))
            t = torch.einsum(ein, D, F)
            acc = t if acc is None else acc + t
        R.append(acc)
    return assemble_each(R, ibool, like)

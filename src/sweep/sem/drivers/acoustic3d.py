"""The 3-D acoustic C-PML strip driver (flat Hex8 and deformed Hex27): the
Function :class:`~sweep.sem.dd.SEMModelParallel` steps over every local
strip with one exchange per assembly. There is no fused twin -- PropSEM and
SEMCudaPropagator run this Function with a single strip.
"""

from __future__ import annotations

import torch

from ..cdrive import (METRIC_NAMES, Alloc, check_filled, dd_grads, dims,
                      fill, give, put,
                      roll, swap, take)

_AC3D_CONSTS = ("ibool", "D", "w_jac", "w_jx", "w_jy", "w_jz", "inv_M",
                "a_x", "b_x", "dbx_dx", "a_y", "b_y", "dby_dy",
                "a_z", "b_z", "dbz_dz", "src_pattern", "rec_G", "rec_W",
                "dirichlet_idx", "ihx", "ihy", "ihz", "band_node",
                "band_idx", "band_idx_x", "band_idx_y", "band_idx_z",
                # which elements may carry a non-zero psi_d, per axis: the
                # assembly skips the a*psi half where they cannot (slab.py)
                "elem_slab")
# read per element-node by the deformed kernels only; on the flat mesh the
# geometry rides in w_j{x,y,z} and 1/2h
_AC3D_DEFORMED = METRIC_NAMES


def _ac3d_metric(s, c, A):
    """The Hex27 metric on ``s``, with the columnar shortcut where the mesh has
    one. ``columnar`` defaults to 0 on the struct, so a path that forgets this
    falls back to the always-correct full 3x3 rather than reading a stale
    scalar -- see ``prop_c._columnar_metric``, which derives the flag."""
    if s.deformed:
        fill(s, c, _AC3D_DEFORMED)
        s.columnar = int(c.get("columnar", 0))
        s.col_ixx, s.col_iey = c.get("col_ixx", 0.0), c.get("col_iey", 0.0)
    else:
        put(s, _AC3D_DEFORMED, A.empty)


_AC3D_MEM = tuple(f"{f}_{ax}" for f in ("psi", "zeta") for ax in "xyz")
_AC3D_STATE_TAPES = ("save_u",) + tuple("save_" + f for f in _AC3D_MEM)
_AC3D_TAPES = _AC3D_STATE_TAPES + ("save_w_sum",)
_AC3D_RINGS = ("band_u",) + tuple("band_" + f for f in _AC3D_MEM)
_AC3D_R6 = tuple("r_" + f for f in _AC3D_MEM)
_AC3D_ADJ = (tuple("l" + f for f in _AC3D_MEM)
             + tuple("l" + f + "_n" for f in _AC3D_MEM) + ("apx", "apy", "apz"))
# what the forward hands the backward, by name
_AC3D_CARRY = (_AC3D_TAPES + _AC3D_RINGS
               + ("fin_u", "fin_u_prev", "band_scale", "band_kind",
                  "band_split", "bs"))


def _ac3d_mode(c, need_bwd):
    """``(mode, wtape, slab)`` of one run.

    ``mode`` is what the forward keeps for the backward: ``'ring'`` (the
    C-PML band per step), ``'tape'`` (save-all: the w_sum tape alone under
    ``wtape``, else the seven state tapes) or ``'none'``. ``slab`` says
    whether the memory fields live on their axis slab (slab.py): they do
    unless a state tape copies them or the rings are not split.
    """
    from ..prop_c import _BS_RING_SPLIT, _PSI_SLAB, _WSUM_TAPE
    mode = ("ring" if bool(c.get("bs")) and need_bwd
            else "tape" if need_bwd else "none")
    wtape = mode == "tape" and _WSUM_TAPE
    slab = (_PSI_SLAB and (mode != "tape" or wtape)
            and (mode != "ring" or _BS_RING_SPLIT))
    return mode, wtape, slab


def _ac3d_forward_input(c, vp2, wavelet, mode, wtape, slab):
    """One strip's filled ``Sem3dCpmlForwardInput`` and its g9 block."""
    import sweep._C as _C
    from ..prop_c import _BS_RING_SPLIT, _band_alloc
    from ..slab import set_slab_fields, field_len
    dev, nt = wavelet.device, c["nt"]
    A = Alloc(dev, c["n_global"], nt)
    s = _C.Sem3dCpmlForwardInput()
    fill(s, c, _AC3D_CONSTS)
    s.deformed = c["deformed"]
    _ac3d_metric(s, c, A)
    s.vp2_node, s.wavelet = vp2, wavelet
    put(s, ("u", "u_prev", "u_next", "w_sum"), A.zeros)
    set_slab_fields(s, c, slab)
    for nm in _AC3D_MEM + tuple(f + "_n" for f in _AC3D_MEM) + ("apsi_x", "apsi_y", "apsi_z"):
        setattr(s, nm, A.zeros(field_len(nm, c, slab)))
    g9, g9buf = A.chunks(_ac3d_row_sizes(c, _AC3D_G9_ROWS, slab),
                         with_parent=True)
    (s.gx, s.gy, s.gz, s.kx, s.ky, s.kz, s.mx, s.my, s.mz) = g9
    put(s, ("gax", "gay", "gaz"), A.empty)   # retired: read by nothing
    s.seis = torch.zeros(c["n_rec"], nt, dtype=torch.float32, device=dev)
    s.bs, s.band_split = int(mode == "ring"), int(_BS_RING_SPLIT)
    s.band_kind, s.band_scale = 0, A.empty
    put(s, _AC3D_TAPES + _AC3D_RINGS + ("fin_u", "fin_u_prev"), A.empty)
    if mode == "ring":
        # rings are per STRIP, O(nt * |local band|): u on the union band,
        # psi_d / zeta_d on their own axis's band when split
        s.band_kind = int(c.get("bs_kind", 0))
        ring, s.band_scale = _band_alloc(s.band_kind, 7, nt,
                                         int(c["band_idx"].numel()), dev)
        s.band_u = ring()
        for nm in _AC3D_MEM:
            key = f"band_idx_{nm[-1]}" if s.band_split else "band_idx"
            setattr(s, "band_" + nm, ring(int(c[key].numel())))
        s.fin_u, s.fin_u_prev = A.zeros(), A.zeros()
    elif mode == "tape":
        put(s, ("save_w_sum",) if wtape else _AC3D_STATE_TAPES,
            lambda: A.hist(zero=False))     # written before read; see Alloc.hist
    dims(s, c, 3, batch=1)
    return check_filled(s), g9buf


def _ac3d_block_flat(c, part, rows, slab):
    """A part's ``(lo, hi)`` flat cut indices inside a heterogeneous block.

    ``rows`` is the block's rows in order, each either ``None`` for a per-node
    row or an axis letter for a row that lives on that axis's slab. A slab row
    is shorter than a node row and carries only the cut nodes that lie on ITS
    slab, so the cut is not one column set repeated K times and the exchange
    has to index the buffer flat (``dd._DDExchange.block``).

    Both sides of a cut list the same physical nodes in the same order, and
    whether a node is on slab d is a property of the NODE -- the profile is
    sliced from one global build and the y/z plane sets are global -- so the
    two filtered subsequences correspond entry for entry, which is what the
    sum needs. The x rows are usually empty: the cut is asserted PML-free, so
    no node on it has b_x != 0. Usually, not always -- on a deformed mesh the
    band edge lands at float residue rather than a clean zero -- so the filter
    is done rather than assumed, on both sides alike.
    """
    from ..slab import slab_slot
    ng = int(c["n_global"])
    dims = (int(c["nx"]), int(c["ny"]), int(c["nz"])) if slab else None
    off, s = [], 0
    for ax in rows:
        off.append(s)
        s += int(c[f"slab_n_{ax}"]) if (slab and ax) else ng
    out = []
    for iface in (part.if_lo, part.if_hi):
        if iface is None:
            out.append(None)
            continue
        flat = []
        for ax, o in zip(rows, off):
            if slab and ax:
                sl = slab_slot(ax, iface, dims, c["tab_" + ax])
                flat.append(sl[sl >= 0] + o)
            else:
                flat.append(iface + o)
        out.append(torch.cat(flat))
    return tuple(out)


# each block's rows, in the order the C entry unpacks them: an axis letter
# where the row lives on that axis's slab, None where it is per node
_AC3D_L4_ROWS = (None, "x", "y", "z")               # lu_G, lap_x, lap_y, lap_z
# g_d and k_d on their axis slabs, then m_d = ga_d - k_d per node
_AC3D_G9_ROWS = ("x", "y", "z", "x", "y", "z", None, None, None)


def _ac3d_row_sizes(c, rows, slab):
    """A block's row lengths: its axis slab where the row lives on one, the
    node count otherwise."""
    return [int(c[f"slab_n_{ax}"]) if (slab and ax) else int(c["n_global"])
            for ax in rows]


def _ac3d_backward_input(c, vp2, wavelet, own, carry, grad_syn, mode, slab):
    """One strip's filled ``Sem3dCpmlBackwardInput``, its g9 block and the
    ONE buffer behind lu_G / lap_x / lap_y / lap_z."""
    import sweep._C as _C
    from ..slab import set_slab_fields, field_len
    dev, nt = wavelet.device, c["nt"]
    A = Alloc(dev, c["n_global"], nt)
    b = _C.Sem3dCpmlBackwardInput()
    fill(b, c, _AC3D_CONSTS)
    b.deformed = c["deformed"]
    _ac3d_metric(b, c, A)
    b.vp2_node, b.wavelet = vp2, wavelet
    # the wavelet gradient is a global reduction: mask the non-owned cut
    # plane (complete on both strips after the adjoint exchange)
    b.src_pattern_own = (c["src_pattern"] * own).contiguous()
    give(b, carry)
    b.grad_syn = grad_syn
    set_slab_fields(b, c, slab)
    put(b, ("u_hi", "u_lo", "u_km1") + _AC3D_R6, A.empty)
    if mode == "ring":
        # reverse-march levels seeded from the forward's finals; the six
        # memory-rebuild buffers contiguous for one memset per step
        b.u_hi, b.u_lo, b.u_km1 = b.fin_u.clone(), b.fin_u_prev.clone(), A.zeros()
        sizes = [field_len(nm, c, slab) for nm in _AC3D_R6]
        for nm, t in zip(_AC3D_R6, A.chunks(sizes)):
            setattr(b, nm, t)
    g9, g9buf = A.chunks(_ac3d_row_sizes(c, _AC3D_G9_ROWS, slab),
                         with_parent=True)
    (b.gx, b.gy, b.gz, b.kx, b.ky, b.kz, b.mx, b.my, b.mz) = g9
    put(b, ("gax", "gay", "gaz"), A.empty)   # retired: read by nothing
    # lu_G is per node; each lap_d is its axis slab long under the slab
    # layout (grads_T writes it there and combine reads it there), so the
    # rows differ in length and the block is one buffer plus a flat index
    l4, l4buf = A.chunks(_ac3d_row_sizes(c, _AC3D_L4_ROWS, slab),
                         with_parent=True)
    b.lu_G, b.lapx, b.lapy, b.lapz = l4
    put(b, ("gA", "gB", "gu_k", "gw", "w_sum", "gbx", "gby", "gbz",
            "gbkx", "gbky", "gbkz", "gbax", "gbay", "gbaz", "grad_vp2"), A.zeros)
    for nm in _AC3D_ADJ:        # a_d*psi_d and the adjoint carries: on the slab
        setattr(b, nm, A.zeros(field_len(nm, c, slab)))
    b.grad_wavelet = torch.zeros(nt, dtype=torch.float32, device=dev)
    dims(b, c, 3, batch=1)
    return check_filled(b), g9buf, l4buf


class _SEMAcoustic3DCpmlDDC(torch.autograd.Function):
    """DD stepped 3-D acoustic C-PML loop over every local part.

    Same shape as the 2-D acoustic twin -- one assembly per forward step
    (nine assembled C-PML grad fields) and two per reverse step -- with the
    cut being a YZ PLANE instead of a column. Differentiable per part:
    ``vp2_node`` (the 3-D path carries the geometric inverse mass as a
    constant) plus the shared wavelet. Boundary saving composes at zero
    extra exchange: each strip rings its OWN C-PML band, and the two strips
    sharing a cut agree on that plane's band membership (the profile is a
    slice of one global build), so both sides reconstruct it the same way --
    from the ring, or by the pointwise march. See :func:`_band_nodes`.
    """

    @staticmethod
    def forward(ctx, wavelet, parts, exchange, bs, *mats):
        import sweep._C as _C
        nt = parts[0].c["nt"]
        need_bwd = wavelet.requires_grad or any(t.requires_grad for t in mats)
        mode, wtape, slab = _ac3d_mode(parts[0].c, need_bwd)
        built = [_ac3d_forward_input(p.c, vp2, wavelet, mode, wtape, slab)
                 for p, vp2 in zip(parts, mats)]
        inps, g9s = [s for s, _ in built], [g for _, g in built]
        gflat = [_ac3d_block_flat(p.c, p, _AC3D_G9_ROWS, slab) for p in parts]

        for n in range(nt):
            for s in inps:
                _C.sem3d_acoustic_cpml_dd_asm(s)
            exchange.block(g9s, flat=gflat)   # the step's ONE assembly
            for s in inps:
                _C.sem3d_acoustic_cpml_dd_step(s, n)
            for s in inps:
                roll(s, ("u_prev", "u", "u_next"))
                swap(s, [(f, f + "_n") for f in _AC3D_MEM])
        ctx.mode, ctx.wtape, ctx.slab = mode, wtape, slab    # read by tests too
        if need_bwd:
            ctx.save_for_backward(wavelet, *mats)
            ctx.parts, ctx.exchange = parts, exchange
            ctx.carry = [take(s, _AC3D_CARRY) for s in inps]
        return tuple(s.seis for s in inps)

    @staticmethod
    def backward(ctx, *grad_seis):
        import sweep._C as _C
        wavelet, *mats = ctx.saved_tensors
        parts, exchange = ctx.parts, ctx.exchange
        nt = parts[0].c["nt"]
        from ..dd import _patch_dup_grad_syn
        gs = _patch_dup_grad_syn(parts, [g.contiguous().float() for g in grad_seis], nt)
        built = [_ac3d_backward_input(p.c, vp2, wavelet, p.own, carry, g,
                                      ctx.mode, ctx.slab)
                 for p, vp2, carry, g in zip(parts, mats, ctx.carry, gs)]
        bs_ = [b for b, _, _ in built]
        g9s, l4s = [g for _, g, _ in built], [l for _, _, l in built]
        gflat = [_ac3d_block_flat(p.c, p, _AC3D_G9_ROWS, ctx.slab) for p in parts]
        lflat = [_ac3d_block_flat(p.c, p, _AC3D_L4_ROWS, ctx.slab) for p in parts]

        ring = ctx.mode == "ring"
        for k in range(nt - 1, -1, -1):
            for b in bs_:
                _C.sem3d_acoustic_cpml_dd_bwd_pre(b, k)
            if k >= 1:
                # under the w_sum tape nothing is rebuilt from the forward
                # state, so the nine assembled fields are never produced
                if not ctx.wtape:
                    exchange.block(g9s, flat=gflat)
                for b in bs_:
                    _C.sem3d_acoustic_cpml_dd_bwd_mid(b, k)
                exchange.block(l4s, flat=lflat)
            for b in bs_:
                _C.sem3d_acoustic_cpml_dd_bwd_end(b, k)
            for b in bs_:
                swap(b, [("gA", "gu_k")] + [("l" + f, "l" + f + "_n") for f in _AC3D_MEM])
                if ring and k >= 1:    # (u^{k+1}, u^k) -> (u^k, u^{k-1})
                    roll(b, ("u_hi", "u_lo", "u_km1"))

        # per-node gradients are complete on both sides of a cut after the
        # adjoint exchange: dedup by ownership; the wavelet's is a global sum
        grads = [b.grad_vp2 * p.own for b, p in zip(bs_, parts)]
        return dd_grads(ctx, bs_, grads)



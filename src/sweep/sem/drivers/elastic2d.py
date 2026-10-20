"""The 2-D elastic C-PML drivers on CUDA (SPECFEM2D-style 2nd-order
displacement CPML): the fused single-domain Function (``prop_c``'s
``_SEMElasticCpmlC``, shot-batchable over one shared model) and the strip
Function of ``dd`` (``_SEMElasticCpmlDDC``). Differentiable inputs: lam, mu,
lam2mu (stress), inv_M (mass), w_rho_J (the rho-weighted quadrature factor
in the PML acceleration term) and the wavelet; the CPML profile
coefficients are geometric constants.

Boundary saving rings the displacement on band NODES and the six
element-local histories (four stretched derivatives, two acceleration
terms) on band ELEMENTS, each family with its own store kind, plus the
final two displacement levels that seed the reverse march.
"""

from __future__ import annotations

import torch

from ..cdrive import (EL_CPML_FLAT, Alloc, check_filled, dd_grads, dims,
                      fill, give, grads_or_none, memory_mode, put, roll,
                      swap, take)

_EL2D_CONSTS = ("ibool", "D", "wJ", "ixx", "ixz", "iex", "iez", "mask_e",
                "outer_edge_idx", "pat_x", "pat_z", "rec_G", "rec_W",
                "band_node_idx", "band_elem_idx", "elem_band") + EL_CPML_FLAT
_EL2D_HIST = ("sd_xx", "sd_zx", "sd_xz", "sd_zz", "apx", "apz")   # per element-node
_EL2D_U = ("ux", "uz")
_EL2D_TAPES = tuple("save_" + f for f in _EL2D_U + _EL2D_HIST)
_EL2D_RINGS = tuple("band_" + f for f in _EL2D_U + _EL2D_HIST)
_EL2D_SCR = ("sd_scr_xx", "sd_scr_zx", "sd_scr_xz", "sd_scr_zz", "ap_scr_x", "ap_scr_z")
_EL2D_FIN = ("fin_ux", "fin_uz", "fin_ux_prev", "fin_uz_prev")
_EL2D_LEVELS = ("u_hi_x", "u_hi_z", "u_lo_x", "u_lo_z", "u_km1_x", "u_km1_z")
_EL2D_REC = tuple("rec_" + f for f in _EL2D_HIST)
_EL2D_REC_R6 = ("rec_rxx", "rec_rxz", "rec_rzx", "rec_rzz", "rec_rdx", "rec_rdz")
_EL2D_CARRY = (_EL2D_TAPES + _EL2D_RINGS + _EL2D_FIN
               + ("band_scale", "band_scale_e", "band_kind", "band_kind_e", "bs"))


def _el2d_forward_input(c, mats, wavelet, mode, B=1):
    """One domain's filled ``SemElasticCpmlForwardInput`` and its Acc block.
    ``B`` shots share one model and one kernel sequence (per-shot
    ``pat_x`` / ``pat_z`` ``(B, ng)``, ``seis_*`` ``(B, nrec, nt)``)."""
    import sweep._C as _C
    from ..prop_c import _BAND_DTYPE
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    E, m = c["n_elements"], c["ngll"] ** 2
    A = Alloc(dev, B * ng, nt)
    s = _C.SemElasticCpmlForwardInput()
    fill(s, c, _EL2D_CONSTS)
    s.B = B
    s.lam, s.mu, s.lam2mu, s.inv_M, s.w_rho_J = mats
    s.wavelet = wavelet
    put(s, ("ux", "uz", "ux_prev", "uz_prev", "ux_next", "uz_next"), A.zeros)
    acc = A.block(2)
    s.Acc_x, s.Acc_z = acc
    put(s, ("r_dux_dx", "r_dux_dz", "r_duz_dx", "r_duz_dz", "r_displ_x",
            "r_displ_z"), lambda: A.zeros(B * E * m * 2))
    seis = (B, c["n_rec"], nt) if B > 1 else (c["n_rec"], nt)
    s.seis_x = torch.zeros(*seis, dtype=torch.float32, device=dev)
    s.seis_z = torch.zeros(*seis, dtype=torch.float32, device=dev)
    s.bs, s.band_kind, s.band_kind_e = int(mode == "ring"), 0, 0
    put(s, _EL2D_TAPES + _EL2D_RINGS + _EL2D_SCR + _EL2D_FIN
        + ("band_scale", "band_scale_e"), A.empty)
    if mode == "ring":
        # two ring widths (u on band nodes, the six histories on band
        # ELEMENTS), each family with its own store kind; the int8 amax
        # tables hold one scale per (field, step, shot) for the node rings
        # and one per band element on top of that for the element rings
        nbn, nbe = int(c["band_node_idx"].numel()), int(c["band_elem_idx"].numel())
        s.band_kind = int(c.get("bs_kind", 0))
        s.band_kind_e = int(c.get("bs_kind_e", s.band_kind))
        lead = (B,) if B > 1 else ()
        if s.band_kind == 2:
            s.band_scale = torch.zeros(2, nt, *lead, dtype=torch.float32, device=dev)
        if s.band_kind_e == 2:
            s.band_scale_e = torch.zeros(6, nt, *lead, nbe, dtype=torch.float32, device=dev)
        rdt, rdt_e = _BAND_DTYPE[s.band_kind], _BAND_DTYPE[s.band_kind_e]
        put(s, ("band_ux", "band_uz"),
            lambda: torch.zeros(nt, B * nbn, dtype=rdt, device=dev))
        put(s, tuple("band_" + f for f in _EL2D_HIST),
            lambda: torch.zeros(nt, B * nbe * m, dtype=rdt_e, device=dev))
        put(s, _EL2D_SCR, lambda: A.zeros(B * E * m))
        put(s, _EL2D_FIN, A.zeros)
    elif mode == "tape":
        # zero=False: the forward writes every row of these before the
        # backward reads one (NaN-poison of the whole suite; see Alloc.hist)
        put(s, ("save_ux", "save_uz"), lambda: A.hist(zero=False))
        put(s, tuple("save_" + f for f in _EL2D_HIST),
            lambda: A.hist(B * E * m, zero=False))
    dims(s, c, 2)
    return check_filled(s), acc


def _el2d_backward_input(c, mats, wavelet, carry, grads, B=1, own=None):
    """One domain's filled ``SemElasticCpmlBackwardInput``. ``own`` (a strip's
    ownership mask) masks the patterns the wavelet gradient reduces over;
    the reverse-march levels are seeded from the forward's finals when it
    is given (the stepped entries expect that; the fused C loop seeds them
    itself)."""
    import sweep._C as _C
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    E, m = c["n_elements"], c["ngll"] ** 2
    A = Alloc(dev, B * ng, nt)
    b = _C.SemElasticCpmlBackwardInput()
    fill(b, c, _EL2D_CONSTS + ("pin_mask",))
    b.B = B
    b.lam, b.mu, b.lam2mu, b.inv_M, b.w_rho_J = mats
    b.wavelet = wavelet
    for nm in ("pat_x", "pat_z"):
        setattr(b, nm + "_own", (c[nm] * own).contiguous() if own is not None else A.empty)
    give(b, carry)
    b.grad_sx, b.grad_sz = grads
    put(b, _EL2D_LEVELS + _EL2D_REC + _EL2D_REC_R6 + ("rec_Acc_x", "rec_Acc_z"), A.empty)
    if b.bs:
        put(b, _EL2D_LEVELS, A.zeros)
        if own is not None:
            b.u_hi_x, b.u_hi_z = b.fin_ux.clone(), b.fin_uz.clone()
            b.u_lo_x, b.u_lo_z = b.fin_ux_prev.clone(), b.fin_uz_prev.clone()
        b.rec_Acc_x, b.rec_Acc_z = A.block(2)
        put(b, _EL2D_REC, lambda: A.zeros(B * E * m))
        # the six memory scratch buffers, re-zeroed every reverse step and
        # never rotating: one memset for six
        (b.rec_rxx, b.rec_rxz, b.rec_rzx,
         b.rec_rzz, b.rec_rdx, b.rec_rdz) = A.block(6, B * E * m * 2)
    put(b, ("lam1_x", "lam1_z", "lamn_x", "lamn_z", "g_x", "g_z",
            "gprev_x", "gprev_z", "y_x", "y_z"), A.zeros)
    # re-zeroed every step: one memset for four. The sc_prev pair SWAPS
    # every step (this step's scatter is the next step's carry), so it gets
    # its own block: after the swap the "a" half is still two adjacent rows
    b.sc_now_x, b.sc_now_z, b.adj_x, b.adj_z = A.block(4)
    (b.sc_prev_a_x, b.sc_prev_a_z, b.sc_prev_b_x, b.sc_prev_b_z) = A.block(4)
    put(b, ("mu_xx", "mu_zx", "mu_xz", "mu_zz", "mu_dx", "mu_dz"),
        lambda: A.zeros(B * E * m * 2))
    # per-shot gradients under the batch's one shared model (summed by the
    # caller), which keeps the kernels free of cross-shot atomics
    lead = (B,) if B > 1 else ()
    for nm, t in (("grad_lam", b.lam), ("grad_mu", b.mu),
                  ("grad_lam2mu", b.lam2mu), ("grad_w_rho_J", b.w_rho_J)):
        setattr(b, nm, torch.zeros(*lead, *t.shape, dtype=torch.float32, device=dev))
    b.grad_inv_M = A.zeros()
    b.grad_wavelet = torch.zeros(*lead, nt, dtype=torch.float32, device=dev)
    dims(b, c, 2)
    return check_filled(b)


def _el2d_rotate_fwd(s):
    roll(s, ("ux_prev", "ux", "ux_next"))
    roll(s, ("uz_prev", "uz", "uz_next"))


def _el2d_rotate_bwd(b):
    swap(b, [("lam1_x", "lamn_x"), ("lam1_z", "lamn_z"), ("gprev_x", "g_x"),
             ("gprev_z", "g_z"), ("sc_prev_b_x", "sc_prev_a_x"),
             ("sc_prev_b_z", "sc_prev_a_z")])


class _SEMElasticCpmlC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, inv_M, w_rho_J, wavelet, c):
        import sweep._C as _C
        B = int(c.get("B", 1))
        mats = (lam, mu, lam2mu, inv_M, w_rho_J)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        s, _ = _el2d_forward_input(c, mats, wavelet, memory_mode(c, need_bwd), B)
        seis_x, seis_z = _C.sem2d_elastic_cpml_forward(s)
        if need_bwd:
            ctx.save_for_backward(*mats, wavelet)
            ctx.c, ctx.B = c, B
            ctx.carry = take(s, _EL2D_CARRY)
        return seis_x, seis_z

    @staticmethod
    def backward(ctx, grad_sx, grad_sz):
        import sweep._C as _C
        *mats, wavelet = ctx.saved_tensors
        B, ng = ctx.B, ctx.c["n_global"]
        b = _el2d_backward_input(ctx.c, mats, wavelet, ctx.carry,
                                 (grad_sx.contiguous().float(),
                                  grad_sz.contiguous().float()), B)
        g_lam, g_mu, g_l2m, g_M, g_wrJ, g_w = _C.sem2d_elastic_cpml_backward(b)
        if B > 1:                  # one model for every shot: sum the per-shot grads
            g_lam, g_mu, g_l2m, g_wrJ = (g_lam.sum(0), g_mu.sum(0),
                                         g_l2m.sum(0), g_wrJ.sum(0))
            g_M = g_M.reshape(B, ng).sum(0)
            if wavelet.dim() == 1:
                g_w = g_w.sum(0)
        return grads_or_none(mats + [wavelet],
                             (g_lam, g_mu, g_l2m, g_M, g_wrJ, g_w)) + (None,)


class _SEMElasticCpmlDDC(torch.autograd.Function):
    """DD stepped elastic C-PML loop over every local part.

    One assembly per step (the element kernel's Acc scatter) -> one 2-field
    exchange forward; one per reverse step (element_T's sc_now/sc_prev) ->
    one 4-field exchange, plus a 2-field one for the boundary-saving march's
    Acc recompute. Differentiable per part: (lam, mu, lam2mu, inv_M,
    w_rho_J) + the shared wavelet.
    """

    @staticmethod
    def forward(ctx, wavelet, parts, exchange, bs, *mats):
        import sweep._C as _C
        nt = parts[0].c["nt"]
        need_bwd = wavelet.requires_grad or any(t.requires_grad for t in mats)
        mode = memory_mode(parts[0].c, need_bwd)
        built = [_el2d_forward_input(p.c, mats[5 * i:5 * i + 5], wavelet, mode)
                 for i, p in enumerate(parts)]
        inps, accs = [s for s, _ in built], [a for _, a in built]

        for n in range(nt):
            for s in inps:
                _C.sem2d_elastic_cpml_dd_asm(s, n)
            exchange.block(accs)         # the step's ONE assembly
            for s in inps:
                _C.sem2d_elastic_cpml_dd_step(s, n)
            for s in inps:
                _el2d_rotate_fwd(s)
        if need_bwd:
            ctx.save_for_backward(wavelet, *mats)
            ctx.parts, ctx.exchange = parts, exchange
            ctx.carry = [take(s, _EL2D_CARRY) for s in inps]
        return tuple(t for s in inps for t in (s.seis_x, s.seis_z))

    @staticmethod
    def backward(ctx, *grad_seis):
        import sweep._C as _C
        from ..dd import _patch_dup_grad_syn
        wavelet, *mats = ctx.saved_tensors
        parts, exchange = ctx.parts, ctx.exchange
        nt = parts[0].c["nt"]
        gs = _patch_dup_grad_syn(parts, [g.contiguous().float() for g in grad_seis],
                                 nt, stride=2)
        bs_ = [_el2d_backward_input(p.c, mats[5 * i:5 * i + 5], wavelet, carry,
                                    gs[2 * i:2 * i + 2], own=p.own)
               for i, (p, carry) in enumerate(zip(parts, ctx.carry))]
        ring = bool(bs_[0].bs)

        def many(names):
            exchange.many([[getattr(b, nm) for b in bs_] for nm in names])

        for n in range(nt - 1, -1, -1):
            if ring and n >= 1:
                for b in bs_:
                    _C.sem2d_elastic_cpml_dd_bwd_march_asm(b)
                many(("rec_Acc_x", "rec_Acc_z"))
                for b in bs_:
                    _C.sem2d_elastic_cpml_dd_bwd_march_fin(b, n)
            for b in bs_:
                _C.sem2d_elastic_cpml_dd_bwd_pre(b, n)
            many(("sc_now_x", "sc_now_z", "sc_prev_a_x", "sc_prev_a_z"))
            for b in bs_:
                _C.sem2d_elastic_cpml_dd_bwd_end(b, n)
            for b in bs_:
                _el2d_rotate_bwd(b)
                if ring and n >= 1:
                    roll(b, ("u_hi_x", "u_lo_x", "u_km1_x"))
                    roll(b, ("u_hi_z", "u_lo_z", "u_km1_z"))

        # element-local gradients need no dedup (elements are uniquely
        # owned); inv_M is per node, so mask the non-owned cut column
        grads = []
        for b, p in zip(bs_, parts):
            grads += [b.grad_lam, b.grad_mu, b.grad_lam2mu, b.grad_inv_M * p.own,
                      b.grad_w_rho_J]
        return dd_grads(ctx, bs_, grads)

"""The coupled fluid-solid drivers on CUDA (acoustic potential in the water,
displacement in the rock, coupled across the seabed faces): the fused
single-domain Function (``prop_c``'s ``_SEMFluidSolidC``, shot-batchable over
one shared model, C-PML optional) and the strip Function of ``dd``
(``_SEMFluidSolidDDC``, always C-PML). Differentiable inputs: the fluid
(inv_rho, inv_M, kappa) and solid (lam, mu, lam2mu, inv_M, w_rho_J)
materials and the wavelet.

Boundary saving rings BOTH domains' bands per domain -- the fluid band on
nodes, the solid band on nodes (u) and on ELEMENTS (its six histories) --
and re-anchors the reverse march on full-state checkpoints every
``ckpt_every`` steps, because the coupled march accumulates error the
acoustic and elastic ones do not.
"""

from __future__ import annotations

import torch

from ..cdrive import (AC_CPML_FIELDS, EL_CPML_FLAT, Alloc, check_filled,
                      dd_grads, dims, fill, give, grads_or_none, memory_mode,
                      put, roll, swap, take)

_FS_CONSTS = ("ibool", "D", "wJ", "w1d", "ixx", "ixz", "iex", "iez",
              "ac_mask_e", "el_mask_e", "dirichlet_idx",
              "ifc_idx", "ifc_w", "ifc_nx", "ifc_nz",
              "pat_chi", "pat_x", "pat_z", "rec_G", "rec_W")
_FS_CPML = (("ac_inv_Mg",) + AC_CPML_FIELDS + EL_CPML_FLAT
            + ("outer_edge_idx",))
_FS_MATS = ("ac_inv_rho", "ac_inv_M", "ac_kappa",
            "el_lam", "el_mu", "el_lam2mu", "el_inv_M", "el_w_rho_J")
_FS_MEM = ("psi_x", "psi_z", "zeta_x", "zeta_z")             # fluid C-PML memories
_FS_HIST = ("sd_xx", "sd_zx", "sd_xz", "sd_zz", "apx", "apz")  # solid, per element-node
_FS_TAPES = (("save_chi", "save_ux", "save_uz")
             + tuple("save_" + f for f in _FS_MEM + _FS_HIST))
_FS_RINGS_AC = ("band_chi",) + tuple("band_" + f for f in _FS_MEM)
_FS_RINGS_EL = ("band_ux", "band_uz") + tuple("band_" + f for f in _FS_HIST)
_FS_SCR = ("sd_scr_xx", "sd_scr_zx", "sd_scr_xz", "sd_scr_zz", "ap_scr_x", "ap_scr_z")
_FS_FIN = ("fin_chi", "fin_chi_prev", "fin_ux", "fin_uz", "fin_ux_prev", "fin_uz_prev")
_FS_CK = ("ck_chi", "ck_chi_prev", "ck_ux", "ck_uz", "ck_ux_prev", "ck_uz_prev")
_FS_BANDS = ("band_idx", "band_node_idx", "band_elem_idx")
_FS_LEVELS = ("chi_hi", "chi_lo", "chi_km1", "u_hi_x", "u_hi_z", "u_lo_x", "u_lo_z",
              "u_km1_x", "u_km1_z")
_FS_R4 = tuple("r_" + f for f in _FS_MEM)
_FS_REC = tuple("rec_" + f for f in _FS_HIST)
_FS_REC_R6 = ("rec_rxx", "rec_rxz", "rec_rzx", "rec_rzz", "rec_rdx", "rec_rdz")
_FS_REC_NODE = ("rec_chi_dd", "rec_Acc_x", "rec_Acc_z", "rec_f_x", "rec_f_z")
_FS_CARRY = (_FS_TAPES + _FS_RINGS_AC + _FS_RINGS_EL + _FS_FIN + _FS_CK + _FS_BANDS
             + ("ckpt_every", "band_scale", "band_scale_e", "band_kind",
                "band_kind_e", "bs"))


def _fs_forward_input(c, mats, wavelet, mode, B=1):
    """One domain's filled ``SemFluidSolidForwardInput``. Without a C-PML
    (``c["has_cpml"] == 0``, fused path only) the memory fields and the ring
    mode do not exist: ``mode`` falls back to the tape."""
    import sweep._C as _C
    from ..prop_c import _BAND_DTYPE
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    E, m = c["n_elements"], c["ngll"] ** 2
    cp = bool(c.get("has_cpml", 1))
    if mode == "ring" and not cp:
        mode = "tape"
    A = Alloc(dev, B * ng, nt)
    s = _C.SemFluidSolidForwardInput()
    fill(s, c, _FS_CONSTS)
    if cp:
        fill(s, c, _FS_CPML)
    else:
        put(s, _FS_CPML, A.empty)
    s.B = B
    for nm, t in zip(_FS_MATS, mats):
        setattr(s, nm, t)
    s.wavelet = wavelet
    put(s, ("chi", "chi_prev", "chi_next", "s_ac", "chi_dd", "Kchi",
            "ux", "uz", "ux_prev", "uz_prev", "ux_next", "uz_next",
            "Acc_x", "Acc_z", "f_x", "f_z"), A.zeros)
    put(s, _FS_MEM + tuple(f + "_n" for f in _FS_MEM)
        + ("apsi_x", "apsi_z", "w_sum", "gx", "gz", "kx", "kz", "gax", "gaz"),
        A.zeros if cp else A.empty)
    put(s, ("r_dux_dx", "r_dux_dz", "r_duz_dx", "r_duz_dz", "r_displ_x", "r_displ_z"),
        (lambda: A.zeros(B * E * m * 2)) if cp else A.empty)
    seis = (B, c["n_rec"], nt) if B > 1 else (c["n_rec"], nt)
    for nm in ("seis_chi", "seis_x", "seis_z"):
        setattr(s, nm, torch.zeros(*seis, dtype=torch.float32, device=dev))
    s.bs, s.ckpt_every, s.band_kind, s.band_kind_e = int(mode == "ring"), 0, 0, 0
    # built in both memory modes wherever there IS a C-PML; the bare box has
    # no band, and hands the entry the empty tensor it always did
    if "band_idx" in c:
        fill(s, c, _FS_BANDS)
    else:
        put(s, _FS_BANDS, A.empty.to(torch.int64))
    # the shared elastic element kernels' interior fast-path flag; empty where
    # there is no C-PML, which means "every element is band"
    if "elem_band" in c:
        fill(s, c, ("elem_band",))
    else:
        put(s, ("elem_band",), lambda: A.empty.to(torch.int32))
    put(s, _FS_TAPES + _FS_RINGS_AC + _FS_RINGS_EL + _FS_SCR + _FS_FIN + _FS_CK
        + ("band_scale", "band_scale_e"), A.empty)
    if mode == "ring":
        nband = int(c["band_idx"].numel())
        nbn, nbe = int(c["band_node_idx"].numel()), int(c["band_elem_idx"].numel())
        # the seven NODE rings (5 fluid + 2 solid) share one amax table, the
        # six ELEMENT rings get one amax per band element; fin_* / ck_* stay
        # fp32 (they seed and re-anchor the march, they are not ring rows)
        s.band_kind = int(c.get("bs_kind", 0))
        s.band_kind_e = int(c.get("bs_kind_e", s.band_kind))
        lead = (B,) if B > 1 else ()
        if s.band_kind == 2:
            s.band_scale = torch.zeros(7, nt, *lead, dtype=torch.float32, device=dev)
        if s.band_kind_e == 2:
            s.band_scale_e = torch.zeros(6, nt, *lead, nbe, dtype=torch.float32, device=dev)
        rdt, rdt_e = _BAND_DTYPE[s.band_kind], _BAND_DTYPE[s.band_kind_e]
        put(s, _FS_RINGS_AC, lambda: torch.zeros(nt, B * nband, dtype=rdt, device=dev))
        put(s, ("band_ux", "band_uz"), lambda: torch.zeros(nt, B * nbn, dtype=rdt, device=dev))
        put(s, tuple("band_" + f for f in _FS_HIST),
            lambda: torch.zeros(nt, B * nbe * m, dtype=rdt_e, device=dev))
        put(s, _FS_SCR, lambda: A.zeros(B * E * m))
        put(s, _FS_FIN, A.zeros)
        s.ckpt_every = int(c.get("ckpt_every", 50))
        nck = nt // s.ckpt_every if s.ckpt_every > 0 else 0
        put(s, _FS_CK, lambda: torch.zeros(max(nck, 1), B * ng,
                                           dtype=torch.float32, device=dev))
    elif mode == "tape":
        # zero=False: written before read (NaN-poison; see Alloc.hist)
        put(s, ("save_chi", "save_ux", "save_uz"), lambda: A.hist(zero=False))
        put(s, tuple("save_" + f for f in _FS_MEM),
            (lambda: A.hist(zero=False)) if cp else A.empty)
        put(s, tuple("save_" + f for f in _FS_HIST),
            (lambda: A.hist(B * E * m, zero=False)) if cp else A.empty)
    dims(s, c, 2, has_cpml=int(cp))
    return check_filled(s)


def _fs_backward_input(c, mats, wavelet, carry, grads, B=1, own=None):
    """One domain's filled ``SemFluidSolidBackwardInput``. ``own`` (a strip's
    ownership mask) masks the three patterns the wavelet gradient reduces
    over; the reverse-march levels are seeded from the forward's finals when
    it is given (the stepped entries expect that; the fused C loop seeds
    them itself)."""
    import sweep._C as _C
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    E, m = c["n_elements"], c["ngll"] ** 2
    cp = bool(c.get("has_cpml", 1))
    A = Alloc(dev, B * ng, nt)
    b = _C.SemFluidSolidBackwardInput()
    fill(b, c, _FS_CONSTS)
    if cp:
        fill(b, c, _FS_CPML)
    else:
        put(b, _FS_CPML, A.empty)
    b.B = B
    for nm, t in zip(_FS_MATS, mats):
        setattr(b, nm, t)
    b.wavelet = wavelet
    for nm in ("pat_chi", "pat_x", "pat_z"):
        setattr(b, nm + "_own", (c[nm] * own).contiguous() if own is not None else A.empty)
    if "elem_band" in c:
        fill(b, c, ("elem_band",))
    else:
        put(b, ("elem_band",), lambda: A.empty.to(torch.int32))
    give(b, carry)
    b.grad_schi, b.grad_sx, b.grad_sz = grads
    put(b, _FS_LEVELS + _FS_R4 + _FS_REC + _FS_REC_R6 + _FS_REC_NODE, A.empty)
    if b.bs:
        put(b, _FS_LEVELS + _FS_REC_NODE, A.zeros)
        if own is not None:
            b.chi_hi, b.chi_lo = b.fin_chi.clone(), b.fin_chi_prev.clone()
            b.u_hi_x, b.u_hi_z = b.fin_ux.clone(), b.fin_uz.clone()
            b.u_lo_x, b.u_lo_z = b.fin_ux_prev.clone(), b.fin_uz_prev.clone()
        # the rebuilt fluid memories and the six solid scratch buffers,
        # each family contiguous for one memset per reverse step
        (b.r_psi_x, b.r_psi_z, b.r_zeta_x, b.r_zeta_z) = A.block(4)
        put(b, _FS_REC, lambda: A.zeros(B * E * m))
        (b.rec_rxx, b.rec_rxz, b.rec_rzx,
         b.rec_rzz, b.rec_rdx, b.rec_rdz) = A.block(6, B * E * m * 2)
    put(b, ("gA", "gB", "gchi_k", "lchidd", "s_ac", "Kchi", "yf", "zeros", "lchi_G",
            "lam1_x", "lam1_z", "lamn_x", "lamn_z", "g_x", "g_z",
            "gprev_x", "gprev_z", "y_x", "y_z", "adj_x", "adj_z",
            "tmp_x", "tmp_z", "sc_now_x", "sc_now_z",
            "sc_prev_a_x", "sc_prev_a_z", "sc_prev_b_x", "sc_prev_b_z"), A.zeros)
    put(b, tuple("l" + f for f in _FS_MEM) + tuple("l" + f + "_n" for f in _FS_MEM)
        + ("apx", "apz", "gx", "gz", "kx", "kz", "gax", "gaz", "w_sum", "gw",
           "gbx", "gbz", "gbkx", "gbkz", "gbax", "gbaz", "lapx", "lapz"),
        A.zeros if cp else A.empty)
    put(b, ("mu_xx", "mu_zx", "mu_xz", "mu_zz", "mu_dx", "mu_dz"),
        (lambda: A.zeros(B * E * m * 2)) if cp else A.empty)
    # per-shot gradients under the batch's one shared model (summed by the
    # caller), which keeps the kernels free of cross-shot atomics
    lead = (B,) if B > 1 else ()
    for nm in ("ac_inv_rho", "el_lam", "el_mu", "el_lam2mu", "el_w_rho_J"):
        setattr(b, "grad_" + nm, torch.zeros(*lead, *getattr(b, nm).shape,
                                             dtype=torch.float32, device=dev))
    put(b, ("grad_ac_inv_M", "grad_ac_kappa", "grad_el_inv_M"), A.zeros)
    b.grad_ac_inv_Mg = A.zeros() if cp else A.empty
    b.grad_wavelet = torch.zeros(*lead, nt, dtype=torch.float32, device=dev)
    dims(b, c, 2, has_cpml=int(cp))
    return check_filled(b)


def _fs_rotate_fwd(s):
    roll(s, ("chi_prev", "chi", "chi_next"))
    roll(s, ("ux_prev", "ux", "ux_next"))
    roll(s, ("uz_prev", "uz", "uz_next"))
    swap(s, [(f, f + "_n") for f in _FS_MEM])


def _fs_rotate_bwd(b, march):
    swap(b, [("gA", "gchi_k")] + [("l" + f, "l" + f + "_n") for f in _FS_MEM])
    if march:                       # (f^{k+1}, f^k) -> (f^k, f^{k-1})
        roll(b, ("chi_hi", "chi_lo", "chi_km1"))
        roll(b, ("u_hi_x", "u_lo_x", "u_km1_x"))
        roll(b, ("u_hi_z", "u_lo_z", "u_km1_z"))
    swap(b, [("lam1_x", "lamn_x"), ("lam1_z", "lamn_z"), ("gprev_x", "g_x"),
             ("gprev_z", "g_z"), ("sc_prev_b_x", "sc_prev_a_x"),
             ("sc_prev_b_z", "sc_prev_a_z")])


class _SEMFluidSolidC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ac_inv_rho, ac_inv_M, ac_kappa,
                el_lam, el_mu, el_lam2mu, el_inv_M, el_w_rho_J, wavelet, c):
        import sweep._C as _C
        B = int(c.get("B", 1))
        mats = (ac_inv_rho, ac_inv_M, ac_kappa, el_lam, el_mu, el_lam2mu,
                el_inv_M, el_w_rho_J)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        s = _fs_forward_input(c, mats, wavelet, memory_mode(c, need_bwd), B)
        seis_chi, seis_x, seis_z = _C.sem2d_fluid_solid_forward(s)
        if need_bwd:
            ctx.save_for_backward(*mats, wavelet)
            ctx.c, ctx.B = c, B
            ctx.carry = take(s, _FS_CARRY)
        return seis_chi, seis_x, seis_z

    @staticmethod
    def backward(ctx, g_chi, g_sx, g_sz):
        import sweep._C as _C
        *mats, wavelet = ctx.saved_tensors
        B, ng = ctx.B, ctx.c["n_global"]
        b = _fs_backward_input(ctx.c, mats, wavelet, ctx.carry,
                               tuple(g.contiguous().float() for g in (g_chi, g_sx, g_sz)), B)
        (g_air, g_aim, g_akp, g_lam, g_mu, g_l2m,
         g_eim, g_wrj, g_w) = _C.sem2d_fluid_solid_backward(b)
        if B > 1:                  # one model for every shot: sum the per-shot grads
            g_air, g_lam, g_mu, g_l2m, g_wrj = (
                g_air.sum(0), g_lam.sum(0), g_mu.sum(0), g_l2m.sum(0), g_wrj.sum(0))
            g_aim, g_akp, g_eim = (g.reshape(B, ng).sum(0) for g in (g_aim, g_akp, g_eim))
            if wavelet.dim() == 1:
                g_w = g_w.sum(0)
        return grads_or_none(mats + [wavelet],
                             (g_air, g_aim, g_akp, g_lam, g_mu, g_l2m, g_eim, g_wrj, g_w)
                             ) + (None,)


class _SEMFluidSolidDDC(torch.autograd.Function):
    """DD stepped coupled fluid-solid C-PML loop over every local part.

    Three assembly points per forward step group into TWO exchanges (the
    fluid grads ride with ifc_ac; the solid element rides with ifc_el), and
    the reverse step needs three. The seabed interface crossing a vertical
    cut needs no special machinery: the exchange sums the whole cut COLUMN,
    and the interface node on it is one of those nodes -- faces on either
    side contribute exactly as elements do (see ``_ifc_subset``).

    Differentiable per part: the eight materials + the shared wavelet.
    Boundary saving composes: each strip rings BOTH domains' bands and
    re-anchors on full-state checkpoints. It is the one path where BS ADDS
    an exchange -- the reverse march has to recompute the solid Acc, which
    is an assembly.
    """

    NMAT = len(_FS_MATS)

    @staticmethod
    def forward(ctx, wavelet, parts, exchange, bs, *mats):
        import sweep._C as _C
        nt, K = parts[0].c["nt"], len(_FS_MATS)
        need_bwd = wavelet.requires_grad or any(t.requires_grad for t in mats)
        mode = memory_mode(parts[0].c, need_bwd)
        inps = [_fs_forward_input(p.c, mats[K * i:K * i + K], wavelet, mode)
                for i, p in enumerate(parts)]

        def many(names):
            exchange.many([[getattr(s, nm) for s in inps] for nm in names])

        for n in range(nt):
            for s in inps:
                _C.sem2d_fluid_solid_dd_asm(s)
            many(("gx", "gz", "kx", "kz", "gax", "gaz", "s_ac"))
            for s in inps:
                _C.sem2d_fluid_solid_dd_mid(s, n)
            many(("f_x", "f_z", "Acc_x", "Acc_z"))
            for s in inps:
                _C.sem2d_fluid_solid_dd_step(s, n)
            for s in inps:
                _fs_rotate_fwd(s)
        if need_bwd:
            ctx.save_for_backward(wavelet, *mats)
            ctx.parts, ctx.exchange = parts, exchange
            ctx.carry = [take(s, _FS_CARRY) for s in inps]
        return tuple(t for s in inps for t in (s.seis_chi, s.seis_x, s.seis_z))

    @staticmethod
    def backward(ctx, *grad_seis):
        import sweep._C as _C
        from ..dd import _patch_dup_grad_syn
        wavelet, *mats = ctx.saved_tensors
        parts, exchange = ctx.parts, ctx.exchange
        nt, K = parts[0].c["nt"], len(_FS_MATS)
        gs = _patch_dup_grad_syn(parts, [g.contiguous().float() for g in grad_seis],
                                 nt, stride=3)
        bs_ = [_fs_backward_input(p.c, mats[K * i:K * i + K], wavelet, carry,
                                  gs[3 * i:3 * i + 3], own=p.own)
               for i, (p, carry) in enumerate(zip(parts, ctx.carry))]
        ring = bool(bs_[0].bs)

        def many(names):
            exchange.many([[getattr(b, nm) for b in bs_] for nm in names])

        for k in range(nt - 1, -1, -1):
            for b in bs_:
                _C.sem2d_fluid_solid_dd_bwd_recon(b, k)   # no-op unless bs
            for b in bs_:
                _C.sem2d_fluid_solid_dd_bwd_a(b, k)
            many(("lchidd", "s_ac"))
            for b in bs_:
                _C.sem2d_fluid_solid_dd_bwd_b(b, k)
            if k >= 1:
                many(("gx", "gz", "kx", "kz", "gax", "gaz"))
                for b in bs_:
                    _C.sem2d_fluid_solid_dd_bwd_c(b, k)
                many(("lchi_G", "lapx", "lapz"))
            if ring and k >= 1:
                # the reverse march recomputes the solid Acc, which is an
                # ASSEMBLY -- the one exchange boundary saving ADDS to this
                # path (the same four fields the forward exchanges after
                # dd_mid). bwd_d then reads the histories it just rebuilt.
                for b in bs_:
                    _C.sem2d_fluid_solid_dd_bwd_march_asm(b, k)
                many(("rec_f_x", "rec_f_z", "rec_Acc_x", "rec_Acc_z"))
                for b in bs_:
                    _C.sem2d_fluid_solid_dd_bwd_march_fin(b, k)
            for b in bs_:
                _C.sem2d_fluid_solid_dd_bwd_d(b, k)
            many(("sc_now_x", "sc_now_z", "sc_prev_a_x", "sc_prev_a_z"))
            for b in bs_:
                _C.sem2d_fluid_solid_dd_bwd_e(b, k)
            for b in bs_:
                _fs_rotate_bwd(b, ring and k >= 1)

        # element-local grads need no dedup (elements are uniquely owned);
        # the per-NODE ones carry the complete cut value on both strips
        grads = []
        for b, p in zip(bs_, parts):
            grads += [b.grad_ac_inv_rho, b.grad_ac_inv_M * p.own,
                      b.grad_ac_kappa * p.own, b.grad_el_lam, b.grad_el_mu,
                      b.grad_el_lam2mu, b.grad_el_inv_M * p.own, b.grad_el_w_rho_J]
        return dd_grads(ctx, bs_, grads)

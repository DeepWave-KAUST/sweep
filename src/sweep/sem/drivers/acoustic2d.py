"""The 2-D acoustic-potential C-PML drivers on CUDA: the fused single-domain
Function (``prop_c``'s ``_SEMAcousticCpmlC``, shot-batchable) and the strip
Function of ``dd`` (``_SEMAcousticCpmlDDC``). Both fill the same two pybind
structs and keep the same forward-to-backward carry; they differ in who runs
the time loop (the C entry, or Python over the stepped entries with an
exchange per assembly) and in how the gradients are reduced (over the shot
batch, or by cut ownership across strips).
"""

from __future__ import annotations

import torch

from ..cdrive import (Alloc, check_filled, dd_grads, dims, fill, give,
                      grads_or_none, memory_mode, put, roll, swap, take)

_AC2D_CONSTS = ("ibool", "D", "wJ", "w1d", "ixx", "ixz", "iex", "iez", "mask_e",
                "inv_Mg", "a_x", "b_x", "dbx_dx", "a_z", "b_z", "dbz_dz",
                "src_pattern", "rec_G", "rec_W", "dirichlet_idx",
                "band_idx", "band_idx_x", "band_idx_z")
_AC2D_MEM = ("psi_x", "psi_z", "zeta_x", "zeta_z")
_AC2D_STATE_TAPES = ("save_wavefield",) + tuple("save_" + f for f in _AC2D_MEM)
# the w_sum tape (see the C struct): the save-all mode keeps chi + w_sum
# instead of the five state tapes, and the backward rebuilds nothing
_AC2D_TAPES = _AC2D_STATE_TAPES + ("save_w_sum",)
_AC2D_RINGS = ("band_chi",) + tuple("band_" + f for f in _AC2D_MEM)
_AC2D_R4 = tuple("r_" + f for f in _AC2D_MEM)
# what the forward hands the backward, by name; the backward struct calls
# the chi tape save_chi
_AC2D_CARRY = (_AC2D_TAPES + _AC2D_RINGS
               + ("fin_chi", "fin_chi_prev", "band_scale", "band_kind",
                  "band_split", "bs"))
_AC2D_RENAME = (("save_wavefield", "save_chi"),)


def _ac2d_mode(c, need_bwd):
    """``(mode, wtape)`` of one run: what the forward keeps for the backward
    (:func:`~sweep.sem.cdrive.memory_mode`), and whether the save-all mode
    keeps the w_sum tape alone instead of the four state tapes."""
    from ..prop_c import _WSUM_TAPE_2D
    mode = memory_mode(c, need_bwd)
    return mode, mode == "tape" and _WSUM_TAPE_2D


def _ac2d_forward_input(c, mats, wavelet, mode, B=1, Bm=1, wtape=False):
    """One domain's filled ``SemAcousticCpmlForwardInput`` and its g6 block.

    ``B`` shots share one kernel sequence (``src_pattern`` ``(B, ng)``,
    ``seis`` ``(B, nrec, nt)``); ``Bm == B`` gives each its own materials,
    ``Bm == 1`` broadcasts one model. ``B == 1`` is the historical layout.
    """
    import sweep._C as _C
    from ..prop_c import _BS_RING_SPLIT, _band_alloc
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    A = Alloc(dev, B * ng, nt)
    s = _C.SemAcousticCpmlForwardInput()
    fill(s, c, _AC2D_CONSTS)
    s.B, s.Bm = B, Bm
    s.inv_rho, s.inv_M, s.kappa_node = mats
    s.wavelet = wavelet
    put(s, ("chi", "chi_prev", "chi_next") + _AC2D_MEM
        + tuple(f + "_n" for f in _AC2D_MEM), A.zeros)
    # never read or written by the 2-D entries (checked by NaN-poisoning them)
    put(s, ("apsi_x", "apsi_z", "w_sum"), A.empty)
    g6 = A.block(6)
    (s.gx, s.gz, s.kx, s.kz, s.gax, s.gaz) = g6
    s.seis = torch.zeros(*((B,) if B > 1 else ()), c["n_rec"], nt,
                         dtype=torch.float32, device=dev)
    s.bs, s.band_split = int(mode == "ring"), int(_BS_RING_SPLIT)
    s.band_kind, s.band_scale = 0, A.empty
    put(s, _AC2D_TAPES + _AC2D_RINGS + ("fin_chi", "fin_chi_prev"), A.empty)
    if mode == "ring":
        # chi rings on the union band, psi_d / zeta_d on their own axis's
        # band when split; bf16 / int8 storage divides the ring again
        s.band_kind = int(c.get("bs_kind", 0))
        ring, s.band_scale = _band_alloc(s.band_kind, 5, nt,
                                         int(c["band_idx"].numel()), dev, B)
        s.band_chi = ring()
        for nm in _AC2D_MEM:
            key = f"band_idx_{nm[-1]}" if s.band_split else "band_idx"
            setattr(s, "band_" + nm, ring(int(c[key].numel())))
        s.fin_chi, s.fin_chi_prev = A.zeros(), A.zeros()
    elif mode == "tape":
        # the w_sum tape: 2 x 4 B per node per step instead of 5 x 4, and the
        # backward's reverse step loses an element kernel (the chi tape stays:
        # grads_T images inv_rho from chi^k)
        put(s, ("save_wavefield", "save_w_sum") if wtape else _AC2D_STATE_TAPES,
            A.hist)
    dims(s, c, 2)
    return check_filled(s), g6


def _ac2d_backward_input(c, mats, wavelet, carry, grad_syn, B=1, Bm=1,
                         own=None):
    """One domain's filled ``SemAcousticCpmlBackwardInput`` and its g6 / l3
    blocks. ``own`` (a strip's ownership mask) masks the source pattern the
    wavelet gradient reduces over; the reverse-march levels are seeded from
    the forward's finals when it is given (the stepped entries expect that;
    the fused C loop seeds them itself)."""
    import sweep._C as _C
    dev, ng, nt = wavelet.device, c["n_global"], c["nt"]
    A = Alloc(dev, B * ng, nt)
    b = _C.SemAcousticCpmlBackwardInput()
    fill(b, c, _AC2D_CONSTS)
    b.B, b.Bm = B, Bm
    b.inv_rho, b.inv_M, b.kappa_node = mats
    b.wavelet = wavelet
    b.src_pattern_own = ((c["src_pattern"] * own).contiguous() if own is not None
                         else A.empty)
    give(b, carry, _AC2D_RENAME)
    b.grad_syn = grad_syn
    put(b, ("chi_hi", "chi_lo", "chi_km1") + _AC2D_R4, A.empty)
    if b.bs:
        b.chi_hi, b.chi_lo, b.chi_km1 = A.zeros(), A.zeros(), A.zeros()
        if own is not None:
            b.chi_hi, b.chi_lo = b.fin_chi.clone(), b.fin_chi_prev.clone()
        # re-zeroed every reverse step and never rotating: one memset for four
        (b.r_psi_x, b.r_psi_z, b.r_zeta_x, b.r_zeta_z) = A.block(4)
    g6 = A.block(6)
    (b.gx, b.gz, b.kx, b.kz, b.gax, b.gaz) = g6
    put(b, ("gA", "gB", "gchi_k", "apx", "apz", "w_sum", "gw",
            "gbx", "gbz", "gbkx", "gbkz", "gbax", "gbaz", "grad_inv_Mg")
        + tuple("l" + f for f in _AC2D_MEM)
        + tuple("l" + f + "_n" for f in _AC2D_MEM), A.zeros)
    # the transpose-gather outputs, re-zeroed twice per reverse step; the
    # rotating carries (gA, lpsi, lzeta) stay out of this block
    l3 = A.block(3)
    b.lchi_G, b.lapx, b.lapz = l3
    # per-shot gradients under a batch-shared model (summed by the caller);
    # the kernels stay free of cross-shot atomics, which keeps the B == 1
    # bit-path
    lead = (B,) if (B > 1 and Bm == 1) else ()
    for nm, t in (("grad_inv_rho", b.inv_rho), ("grad_inv_M", b.inv_M),
                  ("grad_kappa", b.kappa_node)):
        setattr(b, nm, torch.zeros(*lead, *t.shape, dtype=torch.float32, device=dev))
    b.grad_wavelet = torch.zeros(*((B,) if B > 1 else ()), nt,
                                 dtype=torch.float32, device=dev)
    dims(b, c, 2)
    return check_filled(b), g6, l3


def _ac2d_rotate_fwd(s):
    roll(s, ("chi_prev", "chi", "chi_next"))
    swap(s, [(f, f + "_n") for f in _AC2D_MEM])


def _ac2d_rotate_bwd(b):
    swap(b, [("gA", "gchi_k")] + [("l" + f, "l" + f + "_n") for f in _AC2D_MEM])


# --------------------------------------------------------------------------- #
# the fused single-domain Function: the C entry runs the whole time loop.
# Differentiable inputs: inv_rho, inv_M, kappa (material); inv_Mg is
# geometric (no model grad). PyTorch chains them through build() to vp/rho.
# --------------------------------------------------------------------------- #
class _SEMAcousticCpmlC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inv_rho, inv_M, kappa, wavelet, c):
        import sweep._C as _C
        B, Bm = int(c.get("B", 1)), int(c.get("Bm", 1))
        need_bwd = any(t.requires_grad for t in (inv_rho, inv_M, kappa, wavelet))
        mode, wtape = _ac2d_mode(c, need_bwd)
        s, _ = _ac2d_forward_input(c, (inv_rho, inv_M, kappa), wavelet,
                                   mode, B, Bm, wtape)
        seis = _C.sem2d_acoustic_cpml_forward(s)
        if need_bwd:
            ctx.save_for_backward(inv_rho, inv_M, kappa, wavelet)
            ctx.c, ctx.B, ctx.Bm = c, B, Bm
            ctx.carry = take(s, _AC2D_CARRY)
        return seis

    @staticmethod
    def backward(ctx, grad_seis):
        import sweep._C as _C
        inv_rho, inv_M, kappa, wavelet = ctx.saved_tensors
        B, Bm = ctx.B, ctx.Bm
        b, _, _ = _ac2d_backward_input(ctx.c, (inv_rho, inv_M, kappa), wavelet,
                                       ctx.carry, grad_seis.contiguous().float(),
                                       B, Bm)
        g_invM, g_invrho, g_kappa, _g_invMg, g_wav = _C.sem2d_acoustic_cpml_backward(b)
        if B > 1 and Bm == 1:       # one model for every shot: sum the per-shot grads
            g_invM, g_invrho, g_kappa = g_invM.sum(0), g_invrho.sum(0), g_kappa.sum(0)
        if B > 1 and wavelet.dim() == 1:
            g_wav = g_wav.sum(0)
        return grads_or_none((inv_rho, inv_M, kappa, wavelet),
                             (g_invrho, g_invM, g_kappa, g_wav)) + (None,)


# --------------------------------------------------------------------------- #
# the strip Function: Python steps the stepped entries over every local part
# with one exchange per assembly. Differentiable inputs: the wavelet plus each
# part's (inv_rho, inv_M, kappa) triple (sliced from the global build by the
# wrapper, so their gradients ride the slice chain back to the model).
# --------------------------------------------------------------------------- #
class _SEMAcousticCpmlDDC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wavelet, parts, exchange, bs, *mats):
        import sweep._C as _C
        nt = parts[0].c["nt"]
        need_bwd = wavelet.requires_grad or any(t.requires_grad for t in mats)
        mode, wtape = _ac2d_mode(parts[0].c, need_bwd)
        built = [_ac2d_forward_input(p.c, mats[3 * i:3 * i + 3], wavelet, mode,
                                     wtape=wtape)
                 for i, p in enumerate(parts)]
        inps, g6s = [s for s, _ in built], [g for _, g in built]

        for n in range(nt):
            for s in inps:
                _C.sem2d_acoustic_cpml_dd_grads(s)
            exchange.block(g6s)          # the step's ONE assembly
            for s in inps:
                _C.sem2d_acoustic_cpml_dd_step(s, n)
            for s in inps:
                _ac2d_rotate_fwd(s)
        if need_bwd:
            ctx.save_for_backward(wavelet, *mats)
            ctx.parts, ctx.exchange = parts, exchange
            ctx.wtape = wtape
            ctx.carry = [take(s, _AC2D_CARRY) for s in inps]
        return tuple(s.seis for s in inps)

    @staticmethod
    def backward(ctx, *grad_seis):
        import sweep._C as _C
        from ..dd import _patch_dup_grad_syn
        wavelet, *mats = ctx.saved_tensors
        parts, exchange = ctx.parts, ctx.exchange
        nt = parts[0].c["nt"]
        # a cut-column receiver is registered on both strips but the loss
        # reads the owned copy: hand the duplicate row the owner's gradient
        gs = _patch_dup_grad_syn(parts, [g.contiguous().float() for g in grad_seis], nt)
        built = [_ac2d_backward_input(p.c, mats[3 * i:3 * i + 3], wavelet, carry,
                                      g, own=p.own)
                 for i, (p, carry, g) in enumerate(zip(parts, ctx.carry, gs))]
        bs_ = [b for b, _, _ in built]
        g6s, l3s = [g for _, g, _ in built], [l for _, _, l in built]

        for k in range(nt - 1, -1, -1):
            for b in bs_:
                _C.sem2d_acoustic_cpml_dd_bwd_pre(b, k)
            if k >= 1:
                # under the w_sum tape nothing is rebuilt from the forward
                # state, so the six assembled fields are never produced
                if not ctx.wtape:
                    exchange.block(g6s)
                for b in bs_:
                    _C.sem2d_acoustic_cpml_dd_bwd_mid(b, k)
                exchange.block(l3s)
            for b in bs_:
                _C.sem2d_acoustic_cpml_dd_bwd_end(b, k)
            for b in bs_:
                _ac2d_rotate_bwd(b)
                if b.bs and k >= 1:   # (chi^{k+1}, chi^k) -> (chi^k, chi^{k-1})
                    roll(b, ("chi_hi", "chi_lo", "chi_km1"))

        # per-node gradients are complete on both sides of a cut after the
        # adjoint exchange: dedup by ownership (grad_inv_rho is per element,
        # uniquely owned); the wavelet's is a global sum
        grads = []
        for b, p in zip(bs_, parts):
            grads += [b.grad_inv_rho, b.grad_inv_M * p.own, b.grad_kappa * p.own]
        return dd_grads(ctx, bs_, grads)

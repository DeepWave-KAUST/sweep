"""The 2-D Biot driver on CUDA: BiotVSSEM2DDeformed and BiotAnisoPermSEM2DDeformed
(``sem2d_biot_forward`` / ``_backward``, ``sem2d/biot2d.cu``), bare or with the C-PML,
save-all or in chunks (:mod:`.vs_chunks`).

The leaves the kernels return gradients for, in ``_LEAVES`` order: the inverse
masses ``inv_M`` (solid), ``inv_Mp`` (pressure), ``inv_M1`` (an injected pressure),
``inv_Mm`` (an injected flux), the frame's ``lam_d`` and ``mu``, ``alpha = C / M``,
the flux copies' ``wr`` and ``share``, the drag's exact step ``dragE`` / ``dragP``
(the equation's ``drag_operators`` for this dt), the relaxed drag ``b_xx``, ``b_xz``,
``b_zz`` the viscous memory runs on, and ViscoBiotVSSEM2DDeformed's GSLS frame
``ptau`` / ``mtau`` ``(n_sls, E, ngll^2)`` (empty without it). Autograd takes them on to the eight
Biot coefficients through the equation's build and its matrix exponentials. The
records are the eight receiver fields ``(8, n_rec, nt)``: v_x v_z p q_x q_z
sigma_xx sigma_zz sigma_xz.
"""

from __future__ import annotations

import torch

from ..cdrive import (Alloc, check_filled, dims, fill, give, grads_or_none,
                      need_backward, put, take)
from .vs_chunks import backward_chunks, chunk_of, forward_chunks

_GEOM = ("ibool", "D", "wJ", "ixx", "ixz", "iex", "iez", "qmask", "pin_n", "slot_elem",
         "dirichlet_idx", "ifc_idx", "ifc_w", "ifc_nx", "ifc_nz",
         "pat_vx", "pat_vz", "pat_p", "pat_qx", "pat_qz", "rec_G", "rec_W", "rec_Ge")
_CPML = ("a_x", "b_x", "a_z", "b_z", "a_n")
_LEAVES = ("inv_M", "inv_Mp", "inv_M1", "inv_Mm", "lam_d", "mu", "alpha", "wr", "share",
           "dragE", "dragP", "b_xx", "b_xz", "b_zz", "ptau", "mtau")
_HIST = ("save_f", "save_qs", "save_rs", "save_rhs", "save_gv", "save_q")


def _flags(s, c):
    s.kmax, s.tensor_drag = c["kmax"], c["tensor_drag"]
    s.velocity_mode, s.viscous = c["velocity_mode"], c["viscous"]
    s.decay, s.gain = c["decay"], c["gain"]


def _gsls(s, c, A, leaves, backward):
    """ViscoBiotVSSEM2DDeformed's frame memories (``n_sls`` = 0: none)."""
    L = int(c.get("nsls", 0))
    s.nsls = L
    s.gsls_w = c["gsls_w"] if L else A.empty
    Em = c["n_elements"] * c["ngll"] ** 2
    mem = A.block(3 * L, Em) if L else A.empty
    if backward:
        s.lrmem = mem
    else:
        s.rmem = mem


def _forward_input(c, A, leaves, wavelet, hist):
    """The forward struct, with the histories of ``hist`` steps (``None``: none)."""
    import sweep._C as _C
    E, m, ng, k = c["n_elements"], c["ngll"] ** 2, c["n_global"], c["kmax"]
    Em = E * m
    s = _C.Sem2dBiotForwardInput()
    fill(s, c, _GEOM)
    if c["has_cpml"]:
        fill(s, c, _CPML)
        s.mp, s.mv = A.block(2, Em), A.block(4, Em)
        s.msig, s.mq = A.block(4, ng), A.block(2, ng)
        s.P, s.Pq = A.block(4, ng), A.block(2, ng)
    else:
        put(s, _CPML + ("mp", "mv", "msig", "mq", "P", "Pq"), A.empty)
    for nm, t in zip(_LEAVES, leaves):
        setattr(s, nm, t)
    _gsls(s, c, A, leaves, backward=False)
    s.wavelet = wavelet
    put(s, ("vx", "vz", "p", "Rx", "Rz", "adiv", "divq", "dp"), A.zeros)
    s.q, s.sig, s.r = A.block(2, Em), A.block(3, Em), A.block(2, Em)
    s.gp, s.f = A.block(2, Em), A.block(2, ng)
    s.seis = torch.zeros(8, c["n_rec"], c["nt"], dtype=torch.float32, device=A.dev)
    if hist is None:
        put(s, _HIST, A.empty)
    else:
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=A.dev)
        s.save_f, s.save_qs, s.save_rs = z(hist, 2, ng), z(hist, 2, ng, k), z(hist, 2, ng, k)
        s.save_rhs, s.save_gv = z(hist, ng), z(hist, 4, Em)
        s.save_q = z(hist, 2, Em) if c["viscous"] else A.empty
    _flags(s, c)
    dims(s, c, 2, batch=1)
    return check_filled(s)


class _SEMBiot2DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, wavelet, *leaves):
        import sweep._C as _C
        need_bwd = need_backward(c)
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        K = chunk_of(c, need_bwd)
        s = _forward_input(c, A, leaves, wavelet, c["nt"] if need_bwd and K is None else None)
        if K is None:
            _C.sem2d_biot_forward(s)
            carry = take(s, _HIST)
        else:
            names, ckpts = forward_chunks(_C.sem2d_biot_forward, s, c["nt"], K)
            carry = {"K": K, "names": names, "ckpts": ckpts}
        if need_bwd:
            ctx.save_for_backward(wavelet, *leaves)
            ctx.c, ctx.carry = c, carry
        return s.seis

    @staticmethod
    def backward(ctx, gseis):
        import sweep._C as _C
        wavelet, *leaves = ctx.saved_tensors
        c, carry = ctx.c, ctx.carry
        E, m, ng = c["n_elements"], c["ngll"] ** 2, c["n_global"]
        Em = E * m
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.Sem2dBiotBackwardInput()
        fill(b, c, _GEOM)
        if c["has_cpml"]:
            fill(b, c, _CPML)
            b.lmp, b.lmv = A.block(2, Em), A.block(4, Em)
            b.lmsig, b.lmq = A.block(4, ng), A.block(2, ng)
            b.lP, b.lPq = A.block(4, ng), A.block(2, ng)
        else:
            put(b, _CPML + ("lmp", "lmv", "lmsig", "lmq", "lP", "lPq"), A.empty)
        for nm, t in zip(_LEAVES, leaves):
            setattr(b, nm, t)
            setattr(b, "grad_" + nm, torch.zeros_like(t))
        _gsls(b, c, A, leaves, backward=True)
        b.wavelet = wavelet
        b.grad_seis = gseis.contiguous().float()
        put(b, ("lvx", "lvz", "lp", "lRx", "lRz", "ldp", "ldivq"), A.zeros)
        b.lq, b.lsig, b.lr = A.block(2, Em), A.block(3, Em), A.block(2, Em)
        b.lf, b.lgp, b.lgv, b.lg = A.block(2, ng), A.block(2, Em), A.block(4, Em), A.block(2, Em)
        b.grad_wavelet = A.zeros(c["nt"])
        _flags(b, c)
        dims(b, c, 2, batch=1)
        if "ckpts" in carry:
            K = carry["K"]
            f = _forward_input(c, A, tuple(leaves), wavelet, K)
            give(b, take(f, _HIST))
            backward_chunks(_C.sem2d_biot_forward, _C.sem2d_biot_backward, f, check_filled(b),
                            c["nt"], K, carry["names"], carry["ckpts"])
        else:
            give(b, carry)
            _C.sem2d_biot_backward(check_filled(b))
        grads = grads_or_none((wavelet,) + tuple(leaves),
                              (b.grad_wavelet,) + tuple(getattr(b, "grad_" + nm)
                                                        for nm in _LEAVES))
        return (None,) + grads

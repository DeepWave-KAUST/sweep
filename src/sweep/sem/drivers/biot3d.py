"""The 3-D Biot driver on CUDA: BiotVSSEM3DDeformed (``sem3d_biot_forward`` /
``_backward``, ``sem3d/biot3d.cu``), bare or with the C-PML, save-all or in chunks
(:mod:`.vs_chunks`). :mod:`.biot2d` with y: the leaves are the inverse masses, the
frame's ``lam_d`` and ``mu``, ``alpha``, the isotropic drag ``b`` (the viscous
memory's), the flux copies' ``wr`` and ``share`` in the eager class's perm order and
the groups' drag blocks ``dragE`` / ``dragP`` flattened one after the other. The
records are the thirteen receiver fields ``(13, n_rec, nt)``: v_x v_y v_z p q_x q_y
q_z and the six stresses.
"""

from __future__ import annotations

import torch

from ..cdrive import (Alloc, check_filled, dims, fill, give, grads_or_none,
                      need_backward, put, take)
from .vs_chunks import backward_chunks, chunk_of, forward_chunks

_METRIC = ("ixx", "ixy", "ixz", "iex", "iey", "iez", "igx", "igy", "igz")
_GEOM = ("ibool", "D", "wJ") + _METRIC + (
    "qmask", "pin_n", "perm", "node_order", "dirichlet_idx",
    "ifc_idx", "ifc_w", "ifc_nx", "ifc_ny", "ifc_nz",
    "pat_vx", "pat_vy", "pat_vz", "pat_p", "pat_qx", "pat_qy", "pat_qz",
    "rec_G", "rec_W", "rec_Ge")
_CPML = ("a_x", "b_x", "a_y", "b_y", "a_z", "b_z", "a_n")
_LEAVES = ("inv_M", "inv_Mp", "inv_M1", "inv_Mm", "lam_d", "mu", "alpha", "b", "wr",
           "share", "dragE", "dragP")
_HIST = ("save_f", "save_qs", "save_rs", "save_rhs", "save_gv", "save_q")
_GROUPS = ("g0_k", "g0_n", "g1_k", "g1_n", "g2_k", "g2_n", "g3_k", "g3_n")


def _flags(s, c):
    s.velocity_mode, s.viscous = c["velocity_mode"], c["viscous"]
    s.decay, s.gain = c["decay"], c["gain"]
    for nm in _GROUPS:
        setattr(s, nm, c[nm])


def _forward_input(c, A, leaves, wavelet, hist):
    """The forward struct, with the histories of ``hist`` steps (``None``: none)."""
    import sweep._C as _C
    E, m, ng = c["n_elements"], c["ngll"] ** 3, c["n_global"]
    Em = E * m
    s = _C.Sem3dBiotForwardInput()
    fill(s, c, _GEOM)
    if c["has_cpml"]:
        fill(s, c, _CPML)
        s.mp, s.mv = A.block(3, Em), A.block(9, Em)
        s.msig, s.mq = A.block(9, ng), A.block(3, ng)
        s.P, s.Pq = A.block(9, ng), A.block(3, ng)
    else:
        put(s, _CPML + ("mp", "mv", "msig", "mq", "P", "Pq"), A.empty)
    for nm, t in zip(_LEAVES, leaves):
        setattr(s, nm, t)
    s.wavelet = wavelet
    put(s, ("vx", "vy", "vz", "p", "Rx", "Ry", "Rz", "adiv", "divq", "dp"), A.zeros)
    s.q, s.sig, s.r = A.block(3, Em), A.block(6, Em), A.block(3, Em)
    s.gp, s.f = A.block(3, Em), A.block(3, ng)
    s.seis = torch.zeros(13, c["n_rec"], c["nt"], dtype=torch.float32, device=A.dev)
    if hist is None:
        put(s, _HIST, A.empty)
    else:
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=A.dev)
        s.save_f, s.save_qs, s.save_rs = z(hist, 3, ng), z(hist, 3, Em), z(hist, 3, Em)
        s.save_rhs, s.save_gv = z(hist, ng), z(hist, 9, Em)
        s.save_q = z(hist, 3, Em) if c["viscous"] else A.empty
    _flags(s, c)
    dims(s, c, 3, batch=1)
    return check_filled(s)


class _SEMBiot3DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, wavelet, *leaves):
        import sweep._C as _C
        need_bwd = need_backward(c)
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        K = chunk_of(c, need_bwd)
        s = _forward_input(c, A, leaves, wavelet, c["nt"] if need_bwd and K is None else None)
        if K is None:
            _C.sem3d_biot_forward(s)
            carry = take(s, _HIST)
        else:
            names, ckpts = forward_chunks(_C.sem3d_biot_forward, s, c["nt"], K)
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
        E, m, ng = c["n_elements"], c["ngll"] ** 3, c["n_global"]
        Em = E * m
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.Sem3dBiotBackwardInput()
        fill(b, c, _GEOM)
        if c["has_cpml"]:
            fill(b, c, _CPML)
            b.lmp, b.lmv = A.block(3, Em), A.block(9, Em)
            b.lmsig, b.lmq = A.block(9, ng), A.block(3, ng)
            b.lP, b.lPq = A.block(9, ng), A.block(3, ng)
        else:
            put(b, _CPML + ("lmp", "lmv", "lmsig", "lmq", "lP", "lPq"), A.empty)
        for nm, t in zip(_LEAVES, leaves):
            setattr(b, nm, t)
            setattr(b, "grad_" + nm, torch.zeros_like(t))
        b.wavelet = wavelet
        b.grad_seis = gseis.contiguous().float()
        put(b, ("lvx", "lvy", "lvz", "lp", "lRx", "lRy", "lRz", "ldp", "ldivq"), A.zeros)
        b.lq, b.lsig, b.lr = A.block(3, Em), A.block(6, Em), A.block(3, Em)
        b.lf, b.lgp, b.lgv, b.lg = A.block(3, ng), A.block(3, Em), A.block(9, Em), A.block(3, Em)
        b.grad_wavelet = A.zeros(c["nt"])
        _flags(b, c)
        dims(b, c, 3, batch=1)
        if "ckpts" in carry:
            K = carry["K"]
            f = _forward_input(c, A, tuple(leaves), wavelet, K)
            give(b, take(f, _HIST))
            backward_chunks(_C.sem3d_biot_forward, _C.sem3d_biot_backward, f, check_filled(b),
                            c["nt"], K, carry["names"], carry["ckpts"])
        else:
            give(b, carry)
            _C.sem3d_biot_backward(check_filled(b))
        grads = grads_or_none((wavelet,) + tuple(leaves),
                              (b.grad_wavelet,) + tuple(getattr(b, "grad_" + nm)
                                                        for nm in _LEAVES))
        return (None,) + grads

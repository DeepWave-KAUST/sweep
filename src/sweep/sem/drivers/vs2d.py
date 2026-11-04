"""The 2-D velocity-stress drivers on CUDA: ElasticVSSEM2DDeformed and its VTI,
TTI and visco-elastic subclasses (``sem2d_elastic_vs_forward`` / ``_backward``,
``sem2d/vs2d.cu``), bare or with the C-PML, the natural free surface, save-all.

One autograd Function serves the four classes; ``c["mode"]`` names what the
``mats`` leaves are:

* ``"iso"``:   ``(lam, mu, lam2mu)``                       the Lame set;
* ``"aniso"``: ``(cij,)``  ``(4 or 6, E, ngll^2)``          the VTI / TTI stiffnesses;
* ``"visco"``: ``(lam, mu, lam2mu, ptau, mtau)``           the unrelaxed Lame set and
  the relaxed moduli times the strengths, ``(n_sls, E, ngll^2)`` each, with the
  trapezoidal weights in ``c["gsls_w"]``.

The save-all keeps the velocity history and the four stretched velocity gradients
per step, what the moduli's imaging and the adjoint need.
"""

from __future__ import annotations

import torch

from ..cdrive import Alloc, check_filled, dims, fill, give, grads_or_none, put, take
from .vs_chunks import backward_chunks, chunk_of, forward_chunks

_VS2D_GEOM = ("ibool", "D", "wJ", "ixx", "ixz", "iex", "iez", "pat_x", "pat_z",
              "rec_G", "rec_W")
_VS2D_CPML = ("a_x", "b_x", "a_z", "b_z", "a_n")
_VS2D_CARRY = ("save_vx", "save_vz", "save_gv")
_LAME = ("lam", "mu", "lam2mu")
_GSLS = ("ptau", "mtau", "gsls_w")


def _record(A, c, *names):
    return {nm: torch.zeros(c["n_rec"], A.nt, dtype=torch.float32, device=A.dev)
            for nm in names}


def _moduli(s, c, A, mats, backward):
    """The constitutive fields of either struct from the ``mats`` leaves (see the
    module doc), empties for the rest; on the backward the gradient buffers."""
    mode = c["mode"]
    s.aniso, s.nsls = 0, 0
    put(s, ("cij",) + _GSLS, A.empty)
    put(s, _LAME, A.empty)
    if backward:
        put(s, ("grad_cij", "grad_ptau", "grad_mtau", "lrmem") + tuple(
            "grad_" + n for n in _LAME), A.empty)
    else:
        s.rmem = A.empty
    if mode == "aniso":
        (cij,) = mats
        s.aniso, s.cij = int(cij.shape[0]), cij
        if backward:
            s.grad_cij = torch.zeros_like(cij)
        return
    lam, mu, lam2mu = mats[:3]
    s.lam, s.mu, s.lam2mu = lam, mu, lam2mu
    if backward:
        s.grad_lam, s.grad_mu, s.grad_lam2mu = (torch.zeros_like(t) for t in mats[:3])
    if mode == "visco":
        ptau, mtau = mats[3:]
        L, E, m = int(ptau.shape[0]), c["n_elements"], c["ngll"] ** 2
        s.nsls, s.ptau, s.mtau, s.gsls_w = L, ptau, mtau, c["gsls_w"]
        if backward:
            s.lrmem = A.block(3 * L, E * m)
            s.grad_ptau, s.grad_mtau = torch.zeros_like(ptau), torch.zeros_like(mtau)
        else:
            s.rmem = A.block(3 * L, E * m)


def _cpml(s, c, A, scratch):
    """The C-PML coefficients and the two nodal blocks ``scratch`` names (the
    divergence by direction and its memories), or empties on a bare run."""
    if c["has_cpml"]:
        fill(s, c, _VS2D_CPML)
        for nm in scratch:
            setattr(s, nm, A.block(4, c["n_global"]))
    else:
        put(s, _VS2D_CPML + scratch, A.empty)


def _forward_input(c, A, mats, inv_M, wavelet, hist):
    """The forward struct: geometry, C-PML, moduli, state, records, and the
    histories of ``hist`` steps (``None``: none; see :mod:`.vs_chunks`)."""
    import sweep._C as _C
    E, m, ng = c["n_elements"], c["ngll"] ** 2, c["n_global"]
    s = _C.Sem2dVsForwardInput()
    fill(s, c, _VS2D_GEOM)
    _cpml(s, c, A, ("P", "msig"))
    _moduli(s, c, A, mats, backward=False)
    s.inv_M, s.wavelet = inv_M, wavelet
    put(s, ("vx", "vz", "Rx", "Rz"), A.zeros)
    s.sig, s.mvel = A.block(3, E * m), A.block(4, E * m)
    give(s, _record(A, c, "seis_x", "seis_z"))
    if hist is None:
        put(s, ("save_vx", "save_vz", "save_gv"), A.empty)
    else:               # v: the start of the run, then every step; gv: every step
        put(s, ("save_vx", "save_vz"), lambda: torch.zeros(
            hist + 1, ng, dtype=torch.float32, device=A.dev))
        s.save_gv = torch.zeros(hist, 4, E * m, dtype=torch.float32, device=A.dev)
    dims(s, c, 2, batch=1)
    return check_filled(s)


class _SEMElasticVS2DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inv_M, wavelet, c, *mats):
        import sweep._C as _C
        need_bwd = any(t.requires_grad for t in (inv_M, wavelet) + mats)
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        K = chunk_of(c, need_bwd)
        s = _forward_input(c, A, mats, inv_M, wavelet,
                           c["nt"] if need_bwd and K is None else None)
        if K is None:
            seis = _C.sem2d_elastic_vs_forward(s)
            carry = take(s, _VS2D_CARRY)
        else:
            names, ckpts = forward_chunks(_C.sem2d_elastic_vs_forward, s, c["nt"], K)
            seis = (s.seis_x, s.seis_z)
            carry = {"K": K, "names": names, "ckpts": ckpts}
        if need_bwd:
            ctx.save_for_backward(inv_M, wavelet, *mats)
            ctx.c, ctx.carry = c, carry
        return seis

    @staticmethod
    def backward(ctx, gsx, gsz):
        import sweep._C as _C
        inv_M, wavelet, *mats = ctx.saved_tensors
        c, carry = ctx.c, ctx.carry
        E, m = c["n_elements"], c["ngll"] ** 2
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.Sem2dVsBackwardInput()
        fill(b, c, _VS2D_GEOM)
        _cpml(b, c, A, ("lP", "lmsig"))
        _moduli(b, c, A, mats, backward=True)
        b.inv_M, b.wavelet = inv_M, wavelet
        b.grad_sx, b.grad_sz = (g.contiguous().float() for g in (gsx, gsz))
        put(b, ("lvx", "lvz", "lRx", "lRz", "grad_inv_M"), A.zeros)
        b.lsig, b.lmvel = A.block(3, E * m), A.block(4, E * m)
        b.grad_wavelet = A.zeros(c["nt"])
        dims(b, c, 2, batch=1)
        if "ckpts" in carry:
            K = carry["K"]
            f = _forward_input(c, A, tuple(mats), inv_M, wavelet, K)
            give(b, take(f, _VS2D_CARRY))
            backward_chunks(_C.sem2d_elastic_vs_forward, _C.sem2d_elastic_vs_backward,
                            f, check_filled(b), c["nt"], K, carry["names"], carry["ckpts"])
        else:
            give(b, carry)
            _C.sem2d_elastic_vs_backward(check_filled(b))
        mode = c["mode"]
        if mode == "aniso":
            g_mats = (b.grad_cij,)
        elif mode == "visco":
            g_mats = (b.grad_lam, b.grad_mu, b.grad_lam2mu, b.grad_ptau, b.grad_mtau)
        else:
            g_mats = (b.grad_lam, b.grad_mu, b.grad_lam2mu)
        grads = grads_or_none((inv_M, wavelet) + tuple(mats),
                              (b.grad_inv_M, b.grad_wavelet) + g_mats)
        return grads[:2] + (None,) + grads[2:]

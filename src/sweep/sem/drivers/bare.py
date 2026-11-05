"""The single-domain drivers without a C-PML: 2-D acoustic potential and
elastic, 3-D acoustic (flat Hex8 and deformed Hex27), 3-D elastic
displacement and 3-D velocity-stress elastic (bare or simple CPML). The C
entry runs the whole time loop; the backward replays it from the save-all
tapes (the flat and deformed 3-D acoustic can ring the two final levels
instead and march the field back).
"""

from __future__ import annotations

import torch

from ..cdrive import (METRIC_NAMES, Alloc, check_filled, dims, fill, give,
                      grads_or_none,
                      memory_mode, put, take)
from .vs_chunks import backward_chunks, chunk_of, forward_chunks


def _record(A, c, *names):
    return {nm: torch.zeros(c["n_rec"], A.nt, dtype=torch.float32, device=A.dev)
            for nm in names}


# --------------------------------------------------------------------------- #
# 2-D acoustic potential: inv_M, inv_rho and the wavelet carry gradients
# --------------------------------------------------------------------------- #
_AC2D_BARE = ("ibool", "D", "wJ", "ixx", "ixz", "iex", "iez", "mask_e",
              "dirichlet_idx", "src_pattern", "rec_G", "rec_W")


class _SEMAcousticC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inv_M, inv_rho, wavelet, c):
        import sweep._C as _C
        need_bwd = any(t.requires_grad for t in (inv_M, inv_rho, wavelet))
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        s = _C.SemForwardInput()
        fill(s, c, _AC2D_BARE)
        s.inv_rho, s.inv_M, s.wavelet = inv_rho, inv_M, wavelet
        put(s, ("chi", "chi_prev", "chi_next", "K"), A.zeros)
        give(s, _record(A, c, "seis"))
        s.save_wavefield = A.hist() if need_bwd else A.empty
        dims(s, c, 2)
        seis = _C.sem2d_acoustic_forward(check_filled(s))
        if need_bwd:
            ctx.save_for_backward(inv_M, inv_rho, wavelet)
            ctx.c, ctx.carry = c, take(s, ("save_wavefield",))
        return seis

    @staticmethod
    def backward(ctx, grad_seis):
        import sweep._C as _C
        inv_M, inv_rho, wavelet = ctx.saved_tensors
        c = ctx.c
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.SemBackwardInput()
        fill(b, c, _AC2D_BARE)
        b.inv_rho, b.inv_M, b.wavelet = inv_rho, inv_M, wavelet
        give(b, ctx.carry)
        b.grad_syn = grad_seis.contiguous().float()
        put(b, ("a_p1", "a_p2", "a_m", "tmp", "Ktmp", "Ku", "adj_src", "grad_inv_M"),
            A.zeros)
        b.grad_inv_rho = torch.zeros_like(inv_rho)
        b.grad_wavelet = A.zeros(c["nt"])
        dims(b, c, 2)
        g_invM, g_invrho, g_wav = _C.sem2d_acoustic_backward(check_filled(b))
        return grads_or_none((inv_M, inv_rho, wavelet), (g_invM, g_invrho, g_wav)) + (None,)


# --------------------------------------------------------------------------- #
# 2-D elastic (two displacement components)
# --------------------------------------------------------------------------- #
_EL2D_BARE = ("ibool", "D", "wJ", "ixx", "ixz", "iex", "iez", "mask_e",
              "pat_x", "pat_z", "rec_G", "rec_W")


class _SEMElasticC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, inv_M, wavelet, c):
        import sweep._C as _C
        mats = (lam, mu, lam2mu, inv_M)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        s = _C.SemElasticForwardInput()
        fill(s, c, _EL2D_BARE)
        s.lam, s.mu, s.lam2mu, s.inv_M, s.wavelet = lam, mu, lam2mu, inv_M, wavelet
        put(s, ("ux", "uz", "ux_prev", "uz_prev", "ux_next", "uz_next", "Kux", "Kuz"),
            A.zeros)
        give(s, _record(A, c, "seis_x", "seis_z"))
        put(s, ("save_ux", "save_uz"), A.hist if need_bwd else A.empty)
        dims(s, c, 2)
        seis_x, seis_z = _C.sem2d_elastic_forward(check_filled(s))
        if need_bwd:
            ctx.save_for_backward(*mats, wavelet)
            ctx.c, ctx.carry = c, take(s, ("save_ux", "save_uz"))
        return seis_x, seis_z

    @staticmethod
    def backward(ctx, grad_sx, grad_sz):
        import sweep._C as _C
        *mats, wavelet = ctx.saved_tensors
        c = ctx.c
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.SemElasticBackwardInput()
        fill(b, c, _EL2D_BARE)
        b.lam, b.mu, b.lam2mu, b.inv_M = mats
        b.wavelet = wavelet
        give(b, ctx.carry)
        b.grad_sx, b.grad_sz = grad_sx.contiguous().float(), grad_sz.contiguous().float()
        put(b, ("ax1", "az1", "ax2", "az2", "axm", "azm", "tmp_x", "tmp_z",
                "Kx", "Kz", "adj_x", "adj_z", "grad_inv_M"), A.zeros)
        for nm, t in zip(("grad_lam", "grad_mu", "grad_lam2mu"), mats):
            setattr(b, nm, torch.zeros_like(t))
        b.grad_wavelet = A.zeros(c["nt"])
        dims(b, c, 2)
        grads = _C.sem2d_elastic_backward(check_filled(b))
        return grads_or_none(mats + [wavelet], grads) + (None,)


# --------------------------------------------------------------------------- #
# 3-D acoustic, velocity form: u_tt = c^2 Laplace u. The stiffness K0 is
# c^2-free -- flat Hex8 with the separable weights w3 G_axis (geometry,
# constants), or deformed Hex27 with w_J and the 3x3 inverse-Jacobian metric --
# and the kernels multiply K0 u by the per-node c^2 (``c2_node``, the
# equation's one-copy ``vp_squared_node``), as the C-PML strip kernels do. The
# only material input is c2_node; its gradient comes back in grad_c2_jx.
# --------------------------------------------------------------------------- #
_AC3D_BARE = ("ibool", "D", "dirichlet_idx", "src_pattern", "rec_G", "rec_W")
_AC3D_METRIC = ("wJ",) + METRIC_NAMES
_AC3D_WEIGHTS = (("c2_jx", "w_jx"), ("c2_jy", "w_jy"), ("c2_jz", "w_jz"))
_AC3D_BARE_CARRY = ("save_wavefield", "fin_u", "fin_u_prev", "bs")


def _ac3d_geometry(s, c, A):
    """The stiffness geometry on both structs: the metric (deformed) or the
    c^2-free axis weights (flat); the other set is empty."""
    s.deformed = int(c.get("deformed", 0))
    if s.deformed:
        fill(s, c, _AC3D_METRIC)
        put(s, tuple(f for f, _ in _AC3D_WEIGHTS), A.empty)
    else:
        put(s, _AC3D_METRIC, A.empty)
        for f, k in _AC3D_WEIGHTS:
            setattr(s, f, c[k])


def _ac3d_bare_forward_input(c, c2_node, inv_M, wavelet, mode):
    import sweep._C as _C
    A = Alloc(wavelet.device, c["n_global"], c["nt"])
    s = _C.Sem3dForwardInput()
    fill(s, c, _AC3D_BARE)
    _ac3d_geometry(s, c, A)
    s.c2_node, s.inv_M, s.wavelet = c2_node, inv_M, wavelet
    put(s, ("u", "u_prev", "u_next", "K"), A.zeros)
    give(s, _record(A, c, "seis"))
    s.bs = int(mode == "ring")
    s.save_wavefield = A.hist() if mode == "tape" else A.empty
    put(s, ("fin_u", "fin_u_prev"), A.zeros if mode == "ring" else A.empty)
    dims(s, c, 3, batch=1)
    return check_filled(s)


def _ac3d_bare_backward_input(c, c2_node, inv_M, wavelet, carry, grad_syn):
    import sweep._C as _C
    A = Alloc(wavelet.device, c["n_global"], c["nt"])
    b = _C.Sem3dBackwardInput()
    fill(b, c, _AC3D_BARE)
    _ac3d_geometry(b, c, A)
    b.c2_node, b.inv_M, b.wavelet = c2_node, inv_M, wavelet
    b.grad_c2_jx = torch.zeros_like(c2_node)
    put(b, ("grad_c2_jy", "grad_c2_jz"), A.empty)
    give(b, carry)
    b.grad_syn = grad_syn
    put(b, ("u_hi", "u_lo", "u_km1"), A.zeros if b.bs else A.empty)
    put(b, ("a_p1", "a_p2", "a_m", "tmp", "Ktmp", "Ku", "adj_src", "grad_inv_M"),
        A.zeros)
    b.grad_wavelet = A.zeros(c["nt"])
    dims(b, c, 3, batch=1)
    return check_filled(b)


class _SEMAcoustic3DC(torch.autograd.Function):
    """AcousticSEM3D / AcousticSEM3DDeformed without a C-PML (``deformed`` in
    the consts picks the stiffness)."""
    @staticmethod
    def forward(ctx, c2_node, inv_M, wavelet, c):
        import sweep._C as _C
        need_bwd = any(t.requires_grad for t in (c2_node, inv_M, wavelet))
        s = _ac3d_bare_forward_input(c, c2_node, inv_M, wavelet,
                                     memory_mode(c, need_bwd))
        seis = _C.sem3d_acoustic_forward(s)
        if need_bwd:
            ctx.save_for_backward(c2_node, inv_M, wavelet)
            ctx.c, ctx.carry = c, take(s, _AC3D_BARE_CARRY)
        return seis

    @staticmethod
    def backward(ctx, grad_seis):
        import sweep._C as _C
        c2_node, inv_M, wavelet = ctx.saved_tensors
        b = _ac3d_bare_backward_input(ctx.c, c2_node, inv_M, wavelet, ctx.carry,
                                      grad_seis.contiguous().float())
        g_c2, _gjy, _gjz, g_M, g_w = _C.sem3d_acoustic_backward(b)
        return grads_or_none((c2_node, inv_M, wavelet), (g_c2, g_M, g_w)) + (None,)


# --------------------------------------------------------------------------- #
# 3-D elastic displacement (K self-adjoint), flat or deformed
# --------------------------------------------------------------------------- #
_EL3D_BARE = ("ibool", "D", "w3", "jac", "pat_x", "pat_y", "pat_z", "rec_G", "rec_W",
              "Gxx", "Gyy", "Gzz", "Gxy", "Gxz", "Gyz")


def _el3d_geometry(s, c, A):
    """The Hex27 metric on both elastic 3-D structs -- this one helper serves
    the forward and the backward -- with the columnar shortcut where the mesh
    has one. ``columnar`` defaults to 0 on the struct, so a path that forgets
    it falls back to the always-correct full 3x3 rather than reading a stale
    scalar; that is the quiet failure ``prop_c.metric_consts`` exists to make
    impossible on the producer side."""
    s.deformed = int(c.get("deformed", 0))
    if s.deformed:
        fill(s, c, _AC3D_METRIC)
        s.columnar = int(c.get("columnar", 0))
        s.col_ixx, s.col_iey = c.get("col_ixx", 0.0), c.get("col_iey", 0.0)
    else:
        put(s, _AC3D_METRIC, A.empty)


class _SEMElastic3DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, inv_M, wavelet, c):
        import sweep._C as _C
        mats = (lam, mu, lam2mu, inv_M)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        s = _C.Sem3dElasticForwardInput()
        fill(s, c, _EL3D_BARE)
        _el3d_geometry(s, c, A)
        s.lam, s.mu, s.lam2mu, s.inv_M, s.wavelet = lam, mu, lam2mu, inv_M, wavelet
        put(s, ("ux", "uy", "uz", "ux_prev", "uy_prev", "uz_prev",
                "ux_next", "uy_next", "uz_next", "Kx", "Ky", "Kz"), A.zeros)
        give(s, _record(A, c, "seis_x", "seis_y", "seis_z"))
        put(s, ("save_ux", "save_uy", "save_uz"), A.hist if need_bwd else A.empty)
        dims(s, c, 3, batch=1)
        seis = _C.sem3d_elastic_forward(check_filled(s))
        if need_bwd:
            ctx.save_for_backward(*mats, wavelet)
            ctx.c, ctx.carry = c, take(s, ("save_ux", "save_uy", "save_uz"))
        return seis

    @staticmethod
    def backward(ctx, gsx, gsy, gsz):
        import sweep._C as _C
        *mats, wavelet = ctx.saved_tensors
        c = ctx.c
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.Sem3dElasticBackwardInput()
        fill(b, c, _EL3D_BARE)
        _el3d_geometry(b, c, A)
        b.lam, b.mu, b.lam2mu, b.inv_M = mats
        b.wavelet = wavelet
        give(b, ctx.carry)
        b.grad_sx, b.grad_sy, b.grad_sz = (g.contiguous().float() for g in (gsx, gsy, gsz))
        put(b, ("ax1", "ay1", "az1", "ax2", "ay2", "az2", "axm", "aym", "azm",
                "tmp_x", "tmp_y", "tmp_z", "Kx", "Ky", "Kz",
                "adj_x", "adj_y", "adj_z", "yx", "yy", "yz", "grad_inv_M"), A.zeros)
        for nm, t in zip(("grad_lam", "grad_mu", "grad_lam2mu"), mats):
            setattr(b, nm, torch.zeros_like(t))
        b.grad_wavelet = A.zeros(c["nt"])
        dims(b, c, 3, batch=1)
        grads = _C.sem3d_elastic_backward(check_filled(b))
        return grads_or_none(mats + [wavelet], grads) + (None,)


# --------------------------------------------------------------------------- #
# 3-D velocity-stress elastic (bare or simple CPML): the save-all keeps the
# velocity history and the nine stretched velocity gradients per step
# --------------------------------------------------------------------------- #
_VS3D_BARE = ("ibool", "D", "wHx", "wHy", "wHz", "pat_x", "pat_y", "pat_z",
              "rec_G", "rec_W", "ihx", "ihy", "ihz")
_VS3D_CPML = ("a_x", "b_x", "a_y", "b_y", "a_z", "b_z")


def _vs3d_geometry(s, c, A):
    """The Hex27 metric on both velocity-stress 3-D structs
    (ElasticVSSEM3DDeformed: ``deformed`` = 1), or empties on the Hex8 path,
    whose kernels read wH* / ih* instead."""
    s.deformed = int(c.get("deformed", 0))
    if s.deformed:
        fill(s, c, _AC3D_METRIC)
    else:
        put(s, _AC3D_METRIC, A.empty)


def _vs3d_forward_input(c, A, hist):
    """The forward struct of the 3-D velocity-stress Functions, all but the
    moduli and the ``inv_M`` / ``wavelet`` leaves, which the caller sets; with
    the histories of ``hist`` steps (``None``: none), which a save-all run and a
    chunk's recompute keep (see :mod:`.vs_chunks`)."""
    import sweep._C as _C
    E, m, ng = c["n_elements"], c["ngll"] ** 3, c["n_global"]
    s = _C.Sem3dVsForwardInput()
    fill(s, c, _VS3D_BARE)
    _vs3d_geometry(s, c, A)
    if c["has_cpml"]:
        fill(s, c, _VS3D_CPML + ("a_n",))
        s.P, s.msig = A.block(9, ng), A.block(9, ng)      # the divergence memories are nodal
    else:
        put(s, _VS3D_CPML + ("a_n", "P", "msig"), A.empty)
    put(s, ("vx", "vy", "vz", "Rx", "Ry", "Rz"), A.zeros)
    s.sig, s.mvel = A.block(6, E * m), A.block(9, E * m)
    give(s, _record(A, c, "seis_x", "seis_y", "seis_z"))
    if hist is None:
        put(s, ("save_vx", "save_vy", "save_vz", "save_gv"), A.empty)
    else:               # v: the start of the run, then every step; gv: every step
        put(s, ("save_vx", "save_vy", "save_vz"), lambda: torch.zeros(
            hist + 1, ng, dtype=torch.float32, device=A.dev))
        s.save_gv = torch.zeros(hist, 9, E * m, dtype=torch.float32, device=A.dev)
    # the isotropic elastic default; the anisotropic / visco Functions override
    s.aniso, s.cij = 0, A.empty
    s.nsls = 0
    put(s, ("ptau", "mtau", "gsls_w", "rmem"), A.empty)
    dims(s, c, 3, batch=1)
    return s


def _vs3d_backward_input(c, A, gs):
    """The backward struct of the Functions, all but the histories, the moduli,
    their gradient buffers and the ``inv_M`` / ``wavelet`` leaves."""
    import sweep._C as _C
    E, m, ng = c["n_elements"], c["ngll"] ** 3, c["n_global"]
    b = _C.Sem3dVsBackwardInput()
    fill(b, c, _VS3D_BARE)
    _vs3d_geometry(b, c, A)
    if c["has_cpml"]:
        fill(b, c, _VS3D_CPML + ("a_n",))
        b.lP, b.lmsig = A.block(9, ng), A.block(9, ng)
    else:
        put(b, _VS3D_CPML + ("a_n", "lP", "lmsig"), A.empty)
    b.grad_sx, b.grad_sy, b.grad_sz = (g.contiguous().float() for g in gs)
    put(b, ("lvx", "lvy", "lvz", "lRx", "lRy", "lRz", "adj_x", "adj_y", "adj_z",
            "grad_inv_M"), A.zeros)
    b.lsig, b.lmvel = A.block(6, E * m), A.block(9, E * m)
    b.lgv = A.empty
    b.grad_wavelet = A.zeros(c["nt"])
    b.aniso, b.cij, b.grad_cij = 0, A.empty, A.empty
    b.nsls = 0
    put(b, ("ptau", "mtau", "gsls_w", "lrmem", "grad_ptau", "grad_mtau"), A.empty)
    dims(b, c, 3, batch=1)
    return b


_VS3D_CARRY = ("save_vx", "save_vy", "save_vz", "save_gv")


def _vs3d_run(ctx, c, A, need_bwd, set_moduli, saved):
    """One forward of a 3-D velocity-stress Function: the save-all, a run with no
    adjoint, or the chunked checkpoints (``c["ckpt"]``). ``set_moduli(s)`` puts
    the Function's constitutive leaves on a forward struct; ``saved`` are the
    tensors the backward takes back."""
    import sweep._C as _C
    K = chunk_of(c, need_bwd)
    s = _vs3d_forward_input(c, A, c["nt"] if need_bwd and K is None else None)
    set_moduli(s)
    if K is None:
        seis = _C.sem3d_elastic_vs_forward(check_filled(s))
        carry = take(s, _VS3D_CARRY)
    else:
        names, ckpts = forward_chunks(_C.sem3d_elastic_vs_forward, check_filled(s),
                                      c["nt"], K)
        seis = (s.seis_x, s.seis_y, s.seis_z)
        carry = {"K": K, "names": names, "ckpts": ckpts}
    if need_bwd:
        ctx.save_for_backward(*saved)
        ctx.c, ctx.carry, ctx.set_moduli = c, carry, set_moduli
    return seis


def _vs3d_adjoint(ctx, A, b):
    """The adjoint of :func:`_vs3d_run` on the backward struct ``b`` (moduli,
    gradient buffers and leaves set): the save-all histories, or the chunks
    recomputed from their checkpoints."""
    import sweep._C as _C
    c, carry = ctx.c, ctx.carry
    if "ckpts" not in carry:
        give(b, carry)
        _C.sem3d_elastic_vs_backward(check_filled(b))
        return
    K = carry["K"]
    f = _vs3d_forward_input(c, A, K)
    ctx.set_moduli(f)
    give(b, take(f, _VS3D_CARRY))
    backward_chunks(_C.sem3d_elastic_vs_forward, _C.sem3d_elastic_vs_backward,
                    check_filled(f), check_filled(b), c["nt"], K,
                    carry["names"], carry["ckpts"])


class _SEMElasticVS3DC(torch.autograd.Function):
    """ElasticVSSEM3D / ElasticVSSEM3DDeformed: the isotropic Lame set."""
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, inv_M, wavelet, c):
        mats = (lam, mu, lam2mu, inv_M)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        A = Alloc(wavelet.device, c["n_global"], c["nt"])

        def set_moduli(s):
            s.lam, s.mu, s.lam2mu, s.inv_M, s.wavelet = lam, mu, lam2mu, inv_M, wavelet
        return _vs3d_run(ctx, c, A, need_bwd, set_moduli, mats + (wavelet,))

    @staticmethod
    def backward(ctx, gsx, gsy, gsz):
        *mats, wavelet = ctx.saved_tensors
        c = ctx.c
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _vs3d_backward_input(c, A, (gsx, gsy, gsz))
        b.lam, b.mu, b.lam2mu, b.inv_M = mats
        b.wavelet = wavelet
        for nm, t in zip(("grad_lam", "grad_mu", "grad_lam2mu"), mats):
            setattr(b, nm, torch.zeros_like(t))
        _vs3d_adjoint(ctx, A, b)
        grads = (b.grad_lam, b.grad_mu, b.grad_lam2mu, b.grad_inv_M, b.grad_wavelet)
        return grads_or_none(mats + [wavelet], grads) + (None,)


class _SEMElasticVS3DAnisoC(torch.autograd.Function):
    """ElasticVTISEM3D / ElasticTTISEM3D and their Hex27 versions: the same
    kernels with the Voigt stiffnesses ``cij`` -- ``(9 or 21, E, ngll^3)``, in
    the class's ``bs_model_attrs`` order -- in place of the Lame set. The
    stiffness gradients come back as one block; autograd takes them on to the
    Thomsen parameters through the class's ``thomsen_to_cij3d``."""
    @staticmethod
    def forward(ctx, cij, inv_M, wavelet, c):
        need_bwd = any(t.requires_grad for t in (cij, inv_M, wavelet))
        A = Alloc(wavelet.device, c["n_global"], c["nt"])

        def set_moduli(s):
            s.aniso, s.cij = int(cij.shape[0]), cij
            s.lam = s.mu = s.lam2mu = A.empty
            s.inv_M, s.wavelet = inv_M, wavelet
        return _vs3d_run(ctx, c, A, need_bwd, set_moduli, (cij, inv_M, wavelet))

    @staticmethod
    def backward(ctx, gsx, gsy, gsz):
        cij, inv_M, wavelet = ctx.saved_tensors
        c = ctx.c
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _vs3d_backward_input(c, A, (gsx, gsy, gsz))
        b.aniso, b.cij, b.grad_cij = int(cij.shape[0]), cij, torch.zeros_like(cij)
        b.lam = b.mu = b.lam2mu = A.empty
        b.grad_lam = b.grad_mu = b.grad_lam2mu = A.empty
        b.inv_M, b.wavelet = inv_M, wavelet
        _vs3d_adjoint(ctx, A, b)
        return grads_or_none((cij, inv_M, wavelet),
                             (b.grad_cij, b.grad_inv_M, b.grad_wavelet)) + (None,)


class _SEMViscoElasticVS3DC(torch.autograd.Function):
    """ViscoElasticVSSEM3D / ViscoElasticVSSEM3DDeformed: the unrelaxed Lame set
    plus the GSLS memories on the stress update. ``ptau`` / ``mtau`` are
    ``(n_sls, E, ngll^3)``: the relaxed P / shear moduli times the strengths;
    ``c["gsls_w"]`` the trapezoidal weights ``(a_l .., c_l ..)`` for this dt.
    The memories' history is not needed by the adjoint (their update is linear
    with constant weights), only their adjoint state."""
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, ptau, mtau, inv_M, wavelet, c):
        mats = (lam, mu, lam2mu, ptau, mtau, inv_M)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        E, m = c["n_elements"], c["ngll"] ** 3
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        L = int(ptau.shape[0])

        def set_moduli(s):
            s.lam, s.mu, s.lam2mu, s.inv_M, s.wavelet = lam, mu, lam2mu, inv_M, wavelet
            s.nsls, s.ptau, s.mtau, s.gsls_w = L, ptau, mtau, c["gsls_w"]
            s.rmem = A.block(6 * L, E * m)
        return _vs3d_run(ctx, c, A, need_bwd, set_moduli, mats + (wavelet,))

    @staticmethod
    def backward(ctx, gsx, gsy, gsz):
        lam, mu, lam2mu, ptau, mtau, inv_M, wavelet = ctx.saved_tensors
        c = ctx.c
        E, m = c["n_elements"], c["ngll"] ** 3
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _vs3d_backward_input(c, A, (gsx, gsy, gsz))
        b.lam, b.mu, b.lam2mu, b.inv_M, b.wavelet = lam, mu, lam2mu, inv_M, wavelet
        for nm, t in zip(("grad_lam", "grad_mu", "grad_lam2mu"), (lam, mu, lam2mu)):
            setattr(b, nm, torch.zeros_like(t))
        L = int(ptau.shape[0])
        b.nsls, b.ptau, b.mtau, b.gsls_w = L, ptau, mtau, c["gsls_w"]
        b.lrmem = A.block(6 * L, E * m)
        b.grad_ptau, b.grad_mtau = torch.zeros_like(ptau), torch.zeros_like(mtau)
        _vs3d_adjoint(ctx, A, b)
        return grads_or_none((lam, mu, lam2mu, ptau, mtau, inv_M, wavelet),
                             (b.grad_lam, b.grad_mu, b.grad_lam2mu, b.grad_ptau,
                              b.grad_mtau, b.grad_inv_M, b.grad_wavelet)) + (None,)

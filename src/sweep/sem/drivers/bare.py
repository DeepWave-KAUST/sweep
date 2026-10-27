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
# 3-D acoustic, velocity form: flat Hex8 with the separable stiffness weights
# c2_j{x,y,z} (= vp^2 w3 G_axis, so build() chains their grads to vp_e) or
# deformed Hex27 with one per-node c^2 and the 3x3 inverse-Jacobian metric.
# The same two C entries serve both, switched by ``deformed``.
# --------------------------------------------------------------------------- #
_AC3D_BARE = ("ibool", "D", "dirichlet_idx", "src_pattern", "rec_G", "rec_W")
_AC3D_METRIC = ("wJ",) + METRIC_NAMES
_AC3D_BARE_CARRY = ("save_wavefield", "fin_u", "fin_u_prev", "bs")


def _ac3d_bare_forward_input(c, mats, inv_M, wavelet, mode, deformed):
    import sweep._C as _C
    A = Alloc(wavelet.device, c["n_global"], c["nt"])
    s = _C.Sem3dForwardInput()
    fill(s, c, _AC3D_BARE)
    s.deformed = int(deformed)
    if deformed:
        fill(s, c, _AC3D_METRIC)
        (s.c2_node,) = mats
        put(s, ("c2_jx", "c2_jy", "c2_jz"), A.empty)
    else:
        put(s, _AC3D_METRIC + ("c2_node",), A.empty)
        s.c2_jx, s.c2_jy, s.c2_jz = mats
    s.inv_M, s.wavelet = inv_M, wavelet
    put(s, ("u", "u_prev", "u_next", "K"), A.zeros)
    give(s, _record(A, c, "seis"))
    s.bs = int(mode == "ring")
    s.save_wavefield = A.hist() if mode == "tape" else A.empty
    put(s, ("fin_u", "fin_u_prev"), A.zeros if mode == "ring" else A.empty)
    dims(s, c, 3, batch=1)
    return check_filled(s)


def _ac3d_bare_backward_input(c, mats, inv_M, wavelet, carry, grad_syn, deformed):
    import sweep._C as _C
    A = Alloc(wavelet.device, c["n_global"], c["nt"])
    b = _C.Sem3dBackwardInput()
    fill(b, c, _AC3D_BARE)
    b.deformed = int(deformed)
    if deformed:
        fill(b, c, _AC3D_METRIC)
        (b.c2_node,) = mats
        put(b, ("c2_jx", "c2_jy", "c2_jz"), A.empty)
        # grad_c2_node comes back in the jx slot; jy / jz are unused
        put(b, ("grad_c2_jx", "grad_c2_jy", "grad_c2_jz"),
            lambda: torch.zeros_like(mats[0]))
    else:
        put(b, _AC3D_METRIC + ("c2_node",), A.empty)
        b.c2_jx, b.c2_jy, b.c2_jz = mats
        for nm, t in zip(("grad_c2_jx", "grad_c2_jy", "grad_c2_jz"), mats):
            setattr(b, nm, torch.zeros_like(t))
    b.inv_M, b.wavelet = inv_M, wavelet
    give(b, carry)
    b.grad_syn = grad_syn
    put(b, ("u_hi", "u_lo", "u_km1"), A.zeros if b.bs else A.empty)
    put(b, ("a_p1", "a_p2", "a_m", "tmp", "Ktmp", "Ku", "adj_src", "grad_inv_M"),
        A.zeros)
    b.grad_wavelet = A.zeros(c["nt"])
    dims(b, c, 3, batch=1)
    return check_filled(b)


# NOT ROUTED (prop_c._C_ROUTES): the sem3d_acoustic_* kernels behind the two
# Functions below put c² inside the element integral, i.e. solve div(c² grad u)
# instead of the c² Laplace u of AcousticSEM3D{,Deformed} -- every reflection
# coefficient comes out with the wrong sign. Kept for when the kernels apply c²
# per node (forward + adjoint), as the C-PML strip kernels already do.
class _SEMAcoustic3DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c2_jx, c2_jy, c2_jz, inv_M, wavelet, c):
        import sweep._C as _C
        mats = (c2_jx, c2_jy, c2_jz)
        need_bwd = any(t.requires_grad for t in mats + (inv_M, wavelet))
        s = _ac3d_bare_forward_input(c, mats, inv_M, wavelet, memory_mode(c, need_bwd), False)
        seis = _C.sem3d_acoustic_forward(s)
        if need_bwd:
            ctx.save_for_backward(*mats, inv_M, wavelet)
            ctx.c, ctx.carry = c, take(s, _AC3D_BARE_CARRY)
        return seis

    @staticmethod
    def backward(ctx, grad_seis):
        import sweep._C as _C
        *mats, inv_M, wavelet = ctx.saved_tensors
        b = _ac3d_bare_backward_input(ctx.c, mats, inv_M, wavelet, ctx.carry,
                                      grad_seis.contiguous().float(), False)
        grads = _C.sem3d_acoustic_backward(b)
        return grads_or_none(mats + [inv_M, wavelet], grads) + (None,)


class _SEMAcoustic3DDeformedC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c2_node, inv_M, wavelet, c):
        import sweep._C as _C
        need_bwd = any(t.requires_grad for t in (c2_node, inv_M, wavelet))
        s = _ac3d_bare_forward_input(c, (c2_node,), inv_M, wavelet,
                                     memory_mode(c, need_bwd), True)
        seis = _C.sem3d_acoustic_forward(s)
        if need_bwd:
            ctx.save_for_backward(c2_node, inv_M, wavelet)
            ctx.c, ctx.carry = c, take(s, _AC3D_BARE_CARRY)
        return seis

    @staticmethod
    def backward(ctx, grad_seis):
        import sweep._C as _C
        c2_node, inv_M, wavelet = ctx.saved_tensors
        b = _ac3d_bare_backward_input(ctx.c, (c2_node,), inv_M, wavelet, ctx.carry,
                                      grad_seis.contiguous().float(), True)
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


class _SEMElasticVS3DC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, mu, lam2mu, inv_M, wavelet, c):
        import sweep._C as _C
        mats = (lam, mu, lam2mu, inv_M)
        need_bwd = any(t.requires_grad for t in mats + (wavelet,))
        E, m = c["n_elements"], c["ngll"] ** 3
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        s = _C.Sem3dVsForwardInput()
        fill(s, c, _VS3D_BARE)
        if c["has_cpml"]:
            fill(s, c, _VS3D_CPML)
        else:
            put(s, _VS3D_CPML, A.empty)
        s.lam, s.mu, s.lam2mu, s.inv_M, s.wavelet = lam, mu, lam2mu, inv_M, wavelet
        put(s, ("vx", "vy", "vz", "Rx", "Ry", "Rz"), A.zeros)
        s.sig, s.msig, s.mvel = A.block(6, E * m), A.block(9, E * m), A.block(9, E * m)
        give(s, _record(A, c, "seis_x", "seis_y", "seis_z"))
        put(s, ("save_vx", "save_vy", "save_vz"), A.hist if need_bwd else A.empty)
        s.save_gv = (torch.zeros(c["nt"], 9, E * m, dtype=torch.float32, device=A.dev)
                     if need_bwd else A.empty)
        dims(s, c, 3, batch=1)
        seis = _C.sem3d_elastic_vs_forward(check_filled(s))
        if need_bwd:
            ctx.save_for_backward(*mats, wavelet)
            ctx.c, ctx.carry = c, take(s, ("save_vx", "save_vy", "save_vz", "save_gv"))
        return seis

    @staticmethod
    def backward(ctx, gsx, gsy, gsz):
        import sweep._C as _C
        *mats, wavelet = ctx.saved_tensors
        c = ctx.c
        E, m = c["n_elements"], c["ngll"] ** 3
        A = Alloc(wavelet.device, c["n_global"], c["nt"])
        b = _C.Sem3dVsBackwardInput()
        fill(b, c, _VS3D_BARE)
        if c["has_cpml"]:
            fill(b, c, _VS3D_CPML)
        else:
            put(b, _VS3D_CPML, A.empty)
        b.lam, b.mu, b.lam2mu, b.inv_M = mats
        b.wavelet = wavelet
        give(b, ctx.carry)
        b.grad_sx, b.grad_sy, b.grad_sz = (g.contiguous().float() for g in (gsx, gsy, gsz))
        put(b, ("lvx", "lvy", "lvz", "lRx", "lRy", "lRz", "adj_x", "adj_y", "adj_z",
                "grad_inv_M"), A.zeros)
        b.lsig, b.lmsig, b.lmvel = A.block(6, E * m), A.block(9, E * m), A.block(9, E * m)
        b.lgv = A.empty
        for nm, t in zip(("grad_lam", "grad_mu", "grad_lam2mu"), mats):
            setattr(b, nm, torch.zeros_like(t))
        b.grad_wavelet = A.zeros(c["nt"])
        dims(b, c, 3, batch=1)
        grads = _C.sem3d_elastic_vs_backward(check_filled(b))
        return grads_or_none(mats + [wavelet], grads) + (None,)

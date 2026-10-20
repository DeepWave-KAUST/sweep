"""The 2-D acoustic C-PML save-all backward on a w_sum tape.

The save-all backward used to keep five tapes (chi, psi_d, zeta_d; 20 B per
node per step) and, at every reverse step, re-run the forward's assembly on
the taped state -- six memsets and the element kernel -- only to recompute
w_sum^k. Its two consumers were ``combine_f`` (grad_kappa and the chi
adjoint) and ``grad_inv_Mg``, which every driver discards. So w_sum^k and
chi^k are the only forward quantities the backward needs: chi^k because
``grads_T`` images inv_rho from it (that tape stays), w_sum^k because the
forward already has it in a register. The forward now stores it (one more
4 B/node/step tape, four fewer) and the reverse step drops the assembly --
and, under DD, the exchange of the six assembled fields.

What is held here:
1. Row indexing: forward step n writes row n = w_sum^n, and the chi tape's
   own leapfrog identity ties the two together (a wrong row is O(1) wrong).
   The four state tapes are not allocated at all.
2. Given the same w_sum^k, the WTAPE code path computes the identical
   adjoint arithmetic: one reverse step on the live (old-path) struct and on
   a twin carrying that w_sum in its tape row -- every node-kernel output
   bit for bit. grad_inv_Mg is the one deliberate difference.
3. The old recompute and the tape draw from the SAME distribution as the
   assembly's own run-to-run floor. Nothing here is bitwise (SEM assembles
   with atomicAdd) and the floor is heavy-tailed, so the bars are pooled
   over samples, not read off one. Measured on this fixture when written (nt=120, three
   samples each): A/A record 1.4e-7 .. 4.8e-7, A/A d_vp 7.1e-7 .. 3.5e-6;
   A/B (recompute vs tape) record 1.3e-10 .. 4.8e-7, d_vp 6.7e-8 .. 3.5e-6.
   The gradients' 1-cos stays at 1e-16 .. 1e-12 throughout.
4. DD: the backward does one exchange per reverse step instead of two.
5. Boundary saving is untouched: it still takes rings and no w_sum tape.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
import sweep.sem.prop_c as PC
import sweep.sem.dd as DD
from sweep.propagator import BoundarySaving
from sweep.sem.drivers.acoustic2d import _AC2D_STATE_TAPES
from sweep.sem.equations import AcousticPotentialSEM2D

try:
    import sweep._C as _C
    _HAS_C = hasattr(_C, "sem2d_acoustic_cpml_dd_bwd_mid")
except Exception:                                   # pragma: no cover
    _HAS_C = False

needs_c = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_C),
                             reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, DT, F0 = 4, 25.0, 5e-4, 8.0
NZ, NX = 25, 49                      # 6 x 12 elements
STATE_TAPES = _AC2D_STATE_TAPES[1:]  # the four psi/zeta tapes


def _prop(nt=60, abcn=3 * NO, memory=None):
    return S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=nt, n_order=NO, free_surface=("top",),
                     abcn=abcn, f_dominant=F0, dev="cuda", impl="c",
                     memory=memory)


def _vp(grad=False):
    torch.manual_seed(0)
    v = (2000.0 + 300.0 * torch.rand(NZ, NX)).cuda()
    return v.requires_grad_(True) if grad else v


def _wav(nt):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")


def _geometry(prop, vp):
    eq, _ = prop._resolve_model_batch([vp.detach()])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + 5 * NO + 1
    rec = (torch.arange(3, nxg - 3, 4) + nxg).cuda()
    return src, rec


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def _gap(a, b):
    x, y = a.double().reshape(-1), b.double().reshape(-1)
    return abs(1.0 - float(x @ y / (x.norm() * y.norm())))


def _run(prop, vp, wav, src, rec, wtape, px=1):
    old = PC._WSUM_TAPE_2D
    PC._WSUM_TAPE_2D = wtape
    try:
        return DD.SEMModelParallel(prop, DD.StripTopology(px))(
            wav, src, rec, [vp])
    finally:
        PC._WSUM_TAPE_2D = old


# --------------------------------------------------------------------------- #
# 1. what the forward stores: row n = w_sum^n, and only two tapes
# --------------------------------------------------------------------------- #
@needs_c
def test_the_tape_rows_satisfy_the_forwards_leapfrog_and_replace_four_tapes():
    """chi^{n+1} = 2 chi^n - chi^{n-1} + dt2 (kappa w_sum^n + invM pat wav[n])
    is the forward step itself, so with save_chi[n] = chi^{n+1} and
    save_w_sum[n] = w_sum^n it must hold ROW BY ROW. A row-index slip is
    O(1) wrong, far outside the fp32 rounding this is compared to."""
    nt = 60
    p = _prop(nt)
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    got = {}
    real = DD._SEMAcousticCpmlDDC.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        got["carry"] = ctx.carry[0]
        got["c"] = ctx.parts[0].c
        return out

    DD._SEMAcousticCpmlDDC.forward = staticmethod(spy)
    try:
        _run(p, vp, _wav(nt), src, rec, True).pow(2).sum().backward()
    finally:
        DD._SEMAcousticCpmlDDC.forward = staticmethod(real)

    cr, c = got["carry"], got["c"]
    for nm in STATE_TAPES:
        assert cr[nm].numel() == 0, f"{nm} allocated under the w_sum tape"
    chi, ws = cr["save_wavefield"], cr["save_w_sum"]
    assert tuple(ws.shape) == (nt, c["n_global"])
    eq, _ = p._resolve_model_batch([vp.detach()])
    kappa = eq.kappa_node.to("cuda", torch.float32)
    inv_M = eq._inv_M.to("cuda", torch.float32)
    pat = c["src_pattern"]
    wav = _wav(nt)
    dt2 = float(DT) ** 2
    keep = torch.ones(c["n_global"], dtype=torch.bool, device="cuda")
    keep[c["dirichlet_idx"]] = False          # chi is re-zeroed there
    n = torch.arange(2, nt, device="cuda")
    want = (2.0 * chi[n - 1] - chi[n - 2]
            + dt2 * (kappa * ws[n] + inv_M * pat * wav[n][:, None]))
    err = (want[:, keep] - chi[n][:, keep]).abs().max()
    scale = chi[n][:, keep].abs().max()
    assert float(err / scale) < 1e-5, f"leapfrog residual {float(err / scale):.2e}"
    assert float(ws.abs().max()) > 0, "the tape never rang"


# --------------------------------------------------------------------------- #
# 2. twin check: the same w_sum^k, two code paths, the same bits
# --------------------------------------------------------------------------- #
MID_OUT = ("gbx", "gbz", "gbkx", "gbkz", "gbax", "gbaz")
CARRIES_N = ("lpsi_x_n", "lpsi_z_n", "lzeta_x_n", "lzeta_z_n")
END_OUT = ("gchi_k", "grad_kappa", "grad_inv_M") + CARRIES_N


def _fields(b):
    return [n for n in dir(b) if not n.startswith("_") and not callable(getattr(b, n))]


def _clone(b):
    nb = _C.SemAcousticCpmlBackwardInput()
    for n in _fields(b):
        v = getattr(b, n)
        setattr(nb, n, v.clone() if torch.is_tensor(v) else v)
    return nb


def _bits(a, b):
    """(value differences, signed-zero-only differences)."""
    da = a.view(torch.int32) != b.view(torch.int32)
    eq = a == b
    return int((da & ~eq).sum()), int((da & eq).sum())


def _twin(nt=24, k_at=None):
    """The old path (recompute, five tapes) drives; at reverse step K a twin
    of the pre-mid struct gets the old path's freshly recomputed w_sum^K as
    its tape row and runs the WTAPE path. Same w_sum, same inputs -> same
    bits."""
    p = _prop(nt)
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    K = nt // 2 if k_at is None else k_at
    got = {}
    real_mid, real_end = (_C.sem2d_acoustic_cpml_dd_bwd_mid,
                          _C.sem2d_acoustic_cpml_dd_bwd_end)

    def spy_mid(b, k):
        if k != K:
            return real_mid(b, k)
        assert b.save_w_sum.numel() == 0, "the driver must be on the old path here"
        t = _clone(b)                                  # pre-mid state
        real_mid(b, k)                                 # old: recompute + adjoint
        torch.cuda.synchronize()
        tape = torch.zeros(int(b.nt), int(b.n_global), dtype=torch.float32,
                           device="cuda")
        tape[k] = b.w_sum                              # the twin's tape row
        t.save_w_sum = tape
        real_mid(t, k)                                 # WTAPE: adjoint algebra only
        torch.cuda.synchronize()
        got["mid"] = {nm: _bits(getattr(b, nm), getattr(t, nm)) for nm in MID_OUT}
        got["gInvMg"] = (b.grad_inv_Mg.clone(), t.grad_inv_Mg.clone())
        for nm in ("lchi_G", "lapx", "lapz"):          # grads_T atomics: same bits in
            getattr(t, nm).copy_(getattr(b, nm))
        got["twin"] = t
        return None

    def spy_end(b, k):
        if k != K:
            return real_end(b, k)
        t = got["twin"]
        real_end(b, k)
        real_end(t, k)
        torch.cuda.synchronize()
        got["end"] = {nm: _bits(getattr(b, nm), getattr(t, nm)) for nm in END_OUT}
        return None

    _C.sem2d_acoustic_cpml_dd_bwd_mid = spy_mid
    _C.sem2d_acoustic_cpml_dd_bwd_end = spy_end
    try:
        _run(p, vp, _wav(nt), src, rec, False).pow(2).sum().backward()
    finally:
        _C.sem2d_acoustic_cpml_dd_bwd_mid = real_mid
        _C.sem2d_acoustic_cpml_dd_bwd_end = real_end
    assert "mid" in got and "end" in got
    return got


@needs_c
@pytest.mark.parametrize("k_at", [None, 3], ids=["mid-time", "late-reverse"])
def test_wtape_path_is_bitwise_given_the_same_w_sum(k_at):
    got = _twin(k_at=k_at)
    bad = {nm: v for nm, v in {**got["mid"], **got["end"]}.items() if v[0]}
    assert not bad, f"value differences: {bad}"
    signed = {nm: v[1] for nm, v in {**got["mid"], **got["end"]}.items() if v[1]}
    assert not signed, f"signed-zero differences (unexpected in 2-D): {signed}"
    # the one deliberate difference: grad_inv_Mg needs the assembly, so the
    # tape path never accumulates it (every driver discards the value)
    old_g, new_g = got["gInvMg"]
    assert float((old_g - new_g).abs().max()) > 0, \
        "grad_inv_Mg unchanged: the WTAPE instantiation still accumulated it"


# --------------------------------------------------------------------------- #
# 3. the two paths' gradients agree to the assembly's floor
# --------------------------------------------------------------------------- #
def _grad(wtape, px=1, nt=120, fused=False):
    p = _prop(nt)
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    wav = _wav(nt).requires_grad_(True)
    if fused:
        old = PC._WSUM_TAPE_2D
        PC._WSUM_TAPE_2D = wtape
        try:
            eq, _ = p._resolve_model_batch([vp])
            pattern = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                                  device=eq.mesh.ibool.device)
            pattern[int(src)] = 1.0
            parts, mats = DD.SEMModelParallel(p, DD.StripTopology(1))._parts(
                eq, pattern, rec)
            c = dict(parts[0].c)
            c["bs"], c["bs_kind"] = False, 0
            out = PC._SEMAcousticCpmlC.apply(mats[0], mats[1], mats[2], wav, c)
        finally:
            PC._WSUM_TAPE_2D = old
    else:
        out = _run(p, vp, wav, src, rec, wtape, px=px)
    out.pow(2).sum().backward()
    return out.detach(), vp.grad.clone(), wav.grad.clone()


def _agree(s0, s1, g0, g1, w0, w1):
    """The bars are the pooled A/A floor of this fixture, one decade up (see
    the module docstring): the assembly has no bitwise criterion, and the
    tape moves the result no more than re-running the same code does."""
    assert _rel(s1, s0) < 5e-6, f"record rel {_rel(s1, s0):.2e}"
    assert _gap(g0, g1) < 1e-10, f"d_vp 1-cos {_gap(g0, g1):.2e}"
    assert _rel(g1, g0) < 3e-5, f"d_vp rel {_rel(g1, g0):.2e}"
    assert _rel(w1, w0) < 3e-5, f"d_wav rel {_rel(w1, w0):.2e}"


@needs_c
@pytest.mark.parametrize("px", [1, 2])
def test_dd_gradient_matches_the_recompute_path(px):
    s0, g0, w0 = _grad(False, px=px)
    s1, g1, w1 = _grad(True, px=px)
    _agree(s0, s1, g0, g1, w0, w1)


@needs_c
def test_fused_gradient_matches_the_recompute_path():
    s0, g0, w0 = _grad(False, fused=True)
    s1, g1, w1 = _grad(True, fused=True)
    _agree(s0, s1, g0, g1, w0, w1)


# --------------------------------------------------------------------------- #
# 4. DD: one exchange per reverse step instead of two
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("wtape", [False, True], ids=["recompute", "tape"])
def test_dd_backward_exchange_count(wtape):
    nt = 30
    p = _prop(nt)
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    seen = {"n": 0}
    real = DD._DDExchange.block

    def spy(self, parents):
        seen["n"] += 1
        return real(self, parents)

    DD._DDExchange.block = spy
    try:
        out = _run(p, vp, _wav(nt), src, rec, wtape, px=2)
        seen["fwd"] = seen["n"]
        out.pow(2).sum().backward()
    finally:
        DD._DDExchange.block = real
    bwd = seen["n"] - seen["fwd"]
    assert seen["fwd"] == nt, f"forward exchanges {seen['fwd']} != {nt}"
    want = (nt - 1) if wtape else 2 * (nt - 1)
    assert bwd == want, f"backward exchanges {bwd} != {want}"


# --------------------------------------------------------------------------- #
# 5. boundary saving is untouched
# --------------------------------------------------------------------------- #
@needs_c
def test_boundary_saving_takes_no_w_sum_tape():
    """The reverse march rebuilds w_sum^k on the fly to step chi back, so bs
    keeps the recompute; the tape must not be allocated there."""
    nt = 40
    p = _prop(nt, memory=BoundarySaving())
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    got = {}
    real = DD._SEMAcousticCpmlDDC.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        got["carry"] = ctx.carry[0]
        got["wtape"] = ctx.wtape
        return out

    DD._SEMAcousticCpmlDDC.forward = staticmethod(spy)
    try:
        _run(p, vp, _wav(nt), src, rec, True, px=2).pow(2).sum().backward()
    finally:
        DD._SEMAcousticCpmlDDC.forward = staticmethod(real)
    assert got["wtape"] is False
    cr = got["carry"]
    assert cr["save_w_sum"].numel() == 0, "boundary saving allocated a w_sum tape"
    assert cr["band_chi"].numel() > 0, "the ring is gone"


# --------------------------------------------------------------------------- #
# 6. decisive: under the tape the reverse step never READS the assembly
# --------------------------------------------------------------------------- #
@needs_c
def test_tape_backward_never_reads_the_assembled_fields():
    """NaN-poison the six assembled fields before every reverse step: NaN
    propagates unconditionally, so a finite gradient is proof that the tape
    path read none of them. They are also never written, which is why
    grad_inv_Mg -- the assembly's only other consumer -- stays zero. The
    w_sum scratch is NOT poisoned: at k == 0 the combine legitimately reads
    the zeros bwd_pre memsets there (row 0 is never taped, w_sum^0 == 0)."""
    nt = 30
    p = _prop(nt)
    vp = _vp(grad=True)
    src, rec = _geometry(p, vp)
    seen = {}
    real_pre = _C.sem2d_acoustic_cpml_dd_bwd_pre

    def spy(b, k):
        out = real_pre(b, k)
        for nm in ("gx", "gz", "kx", "kz", "gax", "gaz"):
            getattr(b, nm).fill_(float("nan"))
        seen["gInvMg"] = b.grad_inv_Mg
        return out

    _C.sem2d_acoustic_cpml_dd_bwd_pre = spy
    try:
        _run(p, vp, _wav(nt), src, rec, True).pow(2).sum().backward()
    finally:
        _C.sem2d_acoustic_cpml_dd_bwd_pre = real_pre
    assert torch.isfinite(vp.grad).all(), "the tape backward read a poisoned field"
    assert float(vp.grad.abs().max()) > 0, "the gradient is trivially zero"
    assert float(seen["gInvMg"].abs().max()) == 0.0, \
        "grad_inv_Mg was accumulated: the assembly is still being used"

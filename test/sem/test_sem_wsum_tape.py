"""The 3-D save-all backward on a w_sum tape.

The old save-all backward kept seven tapes (u, psi_d, zeta_d; 28 B per node
per step) and, at every reverse step, ran a*psi, nine memsets and the
element assembly again -- everything the forward did at step k -- only to
recompute w_sum^k for the u reverse march (unused here) and for
grad_vp2 += dt2 * w_sum^k * gA. The 3-D material gradient enters through
w_sum alone and grads_T reads no forward field, so w_sum^k is the ONLY
forward quantity the backward needs. The forward now saves it (one tape,
4 B/node/step) and the backward rebuilds nothing: no a*psi, no assembly,
no zeta, and under DD no exchange of the nine assembled fields.

What is held here:
1. The tape row is the forward's own w_sum (bitwise, on the last step
   where the forward's buffer is still around), the seven old tapes are
   not taken, and the memory-field slabs now apply in this mode too.
2. Given the same w_sum^k, the WTAPE code path computes the identical
   adjoint arithmetic: one reverse step on the live (old-path) struct and
   on a twin that carries that w_sum in its tape row -- every node-kernel
   output bit for bit.
3. The old recompute and the tape differ by the assembly's run-to-run
   floor (measured), and the gradients of the two paths agree to it.
4. DD: the backward does one exchange per step instead of two.
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
import sweep.sem.prop_c as PC
import sweep.sem.dd as DD
from sweep.sem.equations import AcousticSEM3D, AcousticSEM3DDeformed

try:
    import sweep._C as _C
    _HAS_C = hasattr(_C, "sem3d_acoustic_cpml_dd_bwd_mid")
except Exception:                                   # pragma: no cover
    _HAS_C = False

needs_c = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_C),
                             reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, DT, F0 = 4, 40.0, 6e-4, 15.0
TAPES = ("save_u", "save_psi_x", "save_psi_y", "save_psi_z",
         "save_zeta_x", "save_zeta_y", "save_zeta_z")


def _hill(x, y):
    return 120.0 * torch.exp(-(((x - 1300.0) / 700.0) ** 2
                               + ((y - 1300.0) / 700.0) ** 2))


def _hill_rows(n):
    """``_hill`` on the model columns: the surface row of each, (n, n)."""
    a = torch.arange(n, dtype=torch.float64) * DH
    Y, X = torch.meshgrid(a, a, indexing="ij")
    return (_hill(X, Y) / DH).numpy()


def _prop(nex, deformed, nt, abcn=2):
    """A model of ``nex`` elements a side; ``abcn`` ELEMENTS of absorbing
    layer outside it."""
    n = nex * NO + 1
    return S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D,
                     shape=(n, n, n), dh=DH, dt=DT, nt=nt, n_order=NO,
                     free_surface=("top",), abcn=abcn * NO, f_dominant=F0,
                     dev="cuda", impl="c",
                     **(dict(topography=_hill_rows(n)) if deformed else {}))


def _gid(p, ix, iy, iz):
    """Model node ``(ix, iy, iz)`` -> its GLOBAL id, on the row-major node
    grid of the model AND its absorbing layer."""
    _, nyg, nxg = p._grid_shape()
    z0, y0, x0 = p._pad0
    return ((iz + z0) * nyg + iy + y0) * nxg + ix + x0


def _wav(nt):
    t = torch.arange(nt, dtype=torch.float32, device="cuda") * DT - 1.2 / F0
    a = (3.1415926 * F0 * t) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _vp(n, seed=0):
    torch.manual_seed(seed)
    return 2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")


def _rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-300))


def _gap(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return abs(1.0 - float(a @ b / (a.norm() * b.norm())))


REL_BAR, GAP_BAR = 1e-5, 1e-10     # ten forward floors, see test_sem_psi_slab


# --------------------------------------------------------------------------- #
# 1. the tape is the forward's w_sum; the seven old tapes are gone
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_tape_row_is_the_forwards_w_sum_and_the_old_tapes_are_gone(deformed):
    nex, nt = 8, 12
    n = nex * NO + 1
    p = _prop(nex, deformed, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0, c0, z0]]], device="cuda")
    seen = {}
    real = _C.sem3d_acoustic_cpml_dd_step             # PropSEM runs the one-strip driver

    def spy(inp, n_):
        real(inp, n_)
        if n_ == nt - 1:
            torch.cuda.synchronize()
            seen["tape"] = inp.save_w_sum.clone()
            seen["last"] = inp.w_sum.clone()      # w_sum of step nt-1, still in its buffer
            seen["old"] = {nm: int(getattr(inp, nm).numel()) for nm in TAPES}
            seen["slab"] = int(inp.psi_slab)
            seen["ng"] = int(inp.n_global)

    _C.sem3d_acoustic_cpml_dd_step = spy
    try:
        out = p.forward(_wav(nt), src, rec, [vp])        # save-all mode
        out.pow(2).sum().backward()
    finally:
        _C.sem3d_acoustic_cpml_dd_step = real
    assert seen["tape"].shape == (nt, seen["ng"])
    assert torch.equal(seen["tape"][nt - 1].view(torch.int32), seen["last"].view(torch.int32))
    assert all(v == 0 for v in seen["old"].values()), seen["old"]
    assert seen["slab"] == 1, "the slabs apply once the full tapes are gone"
    assert float(seen["tape"].abs().max()) > 0, "vacuous: w_sum never rang"
    assert torch.isfinite(vp.grad).all() and float(vp.grad.abs().max()) > 0


# --------------------------------------------------------------------------- #
# 2. the twin: the WTAPE path's adjoint arithmetic, bit for bit
# --------------------------------------------------------------------------- #
def _fields(b):
    return [n for n in dir(b) if not n.startswith("_") and not callable(getattr(b, n))]


def _clone(b):
    nb = _C.Sem3dCpmlBackwardInput()
    for n in _fields(b):
        v = getattr(b, n)
        setattr(nb, n, v.clone() if torch.is_tensor(v) else v)
    return nb


def _bits(a, b):
    da = a.view(torch.int32) != b.view(torch.int32)
    eq = a == b
    return int((da & ~eq).sum()), int((da & eq).sum())


MID_OUT = ("gbx", "gby", "gbz", "gbkx", "gbky", "gbkz", "gbax", "gbay", "gbaz")
CARRIES_N = ("lpsi_x_n", "lpsi_y_n", "lpsi_z_n", "lzeta_x_n", "lzeta_y_n", "lzeta_z_n")
END_OUT = ("gu_k", "grad_vp2") + CARRIES_N


def twin_check(deformed, nex=8, nt=24, k_at=None, slab_twin=False):
    """Old path (recompute, full layout) drives; at reverse step K a twin of
    the pre-mid struct gets the old path's freshly recomputed w_sum^K as its
    tape row and runs the WTAPE path -- in the full layout, or (slab_twin)
    converted to the slab layout, i.e. the production instantiation
    <SLAB=1, WTAPE=1> against the live <0, 0>. Same w_sum, same inputs ->
    same bits (the slab twin may differ in the sign of exact zeros in the
    nine step_T scratch outputs, see test_sem_adjoint_slab)."""
    from test_sem_adjoint_slab import _slots, _to_compact
    from sweep.sem.slab import FIELD_AXIS
    n = nex * NO + 1
    p = _prop(nex, deformed, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    K = nt // 2 if k_at is None else k_at
    got, parts = {}, {}
    real_mid, real_end = _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end
    real_fwd = DD._SEMAcoustic3DCpmlDDC.forward

    def spy_fwd(ctx, *a):
        out = real_fwd(ctx, *a)
        parts["c"] = ctx.parts[0].c
        return out

    def carries_compact(x, nm):
        c = parts["c"]
        ax = FIELD_AXIS[nm]
        return _to_compact(x, _slots(c, ax), int(c[f"slab_n_{ax}"]))

    def spy_mid(b, k):
        if k != K:
            return real_mid(b, k)
        assert b.save_w_sum.numel() == 0 and b.psi_slab == 0, "the driver must be on the old path here"
        t = _clone(b)                                  # pre-mid state
        real_mid(b, k)                                 # old: recompute w_sum^K + adjoint
        torch.cuda.synchronize()
        ng = int(b.n_global)
        tape = torch.zeros(int(b.nt), ng, dtype=torch.float32, device="cuda")
        tape[k] = b.w_sum                              # the twin's tape row
        t.save_w_sum = tape
        if slab_twin:                                  # the production layout
            # the twelve carries, lap_d, g_d and k_d: everything that
            # lives on an axis slab in the production layout
            for nm in FIELD_AXIS:
                if nm.startswith("l") or nm in ("gx", "gy", "gz",
                                                "kx", "ky", "kz"):
                    setattr(t, nm, carries_compact(getattr(t, nm), nm))
            t.psi_slab = 1
        real_mid(t, k)                                 # WTAPE: adjoint algebra only
        torch.cuda.synchronize()
        got["mid"] = {nm: _bits(getattr(b, nm), getattr(t, nm)) for nm in MID_OUT}
        for nm in ("lu_G", "lapx", "lapy", "lapz"):    # grads_T atomics: same bits in
            src = getattr(b, nm)
            if slab_twin and nm in FIELD_AXIS:         # lap_d is on its slab
                src = carries_compact(src, nm)
            getattr(t, nm).copy_(src)
        got["twin"] = t
        return None

    def spy_end(b, k):
        if k != K:
            return real_end(b, k)
        t = got["twin"]
        real_end(b, k)
        real_end(t, k)
        torch.cuda.synchronize()
        res = {}
        for nm in END_OUT:
            x, y = getattr(b, nm), getattr(t, nm)
            if nm in CARRIES_N and slab_twin:          # compare on the slab
                x = carries_compact(x, nm)
            res[nm] = _bits(x, y)
        got["end"] = res
        return None

    old = PC._WSUM_TAPE
    PC._WSUM_TAPE = False
    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy_fwd)
    _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end = spy_mid, spy_end
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(1))(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        PC._WSUM_TAPE = old
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real_fwd)
        _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end = real_mid, real_end
    assert "mid" in got and "end" in got
    return got


@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_wtape_path_is_bitwise_given_the_same_w_sum(deformed):
    got = twin_check(deformed)
    for stage in ("mid", "end"):
        bad = {k: v for k, v in got[stage].items() if v != (0, 0)}
        assert not bad, f"{stage}: bit differences {bad}"


@needs_c
@pytest.mark.parametrize("k_at", [None, 3], ids=["mid-time", "late-reverse"])
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_production_instantiation_is_bitwise_up_to_the_slabs_signed_zeros(deformed, k_at):
    """<SLAB=1, WTAPE=1> (what production runs) against the old <0, 0> on
    the same w_sum^k and inputs: no value differences anywhere; signed-zero
    differences only in the nine step_T scratch outputs, the slab's known
    effect (test_sem_adjoint_slab)."""
    from test_sem_adjoint_slab import SIGN_OK
    got = twin_check(deformed, k_at=k_at, slab_twin=True)
    for stage in ("mid", "end"):
        real = {k: v[0] for k, v in got[stage].items() if v[0]}
        assert not real, f"{stage}: VALUE differences {real}"
        stray = {k: v[1] for k, v in got[stage].items() if v[1] and k not in SIGN_OK}
        assert not stray, f"{stage}: signed zeros outside step_T's scratch: {stray}"


@needs_c
@pytest.mark.parametrize("px", [2, 3])
def test_dd_tape_row_is_each_strips_w_sum(px):
    """Pins the row index under DD without the fused loop: at forward step K
    each strip's live w_sum (written by dd_step) must be its tape row K."""
    # uniform tiles: px | 8 + 2 * 2, the model's and layer's elements
    nex, nt, K = 8, 12, 7
    n = nex * NO + 1
    p = _prop(nex, False, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    seen = {}
    real = _C.sem3d_acoustic_cpml_dd_step

    def spy(inp, n_):
        real(inp, n_)
        if n_ == K:
            torch.cuda.synchronize()
            seen[id(inp)] = (inp.w_sum.clone(), inp.save_w_sum)

    _C.sem3d_acoustic_cpml_dd_step = spy
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(px))(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        _C.sem3d_acoustic_cpml_dd_step = real
    assert len(seen) == px
    for live, tape in seen.values():
        assert tape.shape[0] == nt
        assert torch.equal(tape[K].view(torch.int32), live.view(torch.int32))
        assert float(live.abs().max()) > 0


# --------------------------------------------------------------------------- #
# 3. old recompute vs tape: the assembly floor, and the gradients
# --------------------------------------------------------------------------- #
def _fused_grad(deformed, wtape, nex=8, nt=40):
    n = nex * NO + 1
    p = _prop(nex, deformed, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0 - 6 + 2 * i, c0, z0] for i in range(6)]], device="cuda")
    old = PC._WSUM_TAPE
    PC._WSUM_TAPE = wtape
    try:
        out = p.forward(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        PC._WSUM_TAPE = old
    return out.detach(), vp.grad.detach()


@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_fused_gradient_matches_the_recompute_path(deformed):
    r0, g0 = _fused_grad(deformed, False)
    r1, g1 = _fused_grad(deformed, True)
    assert torch.equal(r0, r1) or _rel(r0, r1) <= REL_BAR      # the forward is untouched
    assert _rel(g0, g1) <= REL_BAR and _gap(g0, g1) <= GAP_BAR, \
        f"d_vp: rel {_rel(g0, g1):.3e} gap {_gap(g0, g1):.3e}"


@needs_c
def test_recompute_differs_from_the_tape_only_by_the_assembly_floor():
    """The number behind the argument: what the old backward rebuilt as
    w_sum^k is the forward's value up to the atomics' run-to-run floor --
    the tape hands the backward the forward's own bits instead."""
    nex, nt, K = 8, 24, 12
    n = nex * NO + 1
    p = _prop(nex, False, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    got = {}
    real_mid = _C.sem3d_acoustic_cpml_dd_bwd_mid
    real_fwd = _C.sem3d_acoustic_cpml_dd_step

    def spy_fwd(inp, n_):
        real_fwd(inp, n_)
        if n_ == K:
            torch.cuda.synchronize()
            got["fwd"] = inp.w_sum.clone()             # the forward's w_sum^K

    def spy_mid(b, k):
        real_mid(b, k)
        if k == K:
            torch.cuda.synchronize()
            got["bwd"] = b.w_sum.clone()               # the old path's recompute

    old = PC._WSUM_TAPE
    PC._WSUM_TAPE = False
    _C.sem3d_acoustic_cpml_dd_step, _C.sem3d_acoustic_cpml_dd_bwd_mid = spy_fwd, spy_mid
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(1))(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        PC._WSUM_TAPE = old
        _C.sem3d_acoustic_cpml_dd_step, _C.sem3d_acoustic_cpml_dd_bwd_mid = real_fwd, real_mid
    rel = _rel(got["fwd"], got["bwd"])
    assert float(got["fwd"].abs().max()) > 0
    assert rel <= REL_BAR, f"recompute vs forward w_sum: rel {rel:.3e}"
    print(f"old backward's recomputed w_sum^{K} vs the forward's: rel {rel:.3e}")


def _dd_grad(deformed, wtape, px, nex=8, nt=40):
    n = nex * NO + 1
    p = _prop(nex, deformed, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    calls = {"block": 0}
    real_block = DD._DDExchange.block if hasattr(DD, "_DDExchange") else None
    old = PC._WSUM_TAPE
    PC._WSUM_TAPE = wtape
    try:
        ddp = DD.SEMModelParallel(p, DD.StripTopology(px))
        out = ddp(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        PC._WSUM_TAPE = old
    return out.detach(), vp.grad.detach()


@needs_c
@pytest.mark.parametrize("px", [2, 3])
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_dd_gradient_matches_the_recompute_path(deformed, px):
    nex = 8         # uniform tiles: px | 8 + 2 * 2, the model's and layer's elements
    r0, g0 = _dd_grad(deformed, False, px, nex=nex)
    r1, g1 = _dd_grad(deformed, True, px, nex=nex)
    assert _rel(r0, r1) <= REL_BAR
    assert _rel(g0, g1) <= REL_BAR and _gap(g0, g1) <= GAP_BAR, \
        f"d_vp: rel {_rel(g0, g1):.3e} gap {_gap(g0, g1):.3e}"


# --------------------------------------------------------------------------- #
# 4. DD: one exchange per reverse step instead of two
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("wtape", [False, True], ids=["recompute", "tape"])
def test_dd_backward_exchange_count(wtape):
    nex, nt, px = 8, 10, 2
    n = nex * NO + 1
    p = _prop(nex, False, nt)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    n_bwd = {"count": 0}
    real_bwd = DD._SEMAcoustic3DCpmlDDC.backward
    old = PC._WSUM_TAPE
    PC._WSUM_TAPE = wtape

    def spy_bwd(ctx, *g):
        ex = ctx.exchange
        real_block = ex.block

        def counting(blocks, **kw):
            n_bwd["count"] += 1
            return real_block(blocks, **kw)

        ex.block = counting
        try:
            return real_bwd(ctx, *g)
        finally:
            ex.block = real_block

    DD._SEMAcoustic3DCpmlDDC.backward = staticmethod(spy_bwd)
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(px))(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        PC._WSUM_TAPE = old
        DD._SEMAcoustic3DCpmlDDC.backward = staticmethod(real_bwd)
    # k = nt-1 .. 1 exchange; the recompute path exchanges g9 AND l4, the tape l4 only
    want = (nt - 1) * (1 if wtape else 2)
    assert n_bwd["count"] == want, (n_bwd["count"], want)

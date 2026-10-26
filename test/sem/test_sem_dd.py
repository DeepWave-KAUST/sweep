"""SEM domain decomposition on impl='c': DD stepped loop vs the fused driver.

Single-process harness (the FD PLAN.md D5 pattern): one process steps every
strip, the exchange is a tensor add -- numerically identical to the NCCL
path. The reference is the SAME consts run through the FUSED single-domain
C driver, so px=1 pins the stepped SPLIT itself and px>1 pins the
decomposition. SEM assembles by summing (atomicAdd), so thresholds anchor
to the fused driver's own run-to-run floor, never bitwise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology, _SEMAcousticCpmlDDC
from sweep.sem.drivers.acoustic2d import _AC2D_RINGS
from sweep.sem.equations import AcousticPotentialSEM2D
from sweep.sem.prop_c import _SEMAcousticCpmlC, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 25.0, 8.0
NZ, NX = 25, 49                     # 6 x 12 elements
NT, DT = 250, 5e-4
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4


def _prop(memory=None):
    return S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=3 * NO, memory=memory, f_dominant=F0,
                     dev="cuda", impl="c")


def _vp(grad=False):
    torch.manual_seed(0)
    v = (2000.0 + 300.0 * torch.rand(NZ, NX)).cuda()
    return v.requires_grad_(True) if grad else v


def _wav(grad=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                     device="cuda")
    return w.requires_grad_(True) if grad else w


def _geometry(prop, vp):
    eq, _ = prop._resolve_model_batch([vp.detach()])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + 5 * NO + 1
    rec = (torch.arange(3, nxg - 3, 4) + 1 * nxg).cuda()
    return src, rec


def _fused_reference(prop, vp, wav, src, rec):
    """The SAME px=1 consts through the FUSED single-domain C driver."""
    eq_g, _ = prop._resolve_model_batch([vp])
    pattern = torch.zeros(eq_g.mesh.n_global, dtype=torch.float32,
                          device=eq_g.mesh.ibool.device)
    pattern[int(src)] = 1.0
    ddp1 = SEMModelParallel(prop, StripTopology(1))
    parts, mats = ddp1._parts(eq_g, pattern, rec)
    c = dict(parts[0].c)
    c["bs"], c["bs_kind"] = False, 0
    # px=1: rec_slot is the identity ordering, so the fused seis rows are
    # already in input order
    seis = _SEMAcousticCpmlC.apply(mats[0], mats[1], mats[2], wav, c)
    return seis.transpose(0, 1)


def _run_dd(prop, vp, wav, src, rec, px):
    return SEMModelParallel(prop, StripTopology(px))(wav, src, rec, [vp])


@pytest.mark.parametrize("px", [1, 2, 3])
def test_dd_forward_matches_fused(px):
    """px=1 isolates the stepped split; px>1 adds the decomposition."""
    prop = _prop()
    vp = _vp()
    src, rec = _geometry(prop, vp)
    wav = _wav()
    with torch.no_grad():
        ref1 = _fused_reference(prop, vp, wav, src, rec)
        ref2 = _fused_reference(prop, vp, wav, src, rec)   # atomics floor
        got = _run_dd(prop, vp, wav, src, rec, px)
    floor = float((ref1 - ref2).abs().max() / ref1.abs().max())
    err = float((got - ref1).abs().max() / ref1.abs().max())
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"px={px}: DD vs fused rel {err:.2e}, floor {floor:.2e}")


def test_dd_gradients_match_fused():
    """vp AND wavelet gradients. The wavelet gradient is the falsifier for
    the ownership mask (src_pattern_own): without the dedup, every interface
    column double-counts and the error sits far above the floor."""
    prop = _prop()

    def grads(runner):
        vp, wav = _vp(grad=True), _wav(grad=True)
        src, rec = _geometry(prop, vp)
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone(), wav.grad.clone()

    gv_ref, gw_ref = grads(_fused_reference)
    gv_ref2, gw_ref2 = grads(_fused_reference)             # atomics floor
    gv_dd, gw_dd = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, 3))
    for nm, ref, ref2, got in (("vp", gv_ref, gv_ref2, gv_dd),
                               ("wavelet", gw_ref, gw_ref2, gw_dd)):
        floor = float((ref - ref2).abs().max() / ref.abs().max())
        err = float((got - ref).abs().max() / ref.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{nm}: DD grad vs fused rel {err:.2e}, floor {floor:.2e}")


def test_interface_receiver_owned_once():
    """A receiver EXACTLY on a cut column: owned by the RIGHT strip, traced
    once, identical to the fused reference."""
    prop = _prop()
    vp = _vp()
    eq, _ = prop._resolve_model_batch([vp])
    nxg = eq.mesh.nex * NO + 1
    cut_col = (eq.mesh.nex // 2) * NO
    src, _ = _geometry(prop, vp)
    rec = torch.tensor([2 * nxg + cut_col], device="cuda")
    wav = _wav()
    with torch.no_grad():
        ref = _fused_reference(prop, vp, wav, src, rec)
        got = _run_dd(prop, vp, wav, src, rec, 2)
    err = float((got - ref).abs().max() / ref.abs().max())
    assert err <= ABS_CEILING


def _cut_cols(prop, vp, px):
    eq, _ = prop._resolve_model_batch([vp])
    nex = eq.mesh.nex
    return [(nex // px) * r * NO for r in range(1, px)], nex * NO + 1


@pytest.mark.parametrize("case", ["rec_on_cut", "rec_on_both_cuts",
                                  "src_on_cut", "src_and_rec_on_cut"],
                         ids=str)
def test_dd_gradients_with_cut_geometry(case):
    """Gradient consistency with sources/receivers EXACTLY on cut columns.

    Both falsifiers here caught real bugs:
    * a cut-column receiver injected the adjoint source only on the owning
      strip -- the neighbour's interface adjoint missed it and the vp
      gradient was off by rel 0.94 (fix: register on both strips, patch the
      duplicate's grad_syn from the owner);
    * a cut-column source was DOUBLED by the setup-time pattern exchange --
      the sliced global pattern is already complete on both copies (fix:
      no setup exchange; forward was off by rel 1.0).
    """
    prop = _prop()
    px = 3 if case == "rec_on_both_cuts" else 2
    cuts, nxg = _cut_cols(prop, _vp(), px)

    def geometry():
        src_col = cuts[0] if "src" in case else 5 * NO + 1
        if case == "rec_on_cut":
            rec_cols = [cuts[0], 7, 25]
        elif case == "rec_on_both_cuts":
            rec_cols = [cuts[0], cuts[1], 25]
        elif case == "src_on_cut":
            rec_cols = [3, 7, 25, 40]
        else:                                   # src_and_rec_on_cut
            rec_cols = [cuts[0], 25]
        return (2 * NO * nxg + src_col,
                (torch.tensor(rec_cols) + 2 * nxg).cuda())

    def grads(runner):
        vp, wav = _vp(grad=True), _wav(grad=True)
        src, rec = geometry()
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), vp.grad.clone(), wav.grad.clone()

    s_r, gv_r, gw_r = grads(_fused_reference)
    s_r2, gv_r2, gw_r2 = grads(_fused_reference)           # atomics floor
    s_d, gv_d, gw_d = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px))
    for nm, ref, ref2, got in (("fwd", s_r, s_r2, s_d),
                               ("vp", gv_r, gv_r2, gv_d),
                               ("wavelet", gw_r, gw_r2, gw_d)):
        floor = float((ref - ref2).abs().max() / ref.abs().max())
        err = float((got - ref).abs().max() / ref.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{case}/{nm}: DD vs fused rel {err:.2e}, floor {floor:.2e}")


def test_rebuilt_profile_would_be_wrong():
    """Red-light drill for the PML-hole mirror: a per-strip profile rebuild
    grows PML at the cut; the forward's interface assertion must fire."""
    from sweep.sem.cpml.acoustic import build_cpml
    from sweep.sem.mesh.quad9 import build_axis_aligned_quad9
    prop = _prop()
    vp = _vp()
    src, rec = _geometry(prop, vp)
    ddp = SEMModelParallel(prop, StripTopology(2))
    orig = ddp._parts

    def strip_mesh(eq_g, pp):
        nzl = eq_g.mesh.nez * NO + 1
        nex_l = (pp.c["n_global"] // nzl - 1) // NO
        return build_axis_aligned_quad9(
            nex=nex_l, nez=eq_g.mesh.nez,
            x_max=float(eq_g.mesh.x_max) * nex_l / eq_g.mesh.nex,
            z_max=float(eq_g.mesh.z_max), n_order=NO,
            dtype=torch.float32, device="cuda")

    def sabotaged(eq_g, pattern_g, receivers):
        parts, mats = orig(eq_g, pattern_g, receivers)
        for pp in parts:
            cp = build_cpml(  # rebuild on the STRIP mesh: PML at the cut
                strip_mesh(eq_g, pp), dt=DT,
                sides=("left", "right", "bottom"), pml_thickness_elements=3,
                c_max=2400.0, f_dominant=F0)
            pp.c["b_x"] = cp.b_x.to("cuda", torch.float32)
        return parts, mats

    ddp._parts = sabotaged
    with pytest.raises(AssertionError):
        ddp(_wav(), src, rec, [vp])


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(x @ y / (x.norm() * y.norm()))


@pytest.mark.parametrize("px", [1, 2, 3])
def test_dd_boundary_saving_matches_save_all(px):
    """Boundary saving composes with the decomposition -- not because a cut
    column avoids the ring (it does not: the bottom rim reaches it) but
    because the profile is sliced from ONE global
    build, so both strips give the shared column the same band membership
    and reconstruct it the same way. No extra interface exchange needed.

    Bars follow the SINGLE-DOMAIN boundary-saving convention (cos + relative
    L2, see test_sem_boundary_saving.py), not the pointwise-max bar the rest
    of this file uses against the fused driver: BS is a RECONSTRUCTION, and
    the single-domain path itself sits at ~6e-4 pointwise / ~1e-5 in L2
    against its own save-all tape. The DD-specific question is whether the
    decomposition ADDS error on top of that, which the second block below
    isolates by comparing DD-bs against FUSED-bs directly.
    """
    prop = _prop()
    src, rec = _geometry(prop, _vp())
    if px > 1:                       # make the cut columns load-bearing
        cuts, nxg = _cut_cols(prop, _vp(), px)
        rec = torch.cat([rec, (torch.tensor(cuts, device="cuda") + 2 * nxg)])

    def grads(prop):
        vp, wav = _vp(grad=True), _wav(grad=True)
        out = _run_dd(prop, vp, wav, src, rec, px)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), vp.grad.clone(), wav.grad.clone()

    s_a, gv_a, gw_a = grads(prop)             # save-all
    s_b, gv_b, gw_b = grads(_prop(BoundarySaving()))
    # the forward is NOT a reconstruction: it must match at the atomics floor
    assert float((s_a - s_b).abs().max() / s_a.abs().max()) <= ABS_CEILING
    for nm, ref, got in (("vp", gv_a, gv_b), ("wavelet", gw_a, gw_b)):
        assert _cos(ref, got) > 0.9999, f"px={px}/{nm}: cos {_cos(ref, got)}"
        l2 = float((got - ref).norm() / ref.norm())
        assert l2 < 1e-3, f"px={px}/{nm}: DD-bs vs DD-save-all L2 {l2:.2e}"


@pytest.mark.parametrize("px", [2, 3])
def test_dd_boundary_saving_adds_no_error(px):
    """The DD-specific bar: DD-bs must be as close to FUSED-bs as DD-save-all
    is to fused-save-all. Any reconstruction error the cuts introduced would
    show up here even though the previous test's BS-vs-tape bars are loose."""
    prop = _prop()
    src, rec = _geometry(prop, _vp())

    def grads(runner, prop=prop):
        vp, wav = _vp(grad=True), _wav(grad=True)
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone()

    def fused_bs(p, vp, wav, s, r):
        eq, _ = p._resolve_model_batch([vp])
        pattern = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                              device=eq.mesh.ibool.device)
        pattern[int(s)] = 1.0
        d1 = SEMModelParallel(p, StripTopology(1))
        parts, mats = d1._parts(eq, pattern, r)
        c = dict(parts[0].c)
        c["bs"], c["bs_kind"] = True, 0
        return _SEMAcousticCpmlC.apply(mats[0], mats[1], mats[2], wav,
                                       c).transpose(0, 1)

    g_ref_tape = grads(_fused_reference)
    g_dd_tape = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px))
    base = float((g_dd_tape - g_ref_tape).norm() / g_ref_tape.norm())

    g_ref_bs = grads(fused_bs)
    g_dd_bs = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px),
                    _prop(BoundarySaving()))
    with_bs = float((g_dd_bs - g_ref_bs).norm() / g_ref_bs.norm())
    assert with_bs <= max(10 * base, 1e-3), (
        f"px={px}: DD adds error under bs (L2 {with_bs:.2e}) beyond its "
        f"save-all baseline ({base:.2e})")


def test_dd_boundary_saving_saves_memory():
    """The ring must actually be smaller than the tape -- otherwise the flag
    could be silently ignored and every equality test above would still pass.

    Uses a BIGGER grid than the rest of the file on purpose: the ring saves
    O(nt * |band|) against O(nt * ng), so the win only shows once the domain
    is much wider than its C-PML rim (when a 3-element rim covered nearly the
    whole domain, the two were within 5%)."""
    nz, nx, nt = 65, 129, 400

    def build(memory=None):
        return S.PropSEM(AcousticPotentialSEM2D, shape=(nz, nx), dh=DH,
                         dt=DT, nt=nt, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, memory=memory, f_dominant=F0,
                         dev="cuda", impl="c")

    prop, prop_bs = build(), build(BoundarySaving())
    torch.manual_seed(0)
    vp = (2000.0 + 300.0 * torch.rand(nz, nx)).cuda().requires_grad_(True)
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                       device="cuda")
    eq, _ = prop._resolve_model_batch([vp.detach()])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + nxg // 2
    rec = (torch.arange(4, nxg - 4, 4) + 1 * nxg).cuda()

    # What a call allocates, not what earlier calls left behind: an absolute
    # peak also carries the allocations that stay (the 32 MiB cuBLAS workspace
    # on Hopper, whatever earlier tests hold), which sank this ratio on an H200
    # while it passed on Ada.  One warm-up call allocates those first -- one
    # per propagator, since each keeps its own mesh and C-PML cache.
    def peak(prop):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = SEMModelParallel(prop, StripTopology(2))(wav, src, rec, [vp])
        (out ** 2).sum().backward()
        vp.grad = None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    peak(prop)
    peak(prop_bs)
    p_tape, p_ring = peak(prop), peak(prop_bs)
    assert p_ring < 0.85 * p_tape, (
        f"boundary saving did not reduce peak memory: ring {p_ring/2**20:.0f} "
        f"MiB vs tape {p_tape/2**20:.0f} MiB")


def test_topology_validation():
    with pytest.raises(ValueError):
        StripTopology(0)
    with pytest.raises(ValueError):
        StripTopology(2, rank=2)
    with pytest.raises(ValueError):
        StripTopology(5).local_elems(12)
    assert StripTopology(3, rank=1).local_elems(12) == (4, 8)
    t = StripTopology(3, rank=1)
    assert not (t.is_edge_lo or t.is_edge_hi)


# A grid whose C-PML band does NOT swallow the source. That is the single
# thing low-precision rings are sensitive to: int8 keeps one amax per
# (field, step) over the whole band, so a source cell inside the ring makes
# the amax the source spike and quantises every other node to noise
# (measured on a grid whose band held the source: bf16 L2 1.3e+01, int8
# 1.3e+03).
# Move the source out of the band and the same code gives 6e-6 / 3.5e-6.
RING = dict(shape=(49, 97), nt=400, abcn=3, src_elem_z=2)


def _ring_prop(memory=None):
    return S.PropSEM(AcousticPotentialSEM2D, shape=RING["shape"], dh=DH,
                     dt=DT, nt=RING["nt"], n_order=NO, free_surface=("top",),
                     abcn=RING["abcn"] * NO, memory=memory,
                     f_dominant=F0, dev="cuda", impl="c")


def _ring_setup():
    prop = _ring_prop()

    def vp(grad=False):
        torch.manual_seed(0)
        v = (2000.0 + 300.0 * torch.rand(*RING["shape"])).cuda()
        return v.requires_grad_(True) if grad else v

    t = np.arange(RING["nt"]) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                       device="cuda")
    eq, _ = prop._resolve_model_batch([vp()])
    nxg = eq.mesh.nex * NO + 1
    src = RING["src_elem_z"] * NO * nxg + (eq.mesh.nex // 2) * NO
    rec = (torch.arange(4, nxg - 4, 4) + 1 * nxg).cuda()
    return prop, vp, wav, src, rec


@pytest.mark.parametrize("store", ["bf16", "int8"])
def test_dd_low_precision_ring(store):
    """bf16 / int8 ring storage on the 2-D DD path.

    Three bars. (1) the ring really is that dtype and int8 carries its amax
    table; (2) the single-domain suite's accuracy bar -- the gradient stays
    within storage-rounding noise of the fp32 ring; (3) the DD-specific one
    -- the decomposition must not add quantisation error of ITS own, i.e.
    its degradation must match the FUSED driver's on the same problem.

    Plus a guard that the source is outside the band, because that -- not
    the decomposition -- is what decides whether int8 is usable at all."""
    prop, vp, wav, src, rec = _ring_setup()
    eq, _ = prop._resolve_model_batch([vp()])
    pattern = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                          device=eq.mesh.ibool.device)
    pattern[int(src)] = 1.0
    parts, _m = SEMModelParallel(prop, StripTopology(2))._parts(
        eq, pattern, rec)
    assert not any(int(src) in set(p.gidx[p.c["band_idx"]].tolist())
                   for p in parts), (
        "the source is inside the band ring: int8's per-step amax becomes "
        "the source spike and this test would be measuring that, not DD")

    seen = {}
    real = _SEMAcousticCpmlDDC.forward

    def spy(ctx, wavelet, pp, exchange, bs, *mats):
        out = real(ctx, wavelet, pp, exchange, bs, *mats)
        seen["carry"] = ctx.carry
        return out

    def dd(storage_dtype):
        v = vp(grad=True)
        out = SEMModelParallel(
            _ring_prop(BoundarySaving(storage_dtype=storage_dtype)),
            StripTopology(2))(wav, src, rec, [v])
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), v.grad.clone()

    def fused(kind):
        v = vp(grad=True)
        e, _ = prop._resolve_model_batch([v])
        pat = torch.zeros(e.mesh.n_global, dtype=torch.float32,
                          device=e.mesh.ibool.device)
        pat[int(src)] = 1.0
        pp, mats = SEMModelParallel(prop, StripTopology(1))._parts(
            e, pat, rec)
        c = dict(pp[0].c)
        c["bs"], c["bs_kind"] = True, kind
        out = _SEMAcousticCpmlC.apply(mats[0], mats[1], mats[2], wav,
                                      c).transpose(0, 1)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return v.grad.clone()

    s32, g32 = dd("fp32")
    _SEMAcousticCpmlDDC.forward = staticmethod(spy)
    try:
        sq, gq = dd(store)
    finally:
        _SEMAcousticCpmlDDC.forward = staticmethod(real)

    want = torch.bfloat16 if store == "bf16" else torch.int8
    for cr in seen["carry"]:
        for k in _AC2D_RINGS:
            assert cr[k].dtype == want, f"ring is {cr[k].dtype}, wanted {want}"
        scale = cr["band_scale"]
        if store == "int8":            # one amax per (field, step), 5 fields
            assert tuple(scale.shape) == (5, RING["nt"])
            assert float(scale.max()) > 0
        else:
            assert scale.numel() == 0
    assert torch.isfinite(gq).all()
    # the ring is read only in the backward: the record must not move
    assert float((sq - s32).abs().max() / s32.abs().max()) < 1e-3
    l2 = float((gq - g32).norm() / g32.norm())
    assert l2 < 5e-3, f"{store} ring vs fp32 ring: grad L2 {l2:.2e}"
    kind = 1 if store == "bf16" else 2
    f32 = fused(0)
    l2_f = float((fused(kind) - f32).norm() / f32.norm())
    assert l2 <= max(1.5 * l2_f, 1e-4), (
        f"{store}: DD degrades {l2:.2e} where the fused driver degrades "
        f"{l2_f:.2e} on the same problem")


# The "one path still refuses low precision" test used to live here. Every
# SEM boundary-saving path takes bs_store_dtype now, so the guard against a
# path silently storing fp32 flipped into a positive invariant and moved to
# test_sem_bs_store_dtype.py::test_every_boundary_saving_path_honours_the_
# store_dtype -- ask each path for int8, assert the ring really is int8.

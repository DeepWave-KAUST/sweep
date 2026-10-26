"""3-D acoustic C-PML domain decomposition: DD stepped loop vs the fused driver.

Same discipline as the 2-D suites: the reference is the SAME per-strip consts
through the FUSED single-domain C driver, px=1 isolates the stepped split,
px>1 adds the decomposition, thresholds anchor to the fused driver's own
atomics floor.

What is different in 3-D: a cut is a whole YZ PLANE, not a column. The global
numbering ``((iz*ny_g + iy)*nx_g + ix)`` puts x fastest, so the 2-D geometry
generalises by widening the row axis to ``nz_g * ny_g`` -- and the y extent
being > 1 is exactly what these tests exercise (a bug that collapsed the
plane to a single row would still pass any ny=1 check).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import (SEMModelParallel, StripTopology,
                          _SEMAcoustic3DCpmlDDC, _AC3D_RINGS, _AC3D_TAPES)
from sweep.sem.equations import AcousticSEM3D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 40.0, 8.0
# 4 x 3 x 2 model elements; the 2-element absorbing layer outside it makes the
# mesh 6 x 7 x 6, and nex=6 is what px=2 and 3 must divide
NZ, NY, NX = 17, 13, 9
# NT runs past the Ricker peak (t0 = 1.2/F0 = 0.15 s -> step 125): with a
# source that has not really fired the comparisons still hold (they are
# DD-vs-fused on identical inputs) but they grade a thin wavefield.
NT, DT = 170, 1.2e-3
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4


def _prop(memory=None):
    return S.PropSEM(AcousticSEM3D, shape=(NZ, NY, NX), dh=DH, dt=DT, nt=NT,
                     n_order=NO, free_surface=("top",), abcn=2 * NO,
                     memory=memory, f_dominant=F0, dev="cuda", impl="c")


def _vp(grad=False):
    torch.manual_seed(0)
    v = (2500.0 + 200.0 * torch.rand(NZ, NY, NX)).cuda()
    return v.requires_grad_(True) if grad else v


def _wav(grad=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                     device="cuda")
    return w.requires_grad_(True) if grad else w


def _dims(prop, vp):
    eq, _ = prop._resolve_model_batch([vp.detach()])
    m = eq.mesh
    return (m.nex * NO + 1, m.ney * NO + 1, m.nez * NO + 1, m.nex)


def _gid(ix, iy, iz, nxg, nyg):
    return (iz * nyg + iy) * nxg + ix


def _geometry(prop, vp, rec_x=None, src_x=None):
    nxg, nyg, nzg, nex = _dims(prop, vp)
    sx = src_x if src_x is not None else nxg // 2
    src = _gid(sx, nyg // 2, 2 * NO, nxg, nyg)
    xs = rec_x if rec_x is not None else list(range(3, nxg - 3, 3))
    # receivers spread over TWO y rows so a plane collapsed to one row shows up
    rec = torch.tensor([_gid(x, y, NO, nxg, nyg)
                        for y in (nyg // 3, 2 * nyg // 3) for x in xs]).cuda()
    return src, rec


def _fused_reference(prop, vp, wav, src, rec):
    """The single-strip run of the same driver: the fused single-domain
    Function it used to call is gone (PropSEM routes through this too)."""
    return SEMModelParallel(prop, StripTopology(1))(wav, src, rec, [vp])


def _run_dd(prop, vp, wav, src, rec, px):
    return SEMModelParallel(prop, StripTopology(px))(wav, src, rec, [vp])


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(x @ y / (x.norm() * y.norm()))


@pytest.mark.parametrize("px", [1, 2, 3])
def test_dd3d_forward_matches_fused(px):
    prop = _prop()
    vp, wav = _vp(), _wav()
    src, rec = _geometry(prop, vp)
    with torch.no_grad():
        ref1 = _fused_reference(prop, vp, wav, src, rec)
        ref2 = _fused_reference(prop, vp, wav, src, rec)
        got = _run_dd(prop, vp, wav, src, rec, px)
    floor = float((ref1 - ref2).abs().max() / ref1.abs().max())
    err = float((got - ref1).abs().max() / ref1.abs().max())
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"px={px}: 3-D DD vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [2, 3])
def test_dd3d_gradients_match_fused(px):
    prop = _prop()

    def grads(runner):
        vp, wav = _vp(grad=True), _wav(grad=True)
        src, rec = _geometry(prop, vp)
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone(), wav.grad.clone()

    gv, gw = grads(_fused_reference)
    gv2, gw2 = grads(_fused_reference)
    gvd, gwd = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px))
    assert float(gv.abs().max()) > 0.0, "reference gradient is all zeros"
    for nm, a, b, c in (("vp", gv, gv2, gvd), ("wavelet", gw, gw2, gwd)):
        floor = float((a - b).abs().max() / a.abs().max())
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"px={px}/{nm}: 3-D DD grad vs fused rel {err:.2e}, "
            f"floor {floor:.2e}")


@pytest.mark.parametrize("case", ["rec_on_cut", "src_on_cut"], ids=str)
def test_dd3d_cut_plane_geometry(case):
    """Sources / receivers exactly ON the cut PLANE -- the 3-D form of the
    geometry that caught two real bugs on the 2-D acoustic path."""
    prop = _prop()
    px = 2
    nxg, nyg, nzg, nex = _dims(prop, _vp())
    cut = (nex // px) * NO

    def grads(runner):
        vp, wav = _vp(grad=True), _wav(grad=True)
        if case == "src_on_cut":
            src, rec = _geometry(prop, vp, src_x=cut)
        else:
            src, rec = _geometry(prop, vp, rec_x=[cut, cut - 3, 3, nxg - 4])
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), vp.grad.clone(), wav.grad.clone()

    ref = grads(_fused_reference)
    ref2 = grads(_fused_reference)
    got = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px))
    for nm, a, b, c in zip(("fwd", "vp", "wavelet"), ref, ref2, got):
        floor = float((a - b).abs().max() / a.abs().max())
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{case}/{nm}: 3-D DD vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [1, 2, 3])
def test_dd3d_boundary_saving_matches_save_all(px):
    """Boundary saving composes with the 3-D decomposition: the cut PLANE
    carries no C-PML, so it sits OUTSIDE the band ring and the interior
    reverse march recomputes it -- no extra interface exchange.

    Bars follow the single-domain BS convention (cos + relative L2), not the
    pointwise-max bar the rest of this file uses: BS is a RECONSTRUCTION and
    the single-domain path already differs from its own tape at that level.
    Whether the DECOMPOSITION adds error on top is isolated by the next
    test, which compares DD-bs against fused-bs directly.

    NOTE the default grid's band covers 100% of its nodes (a 3-D C-PML rim
    is a shell on five faces and swallows a small domain), so here every
    node is ring-restored and the reverse MARCH is masked. This test checks
    the ring plumbing and the cut geometry; the march arithmetic is checked
    on the wide grid by ``test_dd3d_boundary_saving_on_a_real_rim``."""
    prop = _prop()
    vp0 = _vp()
    src, rec = _geometry(prop, vp0)
    if px > 1:                        # make the cut plane load-bearing
        nxg, nyg, nzg, nex = _dims(prop, vp0)
        cut = (nex // px) * NO
        rec = torch.cat([rec, torch.tensor(
            [_gid(cut, y, 2 * NO, nxg, nyg)
             for y in (nyg // 3, 2 * nyg // 3)]).cuda()])

    def grads(prop):
        vp, wav = _vp(grad=True), _wav(grad=True)
        out = _run_dd(prop, vp, wav, src, rec, px)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), vp.grad.clone(), wav.grad.clone()

    s_a, gv_a, gw_a = grads(prop)                # save-all tape
    s_b, gv_b, gw_b = grads(_prop(BoundarySaving()))
    assert float(s_a.abs().max()) > 0.0, "reference record is all zeros"
    # the forward is NOT a reconstruction -- it must match at the floor
    assert float((s_a - s_b).abs().max() / s_a.abs().max()) <= ABS_CEILING
    for nm, ref, got in (("vp", gv_a, gv_b), ("wavelet", gw_a, gw_b)):
        assert _cos(ref, got) > 0.9999, f"px={px}/{nm}: cos {_cos(ref, got)}"
        l2 = float((got - ref).norm() / ref.norm())
        assert l2 < 1e-3, f"px={px}/{nm}: DD-bs vs DD-save-all L2 {l2:.2e}"


@pytest.mark.parametrize("px", [2, 3])
def test_dd3d_boundary_saving_adds_no_error(px):
    """The DD-specific bar: DD-bs must sit as close to FUSED-bs as DD-tape
    sits to fused-tape. Reconstruction error introduced BY the cuts shows up
    here even though the previous test's BS-vs-tape bars are loose."""
    prop = _prop()
    src, rec = _geometry(prop, _vp())
    # the memory is a constructor option: one propagator per mode
    props = {None: prop, "boundary": _prop(BoundarySaving())}

    def grads(runner, memory):
        vp, wav = _vp(grad=True), _wav(grad=True)
        out = runner(props[memory], vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone()

    g_ref_tape = grads(_fused_reference, None)
    g_dd_tape = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px), None)
    base = float((g_dd_tape - g_ref_tape).norm() / g_ref_tape.norm())
    g_ref_bs = grads(_fused_reference, "boundary")
    g_dd_bs = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px),
                    "boundary")
    with_bs = float((g_dd_bs - g_ref_bs).norm() / g_ref_bs.norm())
    assert with_bs <= max(10 * base, 1e-3), (
        f"px={px}: DD adds error under bs (L2 {with_bs:.2e}) beyond its "
        f"save-all baseline ({base:.2e})")


@pytest.mark.parametrize("px", [2, 3])
def test_dd3d_band_membership_agrees_across_the_cut(px):
    """The invariant that actually makes BS compose with DD.

    It is NOT "the cut plane is outside the ring" -- the rim reaches into
    every cut plane here, as it does into the 2-D cut columns. What
    holds, because the profile is sliced from ONE global build, is that the
    two strips sharing a cut assign that plane IDENTICAL band membership.
    So a shared node is either restored on both sides from rings holding the
    same exchanged forward values, or recomputed on both sides by the same
    pointwise march -- never one of each. A per-strip profile rebuild would
    break exactly this, which is why the sliced-profile assertion and this
    one are the pair that keep boundary saving honest."""
    prop = _prop()
    vp = _vp()
    src, rec = _geometry(prop, vp)
    eq, _ = prop._resolve_model_batch([vp])
    pat = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                      device=eq.mesh.ibool.device)
    pat[int(src)] = 1.0
    parts, _m = SEMModelParallel(prop, StripTopology(px))._parts3d(
        eq, (pat,), rec)
    seen = 0
    for lo, hi in zip(parts[:-1], parts[1:]):
        g_lo, g_hi = lo.gidx[lo.if_hi], hi.gidx[hi.if_lo]
        assert torch.equal(g_lo, g_hi), "the strips do not name the same plane"
        plane = set(g_lo.tolist())
        b_lo = set(lo.gidx[lo.c["band_idx"]].tolist()) & plane
        b_hi = set(hi.gidx[hi.c["band_idx"]].tolist()) & plane
        assert b_lo == b_hi, (
            f"band membership disagrees on the shared plane: "
            f"{len(b_lo)} vs {len(b_hi)} of {len(plane)}")
        seen += 1
    assert seen == px - 1


def test_dd3d_boundary_saving_actually_rings():
    """Mechanism check, independent of any grid: under BoundarySaving() the
    save-all tapes must be EMPTY and the rings must be (nt, |band|) for u and
    (nt, |band & b_ax != 0|) for the C-PML memory fields psi_ax / zeta_ax --
    the per-field ring split (test_sem_bs_ring_split.py owns its proof).
    Without this a silently ignored flag would still pass every equality
    test."""
    prop = _prop(BoundarySaving())
    vp, wav = _vp(grad=True), _wav()
    src, rec = _geometry(prop, vp)
    ddp = SEMModelParallel(prop, StripTopology(2))
    seen = {}
    real = _SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, wavelet, parts, exchange, bs, *mats):
        out = real(ctx, wavelet, parts, exchange, bs, *mats)
        seen["rings"] = [[cr[k] for k in _AC3D_RINGS + ("fin_u", "fin_u_prev")]
                         for cr in ctx.carry]
        seen["saves"] = [[cr[k] for k in _AC3D_TAPES] for cr in ctx.carry]
        seen["nband"] = [int(p.c["band_idx"].numel()) for p in parts]
        seen["support"] = [{ax: int(p.c[f"band_idx_{ax}"].numel()) for ax in "xyz"}
                           for p in parts]
        return out

    _SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    try:
        ddp(wav, src, rec, [vp])
    finally:
        _SEMAcoustic3DCpmlDDC.forward = staticmethod(real)
    assert seen, "the DD Function never ran"
    for tapes in seen["saves"]:
        assert all(t.numel() == 0 for t in tapes), "save-all tape allocated"
    axes = ("x", "y", "z", "x", "y", "z")             # psi_x..z, zeta_x..z
    for ring, nb, sup in zip(seen["rings"], seen["nband"], seen["support"]):
        assert nb > 0
        assert tuple(ring[0].shape) == (NT, nb), f"u ring {tuple(ring[0].shape)}"
        for r, ax in zip(ring[1:7], axes):
            assert 0 < sup[ax] <= nb
            assert tuple(r.shape) == (NT, sup[ax]), f"{ax} ring {tuple(r.shape)}"
        for f in ring[7:9]:                      # fin_u, fin_u_prev
            assert f.numel() > 0 and float(f.abs().max()) > 0.0


# A grid whose band is a REAL rim: the small grid above has band == 100% of
# its nodes, which makes boundary saving trivially exact (nothing is marched)
# and makes int8's per-step amax the source spike. Everything that depends on
# the reverse march actually running, or on the ring's dynamic range, has to
# be measured here instead.
# nt must run PAST the Ricker peak (t0 = 1.2/F0 = 0.15 s at DT = 1.2e-3 is
# step 125): the file's default NT=120 stops before the source has really
# fired, which is harmless for floor-level comparisons but would grade the
# boundary-saving reconstruction against noise.
RIM = dict(shape=(33, 49, 129), nt=250, abcn=2)


def _rim_prop(memory=None):
    return S.PropSEM(AcousticSEM3D, shape=RIM["shape"], dh=DH, dt=DT,
                     nt=RIM["nt"], n_order=NO, free_surface=("top",),
                     abcn=RIM["abcn"] * NO, memory=memory,
                     f_dominant=F0, dev="cuda", impl="c")


def _rim():
    prop = _rim_prop()

    def vp(grad=False):
        torch.manual_seed(0)
        v = (2500.0 + 200.0 * torch.rand(*RIM["shape"])).cuda()
        return v.requires_grad_(True) if grad else v

    t = np.arange(RIM["nt"]) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                       device="cuda")
    eq, _ = prop._resolve_model_batch([vp()])
    m = eq.mesh
    nxg, nyg = m.nex * NO + 1, m.ney * NO + 1
    src = _gid(nxg // 2, nyg // 2, 4 * NO, nxg, nyg)
    rec = torch.tensor([_gid(x, y, NO, nxg, nyg)
                        for y in (nyg // 3, 2 * nyg // 3)
                        for x in range(8, nxg - 8, 8)]).cuda()
    return prop, vp, wav, src, rec


def _rim_band_fraction(prop, vp, src, rec, px=2):
    eq, _ = prop._resolve_model_batch([vp()])
    pat = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                      device=eq.mesh.ibool.device)
    pat[int(src)] = 1.0
    parts, _m = SEMModelParallel(prop, StripTopology(px))._parts3d(
        eq, (pat,), rec)
    return parts, [int(p.c["band_idx"].numel()) / p.c["n_global"]
                   for p in parts]


def test_dd3d_boundary_saving_on_a_real_rim():
    """BS x DD where the reverse march genuinely runs.

    The band must be a real rim (guarded below): otherwise every node is
    ring-restored, the march arithmetic is never executed, and this test
    would silently become a weaker copy of the small-grid one."""
    prop, vp, wav, src, rec = _rim()
    parts, frac = _rim_band_fraction(prop, vp, src, rec)
    assert max(frac) < 0.9, (
        f"band covers {max(frac):.0%} of the strip: the march is masked, so "
        "this grid tests nothing the small-grid test does not")

    def grads(runner, memory):
        v = vp(grad=True)
        out = runner(v, memory)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return v.grad.clone()

    # the memory is a constructor option: one propagator per mode
    props = {None: prop, "boundary": _rim_prop(BoundarySaving())}

    def dd(v, memory):
        return SEMModelParallel(props[memory], StripTopology(2))(
            wav, src, rec, [v])

    def fused(v, memory):
        return SEMModelParallel(props[memory], StripTopology(1))(wav, src, rec, [v])

    g_dd_t, g_dd_b = grads(dd, None), grads(dd, "boundary")
    assert _cos(g_dd_b, g_dd_t) > 0.9999
    assert float((g_dd_b - g_dd_t).norm() / g_dd_t.norm()) < 1e-3
    g_f_t, g_f_b = grads(fused, None), grads(fused, "boundary")
    base = float((g_dd_t - g_f_t).norm() / g_f_t.norm())
    with_bs = float((g_dd_b - g_f_b).norm() / g_f_b.norm())
    assert with_bs <= max(100 * base, 1e-3), (
        f"DD adds error under bs on a real rim (L2 {with_bs:.2e}) beyond its "
        f"save-all baseline ({base:.2e})")


def test_dd3d_boundary_saving_saves_memory():
    """The ring must actually buy memory. The bar is the ANALYTIC saving --
    7 histories x nt x (ng - |band|) x 4 B per strip -- not an arbitrary
    ratio, because in 3-D a C-PML rim is a shell on five of six faces and
    stays a large fraction of the volume at any test size. Asserting a fixed
    ratio would only be asserting the grid; asserting that the predicted
    saving is realised tests the ring and goes to zero if the flag is
    ignored.

    The reference is the SEVEN-tape save-all mode the formula describes
    (prop_c._WSUM_TAPE = False). The default save-all mode now keeps one
    w_sum tape, 4 B/node/step, and on a grid this small that is LESS than
    boundary saving's rings plus its per-node reconstruction buffers (the
    rings win once nt * n_global stops fitting)."""
    import sweep.sem.prop_c as PC
    prop, vp, wav, src, rec = _rim()
    parts, _frac = _rim_band_fraction(prop, vp, src, rec)
    predicted = sum(7 * RIM["nt"]
                    * (p.c["n_global"] - int(p.c["band_idx"].numel())) * 4
                    for p in parts)
    assert predicted > 0, "the band covers the whole domain: pick a wider grid"
    v = vp(grad=True)

    # one propagator per memory mode (a constructor option), each driven
    # through its own strip driver -- built inside its window, so both peaks
    # carry one set of strip meshes, as the shared driver used to give them
    def peak(prop, wtape=True):
        old = PC._WSUM_TAPE
        PC._WSUM_TAPE = wtape
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out = SEMModelParallel(prop, StripTopology(2))(wav, src, rec, [v])
            (out ** 2).sum().backward()
            v.grad = None
            return torch.cuda.max_memory_allocated()
        finally:
            PC._WSUM_TAPE = old

    mem_tape = peak(prop, wtape=False)
    mem_ring = peak(_rim_prop(BoundarySaving()))
    saved = mem_tape - mem_ring
    assert saved >= 0.8 * predicted, (
        f"ring saved {saved/1e6:.0f} MB of the {predicted/1e6:.0f} MB the "
        f"band size predicts (tape {mem_tape/1e6:.0f} MB, "
        f"ring {mem_ring/1e6:.0f} MB)")


@pytest.mark.parametrize("store", ["bf16", "int8"])
def test_dd3d_low_precision_ring(store):
    """bf16 / int8 ring storage on the 3-D DD path.

    Run on the RIM grid on purpose. int8 keeps one amax per (field, step) over
    the whole band, so its accuracy is set by the band's dynamic range -- and
    where the band takes in the source, the SOURCE SPIKE sets the amax and
    quantises the median node to zero (measured on a grid whose band was the
    entire domain: amax/median 1e6..1e9, int8 at L2 ~1e0 for the fused
    single-domain driver too). On a real rim the range is sane and int8 sits
    inside the single-domain suite's 5e-3 bar.

    The DD-specific bar is the second one: the decomposition must not add
    quantisation error of its own, i.e. DD's degradation must match the FUSED
    driver's on the same problem."""
    prop, vp, wav, src, rec = _rim()
    seen = {}
    real = _SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, wavelet, parts, exchange, bs, *mats):
        out = real(ctx, wavelet, parts, exchange, bs, *mats)
        seen["carry"] = ctx.carry
        return out

    def dd(storage_dtype):
        v = vp(grad=True)
        out = SEMModelParallel(
            _rim_prop(BoundarySaving(storage_dtype=storage_dtype)),
            StripTopology(2))(wav, src, rec, [v])
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), v.grad.clone()

    def fused(storage_dtype):
        v = vp(grad=True)
        out = SEMModelParallel(
            _rim_prop(BoundarySaving(storage_dtype=storage_dtype)),
            StripTopology(1))(wav, src, rec, [v])
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return v.grad.clone()

    s32, g32 = dd("fp32")
    _SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    try:
        sq, gq = dd(store)
    finally:
        _SEMAcoustic3DCpmlDDC.forward = staticmethod(real)

    want = torch.bfloat16 if store == "bf16" else torch.int8
    for cr in seen["carry"]:
        for k in _AC3D_RINGS:
            assert cr[k].dtype == want, f"ring is {cr[k].dtype}, wanted {want}"
        scale = cr["band_scale"]
        if store == "int8":       # one amax per (field, step), 7 fields
            assert tuple(scale.shape) == (7, RIM["nt"])
            assert float(scale.max()) > 0
        else:
            assert scale.numel() == 0
    assert torch.isfinite(gq).all()
    # the ring is read only in the backward: the record must not move
    assert float((sq - s32).abs().max() / s32.abs().max()) < 1e-3
    l2 = float((gq - g32).norm() / g32.norm())
    assert l2 < 5e-3, f"{store} ring vs fp32 ring: grad L2 {l2:.2e}"
    # ... and the decomposition must not add quantisation error of its own
    f32, fq = fused("fp32"), fused(store)
    l2_f = float((fq - f32).norm() / f32.norm())
    assert l2 <= max(1.5 * l2_f, 1e-4), (
        f"{store}: DD degrades {l2:.2e} where the fused driver degrades "
        f"{l2_f:.2e} on the same problem")


def test_dd3d_store_dtype_needs_boundary_saving():
    """A store dtype without memory='boundary' silently does nothing, which
    is exactly the kind of no-op flag that hides a mis-set experiment.

    The propagator can no longer express that pairing -- the store dtype is
    ``BoundarySaving(storage_dtype=...)`` and ``forward`` takes no memory
    option -- so the refusal is pinned where it can still be asked for: the
    strip driver's low-level ``run``."""
    prop = _prop()
    vp, wav = _vp(grad=True), _wav()
    src, rec = _geometry(prop, vp)
    eq, _ = prop._resolve_model_batch([vp.detach()])
    with pytest.raises(ValueError, match="memory='boundary'"):
        SEMModelParallel(prop, StripTopology(2)).run(
            eq, wav, src, rec, memory=None, bs_store_dtype="int8")

"""Elastic C-PML domain decomposition: DD stepped loop vs the fused driver.

Same discipline as the acoustic twin (test_sem_dd.py): the reference is the
SAME per-strip consts run through the FUSED single-domain C driver, px=1
isolates the stepped split itself, px>1 adds the decomposition, and the
thresholds anchor to the fused driver's own run-to-run atomics floor.

The elastic step has ONE assembly per time step (the element kernel's
Acc scatter), so DD exchanges 2 fields forward and 4 backward (+2 for the
boundary-saving march) -- fewer than the acoustic path's 6/6/3.
"""
from __future__ import annotations

import numpy as np
import itertools

import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import (SEMModelParallel, StripTopology,
                          _SEMElasticCpmlDDC)
from sweep.sem.equations import ElasticSEM2DDeformed
from sweep.sem.prop_c import _SEMElasticCpmlC, sem_c_available
from sweep.sem.drivers.elastic2d import _EL2D_RINGS

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 25.0, 8.0
NZ, NX = 25, 49                     # 6 x 12 elements
NT, DT = 200, 4e-4
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4
FLOOR_REPEATS = 4          # see _pooled_floor: one sample of it is not stable


def _pooled_floor(samples):
    """The A/A floor of one row, pooled over reference runs, not sampled once.

    Measured on this test when it was written, px=3, 12 rounds: the wavelet row's DD-vs-fused
    difference is 1.02e-4 to 1.89e-4 -- SYSTEMATICALLY above ABS_CEILING,
    because that row's own assembly noise is ~10x the model gradients'
    (floor 2.1e-5 against 2-3e-6, while err/floor is 6.3 against their 8, i.e.
    the wavelet is not anomalous, only larger). The bar is
    max(FLOOR_FACTOR * floor, ABS_CEILING), so the row passes on the floor
    term -- until a single sample of that floor lands low (3.14e-6 was
    observed, 7x under its median), the fixed ceiling takes over, and the test
    goes red having found nothing. Two of eight full-suite runs failed that
    way, never in isolation.

    Pooling the floor over repeats is what every other criterion in this tree
    does (on the columnar A/A floors, pooling only one arm made a change look
    2.25x out of family when it was at the floor).

    Measured, wavelet row at px=3, 14 rounds each: with ONE sample the worst
    bar came to 1.12e-4 against an err that reaches 1.50e-4 -- 1 round in 14
    red, which is the observed full-suite rate. Pooled over 3 the worst bar is
    2.70e-4 (1.8x the worst err); over 4 it is 5.69e-4 (3.8x), which is why
    FLOOR_REPEATS is 4. This does not loosen the criterion -- it is still
    FLOOR_FACTOR times a real floor, and the ratio err/floor the test actually
    sees (6.3 on the wavelet, 8 on the model gradients) is unchanged.
    """
    return max(float((a - b).abs().max() / a.abs().max())
               for a, b in itertools.combinations(samples, 2))


def _prop(memory=None):
    return S.PropSEM(ElasticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=False, abcn=3 * NO,
                     memory=memory, f_dominant=F0, dev="cuda", impl="c")


def _models(grad=False):
    z = torch.linspace(0, 1, NZ)[:, None].expand(NZ, NX)
    ms = [(2400.0 + 600.0 * z).contiguous().float().cuda(),
          (1300.0 + 350.0 * z).contiguous().float().cuda(),
          (2100.0 + 200.0 * z).contiguous().float().cuda()]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def _wav(grad=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e3,
                     device="cuda")
    return w.requires_grad_(True) if grad else w


def _geometry(prop, models):
    eq, _ = prop._resolve_model_batch([m.detach() for m in models])
    nxg = eq.mesh.nex * NO + 1
    src = 3 * NO * nxg + 5 * NO + 1
    rec = (torch.arange(4, nxg - 4, 4) + 2 * nxg).cuda()
    return src, rec


def _cut_cols(prop, models, px):
    eq, _ = prop._resolve_model_batch([m.detach() for m in models])
    return ([(eq.mesh.nex // px) * r * NO for r in range(1, px)],
            eq.mesh.nex * NO + 1)


def _fused_reference(prop, models, wav, src, rec, memory=None):
    """The px=1 consts through the FUSED single-domain elastic C driver."""
    eq, _ = prop._resolve_model_batch(models)
    d1 = SEMModelParallel(prop, StripTopology(1))
    pats = []
    for j in range(2):
        v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                        device=eq.mesh.ibool.device)
        if j == 1:
            v[int(src)] = 1.0
        pats.append(v)
    parts, mats = d1._parts(eq, tuple(pats), rec)
    c = dict(parts[0].c)
    c["bs"], c["bs_kind"] = memory == "boundary", 0
    sx, sz = _SEMElasticCpmlC.apply(mats[0], mats[1], mats[2], mats[3],
                                    mats[4], wav, c)
    return torch.stack([sx, sz], -1).transpose(0, 1)      # (nt, nrec, 2)


def _run_dd(prop, models, wav, src, rec, px):
    return SEMModelParallel(prop, StripTopology(px))(wav, src, rec, models)


@pytest.mark.parametrize("px", [1, 2, 3])
def test_elastic_dd_forward_matches_fused(px):
    prop = _prop()
    models, wav = _models(), _wav()
    src, rec = _geometry(prop, models)
    with torch.no_grad():
        refs = [_fused_reference(prop, models, wav, src, rec)
                for _ in range(FLOOR_REPEATS)]
        got = _run_dd(prop, models, wav, src, rec, px)
    ref1 = refs[0]
    assert got.shape == (NT, rec.numel(), 2)
    floor = _pooled_floor(refs)
    err = float((got - ref1).abs().max() / ref1.abs().max())
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"px={px}: elastic DD vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [2, 3])
def test_elastic_dd_gradients_match_fused(px):
    """vp/vs/rho AND the wavelet gradient."""
    prop = _prop()

    def grads(runner):
        models, wav = _models(grad=True), _wav(grad=True)
        src, rec = _geometry(prop, models)
        out = runner(prop, models, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return [m.grad.clone() for m in models] + [wav.grad.clone()]

    refs = [grads(_fused_reference) for _ in range(FLOOR_REPEATS)]
    ref = refs[0]
    got = grads(lambda p, m, w, s, r: _run_dd(p, m, w, s, r, px))
    # the reference must actually CARRY signal: with too few steps the source
    # has not fired, every gradient is zero and any comparison passes
    # vacuously (that false green hid a real save-offset bug once)
    assert float(ref[0].abs().max()) > 0.0, "reference gradient is all zeros"
    for i, (nm, a, c) in enumerate(zip(("vp", "vs", "rho", "wavelet"),
                                       ref, got)):
        floor = _pooled_floor([r[i] for r in refs])
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"px={px}/{nm}: DD grad vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("case", ["rec_on_cut", "src_on_cut"], ids=str)
def test_elastic_dd_cut_geometry(case):
    """Sources / receivers EXACTLY on a cut column -- the geometry that
    caught two real bugs on the acoustic path (missing adjoint injection on
    the non-owning strip; a doubled cut-column source)."""
    prop = _prop()
    px = 2
    cuts, nxg = _cut_cols(prop, _models(), px)

    def geometry():
        src = 3 * NO * nxg + (cuts[0] if case == "src_on_cut" else 5 * NO + 1)
        cols = [cuts[0], 8, 28] if case == "rec_on_cut" else [4, 12, 28, 40]
        return src, (torch.tensor(cols) + 2 * nxg).cuda()

    def grads(runner):
        models, wav = _models(grad=True), _wav(grad=True)
        src, rec = geometry()
        out = runner(prop, models, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return [out.detach()] + [m.grad.clone() for m in models] + \
               [wav.grad.clone()]

    refs = [grads(_fused_reference) for _ in range(FLOOR_REPEATS)]
    ref = refs[0]
    got = grads(lambda p, m, w, s, r: _run_dd(p, m, w, s, r, px))
    for i, (nm, a, c) in enumerate(zip(("fwd", "vp", "vs", "rho", "wavelet"),
                                       ref, got)):
        floor = _pooled_floor([r[i] for r in refs])
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"{case}/{nm}: DD vs fused rel {err:.2e}, floor {floor:.2e}")


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(x @ y / (x.norm() * y.norm()))


@pytest.mark.parametrize("px", [1, 2])
def test_elastic_dd_boundary_saving(px):
    """Boundary saving composes with the elastic decomposition. Bars follow
    the single-domain BS convention (cos + relative L2): BS is a
    reconstruction, so a pointwise-max bar would flag the reconstruction's
    own approximation rather than anything DD did."""
    prop = _prop()
    src, rec = _geometry(prop, _models())

    def grads(prop):
        models, wav = _models(grad=True), _wav(grad=True)
        out = _run_dd(prop, models, wav, src, rec, px)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), [m.grad.clone() for m in models]

    s_a, g_a = grads(prop)
    s_b, g_b = grads(_prop(BoundarySaving()))
    assert float((s_a - s_b).abs().max() / s_a.abs().max()) <= ABS_CEILING
    for nm, a, b in zip(("vp", "vs", "rho"), g_a, g_b):
        assert _cos(a, b) > 0.999, f"px={px}/{nm}: cos {_cos(a, b)}"
        l2 = float((b - a).norm() / a.norm())
        assert l2 < 5e-2, f"px={px}/{nm}: DD-bs vs DD-tape L2 {l2:.2e}"


# A configuration where boundary saving itself is accurate AND the reverse
# march genuinely runs: a 32x16-element model inside its 5-element rim leaves
# most of the nodes outside the band, and nt=800 runs well past the Ricker
# peak so the gradient is real signal rather than pre-arrival noise. Both
# matter -- where the band covers a whole strip nothing is marched, and with
# too few steps the reconstruction's fixed absolute error swamps a gradient
# that has barely grown.
RING = dict(shape=(65, 129), nt=800, abcn=5, src_elem_z=8)


def _ring_prop(memory=None, bs_store_dtype_elem=None):
    return S.PropSEM(ElasticSEM2DDeformed, shape=RING["shape"], dh=DH, dt=7e-4,
                     nt=RING["nt"], n_order=NO, free_surface=False,
                     abcn=RING["abcn"] * NO, memory=memory,
                     bs_store_dtype_elem=bs_store_dtype_elem,
                     f_dominant=F0, dev="cuda", impl="c")


def _ring_setup():
    prop = _ring_prop()
    nz, nx = RING["shape"]
    z = torch.linspace(0, 1, nz)[:, None].expand(nz, nx)

    def models(grad=False):
        ms = [(2400.0 + 600.0 * z).contiguous().float().cuda(),
              (1300.0 + 350.0 * z).contiguous().float().cuda(),
              (2100.0 + 200.0 * z).contiguous().float().cuda()]
        return [t.requires_grad_(True) for t in ms] if grad else ms

    t = np.arange(RING["nt"]) * 7e-4
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e3,
                       device="cuda")
    eq, _ = prop._resolve_model_batch(models())
    nxg = eq.mesh.nex * NO + 1
    src = RING["src_elem_z"] * NO * nxg + (eq.mesh.nex // 2) * NO
    rec = (torch.arange(4, nxg - 4, 4) + 2 * nxg).cuda()
    return prop, models, wav, src, rec


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(x @ y / (x.norm() * y.norm()))


@pytest.mark.parametrize("store", ["bf16", "int8"])
def test_dd_elastic_low_precision_ring(store):
    """bf16 / int8 rings on the elastic DD path.

    The elastic ring is TWO rings: u on band nodes and six element-local
    history buffers on band ELEMENTS. int8 keeps one amax per (field, step)
    for the node rings and per (field, step, ELEMENT) for the element ones,
    because a single band amax is far too coarse there.

    The bars differ by dtype on purpose, and the difference is the finding
    (measured when written): bf16 lands at ~5e-4, int8 only at a few
    percent, so **bf16 is the
    low-precision setting to reach for on this path**. Bisecting the ring
    families puts the whole int8 cost in the two u NODE rings (2.5e-2) with
    all six ELEMENT rings together at 3.6e-4 -- and the element rings are
    82% of the ring memory, so a per-family dtype would be the fix if this
    ever needs to be cheap. See docs/user-guide/sem.md, "Ring precision".

    The tight bar is the DD-specific one: whatever the store costs, the
    decomposition must not add to it."""
    prop, models, wav, src, rec = _ring_setup()
    eq, _ = prop._resolve_model_batch(models())
    pats = []
    for j in range(2):
        v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                        device=eq.mesh.ibool.device)
        if j == 1:
            v[int(src)] = 1.0
        pats.append(v)
    parts, _m = SEMModelParallel(prop, StripTopology(2))._parts(
        eq, tuple(pats), rec)
    for pp in parts:      # the march must really run, else this tests nothing
        assert int(pp.c["band_node_idx"].numel()) < 0.95 * pp.c["n_global"]
    assert not any(int(src) in set(pp.gidx[pp.c["band_node_idx"]].tolist())
                   for pp in parts), "source inside the band: see test_sem_dd"

    seen = {}
    real = _SEMElasticCpmlDDC.forward

    def spy(ctx, wavelet, pp, exchange, bs, *mats):
        out = real(ctx, wavelet, pp, exchange, bs, *mats)
        seen["carry"] = ctx.carry
        return out

    def dd(storage_dtype, px=2):
        ms, w = models(grad=True), wav.clone().requires_grad_(True)
        out = SEMModelParallel(
            _ring_prop(BoundarySaving(storage_dtype=storage_dtype)),
            StripTopology(px))(w, src, rec, ms)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return out.detach(), [t.grad.clone() for t in ms]

    s32, g32 = dd("fp32")
    _SEMElasticCpmlDDC.forward = staticmethod(spy)
    try:
        sq, gq = dd(store)
    finally:
        _SEMElasticCpmlDDC.forward = staticmethod(real)

    want = torch.bfloat16 if store == "bf16" else torch.int8
    for cr, pp in zip(seen["carry"], parts):
        nbe = int(pp.c["band_elem_idx"].numel())   # differs per strip
        for k in _EL2D_RINGS:
            assert cr[k].dtype == want, f"ring is {cr[k].dtype}, wanted {want}"
        scale, scale_e = cr["band_scale"], cr["band_scale_e"]
        if store == "int8":
            assert tuple(scale.shape) == (2, RING["nt"])
            assert tuple(scale_e.shape) == (6, RING["nt"], nbe)
            assert float(scale.max()) > 0 and float(scale_e.max()) > 0
        else:
            assert scale.numel() == 0 and scale_e.numel() == 0
    assert all(torch.isfinite(g).all() for g in gq)
    # the ring is read only in the backward: the record must not move
    assert float((sq - s32).abs().max() / s32.abs().max()) < 1e-3
    bar = 5e-3 if store == "bf16" else 8e-2
    for nm, a, b in zip(("vp", "vs", "rho"), g32, gq):
        assert _cos(a, b) > 0.998, f"{store}/{nm}: cos {_cos(a, b)}"
        l2 = float((b - a).norm() / a.norm())
        assert l2 < bar, f"{store}/{nm}: ring vs fp32 ring L2 {l2:.2e}"


def test_dd_elastic_per_family_store_dtype():
    """``bs_store_dtype_elem``: store the ELEMENT rings coarser than the node
    rings.

    This exists because of where the int8 cost actually sits. Bisecting the
    ring families put the whole of it in the two u NODE rings, while the six
    ELEMENT rings -- 82% of the ring memory -- quantise ~70x better. Splitting
    the dtype therefore buys most of the memory at the cheap error, and this
    test pins that ordering rather than absolute numbers, so it keeps meaning
    if the reconstruction itself changes:

        all-int8   is much WORSE  than elems-int8-only
        elems-int8 is no worse    than all-bf16
        peak(nodes bf16 / elems int8) < peak(all bf16)

    Measured when written (peak MB / vp L2): fp32 266/-, all-bf16 140/4.4e-4,
    all-int8 89/2.5e-2, nodes-fp32+elems-int8 123/3.6e-4, and the setting to
    reach for, nodes-bf16+elems-int8, 100/5.6e-4."""
    prop, models, wav, src, rec = _ring_setup()

    def run(store, store_e=None, want_peak=False):
        # both stores are constructor options now: one propagator per run
        p = _ring_prop(BoundarySaving(storage_dtype=store or "fp32"),
                       bs_store_dtype_elem=store_e)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ms, w = models(grad=True), wav.clone().requires_grad_(True)
        out = SEMModelParallel(p, StripTopology(2))(w, src, rec, ms)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        g = [t.grad.clone() for t in ms]
        return (g, torch.cuda.max_memory_allocated()) if want_peak else g

    def l2(a, b):
        return max(float((x - y).norm() / y.norm()) for x, y in zip(a, b))

    ref = run(None)
    g_i8 = run("int8")
    g_bf16, p_bf16 = run("bf16", want_peak=True)
    g_elem = run(None, "int8")
    g_mix, p_mix = run("bf16", "int8", want_peak=True)

    e_i8, e_bf16 = l2(g_i8, ref), l2(g_bf16, ref)
    e_elem, e_mix = l2(g_elem, ref), l2(g_mix, ref)
    assert e_i8 > 20 * e_elem, (
        f"element-only int8 ({e_elem:.2e}) should be far better than "
        f"all-int8 ({e_i8:.2e}) -- that is the whole point of the split")
    assert e_elem <= 2 * e_bf16, (
        f"element-only int8 ({e_elem:.2e}) should be bf16-class "
        f"({e_bf16:.2e})")
    assert e_mix <= 3 * e_bf16, f"mixed store {e_mix:.2e} vs bf16 {e_bf16:.2e}"
    assert p_mix < p_bf16, (
        f"mixed store peaked at {p_mix/1e6:.0f} MB, no better than all-bf16 "
        f"({p_bf16/1e6:.0f} MB) -- the element rings are the bulk, so it must")


def test_dd_store_dtype_elem_refused_where_there_are_no_element_rings():
    """A store dtype for rings a path does not have would be a silent no-op --
    the exact failure mode the store-dtype tests exist to prevent."""
    prop = _prop()
    models_ = _models(grad=True)
    src, rec = _geometry(prop, models_)
    from sweep.sem.equations import AcousticPotentialSEM2D
    # the propagator takes the element store (it cannot know which path will
    # run it); the strip driver of a node-ring-only family must refuse it
    ap = S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                   nt=NT, n_order=NO, free_surface=("top",),
                   abcn=3 * NO, memory=BoundarySaving(),
                   bs_store_dtype_elem="int8", f_dominant=F0,
                   dev="cuda", impl="c")
    vp = (2000.0 + 300.0 * torch.rand(NZ, NX)).cuda().requires_grad_(True)
    with pytest.raises(ValueError, match="ELEMENT"):
        SEMModelParallel(ap, StripTopology(2))(_wav(), src, rec, [vp])

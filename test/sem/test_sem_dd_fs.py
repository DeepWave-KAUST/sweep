"""Coupled fluid-solid domain decomposition: DD stepped loop vs the fused driver.

Same discipline as the acoustic/elastic twins: the reference is the SAME
per-strip consts through the FUSED single-domain C driver, px=1 isolates the
stepped split, px>1 adds the decomposition, thresholds anchor to the fused
driver's own atomics floor.

What is NEW here: the horizontal seabed interface necessarily CROSSES every
vertical cut. That needs no dedicated machinery -- the exchange sums the
whole cut COLUMN and the interface node on it is one of those nodes, so
faces on either side contribute exactly as elements do -- but it is the
property most likely to break silently, so ``test_fs_dd_cut_crosses_seabed``
puts a cut right where the interface runs and compares the whole record.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import (SEMModelParallel, StripTopology,
                          _SEMFluidSolidDDC)
from sweep.sem.equations import AcousticElasticSEM2D
from sweep.sem.prop_c import _SEMFluidSolidC, sem_c_available
from sweep.sem.drivers.fluid_solid import (_FS_TAPES, _FS_RINGS_AC, _FS_RINGS_EL,
                                           _FS_FIN, _FS_CK)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 25.0, 6.0
NZ, NX, NW = 25, 49, 8            # 6 x 12 elements, water = 8 rows
NT, DT = 200, 1.2e-3
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4


def _prop(memory=None):
    return S.PropSEM(AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=3 * NO, memory=memory, f_dominant=F0,
                     dev="cuda", impl="c", receiver_type=["p", "vx", "vz"])


def _models(prop, grad=False):
    m = prop.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    ms = [torch.where(fl, 1500.0, 2300.0).float().cuda(),
          torch.where(fl, 0.0, 1250.0).float().cuda(),
          torch.where(fl, 1000.0, 2100.0).float().cuda()]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def _wav(grad=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.3 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    return w.requires_grad_(True) if grad else w


def _nxg(prop, models):
    eq, _ = prop._resolve_model_batch([m.detach() for m in models])
    return eq.mesh.nex * NO + 1, eq.mesh.nex


def _geometry(prop, models, rec_cols=None, src_col=None):
    nxg, _ = _nxg(prop, models)
    src = 2 * NO * nxg + (src_col if src_col is not None else 5 * NO + 1)
    cols = rec_cols if rec_cols is not None else list(range(4, nxg - 4, 4))
    rec = (torch.tensor(cols) + (NW + 1) * nxg).cuda()   # OBS on the seabed
    return src, rec


def _fused_reference(prop, models, wav, src, rec):
    eq, _ = prop._resolve_model_batch(models)
    d1 = SEMModelParallel(prop, StripTopology(1))
    pats = []
    for j in range(3):
        v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                        device=eq.mesh.ibool.device)
        if j == 0:
            v[int(src)] = 1.0
        pats.append(v)
    parts, mats = d1._parts(eq, tuple(pats), rec)
    c = dict(parts[0].c)
    c["bs"] = False
    sc, sx, sz = _SEMFluidSolidC.apply(*mats, wav, c)
    return torch.stack([sc, sx, sz], -1).transpose(0, 1)   # (nt, nrec, 3)


def _run_dd(prop, models, wav, src, rec, px):
    return SEMModelParallel(prop, StripTopology(px))(wav, src, rec, models)


@pytest.mark.parametrize("px", [1, 2, 3])
def test_fs_dd_forward_matches_fused(px):
    prop = _prop()
    models, wav = _models(prop), _wav()
    src, rec = _geometry(prop, models)
    with torch.no_grad():
        ref1 = _fused_reference(prop, models, wav, src, rec)
        ref2 = _fused_reference(prop, models, wav, src, rec)
        got = _run_dd(prop, models, wav, src, rec, px)
    assert got.shape == (NT, rec.numel(), 3)
    floor = float((ref1 - ref2).abs().max() / ref1.abs().max())
    err = float((got - ref1).abs().max() / ref1.abs().max())
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"px={px}: fs DD vs fused rel {err:.2e}, floor {floor:.2e}")


def test_fs_dd_cut_crosses_seabed():
    """The cut column runs THROUGH the fluid-solid interface. Interface faces
    on either side of it must each contribute their partial sum, with the
    column exchange completing the shared node -- if the interface points
    were sliced by node instead of by face, the shared node would be
    double-counted and this comparison would blow up."""
    prop = _prop()
    models, wav = _models(prop), _wav()
    nxg, nex = _nxg(prop, models)
    cut = (nex // 2) * NO
    # receivers straddling the crossing point, plus one exactly on it
    src, rec = _geometry(prop, models,
                         rec_cols=[cut - 4, cut, cut + 4, 8, 40])
    with torch.no_grad():
        ref1 = _fused_reference(prop, models, wav, src, rec)
        ref2 = _fused_reference(prop, models, wav, src, rec)
        got = _run_dd(prop, models, wav, src, rec, 2)
    floor = float((ref1 - ref2).abs().max() / ref1.abs().max())
    err = float((got - ref1).abs().max() / ref1.abs().max())
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"seabed-crossing cut: fs DD vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [2, 3])
def test_fs_dd_gradients_match_fused(px):
    prop = _prop()

    def grads(runner):
        models, wav = _models(prop, grad=True), _wav(grad=True)
        src, rec = _geometry(prop, models)
        out = runner(prop, models, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return [m.grad.clone() for m in models] + [wav.grad.clone()]

    ref = grads(_fused_reference)
    ref2 = grads(_fused_reference)
    got = grads(lambda p, m, w, s, r: _run_dd(p, m, w, s, r, px))
    assert float(ref[0].abs().max()) > 0.0, "reference gradient is all zeros"
    for nm, a, b, c in zip(("vp", "vs", "rho", "wavelet"), ref, ref2, got):
        floor = float((a - b).abs().max() / a.abs().max())
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"px={px}/{nm}: fs DD grad vs fused rel {err:.2e}, "
            f"floor {floor:.2e}")


def test_fs_dd_gradients_with_seabed_crossing_cut():
    """Gradients with the cut through the interface AND a receiver on it."""
    prop = _prop()
    nxg, nex = _nxg(prop, _models(prop))
    cut = (nex // 2) * NO

    def grads(runner):
        models, wav = _models(prop, grad=True), _wav(grad=True)
        src, rec = _geometry(prop, models, rec_cols=[cut, cut - 4, 40])
        out = runner(prop, models, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return [m.grad.clone() for m in models] + [wav.grad.clone()]

    ref = grads(_fused_reference)
    ref2 = grads(_fused_reference)
    got = grads(lambda p, m, w, s, r: _run_dd(p, m, w, s, r, 2))
    assert float(ref[0].abs().max()) > 0.0
    for nm, a, b, c in zip(("vp", "vs", "rho", "wavelet"), ref, ref2, got):
        floor = float((a - b).abs().max() / a.abs().max())
        err = float((c - a).abs().max() / a.abs().max())
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"seabed-cut/{nm}: rel {err:.2e}, floor {floor:.2e}")


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(x @ y / (x.norm() * y.norm()))


def _fused_bs(prop, models, wav, src, rec, memory):
    eq, _ = prop._resolve_model_batch(models)
    pats = []
    for j in range(3):
        v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                        device=eq.mesh.ibool.device)
        if j == 0:
            v[int(src)] = 1.0
        pats.append(v)
    parts, mats = SEMModelParallel(prop, StripTopology(1))._parts(
        eq, tuple(pats), rec)
    c = dict(parts[0].c)
    c["bs"] = memory == "boundary"
    sc, sx, sz = _SEMFluidSolidC.apply(*mats, wav, c)
    return torch.stack([sc, sx, sz], -1).transpose(0, 1)


def _bands(prop, models, src, rec, px=2):
    eq, _ = prop._resolve_model_batch(models)
    pats = []
    for j in range(3):
        v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                        device=eq.mesh.ibool.device)
        if j == 0:
            v[int(src)] = 1.0
        pats.append(v)
    parts, _m = SEMModelParallel(prop, StripTopology(px))._parts(
        eq, tuple(pats), rec)
    return parts


@pytest.mark.parametrize("px", [1, 2, 3])
def test_fs_dd_boundary_saving_matches_save_all(px):
    """Boundary saving on the coupled DD path. Each strip rings BOTH domains
    -- the fluid band on nodes, the solid band on nodes AND elements -- and
    re-anchors on full-state checkpoints, because the coupled reverse map
    amplifies roundoff and the march is only trustworthy over short
    segments.

    Bars follow the single-domain BS convention (cos + relative L2); whether
    the DECOMPOSITION adds anything is isolated by the next test. The band
    guard matters here: on a grid whose rim covers everything, every node
    would be ring-restored and the reverse march -- the part most likely to
    break -- would never run."""
    prop = _prop()
    src, rec = _geometry(prop, _models(prop))
    parts = _bands(prop, _models(prop), src, rec, max(px, 2))
    # the march must run SOMEWHERE: a narrow outer strip can be all band,
    # but the middle strip keeps a genuine interior. The wide-grid test
    # below is the one where every strip has a real rim.
    assert any(int(pp.c["band_idx"].numel()) < pp.c["n_global"]
               for pp in parts), "the band covers every strip: no march runs"

    def grads(p):
        models, wav = _models(prop, grad=True), _wav(grad=True)
        out = SEMModelParallel(p, StripTopology(px))(wav, src, rec, models)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return (out.detach(), *[m.grad.clone() for m in models],
                wav.grad.clone())

    ref = grads(prop)
    got = grads(_prop(BoundarySaving()))
    assert float(ref[0].abs().max()) > 0.0, "reference record is all zeros"
    # the forward is NOT a reconstruction: it must match at the floor
    assert float((ref[0] - got[0]).abs().max()
                 / ref[0].abs().max()) <= ABS_CEILING
    for nm, a, b in zip(("vp", "vs", "rho", "wavelet"), ref[1:], got[1:]):
        assert _cos(a, b) > 0.9999, f"px={px}/{nm}: cos {_cos(a, b)}"
        l2 = float((b - a).norm() / a.norm())
        assert l2 < 1e-3, f"px={px}/{nm}: DD-bs vs DD-save-all L2 {l2:.2e}"


@pytest.mark.parametrize("px", [2, 3])
def test_fs_dd_boundary_saving_adds_no_error(px):
    """The DD-specific bar: DD-bs must sit as close to FUSED-bs as DD-tape
    sits to fused-tape. This is the test that would catch the coupled
    march's one DD hazard -- it recomputes the solid Acc, an ASSEMBLY, so it
    needs its own interface exchange; drop that exchange and the strips
    march apart."""
    prop = _prop()
    src, rec = _geometry(prop, _models(prop))
    # the DD memory is a constructor option: one propagator per mode (the
    # fused reference takes the mode itself, on the consts)
    props = {None: prop, "boundary": _prop(BoundarySaving())}

    def grads(runner, memory):
        models, wav = _models(prop, grad=True), _wav(grad=True)
        out = runner(models, wav, memory)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return [m.grad.clone() for m in models] + [wav.grad.clone()]

    def dd(models, wav, memory):
        return SEMModelParallel(props[memory], StripTopology(px))(
            wav, src, rec, models)

    def fused(models, wav, memory):
        return _fused_bs(prop, models, wav, src, rec, memory)

    g_dd_t, g_f_t = grads(dd, None), grads(fused, None)
    g_dd_b, g_f_b = grads(dd, "boundary"), grads(fused, "boundary")
    for nm, dt_, ft_, db, fb in zip(("vp", "vs", "rho", "wavelet"),
                                    g_dd_t, g_f_t, g_dd_b, g_f_b):
        base = float((dt_ - ft_).norm() / ft_.norm())
        with_bs = float((db - fb).norm() / fb.norm())
        assert with_bs <= max(10 * base, 1e-3), (
            f"px={px}/{nm}: DD adds error under bs (L2 {with_bs:.2e}) beyond "
            f"its save-all baseline ({base:.2e})")


def test_fs_dd_boundary_saving_survives_a_cut_on_the_seabed():
    """The path's signature hazard, now under boundary saving: a cut placed
    exactly where the fluid-solid interface runs. The march recomputes the
    coupling (chi_dd feeds the solid force, u feeds s_ac) on BOTH strips, so
    the interface has to reproduce identically after the exchange."""
    prop = _prop()
    nxg, nex = _nxg(prop, _models(prop))
    cut = (nex // 2) * NO
    src, rec = _geometry(prop, _models(prop),
                         rec_cols=[cut, cut - 4, 4, nxg - 5], src_col=cut)

    def grads(p):
        models, wav = _models(prop, grad=True), _wav(grad=True)
        out = SEMModelParallel(p, StripTopology(2))(wav, src, rec, models)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return (out.detach(), *[m.grad.clone() for m in models],
                wav.grad.clone())

    ref, got = grads(prop), grads(_prop(BoundarySaving()))
    assert float(ref[0].abs().max()) > 0.0
    assert float((ref[0] - got[0]).abs().max()
                 / ref[0].abs().max()) <= ABS_CEILING
    for nm, a, b in zip(("vp", "vs", "rho", "wavelet"), ref[1:], got[1:]):
        assert _cos(a, b) > 0.9999, f"seabed-cut/{nm}: cos {_cos(a, b)}"


def test_fs_dd_boundary_saving_actually_rings():
    """Mechanism check: under BoundarySaving() the save-all tapes must be
    EMPTY, both domains' rings must be allocated at their band widths, and
    the re-anchor checkpoints must hold real state. Without this a silently
    ignored flag would still pass every equality test above."""
    from sweep.sem.dd import _SEMFluidSolidDDC
    prop = _prop(BoundarySaving())
    models, wav = _models(prop, grad=True), _wav()
    src, rec = _geometry(prop, models)
    seen = {}
    real = _SEMFluidSolidDDC.forward

    def spy(ctx, wavelet, parts, exchange, bs, *mats):
        out = real(ctx, wavelet, parts, exchange, bs, *mats)
        seen["carry"], seen["parts"] = ctx.carry, parts
        return out

    _SEMFluidSolidDDC.forward = staticmethod(spy)
    try:
        SEMModelParallel(prop, StripTopology(2))(wav, src, rec, models)
    finally:
        _SEMFluidSolidDDC.forward = staticmethod(real)
    assert seen, "the DD Function never ran"
    m2 = (NO + 1) ** 2
    for cr, pp in zip(seen["carry"], seen["parts"]):
        assert all(cr[k].numel() == 0 for k in _FS_TAPES), "save-all tape allocated"
        assert cr["band_kind"] == 0 and cr["band_kind_e"] == 0    # fp32 by default
        assert cr["band_scale"].numel() == 0 and cr["band_scale_e"].numel() == 0
        nband = int(pp.c["band_idx"].numel())
        nbn = int(pp.c["band_node_idx"].numel())
        nbe = int(pp.c["band_elem_idx"].numel())
        assert nband > 0 and nbn > 0 and nbe > 0
        for k in _FS_RINGS_AC:
            assert tuple(cr[k].shape) == (NT, nband)
        for k in _FS_RINGS_EL[:2]:
            assert tuple(cr[k].shape) == (NT, nbn)
        for k in _FS_RINGS_EL[2:]:
            assert tuple(cr[k].shape) == (NT, nbe * m2)
        every = cr["ckpt_every"]
        assert every > 0 and all(cr[k].shape[0] == NT // every for k in _FS_CK)
        assert any(float(cr[k].abs().max()) > 0.0 for k in _FS_CK), "anchors unwritten"
        assert all(float(cr[k].abs().max()) > 0.0 for k in _FS_FIN), "no march seed"


# A grid wide enough that EVERY strip keeps a real interior (measured when
# written, px=2: fluid band 57%, solid nodes 46%, solid elements 44%), so the coupled
# reverse march -- the part of boundary saving most likely to break -- is
# genuinely exercised on both sides of the cut.
WIDE = dict(shape=(49, 97), nt=300)


def _wide_prop(memory=None):
    return S.PropSEM(AcousticElasticSEM2D, shape=WIDE["shape"], dh=DH, dt=DT,
                     nt=WIDE["nt"], n_order=NO, free_surface=("top",),
                     abcn=3 * NO, memory=memory, f_dominant=F0,
                     dev="cuda", impl="c", receiver_type=["p", "vx", "vz"])


def _wide():
    prop = _wide_prop()
    m = prop.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)

    def models(grad=False):
        ms = [torch.where(fl, 1500.0, 2300.0).float().cuda(),
              torch.where(fl, 0.0, 1250.0).float().cuda(),
              torch.where(fl, 1000.0, 2100.0).float().cuda()]
        return [t.requires_grad_(True) for t in ms] if grad else ms

    t = np.arange(WIDE["nt"]) * DT
    a = (np.pi * F0 * (t - 1.3 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                       device="cuda")
    eq, _ = prop._resolve_model_batch(models())
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + nxg // 2
    rec = (torch.arange(4, nxg - 4, 6) + (NW + 1) * nxg).cuda()
    return prop, models, wav, src, rec


def test_fs_dd_boundary_saving_on_a_real_rim():
    """BS x DD where every strip has an interior the march has to rebuild."""
    prop, models, wav, src, rec = _wide()
    parts = _bands(prop, models(), src, rec, 2)
    for pp in parts:
        assert int(pp.c["band_idx"].numel()) < 0.9 * pp.c["n_global"], (
            "the band is nearly the whole strip: the march is masked and "
            "this test would not be checking it")

    def grads(p):
        ms, w = models(grad=True), wav.clone().requires_grad_(True)
        out = SEMModelParallel(p, StripTopology(2))(w, src, rec, ms)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return (out.detach(), *[t.grad.clone() for t in ms], w.grad.clone())

    ref, got = grads(prop), grads(_wide_prop(BoundarySaving()))
    assert float(ref[0].abs().max()) > 0.0
    assert float((ref[0] - got[0]).abs().max()
                 / ref[0].abs().max()) <= ABS_CEILING
    for nm, a, b in zip(("vp", "vs", "rho", "wavelet"), ref[1:], got[1:]):
        assert _cos(a, b) > 0.9999, f"rim/{nm}: cos {_cos(a, b)}"
        l2 = float((b - a).norm() / a.norm())
        assert l2 < 1e-3, f"rim/{nm}: DD-bs vs DD-save-all L2 {l2:.2e}"


def test_fs_dd_boundary_saving_saves_memory():
    """The rings must actually buy memory: the coupled save-all tape carries
    seven node histories AND six element histories, which is what makes this
    the most expensive path to tape."""
    nz, nx, nt = 49, 97, 300

    def build(memory=None):
        return S.PropSEM(AcousticElasticSEM2D, shape=(nz, nx), dh=DH, dt=DT,
                         nt=nt, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, memory=memory, f_dominant=F0,
                         dev="cuda", impl="c",
                         receiver_type=["p", "vx", "vz"])

    prop = build()
    m = prop.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    models = [torch.where(fl, 1500.0, 2300.0).float().cuda(),
              torch.where(fl, 0.0, 1250.0).float().cuda(),
              torch.where(fl, 1000.0, 2100.0).float().cuda()]
    models[0].requires_grad_(True)
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 1.3 / F0)) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                       device="cuda")
    eq, _ = prop._resolve_model_batch([mm.detach() for mm in models])
    nxg = eq.mesh.nex * NO + 1
    src = 2 * NO * nxg + nxg // 2
    rec = (torch.arange(4, nxg - 4, 6) + (NW + 1) * nxg).cuda()

    # one propagator per memory mode (a constructor option), each built for
    # its own window so the two peaks carry the same set-up allocations
    def peak(memory=None):
        ddp = SEMModelParallel(build(memory), StripTopology(2))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = ddp(wav, src, rec, models)
        (out ** 2).sum().backward()
        models[0].grad = None
        return torch.cuda.max_memory_allocated()

    mem_tape, mem_ring = peak(), peak(BoundarySaving())
    assert mem_ring < 0.75 * mem_tape, (
        f"ring {mem_ring/1e6:.0f} MB vs tape {mem_tape/1e6:.0f} MB")


@pytest.mark.parametrize("store", ["bf16", "int8"])
def test_fs_dd_low_precision_ring(store):
    """bf16 / int8 rings on the coupled DD path.

    Three ring families share the machinery: five fluid node rings and two
    solid node rings on one amax table (slots 0..6), and six solid ELEMENT
    rings on their own, one amax per band element. The full-state march
    seeds and re-anchor checkpoints deliberately stay fp32 -- they are what
    the march restarts from, not ring rows.

    Measured on the wide grid when written: BOTH stores land at the fp32 ring's own
    reconstruction error (fp32-vs-tape 1.4e-06..8.3e-06; bf16 1.2e-06..
    8.3e-06; int8 1.5e-06..9.6e-06). That is a sharp contrast with the
    elastic path, where int8 costs ~2000x its fp32 reconstruction error --
    and it is NOT explained by the checkpoints (disabling them changes
    nothing) nor by march length (the ratio is stable in nt). Left as a
    measured difference rather than a story."""
    prop, models, wav, src, rec = _wide()

    def dd(storage_dtype):
        ms, w = models(grad=True), wav.clone().requires_grad_(True)
        out = SEMModelParallel(
            _wide_prop(BoundarySaving(storage_dtype=storage_dtype)),
            StripTopology(2))(w, src, rec, ms)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return (out.detach(), *[t.grad.clone() for t in ms], w.grad.clone())

    seen = {}
    real = _SEMFluidSolidDDC.forward

    def spy(ctx, wavelet, parts, exchange, bs, *mats):
        out = real(ctx, wavelet, parts, exchange, bs, *mats)
        seen["carry"], seen["parts"] = ctx.carry, parts
        return out

    ref = dd("fp32")
    _SEMFluidSolidDDC.forward = staticmethod(spy)
    try:
        got = dd(store)
    finally:
        _SEMFluidSolidDDC.forward = staticmethod(real)

    want = torch.bfloat16 if store == "bf16" else torch.int8
    for cr, pp in zip(seen["carry"], seen["parts"]):
        nbe = int(pp.c["band_elem_idx"].numel())
        for k in _FS_RINGS_AC + _FS_RINGS_EL:
            assert cr[k].dtype == want, f"ring is {cr[k].dtype}, wanted {want}"
        for k in _FS_FIN + _FS_CK:      # seeds / anchors must stay exact
            assert cr[k].dtype == torch.float32
        kind, scale, scale_e, kind_e = (cr["band_kind"], cr["band_scale"],
                                        cr["band_scale_e"], cr["band_kind_e"])
        want_kind = 1 if store == "bf16" else 2
        assert kind == want_kind and kind_e == want_kind
        if store == "int8":
            assert tuple(scale.shape) == (7, WIDE["nt"])
            assert tuple(scale_e.shape) == (6, WIDE["nt"], nbe)
            assert float(scale.max()) > 0 and float(scale_e.max()) > 0
        else:
            assert scale.numel() == 0 and scale_e.numel() == 0
    # the ring is read only in the backward: the record must not move
    assert float((got[0] - ref[0]).abs().max() / ref[0].abs().max()) < 1e-3
    for nm, a, b in zip(("vp", "vs", "rho", "wavelet"), ref[1:], got[1:]):
        assert _cos(a, b) > 0.9999, f"{store}/{nm}: cos {_cos(a, b)}"
        l2 = float((b - a).norm() / a.norm())
        assert l2 < 5e-3, f"{store}/{nm}: ring vs fp32 ring L2 {l2:.2e}"

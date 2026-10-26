"""Boundary saving for the coupled fluid-solid path on impl='c'.

Design under test (rings + anchors): the acoustic (chi, psi, zeta) ring on the
fluid C-PML band nodes, the elastic u ring + six element-local history rings
on the solid band, AND a full-state re-anchor checkpoint every
``bs_frame_checkpoint_every`` steps -- the coupled reverse map amplifies
roundoff ~x1.15/step (measured on the eager side), so the leapfrog march alone
is only trustworthy over ~100-step segments.

Why the eager coupled BS is NOT the reference here: it runs ring-less
(zero-memory reverse + anchors) and its exactness test builds a BARE box -- the
C-PML configuration was never validated there. The references used instead:
the save-all impl='c' gradient (same kernels, full histories) at the measured
run-to-run floor, and the eager FULL-TAPE gradient.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving, Full
from sweep.sem.equations import AcousticElasticSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, DT = 4, 25.0, 6.0, 260, 0.0016
NZ, NX, NW = 49, 81, 8
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4

SRC = np.array([[NX // 2, 3]], dtype=np.int64)
RIX = np.arange(6, NX - 6, 3)
REC = np.stack([RIX, np.full_like(RIX, NW + 2)], -1)[None]     # OBS, solid


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                        device="cuda")


def _prop(impl="c", abcn=4 * NO, memory=None):
    return S.PropSEM(AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=abcn, f_dominant=F0, dev="cuda", impl=impl,
                     memory=memory, receiver_type=["vx", "vz"])


def _models(p, grad=False):
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    vp = torch.where(fl, 1500.0, 2200.0).float().cuda()
    vs = torch.where(fl, 0.0, 1200.0).float().cuda()
    rho = torch.where(fl, 1000.0, 2100.0).float().cuda()
    ms = [vp, vs, rho]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def _run(memory, impl="c"):
    p = _prop(impl, memory=memory)
    ms = _models(p, grad=True)
    s = p(_wav(), SRC, REC, models=ms)
    (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
    return s.detach(), [t.grad.clone() for t in ms]


def _rel(a, b, scale=None):
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


def test_bs_forward_records_are_unchanged():
    """bs only changes WHAT IS STORED; the forward physics is identical."""
    s_bs, _ = _run(BoundarySaving())
    s_bs2, _ = _run(BoundarySaving())      # run-to-run floor
    s_full, _ = _run(Full())
    floor, got = _rel(s_bs, s_bs2), _rel(s_bs, s_full)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"bs seis vs save-all {got:.2e}, floor {floor:.2e}")


def test_bs_gradients_match_save_all():
    _, g_bs = _run(BoundarySaving())
    _, g_bs2 = _run(BoundarySaving())      # run-to-run floor
    _, g_full = _run(Full())
    for nm, a, b, r in zip(("vp", "vs", "rho"), g_bs, g_bs2, g_full):
        floor, got = _rel(a, b), _rel(a, r)
        assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"d/d{nm}: bs vs save-all {got:.2e}, floor {floor:.2e}")


def test_bs_gradients_match_the_eager_full_tape():
    """External reference: eager keeps the whole autograd tape -- no
    reconstruction anywhere -- so agreement here rules out a bs-vs-save-all
    comparison that is 'consistent but jointly wrong'."""
    _, g_bs = _run(BoundarySaving())
    _, g_bs2 = _run(BoundarySaving())
    _, g_e = _run(Full(), impl="eager")
    for nm, a, b, r in zip(("vp", "vs", "rho"), g_bs, g_bs2, g_e):
        floor, got = _rel(a, b), _rel(a, r)
        assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"d/d{nm}: bs vs eager full tape {got:.2e}, floor {floor:.2e}")


def test_ring_restored_march_is_stable_without_anchors(monkeypatch):
    """The EAGER coupled BS needs its re-anchor checkpoints because it
    reverses with ZEROED memories (ring-less), and that in-band inconsistency
    feeds back through the coupling at ~x1.15/step. The C rings clamp the
    band to exact values every step, which SUPPRESSES that mode: measured
    drift with anchors disabled is flat in nt (probed at 260/600/1200 steps:
    ~1e-5, the atomics-floor scale, no growth). Pin that stability -- if this
    ever goes red, the march has become anchor-dependent in some new
    configuration and ckpt_every needs attention (the anchors themselves stay
    ON by default as insurance; they also measurably tighten the error)."""
    monkeypatch.setattr(AcousticElasticSEM2D, "bs_frame_checkpoint_every",
                        10 ** 9)
    _, g_no = _run(BoundarySaving())
    _, g_full = _run(Full())
    err = max(_rel(a, r) for a, r in zip(g_no, g_full))
    assert err <= ABS_CEILING, (
        f"anchor-free march drifted to {err:.2e} -- the ring restore no "
        f"longer stabilises the coupled reverse iteration")


def test_bs_peak_memory_is_smaller():
    # What a call allocates, not what earlier calls left behind: an absolute
    # peak also carries the allocations that stay (the 32 MiB cuBLAS workspace
    # on Hopper, whatever earlier tests hold), which sank this ratio on an H200
    # while it passed on Ada.  One warm-up call allocates those first.
    def peak(memory):
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _run(memory)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    peak(Full())
    mem_full, mem_bs = peak(Full()), peak(BoundarySaving())
    assert mem_bs < 0.6 * mem_full, (
        f"bs {mem_bs/1e6:.0f} MB vs save-all {mem_full/1e6:.0f} MB")


def test_bare_box_still_refuses_boundary_saving():
    """The bs path needs the coupled C-PML (its rings ARE the C-PML bands);
    the bare box must refuse loudly, not silently run the full tape."""
    p = _prop(abcn=0, memory=BoundarySaving())
    ms = _models(p, grad=True)
    with pytest.raises(NotImplementedError, match="boundary"):
        p(_wav(), SRC, REC, models=ms)


def test_low_precision_ring_matches_the_fp32_ring():
    """This used to assert the coupled ring REFUSED low precision. It takes
    it now, and on this path it is close to free: int8 lands on the fp32
    ring's own reconstruction error rather than adding to it (unlike the
    elastic path, where int8 costs ~2000x that -- see
    test_sem_dd_elastic.py::test_dd_elastic_per_family_store_dtype).

    The bar is relative to what the fp32 ring already costs against the full
    tape, so it stays meaningful if the reconstruction changes."""

    def grads(store):
        p = _prop(memory=BoundarySaving(storage_dtype=store))
        ms = _models(p, grad=True)
        s = p(_wav(), SRC, REC, models=ms)
        (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
        return [t.grad.clone() for t in ms]

    def tape():
        p = _prop()
        ms = _models(p, grad=True)
        s = p(_wav(), SRC, REC, models=ms)
        (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
        return [t.grad.clone() for t in ms]

    l2 = lambda a, b: max(float((x - y).norm() / y.norm().clamp_min(1e-30))
                          for x, y in zip(a, b))
    g_tape, g_f32 = tape(), grads("fp32")
    base = l2(g_f32, g_tape)          # what boundary saving itself costs
    assert base > 0, "the fp32 ring is bit-exact here; pick a harder config"
    for store in ("bf16", "int8"):
        cost = l2(grads(store), g_f32)
        assert cost <= max(5 * base, 1e-4), (
            f"{store} ring costs {cost:.2e}, well beyond the fp32 ring's own "
            f"reconstruction error ({base:.2e})")


def test_the_band_indices_are_built_in_both_memory_modes():
    """The band index sets are GEOMETRY, not a boundary-saving artefact.

    The elastic path has always built ``band_node_idx`` / ``band_elem_idx``
    inside its consts (``prop_c._elastic_cpml_consts``); the coupled path used
    to build them only under ``memory='boundary'`` and hand the C entry a
    zero-length tensor otherwise. Nothing read them in tape mode, so nothing
    broke -- but an element flag or a band-only history added later WOULD read
    them there, and an empty index set means "no band", which turns the
    absorbing layer into a plain stretch: no error, no NaN, just reflections
    off the edge of the model.

    Both halves of the assertion matter. Equality alone would be satisfied by
    two empty sets, which is exactly the bug; non-emptiness alone would not
    catch a mode that builds a DIFFERENT band.
    """
    import sweep._C as _C
    seen = {}
    real = _C.sem2d_fluid_solid_forward

    def spy(inp, *rest):
        seen[spy.tag] = tuple(int(getattr(inp, nm).numel()) for nm in
                              ("band_idx", "band_node_idx", "band_elem_idx"))
        return real(inp, *rest)

    _C.sem2d_fluid_solid_forward = spy
    try:
        for memory in (None, "boundary"):
            spy.tag = str(memory)
            p = _prop(memory=BoundarySaving() if memory else Full())
            p(_wav(), SRC, REC, models=_models(p))
    finally:
        _C.sem2d_fluid_solid_forward = real

    assert all(n > 0 for n in seen["None"]), (
        f"tape mode got empty band index sets {seen['None']}: the absorbing "
        "layer would silently stop absorbing")
    assert seen["None"] == seen["boundary"], seen

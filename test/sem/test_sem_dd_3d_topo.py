"""3-D acoustic DD on a DEFORMED (topography) mesh -- ``AcousticSEM3DDeformed``.

Same discipline as the flat 3-D twin: the reference is the SAME per-strip
consts pushed through the FUSED single-domain driver, px=1 isolates the stepped
split, px>1 adds the decomposition, and the thresholds anchor to the fused
driver's own run-to-run atomics floor rather than to a wished-for constant.

What is new here is only the geometry the strips carry. The flat kernels read
``w_jac`` per THREAD (every element's slice is identical on a flat mesh); the
deformed ones read ``w_J`` and the nine inverse-Jacobian metrics per
ELEMENT-NODE. So a strip cannot rebuild its geometry -- it slices all ten
tensors from the global build by element id, and rebuilds the strip mesh only
for its ibool numbering. ``_assert_strip_ibool`` is what stops that split from
going quietly wrong, and it runs on every strip built here.

``test_topography_actually_deforms_the_result`` is the guard that makes the
rest mean something: with a flat-enough hill every comparison below would pass
against a propagator that ignored the topography entirely.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0 = 4, 40.0, 8.0
# 4 x 3 x 2 model elements; with the 2-element absorbing layer added outside,
# the mesh is 6 x 7 x 6 -- nex=6 divides 2 and 3
NZ, NY, NX = 17, 13, 9
NT, DT = 170, 1.2e-3
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4
HILL = 110.0                     # ~2.75 cells of relief; see the guard test


def _ztop(x, y):
    return HILL * torch.exp(-(((x - NX * DH * 0.5) / 220.0) ** 2
                              + ((y - NY * DH * 0.5) / 190.0) ** 2))


def _rows(z_top):
    """``z_top(x, y)`` (metres, model frame) as PropSEM's topography: the
    surface row of every model column, ``(NY, NX)``."""
    y, x = torch.meshgrid(torch.arange(NY, dtype=torch.float64) * DH,
                          torch.arange(NX, dtype=torch.float64) * DH,
                          indexing="ij")
    return (z_top(x, y) / DH).numpy()


def _prop(hill=True, memory=None):
    return S.PropSEM(AcousticSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=2 * NO, memory=memory, f_dominant=F0,
                     topography=_rows(_ztop if hill else (lambda x, y: 0.0 * x)),
                     dev="cuda", impl="c")


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
    return m.nex * NO + 1, m.ney * NO + 1, m.nez * NO + 1


def _gid(ix, iy, iz, nxg, nyg):
    return (iz * nyg + iy) * nxg + ix


def _geometry(prop, vp):
    nxg, nyg, _ = _dims(prop, vp)
    src = _gid(nxg // 2, nyg // 2, 2 * NO, nxg, nyg)
    rec = torch.tensor([_gid(x, y, NO, nxg, nyg)
                        for y in (nyg // 3, 2 * nyg // 3)
                        for x in range(3, nxg - 3, 3)]).cuda()
    return src, rec


def _fused_reference(prop, vp, wav, src, rec):
    """The single-strip run of the same driver (the fused Function is gone)."""
    return SEMModelParallel(prop, StripTopology(1))(wav, src, rec, [vp])


def _run_dd(prop, vp, wav, src, rec, px):
    return SEMModelParallel(prop, StripTopology(px))(wav, src, rec, [vp])


def _rel(got, ref):
    return float((got - ref).abs().max() / ref.abs().max())


def _cos(x, y):
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-300))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("px", [1, 2, 3])
def test_topo_dd_forward_matches_fused(px):
    prop = _prop()
    vp, wav = _vp(), _wav()
    src, rec = _geometry(prop, vp)
    with torch.no_grad():
        ref1 = _fused_reference(prop, vp, wav, src, rec)
        ref2 = _fused_reference(prop, vp, wav, src, rec)
        got = _run_dd(prop, vp, wav, src, rec, px)
    assert torch.isfinite(got).all()
    floor = _rel(ref1, ref2)
    err = _rel(got, ref1)
    assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"px={px}: topo DD vs fused rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [2, 3])
def test_topo_dd_gradients_match_fused(px):
    prop = _prop()

    def grads(runner):
        vp, wav = _vp(grad=True), _wav(grad=True)
        src, rec = _geometry(prop, vp)
        out = runner(prop, vp, wav, src, rec)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone(), wav.grad.clone()

    ref = grads(_fused_reference)
    ref2 = grads(_fused_reference)
    got = grads(lambda p, v, w, s, r: _run_dd(p, v, w, s, r, px))
    assert float(ref[0].abs().max()) > 0.0, "reference gradient is all zeros"
    for j, name in enumerate(("vp", "wavelet")):
        floor = _rel(ref[j], ref2[j])
        err = _rel(got[j], ref[j])
        assert err <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"px={px} d{name}: rel {err:.2e}, floor {floor:.2e}")


@pytest.mark.parametrize("px", [1, 2])
def test_topo_dd_boundary_saving_matches_the_tape(px):
    """Boundary saving over a deformed strip: the band is sliced from the same
    global profile on both sides of a cut, exactly as on the flat mesh."""
    # the memory is a constructor option: one propagator per mode
    props = {None: _prop(), "boundary": _prop(memory=BoundarySaving())}

    def grads(memory, px_):
        prop = props[memory]
        vp, wav = _vp(grad=True), _wav(grad=True)
        src, rec = _geometry(prop, vp)
        out = _run_dd(prop, vp, wav, src, rec, px_)
        (out ** 2).sum().div(out.detach().pow(2).sum().clamp_min(1e-30)
                             ).backward()
        return vp.grad.clone()

    tape = grads(None, px_=px)
    bs = grads("boundary", px_=px)
    c = _cos(bs, tape)
    assert c >= 1 - 1e-6, f"px={px}: BS vs tape grad cos={c:.8f}"


def test_topography_actually_deforms_the_result():
    """Without this, every comparison above would also pass for a propagator
    that silently dropped the topography and ran the flat geometry."""
    flat, hill = _prop(hill=False), _prop()
    vp, wav = _vp(), _wav()
    src, rec = _geometry(hill, vp)
    with torch.no_grad():
        a = _run_dd(flat, vp, wav, src, rec, 2)
        b = _run_dd(hill, vp, wav, src, rec, 2)
    rel = _rel(b, a)
    assert rel > 0.05, (
        f"topography changed the record by only {rel:.2e} -- this grid is too "
        "flat for the tests above to discriminate")


def test_strip_ibool_guard_is_live():
    """The rebuild-vs-slice guard must actually run on this path (it is what
    makes slicing ten geometry tensors by element id safe)."""
    import sweep.sem.dd as DD
    prop = _prop()
    vp = _vp()
    src, rec = _geometry(prop, vp)
    eq, _ = prop._resolve_model_batch([vp])
    pat = torch.zeros(eq.mesh.n_global, device="cuda")
    pat[int(src)] = 1.0
    calls = []
    orig = DD._assert_strip_ibool
    DD._assert_strip_ibool = lambda *a: (calls.append(1), orig(*a))[1]
    try:
        SEMModelParallel(prop, StripTopology(3))._parts3d(eq, (pat,), rec)
    finally:
        DD._assert_strip_ibool = orig
    assert len(calls) == 3, f"guard ran {len(calls)} times, expected one/strip"

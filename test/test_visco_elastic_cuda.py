"""ViscoElastic impl='c' (CUDA): forward and hand-written adjoint vs the eager
autograd reference, storage modes, determinism, elastic limit and guards.

The eager reference runs with ``use_compile=False`` (the compiled eager path's
fused arithmetic is not a gradient reference).  Geometry keeps a real physical
interior (72 x 88 cells inside a 20-cell PML), heterogeneous vp/vs/rho/Qp/Qs,
and a random linear functional of the record as the loss.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ViscoElastic
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' needs CUDA")
DEV = "cuda"
NZ, NX, DH, DT, NT, ABCN = 72, 88, 10.0, 1e-3, 400, 20
FREF = 12.0


def _setup():
    t = np.arange(NT) * DT - 0.08
    wav = torch.tensor((1e3 * ricker(t, f=FREF)).astype(np.float32), device=DEV)
    src = np.array([[NX // 2, 12]], np.int64)
    rx = np.arange(6, NX - 6, 3)
    rec = np.stack([rx, np.full_like(rx, 4)], -1)[None]
    zz = torch.linspace(0, 1, NZ, device=DEV)[:, None] * torch.ones(1, NX, device=DEV)
    box = torch.zeros(NZ, NX, device=DEV)
    box[30:45, 30:55] = 1
    vp = 2000 + 700 * zz + 250 * box
    vs = vp / 1.8 - 60 * box
    rho = 1800 + 400 * zz + 80 * box
    Qp = 40 + 30 * zz - 20 * box
    Qs = 25 + 20 * zz - 12 * box
    return wav, src, rec, [vp, vs, rho, Qp, Qs]


def _prop(eq, impl, fs, **kw):
    if impl == "eager":
        kw = dict(use_ckpt=False, use_compile=False, **kw)
    return PropTorch(eq, shape=(NZ, NX), free_surface=fs, abcn=ABCN, dh=DH, dt=DT,
                     impl=impl, **kw)


def _run(eq, models, impl, fs, weight_seed=1, rec=None, **kw):
    wav, src, rec0, _ = _setup()
    rec = rec0 if rec is None else rec
    ms = [m.clone().requires_grad_(True) for m in models]
    d = _prop(eq, impl, fs, **kw)(wav, src.copy(), rec.copy(), models=ms)
    w = torch.randn(d.shape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(weight_seed))
    (d * w).sum().backward()
    return d.detach(), [m.grad.detach() for m in ms]


def _cos_rel(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm())), float((a - b).norm() / b.norm())


FULL = dict(memory=Full())


def _metrics(c, e):
    (dc, gc), (de, ge) = c, e
    fwd = float((dc - de).double().norm() / de.double().norm())
    return fwd, [_cos_rel(a, b) for a, b in zip(gc, ge)]


PARAMS = ["vp", "vs", "rho", "Qp", "Qs"]
SOURCES = [["sxx", "szz"], ["sxx"], ["szz"], ["sxz"], ["vx"], ["vz"]]
RECEIVERS = [["vx", "vz"], ["sxx", "szz"], ["vz", "sxx"]]


def _receivers(geometry):
    """``surface_line``: 4 cells below the top; ``colocated``: along the source
    row THROUGH the source cell (the receiver/source rho-imaging corrections)."""
    if geometry == "surface_line":
        rx = np.arange(6, NX - 6, 3)
        return np.stack([rx, np.full_like(rx, 4)], -1)[None]
    rx = np.arange(5, NX - 5, 3)
    rec = np.stack([rx, np.full_like(rx, 12)], -1)[None]
    assert (rec[0, :, 0] == NX // 2).any(), "a receiver must sit on the source cell"
    return rec


# c vs eager for every source x receiver x geometry x free-surface combination.
# Measured on RTX 6000 Ada (fp32, --use_fast_math) over all 72: record <= 6.8e-5,
# gradients <= 8.8e-5 (vp; vs, rho, Qp, Qs <= 7.3e-5), cos = 1.00000 -- on every
# combination the same as Elastic's own c-vs-eager floor there.
@pytest.mark.parametrize("fs", [True, False], ids=["fs", "nofs"])
@pytest.mark.parametrize("geometry", ["surface_line", "colocated"])
@pytest.mark.parametrize("src_type", SOURCES, ids=lambda s: "src_" + "+".join(s))
@pytest.mark.parametrize("rec_type", RECEIVERS, ids=lambda s: "rec_" + "+".join(s))
def test_c_matches_eager_source_receiver_combinations(fs, geometry, src_type, rec_type):
    _, _, _, models = _setup()
    rec = _receivers(geometry)
    kw = dict(rec=rec, source_type=src_type, receiver_type=rec_type)
    c = _run(ViscoElastic(4, DEV, f_ref=FREF), models, "c", fs, **FULL, **kw)
    e = _run(ViscoElastic(4, DEV, f_ref=FREF), models, "eager", fs, **kw)
    fwd, grads = _metrics(c, e)
    assert fwd < 3e-4
    for name, (cos, rel) in zip(PARAMS, grads):
        assert cos > 0.99999 and rel < 3e-4, (name, cos, rel)


def test_c_matches_eager_per_edge_free_surface():
    _, _, _, models = _setup()
    fs = ["top", "left"]
    fwd, grads = _metrics(_run(ViscoElastic(4, DEV, f_ref=FREF), models, "c", fs, **FULL),
                          _run(ViscoElastic(4, DEV, f_ref=FREF), models, "eager", fs))
    assert fwd < 3e-4
    for name, (cos, rel) in zip(PARAMS, grads):
        assert cos > 0.99999 and rel < 3e-4, (name, cos, rel)


def _closed_box_run(impl, src_near_high):
    wav, src, rec, models = _setup()
    if src_near_high:     # exercise the bottom/right (high-side) surface solve
        src = np.array([[NX - 1 - 14, NZ - 1 - 12]], np.int64)
        rx = np.arange(6, NX - 6, 3)
        rz = np.arange(6, NZ - 6, 3)
        rec = np.concatenate([np.stack([rx, np.full_like(rx, NZ - 1 - 4)], -1),
                              np.stack([np.full_like(rz, NX - 1 - 3), rz], -1)])[None]
    ms = [m.clone().requires_grad_(True) for m in models]
    kw = FULL if impl == "c" else {}
    d = _prop(ViscoElastic(4, DEV, f_ref=FREF), impl, ["top", "bottom", "left", "right"], **kw)(
        wav, src.copy(), rec.copy(), models=ms)
    w = torch.randn(d.shape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))
    (d * w).sum().backward()
    return d.detach(), [m.grad.detach() for m in ms]


@pytest.mark.parametrize("src_near_high", [False, True], ids=["low_faces", "high_faces"])
def test_c_matches_eager_closed_box(src_near_high):
    """No PML anywhere: records agree to ~1e-6, gradients to <= 1.3e-5 (the
    eager closed-box backward itself repeats only to ~1e-6).  The surface
    rows/columns next to the source carry 3-30% of each gradient; they are
    checked on their own so an error in the surface-solve adjoint cannot hide
    in a global norm."""
    c = _closed_box_run("c", src_near_high)
    e = _closed_box_run("eager", src_near_high)
    fwd, grads = _metrics(c, e)
    assert fwd < 1e-5
    for name, (cos, rel) in zip(["vp", "vs", "rho", "Qp", "Qs"], grads):
        assert rel < 1e-4, (name, rel)
    faces = [(-1, slice(None)), (slice(None), -1)] if src_near_high else [(0, slice(None)), (slice(None), 0)]
    for sel in faces:
        for name, gc, ge in zip(["vp", "vs", "rho", "Qp", "Qs"], c[1], e[1]):
            a, b = gc[sel].double(), ge[sel].double()
            assert float(b.norm() / ge.double().norm()) > 1e-3, "face must carry gradient"
            assert float((a - b).norm() / b.norm()) < 1e-4, (name, sel)


@pytest.mark.parametrize("storage", ["gpu", "cpu"])
def test_ckpt_bitwise_equals_full(storage):
    """Chunk and recursive checkpointing run the full mode's kernel sequence
    (the imaging is fused into the stress-adjoint prepare in every mode)."""
    _, _, _, models = _setup()
    eq = lambda: ViscoElastic(4, DEV, f_ref=FREF)
    fs = ["top", "left"]
    full = _run(eq(), models, "c", fs, **FULL)
    runs = [Ckpt(chunks=chunks, storage=storage) for chunks in (64, 150)]   # 150: no divisor, tests the carry
    runs.append(Ckpt(mode="recursive", count=6, storage=storage))
    for ckpt in runs:
        ck = _run(eq(), models, "c", fs, memory=ckpt)
        assert torch.equal(full[0], ck[0]), ckpt
        assert all(torch.equal(a, b) for a, b in zip(full[1], ck[1])), ckpt


@pytest.mark.parametrize("n_sls", [1, 2, 4])
def test_c_matches_eager_mechanism_counts(n_sls):
    """The compiled traits are a template on the mechanism count (3 is the
    default every other test runs)."""
    _, _, _, models = _setup()
    eq = lambda: ViscoElastic(4, DEV, f_ref=FREF, n_sls=n_sls)
    fwd, grads = _metrics(_run(eq(), models, "c", True, **FULL), _run(eq(), models, "eager", True))
    assert fwd < 3e-4
    for name, (cos, rel) in zip(PARAMS, grads):
        assert cos > 0.99999 and rel < 3e-4, (n_sls, name, cos, rel)


def test_deterministic_and_default_is_full():
    _, _, _, models = _setup()
    eq = lambda: ViscoElastic(4, DEV, f_ref=FREF)
    a = _run(eq(), models, "c", ["top", "left"], **FULL)
    b = _run(eq(), models, "c", ["top", "left"], **FULL)
    dflt = _run(eq(), models, "c", ["top", "left"])      # boundary default -> full
    for other in (b, dflt):
        assert torch.equal(a[0], other[0])
        assert all(torch.equal(x, y) for x, y in zip(a[1], other[1]))


def test_q_inf_matches_elastic_c():
    _, _, _, (vp, vs, rho, _, _) = _setup()
    inf = torch.full_like(vp, float("inf"))
    e = _run(Elastic(4, DEV), [vp, vs, rho], "c", True, **FULL)
    v = _run(ViscoElastic(4, DEV, f_ref=FREF), [vp, vs, rho, inf, inf], "c", True, **FULL)
    assert float((v[0] - e[0]).norm() / e[0].norm()) < 2e-5
    for a, b in zip(v[1][:3], e[1]):
        assert _cos_rel(a, b)[1] < 5e-5
    assert all(float(g.abs().max()) == 0.0 for g in v[1][3:])


@pytest.mark.parametrize("what", ["boundary", "topography"])
def test_unsupported_modes_raise(what):
    _, _, _, models = _setup()
    kw = {
        "boundary": dict(memory=BoundarySaving()),
        "topography": dict(topography=np.full(NX, 3)),
    }[what]
    with pytest.raises((NotImplementedError, RuntimeError)):
        _run(ViscoElastic(4, DEV, f_ref=FREF), models, "c", True, **kw)

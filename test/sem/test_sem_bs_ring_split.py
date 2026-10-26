"""Boundary-saving rings per field: each C-PML memory field on its own support.

A memory field ``psi_d`` / ``zeta_d`` is updated as ``b_d * (...) + a_d * psi_d``
from a zero start, so it is exactly ``0.0`` at every node where ``b_d == 0``,
for all time. The ring used to store it on the union band anyway. The union
of the three axis slabs is ~2.4x any one slab at production sizes, so the
ring -- 74-90% of a boundary-saved backward's peak -- was 2.2-2.5x larger
than its information.

What has to hold, and is checked here rather than argued:

* the claim itself, on the equation's own arithmetic: every memory field is
  exactly zero off its support at every step of an eager rollout;
* the split ring equals the union ring on the support and the union ring is
  exactly zero outside it, and the int8 scale tables agree -- measured on the
  stored bytes of the C drivers, single-domain and DD, fp32/bf16/int8, so
  the split cannot have dropped a value;
* the eager path gives the SAME record and gradient with either layout,
  ``torch.equal``, on the CPU where it is deterministic;
* rim mode empties the six memory rings (the rim carries no ``b``);
* the C paths' end-to-end gradient with either layout agrees to the cosine
  floor of this path (its assembly is a float atomicAdd; see
  ``lesson_sem_dd_no_bitexact_criterion``).
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
import sweep.sem.prop_c as PC
import sweep.sem.dd as DD
from sweep.propagator import BoundarySaving
from sweep.sem.boundary_saving import band_support_idx, cpml_ring_masks
from sweep.sem.equations import AcousticPotentialSEM2D, AcousticSEM3D
from sweep.sem.prop_c import sem_c_available
from sweep.sem.drivers.acoustic2d import _AC2D_RINGS

needs_c = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, DT = 4, 25.0, 8.0, 6e-4
SUPPORT = {"psi_x": "b_x", "psi_y": "b_y", "psi_z": "b_z",
           "zeta_x": "b_x", "zeta_y": "b_y", "zeta_z": "b_z"}


def _wav(nt, device):
    t = (torch.arange(nt, dtype=torch.float32, device=device) - 6.0) * DT * F0
    return (1.0 - 2.0 * (3.1415926 * t) ** 2) * torch.exp(-(3.1415926 * t) ** 2)


def _prop(eq, shape, dev, impl, mode="full", abcn=2 * NO, nt=24, memory=None):
    return S.PropSEM(eq, shape=shape, dh=DH, dt=DT, nt=nt, n_order=NO,
                     free_surface=("top",), abcn=abcn, f_dominant=F0,
                     dev=dev, impl=impl, memory=memory, bs_band_mode=mode)


def _bs(store):
    """Boundary saving with its ring stored as ``store`` (None: fp32)."""
    return BoundarySaving(
        storage_dtype={None: "fp32", torch.int8: "int8"}.get(store, store))


def _geometry(shape, device):
    torch.manual_seed(0)
    vp = (2200.0 + 200.0 * torch.rand(*shape, device=device)).requires_grad_(True)
    if len(shape) == 2:
        src = torch.tensor([[shape[1] // 2, 8]], device=device)
        rec = torch.tensor([[[4 + 3 * i, 8] for i in range(6)]], device=device)
    else:
        c = shape[2] // 2
        src = torch.tensor([[c, c, 8]], device=device)
        rec = torch.tensor([[[4 + 2 * i, c, 8] for i in range(6)]], device=device)
    return vp, src, rec


# --------------------------------------------------------------------------- #
# 1. The claim, on the equation's own arithmetic (eager, CPU, every step).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eq,shape", [(AcousticPotentialSEM2D, (25, 33)),
                                      (AcousticSEM3D, (21, 21, 21))])
def test_memory_fields_vanish_exactly_off_their_support(eq, shape):
    p = _prop(eq, shape, "cpu", "eager")
    vp, src, rec = _geometry(shape, "cpu")
    e, _ = p._resolve_model_batch([vp.detach()])
    names = [s.name for s in e.field_specs]
    fields = list(e.init_state(p.dt, torch.device("cpu"), torch.float32))
    ng = e.mesh.n_global
    pat = torch.zeros(ng)
    # a source somewhere in the interior: any non-zero drive will do
    pat[ng // 2] = 1.0
    wav = _wav(p.nt, "cpu")
    hit = 0
    for n in range(p.nt):
        fields = list(e.step(tuple(fields), [pat * wav[n]], p.dt))
        for f, nm in enumerate(names):
            if nm not in SUPPORT:
                continue
            b = getattr(e.cpml, SUPPORT[nm]).reshape(-1)
            off = fields[f][b == 0]
            assert bool((off == 0).all()), (
                f"{nm} is non-zero at {int((off != 0).sum())} nodes with "
                f"{SUPPORT[nm]} == 0 at step {n} -- the support claim is false")
            hit += int((fields[f] != 0).any())
    assert hit > 0, "no memory field ever became non-zero: the test saw nothing"


# --------------------------------------------------------------------------- #
# 2. The indices the C paths ring on.
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("mode", ["full", "rim"])
def test_strip_ring_indices_are_the_axis_slabs(mode):
    p = _prop(AcousticSEM3D, (33, 33, 33), "cuda", "c", mode=mode,
              memory=BoundarySaving())
    vp, _, _ = _geometry((33, 33, 33), "cuda")
    src = DD._CompactEq3D.build(p, vp.detach())
    nzg, nyg, nxg = p._grid_shape()        # the model AND its absorbing layer
    ng = nzg * nyg * nxg
    rec = torch.tensor([ng // 3], device="cuda")
    parts, _ = DD.SEMModelParallel(p, DD.StripTopology(2))._parts3d(src, ng // 2, rec)
    for q in parts:
        c = q.c
        band = c["band_idx"]
        for ax in "xyz":
            got = c[f"band_idx_{ax}"]
            want = band[c[f"b_{ax}"][band] != 0]
            assert torch.equal(got, want)
            assert torch.equal(got, torch.sort(got).values)      # a filter, not a re-sort
            if mode == "rim":
                assert got.numel() == 0, "the rim carries no b, so no memory ring"
        if mode == "full":
            u = torch.unique(torch.cat([c[f"band_idx_{ax}"] for ax in "xyz"]))
            assert u.numel() < band.numel(), "supports should not cover the band"


# --------------------------------------------------------------------------- #
# 3. Split vs union layout on the C paths: rings bitwise, gradient to the floor.
# --------------------------------------------------------------------------- #
def _spy_rings(fn, attr):
    seen = {}
    real = fn.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        seen["ctx"] = ctx
        return out

    fn.forward = staticmethod(spy)
    return seen, lambda: setattr(fn, "forward", staticmethod(real))


def _positions(sub, full):
    pos = torch.searchsorted(full, sub)
    assert torch.equal(full[pos], sub)
    return pos


def _cos_gap(a, b):
    x, y = a.double().reshape(-1), b.double().reshape(-1)
    return abs(1.0 - (x @ y / (x.norm() * y.norm())).item())


def _fused(eq, shape, store, split, spy_fn):
    PC._BS_RING_SPLIT = split
    try:
        p = _prop(eq, shape, "cuda", "c", memory=_bs(store))
        vp, src, rec = _geometry(shape, "cuda")
        seen, undo = _spy_rings(spy_fn, "bs_saves")
        try:
            out = p.forward(_wav(p.nt, "cuda"), src, rec, [vp])
            out.pow(2).sum().backward()
        finally:
            undo()
        ctx = seen["ctx"]
        n_ring = 5 if len(shape) == 2 else 7
        if isinstance(ctx.carry, list):    # the strip driver (3-D): one strip
            cr = ctx.carry[0]
            rings, scale, c = ([cr[k] for k in DD._AC3D_RINGS], cr["band_scale"],
                               ctx.parts[0].c)
        else:                              # the fused 2-D Function
            cr = ctx.carry
            rings, scale, c = [cr[k] for k in _AC2D_RINGS], cr["band_scale"], ctx.c
        return {"rings": [r.detach().cpu() for r in rings],
                "scale": scale.detach().cpu(),
                "c": {k: v.detach().cpu() for k, v in c.items()
                      if torch.is_tensor(v) and k.startswith(("band_idx", "b_"))},
                "d_vp": vp.grad.detach().cpu(), "record": out.detach().cpu()}
    finally:
        PC._BS_RING_SPLIT = True


def _rel(a, b):
    a, b = a.float(), b.float()
    if not a.numel():
        return 0.0
    return ((a - b).abs().max() / a.abs().max().clamp_min(1e-300)).item()


# On-support agreement bars, by STORAGE FORMAT, relative to the ring's max.
# The rings hold the forward fields, whose assembly is a float atomicAdd: two
# runs of the same layout differ at ~1e-6 relative (measured worst 1.05e-6 over
# 216 rings), and a bf16 or int8 ring rounds that noise to whole ulps. So the
# bar is one storage ulp: 2^-8 for bf16, 1/127 for int8 (measured worst 9.1e-4
# and 0.0 across two layouts x two runs); fp32 gets ten forward floors. A
# per-field floor from ONE A/A sample is not a bound -- it is a heavy-tailed
# draw (a field's A/A came out 3.5e-9 in one run and 1.9e-6 in the next) --
# which is what the first version of this test tried and why it is gone.
_ON_SUPPORT_BAR = {None: 1e-5, "bf16": 2.0 ** -8, "int8": 1.0 / 127.0}
_SCALE_BAR = 1e-5


def _check_layouts(union, split, names, store):
    """What the split rests on, measured on the stored bytes.

    DECISIVE and bitwise, on the union run alone: every entry the split
    layout no longer stores -- the union ring OUTSIDE the field's support --
    is exactly ``0.0``. Nothing else is needed for "nothing was lost".

    On the support the two layouts hold the same forward fields from two
    runs, so they agree to the storage-format bar above; the int8 scale
    tables (an fp32 amax of those fields) to ten forward floors.
    """
    band = union["c"]["band_idx"]
    assert torch.equal(band, split["c"]["band_idx"])
    bar = _ON_SUPPORT_BAR[store]
    for j, nm in enumerate(names):
        ru, rs = union["rings"][j], split["rings"][j]
        if nm in SUPPORT:
            ax = SUPPORT[nm][-1]
            idx = split["c"][f"band_idx_{ax}"]
            assert rs.shape[-1] == idx.numel(), f"{nm}: ring width != |support|"
            pos = _positions(idx, band)
            keep = torch.zeros(band.numel(), dtype=torch.bool)
            keep[pos] = True
            outside = ru[..., ~keep]
            assert not outside.numel() or bool((outside == 0).all()), (
                f"{nm}: {int((outside != 0).sum())} non-zero entries were stored "
                "outside the support -- the split would lose them")
            on = ru[..., pos]
        else:
            on = ru
        got = _rel(on, rs)
        assert got <= bar, f"{nm}: union-vs-split ring differs by {got:.3e} (bar {bar:.1e})"
    if union["scale"].numel():
        got = _rel(union["scale"], split["scale"])
        assert got <= _SCALE_BAR, f"int8 scale tables differ by {got:.3e}"


@needs_c
@pytest.mark.parametrize("store", [None, "bf16", "int8"])
def test_fused_3d_split_rings_match_the_union_layout(store):
    names = ("u", "psi_x", "psi_y", "psi_z", "zeta_x", "zeta_y", "zeta_z")
    union = _fused(AcousticSEM3D, (33, 33, 33), store, False, DD._SEMAcoustic3DCpmlDDC)
    split = _fused(AcousticSEM3D, (33, 33, 33), store, True, DD._SEMAcoustic3DCpmlDDC)
    a2 = _fused(AcousticSEM3D, (33, 33, 33), store, True, DD._SEMAcoustic3DCpmlDDC)
    _check_layouts(union, split, names, store)
    floor = max(_cos_gap(split["d_vp"], a2["d_vp"]), 1e-12)
    assert _cos_gap(union["d_vp"], split["d_vp"]) <= 10 * floor
    assert torch.isfinite(split["d_vp"]).all()


@needs_c
@pytest.mark.parametrize("store", [None, "int8"])
def test_fused_2d_split_rings_match_the_union_layout(store):
    names = ("chi", "psi_x", "psi_z", "zeta_x", "zeta_z")
    union = _fused(AcousticPotentialSEM2D, (33, 49), store, False, PC._SEMAcousticCpmlC)
    split = _fused(AcousticPotentialSEM2D, (33, 49), store, True, PC._SEMAcousticCpmlC)
    a2 = _fused(AcousticPotentialSEM2D, (33, 49), store, True, PC._SEMAcousticCpmlC)
    _check_layouts(union, split, names, store)
    floor = max(_cos_gap(split["d_vp"], a2["d_vp"]), 1e-12)
    assert _cos_gap(union["d_vp"], split["d_vp"]) <= 10 * floor


@needs_c
def test_layout_check_rejects_a_misaligned_support():
    """Negative control for the criterion above: if the split's index were off
    by ONE position, the on-support comparison must fail loudly -- otherwise
    the fp32/bf16/int8 bars would be tolerances that hide a real misplacement
    rather than bounds that admit only the forward's noise."""
    names = ("u", "psi_x", "psi_y", "psi_z", "zeta_x", "zeta_y", "zeta_z")
    union = _fused(AcousticSEM3D, (33, 33, 33), None, False, DD._SEMAcoustic3DCpmlDDC)
    split = _fused(AcousticSEM3D, (33, 33, 33), None, True, DD._SEMAcoustic3DCpmlDDC)
    band = split["c"]["band_idx"]
    bad = dict(split)
    bad["c"] = dict(split["c"])
    # roll psi_x's support one band position inward: same length, wrong nodes
    pos = _positions(split["c"]["band_idx_x"], band)
    bad["c"]["band_idx_x"] = band[(pos + 1).clamp_max(band.numel() - 1)]
    with pytest.raises(AssertionError):
        _check_layouts(union, bad, names, None)


@needs_c
def test_split_ring_holds_fewer_bytes_than_the_union():
    """The point of the change, on the ring tensors themselves."""
    split = _fused(AcousticSEM3D, (33, 33, 33), None, True, DD._SEMAcoustic3DCpmlDDC)
    union = _fused(AcousticSEM3D, (33, 33, 33), None, False, DD._SEMAcoustic3DCpmlDDC)
    nb_s = sum(r.numel() for r in split["rings"])
    nb_u = sum(r.numel() for r in union["rings"])
    assert nb_u / nb_s > 1.5, f"union/split = {nb_u / nb_s:.2f}"


# --------------------------------------------------------------------------- #
# 4. DD: the strips' rings, split vs union, and the gradient.
# --------------------------------------------------------------------------- #
def _dd3d(store, split, px=2):
    PC._BS_RING_SPLIT = split
    try:
        n = 33
        p = _prop(AcousticSEM3D, (n, n, n), "cuda", "c", memory=_bs(store))
        torch.manual_seed(0)
        vp = (2200.0 + 200.0 * torch.rand(n, n, n, device="cuda")).requires_grad_(True)
        # GLOBAL node ids, on the node grid of the model AND its absorbing layer
        nzg, nyg, nxg = p._grid_shape()
        src = (2 * NO * nyg + nyg // 2) * nxg + nxg // 2
        rec = torch.tensor([src + 5 + 2 * i for i in range(6)], device="cuda")
        seen, undo = _spy_rings(DD._SEMAcoustic3DCpmlDDC, "rings")
        try:
            out = DD.SEMModelParallel(p, DD.StripTopology(px))(
                _wav(p.nt, "cuda"), src, rec, [vp])
            out.pow(2).sum().backward()
        finally:
            undo()
        ctx = seen["ctx"]
        parts = [{"rings": [ctx.carry[i][k].detach().cpu() for k in DD._AC3D_RINGS],
                  "scale": ctx.carry[i]["band_scale"].detach().cpu(),
                  "c": {k: v.detach().cpu() for k, v in pp.c.items()
                        if torch.is_tensor(v) and k.startswith(("band_idx", "b_"))}}
                 for i, pp in enumerate(ctx.parts)]
        return parts, vp.grad.detach().cpu()
    finally:
        PC._BS_RING_SPLIT = True


@needs_c
@pytest.mark.parametrize("store", [None, "int8"])
def test_dd_3d_split_rings_match_the_union_layout(store):
    names = ("u", "psi_x", "psi_y", "psi_z", "zeta_x", "zeta_y", "zeta_z")
    pu, gu = _dd3d(store, False)
    ps, gs = _dd3d(store, True)
    _, g2 = _dd3d(store, True)
    for a, b in zip(pu, ps):
        _check_layouts(a, b, names, store)
    floor = max(_cos_gap(gs, g2), 1e-12)
    assert _cos_gap(gu, gs) <= 10 * floor


# --------------------------------------------------------------------------- #
# 5. Eager: same numbers with either layout, bitwise (one CPU thread is
#    deterministic).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eq,shape", [(AcousticPotentialSEM2D, (25, 33)),
                                      (AcousticSEM3D, (21, 21, 21))])
@pytest.mark.parametrize("mode", ["full", "rim"])
@pytest.mark.parametrize("store", [None, torch.int8])
def test_eager_split_equals_union_layout_bitwise(monkeypatch, eq, shape, mode, store):
    import sweep.sem.boundary_saving as BS

    def run(split):
        if not split:
            monkeypatch.setattr(BS, "cpml_ring_masks", lambda eq_, idx: {})
        else:
            monkeypatch.setattr(BS, "cpml_ring_masks", cpml_ring_masks)
        p = _prop(eq, shape, "cpu", "eager", mode=mode, memory=_bs(store))
        vp, src, rec = _geometry(shape, "cpu")
        out = p.forward(_wav(p.nt, "cpu"), src, rec, [vp])
        out.pow(2).sum().backward()
        return out.detach(), vp.grad.clone()

    # on ONE thread: with several, the CPU's parallel scatters sum in a varying
    # order once the mesh is large enough, and two runs of the SAME layout
    # already differ in the last bit (the 3-D one here: 39.7k nodes)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ru, gu = run(False)
        rs, gs = run(True)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(ru, rs), "eager record differs between ring layouts"
    assert torch.equal(gu, gs), "eager gradient differs between ring layouts"
    assert float(gs.abs().max()) > 0.0


def test_band_support_idx_is_a_filter_of_the_band():
    band = torch.tensor([3, 5, 8, 13, 21])
    b = torch.zeros(30)
    b[5] = b[13] = 0.25
    assert torch.equal(band_support_idx(band, b), torch.tensor([5, 13]))
    assert band_support_idx(band, torch.zeros(30)).numel() == 0

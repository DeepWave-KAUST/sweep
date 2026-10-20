"""The contract every way of building a strip's consts must satisfy.

`_parts*` today builds ONE global equation and slices it per strip. That is
where the decomposition's correctness argument comes from: a single object is
the source of truth, so two strips sharing a cut cannot disagree. Removing the
global build -- the remaining lever on both the DD serial fraction and the
memory ceiling (a rank holds 158.5 B/node for it) -- moves that responsibility
onto "every strip generates from the same compact description", and the
argument has to be re-established by measurement instead.

This file is that measurement, written BEFORE the second path exists so it
cannot be shaped to fit it. It fixes two things:

* **the key set**. Adding or removing a consts entry becomes a deliberate act
  that edits this list, rather than something a refactor can do silently. A
  new entry that a future strip-built path forgets to produce would otherwise
  surface as a wrong kernel argument, not as a failure here.
* **a bit-identity comparator** over every entry, tensor or scalar. When the
  strip-built path lands, `compare_consts` is pointed at it and the bar stays
  `== 0.0` -- the two paths must agree exactly, because they are computing the
  same numbers, not approximating each other.

Until that path exists the comparison runs global-vs-global on two
independently built equations, which is not vacuous: it pins that the current
path is deterministic, and that nothing in it depends on allocation order or
on a previously cached profile.
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import AcousticSEM3D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NEX = 4, 40.0, 8.0, 16
# NEX counts the MESH's elements per axis, absorbing layer included: the node
# ids, strip widths and element counts below are mesh arithmetic. PropSEM adds
# the ABC-element layer outside the model, so the model is smaller by it on
# every side but the free top.
ABC = 4

# The exact consts `_parts3d` hands the 3-D acoustic kernels. Edit deliberately.
CONSTS_3D = {
    "a_x", "a_y", "a_z", "b_x", "b_y", "b_z", "band_idx",
    "band_idx_x", "band_idx_y", "band_idx_z", "D", "dbx_dx",
    "dby_dy", "dbz_dz", "deformed", "dirichlet_idx", "dt", "ibool", "ihx",
    "ihy", "ihz", "inv_M", "n_elements", "n_global", "n_rec", "ngll", "nt",
    "rec_G", "rec_W", "src_pattern", "w_jac", "w_jx", "w_jy", "w_jz",
    # the memory fields' axis-slab tables (sweep/sem/slab.py)
    "nx", "ny", "nz", "nsx", "nsy", "nsz",
    "tab_x", "tab_y", "tab_z", "planes_x", "planes_y", "planes_z",
    "slab_n_x", "slab_n_y", "slab_n_z", "elem_slab",
    # 1 byte per node, 1 where any b_d or db_d is non-zero: the node kernels'
    # interior fast-path test, six fp32 loads replaced by one
    "band_node",
    "ring_slot_x", "ring_slot_y", "ring_slot_z",
    "dir_slot_x", "dir_slot_y", "dir_slot_z",
}


def _build(seed=0):
    n = NEX * NO + 1
    shape = (n - ABC * NO, n - 2 * ABC * NO, n - 2 * ABC * NO)   # mesh: n^3
    p = S.PropSEM(AcousticSEM3D, shape=shape, dh=DH, dt=6e-4, nt=10,
                  n_order=NO, free_surface=("top",), abcn=ABC * NO,
                  f_dominant=F0, dev="cuda", impl="c")
    torch.manual_seed(seed)
    vp = 2500.0 + 200.0 * torch.rand(*shape, device="cuda")
    return p, vp


def _parts(p, vp, px):
    eq, _ = p._resolve_model_batch([vp])
    ng = eq.mesh.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[ng // 2] = 1.0
    rec = torch.tensor([ng // 3, ng // 3 + 5], device="cuda")
    return SEMModelParallel(p, StripTopology(px))._parts3d(eq, (pat,), rec)


def compare_consts(a, b, where=""):
    """Every entry equal, exactly. Returns the list of disagreements."""
    bad = []
    ka, kb = set(a), set(b)
    if ka != kb:
        bad.append(f"{where}: key sets differ, "
                   f"only-in-a={sorted(ka - kb)} only-in-b={sorted(kb - ka)}")
        return bad
    for k in sorted(ka):
        x, y = a[k], b[k]
        if torch.is_tensor(x) != torch.is_tensor(y):
            bad.append(f"{where}:{k}: tensor vs non-tensor")
        elif torch.is_tensor(x):
            if x.shape != y.shape:
                bad.append(f"{where}:{k}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
            elif x.dtype != y.dtype:
                bad.append(f"{where}:{k}: dtype {x.dtype} vs {y.dtype}")
            elif not torch.equal(x, y):
                d = float((x.double() - y.double()).abs().max())
                bad.append(f"{where}:{k}: max |diff| = {d:.3e}")
        elif x != y:
            bad.append(f"{where}:{k}: {x!r} vs {y!r}")
    return bad


@pytest.mark.parametrize("px", [1, 2, 4])
def test_the_consts_key_set_is_what_the_kernels_expect(px):
    p, vp = _build()
    for part in _parts(p, vp, px)[0]:
        assert set(part.c) == CONSTS_3D, (
            f"px={px}: consts key set drifted; "
            f"added={sorted(set(part.c) - CONSTS_3D)} "
            f"removed={sorted(CONSTS_3D - set(part.c))}")


@pytest.mark.parametrize("px", [1, 2, 4])
def test_two_independent_builds_give_identical_consts(px):
    """Same inputs, two separately built equations -> the same consts, exactly.
    This is the bar the strip-built path will have to clear."""
    pa, vpa = _build(seed=0)
    pb, vpb = _build(seed=0)
    parts_a, parts_b = _parts(pa, vpa, px)[0], _parts(pb, vpb, px)[0]
    assert len(parts_a) == len(parts_b) == px
    bad = []
    for r, (qa, qb) in enumerate(zip(parts_a, parts_b)):
        bad += compare_consts(qa.c, qb.c, where=f"px={px} strip={r}")
    assert not bad, "consts are not reproducible:\n  " + "\n  ".join(bad)


def test_a_different_model_does_change_the_consts():
    """Guards the comparator itself: if `compare_consts` reported equality for
    genuinely different inputs, every use of it above would be vacuous."""
    pa, vpa = _build(seed=0)
    pb, vpb = _build(seed=1)
    bad = compare_consts(_parts(pa, vpa, 2)[0][0].c,
                         _parts(pb, vpb, 2)[0][0].c, where="different-model")
    assert bad, "two different media produced identical consts"


# --------------------------------------------------------------------------- #
# The compact source vs the global equation.
# --------------------------------------------------------------------------- #
def _parts_compact(p, vp, px):
    from sweep.sem.dd import _CompactEq3D
    src = _CompactEq3D.build(p, vp)
    ng = src.mesh.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[ng // 2] = 1.0
    rec = torch.tensor([ng // 3, ng // 3 + 5], device="cuda")
    return SEMModelParallel(p, StripTopology(px))._parts3d(src, (pat,), rec)


@pytest.mark.parametrize("px", [1, 2, 4])
def test_compact_source_consts_equal_the_global_equations(px):
    """Every consts entry AND every material tensor, exactly equal. The
    compact source computes the same floats with the same ops -- the mass by
    the same expand-view index_add_, the C-PML by the same pointwise formulas
    on the same coordinates with the same c_max -- so `== 0.0` is the right
    bar and anything else is a bug, not a tolerance question."""
    p, vp = _build()
    parts_g, mats_g = _parts(p, vp, px)
    parts_c, mats_c = _parts_compact(p, vp, px)
    bad = []
    for r, (qg, qc) in enumerate(zip(parts_g, parts_c)):
        bad += compare_consts(qg.c, qc.c, where=f"px={px} strip={r}")
    for r, (mg, mc) in enumerate(zip(mats_g, mats_c)):
        if not torch.equal(mg, mc):
            d = float((mg.double() - mc.double()).abs().max())
            bad.append(f"px={px} mats[{r}]: max |diff| = {d:.3e}")
    assert not bad, "compact != global:\n  " + "\n  ".join(bad)


def test_compact_source_material_gradients_flow():
    """The DD material gradient reaches the INPUT model through the per-strip
    interpolation. A detached gather or scatter would zero it silently."""
    p, vp = _build()
    vp = vp.clone().requires_grad_(True)
    from sweep.sem.dd import _CompactEq3D
    src = _CompactEq3D.build(p, vp)
    ng = src.mesh.n_global
    pat = torch.zeros(ng, device="cuda")
    pat[ng // 2] = 1.0
    rec = torch.tensor([ng // 3], device="cuda")
    _, mats = SEMModelParallel(p, StripTopology(2))._parts3d(src, (pat,), rec)
    sum(m.pow(2).sum() for m in mats).backward()
    g = vp.grad
    assert g is not None and float(g.abs().max()) > 0.0, (
        "no gradient reached the model through the compact source")


def test_compact_source_holds_no_global_node_arrays():
    """The point of the compact source is that nothing n_global-sized or
    (E, ngll^3)-sized is built up front. Every tensor it owns must be tiny."""
    p, vp = _build()
    from sweep.sem.dd import _CompactEq3D
    src = _CompactEq3D.build(p, vp)
    ng = src.mesh.n_global
    big = []
    for k, v in vars(src).items():
        if torch.is_tensor(v) and v is not src.model and v.numel() >= ng // 4:
            big.append(f"{k}: {v.numel():,} elements")
    assert not big, "compact source holds global-sized tensors:\n  " + "\n  ".join(big)


@pytest.mark.parametrize("px", [1, 2, 4])
@pytest.mark.parametrize("where", ["interior", "on_cut"])
def test_node_index_source_equals_the_sliced_global_pattern(px, where):
    """`_parts3d` accepts a bare global node index in place of the pattern
    tuple and each strip slices its own unit pattern. That must give the same
    `src_pattern` as slicing an n_global pattern -- including a source that
    sits exactly ON a cut plane, which the slice puts on both strips."""
    from sweep.sem.dd import _CompactEq3D
    p, vp = _build()
    src = _CompactEq3D.build(p, vp)
    n = NEX * NO + 1
    gx = (NEX // 2) * NO if where == "on_cut" else n // 2 + 1
    node = (2 * NO * (n * n)) + (n // 2) * n + gx
    rec = torch.tensor([node + 7], device="cuda")
    pat = torch.zeros(src.mesh.n_global, device="cuda")
    pat[node] = 1.0
    dd = SEMModelParallel(p, StripTopology(px))
    parts_t, _ = dd._parts3d(src, (pat,), rec)
    parts_i, _ = dd._parts3d(src, node, rec)
    bad = []
    for r, (qt, qi) in enumerate(zip(parts_t, parts_i)):
        bad += compare_consts(qt.c, qi.c, where=f"px={px} strip={r}")
    assert not bad, "node-index source != sliced pattern:\n  " + "\n  ".join(bad)
    hits = sum(int(q.c["src_pattern"].sum()) for q in parts_i)
    assert hits == (2 if (where == "on_cut" and px in (2, 4)) else 1), (
        f"the unit source landed on {hits} strips")


def test_dd_forward_and_backward_never_build_the_global_mesh(monkeypatch):
    """The mesh's arrays were the last per-rank term that did not shrink with
    px (50.7 B/node). The compact path reads the propagator's Hex8Layout
    instead, so a full forward + backward must (a) never read `PropSEM.mesh`
    (the property is replaced by a tripwire for the duration), (b) build no
    Hex8Mesh other than the strips' own connectivity-only ones (every
    construction is spied on, whichever route it takes), and (c) leave the
    propagator with NO mesh -- while building one afterwards still works."""
    from sweep.sem.mesh.hex8 import Hex8Mesh
    p, vp = _build()
    vp = vp.clone().requires_grad_(True)
    n = NEX * NO + 1
    # inside the model, on the px=2 cut: the y = 0 face is absorbing layer
    # now, whose gradient is cut, and 10 steps do not reach the model from it
    src_node = (2 * NO * n + n // 2) * n + n // 2
    rec = torch.tensor([src_node + 7], device="cuda")
    t = torch.arange(10, dtype=torch.float32, device="cuda")
    assert p._mesh is None, "the constructor built the mesh"
    built = []
    orig_init = Hex8Mesh.__init__

    def spy(self, *a, **k):
        orig_init(self, *a, **k)
        built.append((self.nex, self.x_e is None))

    with monkeypatch.context() as mp:
        mp.setattr(Hex8Mesh, "__init__", spy)
        mp.setattr(type(p), "mesh", property(
            lambda s: pytest.fail("the DD rank read PropSEM.mesh")))
        out = SEMModelParallel(p, StripTopology(2))(torch.sin(t), src_node,
                                                    rec, [vp])
        out.pow(2).sum().backward()
    assert vp.grad is not None and float(vp.grad.abs().max()) > 0.0
    assert p._mesh is None, "the DD forward/backward materialised the global mesh"
    assert built, "no strip mesh was built at all"
    assert all(nex == NEX // 2 and bare for nex, bare in built), (
        f"a mesh other than a connectivity-only strip was built: {built}")
    m = p.mesh                                     # the fused path still can
    assert m is p.mesh and m.ibool.shape[0] == NEX ** 3


def test_out_of_range_source_raises_on_every_strip():
    """The n_global unit vector raised IndexError on every rank; a per-strip
    slice must not raise only on the strip that owns the column and hand the
    others a silent all-zero pattern (under dist those survivors would block
    on the first interface exchange). And a negative in-range node keeps the
    old Python-index wrap."""
    from sweep.sem.dd import _CompactEq3D, _strip_source_pattern
    p, vp = _build()
    src = _CompactEq3D.build(p, vp)
    ng = src.mesh.n_global
    n = NEX * NO + 1
    rec = torch.tensor([ng // 3], device="cuda")
    dd = SEMModelParallel(p, StripTopology(4))
    for bad in (ng, ng + 3, -ng - 1):
        with pytest.raises(IndexError, match="out of bounds"):
            dd._parts3d(src, bad, rec)
    # rank-independent: each strip's OWN slice refuses it, including strips
    # whose x-range does not hold the column
    for r in range(4):
        ex0, ex1 = StripTopology(4, r).local_elems(NEX)
        with pytest.raises(IndexError):
            _strip_source_pattern(ng + 3, n, (ex1 - ex0) * NO + 1, ex0 * NO,
                                  ex1 * NO, n * n, torch.device("cuda"))
    a = dd._parts3d(src, -1, rec)[0]
    b = dd._parts3d(src, ng - 1, rec)[0]
    for qa, qb in zip(a, b):
        assert torch.equal(qa.c["src_pattern"], qb.c["src_pattern"])


def test_forward_actually_uses_the_compact_source():
    """The whole point is that the global equation is NOT built. If a later
    edit quietly reroutes forward back through `_resolve_model_batch`, every
    cross-path comparison in the DD suites collapses back to same-vs-same and
    the memory saving silently evaporates -- this is the tripwire."""
    import sweep.sem.dd as DD
    p, vp = _build()
    n = NEX * NO + 1
    src_node = (2 * NO * (n * n)) + (n // 2)
    rec = torch.tensor([src_node + 7], device="cuda")
    t = torch.arange(10, dtype=torch.float32, device="cuda")
    wav = torch.sin(t)
    calls = []
    orig = DD._CompactEq3D.build
    DD._CompactEq3D.build = classmethod(
        lambda cls, prop, m: calls.append(1) or orig.__func__(cls, prop, m))
    try:
        SEMModelParallel(p, StripTopology(2))(wav, src_node, rec, [vp])
    finally:
        DD._CompactEq3D.build = orig
    assert calls, "forward built the global equation instead of the compact source"

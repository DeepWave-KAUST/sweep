"""The 3-D acoustic C-PML memory fields on their axis SLABS (sweep/sem/slab.py).

What the change rests on, and how each piece is held here:

1. DECISIVE, bitwise, on the FULL layout alone: the memory fields psi_d /
   zeta_d are exactly +0.0 at every node off the index slab of axis d, at
   every step -- checked on the stored bytes of the save-all tapes (which
   are the fields themselves), with a guard that the wave actually reached
   the slab (a run in which the layer never saw the wave proves nothing).
   Flat and deformed meshes, single domain and DD strips.
2. The slot arithmetic: the Python reference of the device formula is a
   bijection between the slab's nodes and [0, slab length), -1 elsewhere,
   the support {b_d != 0} lies on the slab, the ring and Dirichlet index
   sets land on it, and the tables invert each other.
3. Both layouts in one process (the `_PSI_SLAB` module switch): records
   and gradients agree to the forward's run-to-run floor. There is no
   bitwise criterion on this path (the assembly is a float atomicAdd), so
   the bar is ten times the measured floor plus a cosine gap.
4. The account: the fifteen forward arrays shrink by the slab fraction.
5. The fallbacks: a save-all tape forces the full layout; rim mode has
   empty rings but the slabs still carry the fields; a middle strip with no
   x slab runs with empty psi_x / zeta_x buffers.
"""
from __future__ import annotations

import pytest
import torch

import sweep.sem as S
import sweep.sem.prop_c as PC
import sweep.sem.dd as DD
from sweep.propagator import BoundarySaving
from sweep.sem.equations import AcousticSEM3D, AcousticSEM3DDeformed
from sweep.sem.slab import (AXES, cpml_slab_tables, node_grid_dims,
                            slab_slot, slab_length)

try:
    import sweep._C as _C
    _HAS_C = hasattr(_C, "sem3d_acoustic_cpml_dd_step")
except Exception:                                   # pragma: no cover
    _HAS_C = False

needs_c = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_C),
                             reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, DT, F0 = 4, 40.0, 6e-4, 15.0
FIELDS = ("u", "psi_x", "psi_y", "psi_z", "zeta_x", "zeta_y", "zeta_z")
AXIS = {"psi_x": "x", "psi_y": "y", "psi_z": "z",
        "zeta_x": "x", "zeta_y": "y", "zeta_z": "z"}
DIM = {"x": 2, "y": 1, "z": 0}
FWD_FIELDS = ("psi_x", "psi_y", "psi_z", "zeta_x", "zeta_y", "zeta_z",
              "psi_x_n", "psi_y_n", "psi_z_n", "zeta_x_n", "zeta_y_n", "zeta_z_n",
              "apsi_x", "apsi_y", "apsi_z")


def _hill(x, y):
    return 120.0 * torch.exp(-(((x - 1300.0) / 700.0) ** 2
                               + ((y - 1300.0) / 700.0) ** 2))


def _hill_rows(n):
    """``_hill`` on the model columns: the surface row of each, (n, n)."""
    a = torch.arange(n, dtype=torch.float64) * DH
    Y, X = torch.meshgrid(a, a, indexing="ij")
    return (_hill(X, Y) / DH).numpy()


def _prop(nex, deformed, nt, abcn=4, mode="full", memory=None):
    """A model of ``nex`` elements a side; ``abcn`` ELEMENTS of absorbing
    layer outside it."""
    n = nex * NO + 1
    return S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D,
                     shape=(n, n, n), dh=DH, dt=DT, nt=nt, n_order=NO,
                     free_surface=("top",), abcn=abcn * NO, f_dominant=F0,
                     dev="cuda", impl="c", memory=memory, bs_band_mode=mode,
                     **(dict(topography=_hill_rows(n)) if deformed else {}))


def _gid(p, ix, iy, iz):
    """Model node ``(ix, iy, iz)`` -> its GLOBAL id, on the row-major node
    grid of the model AND its absorbing layer."""
    _, nyg, nxg = p._grid_shape()
    z0, y0, x0 = p._pad0
    return ((iz + z0) * nyg + iy + y0) * nxg + ix + x0


def _vp(n, seed=0):
    torch.manual_seed(seed)
    return 2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")


def _wav(nt):
    t = torch.arange(nt, dtype=torch.float32, device="cuda") * DT - 1.2 / F0
    a = (3.1415926 * F0 * t) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _corner_src(n):
    """Two nodes inside the inner edge of the x, y and bottom layers -- the
    model's own edges, the layers being outside it."""
    return 2, 2, n - 1 - 2


def _slabs_from_coefs(cp, dims):
    """``dims``: the node grid ``(nz, ny, nx)`` of the model and its layer."""
    out = {}
    for ax in AXES:
        b = getattr(cp, f"b_{ax}").reshape(dims)
        other = tuple(d for d in (0, 1, 2) if d != DIM[ax])
        planes = (b != 0).any(dim=other)
        shape = [1, 1, 1]
        shape[DIM[ax]] = dims[DIM[ax]]
        out[ax] = planes.reshape(shape).expand(*dims).reshape(-1)
    return out


def _zero_off_slab(tapes, slab, gidx=None):
    """(violations, signal) over the memory-field tapes of one propagator or
    strip: non-(+0.0) entries off the slab -- the int32 view must be 0, so
    -0.0 counts too -- and the on-slab amplitudes for the vacuity guard."""
    bad, sig = 0, {}
    u_amp = float(tapes["u"].abs().max())
    for nm, t in tapes.items():
        if nm not in AXIS:
            continue
        m = slab[AXIS[nm]]
        if gidx is not None:
            m = m[gidx]
        m = m.to(t.device)
        bad += int((t[:, ~m].view(torch.int32) != 0).sum())
        sig[nm] = (float(t[:, m].abs().max()) / max(u_amp, 1e-300),
                   float(tapes["u"][:, m].abs().max()) / max(u_amp, 1e-300))
    return bad, sig


def _not_vacuous(sig_list):
    """Every field rang on its slab in at least one strip, and the wave
    itself reached every slab."""
    for nm in AXIS:
        amp = max(s[nm][0] for s in sig_list)
        u_on = max(s[nm][1] for s in sig_list)
        assert u_on > 1e-2, f"the wave never reached {nm}'s slab (|u| {u_on:.1e} of max)"
        assert amp > 1e-8, f"{nm} never rang on its slab ({amp:.1e} of |u|max)"


# --------------------------------------------------------------------------- #
# 1. decisive: the full layout stores exactly +0.0 off the slabs
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_full_layout_memory_fields_are_plus_zero_off_their_slab(deformed):
    nex, nt, abcn = 12, 260, 3
    n = nex * NO + 1
    p = _prop(nex, deformed, nt, abcn)
    vp = _vp(n).requires_grad_(True)
    sx, sy, sz = _corner_src(n)
    src = torch.tensor([[sx, sy, sz]], device="cuda")
    rec = torch.tensor([[[sx + 3 * i, sy, sz] for i in range(6)]], device="cuda")
    got = {}
    real = DD._SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        got["saves"] = {f: ctx.carry[0]["save_" + f] for f in FIELDS}   # one strip
        got["slab"] = int(ctx.slab)
        return out

    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    old_wt = PC._WSUM_TAPE
    PC._WSUM_TAPE = False               # the seven state tapes ARE the check here
    try:
        p.forward(_wav(nt), src, rec, [vp])           # save-all tape mode
    finally:
        PC._WSUM_TAPE = old_wt
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real)
    assert got["slab"] == 0, "the full state tapes must run the full layout"
    eq, _ = p._resolve_model_batch([vp.detach()])
    for ax in AXES:                                    # the coefficient facts
        b, a = getattr(eq.cpml, f"b_{ax}"), getattr(eq.cpml, f"a_{ax}")
        off = b == 0
        assert bool((b[off].view(torch.int32) == 0).all()), "b is not +0.0 off the layer"
        assert bool((a[off] == 1.0).all()), "a is not 1.0 off the layer"
    bad, sig = _zero_off_slab(got["saves"], _slabs_from_coefs(eq.cpml, p._grid_shape()))
    _not_vacuous([sig])
    assert bad == 0, f"{bad} tape entries off the slabs are not +0.0"


@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_full_layout_strips_store_plus_zero_off_their_slab(deformed):
    nex, nt, abcn, px = 12, 260, 3, 2
    n = nex * NO + 1
    p = _prop(nex, deformed, nt, abcn)
    vp = _vp(n).requires_grad_(True)
    sx, sy, sz = _corner_src(n)
    src = _gid(p, sx, sy, sz)
    rec = torch.tensor([src + 3 * i for i in range(6)], device="cuda")
    got = {}
    real = DD._SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        got["saves"] = [{f: cr["save_" + f] for f in FIELDS} for cr in ctx.carry]
        got["gidx"] = [pp.gidx.to("cuda") for pp in ctx.parts]
        got["slab"] = int(ctx.slab)
        return out

    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    old_wt = PC._WSUM_TAPE
    PC._WSUM_TAPE = False               # the seven state tapes ARE the check here
    try:
        DD.SEMModelParallel(p, DD.StripTopology(px))(_wav(nt), src, rec, [vp])
    finally:
        PC._WSUM_TAPE = old_wt
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real)
    assert got["slab"] == 0
    eq, _ = p._resolve_model_batch([vp.detach()])
    slab = _slabs_from_coefs(eq.cpml, p._grid_shape())
    bad, sigs = 0, []
    for tapes, g in zip(got["saves"], got["gidx"]):
        b, s = _zero_off_slab(tapes, slab, gidx=g)
        bad += b
        sigs.append(s)
    _not_vacuous(sigs)
    assert bad == 0, f"{bad} strip tape entries off the slabs are not +0.0"


# --------------------------------------------------------------------------- #
# 2. the slot arithmetic
# --------------------------------------------------------------------------- #
def _consts(nex, deformed, abcn=3, mode="full"):
    n = nex * NO + 1
    p = _prop(nex, deformed, 8, abcn, mode)
    vp = _vp(n)
    eq, _ = p._resolve_model_batch([vp])
    c = p._acoustic3d_consts() if hasattr(p, "_acoustic3d_consts") else None
    if c is None:                      # PropSEM wraps the C propagator
        prop = p._props[0] if hasattr(p, "_props") else p
        c = prop._acoustic3d_consts()
    return p, eq, c


@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_slot_map_is_a_bijection_onto_the_slab(deformed):
    nex = 8
    n = nex * NO + 1
    p = _prop(nex, deformed, 8, 2)
    vp = _vp(n)
    eq, _ = p._resolve_model_batch([vp])
    cp = eq.cpml
    dims = p._grid_shape()[::-1]          # (nx, ny, nz): the model and its layer
    dev = cp.b_x.device
    c = {"ibool": eq.mesh.ibool.reshape(eq.mesh.n_elements, -1).to(torch.int32),
         "n_global": eq.mesh.n_global, "ngll": NO + 1,
         "dirichlet_idx": eq.dirichlet_idx.to(dev),
         "b_x": cp.b_x, "b_y": cp.b_y, "b_z": cp.b_z,
         "dbx_dx": cp.dbx_dx, "dby_dy": cp.dby_dy, "dbz_dz": cp.dbz_dz}
    band = torch.arange(eq.mesh.n_global, device=dev)
    for ax in AXES:                                    # every node "rings"
        c[f"band_idx_{ax}"] = band[getattr(cp, f"b_{ax}") != 0]
    t = cpml_slab_tables(c)
    assert (t["nx"], t["ny"], t["nz"]) == dims
    g = torch.arange(eq.mesh.n_global, device=dev)
    for ax in AXES:
        tab, ns = t[f"tab_{ax}"], t[f"ns{ax}"]
        s = slab_slot(ax, g, dims, tab)
        n_slab = slab_length(ax, dims, ns)
        assert t[f"slab_n_{ax}"] == n_slab
        on = s >= 0
        # bijection: the on-slab slots are exactly a permutation of range(n_slab)
        assert int(on.sum()) == n_slab
        assert torch.equal(torch.sort(s[on]).values,
                           torch.arange(n_slab, device=dev))
        # the support lies on the slab. The slab is b's planes UNION db's,
        # and db reaches one plane further at each band end, so on a flat mesh
        # it equals that union exactly -- not b's support alone.
        sup = getattr(cp, f"b_{ax}") != 0
        assert bool(on[sup].all())
        if not deformed:
            wide = sup | (getattr(cp, f"db{ax}_d{ax}") != 0)
            assert torch.equal(on, wide)
            assert int(wide.sum()) > int(sup.sum()), "db adds no plane here"
        # the tables invert each other
        planes = t[f"planes_{ax}"]
        assert torch.equal(tab[planes.long()], torch.arange(ns, dtype=torch.int32, device=dev))
        # ring and Dirichlet index sets land on the slab, injectively
        ring = t[f"ring_slot_{ax}"]
        assert bool((ring >= 0).all()) and ring.unique().numel() == ring.numel()
        assert torch.equal(ring, s[c[f"band_idx_{ax}"]])
        ds = t[f"dir_slot_{ax}"]
        assert ds.unique().numel() == ds.numel()
        want = s[c["dirichlet_idx"]]
        assert torch.equal(ds, want[want >= 0])


def test_node_grid_dims_rejects_an_unstructured_numbering():
    ngll, nx, ny, nz = 3, 7, 5, 4
    ib = torch.zeros(2, ngll, ngll, ngll, dtype=torch.int32)
    k = torch.arange(ngll)
    ib[0] = k.reshape(1, 1, -1) + (k * nx).reshape(1, -1, 1) + (k * nx * ny).reshape(-1, 1, 1)
    ib[1] = ib[0] + 2
    assert node_grid_dims(ib.reshape(2, -1), ngll, nx * ny * nz) == (nx, ny, nz)
    ib[1, 1, 1, 1] += 1                              # one node out of place
    with pytest.raises(ValueError):
        node_grid_dims(ib.reshape(2, -1), ngll, nx * ny * nz)


# --------------------------------------------------------------------------- #
# 3. both layouts in one process
# --------------------------------------------------------------------------- #
def _rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-300))


def _gap(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return abs(1.0 - float(a @ b / (a.norm() * b.norm())))


# Ten times the measured run-to-run floor of the fused 3-D path (record
# 1.1e-7, d_vp 7.9e-7 relative) -- the assembly is a float atomicAdd,
# so two runs of ONE layout already differ by this much. A misaddressed slot
# would put O(1) here.
REL_BAR, GAP_BAR = 1e-5, 1e-10


def _fused_run(deformed, slab, bs, store=None, mode="full", nex=8, nt=40, abcn=2):
    n = nex * NO + 1
    p = _prop(nex, deformed, nt, abcn, mode,
              memory=BoundarySaving(storage_dtype=store or "fp32") if bs else None)
    vp = _vp(n).requires_grad_(bs)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0 - 6 + 2 * i, c0, z0] for i in range(6)]], device="cuda")
    seen = {}
    real = DD._SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, *a):
        out = real(ctx, *a)
        seen["slab"] = int(ctx.slab)
        return out

    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    old = PC._PSI_SLAB
    PC._PSI_SLAB = slab
    try:
        out = p.forward(_wav(nt), src, rec, [vp])
        if bs:
            out.pow(2).sum().backward()
    finally:
        PC._PSI_SLAB = old
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real)
    if bs:
        assert seen["slab"] == int(slab)
    return out.detach(), (vp.grad.detach() if bs else None)


def _check(full, slab):
    rf, gf = full
    rs, gs = slab
    assert torch.isfinite(rs).all()
    assert _rel(rf, rs) <= REL_BAR and _gap(rf, rs) <= GAP_BAR, \
        f"record: rel {_rel(rf, rs):.3e} gap {_gap(rf, rs):.3e}"
    if gf is not None:
        assert torch.isfinite(gs).all()
        assert _rel(gf, gs) <= REL_BAR and _gap(gf, gs) <= GAP_BAR, \
            f"d_vp: rel {_rel(gf, gs):.3e} gap {_gap(gf, gs):.3e}"


@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_fused_forward_only_matches_the_full_layout(deformed):
    _check(_fused_run(deformed, False, False), _fused_run(deformed, True, False))


@needs_c
@pytest.mark.parametrize("store", [None, "bf16", "int8"])
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_fused_boundary_saving_matches_the_full_layout(deformed, store):
    _check(_fused_run(deformed, False, True, store), _fused_run(deformed, True, True, store))


@needs_c
def test_rim_mode_runs_with_empty_memory_rings():
    # rim: the interface line carries no b, so band_idx_d and ring_slot_d are
    # empty; the fields still live on their slabs and the backward reconstructs
    full = _fused_run(False, False, True, None, "rim", nex=10, abcn=2)
    slab = _fused_run(False, True, True, None, "rim", nex=10, abcn=2)
    _check(full, slab)


def _dd_run(deformed, slab, bs, px, nex=8, nt=40, abcn=2):
    n = nex * NO + 1
    p = _prop(nex, deformed, nt, abcn, memory=BoundarySaving() if bs else None)
    vp = _vp(n).requires_grad_(bs)
    c0, z0 = n // 2, 2 * NO
    src = _gid(p, c0, c0, z0)
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    seen = {}
    real = DD._SEMAcoustic3DCpmlDDC.forward

    def spy(ctx, wavelet, parts, exchange, bs_, *mats):
        out = real(ctx, wavelet, parts, exchange, bs_, *mats)
        seen["slab"] = int(ctx.slab)
        seen["nsx"] = [int(pp.c["nsx"]) for pp in parts]
        seen["lens"] = [[int(getattr(i, nm).numel()) for nm in ("psi_x", "psi_y", "psi_z")]
                        for i in seen.get("inps", [])]
        return out

    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy)
    old = PC._PSI_SLAB
    PC._PSI_SLAB = slab
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(px))(_wav(nt), src, rec, [vp])
        if bs:
            out.pow(2).sum().backward()
    finally:
        PC._PSI_SLAB = old
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real)
    if bs:
        assert seen["slab"] == int(slab)
    return (out.detach(), vp.grad.detach() if bs else None), seen


@needs_c
@pytest.mark.parametrize("px", [2, 3])
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_dd_strips_match_the_full_layout(deformed, px):
    nex = 8         # uniform tiles: px | 8 + 2 * 2, the model's and layer's elements
    full, _ = _dd_run(deformed, False, True, px, nex=nex)
    slab, seen = _dd_run(deformed, True, True, px, nex=nex)
    _check(full, slab)
    if px == 3:
        # the middle strip touches no x face: no x slab at all
        assert seen["nsx"][1] == 0 and seen["nsx"][0] > 0 and seen["nsx"][2] > 0


@needs_c
def test_dd_forward_only_matches_the_full_layout():
    full, _ = _dd_run(False, False, False, 2)
    slab, _ = _dd_run(False, True, False, 2)
    _check(full, slab)


# --------------------------------------------------------------------------- #
# 4. the account, 5. the fallbacks
# --------------------------------------------------------------------------- #
@needs_c
def test_forward_arrays_shrink_by_the_slab_fraction():
    nex, abcn = 12, 2
    n = nex * NO + 1
    p = _prop(nex, False, 12, abcn)
    vp = _vp(n)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0, c0, z0]]], device="cuda")
    seen = {}
    real = _C.sem3d_acoustic_cpml_dd_step

    def spy(inp, *rest):
        seen["lens"] = {nm: int(getattr(inp, nm).numel()) for nm in FWD_FIELDS}
        seen["slab"] = int(inp.psi_slab)
        seen["ng"] = int(inp.n_global)
        seen["ns"] = (inp.nsx, inp.nsy, inp.nsz)
        return real(inp, *rest)

    _C.sem3d_acoustic_cpml_dd_step = spy
    try:
        p.forward(_wav(12), src, rec, [vp])
    finally:
        _C.sem3d_acoustic_cpml_dd_step = real
    assert seen["slab"] == 1
    ng = seen["ng"]
    total = sum(seen["lens"].values())
    # each axis: b spans 2*abcn*NO planes and db reaches ONE further at each
    # end of the band, and the storage slab is their union -> 18 of 65 for x
    # and y, 9 of 57 for z (bottom only, the top is the free surface) -- planes
    # of the node grid of the model AND its absorbing layer
    nzg, nyg, nxg = p._grid_shape()
    want = (5 * (2 * abcn * NO + 2) * nzg * (nyg + nxg)
            + 5 * (abcn * NO + 1) * nyg * nxg)
    assert total == want, (seen["lens"], want)
    assert total < 0.5 * 15 * ng
    assert seen["ns"] == (2 * abcn * NO + 2, 2 * abcn * NO + 2, abcn * NO + 1)


@needs_c
def test_the_full_state_tapes_force_the_full_layout():
    """With the seven state tapes (the pre-w_sum-tape save-all mode, kept
    behind prop_c._WSUM_TAPE = False) the memory fields must be full-length:
    the tapes copy whole arrays. With the w_sum tape the slabs apply."""
    nex = 8
    n = nex * NO + 1
    p = _prop(nex, False, 8, 2)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0, c0, z0]]], device="cuda")
    nzg, nyg, nxg = p._grid_shape()
    ng = nzg * nyg * nxg                  # the model's nodes AND its layer's
    seen = {}
    real = _C.sem3d_acoustic_cpml_dd_step

    def spy(inp, *rest):
        seen["slab"] = int(inp.psi_slab)
        seen["len"] = int(inp.psi_x.numel())
        seen["tape"] = int(inp.save_psi_x.numel())
        seen["wtape"] = int(inp.save_w_sum.numel())
        return real(inp, *rest)

    _C.sem3d_acoustic_cpml_dd_step = spy
    old_wt = PC._WSUM_TAPE
    try:
        PC._WSUM_TAPE = False
        p.forward(_wav(8), src, rec, [vp])
        assert seen["tape"] > 0 and seen["wtape"] == 0
        assert seen["slab"] == 0 and seen["len"] == ng
        PC._WSUM_TAPE = True
        p.forward(_wav(8), src, rec, [vp])
        assert seen["tape"] == 0 and seen["wtape"] == 8 * ng
        assert seen["slab"] == 1 and seen["len"] < ng
    finally:
        PC._WSUM_TAPE = old_wt
        _C.sem3d_acoustic_cpml_dd_step = real


@needs_c
def test_the_c_driver_refuses_a_slab_with_a_tape():
    """The guard behind the fallback above: if Python ever hands the driver
    a tape AND the slab layout, the C side must refuse, not copy garbage."""
    nex = 8
    n = nex * NO + 1
    p = _prop(nex, False, 8, 2)
    vp = _vp(n).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0, c0, z0]]], device="cuda")
    real = _C.sem3d_acoustic_cpml_dd_step

    def spy(inp, *rest):
        inp.psi_slab = 1                 # lie: slab flag with full arrays + tape
        return real(inp, *rest)

    _C.sem3d_acoustic_cpml_dd_step = spy
    old_wt = PC._WSUM_TAPE
    PC._WSUM_TAPE = False                # the seven full-length state tapes
    try:
        with pytest.raises(RuntimeError, match="psi_slab"):
            p.forward(_wav(8), src, rec, [vp])
    finally:
        PC._WSUM_TAPE = old_wt
        _C.sem3d_acoustic_cpml_dd_step = real

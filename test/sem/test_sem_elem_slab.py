"""The assembly's per-element a*psi skip.

``psi_d`` starts at zero and grows only through ``b_d``, so an element whose
every node has ``b_d == 0`` carries ``a_d*psi_d == +0.0`` on all of them and
its whole contribution to the ``ga_d`` half of the assembly is exactly +0.0.
``slab.cpml_slab_tables`` therefore hands the kernels a per-element bitmask
(``elem_slab``, bit d per axis) and both 3-D assembly kernels skip that
axis's slab read, D-loop terms and atomicAdd where the bit is clear. At
production sizes that is most of the mesh.

What is held here:
1. The derivation: the flag equals the direct per-node test
   ``any(b_d != 0 over the element's nodes)``, on an asymmetric fixture and
   on every real strip of a DD decomposition.
2. The premise: where a flag is clear, ``a_d*psi_d`` really is +0.0 on every
   node of that element THROUGHOUT a run (not just at t=0).
3. The consequence: record and gradient are unchanged (bitwise where the
   assembly allows it, else inside the assembly's own run-to-run floor).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
import sweep.sem.dd as DD
from sweep.sem.slab import cpml_slab_tables
from sweep.sem.equations import AcousticSEM3D, AcousticSEM3DDeformed

try:
    import sweep._C as _C
    _HAS_C = hasattr(_C, "sem3d_acoustic_cpml_dd_asm")
except Exception:                                   # pragma: no cover
    _HAS_C = False

needs_c = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_C),
                             reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, DT, F0 = 4, 40.0, 6e-4, 15.0


def _hill(x, y):
    return 120.0 * torch.exp(-(((x - 1300.0) / 700.0) ** 2
                               + ((y - 1300.0) / 700.0) ** 2))


def _hill_rows(n):
    """``_hill`` on the model columns: the surface row of each, (n, n)."""
    a = torch.arange(n, dtype=torch.float64) * DH
    Y, X = torch.meshgrid(a, a, indexing="ij")
    return (_hill(X, Y) / DH).numpy()


def _prop(nex, nt, deformed=False, abcn=2):
    n = nex * NO + 1
    return S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D,
                     shape=(n, n, n), dh=DH, dt=DT, nt=nt, n_order=NO,
                     free_surface=("top",),
                     abcn=abcn * NO, f_dominant=F0, dev="cuda", impl="c",
                     **(dict(topography=_hill_rows(n)) if deformed else {}))


def _vp(n, seed=0):
    torch.manual_seed(seed)
    return (2500.0 + 200.0 * torch.rand(n, n, n, device="cuda"))


def _wav(nt):
    t = torch.arange(nt, dtype=torch.float32, device="cuda") * DT - 1.2 / F0
    a = (3.1415926 * F0 * t) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _geometry(p):
    """GLOBAL node ids, on the node grid of the model AND its absorbing
    layer: the source mid-model two elements down, six receivers after it."""
    nz, ny, nx = p._grid_shape()
    c0y, c0x, z0 = ny // 2, nx // 2, 2 * NO
    src = (z0 * ny + c0y) * nx + c0x
    return src, torch.tensor([src + 2 * i for i in range(6)], device="cuda")


# --------------------------------------------------------------------------- #
# 1. the derivation: the bitmask == the direct per-node test
# --------------------------------------------------------------------------- #
def test_flags_match_the_direct_per_node_test_on_an_asymmetric_fixture():
    """CPU, no propagator: three different asymmetric profiles, so a flag that
    silently used one axis's answer for another would show."""
    no, nex, ney, nez = 4, 6, 5, 4
    nx, ny, nz = nex * no + 1, ney * no + 1, nez * no + 1
    k = torch.arange(no + 1)
    ez, ey, ex = torch.meshgrid(torch.arange(nez), torch.arange(ney),
                                torch.arange(nex), indexing="ij")
    gz = (ez[..., None] * no + k)[..., :, None, None]
    gy = (ey[..., None] * no + k)[..., None, :, None]
    gx = (ex[..., None] * no + k)[..., None, None, :]
    ib = ((gz * ny + gy) * nx + gx).reshape(nez * ney * nex, -1).to(torch.int32)

    def prof(axis, lo, hi):
        v = torch.zeros(nz, ny, nx)
        idx = [slice(None)] * 3
        idx[axis] = slice(0, lo); v[tuple(idx)] = 1.0
        idx[axis] = slice(hi, None); v[tuple(idx)] = 1.0
        return v.reshape(-1)

    empty = torch.zeros(0, dtype=torch.int64)
    c = {"ibool": ib, "ngll": no + 1, "n_global": nx * ny * nz,
         "b_x": prof(2, 5, nx - 5), "b_y": prof(1, 9, ny - 3),
         "b_z": prof(0, 4, nz), "dirichlet_idx": empty,
         "band_idx_x": empty, "band_idx_y": empty, "band_idx_z": empty,
         # the storage slab is b's planes union db's; the FLAGS under test are
         # b's alone, so a zero db keeps this fixture about the flags
         "dbx_dx": torch.zeros(nx * ny * nz),
         "dby_dy": torch.zeros(nx * ny * nz),
         "dbz_dz": torch.zeros(nx * ny * nz)}
    flags = cpml_slab_tables(c)["elem_slab"]
    for bit, ax in enumerate("xyz"):
        want = (c[f"b_{ax}"][ib.long()] != 0).any(1)
        got = (flags & (1 << bit)) != 0
        assert torch.equal(want, got), f"axis {ax}"
        assert 0.0 < float(got.float().mean()) < 1.0, f"axis {ax} is degenerate"


@needs_c
@pytest.mark.parametrize("px", [1, 2])
def test_flags_match_the_direct_per_node_test_on_every_strip(px):
    nex, nt = 8, 8
    n = nex * NO + 1
    p = _prop(nex, nt)
    src, rec = _geometry(p)
    dd = DD.SEMModelParallel(p, DD.StripTopology(px))
    eq = DD._CompactEq3D.build(p, _vp(n))
    parts, _ = dd._parts3d(eq, int(src), rec)
    for pp in parts:
        c = pp.c
        ib = c["ibool"].reshape(c["n_elements"], -1).long()
        for bit, ax in enumerate("xyz"):
            want = (c[f"b_{ax}"][ib] != 0).any(1)
            got = (c["elem_slab"] & (1 << bit)) != 0
            assert torch.equal(want, got), f"px={px} axis {ax}"


# --------------------------------------------------------------------------- #
# 2. the premise, during a run: a cleared flag really means a*psi == +0.0
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_a_cleared_flag_means_plus_zero_apsi_at_every_step(deformed):
    """Read a*psi_d off the forward's own buffers at every step (the FULL
    layout, so the array is addressable everywhere) and check the bytes: on
    the nodes of an element whose bit is clear it must be exactly +0.0, and
    the wave must have reached the layer so the check is not vacuous."""
    import sweep.sem.prop_c as PC
    # a corner source two nodes inside the x/y/bottom layers and a record long
    # enough for the Ricker to have peaked (t0 = 1.2/F0): otherwise a*psi is
    # 1e-19 and the check passes vacuously
    nex, nt, abcn = 12, 300, 3
    n = nex * NO + 1
    p = _prop(nex, nt, deformed, abcn=abcn)
    nzg, nyg, nxg = p._grid_shape()    # global node grid: model + layer
    L = abcn * NO
    sx, sy, sz = L + 2, L + 2, nzg - 1 - L - 2
    src = (sz * nyg + sy) * nxg + sx
    rec = torch.tensor([src + 3 * i for i in range(6)], device="cuda")
    got = {"bad": 0, "amp": 0.0, "u": 0.0, "seen": 0}
    real = _C.sem3d_acoustic_cpml_dd_step

    def spy(inp, k):
        out = real(inp, k)
        if k % 37 == 0:
            ib = inp.ibool.reshape(inp.n_elements, -1).long()
            for bit, ax in enumerate("xyz"):
                ap = getattr(inp, f"apsi_{ax}")
                off = (inp.elem_slab & (1 << bit)) == 0
                if not bool(off.any()):
                    continue
                v = ap[ib[off].reshape(-1)]
                got["bad"] += int((v.view(torch.int32) != 0).sum())
                got["amp"] = max(got["amp"], float(ap.abs().max()))
                got["u"] = max(got["u"], float(inp.u.abs().max()))
                got["seen"] += 1
        return out

    old_slab, old_tape = PC._PSI_SLAB, PC._WSUM_TAPE
    PC._PSI_SLAB, PC._WSUM_TAPE = False, False    # the full layout addresses every node
    _C.sem3d_acoustic_cpml_dd_step = spy
    try:
        DD.SEMModelParallel(p, DD.StripTopology(1))(
            _wav(nt), src, rec, [_vp(n).requires_grad_(True)])
    finally:
        PC._PSI_SLAB, PC._WSUM_TAPE = old_slab, old_tape
        _C.sem3d_acoustic_cpml_dd_step = real
    assert got["seen"] > 0, "no element had a cleared flag: the check is vacuous"
    # relative: u is a potential and is small in absolute terms (1e-10 here),
    # so the guard is that the layer really rang, not an absolute amplitude
    assert got["amp"] > 1e-6 * got["u"], (
        f"a*psi never rang: {got['amp']:.1e} against |u| {got['u']:.1e}")
    assert got["bad"] == 0, f"{got['bad']} nodes of unflagged elements are not +0.0"


# --------------------------------------------------------------------------- #
# 3. the consequence: the skip does not move the answer
# --------------------------------------------------------------------------- #
def _run(p, n, nt, px, flags):
    vp = _vp(n).requires_grad_(True)
    src, rec = _geometry(p)
    dd = DD.SEMModelParallel(p, DD.StripTopology(px))
    real = dd._parts3d

    def blank(*a, **kw):          # every element flagged = the pre-flag path
        parts, mats = real(*a, **kw)
        for pp in parts:
            pp.c["elem_slab"] = torch.full_like(pp.c["elem_slab"], 7)
        return parts, mats

    if not flags:
        dd._parts3d = blank
    out = dd(_wav(nt), src, rec, [vp])
    out.pow(2).sum().backward()
    return out.detach(), vp.grad.clone()


@needs_c
@pytest.mark.parametrize("px", [1, 2])
def test_the_skip_does_not_move_the_record_or_the_gradient(px):
    """The skipped contributions are exactly +0.0, so the only difference is
    the assembly's own atomicAdd ordering -- the same floor two runs of the
    SAME path show (measured here, both ways round)."""
    nex, nt = 8, 60
    n = nex * NO + 1
    p = _prop(nex, nt)
    s_on, g_on = _run(p, n, nt, px, True)
    s_off, g_off = _run(p, n, nt, px, False)
    s_on2, g_on2 = _run(p, n, nt, px, True)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
    floor_s, floor_g = rel(s_on2, s_on), rel(g_on2, g_on)
    assert rel(s_off, s_on) <= max(10 * floor_s, 1e-6), (
        f"record {rel(s_off, s_on):.2e} vs the A/A floor {floor_s:.2e}")
    assert rel(g_off, g_on) <= max(10 * floor_g, 1e-5), (
        f"d_vp {rel(g_off, g_on):.2e} vs the A/A floor {floor_g:.2e}")


# --------------------------------------------------------------------------- #
# 4. the transpose's half: under the slab layout lap_d off slab d is unread
# --------------------------------------------------------------------------- #
@needs_c
def test_lap_lives_on_its_slab_and_the_gradient_does_not_move():
    """``combine`` writes the adjoint carries on their slab alone, so the
    transpose need neither produce nor STORE lap_d off slab d.

    Two statements. The allocation: each lap_d is exactly its axis slab long
    under the production layout, and strictly shorter than the node grid --
    that is where the memset and the ``combine`` reads were saved. And the
    consequence: the gradient the slab layout produces sits within the
    assembly's own A/A floor of the full-length one, so nothing that was
    dropped was ever read. (The earlier form of this test poisoned lap_d off
    the slab with NaN; the array no longer HAS an off-slab region, which is
    the whole point, so the statement moved to the layouts' agreement.)
    """
    import sweep.sem.prop_c as PC
    nex, nt = 8, 60
    n = nex * NO + 1
    src, rec = _geometry(_prop(nex, nt))
    real = _C.sem3d_acoustic_cpml_dd_bwd_mid
    sizes = {}

    def measure(b, k):
        sizes["n_global"] = int(b.n_global)
        for ax in "xyz":
            sizes[ax] = (int(getattr(b, f"lap{ax}").numel()), _slab_len(b, ax))
        return real(b, k)

    def run(slab):
        vp = _vp(n).requires_grad_(True)
        old = PC._PSI_SLAB
        PC._PSI_SLAB = slab
        _C.sem3d_acoustic_cpml_dd_bwd_mid = measure if slab else real
        try:
            out = DD.SEMModelParallel(_prop(nex, nt), DD.StripTopology(1))(
                _wav(nt), src, rec, [vp])
            out.pow(2).sum().backward()
        finally:
            PC._PSI_SLAB = old
            _C.sem3d_acoustic_cpml_dd_bwd_mid = real
        return out.detach(), vp.grad.clone()

    s_on, g_on = run(True)
    s_off, g_off = run(False)
    s_on2, g_on2 = run(True)

    for ax in "xyz":
        got, want = sizes[ax]
        assert got == want, f"lap{ax} is {got} long, its slab is {want}"
        assert got < sizes["n_global"], f"lap{ax} is the whole node grid"

    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
    floor_s, floor_g = rel(s_on2, s_on), rel(g_on2, g_on)
    assert rel(s_off, s_on) <= max(10 * floor_s, 1e-6), (
        f"record {rel(s_off, s_on):.2e} vs the A/A floor {floor_s:.2e}")
    assert rel(g_off, g_on) <= max(10 * floor_g, 1e-5), (
        f"d_vp {rel(g_off, g_on):.2e} vs the A/A floor {floor_g:.2e}")


def _slab_len(b, ax):
    """The axis slab's length from the struct's own dims (the C twin is
    ``sem3d_slab_len``)."""
    nx, ny, nz = int(b.nx), int(b.ny), int(b.nz)
    return {"x": nz * ny * int(b.nsx), "y": nz * int(b.nsy) * nx,
            "z": int(b.nsz) * ny * nx}[ax]

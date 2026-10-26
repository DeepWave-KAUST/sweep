"""The 3-D adjoint C-PML carries on their axis slabs, and the 2-D scratch
arrays that were never used.

ADJOINT CARRIES. lpsi_d / lzeta_d (and their double buffers) are NOT zero
off the absorbing layer: `combine` writes a_d·(l + lap) there with a_d = 1.
They are dead there -- `step_T` is their only reader and multiplies them by
b_d, which is exactly +0.0 off the layer. Storing them only on the slab
therefore changes VALUES nowhere; it can change the SIGN of an exact-zero
product (old: +0.0 · (-3.2) = -0.0; new: +0.0 · 0.0f = +0.0), which no
consumer can distinguish once anything non-zero is added. The claim is
checked where it is attainable bit for bit: the node-wise kernels `step_T`
and `combine` are deterministic given identical inputs, so one reverse step
is run on two copies of the SAME input struct, one per layout -- the full
copy carrying arbitrary finite values off the slab, where the old code
would have had its dead accumulations -- and every output is compared bit
for bit. Bits may differ only where the floats compare equal (±0.0).

2-D SCRATCH. The 2-D acoustic C entries never read or write `apsi_x`,
`apsi_z`, `w_sum` -- shown by NaN-poisoning them (finite outputs, arrays
still all-NaN afterwards), so the drivers now hand empty tensors.
"""
from __future__ import annotations

import math

import pytest
import torch

import sweep.sem as S
import sweep.sem.prop_c as PC
import sweep.sem.dd as DD
from sweep.propagator import BoundarySaving
from sweep.sem.equations import (AcousticPotentialSEM2D, AcousticSEM3D,
                                 AcousticSEM3DDeformed)
from sweep.sem.slab import AXES, FIELD_AXIS, slab_slot

try:
    import sweep._C as _C
    _HAS_C = hasattr(_C, "sem3d_acoustic_cpml_dd_bwd_mid")
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


def _prop3(nex, deformed, nt, abcn=2, mode="full"):
    """Every run here is a boundary-saved one."""
    n = nex * NO + 1
    return S.PropSEM(AcousticSEM3DDeformed if deformed else AcousticSEM3D,
                     shape=(n, n, n), dh=DH, dt=DT, nt=nt, n_order=NO,
                     free_surface=("top",),
                     abcn=abcn * NO, f_dominant=F0, dev="cuda", impl="c",
                     memory=BoundarySaving(), bs_band_mode=mode,
                     **(dict(topography=_hill_rows(n)) if deformed else {}))


def _wav(nt):
    t = torch.arange(nt, dtype=torch.float32, device="cuda") * DT - 1.2 / F0
    a = (3.1415926 * F0 * t) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


# --------------------------------------------------------------------------- #
# the node-kernel twin check
# --------------------------------------------------------------------------- #
FULL_OFF_SLAB = ("lpsi_x", "lpsi_y", "lpsi_z", "lzeta_x", "lzeta_y", "lzeta_z",
                 "lpsi_x_n", "lpsi_y_n", "lpsi_z_n", "lzeta_x_n", "lzeta_y_n", "lzeta_z_n")
# lap_d joins them: it is assembled on its slab and read there, and off the
# slab the full layout holds a real value the slab layout never stores
LAP = ("lapx", "lapy", "lapz")
# g_d and k_d join them for the same reason: g_d is read only where b_d or
# db_d is non-zero, k_d alone only ever as b_d*k_d
GRAD = ("gx", "gy", "gz", "kx", "ky", "kz")
ZERO_OFF_SLAB = ("r_psi_x", "r_psi_y", "r_psi_z", "r_zeta_x", "r_zeta_y", "r_zeta_z",
                 "apx", "apy", "apz")
PRE_OUT = FULL_OFF_SLAB[:6] + ZERO_OFF_SLAB                # carries after Dirichlet; restores; a*psi
MID_OUT = ("gbx", "gby", "gbz", "gbkx", "gbky", "gbkz", "gbax", "gbay", "gbaz", "w_sum", "u_km1")
END_OUT = ("gu_k", "grad_vp2") + FULL_OFF_SLAB[6:]          # the _n buffers combine writes


def _fields(b):
    return [n for n in dir(b) if not n.startswith("_") and not callable(getattr(b, n))]


def _clone(b):
    nb = _C.Sem3dCpmlBackwardInput()
    for n in _fields(b):
        v = getattr(b, n)
        setattr(nb, n, v.clone() if torch.is_tensor(v) else v)
    return nb


def _slots(c, ax):
    dims = (c["nx"], c["ny"], c["nz"])
    g = torch.arange(int(c["n_global"]), device=c["b_x"].device)
    return slab_slot(ax, g, dims, c[f"tab_{ax}"])          # (ng,) slot or -1


def _to_full(compact, s, ng, fill=None):
    """Full-length array from a slab array; off-slab entries `fill` (a
    tensor of ng values) or zero."""
    full = torch.zeros(ng, dtype=compact.dtype, device=compact.device) if fill is None else fill.clone()
    on = s >= 0
    full[on] = compact[s[on]]
    return full


def _to_compact(full, s, n):
    out = torch.zeros(n, dtype=full.dtype, device=full.device)
    on = s >= 0
    out[s[on]] = full[on]
    return out


def _twin(b, c, seed=0):
    """The other layout of the same input: bit-identical on the slabs; the
    full copy carries random finite values off the slab in the adjoint
    carries (where the old code accumulated a·(l + lap)) and +0.0 in the
    restores (what the old code held there)."""
    ng = int(c["n_global"])
    t = _clone(b)
    slab_now = bool(b.psi_slab)
    t.psi_slab = int(not slab_now)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for nm in FULL_OFF_SLAB + ZERO_OFF_SLAB + LAP + GRAD:
        ax = FIELD_AXIS[nm]
        s = _slots(c, ax)
        v = getattr(b, nm)
        if slab_now:                                   # b compact -> twin full
            fill = None
            if nm in FULL_OFF_SLAB:
                scale = float(v.abs().max()) if v.numel() else 1.0
                fill = (torch.rand(ng, generator=gen, device="cuda") - 0.5) * 2.0 * max(scale, 1.0)
            setattr(t, nm, _to_full(v, s, ng, fill))
        else:                                          # b full -> twin compact
            setattr(t, nm, _to_compact(v, s, int(c[f"slab_n_{ax}"])))
    return t


def _bits(a, b):
    """(real differences, sign-of-zero differences) between two arrays."""
    da = a.view(torch.int32) != b.view(torch.int32)
    eq = a == b
    return int((da & ~eq).sum()), int((da & eq).sum())


def _compare(b, t, c, names):
    """Outputs of b (its layout) vs t (the other), in a common layout."""
    real, sz = {}, {}
    for nm in names:
        x, y = getattr(b, nm), getattr(t, nm)
        if nm in FIELD_AXIS:                           # slab arrays: compare on the slab
            s = _slots(c, FIELD_AXIS[nm])
            n = int(c[f"slab_n_{FIELD_AXIS[nm]}"])
            if not b.psi_slab:
                x = _to_compact(x, s, n)
            if not t.psi_slab:
                y = _to_compact(y, s, n)
        r, z = _bits(x, y)
        real[nm], sz[nm] = r, z
    return real, sz


def twin_check(deformed, nex=8, nt=30, abcn=2, k_at=None, mode="full"):
    """Run one DD (px=1) forward+backward; at reverse step K run bwd_mid and
    bwd_end on the live struct AND on its other-layout twin with identical
    inputs, and compare every node-kernel output bit for bit."""
    n = nex * NO + 1
    p = _prop3(nex, deformed, nt, abcn, mode)
    torch.manual_seed(0)
    vp = (2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")).requires_grad_(True)
    # GLOBAL node ids, on the node grid of the model AND its absorbing layer
    nzg, nyg, nxg = p._grid_shape()
    c0y, c0x, z0 = nyg // 2, nxg // 2, 2 * NO
    src = (z0 * nyg + c0y) * nxg + c0x
    rec = torch.tensor([src + 2 * i for i in range(6)], device="cuda")
    K = nt // 2 if k_at is None else k_at
    got = {"pre": None, "mid": None, "end": None, "layout": None}
    real_pre = _C.sem3d_acoustic_cpml_dd_bwd_pre
    real_mid, real_end = _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end
    parts = {}
    real_fwd = DD._SEMAcoustic3DCpmlDDC.forward

    def spy_fwd(ctx, *a):
        out = real_fwd(ctx, *a)
        parts["c"] = ctx.parts[0].c
        return out

    def spy_pre(b, k):
        if k != K:
            return real_pre(b, k)
        # the twin is made BEFORE this step's pre, so pre's Dirichlet fill
        # (didx on the full copy, dir_slot_d on the slab copy), the ring
        # restores and a_d*psi_d are compared too
        c = parts["c"]
        t = _twin(b, c)
        real_pre(b, k)
        real_pre(t, k)
        got["layout"] = int(b.psi_slab)
        got["pre"] = _compare(b, t, c, PRE_OUT)
        # pre's assembly is an atomicAdd: hand b's sums (and its gA/gw) to
        # the twin so step_T sees identical inputs
        for nm in GRAD + ("mx", "my", "mz", "gA", "gw"):
            src = getattr(b, nm)
            if nm in GRAD:                    # g_d and k_d are on their slab
                ax = FIELD_AXIS[nm]
                s = _slots(c, ax)
                src = (_to_full(src, s, int(b.n_global), None) if bool(b.psi_slab)
                       else _to_compact(src, s, int(c[f"slab_n_{ax}"])))
            getattr(t, nm).copy_(src)
        got["twin"] = t
        return None

    def spy_mid(b, k):
        if k != K:
            return real_mid(b, k)
        t = got["twin"]
        real_mid(b, k)
        real_mid(t, k)
        got["mid"] = _compare(b, t, parts["c"], MID_OUT)
        # combine must see identical assembled inputs: copy b's l4 into the
        # twin, THROUGH the layout change for the three lap_d
        c = parts["c"]
        getattr(t, "lu_G").copy_(getattr(b, "lu_G"))
        for nm in LAP:
            ax = FIELD_AXIS[nm]
            s, v = _slots(c, ax), getattr(b, nm)
            setattr(t, nm, _to_full(v, s, int(b.n_global), None)
                    if bool(b.psi_slab)
                    else _to_compact(v, s, int(c[f"slab_n_{ax}"])))
        return None

    def spy_end(b, k):
        if k != K:
            return real_end(b, k)
        t = got["twin"]
        real_end(b, k)
        real_end(t, k)
        got["end"] = _compare(b, t, parts["c"], END_OUT)
        return None

    DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(spy_fwd)
    _C.sem3d_acoustic_cpml_dd_bwd_pre = spy_pre
    _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end = spy_mid, spy_end
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(1))(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        DD._SEMAcoustic3DCpmlDDC.forward = staticmethod(real_fwd)
        _C.sem3d_acoustic_cpml_dd_bwd_pre = real_pre
        _C.sem3d_acoustic_cpml_dd_bwd_mid, _C.sem3d_acoustic_cpml_dd_bwd_end = real_mid, real_end
    assert got["pre"] is not None and got["mid"] is not None and got["end"] is not None
    assert torch.isfinite(vp.grad).all()
    return got


# Where a signed zero may legitimately appear: the nine step_T scratch
# outputs at a slow-path node off axis d's slab. gt_d = o_d*gw + b_d*lzeta_d
# -- old: +0.0*(dead value) = +/-0.0, new: +0.0 -- so gbk_d = -im*o_d*gt_d and
# gba_d = im*gt_d can differ in sign where gw is itself an exact zero (ahead
# of the adjoint front), and gb_d = im*(db_d*gt_d + b_d*lpsi_d) where
# db_d*gt_d is. Nowhere else: w_sum, the u march, the carries, the restores
# and combine's outputs are bit-identical, and grads_T absorbs these nine
# (running sums from +0.f: +0 + -0 = +0), so no carried array ever differs.
SIGN_OK = {"gbx", "gby", "gbz", "gbkx", "gbky", "gbkz", "gbax", "gbay", "gbaz"}


def _assert_bitwise(got):
    assert got["layout"] == 1, "the production layout under test is the slab one"
    for stage in ("pre", "mid", "end"):
        real, sz = got[stage]
        assert all(v == 0 for v in real.values()), f"{stage}: VALUE differences {real}"
        stray = {k: v for k, v in sz.items() if v and k not in SIGN_OK}
        assert not stray, f"{stage}: sign-of-zero differences outside gb*: {stray}"


@needs_c
@pytest.mark.parametrize("k_at", [None, 3], ids=["mid-time", "late-reverse"])
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "topo"])
def test_node_kernels_are_bitwise_up_to_the_sign_of_zero(deformed, k_at):
    got = twin_check(deformed, k_at=k_at)
    _assert_bitwise(got)
    print(f"sign-of-zero entries: pre {got['pre'][1]}  mid {got['mid'][1]}  end {got['end'][1]}")


@needs_c
def test_node_kernels_bitwise_in_rim_mode():
    _assert_bitwise(twin_check(False, nex=10, mode="rim"))


@needs_c
def test_backward_arrays_shrink_by_the_slab_fraction():
    nex, abcn, nt = 12, 2, 12
    n = nex * NO + 1
    p = _prop3(nex, False, nt, abcn)
    torch.manual_seed(0)
    vp = (2500.0 + 200.0 * torch.rand(n, n, n, device="cuda")).requires_grad_(True)
    c0, z0 = n // 2, 2 * NO
    src = torch.tensor([[c0, c0, z0]], device="cuda")
    rec = torch.tensor([[[c0, c0, z0]]], device="cuda")
    seen = {}
    real = _C.sem3d_acoustic_cpml_dd_bwd_pre

    def spy(b, *rest):
        seen["lens"] = {nm: int(getattr(b, nm).numel()) for nm in FULL_OFF_SLAB + ZERO_OFF_SLAB}
        seen["slab"] = int(b.psi_slab)
        seen["ng"] = int(b.n_global)
        return real(b, *rest)

    _C.sem3d_acoustic_cpml_dd_bwd_pre = spy
    try:
        out = p.forward(_wav(nt), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        _C.sem3d_acoustic_cpml_dd_bwd_pre = real
    assert seen["slab"] == 1
    total = sum(seen["lens"].values())
    # 21 arrays, 7 per axis. b spans 2*abcn*NO planes and db reaches ONE
    # further at each band end; the storage slab is their union, so x and y
    # get 2*abcn*NO + 2 planes and z (bottom only) abcn*NO + 1 -- planes of
    # the node grid of the model AND its absorbing layer.
    nzg, nyg, nxg = p._grid_shape()
    planes = 2 * abcn * NO + 2
    want = (7 * planes * nzg * nyg + 7 * nzg * planes * nxg
            + 7 * ((abcn * NO + 1) * nyg * nxg))
    assert total == want, (seen["lens"], want)
    assert total < 0.5 * 21 * seen["ng"]


# --------------------------------------------------------------------------- #
# the 2-D scratch arrays
# --------------------------------------------------------------------------- #
NO2, DH2, F02, DT2, NT2 = 4, 25.0, 8.0, 6e-4, 40
DEAD = ("apsi_x", "apsi_z", "w_sum")


def _wav2(nt):
    t = (torch.arange(nt, dtype=torch.float32, device="cuda") - 6.0) * DT2 * F02
    return (1.0 - 2.0 * (3.1415926 * t) ** 2) * torch.exp(-(3.1415926 * t) ** 2)


def _prop2(nz=33, nx=49):
    """Boundary-saved, as every 2-D run here is."""
    return S.PropSEM(AcousticPotentialSEM2D, shape=(nz, nx), dh=DH2, dt=DT2, nt=NT2,
                     n_order=NO2, free_surface=("top",),
                     abcn=3 * NO2, f_dominant=F02, dev="cuda", impl="c",
                     memory=BoundarySaving())


def _poison(inp, seen):
    ng = int(inp.n_global)
    for nm in DEAD:
        assert getattr(inp, nm).numel() == 0, f"the driver still allocates {nm}"
        setattr(inp, nm, torch.full((ng,), math.nan, dtype=torch.float32, device="cuda"))
    seen.append(tuple(getattr(inp, nm) for nm in DEAD))


@needs_c
@pytest.mark.parametrize("B", [1, 3])
def test_2d_fused_scratch_is_never_read_or_written(B):
    nz, nx = 33, 49
    p = _prop2(nz, nx)
    torch.manual_seed(0)
    vp = (2200.0 + 200.0 * torch.rand(nz, nx, device="cuda")).requires_grad_(True)
    src = torch.tensor([[nx // 2 + 2 * b, 2 * NO2] for b in range(B)], device="cuda")
    rec = torch.tensor([[[nx // 4 + 4 * i, 2 * NO2] for i in range(6)]] * B, device="cuda")
    seen = []
    real = _C.sem2d_acoustic_cpml_forward

    def spy(inp):
        _poison(inp, seen)
        return real(inp)

    _C.sem2d_acoustic_cpml_forward = spy
    try:
        out = p.forward(_wav2(NT2), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        _C.sem2d_acoustic_cpml_forward = real
    assert seen and torch.isfinite(out).all() and torch.isfinite(vp.grad).all()
    assert float(out.abs().max()) > 0 and float(vp.grad.abs().max()) > 0
    assert all(bool(torch.isnan(t).all()) for grp in seen for t in grp), "an entry was written"


@needs_c
def test_2d_dd_scratch_is_never_read_or_written():
    nz, nx = 33, 49
    p = _prop2(nz, nx)
    torch.manual_seed(0)
    vp = (2200.0 + 200.0 * torch.rand(nz, nx, device="cuda")).requires_grad_(True)
    # GLOBAL node ids, on the node grid of the model AND its absorbing layer
    _, nxg = p._grid_shape()
    src = (2 * NO2) * nxg + nxg // 2
    rec = torch.tensor([src + 5 + 3 * i for i in range(6)], device="cuda")
    seen, done = [], set()
    real = _C.sem2d_acoustic_cpml_dd_grads

    def spy(inp):
        if id(inp) not in done:
            done.add(id(inp))
            _poison(inp, seen)
        return real(inp)

    _C.sem2d_acoustic_cpml_dd_grads = spy
    try:
        out = DD.SEMModelParallel(p, DD.StripTopology(2))(
            _wav2(NT2), src, rec, [vp])
        out.pow(2).sum().backward()
    finally:
        _C.sem2d_acoustic_cpml_dd_grads = real
    assert len(seen) == 2 and torch.isfinite(out).all() and torch.isfinite(vp.grad).all()
    assert all(bool(torch.isnan(t).all()) for grp in seen for t in grp), "an entry was written"

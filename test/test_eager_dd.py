"""Domain decomposition of the eager propagator (sweep.parallel.eager_dd).

Single process: every tile in this process (``decomposed_forward``) against
the single-domain run -- record, model and wavelet gradients -- for the
acoustic and the staggered elastic equation, two and three tiles, one and
several steps between exchanges; the admission probes (what is exchanged,
what is refused).  In float64, every model gradient of a set of equations
from each family, checked model by model, near the cut and point by point.  Two gloo ranks on the CPU (one spawn, the checks sharing
the process group as test_model_parallel_halo does): the rank path with its
``HaloSwap`` exchange, on the full tape and under chunk checkpointing, with
global and with per-tile models, against the single-domain run.

Small grids, uncompiled steps: the decomposition is the same code whatever
the size, and a compiled step costs the CI seconds per grid.
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.parallel.eager_dd import (EagerModelParallel, decomposed_forward,  # noqa: E402
                                     exchange_plan, step_reach, tile_grid)
from sweep.propagator.options import BoundarySaving  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

from conftest import ricker  # noqa: E402

NZ, NX, NT, DT, DH, ABCN = 24, 48, 80, 1.0e-3, 10.0, 6
DEV = torch.device("cpu")


def models_for(name, nx=NX):
    z = np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(1800.0 + 700.0 * z, (NZ, nx)).copy()
    vp[8:16, nx // 2 - 6:nx // 2 + 6] += 250.0                   # straddles the mid cut
    if name == "acoustic":
        return [vp]
    return [vp, vp / 1.8, 1800.0 + 0.2 * (vp - 1800.0)]


def make_prop(name, dev=DEV, cls=None, fs=False, use_ckpt=False, abcn=ABCN, **kw):
    if name == "acoustic":
        eq = (cls or Acoustic)(spatial_order=4, device=dev, backend="torch")
        st, pml = ["h1"], "cpmlr"
    else:
        eq = (cls or Elastic)(spatial_order=4, device=dev, backend="torch")
        st, pml = ["vz"], "cpmls"
    return PropTorch(eq, backend="torch", impl="eager", shape=(NZ, NX), dev=dev, dh=DH, dt=DT, nt=NT,
                     abcn=abcn, source_type=st, receiver_type=st, pml_type=pml, B=1,
                     use_ckpt=use_ckpt, ckpt_chunks=16, use_compile=False, free_surface=fs, **kw)


def geometry(src_x=25):
    wav = (ricker(NT, DT, fm=15.0) * 1e3).astype(np.float32)
    src = np.array([[[src_x, 6]]], dtype=np.int32)
    rec = np.array([[[x, 3] for x in range(1, NX - 1, 3)]], dtype=np.int32)
    return wav, src, rec


def run(prop, fn, wav, src, rec, mods, dev=DEV):
    ms = [torch.tensor(m, device=dev, requires_grad=True) for m in mods]
    w = torch.tensor(wav, device=dev, requires_grad=True)
    out = fn(prop, w, src, rec, ms)
    g = torch.Generator().manual_seed(1)
    adj = torch.randn(out.shape, generator=g).to(out)
    (out * adj).sum().backward()
    return out.detach(), [m.grad for m in ms], w.grad


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def single(p, w, s, r, m):
    return p(w, s, r, models=m)


# --------------------------------------------------------------- one process
@pytest.mark.parametrize("name", ["acoustic", "elastic"])
@pytest.mark.parametrize("px,K", [(2, 1), (3, 2)])
def test_tiles_match_single_domain(name, px, K):
    """Record, model and wavelet gradients of the decomposed run against the
    single domain.  The record is the same arithmetic on the owned columns;
    the gradients sum the two sides of a cut in another order (rounding)."""
    wav, src, rec = geometry()
    mods = models_for(name)
    ref = run(make_prop(name), single, wav, src, rec, mods)
    dd = run(make_prop(name), lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=px, exchange_steps=K),
             wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    for x, y in zip(dd[1], ref[1]):
        assert rel(x, y) < 2e-5
    assert rel(dd[2], ref[2]) < 2e-5


def test_source_in_ghost_and_free_surface():
    """A source two columns from the cut lands in the neighbour's ghosts,
    which inject it too; with a free surface on top."""
    wav, src, rec = geometry(src_x=NX // 2 - 2)
    mods = models_for("acoustic")
    ref = run(make_prop("acoustic", fs=True), single, wav, src, rec, mods)
    dd = run(make_prop("acoustic", fs=True),
             lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=2, exchange_steps=2), wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    assert rel(dd[1][0], ref[1][0]) < 2e-5
    assert rel(dd[2], ref[2]) < 2e-5


def test_multi_shot_batched():
    """Two shots (mode A1), one source per tile."""
    wav, _, rec = geometry()
    src = np.array([[10, 6], [NX - 10, 6]], dtype=np.int32)        # (nshots, ndim)
    rec = np.repeat(rec, 2, axis=0)
    mods = models_for("acoustic")
    ref = run(make_prop("acoustic"), single, wav, src, rec, mods)
    dd = run(make_prop("acoustic"), lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=2, exchange_steps=1),
             wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    assert rel(dd[1][0], ref[1][0]) < 2e-5


NZ3, NY3, NX3 = 12, 10, 24


def make_prop3d(name, dev=DEV):
    if name == "acoustic3d":
        eq, st, rt = Acoustic3D(spatial_order=4, device=dev, backend="torch"), ["h1"], ["h1"]
    else:
        eq, st, rt = Elastic3D(spatial_order=4, device=dev, backend="torch"), ["sxx", "syy", "szz"], ["vz"]
    return PropTorch(eq, backend="torch", impl="eager", shape=(NZ3, NY3, NX3), dev=dev, dh=DH, dt=DT, nt=40,
                     abcn=4, source_type=st, receiver_type=rt, B=1, use_ckpt=False, use_compile=False,
                     cpml_strips=False)


def models3d(name):
    z = np.linspace(0, 1, NZ3, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(1800.0 + 700.0 * z, (NZ3, NY3, NX3)).copy()
    vp[4:8, 3:7, 8:16] += 250.0
    return [vp] if name == "acoustic3d" else [vp, vp / 1.8, 1800.0 + 0.2 * (vp - 1800.0)]


@pytest.mark.parametrize("name", ["acoustic3d", "elastic3d"])
@pytest.mark.parametrize("px,K", [(2, 1), (3, 2)])
def test_3d_tiles_match_single_domain(name, px, K):
    """3-D, an x cut (cpml_strips=False: the strips layout is refused)."""
    wav = (ricker(40, DT, fm=15.0) * 1e3).astype(np.float32)
    src = np.array([[[13, NY3 // 2, 3]]], dtype=np.int32)                   # (x, y, z)
    rec = np.array([[[x, y, 2] for x in range(1, NX3 - 1, 3) for y in (2, NY3 - 3)]], dtype=np.int32)
    mods = models3d(name)
    ref = run(make_prop3d(name), single, wav, src, rec, mods)
    dd = run(make_prop3d(name), lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=px, exchange_steps=K),
             wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    for x, y in zip(dd[1], ref[1]):
        assert rel(x, y) < 2e-5
    assert rel(dd[2], ref[2]) < 2e-5


@pytest.mark.parametrize("name", ["acoustic3d", "elastic3d"])
@pytest.mark.parametrize("py,px,K", [(2, 2, 1), (2, 3, 2)])
def test_3d_xy_tiles_match_single_domain(name, py, px, K):
    """3-D, an x/y tile grid: the corners come through the y pass."""
    wav = (ricker(40, DT, fm=15.0) * 1e3).astype(np.float32)
    src = np.array([[[13, NY3 // 2 + 1, 3]]], dtype=np.int32)              # near both cuts
    rec = np.array([[[x, y, 2] for x in range(1, NX3 - 1, 3) for y in range(1, NY3 - 1, 2)]], dtype=np.int32)
    mods = models3d(name)
    ref = run(make_prop3d(name), single, wav, src, rec, mods)
    dd = run(make_prop3d(name),
             lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=px, py=py, exchange_steps=K),
             wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    for x, y in zip(dd[1], ref[1]):
        assert rel(x, y) < 2e-5
    assert rel(dd[2], ref[2]) < 2e-5


# ------------------------------------------------------------- balanced cut
def _extents(cuts):
    """Cells every tile computes along each cut axis (owned + ghosts)."""
    return [[hi - lo for lo, _, _, hi in (c.extent(i) for i in range(c.parts))] for c in cuts]


def _assert_balanced(ext):
    """Whole multiples of ALIGN but for one tile's leftover, so within 2*ALIGN."""
    from sweep.parallel.eager_dd import _Cut
    assert sum(e % _Cut.ALIGN != 0 for e in ext) <= 1, ext
    assert max(ext) - min(ext) < 2 * _Cut.ALIGN, ext


@pytest.mark.parametrize("px,K", [(3, 2), (4, 1)])
def test_balanced_tiles_match_single_domain(px, K):
    """balance=True: every tile computes as many cells (the edge tiles own
    fewer physical columns, their pad counting), and the result is the
    single-domain run's."""
    from sweep.parallel import eager_dd as ed
    seen = []
    orig = ed.admit

    def spy(prop, padded, cuts, K_, **kw):
        orig(prop, padded, cuts, K_, **kw)
        seen.append(cuts)
    wav, src, rec = geometry(src_x=17)
    mods = models_for("elastic")
    ref = run(make_prop("elastic"), single, wav, src, rec, mods)
    with pytest.MonkeyPatch.context() as mp_:
        mp_.setattr(ed, "admit", spy)
        dd = run(make_prop("elastic"), lambda p, w, s, r, m: decomposed_forward(
            p, w, s, r, m, px=px, exchange_steps=K, balance=True), wav, src, rec, mods)
    (ext,) = _extents(seen[0])
    _assert_balanced(ext)
    uniform = tile_grid(make_prop("elastic"), 1, px)[0]
    assert seen[0][0].ranges != uniform.ranges
    assert sum(seen[0][0].phys_extent(i)[1] for i in range(px)) == NX
    assert rel(dd[0], ref[0]) < 1e-6
    for x, y in zip(dd[1], ref[1]):
        assert rel(x, y) < 2e-5
    assert rel(dd[2], ref[2]) < 2e-5


def test_balanced_3d_xy_tiles_match_single_domain():
    from sweep.parallel import eager_dd as ed
    seen = []
    orig = ed.admit
    wav = (ricker(40, DT, fm=15.0) * 1e3).astype(np.float32)
    src = np.array([[[13, NY3 // 2 + 1, 3]]], dtype=np.int32)
    rec = np.array([[[x, y, 2] for x in range(1, NX3 - 1, 3) for y in range(1, NY3 - 1, 2)]], dtype=np.int32)
    mods = models3d("acoustic3d")
    ref = run(make_prop3d("acoustic3d"), single, wav, src, rec, mods)
    with pytest.MonkeyPatch.context() as mp_:
        mp_.setattr(ed, "admit", lambda *a, **kw: (orig(*a, **kw), seen.append(a[2])))
        dd = run(make_prop3d("acoustic3d"), lambda p, w, s, r, m: decomposed_forward(
            p, w, s, r, m, px=3, py=2, exchange_steps=1, balance=True), wav, src, rec, mods)
    for ext in _extents(seen[0]):
        _assert_balanced(ext)
    assert rel(dd[0], ref[0]) < 1e-6
    for x, y in zip(dd[1], ref[1]):
        assert rel(x, y) < 2e-5


def test_balance_refusals():
    from types import SimpleNamespace
    with pytest.raises(NotImplementedError, match="eager tiles' switch"):
        ModelParallel(SimpleNamespace(), None, balance=True)          # the stepped class
    mesh = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    with pytest.raises(TypeError, match="True or False"):
        ModelParallel(make_prop("acoustic"), mesh, balance="yes")
    ddp = ModelParallel(make_prop("acoustic"), mesh, balance=True)
    with pytest.raises(RuntimeError, match="known after it"):
        ddp.local_extent()
    wav, src, rec = geometry()
    tile = [torch.tensor(models_for("acoustic")[0][:, :NX - 1])]
    with pytest.raises(ValueError):
        ddp(torch.tensor(wav), src, rec, models=tile)
    # an absorbing pad wider than an edge tile's share leaves it no physical cell
    with pytest.raises(NotImplementedError, match="no physical cell"):
        decomposed_forward(make_prop("acoustic", abcn=40), torch.tensor(wav), src, rec,
                           [torch.tensor(m) for m in models_for("acoustic")], px=8, balance=True)


def test_2d_refuses_y_cut():
    wav, src, rec = geometry()
    with pytest.raises(ValueError, match="py=1"):
        ModelParallel(make_prop("acoustic"), MeshTopology(py=2, px=1, shot_groups=1, world_size=2, rank=0))


def _padded(prop):
    p = prop._backend_impl
    from sweep.parallel.eager_dd import _pad_global
    return p, _pad_global(p, [torch.tensor(m) for m in models_for("acoustic" if len(p.model_names) == 1 else "elastic")], 1)


def test_reach_and_exchange_plan():
    """The probes: acoustic reach M, one field exchanged at K = 1 (the
    current pressure), more at K = 4; the staggered elastic reach 2M - 1."""
    for name, R, n1 in (("acoustic", 2, 1), ("elastic", 3, 7)):
        prop = make_prop(name)
        p, padded = _padded(prop)
        p.init_abc(fd_pad=p._runtime_fd_pad(), shape=p._runtime_shape())
        rt = p._prepare_runtime_models(padded)
        assert step_reach(p, rt) == R
        fields, H = exchange_plan(p, rt, 1)
        assert len(fields) == n1 and H == R
        if name == "acoustic":
            assert [p.wavefield_names[j] for j in fields] == ["h1"]
            fields4, H4 = exchange_plan(p, rt, 4)
            assert H4 == 4 * R and len(fields4) > 1


class AcousticSponge(Acoustic):
    """A new equation: the pressure damped by a profile along x held as a
    grid-shaped tensor -- cut per tile, admitted."""
    def init_abc(self, type="cpml", **kw):
        super().init_abc(type, **kw)
        x = torch.linspace(0, 1, kw["shape"][-1], device=self.device)
        self.sponge = (2e-3 * torch.sin(7 * x) ** 2).view(1, 1, 1, -1)

    def func(self, wavefields, models, dt, h, b, invariants=None, **kw):
        out = list(super().func(wavefields, models, dt, h, b, invariants=invariants, **kw))
        out[0] = out[0] * (1 - self.sponge)
        return tuple(out)


class AcousticSpongeNumpy(AcousticSponge):
    """The same profile as a numpy array: the cut cannot pass through it."""
    def init_abc(self, type="cpml", **kw):
        super().init_abc(type, **kw)
        self.sponge_np = self.sponge.cpu().numpy()
        del self.sponge

    def func(self, wavefields, models, dt, h, b, invariants=None, **kw):
        out = list(Acoustic.func(self, wavefields, models, dt, h, b, invariants=invariants, **kw))
        out[0] = out[0] * (1 - torch.as_tensor(self.sponge_np, device=out[0].device))
        return tuple(out)


class AcousticNonLocal(Acoustic):
    """Adds a row mean: every column depends on every other."""
    def func(self, wavefields, models, dt, h, b, invariants=None, **kw):
        out = list(super().func(wavefields, models, dt, h, b, invariants=invariants, **kw))
        out[0] = out[0] + 1e-3 * out[0].mean(dim=-1, keepdim=True)
        return tuple(out)


def test_new_equation_with_grid_tensor_is_admitted():
    wav, src, rec = geometry()
    mods = models_for("acoustic")
    ref = run(make_prop("acoustic", cls=AcousticSponge), single, wav, src, rec, mods)
    dd = run(make_prop("acoustic", cls=AcousticSponge),
             lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=2, exchange_steps=2), wav, src, rec, mods)
    assert rel(dd[0], ref[0]) < 1e-6
    assert rel(dd[1][0], ref[1][0]) < 2e-5


@pytest.mark.parametrize("cls,why", [(AcousticSpongeNumpy, "tile"), (AcousticNonLocal, "not local")])
def test_refused_equations(cls, why):
    wav, src, rec = geometry()
    mods = [torch.tensor(m) for m in models_for("acoustic")]
    with pytest.raises(NotImplementedError, match=why):
        decomposed_forward(make_prop("acoustic", cls=cls), wav, src, rec, mods, px=2, exchange_steps=1)


@pytest.mark.parametrize("mode", ["adjoint", "frozen"])
def test_qp_backward_must_be_additive(mode):
    """The qP VTI step's default backward takes S_n from the adjoint field: its
    record decomposes, its gradient does not (3e-4 in float64 near a cut), so
    admission refuses it; the autograd-differentiated 'frozen' mode decomposes."""
    from sweep.equations import AcousticVTI
    wav, src, rec = geometry(src_x=NX // 2 + 5)
    vp = models_for("acoustic")[0]
    mods = [vp, np.full_like(vp, 0.12), np.full_like(vp, 0.05)]

    def make():
        eq = AcousticVTI(spatial_order=4, device=DEV, backend="torch", s_gradient=mode)
        return PropTorch(eq, backend="torch", impl="eager", shape=(NZ, NX), dev=DEV, dh=DH, dt=5e-4,
                         nt=NT, abcn=ABCN, source_type=["h1"], receiver_type=["h1"], B=1,
                         use_compile=False, free_surface=False)
    dd = lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=2, exchange_steps=1)  # noqa: E731
    if mode == "adjoint":
        with pytest.raises(NotImplementedError, match="not additive"):
            dd(make(), wav, src, rec, [torch.tensor(m) for m in mods])
        return
    ref = run(make(), single, wav, src, rec, mods)
    got = run(make(), dd, wav, src, rec, mods)
    assert rel(got[0], ref[0]) < 1e-6
    for x, y in zip(got[1], ref[1]):
        assert rel(x, y) < 2e-5


# ------------------------------------------------- float64, model by model
# A float32 max-relative check pooled over models cannot tell a decomposition
# error from round-off where a gradient is small: Biot's moduli gradients are
# 1e-16 in absolute terms, a VRZ impedance gradient sits four orders under the
# velocity one, and an error confined to the cells near a cut, where the
# gradient may be weak, is divided by the global maximum.  In float64 the
# decomposed run must equal the single domain to round-off (1e-14 measured on
# every admitted equation) for EVERY model, in the cells around the cut, and
# point by point -- the qP adjoint-mode error this exists to catch was 3e-4.
FP64_EQUATIONS = [("Acoustic", {}), ("Acoustic1st", {}), ("AcousticVRZ", {}), ("AcousticBorn", {}),
                  ("AcousticVTI", {"s_gradient": "frozen"}), ("Elastic", {}), ("ElasticTTISG", {}),
                  ("ViscoElastic", {}), ("Biot", {}), ("Acoustic3D", {})]


def _model_value(name, vp):
    """A plausible value for every model name these equations declare."""
    rho = 1800.0 + 0.2 * (vp - 1800.0)
    c = np.ones_like(vp)
    v = {"vp": vp, "vp0": vp, "vs": vp / 1.8, "vs0": vp / 1.8, "rho": rho, "z": vp * rho,
         "mp": 1e-2 * (vp - vp.mean()) / vp.std(), "epsilon": 0.12 * c, "delta": 0.05 * c,
         "gamma": 0.06 * c, "theta": 0.3 * c, "phi": 0.2 * c, "Qp": 80 * c, "Qs": 40 * c,
         "H": 2200 * vp ** 2 * 1.4, "mu": 2200 * (vp / 1.8) ** 2, "C": 4e9 * c, "M": 8e9 * c,
         "rho_f": 1000 * c, "m": 3000 * c, "b": 1e3 * c}
    return v[name]


def _to64(obj, seen):
    """Every float32 tensor an equation holds (derivative operators included)."""
    if torch.is_tensor(obj):
        return obj.double() if obj.dtype == torch.float32 else obj
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        return type(obj)(_to64(o, seen) for o in obj)
    if isinstance(obj, dict):
        return {k: _to64(v, seen) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and type(obj).__module__.startswith("sweep.") and id(obj) not in seen:
        seen.add(id(obj))
        for k, v in list(vars(obj).items()):
            nv = _to64(v, seen)
            if nv is not v:
                setattr(obj, k, nv)
    return obj


@pytest.fixture
def eager_fp64(monkeypatch):
    """The FD eager path hard-codes float32; map it to float64 for this test
    (same discrete operator, double arithmetic)."""
    import sweep.parallel.eager_dd as ed
    import sweep.propagator._torch_eager as te

    class _Torch64:
        float32 = torch.float64

        def __getattr__(self, n):
            return getattr(torch, n)
    monkeypatch.setattr(te, "torch", _Torch64())
    monkeypatch.setattr(ed, "torch", te.torch)


@pytest.mark.parametrize("name,kw", FP64_EQUATIONS, ids=[n for n, _ in FP64_EQUATIONS])
def test_fp64_every_model_gradient_matches(eager_fp64, name, kw):
    import sweep.equations as eqs
    cls = getattr(eqs, name)
    ndim = getattr(cls(device=DEV, backend="torch"), "ndim", 2)
    shape = (NZ, 12, NX) if ndim == 3 else (NZ, NX)

    def make():
        eq = cls(spatial_order=4, device=DEV, backend="torch", **kw)
        p = PropTorch(eq, backend="torch", impl="eager", shape=shape, dev=DEV, dh=DH, dt=5e-4, nt=NT,
                      abcn=ABCN, source_type=list(eq.default_source_fields[:1]),
                      receiver_type=list(eq.default_receiver_fields[:1]), B=1, use_compile=False,
                      free_surface=False, **({"cpml_strips": False} if ndim == 3 else {}))
        _to64(p._backend_impl.equation, set())
        return p

    z = np.linspace(0, 1, NZ).reshape((-1,) + (1,) * (ndim - 1))
    vp = np.broadcast_to(1800.0 + 700.0 * z, shape).copy()
    vp[(slice(8, 16),) + (slice(None),) * (ndim - 2) + (slice(NX // 2 - 6, NX // 2 + 6),)] += 250.0
    names = [m.name for m in make()._backend_impl.equation.model_specs]
    mods = [_model_value(n, vp) for n in names]
    wav = ricker(NT, 5e-4, fm=15.0).astype(np.float64) * 1e3
    xs, mid = NX // 2 + 5, [6] if ndim == 2 else [6, 6]          # source 5 cells from the cut
    src = np.array([[[xs] + mid[1:] + [6]]], dtype=np.int32)
    rec = np.array([[[x] + mid[1:] + [3] for x in range(1, NX - 1, 3)]], dtype=np.int32)

    def go(fn):
        ms = [torch.tensor(m, dtype=torch.float64, requires_grad=True) for m in mods]
        out = fn(make(), torch.tensor(wav, requires_grad=True), src, rec, ms)
        adj = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        (out * adj).sum().backward()
        return out.detach(), [m.grad for m in ms]

    ref_rec, ref = go(single)
    assert ref_rec.dtype == torch.float64
    cut, band = NX // 2, slice(NX // 2 - 10, NX // 2 + 10)
    for K in (1, 2):
        got_rec, got = go(lambda p, w, s, r, m: decomposed_forward(p, w, s, r, m, px=2, exchange_steps=K))
        assert rel(got_rec, ref_rec) < 1e-12, (K, "record")
        for n, x, y in zip(names, got, ref):
            gmax = y.abs().max()
            assert x is not None and gmax > 0, (K, n, "no gradient")
            e = (x - y).abs()
            assert e.max() / gmax < 1e-11, (K, n, "whole grid", float(e.max() / gmax))
            assert e[..., band].max() / y[..., band].abs().max() < 1e-11, (K, n, f"cut {cut}+-10")
            big = y.abs() > 1e-3 * gmax
            assert (e[big] / y.abs()[big]).max() < 1e-10, (K, n, "pointwise")


def test_refused_strips_boundary_saving_snapshots():
    wav, src, rec = geometry()
    mods = [torch.tensor(m) for m in models_for("acoustic")]
    with pytest.raises(NotImplementedError, match="cpml_strips"):
        decomposed_forward(make_prop("acoustic", cpml_strips=True), wav, src, rec, mods, px=2)
    bs = make_prop("acoustic", memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="boundary saving"):
        decomposed_forward(bs, wav, src, rec, mods, px=2)
    mesh = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    ddp = ModelParallel(make_prop("acoustic"), mesh)
    assert isinstance(ddp, EagerModelParallel)
    with pytest.raises(NotImplementedError, match="snapshots"):
        ddp(wav, src, rec, models=mods, return_wavefield=True)
    with pytest.raises(NotImplementedError, match="overlap"):   # the stepped drivers' switch
        ModelParallel(make_prop("acoustic"), mesh, overlap=False)
    with pytest.raises(ValueError, match="halo"):
        ModelParallel(make_prop("acoustic"), mesh, halo="ipc")


def test_exchange_steps_must_fit_compiled_call():
    wav, src, rec = geometry()
    mods = [torch.tensor(m) for m in models_for("acoustic")]
    prop = make_prop("acoustic")
    prop._backend_impl.use_compile, prop._backend_impl.steps_per_call = True, 4
    with pytest.raises(ValueError, match="multiple"):
        decomposed_forward(prop, wav, src, rec, mods, px=2, exchange_steps=3)


def test_world_of_one_is_the_single_domain():
    """A one-rank mesh: no exchange, the record of the plain propagator."""
    wav, src, rec = geometry()
    mods = models_for("acoustic")
    mesh = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    ref = run(make_prop("acoustic"), single, wav, src, rec, mods)
    ddp = ModelParallel(make_prop("acoustic"), mesh)
    dd = run(None, lambda _, w, s, r, m: ddp(w, s, r, models=m), wav, src, rec, mods)
    assert torch.equal(dd[0], ref[0])
    assert ddp.own_receivers.tolist() == list(range(rec.shape[1]))
    assert ddp.own_receiver_indices == tuple(range(rec.shape[1]))


# ------------------------------------------------------------ two gloo ranks
def _init_pg(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world_size))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _rank_check(name, rank, world, fs=False, use_ckpt=False, K=2, tile_models=False, balance=False):
    """The rank path against the single-domain run: this rank's record
    columns and its part of the gradients (global models: the ranks' parts
    sum to the single-domain gradient; tile models: this tile's gradient is
    the single-domain gradient's tile)."""
    torch.manual_seed(0)
    wav, src, rec = geometry(src_x=NX // 2 - 2 if fs else 25)
    mods = models_for(name)
    ref = run(make_prop(name, fs=fs, use_ckpt=use_ckpt), single, wav, src, rec, mods)
    mesh = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    ddp = ModelParallel(make_prop(name, fs=fs, use_ckpt=use_ckpt), mesh, exchange_steps=K, balance=balance)
    (_, nx_loc), (_, ox) = mesh.local_extent((NZ, NX))
    leaves_np = [m[:, ox:ox + nx_loc].copy() for m in mods] if tile_models else mods
    ms = [torch.tensor(m, requires_grad=True) for m in leaves_np]
    w = torch.tensor(wav, requires_grad=True)
    out = ddp(w, src, rec, models=ms)
    if balance:
        (_, nx_loc), (_, ox) = ddp.local_extent()
        assert nx_loc != NX // world, "the balanced cut moved nothing"
    idx = ddp.own_receivers
    g = torch.Generator().manual_seed(1)
    adj = torch.randn(ref[0].shape, generator=g)
    (out * adj[:, :, idx]).sum().backward()
    assert rel(out.detach(), ref[0][:, :, idx]) < 1e-6
    grads = [m.grad.clone() for m in ms]
    assert rel(w.grad, ref[2]) < 2e-5                  # complete on every rank
    if tile_models:
        for gt, gr in zip(grads, ref[1]):
            assert rel(gt, gr[:, ox:ox + nx_loc]) < 2e-5
    else:
        # the stepped class's semantics: this rank's tile filled, zero elsewhere
        for gt, gr in zip(grads, ref[1]):
            assert rel(gt[:, ox:ox + nx_loc], gr[:, ox:ox + nx_loc]) < 2e-5
            outside = gt.clone()
            outside[:, ox:ox + nx_loc] = 0
            assert outside.abs().max() == 0
            dist.all_reduce(gt)
            assert rel(gt, gr) < 2e-5
    full = ddp.gather_record(out)
    if rank == 0:
        assert rel(full, ref[0]) < 1e-6
    else:
        assert full is None


def _rank_check_3d(rank, world, name="acoustic3d", K=2, py=1, tile_models=False):
    """The rank path in 3-D: an x cut (``py=1``) or a ``py x px`` tile grid."""
    wav = (ricker(40, DT, fm=15.0) * 1e3).astype(np.float32)
    src = np.array([[[13, NY3 // 2 + 1, 3]]], dtype=np.int32)
    rec = np.array([[[x, y, 2] for x in range(1, NX3 - 1, 3) for y in range(1, NY3 - 1, 2)]], dtype=np.int32)
    mods = models3d(name)
    ref = run(make_prop3d(name), single, wav, src, rec, mods)
    mesh = MeshTopology(py=py, px=world // py, shot_groups=1, world_size=world, rank=rank)
    if tile_models:
        loc, offs = mesh.local_extent((NZ3, NY3, NX3))
        sl = (slice(None), slice(offs[1], offs[1] + loc[1]), slice(offs[2], offs[2] + loc[2]))
        mods_in = [m[sl].copy() for m in mods]
    else:
        mods_in = mods
    ddp = ModelParallel(make_prop3d(name), mesh, exchange_steps=K)
    ms = [torch.tensor(m, requires_grad=True) for m in mods_in]
    w = torch.tensor(wav, requires_grad=True)
    out = ddp(w, src, rec, models=ms)
    idx = ddp.own_receivers
    g = torch.Generator().manual_seed(1)
    adj = torch.randn(ref[0].shape, generator=g)
    (out * adj[:, :, idx]).sum().backward()
    assert rel(out.detach(), ref[0][:, :, idx]) < 1e-6
    assert rel(w.grad, ref[2]) < 2e-5
    for m, gr in zip(ms, ref[1]):
        if tile_models:
            assert rel(m.grad, gr[sl]) < 2e-5
        else:
            gt = m.grad.clone()
            dist.all_reduce(gt)
            assert rel(gt, gr) < 2e-5
    full = ddp.gather_record(out)
    if rank == 0:
        assert rel(full, ref[0]) < 1e-6


CHECKS = {
    "acoustic_full_global": dict(name="acoustic"),
    "elastic_ckpt_fs": dict(name="elastic", fs=True, use_ckpt=True, K=3),
    "acoustic_tile_models_ckpt": dict(name="acoustic", use_ckpt=True, tile_models=True, K=4),
    "acoustic_3d": dict(name="acoustic3d"),
}


CHECKS_4 = {
    "acoustic_balanced_px4": dict(name="acoustic", K=2, balance=True),
    "acoustic_3d_2x2": dict(name="acoustic3d", py=2),
    "elastic_3d_2x2_tile_models": dict(name="elastic3d", py=2, K=3, tile_models=True),
}


def _worker(rank, world, port, out_dir):
    checks = CHECKS if world == 2 else CHECKS_4
    try:
        _init_pg(rank, world, port)
        torch.set_num_threads(1)
        for key, kw in checks.items():
            try:
                if "_3d" in key:
                    _rank_check_3d(rank, world, **kw)
                else:
                    _rank_check(rank=rank, world=world, **kw)
                outcome = "ok"
            except Exception:
                outcome = traceback.format_exc()
            Path(out_dir, f"{key}.{rank}").write_text(outcome)
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(world, checks, out_dir):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(world, port, str(out_dir)), nprocs=world, join=True)
    return {key: {r: (Path(out_dir, f"{key}.{r}").read_text() if Path(out_dir, f"{key}.{r}").exists()
                      else "no outcome written") for r in range(world)} for key in checks}


@pytest.fixture(scope="module")
def two_ranks(tmp_path_factory):
    return _spawn(2, CHECKS, tmp_path_factory.mktemp("eager_dd_2ranks"))


@pytest.fixture(scope="module")
def four_ranks(tmp_path_factory):
    return _spawn(4, CHECKS_4, tmp_path_factory.mktemp("eager_dd_4ranks"))


def _assert_ok(outcomes, key):
    failed = {r: o for r, o in outcomes[key].items() if o != "ok"}
    assert not failed, "\n".join(f"rank {r}:\n{o}" for r, o in failed.items())


@pytest.mark.parametrize("key", list(CHECKS))
def test_two_ranks(two_ranks, key):
    _assert_ok(two_ranks, key)


@pytest.mark.parametrize("key", list(CHECKS_4))
def test_four_ranks(four_ranks, key):
    _assert_ok(four_ranks, key)

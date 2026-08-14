"""Per-edge free surface on ``impl='c'`` (CUDA): forward record + adjoint gradient
consistency vs the eager reference, for every face and z∩x corner, across ALL
backward memory modes — full, boundary-saving, and checkpointing.

CRITICAL — the geometry must DISCRIMINATE.  A central source with near-surface
receivers and short ``NT`` makes the free-surface reflection never reach the
receivers, so every ``free_surface`` config yields ~the same record/gradient and
a backend that silently ignores a face (e.g. runs top-only) passes trivially.
This test uses a central source, a receiver GRID over the whole physical
interior, and a long ``NT`` (many boundary round-trips), and asserts each
non-top config's record genuinely DIFFERS from the top-only reference before
trusting its c-vs-eager match.  The gradient tolerance (rel_l2 < 3e-3) is well
below the error a top-only-collapsed backend produces (>=2.8e-2 for a single x
face, 0.06+ for a z face), so this test fails loudly on that regression.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, DASMu, DASMu3D, Elastic, Elastic3D
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import (
    CUDAOptions, MemoryOptions, BoundaryOptions, CkptOptions,
)
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="per-edge free surface on impl='c' requires CUDA",
)

DEV = "cuda"
NZ, NX, DH, DT, NT, SO, ABCN = 64, 64, 8.0, 1.0e-3, 450, 4, 16
PHYS = NZ - 2 * (ABCN + SO)      # physical interior of the no-FS layout (coords fit all configs)
DISC = 0.10                      # a non-top config must move the record at least this much

CONFIGS = [
    (True, "top"),
    (["bottom"], "bottom"),
    (["left"], "left"),
    (["right"], "right"),
    (["top", "left"], "corner"),
    (["top", "bottom", "left", "right"], "all4"),
]
MODES = ["full", "bs", "ckpt"]


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT - 0.05
    return torch.tensor((1e3 * ricker(t, f=10.0)).astype(np.float32), device=DEV)


def _src_rec():
    src = np.array([[PHYS // 2, PHYS // 2]], dtype=np.int64)
    g = np.arange(2, PHYS - 1, 4, dtype=np.int64)
    rx, rz = np.meshgrid(g, g, indexing="xy")
    rec = np.stack([rx.ravel(), rz.ravel()], -1)[None]
    return src, rec


def _models(eqkind, req):
    d = np.linspace(0, 1, NZ, dtype=np.float32)
    vp = np.broadcast_to((2200 + 700 * d)[:, None], (NZ, NX)).astype(np.float32).copy()
    vp_t = torch.tensor(vp, device=DEV, requires_grad=req)
    if eqkind == "acoustic":
        return [vp_t]
    return [vp_t,
            torch.tensor((vp / 1.7).astype(np.float32).copy(), device=DEV),
            torch.full((NZ, NX), 1000.0, device=DEV)]


def _prop(eqkind, fs, mode):
    if eqkind == "acoustic":
        eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
        stype, rtype = ["p"], ["p"]
    else:
        eq = Elastic(spatial_order=SO, device=DEV, backend="torch")
        stype, rtype = ["sxx", "szz"], ["vx", "vz"]
    common = dict(shape=(NZ, NX), free_surface=fs, abcn=ABCN, dh=DH, dt=DT,
                  source_type=stype, receiver_type=rtype)
    if mode == "eager":
        return PropTorch(eq, backend="torch", impl="eager", use_ckpt=False, **common)
    if mode == "full":
        return PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **common)
    if mode == "bs":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="boundary", boundary=BoundaryOptions(storage="gpu"))),
                         **common)
    if mode == "ckpt":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=10))),
                         **common)
    raise ValueError(mode)


# --- cache the (expensive, pure-python) eager reference per (eqkind, tag) ------
_CACHE = {}


def _record(eqkind, fs, mode, tag):
    key = ("rec", eqkind, tag, mode)
    if mode == "eager" and key in _CACHE:
        return _CACHE[key]
    prop = _prop(eqkind, fs, mode)
    src, rec = _src_rec()
    out = prop(_wav(), src, rec, models=_models(eqkind, False)).detach()
    if mode == "eager":
        _CACHE[key] = out
    return out


def _grad(eqkind, fs, mode, tag):
    key = ("grad", eqkind, tag, mode)
    if mode == "eager" and key in _CACHE:
        return _CACHE[key]
    prop = _prop(eqkind, fs, mode)
    src, rec = _src_rec()
    models = _models(eqkind, True)
    syn = prop(_wav(), src, rec, models=models)
    (0.5 * (syn ** 2).sum()).backward()
    g = models[0].grad
    if mode == "eager":
        _CACHE[key] = g
    return g


def _cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


@pytest.mark.parametrize("eqkind", ["acoustic", "elastic"])
@pytest.mark.parametrize("fs,tag", CONFIGS)
def test_per_edge_cuda_forward_matches_eager(eqkind, fs, tag):
    """c forward record ~ eager, in a geometry where the free surface matters."""
    re = _record(eqkind, fs, "eager", tag)
    rc = _record(eqkind, fs, "full", tag)
    if tag != "top":
        rtop = _record(eqkind, True, "eager", "top")
        assert _rel(re, rtop) > DISC, (
            f"{eqkind}/{tag}: non-discriminating geometry "
            f"(eager record == top-only, rel_l2={_rel(re, rtop):.3f})")
    r = _rel(rc, re)
    assert r < 5e-3, f"{eqkind}/{tag}: c forward vs eager rel_l2={r:.2e}"


@pytest.mark.parametrize("eqkind", ["acoustic", "elastic"])
@pytest.mark.parametrize("fs,tag", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
def test_per_edge_cuda_gradient_matches_eager(eqkind, fs, tag, mode):
    """c adjoint gradient ~ eager across full / bs / ckpt for every face + corner."""
    ge = _grad(eqkind, fs, "eager", tag)
    gc = _grad(eqkind, fs, mode, tag)
    c = _cos(gc, ge)
    r = _rel(gc, ge)
    assert c > 0.999 and r < 3e-3, f"{eqkind}/{tag}/{mode}: cos={c:.5f} rel_l2={r:.2e}"


@pytest.mark.parametrize("eqcls,rtype,rcomp", [
    (Elastic, ["vx", "vz"], 1),   # elastic2d
    (DASMu,   ["vz"],       0),   # DAS-Mu 2D (its own, separate free-surface code)
])
def test_top_free_surface_rayleigh_near_vr(eqcls, rtype, rcomp):
    """PHYSICAL regression for the surface-row shear-stress (``sxz``) free-surface
    fix (Kristek, Moczo & Archuleta 2002, Table 1).  On a homogeneous half-space
    the top free-surface Rayleigh wave is NON-dispersive at the analytic
    ``vR = 0.9194*vs``.  Spuriously zeroing the surface-row ``sxz`` (the old bug —
    its node is a +h/2 *medium* value) slowed it ~6% (mean ~1248 m/s for vs=1445
    -> vR=1329); the fix keeps ``sxz`` live and restores it to ~1% (mean ~1316).
    Asserted within ~3-4% of ``vR`` — fails loudly if the zeroing regresses.

    Runs for both ``Elastic`` and ``DASMu`` (each has its OWN free-surface code).
    impl='c' (production path); eager is covered by the eager-vs-c gradient suite
    (elastic2d/3d + das_mu2d/3d free_surface scenarios) — a revert on either
    backend is caught (c here, or eager via consistency)."""
    n, dh, dt, nt = 140, 10.0, 0.7e-3, 1500
    vs = 2500.0 / 1.73
    vR = 0.9194 * vs
    eq = eqcls(spatial_order=SO, device=DEV, backend="torch")
    p = PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                  boundary_saving_config={"enabled": False},
                  shape=(n, n), free_surface=["top"], abcn=ABCN, dh=dh, dt=dt,
                  source_type=["sxx", "szz"], receiver_type=rtype)
    vp = np.full((n, n), 2500.0, np.float32)
    models = [torch.tensor(vp, device=DEV),
              torch.tensor((vp / 1.73).astype(np.float32), device=DEV),
              torch.full((n, n), 2000.0, device=DEV)]
    t = np.arange(nt, dtype=np.float32) * dt - 0.05
    wav = torch.tensor((1e3 * ricker(t, f=12.0)).astype(np.float32), device=DEV)
    cen = n // 2
    xs = np.arange(int(0.08 * n), n - int(0.08 * n), 1, dtype=np.int64)
    src = np.array([[cen, 2]], dtype=np.int64)
    rec = np.stack([xs, np.full_like(xs, 2)], -1)[None]
    g = p(wav, src, rec, models=models).detach().cpu().numpy()[0, :, :, rcomp]  # vz
    # phase-shift (Park) dispersion; mean picked phase velocity over 6-16 Hz
    off = (xs - cen) * dh; msk = (xs - cen) > 3; gg = g[:, msk]; offm = off[msk]
    U = np.fft.rfft(gg, axis=0); fall = np.fft.rfftfreq(nt, dt)
    fr = np.arange(5, 20, 0.5); ve = np.arange(900.0, 1700.0, 3.0)
    pick = []
    for f in fr:
        fi = int(np.argmin(np.abs(fall - f))); Un = U[fi] / (np.abs(U[fi]) + 1e-30)
        e = [np.abs(np.sum(Un * np.exp(1j * 2 * np.pi * f * offm / v))) for v in ve]
        pick.append(ve[int(np.argmax(e))])
    pick = np.array(pick); band = (fr >= 6) & (fr <= 16)
    mean_vr = float(pick[band].mean())
    assert 0.96 * vR < mean_vr < 1.05 * vR, (
        f"top-FS Rayleigh mean={mean_vr:.0f} m/s vs analytic vR={vR:.0f} "
        f"(a ~6%-slow value ~{0.94*vR:.0f} means the surface-row sxz zeroing regressed)")


@pytest.mark.parametrize("eqcls", [Elastic3D, DASMu3D])
def test_top_free_surface_rayleigh_near_vr_3d(eqcls):
    """PHYSICAL regression for the 3-D Robertsson tangential free-surface
    correction.  At a z-low free surface ``sigma_zz=0`` implies
    ``dvz_dz = -lam/(lam+2mu) (dvx_dx + dvy_dy)``, so the surface-row tangential
    normal stresses reduce to ``sxx = coef*dvx_dx + coef2*dvy_dy`` (x<->y for
    ``syy``), with ``coef = 4 mu (lam+mu)/(lam+2mu)``, ``coef2 = 2 lam mu/(lam+2mu)``.
    Without it the image-method mirror alone leaves the surface sxx/syy too large
    and the Rayleigh wave runs ~15% SLOW (measured 1118 m/s vs vR=1329); with it
    the 3-D speed matches the 2-D solver (~1314, -1%).

    Measured by time-domain moveout of the |vz| peak over far offsets -- the
    phase-shift estimator is fine too but moveout is the sharper regression
    signal here.  Asserted within 4% of the analytic ``vR``; a revert of the
    Robertsson correction lands at ~-8% moveout and fails loudly.
    """
    nz, ny, nx = 48, 56, 150
    dh, dt, nt, abcn, dc = 10.0, 0.7e-3, 1100, 15, 2
    vs = 2500.0 / 1.73
    vR = 0.9194 * vs
    eq = eqcls(spatial_order=SO, device=DEV, backend="torch")
    p = PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                  boundary_saving_config={"enabled": False},
                  shape=(nz, ny, nx), free_surface=["top"], abcn=abcn, dh=dh, dt=dt,
                  source_type=["sxx", "syy", "szz"], receiver_type=["vx", "vy", "vz"])
    vp = np.full((nz, ny, nx), 2500.0, np.float32)
    models = [torch.tensor(vp, device=DEV),
              torch.tensor((vp / 1.73).astype(np.float32), device=DEV),
              torch.full((nz, ny, nx), 2000.0, device=DEV)]
    t = np.arange(nt, dtype=np.float32) * dt - 0.05
    wav = torch.tensor((1e3 * ricker(t, f=12.0)).astype(np.float32), device=DEV)
    cx, cy = nx // 2, ny // 2
    xs = np.arange(cx + 2, nx - abcn - 2, 1, dtype=np.int64)
    src = np.array([[cx, cy, dc]], dtype=np.int64)          # [x, y, z]; z = depth
    rec = np.stack([xs, np.full_like(xs, cy), np.full_like(xs, dc)], -1)[None]
    g = p(wav, src, rec, models=models).detach().cpu().numpy()[0, :, :, 2]   # vz
    off = (xs - cx) * dh
    tpk = t[np.argmax(np.abs(g), axis=0)]
    far = off > 200
    slope = np.linalg.lstsq(np.vstack([off[far], np.ones(far.sum())]).T,
                            tpk[far], rcond=None)[0][0]
    mean_vr = float(1.0 / slope)
    assert 0.96 * vR < mean_vr < 1.04 * vR, (
        f"3-D top-FS Rayleigh moveout={mean_vr:.0f} m/s vs analytic vR={vR:.0f} "
        f"(~{0.92*vR:.0f} means the Robertsson tangential FS correction regressed)")

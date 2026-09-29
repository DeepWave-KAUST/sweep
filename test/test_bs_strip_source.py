"""Boundary saving must not inject a source twice when it sits in a restore strip.

The boundary-saving backward rebuilds the forward field in reverse. Each step,
the NOPML kernel produces ``w^{it-1} - s^{it}`` everywhere (the reversed
leapfrog without the source term); ``restore_backward`` then overwrites the
restore strips -- ``save_width = M + 1`` cells inward from every non-cut face of
the physical box (``M = spatial_order // 2``; ElasticTTI2nd restores with offset
``-M``, so its strip holds only the outermost physical row/column), the
free-surface face included -- with the saved TRUE ``w^{it-1}``, which already
contains the source. The strip is imaged from those time levels and then
``add_source`` adds ``s^{it}`` once more. A source cell inside a strip thus
carries the source twice, the strip's ``u_tt`` picks up ``s/dt^2``, and the
NOPML stencil leaks the error into the interior on the next step.

Invariant: ``BoundarySaving`` and ``Full()`` store the SAME forward field, so
the model gradients and the source illumination they return agree to the
reconstruction's own floor wherever the source is. The only knob varied against
the out-of-strip controls is where the source sits.

Criterion, cell by cell over the whole physical grid, for every differentiated
model and for ``source_illumination`` (where the backward produces one):
``max|x_bs - x_full| / max|x_full| <= tol``. The source cells are deliberately
NOT masked: the defect lives exactly there, and a source mask -- right for a
c-vs-eager comparison -- would hide it.

Measured on the unfixed core (21b68328, fp32 gpu storage):

* Acoustic / Acoustic3D / AcousticLSRTM / AcousticLSRTM3D: in-strip sources
  3.7e-4..2.5e-2 (illumination 1.3e-2..1.4e-2); out-of-strip controls
  1e-8..3.2e-6 (the top of that range is order 8 + free surface with the
  source 5-7 cells from the left edge; <= 9e-7 elsewhere). ``tol = 1e-5``.
* ElasticTTI2nd: in-strip (physical row/col 0, row nz-1) 3.3e-2..2.3e-1; its
  controls sit at 1.0e-5..1.2e-4 -- a deterministic floor of its own
  reconstruction (Full vs Full is bit-identical), largest on physical row 0
  whether or not the source is near it, so ``1e-5`` is not attainable there and
  it gets ``tol = 1e-3``: 8x above that floor, 33x below the smallest red.

Illumination: ``AcousticLSRTM``/``AcousticLSRTM3D``/``ElasticTTI2nd`` return an
all-zero ``source_illumination`` from the gradient backward under both
strategies (it is an RTM-mode output there), so for them only the gradients are
compared -- but a non-zero illumination from either strategy is compared too,
so wiring one up later cannot slip past this file.

The compiled CPU engine (``SWEEP_JIT_FULL=1``) carried the same defect in its
own boundary-saving backward; its group is at the end of this file and skips,
saying why, wherever that engine is not what ``PropTorch(dev='cpu',
impl='c')`` runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from conftest import requires_binding, ricker
from sweep.equations import (
    Acoustic,
    Acoustic3D,
    AcousticLSRTM,
    AcousticLSRTM3D,
    ElasticTTI2nd,
)
from sweep.propagator.options import BoundarySaving, Full
from sweep.propagator.torch import PropTorch

DEV = "cuda"
DH, DT, ABCN = 10.0, 1.0e-3, 16
FM, DELAY = 12.0, 0.08
SHAPE2D, NT2D = (48, 64), 350          # (nz, nx)
SHAPE3D, NT3D = (24, 20, 28), 300      # (nz, ny, nx)

BINDINGS = ("acoustic2d_backward_bs", "acoustic3d_backward_bs",
            "acoustic_lsrtm2d_backward_bs", "acoustic_lsrtm3d_backward_bs",
            "elastic_tti_2nd2d_backward_bs")


@dataclass(frozen=True)
class Eq:
    key: str
    cls: type
    ndim: int
    source_type: tuple
    receiver_type: tuple
    strip_offset_in_M: int     # restore strip starts at the physical edge + offset*M
    illum: bool                # the gradient backward produces source_illumination
    orders: tuple
    tol: float


EQS = (
    Eq("acoustic2d", Acoustic, 2, ("h1",), ("h1",), 0, True, (4, 8), 1e-5),
    Eq("acoustic3d", Acoustic3D, 3, ("h1",), ("h1",), 0, True, (4,), 1e-5),
    Eq("lsrtm2d", AcousticLSRTM, 2, ("h1",), ("sh1",), 0, False, (4, 8), 1e-5),
    Eq("lsrtm3d", AcousticLSRTM3D, 3, ("h1",), ("sh1",), 0, False, (4,), 1e-5),
    # restore_backward_2d_field(..., offset=-M): strip = physical row/col 0 (and
    # nz-1 / nx-1) plus M pad cells outside the box. tol: see module docstring.
    Eq("tti2nd2d", ElasticTTI2nd, 2, ("uz",), ("ux", "uz"), -1, False, (4, 8), 1e-3),
)
EQ = {e.key: e for e in EQS}


def _shape(eq):
    return SHAPE2D if eq.ndim == 2 else SHAPE3D


def _in_strip(pos, shape, M, offset):
    """Python mirror of boundary_kernel2d/3d's band predicate for one device
    (no DD cut faces, tangent_pad 0), in physical coordinates."""
    width = M + 1
    for p, n in zip(pos, shape):
        if offset <= p < offset + width or n - offset - width <= p < n - offset:
            return True
    return False


def _positions(eq, order, fs, shape=None):
    """(label, physical (z, x) / (z, y, x)) source positions for one config."""
    M = order // 2
    shape = _shape(eq) if shape is None else shape
    nz, nx = shape[0], shape[-1]
    zm, xm = nz // 2, nx // 2

    def at(z, x, y=None):
        if eq.ndim == 2:
            return (z, x)
        return (z, shape[1] // 2 if y is None else y, x)

    out = []
    if eq.strip_offset_in_M < 0:           # TTI2nd: strip = outermost row/col
        out += [("top-z0", at(0, xm)), ("left-x0", at(zm, 0)),
                ("bottom-z%d" % (nz - 1), at(nz - 1, xm)),
                ("top-z1", at(1, xm)), ("left-x1", at(zm, 1))]
    else:
        tops = (1, M) if fs else (1, 2)
        out += [("top-z%d" % z, at(z, xm)) for z in sorted(set(tops))]
        out += [("left-x1", at(zm, 1))]
        if eq.ndim == 3:
            out += [("front-y1", at(zm, xm, y=1))]
    out += [("bottom-z%d" % (nz - 2), at(nz - 2, xm)),
            ("top-z%d" % (M + 2), at(M + 2, xm)),
            ("left-x%d" % (M + 2), at(zm, M + 2)),
            ("mid-z%d" % zm, at(zm, xm))]
    return out


def _cases():
    cases = []
    for eq in EQS:
        for order in eq.orders:
            for fs in (False, True):
                M = order // 2
                for label, pos in _positions(eq, order, fs):
                    inside = _in_strip(pos, _shape(eq), M, eq.strip_offset_in_M * M)
                    cid = (f"{eq.key}-o{order}-{'fs' if fs else 'nofs'}-"
                           f"{'strip' if inside else 'ctrl'}-{label}")
                    cases.append(pytest.param(eq, order, fs, [pos], id=cid))
    # Several shots in one call, each in a DIFFERENT strip: a fix that handles
    # only shot 0, or reads another shot's source, stays red here.
    nz, nx = SHAPE2D
    cases.append(pytest.param(EQ["acoustic2d"], 4, False,
                              [(1, nx // 2), (nz // 2, 1), (nz - 2, nx // 3)],
                              id="acoustic2d-o4-nofs-strip-B3-top-left-bottom"))
    cases.append(pytest.param(EQ["acoustic2d"], 4, False,
                              [(4, nx // 2), (nz // 2, 4), (nz // 2, nx // 3)],
                              id="acoustic2d-o4-nofs-ctrl-B3-interior"))
    return cases


def _models(eq, shape):
    """(true, init, requires_grad) lists. Heterogeneous on purpose: a
    homogeneous model hides operator defects and makes relative numbers
    meaningless."""
    idx = np.indices(shape).astype(np.float32)
    z = idx[0] / (shape[0] - 1)
    vp = (1800.0 + 800.0 * z
          + 40.0 * np.sin(idx[-1] / 5.0) * np.sin(idx[0] / 7.0)).astype(np.float32)
    box = np.ones(shape, bool)
    for ax, n in enumerate(shape):
        keep = np.zeros(n, bool)
        keep[n // 3: max(n // 3 + 2, (2 * n) // 3)] = True
        box &= keep.reshape([-1 if i == ax else 1 for i in range(len(shape))])
    box = box.astype(np.float32)

    if eq.key.startswith("lsrtm"):
        # Non-zero initial reflectivity, so the Born forward is exercised.
        return [vp, 0.08 * box], [vp, 0.04 * box], [False, True]
    if eq.key.startswith("tti2nd"):
        rho = (1000.0 + 200.0 * z).astype(np.float32)
        eps = np.full(shape, 0.12, np.float32)
        eta = np.full(shape, 0.06, np.float32)
        the = np.full(shape, 0.35, np.float32)
        vh = vp + 150.0 * box
        true = [vh, vh / 1.9, rho + 60.0 * box, eps, eta, the]
        init = [vp, vp / 1.9, rho, eps, eta, the]
        return ([a.astype(np.float32) for a in true],
                [a.astype(np.float32) for a in init],
                [True, True, True, False, False, False])
    return [vp + 150.0 * box], [vp], [True]


def _geometry(eq, order, phys_positions, shape=None):
    """sweep order: sources (nshot, ndim) as (x, z) / (x, y, z); receivers
    (nshot, nrec, ndim) on a shallow line just below the widest strip."""
    M = order // 2
    shape = _shape(eq) if shape is None else shape
    nx = shape[-1]
    zr = M + 3
    src = np.array([tuple(reversed(p)) for p in phys_positions], np.int64)
    rx = np.arange(M + 2, nx - M - 2, 3)
    if eq.ndim == 2:
        rec1 = np.stack([rx, np.full(rx.size, zr)], 1)
    else:
        ry = np.arange(M + 2, shape[1] - M - 2, 4)
        gy, gx = np.meshgrid(ry, rx[::2], indexing="ij")
        rec1 = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zr)], 1)
    rec = np.repeat(rec1[None], len(phys_positions), 0).astype(np.int64)
    return src, rec


def _run(eq, order, fs, memory, phys_positions, obs=None, dev=DEV, shape=None, nt=None):
    """One forward + backward. Returns (obs, [grad per differentiated model],
    source_illumination); ``obs`` is computed with this propagator when None.
    ``dev``/``shape``/``nt`` default to the CUDA suite's; the CPU-engine group
    passes its own."""
    shape = _shape(eq) if shape is None else shape
    if nt is None:
        nt = NT2D if eq.ndim == 2 else NT3D
    prop = PropTorch(eq.cls(spatial_order=order, device=dev), backend="torch",
                     impl="c", shape=shape, dh=DH, dt=DT, nt=nt, abcn=ABCN,
                     dev=dev, free_surface=fs, memory=memory,
                     source_type=list(eq.source_type),
                     receiver_type=list(eq.receiver_type))
    assert prop.impl == "c", prop.impl
    prop.compute_illumination = True
    src, rec = _geometry(eq, order, phys_positions, shape)
    wav = torch.tensor(ricker(nt, DT, FM, DELAY, scale=1e3), device=dev)
    true, init, flags = _models(eq, shape)
    if obs is None:
        with torch.no_grad():
            obs = prop(wav, src, rec,
                       models=[torch.tensor(a, device=dev) for a in true]).detach()
    ms = [torch.tensor(a, device=dev, requires_grad=g) for a, g in zip(init, flags)]
    syn = prop(wav, src, rec, models=ms)
    ((syn - obs) ** 2).sum().backward()
    grads = [m.grad.detach().double().cpu() for m, g in zip(ms, flags) if g]
    si = getattr(prop, "source_illumination", None)
    illum = None if si is None else si.detach().double().cpu()
    del prop
    return obs, grads, illum


def _rel(a, b):
    """max|a - b| / max|b|, cell by cell over the physical grid."""
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)


def _is_zero(t):
    return t is None or float(t.abs().max()) == 0.0


@requires_binding(*BINDINGS)
@pytest.mark.parametrize("eq,order,fs,positions", _cases())
def test_boundary_saving_matches_full_with_source_in_restore_strip(eq, order, fs, positions):
    if fs and not eq.cls.supports_free_surface:
        pytest.skip(f"{eq.cls.__name__}.supports_free_surface is False: an "
                    "anisotropic medium refuses the isotropic image free surface")

    obs, g_full, i_full = _run(eq, order, fs, Full(), positions)
    _, g_bs, i_bs = _run(eq, order, fs, BoundarySaving(storage="gpu"), positions, obs=obs)

    grad_rel = [_rel(b, f) for b, f in zip(g_bs, g_full)]
    both_zero = _is_zero(i_full) and _is_zero(i_bs)
    if eq.illum:
        assert not both_zero, (
            f"{eq.cls.__name__}: compute_illumination=True returned no "
            f"source_illumination under either strategy")
    illum_rel = None if both_zero else _rel(i_bs, i_full)

    k = int(np.argmax(grad_rel))
    d = (g_bs[k] - g_full[k]).abs()
    where = tuple(int(i) for i in np.unravel_index(int(d.argmax()), tuple(d.shape)))
    print(f"[bs-strip] grad_rel={' '.join(f'{r:.2e}' for r in grad_rel)} "
          f"illum_rel={'n/a' if illum_rel is None else f'{illum_rel:.2e}'} "
          f"argmax={where} src(phys)={positions}")

    assert grad_rel[k] <= eq.tol, (
        f"{eq.cls.__name__} order {order} fs={fs}, source(s) at physical "
        f"{positions}: the BoundarySaving gradient differs from Full by "
        f"max|d|/max|g_full| = {grad_rel[k]:.3e} > {eq.tol:.0e} (per model: "
        f"{', '.join(f'{r:.2e}' for r in grad_rel)}), worst cell {where}. A "
        f"source inside a restore strip is injected twice by the "
        f"boundary-saving reconstruction.")
    if illum_rel is not None:
        assert illum_rel <= eq.tol, (
            f"{eq.cls.__name__} order {order} fs={fs}, source(s) at physical "
            f"{positions}: source_illumination differs between BoundarySaving "
            f"and Full by rel {illum_rel:.3e} > {eq.tol:.0e}.")


# --------------------------------------------------------------------------- #
# Other restore-kernel variants
# --------------------------------------------------------------------------- #
# restore_backward_2d/3d dispatch on the storage: fp32 on gpu/cpu/async-disk ->
# the compact band kernel, fp32 on synchronous disk -> the full-grid scan kernel,
# fp16/int8 -> dequantise + scan kernel, bf16 -> its own kernels. Any fix has to
# agree with every one of them about which cells are "in the strip".
#
# fp32 variants keep the fp32 floor (out-of-strip controls measured identical on
# gpu/cpu/disk: 2.5e-7..8.3e-7), so they take the same whole-grid bar. The
# reduced dtypes do not: their quantisation floor on the whole grid (grad up to
# 4.3e-4, illumination up to 8.5e-3 for int8) overlaps the defect's 6.7e-4..2.5e-2.
# At the SOURCE CELL it does not: the defect puts 4.4e-3..1.9e-2 there in the
# cases below (relative to that cell's own |g_full|, which is also the grid
# max), while quantisation alone -- measured as BS(dtype) vs BS(fp32) on the
# unfixed core, where the double injection is identical and cancels -- puts at
# most 5.2e-5 (bf16), 4.5e-5 (int8), 1.3e-6 (fp16) over 12 in-strip configs.
# The bar 2e-4 sits 4x above the worst of those and 22x below the smallest
# defect here.
LOCAL_TOL = 2e-4

STORAGE_VARIANTS = (
    ("cpu-fp32", dict(storage="cpu", transfer_interval=4), "grid"),
    ("disk-fp32", dict(storage="disk", transfer_interval=4), "grid"),   # sync -> scan kernel
    ("gpu-fp16", dict(storage="gpu", storage_dtype="fp16"), "srccell"),
    ("gpu-bf16", dict(storage="gpu", storage_dtype="bf16"), "srccell"),
    ("gpu-int8", dict(storage="gpu", storage_dtype="int8"), "srccell"),
)
STORAGE_CASES = (
    ("acoustic2d-o4-nofs-strip-left-x1", "acoustic2d", False, [(SHAPE2D[0] // 2, 1)]),
    ("acoustic2d-o4-fs-strip-top-z1", "acoustic2d", True, [(1, SHAPE2D[1] // 2)]),
    ("acoustic3d-o4-nofs-strip-front-y1", "acoustic3d", False,
     [(SHAPE3D[0] // 2, 1, SHAPE3D[2] // 2)]),
    ("lsrtm2d-o4-fs-strip-top-z1", "lsrtm2d", True, [(1, SHAPE2D[1] // 2)]),
)


@requires_binding(*BINDINGS)
@pytest.mark.parametrize("label,kw,metric", STORAGE_VARIANTS, ids=[v[0] for v in STORAGE_VARIANTS])
@pytest.mark.parametrize("cid,eqkey,fs,positions", STORAGE_CASES, ids=[c[0] for c in STORAGE_CASES])
def test_every_restore_variant_agrees_on_the_strip(cid, eqkey, fs, positions, label, kw, metric, tmp_path):
    eq = EQ[eqkey]
    kw = dict(kw)
    if kw["storage"] == "disk":
        kw["disk_dir"] = str(tmp_path)
    obs, g_full, i_full = _run(eq, 4, fs, Full(), positions)
    _, g_bs, i_bs = _run(eq, 4, fs, BoundarySaving(**kw), positions, obs=obs)

    if metric == "grid":
        grad_rel = max(_rel(b, f) for b, f in zip(g_bs, g_full))
        illum_rel = None if (_is_zero(i_full) and _is_zero(i_bs)) else _rel(i_bs, i_full)
        print(f"[bs-strip:{label}] grid grad_rel={grad_rel:.2e} "
              f"illum_rel={'n/a' if illum_rel is None else f'{illum_rel:.2e}'}")
        assert grad_rel <= eq.tol, (
            f"{cid} with {label} storage: gradient differs from Full by "
            f"{grad_rel:.3e} > {eq.tol:.0e}")
        if illum_rel is not None:
            assert illum_rel <= eq.tol, (
                f"{cid} with {label} storage: source_illumination differs from "
                f"Full by {illum_rel:.3e} > {eq.tol:.0e}")
        return

    cells = [tuple(p) for p in positions]
    local = max(float((b[c] - f[c]).abs()) / max(float(f[c].abs()), 1e-30)
                for b, f in zip(g_bs, g_full) for c in cells)
    print(f"[bs-strip:{label}] source-cell grad_rel={local:.2e}")
    assert local <= LOCAL_TOL, (
        f"{cid} with {label} storage: at the source cell the gradient differs "
        f"from Full by {local:.3e} of its own value (> {LOCAL_TOL:.0e}, 4x the "
        f"{label.split('-')[1]} quantisation floor there). The source is being "
        f"injected twice through this restore variant.")


# --------------------------------------------------------------------------- #
# The compiled CPU engine (csrc/cpu/**)
# --------------------------------------------------------------------------- #
# The CPU engine's boundary-saving backward (backward_*_bs_impl in
# csrc/cpu/equations/{acoustic2d,acoustic3d,acoustic_lsrtm2d,acoustic_lsrtm3d})
# runs the same reverse step -- NOPML, restore_*_boundary(_disk), the strip's
# difference u_tt imaged, add_source -- so it carried the same defect. That
# engine exists only in the pybind shim (SWEEP_JIT_FULL=1, or an AOT-built
# sweep._C): on the default ctypes path PropTorch(dev='cpu', impl='c')
# resolves to eager and this group skips.
#
# What differs from the CUDA group above, and why:
#
# * Small grids, order 4 only (CPU_SHAPE / CPU_NT).
# * Free surface: none of the four has a free-surface backward on the CPU
#   engine (can_use_*_raw_backward refuses free_surface, and the forward falls
#   to the generic engine, which keeps no history / last_two). The fs=True
#   cases skip on the error the engine raises, quoting it, and start running
#   by themselves once the engine grows one.
# * Illumination is not compared. The CPU engine's BS backward accumulates
#   sum(u^2) of the reconstructed forward field, its Full backward
#   sum((vp^2 Lap u)^2): two different quantities, rel ~1.0 on every case,
#   out-of-strip controls included, with or without this fix. That split is
#   pre-existing and unrelated to the strip, so only gradients are compared.
# * The Full gradient must be non-zero. When the entries stopped returning
#   their outputs (51871c0f) every CPU record and gradient came back as the
#   zeros the caller had bound, and zero-vs-zero passes any relative bar.
#
# Measured (fs=False; outputs bound through cpu_binding.cpp):
#
# * unfixed engine: in-strip sources 1.6e-3..2.8e-1 (acoustic2d
#   1.9e-3..8.2e-3, acoustic3d 3.3e-3..2.8e-1, lsrtm2d 2.1e-3..8.0e-3,
#   lsrtm3d 1.6e-3..5.4e-2; disk storage the same as in-memory); controls
#   3.4e-8..4.7e-7 -- except Acoustic, whose controls sat at 1.1e-5..6.4e-4
#   for a reason of its own: its Full store imaged vp^2 * bare Laplacian,
#   while the CPML terms still act on the first M physical rows/cols, so
#   there it was not the u_tt of the update that the BS strip images
#   (acoustic_lsrtm2d images the full operator; the CUDA core's physical box
#   has no CPML terms, so its vp^2 * Lap(u) store is the update). Fixed with
#   it; the Full gradient moved on those 2*M rows/cols only, bit-identical
#   inside.
# * fixed engine: every case here 3.4e-8..1.9e-6 (the top is acoustic3d with
#   the source at y=1); controls' BS gradients bit-identical to the unfixed
#   ones. A sweep of the source over every cell within M+1 of an edge (2-D,
#   512 positions per equation, 396 of them in a strip) finds a floor of its
#   own at the source cell: up to 1.06e-5 (acoustic2d) / 1.30e-5 (lsrtm2d) in
#   a strip and 5.6e-6 / 8.2e-6 outside one, where the fix changes nothing --
#   fp32 cancellation in the difference u_tt of the large field there. The
#   3-D sweep (acoustic3d, 728 positions: every combination of
#   {0..3, mid, n-4..n-1} per axis) tops out at 5.2e-6.
#
# CPU_TOL sits ~4x above that floor and 32x below the smallest red here.
CPU_SHAPE = {2: (32, 40), 3: (16, 14, 18)}    # (nz, nx) / (nz, ny, nx)
CPU_NT = {2: 150, 3: 100}
CPU_EQS = ("acoustic2d", "acoustic3d", "lsrtm2d", "lsrtm3d")
CPU_ORDER = 4
CPU_TOL = 5e-5


def _cpu_engine_status():
    """(ok, reason): is PropTorch(dev='cpu', impl='c') the compiled CPU engine
    in this process? Answers without compiling anything."""
    import warnings

    import sweep
    from sweep.backend.c import jit
    from sweep.backend.torch import binding
    from sweep.equations import Acoustic as _Acoustic

    if not (jit.jit_full() or sweep._prebuilt_binding_present()):
        return False, ("the compiled CPU engine (csrc/cpu/**) is built only under "
                       "SWEEP_JIT_FULL=1 or into an AOT sweep._C; the default ctypes "
                       "path serves the CUDA core alone")
    diag = binding.diagnostics()
    if not diag["usable"]:
        return False, f"compiled sweep._C unavailable: {diag['reason']}"
    if not diag["prebuilt"] and diag["shim"] != "pybind":
        return False, (f"sweep._C is the {diag['shim']} shim, not the pybind one "
                       "that carries the CPU engine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prop = PropTorch(_Acoustic(spatial_order=CPU_ORDER, device="cpu"), backend="torch",
                         impl="c", shape=CPU_SHAPE[2], dh=DH, dt=DT, nt=8, abcn=ABCN,
                         dev="cpu")
    if prop.impl != "c":
        return False, (f"PropTorch(dev='cpu', impl='c') resolved to impl={prop.impl!r}"
                       + "".join(f"; {w.message}" for w in caught))
    return True, ""


@pytest.fixture(scope="module")
def cpu_engine():
    ok, reason = _cpu_engine_status()
    if not ok:
        pytest.skip(reason)


def _cpu_cases():
    cases = []
    for key in CPU_EQS:
        eq = EQ[key]
        shape = CPU_SHAPE[eq.ndim]
        M = CPU_ORDER // 2
        for fs in (False, True):
            for label, pos in _positions(eq, CPU_ORDER, fs, shape):
                inside = _in_strip(pos, shape, M, eq.strip_offset_in_M * M)
                cid = (f"cpu-{key}-o{CPU_ORDER}-{'fs' if fs else 'nofs'}-"
                       f"{'strip' if inside else 'ctrl'}-{label}")
                cases.append(pytest.param(eq, fs, [pos], "cpu", id=cid))
    nz, nx = CPU_SHAPE[2]
    _, ny3, nx3 = CPU_SHAPE[3]
    # Several shots, each in a different strip: per-shot indexing of the fix.
    cases.append(pytest.param(EQ["acoustic2d"], False,
                              [(1, nx // 2), (nz // 2, 1), (nz - 2, nx // 3)], "cpu",
                              id="cpu-acoustic2d-o4-nofs-strip-B3-top-left-bottom"))
    cases.append(pytest.param(EQ["acoustic2d"], False,
                              [(4, nx // 2), (nz // 2, 4), (nz // 2, nx // 3)], "cpu",
                              id="cpu-acoustic2d-o4-nofs-ctrl-B3-interior"))
    # Disk storage restores through restore_*_boundary_disk instead.
    cases.append(pytest.param(EQ["acoustic2d"], False, [(nz // 2, 1)], "disk",
                              id="cpu-acoustic2d-o4-nofs-strip-left-x1-disk"))
    cases.append(pytest.param(EQ["lsrtm3d"], False, [(1, ny3 // 2, nx3 // 2)], "disk",
                              id="cpu-lsrtm3d-o4-nofs-strip-top-z1-disk"))
    cases.append(pytest.param(EQ["lsrtm3d"], False, [(4, ny3 // 2, nx3 // 2)], "disk",
                              id="cpu-lsrtm3d-o4-nofs-ctrl-top-z4-disk"))
    return cases


@pytest.mark.parametrize("eq,fs,positions,storage", _cpu_cases())
def test_cpu_engine_boundary_saving_matches_full_with_source_in_restore_strip(
        cpu_engine, eq, fs, positions, storage, tmp_path):
    shape = CPU_SHAPE[eq.ndim]
    nt = CPU_NT[eq.ndim]
    kw = dict(storage=storage)
    if storage == "disk":
        kw.update(disk_dir=str(tmp_path), transfer_interval=4)
    run = dict(dev="cpu", shape=shape, nt=nt)
    try:
        obs, g_full, _ = _run(eq, CPU_ORDER, fs, Full(), positions, **run)
        _, g_bs, _ = _run(eq, CPU_ORDER, fs, BoundarySaving(**kw), positions, obs=obs, **run)
    except RuntimeError as exc:
        # Skip only the engine's own "no free-surface backward" refusals, by
        # their text; anything else under a free surface is a real failure.
        known = ("the engine produced nothing",
                 "requires the handwritten raw float32 path")
        if not fs or not any(k in str(exc) for k in known):
            raise
        pytest.skip(f"{eq.cls.__name__}: the CPU engine has no free-surface backward "
                    f"here: {str(exc).splitlines()[0]}")

    scale = [float(f.abs().max()) for f in g_full]
    assert min(scale) > 0.0, (
        f"{eq.cls.__name__}: the CPU engine's Full gradient is all zero "
        f"(max per model {scale}); its outputs are not reaching the caller")

    grad_rel = [_rel(b, f) for b, f in zip(g_bs, g_full)]
    k = int(np.argmax(grad_rel))
    d = (g_bs[k] - g_full[k]).abs()
    where = tuple(int(i) for i in np.unravel_index(int(d.argmax()), tuple(d.shape)))
    print(f"[bs-strip:cpu-{storage}] grad_rel={' '.join(f'{r:.2e}' for r in grad_rel)} "
          f"argmax={where} src(phys)={positions}")
    assert grad_rel[k] <= CPU_TOL, (
        f"CPU engine {eq.cls.__name__} order {CPU_ORDER} fs={fs} ({storage} storage), "
        f"source(s) at physical {positions}: the BoundarySaving gradient differs from "
        f"Full by max|d|/max|g_full| = {grad_rel[k]:.3e} > {CPU_TOL:.0e} (per model: "
        f"{', '.join(f'{r:.2e}' for r in grad_rel)}), worst cell {where}. A source "
        f"inside a restore strip is injected twice by the boundary-saving "
        f"reconstruction.")

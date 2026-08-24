"""Bit-exactness of the stepped forward API (``it_begin``/``it_end``).

Strategy: run the public propagator once only to capture a fully-populated
``ForwardInput`` (and force buffer allocation), then replay through the raw
binding twice from zeroed state — once as a single full-range call
(reference), once split into ``k``-step segments with the wavefield list
rotated per :func:`sweep.propagator._stepped.rotate_wavefield_roles` — and
require ``torch.equal`` on every output and every piece of persistent state.
Same kernels, same launch order, same absolute time indices: any mismatch is
a bug, not tolerance.

All cases use a single source: multi-source same-cell atomicAdd would make
even two identical full runs bitwise-nondeterministic.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic, Acoustic3D  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    acoustic_psi_pairs,
    rotate_wavefield_roles,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NT = 120


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def build(ndim, *, abcn=8, free_surface=False, bs=None, use_ckpt=False,
          ckpt_mode="chunk", ckpt_count=0, requires_grad=False, nt=NT):
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    eq_cls = Acoustic if ndim == 2 else Acoustic3D
    equation = eq_cls(spatial_order=4, device=DEV, backend="torch")

    boundary_saving_config = {"enabled": False}
    if bs == "gpu":
        boundary_saving_config = {
            "enabled": True, "storage": "gpu",
            "transfer_interval": 1, "pinned_memory": False,
        }
    elif bs == "cpu":
        boundary_saving_config = {
            "enabled": True, "storage": "cpu",
            "transfer_interval": 4, "ring_buffers": 2, "pinned_memory": False,
        }

    prop = PropTorch(
        equation,
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=0.0015,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=free_surface,
        pml_type="cpmlr",
        nt=nt,
        B=1,
        use_ckpt=use_ckpt,
        ckpt_mode=ckpt_mode,
        ckpt_count=ckpt_count,
        boundary_saving_config=boundary_saving_config,
    )

    if ndim == 2:
        nz, nx = shape
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None]
        vp = np.broadcast_to(vp, shape).copy()
        sources = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )
    else:
        nz, ny, nx = shape
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None, None]
        vp = np.broadcast_to(vp, shape).copy()
        sources = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, ny // 2, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )

    models = [torch.tensor(vp, device=DEV, requires_grad=requires_grad)]
    wavelet = ricker(nt, 0.0015)
    return prop, wavelet, sources, receivers, models


def capture(prop):
    """Wrap the compiled propagator's forward_func so the populated
    ForwardInput is kept (PropTorch delegates to ``_backend_impl``)."""
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(params):
        out = orig(params)
        cap["params"] = params
        cap["raw_out"] = out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def _zero_state(p, L, *, record, u_allt=None, has_bs=False, has_ckpt=False):
    for t in L:
        t.zero_()
    record.zero_()
    if u_allt is not None:
        u_allt.zero_()
    if has_bs:
        for t in p.boundary_gpu:
            t.zero_()
        for t in p.boundary_cpu:
            t.zero_()
        if p.last_two is not None:
            p.last_two.zero_()
    if has_ckpt:
        for t in p.checkpoints:
            t.zero_()


def _replay(func, p, L, schedule, psi_pairs):
    for a, b in schedule:
        p.it_begin, p.it_end = a, b
        p.wavefields = rotate_wavefield_roles(L, a, psi_pairs=psi_pairs)
        func(p)
    # restore full-range defaults so later replays start clean
    p.it_begin, p.it_end = 0, -1
    p.wavefields = L


def _snapshot(p, L, *, record, u_allt=None, has_bs=False, has_ckpt=False):
    snap = {
        "wf": [t.clone() for t in L],
        "record": record.clone(),
    }
    if u_allt is not None:
        snap["u_allt"] = u_allt.clone()
    if has_bs:
        snap["bs_gpu"] = [t.clone() for t in p.boundary_gpu]
        snap["bs_cpu"] = [t.clone() for t in p.boundary_cpu]
        snap["last_two"] = p.last_two.clone()
    if has_ckpt:
        snap["ckpt"] = [t.clone() for t in p.checkpoints]
    return snap


def _assert_equal(snap_ref, snap_step):
    for key in snap_ref:
        ref, got = snap_ref[key], snap_step[key]
        if isinstance(ref, list):
            for i, (r, g) in enumerate(zip(ref, got)):
                assert torch.equal(g, r), f"{key}[{i}] differs (bitwise)"
        else:
            assert torch.equal(got, ref), f"{key} differs (bitwise)"


def run_equivalence(ndim, k, **build_kw):
    prop, wavelet, sources, receivers, models = build(ndim, **build_kw)
    cap = capture(prop)
    with torch.no_grad() if not build_kw.get("requires_grad") else _nullcontext():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    want = 9 if ndim == 2 else 12  # PML + psi double-buffer layout
    if not L:
        # Pure forward runs let C++ allocate internally; bind our own
        # tensors so state persists across stepped calls.
        L = [torch.zeros_like(p.models[0]) for _ in range(want)]
    assert len(L) == want, f"expected psi-double-buffer layout, got {len(L)}"
    psi_pairs = acoustic_psi_pairs(ndim)

    nt = int(p.nt)
    has_bs = bool(p.use_boundary_saving)
    has_ckpt = bool(p.use_checkpoint)

    raw_record = cap["raw_out"][2]
    record = torch.zeros_like(raw_record)
    p.record_out = record

    u_allt = None
    if p.save_all_wavefields:
        u_allt = torch.zeros_like(cap["raw_out"][0])
        p.u_allt_out = u_allt

    state_kw = dict(record=record, u_allt=u_allt, has_bs=has_bs, has_ckpt=has_ckpt)

    # Reference: one full-range call from zeroed state.
    _zero_state(p, L, **state_kw)
    _replay(func, p, L, [(0, nt)], psi_pairs)
    ref = _snapshot(p, L, **state_kw)

    # Sanity: the symmetric replay reproduces the captured public run.
    # Bitwise in 2D AND 3D: internal wavefield allocation now opts into the
    # psi double-buffer, so the public run takes the same race-free kernel
    # branch as the Python-bound replay.  (It used to take the legacy
    # in-place psi branch — an intra-launch RAW race, nondeterministic at
    # ulp level in 3-D; see _dd_cuda/REPORT.md.)
    assert torch.equal(record, raw_record)

    # Stepped: same zeroed state, k-step segments (ragged tail included).
    _zero_state(p, L, **state_kw)
    _replay(func, p, L, [(a, min(a + k, nt)) for a in range(0, nt, k)],
            psi_pairs)
    step = _snapshot(p, L, **state_kw)

    _assert_equal(ref, step)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- T1: plain
@cuda_only
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("k", [1, 7, 40])
@pytest.mark.parametrize("pml", [True, False])
def test_stepped_plain_forward(ndim, k, pml):
    run_equivalence(ndim, k, abcn=8 if pml else 0)


# ------------------------------------------------- T2: boundary saving (gpu)
@cuda_only
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("k", [1, 7])
def test_stepped_boundary_saving_gpu(ndim, k):
    run_equivalence(ndim, k, bs="gpu", requires_grad=True)


# ------------------------------------- T3: boundary saving, staged cpu ring
@cuda_only
def test_stepped_boundary_saving_cpu_staged():
    # k=7 deliberately misaligned with transfer_interval=4 chunk boundaries.
    run_equivalence(2, 7, bs="cpu", requires_grad=True)


# ---------------------------------------------------------- T4: free surface
@cuda_only
@pytest.mark.parametrize("k", [1, 7])
def test_stepped_free_surface(k):
    run_equivalence(2, k, free_surface=True)


# --------------------------------------------------- T6: save_all_wavefields
@cuda_only
@pytest.mark.parametrize("k", [1, 7])
def test_stepped_save_all_wavefields(k):
    run_equivalence(2, k, requires_grad=True)


# ------------------------------------------------------- T7: checkpointing
@cuda_only
def test_stepped_checkpoint_chunk():
    run_equivalence(2, 7, use_ckpt=True, requires_grad=True)


@cuda_only
def test_stepped_checkpoint_recursive():
    run_equivalence(2, 7, use_ckpt=True, ckpt_mode="recursive",
                    ckpt_count=6, requires_grad=True)


# --------------------------------------------------------------- T9: guards
@cuda_only
def test_stepped_guards():
    prop, wavelet, sources, receivers, models = build(2, nt=8)
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    record = torch.zeros_like(cap["raw_out"][2])

    # it range out of bounds
    p.record_out = record
    p.it_begin, p.it_end = 0, 9
    with pytest.raises(RuntimeError, match="it_begin <= it_end <= nt"):
        func(p)

    # continuation without wavefields
    p.it_begin, p.it_end = 1, 2
    p.wavefields = []
    with pytest.raises(RuntimeError, match="requires Python-bound wavefields"):
        func(p)
    p.wavefields = L

    # stepped without record_out: needs a params object whose record_out was
    # never assigned (any tensor set through pybind, even empty, is defined)
    prop2, wavelet2, sources2, receivers2, models2 = build(2, nt=8)
    cap2 = capture(prop2)
    with torch.no_grad():
        prop2(wavelet2, sources2, receivers2, models=models2)
    p2 = cap2["params"]
    p2.it_begin, p2.it_end = 0, 4
    with pytest.raises(RuntimeError, match="requires record_out"):
        cap2["func"](p2)

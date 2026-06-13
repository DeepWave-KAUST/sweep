"""Bit-exactness of the stepped forward API for elastic2d / elastic3d.

Same strategy as ``test_stepped_forward.py`` (acoustic): run the public
propagator once only to capture a fully-populated ``ForwardInput``, then
replay through the raw binding twice from zeroed state — once as a single
full-range call (reference), once split into ``k``-step segments — and
require ``torch.equal`` on every output and every piece of persistent state.

Elastic specifics:

* NO buffer-role rotation: every field updates in place at a fixed slot
  (no ``swap`` calls anywhere in elastic2d/3d forward or ``elastic.h``),
  so the SAME wavefield list is bound for every segment
  (``u_blocks=()``, ``psi_pairs=()`` in ``rotate_wavefield_roles`` terms).

* Wavefield list layout (``elastic.h::bind``):
  - 2D, 15 slots: ``vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_sxxx, m_sxxz, m_szzx, m_szzz, m_sxzx, m_sxzz``
  - 3D, 36 slots: ``vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
    m_vxx, m_vxy, m_vxz, m_vyx, m_vyy, m_vyz, m_vzx, m_vzy, m_vzz,
    m_sxxx, m_sxxy, m_sxxz, m_syyx, m_syyy, m_syyz,
    m_szzx, m_szzy, m_szzz, m_sxyx, m_sxyy, m_sxyz,
    m_sxzx, m_sxzy, m_sxzz, m_syzx, m_syzy, m_syzz``

* Sources inject into multiple stress fields, receivers sample velocities
  (smoke-test conventions: ``['sxx','szz']`` / ``['vx','vz']`` in 2D,
  with ``syy``/``vy`` added in 3D), ``pml_type='cpmls'``,
  ``wavelet_scale=1e6``.  Single source point (bitwise determinism).

* Free surface is the elastic image-method z-top handling inside the
  kernels; segmenting the time loop must not interact with it.
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

from sweep.equations import Elastic, Elastic3D  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import rotate_wavefield_roles  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NT = 120
DT = 0.0015
WAVELET_SCALE = 1.0e6

# elastic.h::bind layouts (see module docstring).
NWF = {2: 15, 3: 36}
ELASTIC_ROTATE_KW = dict(u_blocks=(), psi_pairs=())  # no rotation


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def elastic_models(shape, requires_grad=False):
    """Smoke-test recipe: vp 2200+40g, vs 1200+20g, rho 2000+10g."""
    grid = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32)
    grid = grid.reshape(shape)
    arrays = [2200.0 + 40.0 * grid, 1200.0 + 20.0 * grid, 2000.0 + 10.0 * grid]
    return [
        torch.tensor(a, dtype=torch.float32, device=DEV,
                     requires_grad=requires_grad)
        for a in arrays
    ]


def build(ndim, *, abcn=8, free_surface=False, bs=None, requires_grad=False,
          nt=NT):
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    eq_cls = Elastic if ndim == 2 else Elastic3D
    equation = eq_cls(spatial_order=4, device=DEV, backend="torch")

    boundary_saving_config = {"enabled": False}
    if bs == "gpu":
        boundary_saving_config = {
            "enabled": True, "storage": "gpu",
            "transfer_interval": 1, "pinned_memory": False,
        }

    if ndim == 2:
        source_type, receiver_type = ["sxx", "szz"], ["vx", "vz"]
    else:
        source_type, receiver_type = ["sxx", "syy", "szz"], ["vx", "vy", "vz"]

    prop = PropTorch(
        equation,
        backend="torch",
        impl="c",
        shape=shape,
        dev=DEV,
        dh=10.0,
        dt=DT,
        source_type=source_type,
        receiver_type=receiver_type,
        abcn=abcn,
        free_surface=free_surface,
        pml_type="cpmls",
        nt=nt,
        B=1,
        use_ckpt=False,
        boundary_saving_config=boundary_saving_config,
    )

    if ndim == 2:
        nz, nx = shape
        sources = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )
    else:
        nz, ny, nx = shape
        sources = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array(
            [[[ix, ny // 2, 2] for ix in range(2, nx - 2, 6)]], dtype=np.int32
        )

    models = elastic_models(shape, requires_grad=requires_grad)
    wavelet = ricker(nt, DT, scale=WAVELET_SCALE)
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


def _zero_state(p, L, *, record, u_allt=None, has_bs=False):
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


def _replay(func, p, L, schedule):
    for a, b in schedule:
        p.it_begin, p.it_end = a, b
        # Elastic: identity rotation — the SAME list every segment.
        p.wavefields = rotate_wavefield_roles(L, a, **ELASTIC_ROTATE_KW)
        func(p)
    p.it_begin, p.it_end = 0, -1
    p.wavefields = L


def _snapshot(p, L, *, record, u_allt=None, has_bs=False):
    snap = {
        "wf": [t.clone() for t in L],
        "record": record.clone(),
    }
    if u_allt is not None:
        snap["u_allt"] = u_allt.clone()
    if has_bs:
        snap["bs_gpu"] = [t.clone() for t in p.boundary_gpu]
        snap["last_two"] = p.last_two.clone()
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
    if build_kw.get("requires_grad"):
        prop(wavelet, sources, receivers, models=models)
    else:
        with torch.no_grad():
            prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    want = NWF[ndim]
    if not L:
        # Pure forward runs let C++ allocate internally; bind our own
        # tensors so state persists across stepped calls.
        L = [torch.zeros_like(p.models[0]) for _ in range(want)]
    assert len(L) == want, (
        f"elastic{ndim}d wavefield list must have {want} tensors "
        f"(elastic.h::bind layout), got {len(L)}"
    )

    nt = int(p.nt)
    has_bs = bool(p.use_boundary_saving)

    raw_record = cap["raw_out"][2]
    record = torch.zeros_like(raw_record)
    p.record_out = record

    u_allt = None
    if p.save_all_wavefields:
        u_allt = torch.zeros_like(cap["raw_out"][0])
        p.u_allt_out = u_allt

    state_kw = dict(record=record, u_allt=u_allt, has_bs=has_bs)

    # Reference: one full-range call from zeroed state.
    _zero_state(p, L, **state_kw)
    _replay(func, p, L, [(0, nt)])
    ref = _snapshot(p, L, **state_kw)

    # Sanity: the symmetric replay reproduces the captured public run
    # (elastic binds/allocates the same in-place layout — no psi
    # double-buffer branch divergence as in acoustic history).
    assert torch.equal(record, raw_record)
    assert record.abs().max() > 0, "record is all zero — degenerate setup"

    # Stepped: same zeroed state, k-step segments (ragged tail included).
    _zero_state(p, L, **state_kw)
    _replay(func, p, L, [(a, min(a + k, nt)) for a in range(0, nt, k)])
    step = _snapshot(p, L, **state_kw)

    _assert_equal(ref, step)


# ------------------------------------------------------------ T1: plain PML
@cuda_only
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parametrize("free_surface", [False, True])
def test_stepped_plain_forward(ndim, k, free_surface):
    run_equivalence(ndim, k, free_surface=free_surface)


# ------------------------------------------------- T2: boundary saving (gpu)
@cuda_only
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parametrize("free_surface", [False, True])
def test_stepped_boundary_saving_gpu(ndim, k, free_surface):
    if free_surface and k == 1:
        pytest.skip("FS x BS covered at k=7; k=1 adds runtime, not coverage")
    run_equivalence(ndim, k, bs="gpu", requires_grad=True,
                    free_surface=free_surface)


# --------------------------------------------------- T3: save_all_wavefields
@cuda_only
@pytest.mark.parametrize("ndim", [2, 3])
def test_stepped_save_all_wavefields(ndim):
    run_equivalence(ndim, 7, requires_grad=True)


# --------------------------------------------------------------- T4: guards
@cuda_only
def test_stepped_guards():
    prop, wavelet, sources, receivers, models = build(2, nt=8)
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(NWF[2])]
    record = torch.zeros_like(cap["raw_out"][2])

    # it range out of bounds
    p.record_out = record
    p.it_begin, p.it_end = 0, 9
    with pytest.raises(RuntimeError, match="it_begin <= it_end <= nt"):
        func(p)

    # phase-split is supported but only for a single step
    p.it_begin, p.it_end = 0, -1
    p.wavefields = L
    p.step_phase = 1
    try:
        with pytest.raises(RuntimeError, match="requires a single step"):
            func(p)
    finally:
        p.step_phase = 0

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

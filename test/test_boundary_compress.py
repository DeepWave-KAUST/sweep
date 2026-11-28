"""``BoundarySaving(compress='bitcomp')``: the nvCOMP ring compression.

Lossless, so every storage / storage_dtype must reconstruct the SAME gradient
bit for bit with and without it (``torch.equal``, not a tolerance): the ring
slot the backward dequantizes / restores from holds the same bytes either way.
The option rules are checked without a GPU; the gradient tests need CUDA, the
compiled core and the nvCOMP library (``nvidia-nvcomp-cu12`` or
SWEEP_NVCOMP_LIB), and skip without them.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.propagator.options import (  # noqa: E402
    BoundaryOptions, BoundarySaving, check_memory_supported)


# --------------------------------------------------------------------------- #
# option rules (no GPU)
# --------------------------------------------------------------------------- #
def test_compress_option_rules():
    BoundaryOptions(compress="bitcomp")
    BoundaryOptions(compress="bitcomp", storage="cpu", compress_budget=0.5)
    BoundaryOptions(compress="bitcomp", storage="disk", disk_async_read=True)
    # the compressed gpu store is a ring: chunking knobs apply there
    BoundaryOptions(compress="bitcomp", transfer_interval=8, ring_buffers=2)
    with pytest.raises(ValueError, match="compress must be"):
        BoundaryOptions(compress="zstd")
    with pytest.raises(ValueError, match="compress_budget"):
        BoundaryOptions(compress="bitcomp", compress_budget=0.0)
    with pytest.raises(ValueError, match="compress_budget"):
        BoundaryOptions(compress="bitcomp", compress_budget=1.5)
    # read only with compress: a budget without it is an ignored config
    with pytest.raises(ValueError, match="only read with compress"):
        BoundaryOptions(compress_budget=0.5)
    # the disk file is sparse: no budget there
    with pytest.raises(ValueError, match="only read with storage"):
        BoundaryOptions(compress="bitcomp", storage="disk", disk_async_read=True, compress_budget=0.5)
    # the packed chunks are fetched by the reader thread
    with pytest.raises(ValueError, match="disk_async_read=True"):
        BoundaryOptions(compress="bitcomp", storage="disk")
    # without compress the gpu store has no ring, as before
    with pytest.raises(ValueError, match="transfer_interval"):
        BoundaryOptions(storage="gpu", transfer_interval=8)


def test_compress_refused_where_unimplemented():
    strategy = BoundarySaving(compress="bitcomp")
    with pytest.raises(NotImplementedError, match="compress"):
        check_memory_supported("eager", strategy)
    with pytest.raises(NotImplementedError, match="compress"):
        check_memory_supported("c", strategy, device="cpu")
    check_memory_supported("c", strategy, device="cuda")   # must not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' on a CUDA device")
def test_legacy_dict_route_validates_compress():
    """The dict spelling goes through base.py's normaliser, which must apply
    the same rules (and give the compressed gpu store its ring defaults)."""
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch

    eq = Acoustic(spatial_order=4, device="cuda", backend="torch")
    common = dict(shape=(40, 48), dev="cuda", dh=(10.0, 10.0), dt=0.0015, nt=60, B=1,
                  source_type=["h1"], receiver_type=["h1"], abcn=20, impl="c")
    with pytest.raises(ValueError, match="disk_async_read=True"):
        PropTorch(eq, backend="torch", boundary_saving_config=dict(
            enabled=True, storage="disk", compress="bitcomp"), **common)
    s = PropTorch(eq, backend="torch", boundary_saving_config=dict(
        enabled=True, storage="gpu", compress="bitcomp"), **common)
    cfg = s.boundary_saving_config
    assert cfg["compress"] == "bitcomp" and cfg["transfer_interval"] == 16 and cfg["ring_buffers"] == 2


# --------------------------------------------------------------------------- #
# gradients (CUDA + core + nvCOMP)
# --------------------------------------------------------------------------- #
def _ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        if not is_torch_binding_available():
            return False
        from sweep.backend.c import nvcomp
        nvcomp.find_library()
        return True
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _ready(), reason="CUDA + compiled sweep core + nvCOMP required")

DEV = "cuda"


def _acoustic2d(nt=60, B=1):
    from sweep.equations import Acoustic
    nz, nx = 40, 48
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :] += 300.0
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
    wav = torch.zeros(nt, device=DEV)
    wav[5] = 1.0
    common = dict(shape=(nz, nx), dev=DEV, dh=(10.0, 10.0), dt=0.0015, source_type=["h1"],
                  receiver_type=["h1"], abcn=20, pml_type="cpmlr", nt=nt, B=B, allow_growth=True)
    return (lambda: Acoustic(spatial_order=4, device=DEV, backend="torch")), vp, src, rec, wav, common


def _acoustic3d(nt=40):
    from sweep.equations import Acoustic3D
    nz, ny, nx = 24, 20, 28
    vp = np.full((nz, ny, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:] += 300.0
    src = np.array([[nx // 2, ny // 2, nz // 4]], dtype=np.int32)
    rec = np.array([[[4, ny // 2, 3], [nx - 5, ny // 2, 3], [nx // 2, 4, 3]]], dtype=np.int32)
    wav = torch.zeros(nt, device=DEV)
    wav[4] = 1.0
    common = dict(shape=(nz, ny, nx), dev=DEV, dh=(10.0, 10.0, 10.0), dt=0.0012,
                  source_type=["h1"], receiver_type=["h1"], abcn=8, pml_type="cpmlr",
                  nt=nt, B=1, allow_growth=True)
    return (lambda: Acoustic3D(spatial_order=4, device=DEV, backend="torch")), vp, src, rec, wav, common


def _elastic2d(nt=60):
    from sweep.equations import Elastic
    nz, nx = 40, 48
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :] += 300.0
    vs = vp / 1.8
    rho = np.full((nz, nx), 2000.0, dtype=np.float32)
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
    wav = torch.zeros(nt, device=DEV)
    wav[5] = 1.0
    common = dict(shape=(nz, nx), dev=DEV, dh=(10.0, 10.0), dt=0.0012, source_type=["vz"],
                  receiver_type=["vz"], abcn=20, nt=nt, B=1, allow_growth=True)
    return (lambda: Elastic(spatial_order=4, device=DEV, backend="torch")), (vp, vs, rho), src, rec, wav, common


def _opts(storage, dtype, compress, ti=8, ring=2, **extra):
    from sweep.propagator.options import BoundarySaving
    kw = dict(storage=storage, storage_dtype=dtype, compress=compress, **extra)
    if storage != "gpu" or compress:
        kw.update(transfer_interval=ti, ring_buffers=ring)
    if storage == "disk":
        kw["disk_async_read"] = True
    return BoundarySaving(**kw)


def _grads(problem, opts, batch=None):
    from sweep.propagator.torch import PropTorch
    make_eq, vp, src, rec, wav, common = problem
    s = PropTorch(make_eq(), backend="torch", impl="c", memory=opts, **common)
    models = vp if isinstance(vp, tuple) else (vp,)
    ms = [torch.tensor(m, device=DEV, requires_grad=True) for m in models]
    s(wav, src.copy(), rec.copy(), models=ms).pow(2).mean().backward()
    return [m.grad.clone() for m in ms], s


def _assert_equal(g_off, g_on, what):
    for a, b in zip(g_off, g_on):
        assert torch.isfinite(b).all(), what
        assert torch.equal(a, b), (
            f"{what}: compressed gradient differs, max|d|={(a - b).abs().max().item():.3e} "
            "-- Bitcomp is lossless, so this is a ring / index / fetch bug")


@needs_gpu
@pytest.mark.parametrize("storage", ["gpu", "cpu", "disk"])
@pytest.mark.parametrize("dtype", ["int8", "fp32"])
def test_acoustic2d_bitexact(storage, dtype):
    problem = _acoustic2d()
    g_off, _ = _grads(problem, _opts(storage, dtype, None))
    g_on, s = _grads(problem, _opts(storage, dtype, "bitcomp"))
    _assert_equal(g_off, g_on, f"Acoustic 2-D {storage}/{dtype}")
    assert s.last_boundary_compress_ratio > 1.0


@needs_gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_acoustic2d_halfprec_bitexact(dtype):
    problem = _acoustic2d()
    g_off, _ = _grads(problem, _opts("cpu", dtype, None))
    g_on, _ = _grads(problem, _opts("cpu", dtype, "bitcomp"))
    _assert_equal(g_off, g_on, f"Acoustic 2-D cpu/{dtype}")


@needs_gpu
@pytest.mark.parametrize("storage", ["gpu", "cpu", "disk"])
def test_acoustic3d_bitexact(storage):
    problem = _acoustic3d()
    g_off, _ = _grads(problem, _opts(storage, "int8", None))
    g_on, _ = _grads(problem, _opts(storage, "int8", "bitcomp"))
    _assert_equal(g_off, g_on, f"Acoustic 3-D {storage}/int8")


@needs_gpu
@pytest.mark.parametrize("storage", ["gpu", "disk"])
def test_elastic2d_bitexact(storage):
    """nvar > 1: one region per field per face in 2-D, scale files from index 1."""
    problem = _elastic2d()
    g_off, _ = _grads(problem, _opts(storage, "int8", None))
    g_on, _ = _grads(problem, _opts(storage, "int8", "bitcomp"))
    _assert_equal(g_off, g_on, f"Elastic 2-D {storage}/int8")


@needs_gpu
def test_tail_chunk_and_tail_steps():
    """nt not a multiple of transfer_interval (a shorter last chunk, its own
    piece table) and tail_steps (the ring in saved-step coordinates)."""
    problem = _acoustic2d(nt=61)
    g_off, _ = _grads(problem, _opts("cpu", "int8", None, ti=8, ring=2))
    g_on, _ = _grads(problem, _opts("cpu", "int8", "bitcomp", ti=8, ring=2))
    _assert_equal(g_off, g_on, "tail chunk")
    g_off, _ = _grads(problem, _opts("gpu", "int8", None, tail_steps=30))
    g_on, _ = _grads(problem, _opts("gpu", "int8", "bitcomp", ti=8, ring=2, tail_steps=30))
    _assert_equal(g_off, g_on, "tail_steps")


@needs_gpu
def test_batch_capacity_above_call_batch():
    """Buffers allocated for B=2, call with one shot: the ring is narrowed on
    its batch axis while the pieces span the capacity pitch."""
    problem = _acoustic2d(B=2)
    g_off, _ = _grads(problem, _opts("cpu", "int8", None))
    g_on, _ = _grads(problem, _opts("cpu", "int8", "bitcomp"))
    _assert_equal(g_off, g_on, "B_cap > batch")


@needs_gpu
def test_private_store_while_leased():
    """A second forward before the first backward takes a private store (its
    own workspace and arena); both backwards must still be exact."""
    from sweep.propagator.torch import PropTorch
    make_eq, vp, src, rec, wav, common = _acoustic2d()
    ref, _ = _grads(_acoustic2d(), _opts("gpu", "int8", None))
    s = PropTorch(make_eq(), backend="torch", impl="c",
                  memory=_opts("gpu", "int8", "bitcomp"), **common)
    m1 = torch.tensor(vp, device=DEV, requires_grad=True)
    m2 = torch.tensor(vp, device=DEV, requires_grad=True)
    l1 = s(wav, src.copy(), rec.copy(), models=[m1]).pow(2).mean()
    l2 = s(wav, src.copy(), rec.copy(), models=[m2]).pow(2).mean()
    l2.backward()
    l1.backward()
    assert torch.equal(m1.grad, ref[0]) and torch.equal(m2.grad, ref[0])


@needs_gpu
def test_gpu_budget_overflow_raises():
    problem = _acoustic2d()
    with pytest.raises(RuntimeError, match="compress_budget"):
        _grads(problem, _opts("gpu", "int8", "bitcomp", compress_budget=0.02))


@needs_gpu
@pytest.mark.parametrize("storage", ["gpu", "cpu", "disk"])
def test_report_reads_the_store_and_names_the_budget(storage):
    """The report is a function of the store alone (ModelParallel keeps the
    store on its captured tile params), its ``budget_needed`` is a budget
    that fits, and ``peak_ratio`` == ``last_boundary_compress_ratio``."""
    problem = _acoustic2d()
    _, s = _grads(problem, _opts(storage, "int8", "bitcomp"))
    rep = s._backend_impl.boundary_compress_report()
    for key in ("faces", "raw", "packed", "ratio", "peak_raw", "peak_packed", "peak_ratio",
                "peak_bytes", "budget_needed", "arena_bytes"):
        assert key in rep, key
    assert rep["ratio"] > 1.0 and rep["peak_ratio"] == s.last_boundary_compress_ratio
    # a ring as long as the record: the live set IS the whole forward
    assert rep["peak_raw"] == rep["raw"] and rep["peak_packed"] == rep["packed"]
    assert sum(v[1] for v in rep["faces"].values()) == rep["packed"]
    # the geometry comes from the store, not from the propagator's cache words
    impl = s._backend_impl
    impl._boundary_cache_interval = impl._boundary_cache_ring_buffers = impl._boundary_cache_nt = None
    assert impl.boundary_compress_report() == rep
    if storage == "disk":
        assert rep["budget_needed"] is None
        return
    assert 0 < rep["budget_needed"] <= 0.75
    # the budget it names fits (and is far below the default it ran with)
    g_tight, s2 = _grads(problem, _opts(storage, "int8", "bitcomp", compress_budget=rep["budget_needed"] * 1.02))
    g_ref, _ = _grads(problem, _opts(storage, "int8", None))
    _assert_equal(g_ref, g_tight, f"{storage} at budget_needed")
    assert s2._backend_impl.boundary_compress_report()["budget_needed"] == pytest.approx(rep["budget_needed"])


@needs_gpu
def test_missing_library_raises(monkeypatch):
    monkeypatch.setenv("SWEEP_NVCOMP_LIB", "/nonexistent/libnvcomp.so.6")
    problem = _acoustic2d()
    with pytest.raises(ImportError, match="SWEEP_NVCOMP_LIB"):
        _grads(problem, _opts("gpu", "int8", "bitcomp"))

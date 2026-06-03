"""Regression test: staged boundary ``storage_dtype`` support matrix.

Compute always stays FP32; ``storage_dtype`` only changes how the saved
boundary values are represented in memory.  The C++ staged transfer path
(``storage='cpu'``/``'disk'``) supports half-precision staging (fp16/bf16) AND
int8: the persistent cpu buffer (or, for ``'disk'``, the on-disk files), the
gpu staging ring and the cast/quantize kernels run at the save/restore boundary
are all sized/typed to the requested dtype.

int8 staging adds a uint8 main buffer plus an FP32 per-block scale buffer
(quantize on save, dequantize on restore) -- a uint8 main ring + FP32 scale
ring on the gpu, flushed to a persistent uint8 + scale buffer (cpu) or parallel
uint8 + scale files (disk).  Every (storage, storage_dtype) combination is now
supported; there is no longer any rejected combination.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.propagator.options import BoundaryOptions  # noqa: E402

K = "SWEEP_BOUNDARY_DTYPE"

# Every (storage, storage_dtype) combination must be accepted.
SUPPORTED = [
    ("gpu", "fp32"), ("gpu", "fp16"), ("gpu", "bf16"), ("gpu", "int8"),
    ("cpu", "fp32"), ("cpu", "fp16"), ("cpu", "bf16"), ("cpu", "int8"),
    ("disk", "fp32"), ("disk", "fp16"), ("disk", "bf16"), ("disk", "int8"),
]


def test_boundary_options_support_matrix():
    for storage, dtype in SUPPORTED:
        BoundaryOptions(storage=storage, storage_dtype=dtype)  # must not raise
    BoundaryOptions(storage="cpu")  # default fp32
    BoundaryOptions(storage="gpu")
    # An unknown dtype must still be rejected.
    with pytest.raises(ValueError, match="storage_dtype"):
        BoundaryOptions(storage="cpu", storage_dtype="fp8")


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available

        return bool(is_torch_binding_available())
    except Exception:
        return False


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_dict_and_env_staged_dtype_at_forward(monkeypatch):
    """Dict / env config bypasses ``BoundaryOptions.__post_init__`` -- the
    ``_ensure_boundary_buffers`` chokepoint must accept every staged dtype
    (fp32/fp16/bf16/int8) on both cpu and disk and produce a finite gradient."""
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch

    monkeypatch.delenv(K, raising=False)
    dev = "cuda"
    nz, nx, nt, dt = 40, 48, 60, 0.0015
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
    wav = torch.zeros(nt, device=dev)
    wav[5] = 1.0
    common = dict(shape=(nz, nx), dev=dev, dh=(10.0, 10.0), dt=dt, source_type=["h1"],
                  receiver_type=["h1"], abcn=20, pml_type="cpmlr", nt=nt, B=1, allow_growth=True)

    def eq():
        return Acoustic(spatial_order=4, device=dev, backend="torch")

    def fwd(solver):
        m = torch.tensor(vp, device=dev, requires_grad=True)
        solver(wav, src.copy(), rec.copy(), models=[m]).pow(2).mean().backward()
        return m.grad

    # Every staged combo (incl. int8) runs and gives a finite gradient.
    for storage in ("cpu", "disk"):
        for dtype in ("fp32", "fp16", "bf16", "int8"):
            s = PropTorch(eq(), backend="torch", impl="c",
                          boundary_saving_config={"enabled": True, "storage": storage,
                                                  "storage_dtype": dtype, "transfer_interval": 8}, **common)
            assert torch.isfinite(fwd(s)).all(), f"{storage}/{dtype} produced a non-finite gradient"

    # Env override pushing int8 onto cpu storage must also run (no rejection).
    monkeypatch.setenv(K, "int8")
    s_env = PropTorch(eq(), backend="torch", impl="c",
                      boundary_saving_config={"enabled": True, "storage": "cpu", "transfer_interval": 8}, **common)
    assert torch.isfinite(fwd(s_env)).all()
    monkeypatch.delenv(K, raising=False)


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_staged_halfprec_matches_gpu_halfprec():
    """cpu- and disk-staged fp16/bf16 must reconstruct the *same* gradient as
    gpu-direct fp16/bf16 -- all three run the identical cast kernels, so any
    divergence is a staging byte-count / pitch / disk-I/O bug rather than a
    precision effect."""
    from sweep.equations import Acoustic
    from sweep.propagator.options import CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    dev = "cuda"
    nz, nx, nt, dt = 40, 48, 60, 0.0015
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :] += 300.0  # a contrast so the gradient is non-trivial
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
    wav = torch.zeros(nt, device=dev)
    wav[5] = 1.0
    common = dict(shape=(nz, nx), dev=dev, dh=(10.0, 10.0), dt=dt, source_type=["h1"],
                  receiver_type=["h1"], abcn=20, pml_type="cpmlr", nt=nt, B=1, allow_growth=True)

    def grad(storage, dtype):
        eq = Acoustic(spatial_order=4, device=dev, backend="torch")
        bopt = BoundaryOptions(storage=storage, storage_dtype=dtype,
                               transfer_interval=(8 if storage != "gpu" else None))
        co = CUDAOptions(memory=MemoryOptions(strategy="boundary", boundary=bopt))
        s = PropTorch(eq, backend="torch", impl="c", cuda_options=co, **common)
        m = torch.tensor(vp, device=dev, requires_grad=True)
        s(wav, src.copy(), rec.copy(), models=[m]).pow(2).mean().backward()
        return m.grad.double()

    for dtype in ("bf16", "fp16"):
        g_gpu = grad("gpu", dtype)
        for storage in ("cpu", "disk"):
            g_staged = grad(storage, dtype)
            max_abs = (g_staged - g_gpu).abs().max().item()
            scale = g_gpu.abs().max().item() + 1e-30
            assert max_abs / scale < 1e-4, (
                f"{storage}/{dtype} gradient diverges from gpu/{dtype}: "
                f"max|{storage}-gpu|/scale={max_abs / scale:.2e} (expected bit-for-bit match)"
            )


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_staged_int8_matches_gpu_int8():
    """cpu/disk-staged int8 must reconstruct the *same* gradient as gpu-direct
    int8.  INT8 is lossy, so the reference is gpu-direct int8 (identical
    quantize/dequantize kernels), NOT fp32 -- any divergence here is a staging
    byte-count / scale-ring / disk-I/O bug rather than quantization error.
    cpu staging is lossless (uint8 + scale copied verbatim) so it matches
    bit-for-bit; the 2D disk path matches to a loose tolerance (chunked async
    accumulation reorders the adjoint sum, a pre-existing fp32-era effect)."""
    from sweep.equations import Acoustic
    from sweep.propagator.options import CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    dev = "cuda"
    nz, nx, nt, dt = 40, 48, 60, 0.0015
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :] += 300.0
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
    wav = torch.zeros(nt, device=dev)
    wav[5] = 1.0
    common = dict(shape=(nz, nx), dev=dev, dh=(10.0, 10.0), dt=dt, source_type=["h1"],
                  receiver_type=["h1"], abcn=20, pml_type="cpmlr", nt=nt, B=1, allow_growth=True)

    def grad(storage):
        eq = Acoustic(spatial_order=4, device=dev, backend="torch")
        bopt = BoundaryOptions(storage=storage, storage_dtype="int8",
                               transfer_interval=(8 if storage != "gpu" else None))
        co = CUDAOptions(memory=MemoryOptions(strategy="boundary", boundary=bopt))
        s = PropTorch(eq, backend="torch", impl="c", cuda_options=co, **common)
        m = torch.tensor(vp, device=dev, requires_grad=True)
        s(wav, src.copy(), rec.copy(), models=[m]).pow(2).mean().backward()
        return m.grad.double()

    g_gpu = grad("gpu")
    # cpu int8 staging is lossless w.r.t. gpu-direct int8 -> bit-for-bit.
    g_cpu = grad("cpu")
    max_abs = (g_cpu - g_gpu).abs().max().item()
    scale = g_gpu.abs().max().item() + 1e-30
    assert max_abs / scale < 1e-4, (
        f"cpu/int8 gradient diverges from gpu/int8: max|cpu-gpu|/scale="
        f"{max_abs / scale:.2e} (expected bit-for-bit match)")
    # disk int8 staging: same quantized values, looser tol for chunk reordering.
    g_disk = grad("disk")
    assert torch.isfinite(g_disk).all()
    max_abs = (g_disk - g_gpu).abs().max().item()
    assert max_abs / scale < 1e-3, (
        f"disk/int8 gradient diverges from gpu/int8: max|disk-gpu|/scale="
        f"{max_abs / scale:.2e} (staging / disk-I/O bug, not quantization noise)")


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_disk_3d_halfprec_consistent():
    """3D disk staging exercises a different code path than 2D (contiguous
    ``cudaMemcpyAsync`` + threaded file reads in ``flush_gpu_to_disk_3d`` /
    ``load_disk_to_cpu_3d``).  3D staged gradients are NOT bit-for-bit vs
    gpu-direct -- the async ring-buffered chunking reorders the adjoint FP
    accumulation (a pre-existing ~1e-4 effect, present at fp32 too) -- so this
    only guards against gross byte-offset / size corruption, which would
    produce an O(1) divergence rather than the ~3e-4 seen for a correct path.

    Covers fp16/bf16 and int8 (the int8 path additionally round-trips the
    per-block FP32 scale through its own files)."""
    from sweep.equations import Acoustic3D
    from sweep.propagator.options import CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    dev = "cuda"
    nz, ny, nx, nt, dt = 24, 20, 24, 100, 0.0015
    vp = np.full((nz, ny, nx), 2000.0, dtype=np.float32)
    vp[nz // 2:, :, :] += 300.0
    src = np.array([[nx // 2, ny // 2, nz // 4]], dtype=np.int32)
    rx, ry = np.meshgrid(np.arange(2, nx - 2, 6), np.arange(2, ny - 2, 6))
    rx, ry = rx.ravel(), ry.ravel()
    rec = np.stack([rx, ry, np.full(rx.size, 2)], -1).astype(np.int32)[None]
    wav = torch.zeros(nt, device=dev)
    wav[5] = 1.0
    common = dict(shape=(nz, ny, nx), dev=dev, dh=(10.0, 10.0, 10.0), dt=dt, source_type=["h1"],
                  receiver_type=["h1"], abcn=15, pml_type="cpmlr", nt=nt, B=1, allow_growth=True)

    def grad(storage, dtype):
        eq = Acoustic3D(spatial_order=4, device=dev, backend="torch")
        bopt = BoundaryOptions(storage=storage, storage_dtype=dtype,
                               transfer_interval=(4 if storage != "gpu" else None),
                               ring_buffers=(2 if storage == "disk" else None))
        co = CUDAOptions(memory=MemoryOptions(strategy="boundary", boundary=bopt))
        s = PropTorch(eq, backend="torch", impl="c", cuda_options=co, **common)
        m = torch.tensor(vp, device=dev, requires_grad=True)
        s(wav, src.copy(), rec.copy(), models=[m]).pow(2).mean().backward()
        return m.grad.double()

    for dtype in ("bf16", "fp16", "int8"):
        g_gpu = grad("gpu", dtype)
        g_disk = grad("disk", dtype)
        assert torch.isfinite(g_disk).all()
        max_abs = (g_disk - g_gpu).abs().max().item()
        scale = g_gpu.abs().max().item() + 1e-30
        assert max_abs / scale < 1e-2, (
            f"disk/{dtype} 3D gradient grossly diverges from gpu/{dtype}: "
            f"max|disk-gpu|/scale={max_abs / scale:.2e} (corruption, not chunking noise)"
        )

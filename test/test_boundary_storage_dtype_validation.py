"""Regression test: staged boundary ``storage_dtype`` support matrix.

Compute always stays FP32; ``storage_dtype`` only changes how the saved
boundary values are represented in memory.  The C++ staged transfer path
(``storage='cpu'``/``'disk'``) supports half-precision staging (fp16/bf16):
the persistent cpu buffer, the gpu staging ring and -- for ``'disk'`` -- the
on-disk files are all sized/typed to the requested dtype, and the same cast
kernels as gpu-direct storage run at the save/restore boundary.

The only staged combination NOT implemented yet is ``int8`` (it needs a
uint8 + per-block-scale staging ring); it must fail loudly rather than degrade
silently to fp32 (no compression, no warning).  The rejection is enforced both
at ``BoundaryOptions`` construction and at the ``_ensure_boundary_buffers``
chokepoint that the dict / env-var config paths flow through (those bypass
``BoundaryOptions.__post_init__``).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.propagator.options import BoundaryOptions  # noqa: E402

K = "SWEEP_BOUNDARY_DTYPE"

# (storage, storage_dtype) combinations that must be accepted ...
SUPPORTED = [
    ("gpu", "fp32"), ("gpu", "fp16"), ("gpu", "bf16"), ("gpu", "int8"),
    ("cpu", "fp32"), ("cpu", "fp16"), ("cpu", "bf16"),
    ("disk", "fp32"), ("disk", "fp16"), ("disk", "bf16"),
]
# ... and the ones that must still be rejected (staged int8 is unimplemented).
UNSUPPORTED = [
    ("cpu", "int8"),
    ("disk", "int8"),
]


def test_boundary_options_support_matrix():
    for storage, dtype in SUPPORTED:
        BoundaryOptions(storage=storage, storage_dtype=dtype)  # must not raise
    BoundaryOptions(storage="cpu")  # default fp32
    BoundaryOptions(storage="gpu")
    for storage, dtype in UNSUPPORTED:
        with pytest.raises(ValueError, match="int8 is gpu-only"):
            BoundaryOptions(storage=storage, storage_dtype=dtype)


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
    ``_ensure_boundary_buffers`` chokepoint must reject staged int8 and run the
    supported cpu/disk half-precision combinations."""
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

    # Staged int8 must raise at the chokepoint (not degrade silently to fp32).
    for storage in ("cpu", "disk"):
        s = PropTorch(eq(), backend="torch", impl="c",
                      boundary_saving_config={"enabled": True, "storage": storage, "storage_dtype": "int8"}, **common)
        with pytest.raises(ValueError, match="int8 is gpu-only"):
            fwd(s)

    # Env override pushing int8 onto cpu storage must also be rejected.
    monkeypatch.setenv(K, "int8")
    s_env = PropTorch(eq(), backend="torch", impl="c",
                      boundary_saving_config={"enabled": True, "storage": "cpu"}, **common)
    with pytest.raises(ValueError, match="int8 is gpu-only"):
        fwd(s_env)
    monkeypatch.delenv(K, raising=False)

    # Supported staged combos run and give a finite gradient.
    for storage in ("cpu", "disk"):
        for dtype in ("fp32", "fp16", "bf16"):
            s = PropTorch(eq(), backend="torch", impl="c",
                          boundary_saving_config={"enabled": True, "storage": storage,
                                                  "storage_dtype": dtype, "transfer_interval": 8}, **common)
            assert torch.isfinite(fwd(s)).all(), f"{storage}/{dtype} produced a non-finite gradient"


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
def test_disk_3d_halfprec_consistent():
    """3D disk staging exercises a different code path than 2D (contiguous
    ``cudaMemcpyAsync`` + threaded file reads in ``flush_gpu_to_disk_3d`` /
    ``load_disk_to_cpu_3d``).  3D staged gradients are NOT bit-for-bit vs
    gpu-direct -- the async ring-buffered chunking reorders the adjoint FP
    accumulation (a pre-existing ~1e-4 effect, present at fp32 too) -- so this
    only guards against gross byte-offset / size corruption, which would
    produce an O(1) divergence rather than the ~3e-4 seen for a correct path."""
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

    for dtype in ("bf16", "fp16"):
        g_gpu = grad("gpu", dtype)
        g_disk = grad("disk", dtype)
        assert torch.isfinite(g_disk).all()
        max_abs = (g_disk - g_gpu).abs().max().item()
        scale = g_gpu.abs().max().item() + 1e-30
        assert max_abs / scale < 1e-2, (
            f"disk/{dtype} 3D gradient grossly diverges from gpu/{dtype}: "
            f"max|disk-gpu|/scale={max_abs / scale:.2e} (corruption, not chunking noise)"
        )

"""Regression test: staged boundary ``storage_dtype`` support matrix.

Compute always stays FP32; ``storage_dtype`` only changes how the saved
boundary values are represented in memory.  The C++ staged transfer path
(``storage='cpu'``/``'disk'``) now supports half-precision **cpu** staging
(fp16/bf16): the persistent cpu buffer and the gpu staging ring are allocated
in the requested dtype and the same cast kernels as gpu-direct storage run at
the save/restore boundary.

Two staged combinations are NOT implemented yet and must fail loudly rather
than degrade silently to fp32 (no compression, no warning):

  * ``int8`` on cpu/disk -- needs a uint8 + per-block scale staging ring, and
  * ``fp16``/``bf16`` on disk -- needs dtype-aware disk I/O.

The rejection is enforced both at ``BoundaryOptions`` construction and at the
``_ensure_boundary_buffers`` chokepoint that the dict / env-var config paths
flow through (those bypass ``BoundaryOptions.__post_init__``).
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
    ("disk", "fp32"),
]
# ... and the ones that must still be rejected (staged path can't represent them).
UNSUPPORTED = [
    ("cpu", "int8"),
    ("disk", "fp16"), ("disk", "bf16"), ("disk", "int8"),
]


def test_boundary_options_support_matrix():
    for storage, dtype in SUPPORTED:
        BoundaryOptions(storage=storage, storage_dtype=dtype)  # must not raise
    BoundaryOptions(storage="cpu")  # default fp32
    BoundaryOptions(storage="gpu")
    for storage, dtype in UNSUPPORTED:
        with pytest.raises(ValueError, match="fp32-only"):
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
    ``_ensure_boundary_buffers`` chokepoint must reject the unsupported staged
    combinations and run the supported cpu half-precision ones."""
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

    # Unsupported staged combos must raise at the chokepoint (not degrade silently).
    s_int8 = PropTorch(eq(), backend="torch", impl="c",
                       boundary_saving_config={"enabled": True, "storage": "cpu", "storage_dtype": "int8"}, **common)
    with pytest.raises(ValueError, match="fp32-only"):
        fwd(s_int8)

    s_disk16 = PropTorch(eq(), backend="torch", impl="c",
                         boundary_saving_config={"enabled": True, "storage": "disk", "storage_dtype": "fp16"}, **common)
    with pytest.raises(ValueError, match="fp32-only"):
        fwd(s_disk16)

    # Env override pushing int8 onto cpu storage must also be rejected.
    monkeypatch.setenv(K, "int8")
    s_env = PropTorch(eq(), backend="torch", impl="c",
                      boundary_saving_config={"enabled": True, "storage": "cpu"}, **common)
    with pytest.raises(ValueError, match="fp32-only"):
        fwd(s_env)
    monkeypatch.delenv(K, raising=False)

    # Supported combos run and give a finite gradient.
    for dtype in ("fp32", "fp16", "bf16"):
        s = PropTorch(eq(), backend="torch", impl="c",
                      boundary_saving_config={"enabled": True, "storage": "cpu", "storage_dtype": dtype}, **common)
        assert torch.isfinite(fwd(s)).all(), f"cpu/{dtype} produced a non-finite gradient"


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_cpu_halfprec_matches_gpu_halfprec():
    """cpu-staged fp16/bf16 must reconstruct the *same* gradient as gpu-direct
    fp16/bf16 -- both run the identical cast kernels, so any divergence is a
    staging byte-count / pitch bug rather than a precision effect."""
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
        g_cpu = grad("cpu", dtype)
        max_abs = (g_cpu - g_gpu).abs().max().item()
        scale = g_gpu.abs().max().item() + 1e-30
        assert max_abs / scale < 1e-4, (
            f"cpu/{dtype} gradient diverges from gpu/{dtype}: "
            f"max|cpu-gpu|/scale={max_abs / scale:.2e} (expected bit-for-bit match)"
        )

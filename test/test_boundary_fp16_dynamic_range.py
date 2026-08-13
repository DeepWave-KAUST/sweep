"""fp16 boundary storage must survive the dynamic range of elastic wavefields.

A velocity-stress elastic wavefield spans ~7 decades within one timestep:
velocities sit ~rho*vp (about 2e6) below the stresses.  With a unit-amplitude
stress source the velocity boundary values land at O(1e-8) -- below fp16's
smallest subnormal (2^-24 ~ 6e-8) -- so a bare .half()/__float2half cast
flushes the velocity faces to zero and corrupts the reconstructed gradient
(pre-fix: elastic3d vp rel_l2 0.49 interior / 0.91 free-surface vs the eager
reference; elastic2d ~0.09).  bf16 (fp32 exponent range) and int8 (per-block
scales) never had the problem.

The fix stores fp16 per-256-cell normalized, exactly like int8 but with a
__half payload (see quantize_fp16_kernel / _quantize_fp16): scale = block
max|val|, payload in [-1, 1] where the full 10-bit mantissa applies.  These
tests pin that behaviour on the eager helpers, and on impl='c' for the
gpu-direct, staged-cpu and staged-disk storage paths in 2D and 3D.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.propagator._eager_boundary_saving import (  # noqa: E402
    _dequantize_fp16,
    _quantize_fp16,
)

SO, ABCN = 4, 30
DH, DT, NT = 10.0, 1.5e-3, 120
REL_TOL = 2e-2   # pre-fix: 9e-2 (2D) / 4.9e-1 (3D) -- an order of magnitude above


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available
        return bool(is_torch_binding_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mechanism: the block-scaled roundtrip keeps values a bare fp16 cast destroys.
# ---------------------------------------------------------------------------

def test_quantize_fp16_roundtrip_survives_subnormal_range():
    torch.manual_seed(0)
    # Velocity-like magnitudes: far below fp16's smallest subnormal (2^-24).
    tiny = torch.randn(4096, dtype=torch.float32) * 1e-9
    bare = tiny.half().float()
    bare_rel = (bare - tiny).norm() / tiny.norm()
    assert bare_rel > 0.9, "bare fp16 cast should flush ~all of these to zero"

    codes, scale = _quantize_fp16(tiny)
    back = _dequantize_fp16(codes, scale, tiny.numel())
    rel = ((back - tiny).norm() / tiny.norm()).item()
    assert rel < 1e-3, f"block-scaled fp16 roundtrip rel {rel}"

    # Mixed stress+velocity magnitudes in one buffer (the elastic case).
    mixed = torch.cat([torch.randn(2048) * 1e-2, torch.randn(2048) * 1e-9])
    codes, scale = _quantize_fp16(mixed)
    back = _dequantize_fp16(codes, scale, mixed.numel())
    rel = ((back - mixed).norm() / mixed.norm()).item()
    assert rel < 1e-3, f"mixed-range roundtrip rel {rel}"


# ---------------------------------------------------------------------------
# impl='c': elastic gradients with fp16 boundaries match the fp32-BS gradient
# on every storage path.  Unit-amplitude stress source = the failing regime.
# ---------------------------------------------------------------------------

def _ricker(nt, dt, freq=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return (1.0 - 2.0 * x * x) * np.exp(-(x * x))


def _ramp(shape, top, bottom):
    nz = shape[0]
    ramp = np.linspace(top, bottom, nz, dtype=np.float32)
    return np.broadcast_to(ramp.reshape((nz,) + (1,) * (len(shape) - 1)), shape).copy()


def _elastic_grads(ndim, storage, dtype, tmp_path):
    from sweep.equations import Elastic, Elastic3D
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions

    dev = "cuda"
    if ndim == 2:
        shape, cls = (48, 56), Elastic
        st, rt = ["sxx", "szz"], ["vx", "vz"]
        src = np.array([[shape[1] // 2, shape[0] // 4]], np.int64)
        rx = np.arange(2, shape[1] - 2, 6, dtype=np.int64)
        rec = np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]
    else:
        shape, cls = (24, 20, 24), Elastic3D
        st, rt = ["sxx", "syy", "szz"], ["vx", "vy", "vz"]
        src = np.array([[shape[2] // 2, shape[1] // 2, shape[0] // 4]], np.int64)
        rx = np.arange(2, shape[2] - 2, 6, dtype=np.int64)
        ry = np.arange(2, shape[1] - 2, 6, dtype=np.int64)
        gy, gx = np.meshgrid(ry, rx, indexing="ij")
        rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2, np.int64)], -1)[None]

    vp = _ramp(shape, 1800.0, 2400.0)
    vs = (vp / 1.73).astype(np.float32)
    rho = _ramp(shape, 1000.0, 1200.0)

    boundary = dict(storage=storage, storage_dtype=dtype)
    if storage == "disk":
        boundary["disk_dir"] = str(tmp_path)
    opts = CUDAOptions(memory=MemoryOptions(strategy="boundary",
                                            boundary=BoundaryOptions(**boundary)))
    prop = PropTorch(cls(spatial_order=SO, device=dev, backend="torch"),
                     backend="torch", impl="c", cuda_options=opts,
                     shape=shape, abcn=ABCN, dh=DH, dt=DT,
                     source_type=st, receiver_type=rt)
    wav = torch.tensor(_ricker(NT, DT), device=dev)   # unit amplitude
    m = [torch.tensor(vp, device=dev, requires_grad=True),
         torch.tensor(vs, device=dev, requires_grad=True),
         torch.tensor(rho, device=dev)]
    out = prop(wav, src.copy(), rec.copy(), models=m)
    out.pow(2).sum().backward()
    return m[0].grad.detach().clone(), m[1].grad.detach().clone()


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
@pytest.mark.parametrize("ndim,storage", [(2, "gpu"), (3, "gpu"), (2, "cpu"), (2, "disk")])
def test_elastic_fp16_boundary_gradient_matches_fp32(ndim, storage, tmp_path):
    g32_vp, g32_vs = _elastic_grads(ndim, storage, "fp32", tmp_path)
    g16_vp, g16_vs = _elastic_grads(ndim, storage, "fp16", tmp_path)
    assert torch.isfinite(g16_vp).all() and torch.isfinite(g16_vs).all()
    rel_vp, rel_vs = _rel(g16_vp, g32_vp), _rel(g16_vs, g32_vs)
    assert rel_vp < REL_TOL, f"{ndim}D {storage}: vp rel {rel_vp:.3e}"
    assert rel_vs < REL_TOL, f"{ndim}D {storage}: vs rel {rel_vs:.3e}"


# ---------------------------------------------------------------------------
# eager: the ring storage uses the same block-scaled fp16.  Scaling the source
# down pushes every stored boundary value below 2^-24, which the pre-fix bare
# cast turned into an all-zero ring.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eager_fp16_ring_survives_tiny_amplitudes():
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import BoundaryOptions, EagerOptions, MemoryOptions

    dev = "cuda"
    nz, nx = 32, 40
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    src = np.array([[nx // 2, nz // 4]], np.int64)
    rx = np.arange(2, nx - 2, 6, dtype=np.int64)
    rec = np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]
    wav = torch.tensor(_ricker(NT, DT) * 1e-6, device=dev)   # everything < 2^-24

    def grad(mem):
        eq = Acoustic(spatial_order=SO, device=dev, backend="torch")
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                         memory=mem, shape=(nz, nx), dev=dev, dh=DH, dt=DT,
                         source_type=["h1"], receiver_type=["h1"],
                         abcn=20, pml_type=eq.default_pml_type, nt=NT, B=1)
        m = torch.tensor(vp, device=dev, requires_grad=True)
        prop(wav, src.copy(), rec.copy(), models=[m]).pow(2).sum().backward()
        return m.grad.detach().clone()

    g32 = grad(MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="gpu")))
    g16 = grad(MemoryOptions(strategy="boundary",
                             boundary=BoundaryOptions(storage="gpu", storage_dtype="fp16")))
    assert torch.isfinite(g16).all()
    rel = _rel(g16, g32)
    assert rel < REL_TOL, f"eager fp16 ring at 1e-6 amplitude: rel {rel:.3e}"

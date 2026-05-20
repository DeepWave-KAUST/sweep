"""Regression test for the boundary-saving int idx overflow bug.

Before the fix in bug/boundary-kernel-int-idx-overflow,
``boundary_kernel3d_compact`` (and ``boundary_kernel3d``) computed the
boundary-buffer index as ``int``. With ``spatial_order=8`` plus a large enough
combination of ``nz``/``ny``/``nt``, the byte offset that nvcc emits for
``boundary[idx]`` (``idx * sizeof(float)``) overflowed int32 and wrapped to a
large negative value. The kernel then wrote to memory far outside the LEFT /
RIGHT boundary tensor, surfacing as ``CUDA error: an illegal memory access``
during the forward pass.

The fix widens ``idx`` to ``int64_t`` and forces 64-bit promotion of the
``it * ctx.B`` factor that drives the expression, so both intermediate
arithmetic and the pointer offset stay safe.

This test runs the minimum config that reliably triggered the bug on an A100:
``(nz=401, ny=383, nx=64), nt=2800, spatial_order=8`` with
``strategy='boundary'`` and GPU storage. Peak GPU memory is ~23 GiB, so the
test is skipped on GPUs with less than 32 GiB total.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)


def _device0_total_gib() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or _device0_total_gib() < 32.0,
    reason="requires >=32 GiB GPU memory",
)
def test_boundary_kernel_order8_no_illegal_memory_access():
    """Forward + backward must complete and produce a finite, non-zero gradient.

    Before the fix, this configuration crashed with
    ``CUDA error: an illegal memory access was encountered`` during the very
    first forward time step on order=8.
    """
    import numpy as np

    from sweep.equations import Acoustic3D
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    dev = torch.device("cuda")
    nz, ny, nx, nt = 401, 383, 64, 2800
    dh, dt, freq, delay = 10.0, 1.0e-3, 8.0, 0.12

    vp = torch.full((nz, ny, nx), 2000.0, device=dev).requires_grad_(True)
    t = np.arange(nt, dtype=np.float32) * dt
    wavelet = ricker(t - delay, f=freq).astype(np.float32)
    sources = np.array([[nx // 2, ny // 2, 4]], dtype=np.int64)
    rec_x = np.linspace(20, nx - 20, 64, dtype=np.int64)
    rec_y = np.full_like(rec_x, ny // 2)
    rec_z = np.full_like(rec_x, 6)
    receivers = np.stack([rec_x, rec_y, rec_z], axis=1)[None, ...]

    equation = Acoustic3D(spatial_order=8, device=dev, backend="torch")
    solver = PropTorch(
        equation,
        shape=(nz, ny, nx),
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=20,
        free_surface=True,
        pml_type="cpmlr",
        backend="cuda",
        cuda_options=CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(storage="gpu"),
            )
        ),
    )

    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = syn.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(vp.grad).all(), "gradient has NaN/inf"
    assert vp.grad.abs().max().item() > 0.0, "gradient is identically zero"

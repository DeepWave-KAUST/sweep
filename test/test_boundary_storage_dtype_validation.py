"""Regression test: half-precision / int8 boundary storage is GPU-only.

The C++ staged transfer path (storage='cpu'/'disk') reads its staging ring
buffer as float32 and has no half-precision/int8 code path, so requesting a
non-fp32 ``storage_dtype`` with cpu/disk storage previously degraded silently
to fp32 (no compression, no warning).  It must now fail loudly -- both at
``BoundaryOptions`` construction and at the ``_ensure_boundary_buffers``
chokepoint that the dict / env-var config paths flow through.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sweep.propagator.options import BoundaryOptions  # noqa: E402

K = "SWEEP_BOUNDARY_DTYPE"


def test_boundary_options_rejects_halfprec_on_cpu_disk():
    for storage in ("cpu", "disk"):
        for dtype in ("fp16", "bf16", "int8"):
            with pytest.raises(ValueError, match="requires storage='gpu'"):
                BoundaryOptions(storage=storage, storage_dtype=dtype)
    # Valid combinations must NOT raise.
    BoundaryOptions(storage="gpu", storage_dtype="fp16")
    BoundaryOptions(storage="gpu", storage_dtype="bf16")
    BoundaryOptions(storage="gpu", storage_dtype="int8")
    BoundaryOptions(storage="cpu", storage_dtype="fp32")
    BoundaryOptions(storage="cpu")  # default fp32
    BoundaryOptions(storage="disk", storage_dtype="fp32")


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available

        return bool(is_torch_binding_available())
    except Exception:
        return False


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_dict_and_env_halfprec_on_cpu_raise_at_forward(monkeypatch):
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

    # Dict config bypasses BoundaryOptions.__post_init__; the chokepoint must catch it.
    s = PropTorch(eq(), backend="torch", impl="c",
                  boundary_saving_config={"enabled": True, "storage": "cpu", "storage_dtype": "fp16"}, **common)
    with pytest.raises(ValueError, match="requires storage='gpu'"):
        fwd(s)

    # Env override + cpu storage must also be rejected at the chokepoint.
    monkeypatch.setenv(K, "fp16")
    s2 = PropTorch(eq(), backend="torch", impl="c",
                   boundary_saving_config={"enabled": True, "storage": "cpu"}, **common)
    with pytest.raises(ValueError, match="requires storage='gpu'"):
        fwd(s2)
    monkeypatch.delenv(K, raising=False)

    # Regression: valid combos still run.
    s3 = PropTorch(eq(), backend="torch", impl="c",
                   boundary_saving_config={"enabled": True, "storage": "cpu"}, **common)  # cpu + fp32
    assert torch.isfinite(fwd(s3)).all()

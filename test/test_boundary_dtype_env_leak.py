"""Regression test: per-instance boundary ``storage_dtype`` must not leak into
later default-configured PropTorch instances in the same process.

Before the fix, the storage dtype was passed Python->C++ via a process-global
``SWEEP_BOUNDARY_DTYPE`` env var that ``_ensure_boundary_buffers`` set
permanently; a solver configured with e.g. ``storage_dtype='int8'`` then
silently forced int8 boundary storage on every later default-configured
PropTorch.  The C++ saver now derives the storage dtype from the boundary
buffer tensors it is handed (csrc/.../boundary/saver.cuh), and Python no longer
writes the env -- so the choice is per-instance and cannot leak.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

K = "SWEEP_BOUNDARY_DTYPE"


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available

        return bool(is_torch_binding_available())
    except Exception:
        return False


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_storage_dtype_does_not_leak_across_instances(monkeypatch):
    from sweep.equations import Acoustic
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    monkeypatch.delenv(K, raising=False)
    dev = "cuda"
    nz, nx, nt, dt, dh = 40, 48, 80, 0.0015, 10.0
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[nz // 2 :, :] += 300.0
    sources = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rec_x = np.arange(2, nx - 2, 4, dtype=np.int32)
    receivers = np.stack([rec_x, np.full(rec_x.size, 2, dtype=np.int32)], -1)[None]
    t = np.arange(nt) * dt
    f, delay = 12.0, 0.08
    wav = torch.tensor(
        ((1 - 2 * (np.pi * f * (t - delay)) ** 2) * np.exp(-((np.pi * f * (t - delay)) ** 2))).astype(np.float32),
        device=dev,
    )
    common = dict(shape=(nz, nx), dev=dev, dh=(dh, dh), dt=dt, source_type=["h1"],
                  receiver_type=["h1"], abcn=20, pml_type="cpmlr", nt=nt, B=1, allow_growth=True)

    def acoustic():
        return Acoustic(spatial_order=4, device=dev, backend="torch")

    def run(solver):
        m = torch.tensor(vp, device=dev, requires_grad=True)
        rec = solver(wav, sources.copy(), receivers.copy(), models=[m])
        rec.pow(2).mean().backward()
        assert torch.isfinite(m.grad).all()
        return m.grad.detach().double()

    def int8_solver():
        co = CUDAOptions(memory=MemoryOptions(strategy="boundary",
                boundary=BoundaryOptions(storage="gpu", storage_dtype="int8")))
        return PropTorch(acoustic(), backend="torch", impl="c", cuda_options=co, **common)

    def default_solver():  # no boundary config -> default boundary saving (leak victim)
        return PropTorch(acoustic(), backend="torch", impl="c", **common)

    # Reference fp32 gradient from a default solver on a clean process.
    g_ref = run(default_solver())
    assert K not in os.environ

    # A per-instance int8 storage_dtype must NOT persist in the process env ...
    g_int8 = run(int8_solver())
    assert K not in os.environ, "per-instance storage_dtype leaked into os.environ"

    # ... and a default solver built AFTER it must still run fp32, bit-identical
    # to the clean-process reference (it would differ if it silently inherited
    # int8 via a leaked env).
    g_after = run(default_solver())
    assert K not in os.environ
    max_abs = (g_after - g_ref).abs().max().item()
    assert max_abs < 1e-6, f"default solver after int8 diverged from fp32 ref ({max_abs})"

    # Sanity: int8 boundary storage still produces a valid (near-identical,
    # slightly lossy) gradient -- i.e. the dtype really is being delivered.
    cos = float(torch.nn.functional.cosine_similarity(
        g_int8.flatten().unsqueeze(0), g_ref.flatten().unsqueeze(0))[0])
    assert cos > 0.99, f"int8 boundary gradient diverged unexpectedly (cos={cos})"


@pytest.mark.parametrize("name,value", [
    ("SWEEP_BOUNDARY_DTYPE", "bf16"),
    ("SWEEP_BOUNDARY_DTYPE", "fp32"),      # even the old default: the name is retired
    ("SWEEP_BOUNDARY_DTYPE", ""),
    ("SWEEP_FP16_BOUNDARY", "1"),
])
def test_retired_env_is_refused_at_construction(monkeypatch, name, value):
    """The environment used to set the storage dtype of the default ring.  A
    setting that changes the numbers belongs in the config, so a set variable
    is an error when the propagator is built -- on every backend, before any
    core is loaded (eager on the CPU here), whatever the value."""
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=rf"{name} is no longer read.*storage_dtype"):
        PropTorch(Acoustic(spatial_order=4, device="cpu", backend="torch"),
                  backend="torch", impl="eager", shape=(40, 48), dev="cpu", dh=(10.0, 10.0),
                  dt=0.0015, source_type=["h1"], receiver_type=["h1"], abcn=20,
                  pml_type="cpmlr", nt=60, B=1, allow_growth=True)

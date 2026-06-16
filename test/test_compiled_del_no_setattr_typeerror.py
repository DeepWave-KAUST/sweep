"""Regression: ``_CompiledPropagator.__del__`` must not raise at shutdown.

``__del__`` calls ``_remove_boundary_disk_cache``.  That method used to clear
its book-keeping attrs with plain ``self._boundary_disk_root = None`` /
``self._boundary_disk_files = ()`` assignments, which route through
``nn.Module.__setattr__`` -> ``isinstance(value, Parameter)`` ->
``Parameter.__instancecheck__`` -> ``isinstance(instance, torch.Tensor)``.

At interpreter shutdown ``torch.Tensor`` is torn down to ``None`` before the
surviving propagator's finalizer runs, so that inner ``isinstance`` raises
``TypeError: isinstance() arg 2 must be a type ...`` -- surfaced as the noisy
"Exception ignored in __del__".  The fix writes straight to ``__dict__``.

Two guards:
  * a fast, CUDA-free unit test that reproduces the exact shutdown condition
    (``torch.Tensor is None``) and asserts the clear is silent; and
  * a subprocess end-to-end test (CUDA + compiled binding) that runs a real
    forward/backward and exits without ``del``, asserting clean stderr.
"""
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from sweep.propagator._c import _CompiledPropagator


def test_remove_boundary_disk_cache_silent_when_tensor_is_none():
    """Mimic interpreter shutdown: with torch.Tensor==None the clear must not
    touch nn.Module.__setattr__ (which would call isinstance(.., Parameter))."""
    obj = _CompiledPropagator.__new__(_CompiledPropagator)  # bypass __init__
    obj.__dict__["_boundary_disk_root"] = None
    obj.__dict__["_boundary_disk_files"] = ()

    saved = torch.Tensor
    torch.Tensor = None  # what the interpreter does to module globals at exit
    try:
        obj._remove_boundary_disk_cache()  # must not raise TypeError
    finally:
        torch.Tensor = saved

    assert obj.__dict__["_boundary_disk_root"] is None
    assert obj.__dict__["_boundary_disk_files"] == ()
    # The attrs must live in __dict__, not as registered params/buffers/modules.
    assert "_boundary_disk_root" not in getattr(obj, "_parameters", {})


def test_allocate_then_remove_boundary_disk_cache_roundtrip(tmp_path):
    """``_allocate_boundary_disk_files`` writes the attrs via setattr while the
    fixed ``_remove_boundary_disk_cache`` reads them via __dict__; the two must
    stay consistent.  Allocate a real temp dir, then remove must delete it and
    reset the attrs.  CUDA-free -- only the disk helpers are exercised."""
    obj = _CompiledPropagator.__new__(_CompiledPropagator)
    obj.__dict__["_boundary_disk_root"] = None
    obj.__dict__["_boundary_disk_files"] = ()

    obj._allocate_boundary_disk_files([(4, 5), (4, 5)], str(tmp_path))
    root = obj._boundary_disk_root
    assert isinstance(root, str) and os.path.isdir(root)
    assert len(obj._boundary_disk_files) == 2
    assert all(os.path.exists(p) for p in obj._boundary_disk_files)

    obj._remove_boundary_disk_cache()
    assert not os.path.exists(root), "remove did not delete the allocated dir"
    assert obj._boundary_disk_root is None          # readable via normal attr access
    assert obj._boundary_disk_files == ()


def _binding_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from sweep import is_torch_binding_available

        return bool(is_torch_binding_available())
    except Exception:
        return False


@pytest.mark.skipif(not _binding_ready(), reason="CUDA + compiled sweep._C required")
def test_compiled_propagator_del_clean_at_shutdown():
    """End-to-end: a surviving PropTorch(impl='c') must exit without printing
    an ignored-exception traceback from __del__."""
    script = textwrap.dedent(
        """
        import numpy as np, torch
        from sweep.equations import Acoustic
        from sweep.propagator.torch import PropTorch
        nz, nx, nt, dt, dh = 40, 48, 80, 0.0015, 10.0
        vp = np.full((nz, nx), 2000.0, dtype=np.float32); vp[nz//2:, :] += 300.0
        src = np.array([[nx//2, nz//4]], dtype=np.int32)
        rx = np.arange(2, nx-2, 4, dtype=np.int32)
        rec = np.stack([rx, np.full(rx.size, 2, dtype=np.int32)], -1)[None]
        t = np.arange(nt)*dt; f, d = 12.0, 0.08
        wav = torch.tensor(((1-2*(np.pi*f*(t-d))**2)*np.exp(-((np.pi*f*(t-d))**2))).astype(np.float32), device='cuda')
        s = PropTorch(Acoustic(spatial_order=4, device='cuda', backend='torch'),
                      backend='torch', impl='c', shape=(nz, nx), dev='cuda', dh=(dh, dh),
                      dt=dt, source_type=['h1'], receiver_type=['h1'], abcn=20,
                      pml_type='cpmlr', nt=nt, B=1, allow_growth=True)
        m = torch.tensor(vp, device='cuda', requires_grad=True)
        s(wav, src.copy(), rec.copy(), models=[m]).pow(2).mean().backward()
        assert torch.isfinite(m.grad).all()
        # intentionally do NOT del s -> __del__ fires at interpreter shutdown
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr
    assert "TypeError" not in proc.stderr, proc.stderr

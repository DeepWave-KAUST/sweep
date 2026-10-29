#!/usr/bin/env python3
"""Report whether this Python environment can run sweep, and on which backend.

Prints: Python / sweep / torch versions, torch's CUDA build and the GPU it sees,
whether sweep's compiled CUDA core (impl='c') is usable and loads, and whether
the eager backend runs a one-shot forward on the CPU (and the GPU). Exit code 0
when at least the eager CPU path works.

    python check_env.py
"""
from __future__ import annotations

import platform
import sys
import traceback
import warnings


def line(key, value):
    print(f"{key:<28} {value}")


def tiny_forward(device, impl):
    """One 32x32 Acoustic shot, 50 steps; returns the resolved impl."""
    import numpy as np
    import torch
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    kw = {}
    if impl == "eager":
        from sweep.propagator.options import EagerOptions
        kw["eager_options"] = EagerOptions(use_compile=False)  # keep it fast
    solver = PropTorch(Acoustic(device=device), shape=(32, 32), dh=10.0, dt=1e-3,
                       abcn=10, device=device, impl=impl, **kw)
    w = ricker(np.arange(50) * 1e-3 - 0.02, f=20.0).astype(np.float32)
    vp = torch.full((32, 32), 2000.0, device=device)
    with torch.no_grad():
        rec = solver(w, np.array([[16, 4]]), np.array([[[8, 4], [24, 4]]]), models=[vp])
    assert tuple(rec.shape) == (1, 50, 2, 1), rec.shape
    assert bool(torch.isfinite(rec).all())
    return solver.impl


def main() -> int:
    line("python", f"{sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    try:
        import torch
    except ImportError:
        line("torch", "NOT INSTALLED -- pip install torch (pick the CUDA build for your driver)")
        return 1
    line("torch", torch.__version__)
    line("torch CUDA build", torch.version.cuda or "none (CPU-only wheel)")
    cuda = torch.cuda.is_available()
    line("torch.cuda.is_available()", cuda)
    if cuda:
        cap = torch.cuda.get_device_capability(0)
        line("GPU 0", f"{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        if cap[0] >= 8:
            line("cudnn.allow_tf32", f"{torch.backends.cudnn.allow_tf32}  "
                 "(set False for FP32-accurate impl='eager' gradients)")
    elif torch.version.cuda:
        line("note", "CUDA torch but no usable GPU: driver too old for this torch build, "
                     "or no GPU visible (CUDA_VISIBLE_DEVICES).")

    try:
        import sweep
    except ImportError:
        line("sweep", "NOT INSTALLED -- pip install sweep-solver (or sweepx)")
        return 1
    line("sweep", f"{sweep.__version__}  ({sweep.__file__})")

    try:
        import jax  # noqa: F401
        line("jax (PropJax)", jax.__version__)
    except ImportError:
        line("jax (PropJax)", "not installed (optional: pip install 'sweep-solver[jax]')")

    from sweep.backend.torch import binding
    diag = binding.diagnostics()
    line("impl='c' usable", f"{diag.get('usable')}  (reason: {diag.get('reason')})")
    core = diag.get("shipped_core") or {}
    line("shipped CUDA core", f"{core.get('tag')} -- {core.get('reason')} "
                              f"(available: {core.get('available')})")

    ok = True
    try:
        line("eager forward on cpu", f"OK (impl={tiny_forward('cpu', 'eager')})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        line("eager forward on cpu", f"FAILED: {exc!r}")
        traceback.print_exc()
    line("compiled core on cpu", "none: there is no CPU core; CPU runs use impl='eager'")

    if cuda:
        if diag.get("usable"):
            try:
                sweep.precompile()  # loads the core (no-op build for a fitting shipped core)
                line("compiled core loads (cuda)", f"yes (is_compiled={binding.is_compiled()})")
                line("impl='c' forward on cuda", f"OK (impl={tiny_forward('cuda', 'c')})")
            except Exception as exc:  # noqa: BLE001
                line("impl='c' on cuda", f"FAILED: {exc!r}")
        else:
            line("impl='c' on cuda", "unavailable; PropTorch(impl=None) will run eager")
        try:
            with warnings.catch_warnings():  # the TF32 notice is reported above
                warnings.simplefilter("ignore", UserWarning)
                impl = tiny_forward("cuda", "eager")
            line("eager forward on cuda", f"OK (impl={impl})")
        except Exception as exc:  # noqa: BLE001
            line("eager forward on cuda", f"FAILED: {exc!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

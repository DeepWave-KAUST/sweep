"""Per-step overhead of the stepped forward API vs one monolithic call.

Measures, on one GPU, the same propagation run as (a) a single full-range
binding call and (b) nt calls of one step each with Python-side role
rotation — the DD production pattern minus the halo exchange. The gap,
divided by nt, is the per-step Python/launch overhead a DD rank pays.

Usage: PYTHONPATH=src python _dd_cuda/bench_step_overhead.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "test"))

from test_stepped_forward import build, capture  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
)

DEV = torch.device("cuda")
REPEATS = 5


def setup(ndim, shape, nt, abcn=20):
    import test_stepped_forward as T
    T.NT = nt  # build() reads module-level NT for wavelet length
    prop, wavelet, sources, receivers, models = build(ndim, abcn=abcn, nt=nt)
    # Override the tiny default shape with the benchmark size
    prop2 = None
    from sweep.equations import Acoustic, Acoustic3D
    from sweep.propagator.torch import PropTorch
    eq_cls = Acoustic if ndim == 2 else Acoustic3D
    equation = eq_cls(spatial_order=4, device=DEV, backend="torch")
    prop2 = PropTorch(
        equation, backend="torch", impl="c", shape=shape, dev=DEV,
        dh=10.0, dt=0.0015, source_type=["h1"], receiver_type=["h1"],
        abcn=abcn, free_surface=False, pml_type="cpmlr", nt=nt, B=1,
        use_ckpt=False, boundary_saving_config={"enabled": False},
    )
    if ndim == 2:
        nz, nx = shape
        vp = np.full(shape, 2500.0, dtype=np.float32)
        sources = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array([[[ix, 2] for ix in range(2, nx - 2, 50)]], dtype=np.int32)
    else:
        nz, ny, nx = shape
        vp = np.full(shape, 2500.0, dtype=np.float32)
        sources = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        receivers = np.array([[[ix, ny // 2, 2] for ix in range(2, nx - 2, 20)]], dtype=np.int32)
    cap = capture(prop2)
    with torch.no_grad():
        prop2(wavelet, sources, receivers, models=[torch.tensor(vp, device=DEV)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(9 if ndim == 2 else 12)]
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return func, p, L, acoustic_psi_pairs(ndim), int(p.nt)


def run_full(func, p, L, nt):
    for t in L:
        t.zero_()
    p.it_begin, p.it_end = 0, nt
    p.wavefields = L
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    func(p)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_stepped(func, p, L, psi_pairs, nt):
    for t in L:
        t.zero_()
    runner = SteppedBindingRunner(func, p, L, psi_pairs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for it in range(nt):
            runner.run_to(it + 1)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cases = [
        (2, (2048, 2048), 200),
        (2, (512, 512), 200),
        (3, (256, 256, 256), 60),
    ]
    for ndim, shape, nt in cases:
        func, p, L, psi_pairs, nt = setup(ndim, shape, nt)
        # warm-up >= 1.5 s wall (clock ramp)
        tw = time.perf_counter()
        while time.perf_counter() - tw < 1.6:
            run_full(func, p, L, nt)
        fulls = [run_full(func, p, L, nt) for _ in range(REPEATS)]
        steps = [run_stepped(func, p, L, psi_pairs, nt) for _ in range(REPEATS)]
        tf, ts = min(fulls), min(steps)
        per_step_over = (ts - tf) / nt * 1e6
        print(
            f"ndim={ndim} shape={shape} nt={nt}: "
            f"full={tf*1e3:.1f}ms stepped(k=1)={ts*1e3:.1f}ms "
            f"ratio={ts/tf:.3f} overhead={per_step_over:.1f}us/step "
            f"(step={tf/nt*1e6:.0f}us)"
        )


if __name__ == "__main__":
    main()

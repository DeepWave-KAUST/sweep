"""Finite-difference (directional) gradient check for ViscoAcoustic.

For a random interior direction v, compare the analytic directional derivative
<grad, v> against the central finite difference
    (L(m + eps*v) - L(m - eps*v)) / (2 eps)

Read the eps sweep, not any single number. The solver runs float32 (it
downcasts float64 models), so the FD is cancellation-limited: the ``lost``
column reports how many digits of the ~7 available die in ``lp - lm``. The
gradient is right where the FD *plateaus* -- at small eps the FD collapses into
quantization noise and can even flip sign, which says nothing about the
gradient. Judging accuracy from one small-eps point is how you talk yourself
into a phantom error.

Measured (RTX 6000 Ada, float32):
    vp  Q=30, eps=10  -> rel_err 3.8e-03
    Q   Q=10, eps=1   -> rel_err 8.1e-03

Q needs a low-Q model: at Q=30 attenuation barely moves the misfit and the FD
never climbs out of the noise floor.

Run:  PYTHONPATH=src python test/visco_acoustic_grad_fdcheck.py
"""
from __future__ import annotations

import numpy as np
import torch

from sweep.equations import ViscoAcoustic
from sweep.propagator.torch import PropTorch

NZ, NX, ABCN, SO = 48, 56, 30, 4
DH, DT, NT, F0 = 10.0, 1e-3, 300, 15.0
VP = 2000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def wav():
    t = np.arange(NT) * DT - 1.0 / F0
    a = (np.pi * F0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def geo():
    src = np.array([[NX // 2, 4]], dtype=np.int64)          # (x, z)
    rx = np.arange(4, NX - 4, 2, dtype=np.int64)
    rec = np.stack([rx, np.full(rx.size, 2, np.int64)], -1)[None]
    return src, rec


def models(vp=VP, q=30.0):
    return [torch.full((NZ, NX), float(vp), device=DEVICE),
            torch.full((NZ, NX), float(q), device=DEVICE),
            torch.full((NZ, NX), 2 * np.pi * F0, device=DEVICE)]


def prop():
    eq = ViscoAcoustic(spatial_order=SO, backend="torch", device=DEVICE)
    return PropTorch(eq, shape=(NZ, NX), dh=DH, dt=DT, dev=torch.device(DEVICE),
                     source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
                     impl="eager", use_ckpt=False)


W = wav()
SRC, REC = geo()


def fwd(ms, p):
    s = p.forward(W, SRC, REC, models=ms)
    return s[0] if isinstance(s, (tuple, list)) else s


def sweep(idx, name, q_val, epss):
    with torch.no_grad():
        target = fwd(models(vp=VP + 40.0, q=q_val + 3.0), prop()).detach()

    ms = [m.clone().requires_grad_(True) for m in models(q=q_val)]
    (fwd(ms, prop()) - target).pow(2).sum().backward()
    g = [m.grad for m in ms]

    mask = torch.zeros((NZ, NX), device=DEVICE)
    mask[8:-8, 8:-8] = 1.0          # interior only: keep the FD out of the PML
    v = torch.randn((NZ, NX), device=DEVICE) * mask
    v = v / v.norm()
    analytic = float((g[idx] * v).sum())

    print(f"\n{name}  (Q={q_val})   <grad,v> = {analytic:+.6e}")
    print(f"  {'eps':>8} {'FD':>16} {'rel_err':>10}   lost")
    for eps in epss:
        with torch.no_grad():
            mp = models(q=q_val); mp[idx] = mp[idx] + eps * v
            mm = models(q=q_val); mm[idx] = mm[idx] - eps * v
            lp = float((fwd(mp, prop()) - target).pow(2).sum())
            lm = float((fwd(mm, prop()) - target).pow(2).sum())
        fd = (lp - lm) / (2 * eps)
        rel = abs(analytic - fd) / max(abs(fd), 1e-30)
        lost = -np.log10(max(abs(lp - lm) / max(abs(lp), 1e-30), 1e-30))
        print(f"  {eps:>8g} {fd:>+16.6e} {rel:>10.3e}   1e-{lost:.1f}")


def main():
    torch.manual_seed(0)
    print("=" * 66)
    print(f"device={DEVICE}  grid={NZ}x{NX}  abcn={ABCN}  order={SO}  nt={NT}")
    print("=" * 66)
    sweep(0, "vp", q_val=30.0, epss=(30.0, 10.0, 3.0, 1.0, 0.3))
    sweep(1, "Q", q_val=10.0, epss=(3.0, 1.0, 0.3, 0.1, 0.03))


if __name__ == "__main__":
    main()

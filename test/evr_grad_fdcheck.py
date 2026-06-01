"""Decisive finite-difference (directional) gradient check.

For a random direction v in each model parameter, compare the directional
derivative <grad, v> against the central finite difference
   (L(m + eps*v) - L(m - eps*v)) / (2 eps)
where L is evaluated with the CUDA forward (impl='c').

- If <grad_eager, v> matches the FD  -> eager autograd is the true grad of
  the (cuda~eager) forward.
- If <grad_c, v> does NOT match the FD -> the CUDA adjoint is wrong.

C-only scenario (R=0, const V) so only term C is active, isolating the
core adjoint from the pointwise/chain-rule additions.
"""
from __future__ import annotations
import numpy as np
import torch

from sweep.equations import ElasticVRR
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda"
NZ, NX = 48, 56
DH, DT, NT = 10.0, 1.5e-3, 120
ABCN, SO, FREQ, DELAY = 30, 4, 10.0, 0.06
NAMES = ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z")


def wav():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def geo():
    s = np.array([[NX // 2, NZ // 4]], dtype=np.int64)
    rx = np.arange(2, NX - 2, 6, dtype=np.int64)
    r = np.stack([rx, np.full_like(rx, 2)], axis=-1)[None, ...]
    return s, r


def prop(impl):
    eq = ElasticVRR(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(eq, shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, use_ckpt=False,
                     impl=impl, eager_options={"use_compile": False} if impl == "eager" else None)


def base_models():
    vp = torch.full((NZ, NX), 2000.0, device=DEVICE)
    vs = vp / 1.73
    zer = torch.zeros((NZ, NX), device=DEVICE)
    return [vp.clone(), vs.clone(), zer.clone(), zer.clone(), zer.clone(), zer.clone()]


def loss_c(models, w, s, r, target):
    ms = [m.clone() for m in models]
    syn = prop("c")(w, s, r, models=ms)
    return float((syn - target).pow(2).sum().item())


def grads(impl, w, s, r, target):
    ms = [m.clone().requires_grad_(True) for m in base_models()]
    syn = prop(impl)(w, s, r, models=ms)
    (syn - target).pow(2).sum().backward()
    return {n: m.grad.detach().clone() for n, m in zip(NAMES, ms)}


def main():
    torch.manual_seed(0)
    w = wav(); s, r = geo()
    # target from perturbed vp
    tms = base_models(); tms[0] = tms[0] + 40.0
    with torch.no_grad():
        target = prop("c")(w, s, r, models=tms).detach()

    base = base_models()
    eg = grads("eager", w, s, r, target)
    cg = grads("c", w, s, r, target)

    # restrict the random direction to interior cells (avoid PML edges) to
    # keep the FD clean and physically meaningful.
    mask = torch.zeros((NZ, NX), device=DEVICE)
    mask[6:-6, 6:-6] = 1.0

    print(f"{'param':>6}  {'<g_eager,v>':>14} {'<g_c,v>':>14} {'FD(cuda)':>14}"
          f"  {'eager/FD':>9} {'c/FD':>9}")
    eps_map = {"vp": 1.0, "vs": 1.0, "Rp_x": 1e-3, "Rp_z": 1e-3, "Rs_x": 1e-3, "Rs_z": 1e-3}
    for i, n in enumerate(NAMES):
        v = (torch.randn((NZ, NX), device=DEVICE) * mask)
        v = v / v.norm()
        eps = eps_map[n]
        mp = [m.clone() for m in base]; mp[i] = mp[i] + eps * v
        mm = [m.clone() for m in base]; mm[i] = mm[i] - eps * v
        Lp = loss_c(mp, w, s, r, target)
        Lm = loss_c(mm, w, s, r, target)
        fd = (Lp - Lm) / (2 * eps)
        dir_e = float((eg[n] * v).sum().item())
        dir_c = float((cg[n] * v).sum().item())
        print(f"{n:>6}  {dir_e:>14.6e} {dir_c:>14.6e} {fd:>14.6e}"
              f"  {dir_e/fd if fd!=0 else float('nan'):>9.4f} {dir_c/fd if fd!=0 else float('nan'):>9.4f}")


if __name__ == "__main__":
    main()

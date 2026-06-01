"""Probe: is the EVR grad discrepancy per-step (algebra) or accumulated
(timing)? Sweep nt and watch cos. Also dump a spatial error map for the
simplest pure-pointwise gradient (grad_Rp_z) in the C-only scenario.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sweep.equations import ElasticVectorReflectivity
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda"
NZ, NX = 48, 56
DH, DT = 10.0, 1.5e-3
ABCN, SO, FREQ, DELAY = 30, 4, 10.0, 0.06
NAMES = ("vp", "vs", "Rp_x", "Rp_z", "Rs_x", "Rs_z")
OUT = os.environ.get(
    "EVR_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs", "evr_grad_layered"),
)
os.makedirs(OUT, exist_ok=True)


def wav(nt):
    t = np.arange(nt, dtype=np.float32) * DT - DELAY
    return torch.tensor((1e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def geo():
    s = np.array([[NX // 2, NZ // 4]], dtype=np.int64)
    rx = np.arange(2, NX - 2, 6, dtype=np.int64)
    r = np.stack([rx, np.full_like(rx, 2)], axis=-1)[None, ...]
    return s, r


def prop(impl):
    eq = ElasticVectorReflectivity(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(eq, shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, use_ckpt=False,
                     impl=impl, eager_options={"use_compile": False} if impl == "eager" else None)


def models(requires_grad):
    # C-only: constant V, R=0
    vp = torch.full((NZ, NX), 2000.0, device=DEVICE)
    vs = vp / 1.73
    zer = torch.zeros((NZ, NX), device=DEVICE)
    ms = [vp.clone(), vs.clone(), zer.clone(), zer.clone(), zer.clone(), zer.clone()]
    if requires_grad:
        for m in ms:
            m.requires_grad_(True)
    return ms


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def grads(impl, w, s, r, target):
    ms = models(True)
    syn = prop(impl)(w, s, r, models=ms)
    (syn - target).pow(2).sum().backward()
    return {n: m.grad.detach().clone() for n, m in zip(NAMES, ms)}


def main():
    s, r = geo()
    print("nt-sweep (C-only), cos per parameter:")
    print(f"{'nt':>5} " + " ".join(f"{n:>8}" for n in NAMES))
    for nt in [1, 2, 3, 5, 10, 30, 120]:
        w = wav(nt)
        tms = models(False)
        with torch.no_grad():
            tms[0] = tms[0] + 40.0
            target = prop("eager")(w, s, r, models=tms).detach()
        eg = grads("eager", w, s, r, target)
        cg = grads("c", w, s, r, target)
        row = f"{nt:>5} "
        for n in NAMES:
            if eg[n].abs().max().item() < 1e-30:
                row += f"{'--':>8} "
            else:
                row += f"{cos(cg[n], eg[n]):>+8.4f} "
        print(row)

    # spatial error map at full nt for grad_Rp_z (pure pointwise) and grad_vp
    nt = 120
    w = wav(nt)
    tms = models(False)
    with torch.no_grad():
        tms[0] = tms[0] + 40.0
        target = prop("eager")(w, s, r, models=tms).detach()
    eg = grads("eager", w, s, r, target)
    cg = grads("c", w, s, r, target)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for col, name in enumerate(("Rp_z", "vp", "Rs_x")):
        e = eg[name].squeeze().cpu().numpy()
        c = cg[name].squeeze().cpu().numpy()
        vmax = np.percentile(np.abs(e), 99)
        axes[0, col].imshow(e, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        axes[0, col].set_title(f"eager {name}")
        ratio = c / np.where(np.abs(e) > 0.05 * np.abs(e).max(), e, np.nan)
        im = axes[1, col].imshow(ratio, vmin=0.5, vmax=1.5, cmap="RdBu_r", aspect="auto")
        axes[1, col].set_title(f"c/eager {name} (white=1)")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046)
    out = os.path.join(OUT, "evr_grad_probe_ratio.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved ratio map: {out}")

    # print the c/eager ratio statistics in the interior (exclude PML + source halo)
    for name in ("Rp_z", "vp", "Rs_x"):
        e = eg[name].squeeze().cpu().numpy()
        c = cg[name].squeeze().cpu().numpy()
        m = np.abs(e) > 0.2 * np.abs(e).max()
        ratio = c[m] / e[m]
        print(f"  {name}: c/eager over strong-signal cells  median={np.median(ratio):.4f}  "
              f"mean={np.mean(ratio):.4f}  std={np.std(ratio):.4f}")


if __name__ == "__main__":
    main()

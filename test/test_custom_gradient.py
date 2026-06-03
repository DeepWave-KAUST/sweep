"""Test the register_gradient public API (eager path).

  1. gradient mode + standard imaging condition → vp.grad matches autograd (cos≈1)
  2. gradient mode + raw uf*ub → a different gradient (the customization works)
  3. imaging mode → prop.imaging() returns an image, param.grad untouched
"""
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator import PropTorch


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def build(device, nz=48, nx=56, abcn=30, M=2, dt=0.0015, dh=10.0, nt=120):
    eq = Acoustic(spatial_order=2 * M, device=device, backend="torch")
    prop = PropTorch(eq, shape=(nz, nx), abcn=abcn, dh=dh, dt=dt, nt=nt,
                     dev=device, impl="eager", use_compile=False, use_ckpt=False)
    return eq, prop


def standard_imaging(eq):
    def fn(forward, adjoint, models, dt, h):
        vp = models[0]
        hz, hx = eq._spacings_2d(h)
        lz, lx = eq.separable_d2_2d(forward['h1'], eq.laplace_kernels, hz, hx)
        return 2.0 * dt * dt * vp * (lz + lx) * adjoint['h1']
    return fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    nt = 120
    vp_np = np.full((48, 56), 2000.0, dtype=np.float32)
    vp_np[16:, :] += 200.0
    wavelet = torch.as_tensor(ricker(nt, 0.0015, 10.0, 0.06), device=device)
    sources = torch.as_tensor([[[28, 2]]], device=device)
    receivers = torch.as_tensor([[[rx, 2] for rx in range(0, 56, 4)]], device=device)

    # ---------- reference: pure autograd gradient ----------
    eq, prop = build(device)
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec.pow(2).sum()).backward()
    ref_grad = vp.grad.detach().clone()
    print(f"[ref] autograd grad norm={ref_grad.norm().item():.4e}")

    # ---------- 1. gradient mode + standard imaging ----------
    eq, prop = build(device)
    prop.register_gradient("vp", standard_imaging(eq), mode="gradient")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec_c = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec_c.pow(2).sum()).backward()
    g_custom = vp.grad.detach().clone()
    print(f"\n[1] gradient mode, standard imaging:")
    print(f"    record match vs autograd fwd: max|Δ|={(rec_c - rec).abs().max().item():.2e}")
    print(f"    grad norm={g_custom.norm().item():.4e}  "
          f"cosine vs autograd = {cosine(g_custom, ref_grad):.4f}")

    # ---------- 2. gradient mode + raw uf*ub ----------
    eq, prop = build(device)
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd['h1'] * adj['h1'], mode="gradient")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    rec_x = prop(wavelet, sources, receivers, models=[vp])
    (0.5 * rec_x.pow(2).sum()).backward()
    g_xcorr = vp.grad.detach().clone()
    print(f"\n[2] gradient mode, raw uf*ub:")
    print(f"    grad norm={g_xcorr.norm().item():.4e}  "
          f"cosine vs autograd = {cosine(g_xcorr, ref_grad):.4f}  (expected != 1)")

    # ---------- 3. imaging mode ----------
    eq, prop = build(device)
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd['h1'] * adj['h1'], mode="imaging")
    vp = torch.as_tensor(vp_np, device=device).requires_grad_(True)
    images = prop.imaging(wavelet, sources, receivers, models=[vp])
    print(f"\n[3] imaging mode:")
    print(f"    returned keys: {list(images.keys())}")
    print(f"    image['vp'] shape={tuple(images['vp'].shape)} "
          f"norm={images['vp'].norm().item():.4e}")
    print(f"    vp.grad is None (untouched): {vp.grad is None}")


if __name__ == "__main__":
    main()

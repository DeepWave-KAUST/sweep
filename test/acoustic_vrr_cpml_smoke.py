"""Smoke + correctness check for the CPML-ified AcousticVRR (Python/eager path).

Run with the *edited* checkout on PYTHONPATH, e.g.:
    PYTHONPATH=/ibex/user/wangs0j/sweep-stack/sweep/src \
        python test/acoustic_vrr_cpml_smoke.py
"""
import numpy as np
import torch

import sweep
from sweep.equations.acoustic_vrr import AcousticVRR
from sweep.equations import AcousticVRZ, Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

print("sweep imported from:", sweep.__file__)
print("AcousticVRR from   :", AcousticVRR.__module__)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ---- geometry (small, surface acquisition) --------------------------------
nz, nx = 64, 88
dh, dt, nt = 10.0, 0.001, 220
abcn = 30
so = 4

t = np.arange(nt, dtype=np.float32) * dt
wavelet = ricker(t - 0.08, f=12.0).astype(np.float32)[None, :]          # (B=1, nt)
src = np.array([[nx // 2, 4]], dtype=np.int64)                          # (x, z) surface
rec_x = np.arange(6, nx - 6, 4, dtype=np.int64)
recv = np.stack([rec_x, np.full_like(rec_x, 2)], axis=1)[None]          # (1, nrec, 2)


def run(eq_cls, models, *, return_wf=False, wl=None, **fwd):
    eq = eq_cls(spatial_order=so, device=dev, backend="torch")
    prop = PropTorch(eq, (nz, nx), source_type=["h1"], receiver_type=["h1"],
                     abcn=abcn, dh=(dh, dh), dt=dt, device=dev, impl="eager",
                     use_ckpt=False)
    mdls = [torch.tensor(m, dtype=torch.float32, device=dev) for m in models]
    w = torch.tensor(wavelet if wl is None else wl, device=dev)
    s = torch.tensor(src, device=dev)
    r = torch.tensor(recv, device=dev)
    out = prop(w, s, r, models=mdls, return_wavefield=return_wf, **fwd)
    if return_wf:
        rec, wf = out
        return rec, wf
    return out, None


# ===========================================================================
# Test A — homogeneous cross-check: VRR(rx=rz=0) == VRZ(z=const) == Acoustic
# ===========================================================================
vp0 = np.full((nz, nx), 2000.0, dtype=np.float32)
zero = np.zeros((nz, nx), dtype=np.float32)
zconst = np.full((nz, nx), 2.0e6, dtype=np.float32)   # impedance, any constant

rec_vrr, _ = run(AcousticVRR, [vp0, zero, zero])
rec_vrz, _ = run(AcousticVRZ, [vp0, zconst])
rec_aco, _ = run(Acoustic, [vp0])

rec_vrr = rec_vrr.detach().cpu().numpy()
rec_vrz = rec_vrz.detach().cpu().numpy()
rec_aco = rec_aco.detach().cpu().numpy()

den = np.abs(rec_vrz).max() + 1e-30
rel_vrz = np.abs(rec_vrr - rec_vrz).max() / den
rel_aco = np.abs(rec_vrr - rec_aco).max() / (np.abs(rec_aco).max() + 1e-30)
print(f"\n[A] homogeneous medium, peak |trace| VRR={np.abs(rec_vrr).max():.4e}")
print(f"[A] max rel diff  VRR vs VRZ = {rel_vrz:.3e}   (expect ~1e-6, identical update)")
print(f"[A] max rel diff  VRR vs Acoustic = {rel_aco:.3e} (informational)")
assert np.isfinite(rec_vrr).all(), "VRR produced non-finite receiver data"
assert rel_vrz < 1e-4, f"VRR != VRZ on homogeneous medium (rel {rel_vrz:.3e})"

# ===========================================================================
# Test B — variable medium: stability + CPML absorption (interior energy decays)
# ===========================================================================
z = np.linspace(0, 1, nz, dtype=np.float32)[:, None]
vp = (1800.0 + 600.0 * z + np.zeros((nz, nx), dtype=np.float32)).astype(np.float32)
vp[nz // 2 - 4:nz // 2 + 4, nx // 2 - 6:nx // 2 + 6] += 180.0
# small density-coupling parameters r ~ -0.5 d(ln Z)/dx ; here just a mild field
rho = 1000.0 + 200.0 * z + np.zeros((nz, nx), dtype=np.float32)
Z = (rho * vp).astype(np.float32)
lnZ = np.log(Z)
rz = (-0.5 * np.gradient(lnZ, dh, axis=0)).astype(np.float32)
rx = (-0.5 * np.gradient(lnZ, dh, axis=1)).astype(np.float32)

# Long run so the wavefront fully crosses the ~880x640 m domain and exits
# through the CPML (domain crossing ~0.4 s at ~2 km/s; use 0.9 s).
nt_long = 900
t_long = np.arange(nt_long, dtype=np.float32) * dt
wl_long = ricker(t_long - 0.08, f=12.0).astype(np.float32)[None, :]
_, wf = run(AcousticVRR, [vp, rx, rz], return_wf=True, wl=wl_long,
            snapshot_interval=20)
# wf: (n_snapshots, n_fields, B, 1, nz, nx) -> take h1 (field 0)
wf = wf.detach().cpu().numpy()
h1 = wf[:, 0, 0, 0]                                   # (nsnap, nz, nx)
interior = h1[:, abcn:-abcn, abcn:-abcn] if h1.shape[1] > 2 * abcn else h1
energy = (interior ** 2).sum(axis=(1, 2))
peak = energy.max()
tail = energy[-1]
print(f"\n[B] variable medium, snapshots={h1.shape[0]} (nt={nt_long}, T={nt_long*dt:.2f}s)")
print(f"[B] interior energy  peak={peak:.4e}  final={tail:.4e}  ratio={tail / (peak + 1e-30):.3e}")
assert np.isfinite(h1).all(), "VRR variable-medium wavefield went non-finite"
assert tail < 1e-3 * peak, f"CPML not absorbing: tail/peak={tail/(peak+1e-30):.3e}"

# ===========================================================================
# Test C — autograd gradient flows (eager backward works through CPML step)
# ===========================================================================
eq = AcousticVRR(spatial_order=so, device=dev, backend="torch")
prop = PropTorch(eq, (nz, nx), source_type=["h1"], receiver_type=["h1"],
                 abcn=abcn, dh=(dh, dh), dt=dt, device=dev, impl="eager",
                 use_ckpt=False)
vp_t = torch.tensor(vp, dtype=torch.float32, device=dev, requires_grad=True)
rx_t = torch.tensor(rx, dtype=torch.float32, device=dev, requires_grad=True)
rz_t = torch.tensor(rz, dtype=torch.float32, device=dev, requires_grad=True)
w = torch.tensor(wavelet, device=dev)
s = torch.tensor(src, device=dev)
r = torch.tensor(recv, device=dev)
pred = prop(w, s, r, models=[vp_t, rx_t, rz_t])
loss = (pred ** 2).sum()
loss.backward()
gvp = vp_t.grad
print(f"\n[C] loss={loss.item():.4e}  grad_vp finite={torch.isfinite(gvp).all().item()}  "
      f"|grad_vp|max={gvp.abs().max().item():.3e}")
assert torch.isfinite(gvp).all() and gvp.abs().max() > 0, "grad_vp did not flow"

print("\nALL CHECKS PASSED ✅")

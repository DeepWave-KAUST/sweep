"""Localize the EVR backward gradient discrepancy by isolating the three
gamma sub-terms:

    gamma_a_ij = V(d_i V) p_j        [term A: velocity-gradient, needs chain rule]
               - 2 V^2 R_a,i p_j     [term B: reflectivity pointwise]
               + V^2 d_i p_j         [term C: divergence-of-p, = elastic w/ rho=1]

Scenarios:
  C-only   : R=0, constant V          -> only term C active (pointwise A,B vanish).
             grad_R still tests dGamma/dR = -2V^2 p (nonzero even at R=0).
  C+B      : R!=0, constant V          -> adds term B + grad_R pointwise path.
  C+A      : R=0, variable V           -> adds term A + chain-rule through dV.
  full     : R!=0, variable V          -> everything (the real case).

For each scenario + parameter, report cos and rel_l2 (c vs eager autograd).
Whichever scenario first drops cos below ~0.999 points at the buggy term.
"""

from __future__ import annotations
import numpy as np
import torch

from sweep.equations import ElasticVRR, compute_vector_reflectivity
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


def make_models(scenario, requires_grad):
    z = torch.arange(NZ, device=DEVICE, dtype=torch.float32).view(NZ, 1)
    if "A" in scenario:                      # variable velocity
        vp = (1800.0 + 8.0 * z).expand(NZ, NX).contiguous()
    else:                                    # constant velocity
        vp = torch.full((NZ, NX), 2000.0, device=DEVICE)
    vs = vp / 1.73

    if "B" in scenario:                      # nonzero reflectivity (from a rho contrast)
        rho = torch.full((NZ, NX), 1000.0, device=DEVICE)
        rho[NZ // 2:, :] = 1300.0
        Rp_x, Rp_z, Rs_x, Rs_z = compute_vector_reflectivity(vp, vs, rho, h=DH)
    else:
        zer = torch.zeros((NZ, NX), device=DEVICE)
        Rp_x, Rp_z, Rs_x, Rs_z = zer.clone(), zer.clone(), zer.clone(), zer.clone()

    ms = [vp.clone(), vs.clone(), Rp_x.clone(), Rp_z.clone(), Rs_x.clone(), Rs_z.clone()]
    if requires_grad:
        for m in ms:
            m.requires_grad_(True)
    return ms


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def rl2(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def grads(impl, scenario, w, s, r, target):
    ms = make_models(scenario, requires_grad=True)
    syn = prop(impl)(w, s, r, models=ms)
    (syn - target).pow(2).sum().backward()
    return {n: m.grad.detach().clone() for n, m in zip(NAMES, ms)}, syn.detach()


def run():
    w = wav(); s, r = geo()
    print(f"grid {NZ}x{NX} nt {NT}\n")
    for scenario, label in [("C", "C-only  (R=0, const V)"),
                            ("CB", "C+B     (R!=0, const V)"),
                            ("CA", "C+A     (R=0, var V)"),
                            ("CAB", "full    (R!=0, var V)")]:
        # target from a vp-perturbed model (same scenario structure)
        tms = make_models(scenario, requires_grad=False)
        with torch.no_grad():
            tms[0] = tms[0] + 40.0
            target = prop("eager")(w, s, r, models=tms).detach()

        eg, esyn = grads("eager", scenario, w, s, r, target)
        cg, csyn = grads("c", scenario, w, s, r, target)
        fwd = rl2(csyn, esyn)
        print(f"=== {label}   (forward rel_l2 {fwd:.1e}) ===")
        for n in NAMES:
            em = eg[n].abs().max().item()
            if em < 1e-30:
                print(f"  {n:5s}: eager~0 (inactive)")
                continue
            print(f"  {n:5s}: cos={cos(cg[n], eg[n]):+.4f}  rel_l2={rl2(cg[n], eg[n]):.4f}"
                  f"  emax={em:.3e} cmax={cg[n].abs().max().item():.3e}")
        print()


if __name__ == "__main__":
    run()

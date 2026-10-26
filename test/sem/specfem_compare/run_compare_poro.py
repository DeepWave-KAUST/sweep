"""BiotVSSEM2DDeformed against SPECFEM2D's ``poroelastic_semi_infinite_homogeneous`` example.

The example (Morency & Tromp 2008, Fig. 4): a homogeneous inviscid poroelastic square,
1000 m x 1000 m in 5 m Quad9 elements, free top and Stacey elsewhere, an explosion at
(500, 300) and a receiver at (600, 400) (z up), a Gaussian moment history of f0 = 30 Hz,
dt = 1e-4 and 2500 steps. Its reference solid velocities ``REF_SEIS/AA.S0001.BX[XZ].semv``
ship with SPECFEM2D. The record ends before any boundary echo reaches the receiver.

SPECFEM2D loads a poroelastic moment tensor as ``+M : grad(phi)`` on both the solid and the
fluid equation of Morency & Tromp's (u_s, w) form, i.e. the same vector on the
total-momentum and the relative-flow equation. In PropSEM's names that is a total-stress
source ``["sxx", "szz"]`` of history ``-M(t)`` plus a pore-pressure source ``["p"]`` of
``+M(t)`` (both the injected quantity, FD's sign); PropSEM takes their rates.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_poro.py \\
        --ref <specfem2d>/EXAMPLES/poroelastic_semi_infinite_homogeneous/REF_SEIS --out <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from sweep.equations.biot import biot_from_rock, biot_wave_velocities
from sweep.sem import BiotVSSEM2DDeformed, PropSEM

# Par_file, model 1: rho_s rho_f phi tort kxx kxz kzz K_s K_f K_fr eta_f mu_fr
ROCK = dict(k_s=12.2e9, rho_s=2650.0, k_f=1.985e9, rho_f=880.0, eta=0.0, k_d=9.6e9,
            mu=5.1e9, phi=0.1, kappa=1e-9, tortuosity=2.0)
L, H_EL, DT, NT, F0, FACTOR = 1000.0, 5.0, 1e-4, 2500, 30.0, 1e10
SRC, REC = (500.0, 300.0), (600.0, 400.0)          # SPECFEM (x, z), z up from the bottom


def specfem_stf(t):
    """SPECFEM2D's time_function_type 3: ``-factor * (-exp(-a t^2) / (2 a))``, a = (pi f0)^2,
    with its t0 = 1.2 / f0 already taken off ``t``."""
    a = (math.pi * F0) ** 2
    return FACTOR * np.exp(-a * t**2) / (2.0 * a)


def run(device, dtype, n_order=4):
    dh = H_EL / n_order
    n = int(round(L / dh)) + 1
    models = biot_from_rock(**ROCK)
    t0 = 1.2 / F0
    t = np.arange(NT) * DT
    m_hist = specfem_stf(t - t0)
    rate = torch.from_numpy(np.gradient(m_hist, t)).to(dtype)
    # grid indices [ix, iz], z down from the free top
    src = np.array([[SRC[0] / dh, (L - SRC[1]) / dh]])
    rec = np.array([[[REC[0] / dh, (L - REC[1]) / dh]]])
    coef = [torch.full((n, n), float(c), dtype=dtype) for c in models]
    out = 0.0
    for names, sign in ((["sxx", "szz"], -1.0), (["p"], 1.0)):
        prop = PropSEM(BiotVSSEM2DDeformed, shape=(n, n), dh=dh, dt=DT, n_order=n_order,
                       free_surface=True, f_dominant=F0, dev=device, impl="eager",
                       dtype=dtype, source_type=names, receiver_type=["vx", "vz"])
        with torch.no_grad():
            out = out + prop(sign * rate, src, rec, models=coef)[0, :, 0].double().cpu().numpy()
    vx, vz_down = out[:, 0], out[:, 1]
    # a velocity sample n is at (n + 1) dt for an integrated source (see test_biot_sem.py);
    # SPECFEM sample m is at m dt - t0
    t_sem = (np.arange(NT) + 1) * DT - t0
    return t_sem, vx, -vz_down, models


def read_semv(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--float64", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.float64 else torch.float32

    t_sem, vx, vz, models = run(args.device, dtype)
    vf, vsl, vs = biot_wave_velocities([float(m) for m in models])
    print(f"fast P {vf:.1f}  slow P {vsl:.1f}  S {vs:.1f} m/s")
    np.savez(args.out / "sem_biot_specfem_example.npz", t=t_sem, vx=vx, vz=vz)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, comp, ours in zip(axes, ("BXX", "BXZ"), (vx, vz)):
        t_ref, ref = read_semv(args.ref / f"AA.S0001.{comp}.semv")
        on_ref = np.interp(t_ref, t_sem, ours)
        err = np.linalg.norm(on_ref - ref) / np.linalg.norm(ref)
        corr = float(on_ref @ ref) / float(np.linalg.norm(on_ref) * np.linalg.norm(ref))
        print(f"{comp}: rel L2 {err:.4f}  corr {corr:.5f}  "
              f"amp {np.linalg.norm(on_ref) / np.linalg.norm(ref):.4f}")
        ax.plot(t_ref, ref, "k", lw=1.6, label="SPECFEM2D")
        ax.plot(t_ref, on_ref, "r--", lw=1.0, label="SEM Biot")
        ax.plot(t_ref, 5 * (on_ref - ref), color="0.6", lw=0.8, label="5x diff")
        ax.set_ylabel(f"{'vx' if comp == 'BXX' else 'vz (up)'} (m/s)")
        ax.text(0.01, 0.92, f"({'ab'[comp == 'BXZ']}) rel. L2 {err:.3f}", transform=ax.transAxes)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out / "sem_biot_vs_specfem2d.png", dpi=220)
    print("wrote", args.out / "sem_biot_vs_specfem2d.png")


if __name__ == "__main__":
    main()

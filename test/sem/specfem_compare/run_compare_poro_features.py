"""BiotVSSEM2DDeformed / BiotAnisoPermSEM2DDeformed against SPECFEM2D runs of the poroelastic
features: a point force (``force_z``), an anisotropic permeability (``aniso_k``) and the
viscous attenuation (``visc``, ``visc_atten``), on the box ``make_specfem_poro_cases.py``
builds -- 2000 m x 1600 m of 5 m Quad9 elements, free top, the source 600 m deep, four
receivers (two on the source's grid lines), a Gaussian of f0 = 30 Hz, dt = 1e-4, 2500 steps.
Solid velocity only: SPECFEM2D computes no pressure in poroelastic media.

SPECFEM2D's z points up: its BXZ is -v_z here, and its permeability's kxz flips sign. Its
poroelastic force loads the solid and the fluid equations alike (our ``fz``) and, since its
May 2018 sign fix (``sourcearray = (+sin, -cos) hlagrange``), points down at angle 0 for a
positive source time function; its moment tensor ``+M : grad(phi)`` on both is our total
stress ``-M`` plus pore pressure ``+M``.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_poro_features.py \\
        --runs <dir with the SPECFEM case directories> --out <dir> [--case force_z ...]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from sweep.equations.biot import biot_from_rock
from sweep.equations.biot_params import biot_drag_tensor
from sweep.sem import BiotAnisoPermSEM2DDeformed, BiotVSSEM2DDeformed, PropSEM

LX, LZ, H_EL, DT, NT, F0, FACTOR = 2000.0, 1600.0, 5.0, 1e-4, 2500, 30.0, 1e10
SRC = (1000.0, 1000.0)                                   # SPECFEM (x, z), z up
RECS = [(1100.0, 1100.0), (1100.0, 1000.0), (1000.0, 1100.0), (1150.0, 950.0)]
ROCK = dict(k_s=12.2e9, rho_s=2650.0, k_f=1.985e9, rho_f=880.0, k_d=9.6e9, mu=5.1e9, phi=0.1,
            tortuosity=2.0)
CASES = {   # source, eta, (kxx, kxz, kzz) z up, viscous_q
    "force_z": ("force", 0.0, (1e-9, 0.0, 1e-9), None),
    "aniso_k": ("explosion", 1e-3, (2e-9, 5e-10, 5e-10), None),
    "visc": ("explosion", 1e-3, (1e-9, 0.0, 1e-9), None),
    "visc_atten": ("explosion", 1e-3, (1e-9, 0.0, 1e-9), (5.0, 30.0)),
}


def stf(t):
    """SPECFEM2D's time_function_type 3 with t0 = 1.2 / f0 taken off ``t``."""
    a = (math.pi * F0) ** 2
    return FACTOR * np.exp(-a * t**2) / (2.0 * a)


def run_sem(case, device, dtype, n_order=4):
    source, eta, (kxx, kxz, kzz), viscous_q = CASES[case]
    dh = H_EL / n_order
    nz, nx = int(round(LZ / dh)) + 1, int(round(LX / dh)) + 1
    t0 = 1.2 / F0
    t = np.arange(NT) * DT
    coef = biot_from_rock(eta=eta, kappa=kxx, **ROCK)
    eq = BiotVSSEM2DDeformed
    if kxz != 0.0 or kxx != kzz:
        eq = BiotAnisoPermSEM2DDeformed
        coef = list(coef[:7]) + list(biot_drag_tensor(eta, kxx, -kxz, kzz))
    models = [torch.full((nz, nx), float(c), dtype=dtype) for c in coef]
    src = np.array([[SRC[0] / dh, (LZ - SRC[1]) / dh]])
    rec = np.array([[[x / dh, (LZ - z) / dh] for x, z in RECS]])
    kw = dict(shape=(nz, nx), dh=dh, dt=DT, n_order=n_order, free_surface=True, f_dominant=F0,
              dev=device, impl="eager", dtype=dtype, receiver_type=["vx", "vz", "p"])
    if viscous_q is not None:
        kw["equation_options"] = {"viscous_q": viscous_q}
    if source == "force":
        # a downward force stf(t): fz takes the force itself, down positive; a load at
        # step n is the force at n dt, a velocity sample n is at (n + 1/2) dt
        calls = ((["fz"], stf(t - t0)),)
        shifts = (0.5, 0.5, 1.0)
    else:
        # the stress and pressure sources take the rate; samples at (n + 1) dt (velocity)
        # and (n + 3/2) dt (pressure)
        rate = np.gradient(stf(t - t0), t)
        calls = ((["sxx", "szz"], -rate), (["p"], rate))
        shifts = (1.0, 1.0, 1.5)
    out = 0.0
    for names, w in calls:
        prop = PropSEM(eq, source_type=names, **kw)
        with torch.no_grad():
            out = out + prop(torch.from_numpy(w).to(dtype), src, rec,
                             models=models)[0].double().cpu().numpy()
    out[:, :, 1] *= -1.0                                  # v_z down -> SPECFEM's up
    return [(np.arange(NT) + s) * DT - t0 for s in shifts], out


def read_trace(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", nargs="*", default=list(CASES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case in args.case:
        t_sem, sem = run_sem(case, args.device, torch.float32)
        np.savez(args.out / f"sem_{case}.npz", t=np.stack(t_sem), sem=sem)
        fig, axes = plt.subplots(len(RECS), 2, figsize=(10, 2.3 * len(RECS)), sharex=True)
        for r in range(len(RECS)):
            for k, comp in enumerate(("BXX", "BXZ")):
                ax = axes[r, k]
                t_ref, ref = read_trace(args.runs / case / "OUTPUT_FILES"
                                        / f"AA.S{r + 1:04d}.{comp}.semv")
                ours = np.interp(t_ref, t_sem[k], sem[:, r, k])
                err = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
                corr = float(ours @ ref) / float(np.linalg.norm(ours) * np.linalg.norm(ref))
                amp = np.linalg.norm(ours) / np.linalg.norm(ref)
                print(f"{case} rec{r + 1} {comp}: rel L2 {err:.4f} corr {corr:+.5f} amp {amp:.4f}",
                      flush=True)
                ax.plot(t_ref, ref, "k", lw=1.4, label="SPECFEM2D")
                ax.plot(t_ref, ours, "r--", lw=0.9, label="SEM")
                ax.plot(t_ref, 5 * (ours - ref), color="0.6", lw=0.7, label="5x diff")
                ax.set_title(f"({'abcdefgh'[2 * r + k]}) rec {r + 1} {comp}, rel. L2 {err:.4f}",
                             fontsize=8)
        axes[0, 0].legend(frameon=False, fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(args.out / f"sem_vs_specfem2d_{case}.png", dpi=220)
        plt.close(fig)
        print("wrote", args.out / f"sem_vs_specfem2d_{case}.png", flush=True)


if __name__ == "__main__":
    main()

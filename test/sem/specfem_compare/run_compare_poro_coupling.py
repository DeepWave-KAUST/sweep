"""BiotVSSEM2DDeformed with fluid and dry elastic media against SPECFEM2D:

* ``pa_example``: SPECFEM2D's ``poroelastic_acoustic`` example (Morency & Tromp 2008, Fig. 8)
  -- water over an inviscid porous half-space, a pressure source in the water --, and
  ``pa_example_fine``, the same on elements half as large;
* ``pe_layers``: a viscous porous rock over a dry elastic one, an explosion in the rock;
* ``ape_layers``: water, the porous rock, the elastic rock; a pressure source in the water,

the last two on the box of ``run_compare_poro_features.py`` (cases from
``make_specfem_poro_cases.py``). Velocities only (SPECFEM2D's ``seismotype = 2``): in the water
the fluid's, our ``q``; elsewhere the solid's, our ``v``. The source time function is
SPECFEM2D's own (``plot_source_time_function.txt``).

SPECFEM2D's z points up (its BXZ is -v_z here). A force in an acoustic element is its
pressure source, ``chi'' = kappa div(grad(chi) / rho) + stf delta`` with ``p = -chi''``:
our pore-pressure injection of the wavelet ``-d stf / dt``. Its moment tensor
``+M : grad(phi)`` in a porous element is our total stress ``-M`` plus pore pressure ``+M``.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_poro_coupling.py \\
        --runs <dir with the SPECFEM case directories> --out <dir> [--case pa_example ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sweep.equations.biot import biot_from_rock
from sweep.equations.biot_params import biot_elastic, biot_fluid
from sweep.sem import BiotVSSEM2DDeformed, PropSEM

ROCK = biot_from_rock(k_s=12.2e9, rho_s=2650.0, k_f=1.985e9, rho_f=880.0, eta=1e-3, k_d=9.6e9,
                      mu=5.1e9, phi=0.1, kappa=1e-9, tortuosity=2.0)
PA_ROCK = biot_from_rock(k_s=16.0554e9, rho_s=2500.0, k_f=2.295e9, rho_f=1020.0, eta=0.0,
                         k_d=10.0e9, mu=9.63342e9, phi=0.4, kappa=1e-9, tortuosity=2.0)
WATER = biot_fluid(1020.0 * 1500.0**2, 1020.0)
SHALE = biot_elastic(4000.0, 2300.0, 2500.0)

# name: box (LX, LZ, element), f0, dt, nt, media top-down (coefficients, thickness),
#       source kind, source (x, z up), receivers (x, z up)
CASES = {
    "pa_example": ((4800.0, 4800.0, 4800.0 / 220), 15.0, 4.2e-4, 2500,
                   [(WATER, 2400.0), (PA_ROCK, 2400.0)], "pressure", (1600.0, 2900.0),
                   [(2000.0, 2934.0), (2000.0, 1867.0)]),
    # the example on elements half as large: its inviscid slow wave (1060 m/s) has ~1.2
    # elements per wavelength at 40 Hz on the example's own mesh
    "pa_example_fine": ((4800.0, 4800.0, 4800.0 / 440), 15.0, 2.1e-4, 5000,
                        [(WATER, 2400.0), (PA_ROCK, 2400.0)], "pressure", (1600.0, 2900.0),
                        [(2000.0, 2934.0), (2000.0, 1867.0)]),
    "pe_layers": ((2000.0, 1600.0, 5.0), 30.0, 1e-4, 2500, [(ROCK, 800.0), (SHALE, 800.0)],
                  "explosion", (1000.0, 1000.0),
                  [(1100.0, 1100.0), (1100.0, 850.0), (1050.0, 750.0), (1150.0, 650.0)]),
    "ape_layers": ((2000.0, 1600.0, 5.0), 30.0, 1e-4, 5000,
                   [(WATER, 300.0), (ROCK, 500.0), (SHALE, 800.0)], "pressure",
                   (1000.0, 1450.0),
                   [(1100.0, 1400.0), (1100.0, 1250.0), (1050.0, 1000.0), (1150.0, 700.0)]),
}


def layered(mesh, media, dtype):
    """GLL-native coefficients of horizontal layers, ``media`` top-down as
    ``(coefficients, thickness)``; the last one fills the rest (the C-PML below)."""
    zc = mesh.z_e.mean((1, 2))
    edges = torch.tensor(np.cumsum([t for _, t in media])[:-1], dtype=zc.dtype, device=zc.device)
    k = torch.bucketize(zc, edges)
    E, n = mesh.n_elements, mesh.ngll
    return [torch.stack([torch.as_tensor(m[i], dtype=dtype) for m, _ in media]).to(zc.device)[k]
            .view(E, 1, 1).expand(E, n, n).contiguous() for i in range(8)]


def in_water(case, z_up):
    (_, lz, _), media = CASES[case][0], CASES[case][4]
    return media[0][0] is WATER and lz - z_up < media[0][1]


def run_sem(case, stf, device, dtype, n_order=4):
    (lx, lz, h), f0, dt, nt, media, kind, (xs, zs), recs = CASES[case]
    dh = h / n_order
    nz, nx = int(round(lz / dh)) + 1, int(round(lx / dh)) + 1
    kw = dict(shape=(nz, nx), dh=dh, dt=dt, n_order=n_order, free_surface=True, f_dominant=f0,
              dev=device, impl="eager", dtype=dtype, receiver_type=["vx", "vz", "qx", "qz"])
    src = np.array([[xs / dh, (lz - zs) / dh]])
    rec = np.array([[[x / dh, (lz - z) / dh] for x, z in recs]])
    rate = np.gradient(stf, dt)
    if kind == "explosion":
        calls = ((["sxx", "szz"], -rate), (["p"], rate))
    else:
        calls = ((["p"], -rate),)
    out, models = 0.0, None
    for names, w in calls:
        prop = PropSEM(BiotVSSEM2DDeformed, source_type=names, **kw)
        if models is None:
            models = layered(prop.mesh, media, dtype)
        with torch.no_grad():
            out = out + prop(torch.from_numpy(w).to(dtype), src, rec,
                             models=models)[0].double().cpu().numpy()
    # an integrated source's velocity samples sit at (n + 1) dt
    traces = np.stack([out[:, r, 2:4] if in_water(case, z) else out[:, r, 0:2]
                       for r, (_, z) in enumerate(recs)], 1)
    traces[:, :, 1] *= -1.0                                # z down -> SPECFEM's up
    return (np.arange(nt) + 1.0) * dt, traces


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
        run = args.runs / case / "OUTPUT_FILES"
        t_stf, stf = np.loadtxt(run / "plot_source_time_function.txt", unpack=True)
        t_sem, sem = run_sem(case, stf, args.device, torch.float32)
        t_sem = t_sem + t_stf[0]                           # SPECFEM's time axis, t - t0
        np.savez(args.out / f"sem_{case}.npz", t=t_sem, sem=sem)
        recs = CASES[case][7]
        fig, axes = plt.subplots(len(recs), 2, figsize=(10, 2.3 * len(recs)), sharex=True,
                                 squeeze=False)
        for r, (x, z) in enumerate(recs):
            med = "water" if in_water(case, z) else "solid"
            for k, comp in enumerate(("BXX", "BXZ")):
                ax = axes[r, k]
                t_ref, ref = np.loadtxt(run / f"AA.S{r + 1:04d}.{comp}.semv", unpack=True)
                ours = np.interp(t_ref, t_sem, sem[:, r, k])
                if not (np.linalg.norm(ref) > 0 and np.linalg.norm(ours) > 0):
                    print(f"{case} rec{r + 1} ({med}) {comp}: no arrival in the record", flush=True)
                    continue
                err = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
                corr = float(ours @ ref) / float(np.linalg.norm(ours) * np.linalg.norm(ref))
                amp = np.linalg.norm(ours) / np.linalg.norm(ref)
                print(f"{case} rec{r + 1} ({med}) {comp}: rel L2 {err:.4f} corr {corr:+.5f} "
                      f"amp {amp:.4f}", flush=True)
                ax.plot(t_ref, ref, "k", lw=1.4, label="SPECFEM2D")
                ax.plot(t_ref, ours, "r--", lw=0.9, label="SEM")
                ax.plot(t_ref, 5 * (ours - ref), color="0.6", lw=0.7, label="5x diff")
                ax.set_title(f"({'abcdefgh'[2 * r + k]}) rec {r + 1} {med} {comp}, "
                             f"rel. L2 {err:.4f}", fontsize=8)
        axes[0, 0].legend(frameon=False, fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(args.out / f"sem_vs_specfem2d_{case}.png", dpi=220)
        plt.close(fig)
        print("wrote", args.out / f"sem_vs_specfem2d_{case}.png", flush=True)


if __name__ == "__main__":
    main()

"""ElasticVTISEM2DDeformed and ElasticTTISEM2DDeformed against SPECFEM2D's anisotropic elastic
(material type 2) on the boxes ``make_specfem_aniso_cases.py`` builds: one homogeneous medium,
all four sides free, 10 m elements, a Ricker at the centre, a 500 m ring of receivers and a line
10 m below the top. The medium's Thomsen parameters and tilt come from the case's
``sweep_medium.json`` and are checked against the cij SPECFEM2D read.

An absolute comparison: SPECFEM2D's own source time function (``plot_source_time_function.txt``,
its sign and factor included), its time step, elements (``n_order = 4`` = its NGLLX = 5), source
and stations -- no fitted shift or scale. SPECFEM2D's z points up: its BXZ is -v_z here (and its
c15, c35 flip sign). Its point force at angle 0 points down for a
positive source time function, which our ``vz`` force takes as it is; its moment tensor
``+M : grad(phi)`` is our stress source ``-M``. PropSEM's stress source takes the moment rate and
loads its running sum times dt, so the rate given is the backward difference of SPECFEM2D's
samples: the moment loaded at step n is then SPECFEM2D's at its step n, as the force is. A load at
step n is the one at n dt and gives our velocity at (n + 1/2) dt; SPECFEM2D's Newmark velocity
at n dt is the mean of the two around it, which a linear interpolation of ours reproduces.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_aniso.py \\
        --runs <dir with the case directories> --out <dir> [--case vti_force ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from sweep.sem import ElasticTTISEM2DDeformed, ElasticVTISEM2DDeformed, PropSEM, thomsen_to_tti_cij

CASES = ("vti_force", "vti_explosion", "tti30_force", "tti30_explosion")


def par(text, key):
    return re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", text, re.M).group(1).replace("d", "e")


def setup(run):
    """The box, medium, source and stations of a SPECFEM2D run directory."""
    d = run / "DATA"
    p, s = (d / "Par_file").read_text(), (d / "SOURCE").read_text()
    mat = re.search(r"^1 2 (.*)$", p, re.M).group(1).replace("d", "e").split()
    itf = [line.split() for line in (d / "interfaces.dat").read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    stf = np.loadtxt(run / "OUTPUT_FILES" / "plot_source_time_function.txt")
    return dict(xmax=float(par(p, "xmax")), nx=int(par(p, "nx")),
                ztop=max(float(line[1]) for line in itf if len(line) == 2),
                dt=float(par(p, "DT")), nstep=int(par(p, "NSTEP")),
                cij=tuple(float(v) for v in mat[:7]),             # rho c11 c13 c15 c33 c35 c55
                src=(float(par(s, "xs")), float(par(s, "zs"))), stype=int(par(s, "source_type")),
                angle=float(par(s, "anglesource")), f0=float(par(s, "f0")),
                factor=float(par(s, "factor")),
                moment=tuple(float(par(s, k)) for k in ("Mxx", "Mzz", "Mxz")),
                stations=np.loadtxt(d / "STATIONS", usecols=(2, 3)), stf=stf,
                medium=json.loads((d / "sweep_medium.json").read_text()))


def ricker(g, t):
    """SPECFEM2D's time_function_type 1: -factor (1 - 2 a t^2) exp(-a t^2), a = (pi f0)^2."""
    a = (math.pi * g["f0"]) ** 2
    return -g["factor"] * (1.0 - 2.0 * a * t * t) * np.exp(-a * t * t)


def run_sem(g, device, dtype, n_order=4):
    m = g["medium"]
    theta = math.radians(m["theta_deg"])
    params = (m["vp"], m["vs"], m["rho"], m["epsilon"], m["delta"])
    ours = thomsen_to_tti_cij(*(torch.tensor(v, dtype=torch.float64) for v in
                                (m["vp"], m["vs"], m["epsilon"], m["delta"], m["rho"], theta)))
    sf = np.array(g["cij"][1:]) * np.array([1, 1, -1, 1, -1, 1])     # z up -> down
    assert np.abs(np.array([float(c) for c in ours]) - sf).max() <= 1e-12 * sf[0], (ours, sf)
    eq = ElasticVTISEM2DDeformed if theta == 0.0 else ElasticTTISEM2DDeformed
    dh = g["xmax"] / g["nx"] / n_order
    nz, nx = int(round(g["ztop"] / dh)) + 1, int(round(g["xmax"] / dh)) + 1
    models = [torch.full((nz, nx), v, dtype=dtype)
              for v in params + ((theta,) if theta != 0.0 else ())]
    t = g["stf"][:, 0]
    assert len(t) == g["nstep"]
    w = g["stf"][:, 1]                                    # the loads SPECFEM2D applied
    assert np.abs(w - ricker(g, t)).max() <= 1e-6 * np.abs(w).max(), "not SPECFEM2D's Ricker"
    if g["stype"] == 1:
        th = math.radians(g["angle"])                # SPECFEM2D's (sin, -cos) z up = (sin, cos) down
        calls = [(n, c * w) for n, c in (("vx", math.sin(th)), ("vz", math.cos(th))) if c != 0.0]
    else:
        mxx, mzz, mxz = g["moment"]
        assert mxx == mzz and mxz == 0.0, "an explosion only"
        calls = [(("sxx", "szz"), -mxx * np.diff(w, prepend=0.0) / g["dt"])]
    src = np.array([[g["src"][0] / dh, (g["ztop"] - g["src"][1]) / dh]])
    rec = np.stack([g["stations"][:, 0] / dh, (g["ztop"] - g["stations"][:, 1]) / dh], 1)[None]
    kw = dict(shape=(nz, nx), dh=dh, dt=g["dt"], n_order=n_order, abcn=0, f_dominant=g["f0"],
              dev=device, impl="eager", dtype=dtype, receiver_type=["vx", "vz"])
    out = 0.0
    for names, wav in calls:
        prop = PropSEM(eq, source_type=list(names) if isinstance(names, tuple) else [names],
                       **kw)
        with torch.no_grad():
            out = out + prop(torch.from_numpy(wav).to(dtype), src, rec,
                             models=models)[0].double().cpu().numpy()
    out[..., 1] *= -1.0                                   # v_z down -> SPECFEM's up
    return t + 0.5 * g["dt"], out


def load_specfem(run, n):
    tr = [[np.loadtxt(run / "OUTPUT_FILES" / f"AA.S{i + 1:04d}.BX{c}.semv") for i in range(n)]
          for c in "XZ"]
    return tr[0][0][:, 0], np.stack([np.stack([a[:, 1] for a in tr[c]], -1) for c in range(2)], -1)


def direct_window(g):
    """The end of each station's echo-free window: the first free-edge echo (image sources,
    the fastest qP speed) less a period."""
    rho, c11, _, _, c33, _, _ = g["cij"]
    v = math.sqrt(max(c11, c33) / rho)
    xs, zs = g["src"]
    images = [(-xs, zs), (2 * g["xmax"] - xs, zs), (xs, -zs), (xs, 2 * g["ztop"] - zs)]
    st = g["stations"]
    dist = np.min([np.hypot(st[:, 0] - x, st[:, 1] - z) for x, z in images], 0)
    return dist / v - 1.0 / g["f0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", nargs="*", default=list(CASES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case in args.case:
        run = args.runs / case
        g = setup(run)
        n = len(g["stations"])
        t_ref, ref = load_specfem(run, n)
        t_sem, sem = run_sem(g, args.device, dtype)
        ours = np.stack([np.stack([np.interp(t_ref, t_sem, sem[:, i, c], left=0.0)
                                   for i in range(n)], -1) for c in range(2)], -1)
        np.savez(args.out / f"sem_{case}_{args.dtype}.npz", t_sem=t_sem, sem=sem, t_ref=t_ref,
                 ref=ref)
        ring = np.arange(n) < 24
        t_end = direct_window(g)
        rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
        print(f"{case} ({args.dtype}): shift-free, scale-free", flush=True)
        for lab, sel, win in (("ring, echo-free", ring, True), ("ring, full 1.2 s", ring, False),
                              ("surface line, full 1.2 s", ~ring, False)):
            for c, cn in enumerate(("BXX", "BXZ")):
                errs = []
                for i in np.flatnonzero(sel):
                    m = (t_ref <= t_end[i]) if win else np.ones_like(t_ref, bool)
                    if np.linalg.norm(ref[m, i, c]) > 1e-3 * np.abs(ref[..., c]).max():
                        errs.append(rel(ours[m, i, c], ref[m, i, c]))
                print(f"  {lab} {cn}: rel L2 median {np.median(errs):.2e} max {np.max(errs):.2e} "
                      f"({len(errs)} traces)", flush=True)
        print(f"  all traces, both components: rel L2 {rel(ours, ref):.2e}, "
              f"amp {np.linalg.norm(ours) / np.linalg.norm(ref):.5f}", flush=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
        for c, cn in enumerate(("BXX", "BXZ")):
            for r, (sel, lab) in enumerate(((ring, "ring, 15 deg apart"), (~ring, "surface line"))):
                ax = axes[r, c]
                idx = np.flatnonzero(sel)
                s = 0.5 / max(np.abs(ref[:, idx, c]).max(), 1e-30)
                for k, i in enumerate(idx):
                    ax.plot(t_ref, k + s * ref[:, i, c], "k", lw=1.0)
                    ax.plot(t_ref, k + s * ours[:, i, c], "r--", lw=0.7)
                    ax.plot(t_ref, k + 10 * s * (ours[:, i, c] - ref[:, i, c]), color="C0", lw=0.6)
                ax.set_title(f"({'abcd'[2 * r + c]}) {cn}, {lab}: SPECFEM2D (black), SEM (red), "
                             f"10x diff (blue)", fontsize=9)
                ax.set_xlabel("time (s)")
                ax.set_ylabel("station")
        fig.savefig(args.out / f"sem_vs_specfem2d_{case}.png", dpi=220)
        plt.close(fig)
        print("wrote", args.out / f"sem_vs_specfem2d_{case}.png", flush=True)


if __name__ == "__main__":
    main()

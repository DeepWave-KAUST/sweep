"""Plane waves against SPECFEM2D's ``initialfield``, on the cases ``make_specfem_planewave_cases.py``
builds: SPECFEM2D's run starts with the plane wave in the mesh and feeds it through its Stacey
edges (Bielak's conditions); ours (ElasticSEM2DDeformed, PropSEM's ``PlaneWave``) starts at rest
earlier, the wave entering through the band around the box, and its C-PML takes what the box
scatters. The box is SPECFEM2D's domain in the small cases and a 10 x 5 piece of the large ones
(around the basin); the band is the ring of elements just outside it.

Times: SPECFEM2D's state after step ``it`` is the plane wave's time ``it dt`` (its record of
that step is labelled ``(it - 1) dt - t0``); ours is ``(n + 1) dt - lead`` after step ``n``,
``lead`` the whole steps it starts earlier. SPECFEM2D's incident displacement at its reference
point is the Ricker ``R(t - t0)`` below the critical angle, ``t0 = 1.2 / f0`` or ``USER_T0``;
past it and for the Rayleigh wave (``paco_beyond_critical``) it is ``-0.5 R(t - 1.2 / f0)``:
on SH it is ``cos(a) R`` (the x component of its SV polarization). The factor is fitted
here, and reported, rather than assumed. In the flat cases both runs also
meet the analytic plane wave; with the basin, the scattered field is the difference to
``pw_sv20_big``.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_planewave.py \\
        --runs <dir with the case directories> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from sweep.propagator import EagerOptions, Full
from sweep.sem import ElasticSEM2DDeformed, ElasticSHSEM2DDeformed, PlaneWave, PropSEM
from sweep.sem.plane_wave import rayleigh_speed

NO, PML = 4, 10                     # GLL order, C-PML elements


def ricker(t, f0):
    a = (math.pi * f0) ** 2
    return (1.0 - 2.0 * a * t * t) * np.exp(-a * t * t)


def load(run):
    """The geometry and SPECFEM2D's displacement records ``(nt, nrec, 2)``, z up (SH:
    ``(nt, nrec, 1)``)."""
    d = run / "DATA"
    g = json.loads((d / "sweep_planewave.json").read_text())
    g["stations"] = np.loadtxt(d / "STATIONS", usecols=(2, 3))
    n = len(g["stations"])
    comps = ("BXY",) if g["wave"] == "SH" else ("BXX", "BXZ")
    tr = [np.stack([np.loadtxt(run / "OUTPUT_FILES" / f"AA.S{i + 1:04d}.{c}.semd")[:, 1]
                    for i in range(n)], -1) for c in comps]
    g["specfem"] = np.stack(tr, -1)
    g["t_specfem"] = np.loadtxt(run / "OUTPUT_FILES" / f"AA.S0001.{comps[0]}.semd")[:, 0]
    return g


def frequency_domain(g):
    """SPECFEM2D takes this case's field from paco_beyond_critical (in the frequency domain)."""
    return g["wave"] == "Rayleigh" or (
        g["wave"] == "SV" and math.sin(math.radians(g["angle"])) * g["vp"] / g["vs"] > 1.0)


def geometry(g):
    """Our box ``(x0, x1, depth)`` in SPECFEM2D's x, the model's shape around it (the band
    above it too in a whole space), and its top in SPECFEM2D's z."""
    x0, x1, depth = g["sweep_box"]
    h = g["h"]
    dh = h / NO
    above = h if g["whole_space"] else 0.0
    shape = (int(round((depth + h + above) / dh)) + 1, int(round((x1 - x0 + 2.0 * h) / dh)) + 1)
    return x0, x1, depth, dh, shape, g["box"][1] + above


def lead_steps(g, wave, x0, depth, t_c):
    """Whole steps our run starts ahead of SPECFEM2D's: the band's first node (its outer
    corner the wave reaches first) quiet at our start, a Ricker 1.5 periods out (4.5 for the
    frequency-domain fields, whose Hilbert-transform tails run ahead)."""
    h, (xs, _) = g["h"], g["reference"]
    if wave.wave == "Rayleigh":
        ahead = (xs - (x0 - h)) / rayleigh_speed(g["vp"], g["vs"])
    else:
        nx_, nz_ = wave.direction
        c = g["vp"] if wave.wave == "P" else g["vs"]
        ahead = (nx_ * (xs - (x0 - h)) + nz_ * (0.0 - (depth + h))) / c
    margin = (4.5 if frequency_domain(g) else 1.5) / g["f0"]
    return max(0, math.ceil((ahead + margin - t_c) / g["dt"]))


def run_sem(g, device, dtype=torch.float64):
    """Our records on SPECFEM2D's times ``(nstep, nrec, k)`` (z up), and the analytic plane
    wave there in the flat cases."""
    x0, x1, depth, dh, shape, top = geometry(g)
    h, dt, f0 = g["h"], g["dt"], g["f0"]
    sh = g["wave"] == "SH"
    wave = PlaneWave(g["wave"], 90.0 if g["wave"] == "Rayleigh" else g["angle"],
                     origin=((g["reference"][0] - (x0 - h)) / dh,
                             (top - g["reference"][1]) / dh))
    t_c = 1.2 / f0 if frequency_domain(g) or not g["user_t0"] else g["user_t0"]
    m = lead_steps(g, wave, x0, depth, t_c)
    nt = m + g["nstep"]
    w = torch.from_numpy(ricker(np.arange(nt) * dt - m * dt - t_c, f0)).to(dtype)
    prop = PropSEM(ElasticSHSEM2DDeformed if sh else ElasticSEM2DDeformed, shape=shape, dh=dh,
                   dt=dt, n_order=NO, abcn=PML * NO, free_surface=not g["whole_space"],
                   f_dominant=f0, dev=device, dtype=dtype, impl="eager", memory=Full(),
                   eager_options=EagerOptions(use_compile=False),
                   receiver_type=["u_y"] if sh else ["u_x", "u_z"])
    models = [torch.full(prop._gll_shape, float(g[k]), dtype=dtype)
              for k in (("vs", "rho") if sh else ("vp", "vs", "rho"))]
    if g["basin"] is not None:
        b = g["basin"]
        ez = torch.arange(prop._gll_shape[0]) // prop._elems_pad[-1]
        ex = torch.arange(prop._gll_shape[0]) % prop._elems_pad[-1]
        lo = PML + int(round((b["x"][0] - (x0 - h)) / h))
        hi = PML + int(round((b["x"][1] - (x0 - h)) / h))
        inside = (ez < int(round(b["depth"] / h))) & (ex >= lo) & (ex < hi)
        for mdl, k in zip(models, ("vp", "vs", "rho")):
            mdl[inside] = float(b[k])
    rec = np.stack([(g["stations"][:, 0] - (x0 - h)) / dh, (top - g["stations"][:, 1]) / dh], 1)
    with torch.no_grad():
        out = prop(w, wave, torch.from_numpy(rec)[None], models=models)[0]
        free = None if g["basin"] is not None else prop.free_field(
            w, wave, torch.from_numpy(rec), models)
    keep = slice(m, m + g["nstep"])
    ours = out[keep].double().cpu().numpy()
    if not sh:
        ours[..., 1] *= -1.0                              # z down -> SPECFEM2D's up
    if free is not None:
        free = free.transpose(0, 1)[keep].double().cpu().numpy()
        if not sh:
            free[..., 1] *= -1.0
    return ours, free, m


def misfit(test, ref, groups):
    """Relative L2 over the record and both components, overall and per receiver group, and
    the peak ratio (largest difference over the largest reference)."""
    out, lo = {"all": float(np.linalg.norm(test - ref) / np.linalg.norm(ref)),
               "peak": float(np.abs(test - ref).max() / np.abs(ref).max())}, 0
    for lab, n in groups:
        sl = slice(lo, lo + n)
        out[lab] = float(np.linalg.norm(test[:, sl] - ref[:, sl]) / np.linalg.norm(ref[:, sl]))
        lo += n
    return out


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.4 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        r = results[name]
        t, spec, ours = r["t"], r["specfem"], r["ours"]
        picks = [0, 6, 12]                                  # surface receivers
        off = 2.2 * np.abs(spec[:, picks]).max()
        cols = ("k", "tab:blue")[:spec.shape[-1]]
        for k, i in enumerate(picks):
            for c, col in enumerate(cols):
                row[0].plot(t, spec[:, i, c] + k * off, col, lw=1.0)
                row[0].plot(t, ours[:, i, c] + k * off, "--", color="tab:red", lw=0.9)
                row[1].plot(t, 100.0 * (ours[:, i, c] - spec[:, i, c]) + k * off, col, lw=0.8)
        comps = "uy" if spec.shape[-1] == 1 else "ux black, uz blue"
        row[0].set_title(f"{name}: SPECFEM2D ({comps}), sweep dashed", fontsize=9)
        row[1].set_title(f"{name}: (sweep - SPECFEM2D) x 100, rel. L2 "
                         f"{r['misfit']['all']:.1e}", fontsize=9)
        for a in row:
            a.set_yticks([])
            a.set_xlim(t[0], t[-1])
    for a in axes[-1]:
        a.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("figure", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--plot", action="append", default=None,
                    help="the cases the figure shows (default: every case run)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = args.case or sorted(p.name for p in args.runs.iterdir() if p.name.startswith("pw_"))
    results, summary = {}, {}
    for name in cases:
        g = load(args.runs / name)
        ours, free, m = run_sem(g, args.device)
        spec = g["specfem"][: g["nstep"]]
        # SPECFEM2D's amplitude convention: 1 below the critical angle, -0.5 past it, and
        # cos(angle) for SH (its SV polarization's x component)
        scale = float((spec * ours).sum() / (ours * ours).sum())
        expect = (-0.5 if frequency_domain(g) else
                  math.cos(math.radians(g["angle"])) if g["wave"] == "SH" else 1.0)
        ours_s, free_s = expect * ours, None if free is None else expect * free
        res = {"fitted_scale": scale, "expected_scale": expect, "lead_steps": m,
               "sweep_vs_specfem": misfit(ours_s, spec, g["groups"])}
        if free is not None:
            res["sweep_vs_analytic"] = misfit(ours_s, free_s, g["groups"])
            res["specfem_vs_analytic"] = misfit(spec, free_s, g["groups"])
        summary[name] = res
        print(name, json.dumps(res, indent=1))
        t = g["t_specfem"][: g["nstep"]] + (g["user_t0"] or 1.2 / g["f0"]) + g["dt"]
        results[name] = dict(t=t, specfem=spec, ours=ours_s, misfit=res["sweep_vs_specfem"])
        np.savez(args.out / f"{name}.npz", t=t, specfem=spec, sweep=ours_s,
                 analytic=np.zeros(0) if free_s is None else free_s)
    (args.out / "planewave_metrics.json").write_text(json.dumps(summary, indent=1) + "\n")
    shown = args.plot or list(results)
    plot({k: results[k] for k in shown if k in results}, args.out / "planewave_vs_specfem.png")


if __name__ == "__main__":
    main()

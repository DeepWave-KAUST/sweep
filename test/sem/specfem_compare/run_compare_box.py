"""The 2-D SEM classes against SPECFEM2D on the boxes ``make_specfem_box_cases.py`` builds: one
homogeneous medium (or water over rock), all four sides free, 10 m elements, a Ricker inside an
element and two receiver groups. The medium (and the hill on the top edge) comes from the case's
``sweep_medium.json``; a TI medium's cij are checked against those SPECFEM2D read. The classes
run per medium (``--equation`` picks among the isotropic ones):

* an isotropic rock: ElasticVSSEM2DDeformed, ElasticSEM2DDeformed and, flat, ElasticSEM2D;
* a TI shale: ElasticVTISEM2DDeformed, or ElasticTTISEM2DDeformed for a tilted axis;
* an attenuating rock: ViscoElasticVSSEM2DDeformed with SPECFEM2D's conventions: its
  mechanisms (``n_sls`` over ``f_ref / 10 .. 10 f_ref``), its fit (``n_fit = 2 n_sls - 1``)
  and its reading of the velocities at ``f_ref`` (``velocities="real_modulus"``);
* water: AcousticPotentialSEM2D and AcousticSEM2D, pressure-free on every edge as SPECFEM2D's
  fluid is wherever there is no absorbing condition;
* water over rock: AcousticElasticSEM2D, the water pressure-free on its top and sides;
* layers across curved interfaces: the mesh of ``PropSEM(interfaces=...)``, each layer's
  material per element (``layered_model``): AcousticElasticSEM2D under water, else
  ElasticVSSEM2DDeformed and ElasticSEM2DDeformed.

An absolute comparison: SPECFEM2D's own source time function (``plot_source_time_function.txt``,
its sign and factor included), its time step, elements (``n_order = 4`` = its NGLLX = 5), source
and stations -- no fitted shift or scale. SPECFEM2D's z points up: its BXZ is -v_z here (and its
c15, c35 flip sign). Its point force at angle 0 points down for a positive source time function,
which our ``vz`` force takes as it is; its moment tensor ``+M : grad(phi)`` is our stress source
``-M``. PropSEM's stress source takes the moment rate and loads its running sum times dt, so the
rate given is the backward difference of SPECFEM2D's samples: the moment loaded at step n is then
SPECFEM2D's at its step n, as the force is. Its pressure source adds ``h stf / kappa`` to the
potential's load, our ``chi`` source ``h w``: ``w = stf / kappa``. AcousticSEM2D's field,
``u'' = c^2 M^-1 (-K u) + M^-1 h w``, is SPECFEM2D's potential with ``w = stf`` (its pressure
``-u''`` is taken from the trace as the potential form's is): driving ``u`` with the second
difference of the source instead would start it with a step, since SPECFEM2D's Ricker is not
zero at its first sample. A load at step n is the one at n dt. It gives the velocity-stress classes'
velocity at (n + 1/2) dt, and SPECFEM2D's Newmark velocity at n dt is the mean of the two
around it, which a linear interpolation of ours reproduces; the displacement classes' velocity
and the potential's pressure are the central first and second differences, at (n + 1) dt, as
SPECFEM2D's are. The last sample (a one-sided difference there) is left out.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_box.py \\
        --runs <dir with the case directories> --out <dir> [--case iso_force ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from sweep.sem import (AcousticElasticSEM2D, AcousticPotentialSEM2D, AcousticSEM2D,
                       ElasticSEM2D, ElasticSEM2DDeformed, ElasticTTISEM2DDeformed,
                       ElasticVSSEM2DDeformed, ElasticVTISEM2DDeformed, PropSEM,
                       ViscoElasticVSSEM2DDeformed, thomsen_to_tti_cij)

CASES = ("iso_force", "iso_explosion", "hill_explosion", "vti_force", "vti_explosion",
         "tti30_force", "tti30_explosion", "visco_explosion", "water_pressure", "sea_pressure",
         "sea_curved_pressure", "layers_curved_explosion")
ISOTROPIC = {"vs": ElasticVSSEM2DDeformed, "displacement": ElasticSEM2DDeformed,
             "quad4": ElasticSEM2D}
HALF_STEP = (ElasticVSSEM2DDeformed, ElasticVTISEM2DDeformed, ElasticTTISEM2DDeformed,
             ViscoElasticVSSEM2DDeformed)
ALL_SIDES = {"top": True, "bottom": True, "left": True, "right": True}


def par(text, key):
    return re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", text, re.M).group(1).replace("d", "e")


def setup(run):
    """The box, media, source and stations of a SPECFEM2D run directory."""
    d = run / "DATA"
    p, s = (d / "Par_file").read_text(), (d / "SOURCE").read_text()
    mats = {int(i): (int(k), [float(v) for v in rest.replace("d", "e").split()[:7]])
            for i, k, rest in re.findall(r"^(\d+) ([12]) (.*)$", p, re.M)}
    itf = [line.split() for line in (d / "interfaces.dat").read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    medium = json.loads((d / "sweep_medium.json").read_text())
    if "groups" not in medium:
        medium["groups"] = [["ring", medium["n_ring"]], ["line 10 m below the top", 0]]
    return dict(xmax=float(par(p, "xmax")), nx=int(par(p, "nx")),
                ztop=max(float(line[1]) for line in itf if len(line) == 2),
                dt=float(par(p, "DT")), nstep=int(par(p, "NSTEP")), materials=mats,
                src=(float(par(s, "xs")), float(par(s, "zs"))), stype=int(par(s, "source_type")),
                angle=float(par(s, "anglesource")), f0=float(par(s, "f0")),
                factor=float(par(s, "factor")),
                moment=tuple(float(par(s, k)) for k in ("Mxx", "Mzz", "Mxz")),
                stations=np.loadtxt(d / "STATIONS", usecols=(2, 3)),
                stf=np.loadtxt(run / "OUTPUT_FILES" / "plot_source_time_function.txt"),
                medium=medium)


def ricker(g, t):
    """SPECFEM2D's time_function_type 1: -factor (1 - 2 a t^2) exp(-a t^2), a = (pi f0)^2."""
    a = (math.pi * g["f0"]) ** 2
    return -g["factor"] * (1.0 - 2.0 * a * t * t) * np.exp(-a * t * t)


def surface(g, x):
    """The top edge, z up, as make_specfem_box_cases.py wrote it."""
    h = g["medium"]["hill"]
    if h is None:
        return np.full_like(x, g["ztop"])
    return g["ztop"] - h["height"] + h["height"] * np.exp(-((x - h["x0"]) / h["sigma"]) ** 2)


def interface(g, x, f):
    """An internal interface, z up, as make_specfem_box_cases.py wrote it."""
    if f["shape"] == "gauss":
        return f["z0"] + f["height"] * np.exp(-((x - f["x0"]) / f["sigma"]) ** 2)
    return f["z0"] + f["height"] * np.sin(2.0 * np.pi * (x - f["x0"]) / f["sigma"])


def wet(m):
    """A layered medium whose top layer is water."""
    return m["kind"] == "layered" and m["layers"][0].get("vs", 0.0) == 0.0


def equations(g, picks):
    m = g["medium"]
    if wet(m):
        return [AcousticElasticSEM2D]
    if m["kind"] == "layered":
        return [ISOTROPIC[k] for k in (picks or ["vs", "displacement"])]
    if m["kind"] == "ti":
        return [ElasticVTISEM2DDeformed if m["theta_deg"] == 0.0 else ElasticTTISEM2DDeformed]
    if m["kind"] == "acoustic":
        return [AcousticPotentialSEM2D, AcousticSEM2D]
    if m["kind"] == "coupled":
        return [AcousticElasticSEM2D]
    if m["kind"] == "visco":
        return [ViscoElasticVSSEM2DDeformed]
    keys = picks or [k for k in ISOTROPIC if not (k == "quad4" and m["hill"])]
    return [ISOTROPIC[k] for k in keys]


def check_materials(g):
    """SPECFEM2D's material lines against sweep_medium.json; the TI models' values."""
    m, mats = g["medium"], g["materials"]
    iso = lambda med: [med["rho"], med["vp"], med.get("vs", 0.0)]
    if m["kind"] in ("isotropic", "acoustic"):
        assert mats == {1: (1, iso(m) + [0.0, 0.0, 9999.0, 9999.0])}, mats
    elif m["kind"] == "coupled":
        no_q = [0.0, 0.0, 9999.0, 9999.0]
        assert mats == {1: (1, iso(m["rock"]) + no_q), 2: (1, iso(m["water"]) + no_q)}
    elif m["kind"] == "visco":
        assert mats == {1: (1, iso(m) + [0.0, 0.0, m["q"], m["q"]])}, mats
        return (m["vp"], m["vs"], m["rho"], m["q"], m["q"])
    elif m["kind"] == "layered":                 # SPECFEM2D's materials count up from the bottom
        no_q = [0.0, 0.0, 9999.0, 9999.0]
        assert mats == {i + 1: (1, iso(lay) + no_q) for i, lay in enumerate(m["layers"][::-1])}
    else:
        theta = math.radians(m["theta_deg"])
        ours = thomsen_to_tti_cij(*(torch.tensor(v, dtype=torch.float64) for v in
                                    (m["vp"], m["vs"], m["epsilon"], m["delta"], m["rho"], theta)))
        kind, mat = mats[1]
        sf = np.array(mat[1:7]) * np.array([1, 1, -1, 1, -1, 1])           # z up -> down
        assert kind == 2 and np.abs(np.array([float(c) for c in ours]) - sf).max() <= 1e-12 * sf[0]
        return (m["vp"], m["vs"], m["rho"], m["epsilon"], m["delta"]) + ((theta,) if theta else ())
    return None


def run_sem(g, eq, device, dtype, n_order=4):
    """``(t, {component: (nt, nrec) traces})`` of class ``eq`` on the case ``g``."""
    m = g["medium"]
    ti = check_materials(g)
    dh = g["xmax"] / g["nx"] / n_order
    nz, nx = int(round(g["ztop"] / dh)) + 1, int(round(g["xmax"] / dh)) + 1
    full = lambda v: torch.full((nz, nx), float(v), dtype=dtype)
    t = g["stf"][:, 0]
    assert len(t) == g["nstep"]
    w = g["stf"][:, 1]                                    # the loads SPECFEM2D applied
    assert np.abs(w - ricker(g, t)).max() <= 1e-6 * np.abs(w).max(), "not SPECFEM2D's Ricker"
    src = np.array([[g["src"][0] / dh, (g["ztop"] - g["src"][1]) / dh]])
    rec = np.stack([g["stations"][:, 0] / dh, (g["ztop"] - g["stations"][:, 1]) / dh], 1)[None]
    kw = dict(shape=(nz, nx), dh=dh, dt=g["dt"], n_order=n_order, abcn=0, f_dominant=g["f0"],
              dev=device, impl="eager", dtype=dtype)
    if m["hill"]:
        kw["topography"] = (g["ztop"] - surface(g, np.arange(nx) * dh)) / dh
    layered = None
    if m["kind"] == "layered":
        kw["interfaces"] = [(g["ztop"] - interface(g, np.arange(nx) * dh, f)) / dh
                            for f in m["interfaces"]]
        kw["interface_elements"] = m["rows"]

        def layered(prop):
            return [prop.layered_model([lay.get(k, 0.0) for lay in m["layers"]])
                    for k in ("vp", "vs", "rho")]
    if m["kind"] == "acoustic":
        kappa = m["rho"] * m["vp"] ** 2
        kw.update(free_surface=ALL_SIDES, receiver_type=["p"])
        models = [full(m["vp"])]
        if eq is AcousticPotentialSEM2D:
            kw["density"] = m["rho"]
            calls = [(None, w / kappa)]
        else:
            calls = [(None, w)]
        names = ["PRE"]
    elif m["kind"] == "coupled" or wet(m):
        kw.update(free_surface={"top": True, "left": True, "right": True, "bottom": False},
                  receiver_type=["p", "vx", "vz"])
        water = m["water"] if m["kind"] == "coupled" else m["layers"][0]
        calls = [(["chi"], w / (water["rho"] * water["vp"] ** 2))]
        names = ["PRE", "BXX", "BXZ"]
        if m["kind"] == "coupled":
            rock = m["rock"]

            def layered(prop):
                mesh = prop.mesh
                wet_e = mesh.z_e.mean((1, 2)) < m["depth"]
                E, n = mesh.n_elements, mesh.ngll
                return [torch.where(wet_e, torch.tensor(water.get(k, 0.0), dtype=dtype),
                                    torch.tensor(rock[k], dtype=dtype)).to(mesh.z_e.device)
                        .view(E, 1, 1).expand(E, n, n).contiguous() for k in ("vp", "vs", "rho")]
        models = None
    else:
        kw["receiver_type"] = ["vx", "vz"]
        models = (None if m["kind"] == "layered"
                  else [full(v) for v in (ti or (m["vp"], m["vs"], m["rho"]))])
        if m["kind"] == "visco":
            kw["equation_options"] = {"f_ref": m["f_ref"], "n_sls": m["n_sls"],
                                      "f_band": (m["f_ref"] / 10.0, 10.0 * m["f_ref"]),
                                      "n_fit": 2 * m["n_sls"] - 1,
                                      "velocities": "real_modulus"}
        if g["stype"] == 1:
            th = math.radians(g["angle"])            # SPECFEM2D's (sin, -cos) z up = (sin, cos) down
            calls = [([n], c * w) for n, c in (("vx", math.sin(th)), ("vz", math.cos(th)))
                     if c != 0.0]
        else:
            mxx, mzz, mxz = g["moment"]
            assert mxx == mzz and mxz == 0.0, "an explosion only"
            calls = [(["sxx", "szz"], -mxx * np.diff(w, prepend=0.0) / g["dt"])]
        names = ["BXX", "BXZ"]
    out = 0.0
    for source_type, wav in calls:
        prop = PropSEM(eq, source_type=source_type, **kw)
        if layered is not None:
            models = layered(prop)
        with torch.no_grad():
            out = out + prop(torch.from_numpy(wav).to(dtype), src, rec,
                             models=models)[0].double().cpu().numpy()
    traces = {nm: out[..., c] for c, nm in enumerate(names)}
    if eq is AcousticSEM2D:          # the potential's pressure, as PropSEM derives it from chi
        u = traces["PRE"]
        pad = np.concatenate([np.zeros_like(u[:1]), u, u[-1:]])
        traces["PRE"] = -(pad[2:] - 2.0 * u + pad[:-2]) / g["dt"] ** 2
    if "BXZ" in traces:
        traces["BXZ"] = -traces["BXZ"]                     # v_z down -> SPECFEM's up
    return t + (0.5 if eq in HALF_STEP else 1.0) * g["dt"], traces


def load_specfem(run, n, names):
    """``(t, {component: (nt, n)})``: BXX, BXZ (.semv) and PRE (.semp)."""
    out = {}
    for nm in names:
        ext = "semp" if nm == "PRE" else "semv"
        tr = [np.loadtxt(run / "OUTPUT_FILES" / f"AA.S{i + 1:04d}.{nm}.{ext}") for i in range(n)]
        out[nm] = np.stack([a[:, 1] for a in tr], -1)
    return tr[0][:, 0], out


def direct_window(g):
    """The end of each station's echo-free window: the first free-edge echo (image sources in
    the flat edges, the fastest P speed) less a period."""
    m = g["medium"]
    if m["kind"] == "ti":
        _, mat = g["materials"][1]
        v = math.sqrt(max(mat[1], mat[4]) / mat[0])
    elif m["kind"] == "layered":
        v = max(lay["vp"] for lay in m["layers"])
    else:
        v = (m["rock"] if m["kind"] == "coupled" else m)["vp"]
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
    ap.add_argument("--equation", nargs="*", choices=list(ISOTROPIC),
                    help="the classes run on an isotropic rock (default: all that fit)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    for case in args.case:
        run = args.runs / case
        g = setup(run)
        n = len(g["stations"])
        (lab0, n0), (lab1, _) = g["medium"]["groups"]
        first = np.arange(n) < n0
        t_end = direct_window(g)
        for eq in equations(g, args.equation):
            t_sem, sem = run_sem(g, eq, args.device, dtype)
            t_ref, ref = load_specfem(run, n, list(sem))
            keep = t_ref <= t_sem[-2]
            ours = {nm: np.stack([np.interp(t_ref, t_sem[:-1], tr[:-1, i], left=0.0)
                                  for i in range(n)], -1) for nm, tr in sem.items()}
            np.savez(args.out / f"sem_{case}_{eq.__name__}_{args.dtype}.npz", t_sem=t_sem,
                     t_ref=t_ref, **{f"sem_{k}": v for k, v in sem.items()},
                     **{f"ref_{k}": v for k, v in ref.items()})
            # the components each group records: hydrophones the pressure, geophones the velocity
            comps = {True: list(sem), False: list(sem)}
            if g["medium"]["kind"] == "coupled" or wet(g["medium"]):
                comps = {True: ["PRE"], False: ["BXX", "BXZ"]}
            print(f"{case}, {eq.__name__} ({args.dtype}): shift-free, scale-free", flush=True)
            allo, allr = [], []
            for lab, grp, win in ((f"{lab0}, echo-free", True, True),
                                  (f"{lab0}, full record", True, False),
                                  (f"{lab1}, full record", False, False)):
                sel = first if grp else ~first
                for nm in comps[grp]:
                    errs = []
                    for i in np.flatnonzero(sel):
                        msk = keep & ((t_ref <= t_end[i]) if win else True)
                        if np.linalg.norm(ref[nm][msk, i]) > 1e-3 * np.abs(ref[nm][keep][:, sel]).max():
                            errs.append(rel(ours[nm][msk, i], ref[nm][msk, i]))
                        if not win:
                            allo.append(ours[nm][keep, i] / np.abs(ref[nm][keep][:, sel]).max())
                            allr.append(ref[nm][keep, i] / np.abs(ref[nm][keep][:, sel]).max())
                    if errs:
                        print(f"  {lab} {nm}: rel L2 median {np.median(errs):.2e} max "
                              f"{np.max(errs):.2e} ({len(errs)} traces)", flush=True)
            allo, allr = np.concatenate(allo), np.concatenate(allr)
            print(f"  all traces (each component by its peak): rel L2 {rel(allo, allr):.2e}, "
                  f"amp {np.linalg.norm(allo) / np.linalg.norm(allr):.5f}", flush=True)

            panels = [(grp, nm) for grp in (True, False) for nm in comps[grp]]
            ncol = 2
            fig, axes = plt.subplots((len(panels) + 1) // ncol, ncol, squeeze=False,
                                     figsize=(12, 5.5 * ((len(panels) + 1) // ncol)),
                                     constrained_layout=True)
            for k, (grp, nm) in enumerate(panels):
                ax = axes.flat[k]
                idx = np.flatnonzero(first if grp else ~first)
                s = 0.5 / max(np.abs(ref[nm][:, idx]).max(), 1e-30)
                for j, i in enumerate(idx):
                    ax.plot(t_ref, j + s * ref[nm][:, i], "k", lw=1.0)
                    ax.plot(t_ref, j + s * ours[nm][:, i], "r--", lw=0.7)
                    ax.plot(t_ref, j + 10 * s * (ours[nm][:, i] - ref[nm][:, i]), color="C0",
                            lw=0.6)
                ax.set_title(f"({'abcdefgh'[k]}) {nm}, {lab0 if grp else lab1}", fontsize=9)
                ax.set_xlabel("time (s)")
                ax.set_ylabel("station")
            for ax in axes.flat[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{case}, {eq.__name__}: SPECFEM2D black, SEM red, 10x diff blue",
                         fontsize=10)
            fig.savefig(args.out / f"sem_vs_specfem2d_{case}_{eq.__name__}.png", dpi=220)
            plt.close(fig)
            print("wrote", args.out / f"sem_vs_specfem2d_{case}_{eq.__name__}.png", flush=True)


if __name__ == "__main__":
    main()

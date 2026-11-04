"""How much the absorbing boundaries reflect, on the cases ``make_specfem_absorb_cases.py``
builds. SPECFEM2D's Stacey condition and its C-PML are measured against its free box far
larger (``absorb_ref``); sweep's C-PML -- PropSEM's layer of 10 elements outside the same
interior, its default parameters with ``f_dominant = f0`` -- against sweep's run of that
larger box. The reflection at a receiver is the relative L2 difference from the larger box over
the record, both velocity components; the peak ratio is the largest difference over the
largest velocity of the reference on that receiver group.

Two checks of the set-up: sweep's run of the larger box against SPECFEM2D's (the conventions
of ``run_compare_box.py``: SPECFEM2D's Newmark velocity at n dt is ours at (n + 1) dt on the
displacement class, the mean of the two half steps around it on the velocity-stress one), and
sweep's C-PML against SPECFEM2D's.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_absorb.py \\
        --runs <dir with the case directories> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sweep.sem import ElasticSEM2DDeformed, ElasticVSSEM2DDeformed, PropSEM

CLASSES = {"displacement": ElasticSEM2DDeformed, "velocity-stress": ElasticVSSEM2DDeformed}
HALF_STEP = (ElasticVSSEM2DDeformed,)


def load(run):
    """The geometry, SPECFEM2D's source time function and its records ``(nt, nrec, 2)``,
    ``(vx, vz)`` with z up, of a case directory."""
    d = run / "DATA"
    g = json.loads((d / "sweep_absorb.json").read_text())
    g["stations"] = np.loadtxt(d / "STATIONS", usecols=(2, 3))
    stf = np.loadtxt(run / "OUTPUT_FILES" / "plot_source_time_function.txt")
    g["t"], g["w"] = stf[:, 0], stf[:, 1]
    n = len(g["stations"])
    tr = [np.stack([np.loadtxt(run / "OUTPUT_FILES" / f"AA.S{i + 1:04d}.{c}.semv")[:, 1]
                    for i in range(n)], -1) for c in ("BXX", "BXZ")]
    g["t_specfem"] = np.loadtxt(run / "OUTPUT_FILES" / "AA.S0001.BXX.semv")[:, 0]
    g["specfem"] = np.stack(tr, -1)
    return g


def run_sem(g, eq, dt, device, dtype, n_order=4):
    """``(nt, nrec, 2)`` velocity records of class ``eq`` on case ``g``: the case's box, its
    layer (if any) as PropSEM's absorbing layer outside the model, z up as SPECFEM2D's."""
    L, Z = g["box"]
    lay = g["layer"] if g["kind"] != "free" else 0.0
    dh = 10.0 / n_order
    nz, nx = int(round((Z - lay) / dh)) + 1, int(round((L - 2.0 * lay) / dh)) + 1
    grid = lambda p: np.stack([(p[:, 0] - lay) / dh, (Z - p[:, 1]) / dh], 1)
    src, rec = grid(np.array([g["source"]])), grid(g["stations"])[None]
    prop = PropSEM(eq, shape=(nz, nx), dh=dh, dt=dt, n_order=n_order,
                   abcn=int(round(lay / dh)), free_surface=bool(lay), f_dominant=g["f0"],
                   source_type=["vz"], receiver_type=["vx", "vz"], dev=device, impl="eager",
                   dtype=dtype)
    models = [torch.full((nz, nx), float(g[k]), dtype=dtype) for k in ("vp", "vs", "rho")]
    with torch.no_grad():
        out = prop(torch.from_numpy(g["w"]).to(dtype), src, rec, models=models)
    out = out[0].double().cpu().numpy()
    out[..., 1] *= -1.0                                   # v_z down -> SPECFEM2D's up
    return out


def on_specfem_times(g, eq, ours, dt):
    """Our records at SPECFEM2D's sample times (the last, one-sided, sample left out)."""
    t_sem = g["t"] + (0.5 if eq in HALF_STEP else 1.0) * dt
    return np.stack([np.stack([np.interp(g["t_specfem"], t_sem[:-1], ours[:-1, i, c], left=0.0)
                               for i in range(ours.shape[1])], -1) for c in range(2)], -1)


def reflection(test, ref, groups):
    """Per receiver: relative L2 over the record and both components; per group: the median,
    the largest and the peak ratio (largest difference / largest reference velocity)."""
    per = np.linalg.norm(test - ref, axis=(0, 2)) / np.linalg.norm(ref, axis=(0, 2))
    out, lo = {"per_receiver": per.tolist(),
               "all": float(np.linalg.norm(test - ref) / np.linalg.norm(ref))}, 0
    for lab, n in groups:
        sl = slice(lo, lo + n)
        out[lab] = dict(median=float(np.median(per[sl])), max=float(per[sl].max()),
                        peak=float(np.abs(test[:, sl] - ref[:, sl]).max()
                                   / np.abs(ref[:, sl]).max()))
        lo += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    cases = {c: load(args.runs / c) for c in ("absorb_stacey", "absorb_pml", "absorb_ref")}
    ref = cases["absorb_ref"]
    dt = float(np.diff(ref["t"]).mean())
    assert all(np.array_equal(c["w"], ref["w"]) for c in cases.values())
    groups = ref["groups"]
    res = {"specfem_stacey": reflection(cases["absorb_stacey"]["specfem"], ref["specfem"], groups),
           "specfem_cpml": reflection(cases["absorb_pml"]["specfem"], ref["specfem"], groups)}
    traces = {"specfem_ref": ref["specfem"], "specfem_stacey": cases["absorb_stacey"]["specfem"],
              "specfem_cpml": cases["absorb_pml"]["specfem"]}
    for key, eq in CLASSES.items():
        mine = {c: run_sem(cases[c], eq, dt, args.device, dtype) for c in ("absorb_pml", "absorb_ref")}
        res[f"sweep_cpml_{key}"] = reflection(mine["absorb_pml"], mine["absorb_ref"], groups)
        aligned = {c: on_specfem_times(cases[c], eq, mine[c], dt) for c in mine}
        res[f"check_ref_{key}_vs_specfem"] = float(
            np.linalg.norm(aligned["absorb_ref"] - ref["specfem"]) / np.linalg.norm(ref["specfem"]))
        res[f"check_cpml_{key}_vs_specfem_cpml"] = float(
            np.linalg.norm(aligned["absorb_pml"] - cases["absorb_pml"]["specfem"])
            / np.linalg.norm(ref["specfem"]))
        traces[f"sweep_ref_{key}"], traces[f"sweep_cpml_{key}"] = mine["absorb_ref"], mine["absorb_pml"]
        print(key, json.dumps({k: v for k, v in res.items() if key in k and k.startswith("check")}),
              flush=True)
    np.savez(args.out / f"absorb_traces_{args.dtype}.npz", t=ref["t"], **traces)
    (args.out / f"absorb_reflection_{args.dtype}.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:32s} all {v['all']:.2e}  " + "  ".join(
                f"[{lab}] med {v[lab]['median']:.2e} max {v[lab]['max']:.2e} peak {v[lab]['peak']:.2e}"
                for lab, _ in groups))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n0 = groups[0][1]
    x = ref["stations"][:n0, 0] - ref["shift"] - ref["layer"]          # m from the interior's edge
    t = ref["t"]
    base = ref["specfem"][:, :n0, 1]
    clip = 0.02 * np.abs(base).max()
    panels = [("SPECFEM2D reference (clip 2%)", base, "specfem_ref"),
              ("SPECFEM2D Stacey - reference", cases["absorb_stacey"]["specfem"][:, :n0, 1] - base,
               "specfem_stacey"),
              ("SPECFEM2D C-PML - reference", cases["absorb_pml"]["specfem"][:, :n0, 1] - base,
               "specfem_cpml"),
              ("sweep C-PML - reference",
               traces["sweep_cpml_displacement"][:, :n0, 1] - traces["sweep_ref_displacement"][:, :n0, 1],
               "sweep_cpml_displacement")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=True, constrained_layout=True)
    for k, (title, a, key) in enumerate(panels):
        ax = axes[k]
        ax.imshow(a, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip,
                  extent=[x[0] - 50, x[-1] + 50, t[-1], t[0]], interpolation="nearest")
        sub = "" if key == "specfem_ref" else f"\nrel L2 {res[key][groups[0][0]]['median']:.1e} (median)"
        ax.set_title(f"({'abcd'[k]}) {title}{sub}", fontsize=9)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("time (s)")
    fig.suptitle("vz on the line 10 m below the free top, same colour scale", fontsize=10)
    fig.savefig(args.out / "absorb_line_vz.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    for key, lab in (("specfem_stacey", "SPECFEM2D Stacey"), ("specfem_cpml", "SPECFEM2D C-PML"),
                     ("sweep_cpml_displacement", "sweep C-PML, displacement"),
                     ("sweep_cpml_velocity-stress", "sweep C-PML, velocity-stress")):
        ax.semilogy(res[key]["per_receiver"], "o-", ms=3, lw=1, label=lab)
    lo = 0
    for lab, n in groups:
        ax.axvline(lo - 0.5, color="0.6", lw=0.8)
        ax.text(lo, 1.0, lab, fontsize=8, va="top", transform=ax.get_xaxis_transform())
        lo += n
    ax.set_xlabel("receiver")
    ax.set_ylabel("reflection (rel L2)")
    ax.legend(fontsize=8)
    fig.savefig(args.out / "absorb_per_receiver.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

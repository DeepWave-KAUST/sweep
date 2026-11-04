"""The visco-elastic C-PML, on the attenuating cases ``make_specfem_absorb_cases.py`` builds
(``absorb_visco_pml``, ``absorb_visco_ref``: Qkappa = 30, Qmu = 20, three mechanisms at 15 Hz,
the velocities unrelaxed). As ``run_compare_absorb.py`` does for the elastic layer: SPECFEM2D's
visco-elastic C-PML against its free box far larger, and ViscoElasticSEM2DDeformed's -- the
layer attenuating -- against our run of that box; and ``ElasticLayer``, the same class with
the strengths zero on the band elements, the layer elastic (the class as it was before).

Checks of the set-up: our larger box against SPECFEM2D's, and our C-PML runs against
SPECFEM2D's visco-elastic C-PML. SPECFEM2D runs the product of the stretching and the
relaxation as partial fractions, one recursion per pole; ours runs the relaxation on the
stretched strain -- the same operator to second order in ``dt``.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_absorb_visco.py \\
        --runs <dir with the case directories> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_compare_absorb import load, on_specfem_times, reflection
from sweep.sem import PropSEM, ViscoElasticSEM2DDeformed
from sweep.sem.mesh.quad9 import Quad9Mesh


class ElasticLayer(ViscoElasticSEM2DDeformed):
    """ViscoElasticSEM2DDeformed with the strengths zero on the C-PML band elements: the
    absorbing layer elastic."""

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, vs_e, rho_e, Qk_e, Qmu_e, *, cpml=None, domain=None,
              band_split=None, f_ref=10.0, f_band=None, n_sls=3, n_fit=None,
              velocities="unrelaxed", fit="linear"):
        eq = super().build(mesh, vp_e, vs_e, rho_e, Qk_e, Qmu_e, cpml=cpml, domain=domain,
                           band_split=band_split, f_ref=f_ref, f_band=f_band, n_sls=n_sls,
                           n_fit=n_fit, velocities=velocities, fit=fit)
        if cpml is not None:
            E = mesh.n_elements
            off = (~cpml.node_in_pml[:, 1:-1, 1:-1].reshape(E, -1).any(1)).to(vp_e.dtype)
            off = off.view(E, 1, 1)
            eq._w1 = tuple(w * off for w in eq._w1)
            eq._w2 = tuple(w * off for w in eq._w2)
        return eq


CLASSES = {"attenuating layer": ViscoElasticSEM2DDeformed, "elastic layer": ElasticLayer}


def run_sem(g, eq, dt, device, dtype, n_order=4):
    """``(nt, nrec, 2)`` velocity records (z up) of ``eq`` on the attenuating case ``g``."""
    L, Z = g["box"]
    q = g["visco"]
    lay = g["layer"] if g["kind"] != "free" else 0.0
    dh = 10.0 / n_order
    nz, nx = int(round((Z - lay) / dh)) + 1, int(round((L - 2.0 * lay) / dh)) + 1
    grid = lambda p: np.stack([(p[:, 0] - lay) / dh, (Z - p[:, 1]) / dh], 1)
    src, rec = grid(np.array([g["source"]])), grid(g["stations"])[None]
    prop = PropSEM(eq, shape=(nz, nx), dh=dh, dt=dt, n_order=n_order,
                   abcn=int(round(lay / dh)), free_surface=bool(lay), f_dominant=g["f0"],
                   source_type=["vz"], receiver_type=["vx", "vz"], dev=device, impl="eager",
                   dtype=dtype, equation_options={
                       "f_ref": q["f_ref"], "n_sls": q["n_sls"],
                       "f_band": (q["f_ref"] / 10.0, 10.0 * q["f_ref"]),
                       "n_fit": 2 * q["n_sls"] - 1, "velocities": "unrelaxed"})
    models = [torch.full((nz, nx), float(g[k]), dtype=dtype) for k in ("vp", "vs", "rho")]
    models += [torch.full(prop._gll_shape, float(q[k]), dtype=dtype) for k in ("qkappa", "qmu")]
    with torch.no_grad():
        out = prop(torch.from_numpy(g["w"]).to(dtype), src, rec, models=models)
    out = out[0].double().cpu().numpy()
    out[..., 1] *= -1.0                                   # v_z down -> SPECFEM2D's up
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the saved traces")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    if args.plot_only:
        saved = np.load(args.out / f"absorb_visco_traces_{args.dtype}.npz")
        groups = load(args.runs / "absorb_visco_ref")["groups"]
        plot(args.out, saved["t"], {k: saved[k] for k in saved.files if k != "t"}, groups)
        return

    cases = {c: load(args.runs / c) for c in ("absorb_visco_pml", "absorb_visco_ref")}
    ref, pml = cases["absorb_visco_ref"], cases["absorb_visco_pml"]
    dt = float(np.diff(ref["t"]).mean())
    assert np.array_equal(pml["w"], ref["w"])
    groups = ref["groups"]
    res = {"specfem_cpml": reflection(pml["specfem"], ref["specfem"], groups)}
    traces = {"specfem_ref": ref["specfem"], "specfem_cpml": pml["specfem"]}
    eq = ViscoElasticSEM2DDeformed
    mine_ref = run_sem(ref, eq, dt, args.device, dtype)
    res["check_ref_vs_specfem"] = float(
        np.linalg.norm(on_specfem_times(ref, eq, mine_ref, dt) - ref["specfem"])
        / np.linalg.norm(ref["specfem"]))
    traces["sweep_ref"] = mine_ref
    print("check ref vs specfem", res["check_ref_vs_specfem"], flush=True)
    for key, cls in CLASSES.items():
        mine = run_sem(pml, cls, dt, args.device, dtype)
        tag = key.replace(" ", "_")
        res[f"sweep_cpml_{tag}"] = reflection(mine, mine_ref, groups)
        res[f"check_cpml_{tag}_vs_specfem_cpml"] = float(
            np.linalg.norm(on_specfem_times(pml, cls, mine, dt) - pml["specfem"])
            / np.linalg.norm(ref["specfem"]))
        traces[f"sweep_cpml_{tag}"] = mine
        print(key, res[f"check_cpml_{tag}_vs_specfem_cpml"], flush=True)
    np.savez(args.out / f"absorb_visco_traces_{args.dtype}.npz", t=ref["t"], **traces)
    (args.out / f"absorb_visco_reflection_{args.dtype}.json").write_text(
        json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:34s} all {v['all']:.2e}  " + "  ".join(
                f"[{lab}] med {v[lab]['median']:.2e} max {v[lab]['max']:.2e} peak {v[lab]['peak']:.2e}"
                for lab, _ in groups))
        else:
            print(f"{k:34s} {v:.2e}")

    plot(args.out, ref["t"], traces, groups)


def plot(out, t, traces, groups):
    """``v_z`` on the line below the top and the vertical line in the left layer: the
    reference and each layer's reflection, 20x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [("SPECFEM2D visco-elastic C-PML", "specfem_cpml", "specfem_ref"),
             ("sweep, attenuating layer", "sweep_cpml_attenuating_layer", "sweep_ref"),
             ("sweep, elastic layer", "sweep_cpml_elastic_layer", "sweep_ref")]
    rows = [(0, groups[0]), (2, groups[2])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), sharex=True, constrained_layout=True)
    for r, (g, (glab, n)) in enumerate(rows):
        lo = sum(m for _, m in groups[:g])
        sl = slice(lo, lo + n)
        s = 0.5 / np.abs(traces["specfem_ref"][:, sl, 1]).max()
        for c, (lab, key, base) in enumerate(pairs):
            ax = axes[r, c]
            ref, d = traces[base][:, sl, 1], traces[key][:, sl, 1] - traces[base][:, sl, 1]
            for i in range(n):
                ax.plot(t, i + s * ref[:, i], color="0.75", lw=0.8)
                ax.plot(t, i + 20.0 * s * d[:, i], color="C3", lw=0.7)
            ax.set_title(f"({'abcdef'[3 * r + c]}) {lab}, {glab}", fontsize=9)
            if r == 1:
                ax.set_xlabel("time (s)")
        axes[r, 0].set_ylabel("receiver")
    fig.suptitle("v_z: reference grey, layer reflection 20x red (Qkappa 30, Qmu 20)", fontsize=10)
    fig.savefig(out / "absorb_visco_vz.png", dpi=220)
    print("wrote", out / "absorb_visco_vz.png")


if __name__ == "__main__":
    main()

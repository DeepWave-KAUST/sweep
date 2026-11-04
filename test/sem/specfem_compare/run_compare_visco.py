"""ViscoElasticVSSEM2DDeformed against SPECFEM2D's GSLS attenuation (``ATTENUATION_VISCOELASTIC``).

Two SPECFEM2D runs of one homogeneous model with a flat free top and Stacey sides -- elastic,
and visco-elastic with ``Qkappa = Qmu`` (our ``Qp = Qs``), ``N_SLS`` mechanisms and the
velocities read at the source's dominant frequency (``READ_VELOCITIES_AT_f0``) -- against
ElasticVSSEM2DDeformed and ViscoElasticVSSEM2DDeformed on the same box: x in SPECFEM2D's
elements, ``n_order = 4``, half its time step, an explosion of SPECFEM2D's Ricker (a stress
rate of its derivative, PropSEM's convention) and velocity receivers at its stations.

Each code's visco/elastic spectral ratio cancels the wavelet, the time origin and the
amplitude conventions, so it compares the attenuation alone; the time-domain misfits use one
shift and one scale fitted on the elastic pair and applied unchanged to the visco pair.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_visco.py \
        --elastic <elastic run> --visco <visco run> --out <dir>
"""
import argparse
import re
from pathlib import Path

import numpy as np
import torch

from sweep.sem import ElasticVSSEM2DDeformed, PropSEM, ViscoElasticVSSEM2DDeformed


def par(text, key):
    return re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", text, re.M).group(1).replace("d", "e")


def setup(run):
    """The box, model, source and stations of a SPECFEM2D run directory."""
    d = run / "DATA"
    p, s = (d / "Par_file").read_text(), (d / "SOURCE").read_text()
    model = re.search(r"^1 1 (.*)$", p, re.M).group(1).replace("d", "e").split()
    rho, vp, vs, q_kappa, q_mu = (float(model[i]) for i in (0, 1, 2, 5, 6))
    itf = [line.split() for line in (d / "interfaces.dat").read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    stations = np.loadtxt(d / "STATIONS", usecols=(2, 3))
    ztop = max(float(line[1]) for line in itf if len(line) == 2)    # the top interface
    return dict(xmax=float(par(p, "xmax")), nx=int(par(p, "nx")), ztop=ztop,
                dt=float(par(p, "DT")), nstep=int(par(p, "NSTEP")),
                n_sls=int(par(p, "N_SLS")), rho=rho, vp=vp, vs=vs, q=(q_kappa, q_mu),
                src=(float(par(s, "xs")), float(par(s, "zs"))), f0=float(par(s, "f0")),
                tshift=float(par(s, "tshift")), stations=stations)


def run_sem(g, visco, device):
    n_order = 4
    dh = g["xmax"] / (n_order * g["nx"])
    nz = n_order * int(round(g["ztop"] / (n_order * dh))) + 1
    nx = n_order * g["nx"] + 1
    dt, nt = 0.5 * g["dt"], 2 * g["nstep"]
    t = np.arange(nt) * dt - (1.2 / g["f0"] + g["tshift"])      # SPECFEM2D's time axis
    a = (np.pi * g["f0"] * t) ** 2
    ricker = (1.0 - 2.0 * a) * np.exp(-a)
    kw = dict(shape=(nz, nx), dh=dh, dt=dt, n_order=n_order, free_surface=True,
              f_dominant=g["f0"], dev=device, impl="eager", source_type=["sxx", "szz"],
              receiver_type=["vx", "vz"])
    full = lambda v: torch.full((nz, nx), float(v), device=device)
    models = [full(g["vp"]), full(g["vs"]), full(g["rho"])]
    if visco:
        # SPECFEM2D fits its mechanisms over f0 / 10 .. 10 f0 (compute_attenuation_coeffs)
        prop = PropSEM(ViscoElasticVSSEM2DDeformed, equation_options={
            "f_ref": g["f0"], "f_band": (g["f0"] / 10.0, g["f0"] * 10.0),
            "n_sls": g["n_sls"]}, **kw)
        models += [full(g["q"][0]), full(g["q"][1])]
    else:
        prop = PropSEM(ElasticVSSEM2DDeformed, **kw)
    src = np.array([[g["src"][0] / dh, (g["ztop"] - g["src"][1]) / dh]])
    rec = np.stack([g["stations"][:, 0] / dh, (g["ztop"] - g["stations"][:, 1]) / dh], 1)[None]
    w = torch.from_numpy(np.gradient(ricker, dt).astype(np.float32)).to(device)
    with torch.no_grad():
        out = prop(w, src, rec, models=models)[0].double().cpu().numpy()
    out[..., 1] *= -1.0                                         # z down -> SPECFEM's up
    return t, out


def load_specfem(run, n):
    tr = [[np.loadtxt(run / "OUTPUT_FILES" / f"AA.S{i:03d}.BX{c}.semv") for i in range(n)]
          for c in "XZ"]
    return tr[0][0][:, 0], np.stack([np.stack([a[:, 1] for a in tr[c]], -1) for c in range(2)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elastic", type=Path, required=True)
    ap.add_argument("--visco", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g = setup(args.visco)
    assert g["q"][0] == g["q"][1], "Qkappa = Qmu (our Qp = Qs) only"
    n = len(g["stations"])
    t_sp, spE = load_specfem(args.elastic, n)
    _, spV = load_specfem(args.visco, n)
    t, semE = run_sem(g, False, args.device)
    _, semV = run_sem(g, True, args.device)
    np.savez(args.out / "sem_flat_visco.npz", t=t, elastic=semE, visco=semV)

    # one shift (from the elastic cross-correlation) and one scale, both from the elastic pair
    on = lambda d, s: np.stack([np.stack([np.interp(t, t_sp + s, d[:, i, c], left=0, right=0)
                                          for i in range(n)], -1) for c in range(2)], -1)
    shifts = np.arange(-0.05, 0.05, 1e-4)
    cc = [abs(float((on(spE, s)[..., 1] * semE[..., 1]).sum())) for s in shifts]
    shift = float(shifts[int(np.argmax(cc))])
    aE, aV = on(spE, shift), on(spV, shift)
    win = (t <= t_sp[-1] + shift - 0.02).astype(float)
    iend = int(win.sum())
    win[iend - 80:iend] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, 80)))
    aE, aV, sE, sV = (x * win[:, None, None] for x in (aE, aV, semE, semV))
    scale = float((aE * sE).sum() / (sE * sE).sum())
    sE, sV = sE * scale, sV * scale
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"shift {shift:+.4f} s, scale {scale:.4e}")
    print(f"rel L2 SEM vs SPECFEM2D: elastic {rel(sE, aE):.4f}, visco {rel(sV, aV):.4f}")
    print(f"visco/elastic energy: SPECFEM2D {np.linalg.norm(aV) / np.linalg.norm(aE):.4f}, "
          f"SEM {np.linalg.norm(sV) / np.linalg.norm(sE):.4f}")

    off = g["stations"][:, 0] - g["src"][0]
    nf = 8192
    f = np.fft.rfftfreq(nf, t[1] - t[0])
    F = lambda d: np.fft.rfft(d, nf, axis=0)
    R_sp, R_sem = F(aV) / F(aE), F(sV) / F(sE)
    band = (f >= 4) & (f <= 25)
    for lab, sel in (("|off| <= 200 m", np.abs(off) <= 200),
                     ("200 < |off| <= 450 m", (np.abs(off) > 200) & (np.abs(off) <= 450))):
        for c, cn in enumerate("xz"):
            E = np.abs(F(aE))[band][:, sel, c]
            mask = E > 0.05 * E.max(0, keepdims=True)
            ra = np.abs(np.abs(R_sem[band][:, sel, c]) / np.abs(R_sp[band][:, sel, c]) - 1)[mask]
            ph = np.abs(np.angle(R_sem[band][:, sel, c] / R_sp[band][:, sel, c]))[mask]
            print(f"  v{cn} {lab}: | |R_SEM| / |R_SPECFEM| - 1 | median {np.median(ra):.4f} "
                  f"p95 {np.percentile(ra, 95):.4f}; phase median {np.median(ph):.4f} "
                  f"p95 {np.percentile(ph, 95):.4f} rad")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [int(np.argmin(np.abs(off - o))) for o in (-120, 250, 450)]
    fig, axes = plt.subplots(len(picks) + 1, 2, figsize=(12, 11), constrained_layout=True)
    for r, i in enumerate(picks):
        for c, cn in enumerate("xz"):
            ax = axes[r, c]
            ax.plot(t[:iend], aE[:iend, i, c], color="0.6", lw=2.0, label="SPECFEM2D elastic")
            ax.plot(t[:iend], sE[:iend, i, c], "C0--", lw=1.0, label="SEM elastic")
            ax.plot(t[:iend], aV[:iend, i, c], "k", lw=1.4, label="SPECFEM2D visco")
            ax.plot(t[:iend], sV[:iend, i, c], "C3--", lw=1.0, label="SEM visco")
            ax.set_title(f"({'abcdef'[2 * r + c]}) v{cn}, offset {off[i]:.0f} m", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for r, i in enumerate(picks):
        axes[-1, 0].plot(f[band], np.abs(R_sp[band, i, 1]), color=f"C{r}", lw=2.0, alpha=0.5,
                         label=f"SPECFEM2D {off[i]:.0f} m")
        axes[-1, 0].plot(f[band], np.abs(R_sem[band, i, 1]), color=f"C{r}", ls="--", lw=1.0,
                         label=f"SEM {off[i]:.0f} m")
        axes[-1, 1].plot(f[band], np.unwrap(np.angle(R_sp[band, i, 1])), color=f"C{r}", lw=2.0,
                         alpha=0.5)
        axes[-1, 1].plot(f[band], np.unwrap(np.angle(R_sem[band, i, 1])), color=f"C{r}", ls="--",
                         lw=1.0)
    axes[-1, 0].set_title("(g) |visco / elastic|, vz", fontsize=9)
    axes[-1, 1].set_title("(h) phase(visco / elastic), vz", fontsize=9)
    axes[-1, 0].legend(frameon=False, fontsize=7)
    for ax in axes[-2]:
        ax.set_xlabel("time (s)")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.savefig(args.out / "sem_vs_specfem2d_visco.png", dpi=220)
    print("wrote", args.out / "sem_vs_specfem2d_visco.png")


if __name__ == "__main__":
    main()

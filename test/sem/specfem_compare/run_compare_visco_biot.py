"""ViscoBiotVSSEM2DDeformed against SPECFEM2D's attenuating poroelastic frame.

SPECFEM2D refuses attenuation in a poroelastic-only model, so this compares on its
``poroelastic_acoustic`` example of ``run_compare_poro_coupling.py``: ``pa_example`` (water
over an inviscid porous half-space whose material has ``Qmu = 50``) and the same run with
``ATTENUATION_VISCOELASTIC = .true.`` (``pa_qmu``). It compares the attenuation's own effect,
``pa_qmu - pa_example``, each code over its own ``pa_example`` records.

SPECFEM2D attenuates the frame's shear modulus only, takes the given moduli as unrelaxed (no
``READ_VELOCITIES_AT_f0``) and fits its mechanisms over ``f0 / 10 .. 10 f0`` (``f0`` = 15 Hz,
the source's; ``USE_SOLVOPT = .false.``). Ours does the same: ``Qs = 50``, the drained P
modulus relaxing as the shear modulus does (the 2-D bulk ``pi_d - mu`` does not), and
``f_ref`` far above the band. Our fit then gives an effect about 8% larger than SPECFEM2D's.
Its poroelastic routine scales the memories by the relaxed shear modulus, where its elastic
routine (for which its strengths are fitted) uses the unrelaxed one, so its Q is about 10%
above the requested. With ``--specfem-gsls`` the frame takes SPECFEM2D's own mechanisms, with
that scaling, instead of our fit.

SPECFEM2D 8.1.0 (abf3196) needs three fixes in its poroelastic Newmark path first:

1. ``compute_forces_poro_solid`` and ``compute_forces_poro_fluid`` both update the memory
   variables, each for both phases (``iphase = 1, 2``), so they are updated four times per
   step. Update them once.
2. ``displs_poroelastic_old`` is never set, so the recursive convolution's term of step
   ``n - 1`` is zero. Set it after the update.
3. The fluid routine's attenuated total stress has ``mu_G`` for ``lambdal_G`` in its cross
   terms.

As released, its frame's Q is about 115 where 50 is requested.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_visco_biot.py \\
        --runs <dir with pa_example and pa_qmu> [--specfem-gsls]
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sweep.sem.equations.visco_biot_vs_deformed as visco_biot
from sweep.equations.visco_elastic import gsls_moduli_factors
from sweep.sem import ViscoBiotVSSEM2DDeformed

from run_compare_poro_coupling import CASES, PA_ROCK, run_sem

QS = 50.0                      # the porous material's Qmu in SPECFEM2D's example


def specfem_mechanisms(q, f0, n=3):
    """``(tau_sigma, w)`` of SPECFEM2D's fit for ``Q = q`` (``classical_linear_least_squares``):
    relaxation frequencies log-spaced over ``f0 / 10 .. 10 f0``, strengths the least-squares
    solution of ``Im / Re = 1 / Q`` at ``2n - 1`` log-spaced frequencies. Its elastic modulus
    is our GSLS with these strengths."""
    f_min, f_max = f0 / 10.0, f0 * 10.0
    point = 2 * math.pi * f_min * (f_max / f_min) ** (np.arange(n) / (n - 1))
    om = 2 * math.pi * f_min * (f_max / f_min) ** (np.arange(2 * n - 1) / (2 * n - 2))
    basis = om[:, None] * (point - om[:, None] / q) / (point ** 2 + om[:, None] ** 2)
    return 1.0 / point, np.linalg.lstsq(basis, np.full(len(om), 1.0 / q), rcond=None)[0]


@contextlib.contextmanager
def frame_mechanisms(tau_sigma, ts, mu, pi_d):
    """ViscoBiotVSSEM2DDeformed's frame with these shear mechanisms (relaxation times,
    strengths ``ts``) in place of its fit, the drained P modulus ``pi_d`` relaxing as ``mu``
    does (``pi_d - mu`` unrelaxed); both moduli are given unrelaxed."""
    mu_r = mu / (1.0 + sum(ts))
    pi_r = pi_d - (mu - mu_r)
    tp = [mu_r * t / pi_r for t in ts]
    tau = np.asarray(tau_sigma, dtype=float)

    def setup(Qp_e, Qs_e, f_ref, f_band, n_sls, who):
        assert n_sls == len(tau), (n_sls, len(tau))
        s = [torch.full_like(Qs_e, float(t)) for t in ts]
        p = [torch.full_like(Qp_e, float(t)) for t in tp]
        return SimpleNamespace(f_ref=f_ref, f_band=f_band, n_sls=n_sls, q_min=0.0,
                               tau_sigma=tuple(float(t) for t in tau), tp=p, ts=s,
                               p=gsls_moduli_factors(p, tau, f_ref),
                               s=gsls_moduli_factors(s, tau, f_ref))

    saved = visco_biot.gsls_setup
    visco_biot.gsls_setup = setup
    try:
        yield
    finally:
        visco_biot.gsls_setup = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--case", default="pa_qmu", help="the attenuated SPECFEM2D run")
    ap.add_argument("--specfem-gsls", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    f0, recs = CASES["pa_example"][1], CASES["pa_example"][7]
    h_u, mu, c, m = (float(PA_ROCK[i]) for i in range(4))
    pi_d = h_u - c * c / m

    def specfem(case):
        out = args.runs / case / "OUTPUT_FILES"
        t = np.loadtxt(out / "AA.S0001.BXX.semv")[:, 0]
        return t, np.stack([np.stack([np.loadtxt(out / f"AA.S{r + 1:04d}.{k}.semv")[:, 1]
                                      for k in ("BXX", "BXZ")], -1) for r in range(len(recs))], 1)

    t_sp, sp0 = specfem("pa_example")
    _, sp1 = specfem(args.case)
    t_stf, stf = np.loadtxt(args.runs / "pa_example" / "OUTPUT_FILES" /
                            "plot_source_time_function.txt", unpack=True)
    t, s0 = run_sem("pa_example", stf, args.device, torch.float32)
    visco = dict(equation=ViscoBiotVSSEM2DDeformed, q=(QS * pi_d / mu, QS),
                 equation_options={"f_ref": 1e6, "f_band": (f0 / 10.0, f0 * 10.0), "n_sls": 3})
    if args.specfem_gsls:
        tau, w = specfem_mechanisms(QS, f0)
        c_l = w / (1.0 + w.sum()) ** 2         # its poroelastic M_U - M_U sum c_l / (1 + i w tau_l)
        with frame_mechanisms(tau, c_l / (1.0 - c_l.sum()), mu, pi_d):
            _, s1 = run_sem("pa_example", stf, args.device, torch.float32, **visco)
    else:
        _, s1 = run_sem("pa_example", stf, args.device, torch.float32, **visco)

    def on(x):                                 # onto SPECFEM2D's time axis, t - t0
        return np.stack([np.stack([np.interp(t_sp, t + t_stf[0], x[:, r, k]) for k in range(2)],
                                  -1) for r in range(len(recs))], 1)

    s0, s1 = on(s0), on(s1)
    norm = np.linalg.norm
    d_sp, d_sem = (sp1 - sp0) / norm(sp0), (s1 - s0) / norm(s0)
    print(f"the attenuation changes the records by {norm(d_sp):.4f} (SPECFEM2D), "
          f"{norm(d_sem):.4f} (SEM); the records with it differ by "
          f"{norm(s1 - sp1) / norm(sp1):.4f}")
    print(f"the effect, SEM against SPECFEM2D: |SEM| / |SPECFEM2D| {norm(d_sem) / norm(d_sp):.4f}, "
          f"relative L2 {norm(d_sem - d_sp) / norm(d_sp):.4f}, "
          f"corr {float((d_sem * d_sp).sum()) / (norm(d_sem) * norm(d_sp)):+.5f}")
    for r in range(len(recs)):
        for k in range(2):
            a, b = d_sem[:, r, k], d_sp[:, r, k]
            if norm(b) > 0.05 * norm(d_sp):
                print(f"  rec{r + 1} {('BXX', 'BXZ')[k]}: |SEM| / |SPECFEM2D| "
                      f"{norm(a) / norm(b):.4f}, corr {float(a @ b) / (norm(a) * norm(b)):+.5f}")


if __name__ == "__main__":
    main()

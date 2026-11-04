"""The 3-D SEM against SPECFEM3D on the boxes ``make_specfem3d_box_cases.py`` builds: one homogeneous medium, all six sides free, 10 m Hex8 elements (its NGLLX = 5 =
``n_order = 4``), a vertical point force inside an element and ten stations inside elements, on
element faces and edges and on the surface.

* an elastic rock (``ATTENUATION = .false.``): ElasticVSSEM3D, ElasticVSSEM3DDeformed (a flat
  Hex27 mesh) and the displacement class ElasticSEM3D;
* an attenuating rock: ViscoElasticVSSEM3D with SPECFEM3D's band, its three mechanisms and its
  reading of the velocities (the real part of each modulus at the band's logarithmic centre,
  which the cases make ``ATTENUATION_f0_REFERENCE``: ``velocities="real_modulus"``), our own
  constant-Q fit (``Qp = Qs``, so only ``Qkappa = Qmu``); and ``SpecfemMechanisms``, the same
  class on SPECFEM3D's own relaxation times, strengths and unrelaxed moduli (its simplex fit
  of the strain relaxation times, read from its ``attenuation.bin``), any ``Qkappa``, ``Qmu``:
  a bulk and a shear defect per mechanism are a P and a shear one, ``kappa c_kappa + 4/3 mu
  c_mu`` and ``mu c_mu``; and ``SpecfemScheme``, SpecfemMechanisms with SPECFEM3D's memory
  update in place of ours;
* water over the rock (SPECFEM3D's acoustic domain, its potential ``chi`` coupled to the
  displacement of the rock): BiotVSSEM3DDeformed with a fluid region (``biot_fluid``) over a dry
  elastic one (``biot_elastic``), a pressure-free sea surface, the force in the rock; in the
  water its pressure ``p`` against SPECFEM3D's ``-chi''`` and its fluid velocity ``q`` against
  SPECFEM3D's ``grad(chi') / rho``, in the rock ``v``. Eliminating ``q`` leaves ``p`` the
  second-order pressure equation with the potential's stiffness, and ``-chi''`` of SPECFEM3D's
  potential obeys the same leapfrog, the coupling through the same ``v^(n+1/2)`` and ``p^n``:
  the two are one scheme.

SPECFEM3D's attenuation: ``sigma = C_U eps - sum_l R_l``, ``dR_l/dt = -R_l / tau_sigma_l + 2
mu_U c_l eps_dev / tau_sigma_l`` (and ``kappa_U c_kappa_l tr(eps)`` for the bulk part), ``c_l
= (tau_eps_l / tau_sigma_l - 1) / sum_k tau_eps_k / tau_sigma_k`` (its ``factor_common`` is ``2
c_l / tau_sigma_l``): the modulus ``M_U (1 - sum_l c_l / (1 + i w tau_sigma_l))``, ours with
``M_R t_l = M_U c_l``. Its memories advance by a fourth-order Taylor form of the exponential
integrator on the strain one step ahead (``u + dt v``); ours by the trapezoidal rule on the
strain rate. Both are second order: what separates SPECFEM3D's records from SpecfemMechanisms'
is that time discretisation (on the same mechanisms the two updates differ by 6.5e-4, 1.6e-4,
4.1e-5 as dt halves from 0.25 ms in a small box), and SpecfemScheme, which runs SPECFEM3D's
update, has no other difference to show. SPECFEM3D's strain one step ahead, extrapolated from
the last two displacements (``2 u^n - u^(n-1)``), is ``1 + (w dt)^2`` times the strain at the
next step, so its discrete modulus is off by 1.3e-4 at 25 Hz and dt = 0.25 ms; the trapezoidal
rule only warps the frequency, 3e-6 there.

An absolute comparison: SPECFEM3D's own source time function (``plot_source_time_function.txt``),
time step, mesh, source and stations -- no fitted shift or scale. SPECFEM3D's z points up: its
force ``Z_UP = -1`` is our ``vz`` force, its BXZ is -v_z here; its x (east) and y (north) are
ours. A load at step n is the one at n dt (its external source time function starts at t = 0);
the velocity-stress classes' velocity is at (n + 1/2) dt (their pressure at (n + 1) dt, where
SPECFEM3D's -chi'' at n dt is) and SPECFEM3D's Newmark velocity at
n dt is the mean of the two around it, which a linear interpolation of ours reproduces; the
displacement class's velocity is the central difference, at (n + 1) dt, as SPECFEM3D's is. The
last sample is left out.

    PYTHONPATH=src python test/sem/specfem_compare/run_compare_box3d.py \\
        --runs <dir with the case directories> --out <dir> [--case iso3d_force ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import torch

from sweep.equations.biot_params import biot_elastic, biot_fluid
from sweep.propagator import EagerOptions, Full
from sweep.sem import (BiotVSSEM3DDeformed, ElasticSEM3D, ElasticVSSEM3D,
                       ElasticVSSEM3DDeformed, PropSEM, ViscoElasticVSSEM3D)
from sweep.sem.equations.visco_elastic_vs3d import gsls_memory_specs3
from sweep.sem.fields import FieldSpec

CASES = ("iso3d_force", "visco3d_force", "visco3d_km_force", "sea3d_force")
COMPS = ("X", "Y", "Z")
N_SLS = 3                             # SPECFEM3D's constants.h

# SPECFEM3D's mechanisms for the case at hand (specfem_attenuation)
SPECFEM3D_ATT: dict = {}


class SpecfemMechanisms(ViscoElasticVSSEM3D):
    """ViscoElasticVSSEM3D on SPECFEM3D's own relaxation times, strengths and unrelaxed moduli
    (``SPECFEM3D_ATT``, a homogeneous medium): ``vp``, ``vs`` are SPECFEM3D's input velocities,
    which its scale factors take to the unrelaxed ones; the Q models are not read."""

    @classmethod
    def _build(cls, mesh, vp_e, vs_e, rho_e, Qp_e, Qs_e, *, cpml, f_ref, f_band, n_sls, n_fit,
               velocities):
        a = SPECFEM3D_ATT
        mu_U = rho_e * vs_e * vs_e * a["scale_mu"]
        kappa_U = rho_e * (vp_e * vp_e - 4.0 / 3.0 * vs_e * vs_e) * a["scale_kappa"]
        eq = cls._elastic.build(mesh, ((kappa_U + 4.0 / 3.0 * mu_U) / rho_e).sqrt(),
                                (mu_U / rho_e).sqrt(), rho_e, cpml=cpml, band_split=False)
        eq.__class__ = cls
        eq.Qp_e, eq.Qs_e = Qp_e, Qs_e
        eq.f_ref, eq.f_band, eq.n_sls, eq.q_min = f_ref, f_band, N_SLS, 0.0
        eq.n_fit, eq.velocities = n_fit, velocities
        eq._tau_sigma = tuple(a["tau_sigma"])
        eq._p_tau = tuple(kappa_U * ck + 4.0 / 3.0 * mu_U * cm
                          for ck, cm in zip(a["c_kappa"], a["c_mu"]))
        eq._m_tau = tuple(mu_U * cm for cm in a["c_mu"])
        eq._kappa_U, eq._mu_U = kappa_U, mu_U
        eq.field_specs = ElasticVSSEM3D.field_specs + gsls_memory_specs3(N_SLS)
        return eq


def specfem_taylor(tau_sigma, dt):
    """SPECFEM3D's ``get_attenuation_memory_values``: alpha, beta, gamma per mechanism."""
    out = []
    for ts in tau_sigma:
        t = -1.0 / ts
        out.append((1.0 + dt * t + dt ** 2 * t ** 2 / 2.0 + dt ** 3 * t ** 3 / 6.0
                    + dt ** 4 * t ** 4 / 24.0,
                    dt / 2.0 + dt ** 2 * t / 3.0 + dt ** 3 * t ** 2 / 8.0 + dt ** 4 * t ** 3 / 24.0,
                    dt / 2.0 + dt ** 2 * t / 6.0 + dt ** 3 * t ** 2 / 24.0))
    return out


class SpecfemScheme(SpecfemMechanisms):
    """SpecfemMechanisms with SPECFEM3D's memory update in place of ours (no C-PML): per
    mechanism ``R+ = alpha R + F (beta e_old + gamma e)`` (its fourth-order Taylor coefficients,
    ``compute_element_att_memory_second_order_rk``) on its strain one step ahead, ``e =
    eps(u^n + dt v^(n-1/2))``, bulk (``F = kappa_U factor_common_kappa``, the trace) and
    deviatoric (``F = mu_U factor_common``) apart, and ``sigma^(n+1) = C_U eps(u^(n+1)) - sum
    R`` (``compute_forces_viscoelastic``) -- here ``sigma + dt C_U eps(v) - sum (R+ - R)``. The
    memories are its ``R_trace, R_xx, R_yy, R_xy, R_xz, R_yz``; it carries the strain of the
    displacement and its last two strains one step ahead (trace, deviator)."""

    @classmethod
    def _build(cls, *args, **kw):
        eq = super()._build(*args, **kw)
        eq.field_specs = eq.field_specs + [
            FieldSpec(f"{kind}_{c}", internal=True, layout="element")
            for kind in ("eps", "ahead", "ahead_old") for c in range(6)]
        return eq

    def step(self, fields, src, dt):
        assert self.cpml is None
        n, m = len(ElasticVSSEM3D.field_specs), 6 * N_SLS
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz = fields[:9]
        R = fields[n:n + m]
        eps = fields[n + m:n + m + 6]
        ahead, ahead_old = fields[n + m + 6:n + m + 12], fields[n + m + 12:n + m + 18]
        a = SPECFEM3D_ATT
        new_R, dR = [], [0.0] * 6
        for l, (al, be, ga) in enumerate(specfem_taylor(a["tau_sigma"], dt)):
            f_k = self._kappa_U * a["factor_common_kappa"][l]
            f_m = self._mu_U * a["factor_common"][l]
            for k in range(6):
                r = R[6 * l + k]
                rn = al * r + (f_k if k == 0 else f_m) * (be * ahead_old[k] + ga * ahead[k])
                dR[k] = dR[k] + (rn - r)
                new_R.append(rn)
        Rx, Ry, Rz = self.assemble_stress_divergence(sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz)
        vx = vx + dt * self._inv_M * (-Rx + src[0])
        vy = vy + dt * self._inv_M * (-Ry + src[1])
        vz = vz + dt * self._inv_M * (-Rz + src[2])
        g = self.velocity_gradients(vx, vy, vz)
        inc = self._stress_increment(g, dt)
        d_tr, d_xx, d_yy, d_xy, d_xz, d_yz = dR
        sxx = sxx + inc[0] - d_xx - d_tr
        syy = syy + inc[1] - d_yy - d_tr
        szz = szz + inc[2] + d_xx + d_yy - d_tr
        sxy, sxz, syz = sxy + inc[3] - d_xy, sxz + inc[4] - d_xz, syz + inc[5] - d_yz
        vx_x, vy_y, vz_z, vy_x, vx_y, vz_x, vx_z, vz_y, vy_z = g
        rate = (vx_x, vy_y, vz_z, vx_y + vy_x, vx_z + vz_x, vy_z + vz_y)
        eps = [e + dt * r for e, r in zip(eps, rate)]          # eps(u^(n+1))
        e = [x + dt * r for x, r in zip(eps, rate)]            # eps(u^(n+1) + dt v^(n+1/2))
        tr = e[0] + e[1] + e[2]
        third = tr / 3.0
        new_ahead = (tr, e[0] - third, e[1] - third, 0.5 * e[3], 0.5 * e[4], 0.5 * e[5])
        return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, *fields[9:n], *new_R, *eps,
                *new_ahead, *ahead)


def par(text, key):
    return re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", text, re.M).group(1).replace("d", "e")


def setup(run):
    """The box, medium, source and stations of a SPECFEM3D run directory."""
    d, o = run / "DATA", run / "OUTPUT_FILES"
    p = (d / "Par_file").read_text()
    g = json.loads((d / "sweep_box3d.json").read_text())
    st = [line.split() for line in (o / "output_list_stations.txt").read_text().splitlines()
          if line.strip()]
    stations = np.array([[float(v) for v in s[2:5]] for s in st])          # x, y, z up
    assert np.abs(stations - np.array([[x, y, -z] for x, y, z in g["receivers"]])).max() < 1e-9
    assert float(par(p, "DT")) == g["dt"] and int(par(p, "NSTEP")) == g["nstep"]
    assert (par(p, "ATTENUATION") == ".true.") == g["attenuation"]
    if g["attenuation"]:
        assert par(p, "COMPUTE_FREQ_BAND_AUTOMATIC") == ".false."
        for k in ("min_period", "max_period"):
            assert float(par(p, k.upper().replace("_P", "_ATTENUATION_P"))) == g[k]
        assert float(par(p, "ATTENUATION_f0_REFERENCE")) == g["f0_ref"]
    g["stf"] = np.loadtxt(o / "plot_source_time_function.txt")
    assert len(g["stf"]) == g["nstep"]
    return g


def fortran_records(path):
    """The records of a Fortran unformatted sequential file (4-byte markers)."""
    raw, pos, out = path.read_bytes(), 0, []
    while pos < len(raw):
        (n,) = struct.unpack("<i", raw[pos:pos + 4])
        out.append(raw[pos + 4:pos + 4 + n])
        assert struct.unpack("<i", raw[pos + 4 + n:pos + 8 + n]) == (n,)
        pos += n + 8
    return out


def specfem_attenuation(run, g):
    """SPECFEM3D's mechanisms for a homogeneous case: its relaxation times (its formula, on
    the Par_file's band), ``c_mu``, ``c_kappa`` and the scale factors from its
    ``attenuation.bin`` (kept as ``sweep_attenuation.json`` beside the run's output)."""
    keep = run / "OUTPUT_FILES" / "sweep_attenuation.json"
    binf = run / "OUTPUT_FILES" / "DATABASES_MPI" / "proc000000_attenuation.bin"
    if not binf.exists():
        return json.loads(keep.read_text())
    pi = 3.14159265358979                  # get_attenuation_tau_sigma's own constant
    e1, e2 = math.log10(1.0 / g["max_period"]), math.log10(1.0 / g["min_period"])
    tau = [1.0 / (pi * 2.0 * 10 ** (e1 + i * (e2 - e1) / (N_SLS - 1))) for i in range(N_SLS)]
    recs = fortran_records(binf)
    (nspec,) = np.frombuffer(recs[0], "<i4")
    arr = [np.frombuffer(r, "<f8") for r in recs[1:5]]
    fc, sf, fck, sfk = (arr[0].reshape(-1, N_SLS), arr[1], arr[2].reshape(-1, N_SLS), arr[3])
    assert len(sf) == nspec * g["ngll"] ** 3
    for x in (fc, sf, fck, sfk):
        assert np.ptp(x, axis=0).max() <= 1e-15 * np.abs(x).max(), "not homogeneous"
    out = dict(tau_sigma=tau, c_mu=[0.5 * f * t for f, t in zip(fc[0], tau)],
               c_kappa=[f * t for f, t in zip(fck[0], tau)], scale_mu=float(sf[0]),
               scale_kappa=float(sfk[0]), factor_common=fc[0].tolist(),
               factor_common_kappa=fck[0].tolist())
    keep.write_text(json.dumps(out, indent=1) + "\n")
    return out


def equations(g):
    if "water" in g:
        return [BiotVSSEM3DDeformed]
    if not g["attenuation"]:
        return [ElasticVSSEM3D, ElasticVSSEM3DDeformed, ElasticSEM3D]
    qk, qmu = g["q"]
    return ([ViscoElasticVSSEM3D] if qk == qmu else []) + [SpecfemMechanisms, SpecfemScheme]


def wet(g):
    """The stations in the water."""
    if "water" not in g:
        return np.zeros(len(g["receivers"]), bool)
    return np.array(g["receivers"])[:, 2] < g["water"]["layers"] * g["h"]


def run_sea(g, device, dtype, n_order=4):
    """``{component: (t, (nt, nrec))}`` of BiotVSSEM3DDeformed on the water over the rock."""
    dh = g["h"] / n_order
    n = g["ne"] * n_order + 1
    t, w = g["stf"][:, 0], g["stf"][:, 1]
    prop = PropSEM(BiotVSSEM3DDeformed, shape=(n, n, n), dh=dh, dt=g["dt"], n_order=n_order,
                   abcn=0, free_surface=True, f_dominant=g["f0"], dev=device, dtype=dtype,
                   impl="eager", memory=Full(), eager_options=EagerOptions(use_compile=False),
                   source_type=["fz"],
                   receiver_type=["vx", "vy", "vz", "p", "qx", "qy", "qz"])
    mesh = prop.mesh
    water = (mesh.z_e.flatten(1).mean(1) < g["water"]["layers"] * g["h"]).to(torch.long)
    wat = g["water"]
    media = [biot_elastic(g["vp"], g["vs"], g["rho"]),
             biot_fluid(wat["rho"] * wat["vp"] ** 2, wat["rho"])]
    E, k = mesh.n_elements, mesh.ngll
    models = [torch.stack([torch.as_tensor(m[i], dtype=dtype) for m in media]).to(
        mesh.z_e.device)[water].view(E, 1, 1, 1).expand(E, k, k, k).contiguous()
        for i in range(8)]
    src = np.array([g["source"]]) / dh
    rec = np.array(g["receivers"])[None] / dh
    with torch.no_grad():
        out = prop(torch.from_numpy(w).to(dtype), src, rec, models=models)[0]
    out = out.double().cpu().numpy()
    fl = wet(g)
    vel = np.where(fl[None, :, None], out[..., 4:7], out[..., 0:3])     # q in the water
    half, one = t + 0.5 * g["dt"], t + g["dt"]
    return {"X": (half, vel[..., 0]), "Y": (half, vel[..., 1]), "Z": (half, -vel[..., 2]),
            "P": (one, np.where(fl[None, :], out[..., 3], 0.0))}


def run_sem(g, eq, device, dtype, n_order=4):
    """``{component: (t, (nt, nrec))}`` of class ``eq`` on the case ``g``."""
    if "water" in g:
        return run_sea(g, device, dtype, n_order)
    dh = g["h"] / n_order
    n = g["ne"] * n_order + 1
    full = lambda v: torch.full((n, n, n), float(v), dtype=dtype)
    t, w = g["stf"][:, 0], g["stf"][:, 1]                 # the loads SPECFEM3D applied
    src = np.array([g["source"]]) / dh                     # x, y, depth
    rec = np.array(g["receivers"])[None] / dh
    kw = {}
    models = [full(g[k]) for k in ("vp", "vs", "rho")]
    if g["attenuation"]:
        models += [full(q) for q in g["q"]]
        kw["equation_options"] = {"f_ref": g["f0_ref"], "n_sls": N_SLS,
                                  "f_band": (1.0 / g["max_period"], 1.0 / g["min_period"]),
                                  "velocities": "real_modulus"}
    prop = PropSEM(eq, shape=(n, n, n), dh=dh, dt=g["dt"], n_order=n_order, abcn=0,
                   f_dominant=g["f0"], dev=device, dtype=dtype, impl="eager",
                   memory=Full(), eager_options=EagerOptions(use_compile=False),
                   source_type=["vz"], receiver_type=["vx", "vy", "vz"], **kw)
    with torch.no_grad():
        out = prop(torch.from_numpy(w).to(dtype), src, rec, models=models)[0]
    out = out.double().cpu().numpy()
    t = t + (1.0 if eq is ElasticSEM3D else 0.5) * g["dt"]
    return {"X": (t, out[..., 0]), "Y": (t, out[..., 1]), "Z": (t, -out[..., 2])}  # z up


def load_specfem(run, n, fluid):
    """``(t, {component: (nt, n)})``: SPECFEM3D's velocity seismograms (``.semv``) and, at the
    stations in the water (``fluid``), its pressure (``.semp``; zero elsewhere)."""
    out, t = {}, None
    for c in COMPS:
        tr = []
        for i in range(n):
            (f,) = sorted((run / "OUTPUT_FILES").glob(f"AA.S{i + 1:04d}.?X{c}.semv"))
            a = np.loadtxt(f)
            tr.append(a[:, 1])
            t = a[:, 0]
        out[c] = np.stack(tr, -1)
    if fluid.any():
        out["P"] = np.stack([np.loadtxt(sorted((run / "OUTPUT_FILES").glob(
            f"AA.S{i + 1:04d}.*.semp"))[0])[:, 1] if fluid[i] else np.zeros(len(t))
            for i in range(n)], -1)
    return t, out


def direct_window(g):
    """The end of each station's echo-free window: the first free-side echo (image sources in
    the six sides, the P speed) less a period."""
    L = g["ne"] * g["h"]
    s = np.array(g["source"])
    images = []
    for a in range(3):
        for wall in (0.0, L):
            m = s.copy()
            m[a] = 2.0 * wall - s[a]
            images.append(m)
    r = np.array(g["receivers"])
    dist = np.min([np.linalg.norm(r - m, axis=1) for m in images], 0)
    return dist / g["vp"] - 1.0 / g["f0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", nargs="*", default=list(CASES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
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
        n = len(g["receivers"])
        groups = np.repeat([k for k, _ in g["groups"]], [m for _, m in g["groups"]])
        t_end = direct_window(g)
        if g["attenuation"]:
            SPECFEM3D_ATT.clear()
            SPECFEM3D_ATT.update(specfem_attenuation(run, g))
            a = SPECFEM3D_ATT
            print(f"{case}: SPECFEM3D tau_sigma {np.array(a['tau_sigma'])}, c_mu "
                  f"{np.array(a['c_mu'])}, c_kappa {np.array(a['c_kappa'])}, scale mu "
                  f"{a['scale_mu']:.15f} kappa {a['scale_kappa']:.15f}", flush=True)
        fluid = wet(g)
        for eq in equations(g):
            sem = run_sem(g, eq, args.device, dtype)
            t_ref, ref = load_specfem(run, n, fluid)
            assert np.abs(t_ref - g["stf"][:, 0]).max() < 1e-12
            keep = t_ref <= min(ts[-2] for ts, _ in sem.values())
            ours = {c: np.stack([np.interp(t_ref, ts[:-1], tr[:-1, i], left=0.0)
                                 for i in range(n)], -1) for c, (ts, tr) in sem.items()}
            np.savez(args.out / f"sem_{case}_{eq.__name__}_{args.dtype}.npz", t_ref=t_ref,
                     **{f"t_sem_{k}": v[0] for k, v in sem.items()},
                     **{f"sem_{k}": v[1] for k, v in sem.items()},
                     **{f"ref_{k}": v for k, v in ref.items()})
            print(f"{case}, {eq.__name__} ({args.dtype}): shift-free, scale-free", flush=True)
            comps = {grp: COMPS + (("P",) if "P" in ref and grp == "water" else ())
                     for grp, _ in g["groups"]}
            peak = {c: np.abs(ref[c][keep]).max() for c in ref}
            allo, allr = [], []
            for win in (True, False):
                for grp, _ in g["groups"]:
                    for c in comps[grp]:
                        errs = []
                        for i in np.flatnonzero(groups == grp):
                            msk = keep & ((t_ref <= t_end[i]) if win else True)
                            if np.linalg.norm(ref[c][msk, i]) > 1e-3 * peak[c]:
                                errs.append(rel(ours[c][msk, i], ref[c][msk, i]))
                            if not win:
                                allo.append(ours[c][keep, i] / peak[c])
                                allr.append(ref[c][keep, i] / peak[c])
                        if errs:
                            name = "PRE" if c == "P" else f"BX{c}"
                            print(f"  {grp}, {'echo-free' if win else 'full record'} {name}: "
                                  f"rel L2 median {np.median(errs):.2e} max {np.max(errs):.2e} "
                                  f"({len(errs)} traces)", flush=True)
            allo, allr = np.concatenate(allo), np.concatenate(allr)
            print(f"  all traces (each component by its peak): rel L2 {rel(allo, allr):.2e}, "
                  f"amp {np.linalg.norm(allo) / np.linalg.norm(allr):.5f}", flush=True)

            amp = 1000
            panels = list(ref)
            fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.5),
                                     constrained_layout=True)
            for k, c in enumerate(panels):
                ax = axes[k]
                s = 0.5 / peak[c]
                rows = np.flatnonzero(fluid) if c == "P" else np.arange(n)
                for j, i in enumerate(rows):
                    ax.plot(t_ref[keep], j + s * ref[c][keep, i], "k", lw=1.0)
                    ax.plot(t_ref[keep], j + s * ours[c][keep, i], "r--", lw=0.7)
                    ax.plot(t_ref[keep], j + amp * s * (ours[c][keep, i] - ref[c][keep, i]),
                            color="C0", lw=0.6)
                ax.set_yticks(range(len(rows)), [f"{groups[i]} {i + 1}" for i in rows],
                              fontsize=7)
                ax.set_title(f"({'abcd'[k]}) {'PRE' if c == 'P' else 'BX' + c}", fontsize=9)
                ax.set_xlabel("time (s)")
            fig.suptitle(f"{case}, {eq.__name__}: SPECFEM3D black, SEM red, {amp}x diff blue",
                         fontsize=10)
            fig.savefig(args.out / f"sem_vs_specfem3d_{case}_{eq.__name__}.png", dpi=220)
            plt.close(fig)
            print("wrote", args.out / f"sem_vs_specfem3d_{case}_{eq.__name__}.png", flush=True)


if __name__ == "__main__":
    main()

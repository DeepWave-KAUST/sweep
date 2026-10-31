"""SPECFEM2D input directories for ``run_compare_aniso.py``: one homogeneous anisotropic
medium (SPECFEM2D's material type 2, ``c11 c13 c15 c33 c35 c55``) in a 3000 m x 3000 m box of
10 m elements with all four sides free -- SPECFEM2D's C-PML and Stacey take the isotropic
moduli inside an anisotropic element, so no absorbing layer: both codes then solve the same
problem, the free-edge echoes included. A Ricker of f0 = 15 Hz at the centre (inside an
element, off its edges), 24 receivers on a 500 m ring around it and 5 at 10 m below the top
edge, dt = 2e-4 s, 6000 steps.

Cases: a vertical point force (``*_force``) and an explosion (``*_explosion``, Mxx = Mzz) in
a transversely isotropic shale -- Thomsen (1986) epsilon = 0.25, delta = 0.10 about its
symmetry axis -- with the axis vertical (``vti_*``) and tilted by 30 degrees (``tti30_*``,
sweep's tilt: sweep.sem.thomsen_to_tti_cij, z down). The cij go to SPECFEM2D in its z-up axes,
where c15 and c35 change sign; ``DATA/sweep_medium.json`` keeps the parameters they came from.

Run ``xmeshfem2D`` and then ``xspecfem2D`` in each ``<out>/<case>``.

    python make_specfem_aniso_cases.py <specfem2d>/EXAMPLES <out>
"""
import json
import math
import re
import sys
from pathlib import Path

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

L, H, DT, NSTEP, F0, FACTOR = 3000.0, 10.0, 2e-4, 6000, 15.0, 1e10
XS, ZS = 1505.0, 1495.0                        # z up from the bottom; an element's centre
RING = [(XS + 500.0 * math.cos(math.radians(a)), ZS + 500.0 * math.sin(math.radians(a)))
        for a in range(0, 360, 15)]
SURFACE = [(XS + d, L - 10.0) for d in (0.0, 250.0, 500.0, 750.0, 1000.0)]
SHALE = dict(rho=2400.0, vp=3000.0, vs=1500.0, epsilon=0.25, delta=0.10)
CASES = {"vti_force": (dict(SHALE, theta_deg=0.0), "force"),
         "vti_explosion": (dict(SHALE, theta_deg=0.0), "explosion"),
         "tti30_force": (dict(SHALE, theta_deg=30.0), "force"),
         "tti30_explosion": (dict(SHALE, theta_deg=30.0), "explosion")}


def cij_z_down(rho, vp, vs, epsilon, delta, theta_deg):
    """``(c11, c13, c15, c33, c35, c55)``: sweep.sem.thomsen_to_tti_cij's formulas."""
    c33, c55 = rho * vp * vp, rho * vs * vs
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c13 = math.sqrt(max(2.0 * delta * c33 * (c33 - c55) + (c33 - c55) ** 2, 0.0)) - c55
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    ct2, st2 = ct * ct, st * st
    s2c2, mix = st2 * ct2, 2.0 * c13 + 4.0 * c55
    a, b = c13 - c11 + 2.0 * c55, c13 - c33 + 2.0 * c55
    return (c11 * ct2 * ct2 + c33 * st2 * st2 + mix * s2c2,
            (c11 + c33 - 4.0 * c55) * s2c2 + c13 * (st2 * st2 + ct2 * ct2),
            a * ct2 * ct * st - b * ct * st2 * st,
            c11 * st2 * st2 + c33 * ct2 * ct2 + mix * s2c2,
            a * st2 * st * ct - b * st * ct2 * ct,
            (c11 + c33 - 2.0 * c13) * s2c2 + c55 * (ct2 - st2) ** 2)


def setp(text, key, value):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*)(\S+)", re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def dble(x):
    return f"{x:.15e}".replace("e", "d")


base = ex / "anisotropic_isotropic_model" / "DATA"
par0, src0 = (base / "Par_file").read_text(), (base / "SOURCE").read_text()
n = int(L / H)
for k, v in (("NPROC", 1), ("NSTEP", NSTEP), ("DT", dble(DT)), ("NGNOD", 4), ("seismotype", 2),
             ("USER_T0", "0.d0"), ("NTSTEP_BETWEEN_OUTPUT_SEISMOS", NSTEP),
             ("NTSTEP_BETWEEN_OUTPUT_INFO", 1000), ("use_existing_STATIONS", ".true."),
             ("PML_BOUNDARY_CONDITIONS", ".false."), ("STACEY_ABSORBING_CONDITIONS", ".false."),
             ("absorbbottom", ".false."), ("absorbright", ".false."), ("absorbtop", ".false."),
             ("absorbleft", ".false."), ("OUTPUT_ENERGY", ".false."),
             ("output_color_image", ".false."), ("NTSTEP_BETWEEN_OUTPUT_IMAGES", 100000),
             ("xmax", dble(L)), ("nx", n), ("nbmodels", 1), ("nbregions", 1)):
    par0 = setp(par0, k, v)

for name, (medium, source) in CASES.items():
    c11, c13, c15, c33, c35, c55 = cij_z_down(**medium)
    cij = (c11, c13, -c15, c33, -c35, c55)           # SPECFEM2D's z up
    mat = f"1 2 {dble(medium['rho'])} " + " ".join(dble(c) for c in cij) + " 0 0 0 0 9999 9999"
    par = re.sub(r"^1 1 .*\n^2 2 .*\n^3 1 .*$", mat, par0, count=1, flags=re.M)
    par = re.sub(r"^1 50 1   25 1\n^1 50 26  30 2\n^1 50 31  50 3$", f"1 {n} 1 {n} 1", par,
                 count=1, flags=re.M)
    assert mat in par and f"1 {n} 1 {n} 1" in par, name
    src = src0
    for k, v in (("xs", dble(XS)), ("zs", dble(ZS)), ("source_type", 1 if source == "force" else 2),
                 ("time_function_type", 1), ("f0", F0), ("tshift", "0.d0"), ("anglesource", "0.d0"),
                 ("Mxx", "1.d0"), ("Mzz", "1.d0"), ("Mxz", "0.d0"), ("factor", dble(FACTOR))):
        src = setp(src, k, v)
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "SOURCE").write_text(src)
    (d / "interfaces.dat").write_text(f"2\n2\n 0 0\n {L:.0f} 0\n2\n 0 {L:.0f}\n {L:.0f} {L:.0f}\n{n}\n")
    (d / "STATIONS").write_text("".join(f"S{i + 1:04d} AA {x:.6f} {z:.6f} 0.0 0.0\n"
                                        for i, (x, z) in enumerate(RING + SURFACE)))
    (d / "sweep_medium.json").write_text(json.dumps(medium, indent=1) + "\n")
    print("wrote", d)

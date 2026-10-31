"""SPECFEM2D input directories for ``run_compare_box.py``: one homogeneous elastic medium in a
3000 m x 3000 m box of 10 m elements with all four sides free -- no absorbing layer, so that
both codes solve the same problem, the free-edge echoes included (SPECFEM2D's C-PML and Stacey
conditions also take the isotropic moduli inside an anisotropic element). A Ricker of
f0 = 15 Hz inside an element, 24 receivers on a 500 m ring around it and 5 at 10 m below the
top edge, dt = 2e-4 s, 6000 steps.

Cases, a vertical point force (``*_force``) or an explosion (``*_explosion``, Mxx = Mzz):

* ``iso_*``: an isotropic rock (vp 3000, vs 1500, rho 2400);
* ``vti_*`` and ``tti30_*``: a transversely isotropic shale -- Thomsen (1986) epsilon = 0.25,
  delta = 0.10 about its symmetry axis -- with the axis vertical and tilted by 30 degrees
  (sweep's tilt: sweep.sem.thomsen_to_tti_cij, z down), SPECFEM2D's material type 2, whose
  c15 and c35 change sign in its z-up axes;
* ``visco_explosion``: the rock with Qkappa = Qmu = 30 (our Qp = Qs = 30), SPECFEM2D's
  ``ATTENUATION_VISCOELASTIC`` with three mechanisms and the velocities read at 15 Hz;
* ``water_pressure``: water (vp 1500, rho 1020) filling the box, a pressure source (SPECFEM2D's
  force in an acoustic element); every edge of a fluid element is pressure-free there;
* ``sea_pressure``: 600 m of that water over the rock, the pressure source 105 m deep,
  hydrophones 50 m deep and geophones 50 m below the sea floor;
* ``hill_explosion``: the rock under a Gaussian hill (100 m high, sigma 200 m) on the top
  edge, Quad9 elements (``NGNOD = 9``) in 310 rows graded linearly between the flat bottom and
  the hill -- PropSEM's mesh of a 3100 m model with that topography --, the hill given at
  every control-node abscissa so that SPECFEM2D's spline passes through the same surface;
  the explosion at the centre of an element about 145 m below the summit, the line
  receivers 10 m below the hill.

``DATA/sweep_medium.json`` keeps the parameters the material lines and the surface came from,
and the receiver groups.
Run ``xmeshfem2D`` and then ``xspecfem2D`` in each ``<out>/<case>``.

    python make_specfem_box_cases.py <specfem2d>/EXAMPLES <out>
"""
import json
import math
import re
import sys
from pathlib import Path

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

L, H, DT, NSTEP, F0, FACTOR = 3000.0, 10.0, 2e-4, 6000, 15.0, 1e10
XS, ZS = 1505.0, 1495.0                        # z up from the bottom; an element's centre
ROCK = dict(kind="isotropic", rho=2400.0, vp=3000.0, vs=1500.0)
WATER = dict(kind="acoustic", rho=1020.0, vp=1500.0)
SEA = dict(kind="coupled", water=WATER, rock=ROCK, depth=600.0)
VISCO = dict(ROCK, kind="visco", q=30.0, n_sls=3, f_ref=F0)
SHALE = dict(kind="ti", rho=2400.0, vp=3000.0, vs=1500.0, epsilon=0.25, delta=0.10)
HILL = dict(height=100.0, x0=1500.0, sigma=200.0)
CASES = {"iso_force": (ROCK, "force", None),
         "iso_explosion": (ROCK, "explosion", None),
         "vti_force": (dict(SHALE, theta_deg=0.0), "force", None),
         "vti_explosion": (dict(SHALE, theta_deg=0.0), "explosion", None),
         "tti30_force": (dict(SHALE, theta_deg=30.0), "force", None),
         "tti30_explosion": (dict(SHALE, theta_deg=30.0), "explosion", None),
         "hill_explosion": (ROCK, "explosion", HILL),
         "visco_explosion": (VISCO, "explosion", None),
         "water_pressure": (WATER, "pressure", None),
         "sea_pressure": (SEA, "pressure", None)}


def surface(x, hill):
    """The top edge, z up: flat at L, or L plus the hill."""
    if hill is None:
        return L
    return L + hill["height"] * math.exp(-((x - hill["x0"]) / hill["sigma"]) ** 2)


def cij_z_down(rho, vp, vs, epsilon, delta, theta_deg, **_):
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
for k, v in (("NPROC", 1), ("NSTEP", NSTEP), ("DT", dble(DT)), ("seismotype", 2),
             ("USER_T0", "0.d0"), ("NTSTEP_BETWEEN_OUTPUT_SEISMOS", NSTEP),
             ("NTSTEP_BETWEEN_OUTPUT_INFO", 1000), ("use_existing_STATIONS", ".true."),
             ("PML_BOUNDARY_CONDITIONS", ".false."), ("STACEY_ABSORBING_CONDITIONS", ".false."),
             ("absorbbottom", ".false."), ("absorbright", ".false."), ("absorbtop", ".false."),
             ("absorbleft", ".false."), ("OUTPUT_ENERGY", ".false."),
             ("output_color_image", ".false."), ("NTSTEP_BETWEEN_OUTPUT_IMAGES", 100000),
             ("xmax", dble(L)), ("nx", n), ("nbmodels", 1), ("nbregions", 1)):
    par0 = setp(par0, k, v)

def isotropic(i, m):
    q = dble(m["q"]) if "q" in m else "9999"
    return (f"{i} 1 {dble(m['rho'])} {dble(m['vp'])} {dble(m.get('vs', 0.0))} 0 0 "
            f"{q} {q} 0 0 0 0 0 0")


for name, (medium, source, hill) in CASES.items():
    nz = n if hill is None else int(round((L + hill["height"]) / H))   # element rows
    nw = int(round(medium["depth"] / H)) if medium["kind"] == "coupled" else 0
    if medium["kind"] in ("isotropic", "acoustic", "visco"):
        mats, regions = [isotropic(1, medium)], [f"1 {n} 1 {nz} 1"]
    elif medium["kind"] == "coupled":            # rows count up from the bottom
        mats = [isotropic(1, medium["rock"]), isotropic(2, medium["water"])]
        regions = [f"1 {n} 1 {nz - nw} 1", f"1 {n} {nz - nw + 1} {nz} 2"]
    else:
        c11, c13, c15, c33, c35, c55 = cij_z_down(**medium)
        cij = (c11, c13, -c15, c33, -c35, c55)           # SPECFEM2D's z up
        mats = [f"1 2 {dble(medium['rho'])} " + " ".join(dble(c) for c in cij)
                + " 0 0 0 0 9999 9999"]
        regions = [f"1 {n} 1 {nz} 1"]
    par = setp(par0, "NGNOD", 4 if hill is None else 9)
    par = setp(par, "nbmodels", len(mats))
    par = setp(par, "nbregions", len(regions))
    if source == "pressure":
        par = setp(par, "seismotype", "2,4")
    if medium["kind"] == "visco":
        for k, v in (("ATTENUATION_VISCOELASTIC", ".true."), ("N_SLS", medium["n_sls"]),
                     ("ATTENUATION_f0_REFERENCE", dble(medium["f_ref"])),
                     ("READ_VELOCITIES_AT_f0", ".true."), ("USE_SOLVOPT", ".false.")):
            par = setp(par, k, v)
    par = re.sub(r"^1 1 .*\n^2 2 .*\n^3 1 .*$", "\n".join(mats), par, count=1, flags=re.M)
    par = re.sub(r"^1 50 1   25 1\n^1 50 26  30 2\n^1 50 31  50 3$", "\n".join(regions), par,
                 count=1, flags=re.M)
    assert all(m in par for m in mats + regions), name
    # a flat box's source sits at an element's centre; under the hill, the centre of row
    # nz - 15 of the column there (rows graded linearly up to the surface); in the sea, the
    # centre of the eleventh row of water
    zs = ZS if hill is None else (nz - 15 + 0.5) * surface(XS, hill) / nz
    if nw:
        zs = L - 105.0
    src = src0
    for k, v in (("xs", dble(XS)), ("zs", dble(zs)), ("source_type", 2 if source == "explosion" else 1),
                 ("time_function_type", 1), ("f0", F0), ("tshift", "0.d0"), ("anglesource", "0.d0"),
                 ("Mxx", "1.d0"), ("Mzz", "1.d0"), ("Mxz", "0.d0"), ("factor", dble(FACTOR))):
        src = setp(src, k, v)
    ring = [(XS + 500.0 * math.cos(math.radians(a)), zs + 500.0 * math.sin(math.radians(a)))
            for a in range(0, 360, 15)]
    if hill is not None:                 # the upper half of the ring would leave the box
        ring = [(x, z) for x, z in ring if z < surface(x, hill) - 10.0]
    line = [(XS + d, surface(XS + d, hill) - 10.0) for d in (0.0, 250.0, 500.0, 750.0, 1000.0)]
    groups = [["ring", len(ring)], ["line 10 m below the top", len(line)]]
    if nw:                               # hydrophones and geophones
        ring = [(XS + d, L - 50.0) for d in range(0, 1001, 125)]
        line = [(XS + d, L - medium["depth"] - 50.0) for d in range(0, 1001, 125)]
        groups = [["hydrophones 50 m deep", len(ring)],
                  ["geophones 50 m below the sea floor", len(line)]]
    # the top at every control-node abscissa (half an element apart on Quad9)
    xs_top = [L * i / (2 * n) for i in range(2 * n + 1)] if hill is not None else [0.0, L]
    top = "".join(f" {x:.10f} {surface(x, hill):.10f}\n" for x in xs_top)
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "SOURCE").write_text(src)
    if nw:
        (d / "interfaces.dat").write_text(
            f"3\n2\n 0 0\n {L:.0f} 0\n2\n 0 {L - medium['depth']:.0f}\n {L:.0f} "
            f"{L - medium['depth']:.0f}\n2\n 0 {L:.0f}\n {L:.0f} {L:.0f}\n{nz - nw}\n{nw}\n")
    else:
        (d / "interfaces.dat").write_text(
            f"2\n2\n 0 0\n {L:.0f} 0\n{len(xs_top)}\n{top}{nz}\n")
    (d / "STATIONS").write_text("".join(f"S{i + 1:04d} AA {x:.6f} {z:.6f} 0.0 0.0\n"
                                        for i, (x, z) in enumerate(ring + line)))
    (d / "sweep_medium.json").write_text(json.dumps(dict(medium, hill=hill, groups=groups),
                                                    indent=1) + "\n")
    print("wrote", d)

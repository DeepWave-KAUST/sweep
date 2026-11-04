"""SPECFEM2D input directories for ``run_compare_planewave.py``: plane waves by
``initialfield``. SPECFEM2D starts the run with the plane wave in the mesh and feeds it through
its Stacey edges (Bielak's conditions, on the left, right and bottom; the top free); sweep
injects it through a band of elements with the domain reduction method and absorbs the rest in
its C-PML.

Units without dimension, as SPECFEM2D's own plane-wave examples: the rock vp 2, vs 1, rho 1, a
Ricker of f0 = 1, elements of 0.1 (NGLL 5), dt = 2^-9 -- a power of two, which SPECFEM2D's
frequency-domain plane wave past the critical angle (``paco_beyond_critical``) asks for, with
vs = 1. The plane wave's reference point is on the surface.

* ``pw_p30``, ``pw_sv20``: a P wave at 30 degrees and an SV wave at 20 degrees in the flat
  half-space, 8 x 4 (80 x 40 elements), the reference at x = 2;
* ``pw_sv40``: an SV wave at 40 degrees, past the critical angle (30 degrees): the reflected P
  wave evanescent, SPECFEM2D's frequency-domain field;
* ``pw_rayleigh``: a Rayleigh wave along the surface, from x = 1, SPECFEM2D's frequency-domain
  field too;
* ``pw_sv20_basin``: the SV wave at 20 degrees under a soft basin (vp 1.2, vs 0.5, rho 0.8;
  2 wide and 0.4 deep, whole elements) in a 24 x 12 domain, large enough that the Stacey edges'
  echoes of the basin's scattered waves come back after the record; the reference at x = 10.
  ``USER_T0 = 2``: at t = 0, when SPECFEM2D puts the background's plane wave in the whole mesh,
  basin included, its pulse is still deep below the basin;
* ``pw_sv20_big``: that case without the basin -- the plane wave alone in the large domain,
  whose Stacey edges are far from the receivers;
* ``pw_sh30``: an SH wave at 30 degrees (``P_SV = .false.``) in the flat half-space;
* ``pw_p20_whole``: a P wave at 20 degrees in a whole space (``source_type = 4``, no
  reflected waves): Stacey and Bielak's conditions on all four sides, the top too;
* ``pw_p30_big``, ``pw_sh30_big``: those flat cases in the large domain, the reference at
  x = 10. Not the frequency-domain ones: SPECFEM2D computes that field point by point, at
  32768 frequencies in the large domain -- under 500 of its 460000 points in ten minutes.

Our box (``sweep_box``: x0, x1 and its depth) is the domain in the small cases and its piece
from x = 7 to 17, 5 deep, in the large ones.

Receivers: 13 on the surface, 7 on a line 1 below it and 4 more (a vertical line in the flat
cases, a line inside the basin). ``DATA/sweep_planewave.json`` keeps the geometry. Run
``xmeshfem2D`` and then ``xspecfem2D`` in each ``<out>/<case>``.

    python make_specfem_planewave_cases.py <specfem2d>/EXAMPLES <out>
"""
import json
import math
import re
import sys
from pathlib import Path

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

H, DT, F0 = 0.1, 2.0 ** -9, 1.0
ROCK = dict(rho=1.0, vp=2.0, vs=1.0)
BASIN = dict(rho=0.8, vp=1.2, vs=0.5, x=(11.0, 13.0), depth=0.4)
FLAT = dict(W=8.0, D=4.0)
BIG = dict(W=24.0, D=12.0)
# name: (source_type, angle, x of the reference point, domain, basin, NSTEP, USER_T0)
CASES = {"pw_p30": (1, 30.0, 2.0, FLAT, None, 3072, 0.0),
         "pw_sv20": (2, 20.0, 2.0, FLAT, None, 3072, 0.0),
         "pw_sv40": (2, 40.0, 2.0, FLAT, None, 3072, 0.0),
         "pw_rayleigh": (3, 0.0, 1.0, FLAT, None, 3072, 0.0),
         "pw_sv20_basin": (2, 20.0, 10.0, BIG, BASIN, 4096, 2.0),
         "pw_sv20_big": (2, 20.0, 10.0, BIG, None, 4096, 2.0),
         "pw_sh30": (2, 30.0, 2.0, FLAT, None, 3072, 0.0),
         "pw_p20_whole": (4, 20.0, 2.0, FLAT, None, 3072, 0.0),
         "pw_p30_big": (1, 30.0, 10.0, BIG, None, 4096, 2.0),
         "pw_sh30_big": (2, 30.0, 10.0, BIG, None, 4096, 2.0)}
WAVES = {1: "P", 2: "SV", 3: "Rayleigh", 4: "P"}


def setp(text, key, value):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*)(\S+)", re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def dble(x):
    return f"{x:.15e}".replace("e", "d")


def material(i, m):
    return (f"{i} 1 {dble(m['rho'])} {dble(m['vp'])} {dble(m['vs'])} 0 0 9999 9999 "
            "0 0 0 0 0 0")


base = ex / "initial_plane_wave_with_free_surface" / "DATA"
par0, src0 = (base / "Par_file").read_text(), (base / "SOURCE").read_text()
for k, v in (("NPROC", 1), ("DT", dble(DT)), ("seismotype", 1),
             ("NTSTEP_BETWEEN_OUTPUT_INFO", 1000), ("use_existing_STATIONS", ".true."),
             ("NGNOD", 4), ("OUTPUT_ENERGY", ".false."), ("output_color_image", ".false."),
             ("output_postscript_snapshot", ".false."), ("NTSTEP_BETWEEN_OUTPUT_IMAGES", 100000),
             ("initialfield", ".true."), ("add_Bielak_conditions_bottom", ".true."),
             ("add_Bielak_conditions_right", ".true."), ("add_Bielak_conditions_left", ".true."),
             ("add_Bielak_conditions_top", ".false."), ("STACEY_ABSORBING_CONDITIONS", ".true."),
             ("PML_BOUNDARY_CONDITIONS", ".false."), ("absorbbottom", ".true."),
             ("absorbright", ".true."), ("absorbleft", ".true."), ("absorbtop", ".false."),
             ("ATTENUATION_VISCOELASTIC", ".false.")):
    par0 = setp(par0, k, v)

for name, (stype, angle, xs, dom, basin, nstep, user_t0) in CASES.items():
    W, D = dom["W"], dom["D"]
    nx, nz = int(round(W / H)), int(round(D / H))
    par = setp(par0, "xmax", dble(W))
    par = setp(par, "nx", nx)
    par = setp(par, "NSTEP", nstep)
    par = setp(par, "NTSTEP_BETWEEN_OUTPUT_SEISMOS", nstep)
    par = setp(par, "USER_T0", dble(user_t0))
    sh, whole = name.startswith("pw_sh30"), stype == 4
    par = setp(par, "P_SV", ".false." if sh else ".true.")
    par = setp(par, "absorbtop", ".true." if whole else ".false.")
    par = setp(par, "add_Bielak_conditions_top", ".true." if whole else ".false.")
    mats, regions = [material(1, ROCK)], [f"1 {nx} 1 {nz} 1"]
    if basin is not None:                      # element blocks, z up: the basin's rows on top
        i0, i1 = int(round(basin["x"][0] / H)) + 1, int(round(basin["x"][1] / H))
        k0 = nz - int(round(basin["depth"] / H)) + 1
        mats.append(material(2, basin))
        regions = [f"1 {nx} 1 {k0 - 1} 1", f"1 {i0 - 1} {k0} {nz} 1",
                   f"{i1 + 1} {nx} {k0} {nz} 1", f"{i0} {i1} {k0} {nz} 2"]
    par = setp(par, "nbmodels", len(mats))
    par = setp(par, "nbregions", len(regions))
    par = re.sub(r"^1 1 2800\.d0 6500\.d0 3700d0 .*$", "\n".join(mats), par, count=1, flags=re.M)
    par = re.sub(r"^1 120  1 50 1$", "\n".join(regions), par, count=1, flags=re.M)
    assert all(m in par for m in mats + regions), name
    src = src0
    for k, v in (("xs", dble(xs)), ("zs", dble(D)), ("source_type", stype),
                 ("time_function_type", 1), ("f0", dble(F0)), ("tshift", "0.d0"),
                 ("anglesource", dble(angle)), ("factor", "1.d0")):
        src = setp(src, k, v)
    big = dom is BIG
    if not big:
        x0, our_box = 1.0, (0.0, W, D)
        side = [(4.0, D - d) for d in (0.5, 1.5, 2.5, 3.5)]
    else:
        x0, our_box = 9.0, (7.0, 17.0, 5.0)
        side = [(11.25 + 0.5 * i, D - 0.2) for i in range(4)]          # inside the basin
    surf = [(x0 + 0.5 * i, D) for i in range(13)]
    deep = [(x0 + i, D - 1.0) for i in range(7)]
    groups = [["surface", len(surf)], ["line 1 below the surface", len(deep)],
              ["vertical line" if not big else "line 0.2 deep (in the basin)", len(side)]]
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "SOURCE").write_text(src)
    (d / "interfaces.dat").write_text(f"2\n2\n 0 0\n {W:.6f} 0\n2\n 0 {D:.6f}\n {W:.6f} "
                                      f"{D:.6f}\n{nz}\n")
    (d / "STATIONS").write_text("".join(f"S{i + 1:04d} AA {x:.6f} {z:.6f} 0.0 0.0\n"
                                        for i, (x, z) in enumerate(surf + deep + side)))
    (d / "sweep_planewave.json").write_text(json.dumps(dict(
        ROCK, wave="SH" if sh else WAVES[stype], angle=angle, reference=[xs, D], box=[W, D],
        h=H, dt=DT, nstep=nstep, f0=F0, user_t0=user_t0, basin=basin, sweep_box=list(our_box),
        whole_space=whole, groups=groups), indent=1) + "\n")
    print("wrote", d)

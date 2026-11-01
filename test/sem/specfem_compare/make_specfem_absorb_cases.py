"""SPECFEM2D input directories for ``run_compare_absorb.py``: how much the absorbing boundaries
reflect. An elastic half-space (vp 3000, vs 1500, rho 2400; the top free) of 10 m elements, a
vertical point force of a 15 Hz Ricker 35 m below the top, dt = 2e-4 s, 6000 steps:

* ``absorb_stacey``: a 2200 m x 1100 m box, SPECFEM2D's Stacey condition on the left, right and
  bottom edges;
* ``absorb_pml``: the same box with its C-PML in the outer 10 elements of those three sides
  (its default parameters);
* ``absorb_ref``: a 4800 m x 2400 m box with free edges, far enough that no echo of its edges
  reaches a receiver within the record -- the response the two others should give;
* ``absorb_visco_pml`` and ``absorb_visco_ref``: the C-PML box and the reference with the rock
  attenuating, Qkappa = 30 and Qmu = 20 (``ATTENUATION_VISCOELASTIC``, three mechanisms at
  15 Hz, the velocities read as the unrelaxed ones): SPECFEM2D's visco-elastic C-PML.

The box of the first two is the reference's interior, 2000 m x 1000 m, with the 100 m layer
outside it, so both run on the same 220 x 110 elements; the reference holds that interior
1400 m from its left and bottom edges. Receivers in the interior: 20 on a line 10 m below the
top, 10 on a line 50 m above the layer and 9 on a vertical line 50 m inside the left layer.
``DATA/sweep_absorb.json`` keeps the geometry. Run ``xmeshfem2D`` and then ``xspecfem2D`` in
each ``<out>/<case>``.

    python make_specfem_absorb_cases.py <specfem2d>/EXAMPLES <out>
"""
import json
import re
import sys
from pathlib import Path

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

H, DT, NSTEP, F0, FACTOR = 10.0, 2e-4, 6000, 15.0, 1e10
ROCK = dict(rho=2400.0, vp=3000.0, vs=1500.0)
W, D, LAYER, PAD = 2000.0, 1000.0, 100.0, 1400.0         # interior, layer, reference margin
XS, ZS = LAYER + 1005.0, LAYER + D - 35.0                 # an element's centre, z up
VISCO = dict(qkappa=30.0, qmu=20.0, n_sls=3, f_ref=F0)
CASES = {"absorb_stacey": ("stacey", 0.0, None), "absorb_pml": ("pml", 0.0, None),
         "absorb_ref": ("free", PAD - LAYER, None),
         "absorb_visco_pml": ("pml", 0.0, VISCO), "absorb_visco_ref": ("free", PAD - LAYER, VISCO)}


def setp(text, key, value):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*)(\S+)", re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def dble(x):
    return f"{x:.15e}".replace("e", "d")


base = ex / "anisotropic_isotropic_model" / "DATA"
par0, src0 = (base / "Par_file").read_text(), (base / "SOURCE").read_text()
for k, v in (("NPROC", 1), ("NSTEP", NSTEP), ("DT", dble(DT)), ("seismotype", 2),
             ("USER_T0", "0.d0"), ("NTSTEP_BETWEEN_OUTPUT_SEISMOS", NSTEP),
             ("NTSTEP_BETWEEN_OUTPUT_INFO", 1000), ("use_existing_STATIONS", ".true."),
             ("NGNOD", 4), ("OUTPUT_ENERGY", ".false."), ("output_color_image", ".false."),
             ("NTSTEP_BETWEEN_OUTPUT_IMAGES", 100000), ("nbmodels", 1), ("nbregions", 1),
             ("NELEM_PML_THICKNESS", int(LAYER / H))):
    par0 = setp(par0, k, v)

def material(q):
    qk, qmu = ("9999", "9999") if q is None else (dble(q["qkappa"]), dble(q["qmu"]))
    return (f"1 1 {dble(ROCK['rho'])} {dble(ROCK['vp'])} {dble(ROCK['vs'])} 0 0 {qk} {qmu} "
            "0 0 0 0 0 0")


# receivers in the interior, z up, in the box of the first two cases
line = [(LAYER + 50.0 + 100.0 * i, LAYER + D - 10.0) for i in range(20)]
deep = [(LAYER + 50.0 + 200.0 * i, LAYER + 50.0) for i in range(10)]
side = [(LAYER + 50.0, LAYER + 100.0 * k) for k in range(1, 10)]
groups = [["line 10 m below the top", len(line)], ["line 50 m above the layer", len(deep)],
          ["vertical line 50 m inside the left layer", len(side)]]

for name, (kind, shift, q) in CASES.items():
    mat = material(q)
    L, Z = (W + 2.0 * LAYER, D + LAYER) if kind != "free" else (W + 2.0 * PAD, D + PAD)
    n, nz = int(round(L / H)), int(round(Z / H))
    par = setp(par0, "xmax", dble(L))
    par = setp(par, "nx", n)
    absorbing = ".true." if kind != "free" else ".false."
    for k in ("absorbbottom", "absorbright", "absorbleft"):
        par = setp(par, k, absorbing)
    par = setp(par, "absorbtop", ".false.")
    par = setp(par, "STACEY_ABSORBING_CONDITIONS", ".true." if kind == "stacey" else ".false.")
    par = setp(par, "PML_BOUNDARY_CONDITIONS", ".true." if kind == "pml" else ".false.")
    if q is not None:
        for k, v in (("ATTENUATION_VISCOELASTIC", ".true."), ("N_SLS", q["n_sls"]),
                     ("ATTENUATION_f0_REFERENCE", dble(q["f_ref"])),
                     ("READ_VELOCITIES_AT_f0", ".false."), ("USE_SOLVOPT", ".false.")):
            par = setp(par, k, v)
    par = re.sub(r"^1 1 .*\n^2 2 .*\n^3 1 .*$", mat, par, count=1, flags=re.M)
    par = re.sub(r"^1 50 1   25 1\n^1 50 26  30 2\n^1 50 31  50 3$", f"1 {n} 1 {nz} 1", par,
                 count=1, flags=re.M)
    assert mat in par and f"1 {n} 1 {nz} 1" in par, name
    src = src0
    for k, v in (("xs", dble(XS + shift)), ("zs", dble(ZS + shift)), ("source_type", 1),
                 ("time_function_type", 1), ("f0", F0), ("tshift", "0.d0"),
                 ("anglesource", "0.d0"), ("factor", dble(FACTOR))):
        src = setp(src, k, v)
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "SOURCE").write_text(src)
    (d / "interfaces.dat").write_text(f"2\n2\n 0 0\n {L:.0f} 0\n2\n 0 {Z:.0f}\n {L:.0f} {Z:.0f}\n"
                                      f"{nz}\n")
    (d / "STATIONS").write_text("".join(f"S{i + 1:04d} AA {x + shift:.6f} {z + shift:.6f} 0.0 0.0\n"
                                        for i, (x, z) in enumerate(line + deep + side)))
    (d / "sweep_absorb.json").write_text(json.dumps(dict(
        ROCK, kind=kind, box=[L, Z], shift=shift, interior=[W, D], layer=LAYER,
        source=[XS + shift, ZS + shift], f0=F0, groups=groups, visco=q), indent=1) + "\n")
    print("wrote", d)

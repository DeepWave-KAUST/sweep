"""SPECFEM2D input directories for ``run_compare_poro_features.py`` and
``run_compare_poro_coupling.py``, built from SPECFEM2D's own examples:

* on a 2000 m x 1600 m box of 5 m Quad9 elements (free top, Stacey elsewhere, echoes after
  the 0.25 s record), from ``poroelastic_semi_infinite_homogeneous``: ``force_z`` (a vertical
  point force), ``aniso_k`` (an anisotropic permeability), ``visc`` and ``visc_atten`` (the
  viscous drag, then with ``ATTENUATION_PORO_FLUID_PART``) -- the source 600 m deep, two of
  the four receivers on its grid lines --, ``pe_layers`` (porous rock over a dry elastic one)
  and ``ape_layers`` (water, porous rock, elastic rock);
* ``pa_example``: the ``poroelastic_acoustic`` example as it is, serial and without images,
  and ``pa_example_fine`` on elements half as large.

Run ``xmeshfem2D`` and then ``xspecfem2D`` in each ``<out>/<case>``.

    python make_specfem_poro_cases.py <specfem2d>/EXAMPLES <out>
"""
import re
import shutil
import sys
from pathlib import Path

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

LX, LZ, H = 2000.0, 1600.0, 5.0
XS, ZS = 1000.0, 1000.0                      # z up from the bottom: 600 m deep
RECS = [(1100.0, 1100.0), (1100.0, 1000.0), (1000.0, 1100.0), (1150.0, 950.0)]
ROCK = dict(rho_s=2650.0, rho_f=880.0, phi=0.1, tort=2.0, k_s=12.2e9, k_f=1.985e9,
            k_fr=9.6e9, mu_fr=5.1e9)
SHALE = "1 2500.d0 4000.d0 2300.d0 0 0 9999 9999 0 0 0 0 0 0"      # rho vp vs
WATER = "1 1020.d0 1500.d0 0.d0 0 0 9999 9999 0 0 0 0 0 0"

# name: (source_type, angle, eta, (kxx, kxz, kzz) in SPECFEM's z-up axes, (Q0, f0) or None)
FEATURES = {
    "force_z": (1, 0.0, 0.0, (1e-9, 0.0, 1e-9), None),
    "aniso_k": (2, None, 1e-3, (2e-9, 5e-10, 5e-10), None),
    "visc": (2, None, 1e-3, (1e-9, 0.0, 1e-9), None),
    "visc_atten": (2, None, 1e-3, (1e-9, 0.0, 1e-9), (5.0, 30.0)),
}
# name: (layers top-down as (material line, thickness in m), source type, source (x, z up),
#        receivers (x, z up), steps of 1e-4 s); the porous rock with eta = 1e-3
LAYERS = {
    "pe_layers": (["rock", 800.0, SHALE, 800.0], 2, (1000.0, 1000.0),
                  [(1100.0, 1100.0), (1100.0, 850.0), (1050.0, 750.0), (1150.0, 650.0)], 2500),
    "ape_layers": ([WATER, 300.0, "rock", 500.0, SHALE, 800.0], 1, (1000.0, 1450.0),
                   [(1100.0, 1400.0), (1100.0, 1250.0), (1050.0, 1000.0), (1150.0, 700.0)],
                   5000),
}


def setp(text, key, value):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*)(\S+)", re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def quiet(par):
    for k, v in (("NPROC", 1), ("output_color_image", ".false."),
                 ("NTSTEP_BETWEEN_OUTPUT_IMAGES", 100000), ("seismotype", 2)):
        par = setp(par, k, v)
    return par


def rock(eta, k=(1e-9, 0.0, 1e-9)):
    r = ROCK
    line = (f"3 {r['rho_s']}d0 {r['rho_f']}d0 {r['phi']}d0 {r['tort']} {k[0]:.6e} {k[1]:.6e} "
            f"{k[2]:.6e} {r['k_s']:.6e} {r['k_f']:.6e} {r['k_fr']:.6e} {eta:.6e} {r['mu_fr']:.6e} "
            f"9999999999")
    return line.replace("e+", "d+").replace("e-", "d-")


def write(name, par, src):
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "SOURCE").write_text(src)
    (d / "interfaces.dat").write_text(
        f"2\n2\n 0 0\n {LX:.0f} 0\n2\n 0 {LZ:.0f}\n {LX:.0f} {LZ:.0f}\n{int(LZ / H)}\n")
    print("wrote", d)


def box(par, recs):
    """The 2000 m x 1600 m box and one receiver set per receiver."""
    par = setp(par, "xmax", f"{LX}d0")
    par = setp(par, "nx", int(LX / H))
    block = "".join(f"nrec = 1\nxdeb = {x}\nzdeb = {z}\nxfin = {x}\nzfin = {z}\n"
                    f"record_at_surface_same_vertical = .false.\n" for x, z in recs)
    par = setp(par, "nreceiversets", len(recs))
    return re.sub(r"^nrec\s*=.*\n^xdeb\s*=.*\n^zdeb\s*=.*\n^xfin\s*=.*\n^zfin\s*=.*\n"
                  r"^record_at_surface_same_vertical\s*=.*\n", lambda m: block, par, count=1,
                  flags=re.M)


base = ex / "poroelastic_semi_infinite_homogeneous" / "DATA"
par0, src0 = quiet((base / "Par_file").read_text()), (base / "SOURCE").read_text()
nx, nz = int(LX / H), int(LZ / H)

for name, (stype, ang, eta, k, atten) in FEATURES.items():
    par = box(par0, RECS)
    par = setp(par, "ATTENUATION_PORO_FLUID_PART", ".true." if atten else ".false.")
    if atten:
        par = setp(par, "Q0_poroelastic", atten[0])
        par = setp(par, "freq0_poroelastic", atten[1])
    par = re.sub(r"^1 3 .*$", lambda m: "1 " + rock(eta, k), par, count=1, flags=re.M)
    par = re.sub(r"^1 200 1 200 1$", f"1 {nx} 1 {nz} 1", par, count=1, flags=re.M)
    src = setp(setp(setp(src0, "xs", XS), "zs", ZS), "source_type", stype)
    if stype == 1:
        src = setp(src, "anglesource", ang)
    write(name, par, src)

for name, (layers, stype, (xs, zs), recs, nstep) in LAYERS.items():
    mats = [rock(1e-3) if m == "rock" else m for m in layers[::2]]
    par = setp(box(par0, recs), "NSTEP", nstep)
    par = setp(par, "ATTENUATION_PORO_FLUID_PART", ".false.")
    par = setp(par, "nbmodels", len(mats))
    par = re.sub(r"^1 3 .*$", lambda m: "\n".join(f"{i + 1} {line}" for i, line in
                                                  enumerate(mats)), par, count=1, flags=re.M)
    regions, top = [], nz                    # element rows count up from the bottom
    for i, thick in enumerate(layers[1::2]):
        n = int(round(thick / H))
        regions.append(f"1 {nx} {top - n + 1} {top} {i + 1}")
        top -= n
    assert top == 0, name
    par = setp(par, "nbregions", len(mats))
    par = re.sub(r"^1 200 1 200 1$", lambda m: "\n".join(regions), par, count=1, flags=re.M)
    src = setp(setp(setp(src0, "xs", xs), "zs", zs), "source_type", stype)
    if stype == 1:
        src = setp(src, "anglesource", 0.0)
    write(name, par, src)

# the poroelastic_acoustic example, and on elements half as large
pa = ex / "poroelastic_acoustic" / "DATA"
for name, scale in (("pa_example", 1), ("pa_example_fine", 2)):
    d = out / name / "DATA"
    d.mkdir(parents=True, exist_ok=True)
    par = quiet((pa / "Par_file").read_text())
    itf = (pa / "interfaces.dat").read_text()
    if scale > 1:
        par = setp(par, "NSTEP", 2500 * scale)
        par = setp(par, "DT", f"{4.2 / scale}d-4")
        par = setp(par, "nx", 220 * scale)
        par = re.sub(r"^1 220 1 110 1$", f"1 {220 * scale} 1 {110 * scale} 1", par, count=1,
                     flags=re.M)
        par = re.sub(r"^1 220 111 220 2$", f"1 {220 * scale} {110 * scale + 1} {220 * scale} 2",
                     par, count=1, flags=re.M)
        itf = re.sub(r"^\s*110\s*$", str(110 * scale), itf, flags=re.M)
    (d / "Par_file").write_text(par)
    (d / "interfaces.dat").write_text(itf)
    shutil.copy(pa / "SOURCE", d / "SOURCE")
    print("wrote", d)

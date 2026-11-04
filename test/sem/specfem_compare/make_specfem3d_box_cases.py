"""SPECFEM3D input directories for ``run_compare_box3d.py``: homogeneous boxes free on every
side (no absorbing condition, the top the free surface), meshed by SPECFEM3D's internal mesher
into regular Hex8 elements, a vertical point force driven by an external source time function.

Units SI. A 320 m cube of 10 m elements (NGLL 5), dt = 0.25 ms, 1600 steps; vp 2000, vs 1000,
rho 2000; a 25 Hz Ricker delayed by 1.2 / f0, the same samples on both sides
(``USE_EXTERNAL_SOURCE_FILE``: SPECFEM3D's run then starts at t = 0 and takes sample n at step
n). The force sits inside an element; receivers inside elements, on element faces and edges,
and on the surface. ``DATA/sweep_box3d.json`` keeps the geometry for the comparison.

* ``iso3d_force``: elastic (``ATTENUATION = .false.``);
* ``visco3d_force``: attenuating, ``Qkappa = Qmu = 30``;
* ``visco3d_km_force``: attenuating, ``Qkappa = 60``, ``Qmu = 20``;
* ``sea3d_force``: 80 m of water (rho 1020, vp 1500, SPECFEM3D's acoustic domain) over the rock,
  the force in the rock, stations in both and the pressure recorded too. SPECFEM3D's sea
  surface is pressure-free; its other faces, wherever there is no absorbing condition, are the
  potential's natural condition, rigid;
* ``vti3d_force``, ``tti3d_force``: the rock transversely isotropic (``ANISOTROPY``),
  Thomsen ``epsilon = 0.25``, ``delta = 0.1``, ``gamma = 0.15``, the axis vertical or tilted
  by 30 degrees toward the azimuth 40 degrees. SPECFEM3D takes an anisotropic medium from its
  ``model_aniso.f90`` by the material's anisotropy flag (1, 2), which a user edits: this script
  writes ``sweep_aniso_cij.inc``, the 21 stiffnesses of each case (in SPECFEM3D's axes, z up),
  for that routine to include (``include 'sweep_aniso_cij.inc'`` at its end; then rebuild
  ``xgenerate_databases``).

The attenuating cases take SPECFEM3D's three mechanisms (its ``N_SLS``) over a band given
(``COMPUTE_FREQ_BAND_AUTOMATIC = .false.``), periods 4 ms .. 0.4 s (2.5 .. 250 Hz, whose
logarithmic centre is the source's 25 Hz), and read the velocities at
``ATTENUATION_f0_REFERENCE`` = 25 Hz.

Run ``xmeshfem3D``, ``xgenerate_databases`` and ``xspecfem3D`` in each ``<out>/<case>`` (serial:
``NPROC = 1``).

    python make_specfem3d_box_cases.py <specfem3d>/EXAMPLES <out>
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ex, out = Path(sys.argv[1]), Path(sys.argv[2])

H, NE, NGLL = 10.0, 32, 5
DT, NSTEP, F0 = 2.5e-4, 1600, 25.0
ROCK = dict(rho=2000.0, vp=2000.0, vs=1000.0)
SRC = (157.3, 163.1, 141.7)                         # x, y, depth (m), inside an element
RECS = {"inside": [(200.7, 163.1, 141.7), (157.3, 231.9, 98.3), (101.1, 96.4, 51.3),
                   (243.5, 88.2, 252.4)],
        "faces": [(210.0, 163.1, 141.7), (157.3, 230.0, 100.0), (120.0, 200.0, 60.0)],
        "surface": [(100.3, 200.7, 0.0), (250.0, 120.0, 0.0), (157.3, 163.1, 0.0)]}
ATT = dict(f0_ref=25.0, min_period=4e-3, max_period=0.4)
WATER = dict(rho=1020.0, vp=1500.0, layers=8)              # the top 80 m
RECS_SEA = {"water": [(200.7, 163.1, 41.7), (101.1, 96.4, 51.3), (243.5, 88.2, 23.4),
                      (210.0, 163.1, 40.0), (120.0, 200.0, 60.0)],
            "rock": [(157.3, 231.9, 98.3), (243.5, 88.2, 252.4), (200.7, 163.1, 141.7),
                     (210.0, 120.0, 130.0)]}
ANISO = dict(epsilon=0.25, delta=0.1, gamma=0.15)
CASES = {"iso3d_force": dict(attenuation=False),
         "visco3d_force": dict(attenuation=True, q=(30.0, 30.0)),
         "visco3d_km_force": dict(attenuation=True, q=(60.0, 20.0)),
         "sea3d_force": dict(attenuation=False, water=WATER, recs=RECS_SEA),
         "vti3d_force": dict(attenuation=False, aniso=dict(ANISO, flag=1, theta=0.0, phi=0.0)),
         "tti3d_force": dict(attenuation=False, aniso=dict(ANISO, flag=2, theta=math.radians(30.0),
                                                           phi=math.radians(40.0)))}
# the Voigt entries (rows, columns xx, yy, zz, yz, xz, xy) of the FD ElasticTTISG3D's order
KEYS = ("11", "12", "13", "14", "15", "16", "22", "23", "24", "25", "26", "33", "34", "35", "36",
        "44", "45", "46", "55", "56", "66")


def tti_cij(rho, vp, vs, epsilon, delta, gamma, theta, phi):
    """The 21 stiffnesses (z down) of sweep.sem.constitutive.thomsen_to_cij3d, by numpy."""
    c33, c44 = rho * vp * vp, rho * vs * vs
    c11, c66 = c33 * (1 + 2 * epsilon), c44 * (1 + 2 * gamma)
    c13 = math.sqrt(max(2 * delta * c33 * (c33 - c44) + (c33 - c44) ** 2, 0.0)) - c44
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = c11
    C[0, 1] = C[1, 0] = c11 - 2 * c66
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = c13
    C[2, 2], C[3, 3], C[4, 4], C[5, 5] = c33, c44, c44, c66
    ct, st, cp, sp = math.cos(theta), math.sin(theta), math.cos(phi), math.sin(phi)
    R = np.array([[cp * ct, -sp, cp * st], [sp * ct, cp, sp * st], [-st, 0.0, ct]])
    M = np.zeros((6, 6))
    for i in range(3):
        M[i] = [R[i, 0] ** 2, R[i, 1] ** 2, R[i, 2] ** 2, 2 * R[i, 1] * R[i, 2],
                2 * R[i, 0] * R[i, 2], 2 * R[i, 0] * R[i, 1]]
    for r, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
        M[3 + r] = [R[a, 0] * R[b, 0], R[a, 1] * R[b, 1], R[a, 2] * R[b, 2],
                    R[a, 1] * R[b, 2] + R[a, 2] * R[b, 1], R[a, 0] * R[b, 2] + R[a, 2] * R[b, 0],
                    R[a, 0] * R[b, 1] + R[a, 1] * R[b, 0]]
    if theta != 0.0:
        C = M @ C @ M.T
    return {k: C[int(k[0]) - 1, int(k[1]) - 1] for k in KEYS}


# entries with one index in yz or xz change sign with z (z down here, up in SPECFEM3D)
Z_ODD = ("14", "15", "24", "25", "34", "35", "46", "56")


def setp(text, key, value):
    pat = re.compile(rf"^({re.escape(key)}\s*=\s*)(\S+)", re.M)
    assert pat.search(text), key
    return pat.sub(lambda m: m.group(1) + str(value), text, count=1)


def dble(x):
    return f"{x:.15e}".replace("e", "d")


base = ex / "applications" / "meshfem3D_examples" / "regular_element_mesh" / "DATA"
par0 = (base / "Par_file").read_text()
mesh0 = (base / "meshfem3D_files" / "Mesh_Par_file").read_text()
for k, v in (("NPROC", 1), ("SIMULATION_TYPE", 1), ("NOISE_TOMOGRAPHY", 0),
             ("SAVE_FORWARD", ".false."), ("NSTEP", NSTEP), ("DT", dble(DT)), ("NGNOD", 8),
             ("MODEL", "default"), ("ANISOTROPY", ".false."), ("PML_CONDITIONS", ".false."),
             ("PML_INSTEAD_OF_FREE_SURFACE", ".false."),
             ("STACEY_ABSORBING_CONDITIONS", ".false."),
             ("STACEY_INSTEAD_OF_FREE_SURFACE", ".false."),
             ("USE_FORCE_POINT_SOURCE", ".true."), ("USE_RICKER_TIME_FUNCTION", ".false."),
             ("USE_EXTERNAL_SOURCE_FILE", ".true."), ("PRINT_SOURCE_TIME_FUNCTION", ".true."),
             ("SAVE_SEISMOGRAMS_DISPLACEMENT", ".false."), ("SAVE_SEISMOGRAMS_VELOCITY", ".true."),
             ("NTSTEP_BETWEEN_OUTPUT_SEISMOS", NSTEP), ("USE_BINARY_FOR_SEISMOGRAMS", ".false."),
             ("SAVE_MESH_FILES", ".false."), ("LOCAL_PATH", "./OUTPUT_FILES/DATABASES_MPI"),
             ("TOPOGRAPHY", ".false."), ("MOVIE_VOLUME", ".false."), ("MOVIE_SURFACE", ".false."),
             ("OUTPUT_ENERGY", ".false."), ("NTSTEP_BETWEEN_OUTPUT_SAMPLE", 1),
             ("USE_SOURCES_RECEIVERS_Z", ".false."), ("GPU_MODE", ".false."),
             ("USE_LDDRK", ".false."), ("LTS_MODE", ".false.")):
    par0 = setp(par0, k, v)
for k, v in (("LATITUDE_MIN", dble(0.0)), ("LATITUDE_MAX", dble(NE * H)),
             ("LONGITUDE_MIN", dble(0.0)), ("LONGITUDE_MAX", dble(NE * H)),
             ("DEPTH_BLOCK_KM", dble(NE * H / 1000.0)), ("NEX_XI", NE), ("NEX_ETA", NE),
             ("NPROC_XI", 1), ("NPROC_ETA", 1), ("USE_REGULAR_MESH", ".true."),
             ("NDOUBLINGS", 0), ("CREATE_VTK_FILES", ".false."),
             ("LOCAL_PATH", "./OUTPUT_FILES/DATABASES_MPI")):
    mesh0 = setp(mesh0, k, v)
mesh0 = mesh0[:mesh0.index("# number of materials")]       # the materials and regions: ours

t = np.arange(NSTEP) * DT - 1.2 / F0
a = (np.pi * F0 * t) ** 2
stf = (1.0 - 2.0 * a) * np.exp(-a)

for name, case in CASES.items():
    par = setp(par0, "ATTENUATION", ".true." if case["attenuation"] else ".false.")
    for k, v in (("ATTENUATION_f0_REFERENCE", dble(ATT["f0_ref"])),
                 ("MIN_ATTENUATION_PERIOD", dble(ATT["min_period"])),
                 ("MAX_ATTENUATION_PERIOD", dble(ATT["max_period"])),
                 ("COMPUTE_FREQ_BAND_AUTOMATIC", ".false."), ("USE_OLSEN_ATTENUATION", ".false.")):
        par = setp(par, k, v)
    q = case.get("q", (9999.0, 9999.0))
    an = case.get("aniso")
    mat = f"1 {ROCK['rho']} {ROCK['vp']} {ROCK['vs']} {q[0]} {q[1]} {an['flag'] if an else 0} 2"
    wat = case.get("water")
    if wat is None:
        mesh = (mesh0 + f"NMATERIALS                      = 1\n{mat}\n"
                f"NREGIONS                        = 1\n1 {NE} 1 {NE} 1 {NE} 1\n")
    else:                                 # SPECFEM3D counts the layers from the bottom
        nr = NE - wat["layers"]
        mesh = (mesh0 + f"NMATERIALS                      = 2\n{mat}\n"
                f"2 {wat['rho']} {wat['vp']} 0.0 9999.0 9999.0 0 1\n"
                f"NREGIONS                        = 2\n1 {NE} 1 {NE} 1 {nr} 1\n"
                f"1 {NE} 1 {NE} {nr + 1} {NE} 2\n")
    par = setp(par, "SAVE_SEISMOGRAMS_PRESSURE", ".true." if wat else ".false.")
    par = setp(par, "ANISOTROPY", ".true." if an else ".false.")
    d = out / name / "DATA"
    (d / "meshfem3D_files").mkdir(parents=True, exist_ok=True)
    (out / name / "OUTPUT_FILES" / "DATABASES_MPI").mkdir(parents=True, exist_ok=True)
    (d / "Par_file").write_text(par)
    (d / "meshfem3D_files" / "Mesh_Par_file").write_text(mesh)
    (d / "meshfem3D_files" / "interfaces.dat").write_text(
        f"1\n .true. 2 2 0.0 0.0 {NE * H:.1f} {NE * H:.1f}\n interface.dat\n {NE}\n")
    (d / "meshfem3D_files" / "interface.dat").write_text("0.0\n0.0\n0.0\n0.0\n")
    (d / "FORCESOLUTION").write_text(
        "FORCE  001\ntime shift:     0.0\nf0:             25.0\n"
        f"latorUTM:       {SRC[1]}\nlongorUTM:      {SRC[0]}\ndepth:          {SRC[2] / 1000.0}\n"
        "source time function:            0\nfactor force source:             1.d0\n"
        "component dir vect source E:     0.d0\ncomponent dir vect source N:     0.d0\n"
        "component dir vect source Z_UP:  -1.d0\nDATA/stf.dat\n")
    (d / "stf.dat").write_text("".join(f"{v:.17e}\n" for v in stf))
    groups = case.get("recs", RECS)
    recs = [p for g in groups.values() for p in g]
    (d / "STATIONS").write_text("".join(f"S{i + 1:04d} AA {y:.6f} {x:.6f} 0.0 {z:.6f}\n"
                                        for i, (x, y, z) in enumerate(recs)))
    (d / "sweep_box3d.json").write_text(json.dumps(dict(
        ROCK, h=H, ne=NE, ngll=NGLL, dt=DT, nstep=NSTEP, f0=F0, source=SRC, receivers=recs,
        groups=[[g, len(v)] for g, v in groups.items()], attenuation=case["attenuation"], q=q,
        **(ATT if case["attenuation"] else {}), **({"water": wat} if wat else {}),
        **({"aniso": dict(an, cij=tti_cij(ROCK["rho"], ROCK["vp"], ROCK["vs"], an["epsilon"],
                                          an["delta"], an["gamma"], an["theta"], an["phi"]))}
           if an else {})),
        indent=1) + "\n")
    print("wrote", d)

inc = ["! the stiffnesses of make_specfem3d_box_cases.py's anisotropic cases (Pa, z up)"]
for name, case in CASES.items():
    an = case.get("aniso")
    if an is None:
        continue
    c = tti_cij(ROCK["rho"], ROCK["vp"], ROCK["vs"], an["epsilon"], an["delta"], an["gamma"],
                an["theta"], an["phi"])
    inc.append(f"  if (iflag_aniso == {an['flag']}) then      ! {name}")
    inc += [f"    c{k} = {(-c[k] if k in Z_ODD else c[k]):.17e}_CUSTOM_REAL" for k in KEYS]
    inc.append("  endif")
(out / "sweep_aniso_cij.inc").write_text("\n".join(inc) + "\n")
print("wrote", out / "sweep_aniso_cij.inc")

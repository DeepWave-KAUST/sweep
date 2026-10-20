"""Generate SPECFEM2D + matching geometry for an acoustic flat-domain case.

Writes everything into a workdir (default ``_smoketest/specfem_acoustic_flat/``)
that is a valid SPECFEM2D run directory: ``DATA/{Par_file, SOURCE, STATIONS,
interfaces.dat}`` and an empty ``OUTPUT_FILES/``.

The same physical setup will later be exercised by the SEM and the sweep FD
solver via ``run_compare.py`` so all three see identical source, receivers,
mesh size, dt, and BCs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class Setup:
    # Physical
    x_max: float = 1000.0
    z_max: float = 750.0       # in SEM coords z increases DOWN, top is z=0
    vp: float = 2000.0
    rho: float = 1000.0
    # Mesh: 40×30 elements gives dh_e = 25 m, effective GLL spacing ~6 m
    # (finer than FD's dh=10 m); same N=4 → 5 GLL nodes per direction.
    nex: int = 40
    nez: int = 30
    n_order: int = 4
    # Time — long enough for several boundary round-trips so the BC
    # accumulation behaviour shows up (~2.4 s, c=2000 m/s, x_max=1000 m
    # → first round-trip back to centre ≈ 0.5 s; we want to see 3-4 of them).
    dt: float = 0.0008
    nstep: int = 3000
    # Source / receivers (all in SEM coords; the conversion to SPECFEM
    # z-up coords is done in writers)
    src_xz: tuple = (500.0, 100.0)   # 100 m below SEM top
    rec_x_min: float = 50.0
    rec_x_max: float = 950.0
    rec_x_step: float = 50.0
    rec_z: float = 100.0
    # Source spectrum
    f0: float = 12.0
    tshift: float = 0.10
    # SPECFEM output cadence (snapshot interval, in steps)
    image_every: int = 50


def specfem_z(z_sem: float, z_max: float) -> float:
    """Convert SEM z (down) to SPECFEM z (up)."""
    return z_max - z_sem


def write_par_file(path: Path, s: Setup) -> None:
    # NSTEP_BETWEEN_OUTPUT_SEISMOS must be ≥ nstep so output flushed at end.
    text = f"""# SPECFEM2D Par_file — acoustic flat-domain control (matching sweep SEM PoC)

title                           = SEM acoustic flat control
SIMULATION_TYPE                 = 1
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

NPROC                           = 1
NSTEP                           = {s.nstep}
DT                              = {s.dt:.7f}d0
time_stepping_scheme            = 1     # Newmark, 2nd-order (SPECFEM default)
P_SV                            = .true.
AXISYM                          = .false.

PARTITIONING_TYPE               = 3
NGNOD                           = 9
setup_with_binary_database      = 0
MODEL                           = default
SAVE_MODEL                      = default

ATTENUATION_VISCOELASTIC        = .false.
ATTENUATION_VISCOACOUSTIC       = .false.
N_SLS                           = 3
ATTENUATION_f0_REFERENCE        = 5.196
READ_VELOCITIES_AT_f0           = .false.
USE_SOLVOPT                     = .false.
ATTENUATION_PORO_FLUID_PART     = .false.
Q0_poroelastic                  = 1
freq0_poroelastic               = 10
UNDO_ATTENUATION_AND_OR_PML     = .false.
NT_DUMP_ATTENUATION             = 500
NO_BACKWARD_RECONSTRUCTION      = .false.

NSOURCES                        = 1
force_normal_to_surface         = .false.
initialfield                    = .false.
add_Bielak_conditions_bottom    = .false.
add_Bielak_conditions_right     = .false.
add_Bielak_conditions_top       = .false.
add_Bielak_conditions_left      = .false.
ACOUSTIC_FORCING                = .false.
noise_source_time_function_type = 4
write_moving_sources_database   = .false.

seismotype                      = 4   # 4 = pressure (acoustic)
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = {s.nstep}
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1
USE_TRICK_FOR_BETTER_PRESSURE   = .false.
USER_T0                         = 0.0d0
save_ASCII_seismograms          = .true.
save_binary_seismograms_single  = .true.
save_binary_seismograms_double  = .false.
SU_FORMAT                       = .false.
use_existing_STATIONS           = .true.
nreceiversets                   = 1
anglerec                        = 0.d0
rec_normal_to_surface           = .false.
nrec                            = 1
xdeb                            = 100.
zdeb                            = 100.
xfin                            = 100.
zfin                            = 100.
record_at_surface_same_vertical = .false.

save_ASCII_kernels              = .true.
NTSTEP_BETWEEN_COMPUTE_KERNELS  = 1
APPROXIMATE_HESS_KL             = .false.

PML_BOUNDARY_CONDITIONS         = .true.
NELEM_PML_THICKNESS             = 12
ROTATE_PML_ACTIVATE             = .false.
ROTATE_PML_ANGLE                = 30.
K_MIN_PML                       = 1.0d0
K_MAX_PML                       = 1.0d0
damping_change_factor_acoustic  = 0.5d0
damping_change_factor_elastic   = 1.0d0
PML_PARAMETER_ADJUSTMENT        = .false.

STACEY_ABSORBING_CONDITIONS     = .false.

ADD_PERIODIC_CONDITIONS         = .false.
PERIODIC_HORIZ_DIST             = 4000.d0

# Acoustic material: rho Vp Vs=0
nbmodels                        = 1
1 1 {s.rho:.1f}d0 {s.vp:.1f}d0 0.0d0 0 0 9999 9999 0 0 0 0 0 0

TOMOGRAPHY_FILE                 = ./DATA/tomo_file.xyz
read_external_mesh              = .false.

mesh_file                       = ./DATA/mesh_file
nodes_coords_file               = ./DATA/nodes_coords_file
materials_file                  = ./DATA/materials_file
free_surface_file               = ./DATA/free_surface_file
axial_elements_file             = ./DATA/axial_elements_file
absorbing_surface_file          = ./DATA/absorbing_surface_file
acoustic_forcing_surface_file   = ./DATA/MSH/Surf_acforcing_Bottom_enforcing_mesh
absorbing_cpml_file             = ./DATA/absorbing_cpml_file
tangential_detection_curve_file = ./DATA/courbe_eros_nodes

interfacesfile                  = ./interfaces.dat
xmin                            = 0.d0
xmax                            = {s.x_max:.1f}d0
nx                              = {s.nex}
absorbbottom                    = .true.
absorbright                     = .true.
absorbtop                       = .false.
absorbleft                      = .true.
nbregions                       = 1
1 {s.nex}  1 {s.nez} 1

NTSTEP_BETWEEN_OUTPUT_INFO      = 200
output_grid_Gnuplot             = .false.
output_grid_ASCII               = .true.
OUTPUT_ENERGY                   = .false.
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10
COMPUTE_INTEGRATED_ENERGY_FIELD = .false.

NTSTEP_BETWEEN_OUTPUT_IMAGES    = {s.image_every}
cutsnaps                        = 1.

output_color_image              = .true.
imagetype_JPEG                  = 10   # 10 = pressure
factor_subsample_image          = 1.0d0
USE_CONSTANT_MAX_AMPLITUDE      = .false.
CONSTANT_MAX_AMPLITUDE_TO_USE   = 1.17d4
POWER_DISPLAY_COLOR             = 0.30d0
DRAW_SOURCES_AND_RECEIVERS      = .true.
DRAW_WATER_IN_BLUE              = .false.
USE_SNAPSHOT_NUMBER_IN_FILENAME = .false.

output_postscript_snapshot      = .false.
imagetype_postscript            = 1
meshvect                        = .true.
modelvect                       = .false.
boundvect                       = .true.
interpol                        = .true.
pointsdisp                      = 6
subsamp_postscript              = 1
sizemax_arrows                  = 1.d0
US_LETTER                       = .false.

# Wavefield dumps: full GLL grid values for numerical comparison.
output_wavefield_dumps          = .true.
imagetype_wavefield_dumps       = 4    # 4 = pressure
use_binary_for_wavefield_dumps  = .true.

NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.
GPU_MODE                        = .false.
"""
    path.write_text(text)


def _ricker_np(t: np.ndarray, f0: float, delay: float) -> np.ndarray:
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def write_source(path: Path, s: Setup, ext_src_file: Path | None = None) -> None:
    xs, zs_sem = s.src_xz
    zs_spec = specfem_z(zs_sem, s.z_max)
    if ext_src_file is None:
        time_function_type = 1
        name_of_source_file = "YYYYYYYY"
    else:
        time_function_type = 8        # 8 = user-supplied external file
        name_of_source_file = str(ext_src_file)
    txt = f"""## acoustic pressure / dilatation source
source_surf                     = .false.
xs                              = {xs:.4f}
zs                              = {zs_spec:.4f}
source_type                     = 1
time_function_type              = {time_function_type}
name_of_source_file             = {name_of_source_file}
burst_band_width                = 0.
f0                              = {s.f0:.4f}
tshift                          = {s.tshift:.4f}
anglesource                     = 0.0
Mxx                             = 1.0d0
Mzz                             = 1.0d0
Mxz                             = 0.0d0
factor                          = 1.d0
vx                              = 0.0
vz                              = 0.0
"""
    path.write_text(txt)


def write_external_source_file(path: Path, s: Setup) -> None:
    """Dump the exact Ricker SEM uses, sampled on the SPECFEM time grid.

    SPECFEM2D's external source file format: one ``time value`` pair per
    line. The internal solver clock starts at -t0 ~= -1.2/f0 - tshift, so
    we sample the full file-time axis to make sure SPECFEM has a value at
    every requested step.
    """
    t0_internal = 1.2 / s.f0 + s.tshift     # SPECFEM auto t0 heuristic
    t = np.arange(s.nstep + 1) * s.dt - t0_internal
    vals = _ricker_np(t, s.f0, 0.0)         # peak at t=0 in this axis
    with open(path, "w") as f:
        for tk, vk in zip(t, vals):
            f.write(f"{tk: .10e}  {vk: .10e}\n")


def write_stations(path: Path, s: Setup) -> list[tuple[float, float]]:
    rec_x = np.arange(s.rec_x_min, s.rec_x_max + 0.5 * s.rec_x_step, s.rec_x_step)
    rec_zspec = specfem_z(s.rec_z, s.z_max)
    lines = []
    rec_xz_sem = []
    for i, xr in enumerate(rec_x):
        lines.append(f"S{i:03d} AA {xr:14.4f} {rec_zspec:14.4f}  0.0 0.0\n")
        rec_xz_sem.append((float(xr), float(s.rec_z)))
    path.write_text("".join(lines))
    return rec_xz_sem


def write_interfaces(path: Path, s: Setup) -> None:
    """Two-interface format: bottom + free top, with nez elements stacked."""
    txt = (
        "# number of interfaces\n"
        " 2\n"
        "#\n"
        "# interface 1 (bottom)\n"
        " 2\n"
        f" 0.0 0.0\n"
        f" {s.x_max:.1f} 0.0\n"
        "# interface 2 (top = free surface, flat)\n"
        " 2\n"
        f" 0.0 {s.z_max:.1f}\n"
        f" {s.x_max:.1f} {s.z_max:.1f}\n"
        "#\n"
        "# nz_elements per layer\n"
        f" {s.nez}\n"
    )
    path.write_text(txt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir", default="_smoketest/specfem_acoustic_flat",
        help="root directory where DATA/ and OUTPUT_FILES/ will be placed",
    )
    args = parser.parse_args()
    root = Path(args.workdir)
    data = root / "DATA"
    out = root / "OUTPUT_FILES"
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    s = Setup()
    write_par_file(data / "Par_file", s)
    ext_src = data / "external_ricker.txt"
    write_external_source_file(ext_src, s)
    # SPECFEM resolves relative paths from the run directory (parent of DATA).
    write_source(data / "SOURCE", s, ext_src_file=Path("./DATA/external_ricker.txt"))
    rec_xz_sem = write_stations(data / "STATIONS", s)
    write_interfaces(data / "interfaces.dat", s)

    geom = {**asdict(s), "rec_xz_sem": rec_xz_sem}
    np.savez(root / "geometry.npz", **{
        k: (np.asarray(v) if isinstance(v, (list, tuple)) and len(v) > 0 else v)
        for k, v in geom.items() if not isinstance(v, dict)
    })
    print(f"Wrote SPECFEM2D inputs to {data}")
    print(f"  domain  : {s.x_max} x {s.z_max} m,  vp={s.vp},  rho={s.rho}")
    print(f"  mesh    : {s.nex} x {s.nez} elements (N={s.n_order})")
    print(f"  source  : SEM xz=({s.src_xz}) -> SPECFEM ({s.src_xz[0]},"
          f" {specfem_z(s.src_xz[1], s.z_max)})")
    print(f"  dt={s.dt}  nstep={s.nstep}  image_every={s.image_every}")
    print(f"  free top + Stacey on left/right/bottom")
    print(f"  next: cd {root} && ln -sf <repo>/_specfem_compare/specfem2d_src/bin/* . && "
          f"./xmeshfem2D && ./xspecfem2D")


if __name__ == "__main__":
    main()

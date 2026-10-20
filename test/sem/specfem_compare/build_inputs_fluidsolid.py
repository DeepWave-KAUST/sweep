"""SPECFEM2D inputs for the flat water-over-solid coupling gate (stage 4c).

Two-layer model, SAME physics both solvers can run today (no interface PML):

  water : vp 1500, rho 1000, vs 0     SEM z (down) in [0, 1000)
  solid : vp 3000, vs 1732, rho 2200  SEM z in [1000, 2500]

Boundary conditions are matched EXACTLY so the all-reflecting comparison is
valid for every reverberation:
  * SPECFEM: PML off, Stacey off, absorb* all false. Verified in
    meshfem2D/determine_acoustic_surface.f90: every NON-absorbing boundary edge
    of an ACOUSTIC element becomes a pressure-release (chi = 0) free surface —
    top AND the fluid part of the side walls; the elastic part of the
    boundaries is natural = traction-free.
  * ours: AcousticElasticSEM2D with dirichlet=("top", "left", "right") on the
    fluid (chi on the solid part of those walls is a dead DOF anyway) and the
    solid's natural traction-free everywhere.

Source: pressure point in the water, EXTERNAL Ricker file so both solvers use
bit-identical source time functions (SPECFEM's clock starts at -t0; the file is
sampled on that axis with the peak at physical t = 0). Note SPECFEM divides the
acoustic source by kappa at the source point (compute_add_sources_acoustic.f90
:226-228) and interpolates a Lagrange stamp — a constant amplitude factor vs our
unit-node injection, absorbed by one global scalar fitted on the hydrophones.

Receivers: 17 hydrophones in the water (SEM z=400) + 17 geophones in the solid
(SEM z=1600), x = 1200..2800 step 100 — all on GLL nodes of the 50 m mesh so
both solvers sample exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FS:
    x_max: float = 4000.0
    z_max: float = 2500.0          # SEM z-down; top is z=0
    z_if: float = 1000.0           # fluid-solid interface depth (SEM)
    vp_w: float = 1500.0
    rho_w: float = 1000.0
    vp_s: float = 3000.0
    vs_s: float = 1732.0
    rho_s: float = 2200.0
    nex: int = 80
    nez: int = 50                  # 50 m elements; 20 water rows / 30 solid rows
    n_order: int = 4
    dt: float = 1.0e-3
    nstep: int = 1700
    src_xz: tuple = (2000.0, 500.0)      # SEM coords, in the water
    rec_x: tuple = tuple(1200.0 + 100.0 * i for i in range(17))
    z_hydro: float = 400.0
    z_geo: float = 1600.0
    f0: float = 10.0
    tshift: float = 0.10
    dump_every: int = 150

    @property
    def t0_internal(self) -> float:      # SPECFEM auto t0 heuristic
        return 1.2 / self.f0 + self.tshift

    @property
    def nez_water(self) -> int:
        return round(self.z_if / (self.z_max / self.nez))


def specfem_z(z_sem: float, z_max: float) -> float:
    return z_max - z_sem


def write_par_file(path: Path, s: FS) -> None:
    nzw = s.nez_water
    text = f"""# SPECFEM2D Par_file — flat fluid-solid coupling gate (matches sweep SEM)
title                           = fluid-solid flat coupling gate
SIMULATION_TYPE                 = 1
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

NPROC                           = 1
NSTEP                           = {s.nstep}
DT                              = {s.dt:.7f}d0
time_stepping_scheme            = 1
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

seismotype                      = 1,4
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = {s.nstep}
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1
USE_TRICK_FOR_BETTER_PRESSURE   = .false.
USER_T0                         = 0.0d0
save_ASCII_seismograms          = .true.
save_binary_seismograms_single  = .false.
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

PML_BOUNDARY_CONDITIONS         = .false.
NELEM_PML_THICKNESS             = 3
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

# model 1 = solid, model 2 = water (acoustic <=> Vs == 0)
nbmodels                        = 2
1 1 {s.rho_s:.1f}d0 {s.vp_s:.1f}d0 {s.vs_s:.4f}d0 0 0 9999 9999 0 0 0 0 0 0
2 1 {s.rho_w:.1f}d0 {s.vp_w:.1f}d0 0.0d0 0 0 9999 9999 0 0 0 0 0 0

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
absorbbottom                    = .false.
absorbright                     = .false.
absorbtop                       = .false.
absorbleft                      = .false.
# SPECFEM z-up: rows 1..{s.nez - nzw} = solid (model 1), rows {s.nez - nzw + 1}..{s.nez} = water (model 2)
nbregions                       = 2
1 {s.nex}  1 {s.nez - nzw} 1
1 {s.nex} {s.nez - nzw + 1} {s.nez} 2

NTSTEP_BETWEEN_OUTPUT_INFO      = 200
output_grid_Gnuplot             = .false.
output_grid_ASCII               = .false.
OUTPUT_ENERGY                   = .false.
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10
COMPUTE_INTEGRATED_ENERGY_FIELD = .false.

NTSTEP_BETWEEN_OUTPUT_IMAGES    = {s.dump_every}
cutsnaps                        = 1.
output_color_image              = .false.
imagetype_JPEG                  = 10
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

output_wavefield_dumps          = .true.
imagetype_wavefield_dumps       = 4
use_binary_for_wavefield_dumps  = .true.

NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.
GPU_MODE                        = .false.
"""
    path.write_text(text)


def _ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def write_external_source_file(path: Path, s: FS) -> None:
    """Ricker sampled on SPECFEM's clock (starts at -t0), peak at t = 0."""
    t = np.arange(s.nstep + 1) * s.dt - s.t0_internal
    with open(path, "w") as f:
        for tk, vk in zip(t, _ricker(t, s.f0, 0.0)):
            f.write(f"{tk: .10e}  {vk: .10e}\n")


def write_source(path: Path, s: FS) -> None:
    xs, zs_sem = s.src_xz
    path.write_text(f"""## pressure source in the water (external STF)
source_surf                     = .false.
xs                              = {xs:.4f}
zs                              = {specfem_z(zs_sem, s.z_max):.4f}
source_type                     = 1
time_function_type              = 8
name_of_source_file             = ./DATA/external_ricker.txt
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
""")


def write_stations(path: Path, s: FS) -> None:
    lines = []
    k = 0
    for z_sem in (s.z_hydro, s.z_geo):          # hydro first, then geo
        zsp = specfem_z(z_sem, s.z_max)
        for xr in s.rec_x:
            lines.append(f"S{k:03d} AA {xr:14.4f} {zsp:14.4f}  0.0 0.0\n")
            k += 1
    path.write_text("".join(lines))


def write_interfaces(path: Path, s: FS) -> None:
    nzw = s.nez_water
    z_if_spec = specfem_z(s.z_if, s.z_max)
    path.write_text(
        "# number of interfaces\n"
        " 3\n"
        "# interface 1 (bottom)\n"
        " 2\n"
        " 0.0 0.0\n"
        f" {s.x_max:.1f} 0.0\n"
        "# interface 2 (fluid-solid)\n"
        " 2\n"
        f" 0.0 {z_if_spec:.1f}\n"
        f" {s.x_max:.1f} {z_if_spec:.1f}\n"
        "# interface 3 (top free surface)\n"
        " 2\n"
        f" 0.0 {s.z_max:.1f}\n"
        f" {s.x_max:.1f} {s.z_max:.1f}\n"
        "# elements per layer (bottom to top)\n"
        f" {s.nez - nzw}\n"
        f" {nzw}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir",
                    default="_smoketest/specfem_fluidsolid")
    args = ap.parse_args()
    root = Path(args.workdir)
    (root / "DATA").mkdir(parents=True, exist_ok=True)
    (root / "OUTPUT_FILES").mkdir(parents=True, exist_ok=True)
    s = FS()
    write_par_file(root / "DATA" / "Par_file", s)
    write_external_source_file(root / "DATA" / "external_ricker.txt", s)
    write_source(root / "DATA" / "SOURCE", s)
    write_stations(root / "DATA" / "STATIONS", s)
    write_interfaces(root / "DATA" / "interfaces.dat", s)
    print(f"wrote {root}/DATA  (water rows {s.nez_water}, solid rows "
          f"{s.nez - s.nez_water}; t0={s.t0_internal})")


if __name__ == "__main__":
    main()

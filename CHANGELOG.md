# Changelog

All notable changes to SWEEP are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Agent Skill for AI coding assistants** in `skills/sweep-solver/`
  ([agentskills.io](https://agentskills.io) format): a `SKILL.md` entry point
  plus on-demand references (equations, propagators, boundaries, gradients,
  imaging, SEM, parallelism, CLI and datasets, generated API signatures) and
  two helper scripts, `check_env.py` and `stability_check.py`.  Claude Code,
  Codex, Claude.ai and other assistants that read the format then write
  `sweep` code with the real API.  The wheel carries it (as
  `sweep/_skill/sweep-solver`), and **`sweep skill install`** copies it to
  `~/.claude/skills/` (Claude Code) and `~/.agents/skills/` (Codex) for the
  assistants it finds, or for those named with `--for claude|codex|all`;
  with none found and none named it installs nothing and lists the options.
  `--project DIR` installs into a project, `sweep skill zip` writes the zip
  Claude.ai takes, and an existing folder is replaced only when it is this
  skill.  The docs page *AI assistants (Agent Skills)* covers the install,
  and `test/test_skill_snippets.py` runs every code block of the skill so
  its examples cannot fall behind the API.
- **`ViscoElasticVSSEM3D`, `ViscoElasticVSSEM3DDeformed`: visco-elastic attenuation on the
  3-D SEM.** The GSLS of `ViscoElasticVSSEM2DDeformed` on `ElasticVSSEM3D` (Hex8) and
  `ElasticVSSEM3DDeformed` (Hex27, a topography): six memory variables per mechanism, the
  models `vp, vs, rho, Qp, Qs`, the 2-D class's options, all five models differentiable,
  `Q = inf` the elastic step bit for bit; with the C-PML, the 3-D classes that combine
  attenuation, a topography and absorbing boundaries. A point force in a whole space matches
  the visco-elastic Green's function (the elastic one with the GSLS complex moduli) to
  0.7e-3 to 1.2e-3 in relative L2, where the attenuation changes the records by 12% to 47%
  (`test/sem/test_sem_visco_elastic3d.py`). No boundary saving (dissipative).
- **Curved internal interfaces in 3-D: `PropSEM(interfaces=...)` on the Hex27 classes and
  `sweep.sem.build_layered_hex27`.** Each interface is `(ny, nx)` rows; the element layers
  follow the interfaces, graded linearly in every column of control nodes, and
  `layered_model` gives each layer its material. The mid-nodes lie on the interfaces, so the
  faces between the layers follow their curvature, where SPECFEM3D's internal mesher builds
  straight-sided elements; on that straight-sided mesh `ElasticVSSEM3DDeformed` agrees with
  SPECFEM3D to 3e-14 across a curved interface. A curved sea floor for
  `BiotVSSEM3DDeformed` (water and rock, each element wholly one medium) is the case that needs
  it (`test/sem/test_sem_interfaces3d.py`); notebook 61: a seamount.
- **`impl='c'` for `BiotVSSEM3DDeformed`** (core entries `sem3d_biot_forward` / `_backward`,
  `sem3d/biot3d.cu`): the 2-D Biot kernels' step with y, the flux copies advanced in groups of
  their number (one launch per group, the eager class's drag blocks), the stress divergence of
  `ElasticVSSEM3DDeformed`. Records of the thirteen receiver fields and every coefficient's
  gradient agree with eager to the run-to-run rounding for each kind of source, with the C-PML
  or an open-pore surface, water over the porous rock and the viscous memory
  (`test/sem/test_sem_c_biot3d.py`). 41^3 for 600 steps: 1.04 s and 2.4 GB against 4.20 s and
  4.5 GB on eager's defaults (A100).
- **`impl='c'` for `BiotVSSEM2DDeformed`, `BiotAnisoPermSEM2DDeformed` and
  `ViscoBiotVSSEM2DDeformed`** (the GSLS frame's memories in the stress kernel; core entries
  `sem2d_biot_forward` / `_backward`, `sem2d/biot2d.cu`): the eager step one kernel per
  sub-step -- the pressure injection and pin, the strong grad p, the stress divergence, the
  flux copies of every node advanced with the drag's exact step, the frame stress, the weak
  div q, the fluid-solid interface, the viscous memory, the C-PML on every derivative -- and
  their transposes; the gradients go on to the eight coefficients through the class's build
  and its `drag_operators`. Records of the eight receiver fields and every coefficient's
  gradient agree with eager to the run-to-run rounding for each kind of source, with the
  C-PML or an open-pore surface, through water, porous and dry regions, with the viscous
  memory and a drag tensor (`test/sem/test_sem_c_biot2d.py`); the adjoint checkpoints in
  chunks by default. 201 x 401 for 2000 steps: 1.58 s and 6.1 GB against 7.00 s and 5.7 GB
  on eager's defaults (A100).
- **`memory=Ckpt(chunks=K)` on `impl='c'` for the velocity-stress classes**, 2-D and 3-D,
  and PropSEM's default there (`K = 64`; it was the save-all). The forward keeps the state
  at the start of every `K`-step chunk; the backward reruns each chunk from its checkpoint
  with the histories on and runs the adjoint through it (`sweep.sem.drivers.vs_chunks`; the
  core entries take a step range `t0`, `t1`). Records and gradients equal the save-all's up
  to the kernels' run-to-run rounding, for chunks that divide the record or not, of one
  step or of all but one (`test/sem/test_sem_c_vs_ckpt.py`). On an A100 a forward and
  backward call of `ElasticVSSEM2DDeformed` on 401 x 801 for 2000 steps took 1.16 s and
  1.8 GB, against 0.81 s and 23.3 GB with `Full()` and 2.28 s and 7.3 GB on eager's
  defaults; `ElasticVSSEM3DDeformed` on 65^3 for 1000 steps 1.32 s and 4.4 GB, against
  0.91 s and 36.4 GB, and 9.84 s and 12.5 GB.
- **`impl='c'` for the 2-D velocity-stress family**: `ElasticVSSEM2DDeformed`,
  `ElasticVTISEM2DDeformed`, `ElasticTTISEM2DDeformed` and `ViscoElasticVSSEM2DDeformed`, through
  two new core entries (`sem2d_elastic_vs_forward` / `_backward`, `sem2d/vs2d.cu`). The 3-D
  velocity-stress kernels with two components: the per-node 2 x 2 inverse Jacobian, the C-PML
  on the divergence split by direction and on the velocity gradients, the four or six
  stiffnesses or the GSLS memories in the stress update, and their transposes. Records and the
  gradients in every model match eager to 2.5e-7 to 4.8e-6 in relative L2, with the C-PML and
  bare, on a mesh with a topography and an interior shear (`test/sem/test_sem_c_vs2d.py`).
  The natural free surface only: `free_top=True`, which `PropSEM` never sets, stays on eager.
- **`impl='c'` for `ViscoElasticVSSEM3D` and `ViscoElasticVSSEM3DDeformed`.** The elastic
  velocity-stress kernels with the unrelaxed moduli and the GSLS memories (six per mechanism,
  element-local) advancing in the stress kernel; the transpose carries the memories' adjoint
  and images the relaxed moduli times the strengths, which autograd takes on to `vp`, `vs`,
  `rho`, `Qp` and `Qs` through the class's fit. Records and the gradients in all five models
  match eager to 2e-7 to 4e-6 in relative L2 with the C-PML, on Hex8 and on a curved Hex27,
  and `Q = inf` gives the elastic kernels' records (`test/sem/test_sem_c_visco3d.py`). Three
  mechanisms take 1.55 to 1.6 times the elastic time (A100, 16^3 elements).
- **`impl='c'` for the 3-D VTI and TTI classes** (`ElasticVTISEM3D`, `ElasticTTISEM3D` and
  their Hex27 versions). The velocity-stress kernels take the nine or 21 Voigt stiffnesses in
  the stress update and its transpose (one template per stiffness count, the isotropic code
  unchanged) and return the stiffness gradients, which go on to the Thomsen parameters, the
  tilt and the azimuth through `thomsen_to_cij3d`, as on eager. Records and the gradients in
  all eight models match eager to 2e-7 to 2.4e-6 in relative L2 with the C-PML, on Hex8 and on
  a curved Hex27 (`test/sem/test_sem_c_tti3d.py`); a medium without anisotropy runs the
  isotropic kernels' numbers. VTI takes 1.1 times and TTI 1.4 times the isotropic time
  (A100, 16^3 elements).
- **`impl='c'` for `ElasticVSSEM3DDeformed`: 3-D elastic with topography and the C-PML on
  CUDA.** The `ElasticVSSEM3D` time loop with four element kernels that apply the full
  3 x 3 inverse Jacobian at every GLL node (the stress divergence, split by direction in
  the C-PML band, the velocity gradients and their transposes); the nodal kernels are
  shared. On a mesh with a topography and an interior shear, so that all nine metric terms
  are non-zero, the records and the gradients in `vp`, `vs`, `rho` match eager to 3e-7 to
  1.2e-6 in relative L2, the size of the CUDA path's own run-to-run differences, and a flat
  Hex27 reproduces the Hex8 kernels (`test/sem/test_sem_c_elastic_vs3d_deformed.py`). It
  takes 1.15 to 1.3 times the Hex8 time on the same elements (A100).
- **`ElasticVTISEM3D`, `ElasticTTISEM3D` (and their `...Deformed` versions on Hex27):
  transversely isotropic media on the 3-D SEM.** The velocity-stress step of `ElasticVSSEM3D`
  with the stiffnesses of a VTI medium (`vp, vs, rho, epsilon, delta, gamma`) or of one whose
  symmetry axis is tilted by `theta` toward the azimuth `phi` (radians): the parameters and the
  Bond rotation of the FD `ElasticTTISG3D` (`sweep.sem.thomsen_to_cij3d`), all 21 stiffnesses
  in the step, the C-PML, the free surface, topography and the gradients of the parent. A TTI
  medium without tilt is the VTI step bit for bit, while the gradient in `theta` there stays the
  rotation's, so an inversion can start from a vertical axis. An axis tilted by 90 degrees
  toward x or y reproduces the turned VTI problem to round-off, and against SPECFEM3D's
  anisotropic medium (its `model_aniso.f90` given the same stiffnesses) the records agree to
  5e-14 for a vertical and a tilted axis (`test/sem/test_sem_tti3d.py`,
  `run_compare_box3d.py`). Notebook 60: wavefronts and shear-wave splitting.
- **The 3-D SEM against SPECFEM3D.** `test/sem/specfem_compare/run_compare_box3d.py` runs the
  3-D classes against SPECFEM3D 4.1.1 in double precision on a box with six free sides
  (`make_specfem3d_box_cases.py`: a point force, stations inside elements, on faces and edges
  and on the surface). `ElasticVSSEM3D`, `ElasticVSSEM3DDeformed` and `ElasticSEM3D` agree to
  5e-14 to 8e-14. On SPECFEM3D's own mechanisms (Qkappa = Qmu = 30; Qkappa = 60, Qmu = 20)
  `ViscoElasticVSSEM3D` agrees to 3.6e-3 and 4.4e-3, which is the time discretization of the
  memory variables: with SPECFEM3D's memory update in our step the records agree to 1.3e-14.
  SPECFEM3D's update overstates the strain it is driven by a factor `1 + (omega dt)^2`, and
  its complex modulus is off by 1.3e-4 at 25 Hz and dt = 0.25 ms; ours is off by 3e-6.
  `BiotVSSEM3DDeformed` with water over the dry rock agrees with SPECFEM3D's acoustic-elastic
  coupling to round-off (the pressure, and the velocities in the rock and inside the water's
  elements, 2e-12): its fluid region is SPECFEM3D's potential scheme once the fluid velocity
  is eliminated. At a station on an element face the fluid velocity across the face differs
  by 2e-3 to 5e-3 (SPECFEM3D takes one element, ours averages them).
- **Plane-wave incidence on the SEM, SPECFEM2D's `initialfield`: `sweep.sem.PlaneWave` in
  place of the source coordinates** (`prop(wavelet, PlaneWave("SV", 30.0), receivers, ...)`).
  A plane P or SV wave from below at any angle, a Rayleigh wave along the surface, or a plane
  SH wave, with the half-space's reflected and converted waves under a free top, the incident
  wave alone under an absorbing one. On `ElasticSEM2DDeformed`, `ViscoElasticSEM2DDeformed`
  and `ElasticSHSEM2DDeformed`, `impl='eager'`, every memory mode. The wave enters through a
  band of elements one element thick around a box (the domain reduction method; the loads
  `-K_be u0_e` and `K_eb u0_b` on the band's nodes): inside the box the run carries the whole
  field, from the band out only what the box scatters, which the C-PML absorbs. SPECFEM2D
  feeds the wave through its Stacey edges instead (Bielak's conditions). The plane wave is
  computed in the frequency domain from the wavelet's samples, one traction-free solve below,
  at and past the critical angle (where the reflected P wave is evanescent);
  `PropSEM.free_field` gives it at any points. Inside the box the medium is free (a
  topography, a basin, attenuation); outside it must be the uniform elastic background under a
  flat surface, and anything else is refused, as are `impl='c'` (one wavelet per shot) and a
  wave already on the band at the first step. In a flat half-space the records inside the box
  are the plane wave to the leapfrog's error. Against SPECFEM2D 8.1.0 (double precision,
  `test/sem/specfem_compare/run_compare_planewave.py`): in a domain whose Stacey edges are far
  from the receivers the two agree to 1.3e-4 to 2.5e-4, under a soft basin too; in a small one
  SPECFEM2D's edges put 2e-3 to 5e-3 of their own into its records, where ours meet the plane
  wave to 1.6e-4 to 4.2e-4, past the critical angle and for the Rayleigh wave as well. In SH
  SPECFEM2D's Stacey edges take no Bielak terms, and its records near them miss the plane wave
  by up to 20% (`test/sem/test_sem_plane_wave.py`). Notebook 58.
- **The nonlinear constant-Q fit, SPECFEM2D's `USE_SOLVOPT`: `fit="nonlinear"`** on
  `ViscoElasticSEM2DDeformed` and `ViscoAcousticPotentialSEM2D` (`equation_options`). The
  relaxation frequencies are fitted with the strengths, for each Q, both kept positive, by
  the least squares of the linear fit's residual `Q Im M - Re M` at `4 n_sls` frequencies
  (`n_fit=None`). Over SPECFEM2D's band at Q = 30 the largest error of `Q(f)` drops from 5.1%
  (the linear fit at its `2 n_sls - 1` frequencies) to 3.3% with three mechanisms and from
  3.4% to 0.7% with four; strengths stay positive down to Q = 1, where the linear fit refuses
  Q below 1.5-1.8. Each node then has its own relaxation times, the bulk ones from `Qkappa`
  and the shear ones from `Qmu`. `sweep.equations.gsls_nonlinear.GSLSNonlinearFit` solves the
  fit by Levenberg-Marquardt at the Chebyshev nodes of `1 / Q` once per number of mechanisms,
  band ratio and number of frequencies, and maps a model's Q through the Chebyshev series, so
  Q stays differentiable and `Q = inf` gives zero strengths. Its mechanisms agree with those of
  SPECFEM2D's own SolvOpt to 2.5e-4, which stops at four digits
  (`test/sem/specfem_compare/specfem_solvopt_driver.f90`, `test/sem/test_sem_gsls_nonlinear.py`);
  against SPECFEM2D run with `USE_SOLVOPT` the records differ by 1.8e-5, and agree to 6e-14 on
  its own mechanisms.
  `fit` also takes an object with `q_min` and `params(Q)`. Notebook 49.
- **Illumination on the SEM: `PropSEM.compute_illumination`.** PropTorch's switch and pair,
  read back after a call: `source_illumination`, `sum_t |a|^2` with `a` the wavefields'
  second time derivative (the pseudo-Hessian of Shin et al. 2001, SPECFEM2D's first
  `APPROXIMATE_HESS_KL` kernel), and after its backward `receiver_illumination`,
  `sum_t |M^-1 dL/df|^2` (the adjoint wavefield). Both are shaped like the models, the way
  the gradient is: `w |J|` at each element node, and the interpolation's transpose for a
  grid model. On `impl='eager'` on the full tape, and on `impl='c'` for the 2-D C-PML paths
  (`AcousticPotentialSEM2D`, `ElasticSEM2DDeformed`, `AcousticElasticSEM2D`), there in the
  shot batch and under boundary saving, too. The eager and CUDA values agree to 1e-5; the
  sums are fp64. Turning it on does not move the gradient (`test/sem/test_sem_illumination.py`).
  Notebook 48 compensates an RTM image under a hill with it.
- **`ViscoElasticSEM2DDeformed`: SPECFEM2D's visco-elastic displacement scheme on the SEM.**
  `ElasticSEM2DDeformed` with SPECFEM2D's `ATTENUATION_VISCOELASTIC` as its Newmark path runs
  it: mechanisms on the bulk and the shear modulus (models `vp, vs, rho, Qk, Qmu`), element
  memories `e = A^2 e + B (s + A s_old)` of the dilatation, the deviatoric and the shear strain,
  and SPECFEM2D's defaults: the band `f_ref / 10 .. 10 f_ref`, the fit at `2 n_sls - 1`
  frequencies, and `vp`, `vs` read as unrelaxed (`"real_modulus"` is its
  `READ_VELOCITIES_AT_f0`). The C-PML layer attenuates too, as SPECFEM2D's visco-elastic C-PML
  does: there the memories take the stretched strains, a second copy of each on the band for
  `sigma_zz` and `sigma_xz`, the relaxation run on the stretched strain where SPECFEM2D expands
  the product in partial fractions (whose coefficients blow up where a relaxation frequency
  meets a C-PML `alpha` or `beta`). With the layer elastic the jump of the modulus at its edge
  reflected 5.1e-2 of the record (Qkappa 30, Qmu 20), attenuating 4.7e-4, below the elastic
  medium's C-PML (1.0e-3); against SPECFEM2D's visco-elastic C-PML the records agree to 2.5e-7
  and the two layers reflect alike, 9.3e-5 (`test/sem/specfem_compare/run_compare_absorb_visco.py`).
  `Q = inf` is the elastic step bit for bit, and all five models are differentiable. Against SPECFEM2D 8.1.0 the records
  agree to 7e-14, with `Qkappa = Qmu = 30` read at 15 Hz and with `Qkappa = 60`, `Qmu = 20`
  read as unrelaxed, where the velocity-stress class keeps 2e-4. Eager, 2-D; no boundary saving
  (dissipative).
- **`ElasticSHSEM2DDeformed`: SH waves on the SEM.** The out-of-plane displacement `u_y`
  (SPECFEM2D's `P_SV = .false.`), as the potential form's scalar equation with `vp = vs` and
  the density `1 / (rho vs^2)`: its operators, C-PML and exact boundary saving, a traction-free
  surface, models `vs, rho`, a force along y and the receivers `u_y`, `vy`. Against SPECFEM2D:
  1e-13 flat, 2e-12 under a hill. `PropSEM` reads whether an equation takes `density=` from its
  `takes_density` flag.
- **`ViscoAcousticPotentialSEM2D`: visco-acoustic attenuation on the SEM.** The potential
  form with SPECFEM2D's `ATTENUATION_VISCOACOUSTIC`: its weak form, node memories and Newmark
  update, models `vp, Qk` and `density=`, and the visco-elastic class's `f_ref`, `f_band`,
  `n_sls`, `n_fit` and `velocities` (with `"unrelaxed"`, SPECFEM2D's reading without
  `READ_VELOCITIES_AT_f0`, now on both classes). `Qk = inf` is the potential step bit for bit,
  and all models are differentiable. Against SPECFEM2D 8.1.0 the records differ by 2.8% at Q =
  30; its `update_memory_var_acous_weak_form` sets `dot_e1_old = dot_e1` inside the loop over
  the mechanisms, and with that line reproduced they agree to 8e-13. Eager, 2-D; no boundary
  saving (dissipative).
- **Curved internal interfaces on the 2-D SEM mesh.** `PropSEM(interfaces=[...])` takes
  internal interfaces like `topography`, one row per column and top-down, and its Quad9 mesh
  follows them: each layer gets whole element rows (`interface_elements`, by default shared
  by the layers' mean thicknesses), graded linearly between its two interfaces in every
  column, as SPECFEM2D's internal mesher grades the layers of its `interfaces.dat`.
  `element_layer` gives each element's layer and `layered_model(values)` a GLL-native model of
  one value per layer, which steps at the interfaces instead of being interpolated across
  them. The mesh is `sweep.sem.build_layered_quad9` (one layer is `build_topography_quad9`'s).
  Against SPECFEM2D's mesh of the same interfaces (a curved sea floor under water, a
  sinusoidal interface between two rocks) the records agree to 4e-12.
- **The 2-D SEM against SPECFEM2D to round-off.** `test/sem/specfem_compare/run_compare_box.py`
  runs the 2-D classes against SPECFEM2D built in double precision on the cases of
  `make_specfem_box_cases.py`: homogeneous media in a box with all four sides free, a Ricker,
  6000 steps, no shift or scale fitted. The elastic classes (`ElasticVSSEM2DDeformed`,
  `ElasticSEM2DDeformed`, `ElasticSEM2D`, the VTI and TTI ones) agree to 1e-13, and to 3e-12
  under a Gaussian hill. The acoustic ones (`AcousticPotentialSEM2D`, `AcousticSEM2D`) agree
  to 1e-12 in pressure, and `AcousticElasticSEM2D` (water over rock) to 2e-12. The
  visco-elastic class agrees to 2e-4 with SPECFEM2D's conventions; its memory variables'
  time scheme differs. The conventions the comparison matches are in the SEM guide's
  "Against SPECFEM2D".
- **`ElasticTTISEM2DDeformed`: tilted transverse isotropy on the SEM.** The VTI
  velocity-stress SEM with its symmetry axis tilted by a sixth model, `theta` (radians), by
  the closed-form Bond rotation of the FD `ElasticTTI2nd`, in its axes and with its sign of
  `theta` (`sweep.sem.thomsen_to_tti_cij`); the tilt adds the `c15`, `c35` couplings to the
  stress update. `theta = 0` is `ElasticVTISEM2DDeformed`, and all six models are
  differentiable, under boundary saving too. Both classes agree with SPECFEM2D's anisotropic
  elastic medium (material type 2) to 1e-13 in relative L2 with SPECFEM2D built in double
  precision: a VTI shale with a vertical and a 30° axis, a force and an explosion, all sides
  free (`test/sem/specfem_compare/run_compare_box.py`, cases from
  `make_specfem_box_cases.py`). Eager, 2-D.
- **`BiotVSSEM3DDeformed`: the Biot SEM in 3-D.** The velocity-stress Biot SEM of
  `BiotVSSEM2DDeformed` on the Hex27 mesh of `ElasticVSSEM3DDeformed` (flat or under a
  topography): the same coefficients, placement and exact per-node drag, the open-pore
  surface, the C-PML and `viscous_q`. Porous, dry and fluid elements (the fluid-solid faces
  through `sweep.sem.coupling.FluidSolidInterface3D`, a sea floor across the C-PML's x-y corner
  columns allowed), an isotropic drag, and the 2-D class's sources with `y` (the explosion the
  default). The drag's exponentials are stored per number of element copies at a node (1, 2,
  4, 8), about 2.7 kB per element. A dry solid is `ElasticVSSEM3DDeformed` to round-off; a point
  pore-pressure injection and the split source match the analytic fast and slow P waves to
  1% (`test/sem/test_biot_sem3d.py`). Eager. `sweep.sem._assembly.vector_divergence3` is the
  weak divergence of an element-local vector.
- **Moment-tensor sources in 3-D.** `PropSEM`'s stress source names (`sxx`, `syy`, `szz`,
  `sxy`, `sxz`, `syz`) on the 3-D elastic equations, which refused them: the point stamp
  `-M : grad(phi)` on Hex8 and Hex27 meshes (`PointStamp3D.moment_tensor_stamp`), shared on a
  face, an edge or a corner among the elements there by the solid angle each holds. On both
  backends, as in 2-D (the kernels see the assembled patterns).
- **`ViscoElasticVSSEM2DDeformed`: visco-elastic attenuation on the SEM.** The velocity-stress
  elastic SEM with the generalized standard linear solid of the FD `ViscoElastic`, which is
  SPECFEM2D's: models `vp, vs, rho, Qp, Qs` (velocities at `f_ref`), `f_ref`, `f_band` and
  `n_sls` through `equation_options`, element-local memory variables advanced with the
  trapezoidal rule, all five models differentiable, `Q = inf` the elastic step bit for bit.
  Against SPECFEM2D's `ATTENUATION_VISCOELASTIC` (Q = 30, half-space, its band
  `f0 / 10 .. 10 f0`) the records agree to 1.2% in relative L2 (1.7% for the elastic pair)
  and the visco/elastic spectral ratios to 0.1-1.1% in amplitude
  (`test/sem/specfem_compare/run_compare_visco.py`). SPECFEM2D's reading of the velocities
  at `f_ref` (`velocities="real_modulus"`: `rho v^2` is the real part of the modulus there,
  which puts the phase velocity about `3 / (8 Q^2)` above `v`) and its fit's frequencies
  (`n_fit = 2 n_sls - 1`) are options; with both, the records agree with SPECFEM2D's to 2e-4
  in a box with free sides, SPECFEM2D in double precision (0.9% with the defaults), the rest
  being the memory variables' time scheme (second order: 4.2e-5 at half the step). Eager,
  2-D; no boundary saving (dissipative). The band and the moduli factors of the fit are shared with the FD class
  (`sweep.equations.visco_elastic.gsls_band`, `gsls_moduli_factors`); the default band, a
  factor 12, is that of SPECFEM2D's original fit, not of its current one.
- **`ViscoBiotVSSEM2DDeformed`: Biot with an attenuating frame.** The models of
  `BiotVSSEM2DDeformed` and the frame's `Qp` (drained P modulus) and `Qs` (shear), the GSLS
  of the visco-elastic class on the frame part of the total stress; the pore fluid's share
  of `H` does not relax and `viscous_q` works as before. A dry solid gives the visco-elastic
  class's records to round-off, `Q = inf` Biot's bit for bit. Against SPECFEM2D's
  attenuating poroelastic frame (its `poroelastic_acoustic` example, `Qmu = 50`) the
  attenuation's effect agrees to 0.5% once three SPECFEM2D 8.1.0 bugs are fixed (memory
  variables updated four times per step, the previous step's term of their convolution
  zero, `mu` for `lambda` in the fluid routine's attenuated stress; as released its frame Q
  is about 115 for 50) and the frame takes SPECFEM2D's mechanisms
  (`test/sem/specfem_compare/run_compare_visco_biot.py`).
- **`PropSEM(impl='eager', memory=Ckpt(chunks=K))` checkpoints single shots and
  per-shot models**, which it used to refuse (`nshots=1`) or loop without
  (per-shot models): `sweep.sem.ckpt_rollout` keeps one state per `K` steps and
  rebuilds the tape chunk by chunk in the backward pass, for every equation --
  Biot included, which has no boundary saving. Peak memory is `nt/K` states plus
  one chunk's tape. On CUDA each chunk is captured once as a CUDA graph and
  replayed `nt/K` times, so the time loop does no host work per kernel: with
  the step compiled, the 2-D equations at 201x401 run 3.5-9x faster forward
  and 2.7-6.4x forward+backward than the compiled full tape (A100, H100, RTX
  6000 Ada). A step that cannot be
  captured (a host sync inside it) runs the same chunks without graphs after one
  warning; `sweep.sem.ckpt_rollout.USE_CUDA_GRAPHS = False` turns graphs off, and
  on CPU it is plain checkpointing. First derivatives only (`create_graph=True`
  raises); an `nt` that `K` does not divide is padded with zero-wavelet steps
  whose records are dropped. Records equal the full tape's; gradients agree to
  round-off.
- **SEM receivers that record a spatial derivative**: the strain (`exx`, `ezz`, `exz`; 3-D
  `eyy`, `exy`, `eyz` too), the rotation `curl(u) / 2` (`rot_y`; 3-D `rot_x`, `rot_y`, `rot_z`),
  their rates (`_t`, what DAS records), the pressure `-kappa div u` in an isotropic solid (`p`)
  and a potential fluid's particle displacement, velocity and acceleration (`ux`, `vx`, `ax`,
  ...) -- SPECFEM2D's `seismotype` 1-5 in both media. A receiver takes the gradient of its
  element's basis (`sweep.sem.GradientStamp`), so every backend, shot batch and memory mode
  records it, and the gradient reaches the model through it and its material factor.
  `AcousticElasticSEM2D`'s `vx`, `vz` and `p` take the medium of each receiver's element: the
  particle velocity and the pressure are the water's above the sea floor and the rock's
  below (before, `vx` read the dead solid field in the water and `p` the dead potential in the
  rock). On an element edge, where the gradient jumps, a receiver takes the elements that meet
  there by their shares, as a moment source is spread over them. Against SPECFEM2D inside an
  element: round-off (4e-12..5e-12); on an edge SPECFEM2D takes one element (1e-4..2e-4).
  `test/sem/test_sem_spatial_receivers.py`, notebook `47_sem_receivers`.

### Changed
- **`impl="eager"` on a 2-D grid on a GPU compiles its step with Inductor's
  `cpp_wrapper` and `combo_kernels`.** The first generates the host code of each compiled call as
  C++ instead of Python, the second launches the independent small kernels of a
  graph as one; both shorten the host's time per call, which sets the pace of
  a 2-D step of one shot. On an RTX 6000 Ada a gradient step under boundary
  saving on a 2-D grid of 1000² cells was 1.10 (acoustic) and 1.21 (elastic)
  times faster for one shot, 1.20 and 1.04 times for 8 shots, with the first
  compile about 10 s longer; in 3-D they gained nothing (the elastic gradient
  was 1 % slower) and stay off. Records and gradients stay those of the compile
  without them bit for bit, except that `combo_kernels` moves gradients by
  float32 round-off. They are on with Inductor for a 2-D grid on a CUDA device
  from torch 2.9 (`cpp_wrapper` only where torch finds the CUDA toolkit, `CUDA_HOME`, and not
  in the compile modes that replay CUDA graphs, which Inductor turns off for
  it) and off elsewhere; the new
  `EagerOptions(compile_cpp_wrapper=..., compile_combo_kernels=...)` turn
  them on or off, and raise where they cannot take effect. A step whose C++
  wrapper does not build compiles without it, with one warning, instead of
  running uncompiled; PropSEM refuses both options. See
  [Inductor options of the eager step](docs/user-guide/propagators.md#inductor-options).
- **`impl=None` on a GPU runs the velocity-stress and Biot SEM classes on the CUDA
  kernels** (`ElasticVSSEM2DDeformed` and its VTI, TTI and visco-elastic variants, the
  3-D velocity-stress classes, the Biot classes), with `Ckpt(chunks=64)` as their default
  memory; they ran on eager. `return_wavefield=True` needs `impl='eager'` there, as on
  the other classes with kernels.
- **One call of the compiled eager step advances several time steps
  (`EagerOptions(steps_per_call=...)`).** A call costs the host its Dynamo
  guards, autograd wrappers and kernel launches whatever it computes, and on a
  2-D grid that took longer than the step on the GPU: at 1000² the GPU was busy
  9-40% of the time. A call now runs `steps_per_call` steps with the source
  injection and the receiver sampling inside, on the full-tape and checkpointed
  rollouts; `None` (the default) takes 4 for rollouts with gradients on a 2-D
  grid on a GPU and 1 otherwise. Without gradients Inductor fuses the steps of
  a call into one another and, on a large grid, spends more GPU time than the
  call saves the host (a 2000² acoustic forward ran twice as long per step),
  so forwards keep one step a call unless `steps_per_call` is set. Calls start
  on multiples of `steps_per_call`; the steps before
  (until the inputs' `requires_grad` settles) and the last
  `nt % steps_per_call` run uncompiled, so no graph is compiled for a
  remainder, and a checkpoint chunk is rounded up to whole calls (boundary
  saving: see the next entry). `return_wavefield=True` advances one step a
  call; an explicit `steps_per_call > 1` is refused without `use_compile` and
  on PropSEM. Records and gradients move by float32
  round-off (~1e-6 relative). On an A100 and an RTX 6000 Ada a 2-D gradient
  (acoustic, elastic) runs 1.8-3.1x faster at 1000² and 1.1-1.5x at 2000²;
  its first compile takes 2-3x as long (11-19 s) and its peak memory grows
  3-12% under checkpointing. The receivers are sampled inside the compiled
  call at one step a call too (a one-shot 2-D forward at 1000² took 0.15
  instead of 0.21 ms a step), except in a rollout with gradients on CPML
  strips, the 3-D default, which compiles the step alone and samples them
  after it as before: there the sampling inside made Inductor split the
  step's backward into more full-grid kernels (a 160³ acoustic gradient took
  2.95 instead of 2.62 ms a step on an RTX 6000 Ada). Records are the same
  bit for bit either way. See
  [Several time steps per compiled call](docs/user-guide/propagators.md#steps-per-call).
- **Eager boundary saving takes `steps_per_call` steps a call too.** A call of
  `k` steps (`_BoundarySaveSteps`) stores the `k` steps' boundary rings
  together; its backward reconstructs the `k` inputs backwards in time and
  differentiates the `k` steps in one compiled call, each step at its own
  reconstructed input, so the gradient is autograd's and that of one step a
  call up to float32 round-off (records bitwise and gradients within 9e-7
  relative across Acoustic, Acoustic3D, Acoustic1st, Elastic, Elastic3D,
  DASMu, DASMu3D and ElasticVRR when both run the same ops). The forward runs
  the `k` steps as one graph, whatever the grid size. The default
  `steps_per_call=None` gives 4 for 2-D gradient rollouts on
  a GPU, as on the other rollouts. Peak memory grows with `k` (1000^2
  acoustic, 8 shots: 1.5 GB at `k = 1`, 2.5 GB at `k = 4`, 1.3 GB at `k = 4`
  with `cpml_strips=True`). On a 2-D grid the backward's reconstruction and
  re-run are one compiled call (2-10% faster there; on a 160^3 elastic grid
  one call ran 2-3x slower, so 3-D keeps two). On an RTX 6000 Ada at 1000^2,
  boundary saving on both sides and the stencil change below included, a 2-D
  gradient ran 3.6x faster at one shot (acoustic 1.83 to 0.50 ms a step,
  elastic 3.34 to 0.92) and 1.2-1.4x at 4-8 shots; with `cpml_strips=True`
  from 4 shots, 2.4-2.5x (acoustic) and 1.65x (elastic), within 1.71x and
  1.21x of `impl='c'` at 8 shots. A 160^3 gradient with `steps_per_call=2`
  ran 1.45x (acoustic) and 1.56x (elastic) faster than before. The first
  compile of a gradient takes 45-150 s. Previously an explicit
  `steps_per_call > 1` was refused with boundary saving.
- **The compiled slice stencils differentiate as their transpose.** The
  zero-padded correlation that `torch.compile`d eager steps evaluate
  (`sweep.operators.torch._apply_taps`) is an `autograd.Function` whose
  derivative is the same correlation with the offsets negated, gathered on the
  field's own grid, instead of autograd's `slice_backward` per tap into a
  padded zero buffer that Inductor could not fuse. The gradient moves by
  float32 round-off; second derivatives go through the same form and
  `torch.func.vmap` still applies (there is no forward-mode rule: Dynamo
  cannot trace a custom `jvp`). An 8-shot 1000^2 boundary-saving gradient with
  CPML strips ran 1.12x (acoustic) and 1.20x (elastic) faster (RTX 6000 Ada).
- **3-D eager runs keep the CPML memory fields on the absorbing strips by
  default.** `EagerOptions(cpml_strips=None)`, the new default, picks the strips
  on a 3-D grid and the full grid on a 2-D one or with `topography=`;
  `True`/`False` force a layout, and `True` with `topography=` now raises at
  construction. The strips cut a forward's peak memory by about a third
  (`abcn=50`, A100: `Acoustic3D` 1.08 to 0.76 GiB at 128^3 inner cells,
  `Elastic3D` 3.99 to 2.59), boundary saving's by 18-32% and checkpointing's,
  the eager default, by 3-9%.
  Their speed depends on the share of the grid the layers take, since each
  strip adds kernels: with `abcn=50` the acoustic forward is faster from about
  128^3 inner cells (0.69-0.92x the full-grid time per step on A100 and V100)
  and its gradient from about 160^3 (0.88-0.96x), while smaller grids and the
  elastic gradients run 5-50% slower; pass `cpml_strips=False` there. In 2-D,
  where a step of 1000^2 cells is launch-bound, the strips cost more than they
  save and stay off by default; a 2-D gradient of about 3 million cells a call
  (shots x padded grid cells) or more runs faster with `cpml_strips=True`
  (measured in the user guide, *C-PML memory fields on the absorbing strips*).
- Snapshots (`return_wavefield=True`) and `last_wavefields` hold full-grid
  fields under CPML strips too: the strips are put back in place, and
  uncompiled the fields equal the full-grid layout's bit for bit.
  `return_wavefield=True` with strips used to raise.
- Staggered CPML strips add the new memory field to the derivative zero-padded
  instead of replacing the derivative on the strips with a `where`, so the
  step's main kernel reads the strips back instead of retaking the derivative
  on them. Inductor had fused that retake into one kernel of 171 loads for the
  order-8 `Elastic3D`: its forward with strips took 10.4 ms of GPU time per step
  against 1.2 ms on the full grid (A100, 64^3 inner cells), now 1.65 ms; on a
  V100 the wall time went from 7.9x to 0.97x the full grid's. The order-4
  forward is 8-12% faster too. The second-order strips read their inner-cell
  mask as a constant built with the layout, not one built in every step.
- **Uncompiled eager rollouts keep far fewer fields for the backward.** A conv's
  autograd node saves its input, which only the kernel's gradient reads, and the
  stencil kernels are constants. The uncompiled stencils (`apply_kernels`,
  `apply_kernels_3d`, `separable_d2_2d`/`3d`, the kernel path of `gradient`) now
  run through an autograd Function that keeps the kernel alone; its backward is
  the transposed conv with the same kernel and padding, the call the conv's own
  backward makes, so records and gradients are bitwise those of before, on every
  eager equation, CPU and CUDA, forward mode included; second derivatives as
  well, or within float round-off where the conv's double backward sums in
  another order (a GitHub CPU runner, float64).
  Fields kept per time step: 2-D Acoustic 5 -> 2, Elastic 13 -> 8, AcousticVTI
  7 -> 5, ViscoElastic 20 -> 15, Elastic with a free surface 24 -> 12; 3-D
  Elastic 28 -> 19. A 1000x1000 full-tape gradient (V100) peaks at 2.8 GB
  instead of 6.8 (Acoustic) and 10.9 GB instead of 17.6 (Elastic), and takes as
  long as before or up to 5% less there. Where the step is bound by host overhead
  (2-D 1000x1000 on an RTX 6000 Ada) the acoustic equations run 7-11% faster
  (`gradient` lost four CopySlices nodes) and the staggered-grid elastic ones
  (Elastic, DAS) 12-14% slower: each derivative is now a Python autograd node,
  ~25 us more host time per call than the conv's. A second derivative
  (`create_graph=True`) no longer runs cuDNN's weight-gradient kernels, which
  the conv's own double backward did: a Hessian-vector product on 2-D Acoustic
  (100x200, 4 shots, V100) takes 6.0 ms a step instead of 84. Compiled steps
  (the slice form) and runs without gradients are unchanged; `ElasticTTI`'s
  rotated-staggered-grid derivative keeps the plain conv.
  `sweep.operators.torch.STENCIL_BACKWARD_KEEPS_KERNEL_ONLY = False` restores
  the plain conv everywhere.
- **An uncompiled eager rollout makes the step's time-invariant coefficients
  once.** `dt / rho`, `vp**2 * dt**2`, `dt * mu`, the VRZ/VRR first-derivative
  coefficients, the free-surface Robertsson factors and the like depend on the
  models and `dt` alone; recomputed every step, each was a new field that the
  product reading it kept for the backward. The full-tape and checkpointed
  rollouts now hand the step a dict in which they are made once (per rollout,
  per checkpointed chunk); every eager equation that had such coefficients
  takes it (all but the qP family, which has none). Records are bitwise the
  same; gradients move by rounding (at most 2e-6 of their maximum across 60
  cases, second derivatives included). Fields kept per time step, on top of
  the stencil change above: 2-D Acoustic 2 -> 1, Elastic 8 -> 5 (12 -> 5 with a
  free surface), ElasticVRR 26 -> 6, AcousticVRZ 14 -> 4, DASZhao 10 -> 4,
  ViscoElastic 15 -> 6; 3-D Elastic 19 -> 10. Compiled steps and boundary
  saving compute the coefficients in place, as before. An equation opts in
  with `takes_invariants = True` and `sweep.equations._invariants.once`;
  `sweep.propagator._torch_eager.HOIST_INVARIANTS = False` turns it off.
- The CPU test workflow runs as four parallel jobs (`sem-biot`, `sem`,
  `rest`, `skill`), each with its own executed-test floor, instead of one
  job.
- **Every new compile of the eager step is announced, on PropTorch and PropSEM.**
  The line saying that `torch.compile` is compiling used to print once per
  process, so a later grid size, shot count or first gradient call compiled again
  in silence -- seconds to minutes that looked like a hang. Now each new compile
  prints one line on stderr before it starts, naming the step and the size
  (`for 2 shots on a 201x401 grid`), the dtype when it is not float32, and
  `(with gradients)` when autograd records the call -- grad mode on and a tensor
  that requires grad; a forward in grad mode with no model that requires grad
  compiles no backward and is not called one. Compiles are told apart as Dynamo
  tells them apart: by the inputs' shape and dtype and, when the call records, by
  which tensors require grad (the models, and on PropSEM the coefficients the
  step reads off the equation), so inverting vp alone and then vp and vs is two
  compiles, two lines. The first line of the process still says how long a
  compile takes and how to turn compiling off. The check costs ~2 us of Python
  per compiled step.
- **`PropSEM(impl='eager')` compiles one graph per rollout with gradients on its
  full tape and vmapped shot batch, as PropTorch's eager rollouts do.** Dynamo
  guards on each input's `requires_grad`, and the fields pick it up over the
  first steps of a rollout that differentiates the models: compiled, each of
  those steps built a graph of its own, three per rollout. They now run
  uncompiled until the flags stop changing (`sweep._compile.SettledSteps`, the
  class PropTorch's eager path already used). On an A100 at 201x401 (nt=1000,
  fresh compile cache) the first FWI call takes 16 s instead of 24 s (elastic,
  one shot, or four shots under `Ckpt(chunks=8)`) and 9 s instead of 15 s
  (acoustic); later calls take as long as before (within 0.7%), and records and
  gradients agree with the uncompiled step as before (FP32: records 1.5-2.8e-6,
  gradients 2-5e-7 relative). Single-shot `Ckpt()` and boundary saving are
  unchanged.
- **`PropSEM` compiles its eager step by default and takes PropTorch's
  `eager_options=`.** `PropSEM(..., eager_options=EagerOptions(...))` (or its
  dict) replaces the call argument `compile_step`, which is gone (passing it
  raises a `TypeError` that names the replacement): `use_compile` -- default
  `True`, as on PropTorch -- `compile_mode`, `compile_dynamic`, `compile_backend`
  and `compile_fullgraph` reach `torch.compile`, behind the FD eager path's guards
  (no `torch.compile`, torch 2.3's Inductor on CPU, Dynamo's recompile cap, the
  compile notices). As on PropTorch, `eager_options` with
  `impl='c'` raises; `store_last_wavefield` and `cpml_strips` have no SEM
  counterpart and raise when set, and so do the CUDA-graph compile modes
  (`reduce-overhead`, `max-autotune`) together with `BoundarySaving()` or `Ckpt()`
  on CUDA, whose rollouts step again from kept states that those graphs overwrite. The low-level `SEMPropagator.forward` takes
  `compile_options=` (an `EagerOptions`; `None` runs the step as it is) instead of
  `compile_step`. Compiled, the step is ~3x faster on GPU and CPU, at a one-time
  compile per grid size (seconds on a GPU, ~40 s on a CPU);
  `EagerOptions(use_compile=False)` gives the uncompiled step, unchanged.
- **`PropSEM` takes PropTorch's default memory strategy.** Without `memory=`,
  `impl='c'` now saves the boundary where its kernels run boundary saving (the
  five C-PML paths; the full tape elsewhere) and eager checkpoints -- in chunks
  of 8 steps rather than `Ckpt()`'s 100, since the CUDA-graph rollout rebuilds
  its graphs per call at a cost that grows with the chunk (2-D elastic, one
  forward+backward call on an A100: 201x401 with nt=1000, 0.51 / 0.81 s for
  chunks of 8 / 100 and 1.24 s on the full tape; 801x1601 with nt=500, 1.27 /
  2.16 s and 1.13 s). Both used to take the full tape. `memory_strategy` reports
  `'full'` / `'boundary'` / `'ckpt'`, as on PropTorch. The default gives way to
  the full tape for `return_wavefield=True` and for the CUDA-graph compile modes;
  `create_graph=True` and a second backward (`retain_graph=True`) need an
  explicit `memory=Full()`, and say so.
- **The compiled SEM eager step is faster on GPU: 1.05-1.75x forward,
  1.07-1.57x forward+backward (H100), most on 3-D and large 2-D** (A100,
  forward | forward+backward: 1.4x | 1.35x for 2-D elastic at 801x1601, 1.4x |
  1.2x for 3-D elastic at 49^3). Every GLL reference-derivative contraction now
  goes through
  `sweep.sem._assembly.contract`: the einsum it was, except while
  `torch.compile` traces for CUDA, where it is written as `n` broadcast
  multiply-adds that Inductor fuses with the gather before it and the pointwise
  work after it (an einsum lowers to a batched GEMM on `n x n` blocks that
  nothing fuses with). Uncompiled runs and CPU compiles keep the einsum, which is
  faster there (unchanged within 3%). At medium 2-D sizes the compiled step is
  still bound by its per-step host work, which caps the gain there. Records and
  gradients agree with the uncompiled step to fp32 round-off for every equation
  family.
- **The eager elastic C-PML steps can run the C-PML on the absorbing band only.**
  `ElasticSEM2DDeformed`, `ElasticVSSEM2DDeformed` (and `ElasticVTISEM2DDeformed`)
  and `ElasticVSSEM3D` (and `ElasticVSSEM3DDeformed`) run the bare operators on
  the interior elements and the C-PML terms on the band elements, as the CUDA
  kernels do, and keep their memory fields on the band rows only: new field
  layouts `band` / `band2`, whose rows are `eq.band_elements` (none without a
  C-PML). The split costs a second element pipeline per step, so it is on where
  it pays: by default on meshes with at least 40000 interior elements (801x1601
  has 80000), compiled or not, and wherever `build(..., band_split=True)` (also
  through `equation_options`) asks for it. On smaller meshes it costs more than
  it saves, compiled too, and on the velocity-stress forms even replayed as CUDA
  graphs. Unsplit, every element is on the band and the step is the full-mesh
  one. Same arithmetic (fp64: records and gradients agree to 1e-15). 2-D elastic
  at 801x1601 (A100), forward | forward+backward: uncompiled 2.6x | 2.0x faster,
  compiled 2.2x | 1.3x on top of the unrolled contractions. A memory field is
  indexed by band row, not by element id.
- **The displacement-form elastic C-PML centres its damping in time, as SPECFEM2D's Newmark
  scheme does** (`ElasticSEM2DDeformed` and the classes on it: `AcousticElasticSEM2D`'s solid,
  `ViscoElasticSEM2DDeformed`). The mass the step divides by takes `dt/2` times the
  acceleration term's velocity coefficient (`La1 = d_x + d_z`), so the damping no longer acts
  on the backward-difference velocity alone. The records now agree with SPECFEM2D's C-PML to
  7e-7 of the record on its half-space (2.5e-4 before), and the layer reflects 1.6e-4 there
  (2.9e-4 before); a two-element layer 2.2e-3 instead of 6.4e-3
  (`test/sem/test_sem_vs_cpml_absorption.py`). The profile keeps the `dt` it was built for.

- **`PropSEM`'s wavelet is the pressure source on the potential forms too**
  (`AcousticPotentialSEM2D`, the water of `AcousticElasticSEM2D`). The
  wavelet used to drive chi, so `p = -d2(chi)/dt2` came out two time
  derivatives off the pressure form's `p` (`AcousticSEM2DDeformed`, FD's
  `Acoustic`) for the same wavelet. The records are now those of chi driven by
  minus the wavelet's double time integral: the same `p` waveform on both
  forms (to 2e-13 in float64; the amplitude still differs by `rho*c^2` at the
  source), and the fluid-solid `vx`/`vz` stay consistent with that `p`.
  **`chi`, `u_x`, `u_z` and `vx`/`vz` records change accordingly**: observed
  data made with the old convention no longer matches. PropSEM takes them from
  the run the wavelet itself drives -- linear and time-invariant, that run
  records `p` sample for sample and the rest are its running integrals -- so
  float32 never differences a double integral; `return_wavefield=True` runs
  the integrated wavelet instead, so the snapshots are chi and u.
  `SEMPropagator` and `SEMModelParallel` keep the wavelet as the chi source.
  **PropSEM's default records are now the pressure** on these forms --
  `['p']` on the potential form, `['p', 'vx', 'vz']` on the fluid-solid class
  -- as on PropTorch: chi is now a near-DC tail (fitting it, bf16/int8
  boundary-saving gradients fell to cos 0.1-0.4 of the full tape's). Ask for
  `chi` / `u_x` / `u_z` by name; the low-level propagators still record chi.

### Removed
- Scripts under `test/` that are not tests: 85 standalone files that no test, CI job,
  gate script, doc or notebook ran -- the `test/efficiency/` benchmarks, one-off
  gradient, DD and memory diagnostics, the SEM smoke scripts and the early SPECFEM2D
  hill and fluid-solid comparison drivers (superseded by `run_compare_box.py` and
  `run_compare_absorb.py`) -- and five benchmark and diagnostic scripts,
  `dd_nccl_bench.py`, `dd_nccl_elastic_bench.py`, `dd_session_bench.py`,
  `dd_cross_gpu_diag.py` and `save_backend_gradient_plots.py`. The multi-GPU checks, the SPECFEM comparisons and
  the gradient suites that docs and the gate scripts run stay.
- Stale examples and repo-root leftovers: `examples/FWI/` and
  `examples/LSRTM/` (unlinked scripts, covered by the notebooks; their
  `_shared` helpers `configure_overthrust`, `fwi3d_overthrust`, `mpi_utils`
  and `models/overthrust/extract_2d_slice.py` go with them),
  `examples/ipynb/`, `convert_3d_overthrust_vites_to_npy.py` (superseded by
  `download_3d_overthrust.py`), `real_traced_model_demo.py` (needed an
  unshipped model), the root `figures/` directory (the Overthrust card script
  moved to `utils/`), `.vscode/`, `utils/generate_eq_init.py` and its backup
  (replaced by the equation registry), and `gate/sync_to_ibex.sh` /
  `gate/run_reorder_checks.sh`.  Notebook outputs `fwi_ddp.py` and
  `15_wavefield_topography.gif` are no longer tracked.
- **The elastic image-method topography staircase** (`topography=` on
  `Elastic`, `Elastic3D`, `ElasticBorn` and `ElasticVRR`).  It grew without
  bound on slopes steeper than about 0.5 (5 m grid, 45° ridge: 2000× the
  physical amplitude by 0.35 s, on every stencil order), and holding its air at
  zero only stopped the growth: against a refined spectral-element reference
  its error rose with refinement (relative L2 0.52, 0.63, 0.80 at 5, 2.5,
  1.25 m).  These equations now refuse `topography=` on every impl and name the
  alternatives, `ElasticCurvilinear` (2-D) and `PropSEM`.  The acoustic
  staircase (`Acoustic`, `Acoustic3D`, `ViscoAcoustic`) is unchanged, and so is
  every flat free surface.  The per-column surface-row paths in the elastic
  CUDA kernels are gone too (records and gradients are bit-identical); the
  `Elastic` and `ElasticVRR` CUDA drivers now refuse topography rows
  themselves.  `examples/wavefields/topography/elastic2d_hill_demo.py` and the
  two elastic topography bitgate configs go with it, and notebook 15 now runs
  the acoustic image method and curvilinear grid, and elastic on the
  curvilinear grid only.

### Fixed
- **`impl='eager'` refuses the compile modes that replay CUDA graphs on a CUDA
  device.** With `EagerOptions(compile_mode='reduce-overhead')` or
  `'max-autotune'`, a replay writes over the outputs of the one before, which the
  time loop still holds (the records of the earlier calls, the fields a call
  passes on), and every rollout failed reading them back ("accessing tensor
  output of CUDAGraphs that has been overwritten by a subsequent run"): under
  `Full()` and `Ckpt()` with and without gradients, and under boundary saving
  without them (with them it ran, and the same propagator then failed on a
  forward). They raise `ValueError` at construction now, whatever the memory;
  `'max-autotune-no-cudagraphs'` keeps the autotuning. On a CPU, and with
  `use_compile=False`, nothing changes.
- **One shot no longer compiles the eager step again for every new `nt`.** The
  time-major wavelet of one shot, `(nt, 1)`, counts as contiguous, so
  `.contiguous()` kept each step's sample at a stride of `nt`, which Dynamo
  guards on: a rollout with a new `nt` (a multiscale stage, another record
  length) compiled its calls again, the boundary-saving backward included
  (20-100 s each on a GPU), and past Dynamo's recompile limit the step ran
  uncompiled. Each sample is a contiguous copy now; several shots compiled once
  before as well.
- **Eager and JAX boundary saving: a source on the boundary ring, and per-face
  PML widths.** Each step's ring was saved before the source went in, and the
  reverse pass puts the saved ring back after re-injecting the source, so a
  source on a ring cell was dropped from the reconstruction. Under a free
  surface the top face has no PML and its ring lies inside the model, at depth
  `abcn - halo` .. `abcn - 1` cells: a source there gave a vp gradient 17-21%
  off the full tape's (relative L2; 1e-3 a few cells away). The ring is now
  saved after the source, as `impl='c'` saves it. With per-face PML widths (a
  per-edge `abcn`) every face had its ring at `max(abcn) + halo`, inside the
  model on a thinner face, and the strip between the ring and the thin PML
  came back wrong (vp gradients 2-22% off, the error next to that face). A face
  with a damping ramp now has its ring in the innermost cells of its own ramp;
  a face without one (a free surface) keeps it at `abcn + halo`, where the band
  also holds the gradient nearer the full tape's next to the surface. Records
  are unchanged, and so are the gradients, bit for bit, unless a source sits on
  the ring or a face is thinner than another.
- **`ElasticVRR` on `impl='eager'`: uncompiled steps compute the velocity gradients
  of their own models.** `vp_x, vp_z, vs_x, vs_z` were cached on `id(vp), id(vs)`, and
  an id is reused once its tensor is freed: boundary saving's backward makes new
  model leaves every call, so a call could take an earlier call's gradients, whose
  graph ends at that call's leaves. At 4 steps a call run uncompiled (what Dynamo
  does when it falls back to eager) the vp and vs gradients were 19% and 55% off
  the full tape. They are now made once per rollout through the time-invariant
  dict, and in every uncompiled step that has none; compiled steps compute them
  in place, as before.
- **The compiled eager step of the 3-D deformed SEM classes with a C-PML is faster:
  forward 4.3-6.4x on `AcousticSEM3DDeformed`, 1.9-3x on `ElasticVSSEM3DDeformed`,
  `ElasticVTISEM3DDeformed`, `ElasticTTISEM3DDeformed` and
  `ViscoElasticVSSEM3DDeformed`, 1.4-1.9x on `BiotVSSEM3DDeformed`; forward +
  backward 3-4.4x, 1.5-2.2x and 1.1-1.3x (A100, RTX 6000 Ada).** Since the GLL
  contractions were unrolled for the CUDA compile, the 3x3 inverse Jacobian made
  every physical derivative read all three reference directions, and Inductor put
  the gathers of every field into each multiply-add: one kernel with ~100 indirect
  loads per GLL node that spilled its registers (93% of the acoustic step), on the
  RTX 6000 Ada a compiled step slower than the uncompiled one. These classes keep
  the batched einsum (`contract(..., unroll=False)`), which gathers each field once.
  `AcousticSEM3DDeformed` at 97^3 on an A100: 12.6 -> 2.7 ms per step forward and
  27.0 -> 8.6 forward + backward, from 24x the CUDA kernels to 5.5x. Records and
  gradients unchanged to fp32 round-off; the other classes keep the unrolled form.
- **Eager boundary saving saves no boundary on a call without gradients.** A
  call autograd does not record (grad mode off, or no model or wavelet
  requiring grad: a line search, forward modelling) has no backward pass; under
  `memory=BoundarySaving()` it now runs the plain forward steps, as `Full()`
  and `Ckpt()` do and as `impl='c'` does, instead of saving every step's
  boundary ring into storage sized for the whole rollout. Its record is
  `Full()`'s bit for bit, and it runs as fast: on an A100 at 1000^2 cells and
  1000 steps 1.1x (elastic, 8 shots: 0.94 to 0.82 ms a step) to 2.3x (elastic,
  one shot: 0.65 to 0.28) faster than before, and the peak memory of an 8-shot
  elastic call fell from 2.3 to 0.96 GB. `return_wavefield=True` still raises
  under boundary saving.
- **`impl='c'` runs under `torch.inference_mode()`.** Every call there failed
  in the C adapter before the forward ran ("Inference tensors do not track
  version counter"): the host copies of the source and receiver field indices
  and of the checkpoint steps are cached while the tensor keeps its identity
  and its `_version`, and a tensor made under inference mode has no version
  counter. Those tensors are now copied on every call and never cached; the
  record is the `torch.no_grad()` one bit for bit, with boundary saving, the
  full history and checkpoints (`test/test_c_inference_mode.py`).
  `ModelParallel` still takes its first call outside inference mode, as
  before.
- Eager `PropTorch` under chunk checkpointing (the eager default): two forwards
  before one backward (`(loss1 + loss2).backward()`, or a `torch.no_grad()`
  call between a forward and its backward) raised "modified by an inplace
  operation". The initial wavefield was a workspace cached on the propagator
  and zeroed in place by every call, and the first chunk's checkpoint saves
  it. It is allocated per call now
  (`test/test_eager_two_forwards_one_backward.py`).
- Eager `register_gradient(mode="gradient")` raises `ValueError` on a
  propagator built with `Ckpt()` (the eager default), which ignored the
  registration and returned the autograd gradient, and with `BoundarySaving()`,
  which kept the full tape anyway; and on a recording call with `adj=True` or
  `return_wavefield=True`, which also ignored it. The hooked path runs only on
  a call that records a registered model's gradient: under `torch.no_grad()`,
  or with no registered model requiring grad, it kept per-step snapshots and
  returned a record that required grad. On it the other models and the
  wavelet lost their gradients (`Acoustic1st` with `vp` registered: no `rho`
  gradient), and the source went into the first `source_type` field only
  (`Elastic` with `['sxx', 'szz']`: the record off by its own size).
  `imaging()` runs under `torch.no_grad()`.
- `PropSEM(impl='eager', memory=BoundarySaving())`: a call that no backward
  can run through (under `torch.no_grad()`, or nothing requiring grad) built
  the reconstruction state and saved the band every step. It takes the
  full-tape runner now, which keeps nothing then, and gives `Full()`'s record
  exactly.
- `PropSEM(impl='c')`: the CUDA drivers decided inside their autograd
  Functions whether a backward would run, from `requires_grad`, which does not
  see an outer `torch.no_grad()`. A wavelet that required grad, passed in under
  `no_grad`, kept the save-all tape or the C-PML band ring: 2.9 to 160 MiB more
  on the test cases. The call site decides now
  (`sweep.sem.cdrive.apply_recorded`), for all 17 Functions
  (`test/sem/test_sem_c_nograd_tapes.py`).
- `PropSEM(impl=None, memory=BoundarySaving())` runs on eager where the CUDA
  kernels have no boundary saving (`ElasticVSSEM3D`, `ElasticSEM3D`,
  `ElasticSEM3DDeformed`, the 2-D paths without their C-PML) instead of
  taking the kernels and refusing at the first call; `impl='c'` refuses it
  when the `PropSEM` is built. The docs listed eager boundary saving for every
  equation; the dissipative ones (visco-elastic, visco-acoustic, Biot) refuse it.
- `PropSEM(memory=BoundarySaving())` on the velocity-stress SEM
  (`ElasticVSSEM2DDeformed` with VTI/TTI, `ElasticVSSEM3D`,
  `ElasticVSSEM3DDeformed`) gives the full-tape gradient with a C-PML. The ring
  holds only the velocities, so the reverse march rebuilt the band's stresses
  and memories wrong, and the per-step recompute put a wrong velocity on the
  band nodes, which the model gradient next to the band reads: 4e-4 of the 3-D
  vp gradient (9e-5 with a free top), 1e-3 of the 2-D rho gradient, records
  unaffected. Not from the C-PML recursion change: dev before it was off by as
  much. The recompute now pins the velocity update's output to the stored ring
  (`boundary_saving.pin_to_ring`, straight through, so the adjoint is
  unchanged): fp64 round-off in both band splits and both ring modes, with
  topography and in the multi-shot rollout. No extra storage; `impl='c'` has no
  boundary saving for these classes.
- `PropSEM(memory=BoundarySaving())` on the scalar SEM (`AcousticSEM2D`,
  `AcousticSEM2DDeformed`, `AcousticPotentialSEM2D`, `ElasticSHSEM2DDeformed`,
  `AcousticSEM3D`, `AcousticSEM3DDeformed`) gives the full-tape gradient with a
  C-PML, eager. The model's edge nodes read the C-PML memories of the band
  element next to them, and the per-step recompute had them wrong: zeroed in
  the classes whose ring holds `u` only, one step stale in the three that ring
  them (restored from `ring[i - 1]`; a step's input memories are `ring[i]`).
  The vp gradient on the model's edge line was off by 4e-5 (2-D), 3e-6 (3-D),
  7e-5 (potential form) and the SH rho gradient by 1e-6, up to 1e-3 with the
  source next to the band; records unaffected. The memories now come back at
  the right level, and the classes that do not ring them pin `u`'s update to
  the ring (`boundary_saving.pin_to_ring`) instead: ringing them would cost
  2.6-3.4x the ring and, with a bf16 or int8 ring, land 2-15x further from the
  full tape. fp64 round-off in both ring modes, with a free top, topography,
  source encoding, in the multi-shot rollout and compiled; with a bf16 or int8
  ring 4-16x closer to the full tape than before. `impl='c'` was not affected
  (its ring is read at the right level).
- The custom-gradient forward (`register_gradient`) on the eager propagator
  read its receivers from the wrong field under CPML strips when the receiver
  field comes after strip-held fields, as `sh1` does in `AcousticBorn` and
  `AcousticBorn3D`: the record came out zero. It steps full-grid fields, so it
  now finds them by name instead of by their position in the strip-held state.
- `PropSEM(memory=BoundarySaving(), bs_band_mode='rim')` gives the full-tape
  gradient over the whole model, its edge included. The rim was meant to be
  the band's inner edge, but the C-PML derivative profiles (`dbx_dx` ...,
  from `np.gradient`) are non-zero on the model's edge line too, so it was
  counted as band: rim zeroed it in the reverse march and stored the model's
  outermost element instead. That element's gradient was wrong (2-D acoustic,
  `impl='c'`: `1 - cos` 0.35 uncropped; elastic 1.7e-5; the fluid-solid
  "2e-5 plateau" was the same error). Rim now stores the band's first element
  layer, the one touching the model. Band elements are told apart by `b_*` /
  `node_in_pml` at their interior GLL nodes only, since on the band's inner
  face the profile is zero only up to round-off (`b_z = -3.8e-16` measured),
  and a model node the ring does not restore is no longer zeroed in the
  eager reverse march (`np.gradient` leaves 1e-17 on `dbz_dz` one node into
  the model). Every path that
  takes the option -- 2-D acoustic, elastic and fluid-solid, 3-D acoustic, DD
  strips, `impl='c'` and eager -- is at round-off, uncropped. It stores more
  than the old rim (2-D acoustic peak 50 → 114 MB on a 97×193 grid, against
  553 MB for the full band), and the C-PML memory rings are no longer empty
  in rim mode. No kernel change.
- `AcousticTariq`: `vv` is the vertical velocity, as the step uses it
  (`vv**2`); the docstring and model spec called it the squared velocity.
  `AcousticAniso(method='alkhalifah', symmetry='tti')` now raises
  `NotImplementedError` and the `AcousticTTIAlkhalifah` alias is gone: the
  class has no tilt parameter, so both silently solved the VTI equation.
- `PropSEM(memory=BoundarySaving(storage_dtype='fp16'))` is refused on both
  backends. The eager ring stored a plain fp16 cast, which flushes the SEM's
  small field values to zero: 2-D elastic gradients came out 15% and 2-D
  acoustic 6% off the full tape (bf16 5e-5, int8 2e-4); `impl='c'` refused it
  already, under an internal option's name. PropTorch's fp16 ring is
  block-normalised; the SEM rings take fp32, bf16 and int8.
- `AcousticCurvilinear`: the z-direction CPML terms took their derivatives
  with `dh`, but one computational row is `D(x) d_eta` metres, so whenever the
  surface lies below row 0 the stretched-coordinate correction was off by
  `(row_spacing / dh)**2` and the bottom layer reflected (1.2% of the record
  with the surface at row 40 of 121, 3% at row 60, against 1e-4 without
  topography). They now use `beta d/deta`. The interior is unchanged;
  `ElasticCurvilinear`'s first-order CPML is scale-free and was right.
- `PropSEM(impl='eager')` compiling its step no longer fails when
  `torch.compile` cannot build it, e.g. on CPU with g++ older than 10
  (`InductorError: CppCompileError`, `-std=c++20` rejected). As on the FD eager
  path, the step runs uncompiled -- single-shot loop, boundary saving and the
  vmapped multi-shot step alike -- with one `RuntimeWarning` that quotes the
  compiler's error and names `torch._inductor.config.cpp.cxx` / `$CXX`; later
  calls skip the attempt. Results are the same; errors of the step itself
  still propagate.
- **The velocity-stress SEMs' C-PML no longer grows at late times.** Its outer edge was left
  traction-free, and from about 2.5 s on a record rose 2 to 300 times every 1.25 s, in 2-D and
  3-D, with or without a free surface (`ElasticVSSEM2DDeformed` and the VTI class built on it,
  `ElasticVSSEM3D`, `ElasticVSSEM3DDeformed`, `BiotVSSEM2DDeformed` with its
  anisotropic-drag class). The outer edge is now pinned at zero velocity, the rigid wall the
  displacement form's C-PML has (`sweep.sem.cpml.elastic_vs.pin_outer_edge`), on both
  backends and in the boundary-saving recompute: over 10 s the records decay to 1e-6 of the
  direct wave or below, and the absorption is unchanged (`test/sem/test_sem_absorption_gates.py`).
  A C-PML side between two free faces -- free top, left and right over an absorbing bottom, or
  free top and bottom with absorbing sides -- still grows, since the solid between the faces is
  a plate whose backward-propagating Lamb modes any PML amplifies, and `PropSEM` refuses it on
  these classes (`ValueError`; `test/sem/test_sem_cpml_outer_edge.py`).
- The velocity-stress SEMs' C-PML (`ElasticVSSEM2DDeformed`,
  `ElasticVTISEM2DDeformed`, `BiotVSSEM2DDeformed`, `ElasticVSSEM3D(Deformed)`,
  eager and `impl='c'`) recursed on the stress -- and Biot's flux -- and then took
  the weak-form derivative of `sigma + psi`: `d(psi*sigma)` instead of
  `psi*d(sigma)`. The extra term in the profile's gradient sent a low-frequency
  echo back from the top of the layer, the same size at any resolution (0.34 % of
  the record peak through a 200 m layer at 12 Hz; FD 0.01 %). The recursion now
  runs on the assembled divergence split by derivative direction (the
  mass-weighted nodal derivative, inter-element jumps included): 0.018 % there,
  and the late tail decays to 1e-6 instead of 1e-4. In 3-D the body-wave echo
  drops 20-70x; a surface wave meeting a side layer under-resolved (40 m
  elements at 30 Hz) still reflects as much as before. `impl='c'` keeps these
  memories per node (`Sem3dVsForwardInput.a_n`, `.P`; backward `.lP`): a core
  built from this tree is needed.
- `ViscoAcoustic` with `dh=(dz, dx)` and `dz != dx`: the spectral attenuation
  and dispersion filters (eager and `impl='c'`) built |k| from dz on both axes,
  so the x wavenumbers were scaled by dz/dx and the Q response depended on the
  grid orientation (a transposed problem's record differed by 28%). They now
  use both spacings; uniform grids are unchanged.
- The last sample of a `PropSEM` derived receiver component no longer spikes.
  `p = -d2(chi)/dt2` and `vx`/`vz` = `du/dt` repeated the last base sample
  past the end of the record, so the final `p` sample was a first difference
  over `dt^2` (717 after 103 on a 97x193 potential-form run) and the final
  velocity sample half a step off. The end is now the second-order one-sided
  difference: a record cut at `nt` ends within 1-2 % of the longer record's
  value there (was up to 9x its peak), on both backends.
- The SEM C-PML placement guard no longer flags a source or receiver on the
  model's edge column (`ix = 0` / `nx - 1`). PropSEM adds the layer outside
  the model, so that column is the band's inner edge, but the band's
  thickness comes out of float32 mesh coordinates (200.00002 for 200 m) and
  the comparison had no tolerance: `"in left band (x<200.0)"`. Points within
  round-off of the edge now count as inside. The warning that does fire says
  that on PropSEM such a point is outside the model grid and names
  `pml_guard='off'` as the forward-call option it is, instead of advising a
  smaller `pml_thickness_elements`, which PropSEM does not take.
- A `PropSEM` source or receiver above the `topography` is refused up front,
  naming the point and the surface row of its column, on every backend. It
  used to reach the curved-mesh locator, which said only `could not locate
  (x=..., z=...) in any Quad9 element of column ex=..`. A point on the
  surface row (fractional rows and columns too) is accepted as before.
- The acoustic C-PML's `db/dz` on a topography or interfaces mesh. The double-convolution
  step takes the derivative of the recursion coefficient across the layer, which both
  builders took with column 0's z axis for the whole mesh; such a mesh grades every column
  differently, so `db/dz` came out off by the ratio of the column heights (1.4 under a 400 m
  hill on a 1000 m model). Every grid line now takes its own coordinates, in 2-D and 3-D
  (`sweep.sem.cpml._axis.gradient_along`); flat meshes are unchanged bit for bit
  (`test/sem/test_sem_acoustic_cpml_profile.py`).
- **`AcousticElasticSEM2D`: a sea floor through a side C-PML band absorbs like the rock
  alone.** The coupling took its plain terms inside the bands, where each domain's C-PML
  carries its stretching into the interface: the solid's form is scaled by `s_x s_z`, so its
  load is `(s_z*p n_x, s_x*p n_z)`, and the fluid's double-convolution form leaves
  `(1/s_x)*u_x n_x + (1/s_z)*u_z n_z` -- on a horizontal sea floor, `s_x*p` on the solid and
  the plain term on the fluid. With 200 m of water over rock, 10 Hz, 20 m elements and a
  ten-element layer, the records 10 m above and below the floor and 300 m below it differed
  from a box with no boundary within reach by up to 9e-3, 8e-3 and 4e-2; with the stretched
  loads 1.3e-3, 2.2e-3 and 3.8e-3, the C-PML's own level on the rock alone (2.4e-3, 2.3e-3,
  3.8e-3). SPECFEM2D stretches both loads by `s_x` (`compute_coupling_viscoelastic_ac`,
  `compute_coupling_acoustic_el`), right for its fluid, which it scales as well; on ours the
  `s_x` on the fluid load brings the error back to 1.3e-2. The interface points in a band
  carry four memories each (`sweep.sem.coupling.InterfaceCPML`, the profiles' new per-axis
  `sigma_*`/`d_*` and `alpha_*`), on both backends, the shot batch, boundary saving and
  domain decomposition (`test/sem/test_sem_fs_cpml_interface.py`).
- The SEM C-PML profiles no longer take a node on an absorbing layer's inner
  edge into the layer. Its coordinate equals `lo + L` (`hi - L`) only up to
  round-off, and `coord < lo + L` counted it in whenever it fell an ulp short:
  `d ~ 1e-30 d_max`, `alpha = alpha_max`, and every element touching it joined
  the band -- one physical element column or row, on that side only (287 band
  elements instead of 272 on a 38 x 19 mesh), in the eager band split, the
  `impl='c'` interior fast-path mask and the boundary-saving rings. It hit 576
  of 1248 meshes scanned, float32 and float64, flat, topography and 3-D. The
  displacement and velocity-stress elastic, acoustic 2-D/3-D and DD axis
  profiles now count a node in only past 16 ulps of the axis extent;
  coefficients inside the layer are bit-identical. Where such a node also lay
  in the other axis's layer it took the corner formulas, whose separation
  could move its `d` by a few percent; that is gone too. Records and gradients
  move by round-off (3e-14 in float64) unless that separation acted: then by
  8e-7 at surface receivers, and 3e-5 with the source and receivers next to
  that corner.
- 2-D `Elastic`, `ElasticBorn` and `ViscoElastic` refuse a PML that runs
  between two faces of the other axis that have none, one of them free --
  `free_surface=["top", "bottom"]`, `["top", "left", "right"]` and the four
  other such layouts, or a free face opposite an `abcn` 0 face -- with a
  `ValueError` naming the faces. The solid between them is a plate whose
  backward-propagating Lamb modes a PML amplifies (Loh et al. 2009): on eager
  and `impl='c'` these runs grew 1e2-1e4 times every 1.25 s from about 2 s on,
  the growth converges under grid refinement at the plate's S1
  zero-group-velocity frequency, and no PML width, CFS shift or damping stops
  it. Keep one face of the pair absorbing, or free all four. `Acoustic` and
  `ViscoAcoustic` accept every layout; an equation declares a solid with
  `free_plate_unstable`.
- **`impl='c'` refuses a wavelet that requires grad where its backward
  returns none**: `Elastic`, `Elastic3D`, `ElasticVRR`, `ElasticTTI2nd`,
  `ElasticTTISG`, `ElasticTTISG3D`, `ViscoElastic`, the four DAS equations
  and `AcousticVTI1st` / `AcousticVTI1st3D` raise `NotImplementedError`
  before anything runs. They left `wavelet.grad` at None without a word, so
  a wavelet inversion through them never moved.  Detach the wavelet, or use
  `impl='eager'`.  `Acoustic`, `Acoustic3D`, `AcousticVTI`, `AcousticTTI` and
  `ViscoAcoustic` return the wavelet gradient as before.
- **`impl='c'`: several gradient forwards on one propagator before their
  backwards give the right gradient.** The boundary strips, `last_two` and the
  checkpoint snapshots a backward reads lived on the propagator, and the next
  gradient forward rewrote them: `(loss(A) + loss(B)).backward()`, or any
  second forward before the first one's backward, reconstructed the earlier
  call from the later one's boundaries.  With `BoundarySaving()` on GPU or CPU,
  in every `storage_dtype`, the gradient came out wrong without a warning
  (relative L2 error 0.13-1.6 on two-shot test problems); disk storage and
  `Ckpt()` raised "modified by an inplace operation".  That storage now belongs
  to the forward whose backward is pending, and a forward that finds it taken
  allocates storage of its own, freed with its graph.  A forward -> backward
  loop allocates nothing extra; `Full()` was always right.
- **`impl='c'` under `torch.no_grad()` prepares no backward**, whatever
  `requires_grad` its inputs carry.  A per-shot wavelet `(nshots, nt)` that
  requires grad reaches the compiled forward as a view, which keeps
  `requires_grad=True` under no_grad, so such a call (a line search or a QC
  forward during a wavelet inversion) saved the boundary ring and allocated the
  adjoint fields, kept the whole `nt` history under `Full()`, and was refused
  by `AcousticVRZ` and the Born equations ("does not return the wavelet
  gradient") and by boundary saving under `topography=`.  Whether a call runs
  a backward is now decided before the autograd Function, from grad mode.
- **`impl='c'`, `BoundarySaving(storage="disk")`: a no_grad forward between a
  gradient forward and its backward no longer breaks that backward.** It
  zeroed the host staging and `last_two` the backward reads, which then raised
  "modified by an inplace operation".  Only a call that saves boundaries
  touches the boundary storage now -- which also ends the "batch growth"
  failure: after a boundary-saving gradient call, a no_grad call with more
  shots narrowed that call's boundary ring past its end and raised
  `start (0) + length (n) exceeds dimension size (m)`.
- **`ModelParallel` refuses a forward while a backward is pending.**  Its
  backward reads the ring and geometry the instance keeps for its last
  forward, so a second forward -- `torch.no_grad()` included -- gave the
  pending backward a wrong gradient with no error.  It now raises
  `RuntimeError`; call `backward()` after each forward.

## [0.3.6] - 2026-10-29

### Added
- **`ElasticBorn`: Born modelling for the 2-D first-order elastic equation**
  (Li, Guo, Li & Alkhalifah 2019, doi:10.1190/geo2017-0722.1, eq. 5 in
  velocity-stress form).  Models `vp, vs, rho` (background) and `dvp, dvs,
  drho` (absolute perturbation); fields: the Elastic set plus its scattered
  twin `svx, svz, ssxx, sszz, ssxz` with its own CPML memories.  The scattered
  step is the exact Jacobian of the Elastic step (stencil, CPML, per-edge free
  surface and topography, the Robertsson surface coefficient included), so the
  scattered record is the directional derivative of the Elastic record.
  `impl='eager'` returns every gradient by autograd, the background ones being
  the elastic RWI gradient; boundary saving reverses both fields exactly.
  `impl='c'` (`elastic_born2d`) runs both fields in one fused velocity and one
  fused stress kernel and reuses elastic2d's backward for the `dvp, dvs, drho`
  gradients in every memory mode; it refuses the background and wavelet
  gradients.  `CUDALayoutSpec.backward_state_slots` lets a driver's backward
  bind a subset of its forward slots.  Notebook 45 (`45_lsrtm_elastic`) models
  Born data, checks them against `Elastic`, images the three perturbations by
  RTM and LSRTM and takes the elastic RWI background gradient.
- **`sweep.datasets` gains `chevron-fwi-2014`**, the SEG 2014 Chevron FWI
  synthetic: 1600 shot gathers (321 receivers, 8 s at 4 ms, isotropic elastic,
  marine, with free-surface multiples), the starting vp, the source wavelet
  and a well log, with their geometry.  Chevron never released the true
  model, so the entry returns `vp_init` and no `vp`; `fields=` and `shots=`
  limit what is read.  The files come from the project's Hugging Face copy,
  else from the bucket's original archive as the Internet Archive captured
  it, all pinned by sha256.
- **`AcousticVTI` / `AcousticTTI` on `impl='c'` (`qp_vti2d` / `qp_tti2d`).**
  The pure-qP equations of Liang et al. (2023, doi:10.1190/geo2022-0292.1,
  eqs 19-20 and A-7) get CUDA kernels for the forward, the full-storage
  backward and the boundary-saving backward (gpu / cpu / disk, fp32 / fp16 /
  bf16; int8 is refused), orders 2-8 and runtime order; records match eager to
  1e-6-2e-4.
  The backward needs the forward state of step it+1 at reverse step it, in the
  absorbing layer too, so full mode stores five fields per step and boundary
  saving stores them on the whole CPML frame (`cuda_layout.boundary_frame`:
  the band starts `abcn` cells outside the physical box and runs `abcn` past it
  tangentially) and rebuilds the deep interior. No checkpointing.
  `PropTorch` warns once when `dt` breaks the scheme's stability condition
  (the paper's eq. B-9 for VTI; a bound from the safe side for TTI).
- **`s_gradient=` on the qP equations: `'adjoint'` (default), `'frozen'`,
  `'exact'`.** S_n takes its direction from the wavefield, so the step is
  nonlinear in `h1`, and its exact derivative is ill-conditioned: eager fp32
  and fp64 differ by 2-3 % on the canonical gradient test (1e-7 when
  epsilon == delta), and two eager runs do not agree bit for bit. `'exact'` is
  that derivative (`memory=Full()` only on `impl='c'`); `'frozen'` transposes the
  step with S_n held at the forward field (linear adjoint, consistent to
  rounding); `'adjoint'` propagates the adjoint wavefield with its own S_n, the
  adjoint of the wavenumber-domain eq. 14. The forward is the same in all
  three. **This changes the default eager gradient of `AcousticVTI` /
  `AcousticTTI`** (it was `'exact'`).
- **`AcousticVTIBorn` / `AcousticTTIBorn`**: the Born equations of the qP
  pair -- background and scattered qP fields, each with its own S_n, the
  scattered one driven by the reflectivity `mp` times the background update
  (derivation in docs/user-guide/qp_born.md). Eager and `impl='c'`
  (`qp_vti_born2d` / `qp_tti_born2d`, full and boundary saving); the compiled
  backward returns the `mp` gradient and refuses the background models' and
  the wavelet's. Example: `docs/notebooks/34_lsrtm_qp_vti_tti.ipynb` (RTM and
  LSRTM of a tilted anisotropic bed with the isotropic, VTI and TTI operators).
- **`Biot`: 2-D low-frequency Biot poroelastic waves** (eager).  Solid
  velocity, Darcy flux, total stress and pore pressure on the Virieux
  staggered grid; the stiff viscous drag is integrated exactly, so any
  permeability runs at the fast-P time step and the Darcy limit is exact.
  Open-pore top free surface.  `biot_from_rock` builds the eight coefficients
  from rock properties (differentiable).  Boundary saving is refused (the
  drag is dissipative); gradients use checkpointing.
- **Eager boundary saving for `Elastic3D`, `DASMu`, `DASMu3D` and
  `ElasticVRR`**, free surface included: their step now splits into two
  reversible halves.  Against full storage the gradients agree to 3e-5 to
  3e-4.
- **Eager `storage='disk'` for the boundary ring**: pinned staging slots and
  background threads that write and read temp files; I/O errors raise.
  `storage='cpu'` now copies through pinned memory on a side stream and
  prefetches for the reverse pass (`Elastic3D` at 192^3 on a V100: 11.8 s to
  7.8 s, against 7.7 s with the ring on the GPU).  `pinned_memory`,
  `disk_dir`, `transfer_interval` and `ring_buffers` now reach the eager
  backend; they used to be dropped silently.
- **`EagerOptions(cpml_strips=True)`** keeps the CPML memory fields on the
  absorbing strips only, for 17 equations.  Off by default; uncompiled it is
  bit-identical to the full-grid layout.
- **`sweep.sem`: a spectral-element solver for irregular topography.**  Quad9 /
  Hex27 curvilinear meshes fitted to the free surface and GLL points with a
  diagonal mass matrix: the acoustic potential, elastic and coupled
  fluid-solid equations in 2-D, acoustic and elastic in 3-D, all with C-PML
  but the displacement-form elastic `ElasticSEM2D`, `ElasticSEM3D` and
  `ElasticSEM3DDeformed`.  `PropSEM` takes `PropTorch`'s constructor and call:
  `abcn` grid cells of C-PML (whole elements) outside the model, whose
  gradient stops at the model edge as FD's does; the `free_surface` edge
  spec; per-column `topography` rows; `memory=Full()` / `BoundarySaving()` /
  `Ckpt()`; `impl=None` (auto); `nt` from the wavelet; the `source_type` names
  (`["sxx", "szz"]` an explosion, `["vz"]` a vertical force, with FD's
  waveform) and `return_wavefield` snapshots on the grid.  Boundary saving
  (fp32 / bf16 / fp16 / int8 band rings), shot batching, per-shot models,
  local-window FWI and an x-strip domain decomposition (`SEMModelParallel`).
  User guide: `docs/user-guide/sem.md`; examples: the SEM section (notebooks
  41-44).
- **`BiotVSSEM2DDeformed`: Biot poroelastic waves on the SEM.**  The FD `Biot`'s
  velocity-stress-pressure form with the Darcy flux on the Quad9 mesh, its eight
  coefficients and its `source_type` / `receiver_type` names through `PropSEM`.  The
  pore pressure is continuous and the flux element-local, so the pore fluid is the
  acoustic pressure SEM (with the flux continuous, as in SPECFEM2D, a point source
  excites grid-scale modes along its grid lines); the viscous drag integrated
  exactly per node, C-PML, an open-pore free surface (`p = 0`) that a `topography`
  carries too.  Checked against analytic and wavenumber-integration solutions, the
  FD records and SPECFEM2D's `poroelastic_semi_infinite_homogeneous` example.  2-D,
  eager, no boundary saving.  The receiver gather now also reads element-local
  fields (the flux and the stresses).  With SPECFEM2D's other poroelastic
  features: a point force `fx` / `fz` and flux sources `qx` / `qz`; an anisotropic
  permeability (`BiotAnisoPermSEM2DDeformed`, the drag a tensor); the viscous
  attenuation (`equation_options={"viscous_q": (Q0, f0)}`, a standard linear
  solid); fluid (`rho = 0`) and dry elastic (`M = 0`) regions in one model, coupled
  as in SPECFEM2D (open pores to a fluid, sealed to a dry solid), sources loading
  only the fields of their medium.  `sweep.equations.biot_params`: the drag tensor,
  the fluid and dry-solid coefficients, and the parameter families of Morency, Luo &
  Tromp (2009) with their kernels by the exact chain rule.  Checked against
  SPECFEM2D runs of each and its `poroelastic_acoustic` example.
- **`PropSEM(equation_options=...)`**: an equation's own `build` keywords; a name the
  equation does not take is refused.
- **`impl='c'` for `sweep.sem`.**  The SEM CUDA kernels are 43 entries of the
  core (kind 3 of the C API, called through `sweep_sem_call`), their input
  structs generated from `cuda/equations/sem{2,3}d/*_types.h` like the FD ones.
  The core's C ABI is now 3: a core built from an older tree is refused at
  load -- rebuild it.
- **`sweep list equations` and `sweep show` cover the SEM equations**, and
  the API reference has SEM pages (`PropSEM`, the equations, meshes and C-PML,
  `SEMPropagator`, SEM domain decomposition).  `sweep --help` points to
  `sweepx` for task runs.

### Changed
- **The Born equations are named Born, not LSRTM** (no aliases kept):
  `AcousticLSRTM` / `AcousticLSRTM3D` / `AcousticVTILSRTM` / `AcousticTTILSRTM`
  are `AcousticBorn` / `AcousticBorn3D` / `AcousticVTIBorn` / `AcousticTTIBorn`
  (modules `acoustic_born`, `acoustic_born3d`, `qP_born`), and the CUDA side
  follows: `acoustic_born2d` / `acoustic_born3d` / `qp_vti_born2d` /
  `qp_tti_born2d` entries, kernel and slot-table names.  LSRTM is the
  inversion these operators serve; `SWEEP_LSRTM_SPLIT_III` keeps its name.
- **`ElasticCurvilinear` on `impl='eager'` runs 1.8-5.5x faster per step
  compiled, gradients 3-3.5x, and takes `EagerOptions(cpml_strips=True)`.**
  Under `torch.compile` its asymmetric stencils, the free-surface closure block
  and the Kreiss-Oliger filter now run as shifted slices that fuse with the step;
  they were cuDNN convs and an einsum, and the operator's kernel cache, keyed by
  a tuple holding a `torch.dtype`, hid their taps from Dynamo, so the graph
  broke at every derivative.  The filter runs on the surface band only, and the
  CPML memories can sit on the absorbing strips, which on shot batches gives
  another 1.1-1.2x and halves the gradient memory (2048² x 4 shots: the
  gradient used to run out of 48 GB, now 36 GB, 17 GB with strips).  Measured
  at order 16 on an RTX 6000 Ada, 1024²-2048² grids; the gain grows with the
  grid and the shot count.  Uncompiled the record and the gradients are bit for
  bit the old ones; a single shot (1024²) runs 1.3x faster forward and 1.8x
  faster gradient there, from the receiver change below.
- **A receiver footprint (`source_receiver_stencil`: `ElasticCurvilinear`'s
  5x5, `ElasticTTI`'s 3x3) is gathered with one `index_select`** where it took
  one per tap, and its backward scatters the taps into one zeroed field where
  every tap zeroed a full field for autograd to sum: per step and receiver
  field, a third of the forward launches, and one full field written in the
  backward where there were one per tap.  The taps are summed in the old
  order, forward and backward, so records and gradients are bit for bit the
  old ones.
- **`ElasticCurvilinear` follows Bartana, Xu & Rickett: collocated fields,
  asymmetric first-derivative operators and a rotated traction-free surface.**
  The momentum equation is discretised as a divergence, whose operator is the
  negative transpose of the gradient, so the spatial operator is self-adjoint;
  the metric terms are built with that same divergence, so a constant stress
  state accelerates nothing.  The staggered chain-rule version this replaces
  diverged on a Gaussian hill (a 30% slope went to NaN at 1.2 s); over 6 s the
  new one stays finite and decays on every hill tried.  A Kreiss-Oliger filter
  over the surface band (`surface_damping`, default 3e-2) removes the spurious
  surface mode the ghost-free closure supports.  `spatial_order` defaults to 16
  (the paper's L = 8).  On curvilinear equations `topography=` may hold
  fractional rows, and sources and receivers are physical grid indices, as on
  every other equation, fractional rows allowed -- they used to be the
  boundary-fitted grid's own rows, counted from the surface, whose spacing is
  not `dh`.  The propagator maps them onto the nearest computational row and
  refuses one in the air; `PropTorch.to_computational()` shows where a
  position lands.  Snapshots come back on the physical grid too, zero in the
  air (`snapshot_grid='computational'` returns the grid's own rows).  And so
  are models: each column is sampled, linearly and differentiably, at the
  depth of the grid's rows, so the gradient comes back on the physical grid;
  the rows above the surface are not read and get none.  Models used to be
  read as the grid's own rows, which pushed a layer that follows the surface
  down by the depth of the surface.
- **`Acoustic3D` and `AcousticVRZ3D` on `impl='c'` run 1.2-1.8x faster per
  step.**  The forward step, the boundary-saving reconstruction and the
  adjoint step each ran as one launch over the padded grid, every thread
  carrying the CPML path.  The forward and adjoint steps now launch the
  interior and the six PML slabs around it separately, with per-axis CPML
  terms, and the reconstruction launches only its own box.  The
  `AcousticVRZ3D` model gradient is one 2.5-D tiled kernel at orders 2-8
  (z-taps in registers, or through L1/L2 on GPUs with a large L2; the result
  is the same either way).  Boundary-saving step, orders 4-8, 200x240x320 to
  400^3 with a 20-cell PML: `Acoustic3D` 1.4-1.8x on V100, A100 and H200;
  `AcousticVRZ3D` 1.6-1.7x on V100, 1.4-1.8x on A100, 1.2-1.7x on H200;
  order 2 1.2-1.5x.  Full-wavefield mode at orders 4 and 8: 1.2-1.8x.  Under
  `ModelParallel` (4x V100, orders 4-8) 1.1-1.4x for `Acoustic3D` and
  1.4-1.5x for `AcousticVRZ3D`: a tile's cut faces carry no PML.
  `Acoustic3D` records are bit-identical; its gradients and the
  `AcousticVRZ3D` records and gradients move in the last bits (rel <= 1.3e-6).
- **`Acoustic` and `AcousticVRZ` (2-D) on `impl='c'` run 1.05-1.55x faster
  per step.**  Their forward and adjoint steps now launch the interior with
  its own kernel, which carries no CPML path (21-34 registers instead of
  48-80), and the CPML ring around it as one launch.  The `AcousticVRZ` model
  gradient is one 2-D tiled kernel at orders 4-8, and under boundary saving
  it also forms the reverse step's reconstruction (orders 2-8): one pass
  instead of the reconstruction, the source injection and the gradient as
  three launches.  Boundary-saving step with a 20-cell PML, 1024^2 to 4096^2,
  V100 / A100: `Acoustic` 1.07-1.21x / 1.05-1.27x; `AcousticVRZ` 1.27-1.32x /
  1.26-1.38x at order 4, 1.47-1.55x / 1.25-1.30x at order 8.  Full-wavefield
  mode at 2048^2: `Acoustic` 1.13-1.23x, `AcousticVRZ` 1.09-1.29x.  Records
  are bit-identical at orders 2-8 and the `Acoustic` gradients mostly so; the
  `AcousticVRZ` gradients (rel <= 1e-6) and orders above 8 move in the last
  bits, from FMA contraction alone: built with `--fmad=false`, the old and
  new kernels agree bit for bit.
- **`AcousticVRZ3D` under `ModelParallel` matches one card bit for bit at
  every order**; at orders 2 and 4 the two used to differ by ~1 ULP.  At
  orders 2-8 its backward also exchanges halos once per reverse step instead
  of twice (the pressure strips the gradient reads across a cut ride with
  lambda and p), 1.04-1.10x per DD step on 4x V100; a tile cut on both sides
  of an axis needs at least `spatial_order` cells along it.
- **The compiled eager step is faster and compiles once.**  Under
  `torch.compile` (the eager default) every derivative is now a zero-padded
  shifted slice that fuses into the step kernel, instead of a cuDNN
  convolution: on an A100 and a V100, 1.2-4.5x on 2-D forwards, 1.1-7.3x on
  2-D gradients, 2.5-10x at 128^3, and 1.3-3.9x less peak memory for a
  full-tape gradient.  Uncompiled eager keeps the convolution.  Sources are
  scattered from indices built once instead of through a full-grid mask, and
  records are stacked once instead of written step by step (V100: forward
  1.05-1.37x, gradient 1.08-1.22x).  A rollout compiles one graph instead of
  3-6: steps run uncompiled until their inputs' `requires_grad` flags settle,
  so the first gradient call with a cold compile cache takes 35-85 % of the
  time it did.  Uncompiled records and gradients are unchanged bit for bit;
  compiled ones move at fp32 rounding (<= 1.4e-6).
- **The eager step compiles as one graph from torch 2.2 on.**  torch 2.3
  compiles the GPU step with `allow_buffer_reuse=False` and runs the CPU step
  uncompiled, around two Inductor bugs fixed in 2.4 (one turned the
  `ViscoAcoustic` forward to NaN); torch 1.x runs uncompiled and warns once.
- **An eager source off the grid raises `ValueError`**; its flat index used
  to wrap around silently.
- **Eager boundary saving is refused for `ViscoAcoustic`, `AcousticVTI` and
  `AcousticTTI`**: reconstructing their wavefields backwards amplifies the
  round-off into a gradient 2-10 % off.  Use full storage or checkpointing
  (or, for the qP pair, `impl='c'`).

### Removed
- **The APM free surface (`topo_method='apm'`, `ElasticAPM`).**  Elastic
  topography is now `ElasticCurvilinear` (2-D, boundary-fitted) or the
  image-method staircase, which is also what `topo_method='auto'` resolves to
  for `Elastic` and `Elastic3D` (it used to be APM).  `topo_method='apm'`
  raises, and so does any name other than `'auto'` and `'image'` -- it used to
  run the image method.  With APM goes the only `impl='c'` path for 3-D
  elastic topography (forward only; it had no compiled backward); eager runs
  it with the image method.  The core's two APM entries (65, 71) are retired
  and their numbers not reused; the input-struct fields `topo_category` and
  `use_apm` stay, unused.
- **`SWEEP_VRZ_GRAD_SPLIT`.**  It put one card on the two-pass VRZ gradient so
  it matched `ModelParallel`; that now holds without it.  Setting it raises.

### Fixed
- **`PropSEM` no longer spends seconds per call locating its sources and
  receivers on a topography mesh.** A point on a deformed (Quad9 / Hex27)
  mesh was located by a Newton solve in torch scalars, one point at a time
  -- 0.07-0.18 s a point on a 3 km line meshed at 4-8 m -- and `impl='c'` kept the
  stamps keyed on the whole geometry of a call, so every new source position
  located every receiver again: 37 receivers cost 4.7 s per call against
  0.11 s of propagation (2-D elastic, 0.14 M nodes). The points of a call are
  now located in one batched Newton (float64, also on float32 meshes, where
  it used to stop at the float32 round-off) and stamped in one copy to the
  device, and every point's location and stamp are kept on the mesh, least
  recently used out first (`POINT_RECORD_SIZE` points), so a shot at the same
  receivers locates only its source. The same call now costs 0.12 s; 600
  never-seen receivers locate and stamp in 17 ms (were ~80 s). Locations are
  unchanged to 1e-13 on float64 meshes (records to 1e-15); on float32 meshes
  they are now exact, which moves records by ~3e-5, the old locator's own
  error there. `build_point_stamps` and `locate_points` take a list of points.
- **CPU math no longer depends on oneMKL's VML first-call race.** PyTorch's
  x86 wheels run `torch.sqrt` / `exp` / `log` ... on CPU float tensors through
  oneMKL VML, whose CPU detection is not thread-safe: on AVX-512 Intel CPUs the
  first multi-threaded VML op of a process could run a low-precision kernel in
  a worker's OpenMP chunk (`sqrt` up to 3e-4 off, `exp` 1.5e-4) -- one fresh
  process in five at 4 threads on a Xeon Gold 6148 with torch 2.14.1, never on
  AMD EPYC.  A model derived on CPU (`vp = sqrt(H / rho)`) came out wrong in
  that chunk, and `test_biot`'s undrained limit failed in one process out of
  four.  Importing `sweep` (with torch loaded) or `sweep.equations` now settles
  the detection with one serial 1-element call (`sweep/_mkl_vml.py`).
- **The second-order acoustic family's compiled backward is the exact
  transpose of its compiled forward, CPML band included** (`Acoustic`,
  `Acoustic3D`, `AcousticVRZ`, `AcousticVRZ3D`, `AcousticLSRTM`,
  `AcousticLSRTM3D`, `ViscoAcoustic` on `impl='c'`).  `db = np.gradient(b)` is nonzero one cell
  past the band and the 2-D `d(a psi)/dx` stencil reads M cells into it, but
  the forward ran the bare Laplacian there (the 2-D first reflection off the
  CPML was 6x larger than eager's), and the adjoints' interior bounds stopped
  one cell short of the forward's.  The LSRTM adjoint applied the forward CPML
  operator to the adjoint field inside the band, a discretisation of the
  continuous adjoint rather than a transpose (reflectivity dot test 0.11, now
  1e-6); its RWI backward no longer allocates the grid only that operator read
  (one padded grid per shot less in full and boundary-saving modes).  Checkpoint replays now step the CPML memory double-buffered (in
  place they raced the neighbours' reads, so two checkpointed backwards
  differed), and the vp image is the update actually applied,
  `vp^2 * w_sum`.  `ViscoAcoustic` runs the same 2-D forward kernel; its vp
  carrier, recomputed from the raw history, now takes the stored update on the
  M cells next to each CPML band (the history gains a few time slices, about
  1.5 % at production sizes), so with both attenuation terms off it matches
  `Acoustic` bit for bit again, and its checkpoint replay is double-buffered
  too.  Against an eager twin of each compiled forward (outer halo zeroed, the
  kernel's CPML form): records 5e-7 to 2e-5 and gradients 2e-6 to 2e-5 in
  every memory mode, the LSRTM RWI vp gradient at its fp32 floor (7e-5 in
  2-D, 1.8e-4 in 3-D); domain-decomposed runs stay bit-identical to one card.
  Wall time on a V100 changes by -5 % to +2 %.  Records and gradients of these
  equations change accordingly.
- **`AcousticLSRTM` / `AcousticLSRTM3D` and `AcousticVRZ` / `AcousticVRZ3D`
  on `impl='c'` refuse the gradients their backward never computed**: the
  wavelet's (LSRTM 2-D returned zeros, LSRTM 3-D the scattered adjoint sampled
  at the source, VRZ left `wavelet.grad` at None) and, for LSRTM with
  `memory=Ckpt(...)`, the RWI vp gradient (zeros).  Use `impl='eager'` for
  those.
- **`ModelParallel` refuses a wavelet that requires grad**, for every
  equation: the domain-decomposed backward returns the model gradients only
  and left `wavelet.grad` at None (`Acoustic` included, whose single-domain
  backward does return it).  Take that gradient from a single-domain
  `PropTorch` run.
- **`AcousticLSRTM` / `AcousticLSRTM3D` under `ModelParallel` with
  `BoundarySaving(storage='cpu')`: the gradient no longer changes from run to
  run.**  Their stepped backward built a copy stream of its own on every call
  -- DD makes one call per time step -- and the prefetch of the next step's
  boundaries that a call issued last stayed queued on that stream, which
  nothing waited on again; landing late, it overwrote the ring slot a later
  call had just refilled and was restoring from.  In 3-D two identical runs,
  and cpu against gpu staging, differed by up to 12 % of the gradient's peak,
  on one tile and across cuts; the 2-D backward has the same copies, small
  enough at the test sizes to land in time, but not at production sizes.  Both
  now run on the `BoundarySession` `ModelParallel` already holds, as the other
  DD equations do: every copy queues on one stream and every restore waits for
  the copy it reads (a stepped call without a session drains its stream before
  returning).  cpu staging under DD is bit-identical to gpu-direct and to
  itself in fp32 (`test/dd_offload_check.py` with `EQ=lsrtm`,
  `test/lsrtm_staging_check.py`).  `storage='disk'` is refused for the pair at
  construction: the drivers refused it across a cut but ran it on a single
  tile.
- **3-D boundary saving with `storage_dtype='int8'` / `'fp16'`: a race in the
  restore.**  The band-only restore kernel visits a cell on an edge or corner of
  the saved strip (or, on an axis thinner than two bands, a cell both opposite
  bands cover) once per band, and every visit wrote it.  int8 and fp16 storage
  quantize each face with its own per-block scales, so those copies differ, and
  the cell kept whichever thread wrote last: the gradient changed from run to
  run (`Acoustic3D` 2e-7 / 3e-6 relative with fp16 / int8 on
  `test/test_boundary_compact_kernel.py`), and under `ModelParallel` the
  cpu-staged gradient did not match gpu-direct bit for bit; the forward record
  was never affected.  Such a cell is now restored from one band only, the one the
  full-grid restore writes last, so the result equals the full-grid kernel's
  (`SWEEP_BOUNDARY_NO_COMPACT=1`) bit for bit at any block order, and cpu
  staging under DD is bit-identical to gpu-direct in every storage dtype.  It
  affected every compiled 3-D equation with boundary saving on gpu, cpu or
  async-disk storage: int8 since 0.1.0, fp16 since 0.2.0 (when it gained
  per-block scales).  fp32, bf16, synchronous disk and 2-D were not affected.
  New kernel `boundary_kernel3d_compact_owned`; every other kernel's machine
  code is unchanged.
- **Registered gradients under a free surface.**  `register_gradient(mode=
  'gradient')` and `imaging()` crop the accumulated image to the model face by
  face, as `EdgePadding` crops the autograd gradient; they cropped it centred,
  which with a free surface (no pad on top) or uneven per-edge `abcn` put it
  half the difference of the pads off.  On a curvilinear grid they also take
  it back to the model grid, and `imaging()` maps its sources and receivers
  as the forward does.
- **Boundary-saving gradients on a thin physical box.**  Where the physical
  box (PML excluded) is thinner than two restore strips along an axis -- fewer
  than `2 * (spatial_order / 2 + 1)` cells -- the strip imaging of `Acoustic`
  and `Acoustic3D` imaged the cells the two strips share twice, with two
  racing `+=`: the gradient there came out too large and changed from run to
  run (`Acoustic3D`, 3 cells in x: rel 4e-4 against full storage, now 7e-8).
  Each strip cell is now imaged once.  The elastic carrier capture, which
  enumerates the strips the same way (a copy, so the overlap was harmless),
  is clipped the same.
- **`sweep.datasets` downloads work again.**  The SEG open-data S3 bucket that
  served every download entry stopped serving anonymous downloads in 2026.
  The original files of `overthrust`, `seg-eage-salt`, `marmousi` /
  `marmousi2`, `bp-2004` and `bp-2007-tti` now come from the project's Zenodo
  records (one per model), then from the same files on Hugging Face
  (`shaowinw/seismic-benchmark-models`), with the bucket last.  A download is
  kept only if its sha256 matches the original, and a cached file is checked
  again on every load.  `overthrust:3d-acoustic` falls back to the HDF5 copy
  of the model on Zenodo, which gives the same grid; the `datasets` extra now
  installs h5py for it.  Existing caches are reused.  When no source answers,
  the error lists every source tried and the path to put the file at, and
  `sweep datasets download` prints it and exits with status 1 instead of a
  traceback.  `hess-vti` has no public copy left: it loads only from a cache
  or from its original files placed by hand.  `marmousi` (the full-size model,
  read from the Marmousi2 archive, and the embedded demo derived from it) now
  carries Marmousi2's CC-BY-4.0 license and cites Martin et al. (2006).  The
  example download scripts under `examples/models/` use the same sources.
- **Eager boundary saving under `torch.compile` no longer keeps a dropped
  propagator's GPU memory** (its workspace and its last rollout's ring and
  frames: a reference cycle through the autograd graph, which gc cannot
  see).  A second backward of the same rollout now raises; it used to return
  a wrong gradient silently.
- **`ViscoAcoustic` with per-shot models `(B, nz, nx)`.**  The fractional
  exponent was averaged over the whole batch, so each shot's result depended
  on the other shots' Q (1.7e-3 in the data, 8e-4 in the gradient against
  one shot at a time).  Each shot now takes its own, on eager and
  `impl='c'`; a shared model is unchanged bit for bit.
- `backend_concat` and `zero_edge_halo` accept numpy 2 and JAX arrays.

### Documentation
- Notebook 33 (new): eager, eager with CPML strips and `c` on forward,
  full-storage and boundary-saving runs -- time, compile time, peak memory
  and gradient agreement.  On the 3-D acoustic case `c` is 1.9x faster than
  eager for a full-storage gradient and 2.5x for a boundary-saving one.
- Notebook 38: poroelastic traveltime kernels by autograd through `Biot`,
  three parameterizations (Morency et al. 2009).
- Notebook 39: poroelastic radiation patterns, Born analytic against `Biot`,
  four parameterizations (after Yang et al. 2019).
- Notebook 40: `Biot` wavefields -- fast P, slow P and S, four viscous regimes
  against the analytic solution, Dai et al.'s two layers under an open-pore
  free surface.
- API page and equations-table row for `Biot`.

## [0.3.5] - 2026-10-05

### Added
- **`AcousticLSRTM` / `AcousticLSRTM3D` (`impl='c'`) return the velocity
  gradient.**  `vp.grad` used to come back zero; the full and boundary-saving
  backwards now compute the RWI tomographic gradient (Wu & Alkhalifah 2015,
  terms II+III+IV): a background adjoint driven by the coupling transpose,
  plus the scattered field (its `u_tt` stored in full mode, reconstructed in
  boundary-saving mode).  It is paid for only when `vp` requires grad, decided
  per call: a classic LSRTM (reflectivity only) keeps 0.3.4's buffers and
  cost; with the vp gradient a V100 backward takes 1.5-1.8x the time and
  1.4-2.0x the memory.  Against eager, with the reflectivity reaching the CPML
  band: rel 2.4e-4 (2-D), 6.2e-4 (3-D).  The checkpointing modes still return
  `vp.grad = 0`.
- **The image-point term III on its own.**  With `SWEEP_LSRTM_SPLIT_III=1`,
  `vp.grad` holds II+IV and `sweep.propagator.last_grad_split_iii()` returns
  III on the model grid, for the paper's `(II+IV) + beta*III` (eq. 18-20).  The
  core ABI gains `BackwardInput.grad_split_iii_out`: the layout probe refuses a
  0.3.4 core under this package, and the other way round.
- **Domain decomposition for `AcousticLSRTM` / `AcousticLSRTM3D`.**
  `ModelParallel` rotates and halo-exchanges both fields, in the forward and
  in the boundary-saving backward with its reconstructions; record and
  gradients are bit-identical to one card on 4x V100, with and without the vp
  gradient.

### Documentation
- Notebook 31: frequency-selection FWI (Tromp & Bachmann 2019) on a
  towed-streamer Marmousi, multiscale with its own grid and time step per band.
- Notebook 32: where the RWI velocity gradient comes from -- `Acoustic` gives
  term I, `AcousticLSRTM` gives II+III+IV -- in 2-D and 3-D, with the beta-III
  split and the memory modes.
- One citation style for every notebook's Reference cell, fields checked on
  Crossref.

## [0.3.4] - 2026-10-02

### Changed
- **Smaller wheel: every CUDA kernel is compiled once.**  A kernel template
  launched from a header -- a driver hook or a `LAUNCH_*` macro -- was
  instantiated (five stencil orders x every arch of the fat binary) in every
  translation unit that included it, called or not: an equation's forward and
  backward units, and every equation sharing a kernel family (`Elastic` /
  `DASMu` / `ViscoElastic`, `Elastic3D` / `DASMu3D`, `Acoustic` /
  `ViscoAcoustic`).  Kernels defined in headers were compiled into each
  includer the same way.  Each family is now instantiated once, in its
  `kernels.cu`, and launched through a table of its stencil-order
  specializations (`launch/by_order.cuh`): the cu12 core drops from 157.3 to
  70.8 MB, the cu13 core from 189.9 to 85.8 MB and the wheel from 93.1 to
  42.4 MB (PyPI's per-file limit is 100 MB).  The machine code of every kernel
  on every arch is byte-identical to 0.3.3's.

## [0.3.3] - 2026-10-02

### Added
- **ViscoElastic: 2-D visco-elastic equation (generalized standard linear
  solid), eager and CUDA (`impl='c'`) backends.**  The rheology SPECFEM2D
  (and so SeisFlows) uses:
  `n_sls` Zener bodies (default 3) with relaxation times fixed by the band,
  memory variables advanced with the trapezoidal rule, on the `Elastic`
  staggered-grid step (shared, not copied: the velocity-gradient/CPML/free-
  surface block of the stress sub-step is factored into
  `elastic_velocity_gradients`; `Elastic` and `DASMu` stay bit-identical).
  Models are `vp, vs, rho, Qp, Qs`, velocities given at the reference
  frequency `f_ref` (SPECFEM's `READ_VELOCITIES_AT_f0`), and all five carry
  gradients.  Each mechanism's strength is the least-squares constant-Q fit,
  linear in the strengths (a rational function of 1/Q, evaluated elementwise
  with an analytic backward), so Q stays within ~5% of target over the band
  for Q >= 10 (~9% at Q = 3).  `Q = inf` reduces bit-exactly to `Elastic`.
  The flat free surface (per edge) solves the surface normal strain rate so
  the traction stays zero through the memory variables.
  `ViscoElastic.unrelaxed_velocities` gives the velocities the explicit
  update runs at, for the CFL check.  Checked against the exact 2-D
  homogeneous Green's function (P and S spectral ratios to 2e-4 / 2e-3) and
  against SPECFEM2D with a free surface (visco/elastic spectral ratio: median
  deviation 0.3-2% over 4-25 Hz; time-domain misfit no larger than the
  elastic-vs-elastic baseline).  The CUDA backend is a set of staggered-
  skeleton traits derived from elastic2d's (`visco_elastic2d/driver_traits.cuh`,
  a template on the mechanism count): Elastic's velocity kernel, source /
  receiver / rho-correction plumbing and adjoint stencil transposes, plus the
  visco stress update and its hand-derived exact adjoint (memory variables,
  surface solve and its material derivatives) with the gradient imaging fused
  into it in every mode.  `full`, chunk- and recursive-checkpoint backwards,
  all three bitwise equal.  c vs eager: records and all five gradients agree
  at the Elastic c-vs-eager floor (<= 8.8e-5 over 72 source x receiver x
  geometry x free-surface combinations with PML, <= 1.3e-5 in a closed box,
  surface rows included).
  Not supported: boundary saving (default falls back to `full`), topography,
  domain decomposition (all refused explicitly).

## [0.3.2] - 2026-10-01

### Fixed
- `impl='eager'` no longer fails when `torch.compile` cannot build the step,
  e.g. on CPU with g++ older than 10 (`InductorError: CppCompileError`, raised by
  the first forward of a plain CPU run, since `use_compile` defaults to True).
  The step runs uncompiled instead, with one `RuntimeWarning`; results are the
  same. Errors raised by the step itself still propagate.
- A `pml_type` the equation was not written for is refused with a `ValueError`
  at construction. It used to reach the kernels, which read the PML profiles
  positionally: `Elastic` with `pml_type='cpmlr'` and `impl='c'` aborted the
  Python process on a C assert. Each equation's options are in its new
  `supported_pml` property (also in `Equation.defaults()`); every equation has
  one, except `Acoustic1st`, which also takes `'spml'`. Leave `pml_type` unset.
  The README, docs, notebooks and examples no longer pass it.

## [0.3.1] - 2026-10-01

### Added
- `Full`, `BoundarySaving` and `Ckpt` are exported from `sweep.propagator`
  (only the deprecated `MemoryOptions` family was).

### Fixed
- `sweep.backend.torch.binding.diagnostics()["shim"]` read `"ctypes"` on a
  `SWEEP_BUILD_CUDA=1` install, whose compiled `sweep._C` is the pybind shim;
  it now reads `"pybind"` there, as under `SWEEP_JIT_FULL=1`.
- A typed strategy inside `cuda_options=CUDAOptions(memory=...)` did not reach
  the `impl='c'` backend: `Ckpt()` and `Full()` raised a "conflicting
  gradient-memory mode" `ValueError` at construction, and
  `BoundarySaving(storage='cpu', pinned_memory=True)` silently built the
  default GPU ring.  It is now read exactly like `memory=`.
- `AcousticCurvilinear` / `ElasticCurvilinear` counted as compiled bindings
  because their `_C()` stub (which only refuses `impl='c'`) was callable, so
  `impl=None` on a CUDA device picked `'c'` and raised `NotImplementedError`.
  A binding now needs a `C_NAME`: both report no binding (the CLI prints
  `no`), `'auto'` runs them eager, and an explicit `impl='c'` falls back with
  a `UserWarning` like any eager-only equation.
- `ModelParallel` decomposed a propagator built with `topography=` as a
  flat-top problem, silently (the tiles never carried the surface, and
  boundary saving under a surface is wrong anyway).  It now refuses it with
  `NotImplementedError`.

### Documentation
- README and the installation page are shorter; the release-core build rules
  moved to *Developer › Release cores*.
- The API reference, user guide, examples and notebooks were audited against
  the 0.3.0 code: wrong signatures and defaults, removed or renamed APIs,
  deprecated spellings shown as current, and stale JIT-era install claims
  were fixed.

## [0.3.0] - 2026-09-30

### Added
- **The wheel ships a prebuilt CUDA core; first use of `impl='c'` no longer
  needs nvcc.**  The CUDA kernels are built once, ahead of the wheel, as a
  torch-free `libsweep_core.so` (every `.cu`, nvcc) behind a plain C
  interface; the torch side binds to it without compiling (the ctypes layer
  above -- the compiled pybind shim, `module.cpp` + the CPU binding, is the
  `SWEEP_JIT_FULL=1` developer path).  It is one wheel, carrying one core per
  CUDA major as a fat binary (`sweep/lib/cu12/` and `sweep/lib/cu13/`, each a
  `libsweep_core.so` + a `core.json` sidecar with ABI version, CUDA release,
  archs, PTX and sha256; the cu13 entry below has the per-core coverage), so
  `pip install sweepx` followed by the first `impl='c'`
  call, or `sweep.precompile()`, compiles nothing: no nvcc and no toolkit.
  `sweep.backend.c.jit.core_path()` picks the shipped core when its ABI, CUDA major
  and archs fit this process (PTX makes newer cards fit; a GPU outside the
  shipped archs and older than the shipped PTX, e.g. Pascal sm_6x, does not) and
  otherwise falls back to the unchanged local core build, which is also what a
  torch built for another CUDA major, an sdist or a clone gets, since neither
  carries a core; `SWEEP_CORE=<path>` overrides the lookup,
  `sweep.backend.c.jit.shipped_core_info()` reports what was chosen and why, and
  `sweep.backend.torch.binding.diagnostics()` gains a `shipped_core` key with
  the same answer.  `can_compile()` is satisfied by a fitting core with no nvcc
  on the machine, and `python -m sweep.build --no-gpu-required` no longer
  demands `TORCH_CUDA_ARCH_LIST` when a shipped core fits (the arch list only
  names the target of a local core build).  cuFFT is linked dynamically and
  preloaded from torch's `nvidia-cufft` wheel (or `<cuda_home>/lib64`) before
  the core is loaded; the optional extra `sweep-solver[cuda12]` pulls
  `nvidia-cufft-cu12` when a torch wheel did not.
  Release side: `python -m sweep.build --core [--archs ...] [--out DIR]
  [--cuda-home DIR]` produces the core and its sidecar without a GPU (`--out`
  defaults to `sweep/lib/<tag>` inside the installed package, `src/sweep/lib/<tag>`
  in a clone).  The wheel's platform tag is the build host's glibc
  (`manylinux_<glibc>_x86_64`) and the core binds the host's `libstdc++`, so the
  release builds both cores and the wheel inside PyTorch's `manylinux2_28-builder`
  containers (`utils/build_cores_manylinux.sh`): the shipped cores need only
  `GLIBC_2.17` / `GLIBCXX_3.4.22` and the wheel is tagged `manylinux_2_28`.
  Also: the package now really imports on Python 3.9
  (`propagator/options.py` evaluated a `str | None` annotation at runtime),
  and `sweep.build --core` no longer needs torch (its lock and its arch list
  do not import it), which is what lets it run in that container.  Measured
  size (both cores, container build): the cu12 `.so` is 127 MiB, the cu13 `.so`
  155 MiB, and the two-core wheel about 76 MiB, against PyPI's 100 MB per-file
  limit -- keep a single `+PTX` entry per core, the newest arch.  A tree holding a core turns the wheel
  into a platform wheel, otherwise it stays `py3-none-any`.  `src/sweep/lib/` is
  git-ignored and pruned from the sdist, so build the core, then `python -m
  build --wheel` from the same tree: a bare `python -m build` packs the wheel
  from the pruned sdist and ships no core.  `SWEEP_REQUIRE_CORE` (the release
  script sets `cu12,cu13`) makes `setup.py` refuse such a tree instead of quietly
  producing a core-less wheel: `1` asks for at least one core, a comma list names
  the tags that must all be present.
- **A second prebuilt core, for CUDA 13 (`sweep/lib/cu13/`).**  The wheel
  ships cu12 and cu13 side by side and the loader picks the tag torch's CUDA
  major names, so a torch cu130 install compiles nothing either.  Per core:
  cu12 (nvcc 12.9) covers V100 and newer through H100/H200 as SASS
  (sm_70-sm_90) and Blackwell through its sm_90 PTX; cu13 covers T4/RTX 20 and
  newer (sm_75-sm_90) with Blackwell native (sm_100, sm_120 + PTX) -- no V100,
  since nvcc 13 dropped offline compilation for compute capability < 7.5 -- and
  needs driver >= 580, exactly what torch cu130 needs.  The core's rpath now
  reaches cuFFT in both pip layouts (`nvidia/cufft/lib` for the CUDA 12 wheels,
  `nvidia/cu13/lib` for the CUDA 13 ones), and the new extra `sweep-solver[cuda13]`
  pulls `nvidia-cufft` 12.x (the unsuffixed wheel is the CUDA 13 line; `-cu12`
  stays for CUDA 12) when a torch wheel did not.  Release side: `python -m
  sweep.build --core` files the core under the tag of the nvcc it uses
  (`--cuda-home`) and, without `--archs`, builds that major's recommended list
  (cu12 `7.0;7.5;8.0;8.6;8.9;9.0+PTX`, cu13
  `7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX`; an sm_70 entry under nvcc 13 is refused
  up front), so a release is one command per toolkit.  `SWEEP_REQUIRE_CORE`
  accepts a comma list of tags (`cu12,cu13`) that must all be present, naming
  the missing one, next to `1` for "at least one".  The cu13 core is a second
  `.so` (155 MiB against cu12's 127 MiB); the wheel with both is about 76 MiB.
- **`impl='c'` talks to the prebuilt core through a pure-Python `ctypes`
  layer; after `pip install` nothing compiles.**  `sweep.backend.c` fills the
  core's C structs from each tensor's `data_ptr()`, shape, strides and dtype
  and calls into `libsweep_core.so` directly, so the default path no longer
  builds a torch shim at all: no nvcc, no C++ compiler, no CUDA headers, no
  ninja run (the `ninja` dependency serves a local core build only), and no
  torch C++ ABI in the loop -- the same wheel works with any
  torch version by construction.  `nvcc` is needed only when no shipped core
  fits (a torch built for another CUDA major, a GPU outside the shipped archs
  and older than the shipped PTX) or for an sdist/clone install, and then only
  the CUDA core is compiled (`python -m sweep.build`, or the first `impl='c'`
  use).  That local build needs nvcc >= 12.4 (12.0-12.3 ship a broken
  `<cuda/std>` bf16 header; `SWEEP_JIT_ALLOW_OLD_CUDA=1` tries anyway), >= 12.8
  when the target list reaches Blackwell (sm_100 / sm_120), or a CUDA 13 nvcc,
  and is refused up front with the toolkit and the version named otherwise.  A
  local core built once is reused by later runs without nvcc, on the
  strength of a source stamp (`sources.sha256`: the csrc tree plus the nvcc
  flags) written beside it.  `sweep.precompile()` now just makes sure a core
  exists and loads it: a no-op with a shipped core, the local core build
  otherwise.  The compiled pybind shim survives as the
  developer path behind `SWEEP_JIT_FULL=1` (needs torch's C++ headers, the CUDA
  runtime headers -- torch's pip CUDA wheels bring them -- and a C++ compiler;
  nvcc only when no shipped core fits).  A core that fits a GPU only
  through its PTX is refused when the driver's CUDA release is older than the
  toolkit that emitted it (the driver could not JIT it), with the versions in
  the reason.
  `sweep.backend.torch.binding.diagnostics()` gains a `shim` key, `"ctypes"`
  or `"pybind"`.
  A core sitting under `sweep/lib/<cuN>/` is loaded as it is: in a clone that
  holds one (after `utils/build_cores_manylinux.sh`), edits under `csrc/` do not
  reach `impl='c'` until that core is rebuilt or removed (`SWEEP_CORE`,
  `diagnostics()['shipped_core']` show which core is in use).

- **``boundary_buffer=`` on every propagator**: sigma=0 cells between the
  physical box and the PML ramp.  ``None`` (the default) gives an equation that
  declares ``BOUNDARY_BUFFER_REACH`` its need, ``REACH * M + 1`` cells, under
  every memory strategy and backend; only ``AcousticVRZ`` / ``AcousticVRZ3D``
  declare one.  The buffer is cropped with the pad, so shapes and coordinates
  do not change; boundary saving refuses an explicit value below the need.

- **A persistent backward runner for ``AcousticVRZ3D``**
  (``acoustic_vrz3d_backward_bs_runner``), phase-aware, so the domain-decomposed
  backward no longer re-enters the per-call binding three times per step and
  rebuilds its whole setup each time: on a large 3-D grid (2x2 DD, 4xH100)
  an iteration became about 2.5x faster, gradients and records
  bit-identical.  It owns its boundary saver, so it also serves host-staged
  boundaries.

- **A CPU-only test job** (`.github/workflows/tests-cpu.yml`).  Nothing in this
  repository ran the tests before, and nothing could: `pytest test/` was unable
  to return 0.  With that fixed, every push and pull request to `dev` runs the
  suite on a hosted runner -- 313 tests in about three minutes; the rest skip
  for want of a GPU or a compiled binding.  A skip is not a failure, so the job
  also asserts a floor on the number of tests that actually EXECUTED
  (`.github/scripts/assert_executed_floor.py`): without it, "the tests passed"
  and "the tests did not run" print the same summary.  The bit-exactness gate
  stays where the GPUs are.

- **ViscoAcoustic: Zhu & Harris (2014) nearly constant-Q equation, per-edge
  free surface, and a CUDA backend (`impl='c'`).**  The equation now
  implements the paper's decoupled fractional Laplacians (eq. 10/11,
  doi:10.1190/geo2013-0245.1): Kjartansson power-law dispersion
  `c_p = c0*(w/w0)^gamma` (measured exponent matches theory to 0.4% over the
  band; the amplitude decay matches `exp(-pi f r/(Q vp))` to 2%) and a
  `k^(2*gbar+1)` loss filter, replacing the legacy frequency-independent
  effective-velocity coefficients.  Both terms ride on the CPML step as
  spectral corrections that vanish identically at `gamma -> 0`, so both
  switches off still reduces bit-exactly to `Acoustic`.  Heterogeneous media
  freeze the fractional exponent at the average `gbar` (the paper's
  freezing-unfreezing), including its Q-derivative, on every backend.  The
  reference frequency `omega` is a real model parameter with a genuine
  gradient.  Per-edge free surfaces: every `free_surface=` form the acoustic
  solver accepts.  The CUDA backend reuses the acoustic2d CPML kernels with
  the paper's velocity and applies both spectral terms per step through
  ATen/cuFFT with a hand-derived exact adjoint — forward, full /
  chunk-checkpoint / recursive-checkpoint backwards and RTM (closed-box
  c-vs-eager gradients ~1e-6 for wavelet/vp/Q/omega; ckpt matches full
  storage bitwise on the record).  Boundary saving is refused (the
  dissipative global-FFT term is not reverse-reconstructible from boundary
  strips): the impl='c' default memory strategy falls back to 'full', and an
  explicit boundary request raises.  Note: gradient tests reference the
  UNCOMPILED eager step — inductor's fused pow/ln backward perturbs the
  Q-gradient cotangents enough to be amplified to ~5e-3 by the
  `ln(vp/omega)`-weighted chain; the plain eager step matches the CUDA
  adjoint at ~5e-7.
- `ForwardInput/BackwardInput.eq_aux` — equation-specific auxiliary tensors
  (opaque to the shared autograd wrapper); ViscoAcoustic uses it for its |k|
  FFT grid.
- **Equation registry.**  `@register_equation(name=..., aliases=..., method=...)`
  co-locates an equation's public export names with its class, replacing the
  hand-maintained import wall and `globals()` scan in `sweep.equations`.
  `get_equation(name)`, `list_equations()`, `equation_classes()` and
  `equation_method(cls)` are the lookup side.  Adding an equation is now "write
  the module, decorate the class".  Declaring `C_NAME = "<binding prefix>"` is
  what gives a class its compiled bindings: the base derives `_C` from it as
  `{C_NAME}_forward` plus the four backward variants, so the per-equation `_C`
  boilerplate is gone.  Both are OPTIONAL — subclassing `SecondOrderEquation`
  and writing `_C` by hand still works, which is what notebook 18 does.
- **Persistent stepped runners** (`sweep._C.ForwardRunner` / `BackwardRunner`,
  plus `<equation>_forward_runner` / `<equation>_backward_bs_runner` factories
  for all ten templated equations).  A stepped or domain-decomposed propagation
  used to re-enter the extension every time step and rebuild the entire
  prologue; the runner constructs once per propagation and each step is a
  `run(it_begin, it_end, phase)` call.  Reuse requires gpu-direct boundary
  storage and no checkpointing; anything else keeps the per-call path.
  Measured against dev on DD (median of 5 interleaved rounds, Ada): the runner
  alone is worth 3.99x on 2-D elastic 600x900, 1.56x on 3-D elastic, and
  1.04-1.06x on acoustic, because what it removes is a roughly constant ~0.2-1.4
  ms of host cost per step — so the ratio is set by how much GPU work one step
  does, and equations with more fields to rebind gain most.
- **Typed gradient-memory strategies** — `Full()`, `BoundarySaving(...)`,
  `Ckpt(...)` under `sweep.propagator.options`, passed as `memory=`.  The old
  dict/flag spellings still work and are read exactly as before.
- **Declared capabilities instead of inferred ones.**  `C_NAME`,
  `C_HAS_RECURSIVE_CKPT`, `supports_image_topography[_c]`, the CUDA wavefield
  slot table and the DD admission / tail-truncation eligibility are now class
  attributes an equation declares.  Nothing infers behaviour from the class
  name any more.

### Changed
- **``AcousticVRZ`` / ``AcousticVRZ3D`` gradients**: the vp gradient images
  the second time difference ``U_{it+1} - 2U_it + U_{it-1}`` (the forward
  identity) instead of a second spatial derivative of the stored field, so the
  full, checkpointing and boundary-saving gradients agree to rounding; the full
  history gains one slot (6 in 2-D, 8 in 3-D) and the checkpoint replay buffer
  two rows.  With the default ``boundary_buffer`` the runtime grid of a VRZ
  propagator is ``M + 1`` cells wider per absorbing face, which moves the
  gradient near the PML by ~1e-2 relative on a small grid.  The exact CPML
  adjoint (Fixed) makes the backward ~22% slower and adds three adjoint zeta
  shadows (a small fraction of the wavefield memory).
- **Domain decomposition**: a multi-axis mesh ships every halo field of a step
  in ONE batched P2P across the cut axes instead of one round per axis (the
  strips are latency-bound: a six-field shipment on a large 3-D grid ran at
  6.5 GiB/s over NVLink).
- **Boundary saving no longer re-injects a source that sits in the restore
  strip.**  In the boundary-saving reverse pass the strip cells (the M+1 cells
  inside every non-cut physical edge, i.e. physical rows 0..M under a free
  surface) are restored to the true, source-included field, and the source was
  then added a second time; the strip's finite-difference `u_tt` also kept the
  source term, and the error leaked into the interior through the stencil.
  Any shot within M+1 cells of an edge was affected -- in particular a shallow
  source under a free surface, the common marine setup.  Measured before the
  fix: vp gradient vs `Full()` 4e-4..2e-2 of max|g| (source-row), ~1e-3 in the
  interior, source illumination 1.4e-2; after: <=1.2e-6, and vs eager (FP32)
  identical to `Full()` vs eager.  Fixed for Acoustic, Acoustic3D, AcousticLSRTM
  (2-D/3-D) and ElasticTTI2nd (whose strip is the outermost row/column only);
  VRZ, VTI, DAS, the staggered-grid family and the eager/JAX paths inject in an
  order that never had it.  Sources outside the strips are bit-identical; the
  restore kernels and the new un-injection share one strip predicate
  (`csrc/cuda/common/boundary/strip.cuh`).  `test/test_bs_strip_source.py`
  covers every restore variant (gpu/cpu/disk, fp32/fp16/bf16/int8).
- **`impl` resolution by device.**  `impl='auto'` picks the compiled backend
  on CUDA only; on a CPU it is eager, and an explicit `impl='c'` on a CPU falls
  back to eager with a warning, in every build (see Removed: the compiled CPU
  engine).  Also fixed: a device given as a string with an index
  (`dev='cuda:0'`) was read as "not CUDA", so `impl='auto'` silently resolved
  to eager there.
- **A legacy `boundary_saving_config` dict that sets staging knobs its storage
  ignores reads again.**  `{'storage': 'gpu', 'pinned_memory': True, ...}` was
  accepted (the knob ignored) until the dict route gained the typed capability
  check, which then raised `pinned_memory is only valid when storage='cpu'`.
  The check now validates only the knobs the chosen storage accepts; the typed
  `BoundarySaving(...)` stays strict.
- **Host-staged boundaries under DD no longer pay a per-step Python call.**
  The persistent stepped runners (forward and backward) now accept
  `storage='cpu'`; the core keyed its saves and flushes on the global step
  already, and the backward primes its first chunk once per propagation instead
  of once per step (which re-copied the whole chunk prefix every step).  The
  per-call path caches the host copies of the index tensors per parameter
  object, so it no longer synchronizes the device twice per step.  Measured on
  A100 vs dev: cpu-staged DD forward from 1.4-4x slower to parity, cpu-staged
  DD forward+backward 0.74x (faster), all records and gradients bit-identical.
- **The ctypes layer is the package `sweep/backend/c/`** (`loader.py`,
  `adapt.py`, `entries.py`, `runners.py`, the generated `abi.py`, and `jit.py`,
  which decides where the core comes from), replacing `sweep/_capi.py`,
  `_core_abi.py` and `_jit.py`; `sweep._C` stays the lazy entry point.  It
  lost its output mapping and per-call cache: entries and runners return
  nothing, callers read the tensors they bound.
- **`source_illumination` means the same thing under every memory strategy.**
  It was accumulated by one kernel that both backward paths call, but they
  handed it different fields: `Full()` / `Ckpt()` passed the forward store,
  which for acoustic *is* `u_tt = vp^2*Lap(u)`, while `BoundarySaving()` passed
  the reconstructed raw pressure — so one attribute returned
  `sum_t u_tt^2` or `sum_t u^2` depending on a knob that is supposed to be a
  pure space/time trade, and the two differ by ~3e10 (measured: 5.24e18 vs
  1.59e8 in 2-D, 4.94e18 vs 1.61e7 in 3-D, on a heterogeneous model).  Both are
  legitimate quantities — the Shin (2001) pseudo-Hessian and the RTM
  amplitude-compensation illumination — but nothing said which you were
  getting, and nothing said it changed with `memory=`.  All strategies now
  return the pseudo-Hessian `sum_t u_tt^2` (dimensionally paired with the
  gradient `sum_t u_tt*lambda`, which is what makes `grad/(illum+eps)` sane),
  accumulated over the physical box like the gradient kernels.  The boundary
  path forms `u_tt` from the same three reconstruction time levels
  `calculate_grad_utt_band` uses, in its **own** kernel: no gradient arithmetic
  is touched, and `test/test_illumination_pin.py` pins that enabling
  illumination cannot move the gradient.  `receiver_illumination` stays
  `sum_t lambda^2`; under `BoundarySaving()` it used to differ from `Full()` by
  ~0.5% (measured on dev) and now matches it bit for bit.  The contract is now
  written down in `_CompiledPropagator.__init__`.  Notebook 08 (RTM) shows the
  consequence: its illumination-compensated image is now the image divided by
  the illumination of the same (original-wavelet) field, a dimensionless ratio
  (colour scale ~0.05 instead of ~4e5).
- `SecondOrderEquation._apply_free_surface` — the per-edge pressure-release
  zeroing moved from `Acoustic` to the shared base (bit-identical) so
  ViscoAcoustic reuses it.
- `ViscoAcoustic.prepare_models` maps (vp, Q, omega) -> (vp_step, A) once per
  forward (shared by the eager and CUDA paths; the eager step no longer
  recomputes the dispersion/damping coefficients every time step).
- **Unknown keyword arguments are rejected instead of ignored.**  Both the
  propagator constructor and the call now raise `TypeError` naming the
  offending keywords.  A misspelled option previously did nothing at all and
  the run silently used the default, which is how `full_mode=` survived as a
  dead parameter for as long as it did.
- **Configurations that used to fail silently, late, or not at all now refuse
  at construction**, each with a message saying what is unsupported: recursive
  checkpointing when the compiled binding is absent; image-method topography on
  an equation that does not implement it; a free surface on any anisotropic
  equation (those branches were unreachable and are now an explicit refusal);
  and the legacy dict/flag memory spellings, which now go through the same
  capability check as the typed options.  Illumination/ADCIG crop failures warn
  instead of silently returning zeros.
- **The CUDA drivers are two shared skeletons instead of ten hand-written
  pairs.**  `common/eq_driver.cuh` (acoustic family) and `common/sg_driver.cuh`
  (staggered family) hold the forward and the four backward modes; each
  equation supplies a `driver_traits.cuh` of hooks and a thin forward.cu /
  backward.cu.  All ten templated equations are on it.  Contributor-facing
  only — every mode is bit-identical to the code it replaced, and each skeleton
  carries a HOOK TIMING MAP listing the hooks in call order.  See
  `docs/dev/cuda_drivers.md`.
- **Domain decomposition is declarative.**  A `DDSpec` interpreter drives the
  forward and backward loops from the equation's declared wavefield roles,
  grads layout and coupling exchange; the per-family branches and the
  name-sniffing `_FAMILIES` table are gone.
- `PropBase.__init__` is decomposed into grid geometry, IO validation,
  topography, argument resolution, boundary spec and the two DD blocks, and
  exposes `memory_strategy` and `use_ckpt` as read-only properties.
- **Kernel-level performance, all bit-exact** (measured on RTX 6000 Ada, and
  the direction re-checked on V100): imaging is computed over the physical box
  only rather than the padded box, acoustic boundary-saving imaging is fused
  into the reverse reconstruction, the band imaging kernels put the x offset
  innermost, elastic boundary-saving imaging folds into `stress_adjoint_prepare`,
  the elastic velocity carriers are captured inside the NOPML kernel, and the
  elastic3d forward kernels take `__launch_bounds__(256, 4)` (V100 stress kernel
  -13%, Ada -4%, no register spill).  Together with the runner these are worth
  1.20-1.34x on DD acoustic end-to-end, on top of what the runner itself gives.

### Deprecated
- `boundary_saving_config={...}` and `MemoryOptions(strategy=..., boundary=...,
  ckpt=...)`, in favour of `memory=Full()` / `BoundarySaving(...)` / `Ckpt(...)`.
  Both still work, are still read exactly as before, and now emit a
  `DeprecationWarning`; the removal criterion is written next to the shim.
  (`use_ckpt=` is NOT deprecated — it is still a plain supported keyword.)

### Removed
- **The compiled CPU engine** (`csrc/cpu/**`, about 20k lines), which the
  `SWEEP_JIT_FULL=1` pybind shim and an AOT-built `sweep._C` compiled in, with
  `SWEEP_SKIP_CPU` and its stub.  The wheel never shipped it, and its gradients
  disagreed with eager and with the CUDA core (adjoint-source scaling,
  checkpoint modes that crashed, stress receivers with the wrong sign, VRZ, no
  free-surface backward): the CUDA backward moved to exact discrete adjoints
  and the CPU copy never followed.  **Migration:** run CPU propagators with
  `impl='eager'`, which is what `impl='auto'` already picked there.  An
  explicit `impl='c'` on a CPU falls back to eager with a warning, and a host
  tensor handed to a compiled entry is refused with the same message by the
  ctypes layer and by the pybind shim.  The examples' legacy `--backend cpu`
  now means eager on the CPU, and `--impl c --device cpu` is refused.
  `test/cpu_binding_gradient_consistency.py` became `test/gradient_cases.py`,
  the case library of `test/backend_gradient_matrix.py`, which lost its
  `c-cpu` column.
- **`solver.rtm()` and its three compiled bindings** (`acoustic2d_rtm`,
  `acoustic3d_rtm`, `visco_acoustic2d_rtm`), together with `_C_rtm` on the
  equations and the 2-D host wrapper, the four 3-D `rtm_*_impl` wrappers and
  their dispatcher.  The entry had no caller left: the RTM notebook computes
  the image as the gradient of an inner-product loss, sweep-tasks' production
  RTM migrated to forward+backward with `compute_illumination`, and no
  bit-exact gate ever covered it.  **Migration:** run the ordinary forward and
  backward with `compute_illumination=True` and read the image off the
  illumination path — that is what `rtm()` did internally.  `RTMOutput`,
  `accumulate_rtm_image_*`, `init_rtm_output_*` and the `run_*_imaging`
  machinery are deliberately KEPT: despite the names they are the illumination
  path of the ordinary backward, which production RTM uses.  An old script
  calling `rtm()` now gets a plain `AttributeError` rather than a tombstone,
  so it cannot quietly return the wrong thing.
- **`full_mode=` propagator keyword** (and `PropagatorDefaults.full_mode`).  It
  was stored on the instance and never read by anything — a dead parameter.
  Because unknown keywords are now rejected, passing it raises `TypeError`
  instead of being silently ignored; delete it from the call.  Nothing replaces
  it; the memory strategy is chosen with `memory=Full()` / `BoundarySaving(...)`
  / `Ckpt(...)`.

### Fixed
- **Boundary saving held a whole forward history in the backward** for
  ``AcousticVRZ`` / ``AcousticVRZ3D``, ``AcousticVTI1st`` / ``AcousticVTI1st3D``,
  ``ElasticTTI2nd`` and ``DASZhao``.  Every boundary-saving backward bound a
  per-call replay buffer of ``nt`` padded grids times one to five fields that
  their drivers never read -- boundary saving still paid one to five full
  grids per time step (notebook 17 asked for 68.6 GiB and failed on a 32 GB
  GPU).  Introduced with the Python-side allocation.  Only a driver whose
  boundary-saving backward re-runs the forward into that buffer declares
  ``CUDALayoutSpec.bs_backward_replays_forward`` and gets it (``DASZhao3D``,
  through a call-time ``boundary_saving_config``); every other backward's own
  allocations no longer grow with ``nt`` (nt 150 -> 450: 3-D VTI 185 -> 547 MiB
  before, 3.2 MiB after).
- **``AcousticVRZ`` / ``AcousticVRZ3D`` imaged the wrong time step.**  The
  deferred history capture ran after the buffer rotation, so full and
  checkpointing imaged ``U_{it+1}`` while boundary saving imaged ``U_{it-1}``
  (boundary saving vs full rel 0.12); every mode now images ``U_it``
  (4e-4, the rest of which was the 2-D seed below).  Only VRZ used that hook.
- **VRZ boundary saving imaged unrestored cells.**  Its gradient reaches ``2M``,
  one ``M`` past the ``M + 1`` shell, so the outermost physical cells read what
  the reverse step left there (up to 9x the largest full-storage gradient on a
  small 3-D grid; 99.6% of the Marmousi residual in the outermost cell).  The
  sigma=0 ``boundary_buffer`` moves the shell out of the stencil's reach
  instead of widening it (peak memory about a fifth lower than with a
  ``2M + 1`` shell on a 3-D single shot).
- **2-D VRZ boundary saving started its reverse pass from a damaged state.**
  The reconstruction seed zeroed the pad ring of the two last wavefields, as
  ``Acoustic`` does (harmless there), but the VRZ restore band starts ``M``
  cells inside the pad and the first reverse step reads those cells before any
  restore.  Boundary saving vs full was 3.5e-4 to 2.6e-3 whenever the wavefield
  still reached the pad at the last step; it is now 2e-7 to 3e-6, the level of
  the 3-D runner, which never zeroed.
- **The compiled VRZ adjoint was an approximation in the PML band** (it applied
  the forward CPML operator to the adjoint field).  It is now the exact
  discrete transpose, 2-D and 3-D: compiled vs eager 1.7e-7 (2-D) and 8e-7
  (3-D) with TF32 off, finite differences to 1e-4.
- **DD sub-groups ignored the configured NCCL timeout** and fell back to the
  default one.
- **`pytest test/` could never exit 0.**  `test_import_does_not_pull_optional_deps`
  asserted `mod not in sys.modules`, which is process-global: by the time it
  runs it is a statement about everything the preceding ~600 tests imported,
  not about `sweep.datasets`.  Every recorded full-suite run ended
  `1 failed, 886 passed`, so no script, hook or CI job could gate on the suite.
  The check now runs the import in a fresh interpreter and asserts on *its*
  `sys.modules`, which is both order-independent and what the test always meant
  to say.
- **The JIT staging directory ignored edits to `csrc`.**  `_stage()` mirrors
  the CUDA tree into the torch-extension build dir with unique compiled-source
  basenames; the directory was named after the installed `sweep-solver`
  version and the staleness check was "does `.staged` exist".  Editing a kernel
  without bumping the version therefore left the previous copy in place, ninja
  compiled the OLD source, and the `.so` silently did not contain the edit --
  which is why working on `csrc` came with a "delete the extension directory
  first" ritual.  The sentinel is now a manifest of per-file SHA-256 digests:
  only files whose contents changed are re-staged (so ninja still rebuilds just
  the affected translation units, and through its header depfiles whatever
  includes a changed `.cuh`), files whose source disappeared are removed from
  the stage, and a re-staged copy is stamped with the current time so a source
  that moves BACKWARDS in time (`git checkout` of an older revision) still
  invalidates the object built from it.  The staged tree is byte-identical to
  what the previous implementation produced (150 files, the same 50 compiled
  sources); the manifest costs ~44 ms once per process.
  The staging is also taken under `torch.utils.file_baton.FileBaton`, the
  mechanism torch uses for concurrent extension builds.  `torchrun` starts
  one process per GPU and they all stage before `cpp_extension.load` takes
  its own lock, so nothing serialised them: two ranks copied the tree on
  top of each other, and on a real 150-file tree over a shared filesystem
  one lost with `FileExistsError` on the stage directory -- which is what
  killed a 2-rank domain-decomposition benchmark before it measured
  anything.  The regression test asserts on work rather than timing:
  staging from two processes must copy no more than staging from one.
- **Three test files turned a build failure into a green skip.**  They probed
  the extension as `try: import sweep._C as _C; return hasattr(_C, "sym")
  except Exception: return False`.  `sweep._C` is a lazy shim whose attribute
  access triggers the JIT, so when nvcc failed torch raised, the `except`
  swallowed it, the module-level `skipif` fired, and 38 collected items
  reported green on a build that does not exist.  `test/conftest.py` now owns
  the one decision: `requires_binding(*symbols)` answers from
  `is_torch_binding_available()`, which compiles nothing; a binding that IS
  present but lacks a named symbol raises, because that is a stale or partial
  build rather than a missing capability; and `SWEEP_TEST_REQUIRE_CUDA=1` turns
  the skip itself into a collection error, so a machine that is supposed to
  have a GPU cannot report green over half the suite.

- **The staggered family refused host-staged boundary storage under stepping
  and domain decomposition.**  `sg_check_stepped_backward` rejected
  `storage='cpu'` and `storage='disk'` for every stepped or DD `backward_bs`
  across the seven equations on that skeleton (elastic2d/3d, das_mu2d/3d,
  elastic_tti_sg2d/3d, elastic_vr2d), which capped the largest elastic 3-D DD
  tile at whatever boundary ring fits in VRAM.  Two things actually blocked it,
  neither a design limit: the skeleton built its own copy stream per call
  instead of taking the Python-owned `BoundarySession`, so under DD -- where
  every time step is a separate entry into the extension -- the stream and its
  ring events were destroyed and rebuilt every step and no transfer could stay
  in flight; and it primed the TAIL chunk on every segment
  (`prefetch_initial_backward_chunk` with no `it_hi`), so a stepped reverse
  loop fetched the wrong slabs on all but its first call.  Both are fixed and
  the refusal is narrowed to match the acoustic family: disk is still refused
  under a cut mask, cpu is not.  Gate tiers A/C/B/dd1 unchanged.

- **Every backward allocated the RTM/illumination buffers, requested or not.**
  `init_rtm_output` did three `torch::zeros_like(vp)` on the runtime-padded
  grid unconditionally, while the kernels that read them are gated on
  `compute_illumination || compute_adcig` -- and `compute_illumination` is
  itself derived from whether Python allocated a real buffer.  So a plain FWI
  gradient allocated and memset three full padded fields no kernel touched.
  The allocation now takes the same predicate the launch already takes.
  Measured backward-phase peak: `Acoustic` at 724x324 padded 0.024 -> 0.021 GB
  and `Acoustic3D` at 212^3 padded 1.222 -> 1.108 GB, i.e. exactly the 3 MB /
  114 MB the three fields occupied, with records and gradients bit-exact and
  gate tiers A/C/dd1 unchanged.

- **A demoted `impl='c'` lost the memory strategy the caller asked for.**  When
  the compiled binding is unavailable, or the equation has no `_C`, `impl='c'`
  falls back to eager and the cuda-only knobs are dropped with it.  The
  gradient-memory strategy is not a cuda-only knob, though -- it just travels
  inside `cuda_options` -- so dropping the object took the request with it and
  the eager backend fell back to its own default, checkpointing.  A caller who
  asked for boundary saving got `'ckpt'`, silently.  The request is now carried
  across the demotion (an explicit `memory=` still wins), so it reaches
  `check_memory_supported` and the backend actually runs what was asked for.

- **The staggered backward allocated a dead wavefield buffer per call.**
  `SgBackwardBsRunner` passed `{}` for the saver's `last_two`, so
  `allocate_last_two` took its self-allocating branch and built a
  `{BS_NVAR, 1, B, 1, nz[, ny], nx}` FP32 buffer -- 9 padded 3-D grids for
  `Elastic3D` and `ElasticTTI_SG3D`, 15 for `DASMu3D` -- on the GPU, or in
  PINNED HOST memory on the staged path.  Nothing touches it: `save_last_state`
  writes `last_two` in the FORWARD, and the backward seeds from
  `p.u_last_two` directly.  Binding the tensor Python already owns returns all
  of it.  Measured at a 180^3 padded grid, backward-phase peak: `Elastic3D`
  2.737 -> 2.527 GB and `DASMu3D` 4.278 -> 3.928 GB, i.e. exactly the 210 MB /
  350 MB the buffer occupied, with records and every model gradient bit-exact.
  Same defect and same fix the acoustic skeleton already had.
- **Domain decomposition re-emitted staging knobs the storage rejects.**
  `ModelParallel` rebuilt the wrapped propagator's boundary strategy as a typed
  `BoundarySaving` carrying whatever it inherited, including
  `transfer_interval` / `ring_buffers` / `pinned_memory` for `storage='gpu'`,
  which `BoundaryOptions.__post_init__` refuses -- so a configuration the legacy
  dict route accepted became a construction error, including the gpu baseline
  of `test/dd_session_bench.py` at that file's own default, which is the
  reference its bit-exactness gates compare against.  Only the knobs that apply
  to the chosen storage are passed on now.

- **3-D DAS boundary saving cost more than full storage, and was the default.**
  `das3d::backward_bs` is three lines that re-run the forward and allocate the
  entire `{nt, 3, B, nz, ny, nx}` strain history -- exactly what full storage
  holds -- while the forward writes no boundary strips at all (it returns an
  empty `last_two`).  On top of that the Python side allocated a 13-field
  boundary ring and a 13-grid `last_two` that nothing writes and nothing reads.
  Since boundary saving is the implicit `impl='c'` strategy, every plain 3-D DAS
  gradient paid strictly MORE than `'full'` for a "memory-saving" mode, and
  every gradient test passed because the answer was right and nothing asserted
  on memory.  `DASZhao3D.supports_boundary_saving_c = False` now routes the
  default to `'full'` and makes an explicit boundary request raise, the contract
  `ViscoAcoustic` already had.  This is "not implemented", not "impossible":
  `das2d` writes real strips and `DASMu` / `DASMu3D` get them from the shared
  staggered skeleton.  `test/solver_gradient_mode_suite.py` drops the 14 `bs_*`
  entries that all ran the same code as `full`.
- **A propagator could step with another propagator's CPML profiles.**
  `PropBase.init_abc` caches the profiles on the EQUATION (`equation.b`) but
  held the freshness key on the PROPAGATOR.  Sharing one equation between
  propagators is a supported pattern -- `ModelParallel` builds a second
  propagator over the wrapped one's equation -- and anything that changes the
  pad (a free surface, a different `abcn`, a DD tile) changes the profile
  length.  Two propagators therefore each started with their own `None` key,
  both built, and whichever ran last owned `equation.b` while the other's key
  still matched, so its rebuild was skipped and it handed the kernel profiles
  built for a different padded shape.  Nothing validates the length on the way
  in.  Measured on a shared `Acoustic`: `abcn=10` (padded 60x70) then
  `abcn=30` (padded 100x110), then the first propagator again -- its z profile
  stayed 100 long instead of 60.  `docs/notebooks/02_fwi_elastic_marmousi.ipynb`
  builds `solver`, then `solver_fs`, then runs `solver` again, so it hits this.
  The key now lives on the equation next to the value it describes.

- **`AcousticVTI1st` on `impl='c'` read the grid spacing with the axes
  swapped.** `PropBase` stores spacing in model-axis order `(dz, dx)` and
  `_cuda_spacing()` reverses it, so `p.spacing` reaches the kernels as
  `[dx, dz]` -- as every other driver reads it, including this equation's own
  3-D twin. `acoustic_vti_1st_2d` read it as `(dz, dx)` in all four entry
  points, which is identical whenever `dx == dz` and wrong the moment they
  differ: at `dh=(dz=10, dx=20)` the compiled record's cosine against eager was
  **-0.62** (and **0.10** at `(20, 10)`), with the model gradients scattered.
  With `dh=10` it was 1.000000, which is why nothing caught it -- the gate, the
  gradient matrix and every test pass a scalar `dh`, though `normalise_spacing`
  has always accepted a per-axis sequence. Isotropic results are unchanged (the
  two values are equal, so the swap is a no-op there).
- **`gather_record` crossed shot groups.**  The record gather ran over the
  WORLD process group, but with `shot_groups > 1` every group propagates a
  different shot through the same tile grid, so ranks sharing a tile
  coordinate carry the same global receiver indices holding different shots'
  traces.  Rank 0 wrote all of them into one array and whichever tile was
  assembled last silently won, so the "global record" was spliced from several
  shots and nothing raised.  The gather now runs over `mesh.model_pg` -- the
  `py * px` ranks that decompose ONE shot -- and each group assembles its own
  record on its own root (`shot_group * py * px`), which is rank 0 for the
  `shot_groups == 1` case the guides describe.  The collective and the
  index placement moved to `sweep.parallel.gather_tile_records` /
  `assemble_tile_records` (the inverse of `partition_global_coords`), so the
  regression test `test/test_dd_gather_record_shot_groups.py` exercises the
  real collective on gloo/CPU without a GPU.
- **Domain decomposition silently answered a geometry it cannot solve.**  A
  3-D source/receiver array means source encoding, whose leading axis is 1;
  anything longer is what a caller writes when they mean "several shots", and
  the single-domain propagator rejects it.  `ModelParallel._prepare_call` did
  not: it boolean-indexes the owned sources of EVERY leading entry into one
  flat array, and reads the receivers -- and their ownership, which
  `own_receiver_indices` publishes and the parallel guide promises is a
  partition of the global receiver list -- from index 0 alone.  So several
  shots were fired as one fused supershot, recorded at the first entry's
  receivers, and returned without an error.  Both arrays are now checked, with
  a message naming the three supported ways to run several shots.
- **`downsample=` returned a view that pinned the full-resolution array.**
  `decimate` ended in basic slicing, so the decimated model kept the array it
  came from alive through `.base` for as long as the caller held it -- a few
  megabytes pinning a few hundred on the 3-D benchmarks.  It returns a
  C-contiguous copy now, matching `read_model`; the `factor == 1` path stays
  zero-copy.
- **The Dynamo recompile-cap bump did nothing on torch < 2.6.**  The eager step
  raises Dynamo's cap before compiling, because each wavefield's
  `requires_grad` flips on first use and the many-wavefield equations exhaust
  the default of 8 before specialization settles -- at which point Dynamo falls
  back to eager silently.  The knob was renamed (`cache_size_limit` up to torch
  2.5, `recompile_limit` from 2.6) and the bump keyed on the new name behind a
  `hasattr` guard, so on an older torch it was a no-op and the fallback it
  exists to prevent happened anyway.  `torch` is unpinned here, so that is a
  shipped configuration.  Whichever name the installed Dynamo exposes is raised
  now, including the `accumulated_` secondary cap, and a Dynamo with neither
  warns once instead of passing in silence.
- **The eager wavefield-snapshot buffer was cached forever and then copied.**
  `return_wavefield=True` took its buffer from `_workspace_cache`, which has no
  eviction policy, so a buffer sized
  `nsnapshots x nwavefields x B x prod(padded_shape) x 4` -- and the default is
  a snapshot at EVERY time step -- stayed pinned to the propagator for its
  whole lifetime, long after the caller had finished with the snapshots.
  Because it was workspace, it then had to be cloned on the way out, so the
  path held two copies at once.  Measured on `Acoustic` at 136x176 padded with
  `nt=200`: 115 MB per copy.  The buffer is allocated per call now and returned
  directly.
- **The eager record was built by `nt` in-place slice writes into a live
  autograd tensor.**  That builds a chain of `nt` `CopySlices` nodes, and each
  one allocates a full-record buffer and copies the incoming gradient through
  it, so the record's own backward cost was `2 * nt * |record|` -- 64 GB per
  shot at `nt=4000` with 500 receivers, terabytes for a large node gather.  The default
  rollout collects the per-step gather and stacks once, making it
  `2 * |record|` and removing `nt` full-record allocations.  Records and
  gradients are bit-exact (`torch.equal`); the whole backward, on CPU with 500
  receivers on a 40x60 grid, goes 136 -> 129 ms at `nt=250`, 280 -> 235 ms at
  `nt=500` and 616 -> 497 ms at `nt=1000` -- the share grows with `nt` because
  the removed term is the quadratic one.  The stacked record is already fresh,
  so it is returned without the defensive clone the workspace buffer needed.
  The checkpointing and eager-boundary-saving rollouts still fill a
  preallocated record: their write is per CHUNK, not per step.
- **The source injection mask was built on paths that never read it.**
  `SourceTorch` allocated a whole padded wavefield, filled it with an
  `index_put_` and, with a spread kernel, convolved it -- on every
  construction.  Exactly one place reads it, `SourceBase.forward`, which
  `SourceTorch.forward` reaches only when neither source encoding nor adjoint
  modelling is active; both of those inject through `_add_indexed_sources` and
  never look at it.  So every encoded forward and every adjoint construction
  paid `prod(padded_shape) * 4` bytes plus the scatter and the convolution for
  a buffer nothing read.  It is built on the branch that reads it now.
- **A SEG-Y read peaked at 7.3x its payload.**  `segy_to_array` made the whole
  file's traces contiguous in one go -- the memmap rows are strided, so dropping
  the 240-byte trace headers needs a copy -- and handed the result to
  `_ibm_to_ieee`, which holds five temporaries the size of what it is given
  (sign, exponent, mantissa, the `ldexp` result and the negation).  Both scaled
  with the file rather than with a bound.  Measured on a synthetic IBM-float
  file, resident growth over the read: 60 MB payload 440 MB (7.33x), 120 MB
  payload 879 MB (7.33x).  Reading in blocks of ~16 MiB of samples caps the
  temporaries: the same files now grow 224 MB (3.73x) and 356 MB (2.96x), and
  the ratio keeps falling toward the inherent floor of the output array plus the
  mapped pages, because the temporary budget no longer scales with the file.

- **Disk-staged boundary saving reconstructed a wrong gradient.**  Since the
  persistent staging session / non-blocking copy stream (PR #81), every
  `storage='disk'` gradient was wrong: max|disk-gpu|/scale 0.2-0.5 for acoustic
  2-D/3-D and 0.8-5.0 for nvar>1 3-D (Elastic3D, DASMu3D); cpu staging with
  `ring_buffers >= 2` was off by 1.4-2.4x on the same equations.  Three defects in
  `boundary/runtime.cuh`: (1) `prefetch_next_backward_chunk_if_needed` issued
  the next chunk early on `ring_buffers >= 2`, but the synchronous disk path is
  pinned to slot 0 whatever `ring_buffers` says (and defaults to 3 / 2), so the
  early H2D overwrote the chunk still being restored -- the predicate is now the
  slot assignment; (2) the synchronous-disk enqueue never got the
  `cudaStreamWaitEvent(compute_ready_)` write-after-read fence the host-staging
  branch has; (3) the two nvar>1 restore readers kept the slot-0 override for
  cpu staging with `ring_buffers >= 2`.  Regression test
  `test/test_boundary_disk_staging_slot.py`: disk (default knobs and short
  chunks) and cpu ring 2 must be bit-exact against gpu-direct.

## [0.2.0] - 2026-08-24

### Added
- **Domain decomposition (`sweep.parallel`).**  `ModelParallel` splits one
  model into tiles — one GPU per tile — and exchanges a halo every time step,
  so a single shot is solved cooperatively instead of replicated.
  `MeshTopology(py, px, shot_groups=...)` describes the rank grid and composes
  with shot parallelism; `pad_to_mesh` / `unpad_from_mesh` size a model to the
  tile multiple.  Forward and backward are plain autograd, and the gradient is
  **bit-identical** to the single-domain gradient on fp32 GPU boundaries.
  Supported equations: `Acoustic` (2-D), `Acoustic3D`, `AcousticVRZ3D`,
  `Elastic` (2-D), `Elastic3D`; anything without stepped kernels is refused at
  construction.  See the [Domain decomposition](docs/user-guide/parallel.md)
  guide and notebooks 25 / 26.
- **CPU-staged boundary storage under domain decomposition.**
  `BoundaryOptions(storage="cpu")` now works for the Acoustic 2-D/3-D DD
  backward, so a tile whose boundary ring does not fit in GPU memory has a
  fallback instead of a hard stop.  The gradient is **bit-identical** to
  gpu-direct on fp32 and bf16, and within each dtype's own run-to-run floor on
  fp16/int8; it composes with `tail_steps`.  `storage="disk"` under DD, and
  cpu staging on a single-tile mesh, are still refused — by name, at the first
  backward.  Elastic DD remains gpu-direct only.  See
  [Domain decomposition](docs/user-guide/parallel.md#boundary-storage-under-dd).
- **`BoundaryOptions.tail_steps`** (dict spelling:
  `boundary_saving_config={'tail_steps': K}`): keep only the last `K` steps of
  the boundary ring and stop the reverse loop there.  For steady-state
  objectives (frequency-selection / encoded FWI) whose adjoint source is zero
  outside a probe window, the truncated gradient is the same gradient; the
  reverse pass and the ring both shrink proportionally.  Acoustic 2-D/3-D,
  boundary-saving backward only, and it composes with domain decomposition.
- **CPML aux strip allocation.**  `psi`/`zeta` (acoustic) and the elastic
  memory variables now live in per-axis slabs — the PML band plus stencil
  reach — instead of full grids, for `Acoustic`, `Acoustic3D`, `Elastic` and
  `Elastic3D` on `impl='c'`.  Gradients are bit-for-bit unchanged; only the
  allocation shrinks.  Equation authors opt in via the new `CUDALayoutSpec`
  fields `pml_slot_axes`, `checkpoint_slot_axes` and `adjoint_pml_slab`.
- Per-edge free surface (deepwave-style).  `Propagator(free_surface=...)` now
  accepts a per-edge spec — an edge-name list (`['top', 'left']`), a
  length-`2*ndim` bool mask (`[z0, z1, x0, x1]`), or a dict — in addition to the
  historical `bool` (top-only): a free surface on any subset of the domain
  faces.  `abcn` likewise accepts a per-edge list for an independent PML
  thickness per face.  The **eager** backend supports it for **Acoustic and
  Elastic 2-D** (all four edges, gradient-consistent); the compiled `impl='c'`
  backend supports it for **Acoustic and Elastic 2-D on CUDA** (all four edges,
  including z∩x corners) — bit-exact vs eager forward, adjoint-gradient cosine
  ~1.  `free_surface=True` / a scalar `abcn` stay bit-for-bit unchanged.
  On `impl='c'` all three CUDA backward memory modes — **full**
  (`use_ckpt=False`), **checkpointing** (`use_ckpt=True`), and **boundary
  saving** — are gradient-consistent for every edge and z∩x corner
  (adjoint cosine ~1 vs eager).  Other unimplemented requests raise a clear
  `NotImplementedError` pointing at `impl='eager'`: per-edge on 3-D or on
  non-migrated equations, per-edge on the CPU `impl='c'` backend, and per-edge
  PML *thickness* on `impl='c'`.
- Documentation overhaul (Phase 1, facade): rewritten landing page with
  capability cards and audience-routed navigation; README and README.zh-CN
  gained badges and a tagline block.
- `CHANGELOG.md` and `CONTRIBUTING.md` scaffolding.

### Changed
- **The gradient-memory mode is now one three-way choice** — `'full'`,
  `'boundary'` or `'ckpt'` — resolved identically for the eager and CUDA
  backends by `resolve_memory_strategy`, and selected in one place with
  `memory=MemoryOptions(strategy=...)`.  The legacy `use_ckpt` /
  `boundary_saving_config` knobs still work and resolve into the same choice.
  Three behaviour changes come with it:
  - `boundary_saving_config={'enabled': True}` now actually runs the boundary
    backward on both backends.  It used to lose silently to the `use_ckpt=True`
    default, so scripts that believed they were using boundary saving were
    checkpointing (`impl='c'`) or ignoring the dict entirely (`impl='eager'`).
  - Contradictory requests raise `ValueError` instead of one path winning
    silently — `use_ckpt=True` together with an enabled `boundary_saving_config`,
    or `memory=` contradicting a legacy knob.  Knobs that *agree*
    (`memory=MemoryOptions(strategy='boundary')` with `use_ckpt=False`) are
    accepted.
  - A dict passed without `enabled=True` (e.g. `{'storage': 'cpu'}`) selects
    `'full'`, not checkpointing.
  Unchanged on purpose: no knobs at all still means boundary saving for
  `impl='c'` and checkpointing for the eager backend, and an explicit
  off-switch (`use_ckpt=False`) still means full-wavefield storage.
- `docs/user-guide/equations.md`: summary table expanded from 3 rows to cover
  all 20+ exported equation classes, grouped by physics family. Template
  reminder at the bottom replaced with a "See Also" cross-reference block.
- `mkdocs.yml`: enabled `attr_list` and `md_in_html` Markdown extensions to
  support Material grid-card layouts.

### Fixed
- **CPML aux writes on a domain-decomposition cut tile.**  With the strip
  allocation, the per-axis PML compute band reaches columns on a cut face that
  carry no slab storage; the unclamped index produced a negative offset and the
  ungated store aliased `±0` into another row's slab cell, racing its owner.
  Writes are now gated on `stored()` and read through the clamped accessor
  (`aux_rd_*`).  Only reachable with `impl='c'` acoustic + multi-GPU DD, and
  never on a released build; single-domain runs are bit-for-bit unchanged.
- **The boundary spec can no longer be changed after construction.**
  `prop.free_surface = ...` (and `fs_faces`, `abcn`, `pad`, `pml_type`,
  `topography`) used to land on the `PropTorch` wrapper, where it shadowed the
  backend's value: the read-back reported the new setting while every kernel
  kept the old one — a script could believe it had switched a free surface on
  and quietly model without one.  The write now raises `AttributeError` and
  points at the constructor.
- **DAS Mu 2-D/3-D were non-deterministic on `impl='c'`.**  `das_mu*/kernels.cuh`
  includes the elastic kernels, which address the CPML memory variables through
  the solver's aux slabs, but the DAS drivers never installed them: the row
  stride collapsed to zero and every row aliased the first, so the same input
  gave a different answer each run (plain single-GPU forward, with or without a
  free surface).  `AcousticLSRTM` and `AcousticVRZ3D` borrow the acoustic
  kernels the same way but never launch the slab-addressed ones and were
  unaffected.  `test/test_c_aux_slab_repeatability.py` now pins the class.
- **`AcousticVRZ3D` boundary staging with `storage_dtype='fp16'`/`'int8'`.**
  The 2-D and 3-D VRZ paths now pass `boundary_tangent_pad = M` into the
  effective-boundary saver, fixing an out-of-bounds staging copy on the
  low-precision ring.

## Earlier history

Earlier release notes will be backfilled from the commit history. For now,
see the
[GitHub commit history](https://github.com/DeepWave-KAUST/sweep/commits/dev)
for changes prior to this entry.

[Unreleased]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.6...dev
[0.3.6]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/DeepWave-KAUST/sweep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DeepWave-KAUST/sweep/compare/v0.1.0...v0.2.0

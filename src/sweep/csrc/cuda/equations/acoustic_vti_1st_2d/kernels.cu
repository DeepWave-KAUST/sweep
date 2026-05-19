// All kernels for acoustic_vti_1st_2d live in kernels.cuh (templated on Order
// and instantiated inline via the LAUNCH_VTI_* macros in forward.cu).
// This translation unit is intentionally empty — kept for parity with the
// other equation directories so the build glob `**/*.cu` finds the path.

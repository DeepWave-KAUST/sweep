// Translation unit anchor for the 3D acoustic VTI first-order kernels.  All
// kernels are templated in kernels.cuh; the actual instantiations are emitted
// from forward.cu and backward.cu through the LAUNCH macros.  This empty
// .cu keeps the build system's source-file enumeration consistent with the
// 2D companion (acoustic_vti_1st_2d/kernels.cu).

// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(acoustic_vrz3nd_fn, acoustic_vrz3nd_by_order, acoustic_vrz3nd);
SWEEP_BY_ORDER_TABLE(acoustic_vrz3nd_nopml_fn, acoustic_vrz3nd_nopml_by_order, acoustic_vrz3nd_nopml);
SWEEP_BY_ORDER_TABLE(build_vrz_grad_fields_3d_fn, build_vrz_grad_fields_3d_by_order, build_vrz_grad_fields_3d);
SWEEP_BY_ORDER_TABLE(calculate_grad_vrz3d_fn, calculate_grad_vrz3d_by_order, calculate_grad_vrz3d);
SWEEP_BY_ORDER_TABLE(build_vrz_adjoint_coeffs_3d_fn, build_vrz_adjoint_coeffs_3d_by_order, build_vrz_adjoint_coeffs_3d);
SWEEP_BY_ORDER_TABLE(acoustic_vrz3nd_adjoint_fused_fn, acoustic_vrz3nd_adjoint_fused_by_order, acoustic_vrz3nd_adjoint_fused);
SWEEP_BY_ORDER_TABLE(calculate_grad_vrz3d_fused_fn, calculate_grad_vrz3d_fused_by_order, calculate_grad_vrz3d_fused);

// Source-cell correction of the p_tt imaging (see the 2-D twin): the stored
// second time difference contains the wavelet add_source_3d added at it;
// add back the 2·λ·S/v the imaging subtracted.  Same indexing as add_source_3d.
__global__ void vrz3d_utt_source_correction(
    float* __restrict__ grad_vp,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 3)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= solver.B || s >= nsrc) return;
    int base = (b * nsrc + s) * 3;
    int ix = sources_loc[base + 0];
    int iy = sources_loc[base + 1];
    int iz = sources_loc[base + 2];
    if (ix < 0 || ix >= solver.nx || iy < 0 || iy >= solver.ny || iz < 0 || iz >= solver.nz) return;
    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    long long idx = (long long)b * spatial_size
                  + ((long long)iz * solver.ny + iy) * solver.nx + ix;
    long long src_idx = ((long long)b * nsrc + s) * solver.nt + it;
    atomicAdd(&grad_vp[idx], 2.0f * lambda_now[idx] * source[src_idx] / vp[idx]);
}

__global__ void build_kappa_lambda_vrz3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<2>(solver, ix, iy, iz, b);
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = b * spatial_size + iz * solver.nx * solver.ny + iy * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

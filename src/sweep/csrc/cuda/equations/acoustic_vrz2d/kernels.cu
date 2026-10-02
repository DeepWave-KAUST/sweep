// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(acoustic_vrz2nd_fn, acoustic_vrz2nd_by_order, acoustic_vrz2nd);
SWEEP_BY_ORDER_TABLE(acoustic_vrz2nd_nopml_fn, acoustic_vrz2nd_nopml_by_order, acoustic_vrz2nd_nopml);
SWEEP_BY_ORDER_TABLE(calculate_grad_vrz2d_fn, calculate_grad_vrz2d_by_order, calculate_grad_vrz2d);
SWEEP_BY_ORDER_TABLE(build_vrz_grad_fields_fn, build_vrz_grad_fields_by_order, build_vrz_grad_fields);
SWEEP_BY_ORDER_TABLE(build_vrz_adjoint_coeffs_fn, build_vrz_adjoint_coeffs_by_order, build_vrz_adjoint_coeffs);
SWEEP_BY_ORDER_TABLE(acoustic_vrz2nd_adjoint_fused_fn, acoustic_vrz2nd_adjoint_fused_by_order, acoustic_vrz2nd_adjoint_fused);
SWEEP_BY_ORDER_TABLE(calculate_grad_vrz2d_fused_fn, calculate_grad_vrz2d_fused_by_order, calculate_grad_vrz2d_fused);

// The p_tt imaging uses U_{it+1} − 2U_it + U_{it−1}, which at a source cell
// is dt²·rhs + S_it (add_source adds the wavelet to u_next after the step,
// unscaled).  The exact adjoint wants dt²·rhs alone: add back the 2·λ·S/v the
// imaging subtracted.  Same indexing as add_source.
__global__ void vrz2d_utt_source_correction(
    float* __restrict__ grad_vp,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= solver.B || s >= nsrc) return;
    int base = (b * nsrc + s) * 2;
    int ix = sources_loc[base + 0];
    int iz = sources_loc[base + 1];
    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) return;
    long long spatial_size = (long long)solver.nx * solver.nz;
    long long idx = (long long)b * spatial_size + (long long)iz * solver.nx + ix;
    long long src_idx = ((long long)b * nsrc + s) * solver.nt + it;
    atomicAdd(&grad_vp[idx], 2.0f * lambda_now[idx] * source[src_idx] / vp[idx]);
}

__global__ void build_kappa_lambda_vrz2d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = b * spatial_size + iz * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

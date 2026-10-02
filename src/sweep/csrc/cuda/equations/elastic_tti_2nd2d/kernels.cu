// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

namespace elastic_tti_2nd2d {
SWEEP_BY_ORDER_TABLE(tti2nd_stress_kernel_fn, tti2nd_stress_kernel_by_order, tti2nd_stress_kernel);
SWEEP_BY_ORDER_TABLE(tti2nd_displacement_kernel_fn, tti2nd_displacement_kernel_by_order, tti2nd_displacement_kernel);
SWEEP_BY_ORDER_TABLE(tti2nd_stress_kernel_nopml_fn, tti2nd_stress_kernel_nopml_by_order, tti2nd_stress_kernel_nopml);
SWEEP_BY_ORDER_TABLE(tti2nd_displacement_kernel_nopml_rev_fn, tti2nd_displacement_kernel_nopml_rev_by_order, tti2nd_displacement_kernel_nopml_rev);
SWEEP_BY_ORDER_TABLE(tti2nd_adjoint_div_prepare_fn, tti2nd_adjoint_div_prepare_by_order, tti2nd_adjoint_div_prepare);
SWEEP_BY_ORDER_TABLE(tti2nd_adjoint_strain_prepare_fn, tti2nd_adjoint_strain_prepare_by_order, tti2nd_adjoint_strain_prepare);
SWEEP_BY_ORDER_TABLE(tti2nd_adjoint_displacement_apply_fn, tti2nd_adjoint_displacement_apply_by_order, tti2nd_adjoint_displacement_apply);
SWEEP_BY_ORDER_TABLE(tti2nd_calculate_grad_fn, tti2nd_calculate_grad_by_order, tti2nd_calculate_grad);
} // namespace elastic_tti_2nd2d

namespace elastic_tti_2nd2d {
// Source-cell compensation for the rho imaging: the stored U_{t+1} contains
// the injected wavelet S_t, which is rho-independent; add back lam*S/rho.
__global__ void tti2nd_rho_grad_source_correction(
    float* __restrict__ grad_rho,
    const float* __restrict__ adj_field,   // lam_{t+1} component matching the source field
    const float* __restrict__ rho,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
)
{
    int isrc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (isrc >= nsrc || b >= solver.B) return;

    const int spatial_size = solver.nx * solver.nz;
    const int* loc = sources_loc + (b * nsrc + isrc) * 2;
    const int ix = loc[0];
    const int iz = loc[1];
    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) return;

    const int idx = b * spatial_size + iz * solver.nx + ix;
    const float s = source[((long long)b * nsrc + isrc) * solver.nt + it];
    atomicAdd(&grad_rho[idx], adj_field[idx] * s / rho[idx]);
}
} // namespace elastic_tti_2nd2d

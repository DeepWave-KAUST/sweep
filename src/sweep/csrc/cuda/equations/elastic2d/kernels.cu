// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(elastic_velocity_kernel_fn, elastic_velocity_kernel_by_order, elastic_velocity_kernel);
SWEEP_BY_ORDER_TABLE(elastic_stress_kernel_fn, elastic_stress_kernel_by_order, elastic_stress_kernel);
SWEEP_BY_ORDER_TABLE(elastic_velocity_kernel_nopml_fn, elastic_velocity_kernel_nopml_by_order, elastic_velocity_kernel_nopml);
SWEEP_BY_ORDER_TABLE(elastic_stress_kernel_nopml_fn, elastic_stress_kernel_nopml_by_order, elastic_stress_kernel_nopml);
SWEEP_BY_ORDER_TABLE(calculate_grad_elastic_bs_fn, calculate_grad_elastic_bs_by_order, calculate_grad_elastic_bs);
SWEEP_BY_ORDER_TABLE(calculate_grad_elastic_nobs_fn, calculate_grad_elastic_nobs_by_order, calculate_grad_elastic_nobs);
SWEEP_BY_ORDER_TABLE(elastic_stress_adjoint_prepare_fn, elastic_stress_adjoint_prepare_by_order, elastic_stress_adjoint_prepare);
SWEEP_BY_ORDER_TABLE(elastic_stress_adjoint_apply_fn, elastic_stress_adjoint_apply_by_order, elastic_stress_adjoint_apply);
SWEEP_BY_ORDER_TABLE(elastic_velocity_adjoint_prepare_fn, elastic_velocity_adjoint_prepare_by_order, elastic_velocity_adjoint_prepare);
SWEEP_BY_ORDER_TABLE(elastic_velocity_adjoint_apply_fn, elastic_velocity_adjoint_apply_by_order, elastic_velocity_adjoint_apply);
SWEEP_BY_ORDER_TABLE(elastic_velocity_kernel_apm_fn, elastic_velocity_kernel_apm_by_order, elastic_velocity_kernel_apm);
SWEEP_BY_ORDER_TABLE(elastic_stress_kernel_apm_fn, elastic_stress_kernel_apm_by_order, elastic_stress_kernel_apm);

// Compact copy of the physical box's restore strips (width w* per non-cut
// face) of vx/vz into the boundary-saving carriers: the box cells the NOPML
// kernel below does not compute.  Launched BEFORE the reverse update.
__global__ void elastic_capture_strips_2d(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ fvx, float* __restrict__ fvz,
    int nx, int nz, int x0, int x1, int z0, int z1,
    int wxl, int wxh, int wzl, int wzh)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    const int bw = x1 - x0;
    const int zi0 = z0 + wzl, zi1 = z1 - wzh;
    const int bh = zi1 - zi0;
    const int n_top = wzl * bw, n_bot = wzh * bw;
    const int n_left = wxl * bh, n_right = wxh * bh;
    int ix, iz;
    if (t < n_top) { iz = z0 + t / bw; ix = x0 + t % bw; }
    else if ((t -= n_top) < n_bot) { iz = zi1 + t / bw; ix = x0 + t % bw; }
    else if ((t -= n_bot) < n_left) { ix = x0 + t % wxl; iz = zi0 + t / wxl; }
    else if ((t -= n_left) < n_right) { ix = (x1 - wxh) + t % wxh; iz = zi0 + t / wxh; }
    else return;
    const int idx = b * nx * nz + iz * nx + ix;
    fvx[idx] = vx[idx];
    fvz[idx] = vz[idx];
}

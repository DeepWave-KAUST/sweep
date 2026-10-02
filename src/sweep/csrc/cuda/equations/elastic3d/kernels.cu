// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(elastic_velocity_kernel_3d_fn, elastic_velocity_kernel_3d_by_order, elastic_velocity_kernel_3d);
SWEEP_BY_ORDER_TABLE(elastic_stress_kernel_3d_fn, elastic_stress_kernel_3d_by_order, elastic_stress_kernel_3d);
SWEEP_BY_ORDER_TABLE(elastic_velocity_kernel_3d_nopml_fn, elastic_velocity_kernel_3d_nopml_by_order, elastic_velocity_kernel_3d_nopml);
SWEEP_BY_ORDER_TABLE(elastic_stress_kernel_3d_nopml_fn, elastic_stress_kernel_3d_nopml_by_order, elastic_stress_kernel_3d_nopml);
SWEEP_BY_ORDER_TABLE(elastic_velocity_adjoint_prepare_3d_fn, elastic_velocity_adjoint_prepare_3d_by_order, elastic_velocity_adjoint_prepare_3d);
SWEEP_BY_ORDER_TABLE(elastic_velocity_adjoint_apply_3d_fn, elastic_velocity_adjoint_apply_3d_by_order, elastic_velocity_adjoint_apply_3d);
SWEEP_BY_ORDER_TABLE(elastic_stress_adjoint_prepare_3d_fn, elastic_stress_adjoint_prepare_3d_by_order, elastic_stress_adjoint_prepare_3d);
SWEEP_BY_ORDER_TABLE(elastic_stress_adjoint_apply_3d_fn, elastic_stress_adjoint_apply_3d_by_order, elastic_stress_adjoint_apply_3d);
SWEEP_BY_ORDER_TABLE(calculate_grad_elastic3d_bs_fn, calculate_grad_elastic3d_bs_by_order, calculate_grad_elastic3d_bs);
SWEEP_BY_ORDER_TABLE(elastic3d_velocity_kernel_apm_fn, elastic3d_velocity_kernel_apm_by_order, elastic3d_velocity_kernel_apm);
SWEEP_BY_ORDER_TABLE(elastic3d_stress_kernel_apm_fn, elastic3d_stress_kernel_apm_by_order, elastic3d_stress_kernel_apm);

// 3-D twin of elastic_capture_strips_2d: the six restore slabs of the
// physical box (x offset innermost so a warp shares memory sectors).
__global__ void elastic_capture_strips_3d(
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    float* __restrict__ fvx, float* __restrict__ fvy, float* __restrict__ fvz,
    int nx, int ny, int nz,
    int x0, int x1, int y0, int y1, int z0, int z1,
    int wxl, int wxh, int wyl, int wyh, int wzl, int wzh)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    const int bx = x1 - x0, by = y1 - y0;
    const int zi0 = z0 + wzl, zi1 = z1 - wzh, bzi = zi1 - zi0;
    const int yi0 = y0 + wyl, yi1 = y1 - wyh, byi = yi1 - yi0;
    const int n_zl = wzl * bx * by, n_zh = wzh * bx * by;
    const int n_yl = wyl * bx * bzi, n_yh = wyh * bx * bzi;
    const int n_xl = wxl * byi * bzi, n_xh = wxh * byi * bzi;
    int ix, iy, iz;
    if (t < n_zl) { iz = z0 + t / (bx * by); t %= bx * by; iy = y0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_zl) < n_zh) { iz = zi1 + t / (bx * by); t %= bx * by; iy = y0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_zh) < n_yl) { iy = y0 + t / (bx * bzi); t %= bx * bzi; iz = zi0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_yl) < n_yh) { iy = yi1 + t / (bx * bzi); t %= bx * bzi; iz = zi0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_yh) < n_xl) { ix = x0 + t % wxl; t /= wxl; iy = yi0 + t % byi; iz = zi0 + t / byi; }
    else if ((t -= n_xl) < n_xh) { ix = (x1 - wxh) + t % wxh; t /= wxh; iy = yi0 + t % byi; iz = zi0 + t / byi; }
    else return;
    const int idx = b * nx * ny * nz + iz * nx * ny + iy * nx + ix;
    fvx[idx] = vx[idx];
    fvy[idx] = vy[idx];
    fvz[idx] = vz[idx];
}

#include "kernels.cuh"

__global__ void calculate_grad_lsrtm3d_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz) {
        return;
    }

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = stride_z * nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_tt_b = u_tt_bg + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // grad[mp] = sum_t adjoint_sc * d(sc_next)/d(mp) = sum_t u_back * dt^2 * bg_utt,
    // where saved u_tt_bg == bg_utt (= vp^2 * bg_w_sum).  (Was u_tt_bg/vp^2, dropping
    // dt^2*vp^2 -> grad ~1/(dt^2 vp^2) too small vs eager autograd.)
    grad_b[idx] += dt * dt * u_tt_b[idx] * u_backward_b[idx];
}

__global__ void calculate_grad_lsrtm3d_mp_utt(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz) {
        return;
    }

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = stride_z * nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_next_b = u_forward_next + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // u_tt = d^2(bg)/dt^2 = bg_utt (= vp^2 * bg_w_sum); grad[mp] = sum_t u_back * dt^2 * bg_utt.
    float u_tt = (u_now_b[idx] - 2.0f * u_prev_b[idx] + u_next_b[idx]) / (dt * dt);
    grad_b[idx] += dt * dt * u_tt * u_backward_b[idx];
}

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).

SWEEP_BY_ORDER_TABLE(acoustic3d_single_fn, acoustic3d_single_by_order, acoustic3d_single);
SWEEP_BY_ORDER_TABLE(acoustic3d_single_nopml_fn, acoustic3d_single_nopml_by_order, acoustic3d_single_nopml);
SWEEP_BY_ORDER_TABLE(acoustic_lsrtm3d_coupled_fn, acoustic_lsrtm3d_coupled_by_order, acoustic_lsrtm3d_coupled);
SWEEP_BY_ORDER_TABLE(acoustic_lsrtm3d_adjoint_fn, acoustic_lsrtm3d_adjoint_by_order, acoustic_lsrtm3d_adjoint);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
__global__ void compute_v2_lambda_lsrtm3d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int ny, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    int sp = nx * ny * nz; int idx = b * sp + iz * (nx * ny) + iy * nx + ix;
    float v = vp[idx];
    v2_lambda[idx] = v * v * lambda[idx];
}

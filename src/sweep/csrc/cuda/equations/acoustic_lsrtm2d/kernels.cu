#include "kernels.cuh"

__global__ void calculate_grad_lsrtm_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_tt_b = u_tt_bg + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // grad[mp] = sum_t adjoint_sc * d(sc_next)/d(mp).  Forward couples
    //   sc_next += dt^2 * mp * bg_utt   (bg_utt = vp^2 * bg_w_sum, == saved u_tt_bg),
    // so d(sc_next)/d(mp) = dt^2 * bg_utt = dt^2 * u_tt_bg.  (Was u_tt_bg/vp^2, which
    // dropped dt^2*vp^2 -> grad ~1/(dt^2 vp^2) too small vs eager autograd.)
    grad_b[idx] += dt * dt * u_tt_b[idx] * u_backward_b[idx];
}

__global__ void calculate_grad_lsrtm_mp_utt(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

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

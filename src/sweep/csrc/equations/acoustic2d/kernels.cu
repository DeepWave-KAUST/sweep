#include "kernels.cuh"

__global__ void calculate_grad(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 8*u_forward_b[idx] * u_backward_b[idx]/vp3;

}

__global__ void calculate_grad_utt(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 8*u_tt * u_backward_b[idx]/vp3;

}
#include "kernels.cuh"

__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz)
        return;

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = nx * ny * nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 32*u_forward_b[idx] * u_backward_b[idx]/vp3;

}

__global__ void calculate_grad_utt_3d(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz)
        return;

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = nx * ny * nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx])/ (dt*dt); //

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 32*u_tt * u_backward_b[idx]/vp3;

}

__global__ void accumulate_rtm_image_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B, int nx, int ny, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz)
        return;

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = nx * ny * nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_forward_b = u_forward + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float* image_b = image + b * spatial_size;
    float* src_b = source_illumination + b * spatial_size;
    float* rec_b = receiver_illumination + b * spatial_size;

    float uf = u_forward_b[idx];
    float ub = u_backward_b[idx];

    image_b[idx] += uf * ub;
    src_b[idx] += uf * uf;
    rec_b[idx] += ub * ub;
}

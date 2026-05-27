#include "kernels.cuh"

__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx); stored as vp^2 * Lap(u)
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

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    // Discrete adjoint of u_next = 2 u_now - u_prev + dt^2 vp^2 Lap(u_now):
    //   dL/dvp = (2 dt^2 / vp) * sum_t u_forward_saved * u_adj
    grad_b[idx] += 2.f * dt * dt * u_forward_b[idx] * u_backward_b[idx] / vp_b[idx];

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

    // After the forward.swap() in backward_bs the buffer roles are rotated so
    // that this expression evaluates to the centered second time derivative
    // (u(t-1) - 2 u(t) + u(t+1)) / dt^2 at the physical middle time.
    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);

    // u_tt = vp^2 * Lap(u) in the interior, same form as calculate_grad_3d:
    //   dL/dvp += (2 dt^2 / vp) * u_tt * u_adj
    grad_b[idx] += 2.f * dt * dt * u_tt * u_backward_b[idx] / vp_b[idx];

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

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= solver.B || s >= nsrc) {
        return;
    }

    int base = (b * nsrc + s) * 3;
    int ix = sources_loc[base + 0];
    int iy = sources_loc[base + 1];
    int iz = sources_loc[base + 2];

    if (ix < 0 || ix >= solver.nx ||
        iy < 0 || iy >= solver.ny ||
        iz < 0 || iz >= solver.nz) {
        return;
    }

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int u_idx = b * spatial_size + iz * (solver.nx * solver.ny) + iy * solver.nx + ix;
    int grad_idx = (b * nsrc + s) * solver.nt + it;

    grad_source[grad_idx] = u_backward[u_idx];
}

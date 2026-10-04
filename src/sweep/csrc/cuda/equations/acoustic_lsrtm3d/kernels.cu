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

// ---- RWI tomographic vp gradient (Wu & Alkhalifah 2015); 3-D twins of the
// acoustic_lsrtm2d kernels -- see there for the derivation of each term. ----

// Background adjoint lambda_bg = mu: its own propagation plus the transpose of the
// scattered-source coupling, v2 = vp^2 * g with g = lambda_bg + mp * lambda_sc;
// g is kept too, for the adjoint step's CPML band (its pml_field).
__global__ void compute_v2_lambda_bg_lsrtm3d(
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    float* __restrict__ g,
    int nx, int ny, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    int sp = nx * ny * nz; int idx = b * sp + iz * (nx * ny) + iy * nx + ix;
    float v2 = vp[idx] * vp[idx];
    float gi = lambda_bg[idx] + mp[idx] * lambda_sc[idx];
    g[idx] = gi;
    v2_lambda[idx] = v2 * gi;
}

// Full mode, from the stored second time derivatives:
//   grad_v += (2 dt^2 / v) [ sc_utt*lam_sc (II) + bg_utt*lam_bg (IV) + mp*bg_utt*lam_sc (III) ]
__global__ void calculate_grad_lsrtm3d_vp_utt(
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    int B, int nx, int ny, int nz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    float c   = 2.0f * dt * dt / vp[o];
    float bgu = bg_utt[o];
    float ls  = lam_sc[o];
    float ii_iv = c * (sc_utt[o] * ls + bgu * lam_bg[o]);
    float iii   = c * (mp[o] * bgu * ls);
    grad_vp[o] += ii_iv + iii;
}

// Boundary-saving: the scattered reverse recursion's coupling source
// mp * vp^2 * Lap(bg) = mp * (source-free bg second difference).  Interior only.
__global__ void add_lsrtm3d_scattered_coupling(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int B, int nx, int ny, int nz, int halo
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B) return;
    if (ix < halo || ix >= nx - halo || iy < halo || iy >= ny - halo ||
        iz < halo || iz >= nz - halo) return;
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    sc_next[o] += mp[o] * (bg_next[o] - 2.0f * bg_now[o] + bg_prev[o]);
}

// Boundary-saving vp gradient from the reconstructed second differences:
// bg 2nd diff -> IV, sc 2nd diff -> II + III.
__global__ void calculate_grad_lsrtm3d_vp_2diff(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    int B, int nx, int ny, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    float inv = 2.0f / vp[o];
    float bg2 = bg_prev[o] - 2.0f * bg_now[o] + bg_next[o];
    float sc2 = sc_prev[o] - 2.0f * sc_now[o] + sc_next[o];
    float ls  = lam_sc[o];
    float total = inv * (bg2 * lam_bg[o] + sc2 * ls);
    grad_vp[o] += total;
}

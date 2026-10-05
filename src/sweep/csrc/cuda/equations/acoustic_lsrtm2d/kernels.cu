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

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).

SWEEP_BY_ORDER_TABLE(acoustic2d_single_fn, acoustic2d_single_by_order, acoustic2d_single);
SWEEP_BY_ORDER_TABLE(acoustic2d_single_nopml_fn, acoustic2d_single_nopml_by_order, acoustic2d_single_nopml);
SWEEP_BY_ORDER_TABLE(acoustic_lsrtm2nd_fn, acoustic_lsrtm2nd_by_order, acoustic_lsrtm2nd);
SWEEP_BY_ORDER_TABLE(acoustic_lsrtm2nd_adjoint_fn, acoustic_lsrtm2nd_adjoint_by_order, acoustic_lsrtm2nd_adjoint);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
__global__ void compute_v2_lambda_lsrtm2d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz || b >= B) return;
    int sp = nx * nz; int idx = b * sp + iz * nx + ix;
    float v = vp[idx];
    v2_lambda[idx] = v * v * lambda[idx];
}

// RWI (Wu & Alkhalifah 2015): the background adjoint lambda_bg = mu.  Its own
// propagation plus the transpose of the scattered-source coupling arrive in one
// combined right-hand side, so the shared lsrtm adjoint step can inject both:
//   v2 = vp^2 * g,  g = lambda_bg + mp * lambda_sc.
// g is kept too: the adjoint step's CPML band does not read v2 and differentiates
// g instead (its pml_field).
__global__ void compute_v2_lambda_bg_lsrtm2d(
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    float* __restrict__ g,
    int nx, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz || b >= B) return;
    int sp = nx * nz; int idx = b * sp + iz * nx + ix;
    float v2 = vp[idx] * vp[idx];
    float gi = lambda_bg[idx] + mp[idx] * lambda_sc[idx];
    g[idx] = gi;
    v2_lambda[idx] = v2 * gi;
}

// RWI tomographic vp gradient from the stored second time derivatives
// (the full mode, where bg_utt and sc_utt are stored for every step):
//   grad_v += (2 dt^2 / v) [ sc_utt*lam_sc + bg_utt*lam_bg + mp*bg_utt*lam_sc ]
//               \_____ II _____/  \___ IV ___/  \______ III ______/
// II  = <lam, B'(v) q>        (scattered field self-propagation)
// IV  = <mu,  B'(v) p>        (background field via the second adjoint)
// III = <lam, B'(v) p .* w>   (the singular image-point term)
// grad_iii != nullptr keeps III apart so the caller can apply the paper's beta
// weight (eq. 18-20); otherwise all three are summed into grad_vp.
__global__ void calculate_grad_lsrtm_vp_utt(
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int nx, int nz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz) return;
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    float c   = 2.0f * dt * dt / vp[o];
    float bgu = bg_utt[o];
    float ls  = lam_sc[o];
    float ii_iv = c * (sc_utt[o] * ls + bgu * lam_bg[o]);
    float iii   = c * (mp[o] * bgu * ls);
    if (grad_iii != nullptr) {
        grad_vp[o]  += ii_iv;
        grad_iii[o] += iii;
    } else {
        grad_vp[o]  += ii_iv + iii;
    }
}

// Boundary-saving mode drives the scattered field's reverse recursion with the
// coupling source mp * vp^2 * Lap(bg), reusing the background reconstruction's
// own second difference (bg_next - 2 bg_now + bg_prev = dt^2 vp^2 Lap(bg)) so no
// Laplacian is re-evaluated.  Interior only; the outer ring is overwritten by
// the scattered boundary restore right afterwards.
__global__ void add_lsrtm_scattered_coupling_2d(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int nx, int nz, int halo
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix < halo || ix >= nx - halo || iz < halo || iz >= nz - halo) return;
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    sc_next[o] += mp[o] * (bg_next[o] - 2.0f * bg_now[o] + bg_prev[o]);
}

// Boundary-saving vp gradient, from the reconstructed second differences:
//   bg 2nd diff = dt^2 vp^2 Lap(bg) = dt^2 * bg_utt                 -> IV
//   sc 2nd diff = dt^2 (sc_utt + mp*bg_utt)                         -> II + III
// so the total matches the full mode, and III is recovered separately as
// (2/v) * mp * (bg 2nd diff) * lam_sc for the same beta split.
__global__ void calculate_grad_lsrtm_vp_2diff_2d(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int nx, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz) return;
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    float inv = 2.0f / vp[o];
    float bg2 = bg_prev[o] - 2.0f * bg_now[o] + bg_next[o];
    float sc2 = sc_prev[o] - 2.0f * sc_now[o] + sc_next[o];
    float ls  = lam_sc[o];
    float total = inv * (bg2 * lam_bg[o] + sc2 * ls);
    if (grad_iii != nullptr) {
        float iii = inv * (mp[o] * bg2 * ls);
        grad_vp[o]  += total - iii;
        grad_iii[o] += iii;
    } else {
        grad_vp[o]  += total;
    }
}

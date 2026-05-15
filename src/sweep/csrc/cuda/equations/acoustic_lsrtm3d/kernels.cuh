#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

#define ACOUSTIC_LSRTM3D_SINGLE(order, grid, block, ...)                             \
    do {                                                                             \
        if      ((order) == 2) acoustic3d_single<2><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 4) acoustic3d_single<4><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 6) acoustic3d_single<6><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 8) acoustic3d_single<8><<<grid, block>>>(__VA_ARGS__);  \
        else                   acoustic3d_single<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define ACOUSTIC_LSRTM3D_SINGLE_NOPML(order, grid, block, ...)                             \
    do {                                                                                   \
        if      ((order) == 2) acoustic3d_single_nopml<2><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 4) acoustic3d_single_nopml<4><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 6) acoustic3d_single_nopml<6><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 8) acoustic3d_single_nopml<8><<<grid, block>>>(__VA_ARGS__);  \
        else                   acoustic3d_single_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define ACOUSTIC_LSRTM3D_COUPLED(order, grid, block, ...)                              \
    do {                                                                                \
        if      ((order) == 2) acoustic_lsrtm3d_coupled<2><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 4) acoustic_lsrtm3d_coupled<4><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 6) acoustic_lsrtm3d_coupled<6><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 8) acoustic_lsrtm3d_coupled<8><<<grid, block>>>(__VA_ARGS__);  \
        else                   acoustic_lsrtm3d_coupled<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template <int Order>
__device__ inline float acoustic_cpml_update_3d(
    AcousticWavefieldPointer f,
    int ix,
    int iy,
    int iz,
    int idx,
    const AcousticCPMLPointer& cpml,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    bool use_pml,
    const SolverContext& solver,
    int halo
) {
    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);
    if (!use_pml) {
        return lap_x + lap_y + lap_z;
    }

    // Interior fast-path: ax/bx/dbxdx vanish, so w_sum reduces to the sum of
    // laplacians and the aux fields stay zero. Skip the 14+ extra loads/stores.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        return lap_x + lap_y + lap_z;
    }

    float w_sum = 0.0f;

    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float ax_ = cpml.ax[ix];
    float bx_ = cpml.bx[ix];
    float dbxdx_ = cpml.dbxdx[ix];
    float ay_ = cpml.ay[iy];
    float by_ = cpml.by[iy];
    float dbydy_ = cpml.dbydy[iy];
    float az_ = cpml.az[iz];
    float bz_ = cpml.bz[iz];
    float dbzdz_ = cpml.dbzdz[iz];

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + ax_ * dpsixdx + daxdx * f.psix[idx];
    w_sum += (1.0f + bx_) * tmpx + ax_ * f.zetax[idx];
    f.psix[idx] = bx_ * dudx + ax_ * f.psix[idx];
    f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpy = ((1.0f + by_) * lap_y + dbydy_ * dudy) + ay_ * dpsiydy + daydy * f.psiy[idx];
    w_sum += (1.0f + by_) * tmpy + ay_ * f.zetay[idx];
    f.psiy[idx] = by_ * dudy + ay_ * f.psiy[idx];
    f.zetay[idx] = by_ * tmpy + ay_ * f.zetay[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + az_ * dpsizdz + dazdz * f.psiz[idx];
    w_sum += (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];
    f.psiz[idx] = bz_ * dudz + az_ * f.psiz[idx];
    f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];

    return w_sum;
}

template <int Order>
__global__ void acoustic3d_single(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo || iz < halo || iz >= solver.nz - halo) {
        return;
    }

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float utt = (vp_b[idx] * vp_b[idx]) * acoustic_cpml_update_3d<Order>(
        f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, true, solver, halo
    );
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * utt;
    if (save_all_wavefields && u_this_b != nullptr) {
        u_this_b[idx] = utt;
    }
}

template <int Order>
__global__ void acoustic3d_single_nopml(
    AcousticWavefieldPointer wf,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    int m = (Order == -1) ? solver.M : (Order / 2);
    int halo = solver.abcn > 0 ? solver.abcn + 2 * m + 1 : 2 * m;
    int top_halo = solver.free_surface ? 2 * m : halo;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo || iz < top_halo || iz >= solver.nz - halo) {
        return;
    }

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float utt = (vp_b[idx] * vp_b[idx]) * acoustic_cpml_update_3d<Order>(
        f, ix, iy, iz, idx, AcousticCPMLPointer{}, lap_ctx, GradParam{}, GradParam{}, GradParam{}, GradParam{}, false, solver, halo
    );
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * utt;
    if (u_this_b != nullptr) {
        u_this_b[idx] = utt;
    }
}

template <int Order>
__global__ void acoustic_lsrtm3d_coupled(
    AcousticWavefieldPointer bg,
    AcousticWavefieldPointer sc,
    bool save_all_wavefields,
    float* __restrict__ bg_utt,
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo || iz < halo || iz >= solver.nz - halo) {
        return;
    }

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto bg_f = bg.offset(b, spatial_size);
    auto sc_f = sc.offset(b, spatial_size);
    float* bg_utt_b = bg_utt ? bg_utt + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;
    const float* mp_b = mp + b * spatial_size;

    float v2 = vp_b[idx] * vp_b[idx];
    float bg_utt_val = v2 * acoustic_cpml_update_3d<Order>(
        bg_f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, true, solver, halo
    );
    float sc_utt_val = v2 * acoustic_cpml_update_3d<Order>(
        sc_f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, true, solver, halo
    );

    bg_f.u_next[idx] = 2.0f * bg_f.u_now[idx] - bg_f.u_prev[idx] + solver.dt * solver.dt * bg_utt_val;
    sc_f.u_next[idx] = 2.0f * sc_f.u_now[idx] - sc_f.u_prev[idx] + solver.dt * solver.dt * (sc_utt_val + mp_b[idx] * bg_utt_val);

    if (save_all_wavefields && bg_utt_b != nullptr) {
        bg_utt_b[idx] = bg_utt_val;
    }
}

__global__ void calculate_grad_lsrtm3d_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz
);

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
);

__global__ void accumulate_rtm_image_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B,
    int nx,
    int ny,
    int nz
);

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
);

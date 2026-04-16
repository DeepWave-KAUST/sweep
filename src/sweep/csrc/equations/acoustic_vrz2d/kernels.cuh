#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

template<int Order>
__global__ void acoustic_vrz2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
);

template<int Order>
__global__ void acoustic_vrz2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void calculate_grad_vrz2d(
    const float* __restrict__ u_now,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    SolverContext solver
);

template<int Order>
__global__ void calculate_grad_vrz2d_nopml(
    const float* __restrict__ u_now,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
);

#define ACOUSTIC_VRZ2D(order, grid, block, ...)                                      \
    do {                                                                             \
        if      ((order) == 2) acoustic_vrz2nd<2><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 4) acoustic_vrz2nd<4><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 6) acoustic_vrz2nd<6><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 8) acoustic_vrz2nd<8><<<grid, block>>>(__VA_ARGS__);     \
        else                   acoustic_vrz2nd<-1><<<grid, block>>>(__VA_ARGS__);    \
    } while (0)

#define ACOUSTIC_VRZ2D_NOPML(order, grid, block, ...)                                        \
    do {                                                                                      \
        if      ((order) == 2) acoustic_vrz2nd_nopml<2><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 4) acoustic_vrz2nd_nopml<4><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 6) acoustic_vrz2nd_nopml<6><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 8) acoustic_vrz2nd_nopml<8><<<grid, block>>>(__VA_ARGS__);       \
        else                   acoustic_vrz2nd_nopml<-1><<<grid, block>>>(__VA_ARGS__);      \
    } while (0)

#define CALCULATE_GRAD_VRZ2D(order, grid, block, ...)                                         \
    do {                                                                                       \
        if      ((order) == 2) calculate_grad_vrz2d<2><<<grid, block>>>(__VA_ARGS__);         \
        else if ((order) == 4) calculate_grad_vrz2d<4><<<grid, block>>>(__VA_ARGS__);         \
        else if ((order) == 6) calculate_grad_vrz2d<6><<<grid, block>>>(__VA_ARGS__);         \
        else if ((order) == 8) calculate_grad_vrz2d<8><<<grid, block>>>(__VA_ARGS__);         \
        else                   calculate_grad_vrz2d<-1><<<grid, block>>>(__VA_ARGS__);        \
    } while (0)

#define CALCULATE_GRAD_VRZ2D_NOPML(order, grid, block, ...)                                   \
    do {                                                                                       \
        if      ((order) == 2) calculate_grad_vrz2d_nopml<2><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 4) calculate_grad_vrz2d_nopml<4><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 6) calculate_grad_vrz2d_nopml<6><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 8) calculate_grad_vrz2d_nopml<8><<<grid, block>>>(__VA_ARGS__);   \
        else                   calculate_grad_vrz2d_nopml<-1><<<grid, block>>>(__VA_ARGS__);  \
    } while (0)

template<int Order>
__device__ __forceinline__
float vrz_grad_coeff(int k, const float* coeff)
{
    if constexpr (Order == 2) {
        return (k == 1) ? 0.5f : 0.0f;
    } else if constexpr (Order == 4) {
        return (k == 1) ? (8.f / 12.f) : (k == 2 ? (-1.f / 12.f) : 0.0f);
    } else if constexpr (Order == 6) {
        return (k == 1) ? (3.f / 4.f) : (k == 2 ? (-3.f / 20.f) : (k == 3 ? (1.f / 60.f) : 0.0f));
    } else if constexpr (Order == 8) {
        return (k == 1) ? (4.f / 5.f)
             : (k == 2) ? (-1.f / 5.f)
             : (k == 3) ? (4.f / 105.f)
             : (k == 4) ? (-1.f / 280.f)
             : 0.0f;
    } else {
        return coeff[k];
    }
}

template<int Order>
__device__ __forceinline__
int vrz_half_order(const SolverContext& solver)
{
    if constexpr (Order == -1)
        return solver.M;
    else
        return Order / 2;
}

template<int Order>
__device__ __forceinline__
void vrz_forward_terms_at(
    const float* __restrict__ u,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    int ix,
    int iz,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    float& px,
    float& pz,
    float& w_sum,
    float& v,
    float& dvpdx,
    float& dvpdz,
    float& dzdx,
    float& dzdz
)
{
    int idx = iz * lap_ctx.nx + ix;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float lap_x = laplace<2, Order, X>(u, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(u, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(u, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(u, ix, 0, iz, grad_ctx);

    float dpsixdx = gradient<2, Order, X>(psix, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(psiz, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    dvpdx = gradient<2, Order, X>(vp, ix, 0, iz, grad_ctx);
    dvpdz = gradient<2, Order, Z>(vp, ix, 0, iz, grad_ctx);
    dzdx = gradient<2, Order, X>(z, ix, 0, iz, grad_ctx);
    dzdz = gradient<2, Order, Z>(z, ix, 0, iz, grad_ctx);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + (daxdx * psix[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dudx + ax_ * psix[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + (dazdz * psiz[idx] + az_ * dpsizdz);
    float psizn = bz_ * dudz + az_ * psiz[idx];

    px = dudx + psixn;
    pz = dudz + psizn;
    w_sum = (1.0f + bx_) * tmpx + ax_ * zetax[idx]
          + (1.0f + bz_) * tmpz + az_ * zetaz[idx];

    v = vp[idx];
}

template<int Order>
__device__ __forceinline__
float vrz_qx_v_at(
    const float* __restrict__ u,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    int ix,
    int iz,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    const SolverContext& solver
)
{
    float px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz;
    vrz_forward_terms_at<Order>(
        u, psix, psiz, zetax, zetaz, vp, z, ix, iz,
        cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx,
        px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz
    );
    int idx = iz * solver.nx + ix;
    return solver.dt * solver.dt * u_adj[idx] * v * px;
}

template<int Order>
__device__ __forceinline__
float vrz_qz_v_at(
    const float* __restrict__ u,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    int ix,
    int iz,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    const SolverContext& solver
)
{
    float px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz;
    vrz_forward_terms_at<Order>(
        u, psix, psiz, zetax, zetaz, vp, z, ix, iz,
        cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx,
        px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz
    );
    int idx = iz * solver.nx + ix;
    return solver.dt * solver.dt * u_adj[idx] * v * pz;
}

template<int Order>
__device__ __forceinline__
float vrz_qx_z_at(
    const float* __restrict__ u,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    int ix,
    int iz,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    const SolverContext& solver
)
{
    float px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz;
    vrz_forward_terms_at<Order>(
        u, psix, psiz, zetax, zetaz, vp, z, ix, iz,
        cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx,
        px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz
    );
    int idx = iz * solver.nx + ix;
    float inv_z = 1.0f / z[idx];
    return solver.dt * solver.dt * u_adj[idx] * (v * v) * inv_z * px;
}

template<int Order>
__device__ __forceinline__
float vrz_qz_z_at(
    const float* __restrict__ u,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    int ix,
    int iz,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    const SolverContext& solver
)
{
    float px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz;
    vrz_forward_terms_at<Order>(
        u, psix, psiz, zetax, zetaz, vp, z, ix, iz,
        cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx,
        px, pz, w_sum, v, dvpdx, dvpdz, dzdx, dzdz
    );
    int idx = iz * solver.nx + ix;
    float inv_z = 1.0f / z[idx];
    return solver.dt * solver.dt * u_adj[idx] * (v * v) * inv_z * pz;
}

template<int Order>
__global__ void acoustic_vrz2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dpsixdx = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float dzdx = gradient<2, Order, X>(z_b, ix, 0, iz, grad_ctx);
    float dzdz = gradient<2, Order, Z>(z_b, ix, 0, iz, grad_ctx);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + (daxdx * f.psix[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dudx + ax_ * f.psix[idx];
    float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + (dazdz * f.psiz[idx] + az_ * dpsizdz);
    float psizn = bz_ * dudz + az_ * f.psiz[idx];
    float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];

    float w_sum = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx]
                + (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];

    float v = vp_b[idx];
    float imp = z_b[idx];
    float inv_imp = 1.0f / imp;
    float vx = v * dvpdx - (v * v) * dzdx * inv_imp;
    float vz = v * dvpdz - (v * v) * dzdz * inv_imp;
    float grad_term = vx * (dudx + psixn) + vz * (dudz + psizn);

    f.psix[idx] = psixn;
    f.psiz[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetaz[idx] = zetazn;

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        solver.dt * solver.dt * ((v * v) * w_sum + grad_term);

    if (save_all_wavefields && u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum + grad_term;
}

template<int Order>
__global__ void acoustic_vrz2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) M = solver.M;
    else M = Order / 2;

    // In boundary-saving mode we store the outer band with offset=-M.
    // That gives us a valid stencil starting one cell inside the physical domain,
    // so the nopml replay halo should match elastic's outer-band reconstruction.
    int halo = solver.abcn > 0 ? solver.abcn + M + 1 : M + 1;
    int top_halo = solver.free_surface ? M + 1 : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);
    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float dzdx = gradient<2, Order, X>(z_b, ix, 0, iz, grad_ctx);
    float dzdz = gradient<2, Order, Z>(z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z = 1.0f / z_b[idx];
    float grad_term = (v * dvpdx - v * v * dzdx * inv_z) * dudx
                    + (v * dvpdz - v * v * dzdz * inv_z) * dudz;

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        solver.dt * solver.dt * ((v * v) * (lap_x + lap_z) + grad_term);
}

template<int Order>
__global__ void calculate_grad_vrz2d(
    const float* __restrict__ u_now,
    const float* __restrict__ psix,
    const float* __restrict__ psiz,
    const float* __restrict__ zetax,
    const float* __restrict__ zetaz,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    AcousticCPMLPointer cpml,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int fd_halo = is_runtime ? solver.M : M_static;
    // This kernel evaluates divergence terms using neighbors at ix/iz +/- k,
    // and each neighbor in turn applies another Order/2 finite-difference stencil.
    // We therefore need a 2*M safety halo to avoid out-of-bounds accesses.
    int halo = 2 * fd_halo;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int shift = b * spatial_size;

    const float* u_b = u_now + shift;
    const float* psix_b = psix + shift;
    const float* psiz_b = psiz + shift;
    const float* zetax_b = zetax + shift;
    const float* zetaz_b = zetaz + shift;
    const float* adj_b = u_adj + shift;
    const float* vp_b = vp + shift;
    const float* z_b = z + shift;
    float* gvp_b = grad_vp + shift;
    float* gz_b = grad_z + shift;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float lapx = laplace<2, Order, X>(u_b, ix, 0, iz, lap_ctx);
    float lapz = laplace<2, Order, Z>(u_b, ix, 0, iz, lap_ctx);
    float dudx = gradient<2, Order, X>(u_b, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(u_b, ix, 0, iz, grad_ctx);

    float dpsixdx = gradient<2, Order, X>(psix_b, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(psiz_b, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float dzdx = gradient<2, Order, X>(z_b, ix, 0, iz, grad_ctx);
    float dzdz = gradient<2, Order, Z>(z_b, ix, 0, iz, grad_ctx);

    float tmpx = ((1.0f + bx_) * lapx + dbxdx_ * dudx) + (daxdx * psix_b[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dudx + ax_ * psix_b[idx];

    float tmpz = ((1.0f + bz_) * lapz + dbzdz_ * dudz) + (dazdz * psiz_b[idx] + az_ * dpsizdz);
    float psizn = bz_ * dudz + az_ * psiz_b[idx];

    float px = dudx + psixn;
    float pz = dudz + psizn;
    float w_sum = (1.0f + bx_) * tmpx + ax_ * zetax_b[idx]
                + (1.0f + bz_) * tmpz + az_ * zetaz_b[idx];

    float v = vp_b[idx];
    float lambda = adj_b[idx] * solver.dt * solver.dt;
    float inv_z = 1.0f / z_b[idx];
    float inv_z2 = inv_z * inv_z;

    float div_qv = 0.0f;
    float div_qz = 0.0f;
    int M = vrz_half_order<Order>(solver);
    for (int k = 1; k <= M; ++k) {
        float ck = vrz_grad_coeff<Order>(k, grad_ctx.coeff);

        float qxv_p = vrz_qx_v_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix + k, iz, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qxv_m = vrz_qx_v_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix - k, iz, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qzv_p = vrz_qz_v_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix, iz + k, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qzv_m = vrz_qz_v_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix, iz - k, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        div_qv += ck * (qxv_p - qxv_m) / grad_ctx.dx + ck * (qzv_p - qzv_m) / grad_ctx.dz;

        float qxz_p = vrz_qx_z_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix + k, iz, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qxz_m = vrz_qx_z_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix - k, iz, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qzz_p = vrz_qz_z_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix, iz + k, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        float qzz_m = vrz_qz_z_at<Order>(
            u_b, psix_b, psiz_b, zetax_b, zetaz_b, adj_b, vp_b, z_b,
            ix, iz - k, cpml, grad_ctx, grad_ctx_x, grad_ctx_z, lap_ctx, solver
        );
        div_qz += ck * (qxz_p - qxz_m) / grad_ctx.dx + ck * (qzz_p - qzz_m) / grad_ctx.dz;
    }

    gvp_b[idx] += lambda * (
        2.0f * v * w_sum +
        dvpdx * px +
        dvpdz * pz -
        2.0f * v * inv_z * (dzdx * px + dzdz * pz)
    ) - div_qv;

    gz_b[idx] += lambda * (v * v) * inv_z2 * (dzdx * px + dzdz * pz) + div_qz;
}

template<int Order>
__global__ void calculate_grad_vrz2d_nopml(
    const float* __restrict__ u_now,
    const float* __restrict__ u_adj,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int fd_halo = is_runtime ? solver.M : M_static;
    // BS now stores an outer band covering [phys - 2*M, phys]. That makes the
    // nested div(q) stencil valid all the way to the first physical x/bottom
    // cell, so left/right/bottom only need one additional M relative to the
    // physical-domain boundary. A free-surface top still has no exterior padding
    // above z=0, so it remains at the more conservative 2*M crop.
    int halo = solver.abcn > 0 ? solver.abcn + fd_halo : 2 * fd_halo;
    int top_halo = solver.free_surface ? 2 * fd_halo : halo;

    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int shift = b * spatial_size;

    const float* u_b = u_now + shift;
    const float* adj_b = u_adj + shift;
    const float* vp_b = vp + shift;
    const float* z_b = z + shift;
    float* gvp_b = grad_vp + shift;
    float* gz_b = grad_z + shift;

    float lapx = laplace<2, Order, X>(u_b, ix, 0, iz, lap_ctx);
    float lapz = laplace<2, Order, Z>(u_b, ix, 0, iz, lap_ctx);
    float dudx = gradient<2, Order, X>(u_b, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(u_b, ix, 0, iz, grad_ctx);
    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float dzdx = gradient<2, Order, X>(z_b, ix, 0, iz, grad_ctx);
    float dzdz = gradient<2, Order, Z>(z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z = 1.0f / z_b[idx];
    float inv_z2 = inv_z * inv_z;
    float px = dudx;
    float pz = dudz;
    float w_sum = lapx + lapz;
    float lambda = adj_b[idx] * solver.dt * solver.dt;

    float div_qv = 0.0f;
    float div_qz = 0.0f;
    int M = vrz_half_order<Order>(solver);
    for (int k = 1; k <= M; ++k) {
        float ck = vrz_grad_coeff<Order>(k, grad_ctx.coeff);
        int idx_xp = idx + k;
        int idx_xm = idx - k;
        int idx_zp = idx + k * solver.nx;
        int idx_zm = idx - k * solver.nx;

        float qxv_p = solver.dt * solver.dt * adj_b[idx_xp] * vp_b[idx_xp] *
                      gradient<2, Order, X>(u_b, ix + k, 0, iz, grad_ctx);
        float qxv_m = solver.dt * solver.dt * adj_b[idx_xm] * vp_b[idx_xm] *
                      gradient<2, Order, X>(u_b, ix - k, 0, iz, grad_ctx);
        float qzv_p = solver.dt * solver.dt * adj_b[idx_zp] * vp_b[idx_zp] *
                      gradient<2, Order, Z>(u_b, ix, 0, iz + k, grad_ctx);
        float qzv_m = solver.dt * solver.dt * adj_b[idx_zm] * vp_b[idx_zm] *
                      gradient<2, Order, Z>(u_b, ix, 0, iz - k, grad_ctx);
        div_qv += ck * (qxv_p - qxv_m) / grad_ctx.dx + ck * (qzv_p - qzv_m) / grad_ctx.dz;

        float inv_z_xp = 1.0f / z_b[idx_xp];
        float inv_z_xm = 1.0f / z_b[idx_xm];
        float inv_z_zp = 1.0f / z_b[idx_zp];
        float inv_z_zm = 1.0f / z_b[idx_zm];

        float qxz_p = solver.dt * solver.dt * adj_b[idx_xp] * vp_b[idx_xp] * vp_b[idx_xp] *
                      inv_z_xp * gradient<2, Order, X>(u_b, ix + k, 0, iz, grad_ctx);
        float qxz_m = solver.dt * solver.dt * adj_b[idx_xm] * vp_b[idx_xm] * vp_b[idx_xm] *
                      inv_z_xm * gradient<2, Order, X>(u_b, ix - k, 0, iz, grad_ctx);
        float qzz_p = solver.dt * solver.dt * adj_b[idx_zp] * vp_b[idx_zp] * vp_b[idx_zp] *
                      inv_z_zp * gradient<2, Order, Z>(u_b, ix, 0, iz + k, grad_ctx);
        float qzz_m = solver.dt * solver.dt * adj_b[idx_zm] * vp_b[idx_zm] * vp_b[idx_zm] *
                      inv_z_zm * gradient<2, Order, Z>(u_b, ix, 0, iz - k, grad_ctx);
        div_qz += ck * (qxz_p - qxz_m) / grad_ctx.dx + ck * (qzz_p - qzz_m) / grad_ctx.dz;
    }

    gvp_b[idx] += lambda * (
        2.0f * v * w_sum +
        dvpdx * px +
        dvpdz * pz -
        2.0f * v * inv_z * (dzdx * px + dzdz * pz)
    ) - div_qv;

    gz_b[idx] += lambda * (v * v) * inv_z2 * (dzdx * px + dzdz * pz) + div_qz;
}

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../operators/staggered.cuh"

// Displacement-based (second-order in time) 2-D elastic TTI of Oh et al.
// (2020, GJI 223, eqs 11-12).  Spatial terms are nested staggered
// forward/backward first-derivative pairs, so the stiffness operator is the
// exact negative-transpose sandwich K = -D^T C D; the leapfrog update is
//   U_{t+1} = 2 U_t - U_{t-1} + (dt^2/rho) * div(sigma(grad U_t)) + S_t.
// Each first derivative carries its own CPML recursive-convolution memory
// (half-node profiles on the inner/forward layer, integer-node on the
// outer/backward layer).  The forward step is split into a stress kernel
// (writes the three stresses to scratch fields) and a displacement kernel
// (their divergence -> u_next buffer); the host rotates the (now, pre, next)
// triple buffer.  The adjoint runs the same structure transposed
// (prepare/apply split, forward<->backward stencils swapped, no extra signs:
// the two spatial-transpose minus signs cancel; the stiffness imaging keeps
// a single -1).

namespace elastic_tti_2nd2d {

struct WavefieldPointer {
    float* __restrict__ ux;
    float* __restrict__ uz;
    float* __restrict__ ux_pre;
    float* __restrict__ uz_pre;
    float* __restrict__ ux_nxt;
    float* __restrict__ uz_nxt;

    float* __restrict__ m_gxux;
    float* __restrict__ m_gzux;
    float* __restrict__ m_gxuz;
    float* __restrict__ m_gzuz;
    float* __restrict__ m_sxxx;
    float* __restrict__ m_sxzz;
    float* __restrict__ m_sxzx;
    float* __restrict__ m_szzz;

    __device__ WavefieldPointer offset(int b, int spatial_size) const
    {
        WavefieldPointer out = *this;
        const int shift = b * spatial_size;
        out.ux += shift;
        out.uz += shift;
        out.ux_pre += shift;
        out.uz_pre += shift;
        out.ux_nxt += shift;
        out.uz_nxt += shift;
        out.m_gxux += shift;
        out.m_gzux += shift;
        out.m_gxuz += shift;
        out.m_gzuz += shift;
        out.m_sxxx += shift;
        out.m_sxzz += shift;
        out.m_sxzx += shift;
        out.m_szzz += shift;
        return out;
    }
};

struct StiffnessPointer {
    const float* __restrict__ rho;
    const float* __restrict__ C11;
    const float* __restrict__ C33;
    const float* __restrict__ C13;
    const float* __restrict__ C55;
    const float* __restrict__ C15;
    const float* __restrict__ C35;

    __device__ StiffnessPointer offset(int b, int spatial_size) const
    {
        StiffnessPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C33 += shift;
        out.C13 += shift;
        out.C55 += shift;
        out.C15 += shift;
        out.C35 += shift;
        return out;
    }
};

struct StiffnessGradPointer {
    float* __restrict__ rho;
    float* __restrict__ C11;
    float* __restrict__ C33;
    float* __restrict__ C13;
    float* __restrict__ C55;
    float* __restrict__ C15;
    float* __restrict__ C35;

    __device__ StiffnessGradPointer offset(int b, int spatial_size) const
    {
        StiffnessGradPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C33 += shift;
        out.C13 += shift;
        out.C55 += shift;
        out.C15 += shift;
        out.C35 += shift;
        return out;
    }
};

// Sources and receivers act on the freshly computed time level: the next
// buffer (W_{t+1}) before the host rotates the triple buffer.
__host__ inline float* field_ptr(const WavefieldPointer& wf, int field)
{
    switch (field) {
        case 0: return wf.ux_nxt;
        case 1: return wf.uz_nxt;
        default: return nullptr;
    }
}

#define TTI2ND_LAUNCH(kernel, order, grid, block, ...) \
    do { \
        if      ((order) == 2) kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// Inner layer: strains from u_now, CPML memory on the four forward
// derivatives, constitutive row -> three stress scratch fields.
template<int Order>
__global__ void tti2nd_stress_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float gxux = sgradient<2, Order, X, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);
    float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);

    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (in_pml) {
        const float azh = cpml.azh[iz];
        const float bzh = cpml.bzh[iz];
        const float axh = cpml.axh[ix];
        const float bxh = cpml.bxh[ix];

        f.m_gxux[idx] = axh * f.m_gxux[idx] + bxh * gxux;
        gxux += f.m_gxux[idx];
        f.m_gzux[idx] = azh * f.m_gzux[idx] + bzh * gzux;
        gzux += f.m_gzux[idx];
        f.m_gxuz[idx] = axh * f.m_gxuz[idx] + bxh * gxuz;
        gxuz += f.m_gxuz[idx];
        f.m_gzuz[idx] = azh * f.m_gzuz[idx] + bzh * gzuz;
        gzuz += f.m_gzuz[idx];
    }

    const float exz = gzux + gxuz;
    sxx_ws[shift + idx] = m.C11[idx] * gxux + m.C13[idx] * gzuz + m.C15[idx] * exz;
    szz_ws[shift + idx] = m.C13[idx] * gxux + m.C33[idx] * gzuz + m.C35[idx] * exz;
    sxz_ws[shift + idx] = m.C15[idx] * gxux + m.C35[idx] * gzuz + m.C55[idx] * exz;
}

// Outer layer: stress divergence with CPML on the four backward derivatives,
// leapfrog displacement update written to the u_next buffer.
template<int Order>
__global__ void tti2nd_displacement_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    float* __restrict__ u_this_t,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    const float* sxx = sxx_ws + shift;
    const float* szz = szz_ws + shift;
    const float* sxz = sxz_ws + shift;

    float dsxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(szz, ix, 0, iz, grad_ctx);

    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (in_pml) {
        const float az = cpml.az[iz];
        const float bz = cpml.bz[iz];
        const float ax = cpml.ax[ix];
        const float bx = cpml.bx[ix];

        f.m_sxxx[idx] = ax * f.m_sxxx[idx] + bx * dsxx_dx;
        dsxx_dx += f.m_sxxx[idx];
        f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
        dsxz_dz += f.m_sxzz[idx];
        f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
        dsxz_dx += f.m_sxzx[idx];
        f.m_szzz[idx] = az * f.m_szzz[idx] + bz * dszz_dz;
        dszz_dz += f.m_szzz[idx];
    }

    const float scale = solver.dt * solver.dt / m.rho[idx];
    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + scale * (dsxx_dx + dsxz_dz);
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + scale * (dsxz_dx + dszz_dz);

    if (u_this_t) {
        float* u_this_b = u_this_t + b * spatial_size;
        const int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.ux_nxt[idx];
        u_this_b[1 * comp_stride + idx] = f.uz_nxt[idx];
    }
}

// nopml stress: raw strains from u_now (no memory update) -> stress scratch.
// Shared by the bs time-reversal replay and by the checkpoint re-forward
// interior; identical to tti2nd_stress_kernel outside the PML band.
template<int Order>
__global__ void tti2nd_stress_kernel_nopml(
    WavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    if (ix < halo - M || ix >= solver.nx - halo + M || iz < halo - M || iz >= solver.nz - halo + M)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float gxux = sgradient<2, Order, X, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);
    float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);

    const float exz = gzux + gxuz;
    sxx_ws[shift + idx] = m.C11[idx] * gxux + m.C13[idx] * gzuz + m.C15[idx] * exz;
    szz_ws[shift + idx] = m.C13[idx] * gxux + m.C33[idx] * gzuz + m.C35[idx] * exz;
    sxz_ws[shift + idx] = m.C15[idx] * gxux + m.C35[idx] * gzuz + m.C55[idx] * exz;
}

// nopml time-REVERSED displacement update: with (u_now = U_t, u_pre = U_{t+1})
// writes U_{t-1} (up to the source term the caller re-adds) into u_next.
template<int Order>
__global__ void tti2nd_displacement_kernel_nopml_rev(
    WavefieldPointer wf,
    StiffnessPointer model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    const float* sxx = sxx_ws + shift;
    const float* szz = szz_ws + shift;
    const float* sxz = sxz_ws + shift;

    float dsxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(szz, ix, 0, iz, grad_ctx);

    const float scale = solver.dt * solver.dt / m.rho[idx];
    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + scale * (dsxx_dx + dsxz_dz);
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + scale * (dsxz_dx + dszz_dz);
}

// ---------------------------------------------------------------------------
// Adjoint (exact discrete transpose; unsigned convention — the two spatial
// transpose signs cancel between the sigma-side and u-side scatters)
// ---------------------------------------------------------------------------

// K1: bar of the four CPML-corrected stress-divergence terms from the adjoint
// displacement lam_{t+1} (in wf.ux/uz), transposing the outer CPML recursion.
template<int Order>
__global__ void tti2nd_adjoint_div_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ q_sxxx,
    float* __restrict__ q_sxzz,
    float* __restrict__ q_sxzx,
    float* __restrict__ q_szzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    const float scale = solver.dt * solver.dt / m.rho[idx];
    const float bar_dsxx_dx = scale * f.ux[idx];
    const float bar_dsxz_dz = scale * f.ux[idx];
    const float bar_dsxz_dx = scale * f.uz[idx];
    const float bar_dszz_dz = scale * f.uz[idx];

    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        q_sxxx[shift] = bar_dsxx_dx;
        q_sxzz[shift] = bar_dsxz_dz;
        q_sxzx[shift] = bar_dsxz_dx;
        q_szzz[shift] = bar_dszz_dz;
        return;
    }

    const float az = cpml.az[iz];
    const float bz = cpml.bz[iz];
    const float ax = cpml.ax[ix];
    const float bx = cpml.bx[ix];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    q_sxxx[shift] = bar_dsxx_dx + bx * tmp_sxxx;
    q_sxzz[shift] = bar_dsxz_dz + bz * tmp_sxzz;
    q_sxzx[shift] = bar_dsxz_dx + bx * tmp_sxzx;
    q_szzz[shift] = bar_dszz_dz + bz * tmp_szzz;

    f.m_sxxx[idx] = ax * tmp_sxxx;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_szzz[idx] = az * tmp_szzz;
}

// K2: scatter q through the transposed outer derivatives into bar-sigma,
// transpose the constitutive row into bar-strains, transpose the inner CPML
// recursion -> p workspace.
template<int Order>
__global__ void tti2nd_adjoint_strain_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    float* __restrict__ p_gxux,
    float* __restrict__ p_gzux,
    float* __restrict__ p_gxuz,
    float* __restrict__ p_gzuz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    const float* qxx = q_sxxx + shift;
    const float* qxzz = q_sxzz + shift;
    const float* qxzx = q_sxzx + shift;
    const float* qzz = q_szzz + shift;

    // Transpose of DIFF_BACKWARD is the (unsigned) DIFF_FORWARD scatter.
    float bar_sxx = sgradient<2, Order, X, DIFF_FORWARD>(qxx, ix, 0, iz, grad_ctx);
    float bar_szz = sgradient<2, Order, Z, DIFF_FORWARD>(qzz, ix, 0, iz, grad_ctx);
    float bar_sxz = sgradient<2, Order, Z, DIFF_FORWARD>(qxzz, ix, 0, iz, grad_ctx)
                  + sgradient<2, Order, X, DIFF_FORWARD>(qxzx, ix, 0, iz, grad_ctx);

    float bar_gxux = m.C11[idx] * bar_sxx + m.C13[idx] * bar_szz + m.C15[idx] * bar_sxz;
    float bar_gzuz = m.C13[idx] * bar_sxx + m.C33[idx] * bar_szz + m.C35[idx] * bar_sxz;
    const float bar_exz = m.C15[idx] * bar_sxx + m.C35[idx] * bar_szz + m.C55[idx] * bar_sxz;
    float bar_gzux = bar_exz;
    float bar_gxuz = bar_exz;

    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        p_gxux[shift + idx] = bar_gxux;
        p_gzux[shift + idx] = bar_gzux;
        p_gxuz[shift + idx] = bar_gxuz;
        p_gzuz[shift + idx] = bar_gzuz;
        return;
    }

    const float azh = cpml.azh[iz];
    const float bzh = cpml.bzh[iz];
    const float axh = cpml.axh[ix];
    const float bxh = cpml.bxh[ix];

    float tmp_gxux = f.m_gxux[idx] + bar_gxux;
    float tmp_gzux = f.m_gzux[idx] + bar_gzux;
    float tmp_gxuz = f.m_gxuz[idx] + bar_gxuz;
    float tmp_gzuz = f.m_gzuz[idx] + bar_gzuz;

    p_gxux[shift + idx] = bar_gxux + bxh * tmp_gxux;
    p_gzux[shift + idx] = bar_gzux + bzh * tmp_gzux;
    p_gxuz[shift + idx] = bar_gxuz + bxh * tmp_gxuz;
    p_gzuz[shift + idx] = bar_gzuz + bzh * tmp_gzuz;

    f.m_gxux[idx] = axh * tmp_gxux;
    f.m_gzux[idx] = azh * tmp_gzux;
    f.m_gxuz[idx] = axh * tmp_gxuz;
    f.m_gzuz[idx] = azh * tmp_gzuz;
}

// K3: scatter p through the transposed inner derivatives and advance the
// adjoint leapfrog: lam_t = 2 lam_{t+1} - lam_{t+2} + contribution, written
// to the (recycled) u_next buffer; the host then rotates the triple buffer.
template<int Order>
__global__ void tti2nd_adjoint_displacement_apply(
    WavefieldPointer wf,
    const float* __restrict__ p_gxux,
    const float* __restrict__ p_gzux,
    const float* __restrict__ p_gxuz,
    const float* __restrict__ p_gzuz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    const float* pxx = p_gxux + shift;
    const float* pzx = p_gzux + shift;
    const float* pxz = p_gxuz + shift;
    const float* pzz = p_gzuz + shift;

    // Transpose of DIFF_FORWARD is the (unsigned) DIFF_BACKWARD scatter.
    const float contrib_x =
        sgradient<2, Order, X, DIFF_BACKWARD>(pxx, ix, 0, iz, grad_ctx) +
        sgradient<2, Order, Z, DIFF_BACKWARD>(pzx, ix, 0, iz, grad_ctx);
    const float contrib_z =
        sgradient<2, Order, X, DIFF_BACKWARD>(pxz, ix, 0, iz, grad_ctx) +
        sgradient<2, Order, Z, DIFF_BACKWARD>(pzz, ix, 0, iz, grad_ctx);

    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + contrib_x;
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + contrib_z;
}

// Imaging: the six stiffness gradients correlate bar-sigma (rebuilt from the
// q workspace, one unsigned transpose layer => single -1 scale) with the raw
// strains of the forward displacement U_t; the rho gradient correlates
// lam_{t+1} with the leapfrog second time difference (source term excluded —
// compensated at the source cells by tti2nd_rho_grad_source_correction).
template<int Order>
__global__ void tti2nd_calculate_grad(
    WavefieldPointer adj,         // lam_{t+1} in .ux/.uz
    StiffnessPointer model,
    StiffnessGradPointer grad,
    const float* __restrict__ fux,       // U_t
    const float* __restrict__ fuz,
    const float* __restrict__ fux_next,  // U_{t+1} (with its source)
    const float* __restrict__ fuz_next,
    const float* __restrict__ fux_prev,  // U_{t-1} (with its source)
    const float* __restrict__ fuz_prev,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto a = adj.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    auto g = grad.offset(b, spatial_size);

    const float* ux = fux + shift;
    const float* uz = fuz + shift;
    const float* qxx = q_sxxx + shift;
    const float* qxzz = q_sxzz + shift;
    const float* qxzx = q_sxzx + shift;
    const float* qzz = q_szzz + shift;

    const float gxux = sgradient<2, Order, X, DIFF_FORWARD>(ux, ix, 0, iz, grad_ctx);
    const float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(ux, ix, 0, iz, grad_ctx);
    const float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(uz, ix, 0, iz, grad_ctx);
    const float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(uz, ix, 0, iz, grad_ctx);
    const float exz = gzux + gxuz;

    const float bar_sxx = sgradient<2, Order, X, DIFF_FORWARD>(qxx, ix, 0, iz, grad_ctx);
    const float bar_szz = sgradient<2, Order, Z, DIFF_FORWARD>(qzz, ix, 0, iz, grad_ctx);
    const float bar_sxz = sgradient<2, Order, Z, DIFF_FORWARD>(qxzz, ix, 0, iz, grad_ctx)
                        + sgradient<2, Order, X, DIFF_FORWARD>(qxzx, ix, 0, iz, grad_ctx);

    g.C11[idx] -= bar_sxx * gxux;
    g.C33[idx] -= bar_szz * gzuz;
    g.C13[idx] -= bar_sxx * gzuz + bar_szz * gxux;
    g.C55[idx] -= bar_sxz * exz;
    g.C15[idx] -= bar_sxx * exz + bar_sxz * gxux;
    g.C35[idx] -= bar_szz * exz + bar_sxz * gzuz;

    // d U_{t+1} / d rho = -(U_{t+1} - 2 U_t + U_{t-1} - S_t) / rho; the S_t
    // part is handled at the source cells by the correction kernel.
    g.rho[idx] += (
        a.ux[idx] * (2.f * ux[idx] - fux_next[shift + idx] - fux_prev[shift + idx]) +
        a.uz[idx] * (2.f * uz[idx] - fuz_next[shift + idx] - fuz_prev[shift + idx])
    ) / m.rho[idx];
}

// Source-cell compensation for the rho imaging: the stored U_{t+1} contains
// the injected wavelet S_t, which is rho-independent; add back lam*S/rho.
static __global__ void tti2nd_rho_grad_source_correction(
    float* __restrict__ grad_rho,
    const float* __restrict__ adj_field,   // lam_{t+1} component matching the source field
    const float* __restrict__ rho,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
)
{
    int isrc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (isrc >= nsrc || b >= solver.B) return;

    const int spatial_size = solver.nx * solver.nz;
    const int* loc = sources_loc + (b * nsrc + isrc) * 2;
    const int ix = loc[0];
    const int iz = loc[1];
    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) return;

    const int idx = b * spatial_size + iz * solver.nx + ix;
    const float s = source[(b * nsrc + isrc) * solver.nt + it];
    atomicAdd(&grad_rho[idx], adj_field[idx] * s / rho[idx]);
}

} // namespace elastic_tti_2nd2d

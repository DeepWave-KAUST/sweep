#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"
#include "../../operators/staggered.cuh"

namespace elastic_tti_sg2d {

struct WavefieldPointer {
    float* __restrict__ vx;
    float* __restrict__ vy;
    float* __restrict__ vz;
    float* __restrict__ sxx;
    float* __restrict__ szz;
    float* __restrict__ syz;
    float* __restrict__ sxz;
    float* __restrict__ sxy;

    float* __restrict__ m_vxx;
    float* __restrict__ m_vxz;
    float* __restrict__ m_vyx;
    float* __restrict__ m_vyz;
    float* __restrict__ m_vzx;
    float* __restrict__ m_vzz;
    float* __restrict__ m_txxx;
    float* __restrict__ m_txzz;
    float* __restrict__ m_txyx;
    float* __restrict__ m_tyzz;
    float* __restrict__ m_txzx;
    float* __restrict__ m_tzzz;

    __device__ WavefieldPointer offset(int b, int spatial_size) const
    {
        WavefieldPointer out = *this;
        const int shift = b * spatial_size;
        out.vx += shift;
        out.vy += shift;
        out.vz += shift;
        out.sxx += shift;
        out.szz += shift;
        out.syz += shift;
        out.sxz += shift;
        out.sxy += shift;
        out.m_vxx += shift;
        out.m_vxz += shift;
        out.m_vyx += shift;
        out.m_vyz += shift;
        out.m_vzx += shift;
        out.m_vzz += shift;
        out.m_txxx += shift;
        out.m_txzz += shift;
        out.m_txyx += shift;
        out.m_tyzz += shift;
        out.m_txzx += shift;
        out.m_tzzz += shift;
        return out;
    }
};

struct StiffnessPointer {
    const float* __restrict__ rho;
    const float* __restrict__ C11;
    const float* __restrict__ C13;
    const float* __restrict__ C14;
    const float* __restrict__ C15;
    const float* __restrict__ C16;
    const float* __restrict__ C33;
    const float* __restrict__ C34;
    const float* __restrict__ C35;
    const float* __restrict__ C36;
    const float* __restrict__ C44;
    const float* __restrict__ C45;
    const float* __restrict__ C46;
    const float* __restrict__ C55;
    const float* __restrict__ C56;
    const float* __restrict__ C66;

    __device__ StiffnessPointer offset(int b, int spatial_size) const
    {
        StiffnessPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C13 += shift;
        out.C14 += shift;
        out.C15 += shift;
        out.C16 += shift;
        out.C33 += shift;
        out.C34 += shift;
        out.C35 += shift;
        out.C36 += shift;
        out.C44 += shift;
        out.C45 += shift;
        out.C46 += shift;
        out.C55 += shift;
        out.C56 += shift;
        out.C66 += shift;
        return out;
    }
};

struct StiffnessGradPointer {
    float* __restrict__ rho;
    float* __restrict__ C11;
    float* __restrict__ C13;
    float* __restrict__ C14;
    float* __restrict__ C15;
    float* __restrict__ C16;
    float* __restrict__ C33;
    float* __restrict__ C34;
    float* __restrict__ C35;
    float* __restrict__ C36;
    float* __restrict__ C44;
    float* __restrict__ C45;
    float* __restrict__ C46;
    float* __restrict__ C55;
    float* __restrict__ C56;
    float* __restrict__ C66;

    __device__ StiffnessGradPointer offset(int b, int spatial_size) const
    {
        StiffnessGradPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C13 += shift;
        out.C14 += shift;
        out.C15 += shift;
        out.C16 += shift;
        out.C33 += shift;
        out.C34 += shift;
        out.C35 += shift;
        out.C36 += shift;
        out.C44 += shift;
        out.C45 += shift;
        out.C46 += shift;
        out.C55 += shift;
        out.C56 += shift;
        out.C66 += shift;
        return out;
    }
};

__host__ inline float* field_ptr(const WavefieldPointer& wf, int field)
{
    switch (field) {
        case 0: return wf.vx;
        case 1: return wf.vy;
        case 2: return wf.vz;
        case 3: return wf.sxx;
        case 4: return wf.szz;
        case 5: return wf.syz;
        case 6: return wf.sxz;
        case 7: return wf.sxy;
        default: return nullptr;
    }
}

#define LAUNCH_ELASTIC_TTI_SG_VELOCITY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_velocity_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_velocity_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_velocity_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_velocity_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_velocity_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_STRESS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_VELOCITY_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_velocity_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_velocity_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_velocity_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_velocity_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_velocity_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_STRESS_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_stress_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_stress_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_stress_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_stress_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_stress_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_stress_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_stress_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_stress_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_stress_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_stress_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_velocity_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_velocity_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_velocity_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_velocity_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_velocity_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg_velocity_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg_velocity_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg_velocity_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg_velocity_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg_velocity_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) calculate_grad_elastic_tti_sg_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_tti_sg_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_tti_sg_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_tti_sg_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_tti_sg_nobs<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void elastic_tti_sg_velocity_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxy_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxy, ix, 0, iz, grad_ctx);
    float dsyz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.syz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.szz, ix, iz, grad_ctx, solver, true);

    const float scale = solver.dt / m.rho[idx];

    // Interior fast-path: ax/bx vanish, so the six aux fields stay zero and the
    // gradients just feed through to the velocity update.
    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                        (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        f.vx[idx] += scale * (dsxx_dx + dsxz_dz);
        f.vy[idx] += scale * (dsxy_dx + dsyz_dz);
        f.vz[idx] += scale * (dsxz_dx + dszz_dz);
        return;
    }

    const float az = cpml.az[iz];
    const float bz = cpml.bz[iz];
    const float azh = cpml.azh[iz];
    const float bzh = cpml.bzh[iz];
    const float ax = cpml.ax[ix];
    const float bx = cpml.bx[ix];
    const float axh = cpml.axh[ix];
    const float bxh = cpml.bxh[ix];

    f.m_txxx[idx] = axh * f.m_txxx[idx] + bxh * dsxx_dx;
    f.m_txzz[idx] = az * f.m_txzz[idx] + bz * dsxz_dz;
    f.m_txyx[idx] = ax * f.m_txyx[idx] + bx * dsxy_dx;
    f.m_tyzz[idx] = az * f.m_tyzz[idx] + bz * dsyz_dz;
    f.m_txzx[idx] = ax * f.m_txzx[idx] + bx * dsxz_dx;
    f.m_tzzz[idx] = azh * f.m_tzzz[idx] + bzh * dszz_dz;

    dsxx_dx += f.m_txxx[idx];
    dsxz_dz += f.m_txzz[idx];
    dsxy_dx += f.m_txyx[idx];
    dsyz_dz += f.m_tyzz[idx];
    dsxz_dx += f.m_txzx[idx];
    dszz_dz += f.m_tzzz[idx];

    f.vx[idx] += scale * (dsxx_dx + dsxz_dz);
    f.vy[idx] += scale * (dsxy_dx + dsyz_dz);
    f.vz[idx] += scale * (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_tti_sg_stress_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    float* __restrict__ u_this,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int comp_stride = solver.B * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dvy_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vy, ix, 0, iz, grad_ctx);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vz, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.vx, ix, iz, grad_ctx, solver, false);
    float dvy_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.vy, ix, iz, grad_ctx, solver, false);

    // Interior fast-path. Conservative for free_surface: keep iz==halo on the
    // full PML path because both (a) the anisotropic FS gradient adjustment
    // and (b) szz/sxz/syz=0 BC fire only there.
    const int top_pml = solver.free_surface ? (halo + 1) : (solver.abcn + halo);
    const bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                        (iz < top_pml) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        const float shear_xz = dvz_dx + dvx_dz;
        f.sxx[idx] += solver.dt * (
            m.C11[idx] * dvx_dx + m.C16[idx] * dvy_dx + m.C15[idx] * shear_xz +
            m.C14[idx] * dvy_dz + m.C13[idx] * dvz_dz
        );
        f.szz[idx] += solver.dt * (
            m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * shear_xz +
            m.C34[idx] * dvy_dz + m.C33[idx] * dvz_dz
        );
        f.syz[idx] += solver.dt * (
            m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * shear_xz +
            m.C44[idx] * dvy_dz + m.C34[idx] * dvz_dz
        );
        f.sxz[idx] += solver.dt * (
            m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * shear_xz +
            m.C45[idx] * dvy_dz + m.C35[idx] * dvz_dz
        );
        f.sxy[idx] += solver.dt * (
            m.C16[idx] * dvx_dx + m.C66[idx] * dvy_dx + m.C56[idx] * shear_xz +
            m.C46[idx] * dvy_dz + m.C36[idx] * dvz_dz
        );
        if (u_this_b) {
            u_this_b[0 * comp_stride + idx] = f.vx[idx];
            u_this_b[1 * comp_stride + idx] = f.vy[idx];
            u_this_b[2 * comp_stride + idx] = f.vz[idx];
            u_this_b[3 * comp_stride + idx] = f.sxx[idx];
            u_this_b[4 * comp_stride + idx] = f.szz[idx];
            u_this_b[5 * comp_stride + idx] = f.syz[idx];
            u_this_b[6 * comp_stride + idx] = f.sxz[idx];
            u_this_b[7 * comp_stride + idx] = f.sxy[idx];
        }
        return;
    }

    const float az = cpml.az[iz];
    const float bz = cpml.bz[iz];
    const float azh = cpml.azh[iz];
    const float bzh = cpml.bzh[iz];
    const float ax = cpml.ax[ix];
    const float bx = cpml.bx[ix];
    const float axh = cpml.axh[ix];
    const float bxh = cpml.bxh[ix];

    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    f.m_vyx[idx] = axh * f.m_vyx[idx] + bxh * dvy_dx;
    f.m_vyz[idx] = azh * f.m_vyz[idx] + bzh * dvy_dz;
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;

    dvx_dx += f.m_vxx[idx];
    dvx_dz += f.m_vxz[idx];
    dvy_dx += f.m_vyx[idx];
    dvy_dz += f.m_vyz[idx];
    dvz_dx += f.m_vzx[idx];
    dvz_dz += f.m_vzz[idx];

    const float C13 = m.C13[idx];
    const float C14 = m.C14[idx];
    const float C15 = m.C15[idx];
    const float C33 = m.C33[idx];
    const float C34 = m.C34[idx];
    const float C35 = m.C35[idx];
    const float C36 = m.C36[idx];
    const float C44 = m.C44[idx];
    const float C45 = m.C45[idx];
    const float C46 = m.C46[idx];
    const float C55 = m.C55[idx];
    const float C56 = m.C56[idx];

    if (elastic_is_top_free_surface_row(solver, iz)) {
        const float rhs_zz = C13 * dvx_dx + C36 * dvy_dx + C35 * dvz_dx;
        const float rhs_yz = C14 * dvx_dx + C46 * dvy_dx + C45 * dvz_dx;
        const float rhs_xz = C15 * dvx_dx + C56 * dvy_dx + C55 * dvz_dx;

        const float a = C35, bb = C34, c = C33;
        const float d = C45, e = C44, fcoef = C34;
        const float g = C55, h = C45, icoef = C35;
        const float r1 = -rhs_zz, r2 = -rhs_yz, r3 = -rhs_xz;
        const float det = a * (e * icoef - fcoef * h) - bb * (d * icoef - fcoef * g) + c * (d * h - e * g);

        if (fabsf(det) > 1.0e-20f) {
            dvx_dz = ((e * icoef - fcoef * h) * r1 + (c * h - bb * icoef) * r2 + (bb * fcoef - c * e) * r3) / det;
            dvy_dz = ((fcoef * g - d * icoef) * r1 + (a * icoef - c * g) * r2 + (c * d - a * fcoef) * r3) / det;
            dvz_dz = ((d * h - e * g) * r1 + (bb * g - a * h) * r2 + (a * e - bb * d) * r3) / det;
        }
    }

    const float shear_xz = dvz_dx + dvx_dz;

    f.sxx[idx] += solver.dt * (
        m.C11[idx] * dvx_dx + m.C16[idx] * dvy_dx + m.C15[idx] * shear_xz +
        m.C14[idx] * dvy_dz + m.C13[idx] * dvz_dz
    );
    f.szz[idx] += solver.dt * (
        m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * shear_xz +
        m.C34[idx] * dvy_dz + m.C33[idx] * dvz_dz
    );
    f.syz[idx] += solver.dt * (
        m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * shear_xz +
        m.C44[idx] * dvy_dz + m.C34[idx] * dvz_dz
    );
    f.sxz[idx] += solver.dt * (
        m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * shear_xz +
        m.C45[idx] * dvy_dz + m.C35[idx] * dvz_dz
    );
    f.sxy[idx] += solver.dt * (
        m.C16[idx] * dvx_dx + m.C66[idx] * dvy_dx + m.C56[idx] * shear_xz +
        m.C46[idx] * dvy_dz + m.C36[idx] * dvz_dz
    );

    if (elastic_is_top_free_surface_row(solver, iz)) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        f.syz[idx] = 0.f;
    }

    if (u_this_b) {
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
        u_this_b[3 * comp_stride + idx] = f.sxx[idx];
        u_this_b[4 * comp_stride + idx] = f.szz[idx];
        u_this_b[5 * comp_stride + idx] = f.syz[idx];
        u_this_b[6 * comp_stride + idx] = f.sxz[idx];
        u_this_b[7 * comp_stride + idx] = f.sxy[idx];
    }
}

template<int Order>
__global__ void elastic_tti_sg_velocity_kernel_nopml(
    WavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    const int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxy_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxy, ix, 0, iz, grad_ctx);
    float dsyz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.syz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.szz, ix, iz, grad_ctx, solver, true);

    const float scale = solver.dt / m.rho[idx];
    f.vx[idx] -= scale * (dsxx_dx + dsxz_dz);
    f.vy[idx] -= scale * (dsxy_dx + dsyz_dz);
    f.vz[idx] -= scale * (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_tti_sg_stress_kernel_nopml(
    WavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    const int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dvy_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vy, ix, 0, iz, grad_ctx);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vz, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.vx, ix, iz, grad_ctx, solver, false);
    float dvy_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.vy, ix, iz, grad_ctx, solver, false);

    const float shear_xz = dvz_dx + dvx_dz;
    f.sxx[idx] -= solver.dt * (
        m.C11[idx] * dvx_dx + m.C16[idx] * dvy_dx + m.C15[idx] * shear_xz +
        m.C14[idx] * dvy_dz + m.C13[idx] * dvz_dz
    );
    f.szz[idx] -= solver.dt * (
        m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * shear_xz +
        m.C34[idx] * dvy_dz + m.C33[idx] * dvz_dz
    );
    f.syz[idx] -= solver.dt * (
        m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * shear_xz +
        m.C44[idx] * dvy_dz + m.C34[idx] * dvz_dz
    );
    f.sxz[idx] -= solver.dt * (
        m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * shear_xz +
        m.C45[idx] * dvy_dz + m.C35[idx] * dvz_dz
    );
    f.sxy[idx] -= solver.dt * (
        m.C16[idx] * dvx_dx + m.C66[idx] * dvy_dx + m.C56[idx] * shear_xz +
        m.C46[idx] * dvy_dz + m.C36[idx] * dvz_dz
    );
}

template<int Order>
__global__ void elastic_tti_sg_stress_adjoint_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ q_vxx,
    float* __restrict__ q_vyx,
    float* __restrict__ q_vzx,
    float* __restrict__ q_vxz,
    float* __restrict__ q_vyz,
    float* __restrict__ q_vzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_syz = f.syz[idx];
    float bar_sxz = f.sxz[idx];
    float bar_sxy = f.sxy[idx];
    if (elastic_is_top_free_surface_row(solver, iz)) {
        bar_szz = 0.f;
        bar_syz = 0.f;
        bar_sxz = 0.f;
        f.szz[idx] = 0.f;
        f.syz[idx] = 0.f;
        f.sxz[idx] = 0.f;
    }

    const float bar_shear = solver.dt * (
        m.C15[idx] * bar_sxx + m.C35[idx] * bar_szz + m.C45[idx] * bar_syz +
        m.C55[idx] * bar_sxz + m.C56[idx] * bar_sxy
    );
    float bar_dvx_dx = solver.dt * (
        m.C11[idx] * bar_sxx + m.C13[idx] * bar_szz + m.C14[idx] * bar_syz +
        m.C15[idx] * bar_sxz + m.C16[idx] * bar_sxy
    );
    float bar_dvy_dx = solver.dt * (
        m.C16[idx] * bar_sxx + m.C36[idx] * bar_szz + m.C46[idx] * bar_syz +
        m.C56[idx] * bar_sxz + m.C66[idx] * bar_sxy
    );
    float bar_dvz_dx = bar_shear;
    // Stress uses (modified) dv*_dz at the free-surface row, so bar of the
    // *stress-side* dvx_dz/dvy_dz/dvz_dz is bar of the MODIFIED values:
    const float bar_dvx_dz_mod = bar_shear;
    const float bar_dvy_dz_mod = solver.dt * (
        m.C14[idx] * bar_sxx + m.C34[idx] * bar_szz + m.C44[idx] * bar_syz +
        m.C45[idx] * bar_sxz + m.C46[idx] * bar_sxy
    );
    const float bar_dvz_dz_mod = solver.dt * (
        m.C13[idx] * bar_sxx + m.C33[idx] * bar_szz + m.C34[idx] * bar_syz +
        m.C35[idx] * bar_sxz + m.C36[idx] * bar_sxy
    );

    // Outside the FS row, modified == full, so the pre-CPML bars are equal to
    // the stress-side bars. On the FS row the forward overwrites the full
    // values via the 3x3 traction-free system, so:
    //   (1) propagate bar(modified) back through the 3x3 system to bar(dv*_dx),
    //   (2) bar(dv*_dz_full) is zero because the full value never reaches the
    //       stress update (the 3x3 mod replaces it).
    float bar_dvx_dz_full = bar_dvx_dz_mod;
    float bar_dvy_dz_full = bar_dvy_dz_mod;
    float bar_dvz_dz_full = bar_dvz_dz_mod;
    if (elastic_is_top_free_surface_row(solver, iz)) {
        const float C13 = m.C13[idx], C14 = m.C14[idx], C15 = m.C15[idx];
        const float C33 = m.C33[idx], C34 = m.C34[idx], C35 = m.C35[idx], C36 = m.C36[idx];
        const float C44 = m.C44[idx], C45 = m.C45[idx], C46 = m.C46[idx];
        const float C55 = m.C55[idx], C56 = m.C56[idx];
        const float a = C35, bb = C34, c = C33;
        const float d = C45, e = C44, fcoef = C34;
        const float g = C55, h = C45, icoef = C35;
        const float det = a * (e * icoef - fcoef * h) - bb * (d * icoef - fcoef * g) + c * (d * h - e * g);
        if (fabsf(det) > 1.0e-20f) {
            const float inv = 1.f / det;
            const float br1 = ((e * icoef - fcoef * h) * bar_dvx_dz_mod +
                                (fcoef * g - d * icoef) * bar_dvy_dz_mod +
                                (d * h - e * g) * bar_dvz_dz_mod) * inv;
            const float br2 = ((c * h - bb * icoef) * bar_dvx_dz_mod +
                                (a * icoef - c * g) * bar_dvy_dz_mod +
                                (bb * g - a * h) * bar_dvz_dz_mod) * inv;
            const float br3 = ((bb * fcoef - c * e) * bar_dvx_dz_mod +
                                (c * d - a * fcoef) * bar_dvy_dz_mod +
                                (a * e - bb * d) * bar_dvz_dz_mod) * inv;
            // r1 = -rhs_zz = -(C13*dvx_dx + C36*dvy_dx + C35*dvz_dx) and similarly
            // for r2 (using C14, C46, C45) and r3 (using C15, C56, C55).
            // ∂L/∂dvx_dx via the 3x3 path is -C13*br1 - C14*br2 - C15*br3, etc.
            bar_dvx_dx -= C13 * br1 + C14 * br2 + C15 * br3;
            bar_dvy_dx -= C36 * br1 + C46 * br2 + C56 * br3;
            bar_dvz_dx -= C35 * br1 + C45 * br2 + C55 * br3;
        }
        bar_dvx_dz_full = 0.f;
        bar_dvy_dz_full = 0.f;
        bar_dvz_dz_full = 0.f;
    }

    // Position-based PML / interior split. Same logic as elastic2d's
    // adjoint prepare: ax/az/bx/bz vanish outside the PML band, m_v* aux
    // fields stay 0, so the six q_v* outputs collapse to bar_dv*_d* and
    // the six m_v* writes become 0 -> 0. Skip them.
    const int shift = b * spatial_size + idx;
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        q_vxx[shift] = bar_dvx_dx;
        q_vxz[shift] = bar_dvx_dz_full;
        q_vyx[shift] = bar_dvy_dx;
        q_vyz[shift] = bar_dvy_dz_full;
        q_vzx[shift] = bar_dvz_dx;
        q_vzz[shift] = bar_dvz_dz_full;
        return;
    }

    const float az = cpml.az[iz];
    const float bz = cpml.bz[iz];
    const float azh = cpml.azh[iz];
    const float bzh = cpml.bzh[iz];
    const float ax = cpml.ax[ix];
    const float bx = cpml.bx[ix];
    const float axh = cpml.axh[ix];
    const float bxh = cpml.bxh[ix];

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz_full;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz_full;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz_full;

    q_vxx[shift] = bar_dvx_dx + bx * tmp_vxx;
    q_vxz[shift] = bar_dvx_dz_full + bzh * tmp_vxz;
    q_vyx[shift] = bar_dvy_dx + bxh * tmp_vyx;
    q_vyz[shift] = bar_dvy_dz_full + bzh * tmp_vyz;
    q_vzx[shift] = bar_dvz_dx + bxh * tmp_vzx;
    q_vzz[shift] = bar_dvz_dz_full + bz * tmp_vzz;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vyx[idx] = axh * tmp_vyx;
    f.m_vyz[idx] = azh * tmp_vyz;
    f.m_vzx[idx] = axh * tmp_vzx;
    f.m_vzz[idx] = az * tmp_vzz;
}

template<int Order>
__global__ void elastic_tti_sg_stress_adjoint_apply(
    WavefieldPointer wf,
    const float* __restrict__ q_vxx,
    const float* __restrict__ q_vyx,
    const float* __restrict__ q_vzx,
    const float* __restrict__ q_vxz,
    const float* __restrict__ q_vyz,
    const float* __restrict__ q_vzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;
    auto f = wf.offset(b, spatial_size);

    const float* vxx = q_vxx + shift;
    const float* vyx = q_vyx + shift;
    const float* vzx = q_vzx + shift;
    const float* vxz = q_vxz + shift;
    const float* vyz = q_vyz + shift;
    const float* vzz = q_vzz + shift;

    f.vx[idx] += sgradient<2, Order, X, DIFF_FORWARD>(vxx, ix, 0, iz, grad_ctx);
    f.vy[idx] += sgradient<2, Order, X, DIFF_BACKWARD>(vyx, ix, 0, iz, grad_ctx);
    f.vz[idx] += sgradient<2, Order, X, DIFF_BACKWARD>(vzx, ix, 0, iz, grad_ctx);
    f.vx[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD>(vxz, ix, iz, grad_ctx, solver, false);
    f.vy[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD>(vyz, ix, iz, grad_ctx, solver, false);
    f.vz[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(vzz, ix, iz, grad_ctx, solver, true);
}

template<int Order>
__global__ void elastic_tti_sg_velocity_adjoint_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ q_txxx,
    float* __restrict__ q_txzz,
    float* __restrict__ q_txyx,
    float* __restrict__ q_tyzz,
    float* __restrict__ q_txzx,
    float* __restrict__ q_tzzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    const float scale = solver.dt / m.rho[idx];
    const float bar_dsxx_dx = scale * f.vx[idx];
    const float bar_dsxz_dz = scale * f.vx[idx];
    const float bar_dsxy_dx = scale * f.vy[idx];
    const float bar_dsyz_dz = scale * f.vy[idx];
    const float bar_dsxz_dx = scale * f.vz[idx];
    const float bar_dszz_dz = scale * f.vz[idx];

    // Position-based PML / interior split. Outside the PML band the
    // ax/az/bx/bz coefficients vanish, m_t* aux fields stay 0, so the six
    // q_t* outputs collapse to bar_ds*_d* and the six m_t* writes become
    // 0 -> 0. Skip them.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        q_txxx[shift] = bar_dsxx_dx;
        q_txzz[shift] = bar_dsxz_dz;
        q_txyx[shift] = bar_dsxy_dx;
        q_tyzz[shift] = bar_dsyz_dz;
        q_txzx[shift] = bar_dsxz_dx;
        q_tzzz[shift] = bar_dszz_dz;
        return;
    }

    const float az = cpml.az[iz];
    const float bz = cpml.bz[iz];
    const float azh = cpml.azh[iz];
    const float bzh = cpml.bzh[iz];
    const float ax = cpml.ax[ix];
    const float bx = cpml.bx[ix];
    const float axh = cpml.axh[ix];
    const float bxh = cpml.bxh[ix];

    float tmp_txxx = f.m_txxx[idx] + bar_dsxx_dx;
    float tmp_txzz = f.m_txzz[idx] + bar_dsxz_dz;
    float tmp_txyx = f.m_txyx[idx] + bar_dsxy_dx;
    float tmp_tyzz = f.m_tyzz[idx] + bar_dsyz_dz;
    float tmp_txzx = f.m_txzx[idx] + bar_dsxz_dx;
    float tmp_tzzz = f.m_tzzz[idx] + bar_dszz_dz;

    q_txxx[shift] = bar_dsxx_dx + bxh * tmp_txxx;
    q_txzz[shift] = bar_dsxz_dz + bz * tmp_txzz;
    q_txyx[shift] = bar_dsxy_dx + bx * tmp_txyx;
    q_tyzz[shift] = bar_dsyz_dz + bz * tmp_tyzz;
    q_txzx[shift] = bar_dsxz_dx + bx * tmp_txzx;
    q_tzzz[shift] = bar_dszz_dz + bzh * tmp_tzzz;

    f.m_txxx[idx] = axh * tmp_txxx;
    f.m_txzz[idx] = az * tmp_txzz;
    f.m_txyx[idx] = ax * tmp_txyx;
    f.m_tyzz[idx] = az * tmp_tyzz;
    f.m_txzx[idx] = ax * tmp_txzx;
    f.m_tzzz[idx] = azh * tmp_tzzz;
}

template<int Order>
__global__ void elastic_tti_sg_velocity_adjoint_apply(
    WavefieldPointer wf,
    const float* __restrict__ q_txxx,
    const float* __restrict__ q_txzz,
    const float* __restrict__ q_txyx,
    const float* __restrict__ q_tyzz,
    const float* __restrict__ q_txzx,
    const float* __restrict__ q_tzzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;
    auto f = wf.offset(b, spatial_size);

    const float* txxx = q_txxx + shift;
    const float* txzz = q_txzz + shift;
    const float* txyx = q_txyx + shift;
    const float* tyzz = q_tyzz + shift;
    const float* txzx = q_txzx + shift;
    const float* tzzz = q_tzzz + shift;

    f.sxx[idx] += sgradient<2, Order, X, DIFF_BACKWARD>(txxx, ix, 0, iz, grad_ctx);
    f.sxz[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(txzz, ix, iz, grad_ctx, solver, true);
    f.sxy[idx] += sgradient<2, Order, X, DIFF_FORWARD>(txyx, ix, 0, iz, grad_ctx);
    f.syz[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(tyzz, ix, iz, grad_ctx, solver, true);
    f.sxz[idx] += sgradient<2, Order, X, DIFF_FORWARD>(txzx, ix, 0, iz, grad_ctx);
    f.szz[idx] += elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD>(tzzz, ix, iz, grad_ctx, solver, true);
}

template<int Order>
__global__ void calculate_grad_elastic_tti_sg_nobs(
    WavefieldPointer adjoint,
    StiffnessPointer model,
    StiffnessGradPointer grad,
    const float* __restrict__ fvx,
    const float* __restrict__ fvy,
    const float* __restrict__ fvz,
    const float* __restrict__ fvx_next,
    const float* __restrict__ fvy_next,
    const float* __restrict__ fvz_next,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    const float* vx = fvx + shift;
    const float* vy = fvy + shift;
    const float* vz = fvz + shift;
    const float* vx_next = fvx_next + shift;
    const float* vy_next = fvy_next + shift;
    const float* vz_next = fvz_next + shift;

    auto a = adjoint.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    auto g = grad.offset(b, spatial_size);

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(vx, ix, 0, iz, grad_ctx);
    float dvy_dx = sgradient<2, Order, X, DIFF_FORWARD>(vy, ix, 0, iz, grad_ctx);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>(vz, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(vx, ix, iz, grad_ctx, solver, false);
    float dvy_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(vy, ix, iz, grad_ctx, solver, false);

    // At the FS row the forward replaces dv*_dz with the 3x3 traction-free
    // solution before the stress update, so ∂σ_**/∂C** must be evaluated
    // against those modified gradients.
    if (elastic_is_top_free_surface_row(solver, iz)) {
        const float C13 = m.C13[idx], C14 = m.C14[idx], C15 = m.C15[idx];
        const float C33 = m.C33[idx], C34 = m.C34[idx], C35 = m.C35[idx], C36 = m.C36[idx];
        const float C44 = m.C44[idx], C45 = m.C45[idx], C46 = m.C46[idx];
        const float C55 = m.C55[idx], C56 = m.C56[idx];
        const float aa = C35, bb = C34, cc = C33;
        const float d = C45, e = C44, fcoef = C34;
        const float g = C55, h = C45, icoef = C35;
        const float det = aa * (e * icoef - fcoef * h) - bb * (d * icoef - fcoef * g) + cc * (d * h - e * g);
        if (fabsf(det) > 1.0e-20f) {
            const float rhs_zz = C13 * dvx_dx + C36 * dvy_dx + C35 * dvz_dx;
            const float rhs_yz = C14 * dvx_dx + C46 * dvy_dx + C45 * dvz_dx;
            const float rhs_xz = C15 * dvx_dx + C56 * dvy_dx + C55 * dvz_dx;
            const float r1 = -rhs_zz, r2 = -rhs_yz, r3 = -rhs_xz;
            dvx_dz = ((e * icoef - fcoef * h) * r1 + (cc * h - bb * icoef) * r2 + (bb * fcoef - cc * e) * r3) / det;
            dvy_dz = ((fcoef * g - d * icoef) * r1 + (aa * icoef - cc * g) * r2 + (cc * d - aa * fcoef) * r3) / det;
            dvz_dz = ((d * h - e * g) * r1 + (bb * g - aa * h) * r2 + (aa * e - bb * d) * r3) / det;
        }
    }
    float shear_xz = dvz_dx + dvx_dz;

    float bar_sxx = a.sxx[idx];
    float bar_szz = a.szz[idx];
    float bar_syz = a.syz[idx];
    float bar_sxz = a.sxz[idx];
    float bar_sxy = a.sxy[idx];
    if (elastic_is_top_free_surface_row(solver, iz)) {
        bar_szz = 0.f;
        bar_syz = 0.f;
        bar_sxz = 0.f;
    }

    const float scale = -solver.dt;
    g.C11[idx] += scale * bar_sxx * dvx_dx;
    g.C13[idx] += scale * (bar_sxx * dvz_dz + bar_szz * dvx_dx);
    g.C14[idx] += scale * (bar_sxx * dvy_dz + bar_syz * dvx_dx);
    g.C15[idx] += scale * (bar_sxx * shear_xz + bar_sxz * dvx_dx);
    g.C16[idx] += scale * (bar_sxx * dvy_dx + bar_sxy * dvx_dx);
    g.C33[idx] += scale * bar_szz * dvz_dz;
    g.C34[idx] += scale * (bar_szz * dvy_dz + bar_syz * dvz_dz);
    g.C35[idx] += scale * (bar_szz * shear_xz + bar_sxz * dvz_dz);
    g.C36[idx] += scale * (bar_szz * dvy_dx + bar_sxy * dvz_dz);
    g.C44[idx] += scale * bar_syz * dvy_dz;
    g.C45[idx] += scale * (bar_syz * shear_xz + bar_sxz * dvy_dz);
    g.C46[idx] += scale * (bar_syz * dvy_dx + bar_sxy * dvy_dz);
    g.C55[idx] += scale * bar_sxz * shear_xz;
    g.C56[idx] += scale * (bar_sxz * dvy_dx + bar_sxy * shear_xz);
    g.C66[idx] += scale * bar_sxy * dvy_dx;

    g.rho[idx] += (
        a.vx[idx] * (vx[idx] - vx_next[idx]) +
        a.vy[idx] * (vy[idx] - vy_next[idx]) +
        a.vz[idx] * (vz[idx] - vz_next[idx])
    ) / m.rho[idx];
}

}

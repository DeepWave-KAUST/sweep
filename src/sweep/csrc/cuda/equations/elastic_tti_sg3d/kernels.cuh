#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../operators/staggered.cuh"

// 3-D elastic TTI on the Virieux SSG (axis-aligned, no interpolation of
// mixed-location stiffness couplings — the 3-D companion of elastic_tti_sg2d).
// Wavefield layout, CPML staging and thread mapping are identical to
// elastic3d; only the constitutive update differs: the six stresses couple to
// all six Voigt strain rates through the 21 Bond-rotated stiffness entries.
// Anisotropy rejects the image-method free surface (guarded on the Python
// side), so all z-derivatives use the plain staggered stencils.

namespace elastic_tti_sg3d {

struct StiffnessPointer {
    const float* __restrict__ rho;
    const float* __restrict__ C11;
    const float* __restrict__ C12;
    const float* __restrict__ C13;
    const float* __restrict__ C14;
    const float* __restrict__ C15;
    const float* __restrict__ C16;
    const float* __restrict__ C22;
    const float* __restrict__ C23;
    const float* __restrict__ C24;
    const float* __restrict__ C25;
    const float* __restrict__ C26;
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
        out.C12 += shift;
        out.C13 += shift;
        out.C14 += shift;
        out.C15 += shift;
        out.C16 += shift;
        out.C22 += shift;
        out.C23 += shift;
        out.C24 += shift;
        out.C25 += shift;
        out.C26 += shift;
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
    float* __restrict__ C12;
    float* __restrict__ C13;
    float* __restrict__ C14;
    float* __restrict__ C15;
    float* __restrict__ C16;
    float* __restrict__ C22;
    float* __restrict__ C23;
    float* __restrict__ C24;
    float* __restrict__ C25;
    float* __restrict__ C26;
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
        out.C12 += shift;
        out.C13 += shift;
        out.C14 += shift;
        out.C15 += shift;
        out.C16 += shift;
        out.C22 += shift;
        out.C23 += shift;
        out.C24 += shift;
        out.C25 += shift;
        out.C26 += shift;
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

#define LAUNCH_ELASTIC_TTI_SG3D_VELOCITY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_velocity_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_velocity_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_velocity_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_velocity_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_velocity_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_STRESS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_velocity_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_velocity_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_velocity_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_velocity_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_velocity_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_STRESS_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_STRESS_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_stress_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_stress_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_stress_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_stress_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_stress_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_STRESS_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_stress_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_stress_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_stress_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_stress_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_stress_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_velocity_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_velocity_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_velocity_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_velocity_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_velocity_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_tti_sg3d_velocity_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_tti_sg3d_velocity_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_tti_sg3d_velocity_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_tti_sg3d_velocity_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_tti_sg3d_velocity_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG3D_NOBS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) calculate_grad_elastic_tti_sg3d_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_tti_sg3d_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_tti_sg3d_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_tti_sg3d_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_tti_sg3d_nobs<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void __launch_bounds__(256) elastic_tti_sg3d_velocity_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* rho_b = rho + b * spatial_size;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);

    float inv_rho = 1.f / rho_b[idx];

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        f.vx[idx] += solver.dt * inv_rho * (dsxx_dx + dsxy_dy + dsxz_dz);
        f.vy[idx] += solver.dt * inv_rho * (dsxy_dx + dsyy_dy + dsyz_dz);
        f.vz[idx] += solver.dt * inv_rho * (dsxz_dx + dsyz_dy + dszz_dz);
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.m_sxyy[idx] = ay * f.m_sxyy[idx] + by * dsxy_dy;
    dsxy_dy += f.m_sxyy[idx];

    f.m_sxyx[idx] = ax * f.m_sxyx[idx] + bx * dsxy_dx;
    dsxy_dx += f.m_sxyx[idx];

    f.m_syyy[idx] = ayh * f.m_syyy[idx] + byh * dsyy_dy;
    dsyy_dy += f.m_syyy[idx];

    f.m_syzz[idx] = az * f.m_syzz[idx] + bz * dsyz_dz;
    dsyz_dz += f.m_syzz[idx];

    f.m_syzy[idx] = ay * f.m_syzy[idx] + by * dsyz_dy;
    dsyz_dy += f.m_syzy[idx];

    f.vx[idx] += solver.dt * inv_rho * (dsxx_dx + dsxy_dy + dsxz_dz);
    f.vy[idx] += solver.dt * inv_rho * (dsxy_dx + dsyy_dy + dsyz_dz);
    f.vz[idx] += solver.dt * inv_rho * (dsxz_dx + dsyz_dy + dszz_dz);
}

template<int Order>
__global__ void __launch_bounds__(256) elastic_tti_sg3d_stress_kernel(
    ElasticWavefieldPointer wf,
    StiffnessPointer model,
    float* __restrict__ u_this_t,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (!interior) {
        float az = cpml.az[iz];
        float bz = cpml.bz[iz];
        float azh = cpml.azh[iz];
        float bzh = cpml.bzh[iz];

        float ay = cpml.ay[iy];
        float by = cpml.by[iy];
        float ayh = cpml.ayh[iy];
        float byh = cpml.byh[iy];

        float ax = cpml.ax[ix];
        float bx = cpml.bx[ix];
        float axh = cpml.axh[ix];
        float bxh = cpml.bxh[ix];

        f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
        dvz_dz += f.m_vzz[idx];
        f.m_vyy[idx] = ay * f.m_vyy[idx] + by * dvy_dy;
        dvy_dy += f.m_vyy[idx];
        f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
        dvx_dx += f.m_vxx[idx];
        f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
        dvx_dz += f.m_vxz[idx];
        f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
        dvz_dx += f.m_vzx[idx];

        f.m_vxy[idx] = ayh * f.m_vxy[idx] + byh * dvx_dy;
        dvx_dy += f.m_vxy[idx];
        f.m_vyx[idx] = axh * f.m_vyx[idx] + bxh * dvy_dx;
        dvy_dx += f.m_vyx[idx];
        f.m_vyz[idx] = azh * f.m_vyz[idx] + bzh * dvy_dz;
        dvy_dz += f.m_vyz[idx];
        f.m_vzy[idx] = ayh * f.m_vzy[idx] + byh * dvz_dy;
        dvz_dy += f.m_vzy[idx];
    }

    // Voigt strain rates (engineering shears).
    float e1 = dvx_dx;
    float e2 = dvy_dy;
    float e3 = dvz_dz;
    float e4 = dvy_dz + dvz_dy;
    float e5 = dvx_dz + dvz_dx;
    float e6 = dvx_dy + dvy_dx;

    f.sxx[idx] += solver.dt * (
        m.C11[idx] * e1 + m.C12[idx] * e2 + m.C13[idx] * e3 +
        m.C14[idx] * e4 + m.C15[idx] * e5 + m.C16[idx] * e6
    );
    f.syy[idx] += solver.dt * (
        m.C12[idx] * e1 + m.C22[idx] * e2 + m.C23[idx] * e3 +
        m.C24[idx] * e4 + m.C25[idx] * e5 + m.C26[idx] * e6
    );
    f.szz[idx] += solver.dt * (
        m.C13[idx] * e1 + m.C23[idx] * e2 + m.C33[idx] * e3 +
        m.C34[idx] * e4 + m.C35[idx] * e5 + m.C36[idx] * e6
    );
    f.syz[idx] += solver.dt * (
        m.C14[idx] * e1 + m.C24[idx] * e2 + m.C34[idx] * e3 +
        m.C44[idx] * e4 + m.C45[idx] * e5 + m.C46[idx] * e6
    );
    f.sxz[idx] += solver.dt * (
        m.C15[idx] * e1 + m.C25[idx] * e2 + m.C35[idx] * e3 +
        m.C45[idx] * e4 + m.C55[idx] * e5 + m.C56[idx] * e6
    );
    f.sxy[idx] += solver.dt * (
        m.C16[idx] * e1 + m.C26[idx] * e2 + m.C36[idx] * e3 +
        m.C46[idx] * e4 + m.C56[idx] * e5 + m.C66[idx] * e6
    );

    if (u_this_t) {
        float* u_this_b = u_this_t + b * spatial_size;
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
    }
}

template<int Order>
__global__ void elastic_tti_sg3d_velocity_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + M + 1;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* rho_b = rho + b * spatial_size;

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);

    float inv_rho = 1.f / rho_b[idx];

    f.vx[idx] -= solver.dt * inv_rho * (dsxx_dx + dsxy_dy + dsxz_dz);
    f.vy[idx] -= solver.dt * inv_rho * (dsxy_dx + dsyy_dy + dsyz_dz);
    f.vz[idx] -= solver.dt * inv_rho * (dsxz_dx + dsyz_dy + dszz_dz);
}

template<int Order>
__global__ void elastic_tti_sg3d_stress_kernel_nopml(
    ElasticWavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + M + 1;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    float e1 = dvx_dx;
    float e2 = dvy_dy;
    float e3 = dvz_dz;
    float e4 = dvy_dz + dvz_dy;
    float e5 = dvx_dz + dvz_dx;
    float e6 = dvx_dy + dvy_dx;

    f.sxx[idx] -= solver.dt * (
        m.C11[idx] * e1 + m.C12[idx] * e2 + m.C13[idx] * e3 +
        m.C14[idx] * e4 + m.C15[idx] * e5 + m.C16[idx] * e6
    );
    f.syy[idx] -= solver.dt * (
        m.C12[idx] * e1 + m.C22[idx] * e2 + m.C23[idx] * e3 +
        m.C24[idx] * e4 + m.C25[idx] * e5 + m.C26[idx] * e6
    );
    f.szz[idx] -= solver.dt * (
        m.C13[idx] * e1 + m.C23[idx] * e2 + m.C33[idx] * e3 +
        m.C34[idx] * e4 + m.C35[idx] * e5 + m.C36[idx] * e6
    );
    f.syz[idx] -= solver.dt * (
        m.C14[idx] * e1 + m.C24[idx] * e2 + m.C34[idx] * e3 +
        m.C44[idx] * e4 + m.C45[idx] * e5 + m.C46[idx] * e6
    );
    f.sxz[idx] -= solver.dt * (
        m.C15[idx] * e1 + m.C25[idx] * e2 + m.C35[idx] * e3 +
        m.C45[idx] * e4 + m.C55[idx] * e5 + m.C56[idx] * e6
    );
    f.sxy[idx] -= solver.dt * (
        m.C16[idx] * e1 + m.C26[idx] * e2 + m.C36[idx] * e3 +
        m.C46[idx] * e4 + m.C56[idx] * e5 + m.C66[idx] * e6
    );
}

// Transpose of the 21-entry constitutive update.  bar of each Voigt strain
// rate is dt * (C row · bar stress); the engineering-shear bars fan out to
// both of their velocity-gradient parents.  CPML adjoint staging is
// identical to elastic3d's stress_adjoint_prepare (same aux fields, same
// interior fast-path); the q* workspace layout is
//   qxx=bar_dvx_dx qxy=bar_dvx_dy qxz=bar_dvx_dz
//   qyx=bar_dvy_dx qyy=bar_dvy_dy qyz=bar_dvy_dz
//   qzx=bar_dvz_dx qzy=bar_dvz_dy qzz=bar_dvz_dz
template<int Order>
__global__ void elastic_tti_sg3d_stress_adjoint_prepare(
    ElasticWavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qxy,
    float* __restrict__ qxz,
    float* __restrict__ qyx,
    float* __restrict__ qyy,
    float* __restrict__ qyz,
    float* __restrict__ qzx,
    float* __restrict__ qzy,
    float* __restrict__ qzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float bar_sxx = f.sxx[idx];
    float bar_syy = f.syy[idx];
    float bar_szz = f.szz[idx];
    float bar_syz = f.syz[idx];
    float bar_sxz = f.sxz[idx];
    float bar_sxy = f.sxy[idx];

    float bar_e1 = solver.dt * (
        m.C11[idx] * bar_sxx + m.C12[idx] * bar_syy + m.C13[idx] * bar_szz +
        m.C14[idx] * bar_syz + m.C15[idx] * bar_sxz + m.C16[idx] * bar_sxy
    );
    float bar_e2 = solver.dt * (
        m.C12[idx] * bar_sxx + m.C22[idx] * bar_syy + m.C23[idx] * bar_szz +
        m.C24[idx] * bar_syz + m.C25[idx] * bar_sxz + m.C26[idx] * bar_sxy
    );
    float bar_e3 = solver.dt * (
        m.C13[idx] * bar_sxx + m.C23[idx] * bar_syy + m.C33[idx] * bar_szz +
        m.C34[idx] * bar_syz + m.C35[idx] * bar_sxz + m.C36[idx] * bar_sxy
    );
    float bar_e4 = solver.dt * (
        m.C14[idx] * bar_sxx + m.C24[idx] * bar_syy + m.C34[idx] * bar_szz +
        m.C44[idx] * bar_syz + m.C45[idx] * bar_sxz + m.C46[idx] * bar_sxy
    );
    float bar_e5 = solver.dt * (
        m.C15[idx] * bar_sxx + m.C25[idx] * bar_syy + m.C35[idx] * bar_szz +
        m.C45[idx] * bar_syz + m.C55[idx] * bar_sxz + m.C56[idx] * bar_sxy
    );
    float bar_e6 = solver.dt * (
        m.C16[idx] * bar_sxx + m.C26[idx] * bar_syy + m.C36[idx] * bar_szz +
        m.C46[idx] * bar_syz + m.C56[idx] * bar_sxz + m.C66[idx] * bar_sxy
    );

    float bar_dvx_dx = bar_e1;
    float bar_dvy_dy = bar_e2;
    float bar_dvz_dz = bar_e3;
    float bar_dvy_dz = bar_e4;
    float bar_dvz_dy = bar_e4;
    float bar_dvx_dz = bar_e5;
    float bar_dvz_dx = bar_e5;
    float bar_dvx_dy = bar_e6;
    float bar_dvy_dx = bar_e6;

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxy = f.m_vxy[idx] + bar_dvx_dy;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyy = f.m_vyy[idx] + bar_dvy_dy;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzy = f.m_vzy[idx] + bar_dvz_dy;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        qxx[b * spatial_size + idx] = bar_dvx_dx;
        qxy[b * spatial_size + idx] = bar_dvx_dy;
        qxz[b * spatial_size + idx] = bar_dvx_dz;
        qyx[b * spatial_size + idx] = bar_dvy_dx;
        qyy[b * spatial_size + idx] = bar_dvy_dy;
        qyz[b * spatial_size + idx] = bar_dvy_dz;
        qzx[b * spatial_size + idx] = bar_dvz_dx;
        qzy[b * spatial_size + idx] = bar_dvz_dy;
        qzz[b * spatial_size + idx] = bar_dvz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    qxx[b * spatial_size + idx] = bar_dvx_dx + bx * tmp_vxx;
    qxy[b * spatial_size + idx] = bar_dvx_dy + byh * tmp_vxy;
    qxz[b * spatial_size + idx] = bar_dvx_dz + bzh * tmp_vxz;
    qyx[b * spatial_size + idx] = bar_dvy_dx + bxh * tmp_vyx;
    qyy[b * spatial_size + idx] = bar_dvy_dy + by * tmp_vyy;
    qyz[b * spatial_size + idx] = bar_dvy_dz + bzh * tmp_vyz;
    qzx[b * spatial_size + idx] = bar_dvz_dx + bxh * tmp_vzx;
    qzy[b * spatial_size + idx] = bar_dvz_dy + byh * tmp_vzy;
    qzz[b * spatial_size + idx] = bar_dvz_dz + bz * tmp_vzz;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vxy[idx] = ayh * tmp_vxy;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vyx[idx] = axh * tmp_vyx;
    f.m_vyy[idx] = ay * tmp_vyy;
    f.m_vyz[idx] = azh * tmp_vyz;
    f.m_vzx[idx] = axh * tmp_vzx;
    f.m_vzy[idx] = ayh * tmp_vzy;
    f.m_vzz[idx] = az * tmp_vzz;
}

template<int Order>
__global__ void elastic_tti_sg3d_stress_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ qxx,
    const float* __restrict__ qxy,
    const float* __restrict__ qxz,
    const float* __restrict__ qyx,
    const float* __restrict__ qyy,
    const float* __restrict__ qyz,
    const float* __restrict__ qzx,
    const float* __restrict__ qzy,
    const float* __restrict__ qzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* qxx_b = qxx + b * spatial_size;
    const float* qxy_b = qxy + b * spatial_size;
    const float* qxz_b = qxz + b * spatial_size;
    const float* qyx_b = qyx + b * spatial_size;
    const float* qyy_b = qyy + b * spatial_size;
    const float* qyz_b = qyz + b * spatial_size;
    const float* qzx_b = qzx + b * spatial_size;
    const float* qzy_b = qzy + b * spatial_size;
    const float* qzz_b = qzz + b * spatial_size;

    float dqxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(qxx_b, ix, iy, iz, grad_ctx);
    float dqxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qxy_b, ix, iy, iz, grad_ctx);
    float dqxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(qxz_b, ix, iy, iz, grad_ctx);

    float dqyx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qyx_b, ix, iy, iz, grad_ctx);
    float dqyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(qyy_b, ix, iy, iz, grad_ctx);
    float dqyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(qyz_b, ix, iy, iz, grad_ctx);

    float dqzx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qzx_b, ix, iy, iz, grad_ctx);
    float dqzy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qzy_b, ix, iy, iz, grad_ctx);
    float dqzz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(qzz_b, ix, iy, iz, grad_ctx);

    f.vx[idx] += dqxx_dx + dqxy_dy + dqxz_dz;
    f.vy[idx] += dqyx_dx + dqyy_dy + dqyz_dz;
    f.vz[idx] += dqzx_dx + dqzy_dy + dqzz_dz;
}

template<int Order>
__global__ void elastic_tti_sg3d_velocity_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pxy,
    float* __restrict__ pxz,
    float* __restrict__ pyx,
    float* __restrict__ pyy,
    float* __restrict__ pyz,
    float* __restrict__ pzx,
    float* __restrict__ pzy,
    float* __restrict__ pzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    const float* rho_b = rho + b * spatial_size;
    auto f = wf.offset(b, spatial_size);

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float inv_rho = 1.f / rho_b[idx];

    float bar_dsxx_dx = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxy_dy = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dz = solver.dt * inv_rho * f.vx[idx];

    float bar_dsxy_dx = solver.dt * inv_rho * f.vy[idx];
    float bar_dsyy_dy = solver.dt * inv_rho * f.vy[idx];
    float bar_dsyz_dz = solver.dt * inv_rho * f.vy[idx];

    float bar_dsxz_dx = solver.dt * inv_rho * f.vz[idx];
    float bar_dsyz_dy = solver.dt * inv_rho * f.vz[idx];
    float bar_dszz_dz = solver.dt * inv_rho * f.vz[idx];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxyy = f.m_sxyy[idx] + bar_dsxy_dy;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxyx = f.m_sxyx[idx] + bar_dsxy_dx;
    float tmp_syyy = f.m_syyy[idx] + bar_dsyy_dy;
    float tmp_syzz = f.m_syzz[idx] + bar_dsyz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_syzy = f.m_syzy[idx] + bar_dsyz_dy;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        pxx[b * spatial_size + idx] = bar_dsxx_dx;
        pxy[b * spatial_size + idx] = bar_dsxy_dy;
        pxz[b * spatial_size + idx] = bar_dsxz_dz;
        pyx[b * spatial_size + idx] = bar_dsxy_dx;
        pyy[b * spatial_size + idx] = bar_dsyy_dy;
        pyz[b * spatial_size + idx] = bar_dsyz_dz;
        pzx[b * spatial_size + idx] = bar_dsxz_dx;
        pzy[b * spatial_size + idx] = bar_dsyz_dy;
        pzz[b * spatial_size + idx] = bar_dszz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    pxx[b * spatial_size + idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxy[b * spatial_size + idx] = bar_dsxy_dy + by * tmp_sxyy;
    pxz[b * spatial_size + idx] = bar_dsxz_dz + bz * tmp_sxzz;
    pyx[b * spatial_size + idx] = bar_dsxy_dx + bx * tmp_sxyx;
    pyy[b * spatial_size + idx] = bar_dsyy_dy + byh * tmp_syyy;
    pyz[b * spatial_size + idx] = bar_dsyz_dz + bz * tmp_syzz;
    pzx[b * spatial_size + idx] = bar_dsxz_dx + bx * tmp_sxzx;
    pzy[b * spatial_size + idx] = bar_dsyz_dy + by * tmp_syzy;
    pzz[b * spatial_size + idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxyy[idx] = ay * tmp_sxyy;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxyx[idx] = ax * tmp_sxyx;
    f.m_syyy[idx] = ayh * tmp_syyy;
    f.m_syzz[idx] = az * tmp_syzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_syzy[idx] = ay * tmp_syzy;
    f.m_szzz[idx] = azh * tmp_szzz;
}

template<int Order>
__global__ void elastic_tti_sg3d_velocity_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ pxx,
    const float* __restrict__ pxy,
    const float* __restrict__ pxz,
    const float* __restrict__ pyx,
    const float* __restrict__ pyy,
    const float* __restrict__ pyz,
    const float* __restrict__ pzx,
    const float* __restrict__ pzy,
    const float* __restrict__ pzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* pxx_b = pxx + b * spatial_size;
    const float* pxy_b = pxy + b * spatial_size;
    const float* pxz_b = pxz + b * spatial_size;
    const float* pyx_b = pyx + b * spatial_size;
    const float* pyy_b = pyy + b * spatial_size;
    const float* pyz_b = pyz + b * spatial_size;
    const float* pzx_b = pzx + b * spatial_size;
    const float* pzy_b = pzy + b * spatial_size;
    const float* pzz_b = pzz + b * spatial_size;

    float dpxx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(pxx_b, ix, iy, iz, grad_ctx);
    float dpxy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(pxy_b, ix, iy, iz, grad_ctx);
    float dpxz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(pxz_b, ix, iy, iz, grad_ctx);

    float dpyx_dx = sgradient<3, Order, X, DIFF_FORWARD >(pyx_b, ix, iy, iz, grad_ctx);
    float dpyy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(pyy_b, ix, iy, iz, grad_ctx);
    float dpyz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(pyz_b, ix, iy, iz, grad_ctx);

    float dpzx_dx = sgradient<3, Order, X, DIFF_FORWARD >(pzx_b, ix, iy, iz, grad_ctx);
    float dpzy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(pzy_b, ix, iy, iz, grad_ctx);
    float dpzz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(pzz_b, ix, iy, iz, grad_ctx);

    f.sxx[idx] += dpxx_dx;
    f.sxy[idx] += dpxy_dy + dpyx_dx;
    f.sxz[idx] += dpxz_dz + dpzx_dx;
    f.syy[idx] += dpyy_dy;
    f.syz[idx] += dpyz_dz + dpzy_dy;
    f.szz[idx] += dpzz_dz;
}

// Per-step imaging of the 21 stiffness gradients and rho.  The forward
// velocities (fvx/fvy/fvz at time it, *_next at it+1) reproduce the Voigt
// strain rates the forward stress update consumed; each dL/dCij is the
// correlation of adjoint stress i with strain rate j (both orderings for
// off-diagonal entries, which are stored once).
template<int Order>
__global__ void calculate_grad_elastic_tti_sg3d_nobs(
    ElasticWavefieldPointer adjoint,
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
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int shift = b * spatial_size;

    const float* vx = fvx + shift;
    const float* vy = fvy + shift;
    const float* vz = fvz + shift;
    const float* vx_next = fvx_next + shift;
    const float* vy_next = fvy_next + shift;
    const float* vz_next = fvz_next + shift;

    auto a = adjoint.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    auto g = grad.offset(b, spatial_size);

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(vx, ix, iy, iz, grad_ctx);
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(vz, ix, iy, iz, grad_ctx);

    float e1 = dvx_dx;
    float e2 = dvy_dy;
    float e3 = dvz_dz;
    float e4 = dvy_dz + dvz_dy;
    float e5 = dvx_dz + dvz_dx;
    float e6 = dvx_dy + dvy_dx;

    float bar_sxx = a.sxx[idx];
    float bar_syy = a.syy[idx];
    float bar_szz = a.szz[idx];
    float bar_syz = a.syz[idx];
    float bar_sxz = a.sxz[idx];
    float bar_sxy = a.sxy[idx];

    const float scale = -solver.dt;
    g.C11[idx] += scale * bar_sxx * e1;
    g.C12[idx] += scale * (bar_sxx * e2 + bar_syy * e1);
    g.C13[idx] += scale * (bar_sxx * e3 + bar_szz * e1);
    g.C14[idx] += scale * (bar_sxx * e4 + bar_syz * e1);
    g.C15[idx] += scale * (bar_sxx * e5 + bar_sxz * e1);
    g.C16[idx] += scale * (bar_sxx * e6 + bar_sxy * e1);
    g.C22[idx] += scale * bar_syy * e2;
    g.C23[idx] += scale * (bar_syy * e3 + bar_szz * e2);
    g.C24[idx] += scale * (bar_syy * e4 + bar_syz * e2);
    g.C25[idx] += scale * (bar_syy * e5 + bar_sxz * e2);
    g.C26[idx] += scale * (bar_syy * e6 + bar_sxy * e2);
    g.C33[idx] += scale * bar_szz * e3;
    g.C34[idx] += scale * (bar_szz * e4 + bar_syz * e3);
    g.C35[idx] += scale * (bar_szz * e5 + bar_sxz * e3);
    g.C36[idx] += scale * (bar_szz * e6 + bar_sxy * e3);
    g.C44[idx] += scale * bar_syz * e4;
    g.C45[idx] += scale * (bar_syz * e5 + bar_sxz * e4);
    g.C46[idx] += scale * (bar_syz * e6 + bar_sxy * e4);
    g.C55[idx] += scale * bar_sxz * e5;
    g.C56[idx] += scale * (bar_sxz * e6 + bar_sxy * e5);
    g.C66[idx] += scale * bar_sxy * e6;

    g.rho[idx] += (
        a.vx[idx] * (vx[idx] - vx_next[idx]) +
        a.vy[idx] * (vy[idx] - vy_next[idx]) +
        a.vz[idx] * (vz[idx] - vz_next[idx])
    ) / m.rho[idx];
}

} // namespace elastic_tti_sg3d

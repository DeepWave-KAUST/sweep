#pragma once
// Born (single-scattering) forward kernels for the 2-D first-order elastic
// equation: one launch steps BOTH the background field ``wf`` (exactly
// elastic2d's velocity / stress kernels) and the scattered field ``sf`` (the
// same staggered operator, CPML and free surface, plus the Born sources).
//
//   velocity:  dv  += dt/rho (div dsigma)  - (drho/rho) * dt/rho (div sigma0)
//   stress:    dsig += dt C : eps(dv)       + dt dC : eps(v0)
//
// ``eps(v0)`` is the background's CPML-corrected strain rate of this step and
// ``dC`` the Lame perturbation of (dvp, dvs, drho) linearised at the
// background.  At a z (x) free-surface row the Robertsson surface sigma_xx
// (sigma_zz) has the coefficient c = 4 mu (lam + mu) / (lam + 2 mu); its Born
// source is dt dc vx_x (vz_z).  This is the exact Jacobian of the elastic2d
// step, the twin of sweep.equations.elastic_born.ElasticBorn.func.
#include "../../launch/by_order.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"

#define LAUNCH_ELASTIC_BORN_VELOCITY(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(elastic_born_velocity_kernel_by_order, order, grid, block, __VA_ARGS__)

#define LAUNCH_ELASTIC_BORN_STRESS(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(elastic_born_stress_kernel_by_order, order, grid, block, __VA_ARGS__)


template<int Order>
__global__ void __launch_bounds__(256) elastic_born_velocity_kernel(
    ElasticWavefieldPointer wf,
    ElasticWavefieldPointer sf,
    const float* __restrict__ rho,
    const float* __restrict__ drho,

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
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    auto g = sf.offset(b, spatial_size);

    float dsxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (f.sxx, ix, iz, grad_ctx, solver, true);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true, true);
    float dsxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true, true);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float esxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (g.sxx, ix, iz, grad_ctx, solver, true);
    float esxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(g.sxz, ix, iz, grad_ctx, solver, true, true);
    float esxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(g.sxz, ix, iz, grad_ctx, solver, true, true);
    float eszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (g.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho[b * spatial_size + idx];
    float r = drho[b * spatial_size + idx] * inv_rho;

    // elastic2d's PML / interior split (see elastic_velocity_kernel).
    bool in_pml = (!solver.cut_x_lo() && ix < solver.padLo(2) + halo) ||
                  (!solver.cut_x_hi() && ix >= solver.nx - solver.padHi(2) - halo) ||
                  (!solver.cut_z_lo() && iz < solver.padLo(0) + halo) ||
                  (!solver.cut_z_hi() && iz >= solver.nz - solver.padHi(0) - halo);

    if (in_pml) {
        float az = cpml.az[iz];
        float bz = cpml.bz[iz];
        float azh = cpml.azh[iz];
        float bzh = cpml.bzh[iz];

        float ax = cpml.ax[ix];
        float bx = cpml.bx[ix];
        float axh = cpml.axh[ix];
        float bxh = cpml.bxh[ix];

        long xi = solver.aux_rd_x2(iz, ix);
        long zi = solver.aux_rd_z2(iz, ix);
        bool st_x = solver.aux_x.stored(ix);
        bool st_z = solver.aux_z.stored(iz);

        float m;
        m = azh * f.m_szzz[zi] + bzh * dszz_dz; if (st_z) f.m_szzz[zi] = m; dszz_dz += m;
        m = ax  * f.m_sxzx[xi] + bx  * dsxz_dx; if (st_x) f.m_sxzx[xi] = m; dsxz_dx += m;
        m = az  * f.m_sxzz[zi] + bz  * dsxz_dz; if (st_z) f.m_sxzz[zi] = m; dsxz_dz += m;
        m = axh * f.m_sxxx[xi] + bxh * dsxx_dx; if (st_x) f.m_sxxx[xi] = m; dsxx_dx += m;

        m = azh * g.m_szzz[zi] + bzh * eszz_dz; if (st_z) g.m_szzz[zi] = m; eszz_dz += m;
        m = ax  * g.m_sxzx[xi] + bx  * esxz_dx; if (st_x) g.m_sxzx[xi] = m; esxz_dx += m;
        m = az  * g.m_sxzz[zi] + bz  * esxz_dz; if (st_z) g.m_sxzz[zi] = m; esxz_dz += m;
        m = axh * g.m_sxxx[xi] + bxh * esxx_dx; if (st_x) g.m_sxxx[xi] = m; esxx_dx += m;
    }

    float dvx = solver.dt * inv_rho * (dsxx_dx + dsxz_dz);
    float dvz = solver.dt * inv_rho * (dsxz_dx + dszz_dz);
    f.vx[idx] += dvx;
    f.vz[idx] += dvz;
    g.vx[idx] += solver.dt * inv_rho * (esxx_dx + esxz_dz) - r * dvx;
    g.vz[idx] += solver.dt * inv_rho * (esxz_dx + eszz_dz) - r * dvz;
}

using elastic_born_velocity_kernel_fn = void (*)(ElasticWavefieldPointer, ElasticWavefieldPointer,
    const float*, const float*, SGradParam, ElasticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(elastic_born_velocity_kernel_fn, elastic_born_velocity_kernel_by_order);


// The model pointers: background Lame pair (the derived_models slots), the
// raw background and perturbation used for the linearised Lame perturbation.
struct ElasticBornModels {
    const float* __restrict__ lambda;
    const float* __restrict__ mu;
    const float* __restrict__ vp;
    const float* __restrict__ vs;
    const float* __restrict__ rho;
    const float* __restrict__ dvp;
    const float* __restrict__ dvs;
    const float* __restrict__ drho;
};

template<int Order>
__global__ void __launch_bounds__(256) elastic_born_stress_kernel(
    ElasticWavefieldPointer wf,
    ElasticWavefieldPointer sf,
    ElasticBornModels mdl,
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
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int bidx = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    auto g = sf.offset(b, spatial_size);

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    float evx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (g.vx, ix, iz, grad_ctx, solver, true, true);
    float evz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(g.vz, ix, iz, grad_ctx, solver, true, true);
    float evx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (g.vx, ix, iz, grad_ctx, solver, false);
    float evz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (g.vz, ix, iz, grad_ctx, solver, false);

    const float lam = mdl.lambda[bidx];
    const float mu_ = mdl.mu[bidx];
    const float lam2mu = lam + 2.f * mu_;
    // Lame perturbation of (dvp, dvs, drho) at the background (vp, vs, rho).
    const float vp = mdl.vp[bidx], vs = mdl.vs[bidx], rho = mdl.rho[bidx];
    const float dvp = mdl.dvp[bidx], dvs = mdl.dvs[bidx], drho = mdl.drho[bidx];
    const float dlam = drho * (vp * vp - 2.f * vs * vs) + rho * (2.f * vp * dvp - 4.f * vs * dvs);
    const float dmu = drho * vs * vs + 2.f * rho * vs * dvs;
    const float dlam2mu = dlam + 2.f * dmu;

    // elastic2d's PML / interior split, free-surface rows kept on the full path.
    int z_lo = solver.fsLo(0) ? (solver.surface_row(ix) + 1) : (solver.padLo(0) + halo);
    int z_hi = solver.fsHi(0) ? elastic_z_bottom_surface_row(solver)
                              : (solver.nz - solver.padHi(0) - halo);
    int x_lo = solver.fsLo(2) ? (elastic_x_left_surface_col(solver) + 1)
                              : (solver.padLo(2) + halo);
    int x_hi = solver.fsHi(2) ? elastic_x_right_surface_col(solver)
                              : (solver.nx - solver.padHi(2) - halo);
    bool in_pml = (!solver.cut_x_lo() && ix < x_lo) ||
                  (!solver.cut_x_hi() && ix >= x_hi) ||
                  (!solver.cut_z_lo() && iz < z_lo) ||
                  (!solver.cut_z_hi() && iz >= z_hi);

    float az = 0.f, bz = 0.f, azh = 0.f, bzh = 0.f;
    float ax = 0.f, bx = 0.f, axh = 0.f, bxh = 0.f;
    long xi = 0, zi = 0;
    bool st_x = false, st_z = false;
    if (in_pml) {
        az = cpml.az[iz]; bz = cpml.bz[iz]; azh = cpml.azh[iz]; bzh = cpml.bzh[iz];
        ax = cpml.ax[ix]; bx = cpml.bx[ix]; axh = cpml.axh[ix]; bxh = cpml.bxh[ix];
        xi = solver.aux_rd_x2(iz, ix);
        zi = solver.aux_rd_z2(iz, ix);
        st_x = solver.aux_x.stored(ix);
        st_z = solver.aux_z.stored(iz);

        float m;
        m = az * f.m_vzz[zi] + bz * dvz_dz; if (st_z) f.m_vzz[zi] = m; dvz_dz += m;
        m = ax * f.m_vxx[xi] + bx * dvx_dx; if (st_x) f.m_vxx[xi] = m; dvx_dx += m;
        m = az * g.m_vzz[zi] + bz * evz_dz; if (st_z) g.m_vzz[zi] = m; evz_dz += m;
        m = ax * g.m_vxx[xi] + bx * evx_dx; if (st_x) g.m_vxx[xi] = m; evx_dx += m;
    }

    f.sxx[idx] += solver.dt * (lam2mu * dvx_dx + lam * dvz_dz);
    f.szz[idx] += solver.dt * (lam2mu * dvz_dz + lam * dvx_dx);
    g.sxx[idx] += solver.dt * (lam2mu * evx_dx + lam * evz_dz);
    g.szz[idx] += solver.dt * (lam2mu * evz_dz + lam * evx_dx);
    float born_xx = solver.dt * (dlam2mu * dvx_dx + dlam * dvz_dz);
    float born_zz = solver.dt * (dlam2mu * dvz_dz + dlam * dvx_dx);

    if (in_pml) {
        float m;
        m = azh * f.m_vxz[zi] + bzh * dvx_dz; if (st_z) f.m_vxz[zi] = m; dvx_dz += m;
        m = axh * f.m_vzx[xi] + bxh * dvz_dx; if (st_x) f.m_vzx[xi] = m; dvz_dx += m;
        m = azh * g.m_vxz[zi] + bzh * evx_dz; if (st_z) g.m_vxz[zi] = m; evx_dz += m;
        m = axh * g.m_vzx[xi] + bxh * evz_dx; if (st_x) g.m_vzx[xi] = m; evz_dx += m;
    }

    f.sxz[idx] += solver.dt * mu_ * (dvx_dz + dvz_dx);
    g.sxz[idx] += solver.dt * mu_ * (evx_dz + evz_dx) + solver.dt * dmu * (dvx_dz + dvz_dx);

    if (in_pml) {
        // elastic2d's per-face traction BC on both fields: Robertsson
        // corrections first, zeroing after.  The scattered surface row takes
        // the Born source of the Robertsson coefficient instead of the bulk one.
        bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
        bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
        if (is_z_fs || is_x_fs) {
            const float dc = (4.f * mu_ * mu_ * dlam
                              + 4.f * (lam * lam + 2.f * lam * mu_ + 2.f * mu_ * mu_) * dmu)
                             / (lam2mu * lam2mu);
            if (is_z_fs) {
                f.sxx[idx] += -solver.dt * lam * (lam / lam2mu * dvx_dx + dvz_dz);
                g.sxx[idx] += -solver.dt * lam * (lam / lam2mu * evx_dx + evz_dz);
                born_xx = solver.dt * dc * dvx_dx;
            }
            if (is_x_fs) {
                f.szz[idx] += -solver.dt * lam * (lam / lam2mu * dvz_dz + dvx_dx);
                g.szz[idx] += -solver.dt * lam * (lam / lam2mu * evz_dz + evx_dx);
                born_zz = solver.dt * dc * dvz_dz;
            }
        }
        g.sxx[idx] += born_xx;
        g.szz[idx] += born_zz;
        if (is_z_fs) { f.szz[idx] = 0.f; g.szz[idx] = 0.f; }
        if (is_x_fs) { f.sxx[idx] = 0.f; g.sxx[idx] = 0.f; }
        if (elastic_fs_zero_shear(solver, ix, iz)) { f.sxz[idx] = 0.f; g.sxz[idx] = 0.f; }
    } else {
        g.sxx[idx] += born_xx;
        g.szz[idx] += born_zz;
    }

    if (u_this) {
        float* u_this_b = u_this + b * spatial_size;
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }
}

using elastic_born_stress_kernel_fn = void (*)(ElasticWavefieldPointer, ElasticWavefieldPointer,
    ElasticBornModels, float*, SGradParam, ElasticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(elastic_born_stress_kernel_fn, elastic_born_stress_kernel_by_order);

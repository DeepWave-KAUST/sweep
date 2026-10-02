#pragma once
#include "../../launch/by_order.cuh"
// 2-D visco-elastic (GSLS) kernels.
//
// The velocity half-step is Elastic's, unchanged (elastic_velocity_kernel), and
// so are the stencil transposes of the adjoint (elastic_stress_adjoint_apply,
// elastic_velocity_adjoint_prepare/apply): the attenuation only changes the
// pointwise stress update, whose forward and exact transpose live here.
// Every kernel defined in this header carries the ``visco_elastic2d_`` prefix;
// like elastic2d's, they are instantiated once, in this directory's kernels.cu.
//
// Forward stress update per cell (strain rates e = CPML-corrected dv/dx):
//   r+_l = a_l r_l - c_l F_l(e),   F_xx = P_l (exx+ezz) - 2 M_l ezz,
//                                  F_zz = P_l (exx+ezz) - 2 M_l exx,
//                                  F_xz = M_l exz
//   s+   = s + dt (C_U e) + dt/2 sum_l (r+_l + r_l)
// with P_l = pi_R tau_l(Qp), M_l = mu_R tau_l(Qs).  On a free-surface row the
// normal strain rate is solved so that the normal traction stays zero through
// the memory update (z faces first from the raw dvx/dx, then x faces); the
// solved value is written to the ``fs_strain`` strip for the gradient.
//
// Adjoint sign convention: as elastic2d, the STRESS multipliers (and here the
// memory-variable multipliers) carry the opposite sign of the true adjoint
// state, the velocity multipliers the true sign.

#include <cuda.h>
#include <cuda_runtime.h>
#include "../elastic2d/kernels.cuh"

#define VISCO_ELASTIC2D_MAX_SLS 4

// kernel<order> through its table in kernels.cu (launch/by_order.cuh).
#define LAUNCH_VISCO_ELASTIC2D(kernel, order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(kernel##_by_order, order, grid, block, __VA_ARGS__)

// Memory variables (full grid, batch-strided like the physical fields).
struct ViscoElastic2dMemory {
    float* rxx[VISCO_ELASTIC2D_MAX_SLS];
    float* rzz[VISCO_ELASTIC2D_MAX_SLS];
    float* rxz[VISCO_ELASTIC2D_MAX_SLS];
};

// Step coefficients: the unrelaxed Lame set and the per-mechanism relaxed
// strengths (all (B, nz, nx), batch-strided), plus the trapezoidal weights.
// lam2mu is an independent input (its gradient is returned separately and
// autograd sums it back into lam and mu).
struct ViscoElastic2dModel {
    const float* lam;
    const float* mu;
    const float* lam2mu;
    const float* P[VISCO_ELASTIC2D_MAX_SLS];
    const float* Mm[VISCO_ELASTIC2D_MAX_SLS];
    float a[VISCO_ELASTIC2D_MAX_SLS];
    float c[VISCO_ELASTIC2D_MAX_SLS];
    int L;
};

// Fused gradient imaging for one reverse step (all-null => off).
struct ViscoElastic2dGrad {
    const float* fvx;        // forward vx, vz at step it (after the velocity half-step)
    const float* fvz;
    const float* fvx_next;   // ... at step it + 1 (zeros past the end)
    const float* fvz_next;
    const float* fs_strain;  // (B, S) solved surface strain rates at step it
    const float* rho;
    float* g_rho;
    float* g_lam;
    float* g_mu;
    float* g_lam2mu;
    float* g_P[VISCO_ELASTIC2D_MAX_SLS];
    float* g_M[VISCO_ELASTIC2D_MAX_SLS];
};

// Surface strip: [z-low row (nx) | z-high row (nx) | x-low col (nz) | x-high col (nz)].
__host__ __device__ __forceinline__ long visco_elastic2d_strip_len(int nx, int nz)
{
    return 2L * nx + 2L * nz;
}

__device__ __forceinline__ long visco_elastic2d_strip_z(const SolverContext& solver, int ix, int iz)
{
    const int face = (solver.fsLo(0) && iz == solver.surface_row(ix)) ? 0 : 1;
    return (long)face * solver.nx + ix;
}

__device__ __forceinline__ long visco_elastic2d_strip_x(const SolverContext& solver, int ix, int iz)
{
    const int face = (solver.fsLo(2) && ix == elastic_x_left_surface_col(solver)) ? 0 : 1;
    return 2L * solver.nx + (long)face * solver.nz + iz;
}

// CPML band of the staggered velocity-derivative memories (as the elastic
// velocity kernel / adjoint prepare kernels): outside it every coefficient is
// exactly zero and the memories stay zero.
__device__ __forceinline__ bool visco_elastic2d_in_cpml(const SolverContext& solver, int ix, int iz, int halo)
{
    return (!solver.cut_x_lo() && ix < solver.padLo(2) + halo) ||
           (!solver.cut_x_hi() && ix >= solver.nx - solver.padHi(2) - halo) ||
           (!solver.cut_z_lo() && iz < solver.padLo(0) + halo) ||
           (!solver.cut_z_hi() && iz >= solver.nz - solver.padHi(0) - halo);
}

template<int Order>
__global__ void __launch_bounds__(256, 4) visco_elastic2d_stress_kernel(
    ElasticWavefieldPointer wf,
    ViscoElastic2dMemory mem,
    ViscoElastic2dModel m,
    float* __restrict__ u_this,     // (2, B, nz, nx) vx/vz snapshot, or null
    float* __restrict__ fs_strain,  // (B, S) surface strip, or null
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
    const long off = (long)b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    if (visco_elastic2d_in_cpml(solver, ix, iz, halo)) {
        // Slab-resident CPML memories: exactly elastic_stress_kernel's update.
        long xi = solver.aux_rd_x2(iz, ix);
        long zi = solver.aux_rd_z2(iz, ix);
        bool st_x = solver.aux_x.stored(ix);
        bool st_z = solver.aux_z.stored(iz);

        float m_vzz = cpml.az[iz] * f.m_vzz[zi] + cpml.bz[iz] * dvz_dz;
        if (st_z) f.m_vzz[zi] = m_vzz;
        dvz_dz += m_vzz;
        float m_vxx = cpml.ax[ix] * f.m_vxx[xi] + cpml.bx[ix] * dvx_dx;
        if (st_x) f.m_vxx[xi] = m_vxx;
        dvx_dx += m_vxx;
        float m_vxz = cpml.azh[iz] * f.m_vxz[zi] + cpml.bzh[iz] * dvx_dz;
        if (st_z) f.m_vxz[zi] = m_vxz;
        dvx_dz += m_vxz;
        float m_vzx = cpml.axh[ix] * f.m_vzx[xi] + cpml.bxh[ix] * dvz_dx;
        if (st_x) f.m_vzx[xi] = m_vzx;
        dvz_dx += m_vzx;
    }

    const int L = m.L;
    const float lam = m.lam[off];
    const float mu_ = m.mu[off];
    const float l2m = m.lam2mu[off];
    float P[VISCO_ELASTIC2D_MAX_SLS], Mm[VISCO_ELASTIC2D_MAX_SLS];
    float rxx[VISCO_ELASTIC2D_MAX_SLS], rzz[VISCO_ELASTIC2D_MAX_SLS], rxz[VISCO_ELASTIC2D_MAX_SLS];
    #pragma unroll
    for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
        if (l >= L) break;
        P[l] = m.P[l][off];
        Mm[l] = m.Mm[l][off];
        rxx[l] = mem.rxx[l][off];
        rzz[l] = mem.rzz[l][off];
        rxz[l] = mem.rxz[l][off];
    }

    float exx = dvx_dx;
    float ezz = dvz_dz;
    const float exz = dvx_dz + dvz_dx;

    const bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    const bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    if (is_z_fs || is_x_fs) {
        float sum_cp = 0.f, sum_cross = 0.f;
        #pragma unroll
        for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
            if (l >= L) break;
            sum_cp    += m.c[l] * P[l];
            sum_cross += m.c[l] * (P[l] - 2.f * Mm[l]);
        }
        const float den   = l2m - 0.5f * sum_cp;
        const float cross = 0.5f * sum_cross;
        const long sbase = (long)b * visco_elastic2d_strip_len(solver.nx, solver.nz);
        if (is_z_fs) {                       // szz = 0: solve dvz/dz
            float hist = 0.f;
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                hist += 0.5f * (1.f + m.a[l]) * rzz[l];
            }
            ezz = -(lam * exx + hist - cross * exx) / den;
            if (fs_strain) fs_strain[sbase + visco_elastic2d_strip_z(solver, ix, iz)] = ezz;
        }
        if (is_x_fs) {                       // sxx = 0: solve dvx/dx
            float hist = 0.f;
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                hist += 0.5f * (1.f + m.a[l]) * rxx[l];
            }
            exx = -(lam * ezz + hist - cross * ezz) / den;
            if (fs_strain) fs_strain[sbase + visco_elastic2d_strip_x(solver, ix, iz)] = exx;
        }
    }

    const float theta = exx + ezz;
    float sum_xx = 0.f, sum_zz = 0.f, sum_xz = 0.f;
    #pragma unroll
    for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
        if (l >= L) break;
        const float tp  = P[l] * theta;
        const float nxx = m.a[l] * rxx[l] - m.c[l] * (tp - 2.f * Mm[l] * ezz);
        const float nzz = m.a[l] * rzz[l] - m.c[l] * (tp - 2.f * Mm[l] * exx);
        const float nxz = m.a[l] * rxz[l] - m.c[l] * (Mm[l] * exz);
        sum_xx += nxx + rxx[l];
        sum_zz += nzz + rzz[l];
        sum_xz += nxz + rxz[l];
        mem.rxx[l][off] = nxx;
        mem.rzz[l][off] = nzz;
        mem.rxz[l][off] = nxz;
    }

    const float dt = solver.dt;
    float szz = f.szz[idx] + dt * (l2m * ezz + lam * exx);
    float sxx = f.sxx[idx] + dt * (l2m * exx + lam * ezz);
    float sxz = f.sxz[idx] + dt * mu_ * exz;
    szz = szz + (0.5f * dt) * sum_zz;
    sxx = sxx + (0.5f * dt) * sum_xx;
    sxz = sxz + (0.5f * dt) * sum_xz;

    // Traction BC: normal stress on each active face, shear on high faces only.
    if (is_z_fs) szz = 0.f;
    if (is_x_fs) sxx = 0.f;
    if (elastic_fs_zero_shear(solver, ix, iz)) sxz = 0.f;
    f.szz[idx] = szz;
    f.sxx[idx] = sxx;
    f.sxz[idx] = sxz;

    if (u_this) {
        const long comp_stride = (long)solver.B * spatial_size;
        u_this[off] = f.vx[idx];
        u_this[comp_stride + off] = f.vz[idx];
    }
}

using visco_elastic2d_stress_kernel_fn = void (*)(ElasticWavefieldPointer,
    ViscoElastic2dMemory, ViscoElastic2dModel, float*, float*, SGradParam,
    ElasticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(visco_elastic2d_stress_kernel_fn, visco_elastic2d_stress_kernel_by_order);

// Transpose of visco_elastic2d_stress_kernel (plus the FS traction zeroing),
// producing the adjoint strain-rate sources q* for elastic_stress_adjoint_apply
// and stepping the adjoint memory variables r^{it+1} -> r^{it}.  Own-cell only.
// With ``g`` set it also accumulates this reverse step's model gradient; the
// operands are the un-mutated post-source adjoint at entry, as elastic2d's
// fused imaging.
template<int Order>
__global__ void visco_elastic2d_stress_adjoint_prepare(
    ElasticWavefieldPointer wf,
    ViscoElastic2dMemory mem,
    ViscoElastic2dModel m,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qzz,
    float* __restrict__ qxz,
    float* __restrict__ qzx,
    SGradParam grad_ctx,
    ViscoElastic2dGrad g
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    const long off = (long)b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);

    const int L = m.L;
    const float dt  = solver.dt;
    const float hdt = 0.5f * dt;
    const float lam = m.lam[off];
    const float mu_ = m.mu[off];
    const float l2m = m.lam2mu[off];
    float P[VISCO_ELASTIC2D_MAX_SLS], Mm[VISCO_ELASTIC2D_MAX_SLS];
    #pragma unroll
    for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
        if (l >= L) break;
        P[l] = m.P[l][off];
        Mm[l] = m.Mm[l][off];
    }

    // Transpose of the traction zeroing (as elastic_stress_adjoint_prepare).
    const bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    const bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_sxz = f.sxz[idx];
    if (is_z_fs) { bar_szz = 0.f; f.szz[idx] = 0.f; }
    if (is_x_fs) { bar_sxx = 0.f; f.sxx[idx] = 0.f; }
    if (elastic_fs_zero_shear(solver, ix, iz)) { bar_sxz = 0.f; f.sxz[idx] = 0.f; }

    // Total multiplier of r^{it+1}: its own adjoint plus the dt/2 stress coupling.
    float rho_xx[VISCO_ELASTIC2D_MAX_SLS], rho_zz[VISCO_ELASTIC2D_MAX_SLS], rho_xz[VISCO_ELASTIC2D_MAX_SLS];
    float nxx[VISCO_ELASTIC2D_MAX_SLS], nzz[VISCO_ELASTIC2D_MAX_SLS], nxz[VISCO_ELASTIC2D_MAX_SLS];
    float bxx = dt * (l2m * bar_sxx + lam * bar_szz);
    float bzz = dt * (l2m * bar_szz + lam * bar_sxx);
    float bxz = dt * mu_ * bar_sxz;
    #pragma unroll
    for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
        if (l >= L) break;
        rho_xx[l] = mem.rxx[l][off] + hdt * bar_sxx;
        rho_zz[l] = mem.rzz[l][off] + hdt * bar_szz;
        rho_xz[l] = mem.rxz[l][off] + hdt * bar_sxz;
        const float s = rho_xx[l] + rho_zz[l];
        bxx -= m.c[l] * (P[l] * s - 2.f * Mm[l] * rho_zz[l]);
        bzz -= m.c[l] * (P[l] * s - 2.f * Mm[l] * rho_xx[l]);
        bxz -= m.c[l] * (Mm[l] * rho_xz[l]);
        nxx[l] = m.a[l] * rho_xx[l] + hdt * bar_sxx;
        nzz[l] = m.a[l] * rho_zz[l] + hdt * bar_szz;
        nxz[l] = m.a[l] * rho_xz[l] + hdt * bar_sxz;
    }

    // Transpose of the surface strain-rate solve, in reverse order (x, then z).
    // gx / gz: multipliers of the solved exx / ezz (for the gradient).
    float gx = 0.f, gz = 0.f, den = 1.f;
    if (is_z_fs || is_x_fs) {
        float sum_cp = 0.f, sum_cross = 0.f;
        #pragma unroll
        for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
            if (l >= L) break;
            sum_cp    += m.c[l] * P[l];
            sum_cross += m.c[l] * (P[l] - 2.f * Mm[l]);
        }
        den = l2m - 0.5f * sum_cp;
        const float k = -(lam - 0.5f * sum_cross) / den;   // d(solved)/d(other strain)
        if (is_x_fs) {
            gx = bxx;
            bzz += bxx * k;
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                nxx[l] += bxx * (-0.5f * (1.f + m.a[l]) / den);
            }
            bxx = 0.f;
        }
        if (is_z_fs) {
            gz = bzz;
            bxx += bzz * k;
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                nzz[l] += bzz * (-0.5f * (1.f + m.a[l]) / den);
            }
            bzz = 0.f;
        }
    }

    // --- fused gradient imaging ----------------------------------------------
    // True gradient = -(multiplier-form correlation) for every stress-side
    // coefficient (the multipliers carry the opposite sign); rho consumes the
    // adjoint velocity, which carries the true sign (as elastic2d).  Physical
    // box only, as elastic2d: EdgePadding.backward crops the rest, and the
    // prepared models are pointwise in the user's.
    if (g.g_lam != nullptr &&
        ix >= solver.phys_x0() && ix < solver.phys_x1() &&
        iz >= solver.phys_z0() && iz < solver.phys_z1()) {
        const float* fvx_b = g.fvx + (long)b * spatial_size;
        const float* fvz_b = g.fvz + (long)b * spatial_size;
        const float fvx_x = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(fvx_b, ix, iz, grad_ctx, solver, true, true);
        const float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(fvz_b, ix, iz, grad_ctx, solver, true, true);
        const float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (fvx_b, ix, iz, grad_ctx, solver, false);
        const float fvz_x = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>(fvz_b, ix, iz, grad_ctx, solver, false);

        const float exx_raw = fvx_x;
        const float exz = fvx_z + fvz_x;
        float exx_f = exx_raw, ezz_f = fvz_z, ezz_in_x = fvz_z;
        const long sbase = (long)b * visco_elastic2d_strip_len(solver.nx, solver.nz);
        if (is_z_fs) { ezz_f = g.fs_strain[sbase + visco_elastic2d_strip_z(solver, ix, iz)]; ezz_in_x = ezz_f; }
        if (is_x_fs) { exx_f = g.fs_strain[sbase + visco_elastic2d_strip_x(solver, ix, iz)]; }
        const float theta = exx_f + ezz_f;

        float G_l2m = dt * (bar_sxx * exx_f + bar_szz * ezz_f);
        float G_lam = dt * (bar_sxx * ezz_f + bar_szz * exx_f);
        const float G_mu = dt * bar_sxz * exz;
        float G_P[VISCO_ELASTIC2D_MAX_SLS], G_M[VISCO_ELASTIC2D_MAX_SLS];
        #pragma unroll
        for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
            if (l >= L) break;
            G_P[l] = -m.c[l] * ((rho_xx[l] + rho_zz[l]) * theta);
            G_M[l] = -m.c[l] * (-2.f * rho_xx[l] * ezz_f - 2.f * rho_zz[l] * exx_f + rho_xz[l] * exz);
        }
        // Material derivatives of the surface solve  e = -((lam - cross) e_in + H) / den.
        if (is_z_fs) {
            const float inv = 1.f / den;
            G_lam += gz * (-exx_raw * inv);
            G_l2m += gz * (-ezz_f * inv);
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                G_P[l] += gz * (0.5f * m.c[l] * (exx_raw + ezz_f) * inv);
                G_M[l] += gz * (-m.c[l] * exx_raw * inv);
            }
        }
        if (is_x_fs) {
            const float inv = 1.f / den;
            G_lam += gx * (-ezz_in_x * inv);
            G_l2m += gx * (-exx_f * inv);
            #pragma unroll
            for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
                if (l >= L) break;
                G_P[l] += gx * (0.5f * m.c[l] * (ezz_in_x + exx_f) * inv);
                G_M[l] += gx * (-m.c[l] * ezz_in_x * inv);
            }
        }
        g.g_lam2mu[off] -= G_l2m;
        g.g_lam[off]    -= G_lam;
        g.g_mu[off]     -= G_mu;
        #pragma unroll
        for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
            if (l >= L) break;
            g.g_P[l][off] -= G_P[l];
            g.g_M[l][off] -= G_M[l];
        }

        const float* fvx_next_b = g.fvx_next + (long)b * spatial_size;
        const float* fvz_next_b = g.fvz_next + (long)b * spatial_size;
        g.g_rho[off] += (f.vx[idx] * (fvx_b[idx] - fvx_next_b[idx]) +
                         f.vz[idx] * (fvz_b[idx] - fvz_next_b[idx])) / g.rho[off];
    }
    // -------------------------------------------------------------------------

    #pragma unroll
    for (int l = 0; l < VISCO_ELASTIC2D_MAX_SLS; ++l) {
        if (l >= L) break;
        mem.rxx[l][off] = nxx[l];
        mem.rzz[l][off] = nzz[l];
        mem.rxz[l][off] = nxz[l];
    }

    // Transpose of the CPML accumulation (elastic_stress_adjoint_prepare).
    float* qxx_b = qxx + (long)b * spatial_size;
    float* qzz_b = qzz + (long)b * spatial_size;
    float* qxz_b = qxz + (long)b * spatial_size;
    float* qzx_b = qzx + (long)b * spatial_size;
    if (!visco_elastic2d_in_cpml(solver, ix, iz, halo)) {
        qxx_b[idx] = bxx;
        qzz_b[idx] = bzz;
        qxz_b[idx] = bxz;
        qzx_b[idx] = bxz;
        return;
    }
    long xi = solver.aux_rd_x2(iz, ix);
    long zi = solver.aux_rd_z2(iz, ix);
    const float tmp_vxx = f.m_vxx[xi] + bxx;
    const float tmp_vzz = f.m_vzz[zi] + bzz;
    const float tmp_vxz = f.m_vxz[zi] + bxz;
    const float tmp_vzx = f.m_vzx[xi] + bxz;
    qxx_b[idx] = bxx + cpml.bx[ix]  * tmp_vxx;
    qzz_b[idx] = bzz + cpml.bz[iz]  * tmp_vzz;
    qxz_b[idx] = bxz + cpml.bzh[iz] * tmp_vxz;
    qzx_b[idx] = bxz + cpml.bxh[ix] * tmp_vzx;
    if (solver.aux_x.stored(ix)) {
        f.m_vxx[xi] = cpml.ax[ix]  * tmp_vxx;
        f.m_vzx[xi] = cpml.axh[ix] * tmp_vzx;
    }
    if (solver.aux_z.stored(iz)) {
        f.m_vzz[zi] = cpml.az[iz]  * tmp_vzz;
        f.m_vxz[zi] = cpml.azh[iz] * tmp_vxz;
    }
}

using visco_elastic2d_stress_adjoint_prepare_fn = void (*)(ElasticWavefieldPointer,
    ViscoElastic2dMemory, ViscoElastic2dModel, ElasticCPMLPointer, SolverContext,
    float*, float*, float*, float*, SGradParam, ViscoElastic2dGrad);
SWEEP_BY_ORDER_DECL(visco_elastic2d_stress_adjoint_prepare_fn, visco_elastic2d_stress_adjoint_prepare_by_order);

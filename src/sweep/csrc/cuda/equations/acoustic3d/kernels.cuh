#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define ACOUSTIC3D(order, grid, block, ...)                                          \
    do {                                                                                    \
        if      ((order) == 2) acoustic_forward_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_forward_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_forward_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_forward_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_forward_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define ACOUSTIC3D_NOPML(order, grid, block, ...)                           \
    do {                                                                           \
        if      ((order) == 2) acoustic_nopml_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_nopml_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_nopml_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_nopml_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_nopml_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)




// FUSED exact 3D adjoint: ONE launch, no scratch (recompute g_* inline per tap),
// writes next psi/zeta to SEPARATE buffers (double-buffer) -> race-free + no
// 9-field scratch round-trip.  Caller swaps psi/zeta via swap_aux() each step.
#define ACOUSTIC3D_ADJOINT_FUSED(order, grid, block, ...)                                      \
    do {                                                                                        \
        if      ((order) == 2) acoustic_adjoint_fused_3d<2><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 4) acoustic_adjoint_fused_3d<4><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 6) acoustic_adjoint_fused_3d<6><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 8) acoustic_adjoint_fused_3d<8><<<grid, block>>>(__VA_ARGS__);      \
        else                   acoustic_adjoint_fused_3d<-1><<<grid, block>>>(__VA_ARGS__);     \
    } while (0)

// Pre-pass: clear air cells (above per-(iy,ix) surface row).  Launched
// BEFORE acoustic_forward_kernel_3d so the main kernel only reads (never
// writes) air cells in the same launch — eliminates intra-launch RAW
// races on PML aux fields (same fix as acoustic2d).  ``static`` so each
// .cu including this header gets its own copy (avoids link conflicts).
static __global__ void acoustic3d_air_clear_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;
    if (!solver.has_topo) return;
    if (iz >= solver.topo_rows[iy * solver.nx + ix]) return;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int idx = iz * stride_z + iy * stride_y + ix;
    auto f = wf.offset(b, spatial_size);
    f.u_next[idx] = 0.f;
    f.psix[idx] = 0.f; f.psiy[idx] = 0.f; f.psiz[idx] = 0.f;
    f.zetax[idx] = 0.f; f.zetay[idx] = 0.f; f.zetaz[idx] = 0.f;
    if (u_this) u_this[b * spatial_size + idx] = 0.f;
}

template<int Order>
__global__ void acoustic_forward_kernel_3d(
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
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);

    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);

    // Air cells were already cleared by acoustic3d_air_clear_kernel
    // (separate pre-pass).  Early-return without writes so stencil reads
    // of psix/psiy/psiz from neighbouring threads see the zeros that the
    // prior launch finished writing — no intra-launch RAW race.
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix]) {
        return;
    }

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    bool in_pml_x = (ix < x0) || (ix >= x1);
    bool in_pml_y = (iy < y0) || (iy >= y1);
    bool in_pml_z = (iz < z0) || (iz >= z1);

    // =========================================================
    // Laplace
    // =========================================================

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float w_sum = 0.f;

    if (!(in_pml_x || in_pml_y || in_pml_z)) {
        w_sum = lap_x + lap_y + lap_z;

        float u0 = f.u_now[idx];
        float v  = vp_b[idx];

        f.u_next[idx] =
            2.f * u0 -
            f.u_prev[idx] +
            (v * v) * solver.dt * solver.dt * w_sum;

        if (u_this_b != nullptr)
            u_this_b[idx] = (v * v) * w_sum;

        return;
    }

    // =========================================================
    // Gradients
    // =========================================================

    // =========================================================
    // X direction
    // =========================================================
    if (in_pml_x) {
        float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
        float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
        float ax_ = cpml.ax[ix];
        float bx_ = cpml.bx[ix];
        float dbxdx_ = cpml.dbxdx[ix];
        float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);

        float tmpx = ((1.f + bx_) * lap_x + dbxdx_ * dudx)
                     + ax_ * dpsixdx + daxdx * f.psix[idx];

        w_sum += (1.f + bx_) * tmpx + ax_ * f.zetax[idx];

        (f.psixn ? f.psixn : f.psix)[idx]  = bx_ * dudx + ax_ * f.psix[idx];
        f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];
    } else {
        w_sum += lap_x;
    }

    // =========================================================
    // Y direction
    // =========================================================
    if (in_pml_y) {
        float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
        float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
        float ay_ = cpml.ay[iy];
        float by_ = cpml.by[iy];
        float dbydy_ = cpml.dbydy[iy];
        float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);

        float tmpy = ((1.f + by_) * lap_y + dbydy_ * dudy)
                     + ay_ * dpsiydy + daydy * f.psiy[idx];

        w_sum += (1.f + by_) * tmpy + ay_ * f.zetay[idx];

        (f.psiyn ? f.psiyn : f.psiy)[idx]  = by_ * dudy + ay_ * f.psiy[idx];
        f.zetay[idx] = by_ * tmpy + ay_ * f.zetay[idx];
    } else {
        w_sum += lap_y;
    }

    // =========================================================
    // Z direction
    // =========================================================
    if (in_pml_z) {
        float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
        float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);
        float az_ = cpml.az[iz];
        float bz_ = cpml.bz[iz];
        float dbzdz_ = cpml.dbzdz[iz];
        float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

        float tmpz = ((1.f + bz_) * lap_z + dbzdz_ * dudz)
                     + az_ * dpsizdz + dazdz * f.psiz[idx];

        w_sum += (1.f + bz_) * tmpz + az_ * f.zetaz[idx];

        (f.psizn ? f.psizn : f.psiz)[idx]  = bz_ * dudz + az_ * f.psiz[idx];
        f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];
    } else {
        w_sum += lap_z;
    }

    // =========================================================
    // Time update
    // =========================================================

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_y + lap_z);
}






// ---------------------------------------------------------------------------
// FUSED exact 3D discrete adjoint — single kernel, NO scratch.  3D twin of the
// 2D acoustic2nd_adjoint_fused: recompute the per-cell g_* (g_l/g_g/g_q) INLINE
// at each stencil tap from the current adjoint fields, instead of materialising
// 9 scratch fields in global memory.  Reads current lambda(u_now)/psi/zeta at
// idx+neighbours; writes next-step lambda (u_next) and the next-step aux into
// SEPARATE out-buffers (psi*_out/zeta*_out) -> read-old/write-new, exactly like
// the forward step, so it is race-free even though aux is read at neighbours.
// Caller must double-buffer the aux and swap_aux() each step.  Transpose math is
// identical to acoustic_adjoint_{prepare,apply}_3d (verified): Lx^T=Lx (sym),
// Dx^T=-Dx (antisym, folded as +np-nm then subtracted).
template<int Order>
__global__ void acoustic_adjoint_fused_3d(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ psix_out,
    float* __restrict__ psiy_out,
    float* __restrict__ psiz_out,
    float* __restrict__ zetax_out,
    float* __restrict__ zetay_out,
    float* __restrict__ zetaz_out,
    const float* __restrict__ grad_forward_img,  // u_forward[it+1] (vp^2 Lap u), or nullptr
    float* __restrict__ grad_out                 // vp gradient accumulator, or nullptr
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) return;
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix]) return;

    int sy = solver.nx;                       // y-stride
    int sz = solver.nx * solver.ny;           // z-stride
    int sp = sz * solver.nz;
    int idx  = iz * sz + iy * sy + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;          // [c0(center), c1, ..., cM]
    const float* gc = grad_ctx_x.coeff;       // antisymmetric grad coeffs
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdy2 = 1.0f / (lap_ctx.dy * lap_ctx.dy);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    float invdx  = 1.0f / lap_ctx.dx;
    float invdy  = 1.0f / lap_ctx.dy;
    float invdz  = 1.0f / lap_ctx.dz;

    float gun = f.u_now[idx];

    // FUSED vp-gradient imaging (lagged one step): at kernel entry u_now holds the
    // post-source adjoint field of step it+1, so accumulate the imaging correlation
    // for that step here instead of a separate calculate_grad_3d pass.  Bit-identical
    // to calculate_grad_3d (same operands, same op order); halo/air cells skipped
    // above carry adjoint==0, so omitting their imaging contributes exactly 0.
    if (grad_out != nullptr)
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];

    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw  = c0c * gun;                     // c * lambda at idx

    #define GW_AT(n) ( vpb[n]*vpb[n]*dt2 * f.u_now[n] )

    // Interior fast-path (aux all 0, g_l*=gw): 2L - L_prev + Lap(gw).
    int x0 = solver.phys_x0(), x1 = solver.phys_x1();
    int y0 = solver.phys_y0(), y1 = solver.phys_y1();
    int z0 = solver.phys_z0(), z1 = solver.phys_z1();
    bool pure_interior =
        (ix >= x0 + halo) && (ix < x1 - halo) &&
        (iy >= y0 + halo) && (iy < y1 - halo) &&
        (iz >= z0 + halo) && (iz < z1 - halo);
    if (pure_interior) {
        float lapx = -lc[0] * gw, lapy = -lc[0] * gw, lapz = -lc[0] * gw;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            lapx += lc[k] * (GW_AT(idx + k)      + GW_AT(idx - k));
            lapy += lc[k] * (GW_AT(idx + k * sy) + GW_AT(idx - k * sy));
            lapz += lc[k] * (GW_AT(idx + k * sz) + GW_AT(idx - k * sz));
        }
        f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                      + lapx * invdx2 + lapy * invdy2 + lapz * invdz2;
        // Interior aux stays 0: both psi/zeta double-buffers are zero-allocated
        // and no path ever writes a non-zero into the deep interior, so the
        // out-buffers are already 0 here — skip the 6 redundant full-grid zero
        // writes (mirrors the forward kernel's interior early-return; the only
        // readers are PML-band stencils, which read the *input* psi=0).
        return;
    }

    float ax_ = cpml.ax[ix], ay_ = cpml.ay[iy], az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix], by_ = cpml.by[iy], bz_ = cpml.bz[iz];
    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw;
    float gtmpy0 = by_ * f.zetay[idx] + (1.0f + by_) * gw;
    float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw;

    #define GTMP_X(n, nix) ( cpml.bx[nix]*f.zetax[n] + (1.0f+cpml.bx[nix])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Y(n, niy) ( cpml.by[niy]*f.zetay[n] + (1.0f+cpml.by[niy])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Z(n, niz) ( cpml.bz[niz]*f.zetaz[n] + (1.0f+cpml.bz[niz])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )

    // X-direction
    float Lx_glx = -lc[0] * ((1.0f + bx_) * gtmpx0);
    float Dx_ggx = 0.0f, Dx_gqx = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
        float tp = GTMP_X(np, nixp), tm = GTMP_X(nm, nixm);
        float bxp = cpml.bx[nixp], bxm = cpml.bx[nixm];
        Lx_glx += lc[k] * ((1.0f + bxp) * tp + (1.0f + bxm) * tm);
        Dx_ggx += gc[k] * ((bxp * f.psix[np] + cpml.dbxdx[nixp] * tp)
                         - (bxm * f.psix[nm] + cpml.dbxdx[nixm] * tm));
        Dx_gqx += gc[k] * (cpml.ax[nixp] * tp - cpml.ax[nixm] * tm);
    }
    Lx_glx *= invdx2; Dx_ggx *= invdx; Dx_gqx *= invdx;

    // Y-direction
    float Ly_gly = -lc[0] * ((1.0f + by_) * gtmpy0);
    float Dy_ggy = 0.0f, Dy_gqy = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k * sy, nm = idx - k * sy, niyp = iy + k, niym = iy - k;
        float tp = GTMP_Y(np, niyp), tm = GTMP_Y(nm, niym);
        float byp = cpml.by[niyp], bym = cpml.by[niym];
        Ly_gly += lc[k] * ((1.0f + byp) * tp + (1.0f + bym) * tm);
        Dy_ggy += gc[k] * ((byp * f.psiy[np] + cpml.dbydy[niyp] * tp)
                         - (bym * f.psiy[nm] + cpml.dbydy[niym] * tm));
        Dy_gqy += gc[k] * (cpml.ay[niyp] * tp - cpml.ay[niym] * tm);
    }
    Ly_gly *= invdy2; Dy_ggy *= invdy; Dy_gqy *= invdy;

    // Z-direction
    float Lz_glz = -lc[0] * ((1.0f + bz_) * gtmpz0);
    float Dz_ggz = 0.0f, Dz_gqz = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
        float tp = GTMP_Z(np, nizp), tm = GTMP_Z(nm, nizm);
        float bzp = cpml.bz[nizp], bzm = cpml.bz[nizm];
        Lz_glz += lc[k] * ((1.0f + bzp) * tp + (1.0f + bzm) * tm);
        Dz_ggz += gc[k] * ((bzp * f.psiz[np] + cpml.dbzdz[nizp] * tp)
                         - (bzm * f.psiz[nm] + cpml.dbzdz[nizm] * tm));
        Dz_gqz += gc[k] * (cpml.az[nizp] * tp - cpml.az[nizm] * tm);
    }
    Lz_glz *= invdz2; Dz_ggz *= invdz; Dz_gqz *= invdz;

    f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                  + (Lx_glx + Ly_gly + Lz_glz - Dx_ggx - Dy_ggy - Dz_ggz);
    psix_out[oidx]  = ax_ * f.psix[idx] + daxdx * gtmpx0 - Dx_gqx;
    psiy_out[oidx]  = ay_ * f.psiy[idx] + daydy * gtmpy0 - Dy_gqy;
    psiz_out[oidx]  = az_ * f.psiz[idx] + dazdz * gtmpz0 - Dz_gqz;
    zetax_out[oidx] = ax_ * (f.zetax[idx] + gw);
    zetay_out[oidx] = ay_ * (f.zetay[idx] + gw);
    zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw);
    #undef GW_AT
    #undef GTMP_X
    #undef GTMP_Y
    #undef GTMP_Z
}


template<int Order>
__global__ void acoustic_nopml_3d(
    AcousticWavefieldPointer wf,
    
    float* __restrict__ u_this,
    const float* __restrict__ vp,

    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn > 0 ? solver.abcn + 2*M+1 : 2*M;

    int top_halo = solver.free_surface ? 2*M : halo;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo)
        return;
    
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    
    auto f = wf.offset(b, spatial_size);

    int idx = iz * stride_z + iy * stride_y + ix;

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    float lap_x, lap_y, lap_z;

    float w_sum = 0.0f;

    lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    w_sum = lap_x + lap_y + lap_z;

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.0f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;
}

__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt
);

__global__ void calculate_grad_utt_3d(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt
);

__global__ void accumulate_rtm_image_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B, int nx, int ny, int nz
);

__global__ void accumulate_adcig_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ adcig,           // (nlag, B, nz, ny, nx) runtime-padded
    int nlag, int max_lag,
    int B, int nx, int ny, int nz
);

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
);

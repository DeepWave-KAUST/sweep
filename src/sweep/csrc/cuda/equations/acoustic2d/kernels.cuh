#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

// Accessor returning (a · psi) at a stencil tap: a is a 1-D profile (neighbour
// stride 1 along the differencing axis), psi a 2-D field at linear index idx
// with the given stride.  Feeding this to centered_gradient_stencil computes the
// FUSED product derivative  d(a·psi)/dx  in a single stencil — matching the
// eager reference grad_op(a*psi) (and deepwave's DIFFX1(AX_PSIX)).  Replaces the
// product-rule split  a·d(psi) + d(a)·psi, which needs d(psi) AND d(a) held live
// at once -> higher register pressure.
struct AProductAccessor {
    const float* __restrict__ a;
    const float* __restrict__ psi;
    int a_pos;
    int idx;
    int psi_stride;
    __device__ __forceinline__ float operator()(int offset) const {
        return a[a_pos + offset] * psi[idx + offset * psi_stride];
    }
};

template<int Order>
__device__ __forceinline__
float fused_d_aPsi(const float* __restrict__ a, const float* __restrict__ psi,
                   int a_pos, int idx, int psi_stride,
                   int M, const float* __restrict__ coeff, float h)
{
    return centered_gradient_stencil<Order>(
        AProductAccessor{a, psi, a_pos, idx, psi_stride}, M, coeff, h);
}

#define ACOUSTIC2D(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic2nd<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic2nd<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic2nd<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic2nd<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic2nd<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)




// FUSED exact adjoint: ONE launch, no scratch (recompute g_* inline at each tap),
// writes next psi/zeta to SEPARATE buffers (double-buffer) -> race-free + ~forward
// bandwidth.  Caller swaps psi/zeta via swap_aux() each step.
#define ACOUSTIC2D_ADJOINT_FUSED(order, grid, block, ...)                                     \
    do {                                                                                       \
        if      ((order) == 2) acoustic2nd_adjoint_fused<2><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 4) acoustic2nd_adjoint_fused<4><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 6) acoustic2nd_adjoint_fused<6><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 8) acoustic2nd_adjoint_fused<8><<<grid, block>>>(__VA_ARGS__);     \
        else                   acoustic2nd_adjoint_fused<-1><<<grid, block>>>(__VA_ARGS__);    \
    } while (0)

// Pre-pass: clear air cells (above per-column topo surface).  Launched
// BEFORE acoustic2nd so the main kernel can early-return on air cells
// without writing any aux field — eliminating intra-launch RAW races
// between air-zeroing and PML stencil reads.  See sweep VTI history.
static __global__ void acoustic2d_air_clear_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + solver.x_base;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.x_end() || iz >= solver.nz) return;
    if (!solver.has_topo) return;
    if (iz >= solver.topo_rows[ix]) return;
    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    f.u_next[idx] = 0.f;
    f.psix[idx] = 0.f; f.psiz[idx] = 0.f;
    f.zetax[idx] = 0.f; f.zetaz[idx] = 0.f;
    if (u_this) u_this[b * spatial_size + idx] = 0.f;
}

#define ACOUSTIC2D_NOPML(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic2nd_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic2nd_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic2nd_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic2nd_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic2nd_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void acoustic2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + solver.x_base;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.x_end() || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    // Irregular free-surface topography (vacuum staircase / image method):
    // any cell strictly above the per-column surface row is air.  Mirror
    // Python's ``zero_above_topo`` — clear the wavefield and any CPML aux
    // fields, then skip the FD update.  Solid cells just below the surface
    // see these zeros through the stencil, which is what reproduces the
    // ``p=0`` boundary condition.
    // Air cells were already cleared by acoustic2d_air_clear_kernel (a
    // separate pre-pass launch).  Just early-return here — no writes —
    // so this kernel only stencil-reads psix/psiz that the prior launch
    // finished writing.
    if (solver.has_topo && iz < solver.topo_rows[ix]) {
        return;
    }

    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    // Position-based PML / interior split. ax/bx/dbxdx coefficient arrays are
    // exactly zero outside the PML band, and the centered gradient of those
    // arrays vanishes once the stencil clears the band. Skipping the full PML
    // update there is bit-equivalent and avoids ~8 aux-field loads/stores per
    // cell. The check is warp-coherent (same outcome for 32 consecutive ix
    // values), so warps diverge only at the abcn boundary.
    //
    // Use the cut-aware physical bounds (phys_x0/x1/z0/z1): a DD cut face
    // carries only the M halo with zero PML coefficients, so its interior must
    // take the fast path too — otherwise the algebraically-equal zero-coeff
    // PML branch reorders the FMAs and seeds ulp drift. For a single domain
    // (cut_mask == 0) every bound collapses to the legacy abcn+M / free-surface
    // form, bit-for-bit.
    bool in_pml = (ix < solver.phys_x0()) || (ix >= solver.phys_x1()) ||
                  (iz < solver.phys_z0()) || (iz >= solver.phys_z1());

    if (!in_pml) {
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] +
                        v2_dt2 * (lap_x + lap_z);
        if (u_this_b != nullptr)
            u_this_b[idx] = (v * v) * (lap_x + lap_z);
        return;
    }

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float w_sum = 0.0f;

    float dudz     = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
    float dudx     = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    // FUSED d(a·psi) (single stencil) instead of the product-rule split
    // a·d(psi) + d(a)·psi.  Bit-matches the eager reference grad_op(a*psi);
    // drops d(psi) and d(a) from the live set -> fewer registers (deepwave-style).
    float daipsiz_dz = fused_d_aPsi<Order>(cpml.az, f.psiz, iz, idx, grad_ctx.sz, halo, grad_ctx.coeff, grad_ctx.dz);
    float daipxix_dx = fused_d_aPsi<Order>(cpml.ax, f.psix, ix, idx, grad_ctx.sx, halo, grad_ctx.coeff, grad_ctx.dx);

    // X direction.  Race-free: read psix at neighbours (above), write the NEXT
    // psix to a separate buffer (psixn) when double-buffering; fall back to
    // in-place when psixn is null (equations that have not opted in).
    float tmpx = ((1.0f+bx_)*lap_x + dbxdx_*dudx) + daipxix_dx;
    w_sum += (1.0f+bx_) * tmpx + ax_ * f.zetax[idx];
    (f.psixn ? f.psixn : f.psix)[idx]  = bx_ * dudx + ax_ * f.psix[idx];
    f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];

    // Z direction
    float tmpz = ((1.0f+bz_)*lap_z + dbzdz_*dudz) + daipsiz_dz;
    w_sum += (1.0f+bz_) * tmpz + az_ * f.zetaz[idx];
    (f.psizn ? f.psizn : f.psiz)[idx]  = bz_ * dudz + az_ * f.psiz[idx];
    f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        v2_dt2 * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_z);
}





// ---------------------------------------------------------------------------
// FUSED exact discrete adjoint — single kernel, NO scratch.
// Same math as prepare/apply but the per-cell scratch (g_l/g_g/g_q) is
// recomputed INLINE at each stencil tap from the *current* adjoint fields,
// instead of being materialised in global memory.  Reads current
// lambda(u_now)/psi/zeta at idx+neighbours, writes next-step lambda (u_next)
// and the next-step aux into SEPARATE buffers (psi*_out/zeta*_out) -> read-old
// / write-new, exactly like the forward time-step, so there is no race even
// though aux is now read at neighbours.  Caller must double-buffer the aux and
// swap (psi*_out/zeta*_out) <-> (f.psi*/f.zeta*) each step.
// Each tap is read once and feeds all three transposed stencils
// (Lx(g_lx), Dx(g_gx), Dx(g_qx)) -> bandwidth-optimal (no 6/9-field round-trip).
template<int Order>
__global__ void acoustic2nd_adjoint_fused(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ psix_out,
    float* __restrict__ psiz_out,
    float* __restrict__ zetax_out,
    float* __restrict__ zetaz_out,
    const float* __restrict__ grad_forward_img,  // u_forward[it+1] (vp^2 Lap u), or nullptr
    float* __restrict__ grad_out                 // vp gradient accumulator, or nullptr
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo) return;
    if (solver.has_topo && iz < solver.topo_rows[ix]) return;

    int sp = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    int sz = solver.nx;                       // z-stride
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;          // [c0(center), c1, ..., cM]
    const float* gc = grad_ctx_x.coeff;       // [_, c1, ..., cM]  (antisymmetric)
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    float invdx  = 1.0f / lap_ctx.dx;
    float invdz  = 1.0f / lap_ctx.dz;

    float gun = f.u_now[idx];

    // FUSED vp-gradient imaging (lagged one step): at kernel entry u_now holds the
    // post-source adjoint field of step it+1, so accumulate the imaging correlation
    // for that step here instead of a separate calculate_grad pass.  Bit-identical
    // to calculate_grad (same operands, same op order); halo/air cells skipped above
    // carry adjoint==0, so omitting their imaging contributes exactly 0.
    if (grad_out != nullptr)
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];

    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw  = c0c * gun;                     // c * lambda at idx

    // gw at a linear index n  (c[n]*lambda[n])
    #define GW_AT(n) ( vpb[n]*vpb[n]*dt2 * f.u_now[n] )

    // Interior fast-path: aux all 0, g_lx=g_lz=gw -> 2L - L_prev + Lap(gw).
    // On a DD cut side the bound collapses from abcn + 2*halo to halo: a
    // cut clears the global PML band, so cut-adjacent cells are genuine
    // interior cells and must take the SAME fast path (same FP expression)
    // as the matching single-domain cells — taps read lambda from the
    // exchanged M-halo and vp from the tile model's halo slice.
    bool pure_interior =
        (ix >= (solver.cut_x_lo() ? halo : solver.abcn + 2 * halo)) &&
        (ix < solver.nx - (solver.cut_x_hi() ? halo : solver.abcn + 2 * halo)) &&
        (iz >= (solver.cut_z_lo() ? halo
                : (solver.free_surface ? 0 : solver.abcn) + 2 * halo)) &&
        (iz < solver.nz - (solver.cut_z_hi() ? halo : solver.abcn + 2 * halo));
    if (pure_interior) {
        float lapx = -lc[0] * gw, lapz = -lc[0] * gw;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            lapx += lc[k] * (GW_AT(idx + k)      + GW_AT(idx - k));
            lapz += lc[k] * (GW_AT(idx + k * sz) + GW_AT(idx - k * sz));
        }
        f.u_next[idx] = 2.0f * gun - f.u_prev[idx] + lapx * invdx2 + lapz * invdz2;
        // Interior aux stays 0: both psi/zeta double-buffers are zero-allocated
        // and no path ever writes a non-zero into the deep interior, so the
        // out-buffers are already 0 here — skip the 4 redundant full-grid zero
        // writes (mirrors the forward kernel's interior early-return; the only
        // readers are PML-band stencils, which read the *input* psi=0).
        return;
    }

    float ax_ = cpml.ax[ix], az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix], bz_ = cpml.bz[iz];
    // Transpose of the FUSED forward term daipxix = d(ax·psix)/dx.  Its exact
    // transpose w.r.t. psix is  -ax·d(gtmp)/dx  (single antisymmetric stencil of
    // gtmp, ax pulled to the centre) — NOT the product-rule pair
    // -d(ax·gtmp)/dx + daxdx·gtmp the old split forward needed.  So daxdx/dazdz
    // are no longer required, and Dx_gqx/Dz_gqz below carry d(gtmp) (un-scaled).

    float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw;
    float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw;

    // local g_tmp / g_l / g_g / g_q at a tap, given the 1D PML index for that axis
    #define GTMP_X(n, nix) ( cpml.bx[nix]*f.zetax[n] + (1.0f+cpml.bx[nix])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Z(n, niz) ( cpml.bz[niz]*f.zetaz[n] + (1.0f+cpml.bz[niz])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )

    // X-direction: Lx(g_lx), Dx(g_gx), Dx(g_qx); one read per tap.
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
        Dx_gqx += gc[k] * (tp - tm);   // d(gtmp)/dx (ax applied at psix_out)
    }
    Lx_glx *= invdx2; Dx_ggx *= invdx; Dx_gqx *= invdx;

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
        Dz_gqz += gc[k] * (tp - tm);   // d(gtmp)/dz (az applied at psiz_out)
    }
    Lz_glz *= invdz2; Dz_ggz *= invdz; Dz_gqz *= invdz;

    f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                  + (Lx_glx + Lz_glz - Dx_ggx - Dz_ggz);
    psix_out[oidx]  = ax_ * (f.psix[idx] - Dx_gqx);
    psiz_out[oidx]  = az_ * (f.psiz[idx] - Dz_gqz);
    zetax_out[oidx] = ax_ * (f.zetax[idx] + gw);
    zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw);
    #undef GW_AT
    #undef GTMP_X
    #undef GTMP_Z
}

template<int Order>
__global__ void acoustic2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    // Per-side exclusion bands.  On a non-cut side keep the legacy width
    // (PML band + strip + halo, or 2M without PML); on a DD cut side the
    // band collapses to the stencil halo M — the cut-adjacent cells are
    // computed by plain reverse leapfrog reading the per-step exchanged
    // M-halo of u_now (no PML there; restore skips the cut-face strip).
    int wide = solver.abcn > 0 ? solver.abcn + 2*M+1 : 2*M;
    int hx_lo = solver.cut_x_lo() ? M : wide;
    int hx_hi = solver.cut_x_hi() ? M : wide;
    int hz_lo = solver.cut_z_lo() ? M : (solver.free_surface ? 2*M : wide);
    int hz_hi = solver.cut_z_hi() ? M : wide;
    if (ix < hx_lo || ix >= solver.nx - hx_hi ||
        iz < hz_lo || iz >= solver.nz - hz_hi)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* vp_b     = vp     + b * spatial_size;

    float w_sum = 0.0f;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    w_sum = lap_x + lap_z;

    float v  = vp_b[idx];

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

}

__global__ void calculate_grad(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt
);

__global__ void calculate_grad_utt(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt
);

__global__ void accumulate_rtm_image_2d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int nx, int nz
);

__global__ void accumulate_source_grad_2d(
    const float* __restrict__ u_backward,   // (B, nz, nx)
    float* __restrict__ grad_source,        // (B, nsrc, nt)
    const int* __restrict__ sources_loc,    // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);

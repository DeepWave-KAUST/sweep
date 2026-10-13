// 3-D VRZ adjoint step as two launches: the pure-interior box and the CPML
// shell around it (same idea as acoustic3d/adjoint_split.cu).
//
//   interior  ∇²(C0·λ) − Σ_d ∂_d(C_d·λ), launched over that box only, so it
//             runs at its own register count instead of the CPML path's.
//   shell     the exact CPML transpose over the six slabs around the box, per
//             axis: an axis whose taps cannot touch a non-zero a/b/db takes the
//             interior form of that axis' terms and neither reads nor writes
//             that axis' ψ*/ζ*.
//
// Per-axis exactness.  Along axis d the CPML transpose reads b_d, and — at taps
// whose forward ran its CPML branch (gated by in_pml of the tap cell, i.e.
// also by the cell's band membership on the OTHER axes) — a_d and db_d.  b_d
// is exactly 0 off-band and a_d is masked to the band, but db_d =
// np.gradient(b_d) leaks one cell past the band edge into the physical box
// (equations/pml.py).  A cell whose axis-d taps stay at least one cell inside
// that edge, i.e. i_d in [phys_lo + M + 1, phys_hi - M - 1), therefore sees
// a = b = db = 0 at every tap on that axis, whatever the other axes say: its
// axis-d terms are the interior ones, and its axis-d aux outputs are 0 (a = 0
// at the cell, da = 0 and every a-tap 0), which is what the zero-allocated
// double buffers already hold.  A DD cut side has no band, so the bound there
// collapses to the halo.
//
// Blocks are 32x1x8, deep in z, so the z-taps of a block's cells are reused in
// L1; the x slabs, a few cells wide, take 8x4x8.
#include "kernels.cuh"
#include "../../launch/box.cuh"
#include "../../../core/check.h"

namespace {

using fdtd::Box;
struct Region { int x0, y0, z0, nx, ny, nz; };   // lo corner + extent: a shell slab

// --------------------------------------------------------------------------
// interior
// --------------------------------------------------------------------------
template<int Order>
__global__ void vrz3d_adj_interior(
    AcousticWavefieldPointer wf,
    const float* __restrict__ C0, const float* __restrict__ Cx,
    const float* __restrict__ Cy, const float* __restrict__ Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;

    const int sy = solver.nx;
    const int sz = solver.nx * solver.ny;
    const int spatial_size = solver.nx * solver.ny * solver.nz;
    const int idx = iz * sz + iy * sy + ix;

    auto f = wf.offset(b, spatial_size);
    const float* C0_b = C0 + b * spatial_size;
    const float* Cx_b = Cx + b * spatial_size;
    const float* Cy_b = Cy + b * spatial_size;
    const float* Cz_b = Cz + b * spatial_size;
    const float* lam = f.u_now;
    const float dt2 = solver.dt * solver.dt;

    VrzProductAccessor a0x{C0_b, lam, idx, 1};
    VrzProductAccessor a0y{C0_b, lam, idx, sy};
    VrzProductAccessor a0z{C0_b, lam, idx, sz};
    float lap_x = centered_laplace1d_stencil<Order>(a0x, lap_ctx.dx, solver.M, lap_ctx.coeff);
    float lap_y = centered_laplace1d_stencil<Order>(a0y, lap_ctx.dy, solver.M, lap_ctx.coeff);
    float lap_z = centered_laplace1d_stencil<Order>(a0z, lap_ctx.dz, solver.M, lap_ctx.coeff);

    VrzProductAccessor axx{Cx_b, lam, idx, 1};
    VrzProductAccessor ayy{Cy_b, lam, idx, sy};
    VrzProductAccessor azz{Cz_b, lam, idx, sz};
    float gx = centered_gradient_stencil<Order>(axx, solver.M, grad_ctx.coeff, grad_ctx.dx);
    float gy = centered_gradient_stencil<Order>(ayy, solver.M, grad_ctx.coeff, grad_ctx.dy);
    float gz = centered_gradient_stencil<Order>(azz, solver.M, grad_ctx.coeff, grad_ctx.dz);

    float rhs = (lap_x + lap_y + lap_z) - gx - gy - gz;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * rhs;
}

// --------------------------------------------------------------------------
// CPML shell
// --------------------------------------------------------------------------
template<int Order>
__global__ void __launch_bounds__(256) vrz3d_adj_shell(
    AcousticWavefieldPointer wf,
    const float* __restrict__ C0, const float* __restrict__ Cx,
    const float* __restrict__ Cy, const float* __restrict__ Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver,
    Box inner, Box clear, Region reg,
    float* __restrict__ psix_out, float* __restrict__ psiy_out, float* __restrict__ psiz_out,
    float* __restrict__ zetax_out, float* __restrict__ zetay_out, float* __restrict__ zetaz_out)
{
    const int rx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ry = blockIdx.y * blockDim.y + threadIdx.y;
    const int izg = blockIdx.z * blockDim.z + threadIdx.z;
    const int b = izg / reg.nz;
    if (b >= solver.B || rx >= reg.nx || ry >= reg.ny) return;
    const int ix = reg.x0 + rx;
    const int iy = reg.y0 + ry;
    const int iz = reg.z0 + izg % reg.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) return;
    if (ix >= inner.xlo && ix < inner.xhi && iy >= inner.ylo && iy < inner.yhi &&
        iz >= inner.zlo && iz < inner.zhi) return;            // the interior launch's
    const bool ax_on = ix < clear.xlo || ix >= clear.xhi;
    const bool ay_on = iy < clear.ylo || iy >= clear.yhi;
    const bool az_on = iz < clear.zlo || iz >= clear.zhi;

    const int sy = solver.nx;
    const int sz = solver.nx * solver.ny;
    const int spatial_size = solver.nx * solver.ny * solver.nz;
    const int idx = iz * sz + iy * sy + ix;
    const int oidx = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    const float* C0_b = C0 + b * spatial_size;
    const float* Cx_b = Cx + b * spatial_size;
    const float* Cy_b = Cy + b * spatial_size;
    const float* Cz_b = Cz + b * spatial_size;
    const float* lam = f.u_now;
    const float dt2 = solver.dt * solver.dt;

    const float* lc = lap_ctx.coeff;
    const float* gc = grad_ctx.coeff;
    const float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    const float invdy2 = 1.0f / (lap_ctx.dy * lap_ctx.dy);
    const float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    const float invdx = 1.0f / grad_ctx.dx;
    const float invdy = 1.0f / grad_ctx.dy;
    const float invdz = 1.0f / grad_ctx.dz;

    const bool px0 = solver.in_pml_x(ix, halo);
    const bool py0 = solver.in_pml_y(iy, halo);
    const bool pz0 = solver.in_pml_z(iz, halo);
    const bool own_pml = px0 || py0 || pz0;

    const float gw0 = C0_b[idx] * dt2 * lam[idx];

    #define VRZ_GW(n)          ( C0_b[n] * dt2 * lam[n] )
    #define VRZ_GTMP_X(n, nix) ( cpml.bx[nix] * f.zetax[n] + (1.0f + cpml.bx[nix]) * VRZ_GW(n) )
    #define VRZ_GTMP_Y(n, niy) ( cpml.by[niy] * f.zetay[n] + (1.0f + cpml.by[niy]) * VRZ_GW(n) )
    #define VRZ_GTMP_Z(n, niz) ( cpml.bz[niz] * f.zetaz[n] + (1.0f + cpml.bz[niz]) * VRZ_GW(n) )

    // X
    float Lx_gl, Dx_gg = 0.0f;
    if (ax_on) {
        const float ax_ = cpml.ax[ix], bx_ = cpml.bx[ix];
        const float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw0;
        Lx_gl = -lc[0] * ((1.0f + bx_) * gtmpx0);
        float Dx_gq = 0.0f;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
            bool Pp = py0 || pz0 || solver.in_pml_x(nixp, halo);
            bool Pm = py0 || pz0 || solver.in_pml_x(nixm, halo);
            float tp = VRZ_GTMP_X(np, nixp), tm = VRZ_GTMP_X(nm, nixm);
            float bp = cpml.bx[nixp], bm = cpml.bx[nixm];
            float dbp = Pp ? cpml.dbxdx[nixp] : 0.0f, dbm = Pm ? cpml.dbxdx[nixm] : 0.0f;
            float ap  = Pp ? cpml.ax[nixp]    : 0.0f, am  = Pm ? cpml.ax[nixm]    : 0.0f;
            Lx_gl += lc[k] * ((1.0f + bp) * tp + (1.0f + bm) * tm);
            Dx_gg += gc[k] * ((bp * f.psix[np] + dbp * tp + (1.0f + bp) * Cx_b[np] * dt2 * lam[np])
                            - (bm * f.psix[nm] + dbm * tm + (1.0f + bm) * Cx_b[nm] * dt2 * lam[nm]));
            Dx_gq += gc[k] * (ap * tp - am * tm);
        }
        Lx_gl *= invdx2; Dx_gg *= invdx; Dx_gq *= invdx;
        if (own_pml) {
            const float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
            psix_out[oidx]  = ax_ * (f.psix[idx] + Cx_b[idx] * dt2 * lam[idx]) + daxdx * gtmpx0 - Dx_gq;
            zetax_out[oidx] = ax_ * (f.zetax[idx] + gw0);
        }
    } else {
        Lx_gl = -lc[0] * gw0;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            Lx_gl += lc[k] * (VRZ_GW(idx + k) + VRZ_GW(idx - k));
            Dx_gg += gc[k] * (Cx_b[idx + k] * dt2 * lam[idx + k] - Cx_b[idx - k] * dt2 * lam[idx - k]);
        }
        Lx_gl *= invdx2; Dx_gg *= invdx;
    }

    // Y
    float Ly_gl, Dy_gg = 0.0f;
    if (ay_on) {
        const float ay_ = cpml.ay[iy], by_ = cpml.by[iy];
        const float gtmpy0 = by_ * f.zetay[idx] + (1.0f + by_) * gw0;
        Ly_gl = -lc[0] * ((1.0f + by_) * gtmpy0);
        float Dy_gq = 0.0f;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            int np = idx + k * sy, nm = idx - k * sy, niyp = iy + k, niym = iy - k;
            bool Pp = px0 || pz0 || solver.in_pml_y(niyp, halo);
            bool Pm = px0 || pz0 || solver.in_pml_y(niym, halo);
            float tp = VRZ_GTMP_Y(np, niyp), tm = VRZ_GTMP_Y(nm, niym);
            float bp = cpml.by[niyp], bm = cpml.by[niym];
            float dbp = Pp ? cpml.dbydy[niyp] : 0.0f, dbm = Pm ? cpml.dbydy[niym] : 0.0f;
            float ap  = Pp ? cpml.ay[niyp]    : 0.0f, am  = Pm ? cpml.ay[niym]    : 0.0f;
            Ly_gl += lc[k] * ((1.0f + bp) * tp + (1.0f + bm) * tm);
            Dy_gg += gc[k] * ((bp * f.psiy[np] + dbp * tp + (1.0f + bp) * Cy_b[np] * dt2 * lam[np])
                            - (bm * f.psiy[nm] + dbm * tm + (1.0f + bm) * Cy_b[nm] * dt2 * lam[nm]));
            Dy_gq += gc[k] * (ap * tp - am * tm);
        }
        Ly_gl *= invdy2; Dy_gg *= invdy; Dy_gq *= invdy;
        if (own_pml) {
            const float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
            psiy_out[oidx]  = ay_ * (f.psiy[idx] + Cy_b[idx] * dt2 * lam[idx]) + daydy * gtmpy0 - Dy_gq;
            zetay_out[oidx] = ay_ * (f.zetay[idx] + gw0);
        }
    } else {
        Ly_gl = -lc[0] * gw0;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            Ly_gl += lc[k] * (VRZ_GW(idx + k * sy) + VRZ_GW(idx - k * sy));
            Dy_gg += gc[k] * (Cy_b[idx + k * sy] * dt2 * lam[idx + k * sy]
                            - Cy_b[idx - k * sy] * dt2 * lam[idx - k * sy]);
        }
        Ly_gl *= invdy2; Dy_gg *= invdy;
    }

    // Z
    float Lz_gl, Dz_gg = 0.0f;
    if (az_on) {
        const float az_ = cpml.az[iz], bz_ = cpml.bz[iz];
        const float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw0;
        Lz_gl = -lc[0] * ((1.0f + bz_) * gtmpz0);
        float Dz_gq = 0.0f;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
            bool Pp = px0 || py0 || solver.in_pml_z(nizp, halo);
            bool Pm = px0 || py0 || solver.in_pml_z(nizm, halo);
            float tp = VRZ_GTMP_Z(np, nizp), tm = VRZ_GTMP_Z(nm, nizm);
            float bp = cpml.bz[nizp], bm = cpml.bz[nizm];
            float dbp = Pp ? cpml.dbzdz[nizp] : 0.0f, dbm = Pm ? cpml.dbzdz[nizm] : 0.0f;
            float ap  = Pp ? cpml.az[nizp]    : 0.0f, am  = Pm ? cpml.az[nizm]    : 0.0f;
            Lz_gl += lc[k] * ((1.0f + bp) * tp + (1.0f + bm) * tm);
            Dz_gg += gc[k] * ((bp * f.psiz[np] + dbp * tp + (1.0f + bp) * Cz_b[np] * dt2 * lam[np])
                            - (bm * f.psiz[nm] + dbm * tm + (1.0f + bm) * Cz_b[nm] * dt2 * lam[nm]));
            Dz_gq += gc[k] * (ap * tp - am * tm);
        }
        Lz_gl *= invdz2; Dz_gg *= invdz; Dz_gq *= invdz;
        if (own_pml) {
            const float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
            psiz_out[oidx]  = az_ * (f.psiz[idx] + Cz_b[idx] * dt2 * lam[idx]) + dazdz * gtmpz0 - Dz_gq;
            zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw0);
        }
    } else {
        Lz_gl = -lc[0] * gw0;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) {
            Lz_gl += lc[k] * (VRZ_GW(idx + k * sz) + VRZ_GW(idx - k * sz));
            Dz_gg += gc[k] * (Cz_b[idx + k * sz] * dt2 * lam[idx + k * sz]
                            - Cz_b[idx - k * sz] * dt2 * lam[idx - k * sz]);
        }
        Lz_gl *= invdz2; Dz_gg *= invdz;
    }
    #undef VRZ_GW
    #undef VRZ_GTMP_X
    #undef VRZ_GTMP_Y
    #undef VRZ_GTMP_Z

    f.u_next[idx] = 2.0f * lam[idx] - f.u_prev[idx]
                  + (Lx_gl + Ly_gl + Lz_gl - Dx_gg - Dy_gg - Dz_gg);
}

struct SplitArgs {
    AcousticWavefieldPointer wf;
    const float *C0, *Cx, *Cy, *Cz;
    LaplaceParam lap; GradParam g, gx, gy, gz;
    AcousticCPMLPointer cpml;
    const SolverContext* ctx;
    float* out[6];
};

template<int Order>
void launch_split(const SplitArgs& a)
{
    const SolverContext& ctx = *a.ctx;
    // the cells whose taps on every axis stay out of the CPML band and its aux
    const Box inner = fdtd::phys_box(ctx, ctx.M);
    // per axis, the range whose taps cannot reach a non-zero a/b/db (see above)
    const Box clear = fdtd::phys_box(ctx, ctx.M + 1);
    if (!fdtd::empty(inner)) {
        const auto L = fdtd::box_launch(inner, ctx.B, fdtd::kBoxBlock);
        vrz3d_adj_interior<Order><<<L.grid, L.block>>>(a.wf, a.C0, a.Cx, a.Cy, a.Cz,
                                                       a.lap, a.g, ctx, inner);
    }
    // the computed cells minus the interior box, as up to six disjoint slabs
    fdtd::for_each_slab(fdtd::computed_box(ctx), inner, [&](const Box& r, bool x_slab) {
        const auto L = fdtd::box_launch(r, ctx.B, x_slab ? fdtd::kNarrowBlock : fdtd::kBoxBlock);
        vrz3d_adj_shell<Order><<<L.grid, L.block>>>(a.wf, a.C0, a.Cx, a.Cy, a.Cz,
            a.lap, a.g, a.gx, a.gy, a.gz, a.cpml, ctx, inner, clear,
            Region{r.xlo, r.ylo, r.zlo, r.xhi - r.xlo, r.yhi - r.ylo, r.zhi - r.zlo},
            a.out[0], a.out[1], a.out[2], a.out[3], a.out[4], a.out[5]);
    });
}

}  // namespace

void launch_acoustic_vrz3d_adjoint(
    int order, AcousticWavefieldPointer wf,
    const float* C0, const float* Cx, const float* Cy, const float* Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    float* psix_out, float* psiy_out, float* psiz_out,
    float* zetax_out, float* zetay_out, float* zetaz_out)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic_vrz3d adjoint: an x-ranged context is not supported (the "
                "adjoint step always covers the whole tile)");
    SplitArgs a{wf, C0, Cx, Cy, Cz, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
                cpml, &ctx, {psix_out, psiy_out, psiz_out, zetax_out, zetay_out, zetaz_out}};
    switch (order) {
        case 2:  launch_split<2>(a); break;
        case 4:  launch_split<4>(a); break;
        case 6:  launch_split<6>(a); break;
        case 8:  launch_split<8>(a); break;
        default: launch_split<-1>(a); break;
    }
}

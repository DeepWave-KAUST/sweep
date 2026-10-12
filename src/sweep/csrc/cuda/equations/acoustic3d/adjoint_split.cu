// 3-D acoustic adjoint step as two launches: the pure-interior box and the PML
// shell around it.
//
// One kernel over the whole grid would carry the CPML transpose in every
// thread, so the interior would run at the PML path's register count and
// occupancy, and every shell cell would read and write all six psi/zeta fields
// although it sits in one axis' band.  Here:
//
//   interior  L_prev = 2 L - L_next + Lap(g),  g = vp^2 dt^2 L, one thread per
//             cell, over the pure-interior box only;
//   shell     the CPML transpose, per axis: an axis whose taps cannot touch a
//             non-zero a/b/db takes the plain Laplacian of g and leaves that
//             axis' psi/zeta untouched.  b is 0 off-band and a is masked to the
//             band, but db = np.gradient(b) leaks one cell into the physical
//             box (equations/pml.py; Acoustic3D has no sigma=0 buffer) and the
//             CPML path reads db at its taps -- so the per-axis test keeps an
//             M+1 margin, one more than the pure-interior box.  Out there the
//             CPML terms vanish and the untouched psi/zeta stay at the zero both
//             double buffers start from.
//
// The shell runs over the six slabs around the box (all the computed cells when
// the box is empty).  Blocks are 32x1x8, deep in z, so the z-taps of a block's
// cells are reused in L1; the x slabs, a few cells wide, take 8x4x8.
#include "kernels.cuh"
#include "../../launch/box.cuh"
#include "../../../core/check.h"

namespace {

using fdtd::Box;

// --------------------------------------------------------------------------
// interior, one thread per cell
// --------------------------------------------------------------------------
template<int Order>
__global__ void adj3d_interior_percell(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, Box box,
    const float* __restrict__ grad_forward_img, float* __restrict__ grad_out)
{
    constexpr bool is_runtime = (Order == -1);
    const int halo = is_runtime ? solver.M : Order / 2;
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix]) return;

    int sy = solver.nx, sz = solver.nx * solver.ny, sp = sz * solver.nz;
    int idx = iz * sz + iy * sy + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdy2 = 1.0f / (lap_ctx.dy * lap_ctx.dy);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);

    float gun = f.u_now[idx];
    if (grad_out != nullptr &&
        ix >= solver.phys_x0() && ix < solver.phys_x1() &&
        iy >= solver.phys_y0() && iy < solver.phys_y1() &&
        iz >= solver.phys_z0() && iz < solver.phys_z1())
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];
    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw = c0c * gun;
#define GW_AT(n) ( vpb[n]*vpb[n]*dt2 * f.u_now[n] )
    float lapx = -lc[0] * gw, lapy = -lc[0] * gw, lapz = -lc[0] * gw;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        lapx += lc[k] * (GW_AT(idx + k)      + GW_AT(idx - k));
        lapy += lc[k] * (GW_AT(idx + k * sy) + GW_AT(idx - k * sy));
        lapz += lc[k] * (GW_AT(idx + k * sz) + GW_AT(idx - k * sz));
    }
#undef GW_AT
    f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                  + lapx * invdx2 + lapy * invdy2 + lapz * invdz2;
}

// --------------------------------------------------------------------------
// PML shell: the CPML transpose, per axis
// --------------------------------------------------------------------------
// The launch covers region `reg` (lo corner + extent): one of the shell slabs.
struct Region { int x0, y0, z0, nx, ny, nz; };

template<int Order>
__global__ void __launch_bounds__(256) adj3d_shell(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    LaplaceParam lap_ctx, GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, Box box, Box clear, Region reg,
    float* __restrict__ psix_out, float* __restrict__ psiy_out, float* __restrict__ psiz_out,
    float* __restrict__ zetax_out, float* __restrict__ zetay_out, float* __restrict__ zetaz_out,
    const float* __restrict__ grad_forward_img, float* __restrict__ grad_out)
{
    int rx = blockIdx.x * blockDim.x + threadIdx.x;
    int ry = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b  = iz_global / reg.nz;
    if (b >= solver.B || rx >= reg.nx || ry >= reg.ny) return;
    int ix = reg.x0 + rx;
    int iy = reg.y0 + ry;
    int iz = reg.z0 + iz_global % reg.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) return;
    if (ix >= box.xlo && ix < box.xhi && iy >= box.ylo && iy < box.yhi &&
        iz >= box.zlo && iz < box.zhi) return;   // pure interior: the interior launch
    // per-axis: can this cell's taps on that axis touch a non-zero a/b/db?
    const bool inx = ix < clear.xlo || ix >= clear.xhi;
    const bool iny = iy < clear.ylo || iy >= clear.yhi;
    const bool inz = iz < clear.zlo || iz >= clear.zhi;
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix]) return;

    int sy = solver.nx;
    int sz = solver.nx * solver.ny;
    int sp = sz * solver.nz;
    int idx  = iz * sz + iy * sy + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;
    const float* gc = grad_ctx_x.coeff;
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdy2 = 1.0f / (lap_ctx.dy * lap_ctx.dy);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    float invdx  = 1.0f / lap_ctx.dx;
    float invdy  = 1.0f / lap_ctx.dy;
    float invdz  = 1.0f / lap_ctx.dz;

    float gun = f.u_now[idx];
    if (grad_out != nullptr &&
        ix >= solver.phys_x0() && ix < solver.phys_x1() &&
        iy >= solver.phys_y0() && iy < solver.phys_y1() &&
        iz >= solver.phys_z0() && iz < solver.phys_z1())
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];

    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw  = c0c * gun;

    #define GW_AT(n) ( vpb[n]*vpb[n]*dt2 * f.u_now[n] )
    #define GTMP_X(n, nix) ( cpml.bx[nix]*f.zetax[n] + (1.0f+cpml.bx[nix])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Y(n, niy) ( cpml.by[niy]*f.zetay[n] + (1.0f+cpml.by[niy])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Z(n, niz) ( cpml.bz[niz]*f.zetaz[n] + (1.0f+cpml.bz[niz])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )

    float Lx_glx, Dx_ggx = 0.0f;
    if (inx) {
        float ax_ = cpml.ax[ix], bx_ = cpml.bx[ix];
        float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
        float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw;
        Lx_glx = -lc[0] * ((1.0f + bx_) * gtmpx0);
        float Dx_gqx = 0.0f;
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
        psix_out[oidx]  = ax_ * f.psix[idx] + daxdx * gtmpx0 - Dx_gqx;
        zetax_out[oidx] = ax_ * (f.zetax[idx] + gw);
    } else {
        Lx_glx = -lc[0] * gw;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) Lx_glx += lc[k] * (GW_AT(idx + k) + GW_AT(idx - k));
        Lx_glx *= invdx2;
    }

    float Ly_gly, Dy_ggy = 0.0f;
    if (iny) {
        float ay_ = cpml.ay[iy], by_ = cpml.by[iy];
        float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
        float gtmpy0 = by_ * f.zetay[idx] + (1.0f + by_) * gw;
        Ly_gly = -lc[0] * ((1.0f + by_) * gtmpy0);
        float Dy_gqy = 0.0f;
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
        psiy_out[oidx]  = ay_ * f.psiy[idx] + daydy * gtmpy0 - Dy_gqy;
        zetay_out[oidx] = ay_ * (f.zetay[idx] + gw);
    } else {
        Ly_gly = -lc[0] * gw;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) Ly_gly += lc[k] * (GW_AT(idx + k * sy) + GW_AT(idx - k * sy));
        Ly_gly *= invdy2;
    }

    float Lz_glz, Dz_ggz = 0.0f;
    if (inz) {
        float az_ = cpml.az[iz], bz_ = cpml.bz[iz];
        float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
        float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw;
        Lz_glz = -lc[0] * ((1.0f + bz_) * gtmpz0);
        float Dz_gqz = 0.0f;
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
        psiz_out[oidx]  = az_ * f.psiz[idx] + dazdz * gtmpz0 - Dz_gqz;
        zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw);
    } else {
        Lz_glz = -lc[0] * gw;
        #pragma unroll
        for (int k = 1; k <= halo; ++k) Lz_glz += lc[k] * (GW_AT(idx + k * sz) + GW_AT(idx - k * sz));
        Lz_glz *= invdz2;
    }
    #undef GW_AT
    #undef GTMP_X
    #undef GTMP_Y
    #undef GTMP_Z

    f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                  + (Lx_glx + Ly_gly + Lz_glz - Dx_ggx - Dy_ggy - Dz_ggz);
}

// the shell: the slabs around the interior box (all the computed cells when the
// box is empty)
template<int Order>
void launch_shell(const AcousticWavefieldPointer& w, const float* vp,
                  const LaplaceParam& lap, const GradParam& gx, const GradParam& gy,
                  const GradParam& gz, const AcousticCPMLPointer& cpml, const SolverContext& ctx,
                  const Box& box, float* const out[6], const float* img, float* grad)
{
    // per axis, the range whose taps cannot reach a non-zero a/b/db (see above)
    const Box clear = fdtd::phys_box(ctx, ctx.M + 1);
    fdtd::for_each_slab(fdtd::computed_box(ctx), box, [&](const Box& r, bool x_slab) {
        const auto L = fdtd::box_launch(r, ctx.B, x_slab ? fdtd::kNarrowBlock : fdtd::kBoxBlock);
        adj3d_shell<Order><<<L.grid, L.block>>>(w, vp, lap, gx, gy, gz, cpml, ctx, box, clear,
            Region{r.xlo, r.ylo, r.zlo, r.xhi - r.xlo, r.yhi - r.ylo, r.zhi - r.zlo},
            out[0], out[1], out[2], out[3], out[4], out[5], img, grad);
    });
}

template<int Order>
void launch_interior(const AcousticWavefieldPointer& w, const float* vp, const LaplaceParam& lap,
                     const SolverContext& ctx, const Box& box, const float* img, float* grad)
{
    const auto L = fdtd::box_launch(box, ctx.B, fdtd::kBoxBlock);
    adj3d_interior_percell<Order><<<L.grid, L.block>>>(w, vp, lap, ctx, box, img, grad);
}

}  // namespace

void launch_acoustic3d_adjoint(
    int order, const AcousticWavefieldPointer& adj_view, const float* vp, const LaplaceParam& lap_ctx,
    const GradParam& gx, const GradParam& gy, const GradParam& gz,
    const AcousticCPMLPointer& cpml, const SolverContext& ctx,
    float* psixn, float* psiyn, float* psizn, float* zetaxn, float* zetayn, float* zetazn,
    const float* grad_forward_img, float* grad_out)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic3d adjoint: an x-ranged context is not supported (the adjoint "
                "step always covers the whole tile)");
    const Box box = fdtd::phys_box(ctx, ctx.M);   // the pure interior
    float* const out[6] = {psixn, psiyn, psizn, zetaxn, zetayn, zetazn};
    if (!fdtd::empty(box)) {
        switch (order) {
            case 2:  launch_interior<2>(adj_view, vp, lap_ctx, ctx, box, grad_forward_img, grad_out); break;
            case 4:  launch_interior<4>(adj_view, vp, lap_ctx, ctx, box, grad_forward_img, grad_out); break;
            case 6:  launch_interior<6>(adj_view, vp, lap_ctx, ctx, box, grad_forward_img, grad_out); break;
            case 8:  launch_interior<8>(adj_view, vp, lap_ctx, ctx, box, grad_forward_img, grad_out); break;
            default: launch_interior<-1>(adj_view, vp, lap_ctx, ctx, box, grad_forward_img, grad_out); break;
        }
    }
#define SHELL(O) launch_shell<O>(adj_view, vp, lap_ctx, gx, gy, gz, cpml, ctx, box, out, grad_forward_img, grad_out)
    switch (order) {
        case 2:  SHELL(2); break;
        case 4:  SHELL(4); break;
        case 6:  SHELL(6); break;
        case 8:  SHELL(8); break;
        default: SHELL(-1); break;
    }
#undef SHELL
}

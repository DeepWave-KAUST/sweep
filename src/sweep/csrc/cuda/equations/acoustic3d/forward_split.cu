// Acoustic3D forward step and boundary-saving reconstruction, launched over
// boxes.
//
// One kernel over the padded grid would make every thread carry the CPML path's
// registers, exit a third of the launch at once, and with a 32x4x2 block fetch
// the z-taps of u from L2.  Here:
//
//   interior  the physical box (no axis in its CPML band): the plain Laplacian;
//   shell     the six slabs around it: per axis, the CPML terms outside the
//             physical range of that axis and the plain Laplacian inside;
//   nopml     the reconstruction (with the fused BS vp-gradient imaging) over
//             its own box, the cells the saved boundary does not restore.
//
// The forward launches clip to the context's x range [x_base, x_end()), so a
// ranged (DD overlap) launch computes every cell exactly as the full launch.
// Topography: the air pre-pass clears the air cells and both forward kernels
// skip them.  Blocks are 32x1x8, deep in z, so the z-taps of a block's cells are
// reused in L1.  The reconstruction also streams the adjoint field and the
// gradient; its rows start their warps on a 128-B line (Box::ali), which the
// forward does not: there it would shift a block's rows against each other and
// lose the z-tap reuse.
#include "kernels.cuh"
#include "../../launch/box.cuh"
#include "../../../core/check.h"
#include <cstdint>

namespace {

using fdtd::Box;

// ali < 0: warps start at xlo.  ali >= 0 (row-aligned launch): ali is the u_now
// base pointer's float offset within a 128-B line, and each row's warps start at
// the last 128-B boundary at or before xlo (the cells before xlo exit), so the
// warps' row loads/stores are whole lines whatever nx is.  Same cells either way.
struct RowBox : Box { int ali = -1; };

__device__ __forceinline__ bool box_index(const RowBox& r, int B, int& ix, int& iy, int& iz, int& b,
                                          int nx = 0, int ny = 0, long spatial = 0)
{
    const int nbz = r.zhi - r.zlo;
    iy = r.ylo + blockIdx.y * blockDim.y + threadIdx.y;
    const int izg = blockIdx.z * blockDim.z + threadIdx.z;
    b = izg / nbz;
    iz = r.zlo + izg % nbz;
    int x0 = r.xlo;
    if (r.ali >= 0) {
        const long row = (long)b * spatial + ((long)iz * ny + iy) * nx;   // element offset of (b, iz, iy, 0)
        x0 -= (int)((r.ali + row + r.xlo) & 31);
    }
    ix = x0 + blockIdx.x * blockDim.x + threadIdx.x;
    return b < B && ix >= r.xlo && ix < r.xhi && iy < r.yhi;
}

// ----------------------------------------------------------------- interior
template<int Order>
__global__ void a3d_fwd_interior(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, RowBox box)
{
    int ix, iy, iz, b;
    if (!box_index(box, solver.B, ix, iy, iz, b, solver.nx, solver.ny, (long)solver.nx * solver.ny * solver.nz)) return;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix])
        return;   // air: cleared by acoustic3d_air_clear_kernel

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float w_sum = 0.f;
    w_sum = lap_x + lap_y + lap_z;

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;
}

// ----------------------------------------------------------------- CPML shell
template<int Order>
__global__ void __launch_bounds__(256) a3d_fwd_shell(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, RowBox reg)
{
    int ix, iy, iz, b;
    if (!box_index(reg, solver.B, ix, iy, iz, b, solver.nx, solver.ny, (long)solver.nx * solver.ny * solver.nz)) return;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix])
        return;   // air: cleared by acoustic3d_air_clear_kernel

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    const bool in_pml_x = (ix < solver.phys_x0()) || (ix >= solver.phys_x1());
    const bool in_pml_y = (iy < solver.phys_y0()) || (iy >= solver.phys_y1());
    const bool in_pml_z = (iz < solver.phys_z0()) || (iz >= solver.phys_z1());

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float w_sum = 0.f;

    if (in_pml_x) {
        float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
        long xi = solver.aux_rd_x3(iz, iy, ix);
        float dpsixdx = gradient_at<Order>(f.psix, (int)xi, grad_ctx.sx,
                                           grad_ctx.M, grad_ctx.coeff, grad_ctx.dx);
        float ax_ = cpml.ax[ix];
        float bx_ = cpml.bx[ix];
        float dbxdx_ = cpml.dbxdx[ix];
        float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);

        float tmpx = ((1.f + bx_) * lap_x + dbxdx_ * dudx)
                     + ax_ * dpsixdx + daxdx * f.psix[xi];

        w_sum += (1.f + bx_) * tmpx + ax_ * f.zetax[xi];

        if (solver.aux_x.stored(ix)) {
            (f.psixn ? f.psixn : f.psix)[xi]  = bx_ * dudx + ax_ * f.psix[xi];
            f.zetax[xi] = bx_ * tmpx + ax_ * f.zetax[xi];
        }
    } else {
        w_sum += lap_x;
    }

    if (in_pml_y) {
        float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
        long yi = solver.aux_rd_y3(iz, iy, ix);
        float dpsiydy = gradient_at<Order>(f.psiy, (int)yi, grad_ctx.sy,
                                           grad_ctx.M, grad_ctx.coeff, grad_ctx.dy);
        float ay_ = cpml.ay[iy];
        float by_ = cpml.by[iy];
        float dbydy_ = cpml.dbydy[iy];
        float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);

        float tmpy = ((1.f + by_) * lap_y + dbydy_ * dudy)
                     + ay_ * dpsiydy + daydy * f.psiy[yi];

        w_sum += (1.f + by_) * tmpy + ay_ * f.zetay[yi];

        if (solver.aux_y.stored(iy)) {
            (f.psiyn ? f.psiyn : f.psiy)[yi]  = by_ * dudy + ay_ * f.psiy[yi];
            f.zetay[yi] = by_ * tmpy + ay_ * f.zetay[yi];
        }
    } else {
        w_sum += lap_y;
    }

    if (in_pml_z) {
        float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
        long zi = solver.aux_rd_z3(iz, iy, ix);
        float dpsizdz = gradient_at<Order>(f.psiz, (int)zi, grad_ctx.sz,
                                           grad_ctx.M, grad_ctx.coeff, grad_ctx.dz);
        float az_ = cpml.az[iz];
        float bz_ = cpml.bz[iz];
        float dbzdz_ = cpml.dbzdz[iz];
        float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

        float tmpz = ((1.f + bz_) * lap_z + dbzdz_ * dudz)
                     + az_ * dpsizdz + dazdz * f.psiz[zi];

        w_sum += (1.f + bz_) * tmpz + az_ * f.zetaz[zi];

        if (solver.aux_z.stored(iz)) {
            (f.psizn ? f.psizn : f.psiz)[zi]  = bz_ * dudz + az_ * f.psiz[zi];
            f.zetaz[zi] = bz_ * tmpz + az_ * f.zetaz[zi];
        }
    } else {
        w_sum += lap_z;
    }

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_y + lap_z);
}

// ----------------------------------------------------------------- nopml
template<int Order>
__global__ void a3d_nopml_box(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver,
    const float* __restrict__ adj, float* __restrict__ grad, int save_width, RowBox box)
{
    int ix, iy, iz, b;
    if (!box_index(box, solver.B, ix, iy, iz, b, solver.nx, solver.ny, (long)solver.nx * solver.ny * solver.nz)) return;
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
    const float up = f.u_prev[idx];

    const float un =
        2.0f * u0 -
        up +
        (v * v) * solver.dt * solver.dt * w_sum;
    f.u_next[idx] = un;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;

    if (grad != nullptr) {
        const int ix0 = solver.cut_x_lo() ? solver.phys_x0() : solver.phys_x0() + save_width;
        const int ix1 = solver.cut_x_hi() ? solver.phys_x1() : solver.phys_x1() - save_width;
        const int iy0 = solver.cut_y_lo() ? solver.phys_y0() : solver.phys_y0() + save_width;
        const int iy1 = solver.cut_y_hi() ? solver.phys_y1() : solver.phys_y1() - save_width;
        const int iz0 = solver.cut_z_lo() ? solver.phys_z0() : solver.phys_z0() + save_width;
        const int iz1 = solver.cut_z_hi() ? solver.phys_z1() : solver.phys_z1() - save_width;
        if (ix >= ix0 && ix < ix1 && iy >= iy0 && iy < iy1 && iz >= iz0 && iz < iz1) {
            const float* adj_b  = adj  + b * spatial_size;
            float*       grad_b = grad + b * spatial_size;
            float u_tt = (un - 2*u0 + up) / (solver.dt*solver.dt);
            grad_b[idx] += 2.f * solver.dt * solver.dt * u_tt * adj_b[idx] / v;
        }
    }
}

// row-aligned: up to 31 lead-in cells per row
fdtd::LaunchConfig launch_of(const RowBox& r, int B)
{
    return fdtd::box_launch(r, B, fdtd::kBoxBlock, r.ali >= 0 ? 31 : 0);
}

// float offset of a device pointer within its 128-B line (row-aligned launches)
int align_phase(const void* p)
{
    return (int)((reinterpret_cast<uintptr_t>(p) / sizeof(float)) & 31);
}

struct FwdArgs {
    AcousticWavefieldPointer wf; float* u_this; const float* vp;
    LaplaceParam lap; GradParam g, gx, gy, gz; AcousticCPMLPointer cpml;
};

template<int Order>
void launch_fwd(const FwdArgs& a, const SolverContext& ctx)
{
    const Box all = fdtd::computed_box(ctx);
    // the cells with no axis in its CPML band
    const RowBox in{fdtd::intersect(fdtd::phys_box(ctx, 0), all)};
    if (!fdtd::empty(in)) {
        const auto L = launch_of(in, ctx.B);
        a3d_fwd_interior<Order><<<L.grid, L.block>>>(a.wf, a.u_this, a.vp, a.lap, ctx, in);
    }
    fdtd::for_each_slab(all, in, [&](const Box& s, bool) {
        const RowBox r{s};
        const auto L = launch_of(r, ctx.B);
        a3d_fwd_shell<Order><<<L.grid, L.block>>>(a.wf, a.u_this, a.vp, a.lap, a.g,
                                                   a.gx, a.gy, a.gz, a.cpml, ctx, r);
    });
}

struct NopmlArgs {
    AcousticWavefieldPointer wf; float* u_this; const float* vp; LaplaceParam lap;
    const float* adj; float* grad; int save_width;
};

template<int Order>
void launch_nopml(const NopmlArgs& a, const SolverContext& ctx)
{
    const int ali = align_phase(a.wf.u_now);
    // per-side exclusion bands: the PML band plus the restored strip
    const int M = ctx.M;
    const int wide = ctx.abcn > 0 ? ctx.abcn + 2*M+1 : 2*M;
    const int hx_lo = ctx.cut_x_lo() ? M : wide;
    const int hx_hi = ctx.cut_x_hi() ? M : wide;
    const int hy_lo = ctx.cut_y_lo() ? M : wide;
    const int hy_hi = ctx.cut_y_hi() ? M : wide;
    const int hz_lo = ctx.cut_z_lo() ? M : (ctx.free_surface ? 2*M : wide);
    const int hz_hi = ctx.cut_z_hi() ? M : wide;
    const RowBox box{{hx_lo, ctx.nx - hx_hi, hy_lo, ctx.ny - hy_hi, hz_lo, ctx.nz - hz_hi}, ali};
    if (fdtd::empty(box)) return;
    const auto L = launch_of(box, ctx.B);
    a3d_nopml_box<Order><<<L.grid, L.block>>>(a.wf, a.u_this, a.vp, a.lap, ctx,
                                              a.adj, a.grad, a.save_width, box);
}

}  // namespace

void launch_acoustic3d_forward(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    const FwdArgs a{wf, u_this, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml};
    switch (order) {
        case 2:  launch_fwd<2>(a, ctx); break;
        case 4:  launch_fwd<4>(a, ctx); break;
        case 6:  launch_fwd<6>(a, ctx); break;
        case 8:  launch_fwd<8>(a, ctx); break;
        default: launch_fwd<-1>(a, ctx); break;
    }
}

void launch_acoustic3d_nopml(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, SolverContext ctx,
    const float* adj, float* grad, int save_width)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic3d reconstruction: an x-ranged context is not supported (the "
                "reconstruction always covers the whole tile)");
    const NopmlArgs a{wf, u_this, vp, lap_ctx, adj, grad, save_width};
    switch (order) {
        case 2:  launch_nopml<2>(a, ctx); break;
        case 4:  launch_nopml<4>(a, ctx); break;
        case 6:  launch_nopml<6>(a, ctx); break;
        case 8:  launch_nopml<8>(a, ctx); break;
        default: launch_nopml<-1>(a, ctx); break;
    }
}

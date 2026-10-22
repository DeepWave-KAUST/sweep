// VRZ3D forward step and boundary-saving reconstruction, launched over boxes.
//
// One kernel over the padded grid would make every thread carry the CPML path's
// registers (72 on sm_70), exit a third of the launch at once, and with a
// 32x4x2 block fetch the z-taps of u, vp and 1/z from L2.  Here:
//
//   interior  the physical box: the plain update, at its own register count;
//   shell     the CPML update over the six slabs around the box, per axis: an
//             axis whose coordinate lies in [phys_lo + M, phys_hi - M) sees
//             a = b = db = 0 at the cell and a = psi = 0 at all its taps (the
//             forward reads db only at the cell, where np.gradient's one-cell
//             leak sits at phys_lo / phys_hi - 1, inside the M margin), so its
//             terms are the interior ones and its psi/zeta stay 0 -- skipped;
//   nopml     the reconstruction over its own box.
//
// The forward launches clip to the context's x range [x_base, x_end()), so a
// ranged (DD overlap) launch computes every cell exactly as the full launch.
// Blocks are 32x1x8, deep in z, so the z-taps of a block's cells are reused in L1.
#include "kernels.cuh"
#include "../../launch/box.cuh"
#include "../../../core/check.h"

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
template<int Order>
__global__ void vrz3d_fwd_interior(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    const float* __restrict__ z, const float* __restrict__ inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);
    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float rhs = kappa * (beta * (lap_x + lap_y + lap_z) + dbdx * dudx + dbdy * dudy + dbdz * dudz);
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// ----------------------------------------------------------------- CPML shell
template<int Order>
__global__ void __launch_bounds__(256) vrz3d_fwd_shell(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    const float* __restrict__ z, const float* __restrict__ inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, Box clear, Box reg)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(reg, solver.B, ix, iy, iz, b)) return;
    const bool on_x = ix < clear.xlo || ix >= clear.xhi;
    const bool on_y = iy < clear.ylo || iy >= clear.yhi;
    const bool on_z = iz < clear.zlo || iz >= clear.zhi;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);
    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    // per axis: the CPML terms, or (clear axis) exactly what they reduce to
    float wx = lap_x, px = dudx;
    if (on_x) {
        float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
        float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
        float ax_ = cpml.ax[ix], bx_ = cpml.bx[ix];
        float tmpx = ((1.0f + bx_) * lap_x + cpml.dbxdx[ix] * dudx) + daxdx * f.psix[idx] + ax_ * dpsixdx;
        float psixn = bx_ * dudx + ax_ * f.psix[idx];
        float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];
        wx = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx];
        px = dudx + psixn;
        (f.psixn ? f.psixn : f.psix)[idx] = psixn;
        f.zetax[idx] = zetaxn;
    }
    float wy = lap_y, py = dudy;
    if (on_y) {
        float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
        float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
        float ay_ = cpml.ay[iy], by_ = cpml.by[iy];
        float tmpy = ((1.0f + by_) * lap_y + cpml.dbydy[iy] * dudy) + daydy * f.psiy[idx] + ay_ * dpsiydy;
        float psiyn = by_ * dudy + ay_ * f.psiy[idx];
        float zetayn = by_ * tmpy + ay_ * f.zetay[idx];
        wy = (1.0f + by_) * tmpy + ay_ * f.zetay[idx];
        py = dudy + psiyn;
        (f.psiyn ? f.psiyn : f.psiy)[idx] = psiyn;
        f.zetay[idx] = zetayn;
    }
    float wz = lap_z, pz = dudz;
    if (on_z) {
        float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);
        float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
        float az_ = cpml.az[iz], bz_ = cpml.bz[iz];
        float tmpz = ((1.0f + bz_) * lap_z + cpml.dbzdz[iz] * dudz) + dazdz * f.psiz[idx] + az_ * dpsizdz;
        float psizn = bz_ * dudz + az_ * f.psiz[idx];
        float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];
        wz = (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];
        pz = dudz + psizn;
        (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
        f.zetaz[idx] = zetazn;
    }
    float w_sum = wx + wy + wz;
    float rhs = kappa * (beta * w_sum + dbdx * px + dbdy * py + dbdz * pz);
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// ----------------------------------------------------------------- nopml
template<int Order>
__global__ void vrz3d_nopml_box(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    const float* __restrict__ z, const float* __restrict__ inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);
    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float rhs = kappa * (beta * (lap_x + lap_y + lap_z) + dbdx * dudx + dbdy * dudy + dbdz * dudz);
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

struct FwdArgs {
    AcousticWavefieldPointer wf; const float *vp, *z, *iz;
    LaplaceParam lap; GradParam g, gx, gy, gz; AcousticCPMLPointer cpml;
};

template<int Order>
void launch_fwd(const FwdArgs& a, const SolverContext& ctx)
{
    const Box all = fdtd::computed_box(ctx);
    // interior = the physical box less the db leak cell next to each band
    // (in_cpml_3d false) within the x range: the shell's per-axis CPML terms
    // carry db = np.gradient(b), nonzero on that cell (dropping it reflected)
    const Box in = fdtd::intersect(fdtd::phys_box(ctx, 1), all);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        vrz3d_fwd_interior<Order><<<L.grid, L.block>>>(a.wf, a.vp, a.z, a.iz, a.lap, a.g, ctx, in);
    }
    // per-axis clear range: no band within M (cut sides carry no band)
    const Box clear = fdtd::phys_box(ctx, ctx.M);
    fdtd::for_each_slab(all, in, [&](const Box& r, bool) {
        const auto L = fdtd::box_launch(r, ctx.B, fdtd::kBoxBlock);
        vrz3d_fwd_shell<Order><<<L.grid, L.block>>>(a.wf, a.vp, a.z, a.iz, a.lap, a.g,
                                                     a.gx, a.gy, a.gz, a.cpml, ctx, clear, r);
    });
}

template<int Order>
void launch_nopml(const FwdArgs& a, const SolverContext& ctx)
{
    // the reconstruction box: the physical box, one cell in from each non-cut face
    const Box box{ctx.phys_x0() + (ctx.cut_x_lo() ? 0 : 1), ctx.phys_x1() - (ctx.cut_x_hi() ? 0 : 1),
                  ctx.phys_y0() + (ctx.cut_y_lo() ? 0 : 1), ctx.phys_y1() - (ctx.cut_y_hi() ? 0 : 1),
                  ctx.phys_z0() + (ctx.cut_z_lo() ? 0 : 1), ctx.phys_z1() - (ctx.cut_z_hi() ? 0 : 1)};
    if (fdtd::empty(box)) return;
    const auto L = fdtd::box_launch(box, ctx.B, fdtd::kBoxBlock);
    vrz3d_nopml_box<Order><<<L.grid, L.block>>>(a.wf, a.vp, a.z, a.iz, a.lap, a.g, ctx, box);
}

}  // namespace

void launch_acoustic_vrz3d_forward(
    int order, AcousticWavefieldPointer wf,
    const float* vp, const float* z, const float* inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    const FwdArgs a{wf, vp, z, inv_z, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml};
    switch (order) {
        case 2:  launch_fwd<2>(a, ctx); break;
        case 4:  launch_fwd<4>(a, ctx); break;
        case 6:  launch_fwd<6>(a, ctx); break;
        case 8:  launch_fwd<8>(a, ctx); break;
        default: launch_fwd<-1>(a, ctx); break;
    }
}

void launch_acoustic_vrz3d_nopml(
    int order, AcousticWavefieldPointer wf,
    const float* vp, const float* z, const float* inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext ctx)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic_vrz3d reconstruction: an x-ranged context is not supported (the "
                "reconstruction always covers the whole tile)");
    const FwdArgs a{wf, vp, z, inv_z, lap_ctx, grad_ctx, grad_ctx, grad_ctx, grad_ctx, AcousticCPMLPointer{}};
    switch (order) {
        case 2:  launch_nopml<2>(a, ctx); break;
        case 4:  launch_nopml<4>(a, ctx); break;
        case 6:  launch_nopml<6>(a, ctx); break;
        case 8:  launch_nopml<8>(a, ctx); break;
        default: launch_nopml<-1>(a, ctx); break;
    }
}

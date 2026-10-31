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
//
// Host core: the interior and the reconstruction kernels are split per
// launch/cell_path.h (decode + guard in the kernel, the body in a _cell) and
// launched through row launchers (SWEEP_LAUNCH_ROWS; A3dFwdInteriorRows and
// A3dNopmlBoxRows, after the kernels) that run each row's interior cells as one
// SIMD loop over that same body.  The shell stays scalar.
#include "kernels.cuh"
#include "../../launch/launch.h"   // SWEEP_LAUNCH, SWEEP_LAUNCH_ROWS
#include "../../launch/box.cuh"
#include "../../launch/cell_path.h"   // CellPath
#include "../../common/host_rows.h"   // (host core only)
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

// The indices of one cell as box_index decodes them, handed to a _cell body in
// one by-value struct rather than as four int parameters: nvcc orders a
// compare's operands by kind, a bare int parameter below a load, so with ints
// the split kernels would not keep their SASS (elastic_tti_sg3d CellIndex3,
// docs/dev/host_core.md rule 7).  On the host it is free.
struct A3dCell { int ix, iy, iz, b; };

// ----------------------------------------------------------------- interior
// One cell of a3d_fwd_interior: everything after the kernel's box_index guard
// (launch/cell_path.h).  Interior path: the cell is not air (the air return is
// dead); optional output: u_this.
//
// The two u_this tests keep the kernel's own spelling on its own path and go
// through optional_output only on the interior ones.  Through the helper -- a
// call while nvcc first simplifies the cell on its own, even __forceinline__ --
// dev's fold of ``u_this_b != nullptr`` into the ``u_this`` test is lost at
// the static orders: one more 64-bit null test (SASS compared; a3d_nopml_box
// alike).
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void a3d_fwd_interior_cell(
    A3dCell cell, const AcousticWavefieldPointer& wf, float* __restrict__ u_this,
    const float* __restrict__ vp, const LaplaceParam& lap_ctx, const SolverContext& solver)
{
    const int ix = cell.ix, iy = cell.iy, iz = cell.iz, b = cell.b;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    if constexpr (!interior_path<Path>) {
        if (solver.has_topo && iz < solver.topo_rows[iy * solver.nx + ix])
            return;   // air: cleared by acoustic3d_air_clear_kernel
    }

    float*       u_this_b = (Path == CellPath::Any ? u_this != nullptr : optional_output<Path>(u_this))
                                ? u_this + spatial_size * b : nullptr;
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

    if (Path == CellPath::Any ? u_this_b != nullptr : optional_output<Path>(u_this_b))
        u_this_b[idx] = (v * v) * w_sum;
}

template<int Order>
__global__ void a3d_fwd_interior(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, RowBox box)
{
    int ix, iy, iz, b;
    if (!box_index(box, solver.B, ix, iy, iz, b, solver.nx, solver.ny, (long)solver.nx * solver.ny * solver.nz)) return;
    a3d_fwd_interior_cell<Order>(A3dCell{ix, iy, iz, b}, wf, u_this, vp, lap_ctx, solver);
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

    // Per axis, the CPML branch runs on the band plus the one cell past it
    // where db = np.gradient(b) is still nonzero (a and the own psi are 0
    // there, so only db * du survives); a DD cut face has no band, no margin.
    const bool in_pml_x = (ix < solver.phys_x0() + (solver.cut_x_lo() ? 0 : 1)) ||
                          (ix >= solver.phys_x1() - (solver.cut_x_hi() ? 0 : 1));
    const bool in_pml_y = (iy < solver.phys_y0() + (solver.cut_y_lo() ? 0 : 1)) ||
                          (iy >= solver.phys_y1() - (solver.cut_y_hi() ? 0 : 1));
    const bool in_pml_z = (iz < solver.phys_z0() + (solver.cut_z_lo() ? 0 : 1)) ||
                          (iz >= solver.phys_z1() - (solver.cut_z_hi() ? 0 : 1));

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

    // the vp image is the update actually applied (boundary saving's u_tt)
    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;
}

// ----------------------------------------------------------------- nopml
// One cell of a3d_nopml_box: everything after the kernel's box_index guard
// (launch/cell_path.h).  Interior path: with grad there, the cell is in the
// fused imaging's box (its test is true); optional outputs: u_this (0),
// grad (1).  The u_this tests: see a3d_fwd_interior_cell.
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void a3d_nopml_box_cell(
    A3dCell cell, const AcousticWavefieldPointer& wf, float* __restrict__ u_this,
    const float* __restrict__ vp, const LaplaceParam& lap_ctx, const SolverContext& solver,
    const float* __restrict__ adj, float* __restrict__ grad, int save_width)
{
    const int ix = cell.ix, iy = cell.iy, iz = cell.iz, b = cell.b;
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;

    auto f = wf.offset(b, spatial_size);

    int idx = iz * stride_z + iy * stride_y + ix;

    float*       u_this_b = (Path == CellPath::Any ? u_this != nullptr : optional_output<Path, 0>(u_this))
                                ? u_this + spatial_size * b : nullptr;
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

    if (Path == CellPath::Any ? u_this_b != nullptr : optional_output<Path, 0>(u_this_b))
        u_this_b[idx] = (v * v) * w_sum;

    if (optional_output<Path, 1>(grad)) {
        bool image = true;
        if constexpr (!interior_path<Path>) {
            const int ix0 = solver.cut_x_lo() ? solver.phys_x0() : solver.phys_x0() + save_width;
            const int ix1 = solver.cut_x_hi() ? solver.phys_x1() : solver.phys_x1() - save_width;
            const int iy0 = solver.cut_y_lo() ? solver.phys_y0() : solver.phys_y0() + save_width;
            const int iy1 = solver.cut_y_hi() ? solver.phys_y1() : solver.phys_y1() - save_width;
            const int iz0 = solver.cut_z_lo() ? solver.phys_z0() : solver.phys_z0() + save_width;
            const int iz1 = solver.cut_z_hi() ? solver.phys_z1() : solver.phys_z1() - save_width;
            image = ix >= ix0 && ix < ix1 && iy >= iy0 && iy < iy1 && iz >= iz0 && iz < iz1;
        }
        if (image) {
            const float* adj_b  = adj  + b * spatial_size;
            float*       grad_b = grad + b * spatial_size;
            float u_tt = (un - 2*u0 + up) / (solver.dt*solver.dt);
            grad_b[idx] += 2.f * solver.dt * solver.dt * u_tt * adj_b[idx] / v;
        }
    }
}

template<int Order>
__global__ void a3d_nopml_box(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver,
    const float* __restrict__ adj, float* __restrict__ grad, int save_width, RowBox box)
{
    int ix, iy, iz, b;
    if (!box_index(box, solver.B, ix, iy, iz, b, solver.nx, solver.ny, (long)solver.nx * solver.ny * solver.nz)) return;
    a3d_nopml_box_cell<Order>(A3dCell{ix, iy, iz, b}, wf, u_this, vp, lap_ctx, solver, adj, grad,
                              save_width);
}

#if defined(SWEEP_HOST_CORE)
// ----------------------------------------------------------------- host rows
// Host launchers of a3d_fwd_interior and a3d_nopml_box (SWEEP_LAUNCH_ROWS
// below): host/launch.h launch_rows over the kernel's own launch, the span of
// each row built from the kernel's own tests (row_box_rows) and the cell body
// run there.  They hold no arithmetic; the cell lambdas capture BY VALUE.
// Order -1 (runtime M) takes the same spans (the boxes do not depend on the
// order); its row loops do not vectorise.

// box_index on the host: the cell of virtual thread (gx, gy, gz) of a launch
// over ``r`` (launch_of; gx, gy, gz = blockIdx * blockDim + threadIdx), with
// the kernel's nx, ny, spatial.  A row-aligned box (ali >= 0) starts row
// (gy, gz) ``shift`` cells before xlo, shift depending on the row alone, so
// the cells of a row are consecutive in gx either way.  The shift is masked,
// not branched on: as a branch it stays a PHI inside the row loop, ix is not
// affine in gx to the vectoriser and the stencil loads become gathers (the
// row does not vectorise).
inline A3dCell row_box_cell_at(const RowBox& r, int nx, int ny, long spatial, int gx, int gy, int gz)
{
    const int nbz = r.zhi - r.zlo;
    A3dCell c;
    c.iy = r.ylo + gy;
    c.b = gz / nbz;
    c.iz = r.zlo + gz % nbz;
    const long row = (long)c.b * spatial + ((long)c.iz * ny + c.iy) * nx;   // element offset of (b, iz, iy, 0)
    const int shift = (int)((r.ali + row + r.xlo) & 31) & -(int)(r.ali >= 0);   // 0 when ali < 0
    c.ix = r.xlo - shift + gx;
    return c;
}

// The span of a launch over ``r`` whose interior cells are those of ``in``: in
// row (gy, gz) the virtual threads of in's x extent [in.x0, in.x1), when the
// row's (iy, iz) is in ``in`` and its shot before the last (box_index's
// guard: b < B, xlo <= ix < xhi, iy < yhi -- ``in`` is clipped to r) and, for
// a kernel with an air return (``air``) under topography, iz is at or below
// the deepest surface row over in's columns of that iy.  Every cell of the
// span therefore passes the guard and takes the branch the interior path
// hard-codes; the rest of the row runs the kernel.
inline auto row_box_rows(const RowBox& r, CellBox3 in, const SolverContext& solver, bool air)
{
    in = in.intersect({r.xlo, r.xhi, r.ylo, r.yhi, r.zlo, r.zhi});
    const bool none = in.x1 <= in.x0 || in.y1 <= in.y0 || in.z1 <= in.z0;
    std::vector<int> first_solid;   // per iy - in.y0: the first iz solid across in's columns
    if (air && solver.has_topo && !none) {
        first_solid.assign(in.y1 - in.y0, 0);
        for (int iy = in.y0; iy < in.y1; ++iy)
            for (int ix = in.x0; ix < in.x1; ++ix)
                first_solid[iy - in.y0] =
                    std::max(first_solid[iy - in.y0], solver.topo_rows[iy * solver.nx + ix]);
    }
    const int nx = solver.nx, ny = solver.ny, nb = solver.B;
    const long spatial = (long)solver.nx * solver.ny * solver.nz;
    return [r, in, none, nx, ny, nb, spatial, first_solid = std::move(first_solid)](int gy, int gz) {
        if (none)
            return sweep::host::RowSpan{};
        const A3dCell c = row_box_cell_at(r, nx, ny, spatial, 0, gy, gz);   // c.ix: the cell of gx = 0
        if (c.b >= nb || c.iy < in.y0 || c.iy >= in.y1 || c.iz < in.z0 || c.iz >= in.z1)
            return sweep::host::RowSpan{};
        if (!first_solid.empty() && c.iz < first_solid[c.iy - in.y0])
            return sweep::host::RowSpan{};
        return sweep::host::RowSpan{in.x0 - c.ix, in.x1 - c.ix};
    };
}

// a3d_fwd_interior: its one test after the guard is the air return, so the
// span is the whole box minus the rows with air (row_box_rows, air = true).
// Nothing else in the cell depends on where it is: DD cut faces, per-edge pads
// and the x range shape only the box the launcher passes, and the span is
// built from that box.
template<int Order>
struct A3dFwdInteriorRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, float* u_this,
                    const float* vp, LaplaceParam lap_ctx, SolverContext solver, RowBox box)
    {
        auto call = [&](auto&&... a) { a3d_fwd_interior<Order>(a...); };
        const int nx = solver.nx, ny = solver.ny;
        const long spatial = (long)solver.nx * solver.ny * solver.nz;
        const auto span = row_box_rows(
            box, CellBox3{box.xlo, box.xhi, box.ylo, box.yhi, box.zlo, box.zhi}, solver, true);
        sweep::host::with_optional(u_this, [&](auto path) {
            constexpr CellPath P = decltype(path)::value;
            sweep::host::launch_rows(
                grid, block, span,
                [=](int gx, int gy, int gz) {
                    a3d_fwd_interior_cell<Order, P>(row_box_cell_at(box, nx, ny, spatial, gx, gy, gz),
                                                    wf, u_this, vp, lap_ctx, solver);
                },
                call,
                wf, u_this, vp, lap_ctx, solver, box);
        });
    }
};

// a3d_nopml_box: no air return; its one test is the fused imaging's box
// (ix0..iz1 above, the same expressions), run only when grad is there.  So
// the span is the box, cut to that imaging box when grad is there -- with a
// free surface the box starts at 2M and the imaging at phys_z0 + save_width,
// and per-edge pads move the two apart on any face.
template<int Order>
struct A3dNopmlBoxRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, float* u_this,
                    const float* vp, LaplaceParam lap_ctx, SolverContext solver, const float* adj,
                    float* grad, int save_width, RowBox box)
    {
        auto call = [&](auto&&... a) { a3d_nopml_box<Order>(a...); };
        const CellBox3 all{box.xlo, box.xhi, box.ylo, box.yhi, box.zlo, box.zhi};
        const CellBox3 image{
            solver.cut_x_lo() ? solver.phys_x0() : solver.phys_x0() + save_width,
            solver.cut_x_hi() ? solver.phys_x1() : solver.phys_x1() - save_width,
            solver.cut_y_lo() ? solver.phys_y0() : solver.phys_y0() + save_width,
            solver.cut_y_hi() ? solver.phys_y1() : solver.phys_y1() - save_width,
            solver.cut_z_lo() ? solver.phys_z0() : solver.phys_z0() + save_width,
            solver.cut_z_hi() ? solver.phys_z1() : solver.phys_z1() - save_width};
        const int nx = solver.nx, ny = solver.ny;
        const long spatial = (long)solver.nx * solver.ny * solver.nz;
        const auto span = row_box_rows(box, grad != nullptr ? all.intersect(image) : all, solver, false);
        sweep::host::with_optionals(
            [&](auto path) {
                constexpr CellPath P = decltype(path)::value;
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        a3d_nopml_box_cell<Order, P>(row_box_cell_at(box, nx, ny, spatial, gx, gy, gz),
                                                     wf, u_this, vp, lap_ctx, solver, adj, grad,
                                                     save_width);
                    },
                    call,
                    wf, u_this, vp, lap_ctx, solver, adj, grad, save_width, box);
            },
            u_this, grad);
    }
};
#endif

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
    // the cells with no axis in its CPML band or on the db leak cell past it
    const RowBox in{fdtd::intersect(fdtd::phys_box(ctx, 1), all)};
    if (!fdtd::empty(in)) {
        const auto L = launch_of(in, ctx.B);
        SWEEP_LAUNCH_ROWS((a3d_fwd_interior<Order>), A3dFwdInteriorRows<Order>::run, L.grid, L.block,
                          a.wf, a.u_this, a.vp, a.lap, ctx, in);
    }
    fdtd::for_each_slab(all, in, [&](const Box& s, bool) {
        const RowBox r{s};
        const auto L = launch_of(r, ctx.B);
        SWEEP_LAUNCH((a3d_fwd_shell<Order>), L.grid, L.block, a.wf, a.u_this, a.vp, a.lap, a.g,
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
    SWEEP_LAUNCH_ROWS((a3d_nopml_box<Order>), A3dNopmlBoxRows<Order>::run, L.grid, L.block,
                      a.wf, a.u_this, a.vp, a.lap, ctx, a.adj, a.grad, a.save_width, box);
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

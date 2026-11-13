#pragma once
// common/host_rows.h -- row spans for the host core's row launcher
// (host/launch.h launch_rows) over a kernel split per launch/cell_path.h:
// the kernel's interior box in the virtual-thread coordinates the launcher
// walks, minus the rows an irregular surface puts air into (the kernels'
// topography early-return cannot be answered by an interior path, so those
// rows go through the kernel).  Host core only.
#if defined(SWEEP_HOST_CORE)
#include <algorithm>
#include <utility>
#include <vector>

#include "context.h"
#include "../launch/box.cuh"
#include "../launch/cell_path.h"
#include "../../host/launch.h"

namespace sweep {
namespace host {

// ``box`` (a Wave2D kernel's interior, (ix, iz)) with z0 raised past the
// deepest surface row over its columns: every row left holds no air cell.
inline CellBox below_surface_2d(CellBox box, const SolverContext& solver)
{
    if (!solver.has_topo) return box;
    for (int ix = std::max(box.x0, 0); ix < std::min(box.x1, solver.nx); ++ix)
        box.z0 = std::max(box.z0, solver.topo_rows[ix]);
    return box;
}

// The span of a Wave3D kernel -- gx = ix - xb, gy = iy, gz = b * nz + iz --
// whose interior is ``box`` (ix, iy, iz): rows past the batch end or outside
// the box are empty, and with a surface so are the rows (iy, iz) above the
// deepest surface row of iy's columns in the box.
inline auto wave3d_rows(CellBox3 box, const SolverContext& solver, int xb)
{
    std::vector<int> first_solid;   // per iy: the first iz that is solid across the box
    if (solver.has_topo) {
        first_solid.assign(solver.ny, 0);
        for (int iy = std::max(box.y0, 0); iy < std::min(box.y1, solver.ny); ++iy)
            for (int ix = std::max(box.x0, 0); ix < std::min(box.x1, solver.nx); ++ix)
                first_solid[iy] = std::max(first_solid[iy], solver.topo_rows[iy * solver.nx + ix]);
    }
    const int nz = solver.nz, nb = solver.B;
    return [box, first_solid = std::move(first_solid), xb, nz, nb](int gy, int gz) {
        const int b = gz / nz, iz = gz % nz;
        if (b >= nb || gy < box.y0 || gy >= box.y1 || iz < box.z0 || iz >= box.z1)
            return RowSpan{};
        if (!first_solid.empty() && iz < first_solid[gy])
            return RowSpan{};
        return RowSpan{box.x0 - xb, box.x1 - xb};
    };
}

// ---------------------------------------------------------------------------
// Box launches (launch/box.cuh: box_launch with lead_x = 0, box_cell).
//
// The virtual thread (gx, gy, gz) of a box launch over ``r`` is the cell
//   ix = r.xlo + gx,  iy = r.ylo + gy,  b = gz / nbz,  iz = r.zlo + gz % nbz
// with nbz = r.zhi - r.zlo; past ``r`` in x or y, or past the last shot, the
// kernel exits.  box_cell_at is that decode for a launcher's cell lambda.
inline void box_cell_at(const fdtd::Box& r, int gx, int gy, int gz, int& ix, int& iy, int& iz, int& b)
{
    const int nbz = r.zhi - r.zlo;
    ix = r.xlo + gx;
    iy = r.ylo + gy;
    b = gz / nbz;
    iz = r.zlo + gz % nbz;
}

// The span of a box launch whose every cell takes the kernel's interior path
// (the split steps' interior kernels, which carry no CPML branch at all): row
// (gy, gz) is the whole x extent of ``r``, except past the last shot and --
// with a surface -- in the rows (iy, iz) above the deepest surface row over
// the box's columns of that iy, where the kernel's air early-return must run
// (topo_rows is per column ix in 2-D, per (iy, ix) in 3-D).  A launch only
// part of whose box is interior (a shell) needs a span of its own.
inline auto box_launch_rows(const fdtd::Box& r, const SolverContext& solver)
{
    const int nbz = r.zhi - r.zlo, wy = r.yhi - r.ylo, wx = r.xhi - r.xlo, nb = solver.B;
    std::vector<int> first_solid;   // per box row iy - r.ylo: the first iz solid across the box
    if (solver.has_topo && !fdtd::empty(r)) {
        first_solid.assign(wy, 0);
        for (int iy = r.ylo; iy < r.yhi; ++iy)
            for (int ix = r.xlo; ix < r.xhi; ++ix)
                first_solid[iy - r.ylo] = std::max(
                    first_solid[iy - r.ylo],
                    solver.topo_rows[solver.ndim == 2 ? ix : iy * solver.nx + ix]);
    }
    const int zlo = r.zlo;
    return [nbz, wy, wx, nb, zlo, first_solid = std::move(first_solid)](int gy, int gz) {
        if (nbz <= 0 || wx <= 0 || gy >= wy || gz / nbz >= nb)
            return RowSpan{};
        if (!first_solid.empty() && zlo + gz % nbz < first_solid[gy])
            return RowSpan{};
        return RowSpan{0, wx};
    };
}

// ---------------------------------------------------------------------------
// Elastic free surface (common/elastic_free_surface.cuh, "Interior paths").
//
// ``box`` (2-D, (ix, iz)) without the cells an active free-surface face
// reaches, for a cell whose derivatives go through the FS helpers with half
// order ``M`` (elastic_stencil_half_order: Order / 2, solver.M for Order -1):
//
//   forward (adjoint = false) -- elastic_top_fs_sgradient_z_2d,
//     elastic_fs_sgradient_x_2d: taps read [i - M, i + M] and the M cells
//     next to a face are order-2 cells, so
//       z in [top + M, bot - M)          x in [left + M, right - M);
//     this also drops the surface rows/columns (traction, Robertsson and
//     zero-shear tests: elastic_is_top_free_surface_row & co.).
//   adjoint (adjoint = true) -- elastic_top_fs_adjoint_sgradient_z_2d,
//     elastic_fs_adjoint_sgradient_x_2d: the reduced transpose reads
//     [i - M, i + M] and skips band taps (2M from a face), and the air fold at
//     the mirrored index must have no tap inside the grid:
//       z in [2 top + M + 1, min(bot - 2M, 2 bot - nz - M))
//       x in [2 left + M + 1, min(right - 2M, 2 right - nx - M)).
//
// top = surface_row: the deepest surface row over the box's columns under
// topography, phys_z0() otherwise; bot = phys_z1() - 1, left = phys_x0(),
// right = phys_x1() - 1.  Only the faces that are on cut the box.
inline CellBox elastic_fs_box_2d(CellBox box, const SolverContext& solver, int M, bool adjoint)
{
    if (solver.fsLo(0)) {
        int top = solver.phys_z0();
        if (solver.has_topo) {
            top = 0;
            for (int ix = std::max(box.x0, 0); ix < std::min(box.x1, solver.nx); ++ix)
                top = std::max(top, solver.topo_rows[ix]);
        }
        box.z0 = std::max(box.z0, adjoint ? 2 * top + M + 1 : top + M);
    }
    if (solver.fsHi(0)) {
        const int bot = solver.phys_z1() - 1;
        box.z1 = std::min(box.z1, adjoint ? std::min(bot - 2 * M, 2 * bot - solver.nz - M) : bot - M);
    }
    if (solver.fsLo(2)) {
        const int left = solver.phys_x0();
        box.x0 = std::max(box.x0, adjoint ? 2 * left + M + 1 : left + M);
    }
    if (solver.fsHi(2)) {
        const int right = solver.phys_x1() - 1;
        box.x1 = std::min(box.x1, adjoint ? std::min(right - 2 * M, 2 * right - solver.nx - M) : right - M);
    }
    return box;
}

// The 3-D twin (one z-low face, solver.free_surface; no topography in 3-D
// elastic): elastic3d_top_fs_sgradient_z mirrors taps above phys_z0, so
// z >= phys_z0 + M; elastic3d_top_fs_adjoint_sgradient_z drops taps above
// phys_z0 and adds ghost taps up to phys_z0 + M, so z >= phys_z0 + M + 1.
// The surface row phys_z0 is outside either way.
inline CellBox3 elastic3d_fs_box(CellBox3 box, const SolverContext& solver, int M, bool adjoint)
{
    if (solver.free_surface)
        box.z0 = std::max(box.z0, solver.phys_z0() + M + (adjoint ? 1 : 0));
    return box;
}

// with_flags (host/launch.h) for the FS helpers' ELASTIC_FS_Z (bit 0) and
// ELASTIC_FS_X (bit 1): ``f(std::integral_constant<unsigned, Mask>)``, for
// add_flags(path, Mask).  A family with flags of its own spells
// with_flags(f, <these two>, <its own from bit 2>) instead.
template <class F>
inline void with_elastic_fs_2d(const SolverContext& solver, F&& f)
{
    with_flags(f, solver.fsLo(0) || solver.fsHi(0), solver.fsLo(2) || solver.fsHi(2));
}

template <class F>
inline void with_elastic_fs_3d(const SolverContext& solver, F&& f)
{
    with_flags(f, solver.free_surface);
}

}  // namespace host
}  // namespace sweep
#endif

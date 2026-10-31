// Acoustic2D forward step launched as the interior box plus one launch over
// the CPML ring around it.
//
// One kernel over the padded grid (acoustic2nd) makes every thread carry the
// CPML path's registers -- 53-56 on sm_70, half occupancy -- although only the
// ring, a few percent of a 2-D grid, takes that path.  Here:
//
//   interior  the physical box (no axis in its CPML band): acoustic2nd's fast
//             path, verbatim, at its own register count;
//   ring      the other computed cells as ONE launch (launch/box.cuh SlabSet;
//             four thin strips as four launches cost more than they compute on
//             small grids): acoustic2nd's per-axis CPML path, verbatim.
//
// Both clip to the context's x range [x_base, x_end()), so a ranged (DD
// phase-split) launch computes every cell exactly as the full launch; both skip
// the air cells the air pre-pass cleared.  acoustic2nd itself stays for
// visco_acoustic2d.
#include "kernels.cuh"
#include "../../launch/launch.h"   // SWEEP_LAUNCH, SWEEP_LAUNCH_ROWS
#include "../../launch/box.cuh"
#include "../../common/host_rows.h"   // box_launch_rows, box_cell_at (host core only)

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
// One cell of a2d_fwd_interior: everything after the kernel's guards
// (launch/cell_path.h).  The kernel has no PML branch and keeps both of its
// early returns -- the box decode and the air test -- which the host span
// answers instead (A2dFwdInteriorRows); optional output: u_this.
//
// The store's test is the kernel's own ``u_this_b != nullptr`` on the Any
// path and the compile-time answer on an interior one.  Spelled through
// optional_output on the Any path as well, nvcc no longer folded the test to
// ``u_this != nullptr`` (it tested u_this and u_this_b both: a different
// instruction mix at every static order, +8 instructions at orders 4 and 6
// and a register more at 6 -- docs/dev/host_core.md rule 7).
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void a2d_fwd_interior_cell(
    Acoustic2dCellIndex cell,
    const AcousticWavefieldPointer& wf, float* __restrict__ u_this, const float* __restrict__ vp,
    const LaplaceParam& lap_ctx, const SolverContext& solver)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] +
                    v2_dt2 * (lap_x + lap_z);
    bool store_u_this;
    if constexpr (interior_path<Path>)
        store_u_this = optional_output<Path>(u_this);
    else
        store_u_this = u_this_b != nullptr;
    if (store_u_this)
        u_this_b[idx] = (v * v) * (lap_x + lap_z);
}

template<int Order>
__global__ void a2d_fwd_interior(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;
    if (solver.has_topo && iz < solver.topo_rows[ix])
        return;   // air: cleared by acoustic2d_air_clear_kernel

    a2d_fwd_interior_cell<Order>(Acoustic2dCellIndex{ix, iz, b}, wf, u_this, vp, lap_ctx, solver);
}

// ----------------------------------------------------------------- CPML ring
template<int Order>
__global__ void __launch_bounds__(256) a2d_fwd_ring(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, fdtd::SlabSet ring)
{
    int ix, iy, iz, b;
    if (!fdtd::slab_cell(ring, solver.B, ix, iy, iz, b)) return;
    if (solver.has_topo && iz < solver.topo_rows[ix])
        return;   // air: cleared by acoustic2d_air_clear_kernel

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    // acoustic2nd's per-axis compute bands (see there): an axis' CPML terms
    // run within halo of its coefficient band, the bare Laplacian beyond.
    int pml_x0 = solver.cut_x_lo() ? halo : solver.padLo(2) + halo;
    int pml_x1 = solver.nx - (solver.cut_x_hi() ? halo : solver.padHi(2) + halo);
    int pml_z0 = solver.cut_z_lo() ? halo : solver.padLo(0) + halo;
    int pml_z1 = solver.nz - (solver.cut_z_hi() ? halo : solver.padHi(0) + halo);
    bool cx = (ix < pml_x0 + halo) || (ix >= pml_x1 - halo);
    bool cz = (iz < pml_z0 + halo) || (iz >= pml_z1 - halo);

    float w_sum = 0.0f;

    if (cx) {
        float ax_ = cpml.ax[ix];
        float bx_ = cpml.bx[ix];
        float dbxdx_ = cpml.dbxdx[ix];
        float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
        long xi = solver.aux_rd_x2(iz, ix);
        float daipxix_dx = fused_d_aPsi<Order>(cpml.ax, f.psix, ix, (int)xi, grad_ctx.sx, halo, grad_ctx.coeff, grad_ctx.dx);
        float tmpx = ((1.0f+bx_)*lap_x + dbxdx_*dudx) + daipxix_dx;
        w_sum += (1.0f+bx_) * tmpx + ax_ * f.zetax[xi];
        if (solver.aux_x.stored(ix)) {
            (f.psixn ? f.psixn : f.psix)[xi]  = bx_ * dudx + ax_ * f.psix[xi];
            f.zetax[xi] = bx_ * tmpx + ax_ * f.zetax[xi];
        }
    } else {
        w_sum += lap_x;
    }

    if (cz) {
        float az_ = cpml.az[iz];
        float bz_ = cpml.bz[iz];
        float dbzdz_ = cpml.dbzdz[iz];
        float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
        long zi = solver.aux_rd_z2(iz, ix);
        float daipsiz_dz = fused_d_aPsi<Order>(cpml.az, f.psiz, iz, (int)zi, grad_ctx.sz, halo, grad_ctx.coeff, grad_ctx.dz);
        float tmpz = ((1.0f+bz_)*lap_z + dbzdz_*dudz) + daipsiz_dz;
        w_sum += (1.0f+bz_) * tmpz + az_ * f.zetaz[zi];
        if (solver.aux_z.stored(iz)) {
            (f.psizn ? f.psizn : f.psiz)[zi]  = bz_ * dudz + az_ * f.psiz[zi];
            f.zetaz[zi] = bz_ * tmpz + az_ * f.zetaz[zi];
        }
    } else {
        w_sum += lap_z;
    }

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        v2_dt2 * w_sum;

    // The vp image is the update actually applied, v^2 * w_sum: on the halo
    // cells next to a band (now on this branch) w_sum is not the bare
    // Laplacian, and boundary saving's u_tt images exactly this.
    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;
}

#if defined(SWEEP_HOST_CORE)
// Host row launcher of a2d_fwd_interior, for SWEEP_LAUNCH_ROWS in launch_fwd
// (launch/cell_path.h).  The kernel has no PML branch, so every virtual
// thread that passes its two guards runs its interior path: the span is the
// launch box, each row its whole x extent, minus the rows past the last shot
// and -- with a surface -- the rows (iz) above the deepest surface row over
// the box's columns (common/host_rows.h box_launch_rows), where the air test
// must run.  The box itself carries every other geometry feature (CPML band
// plus halo, DD cut faces, per-edge pads and free surfaces, the x_base /
// x_end range: launch_fwd builds it), and box_cell_at decodes a virtual
// thread exactly as the kernel's box_cell does.  Optional output: u_this.
// Order -1 (runtime M) takes the same span (the box does not depend on the
// order); its row loop does not vectorise.  No arithmetic of its own; the cell
// lambda captures BY VALUE.
template<int Order>
struct A2dFwdInteriorRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, float* u_this,
                    const float* vp, LaplaceParam lap_ctx, SolverContext solver, Box box)
    {
        auto call = [&](auto&&... a) { a2d_fwd_interior<Order>(a...); };
        sweep::host::with_optional(u_this, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, sweep::host::box_launch_rows(box, solver),
                [=](int gx, int gy, int gz) {
                    int ix, iy, iz, b;
                    sweep::host::box_cell_at(box, gx, gy, gz, ix, iy, iz, b);
                    a2d_fwd_interior_cell<Order, decltype(path)::value>(
                        Acoustic2dCellIndex{ix, iz, b}, wf, u_this, vp, lap_ctx, solver);
                },
                call,
                wf, u_this, vp, lap_ctx, solver, box);
        });
    }
};
#endif

struct FwdArgs {
    AcousticWavefieldPointer wf; float* u_this; const float* vp;
    LaplaceParam lap; GradParam g, gx, gz; AcousticCPMLPointer cpml;
};

template<int Order>
void launch_fwd(const FwdArgs& a, const SolverContext& ctx)
{
    const Box all = fdtd::computed_box(ctx);
    // the cells no axis' CPML terms reach: the coefficient band plus halo (the
    // d(a psi) taps and np.gradient's dbdx reach halo cells past the band; the
    // adjoint's pure_interior uses the same bound).  Cut faces keep the halo.
    const Box in = fdtd::intersect(fdtd::phys_box(ctx, ctx.M), all);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        SWEEP_LAUNCH_ROWS((a2d_fwd_interior<Order>), A2dFwdInteriorRows<Order>::run, L.grid, L.block,
                          a.wf, a.u_this, a.vp, a.lap, ctx, in);
    }
    const fdtd::SlabSet ring = fdtd::slab_set(all, in);
    if (ring.n == 0) return;
    const auto L = fdtd::slab_set_launch(ring, ctx.B);
    SWEEP_LAUNCH((a2d_fwd_ring<Order>), L.grid, L.block, a.wf, a.u_this, a.vp, a.lap, a.g, a.gx, a.gz,
                                             a.cpml, ctx, ring);
}

}  // namespace

void launch_acoustic2d_forward(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    const FwdArgs a{wf, u_this, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml};
    switch (order) {
        case 2:  launch_fwd<2>(a, ctx); break;
        case 4:  launch_fwd<4>(a, ctx); break;
        case 6:  launch_fwd<6>(a, ctx); break;
        case 8:  launch_fwd<8>(a, ctx); break;
        default: launch_fwd<-1>(a, ctx); break;
    }
}

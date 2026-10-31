// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

namespace elastic_vr2d_kernels {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the stencil kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box -- its own test's
// interior, shrunk by the free-surface reach of the FS helpers its stencils
// go through (common/host_rows.h elastic_fs_box_2d) -- and the cell body to
// run there, with the z free surface as the ELASTIC_FS_Z flag.  This
// equation's FS is z only (its x derivatives are plain sgradients), so there
// is no ELASTIC_FS_X flag.  No kernel adds x_base, so the virtual thread gx
// is ix.  They hold no arithmetic of their own; the cell lambdas capture BY
// VALUE (see acoustic2d).
namespace {

static_assert(ELASTIC_FS_Z == 0, "with_evr_fs bit order");

// The FS helpers' half order (elastic_stencil_half_order), which is also the
// kernels' halo (solver.M for Order -1).
template<int Order>
int evr_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The kernels' halo guard: [halo, n - halo) on both axes.
template<int Order>
CellBox evr_halo_box(const SolverContext& s)
{
    const int h = evr_half<Order>(s);
    return {h, s.nx - h, h, s.nz - h};
}

// The interior of evr_momentum_kernel's and evr_stress_kernel's
// solver.in_pml_2d(ix, iz, halo) test (the physical box at halo == M),
// inside the halo box.
template<int Order>
CellBox evr_pml_box(const SolverContext& s)
{
    const int h = evr_half<Order>(s);
    return CellBox{s.phys_x0(h), s.phys_x1(h), s.phys_z0(h), s.phys_z1(h)}
        .intersect(evr_halo_box<Order>(s));
}

// The interior of the two adjoint prepares' legacy symmetric test
// (abcn + halo from every edge: not padLo, not cut-aware).
template<int Order>
CellBox evr_sym_pml_box(const SolverContext& s)
{
    const int w = s.abcn + evr_half<Order>(s);
    return {w, s.nx - w, w, s.nz - w};
}

// The nopml kernels' guard box: abcn + M + 1 from every edge, M at the top
// with a free surface.
template<int Order>
CellBox evr_nopml_box(const SolverContext& s)
{
    const int M = evr_half<Order>(s);
    const int h = s.abcn + M + 1;
    const int top = s.free_surface ? M : h;
    return {h, s.nx - h, top, s.nz - h};
}

// ``box`` without the forward-type (adjoint = false) or adjoint FS reach.
template<int Order>
CellBox evr_fs(CellBox box, const SolverContext& s, bool adjoint)
{
    return sweep::host::elastic_fs_box_2d(box, s, evr_half<Order>(s), adjoint);
}

// ``box`` without the surface rows themselves (elastic_is_top_free_surface_row):
// the forward margin of a one-cell reach.
CellBox evr_no_surface_row(CellBox box, const SolverContext& s)
{
    return sweep::host::elastic_fs_box_2d(box, s, 1, false);
}

// f(std::integral_constant<unsigned, Mask>) with bit ELASTIC_FS_Z = a z face
// is free, for add_flags(path, Mask).
template<class F>
void with_evr_fs(const SolverContext& s, F&& f)
{
    sweep::host::with_flags(f, s.fsLo(0) || s.fsHi(0));
}

template<int Order>
struct EvrMomentumRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_momentum_kernel<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_pml_box<Order>(solver), solver, false);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    evr_momentum_cell<Order, P>(ix, iz, b, wf, grad_ctx, cpml, solver);
                },
                call,
                wf, grad_ctx, cpml, solver);
        });
    }
};

template<int Order>
struct EvrStressRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* vp,
                    const float* vs, const float* Rp_x, const float* Rp_z, const float* Rs_x,
                    const float* Rs_z, float* u_this, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_stress_kernel<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_pml_box<Order>(solver), solver, false);
        with_evr_fs(solver, [&](auto fl) {
            sweep::host::with_optional(u_this, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        evr_stress_cell<Order, P>(ix, iz, b, wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z,
                                                  u_this, grad_ctx, cpml, solver);
                    },
                    call,
                    wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, u_this, grad_ctx, cpml, solver);
            });
        });
    }
};

template<int Order>
struct EvrMomentumNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_momentum_kernel_nopml<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_nopml_box<Order>(solver), solver, false);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    evr_momentum_nopml_cell<Order, P>(ix, iz, b, wf, grad_ctx, solver);
                },
                call,
                wf, grad_ctx, solver);
        });
    }
};

template<int Order>
struct EvrStressNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* vp,
                    const float* vs, const float* Rp_x, const float* Rp_z, const float* Rs_x,
                    const float* Rs_z, float* u_this, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_stress_kernel_nopml<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_nopml_box<Order>(solver), solver, false);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    evr_stress_nopml_cell<Order, P>(ix, iz, b, wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z,
                                                    grad_ctx, solver);
                },
                call,
                wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, u_this, grad_ctx, solver);
        });
    }
};

template<int Order>
struct EvrStressAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* vp,
                    const float* vs, const float* Rp_x, const float* Rp_z, const float* Rs_x,
                    const float* Rs_z, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver, float* qxx, float* qzz, float* qxz, float* qzx,
                    float* point_px, float* point_pz)
    {
        auto call = [&](auto&&... a) { evr_stress_adjoint_prepare<Order>(a...); };
        // No FS derivative helper here: the surface rows (the FS transpose)
        // are the only free-surface cells to keep out.
        const CellBox box = evr_no_surface_row(
            evr_sym_pml_box<Order>(solver).intersect(evr_halo_box<Order>(solver)), solver);
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                evr_stress_adjoint_prepare_cell<Order, CellPath::Interior>(
                    EvrCellIndex{ix, iz, b}, wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, grad_ctx, cpml,
                    solver, qxx, qzz, qxz, qzx, point_px, point_pz);
            },
            call,
            wf, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, grad_ctx, cpml, solver, qxx, qzz, qxz, qzx,
            point_px, point_pz);
    }
};

template<int Order>
struct EvrStressAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* qxx,
                    const float* qzz, const float* qxz, const float* qzx, const float* point_px,
                    const float* point_pz, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_stress_adjoint_apply<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_halo_box<Order>(solver), solver, true);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    evr_stress_adjoint_apply_cell<Order, P>(ix, iz, b, wf, qxx, qzz, qxz, qzx,
                                                            point_px, point_pz, grad_ctx, solver);
                },
                call,
                wf, qxx, qzz, qxz, qzx, point_px, point_pz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct EvrMomentumAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, ElasticCPMLPointer cpml,
                    SolverContext solver, float* pxx, float* pzz, float* pxz, float* pzx)
    {
        auto call = [&](auto&&... a) { evr_momentum_adjoint_prepare<Order>(a...); };
        // No free-surface logic: the in_pml test's interior.
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(evr_sym_pml_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                evr_momentum_adjoint_prepare_cell<Order, CellPath::Interior>(
                    EvrCellIndex{ix, iz, b}, wf, cpml, solver, pxx, pzz, pxz, pzx);
            },
            call,
            wf, cpml, solver, pxx, pzz, pxz, pzx);
    }
};

template<int Order>
struct EvrMomentumAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* pxx,
                    const float* pzz, const float* pxz, const float* pzx, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_momentum_adjoint_apply<Order>(a...); };
        const CellBox box = evr_fs<Order>(evr_halo_box<Order>(solver), solver, true);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    evr_momentum_adjoint_apply_cell<Order, P>(ix, iz, b, wf, pxx, pzz, pxz, pzx,
                                                              grad_ctx, solver);
                },
                call,
                wf, pxx, pzz, pxz, pzx, grad_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradEvrNobsRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer adjoint, const float* fpx,
                    const float* fpz, const float* fpx_prev, const float* fpz_prev,
                    const float* vp, const float* vs, const float* Rp_x, const float* Rp_z,
                    const float* Rs_x, const float* Rs_z, float* grad_vp, float* grad_vs,
                    float* grad_Rp_x, float* grad_Rp_z, float* grad_Rs_x, float* grad_Rs_z,
                    float* L_dVp_dx, float* L_dVp_dz, float* L_dVs_dx, float* L_dVs_dz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_evr_nobs<Order>(a...); };
        // The FS forward box also keeps the surface rows (the FS material
        // derivative) out.
        const CellBox box = evr_fs<Order>(evr_halo_box<Order>(solver), solver, false);
        with_evr_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    calculate_grad_evr_nobs_cell<Order, P>(
                        ix, iz, b, adjoint, fpx, fpz, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, grad_vp,
                        grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z, L_dVp_dx, L_dVp_dz,
                        L_dVs_dx, L_dVs_dz, grad_ctx, solver);
                },
                call,
                adjoint, fpx, fpz, fpx_prev, fpz_prev, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, grad_vp,
                grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z, L_dVp_dx, L_dVp_dz,
                L_dVs_dx, L_dVs_dz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct EvrGradChainApplyRows {
    static void run(dim3 grid, dim3 block, const float* L_dVp_dx, const float* L_dVp_dz,
                    const float* L_dVs_dx, const float* L_dVs_dz, float* grad_vp, float* grad_vs,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { evr_grad_chain_apply<Order>(a...); };
        // The guard box: h = Order / 2, or 2 for Order -1 (the order-4
        // fallback stencil) -- not solver.M.
        const int h = Order == -1 ? 2 : Order / 2;
        const CellBox box{h, solver.nx - h, h, solver.nz - h};
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                evr_grad_chain_apply_cell<Order, CellPath::Interior>(
                    ix, iz, b, L_dVp_dx, L_dVp_dz, L_dVs_dx, L_dVs_dz, grad_vp, grad_vs, grad_ctx,
                    solver);
            },
            call,
            L_dVp_dx, L_dVp_dz, L_dVs_dx, L_dVs_dz, grad_vp, grad_vs, grad_ctx, solver);
    }
};

}  // namespace

// evr_adjoint_zero_top_fs<0> writes only where elastic_is_top_free_surface_row
// holds -- top surface rows (one row flat, [min, max] of the topography rows
// otherwise) and the bottom surface row -- and returns everywhere else.  So
// instead of every virtual thread of the full grid, run its cell on those rows
// alone: (b, iz) rows over the OpenMP threads, ix in each.  A cell visited twice
// (a top range reaching the bottom row) stores the same zeros again.
void evr_adjoint_zero_top_fs_rows(dim3 grid, dim3 block, ElasticWavefieldPointer wf,
                                  SolverContext solver)
{
    if (!sweep::host::rows_enabled()) {
        sweep::host::launch(grid, block, [&](auto&&... a) { evr_adjoint_zero_top_fs<0>(a...); },
                            wf, solver);
        return;
    }
    // The kernel's index space: ix < grid.x * block.x, iz < grid.y * block.y,
    // b = blockIdx.z, and its bounds guard.
    const int nx = std::min<long long>(solver.nx, 1LL * grid.x * block.x);
    const int nz = std::min<long long>(solver.nz, 1LL * grid.y * block.y);
    const int nb = static_cast<int>(grid.z);
    auto zero_rows = [&](int z0, int z1) {   // rows [z0, z1)
        z0 = std::max(z0, 0);
        z1 = std::min(z1, nz);
        if (z1 <= z0 || nx <= 0 || nb <= 0) return;
        const int nr = z1 - z0;
        sweep::host::parallel_for(1LL * nb * nr, [&](long long r) {
            const int b = static_cast<int>(r / nr);
            const int iz = z0 + static_cast<int>(r % nr);
            for (int ix = 0; ix < nx; ++ix)
                evr_adjoint_zero_top_fs_cell<0>(ix, iz, b, wf, solver);
        });
    };
    if (solver.fsLo(0)) {
        int lo = solver.phys_z0(), hi = lo;   // surface_row(ix) over the columns
        if (solver.has_topo && nx > 0) {
            lo = hi = solver.topo_rows[0];
            for (int ix = 1; ix < nx; ++ix) {
                lo = std::min(lo, solver.topo_rows[ix]);
                hi = std::max(hi, solver.topo_rows[ix]);
            }
        }
        zero_rows(lo, hi + 1);
    }
    if (solver.fsHi(0)) {
        const int bot = solver.phys_z1() - 1;
        zero_rows(bot, bot + 1);
    }
}
#endif

SWEEP_BY_ORDER_TABLE_ROWS(evr_momentum_kernel_fn, evr_momentum_kernel_by_order, evr_momentum_kernel,
                          EvrMomentumRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_stress_kernel_fn, evr_stress_kernel_by_order, evr_stress_kernel,
                          EvrStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_momentum_kernel_nopml_fn, evr_momentum_kernel_nopml_by_order,
                          evr_momentum_kernel_nopml, EvrMomentumNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_stress_kernel_nopml_fn, evr_stress_kernel_nopml_by_order,
                          evr_stress_kernel_nopml, EvrStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_stress_adjoint_prepare_fn, evr_stress_adjoint_prepare_by_order,
                          evr_stress_adjoint_prepare, EvrStressAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_stress_adjoint_apply_fn, evr_stress_adjoint_apply_by_order,
                          evr_stress_adjoint_apply, EvrStressAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_momentum_adjoint_prepare_fn, evr_momentum_adjoint_prepare_by_order,
                          evr_momentum_adjoint_prepare, EvrMomentumAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_momentum_adjoint_apply_fn, evr_momentum_adjoint_apply_by_order,
                          evr_momentum_adjoint_apply, EvrMomentumAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_evr_nobs_fn, calculate_grad_evr_nobs_by_order,
                          calculate_grad_evr_nobs, CalculateGradEvrNobsRows);
SWEEP_BY_ORDER_TABLE_ROWS(evr_grad_chain_apply_fn, evr_grad_chain_apply_by_order,
                          evr_grad_chain_apply, EvrGradChainApplyRows);
evr_adjoint_zero_top_fs_fn const evr_adjoint_zero_top_fs_0 = &evr_adjoint_zero_top_fs<0>;
} // namespace elastic_vr2d_kernels

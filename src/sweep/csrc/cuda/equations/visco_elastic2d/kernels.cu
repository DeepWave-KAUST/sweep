// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the two visco kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box -- the
// visco_elastic2d_in_cpml test's interior, shrunk by the forward free-surface
// reach of the FS helpers its stencils go through (common/host_rows.h
// elastic_fs_box_2d, which also keeps every surface row/column out) -- and
// the cell body to run there, with the active FS axes as the ELASTIC_FS_Z /
// ELASTIC_FS_X flags (with_elastic_fs_2d) and the mechanism count m.L as the
// cells' LMech template argument: the per-mechanism loops
// ``for (l < 4) { if (l >= L) break; ... }`` guard stores, so an interior row
// needs L at compile time to run as a SIMD loop.  No visco kernel adds
// x_base, so the virtual thread gx is ix.  They hold no arithmetic of their
// own; the cell lambdas capture BY VALUE (see acoustic2d).  The velocity
// kernel and the three stencil transposes of the adjoint are elastic2d's
// (ported in elastic2d/kernels.cu).
namespace {

static_assert(ELASTIC_FS_Z == 0 && ELASTIC_FS_X == 1, "with_elastic_fs_2d bit order");

// The FS helpers' half order (elastic_stencil_half_order) and the kernels' halo.
template<int Order>
int visco_elastic2d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The interior of visco_elastic2d_in_cpml, inside the halo box [h, n - h)
// on both axes (the stress kernel's guard; the adjoint prepare runs every
// cell, so there the halo only shrinks the box).
template<int Order>
CellBox visco_elastic2d_cpml_box(const SolverContext& s)
{
    const int h = visco_elastic2d_half<Order>(s);
    return CellBox{s.cut_x_lo() ? h : s.padLo(2) + h, s.nx - (s.cut_x_hi() ? h : s.padHi(2) + h),
                   s.cut_z_lo() ? h : s.padLo(0) + h, s.nz - (s.cut_z_hi() ? h : s.padHi(0) + h)}
        .intersect(CellBox{h, s.nx - h, h, s.nz - h});
}

// ... without the forward-type FS reach (both kernels' derivative helpers are
// the forward ones; neither calls an adjoint FS helper).
template<int Order>
CellBox visco_elastic2d_box(const SolverContext& s)
{
    return sweep::host::elastic_fs_box_2d(visco_elastic2d_cpml_box<Order>(s), s,
                                          visco_elastic2d_half<Order>(s), false);
}

// ``f(std::integral_constant<int, L>)`` for the mechanism count L = 1..4,
// or with 0 for anything else (no interior path: every cell through the
// kernel, launch_through -- the same copy of the kernel body as the band
// cells of the row launches, since ``call`` is one closure for all paths).
template<class F>
void visco_elastic2d_with_mech(int L, F&& f)
{
    switch (L) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    default: f(std::integral_constant<int, 0>{}); break;
    }
}

template<int Order>
struct ViscoElastic2dStressRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, ViscoElastic2dMemory mem,
                    ViscoElastic2dModel m, float* u_this, float* fs_strain, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { visco_elastic2d_stress_kernel<Order>(a...); };
        // fs_strain is written on surface cells only: dead on the interior path.
        const CellBox box = visco_elastic2d_box<Order>(solver);
        visco_elastic2d_with_mech(m.L, [&](auto lm) {
            constexpr int LM = decltype(lm)::value;
            if constexpr (LM == 0) {
                sweep::host::launch_through(grid, block, call, wf, mem, m, u_this, fs_strain,
                                            grad_ctx, cpml, solver);
            } else {
                sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
                    sweep::host::with_optional(u_this, [&](auto op) {
                        constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                        sweep::host::launch_rows(
                            grid, block, sweep::host::box_rows(box),
                            [=](int ix, int iz, int b) {
                                visco_elastic2d_stress_cell<Order, P, LM>(
                                    ix, iz, b, wf, mem, m, u_this, fs_strain, grad_ctx, cpml, solver);
                            },
                            call, wf, mem, m, u_this, fs_strain, grad_ctx, cpml, solver);
                    });
                });
            }
        });
    }
};

template<int Order>
struct ViscoElastic2dStressAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, ViscoElastic2dMemory mem,
                    ViscoElastic2dModel m, ElasticCPMLPointer cpml, SolverContext solver,
                    float* qxx, float* qzz, float* qxz, float* qzx, SGradParam grad_ctx,
                    ViscoElastic2dGrad g)
    {
        auto call = [&](auto&&... a) { visco_elastic2d_stress_adjoint_prepare<Order>(a...); };
        // The surface rows/columns (traction and surface-solve transposes,
        // the fs_strain reads) and the FS reach of the imaging stencils are
        // out; imaging is inside the physical box.
        const CellBox box = visco_elastic2d_box<Order>(solver).intersect(
            CellBox{solver.phys_x0(), solver.phys_x1(), solver.phys_z0(), solver.phys_z1()});
        // The one call site (DriverL::image_standalone) always images, so only
        // the InteriorOut path is instantiated (each path costs the TU a copy
        // of the whole kernel); without g every cell runs the kernel itself.
        const int L = g.g_lam != nullptr ? m.L : 0;
        visco_elastic2d_with_mech(L, [&](auto lm) {
            constexpr int LM = decltype(lm)::value;
            if constexpr (LM == 0) {
                sweep::host::launch_through(grid, block, call, wf, mem, m, cpml, solver, qxx, qzz,
                                            qxz, qzx, grad_ctx, g);
            } else {
                sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
                    constexpr CellPath P = add_flags(CellPath::InteriorOut, decltype(fl)::value);
                    sweep::host::launch_rows(
                        grid, block, sweep::host::box_rows(box),
                        [=](int ix, int iz, int b) {
                            visco_elastic2d_stress_adjoint_prepare_cell<Order, P, LM>(
                                ix, iz, b, wf, mem, m, cpml, solver, qxx, qzz, qxz, qzx,
                                grad_ctx, g);
                        },
                        call, wf, mem, m, cpml, solver, qxx, qzz, qxz, qzx, grad_ctx, g);
                });
            }
        });
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(visco_elastic2d_stress_kernel_fn, visco_elastic2d_stress_kernel_by_order,
                          visco_elastic2d_stress_kernel, ViscoElastic2dStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(visco_elastic2d_stress_adjoint_prepare_fn,
                          visco_elastic2d_stress_adjoint_prepare_by_order,
                          visco_elastic2d_stress_adjoint_prepare,
                          ViscoElastic2dStressAdjointPrepareRows);

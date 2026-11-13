// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

namespace elastic_tti_sg2d {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the nine table kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box in virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b) -- box_rows; no x_base, no
// topography (setup_ctx is a no-op for this equation) -- and the cell body
// to run there.  A stencil whose z derivatives go through the free-surface
// helpers gets its box shrunk by their reach (common/host_rows.h
// elastic_fs_box_2d) and the z face as the ELASTIC_FS_Z flag.  They hold no
// arithmetic of their own; the cell lambdas capture BY VALUE (see
// acoustic2d).
namespace {

static_assert(ELASTIC_FS_Z == 0, "sg2d_with_fs sets bit 0 for ELASTIC_FS_Z");

// The stencils' x stride is 1 at every launch (driver_traits.cuh make_state:
// SGradParam{1, 0, nx, ...}).  An interior row gets a copy with sx a
// compile-time 1, so its x taps are unit-stride loads the vectoriser can use;
// ix * 1 == ix * sx, so the cell forms the same indices.  The copy is made in
// the cell lambda, where the constant reaches the inlined stencils; a launch
// with any other stride gives the interior rows an empty box (every cell
// through the kernel).
inline SGradParam unit_sx(SGradParam g)
{
    g.sx = 1;
    return g;
}

inline CellBox unit_sx_box(const SGradParam& g, CellBox box)
{
    return g.sx == 1 ? box : CellBox{0, 0, 0, 0};
}

// The kernels' half order: ``halo`` of the stencil kernels and the FS
// helpers' ``M`` (elastic_stencil_half_order).
template<int Order>
int sg2d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The stencil kernels' halo guard: [halo, n - halo) on both axes.
template<int Order>
CellBox sg2d_halo_box(const SolverContext& s)
{
    const int h = sg2d_half<Order>(s);
    return {h, s.nx - h, h, s.nz - h};
}

// The interior of solver.in_pml_2d(ix, iz, halo): the velocity kernel and
// both adjoint prepares.
template<int Order>
CellBox sg2d_pml_box(const SolverContext& s)
{
    const int h = sg2d_half<Order>(s);
    return {s.phys_x0(h), s.phys_x1(h), s.phys_z0(h), s.phys_z1(h)};
}

// The interior of the stress kernel's LEGACY in_pml test, as that kernel
// spells it (elastic_tti_sg_stress_in_pml): abcn + halo from the x faces and
// the bottom, ``free_surface ? halo + 1 : abcn + halo`` from the top.
template<int Order>
CellBox sg2d_stress_box(const SolverContext& s)
{
    const int h = sg2d_half<Order>(s);
    const int top_pml = s.free_surface ? (h + 1) : (s.abcn + h);
    return {s.abcn + h, s.nx - s.abcn - h, top_pml, s.nz - s.abcn - h};
}

// The two _nopml kernels' guard: abcn + M + 1 from every face, M from the
// top with a free surface.
template<int Order>
CellBox sg2d_nopml_box(const SolverContext& s)
{
    const int M = sg2d_half<Order>(s);
    const int halo = s.abcn + M + 1;
    const int top_halo = s.free_surface ? M : halo;
    return {halo, s.nx - halo, top_halo, s.nz - halo};
}

// calculate_grad's physical-box guard.
CellBox sg2d_phys_box(const SolverContext& s)
{
    return {s.phys_x0(), s.phys_x1(), s.phys_z0(), s.phys_z1()};
}

// ``box`` without the forward-type (adjoint = false) or adjoint FS reach.
template<int Order>
CellBox sg2d_fs(CellBox box, const SolverContext& s, bool adjoint)
{
    return sweep::host::elastic_fs_box_2d(box, s, sg2d_half<Order>(s), adjoint);
}

// ``box`` without the free-surface rows of the legacy
// elastic_is_top_free_surface_row(solver, iz) overload (phys_z0() with a
// z-low face, phys_z1() - 1 with a z-high one): the pointwise stress-adjoint
// prepare reads no neighbour, so only those rows leave its box.
CellBox sg2d_no_fs_rows(CellBox box, const SolverContext& s)
{
    if (s.fsLo(0) && box.z0 < s.phys_z0() + 1) box.z0 = s.phys_z0() + 1;
    if (s.fsHi(0) && box.z1 > s.phys_z1() - 1) box.z1 = s.phys_z1() - 1;
    return box;
}

// f(integral_constant<unsigned, Mask>) with ELASTIC_FS_Z (bit 0) set when a
// z face is free, for add_flags (no x derivative here goes through an FS
// helper, so ELASTIC_FS_X is never needed).
template<class F>
void sg2d_with_fs(const SolverContext& s, F&& f)
{
    sweep::host::with_flags(f, s.fsLo(0) || s.fsHi(0));
}

template<int Order>
struct TtiSgVelocityRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_velocity_kernel<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, sg2d_fs<Order>(sg2d_pml_box<Order>(solver).intersect(sg2d_halo_box<Order>(solver)),
                                     solver, false));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg_velocity_cell<Order, P>(CellIndex2{ix, iz, b}, wf, model, gc, cpml,
                                                           solver);
                },
                call,
                wf, model, grad_ctx, cpml, solver);
        });
    }
};

template<int Order>
struct TtiSgStressRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    float* u_this, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_stress_kernel<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, sg2d_fs<Order>(sg2d_stress_box<Order>(solver).intersect(sg2d_halo_box<Order>(solver)),
                                     solver, false));
        sg2d_with_fs(solver, [&](auto fl) {
            sweep::host::with_optional(u_this, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        const SGradParam gc = unit_sx(grad_ctx);
                        elastic_tti_sg_stress_cell<Order, P>(CellIndex2{ix, iz, b}, wf, model, u_this,
                                                             gc, cpml, solver);
                    },
                    call,
                    wf, model, u_this, grad_ctx, cpml, solver);
            });
        });
    }
};

template<int Order>
struct TtiSgVelocityNopmlRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_velocity_kernel_nopml<Order>(a...); };
        const CellBox box =
            unit_sx_box(grad_ctx, sg2d_fs<Order>(sg2d_nopml_box<Order>(solver), solver, false));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg_velocity_nopml_cell<Order, P>(CellIndex2{ix, iz, b}, wf, model, gc,
                                                                 solver);
                },
                call,
                wf, model, grad_ctx, solver);
        });
    }
};

template<int Order>
struct TtiSgStressNopmlRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_stress_kernel_nopml<Order>(a...); };
        const CellBox box =
            unit_sx_box(grad_ctx, sg2d_fs<Order>(sg2d_nopml_box<Order>(solver), solver, false));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg_stress_nopml_cell<Order, P>(CellIndex2{ix, iz, b}, wf, model, gc,
                                                               solver);
                },
                call,
                wf, model, grad_ctx, solver);
        });
    }
};

template<int Order>
struct TtiSgStressAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SolverContext solver, float* q_vxx, float* q_vyx,
                    float* q_vzx, float* q_vxz, float* q_vyz, float* q_vzz)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_stress_adjoint_prepare<Order>(a...); };
        const CellBox box = sg2d_no_fs_rows(sg2d_pml_box<Order>(solver), solver);
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                elastic_tti_sg_stress_adjoint_prepare_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, cpml, solver, q_vxx, q_vyx, q_vzx, q_vxz,
                    q_vyz, q_vzz);
            },
            call,
            wf, model, cpml, solver, q_vxx, q_vyx, q_vzx, q_vxz, q_vyz, q_vzz);
    }
};

template<int Order>
struct TtiSgStressAdjointApplyRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, const float* q_vxx,
                    const float* q_vyx, const float* q_vzx, const float* q_vxz,
                    const float* q_vyz, const float* q_vzz, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_stress_adjoint_apply<Order>(a...); };
        const CellBox box =
            unit_sx_box(grad_ctx, sg2d_fs<Order>(sg2d_halo_box<Order>(solver), solver, true));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg_stress_adjoint_apply_cell<Order, P>(
                        CellIndex2{ix, iz, b}, wf, q_vxx, q_vyx, q_vzx, q_vxz, q_vyz, q_vzz, gc,
                        solver);
                },
                call,
                wf, q_vxx, q_vyx, q_vzx, q_vxz, q_vyz, q_vzz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct TtiSgVelocityAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SolverContext solver, float* q_txxx, float* q_txzz,
                    float* q_txyx, float* q_tyzz, float* q_txzx, float* q_tzzz)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_velocity_adjoint_prepare<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(sg2d_pml_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                elastic_tti_sg_velocity_adjoint_prepare_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, cpml, solver, q_txxx, q_txzz, q_txyx,
                    q_tyzz, q_txzx, q_tzzz);
            },
            call,
            wf, model, cpml, solver, q_txxx, q_txzz, q_txyx, q_tyzz, q_txzx, q_tzzz);
    }
};

template<int Order>
struct TtiSgVelocityAdjointApplyRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, const float* q_txxx,
                    const float* q_txzz, const float* q_txyx, const float* q_tyzz,
                    const float* q_txzx, const float* q_tzzz, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg_velocity_adjoint_apply<Order>(a...); };
        const CellBox box =
            unit_sx_box(grad_ctx, sg2d_fs<Order>(sg2d_halo_box<Order>(solver), solver, true));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg_velocity_adjoint_apply_cell<Order, P>(
                        CellIndex2{ix, iz, b}, wf, q_txxx, q_txzz, q_txyx, q_tyzz, q_txzx, q_tzzz,
                        gc, solver);
                },
                call,
                wf, q_txxx, q_txzz, q_txyx, q_tyzz, q_txzx, q_tzzz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct TtiSgGradRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer adjoint, StiffnessPointer model,
                    StiffnessGradPointer grad, const float* fvx, const float* fvy,
                    const float* fvz, const float* fvx_next, const float* fvy_next,
                    const float* fvz_next, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_elastic_tti_sg_nobs<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, sg2d_fs<Order>(sg2d_phys_box(solver).intersect(sg2d_halo_box<Order>(solver)),
                                     solver, false));
        sg2d_with_fs(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    calculate_grad_elastic_tti_sg_nobs_cell<Order, P>(
                        CellIndex2{ix, iz, b}, adjoint, model, grad, fvx, fvy, fvz, fvx_next,
                        fvy_next, fvz_next, gc, solver);
                },
                call,
                adjoint, model, grad, fvx, fvy, fvz, fvx_next, fvy_next, fvz_next, grad_ctx,
                solver);
        });
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_velocity_kernel_fn, elastic_tti_sg_velocity_kernel_by_order, elastic_tti_sg_velocity_kernel, TtiSgVelocityRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_stress_kernel_fn, elastic_tti_sg_stress_kernel_by_order, elastic_tti_sg_stress_kernel, TtiSgStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_velocity_kernel_nopml_fn, elastic_tti_sg_velocity_kernel_nopml_by_order, elastic_tti_sg_velocity_kernel_nopml, TtiSgVelocityNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_stress_kernel_nopml_fn, elastic_tti_sg_stress_kernel_nopml_by_order, elastic_tti_sg_stress_kernel_nopml, TtiSgStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_stress_adjoint_prepare_fn, elastic_tti_sg_stress_adjoint_prepare_by_order, elastic_tti_sg_stress_adjoint_prepare, TtiSgStressAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_stress_adjoint_apply_fn, elastic_tti_sg_stress_adjoint_apply_by_order, elastic_tti_sg_stress_adjoint_apply, TtiSgStressAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_velocity_adjoint_prepare_fn, elastic_tti_sg_velocity_adjoint_prepare_by_order, elastic_tti_sg_velocity_adjoint_prepare, TtiSgVelocityAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg_velocity_adjoint_apply_fn, elastic_tti_sg_velocity_adjoint_apply_by_order, elastic_tti_sg_velocity_adjoint_apply, TtiSgVelocityAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_elastic_tti_sg_nobs_fn, calculate_grad_elastic_tti_sg_nobs_by_order, calculate_grad_elastic_tti_sg_nobs, TtiSgGradRows);
} // namespace elastic_tti_sg2d

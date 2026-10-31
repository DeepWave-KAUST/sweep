// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

namespace elastic_tti_sg3d {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the nine table kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box as Wave3D row spans
// (common/host_rows.h: gx = ix, gy = iy, gz = b * nz + iz; no x_base, no
// topography -- setup_ctx is a no-op for this equation) and the cell body to
// run there.  They hold no arithmetic of their own; the cell lambdas capture
// BY VALUE (see acoustic2d).
namespace {

// Two facts every launch of this equation provides, which a row's interior
// loop needs at compile time to vectorise.  Each launcher checks them per
// launch and gives the interior rows an empty box when either fails, so every
// cell then runs through the kernel; neither changes a single operation of
// the cell.
//
//  * The x stride of the stencils is 1 (driver_traits.cuh make_state:
//    SGradParam{1, nx, nx * ny, ...}).  With sx a compile-time 1 the x taps
//    of a row are unit-stride loads; ix * 1 == ix * sx, so the cell forms the
//    same indices.  The pinned copy is made in the cell lambda, where the
//    constant reaches the inlined stencils.
//  * The 3-D-only fields the cells read (vy, syy, sxy, syz) are bound.
//    ElasticWavefieldPointer::offset (common/elastic.h) moves each of them
//    only when it is non-null -- a test the 2-D equations need -- and GCC
//    does not hoist that conditional out of the row loop, so those streams'
//    bases are not affine and the loop does not vectorise.  Saying in the
//    cell lambda that they are bound lets the tests fold.
inline SGradParam unit_sx(SGradParam g)
{
    g.sx = 1;
    return g;
}

inline bool fields_3d_bound(const ElasticWavefieldPointer& w)
{
    return w.vy != nullptr && w.syy != nullptr && w.sxy != nullptr && w.syz != nullptr;
}

inline void assume_fields_3d_bound(const ElasticWavefieldPointer& w)
{
    if (!fields_3d_bound(w))
        __builtin_unreachable();
}

// ``box`` when the launch provides both facts, else nothing.
inline CellBox3 rows_box(const SGradParam* g, const ElasticWavefieldPointer& w, CellBox3 box)
{
    const bool ok = (g == nullptr || g->sx == 1) && fields_3d_bound(w);
    return ok ? box : CellBox3{0, 0, 0, 0, 0, 0};
}

template<int Order>
struct TtiSg3dVelocityRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_velocity_kernel<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf,
                                      elastic_tti_sg3d_interior_box(solver).intersect(
                                          elastic_tti_sg3d_halo_box<Order>(solver)));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                const SGradParam gc = unit_sx(grad_ctx);
                elastic_tti_sg3d_velocity_kernel_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, rho, gc, cpml, solver);
            },
            call,
            wf, rho, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct TtiSg3dStressRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, StiffnessPointer model,
                    float* u_this_t, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_stress_kernel<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf,
                                      elastic_tti_sg3d_interior_box(solver).intersect(
                                          elastic_tti_sg3d_halo_box<Order>(solver)));
        sweep::host::with_optional(u_this_t, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, sweep::host::wave3d_rows(box, solver, 0),
                [=](int gx, int gy, int gz) {
                    assume_fields_3d_bound(wf);
                    const SGradParam gc = unit_sx(grad_ctx);
                    elastic_tti_sg3d_stress_kernel_cell<Order, decltype(path)::value>(
                        CellIndex3{gx, gy, gz % nz, gz / nz}, wf, model, u_this_t, gc, cpml,
                        solver);
                },
                call,
                wf, model, u_this_t, grad_ctx, cpml, solver);
        });
    }
};

template<int Order>
struct TtiSg3dVelocityNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_velocity_kernel_nopml<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf, elastic_tti_sg3d_nopml_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                const SGradParam gc = unit_sx(grad_ctx);
                elastic_tti_sg3d_velocity_kernel_nopml_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, rho, gc, solver);
            },
            call,
            wf, rho, grad_ctx, solver);
    }
};

template<int Order>
struct TtiSg3dStressNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, StiffnessPointer model,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_stress_kernel_nopml<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf, elastic_tti_sg3d_nopml_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                const SGradParam gc = unit_sx(grad_ctx);
                elastic_tti_sg3d_stress_kernel_nopml_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, model, gc, solver);
            },
            call,
            wf, model, grad_ctx, solver);
    }
};

template<int Order>
struct TtiSg3dStressAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SolverContext solver, float* qxx, float* qxy,
                    float* qxz, float* qyx, float* qyy, float* qyz, float* qzx, float* qzy,
                    float* qzz)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_stress_adjoint_prepare<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(nullptr, wf, elastic_tti_sg3d_interior_box(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                elastic_tti_sg3d_stress_adjoint_prepare_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, model, cpml, solver, qxx, qxy, qxz,
                    qyx, qyy, qyz, qzx, qzy, qzz);
            },
            call,
            wf, model, cpml, solver, qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz);
    }
};

template<int Order>
struct TtiSg3dStressAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* qxx,
                    const float* qxy, const float* qxz, const float* qyx, const float* qyy,
                    const float* qyz, const float* qzx, const float* qzy, const float* qzz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_stress_adjoint_apply<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf, elastic_tti_sg3d_halo_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                const SGradParam gc = unit_sx(grad_ctx);
                elastic_tti_sg3d_stress_adjoint_apply_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, qxx, qxy, qxz, qyx, qyy, qyz, qzx,
                    qzy, qzz, gc, solver);
            },
            call,
            wf, qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz, grad_ctx, solver);
    }
};

template<int Order>
struct TtiSg3dVelocityAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    ElasticCPMLPointer cpml, SolverContext solver, float* pxx, float* pxy,
                    float* pxz, float* pyx, float* pyy, float* pyz, float* pzx, float* pzy,
                    float* pzz)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_velocity_adjoint_prepare<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(nullptr, wf, elastic_tti_sg3d_interior_box(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                elastic_tti_sg3d_velocity_adjoint_prepare_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, rho, cpml, solver, pxx, pxy, pxz,
                    pyx, pyy, pyz, pzx, pzy, pzz);
            },
            call,
            wf, rho, cpml, solver, pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz);
    }
};

template<int Order>
struct TtiSg3dVelocityAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* pxx,
                    const float* pxy, const float* pxz, const float* pyx, const float* pyy,
                    const float* pyz, const float* pzx, const float* pzy, const float* pzz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_tti_sg3d_velocity_adjoint_apply<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, wf, elastic_tti_sg3d_halo_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(wf);
                const SGradParam gc = unit_sx(grad_ctx);
                elastic_tti_sg3d_velocity_adjoint_apply_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, wf, pxx, pxy, pxz, pyx, pyy, pyz, pzx,
                    pzy, pzz, gc, solver);
            },
            call,
            wf, pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz, grad_ctx, solver);
    }
};

template<int Order>
struct TtiSg3dGradRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer adjoint,
                    StiffnessPointer model, StiffnessGradPointer grad, const float* fvx,
                    const float* fvy, const float* fvz, const float* fvx_next,
                    const float* fvy_next, const float* fvz_next, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_elastic_tti_sg3d_nobs<Order>(a...); };
        const int nz = solver.nz;
        const CellBox3 box = rows_box(&grad_ctx, adjoint, elastic_tti_sg3d_halo_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(box, solver, 0),
            [=](int gx, int gy, int gz) {
                assume_fields_3d_bound(adjoint);
                const SGradParam gc = unit_sx(grad_ctx);
                calculate_grad_elastic_tti_sg3d_nobs_cell<Order, CellPath::Interior>(
                    CellIndex3{gx, gy, gz % nz, gz / nz}, adjoint, model, grad, fvx, fvy, fvz,
                    fvx_next, fvy_next, fvz_next, gc, solver);
            },
            call,
            adjoint, model, grad, fvx, fvy, fvz, fvx_next, fvy_next, fvz_next, grad_ctx, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_velocity_kernel_fn, elastic_tti_sg3d_velocity_kernel_by_order, elastic_tti_sg3d_velocity_kernel, TtiSg3dVelocityRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_stress_kernel_fn, elastic_tti_sg3d_stress_kernel_by_order, elastic_tti_sg3d_stress_kernel, TtiSg3dStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_velocity_kernel_nopml_fn, elastic_tti_sg3d_velocity_kernel_nopml_by_order, elastic_tti_sg3d_velocity_kernel_nopml, TtiSg3dVelocityNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_stress_kernel_nopml_fn, elastic_tti_sg3d_stress_kernel_nopml_by_order, elastic_tti_sg3d_stress_kernel_nopml, TtiSg3dStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_stress_adjoint_prepare_fn, elastic_tti_sg3d_stress_adjoint_prepare_by_order, elastic_tti_sg3d_stress_adjoint_prepare, TtiSg3dStressAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_stress_adjoint_apply_fn, elastic_tti_sg3d_stress_adjoint_apply_by_order, elastic_tti_sg3d_stress_adjoint_apply, TtiSg3dStressAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_velocity_adjoint_prepare_fn, elastic_tti_sg3d_velocity_adjoint_prepare_by_order, elastic_tti_sg3d_velocity_adjoint_prepare, TtiSg3dVelocityAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_tti_sg3d_velocity_adjoint_apply_fn, elastic_tti_sg3d_velocity_adjoint_apply_by_order, elastic_tti_sg3d_velocity_adjoint_apply, TtiSg3dVelocityAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_elastic_tti_sg3d_nobs_fn, calculate_grad_elastic_tti_sg3d_nobs_by_order, calculate_grad_elastic_tti_sg3d_nobs, TtiSg3dGradRows);
} // namespace elastic_tti_sg3d

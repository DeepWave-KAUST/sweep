// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

namespace acoustic_vti_1st_2d {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the seven stencil kernels (launch/cell_path.h): each
// hands host/launch.h launch_rows the kernel's interior box in virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b) -- no x_base, no topography here --
// and the cell body to run there.  They hold no arithmetic of their own.  The
// cell lambdas capture BY VALUE (see acoustic2d).
//
// sgradient steps x by the run-time grad_ctx.sx, which every driver of this
// equation sets to 1 (SGradParam{1, 0, nx, ...}).  A stride the compiler
// cannot see turns the x taps of a row into strided element loads, so the
// cell lambdas hand the cell a copy with sx = 1 written in -- and when sx is
// NOT 1 the span is empty and every cell runs through the kernel as before.
namespace {

inline auto stencil_rows(const CellBox& box, const SGradParam& grad_ctx)
{
    return sweep::host::box_rows(grad_ctx.sx == 1 ? box : CellBox{0, 0, 0, 0});
}

inline SGradParam unit_sx(SGradParam g)
{
    g.sx = 1;
    return g;
}

template<int Order>
struct VelocityRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer wf, const float* inv_rho,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { velocity_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_pml_interior_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                velocity_kernel_cell<Order, CellPath::Interior>(ix, iz, b, wf, inv_rho,
                                                                unit_sx(grad_ctx), cpml.axh, cpml.bxh,
                                                                cpml.azh, cpml.bzh, solver);
            },
            call,
            wf, inv_rho, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct StressRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer wf, const float* c11,
                    const float* c33, const float* c13, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { stress_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_pml_interior_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                stress_kernel_cell<Order, CellPath::Interior>(ix, iz, b, wf, c11, c33, c13,
                                                              unit_sx(grad_ctx), cpml.ax, cpml.bx,
                                                              cpml.az, cpml.bz, solver);
            },
            call,
            wf, c11, c33, c13, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct VelocityNopmlRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer wf, const float* inv_rho,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { velocity_kernel_nopml<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_nopml_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                velocity_kernel_nopml_cell<Order, CellPath::Interior>(ix, iz, b, wf, inv_rho,
                                                                      unit_sx(grad_ctx), solver);
            },
            call,
            wf, inv_rho, grad_ctx, solver);
    }
};

template<int Order>
struct StressNopmlRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer wf, const float* c11,
                    const float* c33, const float* c13, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { stress_kernel_nopml<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_nopml_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                stress_kernel_nopml_cell<Order, CellPath::Interior>(ix, iz, b, wf, c11, c33, c13,
                                                                    unit_sx(grad_ctx), solver);
            },
            call,
            wf, c11, c33, c13, grad_ctx, solver);
    }
};

template<int Order>
struct AdjointStressToVelRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer adj, const float* px,
                    const float* pz, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { adjoint_stress_to_vel_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_halo_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                adjoint_stress_to_vel_cell<Order, CellPath::Interior>(ix, iz, b, adj, px, pz,
                                                                      unit_sx(grad_ctx), solver);
            },
            call,
            adj, px, pz, grad_ctx, solver);
    }
};

template<int Order>
struct AdjointVelToStressRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer adj, const float* qx,
                    const float* qz, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { adjoint_vel_to_stress_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_halo_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                adjoint_vel_to_stress_cell<Order, CellPath::Interior>(ix, iz, b, adj, qx, qz,
                                                                      unit_sx(grad_ctx), solver);
            },
            call,
            adj, qx, qz, grad_ctx, solver);
    }
};

template<int Order>
struct CalculateGradRows {
    static void run(dim3 grid, dim3 block, const float* fvx_now, const float* fvz_now,
                    const float* fsH_prev, const float* fsV_prev, VTIWavefieldPointer adj,
                    const float* vp, const float* epsilon, const float* delta, const float* rho,
                    float* grad_vp, float* grad_eps, float* grad_delta, float* grad_rho,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti2d_grad_box<Order>(solver), grad_ctx),
            [=](int ix, int iz, int b) {
                calculate_grad_cell<Order, CellPath::Interior>(
                    ix, iz, b, fvx_now, fvz_now, fsH_prev, fsV_prev, adj, vp, epsilon, delta,
                    rho, grad_vp, grad_eps, grad_delta, grad_rho, unit_sx(grad_ctx), solver);
            },
            call,
            fvx_now, fvz_now, fsH_prev, fsV_prev, adj, vp, epsilon, delta, rho, grad_vp,
            grad_eps, grad_delta, grad_rho, grad_ctx, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(velocity_kernel_fn, velocity_kernel_by_order, velocity_kernel, VelocityRows);
SWEEP_BY_ORDER_TABLE_ROWS(stress_kernel_fn, stress_kernel_by_order, stress_kernel, StressRows);
SWEEP_BY_ORDER_TABLE_ROWS(velocity_kernel_nopml_fn, velocity_kernel_nopml_by_order, velocity_kernel_nopml, VelocityNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(stress_kernel_nopml_fn, stress_kernel_nopml_by_order, stress_kernel_nopml, StressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(adjoint_stress_to_vel_kernel_fn, adjoint_stress_to_vel_kernel_by_order, adjoint_stress_to_vel_kernel, AdjointStressToVelRows);
SWEEP_BY_ORDER_TABLE_ROWS(adjoint_vel_to_stress_kernel_fn, adjoint_vel_to_stress_kernel_by_order, adjoint_vel_to_stress_kernel, AdjointVelToStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_kernel_fn, calculate_grad_kernel_by_order, calculate_grad_kernel, CalculateGradRows);
} // namespace acoustic_vti_1st_2d

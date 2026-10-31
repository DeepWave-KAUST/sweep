// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

namespace acoustic_vti_1st_3d {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the seven stencil kernels (launch/cell_path.h): each
// hands host/launch.h launch_rows the kernel's interior box as Wave3D row
// spans (common/host_rows.h wave3d_rows: gx = ix, gy = iy, gz = b * nz + iz;
// no x_base, no topography here) and the cell body to run there.  They hold
// no arithmetic of their own.  The cell lambdas capture BY VALUE (see
// acoustic2d).
//
// sgradient steps x by the run-time grad_ctx.sx, which every driver of this
// equation sets to 1 (SGradParam{1, nx, nx*ny, ...}).  A stride the compiler
// cannot see turns the x taps of a row into strided element loads, so the
// cell lambdas hand the cell a copy with sx = 1 written in -- and when sx is
// NOT 1 the span is empty and every cell runs through the kernel as before.
namespace {

inline auto stencil_rows(const CellBox3& box, const SGradParam& grad_ctx,
                         const SolverContext& solver)
{
    return sweep::host::wave3d_rows(grad_ctx.sx == 1 ? box : CellBox3{0, 0, 0, 0, 0, 0},
                                    solver, 0);
}

inline SGradParam unit_sx(SGradParam g)
{
    g.sx = 1;
    return g;
}

template<int Order>
struct Velocity3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D wf, const float* inv_rho,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { velocity_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_pml_interior_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                velocity_kernel_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, inv_rho, unit_sx(grad_ctx), cpml.axh, cpml.bxh,
                    cpml.ayh, cpml.byh, cpml.azh, cpml.bzh, solver);
            },
            call,
            wf, inv_rho, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct Stress3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D wf, const float* c11,
                    const float* c33, const float* c13, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { stress_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_pml_interior_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                stress_kernel_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, c11, c33, c13, unit_sx(grad_ctx), cpml.ax,
                    cpml.bx, cpml.ay, cpml.by, cpml.az, cpml.bz, solver);
            },
            call,
            wf, c11, c33, c13, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct VelocityNopml3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D wf, const float* inv_rho,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { velocity_kernel_3d_nopml<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_nopml_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                velocity_kernel_3d_nopml_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, inv_rho, unit_sx(grad_ctx), solver);
            },
            call,
            wf, inv_rho, grad_ctx, solver);
    }
};

template<int Order>
struct StressNopml3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D wf, const float* c11,
                    const float* c33, const float* c13, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { stress_kernel_3d_nopml<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_nopml_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                stress_kernel_3d_nopml_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, c11, c33, c13, unit_sx(grad_ctx), solver);
            },
            call,
            wf, c11, c33, c13, grad_ctx, solver);
    }
};

template<int Order>
struct AdjointStressToVel3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D adj, const float* px,
                    const float* py, const float* pz, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { adjoint_stress_to_vel_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_halo_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                adjoint_stress_to_vel_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, adj, px, py, pz, unit_sx(grad_ctx), solver);
            },
            call,
            adj, px, py, pz, grad_ctx, solver);
    }
};

template<int Order>
struct AdjointVelToStress3DRows {
    static void run(dim3 grid, dim3 block, VTIWavefieldPointer3D adj, const float* qx,
                    const float* qy, const float* qz, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { adjoint_vel_to_stress_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_halo_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                adjoint_vel_to_stress_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, adj, qx, qy, qz, unit_sx(grad_ctx), solver);
            },
            call,
            adj, qx, qy, qz, grad_ctx, solver);
    }
};

template<int Order>
struct CalculateGrad3DRows {
    static void run(dim3 grid, dim3 block, const float* fvx_now, const float* fvy_now,
                    const float* fvz_now, const float* fsH_prev, const float* fsV_prev,
                    VTIWavefieldPointer3D adj, const float* vp, const float* epsilon,
                    const float* delta, const float* rho, float* grad_vp, float* grad_eps,
                    float* grad_delta, float* grad_rho, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, stencil_rows(vti3d_grad_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                calculate_grad_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, fvx_now, fvy_now, fvz_now, fsH_prev, fsV_prev, adj,
                    vp, epsilon, delta, rho, grad_vp, grad_eps, grad_delta, grad_rho,
                    unit_sx(grad_ctx), solver);
            },
            call,
            fvx_now, fvy_now, fvz_now, fsH_prev, fsV_prev, adj, vp, epsilon, delta, rho, grad_vp,
            grad_eps, grad_delta, grad_rho, grad_ctx, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(velocity_kernel_3d_fn, velocity_kernel_3d_by_order, velocity_kernel_3d, Velocity3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(stress_kernel_3d_fn, stress_kernel_3d_by_order, stress_kernel_3d, Stress3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(velocity_kernel_3d_nopml_fn, velocity_kernel_3d_nopml_by_order, velocity_kernel_3d_nopml, VelocityNopml3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(stress_kernel_3d_nopml_fn, stress_kernel_3d_nopml_by_order, stress_kernel_3d_nopml, StressNopml3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(adjoint_stress_to_vel_kernel_3d_fn, adjoint_stress_to_vel_kernel_3d_by_order, adjoint_stress_to_vel_kernel_3d, AdjointStressToVel3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(adjoint_vel_to_stress_kernel_3d_fn, adjoint_vel_to_stress_kernel_3d_by_order, adjoint_vel_to_stress_kernel_3d, AdjointVelToStress3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_kernel_3d_fn, calculate_grad_kernel_3d_by_order, calculate_grad_kernel_3d, CalculateGrad3DRows);
} // namespace acoustic_vti_1st_3d

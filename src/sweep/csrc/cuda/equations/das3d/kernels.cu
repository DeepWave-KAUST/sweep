// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the three split kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box as Wave3D row spans
// (common/host_rows.h: gx = ix, gy = iy, gz = b * nz + iz) and the cell body
// to run there.  They hold no arithmetic of their own; the cell lambdas
// capture BY VALUE (see acoustic2d).  No optional outputs, no free-surface
// switch, no topography, no x_base: nothing to dispatch.
namespace {

template<int Order>
struct Das3dFirstRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer3D wf, float* tmp_sxx_x,
                    float* tmp_syy_y, float* tmp_szz_z, float* tmp_txx_y, float* tmp_txx_z,
                    float* tmp_tyy_x, float* tmp_tyy_z, float* tmp_tzz_x, float* tmp_tzz_y,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das3d_first_derivatives_kernel<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(das3d_first_interior_box<Order>(solver), solver, 0),
            [=](int gx, int gy, int gz) {
                das3d_first_derivatives_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, tmp_sxx_x, tmp_syy_y, tmp_szz_z, tmp_txx_y, tmp_txx_z,
                    tmp_tyy_x, tmp_tyy_z, tmp_tzz_x, tmp_tzz_y, grad_ctx, cpml, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_syy_y, tmp_szz_z, tmp_txx_y, tmp_txx_z, tmp_tyy_x, tmp_tyy_z, tmp_tzz_x,
            tmp_tzz_y, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct Das3dUpdateRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer3D wf, const float* tmp_sxx_x,
                    const float* tmp_syy_y, const float* tmp_szz_z, const float* tmp_txx_y,
                    const float* tmp_txx_z, const float* tmp_tyy_x, const float* tmp_tyy_z,
                    const float* tmp_tzz_x, const float* tmp_tzz_y, const float* rho,
                    const float* lambda, const float* mu, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { das3d_update_kernel<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(das3d_update_interior_box<Order>(solver), solver, 0),
            [=](int gx, int gy, int gz) {
                das3d_update_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, tmp_sxx_x, tmp_syy_y, tmp_szz_z, tmp_txx_y, tmp_txx_z,
                    tmp_tyy_x, tmp_tyy_z, tmp_tzz_x, tmp_tzz_y, rho, lambda, mu, grad_ctx, cpml, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_syy_y, tmp_szz_z, tmp_txx_y, tmp_txx_z, tmp_tyy_x, tmp_tyy_z, tmp_tzz_x,
            tmp_tzz_y, rho, lambda, mu, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct Das3dProjectModelGradRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer3D adjoint, const float* exx_now,
                    const float* eyy_now, const float* ezz_now, const float* exx_prev,
                    const float* eyy_prev, const float* ezz_prev, const float* vp, const float* vs,
                    const float* rho, float* gvp, float* gvs, float* grho, float* q_dxx_sxx,
                    float* q_dyy_syy, float* q_dzz_szz, float* q_dyy_txx, float* q_dzz_txx,
                    float* q_dxx_tyy, float* q_dzz_tyy, float* q_dxx_tzz, float* q_dyy_tzz,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { das3d_project_model_grad_kernel<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(das3d_update_halo_box<Order>(solver), solver, 0),
            [=](int gx, int gy, int gz) {
                das3d_project_model_grad_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, adjoint, exx_now, eyy_now, ezz_now, exx_prev, eyy_prev,
                    ezz_prev, vp, vs, rho, gvp, gvs, grho, q_dxx_sxx, q_dyy_syy, q_dzz_szz, q_dyy_txx,
                    q_dzz_txx, q_dxx_tyy, q_dzz_tyy, q_dxx_tzz, q_dyy_tzz, solver);
            },
            call,
            adjoint, exx_now, eyy_now, ezz_now, exx_prev, eyy_prev, ezz_prev, vp, vs, rho, gvp, gvs, grho,
            q_dxx_sxx, q_dyy_syy, q_dzz_szz, q_dyy_txx, q_dzz_txx, q_dxx_tyy, q_dzz_tyy, q_dxx_tzz,
            q_dyy_tzz, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(das3d_first_derivatives_kernel_fn, das3d_first_derivatives_kernel_by_order,
                          das3d_first_derivatives_kernel, Das3dFirstRows);
SWEEP_BY_ORDER_TABLE_ROWS(das3d_update_kernel_fn, das3d_update_kernel_by_order, das3d_update_kernel,
                          Das3dUpdateRows);
SWEEP_BY_ORDER_TABLE_ROWS(das3d_project_model_grad_kernel_fn, das3d_project_model_grad_kernel_by_order,
                          das3d_project_model_grad_kernel, Das3dProjectModelGradRows);

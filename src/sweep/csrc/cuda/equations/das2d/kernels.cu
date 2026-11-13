// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the six split kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box in virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b) -- box_rows -- and the cell body to
// run there.  They hold no arithmetic of their own; the cell lambdas capture
// BY VALUE (see acoustic2d).  No DAS kernel has an optional output, a
// free-surface switch or topography, so there is nothing to dispatch.
namespace {

template<int Order>
struct Das2dFirstRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D wf, float* tmp_sxx_x,
                    float* tmp_szz_z, float* tmp_txx_z, float* tmp_tzz_x, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_first_derivatives_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_first_interior_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_first_derivatives_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, grad_ctx, cpml, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct Das2dUpdateRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D wf, const float* tmp_sxx_x,
                    const float* tmp_szz_z, const float* tmp_txx_z, const float* tmp_tzz_x,
                    const float* rho, const float* lambda, const float* mu, SGradParam grad_ctx,
                    ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_update_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_update_interior_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_update_cell<Order, CellPath::Interior>(ix, iz, b, wf, tmp_sxx_x, tmp_szz_z,
                                                             tmp_txx_z, tmp_tzz_x, rho, lambda, mu,
                                                             grad_ctx, cpml, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, rho, lambda, mu, grad_ctx, cpml, solver);
    }
};

template<int Order>
struct Das2dFirstNopmlRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D wf, float* tmp_sxx_x,
                    float* tmp_szz_z, float* tmp_txx_z, float* tmp_tzz_x, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_first_derivatives_nopml_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_nopml_halo_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_first_derivatives_nopml_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, grad_ctx, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, grad_ctx, solver);
    }
};

template<int Order>
struct Das2dReverseStrainNopmlRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D wf, const float* tmp_sxx_x,
                    const float* tmp_szz_z, const float* tmp_txx_z, const float* tmp_tzz_x,
                    const float* rho, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_reverse_strain_nopml_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_update_halo_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_reverse_strain_nopml_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, rho, grad_ctx, solver);
            },
            call,
            wf, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, rho, grad_ctx, solver);
    }
};

template<int Order>
struct Das2dReverseStressNopmlRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D wf, const float* lambda,
                    const float* mu, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_reverse_stress_nopml_kernel<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_update_halo_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_reverse_stress_nopml_cell<Order, CellPath::Interior>(ix, iz, b, wf, lambda, mu,
                                                                           solver);
            },
            call,
            wf, lambda, mu, solver);
    }
};

template<int Order>
struct Das2dProjectModelGradRows {
    static void run(dim3 grid, dim3 block, DasWavefieldPointer2D adjoint, const float* exx_now,
                    const float* ezz_now, const float* exx_prev, const float* ezz_prev,
                    const float* vp, const float* vs, const float* rho, float* grad_vp,
                    float* grad_vs, float* grad_rho, float* bar_dxx_sxx, float* bar_dzz_szz,
                    float* bar_dzz_txx, float* bar_dxx_tzz, SolverContext solver)
    {
        auto call = [&](auto&&... a) { das2d_project_model_grad_kernel<Order>(a...); };
        // The frame outside the active box (its das* zeroing) runs through the kernel.
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(das2d_update_halo_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das2d_project_model_grad_cell<Order, CellPath::Interior>(
                    ix, iz, b, adjoint, exx_now, ezz_now, exx_prev, ezz_prev, vp, vs, rho, grad_vp,
                    grad_vs, grad_rho, bar_dxx_sxx, bar_dzz_szz, bar_dzz_txx, bar_dxx_tzz, solver);
            },
            call,
            adjoint, exx_now, ezz_now, exx_prev, ezz_prev, vp, vs, rho, grad_vp, grad_vs, grad_rho,
            bar_dxx_sxx, bar_dzz_szz, bar_dzz_txx, bar_dxx_tzz, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(das2d_first_derivatives_kernel_fn, das2d_first_derivatives_kernel_by_order,
                          das2d_first_derivatives_kernel, Das2dFirstRows);
SWEEP_BY_ORDER_TABLE_ROWS(das2d_update_kernel_fn, das2d_update_kernel_by_order, das2d_update_kernel,
                          Das2dUpdateRows);
SWEEP_BY_ORDER_TABLE_ROWS(das2d_first_derivatives_nopml_kernel_fn,
                          das2d_first_derivatives_nopml_kernel_by_order,
                          das2d_first_derivatives_nopml_kernel, Das2dFirstNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(das2d_reverse_strain_nopml_kernel_fn,
                          das2d_reverse_strain_nopml_kernel_by_order,
                          das2d_reverse_strain_nopml_kernel, Das2dReverseStrainNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(das2d_reverse_stress_nopml_kernel_fn,
                          das2d_reverse_stress_nopml_kernel_by_order,
                          das2d_reverse_stress_nopml_kernel, Das2dReverseStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(das2d_project_model_grad_kernel_fn, das2d_project_model_grad_kernel_by_order,
                          das2d_project_model_grad_kernel, Das2dProjectModelGradRows);

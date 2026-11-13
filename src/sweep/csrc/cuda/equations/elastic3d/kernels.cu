// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the stencil kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box as Wave3D row spans
// (common/host_rows.h wave3d_rows: gx = ix, gy = iy, gz = b * nz + iz; no
// elastic kernel adds x_base) -- its own test's interior, below the rows the
// free-surface helpers mirror or fold (elastic3d_fs_box) -- and the cell body
// to run there, with solver.free_surface as the ELASTIC_FS_Z flag
// (with_elastic_fs_3d).  They hold no arithmetic of their own; the cell
// lambdas capture BY VALUE (see acoustic2d).  das_mu3d launches most of these
// kernels through the same tables.
namespace {

// The FS helper's half order (elastic3d_stencil_half_order).
template<int Order>
int elastic3d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The kernels' halo guard: [halo, n - halo) on every axis.
template<int Order>
CellBox3 elastic3d_halo_box(const SolverContext& s)
{
    const int h = elastic3d_half<Order>(s);
    return {h, s.nx - h, h, s.ny - h, h, s.nz - h};
}

// The kernels' ``interior`` test (physical box shrunk by one cell, the halo
// on a DD cut face; z is never cut), inside the halo box.
template<int Order>
CellBox3 elastic3d_interior_box(const SolverContext& s)
{
    const int h = elastic3d_half<Order>(s);
    return CellBox3{s.cut_x_lo() ? h : s.phys_x0() + 1, s.cut_x_hi() ? s.nx - h : s.phys_x1() - 1,
                    s.cut_y_lo() ? h : s.phys_y0() + 1, s.cut_y_hi() ? s.ny - h : s.phys_y1() - 1,
                    s.phys_z0() + 1, s.phys_z1() - 1}
        .intersect(elastic3d_halo_box<Order>(s));
}

// The nopml kernels' guard box.
template<int Order>
CellBox3 elastic3d_nopml_box(const SolverContext& s)
{
    const int M = elastic3d_half<Order>(s);
    const int wide = s.abcn + 1 * M + 1;
    const int hx_lo = s.cut_x_lo() ? M : wide, hx_hi = s.cut_x_hi() ? M : wide;
    const int hy_lo = s.cut_y_lo() ? M : wide, hy_hi = s.cut_y_hi() ? M : wide;
    const int top_halo = s.free_surface ? M : wide;
    return {hx_lo, s.nx - hx_hi, hy_lo, s.ny - hy_hi, top_halo, s.nz - wide};
}

CellBox3 elastic3d_phys_box(const SolverContext& s)
{
    return {s.phys_x0(), s.phys_x1(), s.phys_y0(), s.phys_y1(), s.phys_z0(), s.phys_z1()};
}

// ``box`` without the forward-type (adjoint = false) or adjoint FS reach, as
// launch_rows spans.
template<int Order>
auto elastic3d_rows(CellBox3 box, const SolverContext& s, bool adjoint)
{
    return sweep::host::wave3d_rows(sweep::host::elastic3d_fs_box(box, s, elastic3d_half<Order>(s), adjoint),
                                    s, 0);
}

template<int Order>
struct ElasticVelocity3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_velocity_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, elastic3d_rows<Order>(elastic3d_interior_box<Order>(solver), solver, false),
                [=](int gx, int gy, int gz) {
                    elastic_velocity_3d_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, rho, grad_ctx, cpml,
                                                       solver);
                },
                call,
                wf, rho, grad_ctx, cpml, solver);
        });
    }
};

template<int Order>
struct ElasticStress3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, float* u_this_t, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_kernel_3d<Order>(a...); };
        const int nz = solver.nz;
        const auto span = elastic3d_rows<Order>(elastic3d_interior_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            sweep::host::with_optional(u_this_t, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        elastic_stress_3d_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, lambda, mu,
                                                         u_this_t, grad_ctx, cpml, solver);
                    },
                    call,
                    wf, lambda, mu, u_this_t, grad_ctx, cpml, solver);
            });
        });
    }
};

template<int Order>
struct ElasticVelocity3DNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, SolverContext solver, float* fvx_prev, float* fvy_prev,
                    float* fvz_prev)
    {
        auto call = [&](auto&&... a) { elastic_velocity_kernel_3d_nopml<Order>(a...); };
        const int nz = solver.nz;
        const auto span = elastic3d_rows<Order>(elastic3d_nopml_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            sweep::host::with_optional(fvx_prev, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        elastic_velocity_3d_nopml_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, rho,
                                                                 grad_ctx, solver, fvx_prev, fvy_prev,
                                                                 fvz_prev);
                    },
                    call,
                    wf, rho, grad_ctx, solver, fvx_prev, fvy_prev, fvz_prev);
            });
        });
    }
};

template<int Order>
struct ElasticStress3DNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_kernel_3d_nopml<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, elastic3d_rows<Order>(elastic3d_nopml_box<Order>(solver), solver, false),
                [=](int gx, int gy, int gz) {
                    elastic_stress_3d_nopml_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, lambda, mu,
                                                           grad_ctx, solver);
                },
                call,
                wf, lambda, mu, grad_ctx, solver);
        });
    }
};

template<int Order>
struct ElasticVelocityAdjointPrepare3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    ElasticCPMLPointer cpml, SolverContext solver, float* pxx, float* pxy,
                    float* pxz, float* pyx, float* pyy, float* pyz, float* pzx, float* pzy,
                    float* pzz)
    {
        auto call = [&](auto&&... a) { elastic_velocity_adjoint_prepare_3d<Order>(a...); };
        // No free-surface logic: the interior test's box.
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, sweep::host::wave3d_rows(elastic3d_interior_box<Order>(solver), solver, 0),
            [=](int gx, int gy, int gz) {
                elastic_velocity_adjoint_prepare_3d_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, rho, cpml, solver, pxx, pxy, pxz, pyx, pyy, pyz, pzx,
                    pzy, pzz);
            },
            call,
            wf, rho, cpml, solver, pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz);
    }
};

template<int Order>
struct ElasticVelocityAdjointApply3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* pxx,
                    const float* pxy, const float* pxz, const float* pyx, const float* pyy,
                    const float* pyz, const float* pzx, const float* pzy, const float* pzz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_velocity_adjoint_apply_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, elastic3d_rows<Order>(elastic3d_halo_box<Order>(solver), solver, true),
                [=](int gx, int gy, int gz) {
                    elastic_velocity_adjoint_apply_3d_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, pxx,
                                                                     pxy, pxz, pyx, pyy, pyz, pzx, pzy,
                                                                     pzz, grad_ctx, solver);
                },
                call,
                wf, pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct ElasticStressAdjointPrepare3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, ElasticCPMLPointer cpml, SolverContext solver, float* qxx,
                    float* qxy, float* qxz, float* qyx, float* qyy, float* qyz, float* qzx,
                    float* qzy, float* qzz, SGradParam grad_ctx, const float* grad_fvx,
                    const float* grad_fvy, const float* grad_fvz, const float* grad_fvx_prev,
                    const float* grad_fvy_prev, const float* grad_fvz_prev,
                    const float* grad_vp_model, const float* grad_vs_model,
                    const float* grad_rho_model, float* grad_vp_out, float* grad_vs_out,
                    float* grad_rho_out)
    {
        auto call = [&](auto&&... a) { elastic_stress_adjoint_prepare_3d<Order>(a...); };
        // The surface row (szz zeroing, Robertsson transpose) and the FS
        // reach of the imaging stencils are out; imaging is inside the
        // physical box.
        CellBox3 box = elastic3d_interior_box<Order>(solver);
        if (grad_vp_out != nullptr)
            box = box.intersect(elastic3d_phys_box(solver));
        const int nz = solver.nz;
        const auto span = elastic3d_rows<Order>(box, solver, false);
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            sweep::host::with_optional(grad_vp_out, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        elastic_stress_adjoint_prepare_3d_cell<Order, P>(
                            gx, gy, gz % nz, gz / nz, wf, lambda, mu, cpml, solver, qxx, qxy, qxz, qyx,
                            qyy, qyz, qzx, qzy, qzz, grad_ctx, grad_fvx, grad_fvy, grad_fvz,
                            grad_fvx_prev, grad_fvy_prev, grad_fvz_prev, grad_vp_model, grad_vs_model,
                            grad_rho_model, grad_vp_out, grad_vs_out, grad_rho_out);
                    },
                    call,
                    wf, lambda, mu, cpml, solver, qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz,
                    grad_ctx, grad_fvx, grad_fvy, grad_fvz, grad_fvx_prev, grad_fvy_prev,
                    grad_fvz_prev, grad_vp_model, grad_vs_model, grad_rho_model, grad_vp_out,
                    grad_vs_out, grad_rho_out);
            });
        });
    }
};

template<int Order>
struct ElasticStressAdjointApply3DRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* qxx,
                    const float* qxy, const float* qxz, const float* qyx, const float* qyy,
                    const float* qyz, const float* qzx, const float* qzy, const float* qzz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_adjoint_apply_3d<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, elastic3d_rows<Order>(elastic3d_halo_box<Order>(solver), solver, true),
                [=](int gx, int gy, int gz) {
                    elastic_stress_adjoint_apply_3d_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, qxx,
                                                                   qxy, qxz, qyx, qyy, qyz, qzx, qzy,
                                                                   qzz, grad_ctx, solver);
                },
                call,
                wf, qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz, grad_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradElastic3DBsRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer forward,
                    ElasticWavefieldPointer adjoint, const float* fvx_prev, const float* fvy_prev,
                    const float* fvz_prev, const float* vp, const float* vs, const float* rho,
                    float* grad_vp, float* grad_vs, float* grad_rho, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_elastic3d_bs<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, elastic3d_rows<Order>(elastic3d_phys_box(solver), solver, false),
                [=](int gx, int gy, int gz) {
                    calculate_grad_elastic3d_bs_cell<Order, P>(gx, gy, gz % nz, gz / nz, forward,
                                                               adjoint, fvx_prev, fvy_prev, fvz_prev,
                                                               vp, vs, rho, grad_vp, grad_vs, grad_rho,
                                                               grad_ctx, solver);
                },
                call,
                forward, adjoint, fvx_prev, fvy_prev, fvz_prev, vp, vs, rho, grad_vp, grad_vs,
                grad_rho, grad_ctx, solver);
        });
    }
};

}  // namespace
#endif

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_kernel_3d_fn, elastic_velocity_kernel_3d_by_order,
                          elastic_velocity_kernel_3d, ElasticVelocity3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_kernel_3d_fn, elastic_stress_kernel_3d_by_order,
                          elastic_stress_kernel_3d, ElasticStress3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_kernel_3d_nopml_fn, elastic_velocity_kernel_3d_nopml_by_order,
                          elastic_velocity_kernel_3d_nopml, ElasticVelocity3DNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_kernel_3d_nopml_fn, elastic_stress_kernel_3d_nopml_by_order,
                          elastic_stress_kernel_3d_nopml, ElasticStress3DNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_adjoint_prepare_3d_fn, elastic_velocity_adjoint_prepare_3d_by_order,
                          elastic_velocity_adjoint_prepare_3d, ElasticVelocityAdjointPrepare3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_adjoint_apply_3d_fn, elastic_velocity_adjoint_apply_3d_by_order,
                          elastic_velocity_adjoint_apply_3d, ElasticVelocityAdjointApply3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_adjoint_prepare_3d_fn, elastic_stress_adjoint_prepare_3d_by_order,
                          elastic_stress_adjoint_prepare_3d, ElasticStressAdjointPrepare3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_adjoint_apply_3d_fn, elastic_stress_adjoint_apply_3d_by_order,
                          elastic_stress_adjoint_apply_3d, ElasticStressAdjointApply3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_elastic3d_bs_fn, calculate_grad_elastic3d_bs_by_order,
                          calculate_grad_elastic3d_bs, CalculateGradElastic3DBsRows);

// 3-D twin of elastic_capture_strips_2d: the six restore slabs of the
// physical box (x offset innermost so a warp shares memory sectors).
__global__ void elastic_capture_strips_3d(
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    float* __restrict__ fvx, float* __restrict__ fvy, float* __restrict__ fvz,
    int nx, int ny, int nz,
    int x0, int x1, int y0, int y1, int z0, int z1,
    int wxl, int wxh, int wyl, int wyh, int wzl, int wzh)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    const int bx = x1 - x0, by = y1 - y0;
    const int zi0 = z0 + wzl, zi1 = z1 - wzh, bzi = zi1 - zi0;
    const int yi0 = y0 + wyl, yi1 = y1 - wyh, byi = yi1 - yi0;
    const int n_zl = wzl * bx * by, n_zh = wzh * bx * by;
    const int n_yl = wyl * bx * bzi, n_yh = wyh * bx * bzi;
    const int n_xl = wxl * byi * bzi, n_xh = wxh * byi * bzi;
    int ix, iy, iz;
    if (t < n_zl) { iz = z0 + t / (bx * by); t %= bx * by; iy = y0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_zl) < n_zh) { iz = zi1 + t / (bx * by); t %= bx * by; iy = y0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_zh) < n_yl) { iy = y0 + t / (bx * bzi); t %= bx * bzi; iz = zi0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_yl) < n_yh) { iy = yi1 + t / (bx * bzi); t %= bx * bzi; iz = zi0 + t / bx; ix = x0 + t % bx; }
    else if ((t -= n_yh) < n_xl) { ix = x0 + t % wxl; t /= wxl; iy = yi0 + t % byi; iz = zi0 + t / byi; }
    else if ((t -= n_xl) < n_xh) { ix = (x1 - wxh) + t % wxh; t /= wxh; iy = yi0 + t % byi; iz = zi0 + t / byi; }
    else return;
    const int idx = b * nx * ny * nz + iz * nx * ny + iy * nx + ix;
    fvx[idx] = vx[idx];
    fvy[idx] = vy[idx];
    fvz[idx] = vz[idx];
}

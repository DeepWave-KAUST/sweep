// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the stencil kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box -- its own test's
// interior, shrunk by the free-surface reach of the FS helpers its stencils
// go through (common/host_rows.h elastic_fs_box_2d) -- and the cell body to
// run there, with the active FS axes as the ELASTIC_FS_Z / ELASTIC_FS_X
// flags (with_elastic_fs_2d).  No elastic kernel adds x_base, so the virtual
// thread gx is ix.  They hold no arithmetic of their own; the cell lambdas
// capture BY VALUE (see acoustic2d).  das_mu2d and visco_elastic2d launch
// most of these kernels through the same tables.
namespace {

static_assert(ELASTIC_FS_Z == 0 && ELASTIC_FS_X == 1, "with_elastic_fs_2d bit order");

// The FS helpers' half order (elastic_stencil_half_order).
template<int Order>
int elastic2d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The kernels' halo guard: [halo, n - halo) on both axes.
template<int Order>
CellBox elastic2d_halo_box(const SolverContext& s)
{
    const int h = elastic2d_half<Order>(s);
    return {h, s.nx - h, h, s.nz - h};
}

// The interior of the in_pml test of elastic_velocity_kernel and both adjoint
// prepares, inside the halo box.  elastic_stress_kernel's test keeps the
// surface rows/columns on its PML path too; elastic_fs_box_2d removes those.
template<int Order>
CellBox elastic2d_pml_box(const SolverContext& s)
{
    const int h = elastic2d_half<Order>(s);
    return CellBox{s.cut_x_lo() ? h : s.padLo(2) + h, s.nx - (s.cut_x_hi() ? h : s.padHi(2) + h),
                   s.cut_z_lo() ? h : s.padLo(0) + h, s.nz - (s.cut_z_hi() ? h : s.padHi(0) + h)}
        .intersect(elastic2d_halo_box<Order>(s));
}

// The nopml kernels' guard box (per-face band: M on a cut or free face,
// abcn + M + 1 otherwise).
template<int Order>
CellBox elastic2d_nopml_box(const SolverContext& s)
{
    const int M = elastic2d_half<Order>(s);
    const int w = s.abcn + M + 1;
    const int z_lo = (s.cut_z_lo() || s.fsLo(0)) ? M : w;
    const int z_hi = (s.cut_z_hi() || s.fsHi(0)) ? M : w;
    const int x_lo = (s.cut_x_lo() || s.fsLo(2)) ? M : w;
    const int x_hi = (s.cut_x_hi() || s.fsHi(2)) ? M : w;
    return {x_lo, s.nx - x_hi, z_lo, s.nz - z_hi};
}

CellBox elastic2d_phys_box(const SolverContext& s)
{
    return {s.phys_x0(), s.phys_x1(), s.phys_z0(), s.phys_z1()};
}

// ``box`` without the forward-type (adjoint = false) or adjoint FS reach.
template<int Order>
CellBox elastic2d_fs(CellBox box, const SolverContext& s, bool adjoint)
{
    return sweep::host::elastic_fs_box_2d(box, s, elastic2d_half<Order>(s), adjoint);
}

template<int Order>
struct ElasticVelocityRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, ElasticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_velocity_kernel<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_pml_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    elastic_velocity_cell<Order, P>(ix, iz, b, wf, rho, grad_ctx, cpml, solver);
                },
                call,
                wf, rho, grad_ctx, cpml, solver);
        });
    }
};

template<int Order>
struct ElasticStressRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, float* u_this, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_kernel<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_pml_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            sweep::host::with_optional(u_this, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        elastic_stress_cell<Order, P>(ix, iz, b, wf, lambda, mu, u_this, grad_ctx,
                                                      cpml, solver);
                    },
                    call,
                    wf, lambda, mu, u_this, grad_ctx, cpml, solver);
            });
        });
    }
};

template<int Order>
struct ElasticVelocityNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    SGradParam grad_ctx, SolverContext solver, float* fvx_prev, float* fvz_prev)
    {
        auto call = [&](auto&&... a) { elastic_velocity_kernel_nopml<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_nopml_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            sweep::host::with_optional(fvx_prev, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        elastic_velocity_nopml_cell<Order, P>(ix, iz, b, wf, rho, grad_ctx, solver,
                                                              fvx_prev, fvz_prev);
                    },
                    call,
                    wf, rho, grad_ctx, solver, fvx_prev, fvz_prev);
            });
        });
    }
};

template<int Order>
struct ElasticStressNopmlRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_kernel_nopml<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_nopml_box<Order>(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    elastic_stress_nopml_cell<Order, P>(ix, iz, b, wf, lambda, mu, grad_ctx, solver);
                },
                call,
                wf, lambda, mu, grad_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradElasticBsRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer forward,
                    ElasticWavefieldPointer adjoint, const float* fvx_prev, const float* fvz_prev,
                    const float* vp, const float* vs, const float* rho, float* grad_vp,
                    float* grad_vs, float* grad_rho, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_elastic_bs<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_phys_box(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    calculate_grad_elastic_bs_cell<Order, P>(ix, iz, b, forward, adjoint, fvx_prev,
                                                             fvz_prev, vp, vs, rho, grad_vp,
                                                             grad_vs, grad_rho, grad_ctx, solver);
                },
                call,
                forward, adjoint, fvx_prev, fvz_prev, vp, vs, rho, grad_vp, grad_vs, grad_rho,
                grad_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradElasticNobsRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer adjoint, const float* fvx,
                    const float* fvz, const float* fvx_prev, const float* fvz_prev,
                    const float* vp, const float* vs, const float* rho, float* grad_vp,
                    float* grad_vs, float* grad_rho, SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_elastic_nobs<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_phys_box(solver), solver, false);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    calculate_grad_elastic_nobs_cell<Order, P>(ix, iz, b, adjoint, fvx, fvz,
                                                               fvx_prev, fvz_prev, vp, vs, rho,
                                                               grad_vp, grad_vs, grad_rho,
                                                               grad_ctx, solver);
                },
                call,
                adjoint, fvx, fvz, fvx_prev, fvz_prev, vp, vs, rho, grad_vp, grad_vs, grad_rho,
                grad_ctx, solver);
        });
    }
};

template<int Order>
struct ElasticStressAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* lambda,
                    const float* mu, ElasticCPMLPointer cpml, SolverContext solver, float* qxx,
                    float* qzz, float* qxz, float* qzx, SGradParam grad_ctx,
                    const float* grad_fvx, const float* grad_fvz, const float* grad_fvx_prev,
                    const float* grad_fvz_prev, const float* grad_vp_model,
                    const float* grad_vs_model, const float* grad_rho_model, float* grad_vp_out,
                    float* grad_vs_out, float* grad_rho_out)
    {
        auto call = [&](auto&&... a) { elastic_stress_adjoint_prepare<Order>(a...); };
        // The surface rows/columns (traction transpose) and, with imaging, the
        // FS reach of the imaging stencils are out; imaging is inside the
        // physical box (which the PML box already is).
        CellBox box = elastic2d_fs<Order>(elastic2d_pml_box<Order>(solver), solver, false);
        if (grad_vp_out != nullptr)
            box = box.intersect(elastic2d_phys_box(solver));
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            sweep::host::with_optional(grad_vp_out, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        elastic_stress_adjoint_prepare_cell<Order, P>(
                            ix, iz, b, wf, lambda, mu, cpml, solver, qxx, qzz, qxz, qzx, grad_ctx,
                            grad_fvx, grad_fvz, grad_fvx_prev, grad_fvz_prev, grad_vp_model,
                            grad_vs_model, grad_rho_model, grad_vp_out, grad_vs_out, grad_rho_out);
                    },
                    call,
                    wf, lambda, mu, cpml, solver, qxx, qzz, qxz, qzx, grad_ctx, grad_fvx, grad_fvz,
                    grad_fvx_prev, grad_fvz_prev, grad_vp_model, grad_vs_model, grad_rho_model,
                    grad_vp_out, grad_vs_out, grad_rho_out);
            });
        });
    }
};

template<int Order>
struct ElasticStressAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* qxx,
                    const float* qzz, const float* qxz, const float* qzx, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_stress_adjoint_apply<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_halo_box<Order>(solver), solver, true);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    elastic_stress_adjoint_apply_cell<Order, P>(ix, iz, b, wf, qxx, qzz, qxz, qzx,
                                                                grad_ctx, solver);
                },
                call,
                wf, qxx, qzz, qxz, qzx, grad_ctx, solver);
        });
    }
};

template<int Order>
struct ElasticVelocityAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* rho,
                    ElasticCPMLPointer cpml, SolverContext solver, float* pxx, float* pzz,
                    float* pxz, float* pzx)
    {
        auto call = [&](auto&&... a) { elastic_velocity_adjoint_prepare<Order>(a...); };
        // No free-surface logic: the in_pml test's interior.
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(elastic2d_pml_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                elastic_velocity_adjoint_prepare_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, rho, cpml, solver, pxx, pzz, pxz, pzx);
            },
            call,
            wf, rho, cpml, solver, pxx, pzz, pxz, pzx);
    }
};

template<int Order>
struct ElasticVelocityAdjointApplyRows {
    static void run(dim3 grid, dim3 block, ElasticWavefieldPointer wf, const float* pxx,
                    const float* pzz, const float* pxz, const float* pzx, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { elastic_velocity_adjoint_apply<Order>(a...); };
        const CellBox box = elastic2d_fs<Order>(elastic2d_halo_box<Order>(solver), solver, true);
        sweep::host::with_elastic_fs_2d(solver, [&](auto fl) {
            constexpr CellPath P = add_flags(CellPath::Interior, decltype(fl)::value);
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    elastic_velocity_adjoint_apply_cell<Order, P>(ix, iz, b, wf, pxx, pzz, pxz, pzx,
                                                                  grad_ctx, solver);
                },
                call,
                wf, pxx, pzz, pxz, pzx, grad_ctx, solver);
        });
    }
};

}  // namespace
#endif

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_kernel_fn, elastic_velocity_kernel_by_order,
                          elastic_velocity_kernel, ElasticVelocityRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_kernel_fn, elastic_stress_kernel_by_order,
                          elastic_stress_kernel, ElasticStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_kernel_nopml_fn, elastic_velocity_kernel_nopml_by_order,
                          elastic_velocity_kernel_nopml, ElasticVelocityNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_kernel_nopml_fn, elastic_stress_kernel_nopml_by_order,
                          elastic_stress_kernel_nopml, ElasticStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_elastic_bs_fn, calculate_grad_elastic_bs_by_order,
                          calculate_grad_elastic_bs, CalculateGradElasticBsRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_elastic_nobs_fn, calculate_grad_elastic_nobs_by_order,
                          calculate_grad_elastic_nobs, CalculateGradElasticNobsRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_adjoint_prepare_fn, elastic_stress_adjoint_prepare_by_order,
                          elastic_stress_adjoint_prepare, ElasticStressAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_stress_adjoint_apply_fn, elastic_stress_adjoint_apply_by_order,
                          elastic_stress_adjoint_apply, ElasticStressAdjointApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_adjoint_prepare_fn, elastic_velocity_adjoint_prepare_by_order,
                          elastic_velocity_adjoint_prepare, ElasticVelocityAdjointPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(elastic_velocity_adjoint_apply_fn, elastic_velocity_adjoint_apply_by_order,
                          elastic_velocity_adjoint_apply, ElasticVelocityAdjointApplyRows);

// Compact copy of the physical box's restore strips (width w* per non-cut
// face) of vx/vz into the boundary-saving carriers: the box cells the NOPML
// kernel below does not compute.  Launched BEFORE the reverse update.
__global__ void elastic_capture_strips_2d(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ fvx, float* __restrict__ fvz,
    int nx, int nz, int x0, int x1, int z0, int z1,
    int wxl, int wxh, int wzl, int wzh)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    const int bw = x1 - x0;
    const int zi0 = z0 + wzl, zi1 = z1 - wzh;
    const int bh = zi1 - zi0;
    const int n_top = wzl * bw, n_bot = wzh * bw;
    const int n_left = wxl * bh, n_right = wxh * bh;
    int ix, iz;
    if (t < n_top) { iz = z0 + t / bw; ix = x0 + t % bw; }
    else if ((t -= n_top) < n_bot) { iz = zi1 + t / bw; ix = x0 + t % bw; }
    else if ((t -= n_bot) < n_left) { ix = x0 + t % wxl; iz = zi0 + t / wxl; }
    else if ((t -= n_left) < n_right) { ix = (x1 - wxh) + t % wxh; iz = zi0 + t / wxh; }
    else return;
    const int idx = b * nx * nz + iz * nx + ix;
    fvx[idx] = vx[idx];
    fvz[idx] = vz[idx];
}

// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

namespace elastic_tti_2nd2d {

#if defined(SWEEP_HOST_CORE)
// Host launchers of the eight table kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box in virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b) -- box_rows; no x_base, no
// topography, free surface refused -- and the cell body to run there.  They
// hold no arithmetic of their own; the cell lambdas capture BY VALUE (see
// acoustic2d).
namespace {

// The stencils' x stride is 1 at every launch (forward.cu / backward.cu:
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

template<int Order>
struct Tti2ndStressRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SGradParam grad_ctx, SolverContext solver,
                    float* sxx_ws, float* szz_ws, float* sxz_ws)
    {
        auto call = [&](auto&&... a) { tti2nd_stress_kernel<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, tti2nd_interior_box<Order>(solver).intersect(tti2nd_halo_box<Order>(solver)));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_stress_kernel_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, cpml, gc, solver, sxx_ws, szz_ws, sxz_ws);
            },
            call,
            wf, model, cpml, grad_ctx, solver, sxx_ws, szz_ws, sxz_ws);
    }
};

template<int Order>
struct Tti2ndDisplacementRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    const float* sxx_ws, const float* szz_ws, const float* sxz_ws,
                    float* u_this_t, ElasticCPMLPointer cpml, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { tti2nd_displacement_kernel<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, tti2nd_interior_box<Order>(solver).intersect(tti2nd_halo_box<Order>(solver)));
        sweep::host::with_optional(u_this_t, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    const SGradParam gc = unit_sx(grad_ctx);
                    tti2nd_displacement_kernel_cell<Order, decltype(path)::value>(
                        CellIndex2{ix, iz, b}, wf, model, sxx_ws, szz_ws, sxz_ws, u_this_t, cpml,
                        gc, solver);
                },
                call,
                wf, model, sxx_ws, szz_ws, sxz_ws, u_this_t, cpml, grad_ctx, solver);
        });
    }
};

template<int Order>
struct Tti2ndStressNopmlRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    SGradParam grad_ctx, SolverContext solver, float* sxx_ws, float* szz_ws,
                    float* sxz_ws)
    {
        auto call = [&](auto&&... a) { tti2nd_stress_kernel_nopml<Order>(a...); };
        const CellBox box = unit_sx_box(grad_ctx, tti2nd_stress_nopml_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_stress_kernel_nopml_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, gc, solver, sxx_ws, szz_ws, sxz_ws);
            },
            call,
            wf, model, grad_ctx, solver, sxx_ws, szz_ws, sxz_ws);
    }
};

template<int Order>
struct Tti2ndDisplacementNopmlRevRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    const float* sxx_ws, const float* szz_ws, const float* sxz_ws,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { tti2nd_displacement_kernel_nopml_rev<Order>(a...); };
        const CellBox box = unit_sx_box(grad_ctx, tti2nd_displacement_nopml_rev_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_displacement_kernel_nopml_rev_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, sxx_ws, szz_ws, sxz_ws, gc, solver);
            },
            call,
            wf, model, sxx_ws, szz_ws, sxz_ws, grad_ctx, solver);
    }
};

template<int Order>
struct Tti2ndAdjointDivPrepareRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SolverContext solver, float* q_sxxx, float* q_sxzz,
                    float* q_sxzx, float* q_szzz)
    {
        auto call = [&](auto&&... a) { tti2nd_adjoint_div_prepare<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(tti2nd_interior_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                tti2nd_adjoint_div_prepare_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, cpml, solver, q_sxxx, q_sxzz, q_sxzx, q_szzz);
            },
            call,
            wf, model, cpml, solver, q_sxxx, q_sxzz, q_sxzx, q_szzz);
    }
};

template<int Order>
struct Tti2ndAdjointStrainPrepareRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, StiffnessPointer model,
                    ElasticCPMLPointer cpml, SGradParam grad_ctx, SolverContext solver,
                    const float* q_sxxx, const float* q_sxzz, const float* q_sxzx,
                    const float* q_szzz, float* p_gxux, float* p_gzux, float* p_gxuz,
                    float* p_gzuz)
    {
        auto call = [&](auto&&... a) { tti2nd_adjoint_strain_prepare<Order>(a...); };
        const CellBox box = unit_sx_box(
            grad_ctx, tti2nd_interior_box<Order>(solver).intersect(tti2nd_halo_box<Order>(solver)));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_adjoint_strain_prepare_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, model, cpml, gc, solver, q_sxxx, q_sxzz, q_sxzx,
                    q_szzz, p_gxux, p_gzux, p_gxuz, p_gzuz);
            },
            call,
            wf, model, cpml, grad_ctx, solver, q_sxxx, q_sxzz, q_sxzx, q_szzz, p_gxux, p_gzux,
            p_gxuz, p_gzuz);
    }
};

template<int Order>
struct Tti2ndAdjointDisplacementApplyRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer wf, const float* p_gxux,
                    const float* p_gzux, const float* p_gxuz, const float* p_gzuz,
                    SGradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { tti2nd_adjoint_displacement_apply<Order>(a...); };
        const CellBox box = unit_sx_box(grad_ctx, tti2nd_halo_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_adjoint_displacement_apply_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, wf, p_gxux, p_gzux, p_gxuz, p_gzuz, gc, solver);
            },
            call,
            wf, p_gxux, p_gzux, p_gxuz, p_gzuz, grad_ctx, solver);
    }
};

template<int Order>
struct Tti2ndGradRows {
    static void run(dim3 grid, dim3 block, WavefieldPointer adj, StiffnessPointer model,
                    StiffnessGradPointer grad, const float* fux, const float* fuz,
                    const float* fux_next, const float* fuz_next, const float* fux_prev,
                    const float* fuz_prev, const float* q_sxxx, const float* q_sxzz,
                    const float* q_sxzx, const float* q_szzz, SGradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { tti2nd_calculate_grad<Order>(a...); };
        const CellBox box = unit_sx_box(grad_ctx, tti2nd_halo_box<Order>(solver));
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(box),
            [=](int ix, int iz, int b) {
                const SGradParam gc = unit_sx(grad_ctx);
                tti2nd_calculate_grad_cell<Order, CellPath::Interior>(
                    CellIndex2{ix, iz, b}, adj, model, grad, fux, fuz, fux_next, fuz_next,
                    fux_prev, fuz_prev, q_sxxx, q_sxzz, q_sxzx, q_szzz, gc, solver);
            },
            call,
            adj, model, grad, fux, fuz, fux_next, fuz_next, fux_prev, fuz_prev, q_sxxx, q_sxzz,
            q_sxzx, q_szzz, grad_ctx, solver);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_stress_kernel_fn, tti2nd_stress_kernel_by_order, tti2nd_stress_kernel, Tti2ndStressRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_displacement_kernel_fn, tti2nd_displacement_kernel_by_order, tti2nd_displacement_kernel, Tti2ndDisplacementRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_stress_kernel_nopml_fn, tti2nd_stress_kernel_nopml_by_order, tti2nd_stress_kernel_nopml, Tti2ndStressNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_displacement_kernel_nopml_rev_fn, tti2nd_displacement_kernel_nopml_rev_by_order, tti2nd_displacement_kernel_nopml_rev, Tti2ndDisplacementNopmlRevRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_adjoint_div_prepare_fn, tti2nd_adjoint_div_prepare_by_order, tti2nd_adjoint_div_prepare, Tti2ndAdjointDivPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_adjoint_strain_prepare_fn, tti2nd_adjoint_strain_prepare_by_order, tti2nd_adjoint_strain_prepare, Tti2ndAdjointStrainPrepareRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_adjoint_displacement_apply_fn, tti2nd_adjoint_displacement_apply_by_order, tti2nd_adjoint_displacement_apply, Tti2ndAdjointDisplacementApplyRows);
SWEEP_BY_ORDER_TABLE_ROWS(tti2nd_calculate_grad_fn, tti2nd_calculate_grad_by_order, tti2nd_calculate_grad, Tti2ndGradRows);
} // namespace elastic_tti_2nd2d

namespace elastic_tti_2nd2d {
// Source-cell compensation for the rho imaging: the stored U_{t+1} contains
// the injected wavelet S_t, which is rho-independent; add back lam*S/rho.
__global__ void tti2nd_rho_grad_source_correction(
    float* __restrict__ grad_rho,
    const float* __restrict__ adj_field,   // lam_{t+1} component matching the source field
    const float* __restrict__ rho,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
)
{
    int isrc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (isrc >= nsrc || b >= solver.B) return;

    const int spatial_size = solver.nx * solver.nz;
    const int* loc = sources_loc + (b * nsrc + isrc) * 2;
    const int ix = loc[0];
    const int iz = loc[1];
    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) return;

    const int idx = b * spatial_size + iz * solver.nx + ix;
    const float s = source[((long long)b * nsrc + isrc) * solver.nt + it];
    atomicAdd(&grad_rho[idx], adj_field[idx] * s / rho[idx]);
}
} // namespace elastic_tti_2nd2d

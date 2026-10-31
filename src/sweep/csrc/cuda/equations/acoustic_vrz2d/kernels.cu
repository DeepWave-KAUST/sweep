// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the four per-step table kernels split per
// launch/cell_path.h: the reconstruction step acoustic_vrz2nd_nopml (boundary
// saving) and the three global-memory gradient kernels, which the host core
// takes at every order (grad_tiled.cu: no tiled kernel without shared memory).
// All four are launched over fdtd::Wave2D (block 32 x 8, grid.z = B), so the
// virtual thread (gx, gy, gz) is the cell (ix, iz, b) -- they take no x_base --
// and each hands host/launch.h launch_rows its kernel's interior box in those
// coordinates (box_rows) and the cell body to run there.  Each box is spelled
// from the kernel's own guards (below), so it holds only cells the kernel runs
// to the end; none of the four has a topography or free-surface test (and VRZ
// sets neither: driver_traits.cuh setup_ctx is empty), so no row is cut by a
// surface.
//
// The launchers hold no arithmetic of their own; the cell lambdas capture BY
// VALUE (see acoustic2d), and each launcher builds its ``call`` once, above
// the p_tt dispatch.  Order -1 (runtime M) takes the same boxes with the
// kernels' run-time halo (vrz2d_halo); its row loops do not vectorise.
namespace {

// vrz2d_unit_sx (kernels.cuh) is the cells' unit-stride copy of grad_ctx;
// with any other stride the box is empty.
inline CellBox vrz2d_unit_sx_box(const GradParam& g, CellBox box)
{
    return g.sx == 1 ? box : CellBox{0, 0, 0, 0};
}

// acoustic_vrz2nd_nopml's guards: the cell is in the grid and in
// vrz2d_recon_box (kernels.cuh).
inline CellBox vrz2d_nopml_box(const SolverContext& s)
{
    const Vrz2dReconBox r = vrz2d_recon_box(s);
    return CellBox{r.x0, r.x1, r.z0, r.z1}.intersect({0, s.nx, 0, s.nz});
}

// The kernels' halo: Order / 2, solver.M for Order -1 (runtime M).
template<int Order>
int vrz2d_halo(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The halo guard of build_vrz_grad_fields and calculate_grad_vrz2d:
// [halo, n - halo) on both axes (inside the grid).
template<int Order>
CellBox vrz2d_halo_box(const SolverContext& s)
{
    const int halo = vrz2d_halo<Order>(s);
    return {halo, s.nx - halo, halo, s.nz - halo};
}

// calculate_grad_vrz2d_fused's guards (the grid, the physical box, the halo
// box) and its VrzGradAccessor2D tap test, false for every tap: a tap along x
// is (ix + off, iz) with |off| <= halo, which [halo, n - halo) holds for every
// off exactly when ix is in [2 halo, nx - 2 halo) and iz in [halo, nz - halo);
// along z likewise.  [2 halo, n - 2 halo) on both axes covers both, and the
// halo box with them.
template<int Order>
CellBox vrz2d_fused_grad_box(const SolverContext& s)
{
    const int halo = vrz2d_halo<Order>(s);
    return CellBox{s.phys_x0(), s.phys_x1(), s.phys_z0(), s.phys_z1()}
        .intersect({2 * halo, s.nx - 2 * halo, 2 * halo, s.nz - 2 * halo});
}

template<int Order>
struct AcousticVrz2ndNopmlRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, const float* vp,
                    const float* z, const float* inv_z, LaplaceParam lap_ctx, GradParam grad_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic_vrz2nd_nopml<Order>(a...); };
        sweep::host::launch_rows(
            grid, block,
            sweep::host::box_rows(vrz2d_unit_sx_box(grad_ctx, vrz2d_nopml_box(solver))),
            [=](int ix, int iz, int b) {
                acoustic_vrz2nd_nopml_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, vp, z, inv_z, lap_ctx, vrz2d_unit_sx(grad_ctx), solver);
            },
            call,
            wf, vp, z, inv_z, lap_ctx, grad_ctx, solver);
    }
};

template<int Order>
struct BuildVrzGradFieldsRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* vp, const float* z, float* c_x, float* c_z, float* e_x, float* e_z,
                    GradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { build_vrz_grad_fields<Order>(a...); };
        sweep::host::launch_rows(
            grid, block,
            sweep::host::box_rows(vrz2d_unit_sx_box(grad_ctx, vrz2d_halo_box<Order>(solver))),
            [=](int ix, int iz, int b) {
                build_vrz_grad_fields_cell<Order, CellPath::Interior>(
                    ix, iz, b, f_u_now, lambda_now, vp, z, c_x, c_z, e_x, e_z,
                    vrz2d_unit_sx(grad_ctx), solver);
            },
            call,
            f_u_now, lambda_now, vp, z, c_x, c_z, e_x, e_z, grad_ctx, solver);
    }
};

template<int Order>
struct CalculateGradVrz2dRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* p_tt, const float* p_prev, const float* p_next, const float* c_x,
                    const float* c_z, const float* e_x, const float* e_z, const float* vp,
                    const float* z, const float* inv_z, float* grad_vp, float* grad_z,
                    GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_vrz2d<Order>(a...); };
        const CellBox box = vrz2d_unit_sx_box(grad_ctx, vrz2d_halo_box<Order>(solver));
        // p_tt (the stored second difference) present or not: the cell's one test.
        sweep::host::with_optional(p_tt, [&](auto path) {
            constexpr CellPath P = decltype(path)::value;
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    calculate_grad_vrz2d_cell<Order, P>(
                        ix, iz, b, f_u_now, lambda_now, p_tt, p_prev, p_next, c_x, c_z, e_x,
                        e_z, vp, z, inv_z, grad_vp, grad_z, vrz2d_unit_sx(grad_ctx), solver);
                },
                call,
                f_u_now, lambda_now, p_tt, p_prev, p_next, c_x, c_z, e_x, e_z, vp, z, inv_z,
                grad_vp, grad_z, grad_ctx, lap_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradVrz2dFusedRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* p_tt, const float* p_prev, const float* p_next, const float* vp,
                    const float* z, const float* inv_z, float* grad_vp, float* grad_z,
                    GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_vrz2d_fused<Order>(a...); };
        const CellBox box = vrz2d_unit_sx_box(grad_ctx, vrz2d_fused_grad_box<Order>(solver));
        const int halo = vrz2d_halo<Order>(solver);   // the kernel's own, passed to the cell
        sweep::host::with_optional(p_tt, [&](auto path) {
            constexpr CellPath P = decltype(path)::value;
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    calculate_grad_vrz2d_fused_cell<Order, P>(
                        ix, iz, b, halo, f_u_now, lambda_now, p_tt, p_prev, p_next, vp,
                        z, inv_z, grad_vp, grad_z, vrz2d_unit_sx(grad_ctx), solver);
                },
                call,
                f_u_now, lambda_now, p_tt, p_prev, p_next, vp, z, inv_z, grad_vp, grad_z,
                grad_ctx, lap_ctx, solver);
        });
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(acoustic_vrz2nd_nopml_fn, acoustic_vrz2nd_nopml_by_order,
                          acoustic_vrz2nd_nopml, AcousticVrz2ndNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_vrz2d_fn, calculate_grad_vrz2d_by_order,
                          calculate_grad_vrz2d, CalculateGradVrz2dRows);
SWEEP_BY_ORDER_TABLE_ROWS(build_vrz_grad_fields_fn, build_vrz_grad_fields_by_order,
                          build_vrz_grad_fields, BuildVrzGradFieldsRows);
SWEEP_BY_ORDER_TABLE(build_vrz_adjoint_coeffs_fn, build_vrz_adjoint_coeffs_by_order, build_vrz_adjoint_coeffs);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_vrz2d_fused_fn, calculate_grad_vrz2d_fused_by_order,
                          calculate_grad_vrz2d_fused, CalculateGradVrz2dFusedRows);

// The p_tt imaging uses U_{it+1} − 2U_it + U_{it−1}, which at a source cell
// is dt²·rhs + S_it (add_source adds the wavelet to u_next after the step,
// unscaled).  The exact adjoint wants dt²·rhs alone: add back the 2·λ·S/v the
// imaging subtracted.  Same indexing as add_source.
__global__ void vrz2d_utt_source_correction(
    float* __restrict__ grad_vp,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= solver.B || s >= nsrc) return;
    int base = (b * nsrc + s) * 2;
    int ix = sources_loc[base + 0];
    int iz = sources_loc[base + 1];
    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) return;
    long long spatial_size = (long long)solver.nx * solver.nz;
    long long idx = (long long)b * spatial_size + (long long)iz * solver.nx + ix;
    long long src_idx = ((long long)b * nsrc + s) * solver.nt + it;
    atomicAdd(&grad_vp[idx], 2.0f * lambda_now[idx] * source[src_idx] / vp[idx]);
}

__global__ void build_kappa_lambda_vrz2d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = b * spatial_size + iz * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of the three gradient kernels the host core runs at every
// order (grad_tiled.cu: the tiled kernel is not built there), per
// launch/cell_path.h: each hands host/launch.h launch_rows the kernel's
// interior box as Wave3D row spans (common/host_rows.h wave3d_rows: gx =
// ix - x_base, gy = iy, gz = b * nz + iz; vrz3d_index adds solver.x_base) and
// the cell body to run there.  They hold no arithmetic of their own; the cell
// lambdas capture BY VALUE (see acoustic2d).  VRZ sets no topography, and the
// kernels have no air test anyway, so wave3d_rows cuts no row.  Order -1
// (runtime M) takes the same boxes with the kernels' run-time halo
// (vrz3d_halo); its row loops do not vectorise.
//
// The cells get vrz3d_unit_sx(grad_ctx) (kernels.cuh), the unit-stride copy;
// with any other stride the span is empty.
namespace {

// The kernels' halo: vrz3d_interior's ``halo`` (Order / 2, solver.M for the
// runtime order -1).
template<int Order>
int vrz3d_halo(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// vrz3d_interior's guard as a box: [halo, n - halo) on every axis, and ix
// below x_end() (the cells of a launch start at x_base, vrz3d_index's offset).
template<int Order>
CellBox3 vrz3d_guard_box(const SolverContext& s)
{
    const int h = vrz3d_halo<Order>(s);
    return CellBox3{h, s.nx - h, h, s.ny - h, h, s.nz - h}
        .intersect({s.x_base, s.x_end(), 0, s.ny, 0, s.nz});
}

// The two imaging kernels' guards: the physical-box test inside the halo
// guard (cut-aware through phys_*(), as the kernels spell it).
template<int Order>
CellBox3 vrz3d_image_box(const SolverContext& s)
{
    return CellBox3{s.phys_x0(), s.phys_x1(), s.phys_y0(), s.phys_y1(), s.phys_z0(), s.phys_z1()}
        .intersect(vrz3d_guard_box<Order>(s));
}

// calculate_grad_vrz3d_fused's box: its guards, and every tap of its six
// VrzGradAccessor3D stencils (offsets up to +-Order/2 along the stencil's own
// axis, the other two indices the cell's) passes the accessor's
// vrz3d_interior test -- [halo, n - halo) and below x_end() -- so the cell is
// 2 * halo from every edge and halo short of x_end().
template<int Order>
CellBox3 vrz3d_fused_box(const SolverContext& s)
{
    const int h = vrz3d_halo<Order>(s);
    return vrz3d_image_box<Order>(s)
        .intersect({2 * h, s.nx - 2 * h, 2 * h, s.ny - 2 * h, 2 * h, s.nz - 2 * h})
        .intersect({s.x_base, s.x_end() - h, 0, s.ny, 0, s.nz});
}

// ``box`` as launch_rows spans, empty unless the x stride is 1 (see above).
inline auto vrz3d_rows(const CellBox3& box, const GradParam& g, const SolverContext& s)
{
    return sweep::host::wave3d_rows(g.sx == 1 ? box : CellBox3{0, 0, 0, 0, 0, 0}, s, s.x_base);
}

template<int Order>
struct BuildVrzGradFields3DRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* vp, const float* z, float* c_x, float* c_y, float* c_z,
                    float* e_x, float* e_y, float* e_z, GradParam grad_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { build_vrz_grad_fields_3d<Order>(a...); };
        const int xb = solver.x_base, nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, vrz3d_rows(vrz3d_guard_box<Order>(solver), grad_ctx, solver),
            [=](int gx, int gy, int gz) {
                build_vrz_grad_fields_3d_cell<Order, CellPath::Interior>(
                    Vrz3dCellIndex{gx + xb, gy, gz % nz, gz / nz}, f_u_now, lambda_now, vp, z,
                    c_x, c_y, c_z, e_x, e_y, e_z, vrz3d_unit_sx(grad_ctx), solver);
            },
            call,
            f_u_now, lambda_now, vp, z, c_x, c_y, c_z, e_x, e_y, e_z, grad_ctx, solver);
    }
};

template<int Order>
struct CalculateGradVrz3dRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* p_tt, const float* p_prev, const float* p_next, const float* c_x,
                    const float* c_y, const float* c_z, const float* e_x, const float* e_y,
                    const float* e_z, const float* vp, const float* z, const float* inv_z,
                    float* grad_vp, float* grad_z, GradParam grad_ctx, LaplaceParam lap_ctx,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_vrz3d<Order>(a...); };
        const int xb = solver.x_base, nz = solver.nz;
        const auto span = vrz3d_rows(vrz3d_image_box<Order>(solver), grad_ctx, solver);
        sweep::host::with_optional(p_tt, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, span,
                [=](int gx, int gy, int gz) {
                    calculate_grad_vrz3d_cell<Order, decltype(path)::value>(
                        Vrz3dCellIndex{gx + xb, gy, gz % nz, gz / nz}, f_u_now, lambda_now, p_tt,
                        p_prev, p_next, c_x, c_y, c_z, e_x, e_y, e_z, vp, z, inv_z, grad_vp,
                        grad_z, vrz3d_unit_sx(grad_ctx), solver);
                },
                call,
                f_u_now, lambda_now, p_tt, p_prev, p_next, c_x, c_y, c_z, e_x, e_y, e_z, vp, z,
                inv_z, grad_vp, grad_z, grad_ctx, lap_ctx, solver);
        });
    }
};

template<int Order>
struct CalculateGradVrz3dFusedRows {
    static void run(dim3 grid, dim3 block, const float* f_u_now, const float* lambda_now,
                    const float* p_tt, const float* p_prev, const float* p_next, const float* vp,
                    const float* z, const float* inv_z, float* grad_vp, float* grad_z,
                    GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { calculate_grad_vrz3d_fused<Order>(a...); };
        const int xb = solver.x_base, nz = solver.nz;
        const auto span = vrz3d_rows(vrz3d_fused_box<Order>(solver), grad_ctx, solver);
        sweep::host::with_optional(p_tt, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, span,
                [=](int gx, int gy, int gz) {
                    calculate_grad_vrz3d_fused_cell<Order, decltype(path)::value>(
                        Vrz3dCellIndex{gx + xb, gy, gz % nz, gz / nz}, f_u_now, lambda_now, p_tt,
                        p_prev, p_next, vp, z, inv_z, grad_vp, grad_z, vrz3d_unit_sx(grad_ctx),
                        solver);
                },
                call,
                f_u_now, lambda_now, p_tt, p_prev, p_next, vp, z, inv_z, grad_vp, grad_z,
                grad_ctx, lap_ctx, solver);
        });
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(build_vrz_grad_fields_3d_fn, build_vrz_grad_fields_3d_by_order,
                          build_vrz_grad_fields_3d, BuildVrzGradFields3DRows);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_vrz3d_fn, calculate_grad_vrz3d_by_order, calculate_grad_vrz3d,
                          CalculateGradVrz3dRows);
SWEEP_BY_ORDER_TABLE(build_vrz_adjoint_coeffs_3d_fn, build_vrz_adjoint_coeffs_3d_by_order, build_vrz_adjoint_coeffs_3d);
SWEEP_BY_ORDER_TABLE_ROWS(calculate_grad_vrz3d_fused_fn, calculate_grad_vrz3d_fused_by_order,
                          calculate_grad_vrz3d_fused, CalculateGradVrz3dFusedRows);

// Source-cell correction of the p_tt imaging (see the 2-D twin): the stored
// second time difference contains the wavelet add_source_3d added at it;
// add back the 2·λ·S/v the imaging subtracted.  Same indexing as add_source_3d.
__global__ void vrz3d_utt_source_correction(
    float* __restrict__ grad_vp,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 3)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= solver.B || s >= nsrc) return;
    int base = (b * nsrc + s) * 3;
    int ix = sources_loc[base + 0];
    int iy = sources_loc[base + 1];
    int iz = sources_loc[base + 2];
    if (ix < 0 || ix >= solver.nx || iy < 0 || iy >= solver.ny || iz < 0 || iz >= solver.nz) return;
    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    long long idx = (long long)b * spatial_size
                  + ((long long)iz * solver.ny + iy) * solver.nx + ix;
    long long src_idx = ((long long)b * nsrc + s) * solver.nt + it;
    atomicAdd(&grad_vp[idx], 2.0f * lambda_now[idx] * source[src_idx] / vp[idx]);
}

__global__ void build_kappa_lambda_vrz3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<2>(solver, ix, iy, iz, b);
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = b * spatial_size + iz * solver.nx * solver.ny + iy * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of das_mu3d's own two kernels (launch/cell_path.h): each
// hands host/launch.h launch_rows the kernel's interior box as Wave3D row
// spans (common/host_rows.h wave3d_rows: gx = ix, gy = iy, gz = b * nz + iz;
// neither kernel adds x_base) and the cell body to run there.  The other
// das_mu3d launches go through elastic3d's tables, already split there.  They
// hold no arithmetic of their own; the cell lambdas capture BY VALUE (see
// acoustic2d).
namespace {

// The FS helper's half order (elastic3d_stencil_half_order).
template<int Order>
int das_mu3d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// das_mu3d_stress_strain_kernel's interior: the kernel has no PML branch, so
// it is the whole halo guard [halo, n - halo) on every axis, below the rows
// elastic3d_top_fs_sgradient_z mirrors (common/host_rows.h elastic3d_fs_box,
// forward: z >= phys_z0 + M with a free surface, which also drops the
// Robertsson surface row).
template<int Order>
CellBox3 das_mu3d_stress_strain_interior_box(const SolverContext& s)
{
    const int h = das_mu3d_half<Order>(s);
    return sweep::host::elastic3d_fs_box(CellBox3{h, s.nx - h, h, s.ny - h, h, s.nz - h}, s, h, false);
}

// das_mu3d_stress_strain_adjoint_prepare's interior: its ``interior`` test,
// the physical box shrunk by one cell on every face (not cut-aware, as the
// kernel spells it; the kernel has no halo guard).  The free-surface row
// phys_z0 is outside it.
CellBox3 das_mu3d_stress_strain_adjoint_prepare_interior_box(const SolverContext& s)
{
    return CellBox3{s.phys_x0() + 1, s.phys_x1() - 1, s.phys_y0() + 1, s.phys_y1() - 1,
                    s.phys_z0() + 1, s.phys_z1() - 1}
        .intersect(CellBox3{0, s.nx, 0, s.ny, 0, s.nz});
}

template<int Order>
struct DasMu3dStressStrainRows {
    static void run(dim3 grid, dim3 block, DasMuWavefieldPointer3D wf, const float* lambda,
                    const float* mu, float* u_this, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { das_mu3d_stress_strain_kernel<Order>(a...); };
        const int nz = solver.nz;
        const auto span = sweep::host::wave3d_rows(das_mu3d_stress_strain_interior_box<Order>(solver), solver, 0);
        sweep::host::with_elastic_fs_3d(solver, [&](auto fl) {
            sweep::host::with_optional(u_this, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        das_mu3d_stress_strain_cell<Order, P>(gx, gy, gz % nz, gz / nz, wf, lambda, mu,
                                                              u_this, grad_ctx, cpml, solver);
                    },
                    call,
                    wf, lambda, mu, u_this, grad_ctx, cpml, solver);
            });
        });
    }
};

template<int Order>
struct DasMu3dStressStrainAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, DasMuWavefieldPointer3D wf, const float* lambda,
                    const float* mu, ElasticCPMLPointer cpml, SolverContext solver, float* qxx,
                    float* qxy, float* qxz, float* qyx, float* qyy, float* qyz, float* qzx,
                    float* qzy, float* qzz)
    {
        auto call = [&](auto&&... a) { das_mu3d_stress_strain_adjoint_prepare<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block,
            sweep::host::wave3d_rows(das_mu3d_stress_strain_adjoint_prepare_interior_box(solver), solver, 0),
            [=](int gx, int gy, int gz) {
                das_mu3d_stress_strain_adjoint_prepare_cell<Order, CellPath::Interior>(
                    DasMu3dCellIndex{gx, gy, gz % nz, gz / nz}, wf, lambda, mu, cpml, solver, qxx, qxy, qxz, qyx, qyy,
                    qyz, qzx, qzy, qzz);
            },
            call,
            wf, lambda, mu, cpml, solver, qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(das_mu3d_stress_strain_kernel_fn, das_mu3d_stress_strain_kernel_by_order,
                          das_mu3d_stress_strain_kernel, DasMu3dStressStrainRows);
SWEEP_BY_ORDER_TABLE_ROWS(das_mu3d_stress_strain_adjoint_prepare_fn,
                          das_mu3d_stress_strain_adjoint_prepare_by_order,
                          das_mu3d_stress_strain_adjoint_prepare,
                          DasMu3dStressStrainAdjointPrepareRows);

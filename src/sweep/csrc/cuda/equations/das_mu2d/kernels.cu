// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"
#include "../../common/host_rows.h"

#if defined(SWEEP_HOST_CORE)
// Host launchers of das_mu2d's own two kernels (launch/cell_path.h): each
// hands host/launch.h launch_rows the kernel's interior box and the cell body
// to run there.  The other das_mu2d launches go through elastic2d's tables,
// already split there.  Neither kernel adds x_base, so the virtual thread gx
// is ix; they hold no arithmetic of their own; the cell lambdas capture BY
// VALUE (see acoustic2d).
namespace {

// The FS helper's half order (elastic_stencil_half_order).
template<int Order>
int das_mu2d_half(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// das_mu2d_stress_strain_kernel's interior: its halo guard [halo, n - halo)
// on both axes and the !in_pml side of its legacy test, spelled as the kernel
// spells it (abcn bands, top_pml = free_surface ? halo + 1 : abcn + halo) --
// then without the z free-surface reach of elastic_top_fs_sgradient_z_2d
// (common/host_rows.h elastic_fs_box_2d, forward: the mirror taps and the
// order-2 band; the surface rows are behind in_pml already).
template<int Order>
CellBox das_mu2d_stress_strain_interior_box(const SolverContext& s)
{
    const int h = das_mu2d_half<Order>(s);
    const int top_pml = s.free_surface ? (h + 1) : (s.abcn + h);
    const CellBox box = CellBox{s.abcn + h, s.nx - s.abcn - h, top_pml, s.nz - s.abcn - h}
                            .intersect(CellBox{h, s.nx - h, h, s.nz - h});
    return sweep::host::elastic_fs_box_2d(box, s, h, false);
}

// das_mu2d_stress_strain_adjoint_prepare's interior: the !in_pml_2d box (the
// kernel has no halo guard), without the z free-surface rows its is_fs test
// picks (elastic_is_top_free_surface_row(solver, iz): phys_z0() on a z-low
// face, phys_z1() - 1 on a z-high one).
template<int Order>
CellBox das_mu2d_stress_strain_adjoint_prepare_interior_box(const SolverContext& s)
{
    const int h = das_mu2d_half<Order>(s);
    CellBox box = CellBox{s.phys_x0(h), s.phys_x1(h), s.phys_z0(h), s.phys_z1(h)}
                      .intersect(CellBox{0, s.nx, 0, s.nz});
    if (s.fsLo(0)) box.z0 = std::max(box.z0, s.phys_z0() + 1);
    if (s.fsHi(0)) box.z1 = std::min(box.z1, s.phys_z1() - 1);
    return box;
}

static_assert(ELASTIC_FS_Z == 0, "with_flags bit 0 is the ELASTIC_FS_Z flag");

template<int Order>
struct DasMu2dStressStrainRows {
    static void run(dim3 grid, dim3 block, DasMuWavefieldPointer2D wf, const float* lambda,
                    const float* mu, float* u_this, SGradParam grad_ctx, ElasticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { das_mu2d_stress_strain_kernel<Order>(a...); };
        const CellBox box = das_mu2d_stress_strain_interior_box<Order>(solver);
        // Only the z derivatives go through an FS helper: ELASTIC_FS_Z alone
        // (bit 0), no ELASTIC_FS_X instantiations.
        sweep::host::with_flags([&](auto fl) {
            sweep::host::with_optional(u_this, [&](auto op) {
                constexpr CellPath P = add_flags(decltype(op)::value, decltype(fl)::value);
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        das_mu2d_stress_strain_cell<Order, P>(DasMu2dCellIndex{ix, iz, b}, wf, lambda,
                                                              mu, u_this, grad_ctx, cpml, solver);
                    },
                    call,
                    wf, lambda, mu, u_this, grad_ctx, cpml, solver);
            });
        }, solver.fsLo(0) || solver.fsHi(0));
    }
};

template<int Order>
struct DasMu2dStressStrainAdjointPrepareRows {
    static void run(dim3 grid, dim3 block, DasMuWavefieldPointer2D wf, const float* lambda,
                    const float* mu, ElasticCPMLPointer cpml, SolverContext solver, float* qxx,
                    float* qzz, float* qxz, float* qzx)
    {
        auto call = [&](auto&&... a) { das_mu2d_stress_strain_adjoint_prepare<Order>(a...); };
        sweep::host::launch_rows(
            grid, block,
            sweep::host::box_rows(das_mu2d_stress_strain_adjoint_prepare_interior_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                das_mu2d_stress_strain_adjoint_prepare_cell<Order, CellPath::Interior>(
                    ix, iz, b, wf, lambda, mu, cpml, solver, qxx, qzz, qxz, qzx);
            },
            call,
            wf, lambda, mu, cpml, solver, qxx, qzz, qxz, qzx);
    }
};

}  // namespace
#endif

SWEEP_BY_ORDER_TABLE_ROWS(das_mu2d_stress_strain_kernel_fn, das_mu2d_stress_strain_kernel_by_order,
                          das_mu2d_stress_strain_kernel, DasMu2dStressStrainRows);
SWEEP_BY_ORDER_TABLE_ROWS(das_mu2d_stress_strain_adjoint_prepare_fn,
                          das_mu2d_stress_strain_adjoint_prepare_by_order,
                          das_mu2d_stress_strain_adjoint_prepare,
                          DasMu2dStressStrainAdjointPrepareRows);

// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(visco_acoustic2d_carrier_fn, visco_acoustic2d_carrier_by_order, visco_acoustic2d_carrier);

namespace {
// Strip element r of one shot's ring (kernels.cuh) -> its grid cell.
__device__ inline void visco_ring_cell(int64_t r, const SolverContext& s, int& iz, int& ix)
{
    const int M = s.M;
    const int64_t zstrip = static_cast<int64_t>(M) * s.nx;
    const int64_t xstrip = static_cast<int64_t>(s.nz) * M;
    if (r < zstrip) { iz = s.phys_z0() + static_cast<int>(r / s.nx); ix = static_cast<int>(r % s.nx); return; }
    r -= zstrip;
    if (r < zstrip) { iz = s.phys_z1() - M + static_cast<int>(r / s.nx); ix = static_cast<int>(r % s.nx); return; }
    r -= zstrip;
    if (r < xstrip) { iz = static_cast<int>(r / M); ix = s.phys_x0() + static_cast<int>(r % M); return; }
    r -= xstrip;
    iz = static_cast<int>(r / M); ix = s.phys_x1() - M + static_cast<int>(r % M);
}
}  // namespace

__global__ void visco_acoustic2d_ring_pack(const float* __restrict__ u_this, float* __restrict__ tail,
                                           int64_t step, SolverContext solver)
{
    const int64_t R = visco_ring_floats(solver.M, solver.nz, solver.nx);
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= R || b >= solver.B) return;
    int iz, ix;
    visco_ring_cell(r, solver, iz, ix);
    const int64_t sp = static_cast<int64_t>(solver.nx) * solver.nz;
    tail[(step * solver.B + b) * R + r] = u_this[b * sp + static_cast<int64_t>(iz) * solver.nx + ix];
}

__global__ void visco_acoustic2d_ring_unpack(const float* __restrict__ tail, float* __restrict__ carrier,
                                             int64_t step, SolverContext solver)
{
    const int64_t R = visco_ring_floats(solver.M, solver.nz, solver.nx);
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (r >= R || b >= solver.B) return;
    int iz, ix;
    visco_ring_cell(r, solver, iz, ix);
    // the strips run the full grid length; only their physical cells are the
    // carrier's (a corner cell is in two strips, both holding the same value)
    if (iz < solver.phys_z0() || iz >= solver.phys_z1() || ix < solver.phys_x0() || ix >= solver.phys_x1())
        return;
    const int64_t sp = static_cast<int64_t>(solver.nx) * solver.nz;
    carrier[b * sp + static_cast<int64_t>(iz) * solver.nx + ix] = tail[(step * solver.B + b) * R + r];
}

#pragma once
#include "../../launch/by_order.cuh"
#include "../../launch/cell_path.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../operators/staggered.cuh"

// Displacement-based (second-order in time) 2-D elastic TTI of Oh et al.
// (2020, GJI 223, eqs 11-12).  Spatial terms are nested staggered
// forward/backward first-derivative pairs, so the stiffness operator is the
// exact negative-transpose sandwich K = -D^T C D; the leapfrog update is
//   U_{t+1} = 2 U_t - U_{t-1} + (dt^2/rho) * div(sigma(grad U_t)) + S_t.
// Each first derivative carries its own CPML recursive-convolution memory
// (half-node profiles on the inner/forward layer, integer-node on the
// outer/backward layer).  The forward step is split into a stress kernel
// (writes the three stresses to scratch fields) and a displacement kernel
// (their divergence -> u_next buffer); the host rotates the (now, pre, next)
// triple buffer.  The adjoint runs the same structure transposed
// (prepare/apply split, forward<->backward stencils swapped, no extra signs:
// the two spatial-transpose minus signs cancel; the stiffness imaging keeps
// a single -1).

namespace elastic_tti_2nd2d {

struct WavefieldPointer {
    float* __restrict__ ux;
    float* __restrict__ uz;
    float* __restrict__ ux_pre;
    float* __restrict__ uz_pre;
    float* __restrict__ ux_nxt;
    float* __restrict__ uz_nxt;

    float* __restrict__ m_gxux;
    float* __restrict__ m_gzux;
    float* __restrict__ m_gxuz;
    float* __restrict__ m_gzuz;
    float* __restrict__ m_sxxx;
    float* __restrict__ m_sxzz;
    float* __restrict__ m_sxzx;
    float* __restrict__ m_szzz;

    __device__ WavefieldPointer offset(int b, int spatial_size) const
    {
        WavefieldPointer out = *this;
        const int shift = b * spatial_size;
        out.ux += shift;
        out.uz += shift;
        out.ux_pre += shift;
        out.uz_pre += shift;
        out.ux_nxt += shift;
        out.uz_nxt += shift;
        out.m_gxux += shift;
        out.m_gzux += shift;
        out.m_gxuz += shift;
        out.m_gzuz += shift;
        out.m_sxxx += shift;
        out.m_sxzz += shift;
        out.m_sxzx += shift;
        out.m_szzz += shift;
        return out;
    }
};

struct StiffnessPointer {
    const float* __restrict__ rho;
    const float* __restrict__ C11;
    const float* __restrict__ C33;
    const float* __restrict__ C13;
    const float* __restrict__ C55;
    const float* __restrict__ C15;
    const float* __restrict__ C35;

    __device__ StiffnessPointer offset(int b, int spatial_size) const
    {
        StiffnessPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C33 += shift;
        out.C13 += shift;
        out.C55 += shift;
        out.C15 += shift;
        out.C35 += shift;
        return out;
    }
};

struct StiffnessGradPointer {
    float* __restrict__ rho;
    float* __restrict__ C11;
    float* __restrict__ C33;
    float* __restrict__ C13;
    float* __restrict__ C55;
    float* __restrict__ C15;
    float* __restrict__ C35;

    __device__ StiffnessGradPointer offset(int b, int spatial_size) const
    {
        StiffnessGradPointer out = *this;
        const int shift = b * spatial_size;
        out.rho += shift;
        out.C11 += shift;
        out.C33 += shift;
        out.C13 += shift;
        out.C55 += shift;
        out.C15 += shift;
        out.C35 += shift;
        return out;
    }
};

// Sources and receivers act on the freshly computed time level: the next
// buffer (W_{t+1}) before the host rotates the triple buffer.
__host__ inline float* field_ptr(const WavefieldPointer& wf, int field)
{
    switch (field) {
        case 0: return wf.ux_nxt;
        case 1: return wf.uz_nxt;
        default: return nullptr;
    }
}

// kernel<order> through its table in kernels.cu (launch/by_order.cuh).
#define TTI2ND_LAUNCH(kernel, order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(kernel##_by_order, order, grid, block, __VA_ARGS__)

// The indices of one cell, as the kernels decode them, handed to a _cell body
// in one by-value struct rather than as three int parameters.  That keeps the
// CUDA code what it was before the split: nvcc puts a compare's operands in a
// canonical order by kind, and a bare int parameter ranks below a load, so
// ``ix < solver.abcn + halo`` with ix a parameter is emitted with its operands
// swapped -- the branch polarities flip and, in some specialisations, the
// scheduling with them.  A field of a by-value struct is a load, as the
// kernel's own decoded index ranks, and the split kernels compile to the same
// SASS (checked with cuobjdump, docs/dev/host_core.md rule 7).  On the host it
// is free.  The cells take the SolverContext by value for the same reason:
// through a reference, the legacy ``in_pml`` test's lazy reads of
// abcn / nx / nz cannot be hoisted and the || chain compiles to branches where
// the kernel had one predicate.
struct CellIndex2 {
    int ix, iz, b;
};

// A read of a field the kernel never writes, through a const __restrict__
// parameter.  While its body was the kernel's own, tti2nd_stress_kernel's
// stiffness and CPML-profile reads compiled to read-only-cache loads
// (LDG.E.CONSTANT): nvcc proved them read-only from the kernel's by-value
// restrict struct parameters, which it no longer trusts once they reach a
// __device__ function.  Read through this, the cell's loads stay what they
// were and the kernel keeps its SASS (docs/dev/host_core.md rule 7); on the
// host it is the plain load.
template <class T>
__device__ __forceinline__ T tti2nd_ro(const T* __restrict__ p, int i)
{
    return p[i];
}

// The kernels' boxes in (ix, iz), for the host row launchers
// (launch/cell_path.h).  Each spells a kernel test below as a half-open box:
// the guard of the stencil kernels (cells at least ``halo`` from every edge),
// the legacy interior test of the stress / displacement kernels and both
// adjoint prepares (abcn + halo from every edge, as those kernels write it),
// and the two widened guards of the boundary-saving replay kernels.
template<int Order>
__device__ __forceinline__
CellBox tti2nd_halo_box(const SolverContext& solver)
{
    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    return {halo, solver.nx - halo, halo, solver.nz - halo};
}

template<int Order>
__device__ __forceinline__
CellBox tti2nd_interior_box(const SolverContext& solver)
{
    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    return {solver.abcn + halo, solver.nx - solver.abcn - halo,
            solver.abcn + halo, solver.nz - solver.abcn - halo};
}

template<int Order>
__device__ __forceinline__
CellBox tti2nd_stress_nopml_box(const SolverContext& solver)
{
    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    // as the kernel's guard: never the outer M-cell halo
    const int lo = halo - M > M ? halo - M : M;
    const int x1 = solver.nx - halo + M < solver.nx - M ? solver.nx - halo + M : solver.nx - M;
    const int z1 = solver.nz - halo + M < solver.nz - M ? solver.nz - halo + M : solver.nz - M;
    return {lo, x1, lo, z1};
}

template<int Order>
__device__ __forceinline__
CellBox tti2nd_displacement_nopml_rev_box(const SolverContext& solver)
{
    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    return {halo, solver.nx - halo, halo, solver.nz - halo};
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// Inner layer: strains from u_now, CPML memory on the four forward
// derivatives, constitutive row -> three stress scratch fields.
// One cell of tti2nd_stress_kernel: everything after the kernel's guards
// (see launch/cell_path.h).  Interior box: tti2nd_interior_box<Order>() (the legacy abcn + halo
// test, as the kernel spells it) inside tti2nd_halo_box<Order>() (the guard),
// where the memory-variable block is skipped.
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_stress_kernel_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const ElasticCPMLPointer& cpml,
    const SGradParam& grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float gxux = sgradient<2, Order, X, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);
    float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);

    bool in_pml = false;
    if constexpr (!interior_path<Path>) {
        in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                 (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    }
    if (in_pml) {
        // Multiaxial PML: the damping profiles are FULL 2-D fields, not one
        // per axis -- sigma_x picks up a fraction of sigma_z and vice versa
        // (Li & Bou Matar 2010 eq. 23), which is what keeps this second-order
        // displacement system stable.  Same index for every profile.
        const int pml_idx = iz * solver.nx + ix;
        const float azh = tti2nd_ro(cpml.azh, pml_idx);
        const float bzh = tti2nd_ro(cpml.bzh, pml_idx);
        const float axh = tti2nd_ro(cpml.axh, pml_idx);
        const float bxh = tti2nd_ro(cpml.bxh, pml_idx);

        f.m_gxux[idx] = axh * f.m_gxux[idx] + bxh * gxux;
        gxux += f.m_gxux[idx];
        f.m_gzux[idx] = azh * f.m_gzux[idx] + bzh * gzux;
        gzux += f.m_gzux[idx];
        f.m_gxuz[idx] = axh * f.m_gxuz[idx] + bxh * gxuz;
        gxuz += f.m_gxuz[idx];
        f.m_gzuz[idx] = azh * f.m_gzuz[idx] + bzh * gzuz;
        gzuz += f.m_gzuz[idx];
    }

    const float exz = gzux + gxuz;
    sxx_ws[shift + idx] = tti2nd_ro(m.C11, idx) * gxux + tti2nd_ro(m.C13, idx) * gzuz +
                          tti2nd_ro(m.C15, idx) * exz;
    szz_ws[shift + idx] = tti2nd_ro(m.C13, idx) * gxux + tti2nd_ro(m.C33, idx) * gzuz +
                          tti2nd_ro(m.C35, idx) * exz;
    sxz_ws[shift + idx] = tti2nd_ro(m.C15, idx) * gxux + tti2nd_ro(m.C35, idx) * gzuz +
                          tti2nd_ro(m.C55, idx) * exz;
}

template<int Order>
__global__ void tti2nd_stress_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_stress_kernel_cell<Order>(CellIndex2{ix, iz, b}, wf, model, cpml, grad_ctx, solver,
        sxx_ws, szz_ws, sxz_ws);
}

using tti2nd_stress_kernel_fn = void (*)(WavefieldPointer, StiffnessPointer,
    ElasticCPMLPointer, SGradParam, SolverContext, float*, float*, float*);
SWEEP_BY_ORDER_DECL(tti2nd_stress_kernel_fn, tti2nd_stress_kernel_by_order);

// Outer layer: stress divergence with CPML on the four backward derivatives,
// leapfrog displacement update written to the u_next buffer.
// One cell of tti2nd_displacement_kernel: everything after the kernel's guards
// (see launch/cell_path.h).  Interior box: tti2nd_interior_box<Order>() (the legacy abcn + halo
// test, as the kernel spells it) inside tti2nd_halo_box<Order>() (the guard),
// where the memory-variable block is skipped; optional output: u_this_t
// (null at every call site today).
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_displacement_kernel_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    float* __restrict__ u_this_t,
    const ElasticCPMLPointer& cpml,
    const SGradParam& grad_ctx,
    SolverContext solver
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    const float* sxx = sxx_ws + shift;
    const float* szz = szz_ws + shift;
    const float* sxz = sxz_ws + shift;

    float dsxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(szz, ix, 0, iz, grad_ctx);

    bool in_pml = false;
    if constexpr (!interior_path<Path>) {
        in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                 (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    }
    if (in_pml) {
        const int pml_idx = iz * solver.nx + ix;   // 2-D M-PML profiles
        const float az = cpml.az[pml_idx];
        const float bz = cpml.bz[pml_idx];
        const float ax = cpml.ax[pml_idx];
        const float bx = cpml.bx[pml_idx];

        f.m_sxxx[idx] = ax * f.m_sxxx[idx] + bx * dsxx_dx;
        dsxx_dx += f.m_sxxx[idx];
        f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
        dsxz_dz += f.m_sxzz[idx];
        f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
        dsxz_dx += f.m_sxzx[idx];
        f.m_szzz[idx] = az * f.m_szzz[idx] + bz * dszz_dz;
        dszz_dz += f.m_szzz[idx];
    }

    const float scale = solver.dt * solver.dt / m.rho[idx];
    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + scale * (dsxx_dx + dsxz_dz);
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + scale * (dsxz_dx + dszz_dz);

    if (optional_output<Path>(u_this_t)) {
        float* u_this_b = u_this_t + b * spatial_size;
        const int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.ux_nxt[idx];
        u_this_b[1 * comp_stride + idx] = f.uz_nxt[idx];
    }
}

template<int Order>
__global__ void tti2nd_displacement_kernel(
    WavefieldPointer wf,
    StiffnessPointer model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    float* __restrict__ u_this_t,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_displacement_kernel_cell<Order>(CellIndex2{ix, iz, b}, wf, model, sxx_ws, szz_ws, sxz_ws,
        u_this_t, cpml, grad_ctx, solver);
}

using tti2nd_displacement_kernel_fn = void (*)(WavefieldPointer, StiffnessPointer,
    const float*, const float*, const float*, float*, ElasticCPMLPointer, SGradParam,
    SolverContext);
SWEEP_BY_ORDER_DECL(tti2nd_displacement_kernel_fn, tti2nd_displacement_kernel_by_order);

// nopml stress: raw strains from u_now (no memory update) -> stress scratch.
// Shared by the bs time-reversal replay and by the checkpoint re-forward
// interior; identical to tti2nd_stress_kernel outside the PML band.
// One cell of tti2nd_stress_kernel_nopml: everything after the kernel's guards
// (see launch/cell_path.h).  Straight-line body: the interior box is the kernel's guard box,
// tti2nd_stress_nopml_box<Order>().
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_stress_kernel_nopml_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const SGradParam& grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    float gxux = sgradient<2, Order, X, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(f.ux, ix, 0, iz, grad_ctx);
    float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);
    float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(f.uz, ix, 0, iz, grad_ctx);

    const float exz = gzux + gxuz;
    sxx_ws[shift + idx] = m.C11[idx] * gxux + m.C13[idx] * gzuz + m.C15[idx] * exz;
    szz_ws[shift + idx] = m.C13[idx] * gxux + m.C33[idx] * gzuz + m.C35[idx] * exz;
    sxz_ws[shift + idx] = m.C15[idx] * gxux + m.C35[idx] * gzuz + m.C55[idx] * exz;
}

template<int Order>
__global__ void tti2nd_stress_kernel_nopml(
    WavefieldPointer wf,
    StiffnessPointer model,
    SGradParam grad_ctx,
    SolverContext solver,
    float* __restrict__ sxx_ws,
    float* __restrict__ szz_ws,
    float* __restrict__ sxz_ws
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    // [halo - M, n - halo + M): the stresses the reversed displacement update
    // reads -- but never the outer M-cell halo, where the forward stress kernel
    // writes nothing (its stress stays zero) and where DIFF_FORWARD's taps
    // (ix - (M-1) .. ix + M) would leave the array once abcn < M - 1.
    if (ix < halo - M || ix >= solver.nx - halo + M || iz < halo - M || iz >= solver.nz - halo + M ||
        ix < M || ix >= solver.nx - M || iz < M || iz >= solver.nz - M)
        return;

    tti2nd_stress_kernel_nopml_cell<Order>(CellIndex2{ix, iz, b}, wf, model, grad_ctx, solver,
        sxx_ws, szz_ws, sxz_ws);
}

using tti2nd_stress_kernel_nopml_fn = void (*)(WavefieldPointer, StiffnessPointer,
    SGradParam, SolverContext, float*, float*, float*);
SWEEP_BY_ORDER_DECL(tti2nd_stress_kernel_nopml_fn, tti2nd_stress_kernel_nopml_by_order);

// nopml time-REVERSED displacement update: with (u_now = U_t, u_pre = U_{t+1})
// writes U_{t-1} (up to the source term the caller re-adds) into u_next.
// One cell of tti2nd_displacement_kernel_nopml_rev: everything after the kernel's guards
// (see launch/cell_path.h).  Straight-line body: the interior box is the kernel's guard box,
// tti2nd_displacement_nopml_rev_box<Order>().
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_displacement_kernel_nopml_rev_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    const SGradParam& grad_ctx,
    SolverContext solver
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    const float* sxx = sxx_ws + shift;
    const float* szz = szz_ws + shift;
    const float* sxz = sxz_ws + shift;

    float dsxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(szz, ix, 0, iz, grad_ctx);

    const float scale = solver.dt * solver.dt / m.rho[idx];
    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + scale * (dsxx_dx + dsxz_dz);
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + scale * (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void tti2nd_displacement_kernel_nopml_rev(
    WavefieldPointer wf,
    StiffnessPointer model,
    const float* __restrict__ sxx_ws,
    const float* __restrict__ szz_ws,
    const float* __restrict__ sxz_ws,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int M = is_runtime ? solver.M : M_static;
    const int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_displacement_kernel_nopml_rev_cell<Order>(CellIndex2{ix, iz, b}, wf, model, sxx_ws,
        szz_ws, sxz_ws, grad_ctx, solver);
}

using tti2nd_displacement_kernel_nopml_rev_fn = void (*)(WavefieldPointer,
    StiffnessPointer, const float*, const float*, const float*, SGradParam,
    SolverContext);
SWEEP_BY_ORDER_DECL(tti2nd_displacement_kernel_nopml_rev_fn, tti2nd_displacement_kernel_nopml_rev_by_order);

// ---------------------------------------------------------------------------
// Adjoint (exact discrete transpose; unsigned convention — the two spatial
// transpose signs cancel between the sigma-side and u-side scatters)
// ---------------------------------------------------------------------------

// K1: bar of the four CPML-corrected stress-divergence terms from the adjoint
// displacement lam_{t+1} (in wf.ux/uz), transposing the outer CPML recursion.
// One cell of tti2nd_adjoint_div_prepare: everything after the kernel's guards
// (see launch/cell_path.h).  No halo guard: the interior box is
// tti2nd_interior_box<Order>() itself (the legacy abcn + halo test).
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_adjoint_div_prepare_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const ElasticCPMLPointer& cpml,
    SolverContext solver,
    float* __restrict__ q_sxxx,
    float* __restrict__ q_sxzz,
    float* __restrict__ q_sxzx,
    float* __restrict__ q_szzz
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    const float scale = solver.dt * solver.dt / m.rho[idx];
    const float bar_dsxx_dx = scale * f.ux[idx];
    const float bar_dsxz_dz = scale * f.ux[idx];
    const float bar_dsxz_dx = scale * f.uz[idx];
    const float bar_dszz_dz = scale * f.uz[idx];

    bool in_pml = false;
    if constexpr (!interior_path<Path>) {
        in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                 (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    }
    if (!in_pml) {
        q_sxxx[shift] = bar_dsxx_dx;
        q_sxzz[shift] = bar_dsxz_dz;
        q_sxzx[shift] = bar_dsxz_dx;
        q_szzz[shift] = bar_dszz_dz;
        return;
    }

    const int pml_idx = iz * solver.nx + ix;   // 2-D M-PML profiles
    const float az = cpml.az[pml_idx];
    const float bz = cpml.bz[pml_idx];
    const float ax = cpml.ax[pml_idx];
    const float bx = cpml.bx[pml_idx];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    q_sxxx[shift] = bar_dsxx_dx + bx * tmp_sxxx;
    q_sxzz[shift] = bar_dsxz_dz + bz * tmp_sxzz;
    q_sxzx[shift] = bar_dsxz_dx + bx * tmp_sxzx;
    q_szzz[shift] = bar_dszz_dz + bz * tmp_szzz;

    f.m_sxxx[idx] = ax * tmp_sxxx;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_szzz[idx] = az * tmp_szzz;
}

template<int Order>
__global__ void tti2nd_adjoint_div_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ q_sxxx,
    float* __restrict__ q_sxzz,
    float* __restrict__ q_sxzx,
    float* __restrict__ q_szzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    tti2nd_adjoint_div_prepare_cell<Order>(CellIndex2{ix, iz, b}, wf, model, cpml, solver, q_sxxx,
        q_sxzz, q_sxzx, q_szzz);
}

using tti2nd_adjoint_div_prepare_fn = void (*)(WavefieldPointer, StiffnessPointer,
    ElasticCPMLPointer, SolverContext, float*, float*, float*, float*);
SWEEP_BY_ORDER_DECL(tti2nd_adjoint_div_prepare_fn, tti2nd_adjoint_div_prepare_by_order);

// K2: scatter q through the transposed outer derivatives into bar-sigma,
// transpose the constitutive row into bar-strains, transpose the inner CPML
// recursion -> p workspace.
// One cell of tti2nd_adjoint_strain_prepare: everything after the kernel's guards
// (see launch/cell_path.h).  Interior box: tti2nd_interior_box<Order>() (the legacy abcn + halo
// test, as the kernel spells it) inside tti2nd_halo_box<Order>() (the guard).
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_adjoint_strain_prepare_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const StiffnessPointer& model,
    const ElasticCPMLPointer& cpml,
    const SGradParam& grad_ctx,
    SolverContext solver,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    float* __restrict__ p_gxux,
    float* __restrict__ p_gzux,
    float* __restrict__ p_gxuz,
    float* __restrict__ p_gzuz
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);

    const float* qxx = q_sxxx + shift;
    const float* qxzz = q_sxzz + shift;
    const float* qxzx = q_sxzx + shift;
    const float* qzz = q_szzz + shift;

    // Transpose of DIFF_BACKWARD is the (unsigned) DIFF_FORWARD scatter.
    float bar_sxx = sgradient<2, Order, X, DIFF_FORWARD>(qxx, ix, 0, iz, grad_ctx);
    float bar_szz = sgradient<2, Order, Z, DIFF_FORWARD>(qzz, ix, 0, iz, grad_ctx);
    float bar_sxz = sgradient<2, Order, Z, DIFF_FORWARD>(qxzz, ix, 0, iz, grad_ctx)
                  + sgradient<2, Order, X, DIFF_FORWARD>(qxzx, ix, 0, iz, grad_ctx);

    float bar_gxux = m.C11[idx] * bar_sxx + m.C13[idx] * bar_szz + m.C15[idx] * bar_sxz;
    float bar_gzuz = m.C13[idx] * bar_sxx + m.C33[idx] * bar_szz + m.C35[idx] * bar_sxz;
    const float bar_exz = m.C15[idx] * bar_sxx + m.C35[idx] * bar_szz + m.C55[idx] * bar_sxz;
    float bar_gzux = bar_exz;
    float bar_gxuz = bar_exz;

    bool in_pml = false;
    if constexpr (!interior_path<Path>) {
        in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                 (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    }
    if (!in_pml) {
        p_gxux[shift + idx] = bar_gxux;
        p_gzux[shift + idx] = bar_gzux;
        p_gxuz[shift + idx] = bar_gxuz;
        p_gzuz[shift + idx] = bar_gzuz;
        return;
    }

    const int pml_idx = iz * solver.nx + ix;   // 2-D M-PML profiles
    const float azh = cpml.azh[pml_idx];
    const float bzh = cpml.bzh[pml_idx];
    const float axh = cpml.axh[pml_idx];
    const float bxh = cpml.bxh[pml_idx];

    float tmp_gxux = f.m_gxux[idx] + bar_gxux;
    float tmp_gzux = f.m_gzux[idx] + bar_gzux;
    float tmp_gxuz = f.m_gxuz[idx] + bar_gxuz;
    float tmp_gzuz = f.m_gzuz[idx] + bar_gzuz;

    p_gxux[shift + idx] = bar_gxux + bxh * tmp_gxux;
    p_gzux[shift + idx] = bar_gzux + bzh * tmp_gzux;
    p_gxuz[shift + idx] = bar_gxuz + bxh * tmp_gxuz;
    p_gzuz[shift + idx] = bar_gzuz + bzh * tmp_gzuz;

    f.m_gxux[idx] = axh * tmp_gxux;
    f.m_gzux[idx] = azh * tmp_gzux;
    f.m_gxuz[idx] = axh * tmp_gxuz;
    f.m_gzuz[idx] = azh * tmp_gzuz;
}

template<int Order>
__global__ void tti2nd_adjoint_strain_prepare(
    WavefieldPointer wf,
    StiffnessPointer model,
    ElasticCPMLPointer cpml,
    SGradParam grad_ctx,
    SolverContext solver,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    float* __restrict__ p_gxux,
    float* __restrict__ p_gzux,
    float* __restrict__ p_gxuz,
    float* __restrict__ p_gzuz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_adjoint_strain_prepare_cell<Order>(CellIndex2{ix, iz, b}, wf, model, cpml, grad_ctx,
        solver, q_sxxx, q_sxzz, q_sxzx, q_szzz, p_gxux, p_gzux, p_gxuz, p_gzuz);
}

using tti2nd_adjoint_strain_prepare_fn = void (*)(WavefieldPointer, StiffnessPointer,
    ElasticCPMLPointer, SGradParam, SolverContext, const float*, const float*,
    const float*, const float*, float*, float*, float*, float*);
SWEEP_BY_ORDER_DECL(tti2nd_adjoint_strain_prepare_fn, tti2nd_adjoint_strain_prepare_by_order);

// K3: scatter p through the transposed inner derivatives and advance the
// adjoint leapfrog: lam_t = 2 lam_{t+1} - lam_{t+2} + contribution, written
// to the (recycled) u_next buffer; the host then rotates the triple buffer.
// One cell of tti2nd_adjoint_displacement_apply: everything after the kernel's guards
// (see launch/cell_path.h).  Straight-line body: the interior box is the kernel's guard box,
// tti2nd_halo_box<Order>().
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_adjoint_displacement_apply_cell(
    CellIndex2 cell,
    const WavefieldPointer& wf,
    const float* __restrict__ p_gxux,
    const float* __restrict__ p_gzux,
    const float* __restrict__ p_gxuz,
    const float* __restrict__ p_gzuz,
    const SGradParam& grad_ctx,
    SolverContext solver
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    const float* pxx = p_gxux + shift;
    const float* pzx = p_gzux + shift;
    const float* pxz = p_gxuz + shift;
    const float* pzz = p_gzuz + shift;

    // Transpose of DIFF_FORWARD is the (unsigned) DIFF_BACKWARD scatter.
    const float contrib_x =
        sgradient<2, Order, X, DIFF_BACKWARD>(pxx, ix, 0, iz, grad_ctx) +
        sgradient<2, Order, Z, DIFF_BACKWARD>(pzx, ix, 0, iz, grad_ctx);
    const float contrib_z =
        sgradient<2, Order, X, DIFF_BACKWARD>(pxz, ix, 0, iz, grad_ctx) +
        sgradient<2, Order, Z, DIFF_BACKWARD>(pzz, ix, 0, iz, grad_ctx);

    f.ux_nxt[idx] = 2.f * f.ux[idx] - f.ux_pre[idx] + contrib_x;
    f.uz_nxt[idx] = 2.f * f.uz[idx] - f.uz_pre[idx] + contrib_z;
}

template<int Order>
__global__ void tti2nd_adjoint_displacement_apply(
    WavefieldPointer wf,
    const float* __restrict__ p_gxux,
    const float* __restrict__ p_gzux,
    const float* __restrict__ p_gxuz,
    const float* __restrict__ p_gzuz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_adjoint_displacement_apply_cell<Order>(CellIndex2{ix, iz, b}, wf, p_gxux, p_gzux, p_gxuz,
        p_gzuz, grad_ctx, solver);
}

using tti2nd_adjoint_displacement_apply_fn = void (*)(WavefieldPointer, const float*,
    const float*, const float*, const float*, SGradParam, SolverContext);
SWEEP_BY_ORDER_DECL(tti2nd_adjoint_displacement_apply_fn, tti2nd_adjoint_displacement_apply_by_order);

// Imaging: the six stiffness gradients correlate bar-sigma (rebuilt from the
// q workspace, one unsigned transpose layer => single -1 scale) with the raw
// strains of the forward displacement U_t; the rho gradient correlates
// lam_{t+1} with the leapfrog second time difference (source term excluded —
// compensated at the source cells by tti2nd_rho_grad_source_correction).
// One cell of tti2nd_calculate_grad: everything after the kernel's guards
// (see launch/cell_path.h).  Straight-line body (no physical-box crop): the interior box is the
// kernel's guard box, tti2nd_halo_box<Order>().
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void tti2nd_calculate_grad_cell(
    CellIndex2 cell,
    const WavefieldPointer& adj,
    const StiffnessPointer& model,
    const StiffnessGradPointer& grad,
    const float* __restrict__ fux,
    const float* __restrict__ fuz,
    const float* __restrict__ fux_next,
    const float* __restrict__ fuz_next,
    const float* __restrict__ fux_prev,
    const float* __restrict__ fuz_prev,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    const SGradParam& grad_ctx,
    SolverContext solver
)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    const int spatial_size = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int shift = b * spatial_size;

    auto a = adj.offset(b, spatial_size);
    auto m = model.offset(b, spatial_size);
    auto g = grad.offset(b, spatial_size);

    const float* ux = fux + shift;
    const float* uz = fuz + shift;
    const float* qxx = q_sxxx + shift;
    const float* qxzz = q_sxzz + shift;
    const float* qxzx = q_sxzx + shift;
    const float* qzz = q_szzz + shift;

    const float gxux = sgradient<2, Order, X, DIFF_FORWARD>(ux, ix, 0, iz, grad_ctx);
    const float gzux = sgradient<2, Order, Z, DIFF_FORWARD>(ux, ix, 0, iz, grad_ctx);
    const float gxuz = sgradient<2, Order, X, DIFF_FORWARD>(uz, ix, 0, iz, grad_ctx);
    const float gzuz = sgradient<2, Order, Z, DIFF_FORWARD>(uz, ix, 0, iz, grad_ctx);
    const float exz = gzux + gxuz;

    const float bar_sxx = sgradient<2, Order, X, DIFF_FORWARD>(qxx, ix, 0, iz, grad_ctx);
    const float bar_szz = sgradient<2, Order, Z, DIFF_FORWARD>(qzz, ix, 0, iz, grad_ctx);
    const float bar_sxz = sgradient<2, Order, Z, DIFF_FORWARD>(qxzz, ix, 0, iz, grad_ctx)
                        + sgradient<2, Order, X, DIFF_FORWARD>(qxzx, ix, 0, iz, grad_ctx);

    g.C11[idx] -= bar_sxx * gxux;
    g.C33[idx] -= bar_szz * gzuz;
    g.C13[idx] -= bar_sxx * gzuz + bar_szz * gxux;
    g.C55[idx] -= bar_sxz * exz;
    g.C15[idx] -= bar_sxx * exz + bar_sxz * gxux;
    g.C35[idx] -= bar_szz * exz + bar_sxz * gzuz;

    // d U_{t+1} / d rho = -(U_{t+1} - 2 U_t + U_{t-1} - S_t) / rho; the S_t
    // part is handled at the source cells by the correction kernel.
    g.rho[idx] += (
        a.ux[idx] * (2.f * ux[idx] - fux_next[shift + idx] - fux_prev[shift + idx]) +
        a.uz[idx] * (2.f * uz[idx] - fuz_next[shift + idx] - fuz_prev[shift + idx])
    ) / m.rho[idx];
}

template<int Order>
__global__ void tti2nd_calculate_grad(
    WavefieldPointer adj,         // lam_{t+1} in .ux/.uz
    StiffnessPointer model,
    StiffnessGradPointer grad,
    const float* __restrict__ fux,       // U_t
    const float* __restrict__ fuz,
    const float* __restrict__ fux_next,  // U_{t+1} (with its source)
    const float* __restrict__ fuz_next,
    const float* __restrict__ fux_prev,  // U_{t-1} (with its source)
    const float* __restrict__ fuz_prev,
    const float* __restrict__ q_sxxx,
    const float* __restrict__ q_sxzz,
    const float* __restrict__ q_sxzx,
    const float* __restrict__ q_szzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    const int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    tti2nd_calculate_grad_cell<Order>(CellIndex2{ix, iz, b}, adj, model, grad, fux, fuz, fux_next,
        fuz_next, fux_prev, fuz_prev, q_sxxx, q_sxzz, q_sxzx, q_szzz, grad_ctx, solver);
}

using tti2nd_calculate_grad_fn = void (*)(WavefieldPointer, StiffnessPointer,
    StiffnessGradPointer, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    SGradParam, SolverContext);
SWEEP_BY_ORDER_DECL(tti2nd_calculate_grad_fn, tti2nd_calculate_grad_by_order);

// Source-cell compensation for the rho imaging: the stored U_{t+1} contains
// the injected wavelet S_t, which is rho-independent; add back lam*S/rho.
// (Defined in kernels.cu: a definition here is compiled into every TU that
// includes this header.)
__global__ void tti2nd_rho_grad_source_correction(
    float* __restrict__ grad_rho,
    const float* __restrict__ adj_field,   // lam_{t+1} component matching the source field
    const float* __restrict__ rho,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);

} // namespace elastic_tti_2nd2d

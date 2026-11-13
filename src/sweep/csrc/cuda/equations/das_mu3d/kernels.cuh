#pragma once
#include "../../launch/by_order.cuh"
#include "../../launch/cell_path.h"

#include "../elastic3d/kernels.cuh"

#include "../../common/context.h"
#include "../../common/das_mu.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_DAS_MU3D_STRESS_STRAIN(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(das_mu3d_stress_strain_kernel_by_order, order, grid, block, __VA_ARGS__)

#define LAUNCH_DAS_MU3D_STRESS_STRAIN_ADJOINT_PREPARE(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(das_mu3d_stress_strain_adjoint_prepare_by_order, order, grid, block, __VA_ARGS__)

// ``wf.offset(b, spatial_size)`` for a cell (launch/cell_path.h), as
// elastic3d_cell_offset: offset() shifts the CPML memory fields behind
// ``if (p)``.  Both das_mu3d cells read the nine m_v* on every cell, on their
// own path as on the kernel's (the stress cell updates them everywhere; the
// prepare reads them before its interior test), so on an interior path those
// are non-null and the shift is unconditional: the same pointers, without
// the in-loop selects that keep the row from vectorising.  Spelled in
// integers, like elastic3d_cell_offset.  The m_s* fields neither cell uses
// keep offset()'s shift.  On Any: offset() itself.
template<CellPath Path>
__device__ __forceinline__ DasMuWavefieldPointer3D das_mu3d_cell_offset(
    const DasMuWavefieldPointer3D& wf, int b, int spatial_size)
{
    DasMuWavefieldPointer3D out = wf.offset(b, spatial_size);
    if constexpr (interior_path<Path>) {
        const uintptr_t shift = (uintptr_t)(intptr_t)(b * spatial_size) * sizeof(float);
        out.m_vxx = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vxx) + shift);
        out.m_vxy = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vxy) + shift);
        out.m_vxz = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vxz) + shift);
        out.m_vyx = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vyx) + shift);
        out.m_vyy = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vyy) + shift);
        out.m_vyz = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vyz) + shift);
        out.m_vzx = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vzx) + shift);
        out.m_vzy = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vzy) + shift);
        out.m_vzz = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(wf.m_vzz) + shift);
    }
    return out;
}

// One cell of das_mu3d_stress_strain_kernel: everything after the kernel's
// guards (launch/cell_path.h).  The kernel has no interior branch -- the CPML
// update runs on every cell, with zero coefficients off the band -- so its
// interior box is the whole halo box, below the FS helper's mirror rows
// (common/host_rows.h elastic3d_fs_box; the surface row with its Robertsson
// block is out); free surface as the ELASTIC_FS_Z flag; optional output:
// u_this.
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void das_mu3d_stress_strain_cell(
    int ix, int iy, int iz, int b,
    const DasMuWavefieldPointer3D& wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    float* __restrict__ u_this,
    const SGradParam& grad_ctx,
    const ElasticCPMLPointer& cpml,
    const SolverContext& solver
)
{
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = das_mu3d_cell_offset<Path>(wf, b, spatial_size);
    float* u_this_b = optional_output<Path>(u_this) ? u_this + b * spatial_size : nullptr;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD, Path>(f.vx, ix, iy, iz, grad_ctx, solver, false);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD, Path>(f.vy, ix, iy, iz, grad_ctx, solver, false);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD, Path>(f.vz, ix, iy, iz, grad_ctx, solver, true, true);

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];
    f.m_vyy[idx] = ay * f.m_vyy[idx] + by * dvy_dy;
    dvy_dy += f.m_vyy[idx];
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];

    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];

    f.m_vxy[idx] = ayh * f.m_vxy[idx] + byh * dvx_dy;
    dvx_dy += f.m_vxy[idx];
    f.m_vyx[idx] = axh * f.m_vyx[idx] + bxh * dvy_dx;
    dvy_dx += f.m_vyx[idx];
    f.m_vyz[idx] = azh * f.m_vyz[idx] + bzh * dvy_dz;
    dvy_dz += f.m_vyz[idx];
    f.m_vzy[idx] = ayh * f.m_vzy[idx] + byh * dvz_dy;
    dvz_dy += f.m_vzy[idx];

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];
    float div_v = dvx_dx + dvy_dy + dvz_dz;

    f.sxx[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvx_dx);
    f.syy[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvy_dy);
    f.szz[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvz_dz);
    f.sxy[idx] += solver.dt * mu_ * (dvx_dy + dvy_dx);
    f.sxz[idx] += solver.dt * mu_ * (dvx_dz + dvz_dx);
    f.syz[idx] += solver.dt * mu_ * (dvy_dz + dvz_dy);

    f.exx[idx] += solver.dt * dvx_dx;
    f.eyy[idx] += solver.dt * dvy_dy;
    f.ezz[idx] += solver.dt * dvz_dz;
    f.exy[idx] += 0.5f * solver.dt * (dvx_dy + dvy_dx);
    f.exz[idx] += 0.5f * solver.dt * (dvx_dz + dvz_dx);
    f.eyz[idx] += 0.5f * solver.dt * (dvy_dz + dvz_dy);

    if (elastic3d_is_top_free_surface_row<Path>(solver, iz)) {
        // Robertsson tangential FS correction (3-D): sxx/syy each get
        // -dt*lam*(lam/(lam+2mu)*(dvx_dx+dvy_dy)+dvz_dz) added; matches eager
        // das_mu3d.  Strains exx/eyy/ezz are unchanged.
        float _rob = -solver.dt * lam *
            (lam / (lam + 2.f * mu_) * (dvx_dx + dvy_dy) + dvz_dz);
        f.sxx[idx] += _rob;
        f.syy[idx] += _rob;
        f.szz[idx] = 0.f;   // z-low FS: zero only szz; sxz/syz are +h/2 medium values (fix)
    }

    if (optional_output<Path>(u_this_b)) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
    }
}

template<int Order>
__global__ void das_mu3d_stress_strain_kernel(
    DasMuWavefieldPointer3D wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    float* __restrict__ u_this,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    das_mu3d_stress_strain_cell<Order>(ix, iy, iz, b, wf, lambda, mu, u_this, grad_ctx, cpml, solver);
}

using das_mu3d_stress_strain_kernel_fn = void (*)(DasMuWavefieldPointer3D, const float*,
    const float*, float*, SGradParam, ElasticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(das_mu3d_stress_strain_kernel_fn, das_mu3d_stress_strain_kernel_by_order);

// The prepare cell's index, by value (as elastic_tti_2nd2d's CellIndex2):
// with plain int parameters nvcc emits the interior test's compares with
// their operands swapped; a field of a by-value struct is a load, ranked like
// the kernel's own decoded index, and the split kernel keeps its SASS
// (docs/dev/host_core.md rule 7).  On the host it is free.
struct DasMu3dCellIndex {
    int ix, iy, iz, b;
};

// One cell of das_mu3d_stress_strain_adjoint_prepare: everything after the
// kernel's guards (launch/cell_path.h).  Interior box: the interior test's
// box (the physical box shrunk by one cell on every face, not cut-aware, as
// the kernel spells it), which holds no free-surface row; no FS derivative
// helper, so no flag; no optional output.
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__
void das_mu3d_stress_strain_adjoint_prepare_cell(
    DasMu3dCellIndex cell,
    const DasMuWavefieldPointer3D& wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    const ElasticCPMLPointer& cpml,
    const SolverContext& solver,
    float* __restrict__ qxx,
    float* __restrict__ qxy,
    float* __restrict__ qxz,
    float* __restrict__ qyx,
    float* __restrict__ qyy,
    float* __restrict__ qyz,
    float* __restrict__ qzx,
    float* __restrict__ qzy,
    float* __restrict__ qzz
)
{
    const int ix = cell.ix, iy = cell.iy, iz = cell.iz, b = cell.b;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = das_mu3d_cell_offset<Path>(wf, b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;
    const int out = b * spatial_size + idx;

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];
    float l2m = lam + 2.f * mu_;

    float bar_sxx = f.sxx[idx];
    float bar_syy = f.syy[idx];
    float bar_szz = f.szz[idx];
    float bar_sxy = f.sxy[idx];
    float bar_sxz = f.sxz[idx];
    float bar_syz = f.syz[idx];
    if (elastic3d_is_top_free_surface_row<Path>(solver, iz)) {
        bar_szz = 0.f;   // z-low FS: zero only szz; keep sxz/syz (+h/2 medium values, fix)
        f.szz[idx] = 0.f;
    }

    float bar_exx = -f.exx[idx];
    float bar_eyy = -f.eyy[idx];
    float bar_ezz = -f.ezz[idx];
    float bar_exy = -f.exy[idx];
    float bar_exz = -f.exz[idx];
    float bar_eyz = -f.eyz[idx];

    // Transpose of the stress/strain-from-velocity update.  On a z-low free
    // surface the Robertsson fix makes sxx=coef*dvx_dx+coef2*dvy_dy,
    // syy=coef2*dvx_dx+coef*dvy_dy, szz=0 (bar_szz=0); strains exx/eyy/ezz are
    // unchanged so their bar_e* remain, and dvz_dz keeps only bar_ezz.
    float bar_dvx_dx, bar_dvy_dy, bar_dvz_dz;
    if (elastic3d_is_top_free_surface_row<Path>(solver, iz)) {
        float coef  = 4.f * mu_ * (lam + mu_) / l2m;
        float coef2 = 2.f * lam * mu_ / l2m;
        bar_dvx_dx = solver.dt * (coef  * bar_sxx + coef2 * bar_syy + bar_exx);
        bar_dvy_dy = solver.dt * (coef2 * bar_sxx + coef  * bar_syy + bar_eyy);
        bar_dvz_dz = solver.dt * (bar_ezz);
    } else {
        bar_dvx_dx = solver.dt * (l2m * bar_sxx + lam * bar_syy + lam * bar_szz + bar_exx);
        bar_dvy_dy = solver.dt * (lam * bar_sxx + l2m * bar_syy + lam * bar_szz + bar_eyy);
        bar_dvz_dz = solver.dt * (lam * bar_sxx + lam * bar_syy + l2m * bar_szz + bar_ezz);
    }
    float bar_dvx_dy = solver.dt * (mu_ * bar_sxy + 0.5f * bar_exy);
    float bar_dvy_dx = solver.dt * (mu_ * bar_sxy + 0.5f * bar_exy);
    float bar_dvx_dz = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);
    float bar_dvz_dx = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);
    float bar_dvy_dz = solver.dt * (mu_ * bar_syz + 0.5f * bar_eyz);
    float bar_dvz_dy = solver.dt * (mu_ * bar_syz + 0.5f * bar_eyz);

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxy = f.m_vxy[idx] + bar_dvx_dy;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyy = f.m_vyy[idx] + bar_dvy_dy;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzy = f.m_vzy[idx] + bar_dvz_dy;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();
    bool interior = true;
    if constexpr (!interior_path<Path>)
        interior =
            ix >= x0 + 1 && ix < x1 - 1 &&
            iy >= y0 + 1 && iy < y1 - 1 &&
            iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        qxx[out] = bar_dvx_dx;
        qxy[out] = bar_dvx_dy;
        qxz[out] = bar_dvx_dz;
        qyx[out] = bar_dvy_dx;
        qyy[out] = bar_dvy_dy;
        qyz[out] = bar_dvy_dz;
        qzx[out] = bar_dvz_dx;
        qzy[out] = bar_dvz_dy;
        qzz[out] = bar_dvz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    qxx[out] = bar_dvx_dx + bx * tmp_vxx;
    qxy[out] = bar_dvx_dy + byh * tmp_vxy;
    qxz[out] = bar_dvx_dz + bzh * tmp_vxz;
    qyx[out] = bar_dvy_dx + bxh * tmp_vyx;
    qyy[out] = bar_dvy_dy + by * tmp_vyy;
    qyz[out] = bar_dvy_dz + bzh * tmp_vyz;
    qzx[out] = bar_dvz_dx + bxh * tmp_vzx;
    qzy[out] = bar_dvz_dy + byh * tmp_vzy;
    qzz[out] = bar_dvz_dz + bz * tmp_vzz;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vxy[idx] = ayh * tmp_vxy;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vyx[idx] = axh * tmp_vyx;
    f.m_vyy[idx] = ay * tmp_vyy;
    f.m_vyz[idx] = azh * tmp_vyz;
    f.m_vzx[idx] = axh * tmp_vzx;
    f.m_vzy[idx] = ayh * tmp_vzy;
    f.m_vzz[idx] = az * tmp_vzz;
}

template<int Order>
__global__ void das_mu3d_stress_strain_adjoint_prepare(
    DasMuWavefieldPointer3D wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qxy,
    float* __restrict__ qxz,
    float* __restrict__ qyx,
    float* __restrict__ qyy,
    float* __restrict__ qyz,
    float* __restrict__ qzx,
    float* __restrict__ qzy,
    float* __restrict__ qzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    das_mu3d_stress_strain_adjoint_prepare_cell<Order>(DasMu3dCellIndex{ix, iy, iz, b}, wf, lambda, mu,
                                                       cpml, solver, qxx, qxy, qxz, qyx, qyy, qyz, qzx,
                                                       qzy, qzz);
}

using das_mu3d_stress_strain_adjoint_prepare_fn = void (*)(DasMuWavefieldPointer3D,
    const float*, const float*, ElasticCPMLPointer, SolverContext, float*, float*,
    float*, float*, float*, float*, float*, float*, float*);
SWEEP_BY_ORDER_DECL(das_mu3d_stress_strain_adjoint_prepare_fn, das_mu3d_stress_strain_adjoint_prepare_by_order);

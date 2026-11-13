#pragma once
#include "../../launch/by_order.cuh"
#include "../../launch/cell_path.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

#define ACOUSTIC_BORN3D_SINGLE(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic3d_single_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_BORN3D_SINGLE_NOPML(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic3d_single_nopml_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_BORN3D_COUPLED(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_born3d_coupled_by_order, order, grid, block, __VA_ARGS__)

// Exact transpose of a field's step (acoustic_born3d_adjoint_exact below).
#define ACOUSTIC_BORN3D_ADJOINT(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_born3d_adjoint_exact_by_order, order, grid, block, __VA_ARGS__)

// The four stencil kernels below are split per launch/cell_path.h: the
// __global__ function keeps its index decode and guards and calls a _cell
// body (the rest of the kernel, unchanged and in the same order), which the
// host core's row launchers (kernels.cu) also run on an interior path.  The
// interior boxes, in (ix, iy, iz), are spelled next to the launchers.

template <int Order>
__device__ inline float acoustic_cpml_update_3d(
    AcousticWavefieldPointer f,
    int ix,
    int iy,
    int iz,
    int idx,
    const AcousticCPMLPointer& cpml,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    bool use_pml,
    const SolverContext& solver,
    int halo
) {
    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);
    if (!use_pml) {
        return lap_x + lap_y + lap_z;
    }

    // Interior fast-path: ax/bx/dbxdx vanish, so w_sum reduces to the sum of
    // laplacians and the aux fields stay zero. Skip the 14+ extra loads/stores.
    // in_cpml: the band AND the cell past it, where db = np.gradient(b) is
    // still nonzero (dropping it there reflected off the CPML).
    bool in_pml = solver.in_cpml_3d(ix, iy, iz, halo);
    if (!in_pml) {
        return lap_x + lap_y + lap_z;
    }

    float w_sum = 0.0f;

    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float ax_ = cpml.ax[ix];
    float bx_ = cpml.bx[ix];
    float dbxdx_ = cpml.dbxdx[ix];
    float ay_ = cpml.ay[iy];
    float by_ = cpml.by[iy];
    float dbydy_ = cpml.dbydy[iy];
    float az_ = cpml.az[iz];
    float bz_ = cpml.bz[iz];
    float dbzdz_ = cpml.dbzdz[iz];

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + ax_ * dpsixdx + daxdx * f.psix[idx];
    w_sum += (1.0f + bx_) * tmpx + ax_ * f.zetax[idx];
    (f.psixn ? f.psixn : f.psix)[idx] = bx_ * dudx + ax_ * f.psix[idx];   // race-free fwd double-buffer
    f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpy = ((1.0f + by_) * lap_y + dbydy_ * dudy) + ay_ * dpsiydy + daydy * f.psiy[idx];
    w_sum += (1.0f + by_) * tmpy + ay_ * f.zetay[idx];
    (f.psiyn ? f.psiyn : f.psiy)[idx] = by_ * dudy + ay_ * f.psiy[idx];   // race-free fwd double-buffer
    f.zetay[idx] = by_ * tmpy + ay_ * f.zetay[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + az_ * dpsizdz + dazdz * f.psiz[idx];
    w_sum += (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];
    (f.psizn ? f.psizn : f.psiz)[idx] = bz_ * dudz + az_ * f.psiz[idx];   // race-free fwd double-buffer
    f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];

    return w_sum;
}

// One cell of acoustic3d_single: everything after the kernel's bounds guards
// (launch/cell_path.h).  Interior box: the cells acoustic_cpml_update_3d's
// in_cpml_3d test sends down its laplacian-only branch (kernels.cu
// born3d_forward_box).  There the cell asks the helper for exactly that
// branch with use_pml = false, which returns the same lap_x + lap_y + lap_z
// before any CPML test.  Optional output: u_this, which the kernel stores
// only under save_all_wavefields as well (the host launcher folds the two into
// one pointer).  On an interior path the u_this null test is answered too:
// left as a select, the store's address is not an affine access and the row
// does not vectorise ("not suitable for scatter store").
template <int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void acoustic3d_single_cell(
    int ix, int iy, int iz, int b,
    const AcousticWavefieldPointer& wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& solver
) {
    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b;
    if constexpr (interior_path<Path>)
        u_this_b = optional_output<Path>(u_this) ? u_this + b * spatial_size : nullptr;
    else
        u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float utt = (vp_b[idx] * vp_b[idx]) * acoustic_cpml_update_3d<Order>(
        f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, !interior_path<Path>, solver, halo
    );
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * utt;
    if constexpr (interior_path<Path>) {
        if (optional_output<Path>(u_this_b)) {
            u_this_b[idx] = utt;
        }
    } else {
        if (save_all_wavefields && u_this_b != nullptr) {
            u_this_b[idx] = utt;
        }
    }
}

template <int Order>
__global__ void acoustic3d_single(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo || iz < halo || iz >= solver.nz - halo) {
        return;
    }

    acoustic3d_single_cell<Order>(ix, iy, iz, b, wf, save_all_wavefields, u_this, vp, lap_ctx,
                                  grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, solver);
}

using acoustic3d_single_fn = void (*)(AcousticWavefieldPointer, bool, float*,
    const float*, LaplaceParam, GradParam, GradParam, GradParam, GradParam,
    AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic3d_single_fn, acoustic3d_single_by_order);

// One cell of acoustic3d_single_nopml: everything after the kernel's bounds
// guards (launch/cell_path.h).  Every cell the guards let through takes the
// one branch (acoustic_cpml_update_3d with use_pml = false returns before any
// CPML test), so the interior box is the guard box (kernels.cu
// born3d_nopml_box); ``halo`` is the kernel's own (abcn + 2m + 1 or 2m).
// Optional output: u_this, its null test answered on an interior path (see
// acoustic3d_single_cell).
template <int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void acoustic3d_single_nopml_cell(
    int ix, int iy, int iz, int b,
    const AcousticWavefieldPointer& wf,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const LaplaceParam& lap_ctx,
    const SolverContext& solver,
    int halo
) {
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b;
    if constexpr (interior_path<Path>)
        u_this_b = optional_output<Path>(u_this) ? u_this + b * spatial_size : nullptr;
    else
        u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float utt = (vp_b[idx] * vp_b[idx]) * acoustic_cpml_update_3d<Order>(
        f, ix, iy, iz, idx, AcousticCPMLPointer{}, lap_ctx, GradParam{}, GradParam{}, GradParam{}, GradParam{}, false, solver, halo
    );
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * utt;
    if constexpr (interior_path<Path>) {
        if (optional_output<Path>(u_this_b)) {
            u_this_b[idx] = utt;
        }
    } else {
        if (u_this_b != nullptr) {
            u_this_b[idx] = utt;
        }
    }
}

template <int Order>
__global__ void acoustic3d_single_nopml(
    AcousticWavefieldPointer wf,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    int m = (Order == -1) ? solver.M : (Order / 2);
    // On a physical face the reconstruction starts just inside the boundary-
    // saving restore strip (abcn + m halo + save_width = abcn + 2m + 1).  A DD
    // CUT face has neither the abcn pad nor a restore strip -- its halo is
    // filled by the NCCL exchange -- so the reconstruction must reach in to the
    // m-wide halo there.  Keeping the abcn-based bound on a cut face skipped
    // abcn + m + 1 PHYSICAL cells beside every cut, which were then never
    // reconstructed and fed stale values to both tiles (DD gradients off by
    // 10-30% while the forward record stayed bitwise).  cut_mask == 0 leaves
    // every bound exactly as before.
    int halo = solver.abcn > 0 ? solver.abcn + 2 * m + 1 : 2 * m;
    int top_halo = solver.free_surface ? 2 * m : halo;
    int x_lo = solver.cut_x_lo() ? m : halo, x_hi = solver.cut_x_hi() ? m : halo;
    int y_lo = solver.cut_y_lo() ? m : halo, y_hi = solver.cut_y_hi() ? m : halo;
    int z_lo = solver.cut_z_lo() ? m : top_halo, z_hi = solver.cut_z_hi() ? m : halo;
    if (ix < x_lo || ix >= solver.nx - x_hi || iy < y_lo || iy >= solver.ny - y_hi || iz < z_lo || iz >= solver.nz - z_hi) {
        return;
    }

    acoustic3d_single_nopml_cell<Order>(ix, iy, iz, b, wf, u_this, vp, lap_ctx, solver, halo);
}

using acoustic3d_single_nopml_fn = void (*)(AcousticWavefieldPointer, float*,
    const float*, LaplaceParam, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic3d_single_nopml_fn, acoustic3d_single_nopml_by_order);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
// (Defined in kernels.cu: a definition here is compiled into every TU that
// includes this header.)
__global__ void compute_v2_lambda_born3d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int ny, int nz, int B
);

// Exact transpose of one step of a field advanced by acoustic_cpml_update_3d
// (the product-rule CPML), the 3-D twin of acoustic_born2nd_adjoint_exact
// (acoustic_born2d/kernels.cuh): per axis d, with g = dt^2 * v2l and
// gtmp_d = (1+b_d) g + b_d zeta*_d,
//   lambda_next = 2 lambda - lambda_prev + sum_d [ L_d((1+b_d) gtmp_d) - D_d(b_d psi*_d + db_d gtmp_d) ]
//   psi*_d'     = a_d psi*_d + da_d gtmp_d - D_d(a_d gtmp_d)
//   zeta*_d'    = a_d (zeta*_d + g)
// v2l = vp^2 * lambda for a field's own adjoint, vp^2 * (lambda_bg + mp *
// lambda_sc) for the RWI background adjoint.  An axis none of whose taps is
// in_cpml reduces to L_d(g) and leaves its (zero) aux unwritten.  New psi* /
// zeta* go to the double buffers; the caller rotates with swap_aux().
//
// One cell of acoustic_born3d_adjoint_exact: everything after the kernel's
// bounds guards (launch/cell_path.h).  Interior box: the cells none of whose
// axes has a tap in the CPML (every taps_in_cpml_* false: kernels.cu
// born3d_adjoint_box), which take the pure L* = lap(v^2 . lambda) branch; no
// optional output.  The CPML profiles stay in the AcousticCPMLPointer: the
// kernel loads them as plain LDG.E (nvcc does not take the struct members'
// __restrict__ here), and handed in as separate __restrict__ pointers they
// became LDG.E.CONSTANT -- 54 loads at order 4, 64 -> 72 registers.
template <int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void acoustic_born3d_adjoint_exact_cell(
    int ix, int iy, int iz, int b,
    const AcousticWavefieldPointer& wf,
    const float* __restrict__ v2l,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& solver
) {
    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    const int sy = solver.nx;
    const int sz = solver.nx * solver.ny;
    const int sp = sz * solver.nz;
    const int idx = iz * sz + iy * sy + ix;
    auto f = wf.offset(b, sp);
    const float* g_b = v2l + b * sp;
    const float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;     // [c0 (centre, applied as -lc[0]), c1..cM]
    const float* gc = grad_ctx_x.coeff;  // [_, c1..cM] (antisymmetric)

    // Does the tap range [i - M, i + M] meet a band face's in_cpml cells?
    // (Never in an interior path's box.)
    bool xfull = false, yfull = false, zfull = false;
    if constexpr (!interior_path<Path>) {
        xfull = solver.taps_in_cpml_x(ix, halo, halo);
        yfull = solver.taps_in_cpml_y(iy, halo, halo);
        zfull = solver.taps_in_cpml_z(iz, halo, halo);
    }

    if (!xfull && !yfull && !zfull) {
        // No tap in the CPML: L* = lap(v^2 . lambda), the transpose of the
        // forward v^2 . lap (compile-time stencils, one dt^2 at the end).
        float lap_x = laplace<3, Order, X>(g_b, ix, iy, iz, lap_ctx);
        float lap_y = laplace<3, Order, Y>(g_b, ix, iy, iz, lap_ctx);
        float lap_z = laplace<3, Order, Z>(g_b, ix, iy, iz, lap_ctx);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * (lap_x + lap_y + lap_z);
        return;
    }

    #define BORN3_GW(n) ( dt2 * g_b[n] )
    // One axis: the transpose (full) or the plain L_d(g) (off the band).
    #define BORN3_AXIS(FULL, AXIS, I, STRIDE, A, B_, DB, PSI, ZETA, PSIN, ZETAN, GCTX, INV, INV2, L_OUT, DGG_OUT) \
    {                                                                                                    \
        float Lacc, Dgg = 0.0f;                                                                          \
        if (FULL) {                                                                                      \
            const float a_ = A[I];                                                                       \
            const float da = gradient<2, Order, X>(A, I, 0, 0, GCTX);                                    \
            const float gtmp0 = B_[I] * ZETA[idx] + (1.0f + B_[I]) * BORN3_GW(idx);                     \
            Lacc = -lc[0] * ((1.0f + B_[I]) * gtmp0);                                                    \
            float Dgq = 0.0f;                                                                            \
            _Pragma("unroll")                                                                            \
            for (int k = 1; k <= (Order > 0 ? M_static : 16); ++k) {                                     \
                if (Order <= 0 && k > halo) break;                                                       \
                const int np = idx + k * STRIDE, nm = idx - k * STRIDE, ip = I + k, im = I - k;          \
                const float tp = B_[ip] * ZETA[np] + (1.0f + B_[ip]) * BORN3_GW(np);                    \
                const float tm = B_[im] * ZETA[nm] + (1.0f + B_[im]) * BORN3_GW(nm);                    \
                Lacc += lc[k] * ((1.0f + B_[ip]) * tp + (1.0f + B_[im]) * tm);                           \
                Dgg += gc[k] * ((B_[ip] * PSI[np] + DB[ip] * tp) - (B_[im] * PSI[nm] + DB[im] * tm));    \
                Dgq += gc[k] * (A[ip] * tp - A[im] * tm);                                                \
            }                                                                                            \
            Lacc *= INV2; Dgg *= INV; Dgq *= INV;                                                        \
            PSIN[idx] = a_ * PSI[idx] + da * gtmp0 - Dgq;                                                \
            ZETAN[idx] = a_ * (ZETA[idx] + BORN3_GW(idx));                                              \
        } else {                                                                                         \
            Lacc = dt2 * laplace<3, Order, AXIS>(g_b, ix, iy, iz, lap_ctx);                              \
        }                                                                                                \
        L_OUT = Lacc; DGG_OUT = Dgg;                                                                     \
    }

    float Lx, Dx, Ly, Dy, Lz, Dz;
    BORN3_AXIS(xfull, X, ix, 1,  cpml.ax, cpml.bx, cpml.dbxdx, f.psix, f.zetax, f.psixn, f.zetaxn, grad_ctx_x,
                1.0f / lap_ctx.dx, 1.0f / (lap_ctx.dx * lap_ctx.dx), Lx, Dx)
    BORN3_AXIS(yfull, Y, iy, sy, cpml.ay, cpml.by, cpml.dbydy, f.psiy, f.zetay, f.psiyn, f.zetayn, grad_ctx_y,
                1.0f / lap_ctx.dy, 1.0f / (lap_ctx.dy * lap_ctx.dy), Ly, Dy)
    BORN3_AXIS(zfull, Z, iz, sz, cpml.az, cpml.bz, cpml.dbzdz, f.psiz, f.zetaz, f.psizn, f.zetazn, grad_ctx_z,
                1.0f / lap_ctx.dz, 1.0f / (lap_ctx.dz * lap_ctx.dz), Lz, Dz)
    #undef BORN3_AXIS
    #undef BORN3_GW

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + (Lx + Ly + Lz - Dx - Dy - Dz);
}

template <int Order>
__global__ void acoustic_born3d_adjoint_exact(
    AcousticWavefieldPointer wf,
    const float* __restrict__ v2l,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz; int iz = iz_global % solver.nz;
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    acoustic_born3d_adjoint_exact_cell<Order>(ix, iy, iz, b, wf, v2l, lap_ctx, grad_ctx_x, grad_ctx_y,
                                              grad_ctx_z, cpml, solver);
}

using acoustic_born3d_adjoint_exact_fn = void (*)(AcousticWavefieldPointer, const float*,
    LaplaceParam, GradParam, GradParam, GradParam, AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_born3d_adjoint_exact_fn, acoustic_born3d_adjoint_exact_by_order);

// One cell of acoustic_born3d_coupled: everything after the kernel's bounds
// guards (launch/cell_path.h).  Interior box: as acoustic3d_single_cell's (the
// in_cpml_3d test of acoustic_cpml_update_3d, the same for both fields), where
// both helper calls take the laplacian-only branch through use_pml = false.
// Optional outputs: bg_utt (output 0) and sc_utt (output 1), which the kernel
// stores only under save_all_wavefields as well (the host launcher folds the
// flag into both pointers); their null tests answered on an interior path
// (see acoustic3d_single_cell).
template <int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void acoustic_born3d_coupled_cell(
    int ix, int iy, int iz, int b,
    const AcousticWavefieldPointer& bg,
    const AcousticWavefieldPointer& sc,
    bool save_all_wavefields,
    float* __restrict__ bg_utt,
    float* __restrict__ sc_utt,
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& solver
) {
    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = stride_z * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto bg_f = bg.offset(b, spatial_size);
    auto sc_f = sc.offset(b, spatial_size);
    float* bg_utt_b;
    float* sc_utt_b;
    if constexpr (interior_path<Path>) {
        bg_utt_b = optional_output<Path, 0>(bg_utt) ? bg_utt + b * spatial_size : nullptr;
        sc_utt_b = optional_output<Path, 1>(sc_utt) ? sc_utt + b * spatial_size : nullptr;
    } else {
        bg_utt_b = bg_utt ? bg_utt + b * spatial_size : nullptr;
        sc_utt_b = sc_utt ? sc_utt + b * spatial_size : nullptr;
    }
    const float* vp_b = vp + b * spatial_size;
    const float* mp_b = mp + b * spatial_size;

    float v2 = vp_b[idx] * vp_b[idx];
    float bg_utt_val = v2 * acoustic_cpml_update_3d<Order>(
        bg_f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, !interior_path<Path>, solver, halo
    );
    float sc_utt_val = v2 * acoustic_cpml_update_3d<Order>(
        sc_f, ix, iy, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, !interior_path<Path>, solver, halo
    );

    bg_f.u_next[idx] = 2.0f * bg_f.u_now[idx] - bg_f.u_prev[idx] + solver.dt * solver.dt * bg_utt_val;
    sc_f.u_next[idx] = 2.0f * sc_f.u_now[idx] - sc_f.u_prev[idx] + solver.dt * solver.dt * (sc_utt_val + mp_b[idx] * bg_utt_val);

    if constexpr (interior_path<Path>) {
        if (optional_output<Path, 0>(bg_utt_b)) {
            bg_utt_b[idx] = bg_utt_val;
        }
        if (optional_output<Path, 1>(sc_utt_b)) {
            sc_utt_b[idx] = sc_utt_val;
        }
    } else {
        if (save_all_wavefields && bg_utt_b != nullptr) {
            bg_utt_b[idx] = bg_utt_val;
        }
        if (save_all_wavefields && sc_utt_b != nullptr) {
            sc_utt_b[idx] = sc_utt_val;
        }
    }
}

template <int Order>
__global__ void acoustic_born3d_coupled(
    AcousticWavefieldPointer bg,
    AcousticWavefieldPointer sc,
    bool save_all_wavefields,
    float* __restrict__ bg_utt,
    float* __restrict__ sc_utt,   // RWI: the scattered u_tt, for the tomographic vp gradient
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) {
        return;
    }

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iy < halo || iy >= solver.ny - halo || iz < halo || iz >= solver.nz - halo) {
        return;
    }

    acoustic_born3d_coupled_cell<Order>(ix, iy, iz, b, bg, sc, save_all_wavefields, bg_utt, sc_utt, vp,
                                        mp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
                                        cpml, solver);
}

using acoustic_born3d_coupled_fn = void (*)(AcousticWavefieldPointer,
    AcousticWavefieldPointer, bool, float*, float*, const float*, const float*, LaplaceParam,
    GradParam, GradParam, GradParam, GradParam, AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_born3d_coupled_fn, acoustic_born3d_coupled_by_order);

__global__ void calculate_grad_born3d_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
);

__global__ void calculate_grad_born3d_mp_utt(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
);

// ---- RWI tomographic vp gradient (Wu & Alkhalifah 2015) ----

__global__ void compute_v2_lambda_bg_born3d(
    const float* __restrict__ vp, const float* __restrict__ mp,
    const float* __restrict__ lambda_bg, const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda, int nx, int ny, int nz, int B);

__global__ void calculate_grad_born3d_vp_utt(
    const float* __restrict__ bg_utt, const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int B, int nx, int ny, int nz, float dt);

__global__ void add_born3d_scattered_coupling(
    float* __restrict__ sc_next, const float* __restrict__ bg_next,
    const float* __restrict__ bg_now, const float* __restrict__ bg_prev,
    const float* __restrict__ mp, int B, int nx, int ny, int nz, int halo);

__global__ void calculate_grad_born3d_vp_2diff(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int B, int nx, int ny, int nz);

#if defined(SWEEP_HOST_CORE)
// Host row launchers of the pointwise kernels above (kernels.cu; launched
// through SWEEP_LAUNCH_ROWS in backward.cu).
void calculate_grad_born3d_mp_rows(dim3 grid, dim3 block, const float* u_tt_bg,
                                   const float* u_backward, const float* vp, float* grad_mp, int B,
                                   int nx, int ny, int nz, float dt);
void calculate_grad_born3d_mp_utt_rows(dim3 grid, dim3 block, const float* u_forward_next,
                                       const float* u_forward_now, const float* u_forward_prev,
                                       const float* u_backward, const float* vp, float* grad_mp,
                                       int B, int nx, int ny, int nz, float dt);
void compute_v2_lambda_born3d_rows(dim3 grid, dim3 block, const float* vp, const float* lambda,
                                   float* v2_lambda, int nx, int ny, int nz, int B);
void compute_v2_lambda_bg_born3d_rows(dim3 grid, dim3 block, const float* vp, const float* mp,
                                      const float* lambda_bg, const float* lambda_sc,
                                      float* v2_lambda, int nx, int ny, int nz, int B);
void calculate_grad_born3d_vp_utt_rows(dim3 grid, dim3 block, const float* bg_utt,
                                       const float* sc_utt, const float* lam_bg,
                                       const float* lam_sc, const float* mp, const float* vp,
                                       float* grad_vp, float* grad_iii, int B, int nx, int ny,
                                       int nz, float dt);
void add_born3d_scattered_coupling_rows(dim3 grid, dim3 block, float* sc_next, const float* bg_next,
                                        const float* bg_now, const float* bg_prev, const float* mp,
                                        int B, int nx, int ny, int nz, int halo);
void calculate_grad_born3d_vp_2diff_rows(dim3 grid, dim3 block, const float* bg_prev,
                                         const float* bg_now, const float* bg_next,
                                         const float* sc_prev, const float* sc_now,
                                         const float* sc_next, const float* lam_bg,
                                         const float* lam_sc, const float* mp, const float* vp,
                                         float* grad_vp, float* grad_iii, int B, int nx, int ny,
                                         int nz);
#endif

// Defined in acoustic3d/kernels.cu and shared across translation units -- LSRTM
// re-declares it here rather than pulling in the whole acoustic3d header.
__global__ void accumulate_illumination_3d(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B,
    int nx,
    int ny,
    int nz,
    float dt,
    int x0, int x1, int y0, int y1, int z0, int z1
);

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
);

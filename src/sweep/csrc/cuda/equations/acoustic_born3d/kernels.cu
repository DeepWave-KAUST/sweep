#include "kernels.cuh"

#include "../../common/host_rows.h"

namespace {
__device__ __forceinline__ void calculate_grad_born3d_mp_cell(
    int ix, int iy, int iz, int b,
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
) {
    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = stride_z * nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_tt_b = u_tt_bg + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // grad[mp] = sum_t adjoint_sc * d(sc_next)/d(mp) = sum_t u_back * dt^2 * bg_utt,
    // where saved u_tt_bg == bg_utt (= vp^2 * bg_w_sum).  (Was u_tt_bg/vp^2, dropping
    // dt^2*vp^2 -> grad ~1/(dt^2 vp^2) too small vs eager autograd.)
    grad_b[idx] += dt * dt * u_tt_b[idx] * u_backward_b[idx];
}
}  // namespace

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
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz) {
        return;
    }
    calculate_grad_born3d_mp_cell(ix, iy, iz, b, u_tt_bg, u_backward, vp, grad_mp, B, nx, ny, nz, dt);
}

namespace {
__device__ __forceinline__ void calculate_grad_born3d_mp_utt_cell(
    int ix, int iy, int iz, int b,
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
) {
    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = stride_z * nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_next_b = u_forward_next + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // u_tt = d^2(bg)/dt^2 = bg_utt (= vp^2 * bg_w_sum); grad[mp] = sum_t u_back * dt^2 * bg_utt.
    float u_tt = (u_now_b[idx] - 2.0f * u_prev_b[idx] + u_next_b[idx]) / (dt * dt);
    grad_b[idx] += dt * dt * u_tt * u_backward_b[idx];
}
}  // namespace

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
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz) {
        return;
    }
    calculate_grad_born3d_mp_utt_cell(ix, iy, iz, b, u_forward_next, u_forward_now, u_forward_prev, u_backward, vp, grad_mp, B, nx, ny, nz, dt);
}

#if defined(SWEEP_HOST_CORE)
// Host launchers of the four stencil kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box as Wave3D row spans
// (common/host_rows.h wave3d_rows: gx = ix, gy = iy, gz = b * nz + iz; these
// kernels add no x_base and test no topography) and the cell body to run
// there.  They hold no arithmetic of their own; the cell lambdas capture BY
// VALUE (see acoustic2d).  Order -1 (runtime M) takes the same boxes with the
// kernels' run-time halo (born3d_halo); its row loops do not vectorise.
//
// Each box is built from the kernel's own tests, for every geometry its
// SolverContext can carry: phys_*(halo) and the cut_* faces resolve the
// per-face pads (per-edge PML, a free-surface face's 0 pad) and the DD cut
// faces exactly as the kernels' in_cpml / taps_in_cpml tests do; the kernels
// decode ix without x_base, so a ranged context changes nothing; wave3d_rows
// decodes (b, iz) per row, so a z block straddling two shots is spanned per
// shot, and the rows past the last shot are empty and run through the kernel.
namespace {

// The kernels' halo: the stencil half order, solver.M for Order -1.
template <int Order>
int born3d_halo(const SolverContext& s)
{
    return Order == -1 ? s.M : Order / 2;
}

// The halo guard of acoustic3d_single, acoustic_born3d_coupled and
// acoustic_born3d_adjoint_exact: [halo, n - halo) on every axis.
template <int Order>
CellBox3 born3d_halo_box(const SolverContext& s)
{
    const int h = born3d_halo<Order>(s);
    return {h, s.nx - h, h, s.ny - h, h, s.nz - h};
}

// Where acoustic_cpml_update_3d takes its laplacian-only branch (single,
// coupled): !in_cpml_3d(ix, iy, iz, halo), i.e. per axis
// [phys_0(halo) + 1, phys_1(halo) - 1) -- the band and the db leak cell past
// it are CPML cells -- and [phys_0(halo), phys_1(halo)) on a DD cut face,
// which has neither; inside the halo guard.
template <int Order>
CellBox3 born3d_forward_box(const SolverContext& s)
{
    const int h = born3d_halo<Order>(s);
    return CellBox3{s.phys_x0(h) + (s.cut_x_lo() ? 0 : 1), s.phys_x1(h) - (s.cut_x_hi() ? 0 : 1),
                    s.phys_y0(h) + (s.cut_y_lo() ? 0 : 1), s.phys_y1(h) - (s.cut_y_hi() ? 0 : 1),
                    s.phys_z0(h) + (s.cut_z_lo() ? 0 : 1), s.phys_z1(h) - (s.cut_z_hi() ? 0 : 1)}
        .intersect(born3d_halo_box<Order>(s));
}

// Where acoustic_born3d_adjoint_exact takes its pure L* = lap(v^2 . lambda)
// branch: no taps_in_cpml_*(i, halo, halo), i.e. per axis the taps
// [i - halo, i + halo] stay clear of the in_cpml cells,
// [phys_0(halo) + 1 + halo, phys_1(halo) - 1 - halo); a DD cut face is not
// probed (unbounded on that side); inside the halo guard.
template <int Order>
CellBox3 born3d_adjoint_box(const SolverContext& s)
{
    const int h = born3d_halo<Order>(s);
    const CellBox3 g = born3d_halo_box<Order>(s);
    return CellBox3{s.cut_x_lo() ? g.x0 : s.phys_x0(h) + 1 + h, s.cut_x_hi() ? g.x1 : s.phys_x1(h) - 1 - h,
                    s.cut_y_lo() ? g.y0 : s.phys_y0(h) + 1 + h, s.cut_y_hi() ? g.y1 : s.phys_y1(h) - 1 - h,
                    s.cut_z_lo() ? g.z0 : s.phys_z0(h) + 1 + h, s.cut_z_hi() ? g.z1 : s.phys_z1(h) - 1 - h}
        .intersect(g);
}

// acoustic3d_single_nopml's halo (abcn + 2m + 1, or 2m without a pad).
template <int Order>
int born3d_nopml_halo(const SolverContext& s)
{
    int m = (Order == -1) ? s.M : (Order / 2);
    return s.abcn > 0 ? s.abcn + 2 * m + 1 : 2 * m;
}

// acoustic3d_single_nopml's guard box, the same expressions as the kernel's
// (every cell it lets through takes the one branch).
template <int Order>
CellBox3 born3d_nopml_box(const SolverContext& s)
{
    int m = (Order == -1) ? s.M : (Order / 2);
    int halo = born3d_nopml_halo<Order>(s);
    int top_halo = s.free_surface ? 2 * m : halo;
    int x_lo = s.cut_x_lo() ? m : halo, x_hi = s.cut_x_hi() ? m : halo;
    int y_lo = s.cut_y_lo() ? m : halo, y_hi = s.cut_y_hi() ? m : halo;
    int z_lo = s.cut_z_lo() ? m : top_halo, z_hi = s.cut_z_hi() ? m : halo;
    return {x_lo, s.nx - x_hi, y_lo, s.ny - y_hi, z_lo, s.nz - z_hi};
}

// The kernels' rows over ``box``.  None of them has a topography return, so
// the box is not lowered under a surface (wave3d_rows would, given has_topo):
// an air cell takes the same branch as any other.
inline auto born3d_rows(const CellBox3& box, const SolverContext& solver)
{
    SolverContext flat = solver;
    flat.has_topo = false;
    return sweep::host::wave3d_rows(box, flat, 0);
}

template <int Order>
struct Acoustic3dSingleRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, bool save_all_wavefields,
                    float* u_this, const float* vp, LaplaceParam lap_ctx, GradParam grad_ctx,
                    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
                    AcousticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic3d_single<Order>(a...); };
        const int nz = solver.nz;
        const auto span = born3d_rows(born3d_forward_box<Order>(solver), solver);
        // The kernel stores u_this only under save_all_wavefields as well.
        sweep::host::with_optional(save_all_wavefields ? u_this : nullptr, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, span,
                [=](int gx, int gy, int gz) {
                    acoustic3d_single_cell<Order, decltype(path)::value>(
                        gx, gy, gz % nz, gz / nz, wf, save_all_wavefields, u_this, vp, lap_ctx,
                        grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, solver);
                },
                call,
                wf, save_all_wavefields, u_this, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y,
                grad_ctx_z, cpml, solver);
        });
    }
};

template <int Order>
struct Acoustic3dSingleNopmlRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, float* u_this,
                    const float* vp, LaplaceParam lap_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic3d_single_nopml<Order>(a...); };
        const int nz = solver.nz;
        const int halo = born3d_nopml_halo<Order>(solver);
        const auto span = born3d_rows(born3d_nopml_box<Order>(solver), solver);
        sweep::host::with_optional(u_this, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, span,
                [=](int gx, int gy, int gz) {
                    acoustic3d_single_nopml_cell<Order, decltype(path)::value>(
                        gx, gy, gz % nz, gz / nz, wf, u_this, vp, lap_ctx, solver, halo);
                },
                call,
                wf, u_this, vp, lap_ctx, solver);
        });
    }
};

template <int Order>
struct AcousticBorn3dCoupledRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer bg, AcousticWavefieldPointer sc,
                    bool save_all_wavefields, float* bg_utt, float* sc_utt, const float* vp,
                    const float* mp, LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x,
                    GradParam grad_ctx_y, GradParam grad_ctx_z, AcousticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic_born3d_coupled<Order>(a...); };
        const int nz = solver.nz;
        const auto span = born3d_rows(born3d_forward_box<Order>(solver), solver);
        // The kernel stores bg_utt (output 0) and sc_utt (output 1) only
        // under save_all_wavefields as well.
        sweep::host::with_optionals(
            [&](auto path) {
                sweep::host::launch_rows(
                    grid, block, span,
                    [=](int gx, int gy, int gz) {
                        acoustic_born3d_coupled_cell<Order, decltype(path)::value>(
                            gx, gy, gz % nz, gz / nz, bg, sc, save_all_wavefields, bg_utt, sc_utt,
                            vp, mp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml,
                            solver);
                    },
                    call,
                    bg, sc, save_all_wavefields, bg_utt, sc_utt, vp, mp, lap_ctx, grad_ctx,
                    grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, solver);
            },
            save_all_wavefields ? bg_utt : nullptr, save_all_wavefields ? sc_utt : nullptr);
    }
};

template <int Order>
struct AcousticBorn3dAdjointRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, const float* v2l,
                    LaplaceParam lap_ctx, GradParam grad_ctx_x, GradParam grad_ctx_y,
                    GradParam grad_ctx_z, AcousticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic_born3d_adjoint_exact<Order>(a...); };
        const int nz = solver.nz;
        sweep::host::launch_rows(
            grid, block, born3d_rows(born3d_adjoint_box<Order>(solver), solver),
            [=](int gx, int gy, int gz) {
                acoustic_born3d_adjoint_exact_cell<Order, CellPath::Interior>(
                    gx, gy, gz % nz, gz / nz, wf, v2l, lap_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
                    cpml, solver);
            },
            call,
            wf, v2l, lap_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, solver);
    }
};

}  // namespace
#endif

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).

SWEEP_BY_ORDER_TABLE_ROWS(acoustic3d_single_fn, acoustic3d_single_by_order, acoustic3d_single,
                          Acoustic3dSingleRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic3d_single_nopml_fn, acoustic3d_single_nopml_by_order,
                          acoustic3d_single_nopml, Acoustic3dSingleNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic_born3d_coupled_fn, acoustic_born3d_coupled_by_order,
                          acoustic_born3d_coupled, AcousticBorn3dCoupledRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic_born3d_adjoint_exact_fn, acoustic_born3d_adjoint_exact_by_order,
                          acoustic_born3d_adjoint_exact, AcousticBorn3dAdjointRows);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
namespace {
__device__ __forceinline__ void compute_v2_lambda_born3d_cell(
    int ix, int iy, int iz, int b,
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx,
    int ny,
    int nz,
    int B
) {
    int sp = nx * ny * nz; int idx = b * sp + iz * (nx * ny) + iy * nx + ix;
    float v = vp[idx];
    v2_lambda[idx] = v * v * lambda[idx];
}
}  // namespace

__global__ void compute_v2_lambda_born3d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int ny, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    compute_v2_lambda_born3d_cell(ix, iy, iz, b, vp, lambda, v2_lambda, nx, ny, nz, B);
}

// ---- RWI tomographic vp gradient (Wu & Alkhalifah 2015); 3-D twins of the
// acoustic_born2d kernels -- see there for the derivation of each term. ----

// Background adjoint lambda_bg = mu: its own propagation plus the transpose of the
// scattered-source coupling, v2 = vp^2 * (lambda_bg + mp * lambda_sc).
namespace {
__device__ __forceinline__ void compute_v2_lambda_bg_born3d_cell(
    int ix, int iy, int iz, int b,
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    int nx,
    int ny,
    int nz,
    int B
) {
    int sp = nx * ny * nz; int idx = b * sp + iz * (nx * ny) + iy * nx + ix;
    float v2 = vp[idx] * vp[idx];
    float gi = lambda_bg[idx] + mp[idx] * lambda_sc[idx];
    v2_lambda[idx] = v2 * gi;
}
}  // namespace

__global__ void compute_v2_lambda_bg_born3d(
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    int nx, int ny, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    compute_v2_lambda_bg_born3d_cell(ix, iy, iz, b, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx, ny, nz, B);
}

// Full mode, from the stored second time derivatives:
//   grad_v += (2 dt^2 / v) [ sc_utt*lam_sc (II) + bg_utt*lam_bg (IV) + mp*bg_utt*lam_sc (III) ]
// grad_iii != nullptr keeps III apart for the paper's beta weight.
namespace {
template <CellPath Path = CellPath::Any>
__device__ __forceinline__ void calculate_grad_born3d_vp_utt_cell(
    int ix, int iy, int iz, int b,
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
) {
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    float c   = 2.0f * dt * dt / vp[o];
    float bgu = bg_utt[o];
    float ls  = lam_sc[o];
    float ii_iv = c * (sc_utt[o] * ls + bgu * lam_bg[o]);
    float iii   = c * (mp[o] * bgu * ls);
    if (optional_output<Path>(grad_iii)) {
        grad_vp[o]  += ii_iv;
        grad_iii[o] += iii;
    } else {
        grad_vp[o]  += ii_iv + iii;
    }
}
}  // namespace

__global__ void calculate_grad_born3d_vp_utt(
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int B, int nx, int ny, int nz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    calculate_grad_born3d_vp_utt_cell(ix, iy, iz, b, bg_utt, sc_utt, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, B, nx, ny, nz, dt);
}

// Boundary-saving: the scattered reverse recursion's coupling source
// mp * vp^2 * Lap(bg) = mp * (source-free bg second difference).  Interior only.
namespace {
__device__ __forceinline__ void add_born3d_scattered_coupling_cell(
    int ix, int iy, int iz, int b,
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int B,
    int nx,
    int ny,
    int nz,
    int halo
) {
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    sc_next[o] += mp[o] * (bg_next[o] - 2.0f * bg_now[o] + bg_prev[o]);
}
}  // namespace

__global__ void add_born3d_scattered_coupling(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int B, int nx, int ny, int nz, int halo
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B) return;
    if (ix < halo || ix >= nx - halo || iy < halo || iy >= ny - halo ||
        iz < halo || iz >= nz - halo) return;
    add_born3d_scattered_coupling_cell(ix, iy, iz, b, sc_next, bg_next, bg_now, bg_prev, mp, B, nx, ny, nz, halo);
}

// Boundary-saving vp gradient from the reconstructed second differences:
// bg 2nd diff -> IV, sc 2nd diff -> II + III; III recovered as (2/v)*mp*bg2*lam_sc.
namespace {
template <CellPath Path = CellPath::Any>
__device__ __forceinline__ void calculate_grad_born3d_vp_2diff_cell(
    int ix, int iy, int iz, int b,
    const float* __restrict__ bg_prev,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev,
    const float* __restrict__ sc_now,
    const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int B,
    int nx,
    int ny,
    int nz
) {
    int sp = nx * ny * nz; int o = b * sp + iz * (nx * ny) + iy * nx + ix;
    float inv = 2.0f / vp[o];
    float bg2 = bg_prev[o] - 2.0f * bg_now[o] + bg_next[o];
    float sc2 = sc_prev[o] - 2.0f * sc_now[o] + sc_next[o];
    float ls  = lam_sc[o];
    float total = inv * (bg2 * lam_bg[o] + sc2 * ls);
    if (optional_output<Path>(grad_iii)) {
        float iii = inv * (mp[o] * bg2 * ls);
        grad_vp[o]  += total - iii;
        grad_iii[o] += iii;
    } else {
        grad_vp[o]  += total;
    }
}
}  // namespace

__global__ void calculate_grad_born3d_vp_2diff(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int B, int nx, int ny, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / nz; int iz = iz_global % nz;
    if (b >= B || ix >= nx || iy >= ny || iz >= nz) return;
    calculate_grad_born3d_vp_2diff_cell(ix, iy, iz, b, bg_prev, bg_now, bg_next, sc_prev, sc_now, sc_next, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, B, nx, ny, nz);
}

#if defined(SWEEP_HOST_CORE)
#include "../../common/host_rows.h"
// Host launchers of the pointwise kernels above (launch/cell_path.h; launched
// by name through SWEEP_LAUNCH_ROWS in backward.cu).  Each reads and writes its
// own cell and has no branch an interior path could answer, so the span is
// exactly the kernel's bounds guard, in the launch's virtual-thread
// coordinates (gx, gy, gz) = (ix, iy, b * nz + iz), decoded per row as the kernels do; the
// optional III output (grad_iii) is the one compile-time path.  No arithmetic
// here; the cell lambdas capture by value.

void calculate_grad_born3d_mp_rows(dim3 grid, dim3 block, const float* u_tt_bg,
                                   const float* u_backward, const float* vp, float* grad_mp, int B,
                                   int nx, int ny, int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born3d_mp(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iy, int gz) {
            const int b = gz / nz, iz = gz % nz;
            calculate_grad_born3d_mp_cell(ix, iy, iz, b, u_tt_bg, u_backward, vp, grad_mp, B, nx, ny,
                                          nz, dt);
        },
        call, u_tt_bg, u_backward, vp, grad_mp, B, nx, ny, nz, dt);
}

void calculate_grad_born3d_mp_utt_rows(dim3 grid, dim3 block, const float* u_forward_next,
                                       const float* u_forward_now, const float* u_forward_prev,
                                       const float* u_backward, const float* vp, float* grad_mp,
                                       int B, int nx, int ny, int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born3d_mp_utt(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iy, int gz) {
            const int b = gz / nz, iz = gz % nz;
            calculate_grad_born3d_mp_utt_cell(ix, iy, iz, b, u_forward_next, u_forward_now,
                                              u_forward_prev, u_backward, vp, grad_mp, B, nx, ny, nz,
                                              dt);
        },
        call, u_forward_next, u_forward_now, u_forward_prev, u_backward, vp, grad_mp, B, nx, ny, nz,
    dt);
}

void compute_v2_lambda_born3d_rows(dim3 grid, dim3 block, const float* vp, const float* lambda,
                                   float* v2_lambda, int nx, int ny, int nz, int B)
{
    auto call = [&](auto&&... a) { compute_v2_lambda_born3d(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iy, int gz) {
            const int b = gz / nz, iz = gz % nz;
            compute_v2_lambda_born3d_cell(ix, iy, iz, b, vp, lambda, v2_lambda, nx, ny, nz, B);
        },
        call, vp, lambda, v2_lambda, nx, ny, nz, B);
}

void compute_v2_lambda_bg_born3d_rows(dim3 grid, dim3 block, const float* vp, const float* mp,
                                      const float* lambda_bg, const float* lambda_sc,
                                      float* v2_lambda, int nx, int ny, int nz, int B)
{
    auto call = [&](auto&&... a) { compute_v2_lambda_bg_born3d(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iy, int gz) {
            const int b = gz / nz, iz = gz % nz;
            compute_v2_lambda_bg_born3d_cell(ix, iy, iz, b, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx,
                                             ny, nz, B);
        },
        call, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx, ny, nz, B);
}

void calculate_grad_born3d_vp_utt_rows(dim3 grid, dim3 block, const float* bg_utt,
                                       const float* sc_utt, const float* lam_bg,
                                       const float* lam_sc, const float* mp, const float* vp,
                                       float* grad_vp, float* grad_iii, int B, int nx, int ny,
                                       int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born3d_vp_utt(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::with_optional(grad_iii, [&](auto path) {
        sweep::host::launch_rows(
            grid, block, span,
            [=](int ix, int iy, int gz) {
                const int b = gz / nz, iz = gz % nz;
                calculate_grad_born3d_vp_utt_cell<decltype(path)::value>(ix, iy, iz, b, bg_utt, sc_utt,
                                                                         lam_bg, lam_sc, mp, vp, grad_vp,
                                                                         grad_iii, B, nx, ny, nz, dt);
            },
            call, bg_utt, sc_utt, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, B, nx, ny, nz, dt);
    });
}

void add_born3d_scattered_coupling_rows(dim3 grid, dim3 block, float* sc_next, const float* bg_next,
                                        const float* bg_now, const float* bg_prev, const float* mp,
                                        int B, int nx, int ny, int nz, int halo)
{
    auto call = [&](auto&&... a) { add_born3d_scattered_coupling(a...); };
    const auto span = [B, nx, ny, nz, halo](int iy, int gz) {   // the guards: b < B, the halo box
        const int b = gz / nz, iz = gz % nz;
        return (b < B && iy >= halo && iy < ny - halo && iz >= halo && iz < nz - halo)
                   ? sweep::host::RowSpan{halo, nx - halo} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iy, int gz) {
            const int b = gz / nz, iz = gz % nz;
            add_born3d_scattered_coupling_cell(ix, iy, iz, b, sc_next, bg_next, bg_now, bg_prev, mp, B,
                                               nx, ny, nz, halo);
        },
        call, sc_next, bg_next, bg_now, bg_prev, mp, B, nx, ny, nz, halo);
}

void calculate_grad_born3d_vp_2diff_rows(dim3 grid, dim3 block, const float* bg_prev,
                                         const float* bg_now, const float* bg_next,
                                         const float* sc_prev, const float* sc_now,
                                         const float* sc_next, const float* lam_bg,
                                         const float* lam_sc, const float* mp, const float* vp,
                                         float* grad_vp, float* grad_iii, int B, int nx, int ny,
                                         int nz)
{
    auto call = [&](auto&&... a) { calculate_grad_born3d_vp_2diff(a...); };
    const auto span = [B, nx, ny, nz](int iy, int gz) {   // the guard: b < B, ix < nx, iy < ny
        return (gz / nz < B && iy < ny) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::with_optional(grad_iii, [&](auto path) {
        sweep::host::launch_rows(
            grid, block, span,
            [=](int ix, int iy, int gz) {
                const int b = gz / nz, iz = gz % nz;
                calculate_grad_born3d_vp_2diff_cell<decltype(path)::value>(ix, iy, iz, b, bg_prev, bg_now,
                                                                           bg_next, sc_prev, sc_now,
                                                                           sc_next, lam_bg, lam_sc, mp, vp,
                                                                           grad_vp, grad_iii, B, nx, ny,
                                                                           nz);
            },
            call, bg_prev, bg_now, bg_next, sc_prev, sc_now, sc_next, lam_bg, lam_sc, mp, vp, grad_vp,
    grad_iii, B, nx, ny, nz);
    });
}
#endif

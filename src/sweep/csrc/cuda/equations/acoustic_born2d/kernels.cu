#include "kernels.cuh"

namespace {
__device__ __forceinline__ void calculate_grad_born_mp_cell(
    int ix, int iz, int b,
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_tt_b = u_tt_bg + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    float* grad_b = grad_mp + b * spatial_size;

    // grad[mp] = sum_t adjoint_sc * d(sc_next)/d(mp).  Forward couples
    //   sc_next += dt^2 * mp * bg_utt   (bg_utt = vp^2 * bg_w_sum, == saved u_tt_bg),
    // so d(sc_next)/d(mp) = dt^2 * bg_utt = dt^2 * u_tt_bg.  (Was u_tt_bg/vp^2, which
    // dropped dt^2*vp^2 -> grad ~1/(dt^2 vp^2) too small vs eager autograd.)
    grad_b[idx] += dt * dt * u_tt_b[idx] * u_backward_b[idx];
}
}  // namespace

__global__ void calculate_grad_born_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;
    calculate_grad_born_mp_cell(ix, iz, b, u_tt_bg, u_backward, vp, grad_mp, nx, nz, dt);
}

namespace {
__device__ __forceinline__ void calculate_grad_born_mp_utt_cell(
    int ix, int iz, int b,
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

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

__global__ void calculate_grad_born_mp_utt(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;
    calculate_grad_born_mp_utt_cell(ix, iz, b, u_forward_next, u_forward_now, u_forward_prev, u_backward, vp, grad_mp, nx, nz, dt);
}

#if defined(SWEEP_HOST_CORE)
// Host launchers of the four stencil kernels (launch/cell_path.h): each hands
// host/launch.h launch_rows the kernel's interior box in virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b) -- box_rows: these kernels add no
// x_base and test no topography -- and the cell body to run there.  They hold
// no arithmetic of their own; the cell lambdas capture BY VALUE (see
// acoustic2d).  Every box below is built from the kernel's own tests (the
// SolverContext helpers they call, the expressions they spell), so it follows
// the DD cut faces, per-edge pads and the free surface the context carries.
//
// The pointwise kernels of this file (compute_v2_lambda_*, calculate_grad_born_*,
// add_born_scattered_coupling_2d) have their own row launchers at the end of
// the file.
namespace {

// The halo of the four stencil kernels, as they compute it.
template <int Order>
int born2d_halo(const SolverContext& s)
{
    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    return is_runtime ? s.M : M_static;
}

// The halo guard of acoustic2d_single, acoustic_born2nd and
// acoustic_born2nd_adjoint_exact: [halo, n - halo) on both axes.
template <int Order>
CellBox born2d_halo_box(const SolverContext& s)
{
    const int halo = born2d_halo<Order>(s);
    return {halo, s.nx - halo, halo, s.nz - halo};
}

// acoustic2d_single and acoustic_born2nd: where acoustic_cpml_update_2d's
// solver.in_cpml_2d(ix, iz, halo) is false -- in_cpml_x / in_cpml_z: the band
// plus the db leak cell past it on a band face, nothing past the halo on a DD
// cut face -- inside the halo guard.
template <int Order>
CellBox born2d_cpml_interior_box(const SolverContext& s)
{
    const int halo = born2d_halo<Order>(s);
    return CellBox{s.phys_x0(halo) + (s.cut_x_lo() ? 0 : 1), s.phys_x1(halo) - (s.cut_x_hi() ? 0 : 1),
                   s.phys_z0(halo) + (s.cut_z_lo() ? 0 : 1), s.phys_z1(halo) - (s.cut_z_hi() ? 0 : 1)}
        .intersect(born2d_halo_box<Order>(s));
}

// acoustic_born2nd_adjoint_exact: where neither solver.taps_in_cpml_x(ix,
// halo, halo) nor taps_in_cpml_z(iz, halo, halo) holds -- on a band face the
// in_cpml threshold (phys(halo) + 1 from the low face, phys(halo) - 1 from the
// high one) moved in by the tap reach halo; a DD cut face is not tested --
// inside the halo guard.
template <int Order>
CellBox born2d_adjoint_interior_box(const SolverContext& s)
{
    const int halo = born2d_halo<Order>(s);
    const CellBox guard = born2d_halo_box<Order>(s);
    return CellBox{s.cut_x_lo() ? guard.x0 : s.phys_x0(halo) + 1 + halo,
                   s.cut_x_hi() ? guard.x1 : s.phys_x1(halo) - 1 - halo,
                   s.cut_z_lo() ? guard.z0 : s.phys_z0(halo) + 1 + halo,
                   s.cut_z_hi() ? guard.z1 : s.phys_z1(halo) - 1 - halo}
        .intersect(guard);
}

// acoustic2d_single_nopml: its guard box, the same expressions as the kernel's
// (every cell the guard lets through takes the one straight-line path).
template <int Order>
CellBox born2d_nopml_box(const SolverContext& solver)
{
    int M = (Order == -1) ? solver.M : (Order / 2);
    int halo = solver.abcn > 0 ? solver.abcn + 2 * M + 1 : 2 * M;
    int top_halo = solver.free_surface ? 2 * M : halo;
    int x_lo = solver.cut_x_lo() ? M : halo, x_hi = solver.cut_x_hi() ? M : halo;
    int z_lo = solver.cut_z_lo() ? M : top_halo, z_hi = solver.cut_z_hi() ? M : halo;
    return {x_lo, solver.nx - x_hi, z_lo, solver.nz - z_hi};
}

template <int Order>
struct Acoustic2dSingleRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, bool save_all_wavefields,
                    float* u_this, const float* vp, LaplaceParam lap_ctx, GradParam grad_ctx,
                    GradParam grad_ctx_x, GradParam grad_ctx_z, AcousticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic2d_single<Order>(a...); };
        const CellBox box = born2d_cpml_interior_box<Order>(solver);
        // The kernel stores u_this only when save_all_wavefields is set too.
        sweep::host::with_optional(save_all_wavefields ? u_this : nullptr, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, sweep::host::box_rows(box),
                [=](int ix, int iz, int b) {
                    acoustic_born2d_single_cell<Order, decltype(path)::value>(
                        AcousticBorn2dCellIndex{ix, iz, b}, wf, save_all_wavefields, u_this, vp,
                        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, solver);
                },
                call,
                wf, save_all_wavefields, u_this, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
                cpml, solver);
        });
    }
};

template <int Order>
struct Acoustic2dSingleNopmlRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, const float* vp,
                    LaplaceParam lap_ctx, SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic2d_single_nopml<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(born2d_nopml_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                acoustic_born2d_single_nopml_cell<Order, CellPath::Interior>(
                    AcousticBorn2dCellIndex{ix, iz, b}, wf, vp, lap_ctx, solver);
            },
            call,
            wf, vp, lap_ctx, solver);
    }
};

template <int Order>
struct AcousticBorn2ndRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer bg, AcousticWavefieldPointer sc,
                    bool save_all_wavefields, float* bg_utt, float* sc_utt, const float* vp,
                    const float* mp, LaplaceParam lap_ctx, GradParam grad_ctx,
                    GradParam grad_ctx_x, GradParam grad_ctx_z, AcousticCPMLPointer cpml,
                    SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic_born2nd<Order>(a...); };
        const CellBox box = born2d_cpml_interior_box<Order>(solver);
        // The kernel stores bg_utt (output 0) and sc_utt (output 1) only when
        // save_all_wavefields is set too.
        sweep::host::with_optionals(
            [&](auto path) {
                sweep::host::launch_rows(
                    grid, block, sweep::host::box_rows(box),
                    [=](int ix, int iz, int b) {
                        acoustic_born2nd_cell<Order, decltype(path)::value>(
                            AcousticBorn2dCellIndex{ix, iz, b}, bg, sc, save_all_wavefields,
                            bg_utt, sc_utt, vp, mp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
                            cpml, solver);
                    },
                    call,
                    bg, sc, save_all_wavefields, bg_utt, sc_utt, vp, mp, lap_ctx, grad_ctx,
                    grad_ctx_x, grad_ctx_z, cpml, solver);
            },
            save_all_wavefields ? bg_utt : nullptr, save_all_wavefields ? sc_utt : nullptr);
    }
};

template <int Order>
struct AcousticBorn2ndAdjointExactRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, const float* v2l,
                    LaplaceParam lap_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
                    AcousticCPMLPointer cpml, SolverContext solver)
    {
        auto call = [&](auto&&... a) { acoustic_born2nd_adjoint_exact<Order>(a...); };
        sweep::host::launch_rows(
            grid, block, sweep::host::box_rows(born2d_adjoint_interior_box<Order>(solver)),
            [=](int ix, int iz, int b) {
                acoustic_born2nd_adjoint_exact_cell<Order, CellPath::Interior>(
                    AcousticBorn2dCellIndex{ix, iz, b}, wf, v2l, lap_ctx, grad_ctx_x, grad_ctx_z,
                    cpml, solver);
            },
            call,
            wf, v2l, lap_ctx, grad_ctx_x, grad_ctx_z, cpml, solver);
    }
};

}  // namespace
#endif

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).

SWEEP_BY_ORDER_TABLE_ROWS(acoustic2d_single_fn, acoustic2d_single_by_order, acoustic2d_single,
                          Acoustic2dSingleRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic2d_single_nopml_fn, acoustic2d_single_nopml_by_order,
                          acoustic2d_single_nopml, Acoustic2dSingleNopmlRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic_born2nd_fn, acoustic_born2nd_by_order, acoustic_born2nd,
                          AcousticBorn2ndRows);
SWEEP_BY_ORDER_TABLE_ROWS(acoustic_born2nd_adjoint_exact_fn,
                          acoustic_born2nd_adjoint_exact_by_order, acoustic_born2nd_adjoint_exact,
                          AcousticBorn2ndAdjointExactRows);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
namespace {
__device__ __forceinline__ void compute_v2_lambda_born2d_cell(
    int ix, int iz, int b,
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx,
    int nz,
    int B
) {
    int sp = nx * nz; int idx = b * sp + iz * nx + ix;
    float v = vp[idx];
    v2_lambda[idx] = v * v * lambda[idx];
}
}  // namespace

__global__ void compute_v2_lambda_born2d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz || b >= B) return;
    compute_v2_lambda_born2d_cell(ix, iz, b, vp, lambda, v2_lambda, nx, nz, B);
}

// RWI (Wu & Alkhalifah 2015): the background adjoint lambda_bg = mu.  Its own
// propagation plus the transpose of the scattered-source coupling arrive in one
// combined right-hand side, so the shared lsrtm adjoint step can inject both:
//   v2 = vp^2 * (lambda_bg + mp * lambda_sc).
namespace {
__device__ __forceinline__ void compute_v2_lambda_bg_born2d_cell(
    int ix, int iz, int b,
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    int nx,
    int nz,
    int B
) {
    int sp = nx * nz; int idx = b * sp + iz * nx + ix;
    float v2 = vp[idx] * vp[idx];
    float gi = lambda_bg[idx] + mp[idx] * lambda_sc[idx];
    v2_lambda[idx] = v2 * gi;
}
}  // namespace

__global__ void compute_v2_lambda_bg_born2d(
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz || b >= B) return;
    compute_v2_lambda_bg_born2d_cell(ix, iz, b, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx, nz, B);
}

// RWI tomographic vp gradient from the stored second time derivatives
// (the full mode, where bg_utt and sc_utt are stored for every step):
//   grad_v += (2 dt^2 / v) [ sc_utt*lam_sc + bg_utt*lam_bg + mp*bg_utt*lam_sc ]
//               \_____ II _____/  \___ IV ___/  \______ III ______/
// II  = <lam, B'(v) q>        (scattered field self-propagation)
// IV  = <mu,  B'(v) p>        (background field via the second adjoint)
// III = <lam, B'(v) p .* w>   (the singular image-point term)
// grad_iii != nullptr keeps III apart so the caller can apply the paper's beta
// weight (eq. 18-20); otherwise all three are summed into grad_vp.
namespace {
template <CellPath Path = CellPath::Any>
__device__ __forceinline__ void calculate_grad_born_vp_utt_cell(
    int ix, int iz, int b,
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int nx,
    int nz,
    float dt
) {
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
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

__global__ void calculate_grad_born_vp_utt(
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int nx, int nz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz) return;
    calculate_grad_born_vp_utt_cell(ix, iz, b, bg_utt, sc_utt, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, nx, nz, dt);
}

// Boundary-saving mode drives the scattered field's reverse recursion with the
// coupling source mp * vp^2 * Lap(bg), reusing the background reconstruction's
// own second difference (bg_next - 2 bg_now + bg_prev = dt^2 vp^2 Lap(bg)) so no
// Laplacian is re-evaluated.  Interior only; the outer ring is overwritten by
// the scattered boundary restore right afterwards.
namespace {
__device__ __forceinline__ void add_born_scattered_coupling_2d_cell(
    int ix, int iz, int b,
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int nx,
    int nz,
    int halo
) {
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    sc_next[o] += mp[o] * (bg_next[o] - 2.0f * bg_now[o] + bg_prev[o]);
}
}  // namespace

__global__ void add_born_scattered_coupling_2d(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int nx, int nz, int halo
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix < halo || ix >= nx - halo || iz < halo || iz >= nz - halo) return;
    add_born_scattered_coupling_2d_cell(ix, iz, b, sc_next, bg_next, bg_now, bg_prev, mp, nx, nz, halo);
}

// Boundary-saving vp gradient, from the reconstructed second differences:
//   bg 2nd diff = dt^2 vp^2 Lap(bg) = dt^2 * bg_utt                 -> IV
//   sc 2nd diff = dt^2 (sc_utt + mp*bg_utt)                         -> II + III
// so the total matches the full mode, and III is recovered separately as
// (2/v) * mp * (bg 2nd diff) * lam_sc for the same beta split.
namespace {
template <CellPath Path = CellPath::Any>
__device__ __forceinline__ void calculate_grad_born_vp_2diff_2d_cell(
    int ix, int iz, int b,
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
    int nx,
    int nz
) {
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
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

__global__ void calculate_grad_born_vp_2diff_2d(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int nx, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz) return;
    calculate_grad_born_vp_2diff_2d_cell(ix, iz, b, bg_prev, bg_now, bg_next, sc_prev, sc_now, sc_next, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, nx, nz);
}

#if defined(SWEEP_HOST_CORE)
#include "../../common/host_rows.h"
// Host launchers of the pointwise kernels above (launch/cell_path.h; launched
// by name through SWEEP_LAUNCH_ROWS in backward.cu).  Each reads and writes its
// own cell and has no branch an interior path could answer, so the span is
// exactly the kernel's bounds guard, in the launch's virtual-thread
// coordinates (gx, gy, gz) = (ix, iz, b); the
// optional III output (grad_iii) is the one compile-time path.  No arithmetic
// here; the cell lambdas capture by value.

void calculate_grad_born_mp_rows(dim3 grid, dim3 block, const float* u_tt_bg,
                                 const float* u_backward, const float* vp, float* grad_mp, int nx,
                                 int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born_mp(a...); };
    const auto span = sweep::host::box_rows(CellBox{0, nx, 0, nz});   // the guard: ix < nx, iz < nz
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iz, int b) {
            calculate_grad_born_mp_cell(ix, iz, b, u_tt_bg, u_backward, vp, grad_mp, nx, nz, dt);
        },
        call, u_tt_bg, u_backward, vp, grad_mp, nx, nz, dt);
}

void calculate_grad_born_mp_utt_rows(dim3 grid, dim3 block, const float* u_forward_next,
                                     const float* u_forward_now, const float* u_forward_prev,
                                     const float* u_backward, const float* vp, float* grad_mp,
                                     int nx, int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born_mp_utt(a...); };
    const auto span = sweep::host::box_rows(CellBox{0, nx, 0, nz});   // the guard: ix < nx, iz < nz
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iz, int b) {
            calculate_grad_born_mp_utt_cell(ix, iz, b, u_forward_next, u_forward_now, u_forward_prev,
                                            u_backward, vp, grad_mp, nx, nz, dt);
        },
        call, u_forward_next, u_forward_now, u_forward_prev, u_backward, vp, grad_mp, nx, nz, dt);
}

void compute_v2_lambda_born2d_rows(dim3 grid, dim3 block, const float* vp, const float* lambda,
                                   float* v2_lambda, int nx, int nz, int B)
{
    auto call = [&](auto&&... a) { compute_v2_lambda_born2d(a...); };
    const auto span = [nx, nz, B](int iz, int b) {   // the guard: ix < nx, iz < nz, b < B
        return (iz < nz && b < B) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iz, int b) {
            compute_v2_lambda_born2d_cell(ix, iz, b, vp, lambda, v2_lambda, nx, nz, B);
        },
        call, vp, lambda, v2_lambda, nx, nz, B);
}

void compute_v2_lambda_bg_born2d_rows(dim3 grid, dim3 block, const float* vp, const float* mp,
                                      const float* lambda_bg, const float* lambda_sc,
                                      float* v2_lambda, int nx, int nz, int B)
{
    auto call = [&](auto&&... a) { compute_v2_lambda_bg_born2d(a...); };
    const auto span = [nx, nz, B](int iz, int b) {   // the guard: ix < nx, iz < nz, b < B
        return (iz < nz && b < B) ? sweep::host::RowSpan{0, nx} : sweep::host::RowSpan{};
    };
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iz, int b) {
            compute_v2_lambda_bg_born2d_cell(ix, iz, b, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx, nz,
                                             B);
        },
        call, vp, mp, lambda_bg, lambda_sc, v2_lambda, nx, nz, B);
}

void calculate_grad_born_vp_utt_rows(dim3 grid, dim3 block, const float* bg_utt,
                                     const float* sc_utt, const float* lam_bg, const float* lam_sc,
                                     const float* mp, const float* vp, float* grad_vp,
                                     float* grad_iii, int nx, int nz, float dt)
{
    auto call = [&](auto&&... a) { calculate_grad_born_vp_utt(a...); };
    const auto span = sweep::host::box_rows(CellBox{0, nx, 0, nz});   // the guard: ix < nx, iz < nz
    sweep::host::with_optional(grad_iii, [&](auto path) {
        sweep::host::launch_rows(
            grid, block, span,
            [=](int ix, int iz, int b) {
                calculate_grad_born_vp_utt_cell<decltype(path)::value>(ix, iz, b, bg_utt, sc_utt, lam_bg,
                                                                       lam_sc, mp, vp, grad_vp, grad_iii,
                                                                       nx, nz, dt);
            },
            call, bg_utt, sc_utt, lam_bg, lam_sc, mp, vp, grad_vp, grad_iii, nx, nz, dt);
    });
}

void add_born_scattered_coupling_2d_rows(dim3 grid, dim3 block, float* sc_next,
                                         const float* bg_next, const float* bg_now,
                                         const float* bg_prev, const float* mp, int nx, int nz,
                                         int halo)
{
    auto call = [&](auto&&... a) { add_born_scattered_coupling_2d(a...); };
    const auto span = sweep::host::box_rows(CellBox{halo, nx - halo, halo, nz - halo});   // the guard
    sweep::host::launch_rows(
        grid, block, span,
        [=](int ix, int iz, int b) {
            add_born_scattered_coupling_2d_cell(ix, iz, b, sc_next, bg_next, bg_now, bg_prev, mp, nx,
                                                nz, halo);
        },
        call, sc_next, bg_next, bg_now, bg_prev, mp, nx, nz, halo);
}

void calculate_grad_born_vp_2diff_2d_rows(dim3 grid, dim3 block, const float* bg_prev,
                                          const float* bg_now, const float* bg_next,
                                          const float* sc_prev, const float* sc_now,
                                          const float* sc_next, const float* lam_bg,
                                          const float* lam_sc, const float* mp, const float* vp,
                                          float* grad_vp, float* grad_iii, int nx, int nz)
{
    auto call = [&](auto&&... a) { calculate_grad_born_vp_2diff_2d(a...); };
    const auto span = sweep::host::box_rows(CellBox{0, nx, 0, nz});   // the guard: ix < nx, iz < nz
    sweep::host::with_optional(grad_iii, [&](auto path) {
        sweep::host::launch_rows(
            grid, block, span,
            [=](int ix, int iz, int b) {
                calculate_grad_born_vp_2diff_2d_cell<decltype(path)::value>(ix, iz, b, bg_prev, bg_now,
                                                                            bg_next, sc_prev, sc_now,
                                                                            sc_next, lam_bg, lam_sc, mp, vp,
                                                                            grad_vp, grad_iii, nx, nz);
            },
            call, bg_prev, bg_now, bg_next, sc_prev, sc_now, sc_next, lam_bg, lam_sc, mp, vp, grad_vp,
    grad_iii, nx, nz);
    });
}
#endif

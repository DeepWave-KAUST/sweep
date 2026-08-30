#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/laplace.cuh"

// NOTE (ODR): the CPML stencil / fused-adjoint kernels are REUSED from
// ../acoustic2d/kernels.cuh — same header, token-identical weak template
// definitions in another TU, which is ODR-safe.  Everything defined HERE is
// visco-private and carries the ``visco_acoustic2d_`` prefix so no mangled
// name can collide with a different body elsewhere (see the lsrtm2d ODR
// incident note in ../acoustic_lsrtm2d/kernels.cuh).

// vp-gradient carrier, recomputed from the RAW pressure history.
//
// The acoustic2d forward stores ``vp^2 * Lap(u)`` per step because that is
// all its backward needs.  The visco forward must store RAW ``u`` instead
// (the attenuation adjoint needs du/dt and its |k| filter), so the carrier
// the shared ``calculate_grad`` / ``accumulate_rtm_image_2d`` kernels expect
// is recomputed here on the fly: ``carrier = vp^2 * (Lap_x + Lap_z)(u)``.
// Halo cells are never written (the caller zeroes the scratch), matching the
// acoustic store where halo cells stay 0.
template<int Order>
__global__ void visco_acoustic2d_carrier(
    const float* __restrict__ u_raw,   // (B, nz, nx) raw pressure at one step
    const float* __restrict__ vp,      // (B, nz, nx) dispersion-folded vp_step
    float* __restrict__ carrier,       // (B, nz, nx) out, pre-zeroed
    LaplaceParam lap_ctx,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* u_b = u_raw + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(u_b, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(u_b, ix, 0, iz, lap_ctx);
    float v = vp_b[idx];
    carrier[b * spatial_size + idx] = (v * v) * (lap_x + lap_z);
}

#define VISCO_ACOUSTIC2D_CARRIER(order, grid, block, ...)                                    \
    do {                                                                                     \
        if      ((order) == 2) visco_acoustic2d_carrier<2><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 4) visco_acoustic2d_carrier<4><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 6) visco_acoustic2d_carrier<6><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 8) visco_acoustic2d_carrier<8><<<grid, block>>>(__VA_ARGS__);    \
        else                   visco_acoustic2d_carrier<-1><<<grid, block>>>(__VA_ARGS__);   \
    } while (0)

// ---------------------------------------------------------------------------
// Host-side helpers shared by forward.cu / backward.cu (inline, header-only).
// ---------------------------------------------------------------------------

// The nearly-constant-Q amplitude filter  L(x) = Re(IFFT2(|k| * FFT2(x))).
// |k| is real and even under k -> -k, so L is self-adjoint on real fields —
// the adjoint code applies the SAME operator.  Runs through ATen (cuFFT),
// numerically matching the eager reference step_visco_cpml.
inline torch::Tensor visco_acoustic2d_Lop(
    const torch::Tensor& x, const torch::Tensor& kmul)
{
    auto F = at::fft_fft2(x, c10::nullopt, {-2, -1}, c10::nullopt);
    return at::real(at::fft_ifft2(kmul * F, c10::nullopt, {-2, -1}, c10::nullopt));
}

// Zero the M-wide outer halo bands of a runtime field.  The CUDA stencil
// kernels never write halo cells (they stay 0 = the pressure-release image
// condition on free-surface faces); the global FFT damping term DOES write
// them, so it must be followed by this to (a) keep the free-surface BC and
// (b) preserve the c-backend "halo == 0" invariant the stencil taps assume.
inline void visco_acoustic2d_zero_halo(torch::Tensor u, int M)
{
    const long nz = u.size(-2);
    const long nx = u.size(-1);
    u.narrow(-2, 0, M).zero_();
    u.narrow(-2, nz - M, M).zero_();
    u.narrow(-1, 0, M).zero_();
    u.narrow(-1, nx - M, M).zero_();
}

// One amplitude-damping application on the freshly-written u_next:
//   u_next -= dt^2 * A * L((u_now - u_prev) / dt),  A = tt * vp / 2.
// Mirrors the eager step_visco_cpml ordering exactly: stencil first, damping
// second, source injection / recording after.
inline void visco_acoustic2d_apply_damping(
    AcousticWavefieldTensor& wf,
    const torch::Tensor& kmul,   // (nz, nx) D_loss grid
    const torch::Tensor& dt2A,   // (N, C, nz, nx) dt^2 * A
    float dt)
{
    auto dudt = (wf.u_now_t - wf.u_prev_t).div_(dt);
    wf.u_next_t.sub_(dt2A * visco_acoustic2d_Lop(dudt, kmul));
}

// ---------------------------------------------------------------------------
// Spectral-term bundle: the amplitude damping above plus the Zhu & Harris
// (2014, eq. 10) fractional-Laplacian dispersion remainder
//   u_next += dt^2 * (B1 (.) L_{D_k2}(u_now) - B2 (.) L_{D_frac}(u_now)),
// which upgrades the CPML FD Laplacian's -c^2 k^2 to the paper's
// -c^2 eta_hat k^(2*gbar+2).  The eq_aux composition selects the terms:
//   ()                        none
//   (D_loss)                  damping only
//   (D_k2, D_frac)            dispersion only
//   (D_loss, D_k2, D_frac)    both
// Prepared-model layout: (vp_step, B1, B2, A).
struct ViscoSpectral {
    bool active = false;       // damping term present
    bool disp = false;         // dispersion term present
    torch::Tensor kmul;        // D_loss
    torch::Tensor Dk2, Dfrac;  // dispersion grids
    torch::Tensor Gp;          // dt   * A  (adjoint damping)
    torch::Tensor dt2A;        // dt^2 * A  (forward damping)
    torch::Tensor Gd1, Gd2;    // dt^2 * B1, dt^2 * B2
};

inline void visco_acoustic2d_check_grid(
    const torch::Tensor& g, int nz, int nx, const char* name)
{
    TORCH_CHECK(g.dim() == 2 && g.size(0) == nz && g.size(1) == nx,
                "visco eq_aux grid ", name,
                " must be (nz_runtime, nx_runtime) = (", nz, ", ", nx,
                "), got ", g.sizes());
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kFloat32,
                "visco eq_aux grid ", name, " must be a float32 CUDA tensor");
}

inline ViscoSpectral visco_acoustic2d_make_spectral(
    const std::vector<torch::Tensor>& eq_aux,
    const std::vector<torch::Tensor>& models,
    float dt, int nz, int nx)
{
    ViscoSpectral s;
    const size_t n = eq_aux.size();
    TORCH_CHECK(n <= 3, "visco eq_aux takes at most 3 grids, got ", n);
    s.active = (n == 1 || n == 3);
    s.disp = (n >= 2);
    if (s.active) {
        s.kmul = eq_aux[0];
        visco_acoustic2d_check_grid(s.kmul, nz, nx, "D_loss");
        s.Gp = models[3] * dt;
        s.dt2A = models[3] * (dt * dt);
    }
    if (s.disp) {
        s.Dk2 = eq_aux[n - 2];
        s.Dfrac = eq_aux[n - 1];
        visco_acoustic2d_check_grid(s.Dk2, nz, nx, "D_k2");
        visco_acoustic2d_check_grid(s.Dfrac, nz, nx, "D_frac");
        s.Gd1 = models[1] * (dt * dt);
        s.Gd2 = models[2] * (dt * dt);
    }
    return s;
}

// Forward application on the freshly-written u_next: dispersion first,
// damping second (the eager step's ordering), one halo restore after — the
// FFTs write the halo bands, and the stencil kernels rely on halo == 0
// (= the pressure-release image condition on free-surface faces).
inline void visco_acoustic2d_apply_spectral(
    AcousticWavefieldTensor& wf, const ViscoSpectral& s, float dt, int M)
{
    if (!(s.active || s.disp)) return;
    if (s.disp) {
        auto F = at::fft_fft2(wf.u_now_t, c10::nullopt, {-2, -1}, c10::nullopt);
        wf.u_next_t.add_(s.Gd1 * at::real(at::fft_ifft2(s.Dk2 * F, c10::nullopt, {-2, -1}, c10::nullopt)));
        wf.u_next_t.sub_(s.Gd2 * at::real(at::fft_ifft2(s.Dfrac * F, c10::nullopt, {-2, -1}, c10::nullopt)));
    }
    if (s.active)
        visco_acoustic2d_apply_damping(wf, s.kmul, s.dt2A, dt);
    visco_acoustic2d_zero_halo(wf.u_next_t, M);
}

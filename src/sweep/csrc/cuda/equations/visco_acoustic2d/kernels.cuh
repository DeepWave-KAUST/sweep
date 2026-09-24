#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <memory>
#include <mutex>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "spectral_ops.cuh"
#include "../../common/derived_models.h"
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
// the shared ``calculate_grad`` / ``accumulate_illumination_2d`` kernels expect
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

// Zero the M-wide outer halo bands of a runtime field.  The CUDA stencil
// kernels never write halo cells (they stay 0 = the pressure-release image
// condition on free-surface faces); the global FFT damping term DOES write
// them, so it must be followed by this to (a) keep the free-surface BC and
// (b) preserve the c-backend "halo == 0" invariant the stencil taps assume.
inline void visco_acoustic2d_zero_halo(const Buf& u, int M)
{
    SWEEP_CHECK(u.defined() && u.is_contiguous() && u.dim() >= 2,
                "visco_acoustic2d_zero_halo: a contiguous (..., nz, nx) grid");
    const int nz = static_cast<int>(u.size(-2));
    const int nx = static_cast<int>(u.size(-1));
    visco_ops::zero_halo(u.data_ptr<float>(), u.numel() / (static_cast<int64_t>(nz) * nx), nz, nx, M);
}
inline void visco_acoustic2d_zero_halo(const torch::Tensor& u, int M) { visco_acoustic2d_zero_halo(buf_of(u), M); }

// ---------------------------------------------------------------------------
// Spectral-term bundle: the amplitude damping (the D_loss filter on du/dt,
// visco_acoustic2d_apply_damping_into below) plus the Zhu & Harris (2014,
// eq. 10) fractional-Laplacian dispersion remainder
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
    SWEEP_CHECK(g.dim() == 2 && g.size(0) == nz && g.size(1) == nx,
                "visco eq_aux grid ", name,
                " must be (nz_runtime, nx_runtime) = (", nz, ", ", nx,
                "), got ", g.sizes());
    SWEEP_CHECK(g.is_cuda() && g.scalar_type() == torch::kFloat32,
                "visco eq_aux grid ", name, " must be a float32 CUDA tensor");
}

// The term switches and filter grids from the eq_aux composition (no tables).
inline ViscoSpectral visco_acoustic2d_spectral_grids(
    const std::vector<torch::Tensor>& eq_aux, int nz, int nx)
{
    ViscoSpectral s;
    const size_t n = eq_aux.size();
    SWEEP_CHECK(n <= 3, "visco eq_aux takes at most 3 grids, got ", n);
    s.active = (n == 1 || n == 3);
    s.disp = (n >= 2);
    if (s.active) {
        s.kmul = eq_aux[0];
        visco_acoustic2d_check_grid(s.kmul, nz, nx, "D_loss");
    }
    if (s.disp) {
        s.Dk2 = eq_aux[n - 2];
        s.Dfrac = eq_aux[n - 1];
        visco_acoustic2d_check_grid(s.Dk2, nz, nx, "D_k2");
        visco_acoustic2d_check_grid(s.Dfrac, nz, nx, "D_frac");
    }
    return s;
}

// The tables from the propagator's ``derived_models`` slots (or, for a caller
// that binds nothing, from slots allocated here once per call), filled by
// derived::derive_scale: out = __fmul_rn(model, s) with s the same float
// scalar the torch products they replaced handed ATen -- Gp = A * dt,
// dt2A = A * (dt * dt), Gd1 = B1 * (dt * dt), Gd2 = B2 * (dt * dt), dt as
// float and dt*dt the host float product -- one rounding, bit-identical.
// Only the mode's tables are defined (derived::visco_tables); Gp is never
// derived in forward mode, dt2A never in full/bs mode.
inline ViscoSpectral visco_acoustic2d_make_spectral_from(
    const std::vector<torch::Tensor>& eq_aux,
    const std::vector<torch::Tensor>& models,
    const std::vector<torch::Tensor>& derived_models,
    derived::ViscoMode mode, float dt, int nz, int nx, const char* what)
{
    SWEEP_CHECK(models.size() == 4, what, ": visco_acoustic2d expects the prepared models "
                "(vp_step, B1, B2, A); got ", models.size());
    ViscoSpectral s = visco_acoustic2d_spectral_grids(eq_aux, nz, nx);
    const float dt2 = dt * dt;   // float on the host: the scalar of the replaced ``model * (dt * dt)`` products
    const auto c = derived::visco_coefficients(
        derived_models, models[1], models[2], models[3], s.active, s.disp, mode, dt, dt2, what);
    s.Gp = c.gp;
    s.dt2A = c.dt2a;
    s.Gd1 = c.gd1;
    s.Gd2 = c.gd2;
    return s;
}

// ===========================================================================
// Allocation-free spectral machinery: the same transforms and elementwise
// kernels as the ATen expressions they replaced (each helper below records
// its expression), issued on pool slots the propagator
// hands over (ForwardInput.forward_workspace / BackwardInput.adjoint_workspace,
// laid out by ViscoAcoustic.cuda_layout) and on tables it hands over
// (derived_models).  Nothing below allocates per step; a caller that binds
// nothing gets one allocation per CALL for each slot instead.
//
// Bit-exactness, op by op (torch 2.9.1: ATen/native/SpectralOps.cpp,
// ATen/native/cuda/SpectralOps.cpp, native/cuda/Copy.cu, BinaryMulKernel.cu):
//  * at::fft_fft2(x) on a float32 x is promote_tensor_fft -> x.to(kComplexFloat)
//    = _to_copy: at::empty_strided (contiguous for our operands) + copy_,
//    whose float->complex path is direct_copy_kernel_cuda (fetch_and_cast,
//    x -> (x, 0); no arithmetic, so the source's strides do not matter).
//    Here: C0.copy_(x) -- the same copy_ into the same contiguous layout.
//  * then _fft_c2c(forward) -> _fft_c2c_cufft -> _exec_fft: the plan of
//    CuFFTParams(in_strides, out_strides, signal_size, C2C, kFloat), executed
//    as cufftSetStream(current) + cufftSetWorkArea(at::empty(ws)) +
//    cufftXtExec(plan, in, out, CUFFT_FORWARD), out of place.  Here: ViscoFFT
//    builds the same CuFFTParams and issues the same three calls, with the
//    work area taken from the pool.  The same plan on the same data gives the
//    same bits.
//  * kmul * F, a float (nz, nx) grid times the complex (B, 1, nz, nx)
//    spectrum: mul_kernel_cuda on the common dtype complex64, kmul loaded as
//    (k, 0) and c10::complex operator* evaluated as (a*c - b*d, a*d + b*c)
//    with (a, b) = (k, 0).  Here: at::mul_out(C0, kmul, F) -- functional mul
//    IS this structured kernel on a fresh contiguous output -- with the
//    operands in the same order; swapping them would put F in (a, b) and flip
//    the sign of +-0 wherever an imaginary part is exactly 0.
//  * at::fft_ifft2(G) = _fft_c2c_cufft(inverse) followed by
//    _fft_apply_normalization(out, by_n) = out.mul_(1.0 / double(nz*nx)),
//    i.e. Tensor::mul_(Scalar) with a double Scalar.  Here: ViscoFFT::inverse
//    issues the same exec and then the same mul_ with the same double.
//  * at::real(C) = view_as_real(C).select(-1, 0), a strided view, no copy.
//    Here: the same call on the slot.
//  * coef * real(C) -> mul_kernel_cuda on float; u_next.add_/sub_(T).  Here:
//    at::mul_out(R, coef, real(C)) into a float alias of a complex slot that
//    is dead at that point, then the same add_/sub_.  Each cell is one IEEE
//    multiply / add; the output's layout (contiguous either way) never enters
//    the arithmetic.  No addcmul (one fused lambda = FMA contraction).
// ===========================================================================

// The cuFFT plan of the spectral step's 2-D C2C transform: process-wide, one
// per (device, B, nz, nx), built with EXACTLY the parameters ATen's
// _exec_fft derives for a contiguous complex64 (B, 1, nz, nx) tensor over
// dims {-2, -1}: batch dims folded to B, the permuted+reshaped input and the
// resized output both (B, nz, nx) contiguous, hence
//   CuFFTParams(in_strides = {nz*nx, nx, 1}, out_strides = {nz*nx, nx, 1},
//               signal_size = {B, nz, nx}, CuFFTTransformType::C2C, kFloat)
// which CuFFTConfig turns into cufftSetAutoAllocation(0) +
// cufftXtMakePlanMany(plan, 2, {nz, nx}, nullptr,1,1, CUDA_C_32F,
//                     nullptr,1,1, CUDA_C_32F, B, &ws, CUDA_C_32F)
// (the simple-layout branch both contiguous operands take).  We ISSUE THAT
// CALL OURSELVES rather than construct a CuFFTConfig: that type lives in
// ATen/native/cuda/CuFFTPlanCache.h, a PRIVATE header whose contents differ
// between torch minors (2.5 spells the layout helpers differently again), so
// depending on it is the one thing in this tree a torch bump can break at
// compile time for no reason of ours.  The arguments are the same arguments,
// so the plan is the same plan -- which the bit-exactness gate checks rather
// than this comment asserting it.  Definitions in
// fft.cu (one cache per process); the binding
// visco_acoustic2d_fft_workspace_bytes sizes the pool's work-area slot from
// workspace_bytes().
class ViscoFFT {
public:
    // The cached plan for this geometry on ``device`` (built on first use;
    // never evicted -- a process sees a handful of geometries).
    static std::shared_ptr<ViscoFFT> get(int device, int64_t B, int64_t nz, int64_t nx);

    int device() const { return device_; }
    int64_t batch() const { return B_; }
    int64_t nz() const { return nz_; }
    int64_t nx() const { return nx_; }
    // cufftXtMakePlanMany's work-area size (what ATen at::empty's per call).
    int64_t workspace_bytes() const;

    // out = fft2(in) over the last two axes: cufftSetStream(current stream),
    // cufftSetWorkArea(work_area), cufftXtExec(CUFFT_FORWARD) -- ATen's
    // sequence.  ``in`` / ``out``: contiguous complex64 tensors on device()
    // holding B*nz*nx elements with trailing (nz, nx), in DISTINCT storages
    // (out of place, as ATen always transforms); ``work_area``: a contiguous
    // CUDA slot of at least workspace_bytes() bytes.
    void forward(const Buf& in, const Buf& out, const Buf& work_area);
    // out = ifft2(in) with ATen's default ("backward") normalization: the
    // CUFFT_INVERSE exec, then out.mul_(1.0 / (nz*nx)) -- the very
    // Tensor::mul_(Scalar) with the same double _fft_apply_normalization uses
    // for fft_norm_mode::by_n (skipped, as there, when the scale is 1.0).
    void inverse(const Buf& in, const Buf& out, const Buf& work_area);

    ~ViscoFFT();

    ViscoFFT(const ViscoFFT&) = delete;
    ViscoFFT& operator=(const ViscoFFT&) = delete;

private:
    ViscoFFT(int device, int64_t B, int64_t nz, int64_t nx);
    void exec(const Buf& in, const Buf& out, const Buf& work_area, bool forward);

    int device_;
    int64_t B_, nz_, nx_;
    double inverse_scale_;
    cufftHandle plan_ = 0;      // owned; destroyed in ~ViscoFFT
    int64_t ws_bytes_ = 0;      // cufftXtMakePlanMany's workSize out-param
    std::mutex mutex_;   // SetStream / SetWorkArea / Exec are plan state: one exec at a time
};

// ---------------------------------------------------------------------------
// Workspace slots: the ONE (a, d, mode) -> index mapping of the spectral
// scratch pools, mirroring ViscoAcoustic.cuda_layout verbatim:
//   forward_workspace (mode Forward):
//       spectral ? [C0, C1, (C2 if d), FFT_WS] : []
//   adjoint_workspace (every backward mode):
//       [CARRIER] + (spectral ? [C0, C1, (C2 if d and mode in ckpt/recursive),
//                               (R1 if d), (UPREV if a and mode == ckpt), FFT_WS]
//                            : [])
// C0/C1/C2 are complex64 grids (float32 [B, 1, nz, nx, 2] on the Python side),
// R1/UPREV/CARRIER float32 [B, 1, nz, nx], FFT_WS the flat float32 cuFFT work
// area.  -1 = the slot does not exist for these flags / this mode.
// ---------------------------------------------------------------------------
struct ViscoSlots {
    int carrier = -1;
    int c0 = -1, c1 = -1, c2 = -1;
    int r1 = -1, uprev = -1;
    int fft_ws = -1;
    int count = 0;
};

inline ViscoSlots visco_slots(bool damping, bool dispersion, derived::ViscoMode mode)
{
    using derived::ViscoMode;
    ViscoSlots s;
    const bool forward = (mode == ViscoMode::Forward);
    const bool replay = (mode == ViscoMode::Checkpoint || mode == ViscoMode::Recursive);
    if (!forward) s.carrier = s.count++;
    if (!(damping || dispersion)) return s;
    s.c0 = s.count++;
    s.c1 = s.count++;
    if (dispersion && (forward || replay)) s.c2 = s.count++;
    if (dispersion && !forward) s.r1 = s.count++;
    if (damping && mode == ViscoMode::Checkpoint) s.uprev = s.count++;
    s.fft_ws = s.count++;
    return s;
}

// A complex64 grid over a pool slot laid out as float32 [B, 1, nz, nx, 2]:
// at::view_as_complex gives the contiguous complex64 (B, 1, nz, nx) tensor
// at::empty(kComplexFloat) would, which is also cuFFT's interleaved
// cufftComplex layout.  The slot exists exactly when
// ViscoAcoustic.cuda_layout's forward_workspace_shapes /
// backward_workspace_shapes declared it for this mode and eq_aux composition,
// and the propagator then binds it, so the binding is required.
inline Buf visco_acoustic2d_complex_slot(const std::vector<torch::Tensor>& pool, int idx,
                                         const Buf& like, const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx),
                what, "[", idx, "] (complex spectrum) must be bound by the propagator "
                "(cuda_layout forward_workspace_shapes / backward_workspace_shapes)");
    auto want = like.sizes().vec();
    want.push_back(2);
    const auto& raw = pool[idx];
    SWEEP_CHECK(raw.sizes().vec() == want,
                what, "[", idx, "] has shape ", raw.sizes(),
                " but the complex slot layout is ", want, " (float32 pairs)");
    SWEEP_CHECK(raw.scalar_type() == torch::kFloat && raw.is_cuda() && raw.is_contiguous(),
                what, "[", idx, "] must be a contiguous float32 CUDA tensor");
    return buf_of(raw);   // (..., 2) float pairs: the complex64 grid, as cuFFT reads it
}

// The cuFFT work area over a pool slot: a flat float32 slot of at least
// ceil(workspace_bytes / 4) elements (what visco_acoustic2d_fft_workspace_bytes
// told the Python side).  Declared by ViscoAcoustic.cuda_layout whenever a
// spectral term is on (_fft_work_area_slot asks this very binding for the size),
// so the propagator always binds it and there is no fallback.
inline Buf visco_acoustic2d_work_area_slot(const std::vector<torch::Tensor>& pool, int idx,
                                           const ViscoFFT& fft, const Buf& /*like*/,
                                           const char* what)
{
    const int64_t floats = std::max<int64_t>(1, (fft.workspace_bytes() + 3) / 4);
    SWEEP_CHECK(pool_slot_bound(pool, idx),
                what, "[", idx, "] (cuFFT work area) must be bound by the propagator "
                "(cuda_layout forward_workspace_shapes / backward_workspace_shapes)");
    const auto& raw = pool[idx];
    SWEEP_CHECK(raw.scalar_type() == torch::kFloat && raw.is_cuda() && raw.is_contiguous(),
                what, "[", idx, "] (cuFFT work area) must be a contiguous float32 CUDA tensor");
    SWEEP_CHECK(raw.numel() >= floats,
                what, "[", idx, "] (cuFFT work area) holds ", raw.numel(),
                " floats but the plan needs ", floats,
                " (visco_acoustic2d_fft_workspace_bytes on this device)");
    return buf_of(raw);
}

// A float32 grid aliasing the first half of a complex slot's storage: free
// real scratch whenever that spectrum is dead (the slot is (..., 2)
// contiguous float pairs, so its first numel/2 floats are one contiguous
// grid of the spectrum's shape).
inline Buf visco_acoustic2d_real_alias(const Buf& C)
{
    SWEEP_CHECK(C.defined() && C.dim() >= 2 && C.size(-1) == 2 && C.is_contiguous(),
                "visco_acoustic2d_real_alias: a contiguous (..., 2) complex slot");
    Buf r = C;
    r.ndim_ = C.ndim_ - 1;
    r.numel_ = C.numel() / 2;
    r.sizes_[r.ndim_] = 0;
    r.strides_[r.ndim_] = 0;
    int64_t stride = 1;
    for (int64_t i = r.ndim_ - 1; i >= 0; --i) {
        r.strides_[i] = stride;
        stride *= r.sizes_[i];
    }
    return r;
}

// The complex slot's real parts, as the strided float operand at::real used
// to hand the ATen kernels: pointer + element stride 2.
inline const float* visco_acoustic2d_real_ptr(const Buf& C) { return C.data_ptr<float>(); }
// The slot as cuFFT / the complex kernels read it: float2 pairs over the same
// bytes.  Not Buf::data_ptr<float2>() -- that asserts on the element width,
// and the slot's width is the float's (the JIT build keeps asserts).
inline float2* visco_acoustic2d_complex_ptr(const Buf& C) { return reinterpret_cast<float2*>(C.data_ptr<float>()); }
constexpr int VISCO_REAL_STRIDE = 2;

// The spectral scratch of one call, bound from a workspace pool by
// visco_acoustic2d_bind_scratch.  Members whose slot does not exist for the
// flags / mode stay undefined; ``fft`` is null when no spectral term is on.
struct ViscoScratch {
    ViscoSlots slots;
    std::shared_ptr<ViscoFFT> fft;      // the plan for the bound geometry
    Buf C0, C1, C2;           // complex64 as (like.sizes(), 2) float pairs
    Buf R1, UPREV, CARRIER;   // float32, like.sizes() (backward pools only)
    Buf fft_ws;               // flat float32 cuFFT work area
};

// ``pool``: p.forward_workspace (mode Forward) or p.adjoint_workspace (the
// backward modes); ``like``: the wavefield geometry (u_now_t: (B, 1, nz, nx)).
// The pool must hold exactly visco_slots(...).count tensors.  That count is the
// same function of (damping, dispersion, mode) that ViscoAcoustic.cuda_layout's
// forward_workspace_shapes / backward_workspace_shapes evaluates -- the two
// flags come from the SAME constructor switches on both sides (the eq_aux
// composition c_eq_aux builds is what visco_acoustic2d_spectral_grids reads
// back) -- so a bound pool of any other size means the declaration drifted.
// count == 0 happens only in the forward mode with both spectral terms off,
// where the propagator legitimately hands over an empty tuple.
inline ViscoScratch visco_acoustic2d_bind_scratch(
    const std::vector<torch::Tensor>& pool, const Buf& like,
    bool damping, bool dispersion, derived::ViscoMode mode, const char* what)
{
    ViscoScratch ws;
    ws.slots = visco_slots(damping, dispersion, mode);
    SWEEP_CHECK(static_cast<int>(pool.size()) == ws.slots.count,
                what, ": requires the propagator-bound workspace pool of ", ws.slots.count,
                " tensors (visco_slots for this mode and eq_aux composition), got ", pool.size());
    if (ws.slots.carrier >= 0)
        ws.CARRIER = buf_of(pool_required(pool, ws.slots.carrier, like, what));
    if (!(damping || dispersion))
        return ws;
    SWEEP_CHECK(like.dim() >= 2 && like.is_cuda() && like.element_size() == 4,
                what, ": the wavefield geometry must be a float32 CUDA grid, got ", like.sizes());
    const int64_t nz = like.size(-2);
    const int64_t nx = like.size(-1);
    ws.fft = ViscoFFT::get(like.device_index(), like.numel() / (nz * nx), nz, nx);
    ws.C0 = visco_acoustic2d_complex_slot(pool, ws.slots.c0, like, what);
    ws.C1 = visco_acoustic2d_complex_slot(pool, ws.slots.c1, like, what);
    if (ws.slots.c2 >= 0)
        ws.C2 = visco_acoustic2d_complex_slot(pool, ws.slots.c2, like, what);
    if (ws.slots.r1 >= 0)
        ws.R1 = buf_of(pool_required(pool, ws.slots.r1, like, what));
    if (ws.slots.uprev >= 0)
        ws.UPREV = buf_of(pool_required(pool, ws.slots.uprev, like, what));
    ws.fft_ws = visco_acoustic2d_work_area_slot(pool, ws.slots.fft_ws, *ws.fft, like, what);
    return ws;
}

// ---------------------------------------------------------------------------
// The nearly-constant-Q filter  L(x) = Re(IFFT2(kmul * FFT2(x))) on slots.
// kmul (|k|^(2*gbar+1), or a dispersion grid) is real and even under
// k -> -k, so L is self-adjoint on real fields — the adjoint code applies the
// SAME operator.  Replaces at::real(at::fft_ifft2(kmul * at::fft_fft2(x)))
// (the eager reference step_visco_cpml's transform pair), op for op:
//   C0.copy_(x)            promote float -> complex (== x.to(kComplexFloat))
//   forward(C0 -> C1)      C1 = FFT2(x)
//   mul_out(C0, kmul, C1)  C0 = kmul * F   (kmul FIRST, see the header note)
//   inverse(C0 -> C1)      C1 = IFFT2(C0), normalised by 1/(nz*nx) as ATen
//   return at::real(C1)    a strided view -- valid until C1 is next written
// ``x`` is any float32 view of like.sizes() (a wavefield, at::real(...), a
// product staged in a real alias).  It may live in C1's storage (the alias
// is consumed by the promote before C1 is overwritten) but NOT in C0's.
// ---------------------------------------------------------------------------
// ``x``: a contiguous float grid of the slot's shape (a wavefield, a real
// alias, a history row); ``kmul``: the (nz, nx) filter.  The result is the
// real part of C1 -- visco_acoustic2d_real_ptr(ws.C1) at VISCO_REAL_STRIDE --
// valid until C1 is next written.
inline void visco_acoustic2d_lop_into(const float* x, const float* kmul, ViscoScratch& ws)
{
    const int64_t n = ws.C1.numel() / 2;
    const int64_t ng = ws.C1.size(-3) * ws.C1.size(-2);   // (nz, nx) of the (..., nz, nx, 2) slot
    visco_ops::promote(visco_acoustic2d_complex_ptr(ws.C0), x, n);
    ws.fft->forward(ws.C0, ws.C1, ws.fft_ws);
    visco_ops::cmul_grid(visco_acoustic2d_complex_ptr(ws.C0), kmul, visco_acoustic2d_complex_ptr(ws.C1), n, ng);
    ws.fft->inverse(ws.C0, ws.C1, ws.fft_ws);
}

// One amplitude-damping application on the freshly-written u_next:
//   u_next -= dt2A * L((u_now - u_prev) / dt),  dt2A = dt^2 * A, A = tt * vp / 2
// (the eager step_visco_cpml ordering: stencil first, damping second, source
// injection / recording after).  Replaces
//   dudt = (u_now - u_prev).div_(dt);  u_next.sub_(dt2A * Lop(dudt, kmul));
// dudt is staged in the float alias of C1 (free here: whatever spectrum C1
// held is dead) with the same sub -> div_(dt) pair; the product dt2A * L(...)
// goes into the float alias of C0 (dead after the inverse transform) and is
// subtracted in place, as before.
inline void visco_acoustic2d_apply_damping_into(
    AcousticWavefieldTensor& wf,
    const torch::Tensor& kmul,   // (nz, nx) D_loss grid
    const torch::Tensor& dt2A,   // (B, 1, nz, nx) dt^2 * A (derived table)
    float dt, ViscoScratch& ws)
{
    const Buf dudt = visco_acoustic2d_real_alias(ws.C1);
    const int64_t n = dudt.numel();
    visco_ops::binary<visco_ops::SubOp>(dudt.data_ptr<float>(),
                                        wf.u_now_t.template data_ptr<float>(), 1,
                                        wf.u_prev_t.template data_ptr<float>(), 1, n);
    visco_ops::scale(dudt.data_ptr<float>(), 1.0f / dt, n);            // div_(dt): times the reciprocal
    visco_acoustic2d_lop_into(dudt.data_ptr<float>(), kmul.data_ptr<float>(), ws);   // C1.re = L; dudt is gone
    const Buf R = visco_acoustic2d_real_alias(ws.C0);
    visco_ops::binary<visco_ops::MulOp>(R.data_ptr<float>(), dt2A.data_ptr<float>(), 1,
                                        visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);
    visco_ops::inplace<visco_ops::SubOp>(wf.u_next_t.template data_ptr<float>(), R.data_ptr<float>(), 1, n);
}

// Forward spectral terms on slots: dispersion first, damping second (the
// eager step's ordering), one halo restore after — the FFTs write the halo
// bands, and the stencil kernels rely on halo == 0 (= the pressure-release
// image condition on free-surface faces).  Replaces
//   F = fft2(u_now);  u_next.add_(Gd1 * real(ifft2(Dk2 * F)));
//   u_next.sub_(Gd2 * real(ifft2(Dfrac * F)));  then the damping above.
// Slot liveness
// through the dispersion term (F = FFT2(u_now) stays in C1 across BOTH
// products, so a third slot C2 receives the inverse transforms):
//   C0 <- promote(u_now)          C1 <- F = FFT2(C0)          [C0 dead]
//   C0 <- Dk2 * F                 C2 <- IFFT2(C0) / n         [C0 dead]
//   R := real alias of C0;  R <- Gd1 * real(C2);  u_next += R  [R dead]
//   C0 <- Dfrac * F               C2 <- IFFT2(C0) / n         [C0 dead]
//   R <- Gd2 * real(C2);  u_next -= R                         [F, C2 dead]
// then the damping pipeline reuses C0 / C1 (its dudt is staged in C1's alias,
// which F no longer occupies).  Every ATen call is the one the replaced
// expression made, on the same operands in the same order.
inline void visco_acoustic2d_apply_spectral_into(
    AcousticWavefieldTensor& wf, const ViscoSpectral& s, ViscoScratch& ws, float dt, int M)
{
    if (!(s.active || s.disp)) return;
    if (s.disp) {
        SWEEP_CHECK(ws.C2.defined(), "visco_acoustic2d: the dispersion pipeline needs the C2 slot "
                    "(bind the scratch with dispersion on, in forward / ckpt / recursive mode)");
        const int64_t n = ws.C1.numel() / 2;
        const int64_t ng = ws.C1.size(-3) * ws.C1.size(-2);
        float2* C0 = visco_acoustic2d_complex_ptr(ws.C0);
        visco_ops::promote(C0, wf.u_now_t.template data_ptr<float>(), n);
        ws.fft->forward(ws.C0, ws.C1, ws.fft_ws);            // C1 = F, alive across both products
        visco_ops::cmul_grid(C0, s.Dk2.data_ptr<float>(), visco_acoustic2d_complex_ptr(ws.C1), n, ng);   // C0 = Dk2 * F
        ws.fft->inverse(ws.C0, ws.C2, ws.fft_ws);            // C2 = IFFT2(Dk2 * F)
        const Buf R = visco_acoustic2d_real_alias(ws.C0);     // the product in C0 is dead
        visco_ops::binary<visco_ops::MulOp>(R.data_ptr<float>(), s.Gd1.data_ptr<float>(), 1,
                                            visco_acoustic2d_real_ptr(ws.C2), VISCO_REAL_STRIDE, n);
        visco_ops::inplace<visco_ops::AddOp>(wf.u_next_t.template data_ptr<float>(), R.data_ptr<float>(), 1, n);
        visco_ops::cmul_grid(C0, s.Dfrac.data_ptr<float>(), visco_acoustic2d_complex_ptr(ws.C1), n, ng); // C0 = Dfrac * F (R's bits are dead)
        ws.fft->inverse(ws.C0, ws.C2, ws.fft_ws);            // C2 = IFFT2(Dfrac * F)
        visco_ops::binary<visco_ops::MulOp>(R.data_ptr<float>(), s.Gd2.data_ptr<float>(), 1,
                                            visco_acoustic2d_real_ptr(ws.C2), VISCO_REAL_STRIDE, n);
        visco_ops::inplace<visco_ops::SubOp>(wf.u_next_t.template data_ptr<float>(), R.data_ptr<float>(), 1, n);
    }
    if (s.active)
        visco_acoustic2d_apply_damping_into(wf, s.kmul, s.dt2A, dt, ws);
    visco_acoustic2d_zero_halo(wf.u_next_t, M);
}

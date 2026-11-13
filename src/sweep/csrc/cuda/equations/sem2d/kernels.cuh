// SEM 2-D CUDA kernels (impl='c' backend for sweep.sem).
//
// These reproduce, on the GPU and in the SAME FLOP order, the eager
// AcousticPotentialSEM2D arithmetic (see sweep/sem/equations/acoustic_potential2d.py
// and sweep/sem/_assembly.py). Ported — NOT copied — from SPECFEM2D's
// Kernel_2_acoustic_impl / update kernels: the arithmetic (GLL contraction via the
// derivative matrix, the (1/rho) flux, the weak-form D^T contraction, the Newmark
// update) is faithful to SPECFEM, but the indexing follows sweep.sem's native
// contiguous (E, ngll, ngll), 0-based, (a=eta, b=xi) layout — copying SPECFEM's
// (i=xi, j=eta) order would silently transpose the differentiation.
//
// Axis convention (must match _assembly.py exactly):
//   ibool[e,a,b] laid out row-major -> ibool[e*m + a*ngll + b], a = eta, b = xi.
//   D is (ngll,ngll) row-major, D[i*ngll+j] = l'_j(x_i).
//   d/dxi   at (a,b) = sum_g D[b,g] * f[a,g]     (contract the b/xi axis)
//   d/deta  at (a,b) = sum_g D[a,g] * f[g,b]     (contract the a/eta axis)
//   weak form r[a,b] = sum_d D[d,b]*Fbeta[a,d] + sum_g D[g,a]*Falpha[g,b]
//
// fp32 only (per the project decision); assembly is a plain atomicAdd (the T2
// cos>=1-1e-6 parity bar does not require a deterministic reduction).
//
// Launch shape of the element-local kernels: ONE ELEMENT PER BLOCK, blockDim =
// m = ngll^2. Two "obvious" improvements were measured and both lose; do not
// redo them without new evidence (RTX 6000 Ada, acoustic C-PML, 55296 elements,
// 886657 GLL nodes, order 4, where _grads_f is 29% of the step at 9.89 ms):
//
//   * Packing 5 elements per block to fill the warp. m = 25 means a block is
//     0.78 of a warp and residency caps at 24 blocks x 25 threads = 39% of the
//     1536 threads an SM holds, so packing looks free. It measured 10.78 ms,
//     i.e. 9% SLOWER, and 10.74 ms with a 2-D block (so it is not the runtime
//     integer division by m). The reason is that at m <= 32 a one-element block
//     IS a single warp: __syncthreads() costs almost nothing and the two
//     shared-memory phases are warp-synchronous. Packing turns both barriers
//     into real cross-warp barriers. The narrow block is load-bearing.
//   * Replacing the six assembly atomicAdds with plain stores (numerically
//     wrong; run only to size the prize) measured 9.75 ms, i.e. atomics cost
//     1.4%. Mesh colouring or a gather-based assembly would buy nothing here.
#pragma once

#include "../../../core/device.h"
#include "../../launch/launch.h"   // SWEEP_LAUNCH
#include "../../../core/sem_input_core.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sem2d {

// ---- element stiffness: K += assemble( D^T (w_J * (1/rho) grad chi) ) --------
// One block per element, `m = ngll*ngll` threads (one per GLL node of the element).
// Dynamic shared memory carries s_chi, s_Fbeta, s_Falpha (3*m floats).
__global__ void sem2d_acoustic_stiffness(
    const float* __restrict__ chi,
    const int*    __restrict__ ibool,
    const float* __restrict__ D,
    const float* __restrict__ inv_rho,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ K,
    int ngll, int E);

// ---- Newmark 2nd-order displacement update (bare, no CPML) --------------------
//   chi_next = 2*chi - chi_prev + dt2 * inv_M * (-K + pattern * wavelet[n])
__global__ void sem2d_acoustic_newmark(
    const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const float* __restrict__ K,
    const float* __restrict__ pattern,
    const float* __restrict__ inv_M,
    const float* __restrict__ wavelet,
    int n, float dt2,
    float* __restrict__ chi_next,
    int n_global);

// ---- Dirichlet (acoustic free surface): pin chi = 0 on the listed nodes -------
__global__ void sem2d_dirichlet_fill(
    float* __restrict__ chi_next, const long* __restrict__ didx, int k);

// shot-batched variant (gridDim.y = B): the base form has ~20 single-shot
// call sites across the drivers, so the batched acoustic C-PML reverse march
// gets its own rather than a signature change.
__global__ void sem2d_dirichlet_fill_b(
    float* __restrict__ f, const long* __restrict__ didx, int k, int ng);

// ---- receiver record: seis[r,n] = sum_m field[rec_G[r,m]] * rec_W[r,m] --------
__global__ void sem2d_record(
    const float* __restrict__ field,
    const long*  __restrict__ rec_G,
    const float* __restrict__ rec_W,
    float* __restrict__ seis,
    int n, int nt, int n_rec, int mgll, int ng);

// ======================= backward (adjoint) kernels ==========================
// Adjoint-state gradient of AcousticPotentialSEM2D.step w.r.t. inv_M, inv_rho,
// wavelet. Reverse leap-frog (K self-adjoint; the transpose applies inv_M BEFORE
// the stiffness, i.e. K(inv_M o a)); imaging correlation of forward field u^{k}
// and the adjoint-derived field y = -dt2 inv_M o a. See backward.cu for the loop.

// out[i] = s * a[i] * b[i]   (e.g. inv_M o a, or the -dt2 inv_M o a field)
__global__ void sem2d_scale2(const float* __restrict__ a, const float* __restrict__ b,
                             float s, float* __restrict__ out, int n);

// Shot-batched variants (gridDim.y = B) of three kernels shared by many
// drivers. Separate kernels rather than signature changes: the base forms
// have ~25 single-shot call sites across the elastic / fluid-solid / plain
// acoustic drivers, and only the batched acoustic C-PML path needs these.
// as_/msn are the stride of the possibly-broadcast operand (0 = shared).
__global__ void sem2d_scale2_b(const float* __restrict__ a,
                             const float* __restrict__ b, float s,
                             float* __restrict__ out, int n, int as_);

// adjoint source: adj[node] += grad_syn[r, tcol] * rec_W[r,j]  (R^T of the record)
__global__ void sem2d_adjoint_source(
    const float* __restrict__ grad_syn, const long* __restrict__ rec_G,
    const float* __restrict__ rec_W, float* __restrict__ adj,
    int tcol, int nt, int n_rec, int mgll);

// batched adjoint source (see the _b note above): grad_syn is (B, nrec, nt),
// the adjoint field advances by ng per shot; receivers are per-mesh.
__global__ void sem2d_adjoint_source_b(
    const float* __restrict__ grad_syn, const long* __restrict__ rec_G,
    const float* __restrict__ rec_W, float* __restrict__ adj,
    int tcol, int nt, int n_rec, int mgll, int ng);

// reverse leap-frog: ubar[i] = 2 a1[i] - a2[i] - dt2 Ktmp[i] + adj[i]
__global__ void sem2d_adj_leapfrog(
    const float* __restrict__ a1, const float* __restrict__ a2,
    const float* __restrict__ Ktmp, const float* __restrict__ adj,
    float dt2, float* __restrict__ ubar, int n);

// grad_inv_M[i] += dt2 * a_m[i] * (pattern[i]*wavelet[k] - Ku[i])
__global__ void sem2d_accum_gradM(
    const float* __restrict__ a_m, const float* __restrict__ pattern,
    const float* __restrict__ Ku, const float* __restrict__ wavelet, int k,
    float dt2, float* __restrict__ gradM, int n);

// Block-wide sum -> ONE atomicAdd per block. The wavelet gradient is a single
// scalar per step, so the naive form had every one of the n_global threads
// serialise on the same address (n_global ~ 1.7e5 on the Marmousi benchmark,
// every step, forward AND adjoint). Warp shuffle + one shared slot per warp
// cuts that to gridDim.x atomics. Float addition reorders, but the previous
// all-thread atomic order was already nondeterministic.
static __device__ __forceinline__ float sem2d_block_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    __shared__ float s[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarp) ? s[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;   // meaningful on threadIdx.x == 0
}

// grad_wavelet[k] += dt2 * sum_i pattern[i]*inv_M[i]*a_m[i]  (block reduction)
__global__ void sem2d_accum_gradW(
    const float* __restrict__ pattern, const float* __restrict__ inv_M,
    const float* __restrict__ a_m, float dt2, int k,
    float* __restrict__ gradW, int n);

// batched wavelet gradient (see the _b note above): gradW is (B, nt), the
// pattern/field advance by n per shot, inv_M by msn (0 = shared model).
__global__ void sem2d_accum_gradW_b(
    const float* __restrict__ pattern, const float* __restrict__ inv_M,
    const float* __restrict__ a_m, float dt2, int k,
    float* __restrict__ gradW, int n, int nt, int msn);

// imaging: grad_inv_rho[e,p] += mask_e[e] * w_J[e,p] * (grad u . grad y) at (e,p),
// with u = saved forward field u^{k}, y = -dt2 inv_M o a_m. One block per element.
__global__ void sem2d_imaging_invrho(
    const float* __restrict__ u, const float* __restrict__ y,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e, float* __restrict__ grad_inv_rho,
    int ngll, int E);

// ======================= elastic forward kernels ============================
// Reproduce ElasticSEM2DDeformed.stiffness_apply / _assemble_stress (bare, no
// CPML, so sigma_zx == sigma_xz). 2-component displacement (u_x, u_z). One block
// per element, m = ngll*ngll threads; shared: s_ux, s_uz (gather) + 4 flux arrays.
__global__ void sem2d_elastic_stiffness(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ Kux, float* __restrict__ Kuz, int ngll, int E);

// 2-component Newmark: u?_next = 2 u? - u?_prev + dt2 inv_M (pattern? * wav[n] - K?)
__global__ void sem2d_elastic_newmark(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const float* __restrict__ Kux, const float* __restrict__ Kuz,
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ inv_M, const float* __restrict__ wavelet,
    int n, float dt2, float* __restrict__ ux_next, float* __restrict__ uz_next,
    int n_global, int wst);

// ======================= elastic backward kernels ===========================
// grad_inv_M[i] += dt2 (ax(pat_x*wav - Kux) + az(pat_z*wav - Kuz))  (both components)
__global__ void sem2d_elastic_accum_gradM(
    const float* __restrict__ ax, const float* __restrict__ az,
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ Kux, const float* __restrict__ Kuz,
    const float* __restrict__ wavelet, int k, float dt2,
    float* __restrict__ gradM, int n);

// grad_wavelet[k] += dt2 sum_i inv_M[i](pat_x[i]*ax[i] + pat_z[i]*az[i])
__global__ void sem2d_elastic_accum_gradW(
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ inv_M, const float* __restrict__ ax,
    const float* __restrict__ az, float dt2, int k,
    float* __restrict__ gradW, int n, int ntW);

// elastic imaging: grad_{lam2mu,lam,mu}[e,p] += mask_e w_J * (...), from the phys
// gradients of the forward u=(ux,uz) and the adjoint-derived y=(yx,yz). One block
// per element; shared: s_ux,s_uz,s_yx,s_yz (4m).
__global__ void sem2d_elastic_imaging(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ yx, const float* __restrict__ yz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ grad_lam2mu, float* __restrict__ grad_lam,
    float* __restrict__ grad_mu, int ngll, int E);

// out[i] = s * in[i]  (scalar scale; used for gw = dt2*kappa*gA via scale2, and neg)
__global__ void sem2d_scale1(const float* __restrict__ in, float s,
                             float* __restrict__ out, int n);

// ---- fused variants (perf; the pure-acoustic drivers use these, the coupled
// fluid-solid driver keeps the split kernels) --------------------------------

// grads with the a*psi products folded into the gather (drops 2 scale2
// launches + 2 scratch buffers per step).
// The GLL weight x Jacobian is DERIVED here rather than read: w_J = w1d[a]*w1d[b]
// / det(inv_J), exact to machine precision (verified 2.8e-16 on flat and on
// topography meshes). It trades one 4-byte global read per element-node for a
// handful of FLOPs, which is the right trade on a DRAM-bandwidth-bound kernel.
// SPECFEM does the same: rho_invl_times_jacobianl = 1/(rho*(xix*gammaz-gammax*xiz)).
// Not bit-identical to the stored w_J (fp32 division), so judge it against the
// atomicAdd reordering floor, not by bit-equality.
__global__ void sem2d_acoustic_cpml_grads_f(
    const float* __restrict__ chi, const float* __restrict__ psi_x,
    const float* __restrict__ psi_z, const float* __restrict__ a_x,
    const float* __restrict__ a_z, const int* __restrict__ ibool,
    const float* __restrict__ D, const float* __restrict__ inv_rho,
    const float* __restrict__ w1d,   // 1-D GLL weights (ngll,)
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ gx, float* __restrict__ gz,
    float* __restrict__ kx, float* __restrict__ kz,
    float* __restrict__ gax, float* __restrict__ gaz, int ngll, int E,
    int ng, int ms);

// fused step algebra + Newmark + optional in-kernel save writes (drops one
// launch + 5 D2D memcpys per step). Saves record PRE-dirichlet values here;
// sem2d_dirichlet_fill5 zeroes both the fields and the save slices, so the
// stored history is post-dirichlet as the backward expects.
// Interior fast path. Outside the C-PML profile b_x = b_z = db*_d* = 0, and then
// psi and zeta are identically zero for all time (they are only ever fed by
// b_*, and they start at zero) -- verified on a 1.77 s propagation with a
// 4-sided band: exactly 0.0 on all 169713 interior nodes, not merely small.
// Everything those quantities feed therefore drops out EXACTLY, so this path is
// bit-identical, not an approximation, and the test asserts bit-equality.
//
// The predicate is on b/db only. gax/gaz are still read: a node with b = 0 can
// belong to an element that DOES contain absorbing nodes (a rim node), and the
// assembled gax there is non-zero. Assuming otherwise would silently corrupt the
// gradient in a one-element-wide ring.
//
// Why it matters: at production sizes 92-96% of nodes are outside the band, and
// these two kernels are ~23% of the step while carrying most of the C-PML
// traffic (6 profile arrays + 4 memory fields read, 4 memory fields written).
// WSAVE: the save-all mode stores w_sum^n (the `ws` register) into the tape
// row instead of the four psi/zeta slices -- see sem_types.h. A compile-time
// flag, not a runtime one: storing ws forces the compiler to materialise the
// sum, which costs one FADD it otherwise reassociates away, and <false> must
// stay the pre-tape kernel down to its arithmetic.
// K##_by_flag is the kernel's table, <false> then <true>.
#define SEM2D_LAUNCH_WS(wsave, K, grid, block, shm, stream, ...) \
    SWEEP_LAUNCH_ON(K##_by_flag[(wsave) ? 1 : 0], grid, block, shm, stream, __VA_ARGS__)

using sem2d_acoustic_cpml_step_newmark_f_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, int, float, float*, float*, float*,
    float*, float*, float*, float*, float*, float*, float*, float*, int, int, int);
extern sem2d_acoustic_cpml_step_newmark_f_fn const sem2d_acoustic_cpml_step_newmark_f_by_flag[2];

// zero 5 fields (and, when saving, their save slices) on the dirichlet nodes.
__global__ void sem2d_dirichlet_fill5(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ f3, float* __restrict__ f4,
    float* __restrict__ s0, float* __restrict__ s1, float* __restrict__ s2,
    float* __restrict__ s3, float* __restrict__ s4,   // may be null
    const long* __restrict__ didx, int k, int ng);

// backward combine with the gB rotation folded in (gB_out = -gA).
__global__ void sem2d_acoustic_cpml_combine_f(
    const float* __restrict__ gA, const float* __restrict__ gB, const float* __restrict__ lchi_G,
    const float* __restrict__ kappa, const float* __restrict__ w_sum,
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int k,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_z,
    const float* __restrict__ lapx, const float* __restrict__ lapz,
    const float* __restrict__ gw, const float* __restrict__ a_x, const float* __restrict__ a_z,
    float* __restrict__ lchi_k, float* __restrict__ lpsi_x_n, float* __restrict__ lpsi_z_n,
    float* __restrict__ lzeta_x_n, float* __restrict__ lzeta_z_n,
    float* __restrict__ grad_kappa, float* __restrict__ grad_inv_M,
    float* __restrict__ gB_out, float dt2, int ng, int msn, int wst);

// --------------------------------------------------------------------------- //
// Low-precision band history. The ring is O(nt * nband) and is the dominant
// boundary-saving allocation, so storing it below fp32 is the main memory lever
// after rim mode. bf16 is a plain cast; int8 is per-(step, field) symmetric
// quantisation, matching the eager path in sweep.sem.boundary_saving.
//
// Measured on the eager side (49x65, 4-sided C-PML, 900-7200 steps), cosine
// against the full tape over the model region: bf16 0.9999997, int8 0.999988,
// and the error does NOT grow with nt -- the ring is re-injected on every
// reverse step rather than compounded. Combined with rim mode, int8 is 45.6x
// less history than an fp32 full band.
// --------------------------------------------------------------------------- //
enum : int { SEM_BAND_F32 = 0, SEM_BAND_BF16 = 1, SEM_BAND_I8 = 2 };

struct SemBandHist {
    void* data = nullptr;      // (nt, nband) in the store dtype
    float* scale = nullptr;    // (nt,) amax per step; int8 only
    int kind = SEM_BAND_F32;
};

// Step-indexed save / load. The offset arithmetic lives here so that the call
// sites stay one line each and cannot get the store dtype and the row stride
// out of sync.
// Build a band-history handle: the tensor's data pointer in whatever store
// dtype Python allocated, plus this field's row of the (n_fields, nt) int8 scale
// table. slot is the field's index in that table.
// Shot batch (B > 1): the ring is (nt, B, nband) -- step base step*B*nband,
// per-shot offset bat*nband inside the kernels -- and the int8 scale table is
// (n_fields, nt, B) so each shot quantises against its OWN amplitude. The
// defaults keep the 14 single-shot call sites (sem3d + elastic) untouched.
// ``row`` is how many scales one (field, step, shot) owns: 1 for the node
// rings (a single band amax) and nbe for the ELEMENT rings, which carry one
// amax per band element -- see sem2d_band_amax_elem for why.
static inline SemBandHist sem_hist(const Buf& t, const Buf& sc,
                                   int kind, int slot, long nt, int B = 1,
                                   int row = 1) {
    SemBandHist h;
    if (t.numel() == 0) return h;
    h.kind = kind;
    h.data = const_cast<void*>(t.data_ptr());
    if (kind == SEM_BAND_I8)
        h.scale = const_cast<float*>(sc.data_ptr<float>())
                + static_cast<size_t>(slot) * nt * B * row;
    return h;
}

void sem2d_band_save(const float* field, const long* band_idx,
                     SemBandHist h, long step, int nband,
                     int tpb, cudaStream_t stream,
                     int B = 1, int ng = 0);

void sem2d_band_load(SemBandHist h, long step, const long* band_idx,
                     float* field, int nband, int tpb,
                     cudaStream_t stream, int B = 1, int ng = 0);

// A field's ring index. With ``band_split`` the driver stores each C-PML
// memory field on ITS OWN axis's support (psi_x / zeta_x where b_x != 0, ...)
// instead of on the union band: the update is psi_d' = b_d * q + a_d * psi_d
// from a zero start, so psi_d is exactly 0.0 wherever b_d == 0, for all time,
// and storing those zeros bought nothing -- measured 2.2-2.5x of the ring at
// production sizes (the union of three slabs is ~2.4x any one of them). An
// EMPTY support is a legitimate answer: in rim mode the rim carries no b at
// all, so six of the seven rings vanish. Without band_split every field keeps
// the union index, which is what every driver did before.
struct SemBandIdx { const long* idx = nullptr; int n = 0; };

static inline SemBandIdx sem_band_idx(bool split, const Buf& own,
                                      const long* union_idx, int union_n) {
    SemBandIdx b;
    if (!split) { b.idx = union_idx; b.n = union_n; return b; }
    b.n = static_cast<int>(own.numel());
    b.idx = b.n > 0 ? own.data_ptr<int64_t>() : nullptr;
    return b;
}

// Adjoint carry + pin + accel scale, fused. The reverse loop used to spend six
// launches per step here -- two device-to-device copies (g <- lambda), two
// dirichlet_fill scatters to pin the outer PML wall, and two scale2 passes for
// y = -dt^2 invM . g -- touching 2*n_global three times over. Post-CUDA-graph
// this bookkeeping cost MORE GPU time than the two real element kernels, so it
// is one pass now. The pin arrives as a 0/1 mask over nodes rather than an index
// list, so it can be applied per node instead of scattered.
__global__ void sem2d_elastic_adj_carry(
    const float* __restrict__ lam1_x, const float* __restrict__ lam1_z,
    const float* __restrict__ pin_mask,     // 1 = live node, 0 = pinned wall
    const float* __restrict__ inv_M, float neg_dt2,
    float* __restrict__ g_x, float* __restrict__ g_z,
    float* __restrict__ y_x, float* __restrict__ y_z, int n);

void sem2d_band_save_elem(const float* field,
                          const long* elem_idx, SemBandHist h,
                          long step, int nbe, int m, long Em,
                          int tpb, cudaStream_t stream,
                          int B = 1);

void sem2d_band_load_elem(SemBandHist h, long step,
                          const long* elem_idx, float* field,
                          int nbe, int m, long Em, int tpb,
                          cudaStream_t stream, int B = 1);

// step_cpml algebra (per global node): w_sum + psi/zeta updates.
__global__ void sem2d_acoustic_cpml_step(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ kz,
    const float* __restrict__ gax, const float* __restrict__ gaz,
    const float* __restrict__ inv_Mg,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx_dx,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz_dz,
    const float* __restrict__ psi_x, const float* __restrict__ psi_z,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_z,
    float* __restrict__ w_sum, float* __restrict__ psi_x_n, float* __restrict__ psi_z_n,
    float* __restrict__ zeta_x_n, float* __restrict__ zeta_z_n, int n);

// CPML Newmark: chi_next = 2 chi - chi_prev + dt2 (kappa_node*w_sum + inv_M*pattern*wav[n])
__global__ void sem2d_acoustic_cpml_newmark(
    const float* __restrict__ chi, const float* __restrict__ chi_prev,
    const float* __restrict__ kappa_node, const float* __restrict__ w_sum,
    const float* __restrict__ inv_M, const float* __restrict__ pattern,
    const float* __restrict__ wavelet, int n, float dt2,
    float* __restrict__ chi_next, int ng, int msn, int wst);

// ================== acoustic CPML backward kernels ===========================

// step-algebra transpose + recompute (node kernel). Recomputes qx/dvx/tmp/w_sum
// from the forward grads (gx..gaz), then transposes op4-op11: outputs the 6
// grad-field adjoints (gbx..gbaz), the recomputed w_sum, and accumulates
// grad_inv_Mg. Memory-adjoint finalisation happens in the combine kernel.
// Interior fast path. Outside the C-PML profile b_x = b_z = db*_d* = 0, and then
// psi and zeta are identically zero for all time (they are only ever fed by
// b_*, and they start at zero) -- verified on a 1.77 s propagation with a
// 4-sided band: exactly 0.0 on all 169713 interior nodes, not merely small.
// Everything those quantities feed therefore drops out EXACTLY, so this path is
// bit-identical, not an approximation, and the test asserts bit-equality.
//
// The predicate is on b/db only. gax/gaz are still read: a node with b = 0 can
// belong to an element that DOES contain absorbing nodes (a rim node), and the
// assembled gax there is non-zero. Assuming otherwise would silently corrupt the
// gradient in a one-element-wide ring.
//
// Why it matters: at production sizes 92-96% of nodes are outside the band, and
// these two kernels are ~23% of the step while carrying most of the C-PML
// traffic (6 profile arrays + 4 memory fields read, 4 memory fields written).
// w_sum-tape twin (the 2-D counterpart of SEM3D_LAUNCH_SLAB2; no SLAB axis):
// a compile-time flag, not a runtime branch, so <false> stays the pre-tape
// text and --use_fast_math cannot re-contract the surviving expressions.
// K##_by_flag is the kernel's table, <false> then <true>.
#define SEM2D_LAUNCH_WT(wt, K, grid, block, shm, stream, ...) \
    SWEEP_LAUNCH_ON(K##_by_flag[(wt) ? 1 : 0], grid, block, shm, stream, __VA_ARGS__)

// WTAPE: the forward saved w_sum^k (save_w_sum), so this kernel neither
// recomputes it nor accumulates grad_inv_Mg (which needs the assembled
// fields): it reads none of gx..gaz nor zeta and writes no w_sum_out. Only
// the adjoint algebra (gt/gq from gw and the carries) runs -- the same text
// in both instantiations, <false> being the pre-tape kernel verbatim.
using sem2d_acoustic_cpml_step_T_fn = void (*)(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, float*,
    float*, float*, float*, float*, float*, float*, float*, int, int);
extern sem2d_acoustic_cpml_step_T_fn const sem2d_acoustic_cpml_step_T_by_flag[2];

// cpml_grads transpose (G^T), one block per element. Given the 6 grad-field
// adjoints, produce lchi (adjoint of chi), lapx/lapz (adjoints of a*psi), and
// accumulate grad_inv_rho. Reverses: gx/gz (direct flux), kx/kz (D^T weak-div),
// gax/gaz (direct psi-grad). See derivation in prop_c/P3b notes.
// The GLL weight x Jacobian is DERIVED here rather than read: w_J = w1d[a]*w1d[b]
// / det(inv_J), exact to machine precision (verified 2.8e-16 on flat and on
// topography meshes). It trades one 4-byte global read per element-node for a
// handful of FLOPs, which is the right trade on a DRAM-bandwidth-bound kernel.
// SPECFEM does the same: rho_invl_times_jacobianl = 1/(rho*(xix*gammaz-gammax*xiz)).
// Not bit-identical to the stored w_J (fp32 division), so judge it against the
// atomicAdd reordering floor, not by bit-equality.
__global__ void sem2d_acoustic_cpml_grads_T(
    const float* __restrict__ chi, const float* __restrict__ gbx, const float* __restrict__ gbz,
    const float* __restrict__ gbkx, const float* __restrict__ gbkz,
    const float* __restrict__ gbax, const float* __restrict__ gbaz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ inv_rho, const float* __restrict__ w1d,   // 1-D GLL weights (ngll,)
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ lchi, float* __restrict__ lapx, float* __restrict__ lapz,
    float* __restrict__ grad_inv_rho, int ngll, int E, int ng, int ms);

// ================== elastic C-PML forward kernels (P3e) =====================
// SPECFEM2D-style 2nd-order displacement C-PML — the CUDA port of
// ElasticSEM2DDeformed._step_cpml. All coefficient fields are element-local
// (E, ngll, ngll); the 6 memory fields are (E, ngll, ngll, 2) updated in place
// (each thread owns its node, no races). Passed as one POD struct by value to
// keep the launch signature sane.
struct ElCpmlCoefs {
    // strain stretch of d/dx derivatives [SPECFEM A5-A7] + 2 conv branches (zx)
    const float *Sx0, *Sx1, *Sx2;
    const float *zx1_0, *zx1_1, *zx1_2, *zx2_0, *zx2_1, *zx2_2;
    // strain stretch of d/dz derivatives [A8-A10] + 2 conv branches (xz)
    const float *Sz0, *Sz1, *Sz2;
    const float *xz1_0, *xz1_1, *xz1_2, *xz2_0, *xz2_1, *xz2_2;
    // acceleration / mass contribution [A1-A4] + 2 displ-memory branches
    const float *La1, *La2, *La3, *La4;
    const float *l1_0, *l1_1, *l1_2, *l2_0, *l2_1, *l2_2;
    // (E,) 1 where the element touches the band; nullptr = every element does
    const int *elem_band;
};

// Pack the 27 flattened C-PML coefficient tensors into ElCpmlCoefs. Templated
// on the input struct: the elastic forward / backward / fluid-solid structs
// all name these fields identically, and the packing was copied out by hand in
// each driver.
template <class In>
static inline ElCpmlCoefs sem2d_el_coefs(const In& in) {
    ElCpmlCoefs cf{};
    cf.Sx0 = in.Sx0.template data_ptr<float>();
    cf.Sx1 = in.Sx1.template data_ptr<float>();
    cf.Sx2 = in.Sx2.template data_ptr<float>();
    cf.zx1_0 = in.zx1_0.template data_ptr<float>();
    cf.zx1_1 = in.zx1_1.template data_ptr<float>();
    cf.zx1_2 = in.zx1_2.template data_ptr<float>();
    cf.zx2_0 = in.zx2_0.template data_ptr<float>();
    cf.zx2_1 = in.zx2_1.template data_ptr<float>();
    cf.zx2_2 = in.zx2_2.template data_ptr<float>();
    cf.Sz0 = in.Sz0.template data_ptr<float>();
    cf.Sz1 = in.Sz1.template data_ptr<float>();
    cf.Sz2 = in.Sz2.template data_ptr<float>();
    cf.xz1_0 = in.xz1_0.template data_ptr<float>();
    cf.xz1_1 = in.xz1_1.template data_ptr<float>();
    cf.xz1_2 = in.xz1_2.template data_ptr<float>();
    cf.xz2_0 = in.xz2_0.template data_ptr<float>();
    cf.xz2_1 = in.xz2_1.template data_ptr<float>();
    cf.xz2_2 = in.xz2_2.template data_ptr<float>();
    cf.La1 = in.La1.template data_ptr<float>();
    cf.La2 = in.La2.template data_ptr<float>();
    cf.La3 = in.La3.template data_ptr<float>();
    cf.La4 = in.La4.template data_ptr<float>();
    cf.l1_0 = in.l1_0.template data_ptr<float>();
    cf.l1_1 = in.l1_1.template data_ptr<float>();
    cf.l1_2 = in.l1_2.template data_ptr<float>();
    cf.l2_0 = in.l2_0.template data_ptr<float>();
    cf.l2_1 = in.l2_1.template data_ptr<float>();
    cf.l2_2 = in.l2_2.template data_ptr<float>();
    cf.elem_band = in.elem_band.numel()
        ? in.elem_band.template data_ptr<int32_t>() : nullptr;
    return cf;
}

// Fused per-element step: gather u_now/u_prev (both components), update the 4
// strain memories, form stretched derivatives + non-symmetric PML stress, D^T
// weak-form divergence, update the 2 displ memories + PML acceleration, and
// atomicAdd  mask*R + aPML  into the shared accumulator (Acc_x, Acc_z).
// Optionally records the 4 stretched derivatives and the 2 acceleration paren
// terms (without the w_rho_J factor) for the backward imaging pass.
__global__ void sem2d_elastic_cpml_element(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ wJ,
    const float* __restrict__ wrhoJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e, const ElCpmlCoefs cf,
    float* __restrict__ r_dux_dx, float* __restrict__ r_dux_dz,
    float* __restrict__ r_duz_dx, float* __restrict__ r_duz_dz,
    float* __restrict__ r_displ_x, float* __restrict__ r_displ_z,
    float inv_dt,
    float* __restrict__ Acc_x, float* __restrict__ Acc_z,
    float* __restrict__ save_sd_xx, float* __restrict__ save_sd_zx,
    float* __restrict__ save_sd_xz, float* __restrict__ save_sd_zz,
    float* __restrict__ save_apx, float* __restrict__ save_apz,
    long save_off, int ngll, int E, int ng);

// ================== elastic C-PML backward kernels (P3e) ====================
// Reverse iteration n (forward iteration n produced pinned u^{n+1}):
//   g   = pin-masked adjoint of u^{n+1}          (computed in the driver)
//   y   = -dt^2 inv_M o g                        (adjoint of the accel Acc)
// This kernel is the transpose of sem2d_elastic_cpml_element. Adjoint stress is
// lsig_ij = mask wJ d_j(gather(y_i)); the memory-adjoint recursion mu^n =
// use^n + c0 mu^{n+1} runs in place on the carry buffers; the lambda-d terms
// transpose through cov -> D^T -> scatter into sc_now (-> lambda u^n) and
// sc_prev (-> lambda u^{n-1}, consumed one reverse step later). Imaging uses
// the SAVED stretched derivatives / accel parens + recomputed raw derivatives
// of u^n. u^n may be null (n = 0: zero field).
__global__ void sem2d_elastic_cpml_element_T(
    const float* __restrict__ y_x, const float* __restrict__ y_z,
    const float* __restrict__ ux_n, const float* __restrict__ uz_n,  // may be null
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ wJ,
    const float* __restrict__ wrhoJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e, const ElCpmlCoefs cf,
    const float* __restrict__ sd_xx, const float* __restrict__ sd_zx,   // saved
    const float* __restrict__ sd_xz, const float* __restrict__ sd_zz,   // stretched
    const float* __restrict__ ap_x, const float* __restrict__ ap_z,     // saved parens
    float* __restrict__ mu_xx, float* __restrict__ mu_zx,   // strain-memory adjoints
    float* __restrict__ mu_xz, float* __restrict__ mu_zz,   // (E,ngll,ngll,2) carries
    float* __restrict__ mu_dx, float* __restrict__ mu_dz,   // displ-memory adjoints
    float inv_dt,
    float* __restrict__ sc_now_x, float* __restrict__ sc_now_z,
    float* __restrict__ sc_prev_x, float* __restrict__ sc_prev_z,
    float* __restrict__ gLam, float* __restrict__ gMu, float* __restrict__ gL2m,
    float* __restrict__ gWrhoJ, int ngll, int E, int ng);

// grad_inv_M via the leapfrog identity: dt^2 (b - Acc) = u^{n+1} - 2u^n + u^{n-1}
// (pinned nodes have g = 0, so the pin discrepancy contributes nothing).
__global__ void sem2d_elastic_cpml_gradM(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ u_np1_x, const float* __restrict__ u_np1_z,
    const float* __restrict__ u_n_x,   const float* __restrict__ u_n_z,    // may be null
    const float* __restrict__ u_nm1_x, const float* __restrict__ u_nm1_z,  // may be null
    const float* __restrict__ inv_M,
    float* __restrict__ gradM, int ng);

// ================== fluid-solid coupling kernels (P4) =======================
// Interface quadrature: flat arrays of n_face*ngll GLL points with global node
// index g, weight w (w_1D*jac1D) and fluid-outward normal (nx, nz). All three
// kernels are plain gather/scatter — batch-ready by construction (index by an
// outer offset when a batch dim lands).

// S^ac_i = scatter( w * (nx*u_x[g] + nz*u_z[g]) )   (into the fluid residual)
__global__ void sem2d_ifc_ac(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ s_ac, int n_pts, int ng);

// (S^el_x, S^el_z) = scatter( w * p[g] * (nx, nz) ) with p = -chi_dd; adds into
// the solid force fields (exact discrete transpose of sem2d_ifc_ac).
__global__ void sem2d_ifc_el(
    const float* __restrict__ chi_dd,
    const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ f_x, float* __restrict__ f_z, int n_pts, int ng);

// ---- illumination (sweep.sem.illumination) ---------------------------------
// acc[b, g] += ((f_next - 2 f + f_prev) / dt^2)^2: one field's source term.
// The accumulators are fp64: a sum of squares over every step leaves the fp32
// range for the adjoint values an unnormalised residual gives.
__global__ void sem2d_illum_src(
    const float* __restrict__ f_next, const float* __restrict__ f,
    const float* __restrict__ f_prev, float inv_dt2, double* __restrict__ acc, int ng);

// acc[b, g] += (inv_M[g] lambda[b, g])^2: one field's receiver term (ms: the
// inverse mass's per-shot stride, 0 when the batch shares it)
__global__ void sem2d_illum_rec(
    const float* __restrict__ lam, const float* __restrict__ inv_M, int ms,
    double* __restrict__ acc, int ng);

// ---- the interface loads' C-PML stretching (coupling.InterfaceCPML) ---------
// One thread per (band interface point, shot). A point owns its four memories
// (two for the solid load, e^{-a t}*chi per axis; two for the fluid load,
// e^{-(a+s) t}*u per axis), so a node shared by two faces runs the same
// recursion twice -- the eager per-node memory, point for point. Coefficient
// row (IFC_PML_K floats):
//   0 w*n_x  1 w*n_z
//   2..8   solid, x stretch: A B C D c0 c1 c2   dp_x = -(A chi_dd + B chi + C chi_prev + D r_x)
//   9..15  solid, z stretch                      dp_z likewise; f_x += w n_x dp_z, f_z += w n_z dp_x
//   16..19 fluid, x: s c0 c1 c2                  s_ac -= w n_x s_x q_x + w n_z s_z q_z
//   20..23 fluid, z
// r = c0 r + c1 chi + c2 chi_prev and q = c0 q + c1 u + c2 u_prev at this
// level. The tape slice (B, n_p, 3) takes this step's (fluid, solid x, solid z)
// point loads; nullptr skips it.
#define IFC_PML_K 24

__global__ void sem2d_ifc_pml_ac(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ mem, float* __restrict__ s_ac, float* __restrict__ tape,
    int n_p, int ng);

__global__ void sem2d_ifc_pml_el(
    const float* __restrict__ chi_dd, const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ mem, float* __restrict__ f_x, float* __restrict__ f_z,
    float* __restrict__ tape, int n_p, int ng);

// out0[g] += tape[slot0], out1[g] += tape[slot1] (out1 may be null): the
// forward's point loads back into a recomputed s_ac or solid force
__global__ void sem2d_ifc_pml_tape_add(
    const float* __restrict__ tape, const long* __restrict__ gidx,
    float* __restrict__ out0, int slot0, float* __restrict__ out1, int slot1,
    int n_p, int ng);

// Adjoint of sem2d_ifc_pml_el. lambda_f = -y (y = -dt2 invM g), so
// lambda_dp_z = -w n_x y_x[g], lambda_dp_x = -w n_z y_z[g]; lmem[0..1] carry
// the memories' adjoints (Lambda^k = -D lambda_dp + c0 Lambda^(k+1)). Writes
// lchidd[g] += -(A_x l_x + A_z l_z) and leaves this point's chi^k adjoint in
// scr[0] (its own plus the chi^k part step k+1 left in scr[1]) and its
// chi^(k-1) adjoint in scr[1], for sem2d_ifc_pml_scr_add.
__global__ void sem2d_ifc_pml_el_T(
    const float* __restrict__ y_x, const float* __restrict__ y_z,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ lmem, float* __restrict__ lchidd, float* __restrict__ scr,
    int n_p, int ng);

// lchi_G[g] += scr[0]: the chi^k adjoint of the solid load, before the fluid
// combine (and, on DD, before lchi_G's exchange)
__global__ void sem2d_ifc_pml_scr_add(
    const float* __restrict__ scr, const long* __restrict__ gidx,
    float* __restrict__ lchi_G, int n_p, int ng);

// Adjoint of sem2d_ifc_pml_ac: lambda_s_ac = invM o lchidd (chi_dd = ... +
// invM s_ac); lmem[2..3] carry the memories' adjoints. The u^k adjoint goes
// to sc_now, the u^(k-1) one to sc_prev (the scatter carried to step k-1).
__global__ void sem2d_ifc_pml_ac_T(
    const float* __restrict__ lchidd, const float* __restrict__ inv_M,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ lmem,
    float* __restrict__ sc_now_x, float* __restrict__ sc_now_z,
    float* __restrict__ sc_prev_x, float* __restrict__ sc_prev_z, int n_p, int ng);

// chi_dd = kappa*w_sum + inv_M*(pat*wav[n] + s_ac)   (post-inversion fluid accel)
__global__ void sem2d_fs_chidd(
    const float* __restrict__ kappa, const float* __restrict__ w_sum,
    const float* __restrict__ inv_M, const float* __restrict__ pattern,
    const float* __restrict__ wavelet, int n, const float* __restrict__ s_ac,
    float* __restrict__ chi_dd, int ng, int wst);

// bare-path fluid accel: chi_dd = inv_M * (pat*wav[n] + s_ac - K(chi))
__global__ void sem2d_fs_chidd_bare(
    const float* __restrict__ Kchi, const float* __restrict__ inv_M,
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int n,
    const float* __restrict__ s_ac, float* __restrict__ chi_dd, int ng);

// chi_next = 2 chi - chi_prev + dt2 * chi_dd
__global__ void sem2d_fs_chi_leapfrog(
    const float* __restrict__ chi, const float* __restrict__ chi_prev,
    const float* __restrict__ chi_dd, float dt2,
    float* __restrict__ chi_next, int ng);

// 2-component Newmark with an ARBITRARY per-node force field (source pattern +
// interface term composed by the driver): u_next = 2u - u_prev + dt2 invM (f - Acc)
__global__ void sem2d_elastic_newmark_force(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const float* __restrict__ Acc_x, const float* __restrict__ Acc_z,
    const float* __restrict__ f_x, const float* __restrict__ f_z,
    const float* __restrict__ inv_M, float dt2,
    float* __restrict__ ux_next, float* __restrict__ uz_next, int ng);

// f = pattern * wav[n]   (force-field initialisation before the interface adds)
__global__ void sem2d_fs_pat_wav(
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int n,
    float* __restrict__ f, int ng, int wst);

// ================== fluid-solid backward glue kernels (P4) ==================
// out += s * a
__global__ void sem2d_fs_axpy(
    const float* __restrict__ a, float s, float* __restrict__ out, int n);

// Fluid-accel adjoint -> material grads. lchidd = lambda(chi_dd) (post-Dir).
// CPML: gKappa += w_sum*lchidd (Kchi null); bare: Kchi subtracts (w_sum null).
// gInvM += (pat*wav[k] + s_ac - Kchi)*lchidd   [chi_dd = kappa w_sum + invM(...)]
__global__ void sem2d_fs_ac_gradMK(
    const float* __restrict__ lchidd, const float* __restrict__ w_sum,
    const float* __restrict__ Kchi, const float* __restrict__ pat_chi,
    const float* __restrict__ wavelet, int k, const float* __restrict__ s_ac,
    float* __restrict__ gKappa, float* __restrict__ gInvM, int ng, int wst);

// Fluid combine, CPML flavour (P3b combine minus the grad terms, which moved
// to sem2d_fs_ac_gradMK): lchi_k = gB + 2 gA + lchi_G; psi/zeta adjoint
// recursions with the coupled gw = kappa o lambda(chi_dd).
__global__ void sem2d_fs_ac_combine_cpml(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ lchi_G,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_z,
    const float* __restrict__ lapx, const float* __restrict__ lapz,
    const float* __restrict__ gw,
    const float* __restrict__ a_x, const float* __restrict__ a_z,
    float* __restrict__ lchi_k,
    float* __restrict__ lpsi_x_n, float* __restrict__ lpsi_z_n,
    float* __restrict__ lzeta_x_n, float* __restrict__ lzeta_z_n, int ng);

// Fluid combine, bare flavour: lchi_k = gB + 2 gA + extra (extra = K^T path)
__global__ void sem2d_fs_combine3(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ extra, float* __restrict__ out, int ng);

// lambda^n = 2 g - g_prev + adj_src + sc_now + sc_prev_carry   (per component)
__global__ void sem2d_elastic_cpml_combine(
    const float* __restrict__ g, const float* __restrict__ g_prev,
    const float* __restrict__ adj, const float* __restrict__ sc_now,
    const float* __restrict__ sc_prev_carry,
    float* __restrict__ lam_out, int ng);

} // namespace sem2d

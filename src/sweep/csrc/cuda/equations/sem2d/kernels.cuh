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
#include "../../../core/sem_input_core.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sem2d {

// ---- element stiffness: K += assemble( D^T (w_J * (1/rho) grad chi) ) --------
// One block per element, `m = ngll*ngll` threads (one per GLL node of the element).
// Dynamic shared memory carries s_chi, s_Fbeta, s_Falpha (3*m floats).
static __global__ void sem2d_acoustic_stiffness(
    const float* __restrict__ chi,
    const int*    __restrict__ ibool,
    const float* __restrict__ D,
    const float* __restrict__ inv_rho,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ K,
    int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m   = ngll * ngll;
    const int tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll;          // eta row
    const int b = tid % ngll;          // xi  col
    const int off = e * m + tid;       // global GLL offset into the (E,ngll,ngll) arrays

    extern __shared__ float sh[];
    float* s_chi = sh;                 // [m]
    float* s_Fb  = sh + m;             // [m]  F_beta
    float* s_Fa  = sh + 2 * m;         // [m]  F_alpha

    s_chi[tid] = chi[ ibool[off] ];
    __syncthreads();

    // reference derivatives at (a,b)
    float dxi = 0.f, deta = 0.f;
    for (int g = 0; g < ngll; ++g) {
        dxi  += D[b * ngll + g] * s_chi[a * ngll + g];
        deta += D[a * ngll + g] * s_chi[g * ngll + b];
    }
    // physical grad via inv-Jacobian chain rule, then flux q = (1/rho) grad chi
    const float dchidx = ixx[off] * dxi + iex[off] * deta;
    const float dchidz = ixz[off] * dxi + iez[off] * deta;
    const float qx = inv_rho[off] * dchidx;
    const float qz = inv_rho[off] * dchidz;
    // weights folded onto the flux BEFORE the D^T contraction (matches eager
    // F_beta = w_J*(...), F_alpha = w_J*(...))
    s_Fb[tid] = wJ[off] * (qx * ixx[off] + qz * ixz[off]);
    s_Fa[tid] = wJ[off] * (qx * iex[off] + qz * iez[off]);
    __syncthreads();

    float r = 0.f;
    for (int g = 0; g < ngll; ++g) {
        r += D[g * ngll + b] * s_Fb[a * ngll + g];   // sum_d D[d,b] Fbeta[a,d]
        r += D[g * ngll + a] * s_Fa[g * ngll + b];   // sum_g D[g,a] Falpha[g,b]
    }
    r *= mask_e[e];
    atomicAdd(&K[ ibool[off] ], r);
}

// ---- Newmark 2nd-order displacement update (bare, no CPML) --------------------
//   chi_next = 2*chi - chi_prev + dt2 * inv_M * (-K + pattern * wavelet[n])
static __global__ void sem2d_acoustic_newmark(
    const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const float* __restrict__ K,
    const float* __restrict__ pattern,
    const float* __restrict__ inv_M,
    const float* __restrict__ wavelet,
    int n, float dt2,
    float* __restrict__ chi_next,
    int n_global)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_global) return;
    const float src = pattern[i] * wavelet[n];
    chi_next[i] = 2.f * chi[i] - chi_prev[i] + dt2 * inv_M[i] * (src - K[i]);
}

// ---- Dirichlet (acoustic free surface): pin chi = 0 on the listed nodes -------
static __global__ void sem2d_dirichlet_fill(
    float* __restrict__ chi_next, const long* __restrict__ didx, int k)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    chi_next[ didx[j] ] = 0.f;
}

// shot-batched variant (gridDim.y = B): the base form has ~20 single-shot
// call sites across the drivers, so the batched acoustic C-PML reverse march
// gets its own rather than a signature change.
static __global__ void sem2d_dirichlet_fill_b(
    float* __restrict__ f, const long* __restrict__ didx, int k, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    f[ didx[j] + static_cast<size_t>(blockIdx.y) * ng ] = 0.f;
}

// ---- receiver record: seis[r,n] = sum_m field[rec_G[r,m]] * rec_W[r,m] --------
static __global__ void sem2d_record(
    const float* __restrict__ field,
    const long*  __restrict__ rec_G,
    const float* __restrict__ rec_W,
    float* __restrict__ seis,
    int n, int nt, int n_rec, int mgll, int ng)
{
    // shot batch: gridDim.y = B; a 1-D launch (blockIdx.y == 0) is unchanged.
    // rec_G / rec_W are per-mesh (same receivers for every shot in the batch).
    const int bat = blockIdx.y;
    field += static_cast<size_t>(bat) * ng;
    seis  += static_cast<size_t>(bat) * n_rec * nt;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rec) return;
    float acc = 0.f;
    const int base = r * mgll;
    for (int j = 0; j < mgll; ++j)
        acc += field[ rec_G[base + j] ] * rec_W[base + j];
    seis[r * nt + n] = acc;
}

// ======================= backward (adjoint) kernels ==========================
// Adjoint-state gradient of AcousticPotentialSEM2D.step w.r.t. inv_M, inv_rho,
// wavelet. Reverse leap-frog (K self-adjoint; the transpose applies inv_M BEFORE
// the stiffness, i.e. K(inv_M o a)); imaging correlation of forward field u^{k}
// and the adjoint-derived field y = -dt2 inv_M o a. See backward.cu for the loop.

// out[i] = s * a[i] * b[i]   (e.g. inv_M o a, or the -dt2 inv_M o a field)
static __global__ void sem2d_scale2(const float* __restrict__ a, const float* __restrict__ b,
                             float s, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s * a[i] * b[i];
}

// Shot-batched variants (gridDim.y = B) of three kernels shared by many
// drivers. Separate kernels rather than signature changes: the base forms
// have ~25 single-shot call sites across the elastic / fluid-solid / plain
// acoustic drivers, and only the batched acoustic C-PML path needs these.
// as_/msn are the stride of the possibly-broadcast operand (0 = shared).
static __global__ void sem2d_scale2_b(const float* __restrict__ a,
                             const float* __restrict__ b, float s,
                             float* __restrict__ out, int n, int as_) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t fo = static_cast<size_t>(blockIdx.y) * n;
    if (i < n) out[fo + i] = s * a[static_cast<size_t>(blockIdx.y) * as_ + i]
                               * b[fo + i];
}

// adjoint source: adj[node] += grad_syn[r, tcol] * rec_W[r,j]  (R^T of the record)
static __global__ void sem2d_adjoint_source(
    const float* __restrict__ grad_syn, const long* __restrict__ rec_G,
    const float* __restrict__ rec_W, float* __restrict__ adj,
    int tcol, int nt, int n_rec, int mgll) {
    const int rj = blockIdx.x * blockDim.x + threadIdx.x;
    if (rj >= n_rec * mgll) return;
    const int r = rj / mgll;
    atomicAdd(&adj[ rec_G[rj] ], grad_syn[r * nt + tcol] * rec_W[rj]);
}

// batched adjoint source (see the _b note above): grad_syn is (B, nrec, nt),
// the adjoint field advances by ng per shot; receivers are per-mesh.
static __global__ void sem2d_adjoint_source_b(
    const float* __restrict__ grad_syn, const long* __restrict__ rec_G,
    const float* __restrict__ rec_W, float* __restrict__ adj,
    int tcol, int nt, int n_rec, int mgll, int ng) {
    const int rj = blockIdx.x * blockDim.x + threadIdx.x;
    if (rj >= n_rec * mgll) return;
    const int r = rj / mgll;
    const int bat = blockIdx.y;
    grad_syn += static_cast<size_t>(bat) * n_rec * nt;
    adj += static_cast<size_t>(bat) * ng;
    atomicAdd(&adj[ rec_G[rj] ], grad_syn[r * nt + tcol] * rec_W[rj]);
}

// reverse leap-frog: ubar[i] = 2 a1[i] - a2[i] - dt2 Ktmp[i] + adj[i]
static __global__ void sem2d_adj_leapfrog(
    const float* __restrict__ a1, const float* __restrict__ a2,
    const float* __restrict__ Ktmp, const float* __restrict__ adj,
    float dt2, float* __restrict__ ubar, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ubar[i] = 2.f * a1[i] - a2[i] - dt2 * Ktmp[i] + adj[i];
}

// grad_inv_M[i] += dt2 * a_m[i] * (pattern[i]*wavelet[k] - Ku[i])
static __global__ void sem2d_accum_gradM(
    const float* __restrict__ a_m, const float* __restrict__ pattern,
    const float* __restrict__ Ku, const float* __restrict__ wavelet, int k,
    float dt2, float* __restrict__ gradM, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // BATCH: non-atomic accumulation into a batch-SHARED gradient buffer.
    // Under B > 1 this is a lost-update race -- needs a node index + atomicAdd.
    if (i < n) gradM[i] += dt2 * a_m[i] * (pattern[i] * wavelet[k] - Ku[i]);
}

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
static __global__ void sem2d_accum_gradW(
    const float* __restrict__ pattern, const float* __restrict__ inv_M,
    const float* __restrict__ a_m, float dt2, int k,
    float* __restrict__ gradW, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // every thread must reach the reduction (shuffle masks + __syncthreads)
    const float v = (i < n) ? dt2 * pattern[i] * inv_M[i] * a_m[i] : 0.0f;
    const float bs = sem2d_block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&gradW[k], bs);
}

// batched wavelet gradient (see the _b note above): gradW is (B, nt), the
// pattern/field advance by n per shot, inv_M by msn (0 = shared model).
static __global__ void sem2d_accum_gradW_b(
    const float* __restrict__ pattern, const float* __restrict__ inv_M,
    const float* __restrict__ a_m, float dt2, int k,
    float* __restrict__ gradW, int n, int nt, int msn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int bat = blockIdx.y;
    pattern += static_cast<size_t>(bat) * n;
    a_m += static_cast<size_t>(bat) * n;
    inv_M += static_cast<size_t>(bat) * msn;
    const float v = (i < n) ? dt2 * pattern[i] * inv_M[i] * a_m[i] : 0.0f;
    const float bsum = sem2d_block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&gradW[static_cast<size_t>(bat) * nt + k], bsum);
}

// imaging: grad_inv_rho[e,p] += mask_e[e] * w_J[e,p] * (grad u . grad y) at (e,p),
// with u = saved forward field u^{k}, y = -dt2 inv_M o a_m. One block per element.
static __global__ void sem2d_imaging_invrho(
    const float* __restrict__ u, const float* __restrict__ y,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e, float* __restrict__ grad_inv_rho,
    int ngll, int E) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;

    extern __shared__ float sh[];
    float* s_u = sh;          // [m]
    float* s_y = sh + m;      // [m]
    s_u[tid] = u[ ibool[off] ];
    s_y[tid] = y[ ibool[off] ];
    __syncthreads();

    float dxi_u = 0.f, deta_u = 0.f, dxi_y = 0.f, deta_y = 0.f;
    for (int g = 0; g < ngll; ++g) {
        dxi_u  += D[b * ngll + g] * s_u[a * ngll + g];
        deta_u += D[a * ngll + g] * s_u[g * ngll + b];
        dxi_y  += D[b * ngll + g] * s_y[a * ngll + g];
        deta_y += D[a * ngll + g] * s_y[g * ngll + b];
    }
    const float gux = ixx[off] * dxi_u + iex[off] * deta_u;
    const float guz = ixz[off] * dxi_u + iez[off] * deta_u;
    const float gyx = ixx[off] * dxi_y + iex[off] * deta_y;
    const float gyz = ixz[off] * dxi_y + iez[off] * deta_y;
    grad_inv_rho[off] += mask_e[e] * wJ[off] * (gux * gyx + guz * gyz);
}

// ======================= elastic forward kernels ============================
// Reproduce ElasticSEM2DDeformed.stiffness_apply / _assemble_stress (bare, no
// CPML, so sigma_zx == sigma_xz). 2-component displacement (u_x, u_z). One block
// per element, m = ngll*ngll threads; shared: s_ux, s_uz (gather) + 4 flux arrays.
static __global__ void sem2d_elastic_stiffness(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ Kux, float* __restrict__ Kuz, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;

    extern __shared__ float sh[];
    float* s_ux  = sh;              float* s_uz  = sh + m;
    float* s_Fxb = sh + 2 * m;      float* s_Fxa = sh + 3 * m;
    float* s_Fzb = sh + 4 * m;      float* s_Fza = sh + 5 * m;
    s_ux[tid] = ux[ ibool[off] ];
    s_uz[tid] = uz[ ibool[off] ];
    __syncthreads();

    float dux_dxi = 0.f, dux_deta = 0.f, duz_dxi = 0.f, duz_deta = 0.f;
    for (int g = 0; g < ngll; ++g) {
        const float db = D[b * ngll + g], da = D[a * ngll + g];
        dux_dxi  += db * s_ux[a * ngll + g];  dux_deta += da * s_ux[g * ngll + b];
        duz_dxi  += db * s_uz[a * ngll + g];  duz_deta += da * s_uz[g * ngll + b];
    }
    const float ex = ixx[off], ez = ixz[off], nx = iex[off], nz = iez[off];
    const float dux_dx = ex * dux_dxi + nx * dux_deta;
    const float dux_dz = ez * dux_dxi + nz * dux_deta;
    const float duz_dx = ex * duz_dxi + nx * duz_deta;
    const float duz_dz = ez * duz_dxi + nz * duz_deta;
    const float sxx = lam2mu[off] * dux_dx + lam[off] * duz_dz;
    const float szz = lam[off] * dux_dx + lam2mu[off] * duz_dz;
    const float sxz = mu[off] * (dux_dz + duz_dx);
    const float w = wJ[off];
    s_Fxb[tid] = w * (sxx * ex + sxz * ez);   // F_x_beta
    s_Fxa[tid] = w * (sxx * nx + sxz * nz);   // F_x_alpha
    s_Fzb[tid] = w * (sxz * ex + szz * ez);   // F_z_beta
    s_Fza[tid] = w * (sxz * nx + szz * nz);   // F_z_alpha
    __syncthreads();

    float rx = 0.f, rz = 0.f;
    for (int g = 0; g < ngll; ++g) {
        const float dgb = D[g * ngll + b], dga = D[g * ngll + a];
        rx += dgb * s_Fxb[a * ngll + g] + dga * s_Fxa[g * ngll + b];
        rz += dgb * s_Fzb[a * ngll + g] + dga * s_Fza[g * ngll + b];
    }
    const float mk = mask_e[e];
    atomicAdd(&Kux[ ibool[off] ], mk * rx);
    atomicAdd(&Kuz[ ibool[off] ], mk * rz);
}

// 2-component Newmark: u?_next = 2 u? - u?_prev + dt2 inv_M (pattern? * wav[n] - K?)
static __global__ void sem2d_elastic_newmark(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const float* __restrict__ Kux, const float* __restrict__ Kuz,
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ inv_M, const float* __restrict__ wavelet,
    int n, float dt2, float* __restrict__ ux_next, float* __restrict__ uz_next,
    int n_global, int wst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // shot batch: gridDim.y = B (shared inv_M; fields, patterns and -- via
    // wst -- the wavelet advance). A 1-D launch is unchanged.
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n_global;
      ux += fo; uz += fo; ux_prev += fo; uz_prev += fo; Kux += fo; Kuz += fo;
      pat_x += fo; pat_z += fo; ux_next += fo; uz_next += fo;
      wavelet += static_cast<size_t>(blockIdx.y) * wst; }
    if (i >= n_global) return;
    const float wv = wavelet[n], im = inv_M[i];
    ux_next[i] = 2.f * ux[i] - ux_prev[i] + dt2 * im * (pat_x[i] * wv - Kux[i]);
    uz_next[i] = 2.f * uz[i] - uz_prev[i] + dt2 * im * (pat_z[i] * wv - Kuz[i]);
}

// ======================= elastic backward kernels ===========================
// grad_inv_M[i] += dt2 (ax(pat_x*wav - Kux) + az(pat_z*wav - Kuz))  (both components)
static __global__ void sem2d_elastic_accum_gradM(
    const float* __restrict__ ax, const float* __restrict__ az,
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ Kux, const float* __restrict__ Kuz,
    const float* __restrict__ wavelet, int k, float dt2,
    float* __restrict__ gradM, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float wv = wavelet[k];
    // BATCH: non-atomic accumulation into a batch-SHARED gradient buffer.
    // Under B > 1 this is a lost-update race -- needs a node index + atomicAdd.
    gradM[i] += dt2 * (ax[i] * (pat_x[i] * wv - Kux[i]) + az[i] * (pat_z[i] * wv - Kuz[i]));
}

// grad_wavelet[k] += dt2 sum_i inv_M[i](pat_x[i]*ax[i] + pat_z[i]*az[i])
static __global__ void sem2d_elastic_accum_gradW(
    const float* __restrict__ pat_x, const float* __restrict__ pat_z,
    const float* __restrict__ inv_M, const float* __restrict__ ax,
    const float* __restrict__ az, float dt2, int k,
    float* __restrict__ gradW, int n, int ntW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const int bat = blockIdx.y;
      const size_t fo = static_cast<size_t>(bat) * n;
      pat_x += fo; pat_z += fo; ax += fo; az += fo;
      gradW += static_cast<size_t>(bat) * ntW; }
    const float v = (i < n)
        ? dt2 * inv_M[i] * (pat_x[i] * ax[i] + pat_z[i] * az[i]) : 0.0f;
    const float bs = sem2d_block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&gradW[k], bs);
}

// elastic imaging: grad_{lam2mu,lam,mu}[e,p] += mask_e w_J * (...), from the phys
// gradients of the forward u=(ux,uz) and the adjoint-derived y=(yx,yz). One block
// per element; shared: s_ux,s_uz,s_yx,s_yz (4m).
static __global__ void sem2d_elastic_imaging(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ yx, const float* __restrict__ yz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ grad_lam2mu, float* __restrict__ grad_lam,
    float* __restrict__ grad_mu, int ngll, int E) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    extern __shared__ float sh[];
    float* s_ux = sh; float* s_uz = sh + m; float* s_yx = sh + 2 * m; float* s_yz = sh + 3 * m;
    s_ux[tid] = ux[ ibool[off] ]; s_uz[tid] = uz[ ibool[off] ];
    s_yx[tid] = yx[ ibool[off] ]; s_yz[tid] = yz[ ibool[off] ];
    __syncthreads();
    float uxxi=0.f, uxet=0.f, uzxi=0.f, uzet=0.f, yxxi=0.f, yxet=0.f, yzxi=0.f, yzet=0.f;
    for (int g = 0; g < ngll; ++g) {
        const float db = D[b * ngll + g], da = D[a * ngll + g];
        uxxi += db * s_ux[a*ngll+g]; uxet += da * s_ux[g*ngll+b];
        uzxi += db * s_uz[a*ngll+g]; uzet += da * s_uz[g*ngll+b];
        yxxi += db * s_yx[a*ngll+g]; yxet += da * s_yx[g*ngll+b];
        yzxi += db * s_yz[a*ngll+g]; yzet += da * s_yz[g*ngll+b];
    }
    const float ex = ixx[off], ez = ixz[off], nx = iex[off], nz = iez[off];
    const float uxx = ex*uxxi + nx*uxet, uxz = ez*uxxi + nz*uxet;   // dux/dx, dux/dz
    const float uzx = ex*uzxi + nx*uzet, uzz = ez*uzxi + nz*uzet;   // duz/dx, duz/dz
    const float yxx = ex*yxxi + nx*yxet, yxz = ez*yxxi + nz*yxet;
    const float yzx = ex*yzxi + nx*yzet, yzz = ez*yzxi + nz*yzet;
    const float w = mask_e[e] * wJ[off];
    grad_lam2mu[off] += w * (uxx * yxx + uzz * yzz);
    grad_lam[off]    += w * (uzz * yxx + uxx * yzz);
    grad_mu[off]     += w * (uxz + uzx) * (yxz + yzx);
}

// out[i] = s * in[i]  (scalar scale; used for gw = dt2*kappa*gA via scale2, and neg)
static __global__ void sem2d_scale1(const float* __restrict__ in, float s,
                                    float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n;
      in += fo; out += fo; }
    if (i < n) out[i] = s * in[i];
}

// ================== acoustic CPML forward kernels ============================
// Port of AcousticPotentialSEM2D._cpml_grads / step_cpml (Komatitsch-Martin
// double-convolution, potential flavour). See acoustic_potential2d.py.

// _cpml_grads: from (chi, apsi_x=a_x*psi_x, apsi_z=a_z*psi_z) build 6 assembled
// global fields: gx,gz (= wJ*(1/rho)dchi/d{x,z}), kx,kz (D^T weak-div of that
// flux), gax,gaz (= wJ*d(apsi_x)/dx, wJ*d(apsi_z)/dz). One block per element.
static __global__ void sem2d_acoustic_cpml_grads(
    const float* __restrict__ chi, const float* __restrict__ apsi_x,
    const float* __restrict__ apsi_z, const int* __restrict__ ibool,
    const float* __restrict__ D, const float* __restrict__ inv_rho,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ gx, float* __restrict__ gz,
    float* __restrict__ kx, float* __restrict__ kz,
    float* __restrict__ gax, float* __restrict__ gaz, int ngll, int E) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    extern __shared__ float sh[];
    float* s_c  = sh;            float* s_px = sh + m;      float* s_pz = sh + 2 * m;
    float* s_kxb = sh + 3 * m;   float* s_kxa = sh + 4 * m;
    float* s_kzb = sh + 5 * m;   float* s_kza = sh + 6 * m;
    const int g0 = ibool[off];
    s_c[tid] = chi[g0]; s_px[tid] = apsi_x[g0]; s_pz[tid] = apsi_z[g0];
    __syncthreads();

    float cxi=0.f, cet=0.f, pxxi=0.f, pxet=0.f, pzxi=0.f, pzet=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float db = D[b*ngll+gi], da = D[a*ngll+gi];
        cxi  += db * s_c[a*ngll+gi];  cet  += da * s_c[gi*ngll+b];
        pxxi += db * s_px[a*ngll+gi]; pxet += da * s_px[gi*ngll+b];
        pzxi += db * s_pz[a*ngll+gi]; pzet += da * s_pz[gi*ngll+b];
    }
    const float ex=ixx[off], ez=ixz[off], nx=iex[off], nz=iez[off], w=wJ[off], ir=inv_rho[off];
    const float qx = ir * (ex*cxi + nx*cet);      // (1/rho) dchi/dx
    const float qz = ir * (ez*cxi + nz*cet);      // (1/rho) dchi/dz
    const float mk = mask_e[e];
    atomicAdd(&gx[g0],  mk * w * qx);
    atomicAdd(&gz[g0],  mk * w * qz);
    atomicAdd(&gax[g0], mk * w * (ex*pxxi + nx*pxet));   // d(apsi_x)/dx
    atomicAdd(&gaz[g0], mk * w * (ez*pzxi + nz*pzet));   // d(apsi_z)/dz
    s_kxb[tid] = w * ex * qx;  s_kxa[tid] = w * nx * qx;
    s_kzb[tid] = w * ez * qz;  s_kza[tid] = w * nz * qz;
    __syncthreads();
    float rkx=0.f, rkz=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float dgb = D[gi*ngll+b], dga = D[gi*ngll+a];
        rkx += dgb * s_kxb[a*ngll+gi] + dga * s_kxa[gi*ngll+b];
        rkz += dgb * s_kzb[a*ngll+gi] + dga * s_kza[gi*ngll+b];
    }
    atomicAdd(&kx[g0], mk * rkx);
    atomicAdd(&kz[g0], mk * rkz);
}

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
static __global__ void sem2d_acoustic_cpml_grads_f(
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
    int ng, int ms) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    // shot batch: gridDim.y = B (a 1-D launch, blockIdx.y == 0, is unchanged).
    // Fields/scratch advance by ng per shot; the material stride ms is 0 (one
    // model broadcast over the batch) or m*E (per-shot models). Geometry
    // (ibool, D, metrics, mask_e) and the C-PML profile a_* are per-mesh.
    {
        const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
        chi += fo; psi_x += fo; psi_z += fo;
        gx += fo; gz += fo; kx += fo; kz += fo; gax += fo; gaz += fo;
        inv_rho += static_cast<size_t>(blockIdx.y) * ms;
    }
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    extern __shared__ float sh[];
    float* s_c  = sh;            float* s_px = sh + m;      float* s_pz = sh + 2 * m;
    float* s_kxb = sh + 3 * m;   float* s_kxa = sh + 4 * m;
    float* s_kzb = sh + 5 * m;   float* s_kza = sh + 6 * m;
    const int g0 = ibool[off];
    s_c[tid] = chi[g0];
    s_px[tid] = a_x[g0] * psi_x[g0];
    s_pz[tid] = a_z[g0] * psi_z[g0];
    __syncthreads();

    float cxi=0.f, cet=0.f, pxxi=0.f, pxet=0.f, pzxi=0.f, pzet=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float db = D[b*ngll+gi], da = D[a*ngll+gi];
        cxi  += db * s_c[a*ngll+gi];  cet  += da * s_c[gi*ngll+b];
        pxxi += db * s_px[a*ngll+gi]; pxet += da * s_px[gi*ngll+b];
        pzxi += db * s_pz[a*ngll+gi]; pzet += da * s_pz[gi*ngll+b];
    }
    const float ex=ixx[off], ez=ixz[off], nx=iex[off], nz=iez[off], ir=inv_rho[off];
    const float w = w1d[a] * w1d[b] / (ex*nz - ez*nx);   // = w_J[off]
    const float qx = ir * (ex*cxi + nx*cet);
    const float qz = ir * (ez*cxi + nz*cet);
    const float mk = mask_e[e];
    atomicAdd(&gx[g0],  mk * w * qx);
    atomicAdd(&gz[g0],  mk * w * qz);
    atomicAdd(&gax[g0], mk * w * (ex*pxxi + nx*pxet));
    atomicAdd(&gaz[g0], mk * w * (ez*pzxi + nz*pzet));
    s_kxb[tid] = w * ex * qx;  s_kxa[tid] = w * nx * qx;
    s_kzb[tid] = w * ez * qz;  s_kza[tid] = w * nz * qz;
    __syncthreads();
    float rkx=0.f, rkz=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float dgb = D[gi*ngll+b], dga = D[gi*ngll+a];
        rkx += dgb * s_kxb[a*ngll+gi] + dga * s_kxa[gi*ngll+b];
        rkz += dgb * s_kzb[a*ngll+gi] + dga * s_kza[gi*ngll+b];
    }
    atomicAdd(&kx[g0], mk * rkx);
    atomicAdd(&kz[g0], mk * rkz);
}

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
#define SEM2D_LAUNCH_WS(wsave, K, grid, block, shm, stream, ...)               \
    do {                                                                       \
        if (wsave) K<true><<<grid, block, shm, stream>>>(__VA_ARGS__);         \
        else       K<false><<<grid, block, shm, stream>>>(__VA_ARGS__);        \
    } while (0)

template <bool WSAVE>
static __global__ void sem2d_acoustic_cpml_step_newmark_f(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ kz,
    const float* __restrict__ gax, const float* __restrict__ gaz,
    const float* __restrict__ inv_Mg,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx_dx,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz_dz,
    const float* __restrict__ psi_x, const float* __restrict__ psi_z,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_z,
    const float* __restrict__ chi, const float* __restrict__ chi_prev,
    const float* __restrict__ kappa_node, const float* __restrict__ inv_M,
    const float* __restrict__ pattern, const float* __restrict__ wavelet,
    int n, float dt2,
    float* __restrict__ chi_next,
    float* __restrict__ psi_x_n, float* __restrict__ psi_z_n,
    float* __restrict__ zeta_x_n, float* __restrict__ zeta_z_n,
    float* __restrict__ s_chi, float* __restrict__ s_px, float* __restrict__ s_pz,
    float* __restrict__ s_zx, float* __restrict__ s_zz,   // save slices, may be null
    float* __restrict__ s_ws,        // the w_sum tape row n (sem_types.h), may be null
    int ng, int msn, int wst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // shot batch: gridDim.y = B. Fields, pattern and the save slices advance
    // by ng per shot (the caller already offsets the saves to step n of a
    // (nt, B, ng) history). msn is the node-material stride for kappa_node /
    // inv_M / inv_Mg: 0 = one model broadcast, ng = per-shot models. ws is the
    // wavelet stride: 0 = one wavelet shared by the batch, nt = per-shot rows.
    // The C-PML profile arrays a_*/b_*/db*_d* are per-mesh, never per-shot.
    {
        const int bat = blockIdx.y;
        const size_t fo = static_cast<size_t>(bat) * ng;
        gx += fo; gz += fo; kx += fo; kz += fo; gax += fo; gaz += fo;
        psi_x += fo; psi_z += fo; zeta_x += fo; zeta_z += fo;
        chi += fo; chi_prev += fo; pattern += fo;
        chi_next += fo; psi_x_n += fo; psi_z_n += fo;
        zeta_x_n += fo; zeta_z_n += fo;
        if (WSAVE) {                     // the chi tape and the w_sum row
            if (s_chi) s_chi += fo;
            if (s_ws) s_ws += fo;
        } else if (s_chi) {              // the five state tapes
            s_chi += fo; s_px += fo; s_pz += fo; s_zx += fo; s_zz += fo;
        }
        // inv_Mg is the GEOMETRIC mass -- per-mesh, never per-shot -- so it
        // takes no stride even when the material arrays do (Bm == B).
        const size_t mo = static_cast<size_t>(bat) * msn;
        kappa_node += mo; inv_M += mo;
        wavelet += static_cast<size_t>(bat) * wst;
    }
    const float bxv = b_x[i], bzv = b_z[i], dbxv = dbx_dx[i], dbzv = dbz_dz[i];
    if (bxv == 0.f && bzv == 0.f && dbxv == 0.f && dbzv == 0.f) {
        const float imv = inv_Mg[i];
        const float dx = -kx[i]*imv, dz = -kz[i]*imv;
        const float t_x = dx + gax[i]*imv, t_z = dz + gaz[i]*imv;
        const float cnv = 2.f*chi[i] - chi_prev[i]
                        + dt2 * (kappa_node[i]*(t_x + t_z)
                                 + inv_M[i]*pattern[i]*wavelet[n]);
        chi_next[i] = cnv;
        // psi/zeta stay zero here; both rotation buffers start at zero and only
        // absorbing nodes are ever written, so skipping the store preserves them.
        if (s_chi) s_chi[i] = cnv;
        if (WSAVE && s_ws) s_ws[i] = t_x + t_z;
        return;
    }
    const float im = inv_Mg[i];
    const float qx = gx[i]*im, qz = gz[i]*im;
    const float dvx = -kx[i]*im, dvz = -kz[i]*im;
    const float gapx = gax[i]*im, gapz = gaz[i]*im;
    const float opbx = 1.f + b_x[i], opbz = 1.f + b_z[i];
    const float tx = opbx*dvx + dbx_dx[i]*qx + gapx;
    const float tz = opbz*dvz + dbz_dz[i]*qz + gapz;
    const float ws = opbx*tx + a_x[i]*zeta_x[i] + opbz*tz + a_z[i]*zeta_z[i];
    const float pxn = b_x[i]*qx + a_x[i]*psi_x[i];
    const float pzn = b_z[i]*qz + a_z[i]*psi_z[i];
    const float zxn = b_x[i]*tx + a_x[i]*zeta_x[i];
    const float zzn = b_z[i]*tz + a_z[i]*zeta_z[i];
    const float cn = 2.f*chi[i] - chi_prev[i]
                   + dt2 * (kappa_node[i]*ws + inv_M[i]*pattern[i]*wavelet[n]);
    chi_next[i] = cn;
    psi_x_n[i] = pxn;  psi_z_n[i] = pzn;
    zeta_x_n[i] = zxn; zeta_z_n[i] = zzn;
    if (WSAVE) {
        if (s_chi) s_chi[i] = cn;
        if (s_ws) s_ws[i] = ws;
    } else if (s_chi) {
        s_chi[i] = cn;
        s_px[i] = pxn;  s_pz[i] = pzn;
        s_zx[i] = zxn;  s_zz[i] = zzn;
    }
}

// zero 5 fields (and, when saving, their save slices) on the dirichlet nodes.
static __global__ void sem2d_dirichlet_fill5(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ f3, float* __restrict__ f4,
    float* __restrict__ s0, float* __restrict__ s1, float* __restrict__ s2,
    float* __restrict__ s3, float* __restrict__ s4,   // may be null
    const long* __restrict__ didx, int k, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    // shot batch: gridDim.y = B; the dirichlet node set is per-mesh.
    const size_t o = static_cast<size_t>(blockIdx.y) * ng;
    const long i = didx[j] + o;
    f0[i] = 0.f; f1[i] = 0.f; f2[i] = 0.f; f3[i] = 0.f; f4[i] = 0.f;
    if (s0) s0[i] = 0.f;
    if (s1) { s1[i] = 0.f; s2[i] = 0.f; s3[i] = 0.f; s4[i] = 0.f; }
}

// backward combine with the gB rotation folded in (gB_out = -gA).
static __global__ void sem2d_acoustic_cpml_combine_f(
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
    float* __restrict__ gB_out, float dt2, int ng, int msn, int wst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // shot batch: gridDim.y = B. msn/ws as in step_newmark_f. The gradient
    // outputs are PER-SHOT (B, ng) so the += below stays race-free; a shared
    // model's gradient is the caller's sum over the leading axis.
    {
        const int bat = blockIdx.y;
        const size_t fo = static_cast<size_t>(bat) * ng;
        gA += fo; gB += fo; lchi_G += fo; w_sum += fo; pattern += fo;
        lpsi_x += fo; lpsi_z += fo; lzeta_x += fo; lzeta_z += fo;
        lapx += fo; lapz += fo; gw += fo;
        lchi_k += fo; lpsi_x_n += fo; lpsi_z_n += fo;
        lzeta_x_n += fo; lzeta_z_n += fo;
        grad_kappa += fo; grad_inv_M += fo; gB_out += fo;
        kappa += static_cast<size_t>(bat) * msn;
        wavelet += static_cast<size_t>(bat) * wst;
    }
    const float ga = gA[i];
    grad_kappa[i]  += dt2 * w_sum[i] * ga;
    grad_inv_M[i]  += dt2 * pattern[i] * wavelet[k] * ga;
    lchi_k[i]   = gB[i] + 2.f*ga + lchi_G[i];
    lpsi_x_n[i] = a_x[i] * (lpsi_x[i] + lapx[i]);
    lpsi_z_n[i] = a_z[i] * (lpsi_z[i] + lapz[i]);
    lzeta_x_n[i] = a_x[i] * (lzeta_x[i] + gw[i]);
    lzeta_z_n[i] = a_z[i] * (lzeta_z[i] + gw[i]);
    gB_out[i] = -ga;
}

// boundary-saving band copy: gather field values at the band nodes into a
// compact history row, or scatter a saved row back into a full field.
// Shot batch: gridDim.y = B; the field advances by ng per shot and the
// history row by nband (a (nt, B, nband) layout whose step base the host
// helper computes). A 1-D launch (ng unused, blockIdx.y == 0) is unchanged.
static __global__ void sem2d_band_gather(
    const float* __restrict__ field, const long* __restrict__ band_idx,
    float* __restrict__ out, int nband, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    field += static_cast<size_t>(blockIdx.y) * ng;
    out += static_cast<size_t>(blockIdx.y) * nband;
    if (j < nband) out[j] = field[band_idx[j]];
}

static __global__ void sem2d_band_scatter(
    const float* __restrict__ vals, const long* __restrict__ band_idx,
    float* __restrict__ field, int nband, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    vals += static_cast<size_t>(blockIdx.y) * nband;
    field += static_cast<size_t>(blockIdx.y) * ng;
    if (j < nband) field[band_idx[j]] = vals[j];
}

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

static __device__ __forceinline__ float sem2d_block_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
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
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;   // meaningful on threadIdx.x == 0
}

// One block (per shot: gridDim.y = B), strided over the band: max |field| at
// the band nodes -> scale[shot]. The int8 scale is PER SHOT -- one shot's
// amplitude must not set another's quantisation step.
static __global__ void sem2d_band_amax(
    const float* __restrict__ field, const long* __restrict__ band_idx,
    float* __restrict__ scale, int nband, int ng) {
    field += static_cast<size_t>(blockIdx.y) * ng;
    scale += blockIdx.y;
    float v = 0.0f;
    for (int j = threadIdx.x; j < nband; j += blockDim.x)
        v = fmaxf(v, fabsf(field[band_idx[j]]));
    const float m = sem2d_block_max(v);
    if (threadIdx.x == 0) *scale = m;
}

// bf16 has no implicit conversions in device code; keep the casts in one place.
static __device__ __forceinline__ void sem_band_put(float& d, float v) { d = v; }
static __device__ __forceinline__ void sem_band_put(__nv_bfloat16& d, float v) {
    d = __float2bfloat16(v);
}
static __device__ __forceinline__ void sem_band_put(signed char& d, float v) {
    const int q = __float2int_rn(v);
    d = static_cast<signed char>(q < -127 ? -127 : (q > 127 ? 127 : q));
}
static __device__ __forceinline__ float sem_band_get(float v) { return v; }
static __device__ __forceinline__ float sem_band_get(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
static __device__ __forceinline__ float sem_band_get(signed char v) {
    return static_cast<float>(v);
}

template <typename T>
static __global__ void sem2d_band_gather_q(
    const float* __restrict__ field, const long* __restrict__ band_idx,
    T* __restrict__ out, const float* __restrict__ scale, int nband, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    field += static_cast<size_t>(blockIdx.y) * ng;
    out += static_cast<size_t>(blockIdx.y) * nband;
    if (scale) scale += blockIdx.y;                 // per-shot int8 scale
    if (j >= nband) return;
    const float v = field[band_idx[j]];
    sem_band_put(out[j], scale ? v * (127.0f / (*scale + 1e-30f)) : v);
}

template <typename T>
static __global__ void sem2d_band_scatter_q(
    const T* __restrict__ vals, const long* __restrict__ band_idx,
    float* __restrict__ field, const float* __restrict__ scale, int nband,
    int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    vals += static_cast<size_t>(blockIdx.y) * nband;
    field += static_cast<size_t>(blockIdx.y) * ng;
    if (scale) scale += blockIdx.y;                 // per-shot int8 scale
    if (j >= nband) return;
    const float v = sem_band_get(vals[j]);
    field[band_idx[j]] = scale ? v * ((*scale + 1e-30f) / 127.0f) : v;
}

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

static inline void sem2d_band_save(const float* field, const long* band_idx,
                                   SemBandHist h, long step, int nband,
                                   int tpb, cudaStream_t stream,
                                   int B = 1, int ng = 0) {
    if (!h.data || nband <= 0) return;
    const int nb = (nband + tpb - 1) / tpb;
    const dim3 g(nb, B);
    const size_t off = static_cast<size_t>(step) * nband * B;
    float* sc = nullptr;
    if (h.kind == SEM_BAND_I8) {
        sc = h.scale + static_cast<size_t>(step) * B;
        sem2d_band_amax<<<dim3(1, B), 256, 0, stream>>>(field, band_idx, sc, nband, ng);
    }
    switch (h.kind) {
    case SEM_BAND_BF16:
        sem2d_band_gather_q<__nv_bfloat16><<<g, tpb, 0, stream>>>(
            field, band_idx, static_cast<__nv_bfloat16*>(h.data) + off,
            nullptr, nband, ng);
        break;
    case SEM_BAND_I8:
        sem2d_band_gather_q<signed char><<<g, tpb, 0, stream>>>(
            field, band_idx, static_cast<signed char*>(h.data) + off, sc, nband, ng);
        break;
    default:
        sem2d_band_gather<<<g, tpb, 0, stream>>>(
            field, band_idx, static_cast<float*>(h.data) + off, nband, ng);
    }
}

static inline void sem2d_band_load(SemBandHist h, long step, const long* band_idx,
                                   float* field, int nband, int tpb,
                                   cudaStream_t stream, int B = 1, int ng = 0) {
    if (!h.data || nband <= 0) return;
    const int nb = (nband + tpb - 1) / tpb;
    const dim3 g(nb, B);
    const size_t off = static_cast<size_t>(step) * nband * B;
    switch (h.kind) {
    case SEM_BAND_BF16:
        sem2d_band_scatter_q<__nv_bfloat16><<<g, tpb, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(h.data) + off, band_idx, field,
            nullptr, nband, ng);
        break;
    case SEM_BAND_I8:
        sem2d_band_scatter_q<signed char><<<g, tpb, 0, stream>>>(
            static_cast<const signed char*>(h.data) + off, band_idx, field,
            h.scale + static_cast<size_t>(step) * B, nband, ng);
        break;
    default:
        sem2d_band_scatter<<<g, tpb, 0, stream>>>(
            static_cast<const float*>(h.data) + off, band_idx, field, nband, ng);
    }
}

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
static __global__ void sem2d_elastic_adj_carry(
    const float* __restrict__ lam1_x, const float* __restrict__ lam1_z,
    const float* __restrict__ pin_mask,     // 1 = live node, 0 = pinned wall
    const float* __restrict__ inv_M, float neg_dt2,
    float* __restrict__ g_x, float* __restrict__ g_z,
    float* __restrict__ y_x, float* __restrict__ y_z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // shot batch: gridDim.y = B (pin_mask / inv_M are per-mesh)
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n;
      lam1_x += fo; lam1_z += fo; g_x += fo; g_z += fo; y_x += fo; y_z += fo; }
    if (i >= n) return;
    const float w = pin_mask ? pin_mask[i] : 1.0f;
    const float gx = w * lam1_x[i];
    const float gz = w * lam1_z[i];
    const float s = neg_dt2 * inv_M[i];
    g_x[i] = gx; g_z[i] = gz;
    y_x[i] = s * gx; y_z[i] = s * gz;
}

// Element-block band gather/scatter: the element-local BS histories (stretched
// derivatives, aPML brackets) are stored as m contiguous floats per element, so
// the band copy moves whole element blocks rather than single nodes.
static __global__ void sem2d_band_gather_elem(
    const float* __restrict__ field, const long* __restrict__ elem_idx,
    float* __restrict__ out, int nbe, int m, long Em) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    field += static_cast<size_t>(blockIdx.y) * Em;
    out += static_cast<size_t>(blockIdx.y) * nbe * m;
    if (j < nbe * m) {
        const int e = j / m;
        out[j] = field[elem_idx[e] * m + (j - e * m)];
    }
}

static __global__ void sem2d_band_scatter_elem(
    const float* __restrict__ vals, const long* __restrict__ elem_idx,
    float* __restrict__ field, int nbe, int m, long Em) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    vals += static_cast<size_t>(blockIdx.y) * nbe * m;
    field += static_cast<size_t>(blockIdx.y) * Em;
    if (j < nbe * m) {
        const int e = j / m;
        field[elem_idx[e] * m + (j - e * m)] = vals[j];
    }
}

// Low-precision ELEMENT rings. Same contract as the node versions above --
// one amax per (field, step, shot) for int8, none for bf16 -- but gathering
// through the (element, node) indirection the elastic histories live on.
// PER-ELEMENT amax: the C-PML memory/history buffers vary between rim
// elements, and one block per band element costs nbe extra floats per
// (field, step) -- nothing against a ring row of nbe*m.
//
// It is NOT what limits int8 on the elastic path, contrary to what this
// comment first claimed. Bisecting the ring families (emulating int8 per
// family in Python, which reproduces the real number to three digits) puts
// the WHOLE elastic int8 error in the two u NODE rings -- 2.5e-2 -- while
// all six element rings together cost 3.6e-4. Read the difference as u
// being inside the march recursion (march_fin injects it, march_asm reads
// it back to rebuild the histories and Acc) while the element histories are
// only consumed by element_T for imaging. Note the coupled fluid-solid path
// has the same u rings and shows none of it, so that reading is a plausible
// mechanism, not an established one.
static __global__ void sem2d_band_amax_elem(
    const float* __restrict__ field, const long* __restrict__ elem_idx,
    float* __restrict__ scale, int nbe, int m, long Em) {
    const int e = blockIdx.x;
    if (e >= nbe) return;
    field += static_cast<size_t>(blockIdx.y) * Em;
    scale += static_cast<size_t>(blockIdx.y) * nbe;
    const long base = elem_idx[e] * m;
    float v = 0.0f;
    for (int j = threadIdx.x; j < m; j += blockDim.x)
        v = fmaxf(v, fabsf(field[base + j]));
    const float mx = sem2d_block_max(v);
    if (threadIdx.x == 0) scale[e] = mx;
}

template <typename T>
static __global__ void sem2d_band_gather_elem_q(
    const float* __restrict__ field, const long* __restrict__ elem_idx,
    T* __restrict__ out, const float* __restrict__ scale, int nbe, int m,
    long Em) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    field += static_cast<size_t>(blockIdx.y) * Em;
    out += static_cast<size_t>(blockIdx.y) * nbe * m;
    if (scale) scale += static_cast<size_t>(blockIdx.y) * nbe;
    if (j >= nbe * m) return;
    const int e = j / m;
    const float v = field[elem_idx[e] * m + (j - e * m)];
    sem_band_put(out[j], scale ? v * (127.0f / (scale[e] + 1e-30f)) : v);
}

template <typename T>
static __global__ void sem2d_band_scatter_elem_q(
    const T* __restrict__ vals, const long* __restrict__ elem_idx,
    float* __restrict__ field, const float* __restrict__ scale, int nbe,
    int m, long Em) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    vals += static_cast<size_t>(blockIdx.y) * nbe * m;
    field += static_cast<size_t>(blockIdx.y) * Em;
    if (scale) scale += static_cast<size_t>(blockIdx.y) * nbe;
    if (j >= nbe * m) return;
    const int e = j / m;
    const float v = sem_band_get(vals[j]);
    field[elem_idx[e] * m + (j - e * m)] =
        scale ? v * ((scale[e] + 1e-30f) / 127.0f) : v;
}

static inline void sem2d_band_save_elem(const float* field,
                                        const long* elem_idx, SemBandHist h,
                                        long step, int nbe, int m, long Em,
                                        int tpb, cudaStream_t stream,
                                        int B = 1) {
    if (!h.data || nbe <= 0) return;
    const int n = nbe * m;
    const dim3 g((n + tpb - 1) / tpb, B);
    const size_t off = static_cast<size_t>(step) * n * B;
    float* sc = nullptr;
    if (h.kind == SEM_BAND_I8) {
        sc = h.scale + static_cast<size_t>(step) * B * nbe;
        sem2d_band_amax_elem<<<dim3(nbe, B), 128, 0, stream>>>(
            field, elem_idx, sc, nbe, m, Em);
    }
    switch (h.kind) {
    case SEM_BAND_BF16:
        sem2d_band_gather_elem_q<__nv_bfloat16><<<g, tpb, 0, stream>>>(
            field, elem_idx, static_cast<__nv_bfloat16*>(h.data) + off,
            nullptr, nbe, m, Em);
        break;
    case SEM_BAND_I8:
        sem2d_band_gather_elem_q<signed char><<<g, tpb, 0, stream>>>(
            field, elem_idx, static_cast<signed char*>(h.data) + off, sc,
            nbe, m, Em);
        break;
    default:
        sem2d_band_gather_elem<<<g, tpb, 0, stream>>>(
            field, elem_idx, static_cast<float*>(h.data) + off, nbe, m, Em);
    }
}

static inline void sem2d_band_load_elem(SemBandHist h, long step,
                                        const long* elem_idx, float* field,
                                        int nbe, int m, long Em, int tpb,
                                        cudaStream_t stream, int B = 1) {
    if (!h.data || nbe <= 0) return;
    const int n = nbe * m;
    const dim3 g((n + tpb - 1) / tpb, B);
    const size_t off = static_cast<size_t>(step) * n * B;
    switch (h.kind) {
    case SEM_BAND_BF16:
        sem2d_band_scatter_elem_q<__nv_bfloat16><<<g, tpb, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(h.data) + off, elem_idx, field,
            nullptr, nbe, m, Em);
        break;
    case SEM_BAND_I8:
        sem2d_band_scatter_elem_q<signed char><<<g, tpb, 0, stream>>>(
            static_cast<const signed char*>(h.data) + off, elem_idx, field,
            h.scale + static_cast<size_t>(step) * B * nbe, nbe, m, Em);
        break;
    default:
        sem2d_band_scatter_elem<<<g, tpb, 0, stream>>>(
            static_cast<const float*>(h.data) + off, elem_idx, field, nbe, m,
            Em);
    }
}

// step_cpml algebra (per global node): w_sum + psi/zeta updates.
static __global__ void sem2d_acoustic_cpml_step(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ kz,
    const float* __restrict__ gax, const float* __restrict__ gaz,
    const float* __restrict__ inv_Mg,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx_dx,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz_dz,
    const float* __restrict__ psi_x, const float* __restrict__ psi_z,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_z,
    float* __restrict__ w_sum, float* __restrict__ psi_x_n, float* __restrict__ psi_z_n,
    float* __restrict__ zeta_x_n, float* __restrict__ zeta_z_n, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n;
      gx += fo; gz += fo; kx += fo; kz += fo; gax += fo; gaz += fo;
      psi_x += fo; psi_z += fo; zeta_x += fo; zeta_z += fo;
      w_sum += fo; psi_x_n += fo; psi_z_n += fo;
      zeta_x_n += fo; zeta_z_n += fo; }
    if (i >= n) return;
    const float im = inv_Mg[i];
    const float qx = gx[i]*im, qz = gz[i]*im;
    const float dvx = -kx[i]*im, dvz = -kz[i]*im;
    const float gapx = gax[i]*im, gapz = gaz[i]*im;
    const float opbx = 1.f + b_x[i], opbz = 1.f + b_z[i];
    const float tx = opbx*dvx + dbx_dx[i]*qx + gapx;
    const float tz = opbz*dvz + dbz_dz[i]*qz + gapz;
    w_sum[i]   = opbx*tx + a_x[i]*zeta_x[i] + opbz*tz + a_z[i]*zeta_z[i];
    psi_x_n[i] = b_x[i]*qx + a_x[i]*psi_x[i];
    psi_z_n[i] = b_z[i]*qz + a_z[i]*psi_z[i];
    zeta_x_n[i] = b_x[i]*tx + a_x[i]*zeta_x[i];
    zeta_z_n[i] = b_z[i]*tz + a_z[i]*zeta_z[i];
}

// CPML Newmark: chi_next = 2 chi - chi_prev + dt2 (kappa_node*w_sum + inv_M*pattern*wav[n])
static __global__ void sem2d_acoustic_cpml_newmark(
    const float* __restrict__ chi, const float* __restrict__ chi_prev,
    const float* __restrict__ kappa_node, const float* __restrict__ w_sum,
    const float* __restrict__ inv_M, const float* __restrict__ pattern,
    const float* __restrict__ wavelet, int n, float dt2,
    float* __restrict__ chi_next, int ng, int msn, int wst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // shot batch: gridDim.y = B; msn/wst as in step_newmark_f (0 broadcast).
    {
        const int bat = blockIdx.y;
        const size_t fo = static_cast<size_t>(bat) * ng;
        chi += fo; chi_prev += fo; w_sum += fo; pattern += fo; chi_next += fo;
        const size_t mo = static_cast<size_t>(bat) * msn;
        kappa_node += mo; inv_M += mo;
        wavelet += static_cast<size_t>(bat) * wst;
    }
    chi_next[i] = 2.f*chi[i] - chi_prev[i]
                + dt2 * (kappa_node[i]*w_sum[i] + inv_M[i]*pattern[i]*wavelet[n]);
}

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
#define SEM2D_LAUNCH_WT(wt, K, grid, block, shm, stream, ...)                  \
    do {                                                                       \
        if (wt) K<true><<<grid, block, shm, stream>>>(__VA_ARGS__);            \
        else    K<false><<<grid, block, shm, stream>>>(__VA_ARGS__);           \
    } while (0)

// WTAPE: the forward saved w_sum^k (save_w_sum), so this kernel neither
// recomputes it nor accumulates grad_inv_Mg (which needs the assembled
// fields): it reads none of gx..gaz nor zeta and writes no w_sum_out. Only
// the adjoint algebra (gt/gq from gw and the carries) runs -- the same text
// in both instantiations, <false> being the pre-tape kernel verbatim.
template <bool WTAPE>
static __global__ void sem2d_acoustic_cpml_step_T(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ kz,
    const float* __restrict__ gax, const float* __restrict__ gaz,
    const float* __restrict__ inv_Mg,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx_dx,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz_dz,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_z,
    const float* __restrict__ gw,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_z,
    float* __restrict__ gbx, float* __restrict__ gbz, float* __restrict__ gbkx,
    float* __restrict__ gbkz, float* __restrict__ gbax, float* __restrict__ gbaz,
    float* __restrict__ w_sum_out, float* __restrict__ grad_inv_Mg, int ng,
    int msn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // shot batch: gridDim.y = B; msn = node-material stride (0 broadcast, ng
    // per-shot). grad_inv_Mg is per-shot (B, ng) -- see combine_f.
    {
        const int bat = blockIdx.y;
        const size_t fo = static_cast<size_t>(bat) * ng;
        if (!WTAPE) {      // the recompute's inputs/outputs: unread under the tape
            gx += fo; gz += fo; kx += fo; kz += fo; gax += fo; gaz += fo;
            zeta_x += fo; zeta_z += fo;
            w_sum_out += fo; grad_inv_Mg += fo;
        }
        gw += fo;
        lpsi_x += fo; lpsi_z += fo; lzeta_x += fo; lzeta_z += fo;
        gbx += fo; gbz += fo; gbkx += fo; gbkz += fo; gbax += fo; gbaz += fo;
        // inv_Mg: geometric, per-mesh -- no stride (msn covers materials
        // only; unused here but kept so every batched kernel reads alike).
        (void)msn;
    }
    const float bxv = b_x[i], bzv = b_z[i], dbxv = dbx_dx[i], dbzv = dbz_dz[i];
    if (bxv == 0.f && bzv == 0.f && dbxv == 0.f && dbzv == 0.f) {
        const float imv = inv_Mg[i];
        if (!WTAPE) {
            const float dx = -kx[i]*imv, dz = -kz[i]*imv;
            const float t_x = dx + gax[i]*imv, t_z = dz + gaz[i]*imv;
            w_sum_out[i] = t_x + t_z;
        }
        const float g = gw[i];
        // gtx = gtz = gw here, so gdv = ggp = gw and gq = 0 (gbx/gbz stay zero).
        gbkx[i] = -imv*g; gbkz[i] = -imv*g;
        gbax[i] =  imv*g; gbaz[i] =  imv*g;
        if (!WTAPE)
            grad_inv_Mg[i] += -kx[i]*g - kz[i]*g + gax[i]*g + gaz[i]*g;
        return;
    }
    const float im = inv_Mg[i];
    const float opbx = 1.f + b_x[i], opbz = 1.f + b_z[i];
    if (!WTAPE) {          // recompute w_sum^k from the forward assembly at k
        const float qx = gx[i]*im, qz = gz[i]*im;
        const float dvx = -kx[i]*im, dvz = -kz[i]*im;
        const float gpx = gax[i]*im, gpz = gaz[i]*im;
        const float tx = opbx*dvx + dbx_dx[i]*qx + gpx;
        const float tz = opbz*dvz + dbz_dz[i]*qz + gpz;
        w_sum_out[i] = opbx*tx + a_x[i]*zeta_x[i] + opbz*tz + a_z[i]*zeta_z[i];
    }
    // transpose: gtx from op7 (opbx*gw) + op10 (b_x*lzeta_x)
    const float gwv = gw[i];
    const float gtx = opbx*gwv + b_x[i]*lzeta_x[i];
    const float gtz = opbz*gwv + b_z[i]*lzeta_z[i];
    // op5^T/op6^T (uses full gtx) + op8^T/op9^T (b*lpsi into gq)
    const float gdvx = opbx*gtx, gdvz = opbz*gtz;
    const float gqx = dbx_dx[i]*gtx + b_x[i]*lpsi_x[i];
    const float gqz = dbz_dz[i]*gtz + b_z[i]*lpsi_z[i];
    const float ggpx = gtx, ggpz = gtz;
    // op4^T
    gbx[i] = im*gqx;   gbz[i] = im*gqz;
    gbkx[i] = -im*gdvx; gbkz[i] = -im*gdvz;
    gbax[i] = im*ggpx; gbaz[i] = im*ggpz;
    if (!WTAPE)
        grad_inv_Mg[i] += gx[i]*gqx + gz[i]*gqz - kx[i]*gdvx - kz[i]*gdvz
                        + gax[i]*ggpx + gaz[i]*ggpz;
}

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
static __global__ void sem2d_acoustic_cpml_grads_T(
    const float* __restrict__ chi, const float* __restrict__ gbx, const float* __restrict__ gbz,
    const float* __restrict__ gbkx, const float* __restrict__ gbkz,
    const float* __restrict__ gbax, const float* __restrict__ gbaz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ inv_rho, const float* __restrict__ w1d,   // 1-D GLL weights (ngll,)
    const float* __restrict__ ixx, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iez,
    const float* __restrict__ mask_e,
    float* __restrict__ lchi, float* __restrict__ lapx, float* __restrict__ lapz,
    float* __restrict__ grad_inv_rho, int ngll, int E, int ng, int ms) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    // shot batch: gridDim.y = B; ms = element-material stride for inv_rho (0
    // broadcast, m*E per-shot). grad_inv_rho is per-shot (B, E*m) always --
    // each thread owns its off, so += stays race-free across the batch.
    {
        const int bat = blockIdx.y;
        const size_t fo = static_cast<size_t>(bat) * ng;
        chi += fo; gbx += fo; gbz += fo; gbkx += fo; gbkz += fo;
        gbax += fo; gbaz += fo; lchi += fo; lapx += fo; lapz += fo;
        inv_rho += static_cast<size_t>(bat) * ms;
        grad_inv_rho += static_cast<size_t>(bat) * m * E;
    }
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    const int g0 = ibool[off];
    const float mk = mask_e[e];
    extern __shared__ float sh[];
    float* s_c   = sh;             float* s_ekx = sh + m;       float* s_ekz = sh + 2 * m;
    float* s_qxi = sh + 3 * m;     float* s_qet = sh + 4 * m;   // cov of chi ref-grads
    float* s_pxi = sh + 5 * m;     float* s_pet = sh + 6 * m;   // cov of apx ref-grads
    float* s_zxi = sh + 7 * m;     float* s_zet = sh + 8 * m;   // cov of apz ref-grads
    s_c[tid]   = chi[g0];
    s_ekx[tid] = mk * gbkx[g0];
    s_ekz[tid] = mk * gbkz[g0];
    __syncthreads();
    // ref-grads of ekx, ekz, chi at (a,b)
    float ekx_xi=0.f, ekx_et=0.f, ekz_xi=0.f, ekz_et=0.f, cxi=0.f, cet=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float db = D[b*ngll+gi], da = D[a*ngll+gi];
        ekx_xi += db*s_ekx[a*ngll+gi]; ekx_et += da*s_ekx[gi*ngll+b];
        ekz_xi += db*s_ekz[a*ngll+gi]; ekz_et += da*s_ekz[gi*ngll+b];
        cxi    += db*s_c[a*ngll+gi];   cet    += da*s_c[gi*ngll+b];
    }
    const float ex=ixx[off], ez=ixz[off], nx=iex[off], nz=iez[off], ir=inv_rho[off];
    const float w = w1d[a] * w1d[b] / (ex*nz - ez*nx);   // = w_J[off]
    // cov on q = w*egx (direct) + w*(metric . ref-grad of ekx) (from kx)
    const float cov_qx = w*mk*gbx[g0] + w*(ex*ekx_xi + nx*ekx_et);
    const float cov_qz = w*mk*gbz[g0] + w*(ez*ekz_xi + nz*ekz_et);
    // grad_inv_rho += dchi/dx * cov_qx + dchi/dz * cov_qz
    const float dchidx = ex*cxi + nx*cet, dchidz = ez*cxi + nz*cet;
    grad_inv_rho[off] += dchidx*cov_qx + dchidz*cov_qz;
    // cov on chi ref-grads (inv_rho chain): store for D^T in phase 2
    s_qxi[tid] = ir*(ex*cov_qx + ez*cov_qz);
    s_qet[tid] = ir*(nx*cov_qx + nz*cov_qz);
    // cov on apx/apz ref-grads (from gax/gaz direct)
    const float egax = w*mk*gbax[g0], egaz = w*mk*gbaz[g0];
    s_pxi[tid] = ex*egax; s_pet[tid] = nx*egax;
    s_zxi[tid] = ez*egaz; s_zet[tid] = nz*egaz;
    __syncthreads();
    // D^T of the cov ref-grads -> node adjoints of chi, apx, apz
    float lc=0.f, la_px=0.f, la_pz=0.f;
    for (int gi = 0; gi < ngll; ++gi) {
        const float dgb = D[gi*ngll+b], dga = D[gi*ngll+a];
        lc    += dgb*s_qxi[a*ngll+gi] + dga*s_qet[gi*ngll+b];
        la_px += dgb*s_pxi[a*ngll+gi] + dga*s_pet[gi*ngll+b];
        la_pz += dgb*s_zxi[a*ngll+gi] + dga*s_zet[gi*ngll+b];
    }
    atomicAdd(&lchi[g0], lc);
    atomicAdd(&lapx[g0], la_px);
    atomicAdd(&lapz[g0], la_pz);
}

// combine (node): Newmark^T grads + finalise chi/memory adjoints.
static __global__ void sem2d_acoustic_cpml_combine(
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
    float dt2, int ng) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float ga = gA[i];
    grad_kappa[i]  += dt2 * w_sum[i] * ga;
    grad_inv_M[i]  += dt2 * pattern[i] * wavelet[k] * ga;
    lchi_k[i]   = gB[i] + 2.f*ga + lchi_G[i];
    lpsi_x_n[i] = a_x[i] * (lpsi_x[i] + lapx[i]);
    lpsi_z_n[i] = a_z[i] * (lpsi_z[i] + lapz[i]);
    lzeta_x_n[i] = a_x[i] * (lzeta_x[i] + gw[i]);
    lzeta_z_n[i] = a_z[i] * (lzeta_z[i] + gw[i]);
}

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
static __global__ void sem2d_elastic_cpml_element(
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
    long save_off, int ngll, int E, int ng)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    // shot batch: gridDim.y = B (shared materials; only fields/memories/
    // histories advance). save_off is the step base; the per-shot offset
    // stacks on top for a (nt, B, E*m) layout.
    { const int bat = blockIdx.y;
      const size_t fo = static_cast<size_t>(bat) * ng;
      const size_t eo2 = static_cast<size_t>(bat) * E * m * 2;
      const size_t eo1 = static_cast<size_t>(bat) * E * m;
      ux += fo; uz += fo; ux_prev += fo; uz_prev += fo;
      Acc_x += fo; Acc_z += fo;
      r_dux_dx += eo2; r_dux_dz += eo2; r_duz_dx += eo2; r_duz_dz += eo2;
      r_displ_x += eo2; r_displ_z += eo2;
      if (save_sd_xx) { save_sd_xx += eo1; save_sd_zx += eo1;
                        save_sd_xz += eo1; save_sd_zz += eo1;
                        save_apx += eo1; save_apz += eo1; } }
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    const int o2 = off * 2;
    // Does this element carry any C-PML node? One block is one element, so the
    // branch below is uniform across the block -- no divergence at all. Off
    // the band every C-PML coefficient is exactly 0 and the stretch exactly 1
    // (measured: Sx0 == 1, Sx1/Sx2/La1..4 and all six coefficient triples
    // == 0), so every term the interior path skips is an exact zero and the
    // stretched derivatives ARE the raw ones. nullptr = every element band.
    const bool pml = !cf.elem_band || cf.elem_band[e] != 0;

    extern __shared__ float sh[];
    float* s_ux  = sh;              float* s_uz  = sh + m;
    float* s_uxo = sh + 2 * m;      float* s_uzo = sh + 3 * m;
    float* s_Fxb = sh + 4 * m;      float* s_Fxa = sh + 5 * m;
    float* s_Fzb = sh + 6 * m;      float* s_Fza = sh + 7 * m;
    const int g = ibool[off];
    s_ux[tid]  = ux[g];       s_uz[tid]  = uz[g];
    if (pml) { s_uxo[tid] = ux_prev[g];  s_uzo[tid] = uz_prev[g]; }
    __syncthreads();

    // 1) reference-grads of all 4 fields, then physical derivatives.
    float dux_dxi = 0.f, dux_deta = 0.f, duz_dxi = 0.f, duz_deta = 0.f;
    float duxo_dxi = 0.f, duxo_deta = 0.f, duzo_dxi = 0.f, duzo_deta = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float db = D[b * ngll + q], da = D[a * ngll + q];
        dux_dxi   += db * s_ux [a * ngll + q];  dux_deta  += da * s_ux [q * ngll + b];
        duz_dxi   += db * s_uz [a * ngll + q];  duz_deta  += da * s_uz [q * ngll + b];
        if (pml) {   // the previous step's derivatives feed the recursion ONLY
        duxo_dxi  += db * s_uxo[a * ngll + q];  duxo_deta += da * s_uxo[q * ngll + b];
        duzo_dxi  += db * s_uzo[a * ngll + q];  duzo_deta += da * s_uzo[q * ngll + b];
        }
    }
    const float ex = ixx[off], ez = ixz[off], nx = iex[off], nz = iez[off];
    const float dux_dx = ex * dux_dxi + nx * dux_deta;
    const float dux_dz = ez * dux_dxi + nz * dux_deta;
    const float duz_dx = ex * duz_dxi + nx * duz_deta;
    const float duz_dz = ez * duz_dxi + nz * duz_deta;
    const float dux_dx_o = ex * duxo_dxi + nx * duxo_deta;
    const float dux_dz_o = ez * duxo_dxi + nz * duxo_deta;
    const float duz_dx_o = ex * duzo_dxi + nx * duzo_deta;
    const float duz_dz_o = ez * duzo_dxi + nz * duzo_deta;

    // 2) strain-memory recursion (two branches each; updated value is used).
    //    x-derivatives use the zx coefficient set, z-derivatives the xz set.
    // Interior: every coefficient here is exactly 0, so r_* would be written
    // the zeros they already hold and the stretch is the identity. The four
    // memory buffers are 2 floats PER ELEMENT-NODE each -- 6 arrays x 2, which
    // at nex=160 is 30.7 MB read and 30.7 MB written per step against 1.6 MB
    // of field. Not touching them is where the interior path's saving is.
    float dux_dx_s = dux_dx, duz_dx_s = duz_dx;
    float dux_dz_s = dux_dz, duz_dz_s = duz_dz;
    if (pml) {
    float rxx1 = cf.zx1_0[off] * r_dux_dx[o2]     + cf.zx1_1[off] * dux_dx + cf.zx1_2[off] * dux_dx_o;
    float rxx2 = cf.zx2_0[off] * r_dux_dx[o2 + 1] + cf.zx2_1[off] * dux_dx + cf.zx2_2[off] * dux_dx_o;
    float rzx1 = cf.zx1_0[off] * r_duz_dx[o2]     + cf.zx1_1[off] * duz_dx + cf.zx1_2[off] * duz_dx_o;
    float rzx2 = cf.zx2_0[off] * r_duz_dx[o2 + 1] + cf.zx2_1[off] * duz_dx + cf.zx2_2[off] * duz_dx_o;
    float rxz1 = cf.xz1_0[off] * r_dux_dz[o2]     + cf.xz1_1[off] * dux_dz + cf.xz1_2[off] * dux_dz_o;
    float rxz2 = cf.xz2_0[off] * r_dux_dz[o2 + 1] + cf.xz2_1[off] * dux_dz + cf.xz2_2[off] * dux_dz_o;
    float rzz1 = cf.xz1_0[off] * r_duz_dz[o2]     + cf.xz1_1[off] * duz_dz + cf.xz1_2[off] * duz_dz_o;
    float rzz2 = cf.xz2_0[off] * r_duz_dz[o2 + 1] + cf.xz2_1[off] * duz_dz + cf.xz2_2[off] * duz_dz_o;
    r_dux_dx[o2] = rxx1;  r_dux_dx[o2 + 1] = rxx2;
    r_duz_dx[o2] = rzx1;  r_duz_dx[o2 + 1] = rzx2;
    r_dux_dz[o2] = rxz1;  r_dux_dz[o2 + 1] = rxz2;
    r_duz_dz[o2] = rzz1;  r_duz_dz[o2 + 1] = rzz2;

    // 3) asymmetrically stretched derivatives  d~ = S0 d + S1 r1 + S2 r2.
    dux_dx_s = cf.Sx0[off] * dux_dx + cf.Sx1[off] * rxx1 + cf.Sx2[off] * rxx2;
    duz_dx_s = cf.Sx0[off] * duz_dx + cf.Sx1[off] * rzx1 + cf.Sx2[off] * rzx2;
    dux_dz_s = cf.Sz0[off] * dux_dz + cf.Sz1[off] * rxz1 + cf.Sz2[off] * rxz2;
    duz_dz_s = cf.Sz0[off] * duz_dz + cf.Sz1[off] * rzz1 + cf.Sz2[off] * rzz2;
    }

    // 4) non-symmetric PML stress (raw = unstretched gradient).
    const float sxx = lam2mu[off] * dux_dx_s + lam[off] * duz_dz;
    const float szz = lam2mu[off] * duz_dz_s + lam[off] * dux_dx;
    const float szx = mu[off] * (duz_dx + dux_dz_s);      // -> accel_x
    const float sxz = mu[off] * (dux_dz + duz_dx_s);      // -> accel_z

    // 6) displ-memory recursion + PML acceleration (element-local, no D).
    // Interior: La1..La4 are exactly 0, so ap_* is exactly +0.0 -- checked on
    // the forward's own tape (16 800 interior entries, not one non-+0.0 bit
    // pattern). The four sd_* stores stay: the deviatoric stress is live in
    // the interior and the backward reads it there.
    float ap_x = 0.f, ap_z = 0.f;
    if (pml) {
    const float ue_x = s_ux[tid], ue_z = s_uz[tid];
    const float uo_x = s_uxo[tid], uo_z = s_uzo[tid];
    float rdx1 = cf.l1_0[off] * r_displ_x[o2]     + cf.l1_1[off] * ue_x + cf.l1_2[off] * uo_x;
    float rdx2 = cf.l2_0[off] * r_displ_x[o2 + 1] + cf.l2_1[off] * ue_x + cf.l2_2[off] * uo_x;
    float rdz1 = cf.l1_0[off] * r_displ_z[o2]     + cf.l1_1[off] * ue_z + cf.l1_2[off] * uo_z;
    float rdz2 = cf.l2_0[off] * r_displ_z[o2 + 1] + cf.l2_1[off] * ue_z + cf.l2_2[off] * uo_z;
    r_displ_x[o2] = rdx1;  r_displ_x[o2 + 1] = rdx2;
    r_displ_z[o2] = rdz1;  r_displ_z[o2 + 1] = rdz2;
    ap_x = cf.La1[off] * (ue_x - uo_x) * inv_dt + cf.La2[off] * ue_x
         + cf.La3[off] * rdx1 + cf.La4[off] * rdx2;
    ap_z = cf.La1[off] * (ue_z - uo_z) * inv_dt + cf.La2[off] * ue_z
         + cf.La3[off] * rdz1 + cf.La4[off] * rdz2;
    }

    if (save_sd_xx) {
        save_sd_xx[save_off + off] = dux_dx_s;  save_sd_zx[save_off + off] = duz_dx_s;
        save_sd_xz[save_off + off] = dux_dz_s;  save_sd_zz[save_off + off] = duz_dz_s;
        save_apx[save_off + off] = ap_x;        save_apz[save_off + off] = ap_z;
    }

    // 5) non-symmetric weak-form divergence (matches _assemble_stress).
    const float w = wJ[off];
    s_Fxb[tid] = w * (sxx * ex + szx * ez);
    s_Fxa[tid] = w * (sxx * nx + szx * nz);
    s_Fzb[tid] = w * (sxz * ex + szz * ez);
    s_Fza[tid] = w * (sxz * nx + szz * nz);
    __syncthreads();

    float rx = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float dgb = D[q * ngll + b], dga = D[q * ngll + a];
        rx += dgb * s_Fxb[a * ngll + q] + dga * s_Fxa[q * ngll + b];
        rz += dgb * s_Fzb[a * ngll + q] + dga * s_Fza[q * ngll + b];
    }
    // Newmark consumes -R - aPML; accumulate  mask*R + wrhoJ*ap  (R masked like
    // the eager _assemble_stress; the aPML assembly is unmasked, matching eager).
    const float mk = mask_e[e];
    // The band expression is kept textually as it was so its FMA contraction
    // is too; the interior one drops a term that is exactly +0.0 (and with it
    // the wrhoJ read).
    if (pml) {
    atomicAdd(&Acc_x[g], mk * rx + wrhoJ[off] * ap_x);
    atomicAdd(&Acc_z[g], mk * rz + wrhoJ[off] * ap_z);
    } else {
        atomicAdd(&Acc_x[g], mk * rx);
        atomicAdd(&Acc_z[g], mk * rz);
    }
}

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
static __global__ void sem2d_elastic_cpml_element_T(
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
    float* __restrict__ gWrhoJ, int ngll, int E, int ng)
{
    // shot batch: gridDim.y = B (shared materials; per-shot fields/memories/
    // histories/material-GRADIENTS -- the += on gLam.. stays race-free).
    { const int bat = blockIdx.y;
      const int m0 = ngll * ngll;
      const size_t fo = static_cast<size_t>(bat) * ng;
      const size_t eo2 = static_cast<size_t>(bat) * E * m0 * 2;
      const size_t eo1 = static_cast<size_t>(bat) * E * m0;
      y_x += fo; y_z += fo;
      if (ux_n) { ux_n += fo; uz_n += fo; }
      sd_xx += eo1; sd_zx += eo1; sd_xz += eo1; sd_zz += eo1;
      ap_x += eo1; ap_z += eo1;
      mu_xx += eo2; mu_zx += eo2; mu_xz += eo2; mu_zz += eo2;
      mu_dx += eo2; mu_dz += eo2;
      sc_now_x += fo; sc_now_z += fo; sc_prev_x += fo; sc_prev_z += fo;
      gLam += eo1; gMu += eo1; gL2m += eo1; gWrhoJ += eo1; }
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int a = tid / ngll, b = tid % ngll, off = e * m + tid;
    const int o2 = off * 2;
    // the forward kernel's flag, on the transpose: same premise, same block
    // uniformity. See sem2d_elastic_cpml_element.
    const bool pml = !cf.elem_band || cf.elem_band[e] != 0;

    extern __shared__ float sh[];
    float* s_yx = sh;          float* s_yz = sh + m;
    float* s_ux = sh + 2 * m;  float* s_uz = sh + 3 * m;
    float* c_xb = sh + 4 * m;  float* c_xa = sh + 5 * m;   // cov scratch (x comp)
    float* c_zb = sh + 6 * m;  float* c_za = sh + 7 * m;   // cov scratch (z comp)
    const int g = ibool[off];
    s_yx[tid] = y_x[g];  s_yz[tid] = y_z[g];
    s_ux[tid] = ux_n ? ux_n[g] : 0.f;
    s_uz[tid] = uz_n ? uz_n[g] : 0.f;
    __syncthreads();

    // 1) physical derivatives of the gathered adjoint (-> lsig) and of u^n (raw
    //    strains for the imaging correlations).
    float dyx_dxi = 0.f, dyx_deta = 0.f, dyz_dxi = 0.f, dyz_deta = 0.f;
    float dux_dxi = 0.f, dux_deta = 0.f, duz_dxi = 0.f, duz_deta = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float db = D[b * ngll + q], da = D[a * ngll + q];
        dyx_dxi += db * s_yx[a * ngll + q];  dyx_deta += da * s_yx[q * ngll + b];
        dyz_dxi += db * s_yz[a * ngll + q];  dyz_deta += da * s_yz[q * ngll + b];
        dux_dxi += db * s_ux[a * ngll + q];  dux_deta += da * s_ux[q * ngll + b];
        duz_dxi += db * s_uz[a * ngll + q];  duz_deta += da * s_uz[q * ngll + b];
    }
    const float ex = ixx[off], ez = ixz[off], nx = iex[off], nz = iez[off];
    const float dyx_dx = ex * dyx_dxi + nx * dyx_deta;
    const float dyx_dz = ez * dyx_dxi + nz * dyx_deta;
    const float dyz_dx = ex * dyz_dxi + nx * dyz_deta;
    const float dyz_dz = ez * dyz_dxi + nz * dyz_deta;
    const float dux_dx = ex * dux_dxi + nx * dux_deta;
    const float dux_dz = ez * dux_dxi + nz * dux_deta;
    const float duz_dx = ex * duz_dxi + nx * duz_deta;
    const float duz_dz = ez * duz_dxi + nz * duz_deta;

    // 2) adjoint stress  lsig_ij = mask wJ d_j(y_i)  (transpose of assemble).
    const float wJm = mask_e[e] * wJ[off];
    const float lsxx = wJm * dyx_dx;   // sig_xx (x-flux of x-eq)
    const float lszx = wJm * dyx_dz;   // sig_zx (z-flux of x-eq)
    const float lsxz = wJm * dyz_dx;   // sig_xz (x-flux of z-eq)
    const float lszz = wJm * dyz_dz;   // sig_zz (z-flux of z-eq)

    // 3) imaging (saved stretched derivs + raw derivs of u^n).
    const float v_sd_xx = sd_xx[off], v_sd_zx = sd_zx[off];
    const float v_sd_xz = sd_xz[off], v_sd_zz = sd_zz[off];
    gL2m[off] += v_sd_xx * lsxx + v_sd_zz * lszz;
    gLam[off] += duz_dz * lsxx + dux_dx * lszz;
    gMu[off]  += (duz_dx + v_sd_xz) * lszx + (dux_dz + v_sd_zx) * lsxz;

    // 4) stress^T -> stretched^T -> strain-memory adjoint recursion.
    const float l_sd_xx = lam2mu[off] * lsxx;    // lambda(dux_dx_s)
    const float l_sd_zz = lam2mu[off] * lszz;    // lambda(duz_dz_s)
    const float l_sd_xz = mu[off] * lszx;        // lambda(dux_dz_s)
    const float l_sd_zx = mu[off] * lsxz;        // lambda(duz_dx_s)
    // Interior: Sx1/Sx2/Sz1/Sz2 and every zx/xz coefficient are exactly 0, so
    // the four memory adjoints are zero, the stretch^T is the identity
    // (Sx0 == Sz0 == 1) and the "prev" lambdas vanish entirely -- which takes
    // section 7 below, a whole D^T contraction and two atomicAdds, with them.
    float l_dxx, l_dzx, l_dxz, l_dzz;
    float p_dxx = 0.f, p_dzx = 0.f, p_dxz = 0.f, p_dzz = 0.f;
    if (!pml) {
        l_dxx = l_sd_xx + lam[off] * lszz;
        l_dzx = l_sd_zx + mu[off] * lszx;
        l_dxz = l_sd_xz + mu[off] * lsxz;
        l_dzz = l_sd_zz + lam[off] * lsxx;
    } else {
    // mu_f^n = S-use + c0 mu_f^{n+1}   (in-place on the carry buffers)
    const float m_xx0 = cf.Sx1[off] * l_sd_xx + cf.zx1_0[off] * mu_xx[o2];
    const float m_xx1 = cf.Sx2[off] * l_sd_xx + cf.zx2_0[off] * mu_xx[o2 + 1];
    const float m_zx0 = cf.Sx1[off] * l_sd_zx + cf.zx1_0[off] * mu_zx[o2];
    const float m_zx1 = cf.Sx2[off] * l_sd_zx + cf.zx2_0[off] * mu_zx[o2 + 1];
    const float m_xz0 = cf.Sz1[off] * l_sd_xz + cf.xz1_0[off] * mu_xz[o2];
    const float m_xz1 = cf.Sz2[off] * l_sd_xz + cf.xz2_0[off] * mu_xz[o2 + 1];
    const float m_zz0 = cf.Sz1[off] * l_sd_zz + cf.xz1_0[off] * mu_zz[o2];
    const float m_zz1 = cf.Sz2[off] * l_sd_zz + cf.xz2_0[off] * mu_zz[o2 + 1];
    mu_xx[o2] = m_xx0;  mu_xx[o2 + 1] = m_xx1;
    mu_zx[o2] = m_zx0;  mu_zx[o2 + 1] = m_zx1;
    mu_xz[o2] = m_xz0;  mu_xz[o2 + 1] = m_xz1;
    mu_zz[o2] = m_zz0;  mu_zz[o2 + 1] = m_zz1;

    // lambda of the raw derivatives at n (S0/raw-stress paths + c1 memory path)
    l_dxx = cf.Sx0[off] * l_sd_xx + lam[off] * lszz
                      + cf.zx1_1[off] * m_xx0 + cf.zx2_1[off] * m_xx1;
    l_dzx = cf.Sx0[off] * l_sd_zx + mu[off] * lszx
                      + cf.zx1_1[off] * m_zx0 + cf.zx2_1[off] * m_zx1;
    l_dxz = cf.Sz0[off] * l_sd_xz + mu[off] * lsxz
                      + cf.xz1_1[off] * m_xz0 + cf.xz2_1[off] * m_xz1;
    l_dzz = cf.Sz0[off] * l_sd_zz + lam[off] * lsxx
                      + cf.xz1_1[off] * m_zz0 + cf.xz2_1[off] * m_zz1;
    // lambda of the previous-step derivatives (c2 memory path only)
    p_dxx = cf.zx1_2[off] * m_xx0 + cf.zx2_2[off] * m_xx1;
    p_dzx = cf.zx1_2[off] * m_zx0 + cf.zx2_2[off] * m_zx1;
    p_dxz = cf.xz1_2[off] * m_xz0 + cf.xz2_2[off] * m_xz1;
    p_dzz = cf.xz1_2[off] * m_zz0 + cf.xz2_2[off] * m_zz1;
    }

    // 5) accel-paren^T + displ-memory adjoint recursion (nodal, no derivs).
    // Interior: La1..La4 and l1/l2 are exactly 0, so lu_now / lu_prev are 0 and
    // ap_* is bitwise +0.0 -- the gWrhoJ term adds nothing and the ap_*, wrhoJ
    // and mu_d* reads go with it.
    float lu_now_x = 0.f, lu_now_z = 0.f, lu_prev_x = 0.f, lu_prev_z = 0.f;
    if (pml) {
    const float yx_e = s_yx[tid], yz_e = s_yz[tid];
    const float lap_x = wrhoJ[off] * yx_e;
    const float lap_z = wrhoJ[off] * yz_e;
    gWrhoJ[off] += ap_x[off] * yx_e + ap_z[off] * yz_e;
    const float md_x0 = cf.La3[off] * lap_x + cf.l1_0[off] * mu_dx[o2];
    const float md_x1 = cf.La4[off] * lap_x + cf.l2_0[off] * mu_dx[o2 + 1];
    const float md_z0 = cf.La3[off] * lap_z + cf.l1_0[off] * mu_dz[o2];
    const float md_z1 = cf.La4[off] * lap_z + cf.l2_0[off] * mu_dz[o2 + 1];
    mu_dx[o2] = md_x0;  mu_dx[o2 + 1] = md_x1;
    mu_dz[o2] = md_z0;  mu_dz[o2 + 1] = md_z1;
    lu_now_x = (cf.La1[off] * inv_dt + cf.La2[off]) * lap_x
                         + cf.l1_1[off] * md_x0 + cf.l2_1[off] * md_x1;
    lu_now_z = (cf.La1[off] * inv_dt + cf.La2[off]) * lap_z
                         + cf.l1_1[off] * md_z0 + cf.l2_1[off] * md_z1;
    lu_prev_x = -cf.La1[off] * inv_dt * lap_x
                          + cf.l1_2[off] * md_x0 + cf.l2_2[off] * md_x1;
    lu_prev_z = -cf.La1[off] * inv_dt * lap_z
                          + cf.l1_2[off] * md_z0 + cf.l2_2[off] * md_z1;
    }

    // 6) derivative^T for the "now" lambdas: cov -> D^T -> scatter.
    __syncthreads();                       // done with the gather use of shared
    c_xb[tid] = ex * l_dxx + ez * l_dxz;   // cov_xi  (x component)
    c_xa[tid] = nx * l_dxx + nz * l_dxz;   // cov_eta
    c_zb[tid] = ex * l_dzx + ez * l_dzz;
    c_za[tid] = nx * l_dzx + nz * l_dzz;
    __syncthreads();
    float ax = lu_now_x, az = lu_now_z;
    for (int q = 0; q < ngll; ++q) {
        const float dgb = D[q * ngll + b], dga = D[q * ngll + a];
        ax += dgb * c_xb[a * ngll + q] + dga * c_xa[q * ngll + b];
        az += dgb * c_zb[a * ngll + q] + dga * c_za[q * ngll + b];
    }
    atomicAdd(&sc_now_x[g], ax);
    atomicAdd(&sc_now_z[g], az);

    // 7) derivative^T for the "prev" lambdas (consumed one reverse step later).
    // Interior: every p_d* and lu_prev_* is exactly 0, so this whole section --
    // two shared rounds, a D^T contraction and two atomicAdds -- would scatter
    // +0.0. `pml` is uniform across the block, so skipping it (barriers and
    // all) is legal.
    if (!pml) return;
    __syncthreads();
    c_xb[tid] = ex * p_dxx + ez * p_dxz;
    c_xa[tid] = nx * p_dxx + nz * p_dxz;
    c_zb[tid] = ex * p_dzx + ez * p_dzz;
    c_za[tid] = nx * p_dzx + nz * p_dzz;
    __syncthreads();
    float px = lu_prev_x, pz = lu_prev_z;
    for (int q = 0; q < ngll; ++q) {
        const float dgb = D[q * ngll + b], dga = D[q * ngll + a];
        px += dgb * c_xb[a * ngll + q] + dga * c_xa[q * ngll + b];
        pz += dgb * c_zb[a * ngll + q] + dga * c_za[q * ngll + b];
    }
    atomicAdd(&sc_prev_x[g], px);
    atomicAdd(&sc_prev_z[g], pz);
}

// grad_inv_M via the leapfrog identity: dt^2 (b - Acc) = u^{n+1} - 2u^n + u^{n-1}
// (pinned nodes have g = 0, so the pin discrepancy contributes nothing).
static __global__ void sem2d_elastic_cpml_gradM(
    const float* __restrict__ gx, const float* __restrict__ gz,
    const float* __restrict__ u_np1_x, const float* __restrict__ u_np1_z,
    const float* __restrict__ u_n_x,   const float* __restrict__ u_n_z,    // may be null
    const float* __restrict__ u_nm1_x, const float* __restrict__ u_nm1_z,  // may be null
    const float* __restrict__ inv_M,
    float* __restrict__ gradM, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      gx += fo; gz += fo; u_np1_x += fo; u_np1_z += fo; gradM += fo;
      if (u_n_x) { u_n_x += fo; u_n_z += fo; }
      if (u_nm1_x) { u_nm1_x += fo; u_nm1_z += fo; } }
    if (i >= ng) return;
    const float unx = u_n_x ? u_n_x[i] : 0.f, unz = u_n_z ? u_n_z[i] : 0.f;
    const float umx = u_nm1_x ? u_nm1_x[i] : 0.f, umz = u_nm1_z ? u_nm1_z[i] : 0.f;
    const float im = inv_M[i];
    if (im == 0.f) return;                 // masked-out node (no mass, no grad)
    // g o dt^2 (b - Acc) = g o (u^{n+1} - 2u^n + u^{n-1}) / inv_M  (the leapfrog
    // identity gives dt^2 inv_M (b - Acc); divide the inv_M back out).
    // gradM is PER-SHOT (B, ng) -- bumped by fo above -- so the += stays
    // race-free across the batch; a shared model's gradient is the caller's
    // sum over the leading axis.
    gradM[i] += (gx[i] * (u_np1_x[i] - 2.f * unx + umx)
               + gz[i] * (u_np1_z[i] - 2.f * unz + umz)) / im;
}

// ================== fluid-solid coupling kernels (P4) =======================
// Interface quadrature: flat arrays of n_face*ngll GLL points with global node
// index g, weight w (w_1D*jac1D) and fluid-outward normal (nx, nz). All three
// kernels are plain gather/scatter — batch-ready by construction (index by an
// outer offset when a batch dim lands).

// S^ac_i = scatter( w * (nx*u_x[g] + nz*u_z[g]) )   (into the fluid residual)
static __global__ void sem2d_ifc_ac(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ s_ac, int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      ux += fo; uz += fo; s_ac += fo; }
    if (j >= n_pts) return;
    const long g = gidx[j];
    atomicAdd(&s_ac[g], w[j] * (nx[j] * ux[g] + nz[j] * uz[g]));
}

// (S^el_x, S^el_z) = scatter( w * p[g] * (nx, nz) ) with p = -chi_dd; adds into
// the solid force fields (exact discrete transpose of sem2d_ifc_ac).
static __global__ void sem2d_ifc_el(
    const float* __restrict__ chi_dd,
    const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ f_x, float* __restrict__ f_z, int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      chi_dd += fo; f_x += fo; f_z += fo; }
    if (j >= n_pts) return;
    const long g = gidx[j];
    const float pw = -w[j] * chi_dd[g];
    atomicAdd(&f_x[g], pw * nx[j]);
    atomicAdd(&f_z[g], pw * nz[j]);
}

// chi_dd = kappa*w_sum + inv_M*(pat*wav[n] + s_ac)   (post-inversion fluid accel)
static __global__ void sem2d_fs_chidd(
    const float* __restrict__ kappa, const float* __restrict__ w_sum,
    const float* __restrict__ inv_M, const float* __restrict__ pattern,
    const float* __restrict__ wavelet, int n, const float* __restrict__ s_ac,
    float* __restrict__ chi_dd, int ng, int wst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      w_sum += fo; pattern += fo; s_ac += fo; chi_dd += fo;
      wavelet += static_cast<size_t>(blockIdx.y) * wst; }
    if (i >= ng) return;
    chi_dd[i] = kappa[i] * w_sum[i] + inv_M[i] * (pattern[i] * wavelet[n] + s_ac[i]);
}

// bare-path fluid accel: chi_dd = inv_M * (pat*wav[n] + s_ac - K(chi))
static __global__ void sem2d_fs_chidd_bare(
    const float* __restrict__ Kchi, const float* __restrict__ inv_M,
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int n,
    const float* __restrict__ s_ac, float* __restrict__ chi_dd, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    chi_dd[i] = inv_M[i] * (pattern[i] * wavelet[n] + s_ac[i] - Kchi[i]);
}

// chi_next = 2 chi - chi_prev + dt2 * chi_dd
static __global__ void sem2d_fs_chi_leapfrog(
    const float* __restrict__ chi, const float* __restrict__ chi_prev,
    const float* __restrict__ chi_dd, float dt2,
    float* __restrict__ chi_next, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      chi += fo; chi_prev += fo; chi_dd += fo; chi_next += fo; }
    if (i >= ng) return;
    chi_next[i] = 2.f * chi[i] - chi_prev[i] + dt2 * chi_dd[i];
}

// 2-component Newmark with an ARBITRARY per-node force field (source pattern +
// interface term composed by the driver): u_next = 2u - u_prev + dt2 invM (f - Acc)
static __global__ void sem2d_elastic_newmark_force(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const float* __restrict__ Acc_x, const float* __restrict__ Acc_z,
    const float* __restrict__ f_x, const float* __restrict__ f_z,
    const float* __restrict__ inv_M, float dt2,
    float* __restrict__ ux_next, float* __restrict__ uz_next, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      ux += fo; uz += fo; ux_prev += fo; uz_prev += fo;
      Acc_x += fo; Acc_z += fo; f_x += fo; f_z += fo;
      ux_next += fo; uz_next += fo; }
    if (i >= ng) return;
    const float im = inv_M[i];
    ux_next[i] = 2.f * ux[i] - ux_prev[i] + dt2 * im * (f_x[i] - Acc_x[i]);
    uz_next[i] = 2.f * uz[i] - uz_prev[i] + dt2 * im * (f_z[i] - Acc_z[i]);
}

// f = pattern * wav[n]   (force-field initialisation before the interface adds)
static __global__ void sem2d_fs_pat_wav(
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int n,
    float* __restrict__ f, int ng, int wst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      pattern += fo; f += fo;
      wavelet += static_cast<size_t>(blockIdx.y) * wst; }
    if (i >= ng) return;
    f[i] = pattern[i] * wavelet[n];
}

// ================== fluid-solid backward glue kernels (P4) ==================
// out += s * a
static __global__ void sem2d_fs_axpy(
    const float* __restrict__ a, float s, float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n;
      a += fo; out += fo; }
    if (i >= n) return;
    out[i] += s * a[i];
}

// Fluid-accel adjoint -> material grads. lchidd = lambda(chi_dd) (post-Dir).
// CPML: gKappa += w_sum*lchidd (Kchi null); bare: Kchi subtracts (w_sum null).
// gInvM += (pat*wav[k] + s_ac - Kchi)*lchidd   [chi_dd = kappa w_sum + invM(...)]
static __global__ void sem2d_fs_ac_gradMK(
    const float* __restrict__ lchidd, const float* __restrict__ w_sum,
    const float* __restrict__ Kchi, const float* __restrict__ pat_chi,
    const float* __restrict__ wavelet, int k, const float* __restrict__ s_ac,
    float* __restrict__ gKappa, float* __restrict__ gInvM, int ng, int wst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const int bat = blockIdx.y;
      const size_t fo = static_cast<size_t>(bat) * ng;
      lchidd += fo; pat_chi += fo; s_ac += fo; gKappa += fo; gInvM += fo;
      if (w_sum) w_sum += fo;
      if (Kchi) Kchi += fo;
      wavelet += static_cast<size_t>(bat) * wst; }
    if (i >= ng) return;
    const float l = lchidd[i];
    if (w_sum) gKappa[i] += w_sum[i] * l;
    gInvM[i] += (pat_chi[i] * wavelet[k] + s_ac[i] - (Kchi ? Kchi[i] : 0.f)) * l;
}

// Fluid combine, CPML flavour (P3b combine minus the grad terms, which moved
// to sem2d_fs_ac_gradMK): lchi_k = gB + 2 gA + lchi_G; psi/zeta adjoint
// recursions with the coupled gw = kappa o lambda(chi_dd).
static __global__ void sem2d_fs_ac_combine_cpml(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ lchi_G,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_z,
    const float* __restrict__ lapx, const float* __restrict__ lapz,
    const float* __restrict__ gw,
    const float* __restrict__ a_x, const float* __restrict__ a_z,
    float* __restrict__ lchi_k,
    float* __restrict__ lpsi_x_n, float* __restrict__ lpsi_z_n,
    float* __restrict__ lzeta_x_n, float* __restrict__ lzeta_z_n, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      gA += fo; gB += fo; lchi_G += fo; lpsi_x += fo; lpsi_z += fo;
      lzeta_x += fo; lzeta_z += fo; lapx += fo; lapz += fo; gw += fo;
      lchi_k += fo; lpsi_x_n += fo; lpsi_z_n += fo;
      lzeta_x_n += fo; lzeta_z_n += fo; }
    if (i >= ng) return;
    lchi_k[i] = gB[i] + 2.f * gA[i] + lchi_G[i];
    lpsi_x_n[i] = a_x[i] * (lpsi_x[i] + lapx[i]);
    lpsi_z_n[i] = a_z[i] * (lpsi_z[i] + lapz[i]);
    lzeta_x_n[i] = a_x[i] * (lzeta_x[i] + gw[i]);
    lzeta_z_n[i] = a_z[i] * (lzeta_z[i] + gw[i]);
}

// Fluid combine, bare flavour: lchi_k = gB + 2 gA + extra (extra = K^T path)
static __global__ void sem2d_fs_combine3(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ extra, float* __restrict__ out, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    out[i] = gB[i] + 2.f * gA[i] + extra[i];
}

// lambda^n = 2 g - g_prev + adj_src + sc_now + sc_prev_carry   (per component)
static __global__ void sem2d_elastic_cpml_combine(
    const float* __restrict__ g, const float* __restrict__ g_prev,
    const float* __restrict__ adj, const float* __restrict__ sc_now,
    const float* __restrict__ sc_prev_carry,
    float* __restrict__ lam_out, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * ng;
      g += fo; g_prev += fo; adj += fo; sc_now += fo;
      sc_prev_carry += fo; lam_out += fo; }
    if (i >= ng) return;
    lam_out[i] = 2.f * g[i] - g_prev[i] + adj[i] + sc_now[i] + sc_prev_carry[i];
}

} // namespace sem2d

#include "kernels.cuh"
#include "../../launch/launch.h"   // SWEEP_LAUNCH

namespace sem2d {

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

__global__ void sem2d_acoustic_newmark(
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

__global__ void sem2d_dirichlet_fill(
    float* __restrict__ chi_next, const long* __restrict__ didx, int k)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    chi_next[ didx[j] ] = 0.f;
}

__global__ void sem2d_dirichlet_fill_b(
    float* __restrict__ f, const long* __restrict__ didx, int k, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    f[ didx[j] + static_cast<size_t>(blockIdx.y) * ng ] = 0.f;
}

__global__ void sem2d_record(
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

__global__ void sem2d_scale2(const float* __restrict__ a, const float* __restrict__ b,
                             float s, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s * a[i] * b[i];
}

__global__ void sem2d_scale2_b(const float* __restrict__ a,
                             const float* __restrict__ b, float s,
                             float* __restrict__ out, int n, int as_) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t fo = static_cast<size_t>(blockIdx.y) * n;
    if (i < n) out[fo + i] = s * a[static_cast<size_t>(blockIdx.y) * as_ + i]
                               * b[fo + i];
}

__global__ void sem2d_adjoint_source(
    const float* __restrict__ grad_syn, const long* __restrict__ rec_G,
    const float* __restrict__ rec_W, float* __restrict__ adj,
    int tcol, int nt, int n_rec, int mgll) {
    const int rj = blockIdx.x * blockDim.x + threadIdx.x;
    if (rj >= n_rec * mgll) return;
    const int r = rj / mgll;
    atomicAdd(&adj[ rec_G[rj] ], grad_syn[r * nt + tcol] * rec_W[rj]);
}

__global__ void sem2d_adjoint_source_b(
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

__global__ void sem2d_adj_leapfrog(
    const float* __restrict__ a1, const float* __restrict__ a2,
    const float* __restrict__ Ktmp, const float* __restrict__ adj,
    float dt2, float* __restrict__ ubar, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ubar[i] = 2.f * a1[i] - a2[i] - dt2 * Ktmp[i] + adj[i];
}

__global__ void sem2d_accum_gradM(
    const float* __restrict__ a_m, const float* __restrict__ pattern,
    const float* __restrict__ Ku, const float* __restrict__ wavelet, int k,
    float dt2, float* __restrict__ gradM, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // BATCH: non-atomic accumulation into a batch-SHARED gradient buffer.
    // Under B > 1 this is a lost-update race -- needs a node index + atomicAdd.
    if (i < n) gradM[i] += dt2 * a_m[i] * (pattern[i] * wavelet[k] - Ku[i]);
}

__global__ void sem2d_accum_gradW(
    const float* __restrict__ pattern, const float* __restrict__ inv_M,
    const float* __restrict__ a_m, float dt2, int k,
    float* __restrict__ gradW, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // every thread must reach the reduction (shuffle masks + __syncthreads)
    const float v = (i < n) ? dt2 * pattern[i] * inv_M[i] * a_m[i] : 0.0f;
    const float bs = sem2d_block_sum(v);
    if (threadIdx.x == 0) atomicAdd(&gradW[k], bs);
}

__global__ void sem2d_accum_gradW_b(
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

__global__ void sem2d_imaging_invrho(
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

__global__ void sem2d_elastic_stiffness(
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

__global__ void sem2d_elastic_newmark(
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

__global__ void sem2d_elastic_accum_gradM(
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

__global__ void sem2d_elastic_accum_gradW(
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

__global__ void sem2d_elastic_imaging(
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

__global__ void sem2d_scale1(const float* __restrict__ in, float s,
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
__global__ void sem2d_acoustic_cpml_grads(
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

template <bool WSAVE>
__global__ void sem2d_acoustic_cpml_step_newmark_f(
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
sem2d_acoustic_cpml_step_newmark_f_fn const sem2d_acoustic_cpml_step_newmark_f_by_flag[2] = {
    &sem2d_acoustic_cpml_step_newmark_f<false>, &sem2d_acoustic_cpml_step_newmark_f<true>};

__global__ void sem2d_dirichlet_fill5(
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
__global__ void sem2d_band_gather(
    const float* __restrict__ field, const long* __restrict__ band_idx,
    float* __restrict__ out, int nband, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    field += static_cast<size_t>(blockIdx.y) * ng;
    out += static_cast<size_t>(blockIdx.y) * nband;
    if (j < nband) out[j] = field[band_idx[j]];
}

__global__ void sem2d_band_scatter(
    const float* __restrict__ vals, const long* __restrict__ band_idx,
    float* __restrict__ field, int nband, int ng) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    vals += static_cast<size_t>(blockIdx.y) * nband;
    field += static_cast<size_t>(blockIdx.y) * ng;
    if (j < nband) field[band_idx[j]] = vals[j];
}

__device__ __forceinline__ float sem2d_block_max(float v) {
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
__global__ void sem2d_band_amax(
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
__device__ __forceinline__ void sem_band_put(float& d, float v) { d = v; }
__device__ __forceinline__ void sem_band_put(__nv_bfloat16& d, float v) {
    d = __float2bfloat16(v);
}
__device__ __forceinline__ void sem_band_put(signed char& d, float v) {
    const int q = __float2int_rn(v);
    d = static_cast<signed char>(q < -127 ? -127 : (q > 127 ? 127 : q));
}
__device__ __forceinline__ float sem_band_get(float v) { return v; }
__device__ __forceinline__ float sem_band_get(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ float sem_band_get(signed char v) {
    return static_cast<float>(v);
}

template <typename T>
__global__ void sem2d_band_gather_q(
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
__global__ void sem2d_band_scatter_q(
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

void sem2d_band_save(const float* field, const long* band_idx,
                     SemBandHist h, long step, int nband,
                     int tpb, cudaStream_t stream,
                     int B, int ng) {
    if (!h.data || nband <= 0) return;
    const int nb = (nband + tpb - 1) / tpb;
    const dim3 g(nb, B);
    const size_t off = static_cast<size_t>(step) * nband * B;
    float* sc = nullptr;
    if (h.kind == SEM_BAND_I8) {
        sc = h.scale + static_cast<size_t>(step) * B;
        SWEEP_LAUNCH_ON(sem2d_band_amax, dim3(1, B), 256, 0, stream, field, band_idx, sc, nband, ng);
    }
    switch (h.kind) {
    case SEM_BAND_BF16:
        SWEEP_LAUNCH_ON((sem2d_band_gather_q<__nv_bfloat16>), g, tpb, 0, stream,
            field, band_idx, static_cast<__nv_bfloat16*>(h.data) + off,
            nullptr, nband, ng);
        break;
    case SEM_BAND_I8:
        SWEEP_LAUNCH_ON((sem2d_band_gather_q<signed char>), g, tpb, 0, stream,
            field, band_idx, static_cast<signed char*>(h.data) + off, sc, nband, ng);
        break;
    default:
        SWEEP_LAUNCH_ON(sem2d_band_gather, g, tpb, 0, stream,
            field, band_idx, static_cast<float*>(h.data) + off, nband, ng);
    }
}

void sem2d_band_load(SemBandHist h, long step, const long* band_idx,
                     float* field, int nband, int tpb,
                     cudaStream_t stream, int B, int ng) {
    if (!h.data || nband <= 0) return;
    const int nb = (nband + tpb - 1) / tpb;
    const dim3 g(nb, B);
    const size_t off = static_cast<size_t>(step) * nband * B;
    switch (h.kind) {
    case SEM_BAND_BF16:
        SWEEP_LAUNCH_ON((sem2d_band_scatter_q<__nv_bfloat16>), g, tpb, 0, stream,
            static_cast<const __nv_bfloat16*>(h.data) + off, band_idx, field,
            nullptr, nband, ng);
        break;
    case SEM_BAND_I8:
        SWEEP_LAUNCH_ON((sem2d_band_scatter_q<signed char>), g, tpb, 0, stream,
            static_cast<const signed char*>(h.data) + off, band_idx, field,
            h.scale + static_cast<size_t>(step) * B, nband, ng);
        break;
    default:
        SWEEP_LAUNCH_ON(sem2d_band_scatter, g, tpb, 0, stream,
            static_cast<const float*>(h.data) + off, band_idx, field, nband, ng);
    }
}

__global__ void sem2d_elastic_adj_carry(
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
__global__ void sem2d_band_gather_elem(
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

__global__ void sem2d_band_scatter_elem(
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
__global__ void sem2d_band_amax_elem(
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
__global__ void sem2d_band_gather_elem_q(
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
__global__ void sem2d_band_scatter_elem_q(
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

void sem2d_band_save_elem(const float* field,
                          const long* elem_idx, SemBandHist h,
                          long step, int nbe, int m, long Em,
                          int tpb, cudaStream_t stream,
                          int B) {
    if (!h.data || nbe <= 0) return;
    const int n = nbe * m;
    const dim3 g((n + tpb - 1) / tpb, B);
    const size_t off = static_cast<size_t>(step) * n * B;
    float* sc = nullptr;
    if (h.kind == SEM_BAND_I8) {
        sc = h.scale + static_cast<size_t>(step) * B * nbe;
        SWEEP_LAUNCH_ON(sem2d_band_amax_elem, dim3(nbe, B), 128, 0, stream,
            field, elem_idx, sc, nbe, m, Em);
    }
    switch (h.kind) {
    case SEM_BAND_BF16:
        SWEEP_LAUNCH_ON((sem2d_band_gather_elem_q<__nv_bfloat16>), g, tpb, 0, stream,
            field, elem_idx, static_cast<__nv_bfloat16*>(h.data) + off,
            nullptr, nbe, m, Em);
        break;
    case SEM_BAND_I8:
        SWEEP_LAUNCH_ON((sem2d_band_gather_elem_q<signed char>), g, tpb, 0, stream,
            field, elem_idx, static_cast<signed char*>(h.data) + off, sc,
            nbe, m, Em);
        break;
    default:
        SWEEP_LAUNCH_ON(sem2d_band_gather_elem, g, tpb, 0, stream,
            field, elem_idx, static_cast<float*>(h.data) + off, nbe, m, Em);
    }
}

void sem2d_band_load_elem(SemBandHist h, long step,
                          const long* elem_idx, float* field,
                          int nbe, int m, long Em, int tpb,
                          cudaStream_t stream, int B) {
    if (!h.data || nbe <= 0) return;
    const int n = nbe * m;
    const dim3 g((n + tpb - 1) / tpb, B);
    const size_t off = static_cast<size_t>(step) * n * B;
    switch (h.kind) {
    case SEM_BAND_BF16:
        SWEEP_LAUNCH_ON((sem2d_band_scatter_elem_q<__nv_bfloat16>), g, tpb, 0, stream,
            static_cast<const __nv_bfloat16*>(h.data) + off, elem_idx, field,
            nullptr, nbe, m, Em);
        break;
    case SEM_BAND_I8:
        SWEEP_LAUNCH_ON((sem2d_band_scatter_elem_q<signed char>), g, tpb, 0, stream,
            static_cast<const signed char*>(h.data) + off, elem_idx, field,
            h.scale + static_cast<size_t>(step) * B * nbe, nbe, m, Em);
        break;
    default:
        SWEEP_LAUNCH_ON(sem2d_band_scatter_elem, g, tpb, 0, stream,
            static_cast<const float*>(h.data) + off, elem_idx, field, nbe, m,
            Em);
    }
}

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

__global__ void sem2d_acoustic_cpml_newmark(
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

template <bool WTAPE>
__global__ void sem2d_acoustic_cpml_step_T(
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
sem2d_acoustic_cpml_step_T_fn const sem2d_acoustic_cpml_step_T_by_flag[2] = {
    &sem2d_acoustic_cpml_step_T<false>, &sem2d_acoustic_cpml_step_T<true>};

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
__global__ void sem2d_acoustic_cpml_combine(
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

__global__ void sem2d_elastic_cpml_gradM(
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

__global__ void sem2d_ifc_ac(
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

__global__ void sem2d_ifc_el(
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

__global__ void sem2d_illum_src(
    const float* __restrict__ f_next, const float* __restrict__ f,
    const float* __restrict__ f_prev, float inv_dt2, double* __restrict__ acc, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const size_t o = static_cast<size_t>(blockIdx.y) * ng + i;
    const double a = static_cast<double>((f_next[o] - 2.f * f[o] + f_prev[o]) * inv_dt2);
    acc[o] += a * a;
}

__global__ void sem2d_illum_rec(
    const float* __restrict__ lam, const float* __restrict__ inv_M, int ms,
    double* __restrict__ acc, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const size_t o = static_cast<size_t>(blockIdx.y) * ng + i;
    const double l = static_cast<double>(inv_M[static_cast<size_t>(blockIdx.y) * ms + i])
                   * static_cast<double>(lam[o]);
    acc[o] += l * l;
}

__global__ void sem2d_ifc_pml_ac(
    const float* __restrict__ ux, const float* __restrict__ uz,
    const float* __restrict__ ux_prev, const float* __restrict__ uz_prev,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ mem, float* __restrict__ s_ac, float* __restrict__ tape,
    int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    { const size_t fo = static_cast<size_t>(b) * ng;
      ux += fo; uz += fo; ux_prev += fo; uz_prev += fo; s_ac += fo; }
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    float* m = mem + pj * 4;
    const float* c = cf + static_cast<size_t>(j) * IFC_PML_K;
    const long g = gidx[j];
    const float qx = c[17] * m[2] + c[18] * ux[g] + c[19] * ux_prev[g];
    const float qz = c[21] * m[3] + c[22] * uz[g] + c[23] * uz_prev[g];
    m[2] = qx; m[3] = qz;
    const float ds = c[0] * (-c[16] * qx) + c[1] * (-c[20] * qz);
    atomicAdd(&s_ac[g], ds);
    if (tape) tape[pj * 3] = ds;
}

__global__ void sem2d_ifc_pml_el(
    const float* __restrict__ chi_dd, const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ mem, float* __restrict__ f_x, float* __restrict__ f_z,
    float* __restrict__ tape, int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    { const size_t fo = static_cast<size_t>(b) * ng;
      chi_dd += fo; chi += fo; chi_prev += fo; f_x += fo; f_z += fo; }
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    float* m = mem + pj * 4;
    const float* c = cf + static_cast<size_t>(j) * IFC_PML_K;
    const long g = gidx[j];
    const float dd = chi_dd[g], ch = chi[g], cp = chi_prev[g];
    const float rx = c[6] * m[0] + c[7] * ch + c[8] * cp;
    const float rz = c[13] * m[1] + c[14] * ch + c[15] * cp;
    m[0] = rx; m[1] = rz;
    const float dpx = -(c[2] * dd + c[3] * ch + c[4] * cp + c[5] * rx);
    const float dpz = -(c[9] * dd + c[10] * ch + c[11] * cp + c[12] * rz);
    const float fx = c[0] * dpz, fz = c[1] * dpx;
    atomicAdd(&f_x[g], fx);
    atomicAdd(&f_z[g], fz);
    if (tape) { tape[pj * 3 + 1] = fx; tape[pj * 3 + 2] = fz; }
}

__global__ void sem2d_ifc_pml_tape_add(
    const float* __restrict__ tape, const long* __restrict__ gidx,
    float* __restrict__ out0, int slot0, float* __restrict__ out1, int slot1,
    int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    const size_t fo = static_cast<size_t>(b) * ng;
    const long g = gidx[j];
    atomicAdd(&out0[fo + g], tape[pj * 3 + slot0]);
    if (out1) atomicAdd(&out1[fo + g], tape[pj * 3 + slot1]);
}

__global__ void sem2d_ifc_pml_el_T(
    const float* __restrict__ y_x, const float* __restrict__ y_z,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ lmem, float* __restrict__ lchidd, float* __restrict__ scr,
    int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    { const size_t fo = static_cast<size_t>(b) * ng;
      y_x += fo; y_z += fo; lchidd += fo; }
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    float* L = lmem + pj * 4;
    float* sc = scr + pj * 2;
    const float* c = cf + static_cast<size_t>(j) * IFC_PML_K;
    const long g = gidx[j];
    const float lpz = -c[0] * y_x[g], lpx = -c[1] * y_z[g];
    const float Lx = -c[5] * lpx + c[6] * L[0];
    const float Lz = -c[12] * lpz + c[13] * L[1];
    L[0] = Lx; L[1] = Lz;
    atomicAdd(&lchidd[g], -(c[2] * lpx + c[9] * lpz));
    const float l_now = -(c[3] * lpx + c[10] * lpz) + c[7] * Lx + c[14] * Lz;
    const float l_prev = -(c[4] * lpx + c[11] * lpz) + c[8] * Lx + c[15] * Lz;
    sc[0] = l_now + sc[1];
    sc[1] = l_prev;
}

__global__ void sem2d_ifc_pml_scr_add(
    const float* __restrict__ scr, const long* __restrict__ gidx,
    float* __restrict__ lchi_G, int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    atomicAdd(&lchi_G[static_cast<size_t>(b) * ng + gidx[j]], scr[pj * 2]);
}

__global__ void sem2d_ifc_pml_ac_T(
    const float* __restrict__ lchidd, const float* __restrict__ inv_M,
    const long* __restrict__ gidx, const float* __restrict__ cf,
    float* __restrict__ lmem,
    float* __restrict__ sc_now_x, float* __restrict__ sc_now_z,
    float* __restrict__ sc_prev_x, float* __restrict__ sc_prev_z, int n_p, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    { const size_t fo = static_cast<size_t>(b) * ng;
      lchidd += fo; sc_now_x += fo; sc_now_z += fo; sc_prev_x += fo; sc_prev_z += fo; }
    if (j >= n_p) return;
    const size_t pj = static_cast<size_t>(b) * n_p + j;
    float* L = lmem + pj * 4;
    const float* c = cf + static_cast<size_t>(j) * IFC_PML_K;
    const long g = gidx[j];
    const float ls = inv_M[g] * lchidd[g];
    const float Qx = -c[0] * c[16] * ls + c[17] * L[2];
    const float Qz = -c[1] * c[20] * ls + c[21] * L[3];
    L[2] = Qx; L[3] = Qz;
    atomicAdd(&sc_now_x[g], c[18] * Qx);
    atomicAdd(&sc_now_z[g], c[22] * Qz);
    atomicAdd(&sc_prev_x[g], c[19] * Qx);
    atomicAdd(&sc_prev_z[g], c[23] * Qz);
}

__global__ void sem2d_fs_chidd(
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

__global__ void sem2d_fs_chidd_bare(
    const float* __restrict__ Kchi, const float* __restrict__ inv_M,
    const float* __restrict__ pattern, const float* __restrict__ wavelet, int n,
    const float* __restrict__ s_ac, float* __restrict__ chi_dd, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    chi_dd[i] = inv_M[i] * (pattern[i] * wavelet[n] + s_ac[i] - Kchi[i]);
}

__global__ void sem2d_fs_chi_leapfrog(
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

__global__ void sem2d_elastic_newmark_force(
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

__global__ void sem2d_fs_pat_wav(
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

__global__ void sem2d_fs_axpy(
    const float* __restrict__ a, float s, float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    { const size_t fo = static_cast<size_t>(blockIdx.y) * n;
      a += fo; out += fo; }
    if (i >= n) return;
    out[i] += s * a[i];
}

__global__ void sem2d_fs_ac_gradMK(
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

__global__ void sem2d_fs_combine3(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ extra, float* __restrict__ out, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    out[i] = gB[i] + 2.f * gA[i] + extra[i];
}

__global__ void sem2d_elastic_cpml_combine(
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

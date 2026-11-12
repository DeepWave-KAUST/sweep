#include "kernels.cuh"
#include "vs2d_kernels.cuh"
#include "biot2d_kernels.cuh"

namespace sem2d {

__global__ void biot2d_inject_p(
    float* __restrict__ p, const float* __restrict__ inv_M1, const float* __restrict__ pin_n,
    const float* __restrict__ pat_p, float wv, float dt, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    p[i] = pin_n[i] * (p[i] + dt * inv_M1[i] * (pat_p[i] * wv));
}

__global__ void biot2d_grad_p(
    const float* __restrict__ p, float* __restrict__ gp, float* __restrict__ mp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    extern __shared__ float sh[];
    sh[tid] = p[ibool[off]];
    __syncthreads();
    const int row = kz * ngll;
    float rxi = 0.f, reta = 0.f;
    for (int q = 0; q < ngll; ++q) {
        rxi += D[kx * ngll + q] * sh[row + q];
        reta += D[kz * ngll + q] * sh[q * ngll + kx];
    }
    float gx = Ixx[off] * rxi + Iex[off] * reta;
    float gz = Ixz[off] * rxi + Iez[off] * reta;
    if (mp) {
        const float mx = a_x[off] * mp[off] + b_x[off] * gx;
        const float mz = a_z[off] * mp[Em + off] + b_z[off] * gz;
        mp[off] = mx; mp[Em + off] = mz;
        gx = gx + mx; gz = gz + mz;
    }
    gp[off] = gx; gp[Em + off] = gz;
}

__global__ void biot2d_force(
    const float* __restrict__ Rx, const float* __restrict__ Rz, float* __restrict__ f,
    const float* __restrict__ pat_vx, const float* __restrict__ pat_vz, float wv, int load,
    int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    float fx = -Rx[i], fz = -Rz[i];
    if (load) { fx += pat_vx[i] * wv; fz += pat_vz[i] * wv; }
    f[i] = fx; f[ng + i] = fz;
}

__global__ void biot2d_ifc_el(
    const float* __restrict__ p, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz, float* __restrict__ f,
    int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    const float pw = w[j] * p[g];
    atomicAdd(&f[g], pw * nx[j]);
    atomicAdd(&f[ng + g], pw * nz[j]);
}

// One node's copies: new = E qs + Psi rs, over (x copies, z copies) -- the same
// kmax x kmax E / Psi for both components, or one (2 kmax)^2 pair with a drag tensor.
template <bool TENSOR>
__device__ __forceinline__ void biot2d_advance(
    const float* __restrict__ Ei, const float* __restrict__ Pi, const float* qs,
    const float* rs, float* nw, int kmax)
{
    if constexpr (TENSOR) {
        const int K = 2 * kmax;
        for (int a = 0; a < K; ++a) {
            float s = 0.f;
            for (int b = 0; b < K; ++b) s += Ei[a * K + b] * qs[b] + Pi[a * K + b] * rs[b];
            nw[a] = s;
        }
    } else {
        for (int c = 0; c < 2; ++c)
            for (int a = 0; a < kmax; ++a) {
                float s = 0.f;
                for (int b = 0; b < kmax; ++b)
                    s += Ei[a * kmax + b] * qs[c * kmax + b] + Pi[a * kmax + b] * rs[c * kmax + b];
                nw[c * kmax + a] = s;
            }
    }
}

template <bool TENSOR>
__global__ void biot2d_flux(
    float* __restrict__ vx, float* __restrict__ vz, float* __restrict__ q,
    const float* __restrict__ r, const float* __restrict__ gp, const float* __restrict__ f,
    const float* __restrict__ wJ, const float* __restrict__ qmask,
    const float* __restrict__ inv_m, const float* __restrict__ inv_Mm,
    const int* __restrict__ slot_elem, const float* __restrict__ wr,
    const float* __restrict__ share, const float* __restrict__ dragE,
    const float* __restrict__ dragP, const float* __restrict__ pat_vx,
    const float* __restrict__ pat_vz, const float* __restrict__ pat_qx,
    const float* __restrict__ pat_qz, float wv, int velocity_mode, float dt,
    float* __restrict__ s_f, float* __restrict__ s_qs, float* __restrict__ s_rs,
    int kmax, int m, long Em, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float im = inv_m[i];
    const float fx = f[i], fz = f[ng + i];
    const float fim_x = fx * im, fim_z = fz * im;
    const float sqx = velocity_mode ? 0.f : pat_qx[i] * wv;
    const float sqz = velocity_mode ? 0.f : pat_qz[i] * wv;
    float qs[2 * BIOT2D_KMAX], rs[2 * BIOT2D_KMAX], nw[2 * BIOT2D_KMAX];
    for (int k = 0; k < kmax; ++k) {
        const int e = slot_elem[i * kmax + k];
        float gx = 0.f, gz = 0.f, qx = 0.f, qz = 0.f;
        if (e >= 0) {
            const float w = wJ[e] * qmask[e / m];
            gx = w * (r[e] - gp[e]);
            gz = w * (r[Em + e] - gp[Em + e]);
            qx = q[e]; qz = q[Em + e];
        }
        const float wk = wr[i * kmax + k], sk = share[i * kmax + k];
        rs[k] = gx - wk * fim_x + sk * sqx;
        rs[kmax + k] = gz - wk * fim_z + sk * sqz;
        qs[k] = qx; qs[kmax + k] = qz;
    }
    const int K = TENSOR ? 2 * kmax : kmax;
    biot2d_advance<TENSOR>(dragE + static_cast<size_t>(i) * K * K,
                           dragP + static_cast<size_t>(i) * K * K, qs, rs, nw, kmax);
    float dx = 0.f, dz = 0.f;
    for (int k = 0; k < kmax; ++k) {
        const float wk = wr[i * kmax + k];
        dx += wk * (nw[k] - qs[k]);
        dz += wk * (nw[kmax + k] - qs[kmax + k]);
    }
    float nvx = vx[i] + (dt * fx - dx) * im;
    float nvz = vz[i] + (dt * fz - dz) * im;
    float jqx = 0.f, jqz = 0.f;
    if (velocity_mode) {
        nvx += dt * im * (pat_vx[i] * wv);
        nvz += dt * im * (pat_vz[i] * wv);
        jqx = dt * inv_Mm[i] * (pat_qx[i] * wv);
        jqz = dt * inv_Mm[i] * (pat_qz[i] * wv);
    }
    vx[i] = nvx; vz[i] = nvz;
    for (int k = 0; k < kmax; ++k) {
        const int e = slot_elem[i * kmax + k];
        if (e < 0) continue;
        const float qm = qmask[e / m];
        q[e] = nw[k] + jqx * qm;
        q[Em + e] = nw[kmax + k] + jqz * qm;
    }
    if (s_f) {
        s_f[i] = fx; s_f[ng + i] = fz;
        for (int k = 0; k < kmax; ++k) {
            s_qs[static_cast<size_t>(i) * kmax + k] = qs[k];
            s_qs[(static_cast<size_t>(ng) + i) * kmax + k] = qs[kmax + k];
            s_rs[static_cast<size_t>(i) * kmax + k] = rs[k];
            s_rs[(static_cast<size_t>(ng) + i) * kmax + k] = rs[kmax + k];
        }
    }
}
biot2d_flux_fn const biot2d_flux_by_flag[2] = {
    &biot2d_flux<false>, &biot2d_flux<true>};

__global__ void biot2d_stress(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ sig, float* __restrict__ mv,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ lam_d, const float* __restrict__ mu,
    const float* __restrict__ alpha, const float* __restrict__ wJ,
    int nsls, const float* __restrict__ ptau, const float* __restrict__ mtau,
    const float* __restrict__ gsls_w, float* __restrict__ rmem,   // GSLS frame (nsls > 0)
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D, float dt,
    float* __restrict__ adiv, float* __restrict__ save_gv, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    extern __shared__ float sh[];
    float* s_vx = sh; float* s_vz = sh + m;
    const int g0 = ibool[off];
    s_vx[tid] = vx[g0]; s_vz[tid] = vz[g0];
    __syncthreads();
    const int row = kz * ngll;
    float r[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx * ngll + q], Dz = D[kz * ngll + q];
        const int iz = q * ngll + kx;
        r[0] += Dx * s_vx[row + q]; r[1] += Dz * s_vx[iz];
        r[2] += Dx * s_vz[row + q]; r[3] += Dz * s_vz[iz];
    }
    const float ixx = Ixx[off], ixz = Ixz[off], iex = Iex[off], iez = Iez[off];
    float gv[4] = {ixx * r[0] + iex * r[1], ixz * r[2] + iez * r[3],
                   ixz * r[0] + iez * r[1], ixx * r[2] + iex * r[3]};
    if (mv) {
        const float ax = a_x[off], bx = b_x[off], az = a_z[off], bz = b_z[off];
        const float ca[4] = {ax, az, az, ax};
        const float cb[4] = {bx, bz, bz, bx};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float mm = ca[j] * mv[j * Em + off] + cb[j] * gv[j];
            mv[j * Em + off] = mm;
            gv[j] = gv[j] + mm;
        }
    }
    if (save_gv) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) save_gv[j * Em + off] = gv[j];
    }
    const float div_v = gv[0] + gv[1];
    const float ld = lam_d[off], u_ = mu[off];
    sig[0 * Em + off] += dt * (ld * div_v + 2.f * u_ * gv[0]);
    sig[1 * Em + off] += dt * (ld * div_v + 2.f * u_ * gv[1]);
    sig[2 * Em + off] += dt * u_ * (gv[2] + gv[3]);
    if (nsls) vs2d_gsls_step(sig, rmem, Em, off, gv, dt, nsls, ptau, mtau, gsls_w);
    atomicAdd(&adiv[g0], wJ[off] * alpha[off] * div_v);
}

__global__ void biot2d_div_q(
    const float* __restrict__ q,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ divq, float* __restrict__ Pq, int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const bool layer = a_x && __syncthreads_or(
        a_x[off] != 0.f || a_z[off] != 0.f || b_x[off] != 0.f || b_z[off] != 0.f);
    extern __shared__ float sh[];
    const float w = wJ[off], qx = q[off], qz = q[Em + off];
    const float ixx = Ixx[off], ixz = Ixz[off], iex = Iex[off], iez = Iez[off];
    const int row = kz * ngll, g = ibool[off];
    if (layer) {               // slots (j, a): w q_j I_aj
        sh[0 * m + tid] = w * qx * ixx; sh[1 * m + tid] = w * qx * iex;
        sh[2 * m + tid] = w * qz * ixz; sh[3 * m + tid] = w * qz * iez;
        __syncthreads();
        float px = 0.f, pz = 0.f;
        for (int k = 0; k < ngll; ++k) {
            const float Tx = D[k * ngll + kx], Tz = D[k * ngll + kz];
            const int iz = k * ngll + kx;
            px += Tx * sh[0 * m + row + k] + Tz * sh[1 * m + iz];
            pz += Tx * sh[2 * m + row + k] + Tz * sh[3 * m + iz];
        }
        atomicAdd(&Pq[g], px);
        atomicAdd(&Pq[n_global + g], pz);
        return;
    }
    sh[0 * m + tid] = w * (qx * ixx + qz * ixz);
    sh[1 * m + tid] = w * (qx * iex + qz * iez);
    __syncthreads();
    float d = 0.f;
    for (int k = 0; k < ngll; ++k) {
        const float Tx = D[k * ngll + kx], Tz = D[k * ngll + kz];
        d += Tx * sh[0 * m + row + k] + Tz * sh[1 * m + k * ngll + kx];
    }
    atomicAdd(&divq[g], d);
}

__global__ void biot2d_cpml_q(
    const float* __restrict__ a_n, const float* __restrict__ Pq, float* __restrict__ mq,
    float* __restrict__ divq, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float pj = Pq[j * ng + i];
        const float psi = a_n[(2 * j) * ng + i] * mq[j * ng + i] + a_n[(2 * j + 1) * ng + i] * pj;
        mq[j * ng + i] = psi;
        s += pj + psi;
    }
    divq[i] += s;
}

__global__ void biot2d_ifc_ac(
    const float* __restrict__ vx, const float* __restrict__ vz, const long* __restrict__ gidx,
    const float* __restrict__ w, const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ divq, int n_pts)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    atomicAdd(&divq[g], -w[j] * (nx[j] * vx[g] + nz[j] * vz[g]));
}

__global__ void biot2d_pressure(
    float* __restrict__ p, const float* __restrict__ divq, const float* __restrict__ adiv,
    const float* __restrict__ inv_Mp, const float* __restrict__ pin_n, float dt,
    float* __restrict__ dp, float* __restrict__ s_rhs, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float rhs = divq[i] - adiv[i];
    const float d = pin_n[i] * (dt * inv_Mp[i] * rhs);
    p[i] += d;
    dp[i] = d;
    if (s_rhs) s_rhs[i] = rhs;
}

__global__ void biot2d_stress2(
    float* __restrict__ sig, float* __restrict__ r, const float* __restrict__ q,
    const float* __restrict__ alpha, const float* __restrict__ dp,
    const float* __restrict__ b_xx, const float* __restrict__ b_xz,
    const float* __restrict__ b_zz, float decay, float gain, int viscous,
    float* __restrict__ save_q, const int* __restrict__ ibool, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    const float ds = alpha[off] * dp[ibool[off]];
    sig[off] -= ds;
    sig[Em + off] -= ds;
    if (viscous) {
        const float qx = q[off], qz = q[Em + off];
        r[off] = decay * r[off] + gain * (b_xx[off] * qx + b_xz[off] * qz);
        r[Em + off] = decay * r[Em + off] + gain * (b_xz[off] * qx + b_zz[off] * qz);
        if (save_q) { save_q[off] = qx; save_q[Em + off] = qz; }
    }
}

__global__ void biot2d_stress_T(
    const float* __restrict__ lsig, float* __restrict__ lr, float* __restrict__ lq,
    const float* __restrict__ gvs, const float* __restrict__ qs_new,
    const float* __restrict__ rhs, const float* __restrict__ inv_Mp,
    const float* __restrict__ pin_n, float dt,
    const float* __restrict__ lam_d, const float* __restrict__ mu,
    const float* __restrict__ alpha, const float* __restrict__ b_xx,
    const float* __restrict__ b_xz, const float* __restrict__ b_zz,
    float decay, float gain, int viscous, const int* __restrict__ ibool,
    int nsls, const float* __restrict__ ptau, const float* __restrict__ mtau,
    const float* __restrict__ gsls_w, float* __restrict__ lrmem,
    float* __restrict__ gPt, float* __restrict__ gMt,             // GSLS frame (nsls > 0)
    float* __restrict__ lgv, float* __restrict__ ldp,
    float* __restrict__ gLd, float* __restrict__ gMu, float* __restrict__ gAl,
    float* __restrict__ gBxx, float* __restrict__ gBxz, float* __restrict__ gBzz, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    const float sxx = lsig[off], szz = lsig[Em + off], sxz = lsig[2 * Em + off];
    const float g0 = gvs[off], g1 = gvs[Em + off], g2 = gvs[2 * Em + off], g3 = gvs[3 * Em + off];
    const float div_v = g0 + g1, sd = sxx + szz;
    const float ld = lam_d[off], u_ = mu[off];
    float lg4[4] = {dt * (ld * sd + 2.f * u_ * sxx), dt * (ld * sd + 2.f * u_ * szz),
                    dt * u_ * sxz, dt * u_ * sxz};
    if (nsls) vs2d_gsls_adjoint(lsig, lrmem, Em, static_cast<int>(off), gvs, dt, nsls, ptau,
                                mtau, gsls_w, gPt, gMt, lg4);
    lgv[off] = lg4[0];
    lgv[Em + off] = lg4[1];
    lgv[2 * Em + off] = lg4[2];
    lgv[3 * Em + off] = lg4[3];
    gLd[off] += dt * sd * div_v;
    gMu[off] += dt * (2.f * sxx * g0 + 2.f * szz * g1 + sxz * (g2 + g3));
    const int g = ibool[off];
    const float dpe = pin_n[g] * (dt * inv_Mp[g] * rhs[g]);
    gAl[off] += -sd * dpe;
    atomicAdd(&ldp[g], -alpha[off] * sd);
    if (viscous) {
        const float rbx = lr[off], rbz = lr[Em + off];
        lr[off] = decay * rbx; lr[Em + off] = decay * rbz;
        lq[off] += gain * (b_xx[off] * rbx + b_xz[off] * rbz);
        lq[Em + off] += gain * (b_xz[off] * rbx + b_zz[off] * rbz);
        const float qx = qs_new[off], qz = qs_new[Em + off];
        gBxx[off] += gain * rbx * qx;
        gBxz[off] += gain * (rbx * qz + rbz * qx);
        gBzz[off] += gain * rbz * qz;
    }
}

__global__ void biot2d_pressure_T(
    float* __restrict__ ldp, const float* __restrict__ rhs, const float* __restrict__ inv_Mp,
    const float* __restrict__ pin_n, float dt, float* __restrict__ ldivq,
    float* __restrict__ gMp, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float l = pin_n[i] * ldp[i];
    ldivq[i] = dt * inv_Mp[i] * l;
    gMp[i] += dt * rhs[i] * l;
}

__global__ void biot2d_ifc_ac_T(
    const float* __restrict__ ldivq, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ lvx, float* __restrict__ lvz, int n_pts)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    const float l = -w[j] * ldivq[g];
    atomicAdd(&lvx[g], l * nx[j]);
    atomicAdd(&lvz[g], l * nz[j]);
}

__global__ void biot2d_stress_grad_T(
    float* __restrict__ lgv, const float* __restrict__ gvs, const float* __restrict__ ldivq,
    float* __restrict__ lmv,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ alpha, const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ lvx, float* __restrict__ lvz, float* __restrict__ gAl,
    int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const int g = ibool[off];
    // A(w alpha div v) enters rhs with -1: its adjoint is -ldivq at the node
    const float ladiv = -ldivq[g];
    const float wa = wJ[off] * alpha[off];
    const float div_v = gvs[off] + gvs[Em + off];
    gAl[off] += wJ[off] * div_v * ladiv;
    float lg[4] = {lgv[off] + wa * ladiv, lgv[Em + off] + wa * ladiv,
                   lgv[2 * Em + off], lgv[3 * Em + off]};
    float lraw[4];
    if (lmv) {
        const float ax = a_x[off], bx = b_x[off], az = a_z[off], bz = b_z[off];
        const float ca[4] = {ax, az, az, ax};
        const float cb[4] = {bx, bz, bz, bx};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float tot = lg[j] + lmv[j * Em + off];
            lraw[j] = lg[j] + cb[j] * tot;
            lmv[j * Em + off] = ca[j] * tot;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) lraw[j] = lg[j];
    }
    // (component c, direction j): (x,x) lraw0, (x,z) lraw2, (z,x) lraw3, (z,z) lraw1
    const float lc[4] = {lraw[0], lraw[2], lraw[3], lraw[1]};
    const float I4[4] = {Ixx[off], Ixz[off], Iex[off], Iez[off]};
    extern __shared__ float sh[];
    #pragma unroll
    for (int c = 0; c < 2; ++c)
        #pragma unroll
        for (int a = 0; a < 2; ++a)
            sh[(2 * c + a) * m + tid] = I4[2 * a + 0] * lc[2 * c + 0] + I4[2 * a + 1] * lc[2 * c + 1];
    __syncthreads();
    const int row = kz * ngll;
    float ax_ = 0.f, az_ = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx], Tz = D[q * ngll + kz];
        const int iz = q * ngll + kx;
        ax_ += Tx * sh[0 * m + row + q] + Tz * sh[1 * m + iz];
        az_ += Tx * sh[2 * m + row + q] + Tz * sh[3 * m + iz];
    }
    atomicAdd(&lvx[g], ax_);
    atomicAdd(&lvz[g], az_);
}

__global__ void biot2d_cpml_q_T(
    const float* __restrict__ a_n, const float* __restrict__ ldivq, float* __restrict__ lmq,
    float* __restrict__ lPq, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float l = ldivq[i];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float tot = l + lmq[j * ng + i];
        lmq[j * ng + i] = a_n[(2 * j) * ng + i] * tot;
        lPq[j * ng + i] = l + a_n[(2 * j + 1) * ng + i] * tot;
    }
}

__global__ void biot2d_div_q_T(
    const float* __restrict__ ldivq, const float* __restrict__ lPq, float* __restrict__ lq,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const bool layer = a_x && __syncthreads_or(
        a_x[off] != 0.f || a_z[off] != 0.f || b_x[off] != 0.f || b_z[off] != 0.f);
    extern __shared__ float sh[];
    const int g = ibool[off], row = kz * ngll;
    const float w = wJ[off];
    const float ixx = Ixx[off], ixz = Ixz[off], iex = Iex[off], iez = Iez[off];
    if (layer) {
        sh[tid] = lPq[g]; sh[m + tid] = lPq[n_global + g];
        __syncthreads();
        float dxx = 0.f, dxe = 0.f, dzx = 0.f, dze = 0.f;
        for (int k = 0; k < ngll; ++k) {
            const float Dx = D[kx * ngll + k], Dz = D[kz * ngll + k];
            dxx += Dx * sh[row + k];     dxe += Dz * sh[k * ngll + kx];
            dzx += Dx * sh[m + row + k]; dze += Dz * sh[m + k * ngll + kx];
        }
        lq[off] += w * (ixx * dxx + iex * dxe);
        lq[Em + off] += w * (ixz * dzx + iez * dze);
        return;
    }
    sh[tid] = ldivq[g];
    __syncthreads();
    float dxi = 0.f, deta = 0.f;
    for (int k = 0; k < ngll; ++k) {
        dxi += D[kx * ngll + k] * sh[row + k];
        deta += D[kz * ngll + k] * sh[k * ngll + kx];
    }
    lq[off] += w * (ixx * dxi + iex * deta);
    lq[Em + off] += w * (ixz * dxi + iez * deta);
}

template <bool TENSOR>
__global__ void biot2d_flux_T(
    const float* __restrict__ lvx, const float* __restrict__ lvz, float* __restrict__ lq,
    const float* __restrict__ s_f, const float* __restrict__ s_qs,
    const float* __restrict__ s_rs, const float* __restrict__ qmask,
    const float* __restrict__ inv_m, const float* __restrict__ inv_Mm,
    const int* __restrict__ slot_elem, const float* __restrict__ wr,
    const float* __restrict__ share, const float* __restrict__ dragE,
    const float* __restrict__ dragP, const float* __restrict__ pat_vx,
    const float* __restrict__ pat_vz, const float* __restrict__ pat_qx,
    const float* __restrict__ pat_qz, float wv, int velocity_mode, float dt,
    float* __restrict__ lf, float* __restrict__ lg,
    float* __restrict__ gE, float* __restrict__ gP, float* __restrict__ gwr,
    float* __restrict__ gshare, float* __restrict__ gim, float* __restrict__ gMm,
    float* __restrict__ gW, int kstep, int kmax, int m, long Em, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float wgrad = 0.f;
    if (i < ng) {
        const float im = inv_m[i];
        const float fx = s_f[i], fz = s_f[ng + i];
        float qs[2 * BIOT2D_KMAX], rs[2 * BIOT2D_KMAX], nw[2 * BIOT2D_KMAX];
        float lnw[2 * BIOT2D_KMAX], lqs[2 * BIOT2D_KMAX], lrs[2 * BIOT2D_KMAX];
        int el[BIOT2D_KMAX];
        for (int k = 0; k < kmax; ++k) {
            qs[k] = s_qs[static_cast<size_t>(i) * kmax + k];
            qs[kmax + k] = s_qs[(static_cast<size_t>(ng) + i) * kmax + k];
            rs[k] = s_rs[static_cast<size_t>(i) * kmax + k];
            rs[kmax + k] = s_rs[(static_cast<size_t>(ng) + i) * kmax + k];
            const int e = slot_elem[i * kmax + k];
            el[k] = e;
            lnw[k] = e >= 0 ? lq[e] : 0.f;
            lnw[kmax + k] = e >= 0 ? lq[Em + e] : 0.f;
        }
        const int K = TENSOR ? 2 * kmax : kmax;
        const float* Ei = dragE + static_cast<size_t>(i) * K * K;
        const float* Pi = dragP + static_cast<size_t>(i) * K * K;
        biot2d_advance<TENSOR>(Ei, Pi, qs, rs, nw, kmax);
        const float lx = lvx[i], lz = lvz[i];
        float g_im = 0.f;
        if (velocity_mode) {
            const float pv = pat_vx[i] * lx + pat_vz[i] * lz;
            wgrad += dt * im * pv;
            g_im += dt * wv * pv;
            float sqx = 0.f, sqz = 0.f;
            for (int k = 0; k < kmax; ++k) {
                if (el[k] < 0) continue;
                const float qm = qmask[el[k] / m];
                sqx += qm * lnw[k]; sqz += qm * lnw[kmax + k];
            }
            const float pq = pat_qx[i] * sqx + pat_qz[i] * sqz;
            wgrad += dt * inv_Mm[i] * pq;
            gMm[i] += dt * wv * pq;
        }
        // v' = v + (dt f - sum wr (new - qs)) inv_m
        const float ux = lx * im, uz = lz * im;
        float lfx = dt * ux, lfz = dt * uz;
        float dsx = 0.f, dsz = 0.f;
        for (int k = 0; k < kmax; ++k) {
            const float wk = wr[i * kmax + k];
            const float ex = nw[k] - qs[k], ez = nw[kmax + k] - qs[kmax + k];
            dsx += wk * ex; dsz += wk * ez;
            lnw[k] -= wk * ux; lnw[kmax + k] -= wk * uz;
            lqs[k] = wk * ux; lqs[kmax + k] = wk * uz;
            gwr[i * kmax + k] += -(ex * ux + ez * uz);
        }
        g_im += lx * (dt * fx - dsx) + lz * (dt * fz - dsz);
        // new = E qs + Psi rs
        if constexpr (TENSOR) {
            for (int b = 0; b < K; ++b) {
                float sq = 0.f, sr = 0.f;
                for (int a = 0; a < K; ++a) {
                    sq += Ei[a * K + b] * lnw[a];
                    sr += Pi[a * K + b] * lnw[a];
                    gE[static_cast<size_t>(i) * K * K + a * K + b] += lnw[a] * qs[b];
                    gP[static_cast<size_t>(i) * K * K + a * K + b] += lnw[a] * rs[b];
                }
                lqs[b] += sq; lrs[b] = sr;
            }
        } else {
            for (int b = 0; b < kmax; ++b) {
                float sqx = 0.f, sqz = 0.f, srx = 0.f, srz = 0.f;
                for (int a = 0; a < kmax; ++a) {
                    const float ea = Ei[a * kmax + b], pa = Pi[a * kmax + b];
                    sqx += ea * lnw[a]; sqz += ea * lnw[kmax + a];
                    srx += pa * lnw[a]; srz += pa * lnw[kmax + a];
                    gE[static_cast<size_t>(i) * K * K + a * kmax + b] +=
                        lnw[a] * qs[b] + lnw[kmax + a] * qs[kmax + b];
                    gP[static_cast<size_t>(i) * K * K + a * kmax + b] +=
                        lnw[a] * rs[b] + lnw[kmax + a] * rs[kmax + b];
                }
                lqs[b] += sqx; lqs[kmax + b] += sqz;
                lrs[b] = srx; lrs[kmax + b] = srz;
            }
        }
        // rs = to_slots(g) - wr f inv_m (+ share pat_q w_n)
        const float sqx = velocity_mode ? 0.f : pat_qx[i];
        const float sqz = velocity_mode ? 0.f : pat_qz[i];
        for (int k = 0; k < kmax; ++k) {
            const float wk = wr[i * kmax + k];
            lfx -= wk * lrs[k] * im; lfz -= wk * lrs[kmax + k] * im;
            gwr[i * kmax + k] += -(lrs[k] * fx + lrs[kmax + k] * fz) * im;
            g_im += -wk * (lrs[k] * fx + lrs[kmax + k] * fz);
            if (!velocity_mode) {
                gshare[i * kmax + k] += (lrs[k] * sqx + lrs[kmax + k] * sqz) * wv;
                wgrad += share[i * kmax + k] * (lrs[k] * sqx + lrs[kmax + k] * sqz);
            }
            const int e = el[k];
            if (e < 0) continue;
            lg[e] = lrs[k]; lg[Em + e] = lrs[kmax + k];
            lq[e] = lqs[k]; lq[Em + e] = lqs[kmax + k];     // the adjoint of the old q
        }
        lf[i] = lfx; lf[ng + i] = lfz;
        gim[i] += g_im;
    }
    const float bs = sem2d_block_sum(wgrad);
    if (threadIdx.x == 0) atomicAdd(&gW[kstep], bs);
}
biot2d_flux_T_fn const biot2d_flux_T_by_flag[2] = {
    &biot2d_flux_T<false>, &biot2d_flux_T<true>};

__global__ void biot2d_g_T(
    const float* __restrict__ lg, float* __restrict__ lr, float* __restrict__ lgp,
    const float* __restrict__ wJ, const float* __restrict__ qmask, int m, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    const float w = wJ[off] * qmask[off / m];
    const float gx = lg[off], gz = lg[Em + off];
    lr[off] += w * gx; lr[Em + off] += w * gz;
    lgp[off] = -w * gx; lgp[Em + off] = -w * gz;
}

__global__ void biot2d_force_T(
    const float* __restrict__ lf, float* __restrict__ lRx, float* __restrict__ lRz,
    const float* __restrict__ pat_vx, const float* __restrict__ pat_vz, int load,
    float* __restrict__ gW, int kstep, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float wgrad = 0.f;
    if (i < ng) {
        const float fx = lf[i], fz = lf[ng + i];
        lRx[i] = -fx; lRz[i] = -fz;
        if (load) wgrad = pat_vx[i] * fx + pat_vz[i] * fz;
    }
    const float bs = sem2d_block_sum(wgrad);
    if (threadIdx.x == 0 && load) atomicAdd(&gW[kstep], bs);
}

__global__ void biot2d_ifc_el_T(
    const float* __restrict__ lf, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz, float* __restrict__ lp,
    int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    atomicAdd(&lp[g], w[j] * (nx[j] * lf[g] + nz[j] * lf[ng + g]));
}

__global__ void biot2d_grad_p_T(
    const float* __restrict__ lgp, float* __restrict__ lmp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ lp, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    float lx = lgp[off], lz = lgp[Em + off];
    if (lmp) {
        const float tx = lx + lmp[off], tz = lz + lmp[Em + off];
        lmp[off] = a_x[off] * tx; lmp[Em + off] = a_z[off] * tz;
        lx = lx + b_x[off] * tx; lz = lz + b_z[off] * tz;
    }
    extern __shared__ float sh[];
    sh[tid] = Ixx[off] * lx + Ixz[off] * lz;          // the xi-derivative's adjoint
    sh[m + tid] = Iex[off] * lx + Iez[off] * lz;      // the eta-derivative's adjoint
    __syncthreads();
    const int row = kz * ngll;
    float s = 0.f;
    for (int q = 0; q < ngll; ++q)
        s += D[q * ngll + kx] * sh[row + q] + D[q * ngll + kz] * sh[m + q * ngll + kx];
    atomicAdd(&lp[ibool[off]], s);
}

__global__ void biot2d_inject_p_T(
    float* __restrict__ lp, const float* __restrict__ inv_M1, const float* __restrict__ pin_n,
    const float* __restrict__ pat_p, float wv, float dt, float* __restrict__ gM1,
    float* __restrict__ gW, int kstep, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float wgrad = 0.f;
    if (i < ng) {
        const float l = pin_n[i] * lp[i];
        lp[i] = l;
        gM1[i] += l * dt * (pat_p[i] * wv);
        wgrad = l * dt * inv_M1[i] * pat_p[i];
    }
    const float bs = sem2d_block_sum(wgrad);
    if (threadIdx.x == 0) atomicAdd(&gW[kstep], bs);
}

} // namespace sem2d

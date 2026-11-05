// sweep.sem CUDA (impl='c') backend -- the 2-D velocity-stress SEM kernels:
// ElasticVSSEM2DDeformed and its VTI / TTI / visco-elastic subclasses on Quad9.
//
// The 3-D velocity-stress kernels on Hex27 (sem3d/kernels.cuh, the *_deformed
// set) with two components. Element-local: sig (3, E*m: xx zz xz), the velocity-
// gradient C-PML memories mvel (4, E*m: vx_x vz_z vx_z vz_x, stretched along
// x z z x) and the GSLS memories; nodal: v, the stress divergence R and, where the
// C-PML reaches, the divergence split by physical direction P with its memories
// msig (4, n_global; slot 2i + j = (R_x, x) (R_x, z) (R_z, x) (R_z, z)). An
// element the C-PML reaches assembles P instead of R, so P is complete at every
// node the layer reaches (see sem3d_vs_sigma_div). I_aj = d xi_a / d x_j: ixx =
// dxi/dx, ixz = dxi/dz, iex = deta/dx, iez = deta/dz. Element arrays are (E, z,
// x), x fastest; xi runs along x and eta along z. One shot (B = 1).
#pragma once

namespace sem2d {

// The 2-D constitutive seam. NC = 0: isotropic (lam, mu, lam2mu); 4 (VTI): c11 c13
// c33 c55; 6 (TTI): c11 c13 c15 c33 c35 c55 -- ElasticVTISEM2DDeformed /
// ElasticTTISEM2DDeformed's bs_model_attrs, (NC, E*m) -- on the strain rates
// e = (vx_x, vz_z, vx_z + vz_x) of the stretched gradients gv, with the eager
// classes' sums.
template <int NC>
__device__ __forceinline__ void vs2d_stress_increment(
    float* __restrict__ sig, long Em, int off, const float gv[4], float dt,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ cij)
{
    if constexpr (NC == 0) {
        const float l = lam[off], u_ = mu[off], l2 = lam2mu[off];
        sig[0 * Em + off] += dt * (l2 * gv[0] + l * gv[1]);
        sig[1 * Em + off] += dt * (l * gv[0] + l2 * gv[1]);
        sig[2 * Em + off] += dt * u_ * (gv[2] + gv[3]);
    } else if constexpr (NC == 4) {
        const float c11 = cij[0 * Em + off], c13 = cij[1 * Em + off];
        const float c33 = cij[2 * Em + off], c55 = cij[3 * Em + off];
        sig[0 * Em + off] += dt * (c11 * gv[0] + c13 * gv[1]);
        sig[1 * Em + off] += dt * (c13 * gv[0] + c33 * gv[1]);
        sig[2 * Em + off] += dt * c55 * (gv[2] + gv[3]);
    } else {
        static_assert(NC == 6, "NC is 0, 4 or 6");
        const float c11 = cij[0 * Em + off], c13 = cij[1 * Em + off], c15 = cij[2 * Em + off];
        const float c33 = cij[3 * Em + off], c35 = cij[4 * Em + off], c55 = cij[5 * Em + off];
        const float shear = gv[2] + gv[3];
        sig[0 * Em + off] += dt * (c11 * gv[0] + c13 * gv[1]) + dt * c15 * shear;
        sig[1 * Em + off] += dt * (c13 * gv[0] + c33 * gv[1]) + dt * c35 * shear;
        sig[2 * Em + off] += dt * c55 * shear + dt * (c15 * gv[0] + c35 * gv[1]);
    }
}

// The adjoint of vs2d_stress_increment: lg (the stretched gradients' adjoint,
// slot order vx_x vz_z vx_z vz_x) and the moduli's gradients against this step's
// saved stretched gradients gvs (the save_gv slice).
template <int NC>
__device__ __forceinline__ void vs2d_stress_adjoint(
    const float* __restrict__ lsig, const float* __restrict__ gvs, long Em, int off,
    float dt, const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ cij,
    float* __restrict__ gLam, float* __restrict__ gMu, float* __restrict__ gL2m,
    float* __restrict__ gC, float lg[4])
{
    const float lxx = lsig[0 * Em + off], lzz = lsig[1 * Em + off], lxz = lsig[2 * Em + off];
    const float g0 = gvs[0 * Em + off], g1 = gvs[1 * Em + off];
    const float g2 = gvs[2 * Em + off], g3 = gvs[3 * Em + off];
    if constexpr (NC == 0) {
        const float l = lam[off], u_ = mu[off], l2 = lam2mu[off];
        lg[0] = dt * (l2 * lxx + l * lzz);
        lg[1] = dt * (l * lxx + l2 * lzz);
        lg[2] = dt * u_ * lxz;  lg[3] = dt * u_ * lxz;
        gL2m[off] += dt * (g0 * lxx + g1 * lzz);
        gLam[off] += dt * (g1 * lxx + g0 * lzz);
        gMu[off]  += dt * (g2 + g3) * lxz;
    } else {
        const float e0 = g0, e1 = g1, e2 = g2 + g3;     // Voigt xx, zz, xz
        float le0, le1, le2;
        if constexpr (NC == 4) {
            const float c11 = cij[0 * Em + off], c13 = cij[1 * Em + off];
            const float c33 = cij[2 * Em + off], c55 = cij[3 * Em + off];
            le0 = dt * (c11 * lxx + c13 * lzz);
            le1 = dt * (c13 * lxx + c33 * lzz);
            le2 = dt * c55 * lxz;
            gC[0 * Em + off] += dt * (lxx * e0);
            gC[1 * Em + off] += dt * (lxx * e1 + lzz * e0);
            gC[2 * Em + off] += dt * (lzz * e1);
            gC[3 * Em + off] += dt * (lxz * e2);
        } else {
            static_assert(NC == 6, "NC is 0, 4 or 6");
            const float c11 = cij[0 * Em + off], c13 = cij[1 * Em + off], c15 = cij[2 * Em + off];
            const float c33 = cij[3 * Em + off], c35 = cij[4 * Em + off], c55 = cij[5 * Em + off];
            le0 = dt * (c11 * lxx + c13 * lzz + c15 * lxz);
            le1 = dt * (c13 * lxx + c33 * lzz + c35 * lxz);
            le2 = dt * (c15 * lxx + c35 * lzz + c55 * lxz);
            gC[0 * Em + off] += dt * (lxx * e0);
            gC[1 * Em + off] += dt * (lxx * e1 + lzz * e0);
            gC[2 * Em + off] += dt * (lxx * e2 + lxz * e0);
            gC[3 * Em + off] += dt * (lzz * e1);
            gC[4 * Em + off] += dt * (lzz * e2 + lxz * e1);
            gC[5 * Em + off] += dt * (lxz * e2);
        }
        lg[0] = le0; lg[1] = le1; lg[2] = le2; lg[3] = le2;
    }
}

// The 2-D GSLS memories (ViscoElasticVSSEM2DDeformed): per mechanism l,
// r+ = a_l r - c_l F_l, F_l = (tp - 2 ts vz_z, tp - 2 ts vx_x, ts e_xz) with
// tp = ptau_l (vx_x + vz_z), ts = mtau_l, and the stress takes dt/2 sum (r+ + r).
// rmem (3 nsls, E*m), xx zz xz per mechanism; w = (a_0 .., c_0 ..).
__device__ __forceinline__ void vs2d_gsls_step(
    float* __restrict__ sig, float* __restrict__ rmem, long Em, int off,
    const float gv[4], float dt, int nsls, const float* __restrict__ ptau,
    const float* __restrict__ mtau, const float* __restrict__ w)
{
    const float theta = gv[0] + gv[1], exz = gv[2] + gv[3];
    float sums[3] = {0.f, 0.f, 0.f};
    for (int l = 0; l < nsls; ++l) {
        const float a = w[l], c = w[nsls + l];
        const float tp = ptau[l * Em + off] * theta, ts = mtau[l * Em + off];
        const float F[3] = {tp - 2.f * ts * gv[1], tp - 2.f * ts * gv[0], ts * exz};
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            float* r = rmem + static_cast<long>(3 * l + k) * Em + off;
            const float r0 = *r, rn = a * r0 - c * F[k];
            sums[k] += rn + r0;
            *r = rn;
        }
    }
    const float h = 0.5f * dt;
    #pragma unroll
    for (int k = 0; k < 3; ++k) sig[k * Em + off] += h * sums[k];
}

// The adjoint of vs2d_gsls_step (see vs3d_gsls_adjoint).
__device__ __forceinline__ void vs2d_gsls_adjoint(
    const float* __restrict__ lsig, float* __restrict__ lrmem, long Em, int off,
    const float* __restrict__ gvs, float dt, int nsls, const float* __restrict__ ptau,
    const float* __restrict__ mtau, const float* __restrict__ w,
    float* __restrict__ gPt, float* __restrict__ gMt, float lg[4])
{
    const float h = 0.5f * dt;
    float ls[3], g[4];
    #pragma unroll
    for (int k = 0; k < 3; ++k) ls[k] = lsig[k * Em + off];
    #pragma unroll
    for (int j = 0; j < 4; ++j) g[j] = gvs[j * Em + off];
    const float theta = g[0] + g[1], exz = g[2] + g[3];
    for (int l = 0; l < nsls; ++l) {
        const float a = w[l], c = w[nsls + l];
        const float pt = ptau[l * Em + off], mt = mtau[l * Em + off];
        float mu[3];
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            float* lr = lrmem + static_cast<long>(3 * l + k) * Em + off;
            const float lam = *lr + h * ls[k];
            *lr = a * lam + h * ls[k];
            mu[k] = -c * lam;
        }
        const float sn = mu[0] + mu[1];
        lg[0] += pt * sn - 2.f * mt * mu[1];
        lg[1] += pt * sn - 2.f * mt * mu[0];
        lg[2] += mt * mu[2];  lg[3] += mt * mu[2];
        gPt[l * Em + off] += sn * theta;
        gMt[l * Em + off] += -2.f * (mu[0] * g[1] + mu[1] * g[0]) + mu[2] * exz;
    }
}

// V: the stress divergence R_i = sum_a D_a^T [w_J sum_j sig_ij I_aj]; an element
// the C-PML reaches assembles it split by physical direction into P (slot 2i + j),
// one stress row at a time. Shared 4m.
static __global__ void sem2d_vs_sigma_div(
    const float* __restrict__ sig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ Rx, float* __restrict__ Rz, float* __restrict__ P,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    // block-uniform: the barriers below stay legal
    const bool layer = a_x && __syncthreads_or(
        a_x[off] != 0.f || a_z[off] != 0.f || b_x[off] != 0.f || b_z[off] != 0.f);

    extern __shared__ float sh[];            // 4 flux arrays
    const float sxx = sig[0 * Em + off], szz = sig[1 * Em + off], sxz = sig[2 * Em + off];
    const float s4[4] = {sxx, sxz, sxz, szz};                        // (row i, column j)
    const float w = wJ[off];
    const float I4[4] = {Ixx[off], Ixz[off], Iex[off], Iez[off]};   // (row a, column j)
    const int row = kz * ngll;
    const int g = ibool[off];
    if (layer) {
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {                 // slots (j, a): w sig_ij I_aj
                const float ws = w * s4[2 * i + j];
                sh[(2 * j + 0) * m + tid] = ws * I4[j];
                sh[(2 * j + 1) * m + tid] = ws * I4[2 + j];
            }
            __syncthreads();
            float r2[2] = {0.f, 0.f};
            for (int q = 0; q < ngll; ++q) {
                const float Tx = D[q * ngll + kx], Tz = D[q * ngll + kz];
                const int iz = q * ngll + kx;
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    r2[j] += Tx * sh[(2 * j + 0) * m + row + q] + Tz * sh[(2 * j + 1) * m + iz];
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) atomicAdd(&P[(2 * i + j) * n_global + g], r2[j]);
            __syncthreads();                              // the next row overwrites the fluxes
        }
        return;
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)                           // slots (i, a): w sum_j sig_ij I_aj
        #pragma unroll
        for (int a = 0; a < 2; ++a)
            sh[(2 * i + a) * m + tid] = w * (s4[2 * i + 0] * I4[2 * a + 0]
                                             + s4[2 * i + 1] * I4[2 * a + 1]);
    __syncthreads();
    float rx = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx], Tz = D[q * ngll + kz];
        const int iz = q * ngll + kx;
        rx += Tx * sh[0 * m + row + q] + Tz * sh[1 * m + iz];
        rz += Tx * sh[2 * m + row + q] + Tz * sh[3 * m + iz];
    }
    atomicAdd(&Rx[g], rx);
    atomicAdd(&Rz[g], rz);
}

// C: the C-PML on the divergence by direction, per node: psi_j = a psi_j + b P_j
// (the slot's direction j % 2), R_i += its two slots of P + psi. a_n (4, n_global):
// a_x b_x a_z b_z. A node the layer does not reach has P = psi = 0.
static __global__ void sem2d_vs_cpml_nodal(
    const float* __restrict__ a_n, const float* __restrict__ P,
    float* __restrict__ msig, float* __restrict__ Rx, float* __restrict__ Rz, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    float r[2] = {0.f, 0.f};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = j % 2;
        const float p = P[j * ng + i];
        const float psi = a_n[(2 * d) * ng + i] * msig[j * ng + i] + a_n[(2 * d + 1) * ng + i] * p;
        msig[j * ng + i] = psi;
        r[j / 2] += p + psi;
    }
    Rx[i] += r[0]; Rz[i] += r[1];
}

// N: v += dt invM (pat wav[n] - R), two components + optional v-history save.
static __global__ void sem2d_vs_vel_update(
    float* __restrict__ vx, float* __restrict__ vz,
    const float* __restrict__ Rx, const float* __restrict__ Rz,
    const float* __restrict__ inv_M, const float* __restrict__ pat_x,
    const float* __restrict__ pat_z, const float* __restrict__ wavelet, int n, float dt,
    float* __restrict__ s_vx, float* __restrict__ s_vz, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float f = dt * inv_M[i], wv = wavelet[n];
    const float nvx = vx[i] + f * (pat_x[i] * wv - Rx[i]);
    const float nvz = vz[i] + f * (pat_z[i] * wv - Rz[i]);
    vx[i] = nvx; vz[i] = nvz;
    if (s_vx) { s_vx[i] = nvx; s_vz[i] = nvz; }
}

// S: the reference derivatives of the new v, the physical gradients d v_c / d x_j
// = sum_a r_ca I_aj (slots vx_x vz_z vx_z vz_x), the C-PML memories and stretch,
// the optional record of the stretched gradients, then the stress update
// (vs2d_stress_increment<NC>, plus the GSLS memories when nsls > 0). Shared 2m.
template <int NC>
static __global__ void sem2d_vs_stress_update(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ sig, float* __restrict__ mvel,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu,
    const float* __restrict__ cij,    // (NC, E*m) when NC > 0
    int nsls, const float* __restrict__ ptau, const float* __restrict__ mtau,
    const float* __restrict__ gsls_w, float* __restrict__ rmem,   // nsls = 0: elastic
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D, float dt,
    float* __restrict__ save_gv,      // (4, E*m) slice for this step; may be null
    int ngll, int E)
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
    float r[4] = {0.f, 0.f, 0.f, 0.f};   // (vx, xi) (vx, eta) (vz, xi) (vz, eta)
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx * ngll + q], Dz = D[kz * ngll + q];
        const int iz = q * ngll + kx;
        r[0] += Dx * s_vx[row + q]; r[1] += Dz * s_vx[iz];
        r[2] += Dx * s_vz[row + q]; r[3] += Dz * s_vz[iz];
    }
    const float ixx = Ixx[off], ixz = Ixz[off], iex = Iex[off], iez = Iez[off];
    float gv[4] = {ixx * r[0] + iex * r[1],      // vx_x
                   ixz * r[2] + iez * r[3],      // vz_z
                   ixz * r[0] + iez * r[1],      // vx_z
                   ixx * r[2] + iex * r[3]};     // vz_x
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], az = a_z[off], bz = b_z[off];
        const float ca[4] = {ax, az, az, ax};
        const float cb[4] = {bx, bz, bz, bx};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float mm = ca[j] * mvel[j * Em + off] + cb[j] * gv[j];
            mvel[j * Em + off] = mm;
            gv[j] = gv[j] + mm;
        }
    }
    if (save_gv) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) save_gv[j * Em + off] = gv[j];
    }
    vs2d_stress_increment<NC>(sig, Em, off, gv, dt, lam, mu, lam2mu, cij);
    if (nsls) vs2d_gsls_step(sig, rmem, Em, off, gv, dt, nsls, ptau, mtau, gsls_w);
}

// S^T: the moduli's adjoint and imaging, the GSLS memories' adjoint, the C-PML
// memory recursion, then the transpose of the metric contraction and the D^T
// scatter into lv. Shared 4m.
template <int NC>
static __global__ void sem2d_vs_stress_update_T(
    const float* __restrict__ lsig, const float* __restrict__ save_gv,
    float* __restrict__ lmvel,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu,
    const float* __restrict__ cij,    // (NC, E*m) when NC > 0
    int nsls, const float* __restrict__ ptau, const float* __restrict__ mtau,
    const float* __restrict__ gsls_w, float* __restrict__ lrmem,  // nsls = 0: elastic
    float* __restrict__ gPt, float* __restrict__ gMt,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D, float dt,
    float* __restrict__ lvx, float* __restrict__ lvz,
    float* __restrict__ gLam, float* __restrict__ gMu, float* __restrict__ gL2m,
    float* __restrict__ gC,           // (NC, E*m) when NC > 0
    int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, kz = tid / ngll;
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;

    float lg[4];
    vs2d_stress_adjoint<NC>(lsig, save_gv, Em, off, dt, lam, mu, lam2mu, cij,
                            gLam, gMu, gL2m, gC, lg);
    if (nsls) vs2d_gsls_adjoint(lsig, lrmem, Em, off, save_gv, dt, nsls, ptau, mtau,
                                gsls_w, gPt, gMt, lg);
    float lraw[4];
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], az = a_z[off], bz = b_z[off];
        const float ca[4] = {ax, az, az, ax};
        const float cb[4] = {bx, bz, bz, bx};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float tot = lg[j] + lmvel[j * Em + off];
            lraw[j] = lg[j] + cb[j] * tot;
            lmvel[j * Em + off] = ca[j] * tot;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) lraw[j] = lg[j];
    }
    // the raw gradients' adjoints per (component c, direction j):
    // (x,x) lraw0, (x,z) lraw2, (z,x) lraw3, (z,z) lraw1
    const float lc[4] = {lraw[0], lraw[2], lraw[3], lraw[1]};
    const float I4[4] = {Ixx[off], Ixz[off], Iex[off], Iez[off]};
    extern __shared__ float sh[];
    #pragma unroll
    for (int c = 0; c < 2; ++c)                            // slots (c, a)
        #pragma unroll
        for (int a = 0; a < 2; ++a)
            sh[(2 * c + a) * m + tid] = I4[2 * a + 0] * lc[2 * c + 0]
                                      + I4[2 * a + 1] * lc[2 * c + 1];
    __syncthreads();
    const int row = kz * ngll;
    float ax_ = 0.f, az_ = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx], Tz = D[q * ngll + kz];
        const int iz = q * ngll + kx;
        ax_ += Tx * sh[0 * m + row + q] + Tz * sh[1 * m + iz];
        az_ += Tx * sh[2 * m + row + q] + Tz * sh[3 * m + iz];
    }
    const int g = ibool[off];
    atomicAdd(&lvx[g], ax_);
    atomicAdd(&lvz[g], az_);
}

// N^T (nodal): grad_inv_M through the update identity, lR = -dt invM lv (lv
// carries on unchanged to the previous step as the identity part).
static __global__ void sem2d_vs_vel_update_T(
    const float* __restrict__ lvx, const float* __restrict__ lvz,
    const float* __restrict__ v_k_x, const float* __restrict__ v_k_z,     // saved v^k
    const float* __restrict__ v_km1_x, const float* __restrict__ v_km1_z, // v^{k-1}, may be null
    const float* __restrict__ inv_M, float dt,
    float* __restrict__ lRx, float* __restrict__ lRz, float* __restrict__ gradM, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float im = inv_M[i];
    const float f = -dt * im;
    lRx[i] = f * lvx[i];  lRz[i] = f * lvz[i];
    if (im != 0.f) {
        const float px = v_km1_x ? v_km1_x[i] : 0.f;
        const float pz = v_km1_z ? v_km1_z[i] : 0.f;
        gradM[i] += (lvx[i] * (v_k_x[i] - px) + lvz[i] * (v_k_z[i] - pz)) / im;
    }
}

// C^T (nodal): tot_j = lR_i + lmsig_j, lmsig_j = a tot_j in place, and the slot's
// divergence adjoint lP_j = lR_i + b tot_j (P enters R directly and through psi).
static __global__ void sem2d_vs_cpml_nodal_T(
    const float* __restrict__ a_n, const float* __restrict__ lRx,
    const float* __restrict__ lRz, float* __restrict__ lmsig, float* __restrict__ lP, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float lr[2] = {lRx[i], lRz[i]};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = j % 2;
        const float tot = lr[j / 2] + lmsig[j * ng + i];
        lmsig[j * ng + i] = a_n[(2 * d) * ng + i] * tot;
        lP[j * ng + i] = lr[j / 2] + a_n[(2 * d + 1) * ng + i] * tot;
    }
}

// V^T (element, no atomics): lsig_ij += w_J sum_a I_aj (D_a lR_i), or with each
// slot's own adjoint lP_(2i+j) in an element the C-PML reaches. Shared 4m.
static __global__ void sem2d_vs_sigma_div_T(
    const float* __restrict__ lRx, const float* __restrict__ lRz,
    const float* __restrict__ lP, float* __restrict__ lsig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
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
    const int g = ibool[off];
    const int row = kz * ngll;
    const float w = wJ[off];
    const float I4[4] = {Ixx[off], Ixz[off], Iex[off], Iez[off]};
    float ls[4];                                          // slots (i, j)
    if (layer) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) sh[j * m + tid] = lP[j * n_global + g];
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            float d0 = 0.f, d1 = 0.f;
            for (int q = 0; q < ngll; ++q) {
                const float Dx = D[kx * ngll + q], Dz = D[kz * ngll + q];
                d0 += Dx * sh[s * m + row + q]; d1 += Dz * sh[s * m + q * ngll + kx];
            }
            const int j = s % 2;
            ls[s] = w * (I4[j] * d0 + I4[2 + j] * d1);
        }
    } else {
        float* s_x = sh; float* s_z = sh + m;
        s_x[tid] = lRx[g]; s_z[tid] = lRz[g];
        __syncthreads();
        float r[4] = {0.f, 0.f, 0.f, 0.f};               // (component i, axis a)
        for (int q = 0; q < ngll; ++q) {
            const float Dx = D[kx * ngll + q], Dz = D[kz * ngll + q];
            const int iz = q * ngll + kx;
            r[0] += Dx * s_x[row + q]; r[1] += Dz * s_x[iz];
            r[2] += Dx * s_z[row + q]; r[3] += Dz * s_z[iz];
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                ls[2 * i + j] = w * (I4[j] * r[2 * i + 0] + I4[2 + j] * r[2 * i + 1]);
    }
    // slot (i, j) -> sigma component: (x,x) xx, (x,z) xz, (z,x) xz, (z,z) zz
    lsig[0 * Em + off] += ls[0];
    lsig[1 * Em + off] += ls[3];
    lsig[2 * Em + off] += ls[1] + ls[2];
}

// Launch KERNEL<NC> for the input's stiffness count (0 isotropic, 4 VTI, 6 TTI).
#define SEM2D_VS_LAUNCH_NC(nc, KERNEL, grid, block, shmem, stream, ...)             \
    do {                                                                            \
        if ((nc) == 6) KERNEL<6><<<grid, block, shmem, stream>>>(__VA_ARGS__);      \
        else if ((nc) == 4) KERNEL<4><<<grid, block, shmem, stream>>>(__VA_ARGS__); \
        else KERNEL<0><<<grid, block, shmem, stream>>>(__VA_ARGS__);                \
    } while (0)

} // namespace sem2d

// sweep.sem CUDA (impl='c') backend -- the 3-D Biot SEM kernels: BiotVSSEM3DDeformed
// (sweep.sem.equations.biot_vs3d_deformed) on Hex27. The 2-D kernels'
// (sem2d/biot2d_kernels.cuh) sub-steps with y, in the same order; the dimension-free
// ones (the pressure injection, the pressure update and their transposes) are the
// 2-D kernels themselves, the stress divergence that of ElasticVSSEM3DDeformed
// (sem3d_vs_sigma_div_deformed, sem3d_vs_cpml_nodal). What differs besides the
// components: the flux copies of a node are grouped by their number k (1 inside an
// element ... 8 at a corner), one launch per group with K = k, the per-copy arrays
// in the eager class's perm order. Element arrays (E, z, y, x), x fastest; I9[3a + j]
// = d xi_a / d x_j; the nine velocity gradients in ElasticVSSEM3D's slot order
// (vx_x vy_y vz_z vy_x vx_y vz_x vx_z vz_y vy_z). One shot (B = 1).
#pragma once

namespace sem3d {

// K2: grad p at the element nodes (x, y, z), stretched by the memories mp (null:
// bare). Shared m.
static __global__ void biot3d_grad_p(
    const float* __restrict__ p, float* __restrict__ gp, float* __restrict__ mp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    extern __shared__ float sh[];
    sh[tid] = p[ibool[off]];
    __syncthreads();
    const int zy = (kz * ngll + ky) * ngll;
    float r0 = 0.f, r1 = 0.f, r2 = 0.f;
    for (int q = 0; q < ngll; ++q) {
        r0 += D[kx * ngll + q] * sh[zy + q];
        r1 += D[ky * ngll + q] * sh[(kz * ngll + q) * ngll + kx];
        r2 += D[kz * ngll + q] * sh[(q * ngll + ky) * ngll + kx];
    }
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
    float g[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j) g[j] = I9[j] * r0 + I9[3 + j] * r1 + I9[6 + j] * r2;
    if (mp) {
        const float a3[3] = {a_x[off], a_y[off], a_z[off]};
        const float b3[3] = {b_x[off], b_y[off], b_z[off]};
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float mm = a3[j] * mp[j * Em + off] + b3[j] * g[j];
            mp[j * Em + off] = mm;
            g[j] = g[j] + mm;
        }
    }
    #pragma unroll
    for (int j = 0; j < 3; ++j) gp[j * Em + off] = g[j];
}

// K4: f = -R (+ the momentum load, "load" mode).
static __global__ void biot3d_force(
    const float* __restrict__ Rx, const float* __restrict__ Ry, const float* __restrict__ Rz,
    float* __restrict__ f, const float* __restrict__ pvx, const float* __restrict__ pvy,
    const float* __restrict__ pvz, float wv, int load, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    float fx = -Rx[i], fy = -Ry[i], fz = -Rz[i];
    if (load) { fx += pvx[i] * wv; fy += pvy[i] * wv; fz += pvz[i] * wv; }
    f[i] = fx; f[ng + i] = fy; f[2 * ng + i] = fz;
}

// K4 (points): a fluid's pressure on the solid, f += w p n.
static __global__ void biot3d_ifc_el(
    const float* __restrict__ p, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ ny, const float* __restrict__ nz,
    float* __restrict__ f, int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    const float pw = w[j] * p[g];
    atomicAdd(&f[g], pw * nx[j]);
    atomicAdd(&f[ng + g], pw * ny[j]);
    atomicAdd(&f[2 * ng + g], pw * nz[j]);
}

// K5: one group of nodes, K copies each: the frozen forces r_s, the copies' exact
// step new = E qs + Psi r_s (each component), v, the copies written back, the
// "velocity" jumps; f, qs and r_s recorded (per-copy arrays in perm order).
template <int K>
static __global__ void biot3d_flux(
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ gp,
    const float* __restrict__ f, const float* __restrict__ wJ, const float* __restrict__ qmask,
    const float* __restrict__ inv_m, const float* __restrict__ inv_Mm,
    const int* __restrict__ perm, const int* __restrict__ node_order,
    const float* __restrict__ wr, const float* __restrict__ share,
    const float* __restrict__ dragE, const float* __restrict__ dragP,
    const float* __restrict__ pvx, const float* __restrict__ pvy, const float* __restrict__ pvz,
    const float* __restrict__ pqx, const float* __restrict__ pqy, const float* __restrict__ pqz,
    float wv, int velocity_mode, float dt,
    float* __restrict__ s_f, float* __restrict__ s_qs, float* __restrict__ s_rs,
    int n_k, int coff, long eoff, int noff, int m, long Em, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_k) return;
    const int node = node_order[noff + j];
    const int base = coff + j * K;
    const float* Ei = dragE + eoff + static_cast<long>(j) * K * K;
    const float* Pi = dragP + eoff + static_cast<long>(j) * K * K;
    const float im = inv_m[node];
    const float fd[3] = {f[node], f[ng + node], f[2 * ng + node]};
    float sq[3] = {0.f, 0.f, 0.f};
    if (!velocity_mode) { sq[0] = pqx[node] * wv; sq[1] = pqy[node] * wv; sq[2] = pqz[node] * wv; }
    float qs[3][K], rs[3][K], nw[3][K];
    int el[K];
    #pragma unroll
    for (int c = 0; c < K; ++c) {
        const int e = perm[base + c];
        el[c] = e;
        const float w = wJ[e] * qmask[e / m];
        const float wk = wr[base + c], sk = share[base + c];
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            qs[d][c] = q[d * Em + e];
            rs[d][c] = w * (r[d * Em + e] - gp[d * Em + e]) - wk * (fd[d] * im) + sk * sq[d];
        }
    }
    #pragma unroll
    for (int d = 0; d < 3; ++d)
        #pragma unroll
        for (int a = 0; a < K; ++a) {
            float s = 0.f;
            #pragma unroll
            for (int b = 0; b < K; ++b) s += Ei[a * K + b] * qs[d][b] + Pi[a * K + b] * rs[d][b];
            nw[d][a] = s;
        }
    float dv[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int c = 0; c < K; ++c) {
        const float wk = wr[base + c];
        #pragma unroll
        for (int d = 0; d < 3; ++d) dv[d] += wk * (nw[d][c] - qs[d][c]);
    }
    float* v3[3] = {vx, vy, vz};
    const float* pv[3] = {pvx, pvy, pvz};
    const float* pq[3] = {pqx, pqy, pqz};
    float jq[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        float nv = v3[d][node] + (dt * fd[d] - dv[d]) * im;
        if (velocity_mode) {
            nv += dt * im * (pv[d][node] * wv);
            jq[d] = dt * inv_Mm[node] * (pq[d][node] * wv);
        }
        v3[d][node] = nv;
    }
    #pragma unroll
    for (int c = 0; c < K; ++c) {
        const int e = el[c];
        const float qm = qmask[e / m];
        #pragma unroll
        for (int d = 0; d < 3; ++d) q[d * Em + e] = nw[d][c] + jq[d] * qm;
    }
    if (s_f) {
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            s_f[d * ng + node] = fd[d];
            #pragma unroll
            for (int c = 0; c < K; ++c) {
                s_qs[d * Em + base + c] = qs[d][c];
                s_rs[d * Em + base + c] = rs[d][c];
            }
        }
    }
}

// K6: grad v (9 slots) stretched by its memories, the frame's stress increment and
// A(w alpha div v). Shared 3m.
static __global__ void biot3d_stress(
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    float* __restrict__ sig, float* __restrict__ mv,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ lam_d, const float* __restrict__ mu,
    const float* __restrict__ alpha, const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D, float dt,
    float* __restrict__ adiv, float* __restrict__ save_gv, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    extern __shared__ float sh[];
    float* s_vx = sh; float* s_vy = sh + m; float* s_vz = sh + 2 * m;
    const int g0 = ibool[off];
    s_vx[tid] = vx[g0]; s_vy[tid] = vy[g0]; s_vz[tid] = vz[g0];
    __syncthreads();
    const int zy = (kz * ngll + ky) * ngll;
    float r[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx * ngll + q], Dy = D[ky * ngll + q], Dz = D[kz * ngll + q];
        const int iy = (kz * ngll + q) * ngll + kx, iz = (q * ngll + ky) * ngll + kx;
        r[0] += Dx * s_vx[zy + q]; r[1] += Dy * s_vx[iy]; r[2] += Dz * s_vx[iz];
        r[3] += Dx * s_vy[zy + q]; r[4] += Dy * s_vy[iy]; r[5] += Dz * s_vy[iz];
        r[6] += Dx * s_vz[zy + q]; r[7] += Dy * s_vz[iy]; r[8] += Dz * s_vz[iz];
    }
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
#define BIOT3D_PG(c, j) (r[3 * (c) + 0] * I9[(j)] + r[3 * (c) + 1] * I9[3 + (j)] \
                         + r[3 * (c) + 2] * I9[6 + (j)])
    float gv[9] = {BIOT3D_PG(0, 0), BIOT3D_PG(1, 1), BIOT3D_PG(2, 2),
                   BIOT3D_PG(1, 0), BIOT3D_PG(0, 1), BIOT3D_PG(2, 0),
                   BIOT3D_PG(0, 2), BIOT3D_PG(2, 1), BIOT3D_PG(1, 2)};
#undef BIOT3D_PG
    if (mv) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, ax, az, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bx, bz, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float mm = ca[j] * mv[j * Em + off] + cb[j] * gv[j];
            mv[j * Em + off] = mm;
            gv[j] = gv[j] + mm;
        }
    }
    if (save_gv) {
        #pragma unroll
        for (int j = 0; j < 9; ++j) save_gv[j * Em + off] = gv[j];
    }
    const float div_v = gv[0] + gv[1] + gv[2];
    const float ld = lam_d[off], u_ = mu[off];
    sig[0 * Em + off] += dt * (ld * div_v + 2.f * u_ * gv[0]);
    sig[1 * Em + off] += dt * (ld * div_v + 2.f * u_ * gv[1]);
    sig[2 * Em + off] += dt * (ld * div_v + 2.f * u_ * gv[2]);
    sig[3 * Em + off] += dt * u_ * (gv[4] + gv[3]);
    sig[4 * Em + off] += dt * u_ * (gv[6] + gv[5]);
    sig[5 * Em + off] += dt * u_ * (gv[8] + gv[7]);
    atomicAdd(&adiv[g0], wJ[off] * alpha[off] * div_v);
}

// K7: the weak divergence of q (3, Em): sum_a D_a^T [w sum_j q_j I_aj]; where the
// C-PML reaches, split by direction into Pq (3, ng). Shared 9m.
static __global__ void biot3d_div_q(
    const float* __restrict__ q,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ divq, float* __restrict__ Pq, int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const bool layer = a_x && __syncthreads_or(
        a_x[off] != 0.f || a_y[off] != 0.f || a_z[off] != 0.f ||
        b_x[off] != 0.f || b_y[off] != 0.f || b_z[off] != 0.f);
    extern __shared__ float sh[];
    const float w = wJ[off];
    const float q3[3] = {q[off], q[Em + off], q[2 * Em + off]};
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
    const int zy = (kz * ngll + ky) * ngll, g = ibool[off];
    if (layer) {               // slots (j, a): w q_j I_aj
        #pragma unroll
        for (int j = 0; j < 3; ++j)
            #pragma unroll
            for (int a = 0; a < 3; ++a) sh[(3 * j + a) * m + tid] = w * q3[j] * I9[3 * a + j];
        __syncthreads();
        float pj[3] = {0.f, 0.f, 0.f};
        for (int k = 0; k < ngll; ++k) {
            const float Tx = D[k * ngll + kx], Ty = D[k * ngll + ky], Tz = D[k * ngll + kz];
            const int iy = (kz * ngll + k) * ngll + kx, iz = (k * ngll + ky) * ngll + kx;
            #pragma unroll
            for (int j = 0; j < 3; ++j)
                pj[j] += Tx * sh[(3 * j + 0) * m + zy + k] + Ty * sh[(3 * j + 1) * m + iy]
                       + Tz * sh[(3 * j + 2) * m + iz];
        }
        #pragma unroll
        for (int j = 0; j < 3; ++j) atomicAdd(&Pq[j * n_global + g], pj[j]);
        return;
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a)
        sh[a * m + tid] = w * (q3[0] * I9[3 * a + 0] + q3[1] * I9[3 * a + 1]
                               + q3[2] * I9[3 * a + 2]);
    __syncthreads();
    float d = 0.f;
    for (int k = 0; k < ngll; ++k) {
        const float Tx = D[k * ngll + kx], Ty = D[k * ngll + ky], Tz = D[k * ngll + kz];
        d += Tx * sh[zy + k] + Ty * sh[m + (kz * ngll + k) * ngll + kx]
           + Tz * sh[2 * m + (k * ngll + ky) * ngll + kx];
    }
    atomicAdd(&divq[g], d);
}

// K8: the C-PML on div q by direction (a_n (6, ng): a_x b_x a_y b_y a_z b_z).
static __global__ void biot3d_cpml_q(
    const float* __restrict__ a_n, const float* __restrict__ Pq, float* __restrict__ mq,
    float* __restrict__ divq, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float pj = Pq[j * ng + i];
        const float psi = a_n[(2 * j) * ng + i] * mq[j * ng + i] + a_n[(2 * j + 1) * ng + i] * pj;
        mq[j * ng + i] = psi;
        s += pj + psi;
    }
    divq[i] += s;
}

// K9 (points): div q -= w v.n.
static __global__ void biot3d_ifc_ac(
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const long* __restrict__ gidx, const float* __restrict__ w, const float* __restrict__ nx,
    const float* __restrict__ ny, const float* __restrict__ nz, float* __restrict__ divq,
    int n_pts)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    atomicAdd(&divq[g], -w[j] * (nx[j] * vx[g] + ny[j] * vy[g] + nz[j] * vz[g]));
}

// K11: the normal stresses on the pressure's increment, the viscous memory.
static __global__ void biot3d_stress2(
    float* __restrict__ sig, float* __restrict__ r, const float* __restrict__ q,
    const float* __restrict__ alpha, const float* __restrict__ dp,
    const float* __restrict__ b, float decay, float gain, int viscous,
    float* __restrict__ save_q, const int* __restrict__ ibool, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    const float ds = alpha[off] * dp[ibool[off]];
    sig[off] -= ds; sig[Em + off] -= ds; sig[2 * Em + off] -= ds;
    if (viscous) {
        const float gb = gain * b[off];
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            const float qd = q[d * Em + off];
            r[d * Em + off] = decay * r[d * Em + off] + gb * qd;
            if (save_q) save_q[d * Em + off] = qd;
        }
    }
}

// ============================== transposes ===================================

// K11^T + K6^T (frame part).
static __global__ void biot3d_stress_T(
    const float* __restrict__ lsig, float* __restrict__ lr, float* __restrict__ lq,
    const float* __restrict__ gvs, const float* __restrict__ qs_new,
    const float* __restrict__ rhs, const float* __restrict__ inv_Mp,
    const float* __restrict__ pin_n, float dt,
    const float* __restrict__ lam_d, const float* __restrict__ mu,
    const float* __restrict__ alpha, const float* __restrict__ b,
    float decay, float gain, int viscous, const int* __restrict__ ibool,
    float* __restrict__ lgv, float* __restrict__ ldp,
    float* __restrict__ gLd, float* __restrict__ gMu, float* __restrict__ gAl,
    float* __restrict__ gB, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    float L[6], g[9];
    #pragma unroll
    for (int k = 0; k < 6; ++k) L[k] = lsig[k * Em + off];
    #pragma unroll
    for (int j = 0; j < 9; ++j) g[j] = gvs[j * Em + off];
    const float div_v = g[0] + g[1] + g[2], sd = L[0] + L[1] + L[2];
    const float ld = lam_d[off], u_ = mu[off];
    lgv[0 * Em + off] = dt * (ld * sd + 2.f * u_ * L[0]);
    lgv[1 * Em + off] = dt * (ld * sd + 2.f * u_ * L[1]);
    lgv[2 * Em + off] = dt * (ld * sd + 2.f * u_ * L[2]);
    lgv[3 * Em + off] = dt * u_ * L[3];  lgv[4 * Em + off] = dt * u_ * L[3];
    lgv[5 * Em + off] = dt * u_ * L[4];  lgv[6 * Em + off] = dt * u_ * L[4];
    lgv[7 * Em + off] = dt * u_ * L[5];  lgv[8 * Em + off] = dt * u_ * L[5];
    gLd[off] += dt * sd * div_v;
    gMu[off] += dt * (2.f * (L[0] * g[0] + L[1] * g[1] + L[2] * g[2])
                      + L[3] * (g[3] + g[4]) + L[4] * (g[5] + g[6]) + L[5] * (g[7] + g[8]));
    const int gn = ibool[off];
    const float dpe = pin_n[gn] * (dt * inv_Mp[gn] * rhs[gn]);
    gAl[off] += -sd * dpe;
    atomicAdd(&ldp[gn], -alpha[off] * sd);
    if (viscous) {
        const float gbv = gain * b[off];
        float acc = 0.f;
        #pragma unroll
        for (int d = 0; d < 3; ++d) {
            const float rb = lr[d * Em + off];
            lr[d * Em + off] = decay * rb;
            lq[d * Em + off] += gbv * rb;
            acc += rb * qs_new[d * Em + off];
        }
        gB[off] += gain * acc;
    }
}

// K9^T (points).
static __global__ void biot3d_ifc_ac_T(
    const float* __restrict__ ldivq, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ ny, const float* __restrict__ nz,
    float* __restrict__ lvx, float* __restrict__ lvy, float* __restrict__ lvz, int n_pts)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    const float l = -w[j] * ldivq[g];
    atomicAdd(&lvx[g], l * nx[j]);
    atomicAdd(&lvy[g], l * ny[j]);
    atomicAdd(&lvz[g], l * nz[j]);
}

// K6^T (gradient part): the div v adjoint of -A(w alpha div v), the memories, the
// metric transpose and the D^T scatter into lv. Shared 9m.
static __global__ void biot3d_stress_grad_T(
    float* __restrict__ lgv, const float* __restrict__ gvs, const float* __restrict__ ldivq,
    float* __restrict__ lmv,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ alpha, const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ lvx, float* __restrict__ lvy, float* __restrict__ lvz,
    float* __restrict__ gAl, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const int g = ibool[off];
    const float ladiv = -ldivq[g];
    const float wa = wJ[off] * alpha[off];
    const float div_v = gvs[off] + gvs[Em + off] + gvs[2 * Em + off];
    gAl[off] += wJ[off] * div_v * ladiv;
    float lg[9];
    #pragma unroll
    for (int j = 0; j < 9; ++j) lg[j] = lgv[j * Em + off];
    lg[0] += wa * ladiv; lg[1] += wa * ladiv; lg[2] += wa * ladiv;
    float lraw[9];
    if (lmv) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, ax, az, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bx, bz, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float tot = lg[j] + lmv[j * Em + off];
            lraw[j] = lg[j] + cb[j] * tot;
            lmv[j * Em + off] = ca[j] * tot;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 9; ++j) lraw[j] = lg[j];
    }
    // (component c, direction j): (x,x)0 (x,y)4 (x,z)6 (y,x)3 (y,y)1 (y,z)8 (z,x)5 (z,y)7 (z,z)2
    const float lc[9] = {lraw[0], lraw[4], lraw[6], lraw[3], lraw[1], lraw[8],
                         lraw[5], lraw[7], lraw[2]};
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
    extern __shared__ float sh[];
    #pragma unroll
    for (int c = 0; c < 3; ++c)
        #pragma unroll
        for (int a = 0; a < 3; ++a)
            sh[(3 * c + a) * m + tid] = I9[3 * a + 0] * lc[3 * c + 0] + I9[3 * a + 1] * lc[3 * c + 1]
                                      + I9[3 * a + 2] * lc[3 * c + 2];
    __syncthreads();
    const int zy = (kz * ngll + ky) * ngll;
    float s3[3] = {0.f, 0.f, 0.f};
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx], Ty = D[q * ngll + ky], Tz = D[q * ngll + kz];
        const int iy = (kz * ngll + q) * ngll + kx, iz = (q * ngll + ky) * ngll + kx;
        #pragma unroll
        for (int c = 0; c < 3; ++c)
            s3[c] += Tx * sh[(3 * c + 0) * m + zy + q] + Ty * sh[(3 * c + 1) * m + iy]
                   + Tz * sh[(3 * c + 2) * m + iz];
    }
    atomicAdd(&lvx[g], s3[0]);
    atomicAdd(&lvy[g], s3[1]);
    atomicAdd(&lvz[g], s3[2]);
}

// K8^T (nodal).
static __global__ void biot3d_cpml_q_T(
    const float* __restrict__ a_n, const float* __restrict__ ldivq, float* __restrict__ lmq,
    float* __restrict__ lPq, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float l = ldivq[i];
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float tot = l + lmq[j * ng + i];
        lmq[j * ng + i] = a_n[(2 * j) * ng + i] * tot;
        lPq[j * ng + i] = l + a_n[(2 * j + 1) * ng + i] * tot;
    }
}

// K7^T (element): lq_j += w sum_a I_aj D_a (ldivq, or lPq_j where the C-PML reaches).
// Shared 3m.
static __global__ void biot3d_div_q_T(
    const float* __restrict__ ldivq, const float* __restrict__ lPq, float* __restrict__ lq,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const bool layer = a_x && __syncthreads_or(
        a_x[off] != 0.f || a_y[off] != 0.f || a_z[off] != 0.f ||
        b_x[off] != 0.f || b_y[off] != 0.f || b_z[off] != 0.f);
    extern __shared__ float sh[];
    const int g = ibool[off], zy = (kz * ngll + ky) * ngll;
    const float w = wJ[off];
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
    const int ns = layer ? 3 : 1;
    for (int s = 0; s < ns; ++s) sh[s * m + tid] = layer ? lPq[s * n_global + g] : ldivq[g];
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float* f = sh + (layer ? j : 0) * m;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f;
        for (int k = 0; k < ngll; ++k) {
            d0 += D[kx * ngll + k] * f[zy + k];
            d1 += D[ky * ngll + k] * f[(kz * ngll + k) * ngll + kx];
            d2 += D[kz * ngll + k] * f[(k * ngll + ky) * ngll + kx];
        }
        lq[j * Em + off] += w * (I9[j] * d0 + I9[3 + j] * d1 + I9[6 + j] * d2);
    }
}

// K5^T: one group of nodes, K copies each (see biot3d_flux).
template <int K>
static __global__ void biot3d_flux_T(
    const float* __restrict__ lvx, const float* __restrict__ lvy, const float* __restrict__ lvz,
    float* __restrict__ lq, const float* __restrict__ s_f, const float* __restrict__ s_qs,
    const float* __restrict__ s_rs, const float* __restrict__ qmask,
    const float* __restrict__ inv_m, const float* __restrict__ inv_Mm,
    const int* __restrict__ perm, const int* __restrict__ node_order,
    const float* __restrict__ wr, const float* __restrict__ share,
    const float* __restrict__ dragE, const float* __restrict__ dragP,
    const float* __restrict__ pvx, const float* __restrict__ pvy, const float* __restrict__ pvz,
    const float* __restrict__ pqx, const float* __restrict__ pqy, const float* __restrict__ pqz,
    float wv, int velocity_mode, float dt, float* __restrict__ lf, float* __restrict__ lg,
    float* __restrict__ gE, float* __restrict__ gP, float* __restrict__ gwr,
    float* __restrict__ gshare, float* __restrict__ gim, float* __restrict__ gMm,
    float* __restrict__ gW, int kstep, int n_k, int coff, long eoff, int noff, int m,
    long Em, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    float wgrad = 0.f;
    if (j < n_k) {
        const int node = node_order[noff + j];
        const int base = coff + j * K;
        const float* Ei = dragE + eoff + static_cast<long>(j) * K * K;
        const float* Pi = dragP + eoff + static_cast<long>(j) * K * K;
        float* gEi = gE + eoff + static_cast<long>(j) * K * K;
        float* gPi = gP + eoff + static_cast<long>(j) * K * K;
        const float im = inv_m[node];
        const float fd[3] = {s_f[node], s_f[ng + node], s_f[2 * ng + node]};
        const float lv[3] = {lvx[node], lvy[node], lvz[node]};
        const float* pv[3] = {pvx, pvy, pvz};
        const float* pq[3] = {pqx, pqy, pqz};
        float qs[3][K], rs[3][K], nw[3][K], lnw[3][K], lqs[3][K], lrs[3][K];
        int el[K];
        #pragma unroll
        for (int c = 0; c < K; ++c) {
            const int e = perm[base + c];
            el[c] = e;
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                qs[d][c] = s_qs[d * Em + base + c];
                rs[d][c] = s_rs[d * Em + base + c];
                lnw[d][c] = lq[d * Em + e];
            }
        }
        #pragma unroll
        for (int d = 0; d < 3; ++d)
            #pragma unroll
            for (int a = 0; a < K; ++a) {
                float s = 0.f;
                #pragma unroll
                for (int b = 0; b < K; ++b) s += Ei[a * K + b] * qs[d][b] + Pi[a * K + b] * rs[d][b];
                nw[d][a] = s;
            }
        float g_im = 0.f;
        if (velocity_mode) {
            float pvs = 0.f, pqs = 0.f;
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                pvs += pv[d][node] * lv[d];
                float sq = 0.f;
                #pragma unroll
                for (int c = 0; c < K; ++c) sq += qmask[el[c] / m] * lnw[d][c];
                pqs += pq[d][node] * sq;
            }
            wgrad += dt * im * pvs + dt * inv_Mm[node] * pqs;
            g_im += dt * wv * pvs;
            gMm[node] += dt * wv * pqs;
        }
        float lfd[3], u[3], dsum[3] = {0.f, 0.f, 0.f};
        #pragma unroll
        for (int d = 0; d < 3; ++d) { u[d] = lv[d] * im; lfd[d] = dt * u[d]; }
        #pragma unroll
        for (int c = 0; c < K; ++c) {
            const float wk = wr[base + c];
            float gw = 0.f;
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                const float ex = nw[d][c] - qs[d][c];
                dsum[d] += wk * ex;
                lnw[d][c] -= wk * u[d];
                lqs[d][c] = wk * u[d];
                gw -= ex * u[d];
            }
            gwr[base + c] += gw;
        }
        #pragma unroll
        for (int d = 0; d < 3; ++d) g_im += lv[d] * (dt * fd[d] - dsum[d]);
        // new = E qs + Psi rs, every component with the same E, Psi
        #pragma unroll
        for (int b = 0; b < K; ++b) {
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                float sq = 0.f, sr = 0.f;
                #pragma unroll
                for (int a = 0; a < K; ++a) {
                    sq += Ei[a * K + b] * lnw[d][a];
                    sr += Pi[a * K + b] * lnw[d][a];
                }
                lqs[d][b] += sq;
                lrs[d][b] = sr;
            }
            #pragma unroll
            for (int a = 0; a < K; ++a) {
                float ge = 0.f, gp = 0.f;
                #pragma unroll
                for (int d = 0; d < 3; ++d) {
                    ge += lnw[d][a] * qs[d][b];
                    gp += lnw[d][a] * rs[d][b];
                }
                gEi[a * K + b] += ge;
                gPi[a * K + b] += gp;
            }
        }
        // rs = to copies of w (r - gp) qmask - wr f inv_m (+ share pat_q w_n)
        #pragma unroll
        for (int c = 0; c < K; ++c) {
            const float wk = wr[base + c];
            float lrf = 0.f, lsq = 0.f;
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                lfd[d] -= wk * lrs[d][c] * im;
                lrf += lrs[d][c] * fd[d];
                if (!velocity_mode) lsq += lrs[d][c] * pq[d][node];
            }
            gwr[base + c] += -lrf * im;
            g_im += -wk * lrf;
            if (!velocity_mode) {
                gshare[base + c] += lsq * wv;
                wgrad += share[base + c] * lsq;
            }
            const int e = el[c];
            #pragma unroll
            for (int d = 0; d < 3; ++d) {
                lg[d * Em + e] = lrs[d][c];
                lq[d * Em + e] = lqs[d][c];        // the adjoint of the old q
            }
        }
        #pragma unroll
        for (int d = 0; d < 3; ++d) lf[d * ng + node] = lfd[d];
        gim[node] += g_im;
    }
    const float bs = sem2d::sem2d_block_sum(wgrad);
    if (threadIdx.x == 0) atomicAdd(&gW[kstep], bs);
}

// K5^T (element part).
static __global__ void biot3d_g_T(
    const float* __restrict__ lg, float* __restrict__ lr, float* __restrict__ lgp,
    const float* __restrict__ wJ, const float* __restrict__ qmask, int m, long Em)
{
    const long off = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= Em) return;
    const float w = wJ[off] * qmask[off / m];
    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        const float gd = lg[d * Em + off];
        lr[d * Em + off] += w * gd;
        lgp[d * Em + off] = -w * gd;
    }
}

// K4^T.
static __global__ void biot3d_force_T(
    const float* __restrict__ lf, float* __restrict__ lRx, float* __restrict__ lRy,
    float* __restrict__ lRz, const float* __restrict__ pvx, const float* __restrict__ pvy,
    const float* __restrict__ pvz, int load, float* __restrict__ gW, int kstep, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float wgrad = 0.f;
    if (i < ng) {
        const float fx = lf[i], fy = lf[ng + i], fz = lf[2 * ng + i];
        lRx[i] = -fx; lRy[i] = -fy; lRz[i] = -fz;
        if (load) wgrad = pvx[i] * fx + pvy[i] * fy + pvz[i] * fz;
    }
    const float bs = sem2d::sem2d_block_sum(wgrad);
    if (threadIdx.x == 0 && load) atomicAdd(&gW[kstep], bs);
}

// K4^T (points).
static __global__ void biot3d_ifc_el_T(
    const float* __restrict__ lf, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ ny, const float* __restrict__ nz,
    float* __restrict__ lp, int n_pts, int ng)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pts) return;
    const long g = gidx[j];
    atomicAdd(&lp[g], w[j] * (nx[j] * lf[g] + ny[j] * lf[ng + g] + nz[j] * lf[2 * ng + g]));
}

// K2^T: the memories' adjoint and Grad^T of the scalar p. Shared 3m.
static __global__ void biot3d_grad_p_T(
    const float* __restrict__ lgp, float* __restrict__ lmp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixy,
    const float* __restrict__ Ixz, const float* __restrict__ Iex,
    const float* __restrict__ Iey, const float* __restrict__ Iez,
    const float* __restrict__ Igx, const float* __restrict__ Igy,
    const float* __restrict__ Igz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ lp, int ngll, int E)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    float l3[3] = {lgp[off], lgp[Em + off], lgp[2 * Em + off]};
    if (lmp) {
        const float a3[3] = {a_x[off], a_y[off], a_z[off]};
        const float b3[3] = {b_x[off], b_y[off], b_z[off]};
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float tot = l3[j] + lmp[j * Em + off];
            lmp[j * Em + off] = a3[j] * tot;
            l3[j] = l3[j] + b3[j] * tot;
        }
    }
    const float I9[9] = {Ixx[off], Ixy[off], Ixz[off], Iex[off], Iey[off], Iez[off],
                         Igx[off], Igy[off], Igz[off]};
    extern __shared__ float sh[];
    #pragma unroll
    for (int a = 0; a < 3; ++a)
        sh[a * m + tid] = I9[3 * a + 0] * l3[0] + I9[3 * a + 1] * l3[1] + I9[3 * a + 2] * l3[2];
    __syncthreads();
    const int zy = (kz * ngll + ky) * ngll;
    float s = 0.f;
    for (int q = 0; q < ngll; ++q)
        s += D[q * ngll + kx] * sh[zy + q] + D[q * ngll + ky] * sh[m + (kz * ngll + q) * ngll + kx]
           + D[q * ngll + kz] * sh[2 * m + (q * ngll + ky) * ngll + kx];
    atomicAdd(&lp[ibool[off]], s);
}

// Launch KERNEL<K> for a group's copy count (1 .. 8).
#define BIOT3D_LAUNCH_K(k, KERNEL, grid, block, shmem, stream, ...)                  \
    do {                                                                            \
        switch (k) {                                                                \
            case 1: SWEEP_LAUNCH_ON((KERNEL<1>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 2: SWEEP_LAUNCH_ON((KERNEL<2>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 3: SWEEP_LAUNCH_ON((KERNEL<3>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 4: SWEEP_LAUNCH_ON((KERNEL<4>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 5: SWEEP_LAUNCH_ON((KERNEL<5>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 6: SWEEP_LAUNCH_ON((KERNEL<6>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            case 7: SWEEP_LAUNCH_ON((KERNEL<7>), grid, block, shmem, stream, __VA_ARGS__); break;  \
            default: SWEEP_LAUNCH_ON((KERNEL<8>), grid, block, shmem, stream, __VA_ARGS__); break; \
        }                                                                           \
    } while (0)

} // namespace sem3d

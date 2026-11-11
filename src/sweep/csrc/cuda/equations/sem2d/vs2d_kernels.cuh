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
__global__ void sem2d_vs_sigma_div(
    const float* __restrict__ sig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ Rx, float* __restrict__ Rz, float* __restrict__ P,
    int ngll, int E, long n_global);

// C: the C-PML on the divergence by direction, per node: psi_j = a psi_j + b P_j
// (the slot's direction j % 2), R_i += its two slots of P + psi. a_n (4, n_global):
// a_x b_x a_z b_z. A node the layer does not reach has P = psi = 0.
__global__ void sem2d_vs_cpml_nodal(
    const float* __restrict__ a_n, const float* __restrict__ P,
    float* __restrict__ msig, float* __restrict__ Rx, float* __restrict__ Rz, int ng);

// N: v += dt invM (pat wav[n] - R), two components + optional v-history save.
__global__ void sem2d_vs_vel_update(
    float* __restrict__ vx, float* __restrict__ vz,
    const float* __restrict__ Rx, const float* __restrict__ Rz,
    const float* __restrict__ inv_M, const float* __restrict__ pat_x,
    const float* __restrict__ pat_z, const float* __restrict__ wavelet, int n, float dt,
    float* __restrict__ s_vx, float* __restrict__ s_vz, int ng);

// S: the reference derivatives of the new v, the physical gradients d v_c / d x_j
// = sum_a r_ca I_aj (slots vx_x vz_z vx_z vz_x), the C-PML memories and stretch,
// the optional record of the stretched gradients, then the stress update
// (vs2d_stress_increment<NC>, plus the GSLS memories when nsls > 0). Shared 2m.
using sem2d_vs_stress_update_fn = void (*)(const float*, const float*, float*, float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, int, const float*, const float*, const float*, float*,
    const float*, const float*, const float*, const float*, const int*, const float*,
    float, float*, int, int);
extern sem2d_vs_stress_update_fn const sem2d_vs_stress_update_by_nc[3];

// S^T: the moduli's adjoint and imaging, the GSLS memories' adjoint, the C-PML
// memory recursion, then the transpose of the metric contraction and the D^T
// scatter into lv. Shared 4m.
using sem2d_vs_stress_update_T_fn = void (*)(const float*, const float*, float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, int, const float*, const float*, const float*, float*,
    float*, float*, const float*, const float*, const float*, const float*, const int*,
    const float*, float, float*, float*, float*, float*, float*, float*, int, int);
extern sem2d_vs_stress_update_T_fn const sem2d_vs_stress_update_T_by_nc[3];

// N^T (nodal): grad_inv_M through the update identity, lR = -dt invM lv (lv
// carries on unchanged to the previous step as the identity part).
__global__ void sem2d_vs_vel_update_T(
    const float* __restrict__ lvx, const float* __restrict__ lvz,
    const float* __restrict__ v_k_x, const float* __restrict__ v_k_z,     // saved v^k
    const float* __restrict__ v_km1_x, const float* __restrict__ v_km1_z, // v^{k-1}, may be null
    const float* __restrict__ inv_M, float dt,
    float* __restrict__ lRx, float* __restrict__ lRz, float* __restrict__ gradM, int ng);

// C^T (nodal): tot_j = lR_i + lmsig_j, lmsig_j = a tot_j in place, and the slot's
// divergence adjoint lP_j = lR_i + b tot_j (P enters R directly and through psi).
__global__ void sem2d_vs_cpml_nodal_T(
    const float* __restrict__ a_n, const float* __restrict__ lRx,
    const float* __restrict__ lRz, float* __restrict__ lmsig, float* __restrict__ lP, int ng);

// V^T (element, no atomics): lsig_ij += w_J sum_a I_aj (D_a lR_i), or with each
// slot's own adjoint lP_(2i+j) in an element the C-PML reaches. Shared 4m.
__global__ void sem2d_vs_sigma_div_T(
    const float* __restrict__ lRx, const float* __restrict__ lRz,
    const float* __restrict__ lP, float* __restrict__ lsig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global);

// Launch KERNEL<NC> for the input's stiffness count (0 isotropic, 4 VTI, 6 TTI).
// KERNEL##_by_nc is the kernel's table, <6>, <4>, <0>.
#define SEM2D_VS_LAUNCH_NC(nc, KERNEL, grid, block, shmem, stream, ...) \
    KERNEL##_by_nc[(nc) == 6 ? 0 : (nc) == 4 ? 1 : 2]<<<grid, block, shmem, stream>>>(__VA_ARGS__)

} // namespace sem2d

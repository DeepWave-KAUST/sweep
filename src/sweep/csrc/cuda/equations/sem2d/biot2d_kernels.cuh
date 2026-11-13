// sweep.sem CUDA (impl='c') backend -- the 2-D Biot SEM kernels:
// BiotVSSEM2DDeformed and BiotAnisoPermSEM2DDeformed (sweep.sem.equations.
// biot_vs_deformed; its module docstring has the scheme). One step, in the eager
// step's order:
//
//   K1  p += dt inv_M1 pat_p w_n, pinned                       (nodal)
//   K2  gp = grad p (+ C-PML memory mp), element-local          (element)
//   K3  R = weak div sigma (+ C-PML by direction)               (sem2d_vs_sigma_div,
//                                                                sem2d_vs_cpml_nodal)
//   K4  f = -R (+ pat_v w_n, "load") + the interface's  w p n   (nodal, points)
//   K5  the flux copies at every node, q <- E q + Psi r_s, and v (nodal, the slots)
//   K6  gv = grad v (+ memories mv), the frame stress, A(w alpha div v)   (element)
//   K7  div q weak (by direction where the C-PML reaches)       (element)
//   K8  its C-PML memories mq                                   (nodal)
//   K9  minus the interface's  w v.n                            (points)
//   K10 rhs, dp = pin dt inv_Mp rhs, p += dp                    (nodal)
//   K11 sig -= alpha dp I, the viscous memory r                 (element)
//
// and the transposes in reverse (the *_T kernels). Element arrays are (E, z, x),
// x fastest; I_aj = d xi_a / d x_j (ixx, ixz, iex, iez). The flux copies of node i
// sit in its kmax slots: slot_elem[i kmax + k] is the element node of copy k (-1:
// empty). The histories a step leaves for its adjoint: f (2, ng), the slots of q
// and of the frozen forces r_s (2, ng, kmax), rhs (ng), the stretched grad v
// (4, E m) and, with the viscous memory, the new q (2, E m). One shot (B = 1).
#pragma once

namespace sem2d {

constexpr int BIOT2D_KMAX = 4;          // copies of a node in a quad mesh

// K1: the injected pore pressure, then the open-pore pin.
__global__ void biot2d_inject_p(
    float* __restrict__ p, const float* __restrict__ inv_M1, const float* __restrict__ pin_n,
    const float* __restrict__ pat_p, float wv, float dt, int ng);

// K2: the strong gradient of the nodal p at the element nodes, stretched by the
// C-PML memories (mp null: bare). Shared m.
__global__ void biot2d_grad_p(
    const float* __restrict__ p, float* __restrict__ gp, float* __restrict__ mp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D, int ngll, int E);

// K4: the force on the solid, f = -R (+ the momentum load in "load" mode).
__global__ void biot2d_force(
    const float* __restrict__ Rx, const float* __restrict__ Rz, float* __restrict__ f,
    const float* __restrict__ pat_vx, const float* __restrict__ pat_vz, float wv, int load,
    int ng);

// K4 (points): a fluid's pressure on the solid, f += w p n (FluidSolidInterface.apply_el).
__global__ void biot2d_ifc_el(
    const float* __restrict__ p, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz, float* __restrict__ f,
    int n_pts, int ng);

// K5: per node, the frozen forces on its flux copies r_s = to_slots(w (r - gp) qmask)
// - wr f inv_m (+ share pat_q w_n, "load"), the copies' exact step new = E qs + Psi
// r_s, v += (dt f - sum wr (new - qs)) inv_m, the copies written back; "velocity"
// mode then jumps v and q. Records f, qs and r_s for the adjoint.
using biot2d_flux_fn = void (*)(float*, float*, float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float, int, float, float*, float*, float*, int, int,
    long, int);
extern biot2d_flux_fn const biot2d_flux_by_flag[2];

// K6: grad v (slots vx_x vz_z vx_z vz_x) stretched by its memories, the frame's
// stress increment dt (lam_d div v I + mu (grad v + grad v^T)), and A(w alpha div v).
// Shared 2m.
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
    float* __restrict__ adiv, float* __restrict__ save_gv, int ngll, int E);

// K7: the weak divergence of the element-local q, sum_a D_a^T [w sum_j q_j I_aj];
// where the C-PML reaches, split by direction into Pq (2, n_global). Shared 4m.
__global__ void biot2d_div_q(
    const float* __restrict__ q,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ divq, float* __restrict__ Pq, int ngll, int E, long n_global);

// K8: the C-PML on div q by direction, per node: psi_j = a psi_j + b Pq_j, and
// div q += Pq_x + Pq_z + psi_x + psi_z. a_n (4, ng): a_x b_x a_z b_z.
__global__ void biot2d_cpml_q(
    const float* __restrict__ a_n, const float* __restrict__ Pq, float* __restrict__ mq,
    float* __restrict__ divq, int ng);

// K9 (points): minus the solid's normal velocity on the fluid, div q -= w v.n
// (FluidSolidInterface.apply_ac).
__global__ void biot2d_ifc_ac(
    const float* __restrict__ vx, const float* __restrict__ vz, const long* __restrict__ gidx,
    const float* __restrict__ w, const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ divq, int n_pts);

// K10: rhs = div q - A(w alpha div v) (the interface already in div q), dp = pin dt
// inv_Mp rhs, p += dp.
__global__ void biot2d_pressure(
    float* __restrict__ p, const float* __restrict__ divq, const float* __restrict__ adiv,
    const float* __restrict__ inv_Mp, const float* __restrict__ pin_n, float dt,
    float* __restrict__ dp, float* __restrict__ s_rhs, int ng);

// K11: sigma = sigma' - alpha p (on the pressure's increment), and the viscous
// memory r <- decay r + gain B q at the new flux.
__global__ void biot2d_stress2(
    float* __restrict__ sig, float* __restrict__ r, const float* __restrict__ q,
    const float* __restrict__ alpha, const float* __restrict__ dp,
    const float* __restrict__ b_xx, const float* __restrict__ b_xz,
    const float* __restrict__ b_zz, float decay, float gain, int viscous,
    float* __restrict__ save_q, const int* __restrict__ ibool, long Em);

// ============================== transposes ===================================

// K11^T + K6^T (frame part): from lsig (the adjoint of sig') and lr: the frame's
// stretched-gradient adjoint into lgv, the moduli's gradients, the alpha dp term
// (lam_d / mu / alpha imaging, the pressure increment's adjoint ldp scattered),
// the viscous memory (lr <- decay lr, lq += gain B^T lr, the drag's gradients).
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
    float* __restrict__ gBxx, float* __restrict__ gBxz, float* __restrict__ gBzz, long Em);

// K10^T: ldp (the adjoint of p' through p' = p1 + dp, plus the alpha term) pinned,
// the rhs adjoint into ldivq (rhs = div q - A(w alpha div v) - the interface), and
// inv_Mp's gradient.
__global__ void biot2d_pressure_T(
    float* __restrict__ ldp, const float* __restrict__ rhs, const float* __restrict__ inv_Mp,
    const float* __restrict__ pin_n, float dt, float* __restrict__ ldivq,
    float* __restrict__ gMp, int ng);

// K9^T (points): v' takes minus the interface's transpose of the rhs adjoint.
__global__ void biot2d_ifc_ac_T(
    const float* __restrict__ ldivq, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz,
    float* __restrict__ lvx, float* __restrict__ lvz, int n_pts);

// K6^T (gradient part): lgv += the div v adjoint of -A(w alpha div v) (alpha's
// gradient), the memories' adjoint, then Grad^T scattered into lv. Shared 4m.
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
    int ngll, int E);

// K8^T (nodal): tot_j = ldivq + lmq_j, lmq_j = a tot_j, lPq_j = ldivq + b tot_j.
__global__ void biot2d_cpml_q_T(
    const float* __restrict__ a_n, const float* __restrict__ ldivq, float* __restrict__ lmq,
    float* __restrict__ lPq, int ng);

// K7^T (element): lq_j += w sum_a I_aj (D_a ldivq), or with each direction's own
// adjoint lPq_j where the C-PML reaches. Shared 2m.
__global__ void biot2d_div_q_T(
    const float* __restrict__ ldivq, const float* __restrict__ lPq, float* __restrict__ lq,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ wJ,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global);

// K5^T: per node, from lv (the adjoint of v') and lq at its copies (of q'): the
// adjoints of f (lf) and of the frozen forces' element terms (lg, at the copies),
// lq at the copies overwritten with the adjoint of the old q, and the gradients of
// E, Psi, wr, share, inv_m, inv_Mm and the wavelet (one atomic per block).
using biot2d_flux_T_fn = void (*)(const float*, const float*, float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float, int, float, float*, float*, float*, float*,
    float*, float*, float*, float*, float*, int, int, int, long, int);
extern biot2d_flux_T_fn const biot2d_flux_T_by_flag[2];

// K5^T (element part): g = w (r - gp) qmask: lr += w qmask lg, lgp = -w qmask lg.
__global__ void biot2d_g_T(
    const float* __restrict__ lg, float* __restrict__ lr, float* __restrict__ lgp,
    const float* __restrict__ wJ, const float* __restrict__ qmask, int m, long Em);

// K4^T: lR = -lf; the momentum load's wavelet gradient ("load"; one atomic per block).
__global__ void biot2d_force_T(
    const float* __restrict__ lf, float* __restrict__ lRx, float* __restrict__ lRz,
    const float* __restrict__ pat_vx, const float* __restrict__ pat_vz, int load,
    float* __restrict__ gW, int kstep, int ng);

// K4^T (points): p1 takes the interface's transpose of lf: lp += w n . lf.
__global__ void biot2d_ifc_el_T(
    const float* __restrict__ lf, const long* __restrict__ gidx, const float* __restrict__ w,
    const float* __restrict__ nx, const float* __restrict__ nz, float* __restrict__ lp,
    int n_pts, int ng);

// K2^T: the memories' adjoint and Grad^T of the scalar p, scattered into lp. Shared 2m.
__global__ void biot2d_grad_p_T(
    const float* __restrict__ lgp, float* __restrict__ lmp,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_z, const float* __restrict__ b_z,
    const float* __restrict__ Ixx, const float* __restrict__ Ixz,
    const float* __restrict__ Iex, const float* __restrict__ Iez,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ lp, int ngll, int E);

// K1^T: the pin, inv_M1's gradient and the injection's wavelet gradient (one atomic
// per block); lp then carries on as the adjoint of the old p.
__global__ void biot2d_inject_p_T(
    float* __restrict__ lp, const float* __restrict__ inv_M1, const float* __restrict__ pin_n,
    const float* __restrict__ pat_p, float wv, float dt, float* __restrict__ gM1,
    float* __restrict__ gW, int kstep, int ng);

// Launch KERNEL<TENSOR> for the input's drag.
// KERNEL##_by_flag is the kernel's table, <false> then <true>.
#define BIOT2D_LAUNCH_T(tensor, KERNEL, grid, block, shmem, stream, ...) \
    SWEEP_LAUNCH_ON(KERNEL##_by_flag[(tensor) ? 1 : 0], grid, block, shmem, stream, __VA_ARGS__)

} // namespace sem2d

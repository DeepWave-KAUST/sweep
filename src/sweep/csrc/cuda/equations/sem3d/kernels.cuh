// sem3d CUDA kernels — 3-D acoustic SEM (Hex8, axis-aligned, velocity form).
// Ports the arithmetic of AcousticSEM3D.stiffness_apply into the sweep-native
// layout: ibool[e, kz, ky, kx] flattened with x fastest; forward ref-grads
// contract the OWN axis (du_dxi[kx] = sum_g D[kx,g] u[kz,ky,g]); the weak-form
// transpose contracts with D^T (r[kx] += sum_g D[g,kx] f[kz,ky,g]).
//
// BATCH STATUS -- read this before lifting any SWEEP_CHECK(B == 1).
//
//   ELEMENT kernels ARE batch-ready: they launch grid (E, batch) and stride the
//   field pointers by blockIdx.y * n_global.
//
//   NODAL kernels are NOT. The older note here claimed they "take the flat
//   batch*n_global length"; that is wrong and would fail silently:
//     * they mix per-batch fields (chi, K, u_next) with arrays that are SHARED
//       across the batch and only n_global long -- pattern, inv_M, the CPML
//       coefficients. A flat batch*n_global index reads those OUT OF BOUNDS for
//       every b >= 1 (see sem2d_acoustic_newmark).
//     * the model-gradient accumulators (sem2d_accum_gradM,
//       sem2d_elastic_accum_gradM, sem2d_elastic_cpml_gradM, sem3d's grad_vp2
//       and vs gradM) do a NON-ATOMIC `grad[i] +=` into a buffer that is shared
//       across the batch. Under B > 1 that is a lost-update race: the gradient
//       comes out smooth and right-signed but wrong in magnitude, and different
//       every run. This is the failure mode that is hardest to notice.
//
//   To enable batch the nodal kernels need a real node index (grid (nb, B) with
//   b = blockIdx.y, like the element kernels): per-batch fields at b*n_global,
//   shared arrays at the node index, and atomicAdd into the shared gradient.
//
// All kernels are `static` (the .cuh is included by two TUs).
#pragma once
#include <cuda_runtime.h>
#include "../../launch/launch.h"   // SWEEP_LAUNCH

namespace sem3d {

// K0 u — symmetric separable stiffness with the c^2-FREE per-axis weights
// w3 G_axis. The driver applies c^2 per node (u_tt = c^2 Laplace u, as on the
// C-PML path); c^2 inside the integral would solve div(c^2 grad u) instead,
// which flips every reflection coefficient.
// One block per (element, batch item); m = ngll^3 threads; shared 4m floats.
__global__ void sem3d_acoustic_stiffness(
    const float* __restrict__ u,        // (batch, n_global)
    const int*    __restrict__ ibool,
    const float* __restrict__ D,
    const float* __restrict__ w_jx, const float* __restrict__ w_jy,
    const float* __restrict__ w_jz,
    float* __restrict__ K,              // (batch, n_global)
    int ngll, int E, long n_global);

// grad_c2[i] -= dt2 * a_m[i] * inv_M[i] * K0u[i]: the c^2 gradient of the
// node-wise c^2 the driver applies to K0 u (forward line k, adjoint a_m).
// Nodal like sem2d_accum_gradM, with its batch caveat.
__global__ void sem3d_accum_grad_c2(
    const float* __restrict__ a_m, const float* __restrict__ inv_M,
    const float* __restrict__ K0u, float dt2, float* __restrict__ grad_c2, int n);

// ================== 3-D acoustic C-PML kernels (P6) =========================
// Komatitsch-Martin psi/zeta double convolution, 3-axis (the exact analog of
// the validated 2-D P3a/P3b machinery). Axis-aligned Hex8: the "direct" grads
// g_a = w_jac * ihd_a * d_a(u); the "weak" grads k_a = D^T_a(w_j_a * d_a(u));
// apsi grads differentiate each a*psi_a along its OWN axis only.

// forward 9-field grads: one block per (element, batch); shared 7m floats.
// w_jac / w_jx / w_jy / w_jz are read per THREAD, not per element-node. On the
// axis-aligned Hex8 mesh these four (E, ngll^3) arrays hold w3[k,j,i] times a
// SCALAR (Hex8Mesh.jacobian_det is a float, and Gx/Gy/Gz are (2/dx_e)^2 * |J|),
// so every element's slice is identical -- 125 distinct values stored E times.
// Reading slice 0 lets all blocks share the same lines instead of streaming
// four full arrays. Measured: grads -23%, grads_T -30%, whole 3-D step -7.7%.
//
// This is gated by the driver's `deformed` flag: the curved Hex27 path has a
// genuinely per-node Jacobian and uses the *_deformed kernels below, which
// still index [off]. prop_c asserts the element-invariance when it builds the
// non-deformed constants, so a future non-uniform Hex8 fails loudly.
//
// ---- Per-axis SLAB storage of the C-PML memory fields (psi_slab=1) ----------
// psi_d / zeta_d / a_d*psi_d are exactly +0.0 wherever b_d == 0 (the update
// is psi_d' = b_d*q + a_d*psi_d from a zero start, b_d is +0.0 off the layer
// and a_d = 1.0 there), so they are stored only on the index slab of their
// axis: the planes i_d on which some node has b_d != 0. The node numbering
// g = (iz*ny + iy)*nx + ix (flat Hex8, deformed Hex27 and every DD strip alike)
// gives the slot arithmetically from one table per axis, tab_d[i_d] = the
// plane's index inside the slab or -1 -- no per-node map. A slot of -1 reads
// as the literal 0.0f the full array held, and the +0.0 the kernel would have
// stored there is dropped; every floating-point operation is unchanged, only
// the addresses are. The kernels below are templated on SLAB so the full
// layout (SLAB=false) compiles to exactly what it was.
struct SemSlab { const int* tab; const int* planes; int ns; };
struct SemSlabs { int nx, ny; SemSlab x, y, z; };

// out[s] = 1.f * a[node(s)] * v[s] over one axis's slab: the slab twin of
// sem2d_scale2(a, v, 1.f, out, ng) with the same expression, launched over
// the slab's slots instead of every node.
__global__ void sem3d_slab_scale(const float* __restrict__ a,
                                 const float* __restrict__ v,
                                 float* __restrict__ out,
                                 int axis, SemSlabs sl, int n);

// Host side: the tables from an input struct (either layout; the kernels only
// dereference them under SLAB), and each axis's slab length.
template <typename In>
static inline SemSlabs sem3d_slabs(const In& in) {
    SemSlabs s;
    s.nx = in.nx; s.ny = in.ny;
    s.x.tab = in.tab_x.numel() ? in.tab_x.template data_ptr<int32_t>() : nullptr;
    s.y.tab = in.tab_y.numel() ? in.tab_y.template data_ptr<int32_t>() : nullptr;
    s.z.tab = in.tab_z.numel() ? in.tab_z.template data_ptr<int32_t>() : nullptr;
    s.x.planes = in.planes_x.numel() ? in.planes_x.template data_ptr<int32_t>() : nullptr;
    s.y.planes = in.planes_y.numel() ? in.planes_y.template data_ptr<int32_t>() : nullptr;
    s.z.planes = in.planes_z.numel() ? in.planes_z.template data_ptr<int32_t>() : nullptr;
    s.x.ns = in.nsx; s.y.ns = in.nsy; s.z.ns = in.nsz;
    return s;
}
template <typename In>
static inline int sem3d_slab_len(int axis, const In& in) {
    return axis == 0 ? in.nz * in.ny * in.nsx
         : axis == 1 ? in.nz * in.nsy * in.nx : in.nsz * in.ny * in.nx;
}
// Zero the nine assembly accumulators. gx/gy/gz are their axis slab long
// under the slab layout -- g_d is read only where b_d or db_d is non-zero,
// and the storage slab is exactly that union -- so the family is not nine
// equal rows and the sizes come from here, once, for both entries that zero
// it. (An out-of-bounds cudaMemsetAsync is reported as cudaErrorInvalidValue
// at the NEXT launch, which is a long way from the mistake.)
template <typename In>
static inline void sem3d_zero_g9(const In& in, bool slab, cudaStream_t stream,
                                 float* const g[9]) {
    const size_t ngb = static_cast<size_t>(in.n_global) * sizeof(float);
    // rows 0-5 are g_d and k_d, each on its axis slab under the slab layout;
    // rows 6-8 are m_d = ga_d - k_d, which stays per node (a node off the
    // slab can still belong to an element with absorbing nodes)
    size_t gb[9];
    for (int j = 0; j < 9; ++j)
        gb[j] = (slab && j < 6) ? static_cast<size_t>(sem3d_slab_len(j % 3, in)) * sizeof(float)
                                : ngb;
    bool contig = true;
    size_t total = gb[0];
    for (int j = 1; j < 9; ++j) {
        contig = contig && (reinterpret_cast<const char*>(g[j])
                            == reinterpret_cast<const char*>(g[j - 1]) + gb[j - 1]);
        total += gb[j];
    }
    if (contig) {
        cudaMemsetAsync(g[0], 0, total, stream);
    } else {
        for (int j = 0; j < 9; ++j) cudaMemsetAsync(g[j], 0, gb[j], stream);
    }
}

// one-bool dispatcher: SLAB for the C-PML kernels, COL for the elastic
// deformed pair (columnar metric). The parameter is named for its first user.
// K##_by_flag is the kernel's table, <false> then <true>.
#define SEM3D_LAUNCH_SLAB(slab, K, grid, block, shm, stream, ...) \
    SWEEP_LAUNCH_ON(K##_by_flag[(slab) ? 1 : 0], grid, block, shm, stream, __VA_ARGS__)
// two-flag twin: <SLAB, FLAG>. FLAG is WTAPE for step_T (read w_sum from the
// tape instead of recomputing it) and COL for the deformed grads (columnar
// metric: four inv_J components identically zero, two mesh-wide constants).
// K##_by_flags is the kernel's table, <false, false>, <false, true>,
// <true, false>, <true, true>.
#define SEM3D_LAUNCH_SLAB2(slab, wt, K, grid, block, shm, stream, ...) \
    SWEEP_LAUNCH_ON(K##_by_flags[((slab) ? 2 : 0) + ((wt) ? 1 : 0)], grid, block, shm, stream, __VA_ARGS__)

using sem3d_acoustic_cpml_grads_fn = void (*)(const float*, const float*, const float*,
    const float*, const int*, const float*, const float*, const float*, const float*,
    const float*, float, float, float, float*, float*, float*, float*, float*, float*,
    float*, float*, float*, int, int, long, SemSlabs, const int*);
extern sem3d_acoustic_cpml_grads_fn const sem3d_acoustic_cpml_grads_by_flag[2];

// fused step algebra + Newmark (node kernel):
//   q_a = g_a invM; dv_a = -k_a invM; gp_a = ga_a invM
//   t_a = (1+b_a) dv_a + db_a q_a + gp_a
//   w_sum = sum_a (1+b_a) t_a + a_a zeta_a
//   psi_a' = b_a q_a + a_a psi_a;  zeta_a' = b_a t_a + a_a zeta_a
//   u_next = 2u - u_prev + dt2 (vp2 w_sum + invM pat wav[n])
// Interior fast path -- the 3-axis twin of the 2-D one in sem2d/kernels.cuh.
// Outside the C-PML profile b_* = db* = 0, and then psi and zeta stay exactly
// zero for all time (b is the only thing that feeds them, and they start at
// zero). Everything they carry drops out exactly. 3-D has more to drop than
// 2-D: nine profile arrays and six memory fields read, six written.
//
// m_d = ga_d - k_d is full-length on purpose: a node with b = 0 can belong to
// an element that DOES contain absorbing nodes, and the assembled a*psi there
// is non-zero. Slabbing it would corrupt the gradient in a one-element ring.
// k_d alone is only ever weighted by b_d, so THAT one lives on the slab.
//
// SLAB: the six memory fields are addressed through their axis slab (see
// SemSlabs). A slow-path node off, say, the x slab (it is in the y layer, or
// on the one plane where only db != 0) reads psi_x = zeta_x = 0.0f -- what
// the full array held -- and the +0.0 it would have written back is dropped.
using sem3d_acoustic_cpml_step_newmark_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*,
    const unsigned char*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, int, float, float*, float*, float*, float*, float*, float*, float*,
    float*, int, SemSlabs);
extern sem3d_acoustic_cpml_step_newmark_fn const sem3d_acoustic_cpml_step_newmark_by_flag[2];

// backward: step-algebra transpose + w_sum recompute (3-axis P3b step_T; the
// mass here is geometric so there is NO grad_inv_M accumulation).
// Interior fast path -- the 3-axis twin of the 2-D one in sem2d/kernels.cuh.
// Outside the C-PML profile b_* = db* = 0, and then psi and zeta stay exactly
// zero for all time (b is the only thing that feeds them, and they start at
// zero). Everything they carry drops out exactly. 3-D has more to drop than
// 2-D: nine profile arrays and six memory fields read, six written.
//
// m_d = ga_d - k_d is full-length on purpose: a node with b = 0 can belong to
// an element that DOES contain absorbing nodes, and the assembled a*psi there
// is non-zero. Slabbing it would corrupt the gradient in a one-element ring.
// k_d alone is only ever weighted by b_d, so THAT one lives on the slab.
// SLAB: the reconstructed zeta_d (r_zeta_d) AND the adjoint carries
// lpsi_d / lzeta_d are read through their axis slab. The carries are not
// zero off the layer (combine accumulates a_d*(l + lap) there with a_d = 1)
// but dead: this kernel is their only reader and multiplies them by b_d,
// which is +0.0 there. So the values are equal; only the sign of an exact
// zero term can differ (-0.0 + -0.0 = -0.0 vs +0.0), which no consumer of
// gb*/gbk*/gba* can distinguish once anything non-zero is added.
//
// WTAPE: the forward saved w_sum^k (save_w_sum), so this kernel does not
// recompute it -- it reads none of the nine assembled fields nor zeta and
// writes no w_sum_out; only the adjoint algebra (gt/gq from gw and the
// carries) runs. The adjoint expressions are the same text in both.
using sem3d_acoustic_cpml_step_T_fn = void (*)(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const unsigned char*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*, float*, int, SemSlabs);
extern sem3d_acoustic_cpml_step_T_fn const sem3d_acoustic_cpml_step_T_by_flags[4];

// backward: grads transpose (G^T). Given the 9 grad-field adjoints, produce
// lu (adjoint of u) and lap_a (adjoints of a*psi_a). No material grads here
// (all weights are geometric). Shared 6m floats.
// w_jac / w_jx / w_jy / w_jz are read per THREAD, not per element-node. On the
// axis-aligned Hex8 mesh these four (E, ngll^3) arrays hold w3[k,j,i] times a
// SCALAR (Hex8Mesh.jacobian_det is a float, and Gx/Gy/Gz are (2/dx_e)^2 * |J|),
// so every element's slice is identical -- 125 distinct values stored E times.
// Reading slice 0 lets all blocks share the same lines instead of streaming
// four full arrays. Measured: grads -23%, grads_T -30%, whole 3-D step -7.7%.
//
// This is gated by the driver's `deformed` flag: the curved Hex27 path has a
// genuinely per-node Jacobian and uses the *_deformed kernels below, which
// still index [off]. prop_c asserts the element-invariance when it builds the
// non-deformed constants, so a future non-uniform Hex8 fails loudly.
// SLAB: lap_d is written on slab d ALONE. combine reads lapx only inside
// `if (sx >= 0)`, so a contribution to a node off slab d is never read -- the
// committed NaN-poison test says so decisively. Under <false> the address
// expression is textually unchanged, so its codegen is too.
using sem3d_acoustic_cpml_grads_T_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const int*, const float*, const float*, const float*, const float*,
    const float*, float, float, float, float*, float*, float*, float*, int, int, long,
    const int*, SemSlabs);
extern sem3d_acoustic_cpml_grads_T_fn const sem3d_acoustic_cpml_grads_T_by_flag[2];

// backward combine (node): Newmark^T grads + finalise u/memory adjoints.
// SLAB: the adjoint carries are updated only on their slab (the old code
// wrote a_d*(l + lap) at every node; off the slab that value was never read
// with a non-zero weight -- see step_T). gw stays full-length -- a node-wise
// quantity read at every slab node -- but lap* is on its slab too: this
// kernel is its only reader and reads it only where the slot exists.
using sem3d_acoustic_cpml_combine_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float, int, SemSlabs);
extern sem3d_acoustic_cpml_combine_fn const sem3d_acoustic_cpml_combine_by_flag[2];

// ============ 3-D acoustic DEFORMED-metric kernels (Hex27 topography) ========
// Metric-general analogue of the axis-aligned sem3d_acoustic_* kernels: consume
// the full per-node 3x3 inv_J (I[m,d] = d xi_m/d x_d, m in {xi,eta,gam}, d in
// {x,y,z}) + w_J = w3 |J|, mirroring AcousticSEM3DDeformed. The physical
// gradient P_d(f) = I[xi,d] G_xi(f) + I[eta,d] G_eta(f) + I[gam,d] G_gam(f); the
// stiffness is P_d^T(w_J P_d(u)) (symmetric -> reused for the reverse march).
// On a flat mesh (I diagonal, cross terms 0) these reduce to the axis-aligned
// arithmetic bit-for-bit. All kernels launch grid (E, batch), m = ngll^3 threads.

// K0 u — deformed stiffness, c^2-free like sem3d_acoustic_stiffness (the
// driver applies c^2 per node). Shared 4m (s_u + 3 ref fluxes F_xi/F_eta/F_gam).
__global__ void sem3d_acoustic_stiffness_deformed(
    const float* __restrict__ u, const int* __restrict__ ibool,
    const float* __restrict__ D, const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float* __restrict__ K, int ngll, int E, long n_global);

// deformed CPML forward 9-field grads: direct g_d = w_J P_d(u); weak
// k_d = P_d^T(w_J P_d(u)) = sum_m G_m^T(I[m,d] w_J P_d(u)); apsi
// ga_d = w_J P_d(a psi_d). Shared 4m (s_u + 3 apsi) reused as 3m ref fluxes.
// COL: the columnar metric of the extruded-topography builder (sem3d_types.h).
// Only the SOURCE of the six affected components changes -- every expression
// after it is the text the full-3x3 path already had, so what COL removes is
// six (E, ngll^3) streams, and nvcc folds the resulting exact zeros. Measured
// 1.25x at nex=48; the arithmetic it also removes is worth nothing on its
// own (0.999x for twice the FLOPs).
using sem3d_acoustic_cpml_grads_deformed_fn = void (*)(const float*, const float*,
    const float*, const float*, const int*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float, float, float*, float*, float*, float*, float*,
    float*, float*, float*, float*, int, int, long, SemSlabs, const int*);
extern sem3d_acoustic_cpml_grads_deformed_fn const sem3d_acoustic_cpml_grads_deformed_by_flags[4];

// deformed CPML grads transpose. lp_d = w_J gb_d + w_J P_d(gbk_d); lu =
// sum_d P_d^T(lp_d) = sum_m G_m^T(sum_d I[m,d] lp_d); lap_d = P_d^T(w_J gba_d).
// Shared 3m, reused across the lu / lapx / lapy / lapz D^T rounds.
using sem3d_acoustic_cpml_grads_T_deformed_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const int*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, float, float, float*, float*, float*, float*, int, int, long,
    const int*, SemSlabs);
extern sem3d_acoustic_cpml_grads_T_deformed_fn const sem3d_acoustic_cpml_grads_T_deformed_by_flags[4];

// ================== 3-D elastic kernels (P7, bare) ==========================
// Symmetric stiffness (ElasticSEM3D.stiffness_apply): 9 ref-grads g_{p,a} of
// the 3 displacement components, 9 fluxes F (normal + shear), 3 D^T sums.
// Shared 12m floats (3 gathers + 9 fluxes).
__global__ void sem3d_elastic_stiffness(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ w3,
    float Gxx, float Gyy, float Gzz, float Gxy, float Gxz, float Gyz,
    const float* __restrict__ jac,           // (E, m) |J| (may be uniform)
    float* __restrict__ Kx, float* __restrict__ Ky, float* __restrict__ Kz,
    int ngll, int E, long n_global);

// imaging: grads of lam/mu/lam2mu from the bilinear-form derivative with the
// saved forward field u and y = -dt2 invM o adjoint. Shared 6m.
__global__ void sem3d_elastic_imaging(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const float* __restrict__ yx, const float* __restrict__ yy,
    const float* __restrict__ yz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ w3, const float* __restrict__ jac,
    float Gxx, float Gyy, float Gzz, float Gxy, float Gxz, float Gyz,
    float* __restrict__ gL2m, float* __restrict__ gLam, float* __restrict__ gMu,
    int ngll, int E, long n_global);

// ============ 3-D elastic DEFORMED-metric kernels (Hex27 topography) =========
// Metric-general analogue of sem3d_elastic_stiffness/_imaging: full per-node
// 3x3 inv_J + w_J, mirroring ElasticSEM3DDeformed. Physical strain
// g_{p,d} = I[xi,d] G_xi(u_p) + I[eta,d] G_eta(u_p) + I[gam,d] G_gam(u_p);
// per-equation flux F^p_m = w_J sum_d sigma_pd I[m,d], D^T-contracted. Reduces
// to the axis-aligned kernels bit-for-bit on a flat mesh.

// K u — deformed elastic stiffness. Shared 12m (3 gathers + 9 ref fluxes).
// COL: the columnar metric (sem3d_types.h). The same substitution as the
// acoustic C-PML pair, and one flag covers BOTH uses here -- the chain rule
// and the reference fluxes, which contract the metric over the other index --
// because they read the same nine names. The stress between them is
// metric-free. Only the SOURCE changes; every expression is unchanged.
using sem3d_elastic_stiffness_deformed_fn = void (*)(const float*, const float*,
    const float*, const int*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, float, float, float*,
    float*, float*, int, int, long);
extern sem3d_elastic_stiffness_deformed_fn const sem3d_elastic_stiffness_deformed_by_flag[2];

// deformed elastic imaging: grads of lam/mu/lam2mu from the bilinear form with
// the saved forward field u and y = -dt2 invM o adjoint, physical strains +
// w_J (no direction factors). Shared 6m.
using sem3d_elastic_imaging_deformed_fn = void (*)(const float*, const float*,
    const float*, const float*, const float*, const float*, const int*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, float, float, float*,
    float*, float*, int, int, long);
extern sem3d_elastic_imaging_deformed_fn const sem3d_elastic_imaging_deformed_by_flag[2];

// ============ 3-D velocity-stress elastic kernels (P8, VS + CPML) ===========
// ElasticVSSEM3D: first-order staggered leapfrog. sigma and the 9 velocity-
// gradient memories are ELEMENT-LOCAL (per-node ownership, in-place updates,
// no races); v, the stress-divergence R and its 9 C-PML memories are nodal.
// Memory layout: sig[c*Em + off] (c = xx,yy,zz,xy,xz,yz), mvel[j*Em + off],
// msig[j*n_global + g], in the fixed 9-slot order of ElasticVSSEM3D.step.

// V: the stress divergence (wH weighting + D^T) -> R. An element the C-PML
// reaches writes it split by derivative direction into P (9 parts, slot order
// (xx,x)(xy,y)(xz,z)(xy,x)(yy,y)(yz,z)(xz,x)(yz,y)(zz,z)) instead, for
// sem3d_vs_cpml_nodal to recurse on and add to R: only such elements touch a
// node the layer reaches, so P is complete there. (Recursing on sigma and
// differentiating sigma + psi gave d(psi*sigma), not psi*d(sigma): an extra
// term in the profile's gradient that reflected low frequencies off the layer
// at any resolution.) The element-local a/b only say which elements.
__global__ void sem3d_vs_sigma_div(
    const float* __restrict__ sig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wHx, const float* __restrict__ wHy,
    const float* __restrict__ wHz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ Rx, float* __restrict__ Ry, float* __restrict__ Rz,
    float* __restrict__ P,
    int ngll, int E, long n_global);

// C: the C-PML on the divergence by direction, per node: psi_j = a psi_j + b P_j
// (the slot's axis), and R_i += the three slots of P + psi. a_n (6, n_global):
// a_x b_x a_y b_y a_z b_z. A node the layer does not reach has P = psi = 0.
__global__ void sem3d_vs_cpml_nodal(
    const float* __restrict__ a_n, const float* __restrict__ P,
    float* __restrict__ msig,
    float* __restrict__ Rx, float* __restrict__ Ry, float* __restrict__ Rz, int ng);

// N: v += dt invM (pat wav[n] - R), 3 components + optional v-history save.
__global__ void sem3d_vs_vel_update(
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ Rx, const float* __restrict__ Ry,
    const float* __restrict__ Rz, const float* __restrict__ inv_M,
    const float* __restrict__ pat_x, const float* __restrict__ pat_y,
    const float* __restrict__ pat_z, const float* __restrict__ wavelet,
    int n, float dt,
    float* __restrict__ s_vx, float* __restrict__ s_vy, float* __restrict__ s_vz,
    int ng);

// Launch KERNEL<NC> for the input's stiffness count (0 isotropic, 9 VTI, 21 TTI).
// KERNEL##_by_nc is the kernel's table, <21>, <9>, <0>.
#define SEM3D_VS_LAUNCH_NC(nc, KERNEL, grid, block, shmem, stream, ...) \
    SWEEP_LAUNCH_ON(KERNEL##_by_nc[(nc) == 21 ? 0 : (nc) == 9 ? 1 : 2], grid, block, shmem, stream, __VA_ARGS__)


// G: velocity gradients of the NEW v + v-memory update + stretch + isotropic
// stress increment (in place). Optionally records the 9 STRETCHED gradients.
using sem3d_vs_stress_update_fn = void (*)(const float*, const float*, const float*,
    float*, float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    int, const float*, const float*, const float*, float*, const int*, const float*,
    float, float, float, float, float*, int, int, long);
extern sem3d_vs_stress_update_fn const sem3d_vs_stress_update_by_nc[3];

// ---- P8 backward kernels (transposes of the three forward kernels) ---------

// K2 = stress_update^T: from lsig (6) + saved STRETCHED gv (9) produce the
// gv adjoints, run the mvel adjoint recursion in place, image lam/mu/lam2mu,
// and D^T-scatter the raw-gv adjoints into lv (the transpose of the gather+
// ref-grad). Shared 9m.
using sem3d_vs_stress_update_T_fn = void (*)(const float*, const float*, float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, int, const float*,
    const float*, const float*, float*, float*, float*, const int*, const float*, float,
    float, float, float, float*, float*, float*, float*, float*, float*, float*, int,
    int, long);
extern sem3d_vs_stress_update_T_fn const sem3d_vs_stress_update_T_by_nc[3];

// K4 = vel_update^T (nodal): grad_inv_M via the update identity, lR = -dt invM lv
// (lv then carries unchanged to the previous step as the identity part).
__global__ void sem3d_vs_vel_update_T(
    const float* __restrict__ lvx, const float* __restrict__ lvy,
    const float* __restrict__ lvz,
    const float* __restrict__ v_k_x, const float* __restrict__ v_k_y,
    const float* __restrict__ v_k_z,                       // saved v^k
    const float* __restrict__ v_km1_x, const float* __restrict__ v_km1_y,
    const float* __restrict__ v_km1_z,                     // saved v^{k-1}, may be null
    const float* __restrict__ inv_M, float dt,
    float* __restrict__ lRx, float* __restrict__ lRy, float* __restrict__ lRz,
    float* __restrict__ gradM, int ng);

// K4c = cpml_nodal^T (nodal): the psi adjoint tot_j = lR_i + lmsig_j, the msig
// recursion lmsig_j = a tot_j in place, and the adjoint of the slot's divergence
// part lP_j = lR_i + b tot_j (P enters R directly and through psi).
__global__ void sem3d_vs_cpml_nodal_T(
    const float* __restrict__ a_n,
    const float* __restrict__ lRx, const float* __restrict__ lRy,
    const float* __restrict__ lRz,
    float* __restrict__ lmsig, float* __restrict__ lP, int ng);

// K5 = sigma_div^T (element, no atomics): lsig += wH * refgrad(gather lR); an
// element the C-PML reaches gathers each slot's own adjoint lP instead (the
// transpose of sem3d_vs_sigma_div's split). Shared 9m.
__global__ void sem3d_vs_sigma_div_T(
    const float* __restrict__ lRx, const float* __restrict__ lRy,
    const float* __restrict__ lRz, const float* __restrict__ lP,
    float* __restrict__ lsig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wHx, const float* __restrict__ wHy,
    const float* __restrict__ wHz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global);


// ---- P8 on Hex27 (ElasticVSSEM3DDeformed): the four element kernels with the
// per-node metric. I_aj = d xi_a / d x_j (rows xi, eta, gamma); w_J = w3 |J|. The
// nodal kernels (vel update, C-PML recursion and their transposes) are shared.

// The deformed sem3d_vs_sigma_div: R_i = sum_a D_a^T [w_J sum_j sig_ij I_aj]. An
// element the C-PML reaches assembles the divergence split by PHYSICAL direction,
// P_ij = sum_a D_a^T [w_J sig_ij I_aj] (slot 3i + j, as the flat kernel), one stress
// row at a time so that the shared memory stays 9m at any order.
__global__ void sem3d_vs_sigma_div_deformed(
    const float* __restrict__ sig,
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
    float* __restrict__ Rx, float* __restrict__ Ry, float* __restrict__ Rz,
    float* __restrict__ P,
    int ngll, int E, long n_global);

// The deformed sem3d_vs_stress_update: the reference derivatives of the new v,
// the physical gradients dv_c/dx_j = sum_a r_ca I_aj, then the flat kernel's
// memory update, stretch, optional record and isotropic stress increment.
using sem3d_vs_stress_update_deformed_fn = void (*)(const float*, const float*,
    const float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, int, const float*, const float*, const float*, float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const int*, const float*, float, float*, int, int,
    long);
extern sem3d_vs_stress_update_deformed_fn const sem3d_vs_stress_update_deformed_by_nc[3];

// The deformed sem3d_vs_stress_update_T: the flat kernel's adjoint of the
// stretched gradients, imaging and memory recursion, then the transpose of the
// metric contraction (adjoint of r_ca = sum_j I_aj lraw_cj) and the D^T scatter.
using sem3d_vs_stress_update_T_deformed_fn = void (*)(const float*, const float*,
    float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, int,
    const float*, const float*, const float*, float*, float*, float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const int*, const float*, float, float*, float*, float*,
    float*, float*, float*, float*, int, int, long);
extern sem3d_vs_stress_update_T_deformed_fn const sem3d_vs_stress_update_T_deformed_by_nc[3];

// The deformed sem3d_vs_sigma_div_T: lsig_ij += w_J sum_a I_aj (D_a lR_i), or with
// each slot's own adjoint lP_ij in an element the C-PML reaches. Shared 9m.
__global__ void sem3d_vs_sigma_div_T_deformed(
    const float* __restrict__ lRx, const float* __restrict__ lRy,
    const float* __restrict__ lRz, const float* __restrict__ lP,
    float* __restrict__ lsig,
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
    int ngll, int E, long n_global);

} // namespace sem3d

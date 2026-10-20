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

namespace sem3d {

// K u — symmetric separable stiffness, c^2 baked in the per-axis weights.
// One block per (element, batch item); m = ngll^3 threads; shared 4m floats.
static __global__ void sem3d_acoustic_stiffness(
    const float* __restrict__ u,        // (batch, n_global)
    const int*    __restrict__ ibool,
    const float* __restrict__ D,
    const float* __restrict__ c2_jx, const float* __restrict__ c2_jy,
    const float* __restrict__ c2_jz,
    float* __restrict__ K,              // (batch, n_global)
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const float* ub = u + static_cast<size_t>(blockIdx.y) * n_global;
    float* Kb = K + static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_u  = sh;
    float* s_fx = sh + m;
    float* s_fy = sh + 2 * m;
    float* s_fz = sh + 3 * m;
    const int g = ibool[off];
    s_u[tid] = ub[g];
    __syncthreads();

    float dxi = 0.f, deta = 0.f, dzeta = 0.f;
    const int zy = (kz * ngll + ky) * ngll;            // row base for x-axis
    for (int q = 0; q < ngll; ++q) {
        dxi   += D[kx * ngll + q] * s_u[zy + q];
        deta  += D[ky * ngll + q] * s_u[(kz * ngll + q) * ngll + kx];
        dzeta += D[kz * ngll + q] * s_u[(q * ngll + ky) * ngll + kx];
    }
    s_fx[tid] = c2_jx[off] * dxi;
    s_fy[tid] = c2_jy[off] * deta;
    s_fz[tid] = c2_jz[off] * dzeta;
    __syncthreads();

    float r = 0.f;
    for (int q = 0; q < ngll; ++q) {
        r += D[q * ngll + kx] * s_fx[zy + q]
           + D[q * ngll + ky] * s_fy[(kz * ngll + q) * ngll + kx]
           + D[q * ngll + kz] * s_fz[(q * ngll + ky) * ngll + kx];
    }
    atomicAdd(&Kb[g], r);
}

// Imaging: grad_c2_j{x,y,z}[e,p] += (ref-grad_axis u)(ref-grad_axis y) at p,
// with u the saved forward field and y = -dt2 inv_M o adjoint. atomicAdd so a
// batched launch accumulates safely across batch items.
static __global__ void sem3d_acoustic_imaging(
    const float* __restrict__ u, const float* __restrict__ y,   // (batch, ng)
    const int*    __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ grad_c2_jx, float* __restrict__ grad_c2_jy,
    float* __restrict__ grad_c2_jz,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const float* ub = u + static_cast<size_t>(blockIdx.y) * n_global;
    const float* yb = y + static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_u = sh;
    float* s_y = sh + m;
    const int g = ibool[off];
    s_u[tid] = ub[g];
    s_y[tid] = yb[g];
    __syncthreads();

    float ux_ = 0.f, uy_ = 0.f, uz_ = 0.f, yx_ = 0.f, yy_ = 0.f, yz_ = 0.f;
    const int zy = (kz * ngll + ky) * ngll;
    for (int q = 0; q < ngll; ++q) {
        const float dx = D[kx * ngll + q], dy = D[ky * ngll + q], dz = D[kz * ngll + q];
        ux_ += dx * s_u[zy + q];
        uy_ += dy * s_u[(kz * ngll + q) * ngll + kx];
        uz_ += dz * s_u[(q * ngll + ky) * ngll + kx];
        yx_ += dx * s_y[zy + q];
        yy_ += dy * s_y[(kz * ngll + q) * ngll + kx];
        yz_ += dz * s_y[(q * ngll + ky) * ngll + kx];
    }
    atomicAdd(&grad_c2_jx[off], ux_ * yx_);
    atomicAdd(&grad_c2_jy[off], uy_ * yy_);
    atomicAdd(&grad_c2_jz[off], uz_ * yz_);
}

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

__device__ __forceinline__ int sem3d_slot_x(int g, const SemSlabs& s) {
    const int ix = g % s.nx, r = g / s.nx;                 // r = iz*ny + iy
    const int t = s.x.tab[ix];
    return t < 0 ? -1 : r * s.x.ns + t;
}
__device__ __forceinline__ int sem3d_slot_y(int g, const SemSlabs& s) {
    const int ix = g % s.nx, r = g / s.nx;
    const int iy = r % s.ny, iz = r / s.ny;
    const int t = s.y.tab[iy];
    return t < 0 ? -1 : (iz * s.y.ns + t) * s.nx + ix;
}
__device__ __forceinline__ int sem3d_slot_z(int g, const SemSlabs& s) {
    const int ix = g % s.nx, r = g / s.nx;
    const int iy = r % s.ny, iz = r / s.ny;
    const int t = s.z.tab[iz];
    return t < 0 ? -1 : (t * s.ny + iy) * s.nx + ix;
}
// slot -> node, for kernels that run over the slab instead of the nodes
__device__ __forceinline__ int sem3d_node_x(int sl, const SemSlabs& s) {
    const int t = sl % s.x.ns, r = sl / s.x.ns;
    return r * s.nx + s.x.planes[t];
}
__device__ __forceinline__ int sem3d_node_y(int sl, const SemSlabs& s) {
    const int ix = sl % s.nx, r = sl / s.nx;
    const int t = r % s.y.ns, iz = r / s.y.ns;
    return (iz * s.ny + s.y.planes[t]) * s.nx + ix;
}
__device__ __forceinline__ int sem3d_node_z(int sl, const SemSlabs& s) {
    const int ix = sl % s.nx, r = sl / s.nx;
    const int iy = r % s.ny, t = r / s.ny;
    return (s.z.planes[t] * s.ny + iy) * s.nx + ix;
}
__device__ __forceinline__ float sem3d_slab_read(const float* f, int sl) {
    return sl >= 0 ? f[sl] : 0.f;
}
// One assembled element-node's contribution to the fused m_d = ga_d - k_d and
// to k_d. m_d is per node; k_d lives on its axis slab under SLAB, where a
// contribution off the slab is dropped (b_d, its only weight, is +0.0 there).
template <bool SLAB>
__device__ __forceinline__ void sem3d_add_mk(float* m, float* k, float a_term,
                                             float k_term, size_t node, int slot) {
    atomicAdd(&m[node], a_term - k_term);
    if (SLAB) { if (slot >= 0) atomicAdd(&k[slot], k_term); }
    else atomicAdd(&k[node], k_term);
}

// out[s] = 1.f * a[node(s)] * v[s] over one axis's slab: the slab twin of
// sem2d_scale2(a, v, 1.f, out, ng) with the same expression, launched over
// the slab's slots instead of every node.
static __global__ void sem3d_slab_scale(const float* __restrict__ a,
                                        const float* __restrict__ v,
                                        float* __restrict__ out,
                                        int axis, SemSlabs sl, int n)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    const int g = axis == 0 ? sem3d_node_x(s, sl)
                : axis == 1 ? sem3d_node_y(s, sl) : sem3d_node_z(s, sl);
    out[s] = 1.f * a[g] * v[s];
}

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
#define SEM3D_LAUNCH_SLAB(slab, K, grid, block, shm, stream, ...)             \
    do {                                                                       \
        if (slab) K<true><<<grid, block, shm, stream>>>(__VA_ARGS__);         \
        else      K<false><<<grid, block, shm, stream>>>(__VA_ARGS__);        \
    } while (0)
// two-flag twin: <SLAB, FLAG>. FLAG is WTAPE for step_T (read w_sum from the
// tape instead of recomputing it) and COL for the deformed grads (columnar
// metric: four inv_J components identically zero, two mesh-wide constants).
#define SEM3D_LAUNCH_SLAB2(slab, wt, K, grid, block, shm, stream, ...)        \
    do {                                                                       \
        if (slab) {                                                            \
            if (wt) K<true, true><<<grid, block, shm, stream>>>(__VA_ARGS__);  \
            else    K<true, false><<<grid, block, shm, stream>>>(__VA_ARGS__); \
        } else {                                                               \
            if (wt) K<false, true><<<grid, block, shm, stream>>>(__VA_ARGS__); \
            else    K<false, false><<<grid, block, shm, stream>>>(__VA_ARGS__);\
        }                                                                      \
    } while (0)

template <bool SLAB>
static __global__ void sem3d_acoustic_cpml_grads(
    const float* __restrict__ u,
    const float* __restrict__ apx, const float* __restrict__ apy,
    const float* __restrict__ apz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ w_jac,
    const float* __restrict__ w_jx, const float* __restrict__ w_jy,
    const float* __restrict__ w_jz,
    float ihx, float ihy, float ihz,
    float* __restrict__ gx, float* __restrict__ gy, float* __restrict__ gz,
    float* __restrict__ kx, float* __restrict__ ky, float* __restrict__ kz,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    int ngll, int E, long n_global, SemSlabs sl,
    const int* __restrict__ elem_slab)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;
    // bit d of elem_slab[e]: element e may carry a non-zero psi_d. psi_d
    // grows only through b_d, so where the bit is clear every node of the
    // element has a_d*psi_d == +0.0 and the whole a*psi half of the assembly
    // contributes exactly +0.0 -- skipping it is bitwise, not an
    // approximation. One block is one element, so the branch is uniform.
    const int es = elem_slab ? elem_slab[e] : 7;
    const bool ax_on = (es & 1) != 0, ay_on = (es & 2) != 0, az_on = (es & 4) != 0;

    extern __shared__ float sh[];
    float* s_u  = sh;          float* s_ax = sh + m;
    float* s_ay = sh + 2 * m;  float* s_az = sh + 3 * m;
    float* s_fx = sh + 4 * m;  float* s_fy = sh + 5 * m;  float* s_fz = sh + 6 * m;
    const int g = ibool[off];
    s_u[tid]  = u[bo + g];
    if (SLAB) {   // a_d*psi_d on its slab; off it the array held +0.0 (batch == 1)
        if (ax_on) s_ax[tid] = sem3d_slab_read(apx, sem3d_slot_x(g, sl));
        if (ay_on) s_ay[tid] = sem3d_slab_read(apy, sem3d_slot_y(g, sl));
        if (az_on) s_az[tid] = sem3d_slab_read(apz, sem3d_slot_z(g, sl));
    } else {
        if (ax_on) s_ax[tid] = apx[bo + g];
        if (ay_on) s_ay[tid] = apy[bo + g];
        if (az_on) s_az[tid] = apz[bo + g];
    }
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float dx = 0.f, dy = 0.f, dz = 0.f, dax = 0.f, day = 0.f, daz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i * ngll + q], Dy = D[ky_i * ngll + q], Dz = D[kz_i * ngll + q];
        dx  += Dx * s_u[zy + q];
        dy  += Dy * s_u[(kz_i * ngll + q) * ngll + kx_i];
        dz  += Dz * s_u[(q * ngll + ky_i) * ngll + kx_i];
        if (ax_on) dax += Dx * s_ax[zy + q];
        if (ay_on) day += Dy * s_ay[(kz_i * ngll + q) * ngll + kx_i];
        if (az_on) daz += Dz * s_az[(q * ngll + ky_i) * ngll + kx_i];
    }
    const float wj = w_jac[tid];
    if (SLAB) {
        // g_d is READ only where b_d or db_d is non-zero (step_newmark
        // multiplies it by exactly those two), which is the axis slab -- so
        // it is stored there and a contribution off it is dropped, not kept
        // to be multiplied by +0.0 later
        const int sx = sem3d_slot_x(g, sl), sy = sem3d_slot_y(g, sl),
                  sz = sem3d_slot_z(g, sl);
        if (sx >= 0) atomicAdd(&gx[sx], wj * dx * ihx);
        if (sy >= 0) atomicAdd(&gy[sy], wj * dy * ihy);
        if (sz >= 0) atomicAdd(&gz[sz], wj * dz * ihz);
    } else {
    atomicAdd(&gx[bo + g],  wj * dx * ihx);
    atomicAdd(&gy[bo + g],  wj * dy * ihy);
    atomicAdd(&gz[bo + g],  wj * dz * ihz);
    }
    const float ax_t = ax_on ? wj * dax * ihx : 0.f;
    const float ay_t = ay_on ? wj * day * ihy : 0.f;
    const float az_t = az_on ? wj * daz * ihz : 0.f;
    s_fx[tid] = w_jx[tid] * dx;
    s_fy[tid] = w_jy[tid] * dy;
    s_fz[tid] = w_jz[tid] * dz;
    __syncthreads();
    float rx = 0.f, ry = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        rx += D[q * ngll + kx_i] * s_fx[zy + q];
        ry += D[q * ngll + ky_i] * s_fy[(kz_i * ngll + q) * ngll + kx_i];
        rz += D[q * ngll + kz_i] * s_fz[(q * ngll + ky_i) * ngll + kx_i];
    }
    // m_d = ga_d - k_d in ONE atomic: the node kernels read only the
    // difference on the interior. k_d is still needed for the b_d*k_d term
    // inside the band, and b_d is +0.0 off the slab, so it goes there.
    sem3d_add_mk<SLAB>(mx, kx, ax_t, rx, bo + g, SLAB ? sem3d_slot_x(g, sl) : 0);
    sem3d_add_mk<SLAB>(my, ky, ay_t, ry, bo + g, SLAB ? sem3d_slot_y(g, sl) : 0);
    sem3d_add_mk<SLAB>(mz, kz, az_t, rz, bo + g, SLAB ? sem3d_slot_z(g, sl) : 0);
}

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
template <bool SLAB>
static __global__ void sem3d_acoustic_cpml_step_newmark(
    const float* __restrict__ gx, const float* __restrict__ gy, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ ky, const float* __restrict__ kz,
    const float* __restrict__ mx, const float* __restrict__ my, const float* __restrict__ mz,
    const float* __restrict__ inv_M,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx,
    const float* __restrict__ a_y, const float* __restrict__ b_y, const float* __restrict__ dby,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz,
    const unsigned char* __restrict__ band_node,
    const float* __restrict__ psi_x, const float* __restrict__ psi_y, const float* __restrict__ psi_z,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_y, const float* __restrict__ zeta_z,
    const float* __restrict__ u, const float* __restrict__ u_prev,
    const float* __restrict__ vp2, const float* __restrict__ pattern,
    const float* __restrict__ wavelet, int n, float dt2,
    float* __restrict__ w_sum, float* __restrict__ u_next,
    float* __restrict__ psi_x_n, float* __restrict__ psi_y_n, float* __restrict__ psi_z_n,
    float* __restrict__ zeta_x_n, float* __restrict__ zeta_y_n, float* __restrict__ zeta_z_n,
    int ng, SemSlabs sl)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // ONE byte decides the branch. Testing the six coefficients themselves
    // cost six fp32 loads at every node -- 24 B against the 40 B the interior
    // branch then uses -- and the band path reloads what it needs anyway.
    if (!band_node[i]) {
        const float imv = inv_M[i];
        // on the interior k_d and ga_d enter ONLY as their difference, so
        // the fused m_d answers with one array instead of two
        const float t_x = mx[i]*imv, t_y = my[i]*imv, t_z = mz[i]*imv;
        const float ws = t_x + t_y + t_z;
        w_sum[i] = ws;
        u_next[i] = 2.f*u[i] - u_prev[i]
                  + dt2 * (vp2[i]*ws + imv*pattern[i]*wavelet[n]);
        // psi/zeta stay zero; both rotation buffers start at zero and only
        // absorbing nodes are ever written, so skipping the store preserves them.
        return;
    }
    int sx = i, sy = i, sz = i;
    if (SLAB) { sx = sem3d_slot_x(i, sl); sy = sem3d_slot_y(i, sl); sz = sem3d_slot_z(i, sl); }
    const float px = SLAB ? sem3d_slab_read(psi_x, sx) : psi_x[i];
    const float py = SLAB ? sem3d_slab_read(psi_y, sy) : psi_y[i];
    const float pz = SLAB ? sem3d_slab_read(psi_z, sz) : psi_z[i];
    const float zx = SLAB ? sem3d_slab_read(zeta_x, sx) : zeta_x[i];
    const float zy = SLAB ? sem3d_slab_read(zeta_y, sy) : zeta_y[i];
    const float zz = SLAB ? sem3d_slab_read(zeta_z, sz) : zeta_z[i];
    const float im = inv_M[i];
    const float qx = (SLAB ? sem3d_slab_read(gx, sx) : gx[i])*im,
                qy = (SLAB ? sem3d_slab_read(gy, sy) : gy[i])*im,
                qz = (SLAB ? sem3d_slab_read(gz, sz) : gz[i])*im;
    const float ox = 1.f + b_x[i], oy = 1.f + b_y[i], oz = 1.f + b_z[i];
    const float kvx = (SLAB ? sem3d_slab_read(kx, sx) : kx[i])*im,
                kvy = (SLAB ? sem3d_slab_read(ky, sy) : ky[i])*im,
                kvz = (SLAB ? sem3d_slab_read(kz, sz) : kz[i])*im;
    // (1+b)(-k im) + db q + ga im  ==  m im - b (k im) + db q, and k alone is
    // weighted by b, which is +0.0 off its slab -- so k_d lives there
    const float tx = mx[i]*im - b_x[i]*kvx + dbx[i]*qx;
    const float ty = my[i]*im - b_y[i]*kvy + dby[i]*qy;
    const float tz = mz[i]*im - b_z[i]*kvz + dbz[i]*qz;
    const float ws = ox*tx + a_x[i]*zx + oy*ty + a_y[i]*zy
                   + oz*tz + a_z[i]*zz;
    w_sum[i] = ws;
    const float npx = b_x[i]*qx + a_x[i]*px;
    const float npy = b_y[i]*qy + a_y[i]*py;
    const float npz = b_z[i]*qz + a_z[i]*pz;
    const float nzx = b_x[i]*tx + a_x[i]*zx;
    const float nzy = b_y[i]*ty + a_y[i]*zy;
    const float nzz = b_z[i]*tz + a_z[i]*zz;
    if (!SLAB || sx >= 0) { psi_x_n[sx] = npx; zeta_x_n[sx] = nzx; }
    if (!SLAB || sy >= 0) { psi_y_n[sy] = npy; zeta_y_n[sy] = nzy; }
    if (!SLAB || sz >= 0) { psi_z_n[sz] = npz; zeta_z_n[sz] = nzz; }
    u_next[i] = 2.f*u[i] - u_prev[i]
              + dt2 * (vp2[i]*ws + im*pattern[i]*wavelet[n]);
}

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
template <bool SLAB, bool WTAPE>
static __global__ void sem3d_acoustic_cpml_step_T(
    const float* __restrict__ gx, const float* __restrict__ gy, const float* __restrict__ gz,
    const float* __restrict__ kx, const float* __restrict__ ky, const float* __restrict__ kz,
    const float* __restrict__ mx, const float* __restrict__ my, const float* __restrict__ mz,
    const float* __restrict__ inv_M,
    const float* __restrict__ a_x, const float* __restrict__ b_x, const float* __restrict__ dbx,
    const float* __restrict__ a_y, const float* __restrict__ b_y, const float* __restrict__ dby,
    const float* __restrict__ a_z, const float* __restrict__ b_z, const float* __restrict__ dbz,
    const unsigned char* __restrict__ band_node,
    const float* __restrict__ zeta_x, const float* __restrict__ zeta_y, const float* __restrict__ zeta_z,
    const float* __restrict__ gw,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_y, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_y, const float* __restrict__ lzeta_z,
    float* __restrict__ gbx, float* __restrict__ gby, float* __restrict__ gbz,
    float* __restrict__ gbkx, float* __restrict__ gbky, float* __restrict__ gbkz,
    float* __restrict__ gbax, float* __restrict__ gbay, float* __restrict__ gbaz,
    float* __restrict__ w_sum_out, int ng, SemSlabs sl)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    // ONE byte decides the branch. Testing the six coefficients themselves
    // cost six fp32 loads at every node -- 24 B against the 40 B the interior
    // branch then uses -- and the band path reloads what it needs anyway.
    if (!band_node[i]) {
        const float imv = inv_M[i];
        if (!WTAPE) {
            const float t_x = mx[i]*imv, t_y = my[i]*imv, t_z = mz[i]*imv;
            w_sum_out[i] = t_x + t_y + t_z;
        }
        const float g = gw[i];
        // gt = gw on all three axes here, so gq = 0 and gbx/gby/gbz stay zero.
        gbkx[i] = -imv*g; gbky[i] = -imv*g; gbkz[i] = -imv*g;
        gbax[i] =  imv*g; gbay[i] =  imv*g; gbaz[i] =  imv*g;
        return;
    }
    int sx = i, sy = i, sz = i;
    if (SLAB) { sx = sem3d_slot_x(i, sl); sy = sem3d_slot_y(i, sl); sz = sem3d_slot_z(i, sl); }
    // the adjoint carries: off their slab the old array held a non-zero
    // dead value, consumed only through b_d == +0.0 below; reading 0.0f
    // instead changes at most the sign of that exact-zero product
    const float lpx = SLAB ? sem3d_slab_read(lpsi_x, sx) : lpsi_x[i];
    const float lpy = SLAB ? sem3d_slab_read(lpsi_y, sy) : lpsi_y[i];
    const float lpz = SLAB ? sem3d_slab_read(lpsi_z, sz) : lpsi_z[i];
    const float lzx = SLAB ? sem3d_slab_read(lzeta_x, sx) : lzeta_x[i];
    const float lzy = SLAB ? sem3d_slab_read(lzeta_y, sy) : lzeta_y[i];
    const float lzz = SLAB ? sem3d_slab_read(lzeta_z, sz) : lzeta_z[i];
    const float im = inv_M[i];
    const float ox = 1.f + b_x[i], oy = 1.f + b_y[i], oz = 1.f + b_z[i];
    if (!WTAPE) {   // recompute w_sum^k from the forward assembly at step k
        const float zx = SLAB ? sem3d_slab_read(zeta_x, sx) : zeta_x[i];
        const float zy = SLAB ? sem3d_slab_read(zeta_y, sy) : zeta_y[i];
        const float zz = SLAB ? sem3d_slab_read(zeta_z, sz) : zeta_z[i];
        const float qx = (SLAB ? sem3d_slab_read(gx, sx) : gx[i])*im,
                    qy = (SLAB ? sem3d_slab_read(gy, sy) : gy[i])*im,
                    qz = (SLAB ? sem3d_slab_read(gz, sz) : gz[i])*im;
        const float kvx = (SLAB ? sem3d_slab_read(kx, sx) : kx[i])*im,
                    kvy = (SLAB ? sem3d_slab_read(ky, sy) : ky[i])*im,
                    kvz = (SLAB ? sem3d_slab_read(kz, sz) : kz[i])*im;
        // (1+b)(-k im) + db q + ga im  ==  m im - b (k im) + db q, and k alone is
        // weighted by b, which is +0.0 off its slab -- so k_d lives there
        const float tx = mx[i]*im - b_x[i]*kvx + dbx[i]*qx;
        const float ty = my[i]*im - b_y[i]*kvy + dby[i]*qy;
        const float tz = mz[i]*im - b_z[i]*kvz + dbz[i]*qz;
        w_sum_out[i] = ox*tx + a_x[i]*zx + oy*ty + a_y[i]*zy
                     + oz*tz + a_z[i]*zz;
    }
    const float gwv = gw[i];
    const float gtx = ox*gwv + b_x[i]*lzx;
    const float gty = oy*gwv + b_y[i]*lzy;
    const float gtz = oz*gwv + b_z[i]*lzz;
    const float gqx = dbx[i]*gtx + b_x[i]*lpx;
    const float gqy = dby[i]*gty + b_y[i]*lpy;
    const float gqz = dbz[i]*gtz + b_z[i]*lpz;
    gbx[i] = im*gqx;    gby[i] = im*gqy;    gbz[i] = im*gqz;
    gbkx[i] = -im*ox*gtx; gbky[i] = -im*oy*gty; gbkz[i] = -im*oz*gtz;
    gbax[i] = im*gtx;   gbay[i] = im*gty;   gbaz[i] = im*gtz;
}

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
template <bool SLAB>
static __global__ void sem3d_acoustic_cpml_grads_T(
    const float* __restrict__ gbx, const float* __restrict__ gby, const float* __restrict__ gbz,
    const float* __restrict__ gbkx, const float* __restrict__ gbky, const float* __restrict__ gbkz,
    const float* __restrict__ gbax, const float* __restrict__ gbay, const float* __restrict__ gbaz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ w_jac,
    const float* __restrict__ w_jx, const float* __restrict__ w_jy,
    const float* __restrict__ w_jz,
    float ihx, float ihy, float ihz,
    float* __restrict__ lu,
    float* __restrict__ lapx, float* __restrict__ lapy, float* __restrict__ lapz,
    int ngll, int E, long n_global, const int* __restrict__ elem_slab,
    SemSlabs sl)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;
    const int g = ibool[off];
    // the a*psi element flags (sem3d_types.h), here for the TRANSPOSE: under
    // the slab layout combine writes the carries only on their slab, so
    // lap_d off slab d is never read and an element that touches no plane of
    // it need not produce one. The driver therefore passes the flags only
    // when psi_slab is on; nullptr keeps every element flagged.
    const int es = elem_slab ? elem_slab[e] : 7;
    const bool ax_on = (es & 1) != 0, ay_on = (es & 2) != 0, az_on = (es & 4) != 0;

    extern __shared__ float sh[];
    float* s_ex = sh;          float* s_ey = sh + m;      float* s_ez = sh + 2 * m;
    float* s_cx = sh + 3 * m;  float* s_cy = sh + 4 * m;  float* s_cz = sh + 5 * m;
    s_ex[tid] = gbkx[bo + g];
    s_ey[tid] = gbky[bo + g];
    s_ez[tid] = gbkz[bo + g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float ekx = 0.f, eky = 0.f, ekz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        ekx += D[kx_i * ngll + q] * s_ex[zy + q];
        eky += D[ky_i * ngll + q] * s_ey[(kz_i * ngll + q) * ngll + kx_i];
        ekz += D[kz_i * ngll + q] * s_ez[(q * ngll + ky_i) * ngll + kx_i];
    }
    const float wj = w_jac[tid];
    // lambda(d_a u) = w_jac ihd_a * gb_a  (direct) + w_j_a * refgrad_a(gbk_a)
    const float ldx = wj * ihx * gbx[bo + g] + w_jx[tid] * ekx;
    const float ldy = wj * ihy * gby[bo + g] + w_jy[tid] * eky;
    const float ldz = wj * ihz * gbz[bo + g] + w_jz[tid] * ekz;
    __syncthreads();
    s_cx[tid] = ldx;  s_cy[tid] = ldy;  s_cz[tid] = ldz;
    // reuse s_e* for the apsi adjoint fields
    if (ax_on) s_ex[tid] = wj * ihx * gbax[bo + g];
    if (ay_on) s_ey[tid] = wj * ihy * gbay[bo + g];
    if (az_on) s_ez[tid] = wj * ihz * gbaz[bo + g];
    __syncthreads();
    // the u adjoint: its own loop, textually the pre-flag one, so the guards
    // below cannot make the compiler re-schedule the arithmetic that survives
    float l_u = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx_i], Ty = D[q * ngll + ky_i], Tz = D[q * ngll + kz_i];
        l_u  += Tx * s_cx[zy + q]
              + Ty * s_cy[(kz_i * ngll + q) * ngll + kx_i]
              + Tz * s_cz[(q * ngll + ky_i) * ngll + kx_i];
    }
    atomicAdd(&lu[bo + g], l_u);
    if (ax_on || ay_on || az_on) {
        float l_ax = 0.f, l_ay = 0.f, l_az = 0.f;
        for (int q = 0; q < ngll; ++q) {
            const float Tx = D[q * ngll + kx_i], Ty = D[q * ngll + ky_i], Tz = D[q * ngll + kz_i];
            if (ax_on) l_ax += Tx * s_ex[zy + q];
            if (ay_on) l_ay += Ty * s_ey[(kz_i * ngll + q) * ngll + kx_i];
            if (az_on) l_az += Tz * s_ez[(q * ngll + ky_i) * ngll + kx_i];
        }
        if (SLAB) {
            const int sx = sem3d_slot_x(g, sl), sy = sem3d_slot_y(g, sl),
                      sz = sem3d_slot_z(g, sl);
            if (ax_on && sx >= 0) atomicAdd(&lapx[sx], l_ax);
            if (ay_on && sy >= 0) atomicAdd(&lapy[sy], l_ay);
            if (az_on && sz >= 0) atomicAdd(&lapz[sz], l_az);
        } else {
        if (ax_on) atomicAdd(&lapx[bo + g], l_ax);
        if (ay_on) atomicAdd(&lapy[bo + g], l_ay);
        if (az_on) atomicAdd(&lapz[bo + g], l_az);
        }
    }
}

// backward combine (node): Newmark^T grads + finalise u/memory adjoints.
// SLAB: the adjoint carries are updated only on their slab (the old code
// wrote a_d*(l + lap) at every node; off the slab that value was never read
// with a non-zero weight -- see step_T). gw stays full-length -- a node-wise
// quantity read at every slab node -- but lap* is on its slab too: this
// kernel is its only reader and reads it only where the slot exists.
template <bool SLAB>
static __global__ void sem3d_acoustic_cpml_combine(
    const float* __restrict__ gA, const float* __restrict__ gB, const float* __restrict__ lu_G,
    const float* __restrict__ vp2, const float* __restrict__ w_sum,
    const float* __restrict__ lpsi_x, const float* __restrict__ lpsi_y, const float* __restrict__ lpsi_z,
    const float* __restrict__ lzeta_x, const float* __restrict__ lzeta_y, const float* __restrict__ lzeta_z,
    const float* __restrict__ lapx, const float* __restrict__ lapy, const float* __restrict__ lapz,
    const float* __restrict__ gw,
    const float* __restrict__ a_x, const float* __restrict__ a_y, const float* __restrict__ a_z,
    float* __restrict__ lu_k,
    float* __restrict__ lpsi_x_n, float* __restrict__ lpsi_y_n, float* __restrict__ lpsi_z_n,
    float* __restrict__ lzeta_x_n, float* __restrict__ lzeta_y_n, float* __restrict__ lzeta_z_n,
    float* __restrict__ grad_vp2, float dt2, int ng, SemSlabs sl)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float ga = gA[i];
    // BATCH: non-atomic accumulation into a batch-SHARED gradient buffer.
    // Under B > 1 this is a lost-update race -- needs a node index + atomicAdd.
    grad_vp2[i] += dt2 * w_sum[i] * ga;
    lu_k[i] = gB[i] + 2.f*ga + lu_G[i];
    if (SLAB) {
        const int sx = sem3d_slot_x(i, sl), sy = sem3d_slot_y(i, sl), sz = sem3d_slot_z(i, sl);
        if (sx >= 0) {
            lpsi_x_n[sx] = a_x[i] * (lpsi_x[sx] + lapx[sx]);
            lzeta_x_n[sx] = a_x[i] * (lzeta_x[sx] + gw[i]);
        }
        if (sy >= 0) {
            lpsi_y_n[sy] = a_y[i] * (lpsi_y[sy] + lapy[sy]);
            lzeta_y_n[sy] = a_y[i] * (lzeta_y[sy] + gw[i]);
        }
        if (sz >= 0) {
            lpsi_z_n[sz] = a_z[i] * (lpsi_z[sz] + lapz[sz]);
            lzeta_z_n[sz] = a_z[i] * (lzeta_z[sz] + gw[i]);
        }
    } else {
        lpsi_x_n[i] = a_x[i] * (lpsi_x[i] + lapx[i]);
        lpsi_y_n[i] = a_y[i] * (lpsi_y[i] + lapy[i]);
        lpsi_z_n[i] = a_z[i] * (lpsi_z[i] + lapz[i]);
        lzeta_x_n[i] = a_x[i] * (lzeta_x[i] + gw[i]);
        lzeta_y_n[i] = a_y[i] * (lzeta_y[i] + gw[i]);
        lzeta_z_n[i] = a_z[i] * (lzeta_z[i] + gw[i]);
    }
}

// ============ 3-D acoustic DEFORMED-metric kernels (Hex27 topography) ========
// Metric-general analogue of the axis-aligned sem3d_acoustic_* kernels: consume
// the full per-node 3x3 inv_J (I[m,d] = d xi_m/d x_d, m in {xi,eta,gam}, d in
// {x,y,z}) + w_J = w3 |J|, mirroring AcousticSEM3DDeformed. The physical
// gradient P_d(f) = I[xi,d] G_xi(f) + I[eta,d] G_eta(f) + I[gam,d] G_gam(f); the
// stiffness is P_d^T(w_J c2 P_d(u)) (symmetric -> reused for the reverse march).
// On a flat mesh (I diagonal, cross terms 0) these reduce to the axis-aligned
// arithmetic bit-for-bit. All kernels launch grid (E, batch), m = ngll^3 threads.

// K u — deformed stiffness. Shared 4m (s_u + 3 ref fluxes F_xi/F_eta/F_gam).
static __global__ void sem3d_acoustic_stiffness_deformed(
    const float* __restrict__ u, const int* __restrict__ ibool,
    const float* __restrict__ D, const float* __restrict__ c2,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float* __restrict__ K, int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const float* ub = u + static_cast<size_t>(blockIdx.y) * n_global;
    float* Kb = K + static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_u  = sh;
    float* s_fx = sh + m; float* s_fy = sh + 2 * m; float* s_fz = sh + 3 * m;
    const int g = ibool[off];
    s_u[tid] = ub[g];
    __syncthreads();

    const int zy = (kz * ngll + ky) * ngll;
    float dxi = 0.f, deta = 0.f, dgam = 0.f;
    for (int q = 0; q < ngll; ++q) {
        dxi  += D[kx * ngll + q] * s_u[zy + q];
        deta += D[ky * ngll + q] * s_u[(kz * ngll + q) * ngll + kx];
        dgam += D[kz * ngll + q] * s_u[(q * ngll + ky) * ngll + kx];
    }
    const float Ixx = ixx[off], Ixy = ixy[off], Ixz = ixz[off];
    const float Iex = iex[off], Iey = iey[off], Iez = iez[off];
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    const float dudx = Ixx * dxi + Iex * deta + Igx * dgam;
    const float dudy = Ixy * dxi + Iey * deta + Igy * dgam;
    const float dudz = Ixz * dxi + Iez * deta + Igz * dgam;
    const float c2v = c2[off], w = wJ[off];
    const float qx = c2v * dudx, qy = c2v * dudy, qz = c2v * dudz;
    s_fx[tid] = w * (qx * Ixx + qy * Ixy + qz * Ixz);       // F_xi
    s_fy[tid] = w * (qx * Iex + qy * Iey + qz * Iez);       // F_eta
    s_fz[tid] = w * (qx * Igx + qy * Igy + qz * Igz);       // F_gam
    __syncthreads();

    float r = 0.f;
    for (int q = 0; q < ngll; ++q) {
        r += D[q * ngll + kx] * s_fx[zy + q]
           + D[q * ngll + ky] * s_fy[(kz * ngll + q) * ngll + kx]
           + D[q * ngll + kz] * s_fz[(q * ngll + ky) * ngll + kx];
    }
    atomicAdd(&Kb[g], r);
}

// imaging: grad_c2[e,p] += w_J (P_x u P_x y + P_y u P_y y + P_z u P_z y) with u
// the saved forward field and y = -dt2 invM o adjoint. Shared 2m.
static __global__ void sem3d_acoustic_imaging_deformed(
    const float* __restrict__ u, const float* __restrict__ y,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float* __restrict__ grad_c2, int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx = tid % ngll, ky = (tid / ngll) % ngll, kz = tid / (ngll * ngll);
    const int off = e * m + tid;
    const float* ub = u + static_cast<size_t>(blockIdx.y) * n_global;
    const float* yb = y + static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_u = sh; float* s_y = sh + m;
    const int g = ibool[off];
    s_u[tid] = ub[g]; s_y[tid] = yb[g];
    __syncthreads();

    const int zy = (kz * ngll + ky) * ngll;
    float uxi = 0.f, ueta = 0.f, ugam = 0.f, yxi = 0.f, yeta = 0.f, ygam = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float dx = D[kx * ngll + q], dy = D[ky * ngll + q], dz = D[kz * ngll + q];
        const int iy = (kz * ngll + q) * ngll + kx, iz = (q * ngll + ky) * ngll + kx;
        uxi += dx * s_u[zy + q]; ueta += dy * s_u[iy]; ugam += dz * s_u[iz];
        yxi += dx * s_y[zy + q]; yeta += dy * s_y[iy]; ygam += dz * s_y[iz];
    }
    const float Ixx = ixx[off], Ixy = ixy[off], Ixz = ixz[off];
    const float Iex = iex[off], Iey = iey[off], Iez = iez[off];
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    const float uX = Ixx * uxi + Iex * ueta + Igx * ugam;
    const float uY = Ixy * uxi + Iey * ueta + Igy * ugam;
    const float uZ = Ixz * uxi + Iez * ueta + Igz * ugam;
    const float yX = Ixx * yxi + Iex * yeta + Igx * ygam;
    const float yY = Ixy * yxi + Iey * yeta + Igy * ygam;
    const float yZ = Ixz * yxi + Iez * yeta + Igz * ygam;
    atomicAdd(&grad_c2[off], wJ[off] * (uX * yX + uY * yY + uZ * yZ));
}

// deformed CPML forward 9-field grads: direct g_d = w_J P_d(u); weak
// k_d = P_d^T(w_J P_d(u)) = sum_m G_m^T(I[m,d] w_J P_d(u)); apsi
// ga_d = w_J P_d(a psi_d). Shared 4m (s_u + 3 apsi) reused as 3m ref fluxes.
// COL: the columnar metric of the extruded-topography builder (sem3d_types.h).
// Only the SOURCE of the six affected components changes -- every expression
// after it is the text the full-3x3 path already had, so what COL removes is
// six (E, ngll^3) streams, and nvcc folds the resulting exact zeros. Measured
// 1.25x at nex=48; the arithmetic it also removes is worth nothing on its
// own (0.999x for twice the FLOPs).
template <bool SLAB, bool COL>
static __global__ void sem3d_acoustic_cpml_grads_deformed(
    const float* __restrict__ u,
    const float* __restrict__ apx, const float* __restrict__ apy,
    const float* __restrict__ apz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float col_ixx, float col_iey,
    float* __restrict__ gx, float* __restrict__ gy, float* __restrict__ gz,
    float* __restrict__ kx, float* __restrict__ ky, float* __restrict__ kz,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    int ngll, int E, long n_global, SemSlabs sl,
    const int* __restrict__ elem_slab)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;
    // bit d of elem_slab[e]: element e may carry a non-zero psi_d. psi_d
    // grows only through b_d, so where the bit is clear every node of the
    // element has a_d*psi_d == +0.0 and the whole a*psi half of the assembly
    // contributes exactly +0.0 -- skipping it is bitwise, not an
    // approximation. One block is one element, so the branch is uniform.
    const int es = elem_slab ? elem_slab[e] : 7;
    const bool ax_on = (es & 1) != 0, ay_on = (es & 2) != 0, az_on = (es & 4) != 0;

    extern __shared__ float sh[];
    float* s_u  = sh;         float* s_ax = sh + m;
    float* s_ay = sh + 2 * m; float* s_az = sh + 3 * m;
    const int g = ibool[off];
    s_u[tid]  = u[bo + g];
    if (SLAB) {   // see sem3d_acoustic_cpml_grads
        if (ax_on) s_ax[tid] = sem3d_slab_read(apx, sem3d_slot_x(g, sl));
        if (ay_on) s_ay[tid] = sem3d_slab_read(apy, sem3d_slot_y(g, sl));
        if (az_on) s_az[tid] = sem3d_slab_read(apz, sem3d_slot_z(g, sl));
    } else {
        if (ax_on) s_ax[tid] = apx[bo + g];
        if (ay_on) s_ay[tid] = apy[bo + g];
        if (az_on) s_az[tid] = apz[bo + g];
    }
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float uxi = 0.f, ueta = 0.f, ugam = 0.f;
    float axxi = 0.f, axeta = 0.f, axgam = 0.f;
    float ayxi = 0.f, ayeta = 0.f, aygam = 0.f;
    float azxi = 0.f, azeta = 0.f, azgam = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i * ngll + q], Dy = D[ky_i * ngll + q], Dz = D[kz_i * ngll + q];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        uxi += Dx * s_u[zy + q]; ueta += Dy * s_u[iy]; ugam += Dz * s_u[iz];
        if (ax_on) { axxi += Dx * s_ax[zy + q]; axeta += Dy * s_ax[iy]; axgam += Dz * s_ax[iz]; }
        if (ay_on) { ayxi += Dx * s_ay[zy + q]; ayeta += Dy * s_ay[iy]; aygam += Dz * s_ay[iz]; }
        if (az_on) { azxi += Dx * s_az[zy + q]; azeta += Dy * s_az[iy]; azgam += Dz * s_az[iz]; }
    }
    float Ixx, Ixy, Ixz, Iex, Iey, Iez;
    if (COL) { Ixx = col_ixx; Iey = col_iey; Ixy = Ixz = Iex = Iez = 0.f; }
    else { Ixx = ixx[off]; Ixy = ixy[off]; Ixz = ixz[off];
           Iex = iex[off]; Iey = iey[off]; Iez = iez[off]; }
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    const float w = wJ[off];
    const float dudx = Ixx * uxi + Iex * ueta + Igx * ugam;
    const float dudy = Ixy * uxi + Iey * ueta + Igy * ugam;
    const float dudz = Ixz * uxi + Iez * ueta + Igz * ugam;
    // apsi_d differentiated along its OWN physical axis d
    const float dapx_dx = Ixx * axxi + Iex * axeta + Igx * axgam;
    const float dapy_dy = Ixy * ayxi + Iey * ayeta + Igy * aygam;
    const float dapz_dz = Ixz * azxi + Iez * azeta + Igz * azgam;
    if (SLAB) {
        const int sx = sem3d_slot_x(g, sl), sy = sem3d_slot_y(g, sl),
                  sz = sem3d_slot_z(g, sl);
        if (sx >= 0) atomicAdd(&gx[sx], w * dudx);
        if (sy >= 0) atomicAdd(&gy[sy], w * dudy);
        if (sz >= 0) atomicAdd(&gz[sz], w * dudz);
    } else {
    atomicAdd(&gx[bo + g],  w * dudx);
    atomicAdd(&gy[bo + g],  w * dudy);
    atomicAdd(&gz[bo + g],  w * dudz);
    }
    const float ax_t = ax_on ? w * dapx_dx : 0.f;
    const float ay_t = ay_on ? w * dapy_dy : 0.f;
    const float az_t = az_on ? w * dapz_dz : 0.f;

    // weak k_d: three ref fluxes I[m,d] w_J P_d(u), D^T-contracted. Reuse shared.
    const float qx = w * dudx, qy = w * dudy, qz = w * dudz;
    // k_x from q_x
    __syncthreads();
    s_u[tid]  = qx * Ixx;   // F_xi
    s_ax[tid] = qx * Iex;   // F_eta
    s_ay[tid] = qx * Igx;   // F_gam
    __syncthreads();
    float rk = 0.f;
    for (int q = 0; q < ngll; ++q)
        rk += D[q * ngll + kx_i] * s_u[zy + q]
            + D[q * ngll + ky_i] * s_ax[(kz_i * ngll + q) * ngll + kx_i]
            + D[q * ngll + kz_i] * s_ay[(q * ngll + ky_i) * ngll + kx_i];
    sem3d_add_mk<SLAB>(mx, kx, ax_t, rk, bo + g, SLAB ? sem3d_slot_x(g, sl) : 0);
    // k_y from q_y
    __syncthreads();
    s_u[tid]  = qy * Ixy;
    s_ax[tid] = qy * Iey;
    s_ay[tid] = qy * Igy;
    __syncthreads();
    rk = 0.f;
    for (int q = 0; q < ngll; ++q)
        rk += D[q * ngll + kx_i] * s_u[zy + q]
            + D[q * ngll + ky_i] * s_ax[(kz_i * ngll + q) * ngll + kx_i]
            + D[q * ngll + kz_i] * s_ay[(q * ngll + ky_i) * ngll + kx_i];
    sem3d_add_mk<SLAB>(my, ky, ay_t, rk, bo + g, SLAB ? sem3d_slot_y(g, sl) : 0);
    // k_z from q_z
    __syncthreads();
    s_u[tid]  = qz * Ixz;
    s_ax[tid] = qz * Iez;
    s_ay[tid] = qz * Igz;
    __syncthreads();
    rk = 0.f;
    for (int q = 0; q < ngll; ++q)
        rk += D[q * ngll + kx_i] * s_u[zy + q]
            + D[q * ngll + ky_i] * s_ax[(kz_i * ngll + q) * ngll + kx_i]
            + D[q * ngll + kz_i] * s_ay[(q * ngll + ky_i) * ngll + kx_i];
    sem3d_add_mk<SLAB>(mz, kz, az_t, rk, bo + g, SLAB ? sem3d_slot_z(g, sl) : 0);
}

// deformed CPML grads transpose. lp_d = w_J gb_d + w_J P_d(gbk_d); lu =
// sum_d P_d^T(lp_d) = sum_m G_m^T(sum_d I[m,d] lp_d); lap_d = P_d^T(w_J gba_d).
// Shared 3m, reused across the lu / lapx / lapy / lapz D^T rounds.
template <bool SLAB, bool COL>   // COL: see sem3d_acoustic_cpml_grads_deformed
static __global__ void sem3d_acoustic_cpml_grads_T_deformed(
    const float* __restrict__ gbx, const float* __restrict__ gby, const float* __restrict__ gbz,
    const float* __restrict__ gbkx, const float* __restrict__ gbky, const float* __restrict__ gbkz,
    const float* __restrict__ gbax, const float* __restrict__ gbay, const float* __restrict__ gbaz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float col_ixx, float col_iey,
    float* __restrict__ lu,
    float* __restrict__ lapx, float* __restrict__ lapy, float* __restrict__ lapz,
    int ngll, int E, long n_global, const int* __restrict__ elem_slab,
    SemSlabs sl)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;
    const int g = ibool[off];
    const int zy = (kz_i * ngll + ky_i) * ngll;
    // the a*psi element flags (sem3d_types.h), here for the TRANSPOSE: under
    // the slab layout combine writes the carries only on their slab, so
    // lap_d off slab d is never read and an element that touches no plane of
    // it need not produce one. The driver therefore passes the flags only
    // when psi_slab is on; nullptr keeps every element flagged.
    const int es = elem_slab ? elem_slab[e] : 7;
    const bool ax_on = (es & 1) != 0, ay_on = (es & 2) != 0, az_on = (es & 4) != 0;


    float Ixx, Ixy, Ixz, Iex, Iey, Iez;
    if (COL) { Ixx = col_ixx; Iey = col_iey; Ixy = Ixz = Iex = Iez = 0.f; }
    else { Ixx = ixx[off]; Ixy = ixy[off]; Ixz = ixz[off];
           Iex = iex[off]; Iey = iey[off]; Iez = iez[off]; }
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    const float w = wJ[off];

    extern __shared__ float sh[];
    float* s0 = sh; float* s1 = sh + m; float* s2 = sh + 2 * m;

    // P_d(gbk_d): gather gbk_d then ref-derivs. Compute the three physical
    // derivatives one direction at a time (gbkx along x, etc.).
    s0[tid] = gbkx[bo + g]; s1[tid] = gbky[bo + g]; s2[tid] = gbkz[bo + g];
    __syncthreads();
    float kx_xi = 0.f, kx_eta = 0.f, kx_gam = 0.f;
    float ky_xi = 0.f, ky_eta = 0.f, ky_gam = 0.f;
    float kz_xi = 0.f, kz_eta = 0.f, kz_gam = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i * ngll + q], Dy = D[ky_i * ngll + q], Dz = D[kz_i * ngll + q];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        kx_xi += Dx * s0[zy + q]; kx_eta += Dy * s0[iy]; kx_gam += Dz * s0[iz];
        ky_xi += Dx * s1[zy + q]; ky_eta += Dy * s1[iy]; ky_gam += Dz * s1[iz];
        kz_xi += Dx * s2[zy + q]; kz_eta += Dy * s2[iy]; kz_gam += Dz * s2[iz];
    }
    const float Pkx = Ixx * kx_xi + Iex * kx_eta + Igx * kx_gam;   // P_x(gbkx)
    const float Pky = Ixy * ky_xi + Iey * ky_eta + Igy * ky_gam;   // P_y(gbky)
    const float Pkz = Ixz * kz_xi + Iez * kz_eta + Igz * kz_gam;   // P_z(gbkz)
    const float lpx = w * gbx[bo + g] + w * Pkx;
    const float lpy = w * gby[bo + g] + w * Pky;
    const float lpz = w * gbz[bo + g] + w * Pkz;
    // lu = sum_m G_m^T( sum_d I[m,d] lp_d )
    __syncthreads();
    s0[tid] = Ixx * lpx + Ixy * lpy + Ixz * lpz;   // e_xi
    s1[tid] = Iex * lpx + Iey * lpy + Iez * lpz;   // e_eta
    s2[tid] = Igx * lpx + Igy * lpy + Igz * lpz;   // e_gam
    __syncthreads();
    float lu_ = 0.f;
    for (int q = 0; q < ngll; ++q)
        lu_ += D[q * ngll + kx_i] * s0[zy + q]
             + D[q * ngll + ky_i] * s1[(kz_i * ngll + q) * ngll + kx_i]
             + D[q * ngll + kz_i] * s2[(q * ngll + ky_i) * ngll + kx_i];
    atomicAdd(&lu[bo + g], lu_);

    // lap_d = P_d^T(w gba_d) = sum_m G_m^T(I[m,d] w gba_d), one round each.
    if (ax_on) {
        const float sax = w * gbax[bo + g];
        __syncthreads();
        s0[tid] = Ixx * sax; s1[tid] = Iex * sax; s2[tid] = Igx * sax;
        __syncthreads();
        float la = 0.f;
        for (int q = 0; q < ngll; ++q)
            la += D[q * ngll + kx_i] * s0[zy + q]
                + D[q * ngll + ky_i] * s1[(kz_i * ngll + q) * ngll + kx_i]
                + D[q * ngll + kz_i] * s2[(q * ngll + ky_i) * ngll + kx_i];
        if (SLAB) { const int s = sem3d_slot_x(g, sl);
                     if (s >= 0) atomicAdd(&lapx[s], la); }
        else atomicAdd(&lapx[bo + g], la);
    }

    if (ay_on) {
        const float say = w * gbay[bo + g];
        __syncthreads();
        s0[tid] = Ixy * say; s1[tid] = Iey * say; s2[tid] = Igy * say;
        __syncthreads();
        float la = 0.f;
        for (int q = 0; q < ngll; ++q)
            la += D[q * ngll + kx_i] * s0[zy + q]
                + D[q * ngll + ky_i] * s1[(kz_i * ngll + q) * ngll + kx_i]
                + D[q * ngll + kz_i] * s2[(q * ngll + ky_i) * ngll + kx_i];
        if (SLAB) { const int s = sem3d_slot_y(g, sl);
                     if (s >= 0) atomicAdd(&lapy[s], la); }
        else atomicAdd(&lapy[bo + g], la);
    }

    if (az_on) {
        const float saz = w * gbaz[bo + g];
        __syncthreads();
        s0[tid] = Ixz * saz; s1[tid] = Iez * saz; s2[tid] = Igz * saz;
        __syncthreads();
        float la = 0.f;
        for (int q = 0; q < ngll; ++q)
            la += D[q * ngll + kx_i] * s0[zy + q]
                + D[q * ngll + ky_i] * s1[(kz_i * ngll + q) * ngll + kx_i]
                + D[q * ngll + kz_i] * s2[(q * ngll + ky_i) * ngll + kx_i];
        if (SLAB) { const int s = sem3d_slot_z(g, sl);
                     if (s >= 0) atomicAdd(&lapz[s], la); }
        else atomicAdd(&lapz[bo + g], la);
    }
}

// ================== 3-D elastic kernels (P7, bare) ==========================
// Symmetric stiffness (ElasticSEM3D.stiffness_apply): 9 ref-grads g_{p,a} of
// the 3 displacement components, 9 fluxes F (normal + shear), 3 D^T sums.
// Shared 12m floats (3 gathers + 9 fluxes).
static __global__ void sem3d_elastic_stiffness(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ w3,
    float Gxx, float Gyy, float Gzz, float Gxy, float Gxz, float Gyz,
    const float* __restrict__ jac,           // (E, m) |J| (may be uniform)
    float* __restrict__ Kx, float* __restrict__ Ky, float* __restrict__ Kz,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_ux = sh; float* s_uy = sh + m; float* s_uz = sh + 2 * m;
    float* F = sh + 3 * m;                   // 9 flux arrays, F[j*m + tid]
    const int g = ibool[off];
    s_ux[tid] = ux[bo + g]; s_uy[tid] = uy[bo + g]; s_uz[tid] = uz[bo + g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float gxx=0.f,gyx=0.f,gzx=0.f, gxy=0.f,gyy=0.f,gzy=0.f, gxz=0.f,gyz=0.f,gzz=0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i*ngll+q], Dy = D[ky_i*ngll+q], Dz = D[kz_i*ngll+q];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        gxx += Dx*s_ux[zy+q]; gyx += Dx*s_uy[zy+q]; gzx += Dx*s_uz[zy+q];
        gxy += Dy*s_ux[iy];   gyy += Dy*s_uy[iy];   gzy += Dy*s_uz[iy];
        gxz += Dz*s_ux[iz];   gyz += Dz*s_uy[iz];   gzz += Dz*s_uz[iz];
    }
    const float w = w3[off] * jac[off];
    const float l = lam[off], u_ = mu[off], l2m = lam2mu[off];
    __syncthreads();
    F[0*m+tid] = w*(l2m*Gxx*gxx + l*Gxy*gyy + l*Gxz*gzz);            // F_xx
    F[1*m+tid] = w*(l*Gxy*gxx + l2m*Gyy*gyy + l*Gyz*gzz);            // F_yy
    F[2*m+tid] = w*(l*Gxz*gxx + l*Gyz*gyy + l2m*Gzz*gzz);            // F_zz
    F[3*m+tid] = w*u_*(Gyy*gxy + Gxy*gyx);                           // F_xy
    F[4*m+tid] = w*u_*(Gxy*gxy + Gxx*gyx);                           // F_yx
    F[5*m+tid] = w*u_*(Gzz*gxz + Gxz*gzx);                           // F_xz
    F[6*m+tid] = w*u_*(Gxz*gxz + Gxx*gzx);                           // F_zx
    F[7*m+tid] = w*u_*(Gzz*gyz + Gyz*gzy);                           // F_yz
    F[8*m+tid] = w*u_*(Gyz*gyz + Gyy*gzy);                           // F_zy
    __syncthreads();
    float rx = 0.f, ry = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q*ngll+kx_i], Ty = D[q*ngll+ky_i], Tz = D[q*ngll+kz_i];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        rx += Tx*F[0*m+zy+q] + Ty*F[3*m+iy] + Tz*F[5*m+iz];
        ry += Tx*F[4*m+zy+q] + Ty*F[1*m+iy] + Tz*F[7*m+iz];
        rz += Tx*F[6*m+zy+q] + Ty*F[8*m+iy] + Tz*F[2*m+iz];
    }
    atomicAdd(&Kx[bo + g], rx);
    atomicAdd(&Ky[bo + g], ry);
    atomicAdd(&Kz[bo + g], rz);
}

// imaging: grads of lam/mu/lam2mu from the bilinear-form derivative with the
// saved forward field u and y = -dt2 invM o adjoint. Shared 6m.
static __global__ void sem3d_elastic_imaging(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const float* __restrict__ yx, const float* __restrict__ yy,
    const float* __restrict__ yz,
    const int*    __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ w3, const float* __restrict__ jac,
    float Gxx, float Gyy, float Gzz, float Gxy, float Gxz, float Gyz,
    float* __restrict__ gL2m, float* __restrict__ gLam, float* __restrict__ gMu,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s[6] = {sh, sh+m, sh+2*m, sh+3*m, sh+4*m, sh+5*m};
    const int g = ibool[off];
    s[0][tid] = ux[bo+g]; s[1][tid] = uy[bo+g]; s[2][tid] = uz[bo+g];
    s[3][tid] = yx[bo+g]; s[4][tid] = yy[bo+g]; s[5][tid] = yz[bo+g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float d[18];                            // u: g_{p,a} p*3+a ; y at 9+
    for (int j = 0; j < 18; ++j) d[j] = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i*ngll+q], Dy = D[ky_i*ngll+q], Dz = D[kz_i*ngll+q];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        for (int p = 0; p < 3; ++p) {
            d[p*3+0]     += Dx * s[p][zy+q];
            d[p*3+1]     += Dy * s[p][iy];
            d[p*3+2]     += Dz * s[p][iz];
            d[9 + p*3+0] += Dx * s[3+p][zy+q];
            d[9 + p*3+1] += Dy * s[3+p][iy];
            d[9 + p*3+2] += Dz * s[3+p][iz];
        }
    }
    // aliases: u-grads g_{p,a} = d[p*3+a], y-grads h_{p,a} = d[9+p*3+a]
    #define GU(p,a) d[(p)*3+(a)]
    #define GY(p,a) d[9+(p)*3+(a)]
    const float w = w3[off] * jac[off];
    gL2m[off] += w * (Gxx*GU(0,0)*GY(0,0) + Gyy*GU(1,1)*GY(1,1) + Gzz*GU(2,2)*GY(2,2));
    gLam[off] += w * (Gxy*(GU(1,1)*GY(0,0) + GU(0,0)*GY(1,1))
                    + Gxz*(GU(2,2)*GY(0,0) + GU(0,0)*GY(2,2))
                    + Gyz*(GU(2,2)*GY(1,1) + GU(1,1)*GY(2,2)));
    gMu[off]  += w * (Gyy*GU(0,1)*GY(0,1) + Gxy*(GU(1,0)*GY(0,1) + GU(0,1)*GY(1,0))
                    + Gxx*GU(1,0)*GY(1,0)
                    + Gzz*GU(0,2)*GY(0,2) + Gxz*(GU(2,0)*GY(0,2) + GU(0,2)*GY(2,0))
                    + Gxx*GU(2,0)*GY(2,0)
                    + Gzz*GU(1,2)*GY(1,2) + Gyz*(GU(2,1)*GY(1,2) + GU(1,2)*GY(2,1))
                    + Gyy*GU(2,1)*GY(2,1));
    #undef GU
    #undef GY
}

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
template <bool COL>
static __global__ void sem3d_elastic_stiffness_deformed(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu, const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float col_ixx, float col_iey,
    float* __restrict__ Kx, float* __restrict__ Ky, float* __restrict__ Kz,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_ux = sh; float* s_uy = sh + m; float* s_uz = sh + 2 * m;
    float* F = sh + 3 * m;                    // 9 flux arrays F[j*m + tid]
    const int g = ibool[off];
    s_ux[tid] = ux[bo + g]; s_uy[tid] = uy[bo + g]; s_uz[tid] = uz[bo + g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float uxxi=0.f,uxet=0.f,uxga=0.f, uyxi=0.f,uyet=0.f,uyga=0.f, uzxi=0.f,uzet=0.f,uzga=0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i*ngll+q], Dy = D[ky_i*ngll+q], Dz = D[kz_i*ngll+q];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        uxxi += Dx*s_ux[zy+q]; uxet += Dy*s_ux[iy]; uxga += Dz*s_ux[iz];
        uyxi += Dx*s_uy[zy+q]; uyet += Dy*s_uy[iy]; uyga += Dz*s_uy[iz];
        uzxi += Dx*s_uz[zy+q]; uzet += Dy*s_uz[iy]; uzga += Dz*s_uz[iz];
    }
    float Ixx, Ixy, Ixz, Iex, Iey, Iez;
    if (COL) { Ixx = col_ixx; Iey = col_iey; Ixy = Ixz = Iex = Iez = 0.f; }
    else { Ixx = ixx[off]; Ixy = ixy[off]; Ixz = ixz[off];
           Iex = iex[off]; Iey = iey[off]; Iez = iez[off]; }
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    // physical displacement gradients g_{p,d} = du_p/dx_d
    const float gxx = Ixx*uxxi + Iex*uxet + Igx*uxga;
    const float gxy = Ixy*uxxi + Iey*uxet + Igy*uxga;
    const float gxz = Ixz*uxxi + Iez*uxet + Igz*uxga;
    const float gyx = Ixx*uyxi + Iex*uyet + Igx*uyga;
    const float gyy = Ixy*uyxi + Iey*uyet + Igy*uyga;
    const float gyz = Ixz*uyxi + Iez*uyet + Igz*uyga;
    const float gzx = Ixx*uzxi + Iex*uzet + Igx*uzga;
    const float gzy = Ixy*uzxi + Iey*uzet + Igy*uzga;
    const float gzz = Ixz*uzxi + Iez*uzet + Igz*uzga;
    const float l = lam[off], u_ = mu[off], l2m = lam2mu[off], w = wJ[off];
    const float sxx = l2m*gxx + l*(gyy+gzz);
    const float syy = l2m*gyy + l*(gxx+gzz);
    const float szz = l2m*gzz + l*(gxx+gyy);
    const float sxy = u_*(gxy+gyx), sxz = u_*(gxz+gzx), syz = u_*(gyz+gzy);
    __syncthreads();
    // per-equation ref fluxes F^p_m = w sum_d sigma_pd I[m,d]
    // eq x: (sxx,sxy,sxz); eq y: (sxy,syy,syz); eq z: (sxz,syz,szz)
    F[0*m+tid] = w*(sxx*Ixx + sxy*Ixy + sxz*Ixz);   // Fx_xi
    F[1*m+tid] = w*(sxx*Iex + sxy*Iey + sxz*Iez);   // Fx_eta
    F[2*m+tid] = w*(sxx*Igx + sxy*Igy + sxz*Igz);   // Fx_gam
    F[3*m+tid] = w*(sxy*Ixx + syy*Ixy + syz*Ixz);   // Fy_xi
    F[4*m+tid] = w*(sxy*Iex + syy*Iey + syz*Iez);   // Fy_eta
    F[5*m+tid] = w*(sxy*Igx + syy*Igy + syz*Igz);   // Fy_gam
    F[6*m+tid] = w*(sxz*Ixx + syz*Ixy + szz*Ixz);   // Fz_xi
    F[7*m+tid] = w*(sxz*Iex + syz*Iey + szz*Iez);   // Fz_eta
    F[8*m+tid] = w*(sxz*Igx + syz*Igy + szz*Igz);   // Fz_gam
    __syncthreads();
    float rx = 0.f, ry = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q*ngll+kx_i], Ty = D[q*ngll+ky_i], Tz = D[q*ngll+kz_i];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        rx += Tx*F[0*m+zy+q] + Ty*F[1*m+iy] + Tz*F[2*m+iz];
        ry += Tx*F[3*m+zy+q] + Ty*F[4*m+iy] + Tz*F[5*m+iz];
        rz += Tx*F[6*m+zy+q] + Ty*F[7*m+iy] + Tz*F[8*m+iz];
    }
    atomicAdd(&Kx[bo + g], rx);
    atomicAdd(&Ky[bo + g], ry);
    atomicAdd(&Kz[bo + g], rz);
}

// deformed elastic imaging: grads of lam/mu/lam2mu from the bilinear form with
// the saved forward field u and y = -dt2 invM o adjoint, physical strains +
// w_J (no direction factors). Shared 6m.
template <bool COL>   // COL: see sem3d_elastic_stiffness_deformed
static __global__ void sem3d_elastic_imaging_deformed(
    const float* __restrict__ ux, const float* __restrict__ uy,
    const float* __restrict__ uz,
    const float* __restrict__ yx, const float* __restrict__ yy,
    const float* __restrict__ yz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    const float* __restrict__ wJ,
    const float* __restrict__ ixx, const float* __restrict__ ixy, const float* __restrict__ ixz,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ igx, const float* __restrict__ igy, const float* __restrict__ igz,
    float col_ixx, float col_iey,
    float* __restrict__ gL2m, float* __restrict__ gLam, float* __restrict__ gMu,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s[6] = {sh, sh+m, sh+2*m, sh+3*m, sh+4*m, sh+5*m};
    const int g = ibool[off];
    s[0][tid] = ux[bo+g]; s[1][tid] = uy[bo+g]; s[2][tid] = uz[bo+g];
    s[3][tid] = yx[bo+g]; s[4][tid] = yy[bo+g]; s[5][tid] = yz[bo+g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float rxi[6], ret[6], rga[6];
    for (int j = 0; j < 6; ++j) { rxi[j] = 0.f; ret[j] = 0.f; rga[j] = 0.f; }
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i*ngll+q], Dy = D[ky_i*ngll+q], Dz = D[kz_i*ngll+q];
        const int iy = (kz_i*ngll+q)*ngll + kx_i, iz = (q*ngll+ky_i)*ngll + kx_i;
        for (int j = 0; j < 6; ++j) {
            rxi[j] += Dx * s[j][zy+q]; ret[j] += Dy * s[j][iy]; rga[j] += Dz * s[j][iz];
        }
    }
    float Ixx, Ixy, Ixz, Iex, Iey, Iez;
    if (COL) { Ixx = col_ixx; Iey = col_iey; Ixy = Ixz = Iex = Iez = 0.f; }
    else { Ixx = ixx[off]; Ixy = ixy[off]; Ixz = ixz[off];
           Iex = iex[off]; Iey = iey[off]; Iez = iez[off]; }
    const float Igx = igx[off], Igy = igy[off], Igz = igz[off];
    // physical grads g[p][d]: p in {ux,uy,uz,yx,yy,yz} rows -> here u=0..2, y=3..5
    float gd[6][3];
    for (int j = 0; j < 6; ++j) {
        gd[j][0] = Ixx*rxi[j] + Iex*ret[j] + Igx*rga[j];   // d/dx
        gd[j][1] = Ixy*rxi[j] + Iey*ret[j] + Igy*rga[j];   // d/dy
        gd[j][2] = Ixz*rxi[j] + Iez*ret[j] + Igz*rga[j];   // d/dz
    }
    // GU(p,a) = gd[p][a] (u = component p in 0..2); GY(p,a) = gd[3+p][a]
    #define GU(p,a) gd[(p)][(a)]
    #define GY(p,a) gd[3+(p)][(a)]
    const float w = wJ[off];
    gL2m[off] += w * (GU(0,0)*GY(0,0) + GU(1,1)*GY(1,1) + GU(2,2)*GY(2,2));
    gLam[off] += w * ((GU(1,1)+GU(2,2))*GY(0,0) + (GU(0,0)+GU(2,2))*GY(1,1)
                    + (GU(0,0)+GU(1,1))*GY(2,2));
    gMu[off]  += w * ((GU(0,1)+GU(1,0))*(GY(0,1)+GY(1,0))
                    + (GU(0,2)+GU(2,0))*(GY(0,2)+GY(2,0))
                    + (GU(1,2)+GU(2,1))*(GY(1,2)+GY(2,1)));
    #undef GU
    #undef GY
}

// ============ 3-D velocity-stress elastic kernels (P8, VS + CPML) ===========
// ElasticVSSEM3D: first-order staggered leapfrog. sigma and the 18 CPML
// memories are ELEMENT-LOCAL (per-node ownership, in-place updates, no
// races); only v and the stress-divergence R are assembled. Memory layout:
// sig[c*Em + off] (c = xx,yy,zz,xy,xz,yz), msig/mvel[j*Em + off] in the fixed
// 9-slot order of ElasticVSSEM3D.step.

// V: sigma-memory update + stretch + wH weighting + D^T divergence -> R.
static __global__ void sem3d_vs_sigma_div(
    const float* __restrict__ sig, float* __restrict__ msig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wHx, const float* __restrict__ wHy,
    const float* __restrict__ wHz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float* __restrict__ Rx, float* __restrict__ Ry, float* __restrict__ Rz,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];        // 9 flux arrays F[j*m + tid]
    const float sxx = sig[0 * Em + off], syy = sig[1 * Em + off], szz = sig[2 * Em + off];
    const float sxy = sig[3 * Em + off], sxz = sig[4 * Em + off], syz = sig[5 * Em + off];
    // 9 stretched fields, order (sxx,x)(sxy,y)(sxz,z)(sxy,x)(syy,y)(syz,z)(sxz,x)(syz,y)(szz,z)
    const float s9[9] = {sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz};
    float str[9];
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, az, ax, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bz, bx, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float mm = ca[j] * msig[j * Em + off] + cb[j] * s9[j];
            msig[j * Em + off] = mm;
            str[j] = s9[j] + mm;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 9; ++j) str[j] = s9[j];
    }
    const float wx = wHx[off], wy = wHy[off], wz = wHz[off];
    const float w9[9] = {wx, wy, wz, wx, wy, wz, wx, wy, wz};
    #pragma unroll
    for (int j = 0; j < 9; ++j) sh[j * m + tid] = w9[j] * str[j];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float rx = 0.f, ry = 0.f, rz = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx_i], Ty = D[q * ngll + ky_i], Tz = D[q * ngll + kz_i];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        rx += Tx * sh[0 * m + zy + q] + Ty * sh[1 * m + iy] + Tz * sh[2 * m + iz];
        ry += Tx * sh[3 * m + zy + q] + Ty * sh[4 * m + iy] + Tz * sh[5 * m + iz];
        rz += Tx * sh[6 * m + zy + q] + Ty * sh[7 * m + iy] + Tz * sh[8 * m + iz];
    }
    const int g = ibool[off];
    atomicAdd(&Rx[bo + g], rx);
    atomicAdd(&Ry[bo + g], ry);
    atomicAdd(&Rz[bo + g], rz);
}

// N: v += dt invM (pat wav[n] - R), 3 components + optional v-history save.
static __global__ void sem3d_vs_vel_update(
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ Rx, const float* __restrict__ Ry,
    const float* __restrict__ Rz, const float* __restrict__ inv_M,
    const float* __restrict__ pat_x, const float* __restrict__ pat_y,
    const float* __restrict__ pat_z, const float* __restrict__ wavelet,
    int n, float dt,
    float* __restrict__ s_vx, float* __restrict__ s_vy, float* __restrict__ s_vz,
    int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float f = dt * inv_M[i], wv = wavelet[n];
    const float nvx = vx[i] + f * (pat_x[i] * wv - Rx[i]);
    const float nvy = vy[i] + f * (pat_y[i] * wv - Ry[i]);
    const float nvz = vz[i] + f * (pat_z[i] * wv - Rz[i]);
    vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
    if (s_vx) { s_vx[i] = nvx; s_vy[i] = nvy; s_vz[i] = nvz; }
}

// G: velocity gradients of the NEW v + v-memory update + stretch + isotropic
// stress increment (in place). Optionally records the 9 STRETCHED gradients.
static __global__ void sem3d_vs_stress_update(
    const float* __restrict__ vx, const float* __restrict__ vy,
    const float* __restrict__ vz,
    float* __restrict__ sig, float* __restrict__ mvel,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float ihx, float ihy, float ihz, float dt,
    float* __restrict__ save_gv,             // (9, E*m) slice for this step; may be null
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_vx = sh; float* s_vy = sh + m; float* s_vz = sh + 2 * m;
    const int g0 = ibool[off];
    s_vx[tid] = vx[bo + g0]; s_vy[tid] = vy[bo + g0]; s_vz[tid] = vz[bo + g0];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float dxx = 0.f, dyx = 0.f, dzx = 0.f;   // d/dx of vx,vy,vz
    float dxy = 0.f, dyy = 0.f, dzy = 0.f;   // d/dy
    float dxz = 0.f, dyz = 0.f, dzz = 0.f;   // d/dz
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i * ngll + q], Dy = D[ky_i * ngll + q], Dz = D[kz_i * ngll + q];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        dxx += Dx * s_vx[zy + q]; dyx += Dx * s_vy[zy + q]; dzx += Dx * s_vz[zy + q];
        dxy += Dy * s_vx[iy];     dyy += Dy * s_vy[iy];     dzy += Dy * s_vz[iy];
        dxz += Dz * s_vx[iz];     dyz += Dz * s_vy[iz];     dzz += Dz * s_vz[iz];
    }
    // fixed 9-slot order: gvx_x, gvy_y, gvz_z, gvy_x, gvx_y, gvz_x, gvx_z, gvz_y, gvy_z
    float gv[9] = {dxx * ihx, dyy * ihy, dzz * ihz,
                   dyx * ihx, dxy * ihy, dzx * ihx,
                   dxz * ihz, dzy * ihy, dyz * ihz};
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, ax, az, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bx, bz, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float mm = ca[j] * mvel[j * Em + off] + cb[j] * gv[j];
            mvel[j * Em + off] = mm;
            gv[j] = gv[j] + mm;
        }
    }
    if (save_gv) {
        #pragma unroll
        for (int j = 0; j < 9; ++j) save_gv[j * Em + off] = gv[j];
    }
    const float l = lam[off], u_ = mu[off], l2 = lam2mu[off];
    sig[0 * Em + off] += dt * (l2 * gv[0] + l * (gv[1] + gv[2]));
    sig[1 * Em + off] += dt * (l2 * gv[1] + l * (gv[0] + gv[2]));
    sig[2 * Em + off] += dt * (l2 * gv[2] + l * (gv[0] + gv[1]));
    sig[3 * Em + off] += dt * u_ * (gv[3] + gv[4]);
    sig[4 * Em + off] += dt * u_ * (gv[5] + gv[6]);
    sig[5 * Em + off] += dt * u_ * (gv[7] + gv[8]);
}

// ---- P8 backward kernels (transposes of the three forward kernels) ---------

// K2 = stress_update^T: from lsig (6) + saved STRETCHED gv (9) produce the
// gv adjoints, run the mvel adjoint recursion in place, image lam/mu/lam2mu,
// and D^T-scatter the raw-gv adjoints into lv (the transpose of the gather+
// ref-grad). Shared 9m.
static __global__ void sem3d_vs_stress_update_T(
    const float* __restrict__ lsig, const float* __restrict__ save_gv,
    float* __restrict__ lmvel,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ lam2mu,
    const int* __restrict__ ibool, const float* __restrict__ D,
    float ihx, float ihy, float ihz, float dt,
    float* __restrict__ lvx, float* __restrict__ lvy, float* __restrict__ lvz,
    float* __restrict__ gLam, float* __restrict__ gMu, float* __restrict__ gL2m,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    const float lxx = lsig[0 * Em + off], lyy = lsig[1 * Em + off], lzz = lsig[2 * Em + off];
    const float lxy = lsig[3 * Em + off], lxz = lsig[4 * Em + off], lyz = lsig[5 * Em + off];
    const float l = lam[off], u_ = mu[off], l2 = lam2mu[off];
    // lambda(gv_str) = dt C^T lsig, in the fixed 9-slot order
    float lg[9];
    lg[0] = dt * (l2 * lxx + l * (lyy + lzz));
    lg[1] = dt * (l2 * lyy + l * (lxx + lzz));
    lg[2] = dt * (l2 * lzz + l * (lxx + lyy));
    lg[3] = dt * u_ * lxy;  lg[4] = dt * u_ * lxy;
    lg[5] = dt * u_ * lxz;  lg[6] = dt * u_ * lxz;
    lg[7] = dt * u_ * lyz;  lg[8] = dt * u_ * lyz;
    // imaging with the saved stretched gradients
    const float* gv = save_gv;
    const float g0 = gv[0*Em+off], g1 = gv[1*Em+off], g2 = gv[2*Em+off];
    const float g3 = gv[3*Em+off], g4 = gv[4*Em+off], g5 = gv[5*Em+off];
    const float g6 = gv[6*Em+off], g7 = gv[7*Em+off], g8 = gv[8*Em+off];
    gL2m[off] += dt * (g0 * lxx + g1 * lyy + g2 * lzz);
    gLam[off] += dt * ((g1 + g2) * lxx + (g0 + g2) * lyy + (g0 + g1) * lzz);
    gMu[off]  += dt * ((g3 + g4) * lxy + (g5 + g6) * lxz + (g7 + g8) * lyz);
    // mvel adjoint recursion + raw-gv adjoint
    float lraw[9];
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, ax, az, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bx, bz, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float tot = lg[j] + lmvel[j * Em + off];
            lraw[j] = lg[j] + cb[j] * tot;
            lmvel[j * Em + off] = ca[j] * tot;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 9; ++j) lraw[j] = lg[j];
    }
    // Grad^T: cov per (comp, axis) with ih, then D^T scatter into lv.
    extern __shared__ float sh[];
    // slots: (vx,x)0 (vy,y)1 (vz,z)2 (vy,x)3 (vx,y)4 (vz,x)5 (vx,z)6 (vz,y)7 (vy,z)8
    sh[0 * m + tid] = ihx * lraw[0];   // vx along x
    sh[1 * m + tid] = ihy * lraw[4];   // vx along y
    sh[2 * m + tid] = ihz * lraw[6];   // vx along z
    sh[3 * m + tid] = ihx * lraw[3];   // vy along x
    sh[4 * m + tid] = ihy * lraw[1];   // vy along y
    sh[5 * m + tid] = ihz * lraw[8];   // vy along z
    sh[6 * m + tid] = ihx * lraw[5];   // vz along x
    sh[7 * m + tid] = ihy * lraw[7];   // vz along y
    sh[8 * m + tid] = ihz * lraw[2];   // vz along z
    __syncthreads();
    const int zy = (kz_i * ngll + ky_i) * ngll;
    float ax_ = 0.f, ay_ = 0.f, az_ = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Tx = D[q * ngll + kx_i], Ty = D[q * ngll + ky_i], Tz = D[q * ngll + kz_i];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        ax_ += Tx * sh[0 * m + zy + q] + Ty * sh[1 * m + iy] + Tz * sh[2 * m + iz];
        ay_ += Tx * sh[3 * m + zy + q] + Ty * sh[4 * m + iy] + Tz * sh[5 * m + iz];
        az_ += Tx * sh[6 * m + zy + q] + Ty * sh[7 * m + iy] + Tz * sh[8 * m + iz];
    }
    const int g = ibool[off];
    atomicAdd(&lvx[bo + g], ax_);
    atomicAdd(&lvy[bo + g], ay_);
    atomicAdd(&lvz[bo + g], az_);
}

// K4 = vel_update^T (nodal): grad_inv_M via the update identity, lR = -dt invM lv
// (lv then carries unchanged to the previous step as the identity part).
static __global__ void sem3d_vs_vel_update_T(
    const float* __restrict__ lvx, const float* __restrict__ lvy,
    const float* __restrict__ lvz,
    const float* __restrict__ v_k_x, const float* __restrict__ v_k_y,
    const float* __restrict__ v_k_z,                       // saved v^k
    const float* __restrict__ v_km1_x, const float* __restrict__ v_km1_y,
    const float* __restrict__ v_km1_z,                     // saved v^{k-1}, may be null
    const float* __restrict__ inv_M, float dt,
    float* __restrict__ lRx, float* __restrict__ lRy, float* __restrict__ lRz,
    float* __restrict__ gradM, int ng)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng) return;
    const float im = inv_M[i];
    const float f = -dt * im;
    lRx[i] = f * lvx[i];  lRy[i] = f * lvy[i];  lRz[i] = f * lvz[i];
    if (im != 0.f) {
        const float px = v_km1_x ? v_km1_x[i] : 0.f;
        const float py = v_km1_y ? v_km1_y[i] : 0.f;
        const float pz = v_km1_z ? v_km1_z[i] : 0.f;
        // BATCH: non-atomic accumulation into a batch-SHARED gradient buffer.
        // Under B > 1 this is a lost-update race -- needs a node index + atomicAdd.
        gradM[i] += (lvx[i] * (v_k_x[i] - px) + lvy[i] * (v_k_y[i] - py)
                   + lvz[i] * (v_k_z[i] - pz)) / im;
    }
}

// K5 = sigma_div^T (element, no atomics): lsig_str = wH * refgrad(gather lR);
// msig adjoint recursion in place; lsig += lsig_str + B_s * tot.
static __global__ void sem3d_vs_sigma_div_T(
    const float* __restrict__ lRx, const float* __restrict__ lRy,
    const float* __restrict__ lRz,
    float* __restrict__ lsig, float* __restrict__ lmsig,
    const float* __restrict__ a_x, const float* __restrict__ b_x,
    const float* __restrict__ a_y, const float* __restrict__ b_y,
    const float* __restrict__ a_z, const float* __restrict__ b_z,   // null = bare
    const float* __restrict__ wHx, const float* __restrict__ wHy,
    const float* __restrict__ wHz,
    const int* __restrict__ ibool, const float* __restrict__ D,
    int ngll, int E, long n_global)
{
    const int e = blockIdx.x;
    if (e >= E) return;
    const int m = ngll * ngll * ngll, tid = threadIdx.x;
    if (tid >= m) return;
    const int kx_i = tid % ngll, ky_i = (tid / ngll) % ngll, kz_i = tid / (ngll * ngll);
    const int off = e * m + tid;
    const long Em = static_cast<long>(E) * m;
    const size_t bo = static_cast<size_t>(blockIdx.y) * n_global;

    extern __shared__ float sh[];
    float* s_x = sh; float* s_y = sh + m; float* s_z = sh + 2 * m;
    const int g = ibool[off];
    s_x[tid] = lRx[bo + g]; s_y[tid] = lRy[bo + g]; s_z[tid] = lRz[bo + g];
    __syncthreads();

    const int zy = (kz_i * ngll + ky_i) * ngll;
    float rx_x = 0.f, rx_y = 0.f, rx_z = 0.f;
    float ry_x = 0.f, ry_y = 0.f, ry_z = 0.f;
    float rz_x = 0.f, rz_y = 0.f, rz_z = 0.f;
    for (int q = 0; q < ngll; ++q) {
        const float Dx = D[kx_i * ngll + q], Dy = D[ky_i * ngll + q], Dz = D[kz_i * ngll + q];
        const int iy = (kz_i * ngll + q) * ngll + kx_i, iz = (q * ngll + ky_i) * ngll + kx_i;
        rx_x += Dx * s_x[zy + q]; rx_y += Dy * s_x[iy]; rx_z += Dz * s_x[iz];
        ry_x += Dx * s_y[zy + q]; ry_y += Dy * s_y[iy]; ry_z += Dz * s_y[iz];
        rz_x += Dx * s_z[zy + q]; rz_y += Dy * s_z[iy]; rz_z += Dz * s_z[iz];
    }
    const float wx = wHx[off], wy = wHy[off], wz = wHz[off];
    // slots: (sxx,x)(sxy,y)(sxz,z)(sxy,x)(syy,y)(syz,z)(sxz,x)(syz,y)(szz,z)
    float ls[9];
    ls[0] = wx * rx_x;  ls[1] = wy * rx_y;  ls[2] = wz * rx_z;
    ls[3] = wx * ry_x;  ls[4] = wy * ry_y;  ls[5] = wz * ry_z;
    ls[6] = wx * rz_x;  ls[7] = wy * rz_y;  ls[8] = wz * rz_z;
    float add6[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // map slot -> sigma component: 0->xx,1->xy,2->xz,3->xy,4->yy,5->yz,6->xz,7->yz,8->zz
    const int comp[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
    if (a_x) {
        const float ax = a_x[off], bx = b_x[off], ay = a_y[off], by = b_y[off];
        const float az = a_z[off], bz = b_z[off];
        const float ca[9] = {ax, ay, az, ax, ay, az, ax, ay, az};
        const float cb[9] = {bx, by, bz, bx, by, bz, bx, by, bz};
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float tot = ls[j] + lmsig[j * Em + off];
            add6[comp[j]] += ls[j] + cb[j] * tot;
            lmsig[j * Em + off] = ca[j] * tot;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 9; ++j) add6[comp[j]] += ls[j];
    }
    #pragma unroll
    for (int c = 0; c < 6; ++c) lsig[c * Em + off] += add6[c];
}

} // namespace sem3d

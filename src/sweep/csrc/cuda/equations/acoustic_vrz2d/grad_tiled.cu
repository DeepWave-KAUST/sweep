// 2-D tiled VRZ model gradient, orders 4-8, and -- boundary saving, orders
// 2-8 -- the reverse reconstruction step fused into it.
//
// The gradient replaces calculate_grad_vrz2d_fused (which recomputes D_d p at
// every divergence tap: O(M^2) loads, 96 registers) and the
// build_vrz_grad_fields + calculate_grad_vrz2d pair (which round-trips four
// scratch fields over the padded grid).  Same per-cell result, operand order
// kept:
//   c_d(j) = (lam*vp)(j) * D_d p(j),  e_d(j) = (((lam*vp)*vp)*z)(j) * D_d p(j),
//   both 0 when tap j is outside [M, n-M)^2;
//   grad_vp += -2 lam ptt / vp + dt^2 (lam (grad p . grad vp) + div c)
//   grad_z  += -dt^2 (lam vp^2 (grad p . grad(1/z)) + div e / z^2)
// on the physical box only.  A block loads its 32 x 16 tile of p with a 2M halo
// into shared memory once, forms c and e on the cross strips around the tile
// there, and takes every derivative of p from shared memory, through the same
// operator templates the global-memory kernels use.
//
// Fused reconstruction.  The boundary-saving reverse step forms
//   U_{it-1} = 2 U_it - U_{it+1} + dt^2 rhs(U_it)   (+ the forward source)
// on [phys0 + 1, phys1 - 1)^2 (acoustic_vrz2nd_nopml), and the gradient at it
// reads U_{it-1} only at its own cell, in p_tt.  So the kernel forms U_{it-1}
// from the p tile, adds the forward source, writes it and images with it in
// one pass: u_now, vp, z and 1/z are read once instead of twice.  The strip
// restore runs first; its cells (the physical box's outer row and the pad rows
// outside it) are disjoint from those the kernel writes, and the kernel reads
// them back for p_tt there.  The sources of a block's tile are gathered into
// shared memory once per block, so a shot with many sources costs a scan of
// the list per block, not per cell.
#include "kernels.cuh"
#include "../../../core/check.h"

namespace {

constexpr int BX = 32, BZ = 16;      // output tile, one cell per thread
constexpr int kTileSources = 32;     // gathered per block; more falls back to a full scan

template<int Order> struct SmemLine {          // value_at(off) along a strided smem line
    const float* base; int stride;
    __device__ __forceinline__ float operator()(int off) const { return base[off * stride]; }
};

struct Sources {                               // the forward source of the reconstruction
    const float* amp;                          // (B, nsrc, nt)
    const int* loc;                            // (B, nsrc, 2): ix, iz
    int nsrc, it;
};

template<int Order, bool RECON>
__global__ void __launch_bounds__(BX * BZ) vrz2d_grad_tiled(
    const float* __restrict__ p_all,           // U_it
    const float* __restrict__ lam_all,
    const float* __restrict__ ptt_all,         // stored second difference, or nullptr
    const float* __restrict__ pprev_all,       // U_{it-1} (not RECON)
    const float* __restrict__ pnext_all,       // U_{it+1}
    float* __restrict__ unext_all,             // RECON: U_{it-1}, formed on the box, read on the strips
    const float* __restrict__ vp_all, const float* __restrict__ z_all,
    const float* __restrict__ invz_all,
    float* __restrict__ gvp_all, float* __restrict__ gz_all,
    GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext solver, Sources src)
{
    constexpr int M = Order / 2, R = 2 * M;    // p halo: taps of taps
    constexpr int PW = BX + 2 * R, PH = BZ + 2 * R;
    constexpr int CW = BX + 2 * M, CH = BZ + 2 * M;
    __shared__ float sP[PH][PW];
    __shared__ float sCx[BZ][CW], sEx[BZ][CW];
    __shared__ float sCz[CH][BX], sEz[CH][BX];
    __shared__ int sNs, sSrc[RECON ? kTileSources : 1];

    const int nx = solver.nx, nz = solver.nz;
    const int tx = threadIdx.x, tz = threadIdx.y, tid = tz * BX + tx;
    const int b = blockIdx.z;
    const int X0 = solver.phys_x0() + blockIdx.x * BX;
    const int Z0 = solver.phys_z0() + blockIdx.y * BZ;
    const int ix = X0 + tx, iz = Z0 + tz;
    const long off = (long)b * nx * nz;
    const float* __restrict__ p = p_all + off;
    const float* __restrict__ lam = lam_all + off;
    const float* __restrict__ vp = vp_all + off;
    const float* __restrict__ zz = z_all + off;
    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    auto live = [&](int jx, int jz) { return jx >= M && jx < nx - M && jz >= M && jz < nz - M; };

    if constexpr (RECON) {
        if (tid == 0) sNs = 0;
        __syncthreads();
    }
    // p tile with its 2M halo; a clamped cell is never the tap of a live cell
    for (int i = tid; i < PH * PW; i += BX * BZ) {
        const int r = i / PW, c = i % PW;
        sP[r][c] = p[(long)cl(Z0 - R + r, nz) * nx + cl(X0 - R + c, nx)];
    }
    if constexpr (RECON) {
        for (int s = tid; s < src.nsrc; s += BX * BZ) {
            const int base = (b * src.nsrc + s) * 2;
            const int sx = src.loc[base], sz = src.loc[base + 1];
            if (sx >= X0 && sx < X0 + BX && sz >= Z0 && sz < Z0 + BZ) {
                const int k = atomicAdd(&sNs, 1);
                if (k < kTileSources) sSrc[k] = s;
            }
        }
    }
    __syncthreads();
    // c_x, e_x on the tile rows widened by M in x; c_z, e_z on its columns in z
    for (int i = tid; i < BZ * CW; i += BX * BZ) {
        const int r = i / CW, c = i % CW;
        const int jx = X0 + c - M, jz = Z0 + r;
        float cx = 0.f, ex = 0.f;
        if (live(jx, jz)) {
            const float d = GradientGPU<Order, 2, X>::eval(&sP[0][0], 1, 0, PW, c - M + R, 0, r + R,
                                                           grad_ctx.dx, grad_ctx.dy, grad_ctx.dz);
            const long j = (long)jz * nx + jx;
            const float lam_j = lam[j], v_j = vp[j];
            const float lam_v = lam_j * v_j;
            cx = lam_v * d;
            ex = ((lam_v * v_j) * zz[j]) * d;
        }
        sCx[r][c] = cx; sEx[r][c] = ex;
    }
    for (int i = tid; i < CH * BX; i += BX * BZ) {
        const int r = i / BX, c = i % BX;
        const int jx = X0 + c, jz = Z0 + r - M;
        float cz = 0.f, ez = 0.f;
        if (live(jx, jz)) {
            const float d = GradientGPU<Order, 2, Z>::eval(&sP[0][0], 1, 0, PW, c + R, 0, r - M + R,
                                                           grad_ctx.dx, grad_ctx.dy, grad_ctx.dz);
            const long j = (long)jz * nx + jx;
            const float lam_j = lam[j], v_j = vp[j];
            const float lam_v = lam_j * v_j;
            cz = lam_v * d;
            ez = ((lam_v * v_j) * zz[j]) * d;
        }
        sCz[r][c] = cz; sEz[r][c] = ez;
    }
    __syncthreads();
    if (ix >= solver.phys_x1() || iz >= solver.phys_z1()) return;

    const long idx = (long)iz * nx + ix;
    const float* vp_b = vp;
    const float* inv_z_b = invz_all + off;
    const float pc = sP[tz + R][tx + R];
    const float dpdx = GradientGPU<Order, 2, X>::eval(&sP[0][0], 1, 0, PW, tx + R, 0, tz + R,
                                                      grad_ctx.dx, grad_ctx.dy, grad_ctx.dz);
    const float dpdz = GradientGPU<Order, 2, Z>::eval(&sP[0][0], 1, 0, PW, tx + R, 0, tz + R,
                                                      grad_ctx.dx, grad_ctx.dy, grad_ctx.dz);
    const float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    const float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    const float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    const float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);
    const float v = vp_b[idx];
    const float inv_z0 = inv_z_b[idx];

    float ptt;
    if constexpr (RECON) {
        float* __restrict__ un_b = unext_all + off;
        const float pn = pnext_all[off + idx];             // U_{it+1}
        float un;
        if (ix > solver.phys_x0() && ix < solver.phys_x1() - 1 &&
            iz > solver.phys_z0() && iz < solver.phys_z1() - 1) {
            // acoustic_vrz2nd_nopml, its taps of u_now read from the tile
            const float lap_x = LaplaceGPU<Order, 2, X>::eval(&sP[0][0], PW, 1, tx + R, 0, tz + R,
                                                              lap_ctx.dx, lap_ctx.dy, lap_ctx.dz);
            const float lap_z = LaplaceGPU<Order, 2, Z>::eval(&sP[0][0], PW, 1, tx + R, 0, tz + R,
                                                              lap_ctx.dx, lap_ctx.dy, lap_ctx.dz);
            const float beta = v * inv_z0;
            const float kappa = v * zz[idx];
            const float dbdx = dvpdx * inv_z0 + v * z1x;
            const float dbdz = dvpdz * inv_z0 + v * z1z;
            const float rhs = kappa * (beta * (lap_x + lap_z) + dbdx * dpdx + dbdz * dpdz);
            un = 2.0f * pc - pn + solver.dt * solver.dt * rhs;
            // the forward source, as add_source adds it to u_next after the step
            const int ns = sNs;
            if (ns <= kTileSources) {
                for (int k = 0; k < ns; ++k) {
                    const int s = sSrc[k], base = (b * src.nsrc + s) * 2;
                    if (src.loc[base] == ix && src.loc[base + 1] == iz)
                        un += src.amp[((long)b * src.nsrc + s) * solver.nt + src.it];
                }
            } else {
                for (int s = 0; s < src.nsrc; ++s) {
                    const int base = (b * src.nsrc + s) * 2;
                    if (src.loc[base] == ix && src.loc[base + 1] == iz)
                        un += src.amp[((long)b * src.nsrc + s) * solver.nt + src.it];
                }
            }
            un_b[idx] = un;
        } else {
            un = un_b[idx];                                  // the restored strip
        }
        ptt = pn - 2.0f * pc + un;
    } else {
        ptt = (ptt_all != nullptr)
            ? ptt_all[off + idx]
            : (pnext_all[off + idx] - 2.0f * pc + pprev_all[off + idx]);
    }

    const float div_c = centered_gradient_stencil<Order>(SmemLine<Order>{&sCx[tz][tx + M], 1}, solver.M, grad_ctx.coeff, grad_ctx.dx)
                      + centered_gradient_stencil<Order>(SmemLine<Order>{&sCz[tz + M][tx], BX}, solver.M, grad_ctx.coeff, grad_ctx.dz);
    const float div_e = centered_gradient_stencil<Order>(SmemLine<Order>{&sEx[tz][tx + M], 1}, solver.M, grad_ctx.coeff, grad_ctx.dx)
                      + centered_gradient_stencil<Order>(SmemLine<Order>{&sEz[tz + M][tx], BX}, solver.M, grad_ctx.coeff, grad_ctx.dz);
    const float lk = lam[idx];
    const float dp_dvp   = dpdx * dvpdx + dpdz * dvpdz;
    const float dp_dinvz = dpdx * z1x  + dpdz * z1z;
    const float dt2 = solver.dt * solver.dt;
    gvp_all[off + idx] += -2.0f * lk * ptt / v + dt2 * (lk * dp_dvp + div_c);
    const float g_z = lk * v * v * dp_dinvz + div_e * inv_z0 * inv_z0;
    gz_all[off + idx] += -dt2 * g_z;
}

template<int Order, bool RECON>
void launch_tiled(const float* p, const float* lam, const float* ptt, const float* pprev,
                  const float* pnext, float* unext, const float* vp, const float* z,
                  const float* invz, float* gvp, float* gz, const GradParam& g,
                  const LaplaceParam& lap, const SolverContext& ctx, const Sources& src)
{
    const int wx = ctx.phys_x1() - ctx.phys_x0(), wz = ctx.phys_z1() - ctx.phys_z0();
    if (wx <= 0 || wz <= 0) return;
    dim3 block(BX, BZ), grid((wx + BX - 1) / BX, (wz + BZ - 1) / BZ, ctx.B);
    vrz2d_grad_tiled<Order, RECON><<<grid, block>>>(p, lam, ptt, pprev, pnext, unext, vp, z, invz,
                                                   gvp, gz, g, lap, ctx, src);
}

template<bool RECON>
void launch_order(int order, const float* p, const float* lam, const float* ptt,
                  const float* pprev, const float* pnext, float* unext, const float* vp,
                  const float* z, const float* invz, float* gvp, float* gz, const GradParam& g,
                  const LaplaceParam& lap, const SolverContext& ctx, const Sources& src)
{
    switch (order) {
        case 2:   // alone, order 2 keeps the nested-stencil kernel (launch_acoustic_vrz2d_grad)
            if constexpr (RECON) launch_tiled<2, true>(p, lam, ptt, pprev, pnext, unext, vp, z, invz, gvp, gz, g, lap, ctx, src);
            break;
        case 4: launch_tiled<4, RECON>(p, lam, ptt, pprev, pnext, unext, vp, z, invz, gvp, gz, g, lap, ctx, src); break;
        case 6: launch_tiled<6, RECON>(p, lam, ptt, pprev, pnext, unext, vp, z, invz, gvp, gz, g, lap, ctx, src); break;
        case 8: launch_tiled<8, RECON>(p, lam, ptt, pprev, pnext, unext, vp, z, invz, gvp, gz, g, lap, ctx, src); break;
        default: SWEEP_CHECK(false, "acoustic_vrz2d tiled gradient: order ", order, " is not tiled");
    }
}

}  // namespace

bool acoustic_vrz2d_grad_is_tiled(int order) { return order >= 2 && order <= 8 && !(order & 1); }

void launch_acoustic_vrz2d_grad(
    int order, dim3 grid, dim3 block,
    const float* p, const float* lam, const float* ptt, const float* pprev, const float* pnext,
    const float* vp, const float* z, const float* inv_z,
    float* c_x, float* c_z, float* e_x, float* e_z,
    float* grad_vp, float* grad_z, GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext ctx)
{
    // order 2: the nested-stencil kernel is the cheaper one (taps of taps only
    // one cell deep) wherever the reconstruction does not ride along
    if (order == 2 || !acoustic_vrz2d_grad_is_tiled(order)) {
        CALCULATE_GRAD_VRZ2D_AUTO(order, grid, block, p, lam, ptt, pprev, pnext, vp, z, inv_z,
                                  c_x, c_z, e_x, e_z, grad_vp, grad_z, grad_ctx, lap_ctx, ctx);
        return;
    }
    launch_order<false>(order, p, lam, ptt, pprev, pnext, nullptr, vp, z, inv_z, grad_vp, grad_z,
                        grad_ctx, lap_ctx, ctx, Sources{nullptr, nullptr, 0, 0});
}

void launch_acoustic_vrz2d_recon_grad(
    int order, AcousticWavefieldPointer wf, const float* lam,
    const float* vp, const float* z, const float* inv_z,
    const float* source, const int* sources_loc, int nsrc, int it,
    float* grad_vp, float* grad_z, GradParam grad_ctx, LaplaceParam lap_ctx, SolverContext ctx)
{
    SWEEP_CHECK(acoustic_vrz2d_grad_is_tiled(order),
                "acoustic_vrz2d fused reconstruction: order ", order, " is not tiled");
    launch_order<true>(order, wf.u_now, lam, nullptr, nullptr, wf.u_prev, wf.u_next, vp, z, inv_z,
                       grad_vp, grad_z, grad_ctx, lap_ctx, ctx, Sources{source, sources_loc, nsrc, it});
}

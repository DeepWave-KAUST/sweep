// 2.5-D tiled VRZ3D model gradient, orders 2-8: one kernel, in place of the
// nested-stencil kernel (calculate_grad_vrz3d_fused, which recomputes d_d p at
// every divergence tap -- O(M^2) -- over the padded grid) and the
// build_vrz_grad_fields_3d + calculate_grad_vrz3d pair (which round-trips six
// scratch fields and z-taps nine stencilled arrays through L2).  Those still
// run for runtime orders and for a DD cut tile without the deep strips.
//
// Same per-cell result as calculate_grad_vrz3d_fused (operand order kept):
//   c_d(j) = (lam*vp)(j) * D_d p(j),  e_d(j) = (((lam*vp)*vp)*z)(j) * D_d p(j),
//   both 0 when tap j is outside [M, n-M)^3;
//   grad_vp += -2 lam ptt / vp + dt^2 (lam (grad p . grad vp) + div c)
//   grad_z  += -dt^2 (lam vp^2 (grad p . grad(1/z)) + div e / z^2)
// on the physical box only.
//
// Schedule.  A block owns a 32 x TY column tile of the physical box and walks a
// z-chunk.  Step L handles three planes: the lead plane L+M (one p load per
// column, for D_z p), the tile plane L (every array's centre AND cross halo is
// loaded at this same plane, so neighbouring tiles fetch the shared halo rows
// in the same step and reuse them through L2; c_x/e_x, c_y/e_y and the x/y
// halves of the dot products are formed in shared memory here) and the output
// plane k = L-M (its z-direction terms need c_z/e_z up to k+M = L, which is
// why the x/y results wait M steps in a delay line).
//
// Per column registers: p(L-M..L+M), c_z/e_z/vp/(1/z)(L-2M..L), lam and
// D_z p (L-M..L), plus the M-deep delay line of four x/y partials.
//
// Variant, per device: with an L2 of at least 32 MB (A100, H100/H200, Ada) the
// z-taps of p, vp and 1/z are re-read through L1/L2 instead of carried in
// registers (64 registers, 4 blocks/SM); with a smaller L2 (V100) those taps
// miss and DRAM traffic grows 2-4x, so the register queues stay.  Order 8 keeps
// the register queues everywhere (the L1 variant spills).  Both variants form
// the same expressions in the same order: the result does not depend on the GPU.
#include "kernels.cuh"
#include "../../../core/check.h"
#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>

namespace {

template<int Order> struct SmemRow {          // value_at(off) over a strided smem line
    const float* base; int stride;
    __device__ __forceinline__ float operator()(int off) const { return base[off * stride]; }
};
struct GlobalZ {                               // value_at(off) along z through L1/L2
    const float* base; long sz;
    __device__ __forceinline__ float operator()(int off) const { return __ldg(base + off * sz); }
};
template<int N> struct RegQ {                  // value_at(off) over a register queue centred at C
    const float (&q)[N]; int c;
    __device__ __forceinline__ float operator()(int off) const { return q[c + off]; }
};

// ZL1: the column's z-taps of p, vp and 1/z (and lam, D_z p at the output plane)
// are re-read through L1/L2 instead of being carried in register queues -- only
// the computed c_z/e_z stay in registers.  Fewer registers, more resident blocks.
template<int Order, int TY, int MINB, bool ZL1>
__global__ void __launch_bounds__(32 * TY, MINB) vrz3d_grad_tiled(
    const float* __restrict__ p_all, const float* __restrict__ lam_all,
    const float* __restrict__ ptt_all, const float* __restrict__ pprev_all,
    const float* __restrict__ pnext_all,
    const float* __restrict__ vp_all, const float* __restrict__ z_all,
    const float* __restrict__ invz_all,
    float* __restrict__ gvp_all, float* __restrict__ gz_all,
    const float* __restrict__ deepx_all, const float* __restrict__ deepy_all,
    GradParam grad_ctx, SolverContext solver, int zc, int nchunk)
{
    constexpr int M = Order / 2;
    constexpr int R = 2 * M;                   // p halo (taps of taps)
    constexpr int BX = 32;
    constexpr int PW = BX + 2 * R, PH = TY + 2 * R;     // p tile (cross filled)
    constexpr int QW = BX + 2 * M, QH = TY + 2 * M;     // model tiles (cross filled)
    __shared__ float sP[PH][PW];
    __shared__ float sA[QH][QW], sB[QH][QW];
    __shared__ float sV2[2][QH][QW], sI2[2][QH][QW];   // by L parity: read after the
                                                      // strip barrier, so no trailing one
    // delay line of the x/y partials (planes L-M .. L-1): registers for M <= 2,
    // a per-thread shared ring for M >= 3 (thread-private, no barrier)
    constexpr bool RING = (M >= 3);
    __shared__ float sRing[RING ? 4 : 1][RING ? M : 1][TY][BX];
    __shared__ float sCx[TY][QW], sEx[TY][QW];
    __shared__ float sCy[QH][BX], sEy[QH][BX];

    const int nx = solver.nx, ny = solver.ny, nz = solver.nz;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;
    const int b = blockIdx.z / nchunk;
    const int X0 = solver.phys_x0() + blockIdx.x * BX;
    const int Y0 = solver.phys_y0() + blockIdx.y * TY;
    const int zs = solver.phys_z0() + (blockIdx.z % nchunk) * zc;
    const int ze = min(zs + zc, solver.phys_z1());
    const int ix = X0 + tx, iy = Y0 + ty;
    const bool act = ix < solver.phys_x1() && iy < solver.phys_y1();
    const bool cxl = solver.cut_x_lo(), cxh = solver.cut_x_hi();
    const bool cyl = solver.cut_y_lo(), cyh = solver.cut_y_hi();

    const long sy = nx, sz = (long)nx * ny, sp = sz * nz;
    const long off = (long)b * sp;
    const float* __restrict__ p = p_all + off;
    const float* __restrict__ lam = lam_all + off;
    const float* __restrict__ vp = vp_all + off;
    const float* __restrict__ zz = z_all + off;
    const float* __restrict__ iz_ = invz_all + off;
    const float* __restrict__ dpx = deepx_all ? deepx_all + off : nullptr;
    const float* __restrict__ dpy = deepy_all ? deepy_all + off : nullptr;
    auto ldp = [&](int x, int y, long zo) -> float {     // p with the deep halo
        if (x >= 0 && x < nx && y >= 0 && y < ny) return p[(long)y * sy + x + zo];
        if (dpx != nullptr && y >= 0 && y < ny) {
            if (x < 0 && x >= -M && cxl) return dpx[(long)y * sy + (x + M) + zo];
            if (x >= nx && x < nx + M && cxh) return dpx[(long)y * sy + (x - M) + zo];
        }
        if (dpy != nullptr && x >= 0 && x < nx) {
            if (y < 0 && y >= -M && cyl) return dpy[(long)(y + M) * sy + x + zo];
            if (y >= ny && y < ny + M && cyh) return dpy[(long)(y - M) * sy + x + zo];
        }
        return 0.f;                                      // never reaches a live tap
    };

    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    // DD (phased): across a cut the coupling taps in the halo [0,M) / [n-M,n) are
    // live -- they are the neighbour's interior cells -- and their D p reaches M
    // cells beyond the halo, which the neighbour packed into the deep strips
    // (launch_acoustic_vrz3d_grad_pack) and the exchange after phase 1 delivered:
    //   p(x in [-M,0)) = deep_x[x+M],  p(x in [nx,nx+M)) = deep_x[x-M]   (y likewise).
    // Same expressions as the single-domain interior taps, so DD == single domain.
    auto live_x = [&](int j) { return j >= 0 && j < nx && (j >= M || cxl) && (j < nx - M || cxh); };
    auto live_y = [&](int j) { return j >= 0 && j < ny && (j >= M || cyl) && (j < ny - M || cyh); };
    auto live_z = [&](int j) { return j >= M && j < nz - M; };
    const long col = (long)cl(iy, ny) * sy + cl(ix, nx);
    const float dt2 = solver.dt * solver.dt;
    const float* gc = grad_ctx.coeff;
    const float hx = grad_ctx.dx, hy = grad_ctx.dy, hz = grad_ctx.dz;

    constexpr int NQ = ZL1 ? 1 : 2 * M + 1, NS = ZL1 ? 1 : M + 1;
    float qp[NQ];                         // p      (L-M .. L+M)
    float qc[2 * M + 1], qe[2 * M + 1];   // c_z,e_z (L-2M .. L)
    float qv[NQ], qi[NQ];                 // vp,1/z (L-2M .. L)
    float ql[NS], qd[NS];                 // lam, D_z p (L-M .. L)
    float dXc[RING ? 1 : M], dXe[RING ? 1 : M], dPv[RING ? 1 : M], dPi[RING ? 1 : M];

    // prime: p(zs-2M .. zs-1) so the first step's shift brings in p(zs-M+M)
    #pragma unroll
    for (int k = 0; k < NQ; ++k) qp[k] = qv[k] = qi[k] = 0.f;
    if constexpr (!ZL1) {
        #pragma unroll
        for (int k = 1; k < 2 * M + 1; ++k) qp[k] = p[col + (long)cl(zs - 2 * M - 1 + k, nz) * sz];
    }
    #pragma unroll
    for (int k = 0; k < 2 * M + 1; ++k) { qc[k] = qe[k] = 0.f; }
    #pragma unroll
    for (int k = 0; k < NS; ++k) { ql[k] = qd[k] = 0.f; }
    #pragma unroll
    for (int k = 0; k < (RING ? 1 : M); ++k) { dXc[k] = dXe[k] = dPv[k] = dPi[k] = 0.f; }
    if constexpr (RING) {
        #pragma unroll
        for (int k = 0; k < M; ++k) { sRing[0][k][ty][tx] = sRing[1][k][ty][tx] = sRing[2][k][ty][tx] = sRing[3][k][ty][tx] = 0.f; }
    }

    for (int L = zs - M; L < ze + M; ++L) {
        const int Lc = cl(L, nz);
        const long zo = (long)Lc * sz;
        // ---- loads: lead p, centre model at L ------------------------------
        const float lamL = lam[col + zo], vL = vp[col + zo], zL = zz[col + zo], iL = iz_[col + zo];
        float pcen;                                   // p(L)
        if constexpr (!ZL1) {
            const float pl = p[col + (long)cl(L + M, nz) * sz];
            #pragma unroll
            for (int k = 0; k < 2 * M; ++k) qp[k] = qp[k + 1];
            qp[2 * M] = pl;                           // qp = p(L-M .. L+M)
            pcen = qp[M];
        } else {
            pcen = __ldg(p + col + zo);
        }
        const float aL = lamL * vL;
        const float bL = (aL * vL) * zL;

        // ---- z work at L: D_z p(L), c_z/e_z(L) -----------------------------
        float dzp = 0.f, czL = 0.f, ezL = 0.f;
        if (L >= M && L < nz - M) {
            if constexpr (!ZL1) dzp = centered_gradient_stencil<Order>(RegQ<NQ>{qp, M}, M, gc, hz);
            else dzp = centered_gradient_stencil<Order>(GlobalZ{p + col + (long)L * sz, sz}, M, gc, hz);
            if (live_x(ix) && live_y(iy)) { czL = aL * dzp; ezL = bL * dzp; }
        }
        #pragma unroll
        for (int k = 0; k < 2 * M; ++k) { qc[k] = qc[k + 1]; qe[k] = qe[k + 1]; }
        qc[2 * M] = czL; qe[2 * M] = ezL;
        if constexpr (!ZL1) {
            #pragma unroll
            for (int k = 0; k < 2 * M; ++k) { qv[k] = qv[k + 1]; qi[k] = qi[k + 1]; }
            qv[2 * M] = vL; qi[2 * M] = iL;
            #pragma unroll
            for (int k = 0; k < M; ++k) { ql[k] = ql[k + 1]; qd[k] = qd[k + 1]; }
            ql[M] = lamL; qd[M] = dzp;
        }

        const bool tile_plane = (L >= zs && L < ze);   // uniform across the block
        float xyc = 0.f, xye = 0.f, pv = 0.f, pi = 0.f;
        const int par = L & 1;
        auto sV = sV2[par];
        auto sI = sI2[par];
        if (tile_plane) {
            // ---- fill the plane-L tiles (cross halos) ------------------------
            // a partial last tile's out-of-array columns/rows are deep halo cells
            // under DD (the clamped column load above is not their value)
            sP[ty + R][tx + R] = (ix < nx && iy < ny) ? pcen : ldp(ix, iy, zo);
            sA[ty + M][tx + M] = aL; sB[ty + M][tx + M] = bL;
            sV[ty + M][tx + M] = vL; sI[ty + M][tx + M] = iL;
            // p row halo: x in [-R,0) U [BX,BX+R), rows of the tile
            for (int i = tid; i < 2 * R * TY; i += BX * TY) {
                int r = i / (2 * R), c = i % (2 * R);
                int xo = c < R ? c - R : BX + (c - R);
                sP[r + R][xo + R] = ldp(X0 + xo, Y0 + r, zo);
            }
            // p column halo: y in [-R,0) U [TY,TY+R)
            for (int i = tid; i < 2 * R * BX; i += BX * TY) {
                int r = i / BX, c = i % BX;
                int yo = r < R ? r - R : TY + (r - R);
                sP[yo + R][c + R] = ldp(X0 + c, Y0 + yo, zo);
            }
            // model halos (cross M): rows then columns
            for (int i = tid; i < 2 * M * TY + 2 * M * BX; i += BX * TY) {
                int yo, xo;
                if (i < 2 * M * TY) { int r = i / (2 * M), c = i % (2 * M); yo = r; xo = c < M ? c - M : BX + (c - M); }
                else { int j = i - 2 * M * TY; int r = j / BX, c = j % BX; yo = r < M ? r - M : TY + (r - M); xo = c; }
                long g = (long)cl(Y0 + yo, ny) * sy + cl(X0 + xo, nx) + zo;
                float l_ = lam[g], v_ = vp[g], z_ = zz[g];
                float a_ = l_ * v_;
                sA[yo + M][xo + M] = a_; sB[yo + M][xo + M] = (a_ * v_) * z_;
                sV[yo + M][xo + M] = v_; sI[yo + M][xo + M] = iz_[g];
            }
            __syncthreads();
            // p partials of the output-to-be at (ix, iy, L): read sP HERE, between the
            // two barriers -- after the strip barrier the next plane's fill may already
            // be overwriting sP (single-buffered, unlike sV/sI)
            const float dpdx = centered_gradient_stencil<Order>(SmemRow<Order>{&sP[ty + R][tx + R], 1}, M, gc, hx);
            const float dpdy = centered_gradient_stencil<Order>(SmemRow<Order>{&sP[ty + R][tx + R], PW}, M, gc, hy);
            // ---- c_x/e_x on the row strip, c_y/e_y on the column strip -------
            for (int i = tid; i < TY * QW; i += BX * TY) {
                int r = i / QW, c = i % QW;                  // x offset c - M
                int jx = X0 + c - M, jy = Y0 + r;
                float cx = 0.f, ex = 0.f;
                if (live_x(jx) && live_y(jy) && live_z(L)) {
                    float d = centered_gradient_stencil<Order>(SmemRow<Order>{&sP[r + R][c - M + R], 1}, M, gc, hx);
                    cx = sA[r + M][c] * d; ex = sB[r + M][c] * d;
                }
                sCx[r][c] = cx; sEx[r][c] = ex;
            }
            for (int i = tid; i < QH * BX; i += BX * TY) {
                int r = i / BX, c = i % BX;                  // y offset r - M
                int jx = X0 + c, jy = Y0 + r - M;
                float cy = 0.f, ey = 0.f;
                if (live_x(jx) && live_y(jy) && live_z(L)) {
                    float d = centered_gradient_stencil<Order>(SmemRow<Order>{&sP[r - M + R][c + R], PW}, M, gc, hy);
                    cy = sA[r][c + M] * d; ey = sB[r][c + M] * d;
                }
                sCy[r][c] = cy; sEy[r][c] = ey;
            }
            __syncthreads();
            // ---- x/y partials of the output-to-be at (ix, iy, L) -------------
            const float dvx = centered_gradient_stencil<Order>(SmemRow<Order>{&sV[ty + M][tx + M], 1}, M, gc, hx);
            const float dvy = centered_gradient_stencil<Order>(SmemRow<Order>{&sV[ty + M][tx + M], QW}, M, gc, hy);
            const float dix = centered_gradient_stencil<Order>(SmemRow<Order>{&sI[ty + M][tx + M], 1}, M, gc, hx);
            const float diy = centered_gradient_stencil<Order>(SmemRow<Order>{&sI[ty + M][tx + M], QW}, M, gc, hy);
            xyc = centered_gradient_stencil<Order>(SmemRow<Order>{&sCx[ty][tx + M], 1}, M, gc, hx)
                + centered_gradient_stencil<Order>(SmemRow<Order>{&sCy[ty + M][tx], BX}, M, gc, hy);
            xye = centered_gradient_stencil<Order>(SmemRow<Order>{&sEx[ty][tx + M], 1}, M, gc, hx)
                + centered_gradient_stencil<Order>(SmemRow<Order>{&sEy[ty + M][tx], BX}, M, gc, hy);
            pv = dpdx * dvx + dpdy * dvy;
            pi = dpdx * dix + dpdy * diy;
        }

        // ---- output plane k = L - M ----------------------------------------
        const int k = L - M;
        if (k >= zs && k < ze && act) {
            const long gk = off + (long)k * sz + col;
            const long ck = col + (long)k * sz;
            float pk, dpdz, dvz, diz, v, inv_z0, lk;
            if constexpr (!ZL1) {
                pk = qp[0]; dpdz = qd[0]; v = qv[M]; inv_z0 = qi[M]; lk = ql[0];
                dvz = centered_gradient_stencil<Order>(RegQ<NQ>{qv, M}, M, gc, hz);
                diz = centered_gradient_stencil<Order>(RegQ<NQ>{qi, M}, M, gc, hz);
            } else {
                pk = __ldg(p + ck); v = __ldg(vp + ck); inv_z0 = __ldg(iz_ + ck); lk = __ldg(lam + ck);
                dpdz = centered_gradient_stencil<Order>(GlobalZ{p + ck, sz}, M, gc, hz);
                dvz = centered_gradient_stencil<Order>(GlobalZ{vp + ck, sz}, M, gc, hz);
                diz = centered_gradient_stencil<Order>(GlobalZ{iz_ + ck, sz}, M, gc, hz);
            }
            const float ptt = (ptt_all != nullptr) ? ptt_all[gk]
                                                   : (pnext_all[gk] - 2.0f * pk + pprev_all[gk]);
            float Xc, Xe, Pv, Pi;
            if constexpr (RING) {
                const int slot = (L - (zs - M)) % M;          // written at step L-M
                Xc = sRing[0][slot][ty][tx]; Xe = sRing[1][slot][ty][tx];
                Pv = sRing[2][slot][ty][tx]; Pi = sRing[3][slot][ty][tx];
            } else {
                Xc = dXc[0]; Xe = dXe[0]; Pv = dPv[0]; Pi = dPi[0];
            }
            const float div_c = Xc + centered_gradient_stencil<Order>(RegQ<2 * M + 1>{qc, M}, M, gc, hz);
            const float div_e = Xe + centered_gradient_stencil<Order>(RegQ<2 * M + 1>{qe, M}, M, gc, hz);
            const float dp_dvp = Pv + dpdz * dvz;
            const float dp_dinvz = Pi + dpdz * diz;
            gvp_all[gk] += -2.0f * lk * ptt / v + dt2 * (lk * dp_dvp + div_c);
            const float g_z = lk * v * v * dp_dinvz + div_e * inv_z0 * inv_z0;
            gz_all[gk] += -dt2 * g_z;
        }
        // delay line: drop plane k, append plane L (pop above happened first)
        if constexpr (RING) {
            const int slot = (L - (zs - M)) % M;
            sRing[0][slot][ty][tx] = xyc; sRing[1][slot][ty][tx] = xye;
            sRing[2][slot][ty][tx] = pv;  sRing[3][slot][ty][tx] = pi;
        } else {
            #pragma unroll
            for (int j = 0; j < M - 1; ++j) { dXc[j] = dXc[j + 1]; dXe[j] = dXe[j + 1]; dPv[j] = dPv[j + 1]; dPi[j] = dPi[j + 1]; }
            dXc[M - 1] = xyc; dXe[M - 1] = xye; dPv[M - 1] = pv; dPi[M - 1] = pi;
        }
    }
}

template<int Order, int TY, int MINB, bool ZL1>
void launch_tiled(const float* p, const float* lam, const float* ptt, const float* pprev,
                  const float* pnext, const float* vp, const float* z, const float* invz,
                  float* gvp, float* gz, const GradParam& g, const SolverContext& ctx, int zc,
                  const float* deepx, const float* deepy)
{
    const int wx = ctx.phys_x1() - ctx.phys_x0(), wy = ctx.phys_y1() - ctx.phys_y0();
    const int wz = ctx.phys_z1() - ctx.phys_z0();
    if (wx <= 0 || wy <= 0 || wz <= 0) return;
    const int nchunk = (wz + zc - 1) / zc;
    dim3 block(32, TY), grid((wx + 31) / 32, (wy + TY - 1) / TY, nchunk * ctx.B);
    vrz3d_grad_tiled<Order, TY, MINB, ZL1><<<grid, block>>>(p, lam, ptt, pprev, pnext, vp, z, invz,
                                                       gvp, gz, deepx, deepy, g, ctx, zc, nchunk);
}

template<int Order>
void launch_order(bool l1_taps, const float* p, const float* lam, const float* ptt, const float* pprev,
                  const float* pnext, const float* vp, const float* z, const float* invz,
                  float* gvp, float* gz, const GradParam& g, const SolverContext& ctx, int zc,
                  const float* deepx, const float* deepy)
{
    if constexpr (Order <= 6) {
        if (l1_taps) {
            launch_tiled<Order, 8, 4, true>(p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, g, ctx, zc, deepx, deepy);
            return;
        }
    }
    launch_tiled<Order, 8, 2, false>(p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, g, ctx, zc, deepx, deepy);
}

// Is the current device's L2 at least 32 MB?  Cached per device.
bool large_l2()
{
    static std::atomic<int> cached[64];          // 0 = not yet asked, 1 = no, 2 = yes
    int dev = 0;
    SWEEP_CHECK(cudaGetDevice(&dev) == cudaSuccess, "tiled VRZ3D gradient: cudaGetDevice failed");
    int v = (dev >= 0 && dev < 64) ? cached[dev].load(std::memory_order_relaxed) : 0;
    if (v == 0) {
        int l2 = 0;
        SWEEP_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev) == cudaSuccess,
                    "tiled VRZ3D gradient: cannot read the L2 size of device ", dev);
        v = l2 >= (32 << 20) ? 2 : 1;
        if (dev >= 0 && dev < 64) cached[dev].store(v, std::memory_order_relaxed);
    }
    return v == 2;
}

void check_context(const SolverContext& ctx)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "tiled VRZ3D gradient: an x-ranged context is not supported");
    SWEEP_CHECK(!ctx.cut_z_lo() && !ctx.cut_z_hi(),
                "tiled VRZ3D gradient: z is never a DD cut axis");
}

// Phase 1 of the phased DD gradient: write, into the SEND strips of the deep
// buffers (shipped by the exchange after phase 1), the p cells the neighbour
// needs M beyond its halo:
//   cut x-lo: deep_x[M+j]    = p[2M+j]       (sent to the low neighbour's [n-M, n))
//   cut x-hi: deep_x[n-2M+j] = p[n-3M+j]     (sent to the high neighbour's [0, M))
// for j in [0, M), every (y, z); y likewise into deep_y.
template<int Order>
__global__ void vrz3d_grad_pack_deep(const float* __restrict__ p_all, float* __restrict__ dx_all,
                                     float* __restrict__ dy_all, SolverContext s)
{
    constexpr int M = Order / 2;
    const int nx = s.nx, ny = s.ny, nz = s.nz;
    const long sy = nx, sz = (long)nx * ny, sp = sz * nz;
    const int b = blockIdx.z;
    const float* p = p_all + (long)b * sp;
    // x faces: index space (side, j, y, z) flattened over threads
    const long nfx = 2L * M * ny * nz, nfy = 2L * M * nx * nz;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < nfx + nfy; t += (long)gridDim.x * blockDim.x) {
        if (t < nfx) {
            const int side = (int)(t / ((long)M * ny * nz));
            long r = t % ((long)M * ny * nz);
            const int j = (int)(r % M); r /= M;
            const int y = (int)(r % ny); const int zi = (int)(r / ny);
            if (side == 0 ? !s.cut_x_lo() : !s.cut_x_hi()) continue;
            const int src = side == 0 ? 2 * M + j : nx - 3 * M + j;
            const int dst = side == 0 ? M + j : nx - 2 * M + j;
            const long base = (long)zi * sz + (long)y * sy;
            dx_all[(long)b * sp + base + dst] = p[base + src];
        } else {
            long t2 = t - nfx;
            const int side = (int)(t2 / ((long)M * nx * nz));
            long r = t2 % ((long)M * nx * nz);
            const int x = (int)(r % nx); r /= nx;
            const int j = (int)(r % M); const int zi = (int)(r / M);
            if (side == 0 ? !s.cut_y_lo() : !s.cut_y_hi()) continue;
            const int src = side == 0 ? 2 * M + j : ny - 3 * M + j;
            const int dst = side == 0 ? M + j : ny - 2 * M + j;
            const long base = (long)zi * sz + x;
            dy_all[(long)b * sp + base + (long)dst * sy] = p[base + (long)src * sy];
        }
    }
}

}  // namespace

bool acoustic_vrz3d_grad_is_tiled(int order) { return order >= 2 && order <= 8 && !(order & 1); }

bool launch_acoustic_vrz3d_grad_tiled(
    int order, const float* p, const float* lam, const float* ptt, const float* pprev,
    const float* pnext, const float* vp, const float* z, const float* invz,
    float* gvp, float* gz, const GradParam& grad_ctx, const SolverContext& ctx,
    const float* deep_x, const float* deep_y)
{
    if (!acoustic_vrz3d_grad_is_tiled(order)) return false;   // runtime order: the kernels in kernels.cuh
    check_context(ctx);
    // A cut needs the neighbour's deep p, which only the phased path ships.  A
    // non-phased backward on a cut tile (ModelParallel's one-off capture, whose
    // gradient is discarded) keeps the kernels in kernels.cuh.
    if ((ctx.cut_x_lo() || ctx.cut_x_hi()) && deep_x == nullptr) return false;
    if ((ctx.cut_y_lo() || ctx.cut_y_hi()) && deep_y == nullptr) return false;
    // z-chunk: 64 planes, or 32 when the grid would be under ~4 waves
    const long tiles = (long)((ctx.phys_x1() - ctx.phys_x0() + 31) / 32)
                     * ((ctx.phys_y1() - ctx.phys_y0() + 7) / 8) * ctx.B;
    const int wz = ctx.phys_z1() - ctx.phys_z0();
    const int zc = (tiles * ((wz + 63) / 64) >= 2000) ? 64 : 32;
    const bool l1 = large_l2();
    switch (order) {
        case 2: launch_order<2>(l1, p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, grad_ctx, ctx, zc, deep_x, deep_y); break;
        case 4: launch_order<4>(l1, p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, grad_ctx, ctx, zc, deep_x, deep_y); break;
        case 6: launch_order<6>(l1, p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, grad_ctx, ctx, zc, deep_x, deep_y); break;
        case 8: launch_order<8>(l1, p, lam, ptt, pprev, pnext, vp, z, invz, gvp, gz, grad_ctx, ctx, zc, deep_x, deep_y); break;
    }
    return true;
}

bool launch_acoustic_vrz3d_grad_pack(int order, const float* p, float* deep_x, float* deep_y,
                                     const SolverContext& ctx)
{
    if (!acoustic_vrz3d_grad_is_tiled(order)) return false;   // same gate as the tiled launch
    check_context(ctx);
    const int M = order / 2;
    SWEEP_CHECK((!(ctx.cut_x_lo() || ctx.cut_x_hi()) || ctx.nx >= 4 * M) &&
                (!(ctx.cut_y_lo() || ctx.cut_y_hi()) || ctx.ny >= 4 * M),
                "tiled VRZ3D gradient: a cut tile must be at least 4M cells wide on its cut axis");
    const long n = 2L * M * ((long)ctx.ny * ctx.nz + (long)ctx.nx * ctx.nz);
    dim3 block(256), grid((unsigned)std::min<long>((n + 255) / 256, 4096), 1, ctx.B);
    switch (order) {
        case 2: vrz3d_grad_pack_deep<2><<<grid, block>>>(p, deep_x, deep_y, ctx); break;
        case 4: vrz3d_grad_pack_deep<4><<<grid, block>>>(p, deep_x, deep_y, ctx); break;
        case 6: vrz3d_grad_pack_deep<6><<<grid, block>>>(p, deep_x, deep_y, ctx); break;
        case 8: vrz3d_grad_pack_deep<8><<<grid, block>>>(p, deep_x, deep_y, ctx); break;
    }
    return true;
}

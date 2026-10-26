#pragma once
// Pure-qP VTI / TTI with the "new acoustic approximation"
// (Liang, Cao, Sun & Yin 2023, Geophysics 88(1) WA335, doi:10.1190/geo2022-0292.1,
// eqs 19-20 and A-7): the CUDA twin of equations/qP_vti.py and qP_tti.py.
//
//   p_tt = v^2 [ (A + S) Lpx + (C + S) Lpz - E m ]
//   S    = -2 (eps - delta) X^2 Z^2 / ((1 + 2 eps) X^4 + Z^4 + 2 (1 + delta) X^2 Z^2 + 1e-26)
//
// VTI: X = p_x, Z = p_z, A = 1 + 2 eps, C = 1, E = 0.
// TTI: X = p_x cos t - p_z sin t, Z = p_x sin t + p_z cos t,
//      A = (1 + 2 eps) cos^2 t + sin^2 t, C = (1 + 2 eps) sin^2 t + cos^2 t,
//      E = 2 eps sin 2t, m = d/dz (p_x).
// p_x, p_z, Lpx, Lpz are the CPML-corrected first and second derivatives of
// the eager step (the cpmlr update of acoustic.py, written out per axis):
//
//   tmpx = ((1 + bx) Lx p + dbx Dx p) + Dx(ax psix)
//   Lpx  = (1 + bx) tmpx + ax zetax          p_x  = (1 + bx) Dx p + ax psix
//   psix' = bx Dx p + ax psix                zetax' = bx tmpx + ax zetax
//
// The operator is NONLINEAR in p (S is a function of grad p), so the adjoint
// is the transpose of its linearisation about the forward state of the same
// step: the backward needs the full state (p, psix, psiz, zetax, zetaz) of
// step k to take adjoint step k (State below).  Full mode stores it per step,
// boundary saving rebuilds p in the deep interior and restores the whole CPML
// frame (driver_traits.cuh).
//
// All kernels cover the computed cells [M, n - M) of both axes; the M-wide
// outer halo is never written and stays zero, in the forward and in every
// adjoint / scratch field.  Fields are full grids (B, 1, nz, nx), models are
// strided per batch like the wavefields.
#include <cuda_runtime.h>

#include "../../launch/by_order.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

namespace qp2d {

struct Models {
    const float* __restrict__ vp;
    const float* __restrict__ eps;
    const float* __restrict__ del;
    const float* __restrict__ th;     // TTI only (nullptr for VTI)
};

// Forward state of one step: p and the four CPML memories, batch 0 pointers.
struct State {
    const float* __restrict__ u;
    const float* __restrict__ psix;
    const float* __restrict__ psiz;
    const float* __restrict__ zetax;
    const float* __restrict__ zetaz;
};

// Per-step adjoint scratch written by qp_adjoint_prep, read at the stencil
// taps by qp_adjoint_apply.
struct Scratch {
    float* __restrict__ tx;    // adjoint of tmpx
    float* __restrict__ tz;    // adjoint of tmpz
    float* __restrict__ dux;   // adjoint of Dx p
    float* __restrict__ duz;   // adjoint of Dz p
};

struct Grads {
    float* __restrict__ vp;
    float* __restrict__ eps;
    float* __restrict__ del;
    float* __restrict__ th;    // TTI only
};

// The three Thomsen weights of a cell (plus cos/sin of the tilt for TTI).
struct Weights {
    float A, C, E, c, s;
};

template <bool TILT>
__device__ __forceinline__ Weights weights(float eps, float th)
{
    Weights w;
    if constexpr (TILT) {
        float s, c;
        sincosf(th, &s, &c);
        w.c = c;
        w.s = s;
        w.A = (1.f + 2.f * eps) * (c * c) + s * s;
        w.C = (1.f + 2.f * eps) * (s * s) + c * c;
        w.E = 2.f * eps * sinf(2.f * th);
    } else {
        w.c = 1.f;
        w.s = 0.f;
        w.A = 1.f + 2.f * eps;
        w.C = 1.f;
        w.E = 0.f;
    }
    return w;
}

// S and its partials.  S = num / (den + 1e-26) exactly as the eager step
// (the 1e-26 keeps S finite where grad p vanishes; S is 0 there).
struct SVal {
    float S, SX, SZ, Se, Sd;
};

template <bool DERIV>
__device__ __forceinline__ SVal s_op(float X, float Z, float eps, float del)
{
    const float X2 = X * X, Z2 = Z * Z;
    const float num = -2.f * (eps - del) * X2 * Z2;
    const float den = (1.f + 2.f * eps) * (X2 * X2) + Z2 * Z2 + 2.f * (1.f + del) * X2 * Z2;
    const float Dt = den + 1e-26f;
    SVal r;
    r.S = num / Dt;
    if constexpr (DERIV) {
        const float NX = -4.f * (eps - del) * X * Z2;
        const float NZ = -4.f * (eps - del) * X2 * Z;
        const float DX = 4.f * (1.f + 2.f * eps) * X * X2 + 4.f * (1.f + del) * X * Z2;
        const float DZ = 4.f * Z * Z2 + 4.f * (1.f + del) * X2 * Z;
        r.SX = (NX - r.S * DX) / Dt;
        r.SZ = (NZ - r.S * DZ) / Dt;
        r.Se = (-2.f * X2 * Z2 - r.S * 2.f * (X2 * X2)) / Dt;
        r.Sd = 2.f * X2 * Z2 * (1.f - r.S) / Dt;
    } else {
        r.SX = r.SZ = r.Se = r.Sd = 0.f;
    }
    return r;
}

// The forward operator's per-cell quantities from a state, at (ix, iz).
struct Cell {
    float Lx, Lz, Dux, Duz;      // plain derivatives of p
    float tmpx, tmpz;            // CPML tmp (== Lx / Lz off the bands)
    float Lpx, Lpz, px, pz;      // CPML-corrected derivatives
    float m;                     // TTI mixed term d(p_x)/dz (0 for VTI)
    bool cx, cz;                 // inside the x / z CPML compute band
};

// The x / z CPML compute bands, as the acoustic2d forward draws them: one
// halo past the profile band (a fused d(a psi) tap reaches into it, dbx has
// one extra cell of support).  Outside them every CPML term is exactly zero.
__device__ __forceinline__ bool in_cx(const SolverContext& ctx, int ix, int M)
{
    const int x0 = ctx.padLo(2) + M, x1 = ctx.nx - (ctx.padHi(2) + M);
    return ix < x0 + M || ix >= x1 - M;
}

__device__ __forceinline__ bool in_cz(const SolverContext& ctx, int iz, int M)
{
    const int z0 = ctx.padLo(0) + M, z1 = ctx.nz - (ctx.padHi(0) + M);
    return iz < z0 + M || iz >= z1 - M;
}

template <int Order, bool TILT>
__device__ __forceinline__ Cell cell(const float* __restrict__ u,
                                     const float* __restrict__ psix,
                                     const float* __restrict__ psiz,
                                     const float* __restrict__ zetax,
                                     const float* __restrict__ zetaz,
                                     const AcousticCPMLPointer& cpml,
                                     int ix, int iz, const SolverContext& ctx)
{
    constexpr int MS = Order > 0 ? Order / 2 : 0;
    const int M = Order > 0 ? MS : ctx.M;
    const int nx = ctx.nx;
    const int idx = iz * nx + ix;
    const float* __restrict__ lc = ctx.lap_coeff;    // [sum, c1..cM]
    const float* __restrict__ gc = ctx.grad_coeff;   // [_, c1..cM]
    const float dx = ctx.dx, dz = ctx.dz;

    Cell c;
    const float u0 = u[idx];
    float Lx = -lc[0] * u0, Lz = -lc[0] * u0, Dux = 0.f, Duz = 0.f;
#pragma unroll
    for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
        if (Order <= 0 && k > M) break;
        Lx += lc[k] * (u[idx + k] + u[idx - k]);
        Lz += lc[k] * (u[idx + k * nx] + u[idx - k * nx]);
        Dux += gc[k] * (u[idx + k] - u[idx - k]);
        Duz += gc[k] * (u[idx + k * nx] - u[idx - k * nx]);
    }
    c.Lx = Lx / (dx * dx);
    c.Lz = Lz / (dz * dz);
    c.Dux = Dux / dx;
    c.Duz = Duz / dz;

    c.cx = in_cx(ctx, ix, M);
    c.cz = in_cz(ctx, iz, M);

    if (c.cx) {
        const float ax = cpml.ax[ix], bx = cpml.bx[ix], dbx = cpml.dbxdx[ix];
        float G = 0.f;
#pragma unroll
        for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
            if (Order <= 0 && k > M) break;
            G += gc[k] * (cpml.ax[ix + k] * psix[idx + k] - cpml.ax[ix - k] * psix[idx - k]);
        }
        G /= dx;
        c.tmpx = ((1.f + bx) * c.Lx + dbx * c.Dux) + G;
        c.Lpx = (1.f + bx) * c.tmpx + ax * zetax[idx];
        c.px = (1.f + bx) * c.Dux + ax * psix[idx];
    } else {
        c.tmpx = c.Lx;
        c.Lpx = c.Lx;
        c.px = c.Dux;
    }

    if (c.cz) {
        const float az = cpml.az[iz], bz = cpml.bz[iz], dbz = cpml.dbzdz[iz];
        float G = 0.f;
#pragma unroll
        for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
            if (Order <= 0 && k > M) break;
            G += gc[k] * (cpml.az[iz + k] * psiz[idx + k * nx] - cpml.az[iz - k] * psiz[idx - k * nx]);
        }
        G /= dz;
        c.tmpz = ((1.f + bz) * c.Lz + dbz * c.Duz) + G;
        c.Lpz = (1.f + bz) * c.tmpz + az * zetaz[idx];
        c.pz = (1.f + bz) * c.Duz + az * psiz[idx];
    } else {
        c.tmpz = c.Lz;
        c.Lpz = c.Lz;
        c.pz = c.Duz;
    }

    if constexpr (TILT) {
        // m = Dz(p_x): p_x of the rows iz +- k, same column, same x profile.
        const float ax = c.cx ? cpml.ax[ix] : 0.f;
        const float bx1 = c.cx ? 1.f + cpml.bx[ix] : 1.f;
        float m = 0.f;
#pragma unroll
        for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
            if (Order <= 0 && k > M) break;
            const int ip = idx + k * nx, im = idx - k * nx;
            float dp = 0.f, dm = 0.f;
#pragma unroll
            for (int j = 1; j <= (Order > 0 ? MS : 16); ++j) {
                if (Order <= 0 && j > M) break;
                dp += gc[j] * (u[ip + j] - u[ip - j]);
                dm += gc[j] * (u[im + j] - u[im - j]);
            }
            dp /= dx;
            dm /= dx;
            const float pxp = c.cx ? bx1 * dp + ax * psix[ip] : dp;
            const float pxm = c.cx ? bx1 * dm + ax * psix[im] : dm;
            m += gc[k] * (pxp - pxm);
        }
        c.m = m / dz;
    } else {
        c.m = 0.f;
    }
    return c;
}

// The rotated pair (X, Z) S is a function of.
template <bool TILT>
__device__ __forceinline__ void rotated(const Cell& c, const Weights& w, float& X, float& Z)
{
    if constexpr (TILT) {
        X = c.px * w.c - c.pz * w.s;
        Z = c.px * w.s + c.pz * w.c;
    } else {
        X = c.px;
        Z = c.pz;
    }
}

// v^2 dt^2 times the bracket: the update every forward / reverse step adds.
// One function for both so the reconstruction repeats the forward's
// arithmetic exactly.
template <bool TILT>
__device__ __forceinline__ float rhs(const Cell& c, const Weights& w, float S, float W)
{
    if constexpr (TILT)
        return W * (w.A + S) * c.Lpx + W * (w.C + S) * c.Lpz - w.E * W * c.m;
    else
        return W * ((w.A + S) * c.Lpx + (w.C + S) * c.Lpz);
}

// ---------------------------------------------------------------------------
// Forward step: u_next, psixn / psizn (double buffer), zetax / zetaz in place.
// ---------------------------------------------------------------------------
template <int Order, bool TILT>
__global__ void qp_forward(AcousticWavefieldPointer wf, Models mdl,
                           AcousticCPMLPointer cpml, SolverContext ctx)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x + ctx.x_base;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int M = Order > 0 ? Order / 2 : ctx.M;
    if (ix >= ctx.x_end() || iz >= ctx.nz) return;
    if (ix < M || ix >= ctx.nx - M || iz < M || iz >= ctx.nz - M) return;

    const int sp = ctx.nx * ctx.nz;
    const int idx = iz * ctx.nx + ix;
    const long mo = (long)b * sp;
    auto f = wf.offset(b, sp);

    const Cell c = cell<Order, TILT>(f.u_now, f.psix, f.psiz, f.zetax, f.zetaz, cpml, ix, iz, ctx);
    const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
    const Weights w = weights<TILT>(eps, TILT ? mdl.th[mo + idx] : 0.f);
    float X, Z;
    rotated<TILT>(c, w, X, Z);
    const SVal s = s_op<false>(X, Z, eps, del);
    const float v = mdl.vp[mo + idx];
    const float W = (v * v) * (ctx.dt * ctx.dt);

    f.u_next[idx] = 2.f * f.u_now[idx] - f.u_prev[idx] + rhs<TILT>(c, w, s.S, W);

    if (c.cx) {
        const float ax = cpml.ax[ix], bx = cpml.bx[ix];
        f.psixn[idx] = bx * c.Dux + ax * f.psix[idx];
        f.zetax[idx] = bx * c.tmpx + ax * f.zetax[idx];
    }
    if (c.cz) {
        const float az = cpml.az[iz], bz = cpml.bz[iz];
        f.psizn[idx] = bz * c.Duz + az * f.psiz[idx];
        f.zetaz[idx] = bz * c.tmpz + az * f.zetaz[idx];
    }
}

// ---------------------------------------------------------------------------
// Boundary-saving reverse step on the deep interior (outside the saved frame):
//   u_next = 2 u_now - u_prev + v^2 dt^2 [ ... ](u_now)
// with the buffer roles of the reconstruction (u_prev = p^{it+1},
// u_now = p^{it}, u_next <- p^{it-1} without the step-it source).  The CPML
// terms are exactly zero there, so the cell is the forward's off-band cell.
// ---------------------------------------------------------------------------
template <int Order, bool TILT>
__global__ void qp_reverse(AcousticWavefieldPointer wf, Models mdl,
                           AcousticCPMLPointer cpml, SolverContext ctx, int frame)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int M = Order > 0 ? Order / 2 : ctx.M;
    if (ix >= ctx.nx || iz >= ctx.nz) return;
    // frame = the saved band width from the halo inwards
    if (ix < M + frame || ix >= ctx.nx - M - frame ||
        iz < M + frame || iz >= ctx.nz - M - frame) return;

    const int sp = ctx.nx * ctx.nz;
    const int idx = iz * ctx.nx + ix;
    const long mo = (long)b * sp;
    auto f = wf.offset(b, sp);

    const Cell c = cell<Order, TILT>(f.u_now, f.psix, f.psiz, f.zetax, f.zetaz, cpml, ix, iz, ctx);
    const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
    const Weights w = weights<TILT>(eps, TILT ? mdl.th[mo + idx] : 0.f);
    float X, Z;
    rotated<TILT>(c, w, X, Z);
    const SVal s = s_op<false>(X, Z, eps, del);
    const float v = mdl.vp[mo + idx];
    const float W = (v * v) * (ctx.dt * ctx.dt);

    f.u_next[idx] = 2.f * f.u_now[idx] - f.u_prev[idx] + rhs<TILT>(c, w, s.S, W);
}

// ---------------------------------------------------------------------------
// Adjoint step k, pass 1: the per-cell adjoint intermediates of the
// linearised step about the forward state ``st`` of step k, and the model
// gradient of step k (fused: lambda^{k+1} is the adjoint u_now on entry).
//
// adjoint wavefield roles: u_prev = lambda^{k+2}, u_now = lambda^{k+1},
// psix/psiz/zetax/zetaz = adjoints of the step-(k+1) memories;
// psixn/psizn get the partial adjoint of psi^k (finished by pass 2),
// zetaxn/zetazn the adjoint of zeta^k.
//
//   g    = v^2 dt^2 lambda'
//   a    = g (A + S)              c    = g (C + S)            (Lpx, Lpz)
//   tx   = (1 + bx) a + bx zbx'   tz   = (1 + bz) c + bz zbz' (tmpx, tmpz)
//   pbx  = g Qx [+ Dz(g E), TTI]  pbz  = g Qz                 (p_x, p_z)
//   dux  = dbx tx + (1 + bx) pbx + bx psbx'                   (Dx p)
//   duz  = dbz tz + (1 + bz) pbz + bz psbz'                   (Dz p)
//   Qx = (Lpx + Lpz) dS/dp_x,   Qz = (Lpx + Lpz) dS/dp_z.
// st.u == nullptr: no forward state (the first reverse step, lambda' = 0):
// S = Q = 0 and no gradient.  s_mode (qP_vti.s_gradient_aux):
//   0 'frozen'  -- S of the forward state as a coefficient, no Q terms;
//   1 'exact'   -- S of the forward state and the Q terms (its linearisation);
//   2 'adjoint' -- S taken from lambda' itself (its own gradient direction),
//                  no Q terms.
// The model gradient always images with the forward state's S and its
// epsilon / delta / theta derivatives.
// ---------------------------------------------------------------------------
template <int Order, bool TILT>
__global__ void qp_adjoint_prep(AcousticWavefieldPointer adj, State st, Models mdl,
                                AcousticCPMLPointer cpml, SolverContext ctx,
                                Scratch scr, Grads gr, int s_mode)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    constexpr int MS = Order > 0 ? Order / 2 : 0;
    const int M = Order > 0 ? MS : ctx.M;
    if (ix >= ctx.nx || iz >= ctx.nz) return;
    if (ix < M || ix >= ctx.nx - M || iz < M || iz >= ctx.nz - M) return;

    const int nx = ctx.nx;
    const int sp = nx * ctx.nz;
    const int idx = iz * nx + ix;
    const long mo = (long)b * sp;
    auto f = adj.offset(b, sp);
    const float dt2 = ctx.dt * ctx.dt;

    const float lam = f.u_now[idx];
    const float v = mdl.vp[mo + idx];
    const float W = (v * v) * dt2;
    const float g = W * lam;
    const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
    const float th = TILT ? mdl.th[mo + idx] : 0.f;
    const Weights w = weights<TILT>(eps, th);

    float S = 0.f, Qx = 0.f, Qz = 0.f;
    if (st.u != nullptr) {
        const Cell c = cell<Order, TILT>(st.u + mo, st.psix + mo, st.psiz + mo,
                                         st.zetax + mo, st.zetaz + mo, cpml, ix, iz, ctx);
        float X, Z;
        rotated<TILT>(c, w, X, Z);
        const SVal s = s_op<true>(X, Z, eps, del);
        S = s.S;
        const float Lsum = c.Lpx + c.Lpz;
        float Spx, Spz;
        if constexpr (TILT) {
            Spx = s.SX * w.c + s.SZ * w.s;
            Spz = -s.SX * w.s + s.SZ * w.c;
        } else {
            Spx = s.SX;
            Spz = s.SZ;
        }
        if (s_mode == 1) {
            Qx = Lsum * Spx;
            Qz = Lsum * Spz;
        }

        // Model gradient of step k on the physical box (EdgePadding crops the
        // rest): d u^{k+1} / d m times lambda^{k+1}.
        if (gr.vp != nullptr &&
            ix >= ctx.phys_x0() && ix < ctx.phys_x1() &&
            iz >= ctx.phys_z0() && iz < ctx.phys_z1()) {
            const float R = (w.A + S) * c.Lpx + (w.C + S) * c.Lpz - w.E * c.m;
            gr.vp[mo + idx] += lam * 2.f * v * dt2 * R;
            if constexpr (TILT) {
                const float s2 = sinf(2.f * th), c2 = cosf(2.f * th);
                gr.eps[mo + idx] += lam * W * (2.f * (w.c * w.c) * c.Lpx + 2.f * (w.s * w.s) * c.Lpz
                                               + s.Se * Lsum - 2.f * s2 * c.m);
                gr.del[mo + idx] += lam * W * (s.Sd * Lsum);
                const float Sth = -s.SX * Z + s.SZ * X;
                gr.th[mo + idx] += lam * W * (-2.f * eps * s2 * c.Lpx + 2.f * eps * s2 * c.Lpz
                                              + Sth * Lsum - 4.f * eps * c2 * c.m);
            } else {
                gr.eps[mo + idx] += lam * W * (2.f * c.Lpx + s.Se * Lsum);
                gr.del[mo + idx] += lam * W * (s.Sd * Lsum);
            }
        }
    }

    float Sa = S;
    if (s_mode == 2) {
        // S_n of the adjoint field's own (plain) gradient
        const float* __restrict__ gc = ctx.grad_coeff;
        float lx = 0.f, lz = 0.f;
#pragma unroll
        for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
            if (Order <= 0 && k > M) break;
            lx += gc[k] * (f.u_now[idx + k] - f.u_now[idx - k]);
            lz += gc[k] * (f.u_now[idx + k * nx] - f.u_now[idx - k * nx]);
        }
        lx /= ctx.dx;
        lz /= ctx.dz;
        float X, Z;
        if constexpr (TILT) {
            X = lx * w.c - lz * w.s;
            Z = lx * w.s + lz * w.c;
        } else {
            X = lx;
            Z = lz;
        }
        Sa = s_op<false>(X, Z, eps, del).S;
    }
    const float a = g * (w.A + Sa);
    const float cc = g * (w.C + Sa);
    const float ax = cpml.ax[ix], bx = cpml.bx[ix], dbx = cpml.dbxdx[ix];
    const float az = cpml.az[iz], bz = cpml.bz[iz], dbz = cpml.dbzdz[iz];
    const float psx = f.psix[idx], psz = f.psiz[idx];
    const float zex = f.zetax[idx], zez = f.zetaz[idx];

    const float tx = (1.f + bx) * a + bx * zex;
    const float tz = (1.f + bz) * cc + bz * zez;
    float pbx = g * Qx;
    const float pbz = g * Qz;
    if constexpr (TILT) {
        // transpose of  -E W m,  m = Dz(p_x):  p_x adjoint += Dz(g E)
        const float* __restrict__ gc = ctx.grad_coeff;
        float acc = 0.f;
#pragma unroll
        for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
            if (Order <= 0 && k > M) break;
            const long ip = mo + idx + k * nx, im = mo + idx - k * nx;
            const float vp_p = mdl.vp[ip], vp_m = mdl.vp[im];
            const float gEp = (vp_p * vp_p) * dt2 * f.u_now[idx + k * nx]
                            * (2.f * mdl.eps[ip] * sinf(2.f * mdl.th[ip]));
            const float gEm = (vp_m * vp_m) * dt2 * f.u_now[idx - k * nx]
                            * (2.f * mdl.eps[im] * sinf(2.f * mdl.th[im]));
            acc += gc[k] * (gEp - gEm);
        }
        pbx += acc / ctx.dz;
    }
    const float dux = dbx * tx + (1.f + bx) * pbx + bx * psx;
    const float duz = dbz * tz + (1.f + bz) * pbz + bz * psz;

    scr.tx[mo + idx] = tx;
    scr.tz[mo + idx] = tz;
    scr.dux[mo + idx] = dux;
    scr.duz[mo + idx] = duz;
    f.psixn[idx] = pbx + psx;            // finished by qp_adjoint_apply
    f.psizn[idx] = pbz + psz;
    f.zetaxn[idx] = ax * (a + zex);
    f.zetazn[idx] = az * (cc + zez);
}

// ---------------------------------------------------------------------------
// Adjoint step k, pass 2: the transposed stencils.
//   lambda^k = 2 lambda' - lambda'' + Lx^T((1 + bx) tx) + Lz^T((1 + bz) tz)
//                                   + Dx^T dux + Dz^T duz
//   psix adjoint^k = ax (pbx + psbx' + Dx^T tx)  (likewise z)
// with Lx^T = Lx (symmetric taps) and (Dx^T y)_j = sum_k c_k (y_{j-k} - y_{j+k}).
// ---------------------------------------------------------------------------
template <int Order>
__global__ void qp_adjoint_apply(AcousticWavefieldPointer adj, Scratch scr,
                                 AcousticCPMLPointer cpml, SolverContext ctx)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    constexpr int MS = Order > 0 ? Order / 2 : 0;
    const int M = Order > 0 ? MS : ctx.M;
    if (ix >= ctx.nx || iz >= ctx.nz) return;
    if (ix < M || ix >= ctx.nx - M || iz < M || iz >= ctx.nz - M) return;

    const int nx = ctx.nx;
    const int sp = nx * ctx.nz;
    const int idx = iz * nx + ix;
    const long mo = (long)b * sp;
    auto f = adj.offset(b, sp);
    const float* __restrict__ TX = scr.tx + mo;
    const float* __restrict__ TZ = scr.tz + mo;
    const float* __restrict__ DUX = scr.dux + mo;
    const float* __restrict__ DUZ = scr.duz + mo;
    const float* __restrict__ lc = ctx.lap_coeff;
    const float* __restrict__ gc = ctx.grad_coeff;

    float lx = -lc[0] * ((1.f + cpml.bx[ix]) * TX[idx]);
    float lz = -lc[0] * ((1.f + cpml.bz[iz]) * TZ[idx]);
    float gx = 0.f, gz = 0.f, ttx = 0.f, ttz = 0.f;
#pragma unroll
    for (int k = 1; k <= (Order > 0 ? MS : 16); ++k) {
        if (Order <= 0 && k > M) break;
        lx += lc[k] * ((1.f + cpml.bx[ix + k]) * TX[idx + k] + (1.f + cpml.bx[ix - k]) * TX[idx - k]);
        lz += lc[k] * ((1.f + cpml.bz[iz + k]) * TZ[idx + k * nx] + (1.f + cpml.bz[iz - k]) * TZ[idx - k * nx]);
        gx += gc[k] * (DUX[idx - k] - DUX[idx + k]);
        gz += gc[k] * (DUZ[idx - k * nx] - DUZ[idx + k * nx]);
        ttx += gc[k] * (TX[idx - k] - TX[idx + k]);
        ttz += gc[k] * (TZ[idx - k * nx] - TZ[idx + k * nx]);
    }
    const float dx = ctx.dx, dz = ctx.dz;
    f.u_next[idx] = 2.f * f.u_now[idx] - f.u_prev[idx]
                  + (lx / (dx * dx) + lz / (dz * dz) + gx / dx + gz / dz);
    f.psixn[idx] = cpml.ax[ix] * (f.psixn[idx] + ttx / dx);
    f.psizn[idx] = cpml.az[iz] * (f.psizn[idx] + ttz / dz);
}

// ---------------------------------------------------------------------------
// Born (LSRTM) pair, qP_born.py: a background field and a scattered field
// with their own CPML memories and their own S_n, the scattered one driven by
// mp times the background's update:
//   bg' = 2 bg - bg_prev + r0,   sc' = 2 sc - sc_prev + r1 + mp r0,
//   r0 = v^2 dt^2 [..](bg), r1 = v^2 dt^2 [..](sc).
// rhs0_out (full mode): r0 of this step into the history row.
// ---------------------------------------------------------------------------
template <int Order, bool TILT>
__global__ void qp_born_forward(AcousticWavefieldPointer bg, AcousticWavefieldPointer sc, Models mdl,
                                const float* __restrict__ mp, AcousticCPMLPointer cpml,
                                SolverContext ctx, float* __restrict__ rhs0_out)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x + ctx.x_base;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int M = Order > 0 ? Order / 2 : ctx.M;
    if (ix >= ctx.x_end() || iz >= ctx.nz) return;
    if (ix < M || ix >= ctx.nx - M || iz < M || iz >= ctx.nz - M) return;

    const int sp = ctx.nx * ctx.nz;
    const int idx = iz * ctx.nx + ix;
    const long mo = (long)b * sp;
    auto fb = bg.offset(b, sp);
    auto fs = sc.offset(b, sp);

    const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
    const Weights w = weights<TILT>(eps, TILT ? mdl.th[mo + idx] : 0.f);
    const float v = mdl.vp[mo + idx];
    const float W = (v * v) * (ctx.dt * ctx.dt);

    const Cell cb = cell<Order, TILT>(fb.u_now, fb.psix, fb.psiz, fb.zetax, fb.zetaz, cpml, ix, iz, ctx);
    float X, Z;
    rotated<TILT>(cb, w, X, Z);
    const float r0 = rhs<TILT>(cb, w, s_op<false>(X, Z, eps, del).S, W);
    const Cell cs = cell<Order, TILT>(fs.u_now, fs.psix, fs.psiz, fs.zetax, fs.zetaz, cpml, ix, iz, ctx);
    rotated<TILT>(cs, w, X, Z);
    const float r1 = rhs<TILT>(cs, w, s_op<false>(X, Z, eps, del).S, W);

    fb.u_next[idx] = 2.f * fb.u_now[idx] - fb.u_prev[idx] + r0;
    fs.u_next[idx] = 2.f * fs.u_now[idx] - fs.u_prev[idx] + r1 + mp[mo + idx] * r0;
    if (rhs0_out != nullptr) rhs0_out[mo + idx] = r0;

    if (cb.cx) {
        const float ax = cpml.ax[ix], bx = cpml.bx[ix];
        fb.psixn[idx] = bx * cb.Dux + ax * fb.psix[idx];
        fb.zetax[idx] = bx * cb.tmpx + ax * fb.zetax[idx];
        fs.psixn[idx] = bx * cs.Dux + ax * fs.psix[idx];
        fs.zetax[idx] = bx * cs.tmpx + ax * fs.zetax[idx];
    }
    if (cb.cz) {
        const float az = cpml.az[iz], bz = cpml.bz[iz];
        fb.psizn[idx] = bz * cb.Duz + az * fb.psiz[idx];
        fb.zetaz[idx] = bz * cb.tmpz + az * fb.zetaz[idx];
        fs.psizn[idx] = bz * cs.Duz + az * fs.psiz[idx];
        fs.zetaz[idx] = bz * cs.tmpz + az * fs.zetaz[idx];
    }
}

// Boundary-saving reverse step of the pair on the deep interior (see qp_reverse).
template <int Order, bool TILT>
__global__ void qp_born_reverse(AcousticWavefieldPointer bg, AcousticWavefieldPointer sc, Models mdl,
                                const float* __restrict__ mp, AcousticCPMLPointer cpml,
                                SolverContext ctx, int frame)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    const int M = Order > 0 ? Order / 2 : ctx.M;
    if (ix >= ctx.nx || iz >= ctx.nz) return;
    if (ix < M + frame || ix >= ctx.nx - M - frame ||
        iz < M + frame || iz >= ctx.nz - M - frame) return;

    const int sp = ctx.nx * ctx.nz;
    const int idx = iz * ctx.nx + ix;
    const long mo = (long)b * sp;
    auto fb = bg.offset(b, sp);
    auto fs = sc.offset(b, sp);

    const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
    const Weights w = weights<TILT>(eps, TILT ? mdl.th[mo + idx] : 0.f);
    const float v = mdl.vp[mo + idx];
    const float W = (v * v) * (ctx.dt * ctx.dt);

    const Cell cb = cell<Order, TILT>(fb.u_now, fb.psix, fb.psiz, fb.zetax, fb.zetaz, cpml, ix, iz, ctx);
    float X, Z;
    rotated<TILT>(cb, w, X, Z);
    const float r0 = rhs<TILT>(cb, w, s_op<false>(X, Z, eps, del).S, W);
    const Cell cs = cell<Order, TILT>(fs.u_now, fs.psix, fs.psiz, fs.zetax, fs.zetaz, cpml, ix, iz, ctx);
    rotated<TILT>(cs, w, X, Z);
    const float r1 = rhs<TILT>(cs, w, s_op<false>(X, Z, eps, del).S, W);

    fb.u_next[idx] = 2.f * fb.u_now[idx] - fb.u_prev[idx] + r0;
    fs.u_next[idx] = 2.f * fs.u_now[idx] - fs.u_prev[idx] + r1 + mp[mo + idx] * r0;
}

// Reflectivity gradient of step k: grad_mp += lambda_s^{k+1} r0^k on the
// physical box.  r0 from the full-mode history (rhs0_hist) or recomputed from
// the background state of step k (boundary saving).
template <int Order, bool TILT>
__global__ void qp_born_image(AcousticWavefieldPointer adj, State bgst, const float* __restrict__ rhs0_hist,
                              Models mdl, AcousticCPMLPointer cpml, SolverContext ctx,
                              float* __restrict__ grad_mp)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (ix >= ctx.nx || iz >= ctx.nz) return;
    if (ix < ctx.phys_x0() || ix >= ctx.phys_x1() || iz < ctx.phys_z0() || iz >= ctx.phys_z1()) return;

    const int sp = ctx.nx * ctx.nz;
    const int idx = iz * ctx.nx + ix;
    const long mo = (long)b * sp;
    const float lam = adj.u_now[mo + idx];
    float r0;
    if (rhs0_hist != nullptr) {
        r0 = rhs0_hist[mo + idx];
    } else {
        const float eps = mdl.eps[mo + idx], del = mdl.del[mo + idx];
        const Weights w = weights<TILT>(eps, TILT ? mdl.th[mo + idx] : 0.f);
        const float v = mdl.vp[mo + idx];
        const float W = (v * v) * (ctx.dt * ctx.dt);
        const Cell c = cell<Order, TILT>(bgst.u + mo, bgst.psix + mo, bgst.psiz + mo,
                                         bgst.zetax + mo, bgst.zetaz + mo, cpml, ix, iz, ctx);
        float X, Z;
        rotated<TILT>(c, w, X, Z);
        r0 = rhs<TILT>(c, w, s_op<false>(X, Z, eps, del).S, W);
    }
    grad_mp[mo + idx] += lam * r0;
}

// ---------------------------------------------------------------------------
// By-order tables (instantiated once, in kernels.cu).
// ---------------------------------------------------------------------------
using qp_forward_fn = void (*)(AcousticWavefieldPointer, Models, AcousticCPMLPointer, SolverContext);
using qp_reverse_fn = void (*)(AcousticWavefieldPointer, Models, AcousticCPMLPointer, SolverContext, int);
using qp_adjoint_prep_fn = void (*)(AcousticWavefieldPointer, State, Models, AcousticCPMLPointer,
                                    SolverContext, Scratch, Grads, int);
using qp_adjoint_apply_fn = void (*)(AcousticWavefieldPointer, Scratch, AcousticCPMLPointer, SolverContext);
using qp_born_forward_fn = void (*)(AcousticWavefieldPointer, AcousticWavefieldPointer, Models, const float*,
                                    AcousticCPMLPointer, SolverContext, float*);
using qp_born_reverse_fn = void (*)(AcousticWavefieldPointer, AcousticWavefieldPointer, Models, const float*,
                                    AcousticCPMLPointer, SolverContext, int);
using qp_born_image_fn = void (*)(AcousticWavefieldPointer, State, const float*, Models,
                                  AcousticCPMLPointer, SolverContext, float*);

SWEEP_BY_ORDER_DECL(qp_forward_fn, qp_forward_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_forward_fn, qp_forward_tti_by_order);
SWEEP_BY_ORDER_DECL(qp_reverse_fn, qp_reverse_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_reverse_fn, qp_reverse_tti_by_order);
SWEEP_BY_ORDER_DECL(qp_adjoint_prep_fn, qp_adjoint_prep_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_adjoint_prep_fn, qp_adjoint_prep_tti_by_order);
SWEEP_BY_ORDER_DECL(qp_adjoint_apply_fn, qp_adjoint_apply_by_order);
SWEEP_BY_ORDER_DECL(qp_born_forward_fn, qp_born_forward_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_born_forward_fn, qp_born_forward_tti_by_order);
SWEEP_BY_ORDER_DECL(qp_born_reverse_fn, qp_born_reverse_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_born_reverse_fn, qp_born_reverse_tti_by_order);
SWEEP_BY_ORDER_DECL(qp_born_image_fn, qp_born_image_vti_by_order);
SWEEP_BY_ORDER_DECL(qp_born_image_fn, qp_born_image_tti_by_order);

template <bool TILT> struct Tables;
template <> struct Tables<false> {
    static const qp_forward_fn* forward() { return qp_forward_vti_by_order; }
    static const qp_reverse_fn* reverse() { return qp_reverse_vti_by_order; }
    static const qp_adjoint_prep_fn* prep() { return qp_adjoint_prep_vti_by_order; }
    static const qp_born_forward_fn* born_forward() { return qp_born_forward_vti_by_order; }
    static const qp_born_reverse_fn* born_reverse() { return qp_born_reverse_vti_by_order; }
    static const qp_born_image_fn* born_image() { return qp_born_image_vti_by_order; }
};
template <> struct Tables<true> {
    static const qp_forward_fn* forward() { return qp_forward_tti_by_order; }
    static const qp_reverse_fn* reverse() { return qp_reverse_tti_by_order; }
    static const qp_adjoint_prep_fn* prep() { return qp_adjoint_prep_tti_by_order; }
    static const qp_born_forward_fn* born_forward() { return qp_born_forward_tti_by_order; }
    static const qp_born_reverse_fn* born_reverse() { return qp_born_reverse_tti_by_order; }
    static const qp_born_image_fn* born_image() { return qp_born_image_tti_by_order; }
};

} // namespace qp2d

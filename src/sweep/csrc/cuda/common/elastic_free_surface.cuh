#pragma once
#include "context.h"
#include "../operators/staggered.cuh"

template<int Order>
__device__ __forceinline__ int elastic_stencil_half_order(const SolverContext& solver)
{
    if constexpr (Order == -1) {
        return solver.M;
    } else {
        return Order / 2;
    }
}

// Bottom (z-high) free-surface row: the last physical row.  Air is ``iz > bot``.
__device__ __forceinline__ int elastic_z_bottom_surface_row(const SolverContext& solver)
{
    return solver.phys_z1() - 1;
}

__device__ __forceinline__ float elastic_top_fs_value_2d(
    const float* __restrict__ u,
    int ix,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    // z-low (top) free surface.  Surface row is per-column under irregular
    // topography, falls back to the constant ``phys_z0()`` in the flat case
    // (no topo → ``surface_row`` returns ``phys_z0()``).
    if (solver.fsLo(0)) {
        const int top = solver.surface_row(ix);
        if (iz < top) {
            int m = 2 * top - iz;
            float v = u[m * grad_ctx.sz + ix * grad_ctx.sx];
            return odd ? -v : v;
        }
    }
    // z-high (bottom) free surface: reflect about the last physical row.
    if (solver.fsHi(0)) {
        const int bot = elastic_z_bottom_surface_row(solver);
        if (iz > bot) {
            int m = 2 * bot - iz;
            float v = u[m * grad_ctx.sz + ix * grad_ctx.sx];
            return odd ? -v : v;
        }
    }
    return u[iz * grad_ctx.sz + ix * grad_ctx.sx];
}

template<int Order, int Type>
__device__ __forceinline__ float elastic_top_fs_sgradient_z_2d(
    const float* __restrict__ u,
    int ix,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    if (!solver.fsLo(0) && !solver.fsHi(0)) {
        return sgradient<2, Order, Z, Type>(u, ix, 0, iz, grad_ctx);
    }

    const int M = elastic_stencil_half_order<Order>(solver);

    // Near-surface order reduction (Kristek AFDA): a 2nd-order z-stencil for the
    // M physical rows adjacent to each active z free surface (top band
    // [top, top+M) and/or bottom band (bot-M, bot]) suppresses the high-order FS
    // instability; the interior keeps full order.  Matches the eager blend.
    // FLAT surface only (topography path is full-order, gated on ``!has_topo``).
    bool in_o2_band = false;
    if (!solver.has_topo) {
        if (solver.fsLo(0)) {
            const int top = solver.surface_row(ix);
            if (iz - top >= 0 && iz - top < M) in_o2_band = true;
        }
        if (solver.fsHi(0)) {
            const int bot = elastic_z_bottom_surface_row(solver);
            if (bot - iz >= 0 && bot - iz < M) in_o2_band = true;
        }
    }
    if (in_o2_band) {
        float gz2 = 0.f;
        if constexpr (Type & DIFF_FORWARD) {
            gz2 += elastic_top_fs_value_2d(u, ix, iz + 1, grad_ctx, solver, odd)
                 - elastic_top_fs_value_2d(u, ix, iz,     grad_ctx, solver, odd);
        }
        if constexpr (Type & DIFF_BACKWARD) {
            gz2 += elastic_top_fs_value_2d(u, ix, iz,     grad_ctx, solver, odd)
                 - elastic_top_fs_value_2d(u, ix, iz - 1, grad_ctx, solver, odd);
        }
        return gz2 / grad_ctx.dz;
    }

    float gz = 0.f;

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float c = grad_ctx.coeff[m];
        if constexpr (Type & DIFF_FORWARD) {
            const float up = elastic_top_fs_value_2d(u, ix, iz + m + 1, grad_ctx, solver, odd);
            const float um = elastic_top_fs_value_2d(u, ix, iz - m,     grad_ctx, solver, odd);
            gz += c * (up - um);
        }
        if constexpr (Type & DIFF_BACKWARD) {
            const float up = elastic_top_fs_value_2d(u, ix, iz + m,     grad_ctx, solver, odd);
            const float um = elastic_top_fs_value_2d(u, ix, iz - m - 1, grad_ctx, solver, odd);
            gz += c * (up - um);
        }
    }

    return gz / grad_ctx.dz;
}

// Plain (no-mirror) transpose of the near-surface-reduced forward z-derivative
// ``elastic_top_fs_sgradient_z_2d``, evaluated at output row ``m``, in the SAME
// sign convention as ``sgradient<2,Order,Z, opposite(ForwardType)>(q, m)``
// (i.e. the negated true transpose — the calling kernel's sign expects this).
//
// The forward applied a 2nd-order stencil to output rows in the band
// ``[top, top+M)`` (Kristek near-surface order reduction) and the full M-th
// order stencil below.  Operator-split form:  D' = P_band·D2 + P_below·D_full
// with diagonal output-row projectors P, so  D'^T = D2^T·P_band + D_full^T·P_below.
// Concretely: a forward output row ``j`` contributes to input row ``m`` with the
// FULL coefficient if ``j`` is below the band, or the order-2 coefficient (=1) if
// ``j`` is in the band — and never both.  Bounds are checked per read so this
// also serves the air-row (mirror) evaluations where the stencil dips below 0.
template<int Order, int ForwardType>
__device__ __forceinline__ float elastic_fs_plain_reduced_adjoint_z_2d(
    const float* __restrict__ q,
    int ix,
    int m,
    const SGradParam& grad_ctx,
    const SolverContext& solver
)
{
    const int M   = elastic_stencil_half_order<Order>(solver);
    const int top = solver.surface_row(ix);
    const int bot = elastic_z_bottom_surface_row(solver);
    // Order-2 reduction bands (FLAT surface only; topography path is full-order):
    // top band [top, top+M) and/or bottom band (bot-M, bot].
    const bool band_top = solver.fsLo(0) && !solver.has_topo;
    const bool band_bot = solver.fsHi(0) && !solver.has_topo;
    #define ELASTIC_FS_IN_BAND(j) \
        ((band_top && (j) >= top && (j) < top + M) || (band_bot && (j) > bot - M && (j) <= bot))
    const int sz = grad_ctx.sz;
    const int sx = grad_ctx.sx;
    float acc = 0.f;

    if constexpr (ForwardType & DIFF_FORWARD) {
        // full part (matches sgradient<DIFF_BACKWARD>), excluding band output rows
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            const float c = grad_ctx.coeff[i];
            const int jp = m + i;
            const int jm = m - i - 1;
            if (jp >= 0 && jp < solver.nz && !ELASTIC_FS_IN_BAND(jp))
                acc += c * q[jp * sz + ix * sx];
            if (jm >= 0 && jm < solver.nz && !ELASTIC_FS_IN_BAND(jm))
                acc -= c * q[jm * sz + ix * sx];
        }
        // order-2 part: band output rows j use forward g[j]=(u[j+1]-u[j])/dz,
        // contributing +q[m] (j=m) and -q[m-1] (j=m-1) in sgradient<BACKWARD> sign.
        if (ELASTIC_FS_IN_BAND(m))     acc += q[m * sz + ix * sx];
        if (ELASTIC_FS_IN_BAND(m - 1)) acc -= q[(m - 1) * sz + ix * sx];
    } else {
        // full part (matches sgradient<DIFF_FORWARD>), excluding band output rows
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            const float c = grad_ctx.coeff[i];
            const int jp = m + i + 1;
            const int jm = m - i;
            if (jp >= 0 && jp < solver.nz && !ELASTIC_FS_IN_BAND(jp))
                acc += c * q[jp * sz + ix * sx];
            if (jm >= 0 && jm < solver.nz && !ELASTIC_FS_IN_BAND(jm))
                acc -= c * q[jm * sz + ix * sx];
        }
        // order-2 part: band output rows j use forward g[j]=(u[j]-u[j-1])/dz,
        // contributing +q[m+1] (j=m+1) and -q[m] (j=m) in sgradient<FORWARD> sign.
        if (ELASTIC_FS_IN_BAND(m + 1)) acc += q[(m + 1) * sz + ix * sx];
        if (ELASTIC_FS_IN_BAND(m))     acc -= q[m * sz + ix * sx];
    }
    #undef ELASTIC_FS_IN_BAND

    return acc / grad_ctx.dz;
}

template<int Order, int ForwardType>
__device__ __forceinline__ float elastic_top_fs_adjoint_sgradient_z_2d(
    const float* __restrict__ q,
    int ix,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    if (!solver.fsLo(0) && !solver.fsHi(0)) {
        if constexpr (ForwardType & DIFF_FORWARD) {
            return sgradient<2, Order, Z, DIFF_BACKWARD>(q, ix, 0, iz, grad_ctx);
        } else {
            return sgradient<2, Order, Z, DIFF_FORWARD>(q, ix, 0, iz, grad_ctx);
        }
    }

    const int top = solver.surface_row(ix);                 // z-low surface row
    const int bot = elastic_z_bottom_surface_row(solver);   // z-high surface row

    // Air cells are not independent solid dofs (the forward overwrites them by
    // the mirror), so ∂/∂u[air] = 0.
    if (solver.fsLo(0) && iz < top) return 0.f;
    if (solver.fsHi(0) && iz > bot) return 0.f;

    // Adjoint of D' ∘ extend is M^T ∘ D'^T.  M^T folds each air region back onto
    // its mirror-image interior row:
    //   gz[iz] = D'^T[iz] + parity*D'^T[2*top-iz]  (iz>top, from the top air)
    //                     + parity*D'^T[2*bot-iz]  (iz<bot, from the bottom air)
    const float parity = odd ? -1.f : 1.f;
    float gz = elastic_fs_plain_reduced_adjoint_z_2d<Order, ForwardType>(q, ix, iz, grad_ctx, solver);

    if (solver.fsLo(0) && iz > top) {
        const int miz = 2 * top - iz;   // top-air mirror row
        gz += parity * elastic_fs_plain_reduced_adjoint_z_2d<Order, ForwardType>(q, ix, miz, grad_ctx, solver);
    }
    if (solver.fsHi(0) && iz < bot) {
        const int miz = 2 * bot - iz;   // bottom-air mirror row
        gz += parity * elastic_fs_plain_reduced_adjoint_z_2d<Order, ForwardType>(q, ix, miz, grad_ctx, solver);
    }

    return gz;
}

// True on a z-normal free-surface row: the top (z-low) surface row and/or the
// bottom (z-high) surface row, whichever faces are active.  (Name kept for the
// existing call sites; now covers both z faces.)  Legacy top-only reduces to
// ``free_surface && iz == surface_row``.
__device__ __forceinline__ bool elastic_is_top_free_surface_row(
    const SolverContext& solver,
    int ix,
    int iz
)
{
    return (solver.fsLo(0) && iz == solver.surface_row(ix))
        || (solver.fsHi(0) && iz == elastic_z_bottom_surface_row(solver));
}

// Back-compat overload for callers that don't carry per-column ``ix``.
// Falls back to the constant ``phys_z0()`` (the flat-FS top row).  Safe
// when ``has_topo == false``; in topography mode pass the ``ix`` overload.
__device__ __forceinline__ bool elastic_is_top_free_surface_row(
    const SolverContext& solver,
    int iz
)
{
    return (solver.fsLo(0) && iz == solver.phys_z0())
        || (solver.fsHi(0) && iz == elastic_z_bottom_surface_row(solver));
}

template<int Order>
__device__ __forceinline__ int elastic3d_stencil_half_order(const SolverContext& solver)
{
    if constexpr (Order == -1) {
        return solver.M;
    } else {
        return Order / 2;
    }
}

__device__ __forceinline__ int elastic3d_index(
    int ix,
    int iy,
    int iz,
    const SGradParam& grad_ctx
)
{
    return ix * grad_ctx.sx + iy * grad_ctx.sy + iz * grad_ctx.sz;
}

__device__ __forceinline__ float elastic3d_top_fs_value(
    const float* __restrict__ u,
    int ix,
    int iy,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    const int top = solver.phys_z0();
    if (solver.free_surface && iz < top) {
        iz = 2 * top - iz;
        float v = u[elastic3d_index(ix, iy, iz, grad_ctx)];
        return odd ? -v : v;
    }
    return u[elastic3d_index(ix, iy, iz, grad_ctx)];
}

template<int Order, int Type>
__device__ __forceinline__ float elastic3d_top_fs_sgradient_z(
    const float* __restrict__ u,
    int ix,
    int iy,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    if (!solver.free_surface) {
        return sgradient<3, Order, Z, Type>(u, ix, iy, iz, grad_ctx);
    }

    const int M = elastic3d_stencil_half_order<Order>(solver);
    float gz = 0.f;

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float c = grad_ctx.coeff[m];
        if constexpr (Type & DIFF_FORWARD) {
            const float up = elastic3d_top_fs_value(u, ix, iy, iz + m + 1, grad_ctx, solver, odd);
            const float um = elastic3d_top_fs_value(u, ix, iy, iz - m,     grad_ctx, solver, odd);
            gz += c * (up - um);
        }
        if constexpr (Type & DIFF_BACKWARD) {
            const float up = elastic3d_top_fs_value(u, ix, iy, iz + m,     grad_ctx, solver, odd);
            const float um = elastic3d_top_fs_value(u, ix, iy, iz - m - 1, grad_ctx, solver, odd);
            gz += c * (up - um);
        }
    }

    return gz / grad_ctx.dz;
}

template<int Order, int ForwardType>
__device__ __forceinline__ float elastic3d_top_fs_adjoint_sgradient_z(
    const float* __restrict__ q,
    int ix,
    int iy,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    if (!solver.free_surface) {
        if constexpr (ForwardType & DIFF_FORWARD) {
            return sgradient<3, Order, Z, DIFF_BACKWARD>(q, ix, iy, iz, grad_ctx);
        } else {
            return sgradient<3, Order, Z, DIFF_FORWARD>(q, ix, iy, iz, grad_ctx);
        }
    }

    const int M = elastic3d_stencil_half_order<Order>(solver);
    const int top = solver.phys_z0();
    const float parity = odd ? -1.f : 1.f;
    float gz = 0.f;

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float c = grad_ctx.coeff[m];
        if constexpr (ForwardType & DIFF_FORWARD) {
            const int jp = iz + m;
            if (jp >= top && jp < solver.nz)
                gz += c * q[elastic3d_index(ix, iy, jp, grad_ctx)];

            const int jm = iz - m - 1;
            if (jm >= top)
                gz -= c * q[elastic3d_index(ix, iy, jm, grad_ctx)];

            const int jg = 2 * top + m - iz;
            if (iz > top && iz <= top + m && jg >= top)
                gz += c * parity * q[elastic3d_index(ix, iy, jg, grad_ctx)];
        } else {
            const int jp = iz + m + 1;
            if (jp >= top && jp < solver.nz)
                gz += c * q[elastic3d_index(ix, iy, jp, grad_ctx)];

            const int jm = iz - m;
            if (jm >= top)
                gz -= c * q[elastic3d_index(ix, iy, jm, grad_ctx)];

            const int jg = 2 * top + m + 1 - iz;
            if (iz > top && iz <= top + m + 1 && jg >= top)
                gz += c * parity * q[elastic3d_index(ix, iy, jg, grad_ctx)];
        }
    }

    return gz / grad_ctx.dz;
}

__device__ __forceinline__ bool elastic3d_is_top_free_surface_row(
    const SolverContext& solver,
    int iz
)
{
    return solver.free_surface && iz == solver.phys_z0();
}

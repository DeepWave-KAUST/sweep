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

__device__ __forceinline__ float elastic_top_fs_value_2d(
    const float* __restrict__ u,
    int ix,
    int iz,
    const SGradParam& grad_ctx,
    const SolverContext& solver,
    bool odd
)
{
    // Surface row is per-column under irregular topography, falls back to
    // the constant ``phys_z0()`` in the flat case (no topo → ``surface_row``
    // returns ``phys_z0()``).  See ``SolverContext::surface_row``.
    const int top = solver.surface_row(ix);
    if (solver.free_surface && iz < top) {
        iz = 2 * top - iz;
        float v = u[iz * grad_ctx.sz + ix * grad_ctx.sx];
        return odd ? -v : v;
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
    if (!solver.free_surface) {
        return sgradient<2, Order, Z, Type>(u, ix, 0, iz, grad_ctx);
    }

    const int M = elastic_stencil_half_order<Order>(solver);
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
    if (!solver.free_surface) {
        if constexpr (ForwardType & DIFF_FORWARD) {
            return sgradient<2, Order, Z, DIFF_BACKWARD>(q, ix, 0, iz, grad_ctx);
        } else {
            return sgradient<2, Order, Z, DIFF_FORWARD>(q, ix, 0, iz, grad_ctx);
        }
    }

    // Per-column surface row under irregular topography; ``phys_z0()`` in flat.
    const int top = solver.surface_row(ix);

    // Air cell:  ``extend_top_free_surface_topo`` does NOT read u[iz<top]
    // (those rows are overwritten by mirror values from the solid), so
    // ∂(D_plain(extend(u)))[iz_out]/∂u[iz<top] = 0 for ALL output positions
    // iz_out.  Cross-column coupling at air cells goes through the separate
    // plain x-derivative — it does NOT contribute here.
    if (iz < top) return 0.f;

    // The adjoint of D_plain ∘ extend is M^T ∘ D_plain^T.  Per-element form:
    //
    //   M^T(h)[iz]  =  h[iz]                              if iz == top
    //              =  h[iz] + parity * h[2*top - iz]       if iz > top
    //
    // where h = D_plain^T(q).  Under per-column topography q[iz_out<top]
    // is NOT zero (cross-column x-coupling carries energy into air rows in
    // neighbouring solid columns), so we cannot use the flat-FS shortcut of
    // gating reads on ``>= top``.  We compute D_plain^T at iz directly via
    // sgradient (full stencil), then add the parity-weighted mirror copy.
    float gz;
    if constexpr (ForwardType & DIFF_FORWARD) {
        gz = sgradient<2, Order, Z, DIFF_BACKWARD>(q, ix, 0, iz, grad_ctx);
    } else {
        gz = sgradient<2, Order, Z, DIFF_FORWARD>(q, ix, 0, iz, grad_ctx);
    }

    if (iz == top) return gz;

    // Mirror contribution at row mirror_iz = 2*top - iz.  The stencil there
    // may extend below row 0 if mirror_iz is too small — fall back to a
    // safe per-element loop in that case.  Otherwise use sgradient.
    const int mirror_iz = 2 * top - iz;
    const int M = elastic_stencil_half_order<Order>(solver);
    const float parity = odd ? -1.f : 1.f;

    float gz_mirror;
    if (mirror_iz - M >= 0 && mirror_iz + M < solver.nz) {
        if constexpr (ForwardType & DIFF_FORWARD) {
            gz_mirror = sgradient<2, Order, Z, DIFF_BACKWARD>(q, ix, 0, mirror_iz, grad_ctx);
        } else {
            gz_mirror = sgradient<2, Order, Z, DIFF_FORWARD>(q, ix, 0, mirror_iz, grad_ctx);
        }
    } else {
        float acc = 0.f;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float c = grad_ctx.coeff[m];
            if constexpr (ForwardType & DIFF_FORWARD) {
                const int jp = mirror_iz + m;
                const int jm = mirror_iz - m - 1;
                if (jp >= 0 && jp < solver.nz)
                    acc += c * q[jp * grad_ctx.sz + ix * grad_ctx.sx];
                if (jm >= 0 && jm < solver.nz)
                    acc -= c * q[jm * grad_ctx.sz + ix * grad_ctx.sx];
            } else {
                const int jp = mirror_iz + m + 1;
                const int jm = mirror_iz - m;
                if (jp >= 0 && jp < solver.nz)
                    acc += c * q[jp * grad_ctx.sz + ix * grad_ctx.sx];
                if (jm >= 0 && jm < solver.nz)
                    acc -= c * q[jm * grad_ctx.sz + ix * grad_ctx.sx];
            }
        }
        gz_mirror = acc / grad_ctx.dz;
    }

    return gz + parity * gz_mirror;
}

__device__ __forceinline__ bool elastic_is_top_free_surface_row(
    const SolverContext& solver,
    int ix,
    int iz
)
{
    return solver.free_surface && iz == solver.surface_row(ix);
}

// Back-compat overload for callers that don't carry per-column ``ix``.
// Falls back to the constant ``phys_z0()`` (the flat-FS top row).  Safe
// when ``has_topo == false``; in topography mode pass the ``ix`` overload.
__device__ __forceinline__ bool elastic_is_top_free_surface_row(
    const SolverContext& solver,
    int iz
)
{
    return solver.free_surface && iz == solver.phys_z0();
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

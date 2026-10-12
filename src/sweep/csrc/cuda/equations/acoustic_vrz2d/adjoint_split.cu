// VRZ2D adjoint step launched as the pure-interior box plus one launch over
// the CPML ring around it (the 2-D twin of acoustic_vrz3d/adjoint_split.cu,
// with the ring as one launch: launch/box.cuh SlabSet).
//
// EXACT discrete transpose of the forward CPML step (forward_split.cu).  Per
// axis d the forward reads (u, psi_d, zeta_d) and writes
//   t_d   = (1+b_d)·L_d u + db_d·D_d u + da_d·psi_d + a_d·D_d psi_d
//   psin_d = b_d·D_d u + a_d·psi_d,   zetan_d = b_d·t_d + a_d·zeta_d
//   rhs   = κβ·Σ_d[(1+b_d)·t_d + a_d·zeta_d] + Σ_d κ∂_d b·(D_d u + psin_d)
//         = C0·Σ_d[...] + Σ_d C_d·((1+b_d)·D_d u + a_d·psi_d)
// with κβ = vp² = C0 and κ∂_d b = C_d.  Transposing operation by operation
// (Lᵀ = L, D_dᵀ = −D_d, pointwise coefficients evaluated at the TAP) gives,
// with gw = C0·dt²·λ and gtmp_d = b_d·ζ*_d + (1+b_d)·gw:
//   λ_next = 2λ − λ_prev + Σ_d [ L_d((1+b_d)·gtmp_d) − D_d(b_d·ψ*_d + db_d·gtmp_d + (1+b_d)·C_d·dt²·λ) ]
//   ψ*_d'  = a_d·(ψ*_d + C_d·dt²·λ) + da_d·gtmp_d − D_d(a_d·gtmp_d)
//   ζ*_d'  = a_d·(ζ*_d + gw)
// Verified against the numerical transpose of the forward step
// (test/vrz_adjoint_transpose_check.py, rel 1e-16).  Every neighbour read is of
// the OLD λ/ψ*/ζ*: the new ψ*/ζ* land in the double-buffers.
//
//   interior  every tap has b = 0, aux = 0: the transpose collapses to
//             ∇²(vp²λ) − ∂ₓ((∂ₓb·κ)·λ) − ∂_z((∂_z b·κ)·λ); its aux is a dead
//             accumulator (never feeds λ: every coupling carries b), left 0;
//   ring      the other computed cells: the full transpose above.
#include "kernels.cuh"
#include "../../launch/box.cuh"
#include "../../../core/check.h"

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
template<int Order>
__global__ void vrz2d_adj_interior(
    AcousticWavefieldPointer wf,
    const float* __restrict__ C0, const float* __restrict__ Cx, const float* __restrict__ Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* C0_b = C0 + b * spatial_size;
    const float* Cx_b = Cx + b * spatial_size;
    const float* Cz_b = Cz + b * spatial_size;
    const float* lam = f.u_now;
    const float dt2 = solver.dt * solver.dt;

    VrzProductAccessor a0x{C0_b, lam, idx, 1};
    VrzProductAccessor a0z{C0_b, lam, idx, solver.nx};
    float lap_x = centered_laplace1d_stencil<Order>(a0x, lap_ctx.dx, solver.M, lap_ctx.coeff);
    float lap_z = centered_laplace1d_stencil<Order>(a0z, lap_ctx.dz, solver.M, lap_ctx.coeff);

    VrzProductAccessor axx{Cx_b, lam, idx, 1};
    VrzProductAccessor azz{Cz_b, lam, idx, solver.nx};
    float gx = centered_gradient_stencil<Order>(axx, solver.M, grad_ctx.coeff, grad_ctx.dx);
    float gz = centered_gradient_stencil<Order>(azz, solver.M, grad_ctx.coeff, grad_ctx.dz);

    float rhs = (lap_x + lap_z) - gx - gz;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * rhs;
}

// ----------------------------------------------------------------- CPML ring
template<int Order>
__global__ void __launch_bounds__(256) vrz2d_adj_ring(
    AcousticWavefieldPointer wf,
    const float* __restrict__ C0, const float* __restrict__ Cx, const float* __restrict__ Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, fdtd::SlabSet ring,
    float* __restrict__ psix_out, float* __restrict__ psiz_out,
    float* __restrict__ zetax_out, float* __restrict__ zetaz_out)
{
    int ix, iy, iz, b;
    if (!fdtd::slab_cell(ring, solver.B, ix, iy, iz, b)) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int oidx = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    const float* C0_b = C0 + b * spatial_size;
    const float* Cx_b = Cx + b * spatial_size;
    const float* Cz_b = Cz + b * spatial_size;
    const float* lam = f.u_now;
    const int sz = solver.nx;
    const float dt2 = solver.dt * solver.dt;

    const float* lc = lap_ctx.coeff;    // [c0(centre), c1..cM]
    const float* gc = grad_ctx.coeff;   // [_, c1..cM] (antisymmetric)
    const float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    const float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    const float invdx = 1.0f / grad_ctx.dx;
    const float invdz = 1.0f / grad_ctx.dz;

    // The forward runs its CPML branch only where in_pml_2d(cell) holds; every
    // other cell takes the interior formula and never touches aux.  The
    // transpose must mirror that per TAP: a coefficient enters a tap's term
    // only if the forward evaluated it there.  b is exactly 0 outside the band
    // by construction (sigma = 0), but db = np.gradient(b) leaks one cell past
    // the band edge and a = exp(-alpha dt) is not zero in the interior, so
    // both are gated.  A tap on the x axis shares this row's z answer (and
    // vice versa), so the 2-D predicate costs one compare per tap.
    const bool px0 = solver.in_pml_x(ix, halo);
    const bool pz0 = solver.in_pml_z(iz, halo);
    const bool own_pml = px0 || pz0;

    const float ax_ = cpml.ax[ix], az_ = cpml.az[iz];
    const float bx_ = cpml.bx[ix], bz_ = cpml.bz[iz];

    const float gw0 = C0_b[idx] * dt2 * lam[idx];
    const float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw0;
    const float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw0;

    #define VRZ_GW(n)          ( C0_b[n] * dt2 * lam[n] )
    #define VRZ_GTMP_X(n, nix) ( cpml.bx[nix] * f.zetax[n] + (1.0f + cpml.bx[nix]) * VRZ_GW(n) )
    #define VRZ_GTMP_Z(n, niz) ( cpml.bz[niz] * f.zetaz[n] + (1.0f + cpml.bz[niz]) * VRZ_GW(n) )

    // X: L((1+b)·gtmp), D(b·ψ* + db·gtmp + (1+b)·Cx·dt²·λ), D(a·gtmp)
    float Lx_gl = -lc[0] * ((1.0f + bx_) * gtmpx0);
    float Dx_gg = 0.0f, Dx_gq = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
        bool Pp = pz0 || solver.in_pml_x(nixp, halo);
        bool Pm = pz0 || solver.in_pml_x(nixm, halo);
        float tp = VRZ_GTMP_X(np, nixp), tm = VRZ_GTMP_X(nm, nixm);
        float bxp = cpml.bx[nixp], bxm = cpml.bx[nixm];
        float dbp = Pp ? cpml.dbxdx[nixp] : 0.0f, dbm = Pm ? cpml.dbxdx[nixm] : 0.0f;
        float ap  = Pp ? cpml.ax[nixp]    : 0.0f, am  = Pm ? cpml.ax[nixm]    : 0.0f;
        Lx_gl += lc[k] * ((1.0f + bxp) * tp + (1.0f + bxm) * tm);
        Dx_gg += gc[k] * ((bxp * f.psix[np] + dbp * tp + (1.0f + bxp) * Cx_b[np] * dt2 * lam[np])
                        - (bxm * f.psix[nm] + dbm * tm + (1.0f + bxm) * Cx_b[nm] * dt2 * lam[nm]));
        Dx_gq += gc[k] * (ap * tp - am * tm);
    }
    Lx_gl *= invdx2; Dx_gg *= invdx; Dx_gq *= invdx;

    // Z
    float Lz_gl = -lc[0] * ((1.0f + bz_) * gtmpz0);
    float Dz_gg = 0.0f, Dz_gq = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
        bool Pp = px0 || solver.in_pml_z(nizp, halo);
        bool Pm = px0 || solver.in_pml_z(nizm, halo);
        float tp = VRZ_GTMP_Z(np, nizp), tm = VRZ_GTMP_Z(nm, nizm);
        float bzp = cpml.bz[nizp], bzm = cpml.bz[nizm];
        float dbp = Pp ? cpml.dbzdz[nizp] : 0.0f, dbm = Pm ? cpml.dbzdz[nizm] : 0.0f;
        float ap  = Pp ? cpml.az[nizp]    : 0.0f, am  = Pm ? cpml.az[nizm]    : 0.0f;
        Lz_gl += lc[k] * ((1.0f + bzp) * tp + (1.0f + bzm) * tm);
        Dz_gg += gc[k] * ((bzp * f.psiz[np] + dbp * tp + (1.0f + bzp) * Cz_b[np] * dt2 * lam[np])
                        - (bzm * f.psiz[nm] + dbm * tm + (1.0f + bzm) * Cz_b[nm] * dt2 * lam[nm]));
        Dz_gq += gc[k] * (ap * tp - am * tm);
    }
    Lz_gl *= invdz2; Dz_gg *= invdz; Dz_gq *= invdz;

    f.u_next[idx] = 2.0f * lam[idx] - f.u_prev[idx] + (Lx_gl + Lz_gl - Dx_gg - Dz_gg);

    // Adjoint aux is alive only where the forward ran its CPML branch: a
    // cell outside the band never wrote psi/zeta, so nothing there reads
    // them back (every coupling carries b, which is 0 there).  Its
    // double-buffers were zeroed at allocation and no other cell writes them,
    // so skipping the writes keeps them exactly 0 -- same as pure interior.
    if (!own_pml) return;

    // da_d = D_d a_d, exactly as the forward computes it (product-rule form)
    const float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    const float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
    psix_out[oidx]  = ax_ * (f.psix[idx] + Cx_b[idx] * dt2 * lam[idx]) + daxdx * gtmpx0 - Dx_gq;
    psiz_out[oidx]  = az_ * (f.psiz[idx] + Cz_b[idx] * dt2 * lam[idx]) + dazdz * gtmpz0 - Dz_gq;
    zetax_out[oidx] = ax_ * (f.zetax[idx] + gw0);
    zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw0);
    #undef VRZ_GW
    #undef VRZ_GTMP_X
    #undef VRZ_GTMP_Z
}

struct AdjArgs {
    AcousticWavefieldPointer wf; const float *C0, *Cx, *Cz;
    LaplaceParam lap; GradParam g, gx, gz; AcousticCPMLPointer cpml;
    float *psixn, *psizn, *zetaxn, *zetazn;
};

template<int Order>
void launch_adj(const AdjArgs& a, const SolverContext& ctx)
{
    // the cells whose taps stay out of every band
    const Box in = fdtd::phys_box(ctx, ctx.M);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        vrz2d_adj_interior<Order><<<L.grid, L.block>>>(a.wf, a.C0, a.Cx, a.Cz, a.lap, a.g, ctx, in);
    }
    const fdtd::SlabSet ring = fdtd::slab_set(fdtd::computed_box(ctx), in);
    if (ring.n == 0) return;
    const auto L = fdtd::slab_set_launch(ring, ctx.B);
    vrz2d_adj_ring<Order><<<L.grid, L.block>>>(a.wf, a.C0, a.Cx, a.Cz, a.lap, a.g, a.gx, a.gz,
                                               a.cpml, ctx, ring,
                                               a.psixn, a.psizn, a.zetaxn, a.zetazn);
}

}  // namespace

void launch_acoustic_vrz2d_adjoint(
    int order, AcousticWavefieldPointer wf,
    const float* C0, const float* Cx, const float* Cz,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    float* psix_out, float* psiz_out, float* zetax_out, float* zetaz_out)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic_vrz2d adjoint: an x-ranged context is not supported (the "
                "adjoint step always covers the whole tile)");
    const AdjArgs a{wf, C0, Cx, Cz, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml,
                    psix_out, psiz_out, zetax_out, zetaz_out};
    switch (order) {
        case 2:  launch_adj<2>(a, ctx); break;
        case 4:  launch_adj<4>(a, ctx); break;
        case 6:  launch_adj<6>(a, ctx); break;
        case 8:  launch_adj<8>(a, ctx); break;
        default: launch_adj<-1>(a, ctx); break;
    }
}

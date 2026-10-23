// Acoustic2D forward step launched as the interior box plus one launch over
// the CPML ring around it.
//
// One kernel over the padded grid (acoustic2nd) makes every thread carry the
// CPML path's registers -- 53-56 on sm_70, half occupancy -- although only the
// ring, a few percent of a 2-D grid, takes that path.  Here:
//
//   interior  the physical box (no axis in its CPML band): acoustic2nd's fast
//             path, verbatim, at its own register count;
//   ring      the other computed cells as ONE launch (launch/box.cuh SlabSet;
//             four thin strips as four launches cost more than they compute on
//             small grids): acoustic2nd's per-axis CPML path, verbatim.
//
// Both clip to the context's x range [x_base, x_end()), so a ranged (DD
// phase-split) launch computes every cell exactly as the full launch; both skip
// the air cells the air pre-pass cleared.  acoustic2nd itself stays for
// visco_acoustic2d.
#include "kernels.cuh"
#include "../../launch/box.cuh"

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
template<int Order>
__global__ void a2d_fwd_interior(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;
    if (solver.has_topo && iz < solver.topo_rows[ix])
        return;   // air: cleared by acoustic2d_air_clear_kernel

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] +
                    v2_dt2 * (lap_x + lap_z);
    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_z);
}

// ----------------------------------------------------------------- CPML ring
template<int Order>
__global__ void __launch_bounds__(256) a2d_fwd_ring(
    AcousticWavefieldPointer wf, float* __restrict__ u_this, const float* __restrict__ vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, fdtd::SlabSet ring)
{
    int ix, iy, iz, b;
    if (!fdtd::slab_cell(ring, solver.B, ix, iy, iz, b)) return;
    if (solver.has_topo && iz < solver.topo_rows[ix])
        return;   // air: cleared by acoustic2d_air_clear_kernel

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    // acoustic2nd's per-axis compute bands (see there): an axis' CPML terms
    // run within halo of its coefficient band, the bare Laplacian beyond.
    int pml_x0 = solver.cut_x_lo() ? halo : solver.padLo(2) + halo;
    int pml_x1 = solver.nx - (solver.cut_x_hi() ? halo : solver.padHi(2) + halo);
    int pml_z0 = solver.cut_z_lo() ? halo : solver.padLo(0) + halo;
    int pml_z1 = solver.nz - (solver.cut_z_hi() ? halo : solver.padHi(0) + halo);
    bool cx = (ix < pml_x0 + halo) || (ix >= pml_x1 - halo);
    bool cz = (iz < pml_z0 + halo) || (iz >= pml_z1 - halo);

    float w_sum = 0.0f;

    if (cx) {
        float ax_ = cpml.ax[ix];
        float bx_ = cpml.bx[ix];
        float dbxdx_ = cpml.dbxdx[ix];
        float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
        long xi = solver.aux_rd_x2(iz, ix);
        float daipxix_dx = fused_d_aPsi<Order>(cpml.ax, f.psix, ix, (int)xi, grad_ctx.sx, halo, grad_ctx.coeff, grad_ctx.dx);
        float tmpx = ((1.0f+bx_)*lap_x + dbxdx_*dudx) + daipxix_dx;
        w_sum += (1.0f+bx_) * tmpx + ax_ * f.zetax[xi];
        if (solver.aux_x.stored(ix)) {
            (f.psixn ? f.psixn : f.psix)[xi]  = bx_ * dudx + ax_ * f.psix[xi];
            f.zetax[xi] = bx_ * tmpx + ax_ * f.zetax[xi];
        }
    } else {
        w_sum += lap_x;
    }

    if (cz) {
        float az_ = cpml.az[iz];
        float bz_ = cpml.bz[iz];
        float dbzdz_ = cpml.dbzdz[iz];
        float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
        long zi = solver.aux_rd_z2(iz, ix);
        float daipsiz_dz = fused_d_aPsi<Order>(cpml.az, f.psiz, iz, (int)zi, grad_ctx.sz, halo, grad_ctx.coeff, grad_ctx.dz);
        float tmpz = ((1.0f+bz_)*lap_z + dbzdz_*dudz) + daipsiz_dz;
        w_sum += (1.0f+bz_) * tmpz + az_ * f.zetaz[zi];
        if (solver.aux_z.stored(iz)) {
            (f.psizn ? f.psizn : f.psiz)[zi]  = bz_ * dudz + az_ * f.psiz[zi];
            f.zetaz[zi] = bz_ * tmpz + az_ * f.zetaz[zi];
        }
    } else {
        w_sum += lap_z;
    }

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        v2_dt2 * w_sum;

    // The vp image is the update actually applied, v^2 * w_sum: on the halo
    // cells next to a band (now on this branch) w_sum is not the bare
    // Laplacian, and boundary saving's u_tt images exactly this.
    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * w_sum;
}

struct FwdArgs {
    AcousticWavefieldPointer wf; float* u_this; const float* vp;
    LaplaceParam lap; GradParam g, gx, gz; AcousticCPMLPointer cpml;
};

template<int Order>
void launch_fwd(const FwdArgs& a, const SolverContext& ctx)
{
    const Box all = fdtd::computed_box(ctx);
    // the cells no axis' CPML terms reach: the coefficient band plus halo (the
    // d(a psi) taps and np.gradient's dbdx reach halo cells past the band; the
    // adjoint's pure_interior uses the same bound).  Cut faces keep the halo.
    const Box in = fdtd::intersect(fdtd::phys_box(ctx, ctx.M), all);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        a2d_fwd_interior<Order><<<L.grid, L.block>>>(a.wf, a.u_this, a.vp, a.lap, ctx, in);
    }
    const fdtd::SlabSet ring = fdtd::slab_set(all, in);
    if (ring.n == 0) return;
    const auto L = fdtd::slab_set_launch(ring, ctx.B);
    a2d_fwd_ring<Order><<<L.grid, L.block>>>(a.wf, a.u_this, a.vp, a.lap, a.g, a.gx, a.gz,
                                             a.cpml, ctx, ring);
}

}  // namespace

void launch_acoustic2d_forward(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    const FwdArgs a{wf, u_this, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml};
    switch (order) {
        case 2:  launch_fwd<2>(a, ctx); break;
        case 4:  launch_fwd<4>(a, ctx); break;
        case 6:  launch_fwd<6>(a, ctx); break;
        case 8:  launch_fwd<8>(a, ctx); break;
        default: launch_fwd<-1>(a, ctx); break;
    }
}

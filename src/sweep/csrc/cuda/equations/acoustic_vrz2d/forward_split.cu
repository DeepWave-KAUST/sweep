// VRZ2D forward step launched as the interior box plus one launch over the
// CPML ring around it (the 2-D twin of acoustic2d/forward_split.cu).
//
// One kernel over the padded grid made every thread carry the CPML path's
// registers (61-66 on sm_70) although only the ring, a few percent of a 2-D
// grid, takes it.  Here:
//
//   interior  the physical box (in_pml_2d false): the plain update;
//   ring      the other computed cells as ONE launch (launch/box.cuh SlabSet):
//             the CPML update of both axes, full-grid aux.
//
// The cells are those of the former single kernel, each computed by the same
// expressions.
#include "kernels.cuh"
#include "../../launch/box.cuh"

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
template<int Order>
__global__ void vrz2d_fwd_interior(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    const float* __restrict__ z, const float* __restrict__ inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, SolverContext solver, Box box)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float rhs = kappa * (beta * (lap_x + lap_z) + dbdx * dudx + dbdz * dudz);
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// ----------------------------------------------------------------- CPML ring
template<int Order>
__global__ void __launch_bounds__(256) vrz2d_fwd_ring(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    const float* __restrict__ z, const float* __restrict__ inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, fdtd::SlabSet ring)
{
    int ix, iy, iz, b;
    if (!fdtd::slab_cell(ring, solver.B, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float dpsixdx = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + (daxdx * f.psix[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dudx + ax_ * f.psix[idx];
    float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + (dazdz * f.psiz[idx] + az_ * dpsizdz);
    float psizn = bz_ * dudz + az_ * f.psiz[idx];
    float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];

    float px = dudx + psixn;
    float pz = dudz + psizn;
    float w_sum = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx]
                + (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];

    float rhs = kappa * (beta * w_sum + dbdx * px + dbdz * pz);

    (f.psixn ? f.psixn : f.psix)[idx] = psixn;   // race-free forward psi double-buffer
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetaz[idx] = zetazn;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

struct FwdArgs {
    AcousticWavefieldPointer wf; const float *vp, *z, *iz;
    LaplaceParam lap; GradParam g, gx, gz; AcousticCPMLPointer cpml;
};

template<int Order>
void launch_fwd(const FwdArgs& a, const SolverContext& ctx)
{
    const Box all = fdtd::computed_box(ctx);
    // interior = the physical box (in_pml_2d false)
    const Box in = fdtd::intersect(fdtd::phys_box(ctx, 0), all);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        vrz2d_fwd_interior<Order><<<L.grid, L.block>>>(a.wf, a.vp, a.z, a.iz, a.lap, a.g, ctx, in);
    }
    const fdtd::SlabSet ring = fdtd::slab_set(all, in);
    if (ring.n == 0) return;
    const auto L = fdtd::slab_set_launch(ring, ctx.B);
    vrz2d_fwd_ring<Order><<<L.grid, L.block>>>(a.wf, a.vp, a.z, a.iz, a.lap, a.g, a.gx, a.gz,
                                               a.cpml, ctx, ring);
}

}  // namespace

void launch_acoustic_vrz2d_forward(
    int order, AcousticWavefieldPointer wf,
    const float* vp, const float* z, const float* inv_z,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    const FwdArgs a{wf, vp, z, inv_z, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml};
    switch (order) {
        case 2:  launch_fwd<2>(a, ctx); break;
        case 4:  launch_fwd<4>(a, ctx); break;
        case 6:  launch_fwd<6>(a, ctx); break;
        case 8:  launch_fwd<8>(a, ctx); break;
        default: launch_fwd<-1>(a, ctx); break;
    }
}

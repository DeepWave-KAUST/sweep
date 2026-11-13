// Acoustic2D adjoint step launched as the pure-interior box plus one launch
// over the CPML ring around it.
//
// One kernel over the padded grid (acoustic2nd_adjoint_fused) carries the CPML
// transpose's registers in every thread -- 56-72 on sm_70, 37-50% occupancy --
// although only the ring, a few percent of a 2-D grid, needs them.  Here:
//
//   interior  the cells whose taps stay out of every band: the fused kernel's
//             fast path, L_prev = 2 L - L_next + Lap(vp^2 dt^2 L), verbatim;
//   ring      the other computed cells as ONE launch (launch/box.cuh SlabSet):
//             the fused kernel's CPML transpose, verbatim.
//
// Both fuse the lagged vp-gradient imaging on the physical box (full mode) and
// skip air cells.  acoustic2nd_adjoint_fused itself stays for visco_acoustic2d.
#include "kernels.cuh"
#include "../../launch/launch.h"   // SWEEP_LAUNCH, SWEEP_LAUNCH_ROWS
#include "../../launch/box.cuh"
#include "../../common/host_rows.h"   // box_launch_rows, box_cell_at (host core only)
#include "../../../core/check.h"

namespace {

using fdtd::Box;

// ----------------------------------------------------------------- interior
// One cell of a2d_adj_interior: everything after the kernel's box decode
// (launch/cell_path.h).  The kernel has no PML branch; its run-time tests are
// the air test and the fused imaging's physical-box test.  On an interior
// path the cell is in neither the air (the host span holds no row with an air
// cell) nor outside the physical box (the span is clipped to it), so the
// imaging is whether grad_out is there: optional output grad_out.
template<int Order, CellPath Path = CellPath::Any>
__device__ __forceinline__ void a2d_adj_interior_cell(
    Acoustic2dCellIndex cell,
    const AcousticWavefieldPointer& wf, const float* __restrict__ vp,
    const LaplaceParam& lap_ctx, const SolverContext& solver,
    const float* __restrict__ grad_forward_img, float* __restrict__ grad_out)
{
    const int ix = cell.ix, iz = cell.iz, b = cell.b;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if constexpr (!interior_path<Path>) {
        if (solver.has_topo && iz < solver.topo_rows[ix]) return;
    }

    int sp = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    int sz = solver.nx;                       // z-stride
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;          // [c0(center), c1, ..., cM]
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);

    float gun = f.u_now[idx];

    // fused vp-gradient imaging, lagged one step (see acoustic2nd_adjoint_fused)
    // -- the kernel's own test on the Any path (no optional_output call there:
    // see a2d_fwd_interior_cell), the compile-time answer on an interior one.
    bool image;
    if constexpr (interior_path<Path>)
        image = optional_output<Path>(grad_out);
    else
        image = grad_out != nullptr &&
            ix >= solver.phys_x0() && ix < solver.phys_x1() &&
            iz >= solver.phys_z0() && iz < solver.phys_z1();
    if (image)
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];

    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw  = c0c * gun;                     // c * lambda at idx

    #define GW_AT(n) ( vpb[n]*vpb[n]*dt2 * f.u_now[n] )
    // aux all 0 here and left 0 (both double-buffers are zero-allocated)
    float lapx = -lc[0] * gw, lapz = -lc[0] * gw;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        lapx += lc[k] * (GW_AT(idx + k)      + GW_AT(idx - k));
        lapz += lc[k] * (GW_AT(idx + k * sz) + GW_AT(idx - k * sz));
    }
    f.u_next[idx] = 2.0f * gun - f.u_prev[idx] + lapx * invdx2 + lapz * invdz2;
    #undef GW_AT
}

template<int Order>
__global__ void a2d_adj_interior(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    LaplaceParam lap_ctx, SolverContext solver, Box box,
    const float* __restrict__ grad_forward_img, float* __restrict__ grad_out)
{
    int ix, iy, iz, b;
    if (!fdtd::box_cell(box, solver.B, ix, iy, iz, b)) return;

    a2d_adj_interior_cell<Order>(Acoustic2dCellIndex{ix, iz, b}, wf, vp, lap_ctx, solver,
                                 grad_forward_img, grad_out);
}

// ----------------------------------------------------------------- CPML ring
template<int Order>
__global__ void __launch_bounds__(256) a2d_adj_ring(
    AcousticWavefieldPointer wf, const float* __restrict__ vp,
    LaplaceParam lap_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext solver, fdtd::SlabSet ring,
    float* __restrict__ psix_out, float* __restrict__ psiz_out,
    float* __restrict__ zetax_out, float* __restrict__ zetaz_out,
    const float* __restrict__ grad_forward_img, float* __restrict__ grad_out)
{
    int ix, iy, iz, b;
    if (!fdtd::slab_cell(ring, solver.B, ix, iy, iz, b)) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (solver.has_topo && iz < solver.topo_rows[ix]) return;

    int sp = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int oidx = b * sp + idx;
    auto f = wf.offset(b, sp);
    const float* vpb = vp + b * sp;
    int sz = solver.nx;                       // z-stride
    float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;          // [c0(center), c1, ..., cM]
    const float* gc = grad_ctx_x.coeff;       // [_, c1, ..., cM]  (antisymmetric)
    float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    float invdx  = 1.0f / lap_ctx.dx;
    float invdz  = 1.0f / lap_ctx.dz;

    float gun = f.u_now[idx];

    if (grad_out != nullptr &&
        ix >= solver.phys_x0() && ix < solver.phys_x1() &&
        iz >= solver.phys_z0() && iz < solver.phys_z1())
        grad_out[oidx] += 2.f * solver.dt * solver.dt
                        * grad_forward_img[oidx] * gun / vpb[idx];

    float c0c = vpb[idx]; c0c = c0c * c0c * dt2;
    float gw  = c0c * gun;                     // c * lambda at idx

    float ax_ = cpml.ax[ix], az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix], bz_ = cpml.bz[iz];

    float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw;
    float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw;

    #define GTMP_X(n, nix) ( cpml.bx[nix]*f.zetax[n] + (1.0f+cpml.bx[nix])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )
    #define GTMP_Z(n, niz) ( cpml.bz[niz]*f.zetaz[n] + (1.0f+cpml.bz[niz])*(vpb[n]*vpb[n]*dt2)*f.u_now[n] )

    float Lx_glx = -lc[0] * ((1.0f + bx_) * gtmpx0);
    float Dx_ggx = 0.0f, Dx_gqx = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
        float tp = GTMP_X(np, nixp), tm = GTMP_X(nm, nixm);
        float bxp = cpml.bx[nixp], bxm = cpml.bx[nixm];
        Lx_glx += lc[k] * ((1.0f + bxp) * tp + (1.0f + bxm) * tm);
        Dx_ggx += gc[k] * ((bxp * f.psix[np] + cpml.dbxdx[nixp] * tp)
                         - (bxm * f.psix[nm] + cpml.dbxdx[nixm] * tm));
        Dx_gqx += gc[k] * (tp - tm);   // d(gtmp)/dx (ax applied at psix_out)
    }
    Lx_glx *= invdx2; Dx_ggx *= invdx; Dx_gqx *= invdx;

    float Lz_glz = -lc[0] * ((1.0f + bz_) * gtmpz0);
    float Dz_ggz = 0.0f, Dz_gqz = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
        float tp = GTMP_Z(np, nizp), tm = GTMP_Z(nm, nizm);
        float bzp = cpml.bz[nizp], bzm = cpml.bz[nizm];
        Lz_glz += lc[k] * ((1.0f + bzp) * tp + (1.0f + bzm) * tm);
        Dz_ggz += gc[k] * ((bzp * f.psiz[np] + cpml.dbzdz[nizp] * tp)
                         - (bzm * f.psiz[nm] + cpml.dbzdz[nizm] * tm));
        Dz_gqz += gc[k] * (tp - tm);   // d(gtmp)/dz (az applied at psiz_out)
    }
    Lz_glz *= invdz2; Dz_ggz *= invdz; Dz_gqz *= invdz;

    f.u_next[idx] = 2.0f * gun - f.u_prev[idx]
                  + (Lx_glx + Lz_glz - Dx_ggx - Dz_ggz);
    psix_out[oidx]  = ax_ * (f.psix[idx] - Dx_gqx);
    psiz_out[oidx]  = az_ * (f.psiz[idx] - Dz_gqz);
    zetax_out[oidx] = ax_ * (f.zetax[idx] + gw);
    zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw);
    #undef GTMP_X
    #undef GTMP_Z
}

#if defined(SWEEP_HOST_CORE)
// box_launch_rows (common/host_rows.h) with every row clipped to ``keep``, a
// box in (ix, iz): a row outside keep's z range is empty, the others keep only
// keep's x range.  The span of a box launch whose interior path also answers a
// box test of the cell's own (here a2d_adj_interior's imaging test); a local
// helper, generic in all but its name.
inline auto a2d_box_launch_rows_within(const Box& r, const SolverContext& solver, CellBox keep)
{
    const int nbz = r.zhi - r.zlo, zlo = r.zlo, xlo = r.xlo;
    return [rows = sweep::host::box_launch_rows(r, solver), keep, nbz, zlo, xlo](int gy, int gz) {
        sweep::host::RowSpan s = rows(gy, gz);
        if (s.hi <= s.lo)
            return s;   // past the box, the shots or the surface (and every row of an empty box)
        const int iz = zlo + gz % nbz;
        if (iz < keep.z0 || iz >= keep.z1)
            return sweep::host::RowSpan{};
        s.lo = std::max(s.lo, keep.x0 - xlo);
        s.hi = std::min(s.hi, keep.x1 - xlo);
        return s;
    };
}

// Host row launcher of a2d_adj_interior, for SWEEP_LAUNCH_ROWS in launch_adj
// (launch/cell_path.h).  The kernel has no PML branch; the span is built from
// its three tests: the box decode and the air test -- the launch box minus the
// rows past the last shot and the rows above the deepest surface row over the
// box's columns (box_launch_rows) -- and the imaging test, whose physical box
// [phys_x0, phys_x1) x [phys_z0, phys_z1) every row is clipped to.  (The
// launch box, phys_box(ctx, M + 1), lies inside the physical box on every
// face -- a band face starts M + 1 further in, a cut face at the same M -- so
// the clip removes nothing; it keeps the span a subset should the box ever
// change.)  box_cell_at decodes a virtual thread exactly as box_cell does.
// Optional output: grad_out.  Order -1 (runtime M) takes the same span (the
// box does not depend on the order); its row loop does not vectorise.  No
// arithmetic of its own; the cell lambda captures BY VALUE.
template<int Order>
struct A2dAdjInteriorRows {
    static void run(dim3 grid, dim3 block, AcousticWavefieldPointer wf, const float* vp,
                    LaplaceParam lap_ctx, SolverContext solver, Box box,
                    const float* grad_forward_img, float* grad_out)
    {
        auto call = [&](auto&&... a) { a2d_adj_interior<Order>(a...); };
        const CellBox phys{solver.phys_x0(), solver.phys_x1(), solver.phys_z0(), solver.phys_z1()};
        sweep::host::with_optional(grad_out, [&](auto path) {
            sweep::host::launch_rows(
                grid, block, a2d_box_launch_rows_within(box, solver, phys),
                [=](int gx, int gy, int gz) {
                    int ix, iy, iz, b;
                    sweep::host::box_cell_at(box, gx, gy, gz, ix, iy, iz, b);
                    a2d_adj_interior_cell<Order, decltype(path)::value>(
                        Acoustic2dCellIndex{ix, iz, b}, wf, vp, lap_ctx, solver,
                        grad_forward_img, grad_out);
                },
                call,
                wf, vp, lap_ctx, solver, box, grad_forward_img, grad_out);
        });
    }
};
#endif

struct AdjArgs {
    AcousticWavefieldPointer wf; const float* vp;
    LaplaceParam lap; GradParam gx, gz; AcousticCPMLPointer cpml;
    float *psixn, *psizn, *zetaxn, *zetazn;
    const float* img; float* grad;
};

template<int Order>
void launch_adj(const AdjArgs& a, const SolverContext& ctx)
{
    // the cells whose taps stay out of every band and off the one cell past
    // it where dbdx (np.gradient of b) is still nonzero: the forward applies
    // that dbdx there, so its transpose reaches cells up to M past it.
    const Box in = fdtd::phys_box(ctx, ctx.M + 1);
    if (!fdtd::empty(in)) {
        const auto L = fdtd::box_launch(in, ctx.B, fdtd::kBoxBlock);
        SWEEP_LAUNCH_ROWS((a2d_adj_interior<Order>), A2dAdjInteriorRows<Order>::run, L.grid, L.block,
                          a.wf, a.vp, a.lap, ctx, in, a.img, a.grad);
    }
    const fdtd::SlabSet ring = fdtd::slab_set(fdtd::computed_box(ctx), in);
    if (ring.n == 0) return;
    const auto L = fdtd::slab_set_launch(ring, ctx.B);
    SWEEP_LAUNCH((a2d_adj_ring<Order>), L.grid, L.block, a.wf, a.vp, a.lap, a.gx, a.gz, a.cpml, ctx, ring,
                                             a.psixn, a.psizn, a.zetaxn, a.zetazn, a.img, a.grad);
}

}  // namespace

void launch_acoustic2d_adjoint(
    int order, AcousticWavefieldPointer wf, const float* vp, LaplaceParam lap_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_z, AcousticCPMLPointer cpml, SolverContext ctx,
    float* psix_out, float* psiz_out, float* zetax_out, float* zetaz_out,
    const float* grad_forward_img, float* grad_out)
{
    SWEEP_CHECK(ctx.x_base == 0 && ctx.x_limit < 0,
                "acoustic2d adjoint: an x-ranged context is not supported (the adjoint "
                "step always covers the whole tile)");
    const AdjArgs a{wf, vp, lap_ctx, grad_ctx_x, grad_ctx_z, cpml,
                    psix_out, psiz_out, zetax_out, zetaz_out, grad_forward_img, grad_out};
    switch (order) {
        case 2:  launch_adj<2>(a, ctx); break;
        case 4:  launch_adj<4>(a, ctx); break;
        case 6:  launch_adj<6>(a, ctx); break;
        case 8:  launch_adj<8>(a, ctx); break;
        default: launch_adj<-1>(a, ctx); break;
    }
}

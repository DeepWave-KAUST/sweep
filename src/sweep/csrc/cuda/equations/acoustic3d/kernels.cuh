#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

// Pre-pass: clear air cells (above per-(iy,ix) surface row).  Launched
// BEFORE the forward step so the step only reads (never writes) air cells in
// the same launch — eliminates intra-launch RAW races on PML aux fields (same
// fix as acoustic2d).  (Defined in kernels.cu: a definition here is compiled
// into every TU that includes this header.)
__global__ void acoustic3d_air_clear_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    SolverContext solver
);

// The 3-D acoustic steps (adjoint_split.cu, forward_split.cu): an interior
// launch plus a per-axis CPML launch over the slabs around it, every order
// (orders above 8 through the runtime-order instantiation).
void launch_acoustic3d_adjoint(
    int order, const AcousticWavefieldPointer& adj_view, const float* vp, const LaplaceParam& lap_ctx,
    const GradParam& gx, const GradParam& gy, const GradParam& gz,
    const AcousticCPMLPointer& cpml, const SolverContext& ctx,
    float* psixn, float* psiyn, float* psizn, float* zetaxn, float* zetayn, float* zetazn,
    const float* grad_forward_img, float* grad_out);
void launch_acoustic3d_forward(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx);
// boundary-saving reconstruction, with the fused vp-gradient imaging when grad != nullptr
void launch_acoustic3d_nopml(
    int order, AcousticWavefieldPointer wf, float* u_this, const float* vp,
    LaplaceParam lap_ctx, SolverContext ctx,
    const float* adj, float* grad, int save_width);

__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt,
    int x0, int x1, int y0, int y1, int z0, int z1
);

__global__ void calculate_grad_utt_3d(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt,
    int x0, int x1, int y0, int y1, int z0, int z1
);

// Strip-only twin of calculate_grad_utt_3d (see the 2-D band kernel): the six
// restore slabs of the physical box, each cell once.  grid = (ceil(n/block), B).
__global__ void calculate_grad_utt_3d_band(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad,
    int nx, int ny, int nz, float dt,
    int x0, int x1, int y0, int y1, int z0, int z1,
    int wxl, int wxh, int wyl, int wyh, int wzl, int wzh
);

// 3-D twin; see accumulate_illumination_2d in acoustic2d/kernels.cuh for why
// the forward operand comes in two shapes.
__global__ void accumulate_illumination_3d(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,     // null => u_forward_next IS u_tt
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B, int nx, int ny, int nz, float dt,
    int x0, int x1, int y0, int y1, int z0, int z1
);

__global__ void accumulate_adcig_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ adcig,           // (nlag, B, nz, ny, nx) runtime-padded
    int nlag, int max_lag,
    int B, int nx, int ny, int nz
);

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
);

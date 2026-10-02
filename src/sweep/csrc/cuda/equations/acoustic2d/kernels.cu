#include "kernels.cuh"

__global__ void calculate_grad(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx); stored as vp^2 * Lap(u)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt,
    int x0, int x1, int z0, int z1
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;
    // Physical box only: the model gradient outside [padLo+M, N-padHi-M)
    // per axis is cropped by EdgePadding.backward (never observable), so
    // imaging those cells is pure memory traffic.
    if (ix < x0 || ix >= x1 || iz < z0 || iz >= z1)
        return;

    long long spatial_size = (long long)nx * nz;
    int idx = iz * nx + ix;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    // Discrete adjoint of u_next = 2 u_now - u_prev + dt^2 vp^2 Lap(u_now):
    //   dL/dvp = 2 dt^2 vp * sum_t Lap(u) * u_adj
    //          = (2 dt^2 / vp) * sum_t u_forward_saved * u_adj
    grad_b[idx] += 2.f * dt * dt * u_forward_b[idx] * u_backward_b[idx] / vp_b[idx];

}

__global__ void calculate_grad_utt(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt,
    int x0, int x1, int z0, int z1
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;
    // Physical box only: the model gradient outside [padLo+M, N-padHi-M)
    // per axis is cropped by EdgePadding.backward (never observable), so
    // imaging those cells is pure memory traffic.
    if (ix < x0 || ix >= x1 || iz < z0 || iz >= z1)
        return;

    long long spatial_size = (long long)nx * nz;
    int idx = iz * nx + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    // After the forward.swap() in backward_bs the buffer roles are rotated so
    // that this expression evaluates to the centered second time derivative
    // (u(t-1) - 2 u(t) + u(t+1)) / dt^2 at the physical middle time.
    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);

    // u_tt = vp^2 * Lap(u) in the interior, same form as calculate_grad:
    //   dL/dvp += (2 dt^2 / vp) * u_tt * u_adj
    grad_b[idx] += 2.f * dt * dt * u_tt * u_backward_b[idx] / vp_b[idx];

}

__global__ void calculate_grad_utt_band(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt,
    int x0, int x1, int z0, int z1,
    int wxl, int wxh, int wzl, int wzh
) {
    // Strip cell enumeration: top rows, bottom rows (full box width), then
    // left / right columns over the inner z range (corners counted once).
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int b  = blockIdx.y;
    const int bw = x1 - x0;
    const int zi0 = z0 + wzl, zi1 = z1 - wzh;
    const int bh = zi1 - zi0;
    const int n_top = wzl * bw, n_bot = wzh * bw;
    const int n_left = wxl * bh, n_right = wxh * bh;
    int ix, iz;
    if (t < n_top) { iz = z0 + t / bw; ix = x0 + t % bw; }
    else if ((t -= n_top) < n_bot) { iz = zi1 + t / bw; ix = x0 + t % bw; }
    // Side strips: x offset innermost so a warp's threads share memory sectors.
    else if ((t -= n_bot) < n_left) { ix = x0 + t % wxl; iz = zi0 + t / wxl; }
    else if ((t -= n_left) < n_right) { ix = (x1 - wxh) + t % wxh; iz = zi0 + t / wxh; }
    else return;

    long long spatial_size = (long long)nx * nz;
    int idx = iz * nx + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    // After the forward.swap() in backward_bs the buffer roles are rotated so
    // that this expression evaluates to the centered second time derivative
    // (u(t-1) - 2 u(t) + u(t+1)) / dt^2 at the physical middle time.
    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);

    // u_tt = vp^2 * Lap(u) in the interior, same form as calculate_grad:
    //   dL/dvp += (2 dt^2 / vp) * u_tt * u_adj
    grad_b[idx] += 2.f * dt * dt * u_tt * u_backward_b[idx] / vp_b[idx];

}

__global__ void accumulate_illumination_2d(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,     // null => u_forward_next IS u_tt
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int nx, int nz, float dt,
    int x0, int x1, int z0, int z1
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;
    // The same physical box the gradient kernels image: EdgePadding.backward
    // crops the rest, and both paths must cover the same cells or the two
    // illuminations would still not be comparable.
    if (ix < x0 || ix >= x1 || iz < z0 || iz >= z1)
        return;

    long long spatial_size = (long long)nx * nz;
    int idx = iz * nx + ix;

    float* src_b = source_illumination + b * spatial_size;

    // Same expression and operand order as calculate_grad_utt_band.
    if (receiver_illumination != nullptr) {
        const float* u_backward_b2 = u_backward + b * spatial_size;
        float ub2 = u_backward_b2[idx];
        (receiver_illumination + b * spatial_size)[idx] += ub2 * ub2;
    }
    // Receiver-only is a real call: the it == 0 tail of the boundary-saving
    // reverse loop has the adjoint field but no reconstructed forward, and the
    // store-based paths DO accumulate lambda(0)^2 there.
    if (source_illumination == nullptr)
        return;

    const float* u_next_b = u_forward_next + b * spatial_size;
    float u_tt;
    if (u_forward_now == nullptr) {
        u_tt = u_next_b[idx];                    // the store already holds u_tt
    } else {
        const float* u_now_b  = u_forward_now  + b * spatial_size;
        const float* u_prev_b = u_forward_prev + b * spatial_size;
        u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);
    }
    src_b[idx] += u_tt * u_tt;
}

// Space-lag (horizontal subsurface-offset) extended imaging condition, per step:
//   E(z,x,h) += u_forward(z, x-h) * u_backward(z, x+h),  h in [-max_lag, max_lag]
// The lag is along the contiguous x axis (stride 1).  Each thread writes only
// its own (l, b, idx) cells and reads shifted neighbours of the read-only
// wavefields, so there is no intra-launch race.  Out-of-grid shifts are skipped
// (those halo/air cells carry 0 anyway).  adcig layout: (nlag, B, nz, nx).
__global__ void accumulate_adcig_2d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ adcig,
    int nlag, int max_lag,
    int B, int nx, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    long spatial_size = (long)nx * nz;
    int row = iz * nx;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;

    for (int l = -max_lag; l <= max_lag; ++l) {
        int xs = ix - l;   // u_s(x - h)
        int xr = ix + l;   // u_r(x + h)
        if (xs < 0 || xs >= nx || xr < 0 || xr >= nx)
            continue;
        float prod = u_forward_b[row + xs] * u_backward_b[row + xr];
        int lbin = l + max_lag;
        adcig[((long)lbin * B + b) * spatial_size + row + ix] += prod;
    }
}

__global__ void accumulate_source_grad_2d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || s >= nsrc) {
        return;
    }

    int base = (b * nsrc + s) * 2;
    int ix = sources_loc[base + 0];
    int iz = sources_loc[base + 1];

    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz) {
        return;
    }

    long long spatial_size = (long long)solver.nx * solver.nz;
    long long u_idx = (long long)b * spatial_size + (long long)iz * solver.nx + ix;
    long long grad_idx = ((long long)b * nsrc + s) * solver.nt + it;

    grad_source[grad_idx] += u_backward[u_idx];
}

// Tables behind the order-dispatch macros in kernels.cuh (launch/by_order.cuh).

SWEEP_BY_ORDER_TABLE(acoustic2nd_fn, acoustic2nd_by_order, acoustic2nd);
SWEEP_BY_ORDER_TABLE(acoustic2nd_adjoint_fused_fn, acoustic2nd_adjoint_fused_by_order, acoustic2nd_adjoint_fused);
SWEEP_BY_ORDER_TABLE(acoustic2nd_nopml_fn, acoustic2nd_nopml_by_order, acoustic2nd_nopml);

// Pre-pass: clear air cells (above per-column topo surface).  Launched
// BEFORE acoustic2nd so the main kernel can early-return on air cells
// without writing any aux field — eliminating intra-launch RAW races
// between air-zeroing and PML stencil reads.  See sweep VTI history.
__global__ void acoustic2d_air_clear_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + solver.x_base;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.x_end() || iz >= solver.nz) return;
    if (!solver.has_topo) return;
    if (iz >= solver.topo_rows[ix]) return;
    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    f.u_next[idx] = 0.f;
    // Aux fields live in per-axis slabs; air cells outside a slab hold an
    // implicit zero (the FD kernel never writes them), so only clear the
    // stored part.  Full-domain (legacy) tensors report stored() everywhere.
    if (solver.aux_x.stored(ix)) {
        long xi = solver.aux_idx_x2(iz, ix);
        f.psix[xi] = 0.f; f.zetax[xi] = 0.f;
    }
    if (solver.aux_z.stored(iz)) {
        long zi = solver.aux_idx_z2(iz, ix);
        f.psiz[zi] = 0.f; f.zetaz[zi] = 0.f;
    }
    if (u_this) u_this[b * spatial_size + idx] = 0.f;
}

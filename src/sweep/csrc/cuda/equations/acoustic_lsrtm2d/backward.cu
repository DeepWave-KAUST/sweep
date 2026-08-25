#include <cstdlib>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>

#include "acoustic_lsrtm2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

// ---- bs-mode (boundary-saving) scattered-field reconstruction + vp imaging ----
// In bs mode the scattered field is boundary-saved (field 1) and reconstructed in
// lockstep with the background field (field 0).  Its reverse recursion carries the
// coupling source mp*vp^2*Lap(bg): reusing the background reconstruction's own second
// difference (bg_next - 2 bg_now + bg_prev = dt^2 vp^2 Lap(bg[it])) avoids re-evaluating
// a Laplacian.  Interior only (guard mirrors acoustic2d_single_nopml); the outer ring
// is overwritten by the scattered boundary restore afterwards.
static __global__ void add_lsrtm_scattered_coupling_2d(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int nx, int nz, int halo
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix < halo || ix >= nx - halo || iz < halo || iz >= nz - halo) return;
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    sc_next[o] += mp[o] * (bg_next[o] - 2.0f * bg_now[o] + bg_prev[o]);
}

// bs-mode vp gradient from reconstructed second differences:
//   grad_vp += (2/vp) [ (bg 2nd diff)*lam_bg + (sc 2nd diff)*lam_sc ].
// bg 2nd diff = dt^2 vp^2 Lap(bg) = dt^2*bg_utt  -> term IV via lam_bg;
// sc 2nd diff = dt^2 (vp^2 Lap(sc) + mp vp^2 Lap(bg)) = dt^2 (sc_utt + mp*bg_utt),
// so the sc term folds II (sc_utt*lam_sc) and III (mp*bg_utt*lam_sc) together and the
// total matches the full mode's (2 dt^2/vp)[bg_utt*lam_bg + sc_utt*lam_sc + mp*bg_utt*lam_sc].
// III is recovered as (2/vp)*mp*(bg 2nd diff)*lam_sc, so the same beta split as full mode
// is available here: pass grad_iii != nullptr to keep II+IV and III apart.
static __global__ void calculate_grad_lsrtm_vp_utt_2d(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_ii_iv, float* __restrict__ grad_iii,
    int nx, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz) return;
    int sp = nx * nz; int o = b * sp + iz * nx + ix;
    float v = vp[o];
    float bg2 = bg_prev[o] - 2.0f * bg_now[o] + bg_next[o];
    float sc2 = sc_prev[o] - 2.0f * sc_now[o] + sc_next[o];
    float ls  = lam_sc[o];
    float total = (2.0f / v) * (bg2 * lam_bg[o] + sc2 * ls);
    if (grad_iii != nullptr) {
        float iii = (2.0f / v) * (mp[o] * bg2 * ls);   // singular image-point term
        grad_ii_iv[o] += total - iii;
        grad_iii[o]   += iii;
    } else {
        grad_ii_iv[o] += total;
    }
}

namespace acoustic_lsrtm2d {

namespace {

// Proper transpose adjoint step for the lsrtm scattered field: v2_lambda =
// vp^2 * lambda_now, then L* = lap(v2_lambda) (interior) / forward CPML (PML).
// Replaces the old forward-operator adjoint (acoustic2d_single = vp^2*lap), which is
// non-self-adjoint when vp varies (~15% grad[mp] error in variable velocity).
static inline void run_lsrtm2d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const torch::Tensor& vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    auto v2_lambda = torch::empty_like(vp);   // vp^2 * lambda_now (fully overwritten each step)
    compute_v2_lambda_lsrtm2d<<<grid, block>>>(
        vp.data_ptr<float>(), adj_view.u_now, v2_lambda.data_ptr<float>(), ctx.nx, ctx.nz, ctx.B);
    ACOUSTIC_LSRTM2D_ADJOINT(order, grid, block,
        adj_view, v2_lambda.data_ptr<float>(), vp.data_ptr<float>(),
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);
}

void advance_forward_interval_2d(
    AcousticWavefieldTensor& forward,
    int start,
    int end,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 source_grid,
    dim3 source_block,
    const BackwardInput& p,
    const torch::Tensor& vp,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc
)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();

        ACOUSTIC_LSRTM2D_SINGLE(
            order,
            wave_grid,
            wave_block,
            view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<source_grid, source_block>>>(
            view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();
    }
}

void process_recursive_interval_2d(
    int start,
    int end,
    const AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInput& p,
    const torch::Tensor& vp,
    torch::Tensor& grad_mp,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 forward_source_grid,
    dim3 forward_source_block,
    dim3 adj_source_grid,
    dim3 adj_source_block,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    CheckpointRuntime& checkpoint_runtime,
    int nx,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        AcousticWavefieldTensor forward_step;
        forward_step.allocate(vp, 2, true);
        checkpoint_runtime.copy_state(forward_step.state_tensors(), start_state.state_tensors());

        auto bg_utt = torch::zeros_like(vp);
        auto fwd_view = forward_step.view();

        ACOUSTIC_LSRTM2D_SINGLE(
            order,
            wave_grid,
            wave_block,
            fwd_view,
            true,
            bg_utt.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        forward_step.swap();

        auto adj_view = adjoint.view();
        run_lsrtm2d_adjoint_step(
            order, wave_grid, wave_block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);

        add_source<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        calculate_grad_lsrtm_mp<<<wave_grid, wave_block>>>(
            bg_utt.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx,
            nz,
            ctx.dt
        );
        return;
    }

    int mid = start + (end - start) / 2;

    AcousticWavefieldTensor mid_state;
    mid_state.allocate(vp, 2, true);
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());

    advance_forward_interval_2d(
        mid_state,
        start,
        mid,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        p,
        vp,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc
    );

    process_recursive_interval_2d(
        mid,
        end,
        mid_state,
        adjoint,
        p,
        vp,
        grad_mp,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        adj_source_grid,
        adj_source_block,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        nx,
        nz
    );

    process_recursive_interval_2d(
        start,
        mid,
        start_state,
        adjoint,
        p,
        vp,
        grad_mp,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        adj_source_grid,
        adj_source_block,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        nx,
        nz
    );
}

// grad_iii may be undefined -> III is folded into grad_vp (plain II+III+IV).
void run_full_imaging(const BackwardInput& p, torch::Tensor& grad_mp,
                      torch::Tensor& grad_vp, torch::Tensor& grad_iii)
{
    auto vp = p.models[0];
    auto mp = p.models[1];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    // scattered-field adjoint (lambda_sc): driven by the receiver residual
    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 9), 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);

    // background-field adjoint (lambda_bg = paper's mu): driven only by the scattered-source
    // coupling transpose; needed for the tomographic vp gradient (term IV).
    AcousticWavefieldTensor bg_adjoint;
    bg_adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    auto v2lbg = torch::empty_like(vp);   // vp^2*lambda_bg + mp*vp^2*lambda_sc

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;
    SolverContext ctx{2, nx, 0, nz, B, p.dt, p.nt, M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int64_t sp = (int64_t)B * nz * nx;

    for (int it = p.nt - 1; it >= 0; --it) {
        // ---- background adjoint step FIRST (lambda_bg) ----
        // Its coupling source must be lambda_sc(it+1) = adjoint.u_now BEFORE this step's
        // scattered propagation (the forward scattered source sc(it+1) += mp vp^2 Lap(bg(it))
        // transposes lambda_sc(it+1) back onto lambda_bg(it)).  Doing the scattered step
        // first would use lambda_sc(it) and be off by one timestep.
        auto bg_adj_view = bg_adjoint.view();
        compute_v2_lambda_bg_lsrtm2d<<<launch_config.grid, launch_config.block>>>(
            vp.data_ptr<float>(), mp.data_ptr<float>(),
            bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
            v2lbg.data_ptr<float>(), nx, nz, B);
        ACOUSTIC_LSRTM2D_ADJOINT(
            order, launch_config.grid, launch_config.block,
            bg_adj_view, v2lbg.data_ptr<float>(), vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);
        bg_adjoint.swap_pml();   // bg_adjoint.u_now = lambda_bg(it)

        // ---- scattered adjoint step (lambda_sc) ----
        auto adj_view = adjoint.view();

        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        // u_forward[it] is (2, B, nz, nx): [0] = bg_utt, [1] = sc_utt
        torch::Tensor u_fwd_it = p.u_forward[it];
        float* bg_utt = u_fwd_it.data_ptr<float>();
        float* sc_utt = bg_utt + sp;

        calculate_grad_lsrtm_mp<<<launch_config.grid, launch_config.block>>>(
            bg_utt,
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx,
            nz,
            ctx.dt
        );

        calculate_grad_lsrtm_vp<<<launch_config.grid, launch_config.block>>>(
            bg_utt, sc_utt,
            bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
            mp.data_ptr<float>(), vp.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_iii.defined() ? grad_iii.data_ptr<float>() : nullptr,
            nx, nz, ctx.dt
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    BackwardOutput out;
    TORCH_CHECK(in.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

    auto grad_wavelet = torch::zeros_like(in.forward_source);
    auto grad_vp = torch::zeros_like(in.models[0]);
    auto grad_mp = torch::zeros_like(in.models[1]);

    // Optional per-term split for RWI: with SWEEP_LSRTM_SPLIT_III=1 the singular
    // image-point term III is returned separately (grad_vp then holds II+IV only), so the
    // caller can apply the paper's beta weight: grad_v = II+IV + beta*III.  Off by default:
    // grad_vp is the plain summed II+III+IV and `grads` keeps its usual 3 entries.
    const char* split_env = std::getenv("SWEEP_LSRTM_SPLIT_III");
    bool split_iii = (split_env != nullptr && split_env[0] == '1');
    torch::Tensor grad_iii;
    if (split_iii)
        grad_iii = torch::zeros_like(in.models[0]);

    run_full_imaging(in, grad_mp, grad_vp, grad_iii);

    out.grads = {grad_wavelet, grad_vp, grad_mp};   // arity must stay 3 (autograd contract)
    out.grad_split_iii = grad_iii;                  // side output; undefined unless split_iii
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 9), 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(), p.forward_wavefields.begin() + 7), 2, true);
    else
        forward.allocate(vp, 2, true);
    // last_two_t: [field, time(prev,now), B, nz, nx].  field 0 = bg, field 1 = sc.
    forward.u_prev_t.copy_(p.u_last_two.select(0, 0).select(0, 1));   // bg u_now (last)
    forward.u_now_t.copy_(p.u_last_two.select(0, 0).select(0, 0));    // bg u_prev (2nd-last)

    // Scattered field, reconstructed in lockstep from its own boundary save (field 1) --
    // terms II/III/IV correlate against it, so without it grad_vp would stay zero.
    auto mp = p.models[1];
    AcousticWavefieldTensor forward_sc;
    forward_sc.allocate(vp, 2, true);
    forward_sc.u_prev_t.copy_(p.u_last_two.select(0, 1).select(0, 1));  // sc u_now (last)
    forward_sc.u_now_t.copy_(p.u_last_two.select(0, 1).select(0, 0));   // sc u_prev (2nd-last)

    // Background adjoint (lambda_bg = mu): driven only by the scattered-source coupling.
    AcousticWavefieldTensor bg_adjoint;
    bg_adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    auto v2lbg = torch::empty_like(vp);   // vp^2*lambda_bg + mp*vp^2*lambda_sc

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);
    const char* split_env_bs = std::getenv("SWEEP_LSRTM_SPLIT_III");
    bool split_iii_bs = (split_env_bs != nullptr && split_env_bs[0] == '1');
    torch::Tensor grad_iii_bs;
    if (split_iii_bs)
        grad_iii_bs = torch::zeros_like(p.models[0]);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 2, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 2, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto for_view = forward.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
    auto sc_view0 = forward_sc.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(sc_view0.u_prev, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(sc_view0.u_now, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};
    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = p.nt - 1; it >= 1; --it) {
        auto for_view_iter = forward.view();
        auto sc_view_iter = forward_sc.view();

        // ---- background adjoint step FIRST (lambda_bg) ----
        // Its coupling source is lambda_sc(it+1) = adjoint.u_now BEFORE the scattered adjoint
        // step below; combined v2 = vp^2*lambda_bg + mp*vp^2*lambda_sc lets the shared lsrtm
        // adjoint kernel do self-propagation + coupling injection in one interior pass.
        auto bg_adj_view = bg_adjoint.view();
        compute_v2_lambda_bg_lsrtm2d<<<launch_config.grid, launch_config.block>>>(
            vp.data_ptr<float>(), mp.data_ptr<float>(),
            bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
            v2lbg.data_ptr<float>(), nx, nz, B);
        ACOUSTIC_LSRTM2D_ADJOINT(
            order, launch_config.grid, launch_config.block,
            bg_adj_view, v2lbg.data_ptr<float>(), vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);
        bg_adjoint.swap_pml();   // bg_adjoint.u_now = lambda_bg(it)

        // ---- scattered adjoint step (lambda_sc): receiver residual ----
        auto adj_view = adjoint.view();
        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // adjoint.u_now = lambda_sc(it)

        // ---- reconstruct the background field (bg[it-1]); restore field 0 ----
        ACOUSTIC_LSRTM2D_SINGLE_NOPML(
            order, launch_config.grid, launch_config.block,
            for_view_iter, vp.data_ptr<float>(), lap_ctx, ctx);

        boundary_runtime.restore_backward_2d_field(
            it, for_view_iter.u_next, launch_config.grid, launch_config.block,
            bs, save_width, 0, ctx, /*field_idx=*/0, /*wait_chunk=*/true, /*record_done=*/false);

        // ---- reconstruct the scattered field (sc[it-1]); restore field 1 ----
        // Its reverse recursion carries the coupling source mp*vp^2*Lap(bg[it]) = mp*(bg 2nd
        // difference).  for_view_iter.u_next is still source-free here (the background forward
        // source is re-injected further down), so that second difference is exact.
        ACOUSTIC_LSRTM2D_SINGLE_NOPML(
            order, launch_config.grid, launch_config.block,
            sc_view_iter, vp.data_ptr<float>(), lap_ctx, ctx);
        add_lsrtm_scattered_coupling_2d<<<launch_config.grid, launch_config.block>>>(
            sc_view_iter.u_next, for_view_iter.u_next,
            forward.u_now_t.data_ptr<float>(), forward.u_prev_t.data_ptr<float>(),
            mp.data_ptr<float>(), nx, nz, M);
        boundary_runtime.restore_backward_2d_field(
            it, sc_view_iter.u_next, launch_config.grid, launch_config.block,
            bs, save_width, 0, ctx, /*field_idx=*/1, /*wait_chunk=*/false, /*record_done=*/true);

        // ---- imaging: mp (bg . lambda_sc) + vp (bg . lambda_bg + sc . lambda_sc) ----
        calculate_grad_lsrtm_mp_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_prev_t.data_ptr<float>(),
            for_view_iter.u_next,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx, nz, dt
        );

        calculate_grad_lsrtm_vp_utt_2d<<<launch_config.grid, launch_config.block>>>(
            forward.u_prev_t.data_ptr<float>(),      // bg[it+1]
            forward.u_now_t.data_ptr<float>(),       // bg[it]
            for_view_iter.u_next,                    // bg[it-1] (source-free)
            forward_sc.u_prev_t.data_ptr<float>(),   // sc[it+1]
            forward_sc.u_now_t.data_ptr<float>(),    // sc[it]
            sc_view_iter.u_next,                     // sc[it-1]
            bg_adjoint.u_now_t.data_ptr<float>(),    // lambda_bg(it)
            adjoint.u_now_t.data_ptr<float>(),       // lambda_sc(it)
            mp.data_ptr<float>(), vp.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            split_iii_bs ? grad_iii_bs.data_ptr<float>() : nullptr,
            nx, nz
        );

        // ---- re-inject the background forward source (the scattered field has none) ----
        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view_iter.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();
        forward_sc.swap();
        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();
        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};   // arity must stay 3 (autograd contract)
    out.grad_split_iii = grad_iii_bs;               // side output; undefined unless split_iii
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D checkpointing expects 6 checkpoint tensors");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 9), 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(), p.forward_wavefields.begin() + 7), 2, true);
    else
        forward.allocate(vp, 2, true);

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "acoustic_lsrtm2d"
    );

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    auto chunk_forward = torch::zeros({chunk_size, B, nz, nx}, vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors(), forward.next_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* bg_utt = chunk_forward[it - start].data_ptr<float>();

            ACOUSTIC_LSRTM2D_SINGLE(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                true,
                bg_utt,
                vp.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                for_view.u_next,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );

            forward.swap();
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();

            run_lsrtm2d_adjoint_step(
                order, launch_config.grid, launch_config.block, adj_view,
                vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx);

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

            calculate_grad_lsrtm_mp<<<launch_config.grid, launch_config.block>>>(
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad_mp.data_ptr<float>(),
                nx,
                nz,
                ctx.dt
            );
        }
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    TORCH_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D recursive checkpointing expects 6 checkpoint tensors");

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "acoustic_lsrtm2d"
    );

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 9), 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(
        p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
        "checkpoint_count does not match checkpoint_steps"
    );
    TORCH_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
        "checkpoint buffer is smaller than checkpoint_steps"
    );

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();

    AcousticWavefieldTensor start_state;
    start_state.allocate(vp, 2, true);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_2d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            grad_mp,
            order,
            launch_config.grid,
            launch_config.block,
            fwd_source_config.grid,
            fwd_source_config.block,
            adj_source_config.grid,
            adj_source_config.block,
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx,
            forward_nsrc,
            adjoint_nsrc,
            checkpoint_runtime,
            nx,
            nz
        );
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

} // namespace acoustic_lsrtm2d

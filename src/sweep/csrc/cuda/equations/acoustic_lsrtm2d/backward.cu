#include <algorithm>


#include "acoustic_lsrtm2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"

namespace acoustic_lsrtm2d {

// p.grads_out as the propagator binds it: {grad_wavelet, grad_vp, grad_mp}, in
// BackwardOutputCore.grads order, zeroed per backward on the Python side and
// accumulated here.  _c.py builds it unconditionally for every backward
// (_gradient_buffers from cuda_layout.grads_out_has_wavelet = true plus one
// slot per model), so the binding is mandatory here.
static BufList grad_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(p.grads_out.size() == 3,
                "acoustic_lsrtm2d/backward requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet + one slot per model = 3 tensors: "
                "{grad_wavelet, grad_vp, grad_mp}), got ", p.grads_out.size());
    return p.grads_out;
}

namespace {

// Proper transpose adjoint step for the lsrtm scattered field: v2_lambda =
// vp^2 * lambda_now, then L* = lap(v2_lambda) (interior) / forward CPML (PML).
// Replaces the old forward-operator adjoint (acoustic2d_single = vp^2*lap), which is

// Layout of p.adjoint_workspace, declared on the Python side by
// AcousticLSRTM.cuda_layout.backward_workspace_shapes (one padded grid per shot
// each, zeroed by the propagator before every gradient-bearing forward): the
// vp^2*lambda scratch of every adjoint step, and -- in the
// recursive-checkpoint mode only -- the background u_tt of the replayed step.
enum WorkspaceSlot : int { V2_LAMBDA = 0, BG_UTT, N_SLOTS_RECURSIVE, N_SLOTS_PLAIN = 1 };

// Exactly the count this mode declares.  The propagator allocates the pool for
// every gradient-bearing forward (_ensure_adjoint_workspace_buffers, driven by
// AcousticLSRTM.cuda_layout.backward_workspace_shapes, which returns a non-empty
// list in all four memory modes), so an empty or differently sized pool means
// the Python declaration drifted.
static BufList workspace_slots(const BackwardInputCore& p, int n_slots)
{
    SWEEP_CHECK(static_cast<int>(p.adjoint_workspace.size()) == n_slots,
                "acoustic_lsrtm2d/backward requires the propagator-bound adjoint_workspace "
                "(cuda_layout.backward_workspace_shapes): ", n_slots,
                " tensors for this mode, got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// The checkpoint replay state of the background field: one 2-D
// AcousticWavefieldTensor in its bind order (u_prev, u_now, u_next, psix, psiz,
// zetax, zetaz -- 7 model-shaped grids, no psi double-buffer shadow, so bind()
// keeps double_buffer_psi=false and the replay keeps its u-only swap(): the
// in-place psi path allocate(vp, 2, true) gives).  Declared on the Python side
// as AcousticLSRTM.cuda_layout.checkpoint_state_nvar and handed over as
// p.forward_wavefields, K sets back to back, zeroed per backward call (the
// state allocate() started from): K = 1 in the chunk mode (the replay state),
// K = 1 + depth in the recursive mode (set 0 = the segment start state, sets
// 1..depth = the bisection's scratch states, one per recursion level).
constexpr int REPLAY_STATE_NVAR = 7;

// Bind set `set` of p.forward_wavefields.  The propagator hands the sets over on
// every checkpoint-mode backward (_c.py _forward_state_buffers over
// cp.forward_state_shapes, sized from cuda_layout.checkpoint_state_nvar and
// recursive_state_depth), so there is no unbound caller to allocate for.  Every
// slot is model-shaped (LSRTM declares no per-axis CPML slabs); bind() checks the
// count only, so the geometry is checked here.
static void bind_replay_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                              const Buf& vp, int set, const char* what)
{
    SWEEP_CHECK(!p.forward_wavefields.empty(),
                what, " requires the propagator-bound forward_wavefields "
                "(cuda_layout.checkpoint_state_nvar replay state sets)");
    auto state = wavefield_set(p.forward_wavefields, set, REPLAY_STATE_NVAR, what);
    for (int i = 0; i < REPLAY_STATE_NVAR; ++i)
        pool_slot_checked(state, i, vp, what);
    wf.bind(state, 2, /*use_pml=*/true);
}

// The adjoint state: the first 9 of the propagator's 18 adjoint wavefield slots
// (cuda_layout base_nvar + pml_nvar; the background field's u triple + CPML quad
// + psi double-buffer pair).  _c.py binds cp.adjoint_wavefields on every backward
// -- _ensure_wavefield_buffers allocates them whenever the forward required a
// gradient -- so the list is never empty here.
static void bind_adjoint_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                               const char* mode)
{
    SWEEP_CHECK(p.adjoint_wavefields.size() >= 9,
                "acoustic_lsrtm2d/", mode, " requires the propagator-bound "
                "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar = 18 tensors; "
                "the background field takes the first 9), got ",
                p.adjoint_wavefields.size());
    wf.bind(std::vector<Buf>(p.adjoint_wavefields.begin(),
                                       p.adjoint_wavefields.begin() + 9), 2, true);
}

// Scratch state sets the bisecting recursive backward keeps: one per recursion
// level of the longest segment.  The same loop as eq_driver.cuh
// recursive_checkpoint_scratch_depth (not included by this driver) and the
// propagator's _c.py _recursive_scratch_depth, which sizes p.forward_wavefields
// from it.
static int recursive_checkpoint_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

// non-self-adjoint when vp varies (~15% grad[mp] error in variable velocity).
static inline void run_lsrtm2d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const Buf& vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    const std::vector<Buf>& workspace)
{
    auto v2_lambda = pool_required(workspace, V2_LAMBDA, vp, "adjoint_workspace");   // vp^2 * lambda_now (fully overwritten each step)
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
    const BackwardInputCore& p,
    const Buf& vp,
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

// One bisection node of a checkpoint segment [start, end): advance a copy of
// start_state to mid, recurse into the right half (on that copy), then into the
// left half (on start_state).  scratch_states[level] is the node's mid_state,
// one state per recursion level: copy_state fills it in full before any read,
// and it is dead once the right subtree returns, so sibling nodes share it.
// The leaf steps start_state IN PLACE: a left child's start_state has no
// reader after that leaf, a right child's start_state is its parent's
// mid_state, never re-read -- so the recursion allocates nothing per node or
// per leaf.  Buffer identity is all that differs from a per-node copy; the
// kernels and the values they read are the same.
void process_recursive_interval_2d(
    int start,
    int end,
    AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInputCore& p,
    const Buf& vp,
    Buf& grad_mp,
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
    std::vector<AcousticWavefieldTensor>& scratch_states,
    int level,
    int nx,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        auto bg_utt = pool_required(p.adjoint_workspace, BG_UTT, vp, "adjoint_workspace");
        auto fwd_view = start_state.view();

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

        start_state.swap();

        auto adj_view = adjoint.view();
        run_lsrtm2d_adjoint_step(
            order, wave_grid, wave_block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

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

    SWEEP_CHECK(level < static_cast<int>(scratch_states.size()),
                "Acoustic LSRTM 2D recursive checkpointing: scratch depth exhausted at level ", level);
    AcousticWavefieldTensor& mid_state = scratch_states[level];
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
        scratch_states,
        level + 1,
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
        scratch_states,
        level + 1,
        nx,
        nz
    );
}

void run_full_imaging(const BackwardInputCore& p, Buf& grad_mp)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
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

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

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
            p.u_forward.select(0, it).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx,
            nz,
            ctx.dt
        );
    }
}

} // namespace

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    BackwardOutputCore out;
    SWEEP_CHECK(in.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

    const auto& gs = grad_slots(in);
    workspace_slots(in, N_SLOTS_PLAIN);
    auto grad_wavelet = pool_required(gs, 0, in.forward_source, "grads_out");
    auto grad_vp = pool_required(gs, 1, in.models[0], "grads_out");
    auto grad_mp = pool_required(gs, 2, in.models[1], "grads_out");

    run_full_imaging(in, grad_mp);

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

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
    bind_adjoint_state(adjoint, p, "backward_bs");

    // Background reconstruction: u_prev/u_now/u_next only (the reverse loop
    // injects boundaries, it runs no CPML).  The propagator hands these over on
    // every boundary-saving backward (_c.py _forward_state_buffers over
    // cp.forward_state_shapes, which is cuda_layout.bs_reconstruction_nvar = 3
    // padded grids in bs mode), so the binding is mandatory.
    AcousticWavefieldTensor forward;
    wavefields_required(p.forward_wavefields, 3, vp,
                        "acoustic_lsrtm2d/backward_bs reconstruction "
                        "(cuda_layout.bs_reconstruction_nvar)");
    forward.bind(p.forward_wavefields, 2, /*use_pml=*/false);
    copy_tensor_cuda_async(forward.u_prev_t, p.u_last_two.select(1, 1).squeeze(0));
    copy_tensor_cuda_async(forward.u_now_t, p.u_last_two.select(1, 0).squeeze(0));

    const auto& gs = grad_slots(p);
    workspace_slots(p, N_SLOTS_PLAIN);
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad_mp = pool_required(gs, 2, p.models[1], "grads_out");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? M + 1 : M;
    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
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

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};
    AsyncCopyContext async_copy(staged_boundary);
    const std::vector<std::string> disk_files = p.boundary_disk_files.vec();   // the runtime keeps a pointer to it
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = p.nt - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();
        auto for_view_iter = forward.view();

        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        ACOUSTIC_LSRTM2D_SINGLE_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view_iter,
            vp.data_ptr<float>(),
            lap_ctx,
            ctx
        );

        boundary_runtime.restore_backward_2d(
            it,
            for_view_iter.u_next,
            launch_config.grid,
            launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );

        calculate_grad_lsrtm_mp_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_prev_t.data_ptr<float>(),
            for_view_iter.u_next,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx, nz, dt
        );

        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view_iter.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();
        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();
        run_lsrtm2d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

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

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    SWEEP_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    SWEEP_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D checkpointing expects 6 checkpoint tensors");

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
    bind_adjoint_state(adjoint, p, "backward_ckpt");

    // The chunk replay state: the one set of p.forward_wavefields.  Every chunk
    // re-seeds all 7 (load the 6 checkpointed fields + zero u_next) before any
    // read, so the propagator's per-call zeroing is all the initialisation it
    // ever needs.
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == REPLAY_STATE_NVAR,
                "acoustic_lsrtm2d/backward_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.checkpoint_state_nvar): one replay "
                "state set of ", REPLAY_STATE_NVAR, " tensors, got ",
                p.forward_wavefields.size());
    AcousticWavefieldTensor forward;
    bind_replay_state(forward, p, vp, /*set=*/0, "acoustic_lsrtm2d ckpt replay state");

    const auto& gs = grad_slots(p);
    workspace_slots(p, N_SLOTS_PLAIN);
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad_mp = pool_required(gs, 2, p.models[1], "grads_out");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
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
    // Python-allocated with the checkpoint snapshots; every row is written by the replay before the reverse pass reads it.
    auto chunk_forward = pool_required(p.checkpoint_replay, 0, {chunk_size, B, nz, nx},
                                       "checkpoint_replay (acoustic_lsrtm2d/backward_ckpt, "
                                       "cuda_layout.checkpoint_replay_shapes)");

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors(), forward.next_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* bg_utt = chunk_forward.select(0, it - start).data_ptr<float>();

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
                vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

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
                chunk_forward.select(0, it - start).data_ptr<float>(),
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

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    SWEEP_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D recursive checkpointing expects 6 checkpoint tensors");

    const Buf& checkpoint_steps_cpu = p.checkpoint_steps;   // a host copy, made by the adapter
    SWEEP_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
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
    bind_adjoint_state(adjoint, p, "backward_recursive_ckpt");
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    const auto& gs = grad_slots(p);
    workspace_slots(p, N_SLOTS_RECURSIVE);
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad_mp = pool_required(gs, 2, p.models[1], "grads_out");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    SWEEP_CHECK(
        p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
        "checkpoint_count does not match checkpoint_steps"
    );
    SWEEP_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
        "checkpoint buffer is smaller than checkpoint_steps"
    );

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();

    int max_segment_length = 0;
    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];
        max_segment_length = std::max(max_segment_length, end - start);
    }

    // Replay state sets of p.forward_wavefields: set 0 the segment start state
    // (zeroed or checkpoint-loaded per segment before any read), sets 1..depth
    // the bisection's scratch states (copy_state-filled from their parent
    // before any read).  The propagator hands 1 + depth sets, its depth
    // (_c.py _recursive_scratch_depth) mirroring recursive_checkpoint_scratch_depth
    // on the same longest segment.
    const int scratch_depth = recursive_checkpoint_scratch_depth(max_segment_length);
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == (1 + scratch_depth) * REPLAY_STATE_NVAR,
                "acoustic_lsrtm2d/backward_recursive_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.checkpoint_state_nvar + "
                "recursive_state_depth): ", 1 + scratch_depth, " replay state sets of ",
                REPLAY_STATE_NVAR, " tensors, got ", p.forward_wavefields.size());

    AcousticWavefieldTensor start_state;
    bind_replay_state(start_state, p, vp, /*set=*/0, "acoustic_lsrtm2d recursive start state");

    std::vector<AcousticWavefieldTensor> scratch_states(scratch_depth);
    for (int level = 0; level < scratch_depth; ++level)
        bind_replay_state(scratch_states[level], p, vp, /*set=*/level + 1,
                          "acoustic_lsrtm2d recursive scratch state");

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
            scratch_states,
            /*level=*/0,
            nx,
            nz
        );
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}






} // namespace acoustic_lsrtm2d

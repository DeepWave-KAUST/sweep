#include <algorithm>

#include "acoustic_lsrtm3d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"

namespace acoustic_lsrtm3d {

// p.grads_out as the propagator binds it: {grad_wavelet, grad_vp, grad_mp}, in
// BackwardOutputCore.grads order (zeroed per backward on the Python side and
// accumulated here), or empty for an unbound caller, which then gets fresh
// zeros per slot.
static BufList grad_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(p.grads_out.size() == 3,
                "acoustic_lsrtm3d/backward requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet + one slot per model = 3 tensors: "
                "{grad_wavelet, grad_vp, grad_mp}), got ", p.grads_out.size());
    return p.grads_out;
}

namespace {

// Proper transpose adjoint step for the lsrtm 3D scattered field: v2_lambda =
// vp^2 * lambda_now, then L* = lap(v2_lambda) (interior) / forward CPML (PML).
// Replaces the old forward-operator adjoint (acoustic3d_single = vp^2*lap),

// Layout of p.adjoint_workspace, declared on the Python side by
// AcousticLSRTM3D.cuda_layout.backward_workspace_shapes (one padded grid per
// shot each, zeroed by the propagator before every gradient-bearing forward):
// the vp^2*lambda scratch of every adjoint step, plus one grid the modes use
// differently -- the replayed step's field in the recursive mode, the forward
// step in the boundary-saving mode; the modes never share a pool. The
// checkpoint replay STATE is not here: it rides p.forward_wavefields as state
// sets (bind_replay_state_set below).
enum WorkspaceSlot : int { V2_LAMBDA = 0, U_THIS = 1, F_THIS = 1, N_SLOTS_PLAIN = 1, N_SLOTS_EXTRA = 2 };
// RWI: the background adjoint's v2 = vp^2*g and g = lambda_bg + mp*lambda_sc itself
// (which the adjoint step's CPML band differentiates) are the LAST two slots of the
// full (V2_LAMBDA + them) and bs (V2_LAMBDA, F_THIS + them) pools;
// _adjoint_workspace_shapes declares 3 and 4 slots for those modes.
constexpr int N_SLOTS_RWI_FULL = 3;
constexpr int N_SLOTS_RWI_BS = 4;

// One checkpoint replay state set: the 3-D AcousticWavefieldTensor in its bind
// order (u_prev, u_now, u_next, psix, psiz, zetax, zetaz, psiy, zetay -- no
// psi double-buffer shadow, so bind() keeps double_buffer_psi=false and the
// replay keeps its u-only swap()), declared as
// AcousticLSRTM3D.cuda_layout.checkpoint_state_nvar because the LSRTM forward
// slot list is two acoustic layouts back to back.
static constexpr int CKPT_STATE_NVAR = 9;

// Exactly the count this mode declares.  The propagator allocates the pool for
// every gradient-bearing forward (_ensure_adjoint_workspace_buffers, driven by
// AcousticLSRTM3D.cuda_layout.backward_workspace_shapes, which returns a
// non-empty list in all four memory modes), so an empty or differently sized
// pool means the Python declaration drifted.
static BufList workspace_slots(const BackwardInputCore& p, int n_slots)
{
    SWEEP_CHECK(static_cast<int>(p.adjoint_workspace.size()) == n_slots,
                "acoustic_lsrtm3d/backward requires the propagator-bound adjoint_workspace "
                "(cuda_layout.backward_workspace_shapes): ", n_slots,
                " tensors for this mode, got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// Bind replay state set `set` of p.forward_wavefields (K sets of
// CKPT_STATE_NVAR model-shaped grids back to back, zeroed by the propagator
// per backward call: set 0 the chunk replay / segment start state, sets
// 1..depth the bisection's scratch states) into wf.  The propagator hands the
// sets over on every checkpoint-mode backward (_c.py _forward_state_buffers
// over cp.forward_state_shapes, sized from cuda_layout.checkpoint_state_nvar
// and recursive_state_depth), so there is no unbound caller to allocate for.
// wavefield_set checks count, dtype, device and contiguity; the model shape is
// checked here because bind() counts only and a slot of the wrong shape would
// read as garbage inside the kernels.
static void bind_replay_state_set(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                                  const Buf& vp, int set, const char* what)
{
    SWEEP_CHECK(!p.forward_wavefields.empty(),
                what, " requires the propagator-bound forward_wavefields "
                "(cuda_layout.checkpoint_state_nvar replay state sets)");
    auto tensors = wavefield_set(p.forward_wavefields, set, CKPT_STATE_NVAR, what);
    for (int i = 0; i < CKPT_STATE_NVAR; ++i)
        SWEEP_CHECK(tensors[i].sizes() == vp.sizes(),
                    what, ": set ", set, " slot ", i, " has shape ", tensors[i].sizes(),
                    " but the replay state is model-shaped ", vp.sizes());
    wf.bind(tensors, 3, /*use_pml=*/true);
}

// Scratch state sets the bisecting recursive-checkpoint backward needs for a
// segment of interval_length steps: one per recursion level until the halves
// are single steps. The same loop as eq_driver.cuh
// recursive_checkpoint_scratch_depth (not included here) and the propagator's
// _c.py _recursive_scratch_depth, which sizes the sets it hands over.
static int recursive_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

// non-self-adjoint when vp varies (~15% grad[mp] error in variable velocity).
static inline void run_lsrtm3d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const Buf& vp,
    LaplaceParam lap_ctx, GradParam grad_ctx, GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    const std::vector<Buf>& workspace)
{
    auto v2_lambda = pool_required(workspace, V2_LAMBDA, vp, "adjoint_workspace");   // vp^2 * lambda_now (fully overwritten each step)
    compute_v2_lambda_lsrtm3d<<<grid, block>>>(
        vp.data_ptr<float>(), adj_view.u_now, v2_lambda.data_ptr<float>(), ctx.nx, ctx.ny, ctx.nz, ctx.B);
    ACOUSTIC_LSRTM3D_ADJOINT(order, grid, block,
        adj_view, v2_lambda.data_ptr<float>(), /*pml_field=*/nullptr, vp.data_ptr<float>(),
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx);
}

std::vector<Buf> slice_wavefields(
    const std::vector<Buf>& tensors,
    size_t start,
    size_t count
) {
    SWEEP_CHECK(
        tensors.size() >= start + count,
        "Acoustic LSRTM 3D wavefield buffer does not contain enough tensors."
    );
    return std::vector<Buf>(
        tensors.begin() + static_cast<long>(start),
        tensors.begin() + static_cast<long>(start + count)
    );
}

// The adjoint state: the first 12 of the propagator's 24 adjoint wavefield
// slots (cuda_layout base_nvar + pml_nvar; the background field's u triple +
// CPML sextet + psi double-buffer triple).  _c.py binds cp.adjoint_wavefields
// on every backward -- _ensure_wavefield_buffers allocates them whenever the
// forward required a gradient -- so the list is never empty here.
void bind_adjoint_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                        const char* mode)
{
    SWEEP_CHECK(p.adjoint_wavefields.size() >= 12,
                "acoustic_lsrtm3d/", mode, " requires the propagator-bound "
                "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar = 24 tensors; "
                "the background field takes the first 12), got ",
                p.adjoint_wavefields.size());
    wf.bind(slice_wavefields(p.adjoint_wavefields, 0, 12), 3, true);
}

// The background adjoint lambda_bg = mu (Wu & Alkhalifah 2015): the second 12 of
// the 24 adjoint slots.  LSRTM declares two fields, so the propagator already sizes
// the adjoint pool for both; the scattered adjoint takes 0..11, these were unused.
void bind_bg_adjoint_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                           const char* mode)
{
    SWEEP_CHECK(p.adjoint_wavefields.size() >= 24,
                "acoustic_lsrtm3d/", mode, " needs all 24 adjoint_wavefields "
                "(cuda_layout.base_nvar + pml_nvar) for the RWI background adjoint, got ",
                p.adjoint_wavefields.size());
    wf.bind(slice_wavefields(p.adjoint_wavefields, 12, 12), 3, true);
}

BackwardOutputCore backward_full_imaging_impl(const BackwardInputCore& p);
BackwardOutputCore backward_bs_imaging_impl(const BackwardInputCore& p);
BackwardOutputCore backward_ckpt_imaging_impl(const BackwardInputCore& p);
BackwardOutputCore backward_recursive_imaging_impl(const BackwardInputCore& p);

} // namespace

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    SWEEP_CHECK(in.models.size() == 2, "Acoustic LSRTM 3D backward expects two models.");
    return backward_full_imaging_impl(in);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    SWEEP_CHECK(in.models.size() == 2, "Acoustic LSRTM 3D backward expects two models.");
    return backward_bs_imaging_impl(in);
}

namespace {

void accumulate_rtm_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* forward_ptr,
    const float* adjoint_ptr,
    RTMOutputCore& out,
    int B,
    int nx,
    int ny,
    int nz
)
{
    // LSRTM keeps exactly what it had: its callers hand in the quantity they
    // want squared (so the u_tt branch is off), and the box spans the whole
    // padded grid, which is the region this kernel covered before it learnt
    // about the physical box. dt is unused when u_forward_now is null.
    accumulate_illumination_3d<<<wave_grid, wave_block>>>(
        forward_ptr, nullptr, nullptr,
        adjoint_ptr,
        out.source_illumination.data_ptr<float>(),
        out.receiver_illumination.data_ptr<float>(),
        B, nx, ny, nz, 0.0f,
        0, nx, 0, ny, 0, nz
    );
}

void accumulate_imaging_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* forward_ptr,
    const float* adjoint_ptr,
    const Buf& vp,
    Buf* grad,
    RTMOutputCore* rtm_out,
    int B,
    int nx,
    int ny,
    int nz,
    float dt
)
{
    if (grad != nullptr) {
        calculate_grad_lsrtm3d_mp<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            B, nx, ny, nz, dt
        );
        return;
    }

    SWEEP_CHECK(rtm_out != nullptr, "Imaging accumulation requires grad or RTM output.");
    accumulate_rtm_3d(wave_grid, wave_block, forward_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
}

void accumulate_imaging_utt_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* u_next_ptr,
    const float* u_now_ptr,
    const float* u_prev_ptr,
    const float* adjoint_ptr,
    const Buf& vp,
    float dt,
    Buf* grad,
    RTMOutputCore* rtm_out,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (grad != nullptr) {
        calculate_grad_lsrtm3d_mp_utt<<<wave_grid, wave_block>>>(
            u_next_ptr,
            u_now_ptr,
            u_prev_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            B, nx, ny, nz, dt
        );
        return;
    }

    SWEEP_CHECK(rtm_out != nullptr, "Imaging accumulation requires grad or RTM output.");
    accumulate_rtm_3d(wave_grid, wave_block, u_now_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
}

void accumulate_source_gradient_3d(
    dim3 source_grid,
    dim3 source_block,
    const float* adjoint_ptr,
    const BackwardInputCore& p,
    Buf* grad_wavelet,
    int it,
    const SolverContext& ctx,
    int nsrc
)
{
    if (grad_wavelet == nullptr) {
        return;
    }

    accumulate_source_grad_3d<<<source_grid, source_block>>>(
        adjoint_ptr,
        grad_wavelet->data_ptr<float>(),
        p.forward_sources_loc.data_ptr<int>(),
        it,
        nsrc,
        ctx
    );
}

void advance_forward_interval_3d(
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
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc
)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();

        ACOUSTIC_LSRTM3D_SINGLE(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<source_grid, source_block>>>(
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

// Bisection over one checkpoint segment [start, end): a node copies its
// start_state into scratch_states[level] (one AcousticWavefieldTensor per
// recursion level, replay state set 1 + level of p.forward_wavefields),
// advances that copy to mid, recurses into [mid, end) with it and then into
// [start, mid) with start_state. A leaf steps its start_state IN PLACE: a left
// child is the last reader of its start_state at that level, and a right
// child's start_state is the parent's mid_state, which is never re-read.
// Every scratch state is fully overwritten by copy_state before any read, so
// only the buffers' identity changed against the per-node / per-leaf
// allocations this replaces -- the kernels, their order and their operands'
// values did not.
void process_recursive_interval_3d(
    int start,
    int end,
    AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInputCore& p,
    const Buf& vp,
    Buf* grad,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out,
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
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    CheckpointRuntime& checkpoint_runtime,
    std::vector<AcousticWavefieldTensor>& scratch_states,
    int level,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        // The leaf replays its one step on start_state in place (see above).
        auto u_this = pool_required(p.adjoint_workspace, U_THIS, vp, "adjoint_workspace");
        auto fwd_view = start_state.view();

        ACOUSTIC_LSRTM3D_SINGLE(
            order,
            wave_grid,
            wave_block,
            fwd_view,
            true,
            u_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        start_state.swap();

        auto adj_view = adjoint.view();

        run_lsrtm3d_adjoint_step(
            order, wave_grid, wave_block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

        add_source_3d<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        accumulate_source_gradient_3d(
            forward_source_grid,
            forward_source_block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            start,
            ctx,
            forward_nsrc
        );

        if (grad != nullptr) {
            calculate_grad_lsrtm3d_mp<<<wave_grid, wave_block>>>(
                u_this.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad->data_ptr<float>(),
                B, nx, ny, nz, p.dt
            );
        } else {
            SWEEP_CHECK(rtm_out != nullptr, "Recursive RTM accumulation requested without RTM output.");
            accumulate_rtm_3d(
                wave_grid,
                wave_block,
                u_this.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                *rtm_out,
                B,
                nx,
                ny,
                nz
            );
        }
        return;
    }

    int mid = start + (end - start) / 2;

    SWEEP_CHECK(level < static_cast<int>(scratch_states.size()),
                "Acoustic LSRTM 3D recursive checkpoint scratch depth exhausted: level ", level,
                " of ", scratch_states.size(), " scratch states.");
    AcousticWavefieldTensor& mid_state = scratch_states[level];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
    advance_forward_interval_3d(
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc
    );

    process_recursive_interval_3d(
        mid,
        end,
        mid_state,
        adjoint,
        p,
        vp,
        grad,
        grad_wavelet,
        rtm_out,
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        scratch_states,
        level + 1,
        B,
        nx,
        ny,
        nz
    );

    process_recursive_interval_3d(
        start,
        mid,
        start_state,
        adjoint,
        p,
        vp,
        grad,
        grad_wavelet,
        rtm_out,
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        scratch_states,
        level + 1,
        B,
        nx,
        ny,
        nz
    );
}

void run_full_imaging(
    const BackwardInputCore& p,
    bool rwi,
    Buf* grad_vp,
    Buf* grad,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out
)
{
    auto vp = p.models[0];
    auto mp = p.models[1];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward");
    // rwi (vp needs its gradient): the background adjoint and its scratch.
    AcousticWavefieldTensor bg_adjoint;
    Buf v2lbg, gbg;
    if (rwi) {
        bind_bg_adjoint_state(bg_adjoint, p, "backward");
        v2lbg = pool_required(p.adjoint_workspace, N_SLOTS_RWI_FULL - 2, vp, "adjoint_workspace");
        gbg = pool_required(p.adjoint_workspace, N_SLOTS_RWI_FULL - 1, vp, "adjoint_workspace");
    }
    // RWI beta split (grad_split_iii_out bound): III goes there, grad_vp keeps II+IV.
    SWEEP_CHECK(rwi || !p.grad_split_iii_out.defined(),
                "acoustic_lsrtm3d: grad_split_iii_out needs the RWI vp gradient (vp requires grad)");
    float* grad_iii = p.grad_split_iii_out.defined()
        ? bound_required(p.grad_split_iii_out, vp.sizes().vec(), "grad_split_iii_out").data_ptr<float>()
        : nullptr;

    float* u_thist = nullptr;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {
        // ---- background adjoint FIRST (lambda_bg = mu; rwi only): its coupling
        // source is lambda_sc(it+1) = adjoint.u_now before the scattered step below ----
        if (rwi) {
            auto bg_adj_view = bg_adjoint.view();
            compute_v2_lambda_bg_lsrtm3d<<<launch_config.grid, launch_config.block>>>(
                vp.data_ptr<float>(), mp.data_ptr<float>(),
                bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                v2lbg.data_ptr<float>(), gbg.data_ptr<float>(), nx, ny, nz, B);
            ACOUSTIC_LSRTM3D_ADJOINT(
                order, launch_config.grid, launch_config.block,
                bg_adj_view, v2lbg.data_ptr<float>(), gbg.data_ptr<float>(), vp.data_ptr<float>(),
                lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx);
            bg_adjoint.swap_pml();   // bg_adjoint.u_now = lambda_bg(it)
        }

        auto adj_view = adjoint.view();

        run_lsrtm3d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        accumulate_source_gradient_3d(
            forward_source_config.grid,
            forward_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );

        // u_forward[it] is (B, nz, ny, nx) = bg_utt, or with rwi (2, B, nz, ny, nx) =
        // [bg_utt, sc_utt] (history_fields(2))
        auto u_fwd_it = p.u_forward.select(0, it);
        float* bg_utt = (rwi ? u_fwd_it.select(0, 0) : u_fwd_it).data_ptr<float>();
        if (rwi) {
            float* sc_utt = u_fwd_it.select(0, 1).data_ptr<float>();
            calculate_grad_lsrtm3d_vp_utt<<<launch_config.grid, launch_config.block>>>(
                bg_utt, sc_utt,
                bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                mp.data_ptr<float>(), vp.data_ptr<float>(),
                grad_vp->data_ptr<float>(), grad_iii,
                B, nx, ny, nz, ctx.dt);
        }

        accumulate_imaging_3d(
            launch_config.grid,
            launch_config.block,
            bg_utt,
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            B,
            nx,
            ny,
            nz,
            ctx.dt
        );
    }
}

BackwardOutputCore backward_full_imaging_impl(const BackwardInputCore& p)
{
    sweep::DeviceGuard device_guard(device_index_of(p.models[0]));
    BackwardOutputCore out;
    // The history stacks [bg_utt, sc_utt] exactly when vp needs its RWI gradient
    // (cuda_layout_for_grads); otherwise this is the mp-only backward.
    const bool rwi = p.u_forward.dim() == 6;
    const auto& gs = grad_slots(p);
    workspace_slots(p, rwi ? N_SLOTS_RWI_FULL : N_SLOTS_PLAIN);
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad = pool_required(gs, 2, p.models[1], "grads_out");
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    run_full_imaging(p, rwi, &grad_vp, &grad, &grad_wavelet, nullptr);
    out.grads = {grad_wavelet, grad_vp, grad};
    return out;
}

void run_bs_imaging(
    const BackwardInputCore& p,
    bool rwi,
    Buf* grad_vp,
    Buf* grad,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out
)
{
    auto vp = p.models[0];
    auto mp = p.models[1];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, nullptr, nullptr, dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_bs");

    // Background reconstruction: u_prev/u_now/u_next only (no CPML in the
    // reverse loop).  The propagator hands these over on every boundary-saving
    // backward (_c.py _forward_state_buffers over cp.forward_state_shapes,
    // which is cuda_layout.bs_reconstruction_nvar = 3 padded grids in bs mode),
    // so the binding is mandatory.
    // RWI: 6 grids -- the background's u_prev/u_now/u_next, then the scattered
    // field's, reconstructed in lockstep (terms II/III/IV correlate against it).
    AcousticWavefieldTensor forward;
    AcousticWavefieldTensor forward_sc;
    wavefields_required(p.forward_wavefields, rwi ? 6 : 3, vp,
                        "acoustic_lsrtm3d/backward_bs reconstruction "
                        "(cuda_layout.bs_reconstruction_nvar)");
    forward.bind(slice_wavefields(p.forward_wavefields, 0, 3), 3, /*use_pml=*/false);
    if (rwi)
        forward_sc.bind(slice_wavefields(p.forward_wavefields, 3, 3), 3, /*use_pml=*/false);
    // u_last_two: [field, time(prev,now), B, nz, ny, nx]; time-reversed into the recon.
    copy_tensor_cuda_async(forward.u_prev_t, p.u_last_two.select(0, 0).select(0, 1));
    copy_tensor_cuda_async(forward.u_now_t, p.u_last_two.select(0, 0).select(0, 0));
    if (rwi) {
        copy_tensor_cuda_async(forward_sc.u_prev_t, p.u_last_two.select(0, 1).select(0, 1));
        copy_tensor_cuda_async(forward_sc.u_now_t, p.u_last_two.select(0, 1).select(0, 0));
    }
    AcousticWavefieldTensor bg_adjoint;
    Buf v2lbg, gbg;
    if (rwi) {
        bind_bg_adjoint_state(bg_adjoint, p, "backward_bs");
        v2lbg = pool_required(p.adjoint_workspace, N_SLOTS_RWI_BS - 2, vp, "adjoint_workspace");
        gbg = pool_required(p.adjoint_workspace, N_SLOTS_RWI_BS - 1, vp, "adjoint_workspace");
    }
    SWEEP_CHECK(rwi || !p.grad_split_iii_out.defined(),
                "acoustic_lsrtm3d: grad_split_iii_out needs the RWI vp gradient (vp requires grad)");
    float* grad_iii = p.grad_split_iii_out.defined()
        ? bound_required(p.grad_split_iii_out, vp.sizes().vec(), "grad_split_iii_out").data_ptr<float>()
        : nullptr;

    auto f_this = pool_required(p.adjoint_workspace, F_THIS, vp, "adjoint_workspace");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, rwi ? 2 : 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
    } else {
        boundary_saver.allocate(
            true, 3, rwi ? 2 : 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    AsyncCopyContext async_copy(staged_boundary);
    const std::vector<std::string> disk_files = p.boundary_disk_files.vec();   // the runtime keeps a pointer to it
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
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
        auto for_view = forward.view();
        AcousticWavefieldPointer sc_view{};
        if (rwi)
            sc_view = forward_sc.view();

        // ---- background adjoint FIRST (lambda_bg = mu; rwi only) ----
        if (rwi) {
            auto bg_adj_view = bg_adjoint.view();
            compute_v2_lambda_bg_lsrtm3d<<<launch_config.grid, launch_config.block>>>(
                vp.data_ptr<float>(), mp.data_ptr<float>(),
                bg_adjoint.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                v2lbg.data_ptr<float>(), gbg.data_ptr<float>(), nx, ny, nz, B);
            ACOUSTIC_LSRTM3D_ADJOINT(
                order, launch_config.grid, launch_config.block,
                bg_adj_view, v2lbg.data_ptr<float>(), gbg.data_ptr<float>(), vp.data_ptr<float>(),
                lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx);
            bg_adjoint.swap_pml();   // bg_adjoint.u_now = lambda_bg(it)
        }

        auto adj_view = adjoint.view();

        run_lsrtm3d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        accumulate_source_gradient_3d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );

        ACOUSTIC_LSRTM3D_SINGLE_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            f_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
            ctx
        );

        if (rwi)
            boundary_runtime.restore_backward_3d_field(
                it, for_view.u_next, launch_config.grid, launch_config.block,
                bs, save_width, 0, ctx, /*field_idx=*/0, /*wait_chunk=*/true, /*record_done=*/false);
        else
            boundary_runtime.restore_backward_3d(
                it, for_view.u_next, launch_config.grid, launch_config.block,
                bs, save_width, 0, ctx);

        // Strip source cells back to w^{it-1} - s^{it} before the imaging and
        // the add_source below (see common.cuh).
        sub_source_in_restore_strip_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            save_width, /*offset=*/0, /*tangent_pad=*/0,
            ctx
        );

        // From here until add_source, for_view.u_next is the SOURCE-FREE bg[it-1],
        // so bg_{it-1} - 2 bg_it + bg_{it+1} = dt^2 vp^2 W(bg_it) exactly.
        // ---- scattered reconstruction sc[it-1] (boundary field 1) + vp imaging; rwi only ----
        if (rwi) {
            ACOUSTIC_LSRTM3D_SINGLE_NOPML(
                order, launch_config.grid, launch_config.block,
                sc_view, f_this.data_ptr<float>(), vp.data_ptr<float>(), lap_ctx, ctx);
            add_lsrtm3d_scattered_coupling<<<launch_config.grid, launch_config.block>>>(
                sc_view.u_next, for_view.u_next,
                forward.u_now_t.data_ptr<float>(), forward.u_prev_t.data_ptr<float>(),
                mp.data_ptr<float>(), B, nx, ny, nz, p.M);
            boundary_runtime.restore_backward_3d_field(
                it, sc_view.u_next, launch_config.grid, launch_config.block,
                bs, save_width, 0, ctx, /*field_idx=*/1, /*wait_chunk=*/false, /*record_done=*/true);

            calculate_grad_lsrtm3d_vp_2diff<<<launch_config.grid, launch_config.block>>>(
                forward.u_prev_t.data_ptr<float>(),     // bg[it+1]
                forward.u_now_t.data_ptr<float>(),      // bg[it]
                for_view.u_next,                        // bg[it-1], source-free
                forward_sc.u_prev_t.data_ptr<float>(),  // sc[it+1]
                forward_sc.u_now_t.data_ptr<float>(),   // sc[it]
                sc_view.u_next,                         // sc[it-1]
                bg_adjoint.u_now_t.data_ptr<float>(),   // lambda_bg(it)
                adjoint.u_now_t.data_ptr<float>(),      // lambda_sc(it)
                mp.data_ptr<float>(), vp.data_ptr<float>(),
                grad_vp->data_ptr<float>(), grad_iii,
                B, nx, ny, nz);
        }

        accumulate_imaging_utt_3d(
            launch_config.grid,
            launch_config.block,
            forward.u_prev_t.data_ptr<float>(),
            for_view.u_next,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            p.dt,
            grad,
            rtm_out,
            B,
            nx,
            ny,
            nz
        );

        add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();
        if (rwi)
            forward_sc.swap();
        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();

        run_lsrtm3d_adjoint_step(
            order, launch_config.grid, launch_config.block, adj_view,
            vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        accumulate_source_gradient_3d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            0,
            ctx,
            forward_nsrc
        );
    }
}

BackwardOutputCore backward_bs_imaging_impl(const BackwardInputCore& p)
{
    sweep::DeviceGuard device_guard(device_index_of(p.models[0]));
    BackwardOutputCore out;
    // rwi (vp needs its gradient, cuda_layout_for_grads) binds 6 reconstruction
    // grids, the mp-only backward the background's 3.
    SWEEP_CHECK(p.forward_wavefields.size() == 3 || p.forward_wavefields.size() == 6,
                "acoustic_lsrtm3d/backward_bs reconstruction: 3 grids (mp only) or 6 "
                "(with the RWI vp gradient), got ", p.forward_wavefields.size());
    const bool rwi = p.forward_wavefields.size() == 6;
    const auto& gs = grad_slots(p);
    workspace_slots(p, rwi ? N_SLOTS_RWI_BS : N_SLOTS_EXTRA);
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad = pool_required(gs, 2, p.models[1], "grads_out");
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    run_bs_imaging(p, rwi, &grad_vp, &grad, &grad_wavelet, nullptr);
    out.grads = {grad_wavelet, grad_vp, grad};
    return out;
}

void run_ckpt_imaging(
    const BackwardInputCore& p,
    Buf* grad,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out
)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_ckpt");

    // The chunk replay state: replay state set 0 of p.forward_wavefields, the
    // only set in chunk mode. Every chunk re-seeds all 9 tensors (load the 8
    // checkpointed fields + zero u_next) before any read.
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == CKPT_STATE_NVAR,
                "acoustic_lsrtm3d/backward_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.checkpoint_state_nvar): one replay "
                "state set of ", CKPT_STATE_NVAR, " tensors, got ",
                p.forward_wavefields.size());
    AcousticWavefieldTensor forward;
    bind_replay_state_set(forward, p, vp, /*set=*/0, "acoustic_lsrtm3d ckpt replay state");

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "acoustic_lsrtm3d"
    );

    // Python-allocated with the checkpoint snapshots; every row is written by the replay before the reverse pass reads it.
    auto chunk_forward = pool_required(p.checkpoint_replay, 0, {p.checkpoint_interval, B, nz, ny, nx},
                                       "checkpoint_replay (acoustic_lsrtm3d/backward_ckpt, "
                                       "cuda_layout.checkpoint_replay_shapes)");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors(), forward.next_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* u_this = chunk_forward.select(0, it - start).data_ptr<float>();

            ACOUSTIC_LSRTM3D_SINGLE(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                true,
                u_this,
                vp.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_y,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
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

            run_lsrtm3d_adjoint_step(
                order, launch_config.grid, launch_config.block, adj_view,
                vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx, p.adjoint_workspace);

            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

            accumulate_source_gradient_3d(
                fwd_source_config.grid,
                fwd_source_config.block,
                adjoint.u_now_t.data_ptr<float>(),
                p,
                grad_wavelet,
                it,
                ctx,
                forward_nsrc
            );

            accumulate_imaging_3d(
                launch_config.grid,
                launch_config.block,
                chunk_forward.select(0, it - start).data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp,
                grad,
                rtm_out,
                B,
                nx,
                ny,
                nz,
                ctx.dt
            );
        }
    }
}

BackwardOutputCore backward_ckpt_imaging_impl(const BackwardInputCore& p)
{
    sweep::DeviceGuard device_guard(device_index_of(p.models[0]));
    BackwardOutputCore out;
    const auto& gs = grad_slots(p);
    workspace_slots(p, N_SLOTS_PLAIN);
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad = pool_required(gs, 2, p.models[1], "grads_out");
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    run_ckpt_imaging(p, &grad, &grad_wavelet, nullptr);
    out.grads = {grad_wavelet, grad_vp, grad};
    return out;
}

void run_recursive_imaging(
    const BackwardInputCore& p,
    Buf* grad,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out
)
{
    auto vp = p.models[0];

    const Buf& checkpoint_steps_cpu = p.checkpoint_steps;   // a host copy, made by the adapter
    SWEEP_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "acoustic_lsrtm3d"
    );

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_recursive_ckpt");
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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

    // Replay state sets of p.forward_wavefields: set 0 is the segment start
    // state (zeroed or checkpoint-loaded per segment before any read), sets
    // 1..depth the bisection's scratch states (copy_state-filled from their
    // parent before any read). The propagator hands 1 + depth sets, its depth
    // (_c.py _recursive_scratch_depth) mirroring recursive_scratch_depth on
    // the same longest segment.
    const int scratch_depth = recursive_scratch_depth(max_segment_length);
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == (1 + scratch_depth) * CKPT_STATE_NVAR,
                "acoustic_lsrtm3d/backward_recursive_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.checkpoint_state_nvar + "
                "recursive_state_depth): ", 1 + scratch_depth, " replay state sets of ",
                CKPT_STATE_NVAR, " tensors, got ", p.forward_wavefields.size());

    AcousticWavefieldTensor start_state;
    bind_replay_state_set(start_state, p, vp, /*set=*/0, "acoustic_lsrtm3d recursive start state");

    std::vector<AcousticWavefieldTensor> scratch_states(scratch_depth);
    for (int level = 0; level < scratch_depth; ++level)
        bind_replay_state_set(scratch_states[level], p, vp, /*set=*/level + 1,
                              "acoustic_lsrtm3d recursive scratch state");

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_3d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            grad,
            grad_wavelet,
            rtm_out,
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx,
            forward_nsrc,
            adjoint_nsrc,
            checkpoint_runtime,
            scratch_states,
            /*level=*/0,
            B,
            nx,
            ny,
            nz
        );
    }

}

BackwardOutputCore backward_recursive_imaging_impl(const BackwardInputCore& p)
{
    sweep::DeviceGuard device_guard(device_index_of(p.models[0]));
    BackwardOutputCore out;
    const auto& gs = grad_slots(p);
    workspace_slots(p, N_SLOTS_EXTRA);
    auto grad_vp = pool_required(gs, 1, p.models[0], "grads_out");
    auto grad = pool_required(gs, 2, p.models[1], "grads_out");
    auto grad_wavelet = pool_required(gs, 0, p.forward_source, "grads_out");
    run_recursive_imaging(p, &grad, &grad_wavelet, nullptr);
    out.grads = {grad_wavelet, grad_vp, grad};
    return out;
}

} // namespace

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    SWEEP_CHECK(in.models.size() == 2, "Acoustic LSRTM 3D backward expects two models.");
    return backward_ckpt_imaging_impl(in);
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    SWEEP_CHECK(in.models.size() == 2, "Acoustic LSRTM 3D backward expects two models.");
    return backward_recursive_imaging_impl(in);
}






}

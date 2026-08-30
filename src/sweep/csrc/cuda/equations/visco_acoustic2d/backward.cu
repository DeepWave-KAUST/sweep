#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "visco_acoustic2d.h"
#include "kernels.cuh"
#include "../acoustic2d/kernels.cuh"   // reused fused adjoint + grad kernels (ODR-safe)
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace visco_acoustic2d {

namespace {

// ---------------------------------------------------------------------------
// Spectral terms (Zhu & Harris 2014, decoupled) — adjoint machinery.
// The ViscoSpectral bundle (kernels.cuh) carries the damping filter and the
// fractional-Laplacian dispersion remainder.
//
// Forward, step j (buffers):  u_next -= Gp ⊙ L(u_now - u_prev),  Gp = dt*A,
// followed by the halo zeroing Z.  With λ_j := adjoint of the fully-updated
// u_next of step j, the transpose across the buffer rotation contributes to
// the adjoint recursion at reverse iteration it (u_now = λ_{it+1},
// u_prev = λ_{it+2} BEFORE the fused kernel's swap):
//     λ_it += Z( L(Gp ⊙ (λ_{it+2} - λ_{it+1})) )
// L is self-adjoint (real, even |k| multiplier); Z^T = Z gates every
// accumulation into λ, preserving the "λ halo == 0" invariant the reused
// stencil kernels rely on.
// ---------------------------------------------------------------------------
// The spectral terms of the adjoint recursion; call between the fused adjoint
// kernel (which wrote u_next = λ_it^{S^T}) and swap_aux().  Damping reads the
// lag pair (λ_{it+2} - λ_{it+1}); the dispersion term is memoryless in u_now,
// so its transpose reads λ_{it+1} alone (adj.u_now_t) through the SAME
// self-adjoint operators with the coefficient maps moved inside.
void adjoint_damping_extra(AcousticWavefieldTensor& adj, const ViscoSpectral& d, int M)
{
    if (!(d.active || d.disp)) return;
    torch::Tensor m;
    if (d.active)
        m = visco_acoustic2d_Lop(d.Gp * (adj.u_prev_t - adj.u_now_t), d.kmul);
    if (d.disp) {
        auto e = visco_acoustic2d_Lop(d.Gd1 * adj.u_now_t, d.Dk2)
               - visco_acoustic2d_Lop(d.Gd2 * adj.u_now_t, d.Dfrac);
        m = d.active ? m + e : e;
    }
    visco_acoustic2d_zero_halo(m, M);
    adj.u_next_t.add_(m);
}

// grad_A += -dt^2 * λ_it ⊙ L((u[it] - u[it-1]) / dt)
//         = -dt   * λ_it ⊙ L(du),  du = u[it] - u[it-1].
// ``lam`` is the post-swap adjoint (λ_it, halo == 0 so the halo band of the
// filtered field drops out automatically).
void accumulate_grad_A(torch::Tensor& grad_A,
                       const torch::Tensor& lam,
                       const torch::Tensor& du,
                       const ViscoSpectral& d, float dt)
{
    if (!d.active) return;
    auto Ld = visco_acoustic2d_Lop(du.view(lam.sizes()), d.kmul);
    grad_A.add_(lam * Ld, -static_cast<double>(dt));
}

// grad_B1 += dt^2 * λ ⊙ L_{D_k2}(u[it]);  grad_B2 -= dt^2 * λ ⊙ L_{D_frac}(u[it]).
// Same (λ, u[it]) index pairing as grad_A's u_now slot; valid from it = 0
// (the dispersion term needs no u[it-1]).
void accumulate_grad_disp(torch::Tensor* grad_B1, torch::Tensor* grad_B2,
                          const torch::Tensor& lam,
                          const torch::Tensor& u_it,
                          const ViscoSpectral& d, float dt)
{
    if (!d.disp || grad_B1 == nullptr) return;
    auto uv = u_it.view(lam.sizes());
    const double dt2 = static_cast<double>(dt) * static_cast<double>(dt);
    grad_B1->add_(lam * visco_acoustic2d_Lop(uv, d.Dk2), dt2);
    grad_B2->add_(lam * visco_acoustic2d_Lop(uv, d.Dfrac), -dt2);
}

void init_rtm_output_visco_2d(RTMOutput& out, const torch::Tensor& vp)
{
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
}

// One fused exact-adjoint launch (reused acoustic2d kernel; the damping term
// is layered on top by adjoint_damping_extra).
inline void run_visco2d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const float* vp_ptr,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    TORCH_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
        "fused adjoint needs the adjoint wavefield bound with psi+zeta "
        "double-buffer (11 tensors in 2D); set cuda_layout.adjoint_extra_nvar=2.");
    ACOUSTIC2D_ADJOINT_FUSED(order, grid, block,
        adj_view, vp_ptr, lap_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx,
        adj_view.psixn, adj_view.psizn, adj_view.zetaxn, adj_view.zetazn,
        nullptr, nullptr);
}

// Imaging for one reverse step: recompute the vp_step-gradient carrier
// (vp^2 * Lap u) from the RAW pressure, then reuse the shared acoustic
// calculate_grad / accumulate_rtm_image_2d kernels.
void image_step_from_raw(
    int order, dim3 grid, dim3 block,
    const float* u_raw_ptr,
    const float* lam_ptr,
    const torch::Tensor& vp,
    torch::Tensor& carrier,   // (B, nz, nx) scratch, halo stays 0
    torch::Tensor* grad,
    RTMOutput* rtm_out,
    const LaplaceParam& lap_ctx,
    const SolverContext& ctx,
    int nx, int nz, float dt)
{
    if (grad == nullptr && rtm_out == nullptr) return;
    VISCO_ACOUSTIC2D_CARRIER(order, grid, block,
        u_raw_ptr, vp.data_ptr<float>(), carrier.data_ptr<float>(),
        lap_ctx, ctx);
    if (grad != nullptr) {
        calculate_grad<<<grid, block>>>(
            carrier.data_ptr<float>(), lam_ptr,
            vp.data_ptr<float>(), grad->data_ptr<float>(),
            nx, nz, dt);
    }
    if (rtm_out != nullptr) {
        accumulate_rtm_image_2d<<<grid, block>>>(
            carrier.data_ptr<float>(), lam_ptr,
            rtm_out->image.data_ptr<float>(),
            rtm_out->source_illumination.data_ptr<float>(),
            rtm_out->receiver_illumination.data_ptr<float>(),
            nx, nz);
    }
}

void check_visco_backward(const BackwardInput& p)
{
    TORCH_CHECK(p.models.size() == 4,
                "visco_acoustic2d expects the prepared models "
                "(vp_step, B1, B2, A); got ", p.models.size());
    TORCH_CHECK(!p.bw_stepped() && p.grads_out.empty() && p.step_phase == 0,
                "visco_acoustic2d does not support stepped backward segments");
    TORCH_CHECK(p.cut_face_mask == 0,
                "visco_acoustic2d does not support domain decomposition");
    TORCH_CHECK(!p.has_topo && !p.use_apm,
                "visco_acoustic2d does not support topography on impl='c' yet; "
                "use impl='eager'");
    TORCH_CHECK(!p.compute_adcig,
                "visco_acoustic2d does not support ADCIG yet");
}

// Full-storage reverse sweep, shared by backward() (grads) and rtm() (image).
void run_full_imaging_visco(
    const BackwardInput& p,
    torch::Tensor* grad,
    torch::Tensor* grad_A,
    torch::Tensor* grad_B1,
    torch::Tensor* grad_B2,
    torch::Tensor* grad_wavelet,
    RTMOutput* rtm_out)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    int M = p.M;
    float dt = p.dt;

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    ctx.set_cut_mask(0);
    acoustic_init_aux_slabs(ctx, adjoint);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    ViscoSpectral damping = visco_acoustic2d_make_spectral(p.eq_aux, p.models, dt, nz, nx);
    auto carrier = torch::zeros({B, nz, nx}, vp.options());

    for (int it = p.nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        run_visco2d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        adjoint_damping_extra(adjoint, damping, M);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        if (grad_wavelet != nullptr) {
            accumulate_source_grad_2d<<<forward_source_config.grid,
                                        forward_source_config.block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grad_wavelet->data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );
        }

        image_step_from_raw(
            order, launch_config.grid, launch_config.block,
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp, carrier, grad, rtm_out,
            lap_ctx, ctx, nx, nz, dt);

        if (grad_A != nullptr && damping.active && it >= 1) {
            accumulate_grad_A(*grad_A, adjoint.u_now_t,
                              p.u_forward[it] - p.u_forward[it - 1],
                              damping, dt);
        }
        accumulate_grad_disp(grad_B1, grad_B2, adjoint.u_now_t,
                             p.u_forward[it], damping, dt);
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_visco_backward(in);
    TORCH_CHECK(in.u_forward.defined() && in.u_forward.numel() > 0,
                "visco_acoustic2d backward (full) requires the raw forward "
                "wavefield history");
    BackwardOutput out;
    auto grad = torch::zeros_like(in.models[0]);
    auto grad_A = torch::zeros_like(in.models[3]);
    auto grad_B1 = torch::zeros_like(in.models[1]);
    auto grad_B2 = torch::zeros_like(in.models[2]);
    auto grad_wavelet = torch::zeros_like(in.forward_source);
    RTMOutput illumination;
    init_rtm_output_visco_2d(illumination, in.models[0]);
    run_full_imaging_visco(in, &grad, &grad_A, &grad_B1, &grad_B2,
                           &grad_wavelet,
                           in.compute_illumination ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad, grad_B1, grad_B2, grad_A};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

RTMOutput rtm(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_visco_backward(in);
    TORCH_CHECK(
        in.u_forward.defined() && in.u_forward.numel() > 0,
        "visco_acoustic2d RTM requires full forward wavefields (raw pressure)."
    );
    TORCH_CHECK(
        in.checkpoints.empty(),
        "visco_acoustic2d RTM does not support checkpoint mode."
    );

    RTMOutput out;
    init_rtm_output_visco_2d(out, in.models[0]);
    run_full_imaging_visco(in, nullptr, nullptr, nullptr, nullptr, nullptr, &out);
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    TORCH_CHECK(false,
        "visco_acoustic2d does not support boundary saving: the amplitude "
        "damping is dissipative (reverse-time reconstruction amplifies) and "
        "global (the |k| filter reads the whole padded grid, which boundary "
        "strips cannot restore).  Use memory=MemoryOptions(strategy='ckpt') "
        "or strategy='full'.");
    return {};
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_visco_backward(in);

    const auto& p = in;
    BackwardOutput out;

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "visco_acoustic2d"
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

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, true);
    else
        forward.allocate_from_snapshots(vp, p.checkpoints, 2);
    // Slab geometry follows the FORWARD-state aux layout (the recompute runs
    // the forward kernel); the adjoint aux stays full-domain.
    acoustic_init_aux_slabs(ctx, forward);

    auto grad = torch::zeros_like(vp);
    auto grad_A = torch::zeros_like(p.models[3]);
    auto grad_B1 = torch::zeros_like(p.models[1]);
    auto grad_B2 = torch::zeros_like(p.models[2]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_visco_2d(illumination, vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    ViscoSpectral damping = visco_acoustic2d_make_spectral(p.eq_aux, p.models, dt, nz, nx);
    auto carrier = torch::zeros({B, nz, nx}, vp.options());
    RTMOutput* rtm_out = in.compute_illumination ? &illumination : nullptr;

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    // RAW pressure store for the chunk (the acoustic twin stores the
    // vp^2*Lap(u) carrier; visco recomputes it from raw — see kernels.cuh).
    auto chunk_raw = torch::zeros({chunk_size, B, nz, nx}, vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

        // u(start-1): the loaded state is (u_prev, u_now) = (u(start-1), u(start));
        // the it == start reverse step needs it for du/dt of step ``start``.
        torch::Tensor u_prev_chunk;
        if (damping.active)
            u_prev_chunk = forward.u_prev_t.reshape({B, nz, nx}).clone();

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();

            chunk_raw[it - start].copy_(forward.u_now_t.view({B, nz, nx}));

            ACOUSTIC2D(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
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

            visco_acoustic2d_apply_spectral(forward, damping, dt, M);

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

            run_visco2d_adjoint_step(
                order, launch_config.grid, launch_config.block,
                adj_view, vp.data_ptr<float>(),
                lap_ctx, grad_ctx_x, grad_ctx_z,
                cpml, ctx);

            adjoint_damping_extra(adjoint, damping, M);

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_aux();

            accumulate_source_grad_2d<<<fwd_source_config.grid,
                                        fwd_source_config.block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grad_wavelet.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );

            image_step_from_raw(
                order, launch_config.grid, launch_config.block,
                chunk_raw[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp, carrier, &grad, rtm_out,
                lap_ctx, ctx, nx, nz, dt);

            if (damping.active && it >= 1) {
                accumulate_grad_A(grad_A, adjoint.u_now_t,
                                  chunk_raw[it - start] -
                                      (it > start ? chunk_raw[it - start - 1]
                                                  : u_prev_chunk),
                                  damping, dt);
            }
            accumulate_grad_disp(&grad_B1, &grad_B2,
                                 adjoint.u_now_t, chunk_raw[it - start],
                                 damping, dt);
        }
    }

    out.grads = {grad_wavelet, grad, grad_B1, grad_B2, grad_A};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

} // namespace visco_acoustic2d

// ---------------------------------------------------------------------------
// Recursive (binary) checkpointing.  Mirrors acoustic2d's driver; the leaf
// replays one visco forward step (stencil + damping) from the interval state,
// recomputes the carrier from the raw pressure, and layers the damping terms
// onto the fused adjoint.  The checkpoint state is the same 6-tensor acoustic
// state — the amplitude damping is memoryless in (u_now, u_prev).
// ---------------------------------------------------------------------------
namespace visco_acoustic2d {
namespace {

int visco_recursive_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

void advance_forward_interval_visco_2d(
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
    int forward_nsrc,
    const ViscoSpectral& damping)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();

        ACOUSTIC2D(
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

        visco_acoustic2d_apply_spectral(forward, damping, ctx.dt, ctx.M);

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

void process_recursive_interval_visco_2d(
    int start,
    int end,
    AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInput& p,
    const torch::Tensor& vp,
    torch::Tensor* grad,
    torch::Tensor* grad_A,
    torch::Tensor* grad_B1,
    torch::Tensor* grad_B2,
    torch::Tensor* grad_wavelet,
    RTMOutput* rtm_out,
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
    int scratch_depth,
    torch::Tensor& carrier_scratch,
    const ViscoSpectral& damping,
    int nx,
    int nz)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        // Pre-step captures: the state still holds (u_prev, u_now) =
        // (u(start-1), u(start)).
        torch::Tensor du;
        if (damping.active && start >= 1)
            du = start_state.u_now_t - start_state.u_prev_t;
        // Dispersion gradient bases from the pre-step u(start) (the state
        // rotates before the accumulation point below).
        torch::Tensor Pb, Rb;
        if (damping.disp && grad_B1 != nullptr) {
            Pb = visco_acoustic2d_Lop(start_state.u_now_t, damping.Dk2);
            Rb = visco_acoustic2d_Lop(start_state.u_now_t, damping.Dfrac);
        }
        VISCO_ACOUSTIC2D_CARRIER(order, wave_grid, wave_block,
            start_state.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            carrier_scratch.data_ptr<float>(),
            lap_ctx, ctx);

        auto fwd_view = start_state.view();

        ACOUSTIC2D(
            order,
            wave_grid,
            wave_block,
            fwd_view,
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

        visco_acoustic2d_apply_spectral(start_state, damping, ctx.dt, ctx.M);

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

        run_visco2d_adjoint_step(
            order, wave_grid, wave_block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        adjoint_damping_extra(adjoint, damping, ctx.M);

        add_source<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();

        if (grad_wavelet != nullptr) {
            accumulate_source_grad_2d<<<forward_source_grid, forward_source_block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grad_wavelet->data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                start,
                forward_nsrc,
                ctx
            );
        }

        if (grad != nullptr) {
            calculate_grad<<<wave_grid, wave_block>>>(
                carrier_scratch.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad->data_ptr<float>(),
                nx, nz, ctx.dt);
        }
        if (rtm_out != nullptr) {
            accumulate_rtm_image_2d<<<wave_grid, wave_block>>>(
                carrier_scratch.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                rtm_out->image.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                nx, nz);
        }

        if (grad_A != nullptr && damping.active && start >= 1)
            accumulate_grad_A(*grad_A, adjoint.u_now_t, du, damping, ctx.dt);
        if (damping.disp && grad_B1 != nullptr) {
            const double dt2 = static_cast<double>(ctx.dt) * static_cast<double>(ctx.dt);
            grad_B1->add_(adjoint.u_now_t * Pb, dt2);
            grad_B2->add_(adjoint.u_now_t * Rb, -dt2);
        }
        return;
    }

    int mid = start + (end - start) / 2;

    TORCH_CHECK(
        scratch_depth < static_cast<int>(scratch_states.size()),
        "Recursive checkpoint scratch depth exhausted."
    );
    AcousticWavefieldTensor& mid_state = scratch_states[scratch_depth];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
    advance_forward_interval_visco_2d(
        mid_state, start, mid, order,
        wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        p, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, damping);

    process_recursive_interval_visco_2d(
        mid, end, mid_state, adjoint, p, vp,
        grad, grad_A, grad_B1, grad_B2, grad_wavelet, rtm_out,
        order, wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        adj_source_grid, adj_source_block,
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, adjoint_nsrc,
        checkpoint_runtime, scratch_states, scratch_depth + 1,
        carrier_scratch, damping, nx, nz);

    process_recursive_interval_visco_2d(
        start, mid, start_state, adjoint, p, vp,
        grad, grad_A, grad_B1, grad_B2, grad_wavelet, rtm_out,
        order, wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        adj_source_grid, adj_source_block,
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, adjoint_nsrc,
        checkpoint_runtime, scratch_states, scratch_depth + 1,
        carrier_scratch, damping, nx, nz);
}

} // namespace

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_visco_backward(in);
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(
        p.checkpoints.size() == 6,
        "visco_acoustic2d recursive checkpointing expects 6 checkpoint tensors"
    );

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
        "visco_acoustic2d"
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

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    auto grad = torch::zeros_like(vp);
    auto grad_A = torch::zeros_like(p.models[3]);
    auto grad_B1 = torch::zeros_like(p.models[1]);
    auto grad_B2 = torch::zeros_like(p.models[2]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_visco_2d(illumination, vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    ViscoSpectral damping = visco_acoustic2d_make_spectral(p.eq_aux, p.models, dt, nz, nx);

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

    int max_segment_length = 0;
    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];
        max_segment_length = std::max(max_segment_length, end - start);
    }

    AcousticWavefieldTensor start_state;
    start_state.allocate_from_snapshots(vp, p.checkpoints, 2);
    acoustic_init_aux_slabs(ctx, start_state);

    std::vector<AcousticWavefieldTensor> scratch_states(visco_recursive_scratch_depth(max_segment_length));
    for (auto& scratch_state : scratch_states)
        scratch_state.allocate_like(vp, start_state);
    // zeros (not empty): the carrier kernel writes non-halo cells only and the
    // halo band must stay 0 for the reused grad/RTM kernels.
    auto carrier_scratch = torch::zeros_like(vp);

    RTMOutput* rtm_out = in.compute_illumination ? &illumination : nullptr;

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_visco_2d(
            start, end, start_state, adjoint, p, vp,
            &grad, &grad_A, &grad_B1, &grad_B2,
            &grad_wavelet, rtm_out,
            order, launch_config.grid, launch_config.block,
            fwd_source_config.grid, fwd_source_config.block,
            adj_source_config.grid, adj_source_config.block,
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx, forward_nsrc, adjoint_nsrc,
            checkpoint_runtime, scratch_states, 0,
            carrier_scratch, damping, nx, nz);
    }

    out.grads = {grad_wavelet, grad, grad_B1, grad_B2, grad_A};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

} // namespace visco_acoustic2d

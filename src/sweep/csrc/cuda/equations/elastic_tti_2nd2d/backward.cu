#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <array>

#include "elastic_tti_2nd2d.h"
#include "kernels.cuh"
#include "tensors.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace elastic_tti_2nd2d {

namespace {

struct AdjointWorkspace {
    std::array<torch::Tensor, 8> t;

    void init(const std::vector<torch::Tensor>& external, const torch::Tensor& like)
    {
        if (external.size() == 8) {
            for (int i = 0; i < 8; ++i) {
                t[i] = external[i];
                t[i].zero_();
            }
        } else {
            for (int i = 0; i < 8; ++i)
                t[i] = torch::zeros_like(like);
        }
    }

    float* q(int i) { return t[i].data_ptr<float>(); }
    float* pw(int i) { return t[4 + i].data_ptr<float>(); }
};

// One reverse sweep element: K1 (bar of the CPML-corrected divergence terms
// from lam_{t+1}) must run BEFORE the adjoint advance so the q workspace and
// the adjoint memory transposition see the source-completed lam_{t+1}.
void adjoint_k1(
    int order,
    const fdtd::LaunchConfig& launch_config,
    WavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SolverContext solver,
    AdjointWorkspace& ws
)
{
    TTI2ND_LAUNCH(tti2nd_adjoint_div_prepare, order,
        launch_config.grid, launch_config.block,
        adjoint.view(), model, cpml_view, solver,
        ws.q(0), ws.q(1), ws.q(2), ws.q(3));
}

void adjoint_advance(
    int order,
    const fdtd::LaunchConfig& launch_config,
    WavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    AdjointWorkspace& ws
)
{
    TTI2ND_LAUNCH(tti2nd_adjoint_strain_prepare, order,
        launch_config.grid, launch_config.block,
        adjoint.view(), model, cpml_view, grad_ctx, solver,
        ws.q(0), ws.q(1), ws.q(2), ws.q(3),
        ws.pw(0), ws.pw(1), ws.pw(2), ws.pw(3));

    TTI2ND_LAUNCH(tti2nd_adjoint_displacement_apply, order,
        launch_config.grid, launch_config.block,
        adjoint.view(),
        ws.pw(0), ws.pw(1), ws.pw(2), ws.pw(3),
        grad_ctx, solver);

    adjoint.swap_u();
}

void inject_adjoint_source(
    const BackwardInput& p,
    WavefieldTensor& adjoint,
    const torch::Tensor& receiver_fields,
    const fdtd::LaunchConfig& adj_source_config,
    SolverContext solver,
    int it
)
{
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nrec_fields = p.receiver_field_indices.numel();
    for (int irec = 0; irec < nrec_fields; ++irec) {
        const int fld = receiver_fields[irec].item<int>();
        float* field = nullptr;
        if (fld == 0) field = adjoint.ux_t.data_ptr<float>();
        else if (fld == 1) field = adjoint.uz_t.data_ptr<float>();
        if (field == nullptr) continue;
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            field,
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            solver
        );
    }
}

void rho_source_correction(
    const BackwardInput& p,
    WavefieldTensor& adjoint,
    const torch::Tensor& source_fields,
    torch::Tensor& grad_rho,
    const torch::Tensor& rho,
    const fdtd::LaunchConfig& fwd_source_config,
    SolverContext solver,
    int it
)
{
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
        const int fld = source_fields[isrc].item<int>();
        float* adj_field = nullptr;
        if (fld == 0) adj_field = adjoint.ux_t.data_ptr<float>();
        else if (fld == 1) adj_field = adjoint.uz_t.data_ptr<float>();
        if (adj_field == nullptr) continue;
        tti2nd_rho_grad_source_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            adj_field,
            rho.data_ptr<float>(),
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            solver
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_forward.defined(), "ElasticTTI2nd full backward expects saved forward wavefields");
    TORCH_CHECK(p.u_forward.dim() == 5 && p.u_forward.size(1) == 2,
                "ElasticTTI2nd full backward expects u_forward with shape (nt, 2, B, nz, nx)");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);
    for (auto& tsr : adjoint.state_tensors())
        tsr.zero_();

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    ws.init(p.adjoint_workspace, rho);
    auto zero_field = torch::zeros_like(rho);

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

        adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

        const float* ux_t = (it >= 1) ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* uz_t = (it >= 1) ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* ux_next = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* uz_next = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* ux_prev = (it >= 2) ? p.u_forward.select(0, it - 2).select(0, 0).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* uz_prev = (it >= 2) ? p.u_forward.select(0, it - 2).select(0, 1).data_ptr<float>() : zero_field.data_ptr<float>();

        TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
            launch_config.grid, launch_config.block,
            adjoint.view(), model, grad_view,
            ux_t, uz_t, ux_next, uz_next, ux_prev, uz_prev,
            ws.q(0), ws.q(1), ws.q(2), ws.q(3),
            grad_ctx, solver);

        rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

        if (it == 0)
            continue;

        adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd boundary-saving backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd boundary-saving backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_last_two.defined(), "ElasticTTI2nd boundary-saving backward expects last-two wavefield tensor");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);
    for (auto& tsr : adjoint.state_tensors())
        tsr.zero_();

    WavefieldTensor forward;
    forward.allocate(rho);
    // (storage, level): level 1 = W_nt goes to the pre slot (later time),
    // level 0 = W_{nt-1} becomes the current state — acoustic2d convention.
    forward.ux_pre_t.copy_(p.u_last_two.select(0, 0).select(0, 1));
    forward.ux_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.uz_pre_t.copy_(p.u_last_two.select(0, 1).select(0, 1));
    forward.uz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    ws.init(p.adjoint_workspace, rho);

    auto sxx_ws = torch::zeros_like(rho);
    auto szz_ws = torch::zeros_like(rho);
    auto sxz_ws = torch::zeros_like(rho);

    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 2, 2, solver, rho, save_width, 2,
            true, false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu,
            {}, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true, 2, 2, solver, rho, save_width, 2,
            true, true, 1,
            {}, p.boundary_gpu,
            {}, p.use_pinned_memory
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, rho);
    }
    auto bs = boundary_saver.view();

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

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

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        auto for_view = forward.view();

        inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

        adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

        // Time-reversed reconstruction: with (u_now = W_it, u_pre = W_{it+1})
        // the reversed leapfrog writes W_{it-1} - S_it into u_next; the ring
        // restore fixes the PML-adjacent band, then the source re-add
        // completes W_{it-1}.
        TTI2ND_LAUNCH(tti2nd_stress_kernel_nopml, order,
            launch_config.grid, launch_config.block,
            for_view, model, grad_ctx, solver,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>());

        TTI2ND_LAUNCH(tti2nd_displacement_kernel_nopml_rev, order,
            launch_config.grid, launch_config.block,
            for_view, model,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>(),
            grad_ctx, solver);

        float* rec_fields[2] = { for_view.ux_nxt, for_view.uz_nxt };
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                rec_fields[f],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                f == 0,
                f == 1
            );
        }

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        // W_t = u_now, W_{t+1} = u_pre (later time), W_{t-1} = u_next.
        TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
            launch_config.grid, launch_config.block,
            adjoint.view(), model, grad_view,
            for_view.ux, for_view.uz,
            for_view.ux_pre, for_view.uz_pre,
            for_view.ux_nxt, for_view.uz_nxt,
            ws.q(0), ws.q(1), ws.q(2), ws.q(3),
            grad_ctx, solver);

        rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

        adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);

        forward.swap_u();

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd checkpoint backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd checkpoint backward expects cpmls PML profiles");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 14, "ElasticTTI2nd checkpointing expects 14 checkpoint tensors");

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        14,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic_tti_2nd2d"
    );

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    WavefieldTensor start_state;
    start_state.allocate(rho);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    ws.init(p.adjoint_workspace, rho);

    auto sxx_ws = torch::zeros_like(rho);
    auto szz_ws = torch::zeros_like(rho);
    auto sxz_ws = torch::zeros_like(rho);

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

    const int chunk_size = p.checkpoint_interval;
    const int num_chunks = (static_cast<int>(p.nt) + chunk_size - 1) / chunk_size;

    WavefieldTensor replay;
    if (!p.forward_wavefields.empty())
        replay.bind(p.forward_wavefields);
    else
        replay.allocate(rho);

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        const int start = chunk_id * chunk_size;
        const int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        const int seg_len = end - start;

        if (chunk_id == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(chunk_id, start_state.checkpoint_tensors());

        checkpoint_runtime.copy_state(replay.state_tensors(), start_state.state_tensors());

        // seg[k] = W_{start-1+k}: two history levels + one entry per replayed
        // step, so the reverse pass below has all three time slices in-chunk.
        std::vector<int64_t> seg_shape = rho.sizes().vec();
        seg_shape.insert(seg_shape.begin(), static_cast<int64_t>(seg_len + 2));
        auto seg_ux = torch::zeros(seg_shape, rho.options());
        auto seg_uz = torch::zeros(seg_shape, rho.options());
        seg_ux.select(0, 0).copy_(replay.ux_pre_t);
        seg_uz.select(0, 0).copy_(replay.uz_pre_t);
        seg_ux.select(0, 1).copy_(replay.ux_t);
        seg_uz.select(0, 1).copy_(replay.uz_t);

        for (int it = start; it < end; ++it) {
            auto rep_view = replay.view();

            TTI2ND_LAUNCH(tti2nd_stress_kernel, order,
                launch_config.grid, launch_config.block,
                rep_view, model, cpml_view, grad_ctx, solver,
                sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>());

            TTI2ND_LAUNCH(tti2nd_displacement_kernel, order,
                launch_config.grid, launch_config.block,
                rep_view, model,
                sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>(),
                nullptr, cpml_view, grad_ctx, solver);

            for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
                float* field = field_ptr(rep_view, source_fields[isrc].item<int>());
                if (field == nullptr) continue;
                add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    field,
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it,
                    forward_nsrc,
                    solver
                );
            }

            seg_ux.select(0, it - start + 2).copy_(replay.ux_nxt_t);
            seg_uz.select(0, it - start + 2).copy_(replay.uz_nxt_t);

            replay.swap_u();
        }

        for (int it = end - 1; it >= start; --it) {
            inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

            adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

            const int k = it - start;
            TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
                launch_config.grid, launch_config.block,
                adjoint.view(), model, grad_view,
                seg_ux.select(0, k + 1).data_ptr<float>(),
                seg_uz.select(0, k + 1).data_ptr<float>(),
                seg_ux.select(0, k + 2).data_ptr<float>(),
                seg_uz.select(0, k + 2).data_ptr<float>(),
                seg_ux.select(0, k).data_ptr<float>(),
                seg_uz.select(0, k).data_ptr<float>(),
                ws.q(0), ws.q(1), ws.q(2), ws.q(3),
                grad_ctx, solver);

            rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

            if (it == 0)
                continue;

            adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);
        }
    }

    out.grads = grads;
    return out;
}

} // namespace elastic_tti_2nd2d

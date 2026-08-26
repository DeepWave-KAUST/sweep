#include <torch/extension.h>
#include <cuda_runtime.h>


#include <c10/cuda/CUDAGuard.h>
#include "acoustic_vrz3d.h"
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
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

namespace acoustic_vrz3d {

ForwardOutput forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ3D CUDA forward expects models [vp, z]");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = torch::reciprocal(z);

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;
    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, dy, dz};
    // DD cut-aware boundary: a cut face's boundary buffer is numel-0 (cut-aware
    // gpu_full_shapes) and its halo is reconstructed via NCCL exchange, so the
    // boundary save/restore must SKIP it.  The FP/int8 boundary kernels gate each
    // face on ctx.cut_* — but only if cut_mask is propagated here (acoustic3d does
    // the same).  Missing this makes VRZ write cut faces' null buffers => illegal
    // memory access under DD (PY*/PX* tiles with cut faces).
    ctx.set_cut_mask(p.cut_face_mask);

    // Stepped execution: run [it_begin, it_end) instead of [0, nt) so a DD
    // driver can halo-exchange between single steps.  All per-step indexing
    // stays absolute, so consecutive segments reproduce one full run.
    const int it0 = p.it_begin;
    const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
    TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                "AcousticVRZ3D stepped forward: require 0 <= it_begin <= it_end "
                "<= nt, got [", it0, ", ", it1, ") with nt=", p.nt);
    const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));

    // ---- DD phase-split step (comm/compute overlap) ----
    // Mirrors acoustic3d/forward.cu.  phase 0 = legacy full [0,nx) launch;
    // phase 1 = ONLY the M-wide cut-boundary strips the halo exchange sends;
    // phase 2 = the strict interior complement + source/record/save/swap.  The
    // union of the phase-1 and phase-2 x-ranges equals the legacy [0,nx) launch
    // with NO overlap, and each cell's per-cell physics is independent of the
    // launch range (reads u_now/psi/zeta at its own + neighbour cells, writes
    // u_next/psin at its own cell), so the split is a pure reordering =>
    // bit-identical to phase 0.  Source injection happens in phase 2, hence the
    // driver's ``_src_away_from_cuts`` guard.
    const int phase = p.step_phase;
    const bool cut_x_lo = (p.cut_face_mask & 1) != 0;
    const bool cut_x_hi = (p.cut_face_mask & 2) != 0;
    if (phase != 0) {
        TORCH_CHECK(phase == 1 || phase == 2,
                    "step_phase must be 0 (legacy), 1 (boundary strips) or 2 (interior)");
        TORCH_CHECK(it1 == it0 + 1,
                    "phased forward (step_phase != 0) drives a single step: "
                    "require it_end == it_begin + 1, got [", it0, ", ", it1, ")");
        TORCH_CHECK(p.cut_face_mask != 0,
                    "phased forward requires cut_face_mask != 0");
        TORCH_CHECK((p.cut_face_mask & ~0x3) == 0,
                    "phased forward v1 supports x-face cuts only (bits 0/1), got ",
                    p.cut_face_mask);
        TORCH_CHECK(ctx.phys_x1() - ctx.phys_x0() >= 2 * p.M,
                    "tile too narrow for phase-split strips: nx_phys=",
                    ctx.phys_x1() - ctx.phys_x0(), " < 2M=", 2 * p.M);
    }

    AcousticWavefieldTensor wavefield;
    // A continuation segment (it_begin>0) must keep the same wavefield tensors;
    // the internal allocate() would zero the propagation state mid-run.
    TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                "AcousticVRZ3D stepped continuation (it_begin>0) requires "
                "Python-bound wavefields");
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 3, true);
    else
        wavefield.allocate(vp, 3, true, /*double_buffer_psi=*/true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    // Stepped runs accumulate absolute-it writes into a Python-bound record;
    // a per-call allocation would lose every prior segment.
    TORCH_CHECK(!stepped || p.record_out.defined(),
                "AcousticVRZ3D stepped forward requires record_out bound from Python");
    auto record = p.record_out.defined()
        ? p.record_out
        : torch::zeros({N, nrec, p.nt}, vp.options());
    if (p.record_out.defined())
        TORCH_CHECK(record.is_contiguous() &&
                    record.size(-1) == static_cast<long>(p.nt),
                    "record_out must be contiguous with trailing dim nt");

    // save_all_wavefields keeps a fresh per-call buffer; it is a full-run-only
    // debug path (DD uses boundary saving), so forbid it under stepped where
    // segments would clobber each other.
    TORCH_CHECK(!stepped || !p.save_all_wavefields,
                "AcousticVRZ3D stepped forward does not support save_all_wavefields");
    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 7, B, nz, ny, nx}, vp.options());

    if (p.use_checkpoint)
        TORCH_CHECK(p.checkpoints.size() == 8, "AcousticVRZ3D checkpointing expects 8 checkpoint tensors");
    if (p.use_recursive_checkpoint) {
        TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
        TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
    }

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    // The internal full-storage fallback ring is per-call; segments after the
    // first would lose everything saved before them, so stepped runs need the
    // Python-bound persistent boundary buffers.
    if (stepped && p.use_boundary_saving)
        TORCH_CHECK(!p.boundary_gpu.empty(),
                    "AcousticVRZ3D stepped forward with boundary saving requires "
                    "Python-bound boundary_gpu");
    // tangent_pad MUST match the Python boundary layout (equations/acoustic_vrz.py
    // sets boundary_tangent_pad = so // 2 = M for VRZ; acoustic leaves it 0).  The
    // Python-owned persistent int8 buffers are sized with this pad, so the FP32
    // staging (allocated internally by the saver) must use the same pad or its
    // per-step size falls short of top_t.stride(0) and quantize_step reads past it
    // => illegal memory access (surfaced under DD / large grids; harmless-looking
    // over-read into adjacent allocations on small single-GPU runs).
    const int boundary_tangent_pad = p.M;
    if (staged_boundary) {
        boundary_saver.allocate(p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory,
                                boundary_tangent_pad);
    } else {
        boundary_saver.allocate(p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory,
                                boundary_tangent_pad);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
        p.use_boundary_saving,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "acoustic_vrz3d",
        it0
    );

    for (int it = it0; it < it1; ++it) {
        auto view = wavefield.view();

        // Ranged x stencil launch over [xb, xe); (0, nx) reproduces the legacy
        // full launch bit-identically (same block dims, x_base = 0, x_end = nx).
        // Only the launch RANGE changes per phase — each cell's in_pml branch,
        // aux writes and time update are identical to the full launch.
        auto launch_stencil = [&](int xb, int xe) {
            if (xe <= xb) return;
            SolverContext sctx = ctx;
            sctx.x_base  = xb;
            sctx.x_limit = xe;
            auto lc = fdtd::Wave3D::make(xe - xb, ny, nz, B);
            ACOUSTIC_VRZ3D(
                order,
                lc.grid,
                lc.block,
                view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_y,
                grad_ctx_z,
                cpml,
                sctx
            );
        };

        if (phase == 1) {
            // Boundary phase: ONLY the cut-adjacent M-wide physical edge strips
            // — exactly what the halo exchange sends to the neighbour.
            if (cut_x_lo) launch_stencil(ctx.phys_x0(), ctx.phys_x0() + p.M);
            if (cut_x_hi) launch_stencil(ctx.phys_x1() - p.M, ctx.phys_x1());
        } else if (phase == 2) {
            // Interior phase: strict complement of the phase-1 strips (no
            // overlap — re-running a strip cell would double-write its psi
            // double-buffer / advance zeta twice).
            launch_stencil(cut_x_lo ? ctx.phys_x0() + p.M : 0,
                           cut_x_hi ? ctx.phys_x1() - p.M : nx);
        } else {
            launch_stencil(0, nx);
        }

        if (phase == 1)
            continue;   // no boundary saving / source / record / swap for phase 1

        if (p.use_boundary_saving) {
            boundary_runtime.save_forward_3d(
                it,
                p.nt,
                view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                boundary_offset,
                ctx
            );
        }

        add_source_3d<<<source_config.grid, source_config.block>>>(
            view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel_3d<<<record_config.grid, record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer

        if (u_allt.defined()) {
            u_allt.select(0, it).select(0, 0).copy_(wavefield.u_now_t.squeeze(1));
            u_allt.select(0, it).select(0, 1).copy_(wavefield.psix_t.squeeze(1));
            u_allt.select(0, it).select(0, 2).copy_(wavefield.psiy_t.squeeze(1));
            u_allt.select(0, it).select(0, 3).copy_(wavefield.psiz_t.squeeze(1));
            u_allt.select(0, it).select(0, 4).copy_(wavefield.zetax_t.squeeze(1));
            u_allt.select(0, it).select(0, 5).copy_(wavefield.zetay_t.squeeze(1));
            u_allt.select(0, it).select(0, 6).copy_(wavefield.zetaz_t.squeeze(1));
        }

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt), wavefield.checkpoint_tensors());
    }

    // last_two seeds the backward; only meaningful once the final segment has
    // produced u at nt-1/nt.  Mid-run segments leave it untouched.  Phase 1 has
    // not swapped yet (u_prev/u_now roles would be wrong) — phase 2 of the same
    // step does this copy.
    if (p.use_boundary_saving && it1 == static_cast<int>(p.nt) && phase != 1) {
        boundary_saver.last_two_t.select(1, 0).copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t.select(1, 1).copy_(wavefield.u_now_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace acoustic_vrz3d

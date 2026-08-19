#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>

#include <c10/cuda/CUDAGuard.h>
#include "acoustic2d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"

namespace acoustic2d {

ForwardOutput forward(const ForwardInput& in) {
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx;
    ctx.ndim         = 2;
    ctx.nx           = nx;
    ctx.ny           = 0;
    ctx.nz           = nz;
    ctx.B            = B;
    ctx.dt           = p.dt;
    ctx.nt           = p.nt;
    ctx.M            = p.M;
    ctx.abcn         = p.abcn;
    ctx.free_surface = p.free_surface;
    ctx.lap_coeff    = p.lap_coes.data_ptr<float>();
    ctx.grad_coeff   = p.grad_coes.data_ptr<float>();
    ctx.dx           = dx;
    ctx.dy           = 0.f;
    ctx.dz           = dz;
    ctx.topo_rows    = p.has_topo ? p.topo_rows.data_ptr<int>() : nullptr;
    ctx.has_topo     = p.has_topo;
    ctx.topo_category = nullptr;
    ctx.use_apm      = false;
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    // Cut-aware physical bounds (0 = single domain → legacy per-edge pad + M).
    ctx.cut_mask     = p.cut_face_mask;

    const int it0 = p.it_begin;
    const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
    TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                "stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                it0, ", ", it1, ") with nt=", p.nt);
    const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));

    // ---- DD phase-split step (comm/compute overlap) ----
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
    // On a continuation call the internal allocate() would silently zero the
    // propagation state — the caller must keep binding the same tensors.
    TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                "stepped continuation (it_begin>0) requires Python-bound wavefields");
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 2, true);
    else
        wavefield.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    acoustic_init_aux_slabs(ctx, wavefield);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    TORCH_CHECK(!stepped || p.record_out.defined(),
                "stepped forward requires record_out bound from Python");
    auto record = p.record_out.defined()
        ? p.record_out
        : torch::zeros({N, p.receivers_loc.size(1), p.nt}, vp.options());
    if (p.record_out.defined())
        TORCH_CHECK(record.is_contiguous() &&
                    record.size(-1) == static_cast<long>(p.nt),
                    "record_out must be contiguous with trailing dim nt");

    // Wavefields for all timestep
    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        TORCH_CHECK(!stepped || p.u_allt_out.defined(),
                    "stepped + save_all_wavefields requires u_allt_out bound from Python");
        u_allt = p.u_allt_out.defined()
            ? p.u_allt_out
            : torch::zeros({p.nt, B, nz, nx}, vp.options());
    }

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "acoustic2d",
        it0
    );

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    // The internal full-storage fallback ring is per-call; segments after the
    // first would lose everything saved before them.
    if (stepped && p.use_boundary_saving)
        TORCH_CHECK(!p.boundary_gpu.empty(),
                    "stepped forward with boundary saving requires Python-bound boundary_gpu");
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    else
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, true, 1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    float* u_thist = nullptr;

    LaplaceParam lap_ctx{nx, 1, p.M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};
    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
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
    for (int it = it0; it < it1; ++it) {

        auto view = wavefield.view();

        u_thist = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        // Ranged stencil launch over x in [xb, xe); (0, nx) reproduces the
        // legacy full launch bit-identically (same grid dims, x_base = 0).
        // Pre-pass: clear air cells in a separate kernel launch so the
        // main acoustic2nd kernel only reads (never writes) air cells.
        // Eliminates intra-launch RAW race on PML aux fields that was
        // showing up as ~30% non-deterministic forward output across
        // processes (sweep VTI history pattern).  The air-clear range is
        // widened by the stencil halo M so a phase-split stencil launch
        // still only reads air cells cleared earlier THIS step (re-clearing
        // across phases writes the same zeros — idempotent).
        auto launch_stencil = [&](int xb, int xe) {
            if (xe <= xb) return;
            if (p.has_topo) {
                int axb = std::max(0, xb - p.M);
                int axe = std::min(nx, xe + p.M);
                SolverContext actx = ctx;
                actx.x_base = axb;
                actx.x_limit = axe;
                auto alc = fdtd::Wave2D::make(axe - axb, nz, B);
                acoustic2d_air_clear_kernel<<<alc.grid, alc.block>>>(
                    view, p.save_all_wavefields, u_thist, actx
                );
            }
            SolverContext sctx = ctx;
            sctx.x_base = xb;
            sctx.x_limit = xe;
            auto lc = fdtd::Wave2D::make(xe - xb, nz, B);
            ACOUSTIC2D(
                order,
                lc.grid,
                lc.block,
                view,
                p.save_all_wavefields,
                u_thist,
                vp.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_z,
                cpml,
                sctx
            );
        };

        if (phase == 1) {
            // Boundary phase: ONLY the cut-adjacent M-wide physical edge
            // strips — exactly what the halo exchange sends.
            if (cut_x_lo) launch_stencil(ctx.phys_x0(), ctx.phys_x0() + p.M);
            if (cut_x_hi) launch_stencil(ctx.phys_x1() - p.M, ctx.phys_x1());
        } else if (phase == 2) {
            // Interior phase: the strict complement of the phase-1 strips
            // (no overlap — re-running a strip cell would double-advance
            // its CPML psi double-buffer write).
            launch_stencil(cut_x_lo ? ctx.phys_x0() + p.M : 0,
                           cut_x_hi ? ctx.phys_x1() - p.M : nx);
        } else {
            launch_stencil(0, nx);
        }

        if (phase == 1)
            continue;   // no boundary saving / source / record / swap / ckpt

        if (p.use_boundary_saving) {
            boundary_runtime.save_forward_2d(
                it,
                p.nt,
                view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                0,
                ctx
            );
        }
        
        add_source<<<source_config.grid, source_config.block>>>(
            view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
        
        record_kernel<<<record_config.grid, record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt), wavefield.checkpoint_tensors());

    }

    // Save the last two time steps for backward (only once the final
    // segment has run; mid-run segments leave last_two untouched).
    // Phase 1 has not swapped yet — roles would be wrong; phase 2 of the
    // same step does the copy.
    if (p.use_boundary_saving && it1 == static_cast<int>(p.nt) && phase != 1) {
        boundary_saver.last_two_t.select(1,0).copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t.select(1,1).copy_(wavefield.u_now_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace acoustic2d

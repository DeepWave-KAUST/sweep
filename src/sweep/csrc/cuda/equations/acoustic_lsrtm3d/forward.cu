#include <cuda_runtime.h>


#include "acoustic_lsrtm3d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"

namespace acoustic_lsrtm3d {

namespace {

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

} // namespace

ForwardOutputCore forward_core(const ForwardInputCore& in) {
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    ForwardOutputCore out;

    SWEEP_CHECK(p.models.size() == 2, "Acoustic LSRTM 3D expects two models: vp and mp.");

    auto vp = p.models[0];
    auto mp = p.models[1];

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
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    // DD cut-aware: the CPML band (solver.in_pml_3d) shrinks to the M halo on a
    // cut face, and the boundary save kernels SKIP cut faces -- whose boundary
    // buffers are numel-0 under DD -- only if the mask reaches the context.
    ctx.set_cut_mask(p.cut_face_mask);

    // Stepped execution: run [it_begin, it_end) so a DD driver can exchange the
    // halos of BOTH coupled fields between single steps.  Indexing stays absolute,
    // so consecutive segments reproduce one full run; the wavefield list and
    // record_out are Python-owned (bound on every call), so state carries over.
    const int it0 = p.it_begin;
    const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
    SWEEP_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                "acoustic_lsrtm3d stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                it0, ", ", it1, ") with nt=", p.nt);
    // No phase split here: the LSRTM DD schedule is serial (step, then exchange),
    // so a phased call would silently run the whole step twice.
    SWEEP_CHECK(p.step_phase == 0,
                "acoustic_lsrtm3d forward does not implement step_phase (got ", p.step_phase,
                "); its DD schedule must be the serial one");

    // The propagator binds the forward wavefield state on EVERY call -- the
    // persistent save_all pool or the per-call transient set (_c.py
    // Wrapper.forward, ``params.wavefields = cp.forward_wavefields``, sized by
    // AcousticLSRTM3D.cuda_layout base_nvar 6 + pml_nvar 18) -- so the list is
    // never empty and there is no unbound caller left to allocate for.
    AcousticWavefieldTensor bg;
    AcousticWavefieldTensor sc;
    SWEEP_CHECK(p.wavefields.size() == 24,
                "acoustic_lsrtm3d/forward requires the propagator-bound wavefields "
                "(cuda_layout.base_nvar + pml_nvar = 24 tensors: bg+sc, each 12 with the "
                "psi double-buffer), got ", p.wavefields.size());
    bg.bind(slice_wavefields(p.wavefields, 0, 12), 3, true);
    sc.bind(slice_wavefields(p.wavefields, 12, 12), 3, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    // record_out is bound on every call: AcousticLSRTM3D.cuda_layout declares
    // record_shape, so cp.record_shape is never None and _c.py allocates it.
    auto record = bound_required(p.record_out, {N, nrec, p.nt},
                                 "record_out (acoustic_lsrtm3d/forward, cuda_layout.record_shape)");
    // u_allt_out is bound whenever save_all_wavefields is on: cuda_layout
    // declares save_all_shape, so cp.u_allt_shape is never None.
    // rwi: vp asks for its RWI tomographic gradient (cuda_layout_for_grads), read
    // off what the propagator bound -- the history then stacks [bg_utt, sc_utt]
    // per step (history_fields(2); sc_utt feeds term II) and bs saves both fields'
    // boundaries (last_two holds 2 fields).  Otherwise this is the reflectivity-
    // only forward of the classic LSRTM, exactly as before.
    const bool rwi = p.save_all_wavefields ? p.u_allt_out.dim() == 6
                   : (p.use_boundary_saving && p.last_two.size(0) == 2);
    Buf bg_utt_all;
    if (p.save_all_wavefields && rwi) {
        bg_utt_all = bound_required(p.u_allt_out, {p.nt, 2, B, nz, ny, nx},
                                    "u_allt_out (acoustic_lsrtm3d/forward, cuda_layout.save_all_shape)");
    } else if (p.save_all_wavefields) {
        bg_utt_all = bound_required(p.u_allt_out, {p.nt, B, nz, ny, nx},
                                    "u_allt_out (acoustic_lsrtm3d/forward, cuda_layout.save_all_shape)");
    }

    if (p.use_checkpoint) {
        SWEEP_CHECK(p.checkpoints.size() == 8, "Acoustic LSRTM 3D checkpointing expects 8 checkpoint tensors.");
    }
    if (p.use_recursive_checkpoint) {
        SWEEP_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps.");
        SWEEP_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D.");
    }

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, rwi ? 2 : 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, rwi ? 2 : 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
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
    const std::vector<std::string> disk_files = p.boundary_disk_files.vec();   // the runtime keeps a pointer to it
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
        p.use_boundary_saving,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        disk_files,
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
        "acoustic_lsrtm3d"
    );

    for (int it = it0; it < it1; ++it) {
        auto bg_view = bg.view();
        auto sc_view = sc.view();
        float* bg_utt_ptr = !bg_utt_all.defined() ? nullptr
            : (rwi ? bg_utt_all.select(0, it).select(0, 0) : bg_utt_all.select(0, it)).data_ptr<float>();
        float* sc_utt_ptr = (bg_utt_all.defined() && rwi) ? bg_utt_all.select(0, it).select(0, 1).data_ptr<float>() : nullptr;

        ACOUSTIC_LSRTM3D_COUPLED(
            order,
            launch_config.grid,
            launch_config.block,
            bg_view,
            sc_view,
            p.save_all_wavefields,
            bg_utt_ptr,
            sc_utt_ptr,
            vp.data_ptr<float>(),
            mp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving && rwi) {
            // field 0 = background, field 1 = scattered: the bs backward reconstructs both.
            float* bs_fields[2] = { bg_view.u_now, sc_view.u_now };
            for (int f = 0; f < 2; ++f) {
                boundary_runtime.save_forward_3d_field(
                    it, p.nt, bs_fields[f], launch_config.grid, launch_config.block,
                    bs, save_width, 0, ctx, f, /*flush_chunk=*/f == 1);
            }
        } else if (p.use_boundary_saving) {
            boundary_runtime.save_forward_3d(
                it, p.nt, bg_view.u_now, launch_config.grid, launch_config.block,
                bs, save_width, 0, ctx);
        }

        add_source_3d<<<source_config.grid, source_config.block>>>(
            bg_view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel_3d<<<record_config.grid, record_config.block>>>(
            sc_view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        bg.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer
        sc.swap_pml();

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt), bg.checkpoint_tensors());
    }

    // last_two seeds the backward reconstruction: only once the final segment ran.
    if (p.use_boundary_saving && it1 == static_cast<int>(p.nt)) {
        // [field, time, B, nz, ny, nx]; select(1, t) would broadcast bg into both fields.
        copy_tensor_cuda_async(boundary_saver.last_two.select(0, 0).select(0, 0), bg.u_prev_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0, 0).select(0, 1), bg.u_now_t);
        if (rwi) {
            copy_tensor_cuda_async(boundary_saver.last_two.select(0, 1).select(0, 0), sc.u_prev_t);
            copy_tensor_cuda_async(boundary_saver.last_two.select(0, 1).select(0, 1), sc.u_now_t);
        }
    }

    boundary_runtime.synchronize();

    out.wavefield = bg_utt_all.defined() ? bg_utt_all : Buf{};
    out.last_two = p.use_boundary_saving ? p.last_two : Buf{};   // the tensor Python bound
    out.record = record;
    return out;
}



} // namespace acoustic_lsrtm3d

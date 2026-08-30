#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include "visco_acoustic2d.h"
#include "kernels.cuh"
#include "../acoustic2d/kernels.cuh"   // reused CPML stencil (ODR-safe: same header)
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"

namespace visco_acoustic2d {

// Nearly constant-Q visco-acoustic forward (Zhu & Harris 2014, decoupled):
// the CPML acoustic step run with the dispersion-folded ``vp_step``
// (models[0]) plus a per-step spectral amplitude-damping correction
// (models[1] = A = tt*vp/2; active iff eq_aux = {|k| grid} is present).
//
// v1 scope (all guarded, never silent): no DD / stepped segments, no
// topography / APM, no boundary saving (the dissipative term breaks the
// reverse-time reconstruction; use ckpt / full).  Per-edge free surface is
// inherited from the acoustic2d region logic + the post-damping halo zeroing.
ForwardOutput forward(const ForwardInput& in) {
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 4,
                "visco_acoustic2d expects the prepared models "
                "(vp_step, B1, B2, A); got ", p.models.size());
    TORCH_CHECK(p.models[0].is_cuda(),
                "visco_acoustic2d impl='c' is CUDA-only; use impl='eager' on CPU");
    TORCH_CHECK(p.cut_face_mask == 0 && p.step_phase == 0,
                "visco_acoustic2d does not support domain decomposition (the "
                "amplitude damping is a global FFT)");
    TORCH_CHECK(p.it_begin == 0 && (p.it_end < 0 || p.it_end == (int)p.nt),
                "visco_acoustic2d does not support stepped forward segments");
    TORCH_CHECK(!p.has_topo && !p.use_apm,
                "visco_acoustic2d does not support topography on impl='c' yet; "
                "use impl='eager'");
    TORCH_CHECK(!p.use_boundary_saving,
                "visco_acoustic2d does not support boundary saving (dissipative "
                "step is not reverse-time reconstructible); use "
                "memory=MemoryOptions(strategy='ckpt') or 'full'");

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

    SolverContext ctx{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.topo_rows    = nullptr;
    ctx.has_topo     = false;
    ctx.topo_category = nullptr;
    ctx.use_apm      = false;
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    ctx.set_cut_mask(0);

    AcousticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 2, true);
    else
        wavefield.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    acoustic_init_aux_slabs(ctx, wavefield);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros(
        {N, p.receivers_loc.size(1), p.nt},
        vp.options()
    );

    // Full-mode store: RAW pressure u(t) (NOT the acoustic vp^2*Lap(u)
    // carrier) — the attenuation adjoint needs du/dt and its |k| filter; the
    // vp_step-gradient carrier is recomputed in backward (kernels.cuh).
    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, B, nz, nx}, vp.options());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "visco_acoustic2d",
        0
    );

    // Spectral terms (damping / NCQ dispersion): selected by the eq_aux
    // composition, see visco_acoustic2d_make_spectral (kernels.cuh).
    ViscoSpectral spectral =
        visco_acoustic2d_make_spectral(p.eq_aux, p.models, p.dt, nz, nx);

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    LaplaceParam lap_ctx{nx, 1, p.M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = 0; it < (int)p.nt; ++it) {

        auto view = wavefield.view();

        // Raw store BEFORE the step: u_now == u(t=it).
        if (u_allt.defined())
            u_allt[it].copy_(wavefield.u_now_t.view({B, nz, nx}));

        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
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

        // Dispersion + damping corrections; the FFTs write into the halo
        // bands, so the helper restores the stencil-kernel invariant
        // (halo == 0 == the free-surface image condition) afterwards.
        visco_acoustic2d_apply_spectral(wavefield, spectral, p.dt, p.M);

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

    out.wavefield = u_allt;
    out.last_two = torch::Tensor();
    out.record = record;

    return out;
}

} // namespace visco_acoustic2d

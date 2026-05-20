// ---------------------------------------------------------------------------
// AcousticVTI1st (Duveneck 2008) — 3-D first-order velocity-stress CUDA forward.
//
// Mirrors the 2-D entry point in ../acoustic_vti_1st_2d/forward.cu with the
// y-direction additions:
//
//   * 11 wavefield tensors (5 physical + 6 CPML memory) instead of 8.
//   * `fdtd::Wave3D::make` launch config (block 32×4×2, grid spans batch
//     through gridDim.z).
//   * Spacing list arrives in Cartesian order `[dx, dy, dz]` from
//     PropTorch's `_cuda_spacing()` helper.
//   * Boundary saver tracks 5 physical fields (vx, vy, vz, sH, sV).
//
// Reference: Duveneck E., Milcik P., Bakker P. M., Perkins C. (2008),
// "Acoustic VTI wave equations and their application for anisotropic
// reverse-time migration", SEG Las Vegas Annual Meeting, pp. 2186-2190,
// DOI: 10.1190/1.3059320.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda_runtime.h>


#include <c10/cuda/CUDAGuard.h>
#include "acoustic_vti_1st_3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"     // ElasticCPMLTensor (12 vals in 3D)
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace acoustic_vti_1st_3d {

// Internal wavefield-tensor helper.  Allocates / binds 11 float tensors
// shaped (N, C, nz, ny, nx).
struct VTIWavefieldTensor3D {
    torch::Tensor vx_t, vy_t, vz_t, sH_t, sV_t;
    torch::Tensor m_sHx_t, m_sHy_t, m_sVz_t;
    torch::Tensor m_vxx_t, m_vyy_t, m_vzz_t;
    bool allocated = false;

    void bind(const std::vector<torch::Tensor>& tensors, bool /*use_pml*/)
    {
        TORCH_CHECK(tensors.size() == 11,
                    "AcousticVTI1st3D expects 11 wavefield tensors, got ",
                    tensors.size());
        vx_t    = tensors[0];
        vy_t    = tensors[1];
        vz_t    = tensors[2];
        sH_t    = tensors[3];
        sV_t    = tensors[4];
        m_sHx_t = tensors[5];
        m_sHy_t = tensors[6];
        m_sVz_t = tensors[7];
        m_vxx_t = tensors[8];
        m_vyy_t = tensors[9];
        m_vzz_t = tensors[10];
        allocated = true;
    }

    void allocate(const torch::Tensor& ref, int /*dim*/)
    {
        auto opts = ref.options();
        auto shape = ref.sizes();   // (N, C, nz, ny, nx)
        vx_t    = torch::zeros(shape, opts);
        vy_t    = torch::zeros(shape, opts);
        vz_t    = torch::zeros(shape, opts);
        sH_t    = torch::zeros(shape, opts);
        sV_t    = torch::zeros(shape, opts);
        m_sHx_t = torch::zeros(shape, opts);
        m_sHy_t = torch::zeros(shape, opts);
        m_sVz_t = torch::zeros(shape, opts);
        m_vxx_t = torch::zeros(shape, opts);
        m_vyy_t = torch::zeros(shape, opts);
        m_vzz_t = torch::zeros(shape, opts);
        allocated = true;
    }

    VTIWavefieldPointer3D view() const
    {
        VTIWavefieldPointer3D p{};
        p.vx    = vx_t.data_ptr<float>();
        p.vy    = vy_t.data_ptr<float>();
        p.vz    = vz_t.data_ptr<float>();
        p.sH    = sH_t.data_ptr<float>();
        p.sV    = sV_t.data_ptr<float>();
        p.m_sHx = m_sHx_t.data_ptr<float>();
        p.m_sHy = m_sHy_t.data_ptr<float>();
        p.m_sVz = m_sVz_t.data_ptr<float>();
        p.m_vxx = m_vxx_t.data_ptr<float>();
        p.m_vyy = m_vyy_t.data_ptr<float>();
        p.m_vzz = m_vzz_t.data_ptr<float>();
        return p;
    }

    // 11 tensors saved as one checkpoint "state".
    std::vector<torch::Tensor> state_tensors() const
    {
        return {vx_t, vy_t, vz_t, sH_t, sV_t,
                m_sHx_t, m_sHy_t, m_sVz_t,
                m_vxx_t, m_vyy_t, m_vzz_t};
    }
};


ForwardOutput forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(!p.free_surface,
                "AcousticVTI1st3D CUDA forward: free_surface=True is not yet "
                "implemented (Robertsson 1996 / Mittet 2002 anisotropic FS — "
                "follow-up).");

    // Cartesian-order spacing from PropTorch._cuda_spacing(): [dx, dy, dz].
    TORCH_CHECK(p.spacing.size() >= 3,
                "AcousticVTI1st3D: spacing must have length >= 3");
    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    // Models (vp, epsilon, delta, rho) in this canonical order.
    TORCH_CHECK(p.models.size() == 4,
                "AcousticVTI1st3D expects exactly 4 model tensors "
                "[vp, epsilon, delta, rho]; got ", p.models.size());
    auto vp_t      = p.models[0];
    auto epsilon_t = p.models[1];
    auto delta_t   = p.models[2];
    auto rho_t     = p.models[3];

    int N  = vp_t.size(0);
    int C  = vp_t.size(1);
    int nz = vp_t.size(2);
    int ny = vp_t.size(3);
    int nx = vp_t.size(4);
    int B  = N * C;

    // Cached stiffness tensors (same formulas as Python prepare_models):
    //   c11 = ρ V_P² (1+2ε), c33 = ρ V_P², c13 = ρ V_P² √(1+2δ).
    auto vp_sq   = vp_t * vp_t;
    auto rho_vp2 = rho_t * vp_sq;
    auto c11     = rho_vp2 * (1.0f + 2.0f * epsilon_t);
    auto c33     = rho_vp2;
    auto c13     = rho_vp2 * torch::sqrt(1.0f + 2.0f * delta_t);
    auto inv_rho = 1.0f / rho_t;

    // Wavefield allocation
    VTIWavefieldTensor3D wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, true);
    else
        wavefield.allocate(vp_t, 3);
    auto wf = wavefield.view();

    // CPML (cpmls, 12 vals in 3D — same struct elastic uses)
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    int nsrc        = p.sources_loc.size(1);
    int nrec        = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields   = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    auto record = torch::zeros({nrec_fields, B, nrec, p.nt}, vp_t.options());

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 5, B, nz, ny, nx}, vp_t.options());
        // 5 components: vx, vy, vz, sH, sV

    SolverContext solver{
        3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, dy, dz
    };

    // ----- boundary saving setup (Phase 2: enables backward_bs) ----------
    // For VTI 3-D we save 5 physical fields: vx, vy, vz, sH, sV.  CPML
    // memory variables decay outside the PML band and need not be saved.
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, /*dim=*/3, /*nvar=*/5, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, /*dim=*/3, /*nvar=*/5, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/true, /*transfer_interval=*/1,
            /*boundary_cpu=*/{}, p.boundary_gpu, p.last_two,
            p.use_pinned_memory);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto src_config    = fdtd::Geom::make(nsrc, B);
    auto rec_config    = fdtd::Geom::make(nrec, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(),
                        dx, dy, dz};

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        /*dim=*/3,
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

    // 11 tensors per checkpoint: 5 physical + 6 PML memory.
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        /*expected_tensors=*/11,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "acoustic_vti_1st_3d"
    );

    for (unsigned int it = 0; it < p.nt; ++it) {
        // ----- velocity update (half-step CPML coefficients) -----
        LAUNCH_VTI_3D_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            inv_rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        // ----- stress update (cell-centre CPML coefficients) -----
        LAUNCH_VTI_3D_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            c11.data_ptr<float>(),
            c33.data_ptr<float>(),
            c13.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        // ----- source injection (additive on stress/velocity per the
        //       caller-supplied source_field_indices) -----
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = vti_field_ptr_3d(wf, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<src_config.grid, src_config.block>>>(
                field,
                p.source.data_ptr<float>(),
                p.sources_loc.data_ptr<int>(),
                it,
                nsrc,
                solver
            );
        }

        // ----- checkpoint save (for backward_ckpt) -----
        checkpoint_runtime.save_forward(static_cast<int>(it),
                                        static_cast<int>(p.nt),
                                        wavefield.state_tensors());

        // ----- boundary save (for backward_bs) -----
        if (p.use_boundary_saving) {
            float* fields[5] = { wf.vx, wf.vy, wf.vz, wf.sH, wf.sV };
            for (int f = 0; f < 5; ++f) {
                boundary_runtime.save_forward_3d_field(
                    it, p.nt, fields[f],
                    launch_config.grid, launch_config.block,
                    bs, save_width,
                    -p.M, solver, f, f == 4);
            }
        }

        // ----- snapshot AFTER source injection -----
        //   Matches the 2-D convention.  inv_rho gradient at step it+1 uses
        //   sH/sV input to that step (= state-after-inject at step it), so
        //   we snapshot here.
        if (u_allt.defined()) {
            int spatial_size = solver.nx * solver.ny * solver.nz;
            float* u_this_t = u_allt[it].data_ptr<float>();
            const int comp_stride = B * spatial_size;
            cudaMemcpyAsync(u_this_t + 0 * comp_stride, wf.vx,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(u_this_t + 1 * comp_stride, wf.vy,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(u_this_t + 2 * comp_stride, wf.vz,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(u_this_t + 3 * comp_stride, wf.sH,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(u_this_t + 4 * comp_stride, wf.sV,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
        }

        // ----- receiver recording -----
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = vti_field_ptr_3d(wf, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel_3d<<<rec_config.grid, rec_config.block>>>(
                field,
                record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(),
                it,
                nrec,
                solver
            );
        }
    }

    // ----- save final state for backward_bs (last_two) -----
    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(0, 0).select(0, 0).copy_(wavefield.vx_t);
        boundary_saver.last_two_t.select(0, 1).select(0, 0).copy_(wavefield.vy_t);
        boundary_saver.last_two_t.select(0, 2).select(0, 0).copy_(wavefield.vz_t);
        boundary_saver.last_two_t.select(0, 3).select(0, 0).copy_(wavefield.sH_t);
        boundary_saver.last_two_t.select(0, 4).select(0, 0).copy_(wavefield.sV_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two  = boundary_saver.last_two_t;
    out.record    = record;
    return out;
}

// Backward implementations live in backward.cu.

}  // namespace acoustic_vti_1st_3d

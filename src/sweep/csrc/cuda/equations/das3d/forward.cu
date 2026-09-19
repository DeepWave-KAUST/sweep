#include <torch/extension.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "das3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das3d {

// Layout of p.forward_workspace, declared on the Python side as
// DASZhao3D.cuda_layout.forward_workspace_nvar: the per-step derivative scratch
// (one padded grid per shot each). The step loop zeroes every slot before it
// is written, so nothing here relies on the pool's contents at entry.
enum ForwardWorkspaceSlot : int {
    TMP_SXX_X = 0, TMP_SYY_Y, TMP_SZZ_Z,
    TMP_TXX_Y, TMP_TXX_Z, TMP_TYY_X, TMP_TYY_Z, TMP_TZZ_X, TMP_TZZ_Y,
    N_FORWARD_SLOTS
};

ForwardOutput forward(const ForwardInput& in)
{
    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    const auto lame = derived::lame(p, vp, vs, rho, "das3d::forward");
    auto mu = lame.mu;
    auto lambda = lame.lambda;
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    // Mandatory: the propagator binds every forward state slot
    // (cuda_layout.base_nvar + pml_nvar = 31, propagator/_c.py Wrapper.forward
    // `params.wavefields = cp.forward_wavefields`), in every mode.
    DasWavefieldTensor3D wavefield;
    TORCH_CHECK(!p.wavefields.empty(),
                "das3d/forward requires the propagator-bound wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    wavefield.bind(p.wavefields);
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    // Mandatory: cuda_layout.record_shape is record_multi(), so the propagator
    // always allocates and binds record_out.
    auto record = bound_required(p.record_out, {nrec_fields, B, nrec, p.nt}, vp.options(), "record_out");

    // Mandatory: cuda_layout.forward_workspace_nvar = 9, so
    // _transient_forward_workspace always hands over nine grids.
    TORCH_CHECK(static_cast<int>(p.forward_workspace.size()) == N_FORWARD_SLOTS,
                "das3d/forward requires the propagator-bound forward_workspace "
                "(cuda_layout.forward_workspace_nvar = ",
                static_cast<int>(N_FORWARD_SLOTS), "), got ", p.forward_workspace.size());
    const auto& ws = p.forward_workspace;
    auto tmp_sxx_x = pool_required(ws, TMP_SXX_X, vp, "forward_workspace");
    auto tmp_syy_y = pool_required(ws, TMP_SYY_Y, vp, "forward_workspace");
    auto tmp_szz_z = pool_required(ws, TMP_SZZ_Z, vp, "forward_workspace");
    auto tmp_txx_y = pool_required(ws, TMP_TXX_Y, vp, "forward_workspace");
    auto tmp_txx_z = pool_required(ws, TMP_TXX_Z, vp, "forward_workspace");
    auto tmp_tyy_x = pool_required(ws, TMP_TYY_X, vp, "forward_workspace");
    auto tmp_tyy_z = pool_required(ws, TMP_TYY_Z, vp, "forward_workspace");
    auto tmp_tzz_x = pool_required(ws, TMP_TZZ_X, vp, "forward_workspace");
    auto tmp_tzz_y = pool_required(ws, TMP_TZZ_Y, vp, "forward_workspace");
    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        // Mandatory in this branch: cuda_layout.save_all_shape is
        // history_fields(3), so a save_all forward always binds u_allt_out.
        u_allt = bound_required(p.u_allt_out, {p.nt, 3, B, nz, ny, nx}, vp.options(), "u_allt_out");
    }

    SolverContext solver{
        3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, dy, dz
    };
    SGradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    for (unsigned int it = 0; it < p.nt; ++it) {
        tmp_sxx_x.zero_();
        tmp_syy_y.zero_();
        tmp_szz_z.zero_();
        tmp_txx_y.zero_();
        tmp_txx_z.zero_();
        tmp_tyy_x.zero_();
        tmp_tyy_z.zero_();
        tmp_tzz_x.zero_();
        tmp_tzz_y.zero_();

        LAUNCH_DAS3D_FIRST(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_syy_y.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_y.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tyy_x.data_ptr<float>(),
            tmp_tyy_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            tmp_tzz_y.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS3D_SECOND(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_syy_y.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_y.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tyy_x.data_ptr<float>(),
            tmp_tyy_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            tmp_tzz_y.data_ptr<float>(),
            rho.data_ptr<float>(),
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das3d_field_ptr(wf, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.source.data_ptr<float>(),
                p.sources_loc.data_ptr<int>(),
                it,
                nsrc,
                solver
            );
        }

        if (u_allt.defined()) {
            auto history_t = u_allt.select(0, it);
            history_t.select(0, 0).copy_(wavefield.exx_t.view({B, nz, ny, nx}));
            history_t.select(0, 1).copy_(wavefield.eyy_t.view({B, nz, ny, nx}));
            history_t.select(0, 2).copy_(wavefield.ezz_t.view({B, nz, ny, nx}));
        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das3d_field_ptr(wf, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel_3d<<<record_config.grid, record_config.block>>>(
                field,
                record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(),
                it,
                nrec,
                solver
            );
        }
    }

    out.wavefield = u_allt;
    out.last_two = torch::empty({0}, vp.options());
    out.record = record;
    return out;
}

}

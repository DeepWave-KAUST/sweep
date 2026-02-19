#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic3d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

std::tuple<
    torch::Tensor,   // vp
    std::tuple<      // boundary tuple
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor
    >,
    torch::Tensor,   // u_last_two
    torch::Tensor    // record
>
acoustic_forward3d_cuda(
    torch::Tensor vp,          // (N, C, nz, ny, nx)
    torch::Tensor source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,
    torch::Tensor grad_coes,
    int M,
    int abcn,
    torch::Tensor sources_loc,    // (B, nsrc, 3)
    torch::Tensor receivers_loc,  // (B, nrec, 3)
    const std::vector<torch::Tensor>& pml_vals,
    bool save_all_wavefields,
    bool use_boundary_saving,
    bool free_surface,
    unsigned int nt,
    float dt,
    std::vector<float> spacing
)
{

    float dx = spacing[0];
    float dy = spacing[1];
    float dz = spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B     = N * C;
    int nsrc  = sources_loc.size(1);
    int nrec  = receivers_loc.size(1);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor wavefield;
    wavefield.allocate(vp, 3, true);

    // ----------------------------
    // PML parameters
    // ----------------------------
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(pml_vals, 3);
    auto cpml = cpml_tensor.view();

    // ----------------------------
    // record
    // ----------------------------
    auto record = torch::zeros(
        {N, nrec, nt},
        vp.options()
    );

    // ----------------------------
    // save all wavefields
    // ----------------------------
    torch::Tensor u_allt;
    if (save_all_wavefields)
        u_allt = torch::zeros({nt, B, nz, ny, nx}, vp.options());

    // ----------------------------
    // boundary saving (3D)
    // ----------------------------
    AcousticBoundarySaver boundary_saver;
    boundary_saver.allocate(use_boundary_saving, 3, ctx, vp);
    auto bs = boundary_saver.view();

    // ----------------------------
    // CUDA launch config
    // ----------------------------
    dim3 block(16, 8, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z * B
    );


    float* u_thist = nullptr;

    // ============================================================
    // time stepping
    // ============================================================
    for (int it = 0; it < nt; ++it)
    {

        auto view = wavefield.view();

        u_thist = u_allt.defined()
            ? u_allt[it].data_ptr<float>()
            : nullptr;

        LAUNCH_FORWARD_3D(
            order,
            view,
            save_all_wavefields,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );

        if (use_boundary_saving) {
            save_boundary_kernel_3d<<<grid, block>>>(
                view.u_now,
                bs.top,
                bs.bottom,
                bs.front,
                bs.back,
                bs.left,
                bs.right,
                it,
                ctx
            );
        }

        add_source_3d<<<B, nsrc>>>(
            view.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel_3d<<<B, nrec>>>(
            view.u_next,
            record.data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap();
    }

    if (use_boundary_saving) {
        boundary_saver.last_two_t[0].copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t[1].copy_(wavefield.u_now_t);
    }

    return std::make_tuple(
        u_allt,
        std::make_tuple(
            boundary_saver.top_t,
            boundary_saver.bottom_t,
            boundary_saver.front_t,
            boundary_saver.back_t,
            boundary_saver.left_t,
            boundary_saver.right_t
        ),
        boundary_saver.last_two_t,
        record
    );
}

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic2d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

std::tuple<
    torch::Tensor,   // u_allt
    std::tuple<      // boundary tuple
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor
    >,
    torch::Tensor,   // u_last_two
    torch::Tensor    // record
>
acoustic_forward_cuda(
    torch::Tensor vp,          // velocity (m/s)
    torch::Tensor source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,       // FD coefficients c[0..M]
    torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
    int M,            // half order (order = 2*M)
    int abcn,                 // number of ABC layers
    torch::Tensor sources_loc,   // (B, nsrc, 2) int32
    torch::Tensor receivers_loc, // (B, nrec, 2) int32
    const std::vector<torch::Tensor>& pml_vals,
    bool save_all_wavefields,
    bool use_boundary_saving,
    bool free_surface,
    unsigned int nt,
    float dt,
    std::vector<float> spacing
) {

    float dx = spacing[0];
    float dz = spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int nsrc = sources_loc.size(1);
    int nrec = receivers_loc.size(1);
    int B = N * C;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor wavefield;
    wavefield.allocate(vp, 2, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros(
        {N, receivers_loc.size(1), nt},
        vp.options()
    );

    // Wavefields for all timestep
    torch::Tensor u_allt;
    if (save_all_wavefields)
        u_allt = torch::zeros({nt, B, nz, nx}, vp.options());

    // Wavefields for boundary saving
    AcousticBoundarySaver boundary_saver;
    boundary_saver.allocate(use_boundary_saving, 2, ctx, vp);
    auto bs = boundary_saver.view();

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );

    float* u_thist = nullptr;


    for (int it = 0; it < nt; ++it) {

        auto view = wavefield.view();

        u_thist = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;
        
        LAUNCH_FORWARD(
            order,
            view,
            save_all_wavefields,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );

        if (use_boundary_saving) {
            save_boundary_kernel<<<grid, block>>>(
                view.u_now,
                bs.top,
                bs.bottom,
                bs.left,
                bs.right,
                it,
                ctx
            );
        }
        
        add_source<<<B, nsrc>>>(
            view.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
        
        record_kernel<<<N, nrec>>>(
            view.u_next,
            record.data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap();

    }

    // Save the last two time steps for backward
    if (use_boundary_saving) {
        boundary_saver.last_two_t[0].copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t[1].copy_(wavefield.u_now_t);
    }

    return std::make_tuple(
        u_allt,
        std::make_tuple(
            boundary_saver.top_t,
            boundary_saver.bottom_t,
            boundary_saver.left_t,
            boundary_saver.right_t
        ),
        boundary_saver.last_two_t,
        record
    );

}
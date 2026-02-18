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

    // Wavefield variables
    auto u_prev = torch::zeros_like(vp);
    auto u_now  = torch::zeros_like(vp);
    auto u_next = torch::zeros_like(vp);

    auto psixn = torch::zeros_like(vp);
    auto psizn = torch::zeros_like(vp);
    auto zetax = torch::zeros_like(vp);
    auto zetaz = torch::zeros_like(vp);

    AcousticWavefield wavefield{

        u_prev.data_ptr<float>(), 
        u_now.data_ptr<float>(), 
        u_next.data_ptr<float>(), 

        psixn.data_ptr<float>(), 
        nullptr,
        psizn.data_ptr<float>(), 

        zetax.data_ptr<float>(), 
        nullptr,
        zetaz.data_ptr<float>()
    };

    // cpml parameters
    auto az     = pml_vals[0];
    auto bz     = pml_vals[1];
    auto dbzdz  = pml_vals[2];
    auto ax     = pml_vals[3];
    auto bx     = pml_vals[4];
    auto dbxdx  = pml_vals[5];

    AcousticCPML cpml{
        ax.data_ptr<float>(),
        bx.data_ptr<float>(),
        dbxdx.data_ptr<float>(),

        nullptr,
        nullptr,
        nullptr,

        az.data_ptr<float>(),
        bz.data_ptr<float>(),
        dbzdz.data_ptr<float>()
    };

    auto record = torch::zeros(
        {N, receivers_loc.size(1), nt},
        vp.options()
    );

    // Wavefields for all timestep
    torch::Tensor u_allt;
    if (save_all_wavefields)
        u_allt = torch::zeros({nt, B, nz, nx}, vp.options());

    // Wavefields for boundary saving
    torch::Tensor u_boundary_top, u_boundary_bottom, u_boundary_left, u_boundary_right, u_last_two;

    if (use_boundary_saving) {
        u_boundary_top    = torch::zeros({nt, B, M, nx}, vp.options());
        u_boundary_bottom = torch::zeros({nt, B, M, nx}, vp.options());
        u_boundary_left   = torch::zeros({nt, B, nz, M}, vp.options());
        u_boundary_right  = torch::zeros({nt, B, nz, M}, vp.options());

        u_last_two = torch::zeros({2, B, C, nz, nx}, vp.options());
    }

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    float* u_thist = nullptr;

    SolverContext ctx{nx, 0, nz, B, dt, nt, M, abcn, true, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    for (int it = 0; it < nt; ++it) {

        u_thist = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;
        
        LAUNCH_FORWARD(
            order,
            wavefield,
            save_all_wavefields,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );

        if (use_boundary_saving) {
            save_boundary_kernel<<<grid, block>>>(
                wavefield.u_now,
                u_boundary_top.data_ptr<float>(),
                u_boundary_bottom.data_ptr<float>(),
                u_boundary_left.data_ptr<float>(),
                u_boundary_right.data_ptr<float>(),
                it,
                ctx
            );
        }
        
        add_source<<<B, nsrc>>>(
            wavefield.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            nt,
            nx,
            nz
        );
        
        record_kernel<<<N, nrec>>>(
            wavefield.u_next,
            record.data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            nt,
            nx,
            nz
        );

        // rotate pointers: u_prev <- u_now <- u_next
        auto tmp = wavefield.u_prev;
        wavefield.u_prev = wavefield.u_now;
        wavefield.u_now = wavefield.u_next;
        wavefield.u_next = tmp;

    }

    // Save the last two time steps for backward
    if (use_boundary_saving) {
        cudaMemcpy(
            u_last_two[0].data_ptr<float>(),
            wavefield.u_prev,
            B * C * nz * nx * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            u_last_two[1].data_ptr<float>(),
            wavefield.u_now,
            B * C * nz * nx * sizeof(float),
            cudaMemcpyDeviceToDevice
        );
        // u_last_two[0].copy_(wavefield.u_prev);
        // u_last_two[1].copy_(wavefield.u_now);
    }

    return std::make_tuple(
        u_allt,
        std::make_tuple(
            u_boundary_top,
            u_boundary_bottom,
            u_boundary_left,
            u_boundary_right
        ),
        u_last_two,
        record
    );

}
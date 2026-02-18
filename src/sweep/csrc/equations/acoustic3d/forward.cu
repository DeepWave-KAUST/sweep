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

    // ----------------------------
    // wavefields
    // ----------------------------
    auto u_prev = torch::zeros_like(vp);
    auto u_now  = torch::zeros_like(vp);
    auto u_next = torch::zeros_like(vp);

    auto psixn = torch::zeros_like(vp);
    auto psiyn = torch::zeros_like(vp);
    auto psizn = torch::zeros_like(vp);

    auto zetax = torch::zeros_like(vp);
    auto zetay = torch::zeros_like(vp);
    auto zetaz = torch::zeros_like(vp);

    AcousticWavefield wavefield{

        u_now.data_ptr<float>(),
        u_prev.data_ptr<float>(), 
        u_next.data_ptr<float>(),

        psixn.data_ptr<float>(),
        psiyn.data_ptr<float>(),
        psizn.data_ptr<float>(),

        zetax.data_ptr<float>(),
        zetay.data_ptr<float>(),
        zetaz.data_ptr<float>()
    
    };

    // ----------------------------
    // PML parameters
    // ----------------------------
    auto az     = pml_vals[0];
    auto bz     = pml_vals[1];
    auto dbzdz  = pml_vals[2];

    auto ay     = pml_vals[3];
    auto by     = pml_vals[4];
    auto dbydy  = pml_vals[5];

    auto ax     = pml_vals[6];
    auto bx     = pml_vals[7];
    auto dbxdx  = pml_vals[8];

    AcousticCPML cpml{

        ax.data_ptr<float>(),
        bx.data_ptr<float>(),
        dbxdx.data_ptr<float>(),

        ay.data_ptr<float>(),
        by.data_ptr<float>(),
        dbydy.data_ptr<float>(),

        az.data_ptr<float>(),
        bz.data_ptr<float>(),
        dbzdz.data_ptr<float>()
    };

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
    torch::Tensor u_boundary_xmin, u_boundary_xmax;
    torch::Tensor u_boundary_ymin, u_boundary_ymax;
    torch::Tensor u_boundary_zmin, u_boundary_zmax;
    torch::Tensor u_last_two;

    if (use_boundary_saving) {

        u_boundary_xmin = torch::zeros({nt, B, nz, ny, M}, vp.options());
        u_boundary_xmax = torch::zeros({nt, B, nz, ny, M}, vp.options());

        u_boundary_ymin = torch::zeros({nt, B, nz, M, nx}, vp.options());
        u_boundary_ymax = torch::zeros({nt, B, nz, M, nx}, vp.options());

        u_boundary_zmin = torch::zeros({nt, B, M, ny, nx}, vp.options());
        u_boundary_zmax = torch::zeros({nt, B, M, ny, nx}, vp.options());

        u_last_two = torch::zeros({2, B, 1, nz, ny, nx}, vp.options());
    }

    // ----------------------------
    // CUDA launch config
    // ----------------------------
    dim3 block(16, 8, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z * B
    );

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    float* u_thist = nullptr;

    SolverContext ctx{nx, ny, nz, B, dt, nt, M, abcn, true, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, dy, dz};

    // ============================================================
    // time stepping
    // ============================================================
    for (int it = 0; it < nt; ++it)
    {
        u_thist = u_allt.defined()
            ? u_allt[it].data_ptr<float>()
            : nullptr;

        LAUNCH_FORWARD_3D(
            order,
            wavefield,
            save_all_wavefields,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );

        if (use_boundary_saving) {
            save_boundary_kernel_3d<<<grid, block>>>(
                wavefield.u_now,
                u_boundary_zmin.data_ptr<float>(),
                u_boundary_zmax.data_ptr<float>(),
                u_boundary_ymin.data_ptr<float>(),
                u_boundary_ymax.data_ptr<float>(),
                u_boundary_xmin.data_ptr<float>(),
                u_boundary_xmax.data_ptr<float>(),
                it,
                ctx
            );
        }

        add_source_3d<<<B, nsrc>>>(
            wavefield.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            nt,
            nx, ny, nz
        );

        record_kernel_3d<<<B, nrec>>>(
            wavefield.u_next,
            record.data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            nt,
            nx, ny, nz
        );

        // rotate
        auto tmp = wavefield.u_prev;
        wavefield.u_prev = wavefield.u_now;
        wavefield.u_now  = wavefield.u_next;
        wavefield.u_next = tmp;
    }

    if (use_boundary_saving) {
        cudaMemcpy(
            u_last_two[0].data_ptr<float>(),
            wavefield.u_prev,
            sizeof(float) * B * nz * ny * nx,
            cudaMemcpyDeviceToDevice
        );
        cudaMemcpy(
            u_last_two[1].data_ptr<float>(),
            wavefield.u_now,
            sizeof(float) * B * nz * ny * nx,
            cudaMemcpyDeviceToDevice
        );
    }

    return std::make_tuple(
        u_allt,
        std::make_tuple(
            u_boundary_zmin,
            u_boundary_zmax,
            u_boundary_ymin,
            u_boundary_ymax,
            u_boundary_xmin,
            u_boundary_xmax
        ),
        u_last_two,
        record
    );
}

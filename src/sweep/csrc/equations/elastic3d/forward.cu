#include <torch/extension.h>
#include <cuda_runtime.h>

#include "elastic3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

std::tuple<
    torch::Tensor,   // u_allt
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
forward(
    const std::vector<torch::Tensor>& models,
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
){

    float dx = spacing[0];
    float dy = spacing[1];
    float dz = spacing[2];

    // parse model parameters
    auto vp = models[0];
    auto vs = models[1];
    auto rho = models[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B = N * C;

    ElasticWavefieldTensor wavefield;
    wavefield.allocate(vp, 3);
    auto wf = wavefield.view();

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    ElasticCPMLTensor cpml;
    cpml.allocate(pml_vals, 3);
    auto cpml_view = cpml.view();

    int nsrc = sources_loc.size(1);
    int nrec = receivers_loc.size(1);
    auto record = torch::zeros({B, nrec, nt}, vp.options());

    torch::Tensor u_allt;
    // if (save_all_wavefields) u_allt = torch::zeros({nt, 2, B, nz, ny, nx}, vp.options()); // Only save Vx and Vz.

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    SolverContext solver{3, nx, ny, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, dy, dz};
    
    GeneralBoundarySaverMore boundary_saver;
    boundary_saver.allocate(use_boundary_saving, 3, 9, solver, vp, solver.M, 1);
    auto bs = boundary_saver.view();

    SGradParam grad_ctx{1, nx, nx*ny, M, grad_coes.data_ptr<float>(), dx, dy, dz};

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    float* u_this_t = nullptr;

    for (unsigned int it = 0; it < nt; ++it) {

        // u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_3DELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t+0.5

        LAUNCH_3DELASTIC_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            u_this_t,
            grad_ctx,
            cpml_view,
            solver
        ); // t+1.0

        add_source_3d<<<source_config.grid, source_config.block>>>(
            wf.vz,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            solver
        );

        if (use_boundary_saving) {

            float* fields[9] = {
                wf.vx,
                wf.vy,
                wf.vz,
                wf.sxx,
                wf.syy,
                wf.szz,
                wf.sxy,
                wf.sxz,
                wf.syz
            };

            for (int f = 0; f < 9; ++f) {
                save_boundary_kernel_3d_advance2<<<launch_config.grid, launch_config.block>>>(
                    fields[f],
                    boundary_saver.top_t[f].data_ptr<float>(),
                    boundary_saver.bottom_t[f].data_ptr<float>(),
                    boundary_saver.front_t[f].data_ptr<float>(),
                    boundary_saver.back_t[f].data_ptr<float>(),
                    boundary_saver.left_t[f].data_ptr<float>(),
                    boundary_saver.right_t[f].data_ptr<float>(),
                    it,
                    solver.M,
                    solver
                );
            }
        }

        record_kernel_3d<<<record_config.grid, record_config.block>>>(
            wf.vz,
            record.data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            solver
        );
    }

    if (use_boundary_saving) {
        boundary_saver.last_two_t.select(0,0).select(0,0).copy_(wavefield.vx_t);
        boundary_saver.last_two_t.select(0,1).select(0,0).copy_(wavefield.vy_t);
        boundary_saver.last_two_t.select(0,2).select(0,0).copy_(wavefield.vz_t);
        boundary_saver.last_two_t.select(0,3).select(0,0).copy_(wavefield.sxx_t);
        boundary_saver.last_two_t.select(0,4).select(0,0).copy_(wavefield.syy_t);
        boundary_saver.last_two_t.select(0,5).select(0,0).copy_(wavefield.szz_t);
        boundary_saver.last_two_t.select(0,6).select(0,0).copy_(wavefield.sxy_t);
        boundary_saver.last_two_t.select(0,7).select(0,0).copy_(wavefield.sxz_t);
        boundary_saver.last_two_t.select(0,8).select(0,0).copy_(wavefield.syz_t);
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

}
#include <torch/extension.h>

#include "kernels.cuh"
#include "elastic3d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_bs(
    const std::vector<torch::Tensor>& u_boundary,
    torch::Tensor u_last_two,     // (B, nz, nx)
    const std::vector<torch::Tensor>& models,
    torch::Tensor adjoint_source,      // (B, nsrc, nt)
    torch::Tensor forward_source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,       // FD coefficients c[0..M]
    torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
    int M,            // half order (order = 2*M)
    int abcn,                 // number of ABC layers
    torch::Tensor adjoint_sources_loc,   // (B, nsrc, 2) int32
    torch::Tensor forward_sources_loc,   // (B, nsrc, 2) int32
    const std::vector<torch::Tensor>& pml_vals,
    unsigned int nt,
    float dt,
    std::vector<float> spacing,
    bool free_surface
) {

    float dx = spacing[0];
    float dy = spacing[1];
    float dz = spacing[2];

    auto vp = models[0];
    auto vs = models[1];
    auto rho = models[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B = N * C;

    int adjoint_nsrc = adjoint_sources_loc.size(1);
    int forward_nsrc = forward_sources_loc.size(1);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext solver{3, nx, ny, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, dy, dz};
    
    ElasticWavefieldTensor adjoint;
    adjoint.allocate(vp, 3);
    ElasticWavefieldTensor forward;

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Copy last step of forward wavefield from u_last_two
    forward.allocate(vp, 3, false);
    forward.vx_t.copy_(u_last_two.select(0,0).select(0,0));
    forward.vy_t.copy_(u_last_two.select(0,1).select(0,0));
    forward.vz_t.copy_(u_last_two.select(0,2).select(0,0));
    forward.sxx_t.copy_(u_last_two.select(0,3).select(0,0));
    forward.syy_t.copy_(u_last_two.select(0,4).select(0,0));
    forward.szz_t.copy_(u_last_two.select(0,5).select(0,0));
    forward.sxy_t.copy_(u_last_two.select(0,6).select(0,0));
    forward.sxz_t.copy_(u_last_two.select(0,7).select(0,0));
    forward.syz_t.copy_(u_last_two.select(0,8).select(0,0));

    auto neg_forward_source = -forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    // // Set PML part to zero
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vx, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxx, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.syy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.szz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.syz, solver.abcn+0, nx, ny, nz);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(pml_vals, 3);
    auto cpml_view = cpml.view();

    GeneralBoundarySaverMore boundary_saver;
    boundary_saver.allocate(true, 3, 9, solver, vp, solver.M, 1);
    boundary_saver.load_from_vector(u_boundary, vp);
    auto bs = boundary_saver.view();

    auto fvx_prev = torch::zeros_like(vp);
    auto fvy_prev = torch::zeros_like(vp);
    auto fvz_prev = torch::zeros_like(vp);

    auto u_all_t = torch::zeros({2, B, 1, nz, ny, nx}, vp.options());
    SGradParam grad_ctx{1, nx, nx*ny, M, grad_coes.data_ptr<float>(), dx, dy, dz};

    for (int it = nt - 1; it >= 1; --it) {
        
        // Adjoint modeling
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t-0.5

        LAUNCH_3DELASTIC_STRESS_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t-1.0

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.vz,
            adjoint_source.data_ptr<float>(),
            adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            solver
        );

        // Wavefield reconstruction
        add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.vz,
            neg_forward_source.data_ptr<float>(),
            forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            solver
        );

        LAUNCH_3DELASTIC_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field2[6] = {for_view.sxx, for_view.syy, for_view.szz, for_view.sxy, for_view.sxz, for_view.syz};

        for (int f = 3; f < 9; ++f)
            restore_boundary_kernel_3d_advance2<<<launch_config.grid, launch_config.block>>>(
                field2[f-3],
                boundary_saver.top_t[f].data_ptr<float>(),
                boundary_saver.bottom_t[f].data_ptr<float>(),
                boundary_saver.front_t[f].data_ptr<float>(),
                boundary_saver.back_t[f].data_ptr<float>(),
                boundary_saver.left_t[f].data_ptr<float>(),
                boundary_saver.right_t[f].data_ptr<float>(),
                it-1,
                solver.M,
                solver
            );

        // Gradient calculation
        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            adj_view,

            fvx_prev.data_ptr<float>(),
            fvy_prev.data_ptr<float>(),
            fvz_prev.data_ptr<float>(),

            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),

            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),

            grad_ctx,
            solver
        );

        fvz_prev.copy_(forward.vz_t);
        fvy_prev.copy_(forward.vy_t);
        fvx_prev.copy_(forward.vx_t);

        // Update Velocity components
        LAUNCH_3DELASTIC_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field1[3] = {for_view.vx, for_view.vy, for_view.vz};

        for (int f = 0; f < 3; ++f)
            restore_boundary_kernel_3d_advance2<<<launch_config.grid, launch_config.block>>>(
                field1[f],
                boundary_saver.top_t[f].data_ptr<float>(),
                boundary_saver.bottom_t[f].data_ptr<float>(),
                boundary_saver.front_t[f].data_ptr<float>(),
                boundary_saver.back_t[f].data_ptr<float>(),
                boundary_saver.left_t[f].data_ptr<float>(),
                boundary_saver.right_t[f].data_ptr<float>(),
                it-1,
                solver.M,
                solver
            );
    }

    u_all_t[0].copy_(forward.vz_t); // for visualization
    u_all_t[1].copy_(adjoint.vz_t); // for visualization

    return std::make_tuple(u_all_t, grad_vp, grad_vs, grad_rho);
}

}
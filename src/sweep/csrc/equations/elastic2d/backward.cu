#include <torch/extension.h>

#include "kernels.cuh"
#include "elastic2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
elastic_backward_cuda(
    torch::Tensor u_forward,     // (nt, 4， B, nz, nx)
    const std::vector<torch::Tensor>& models,
    torch::Tensor adjoint_source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,       // FD coefficients c[0..M]
    torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
    int M,            // half order (order = 2*M)
    int abcn,                 // number of ABC layers
    torch::Tensor adjoint_sources_loc,   // (B, nsrc, 2) int32
    const std::vector<torch::Tensor>& pml_vals,
    unsigned int nt,
    float dt,
    std::vector<float> spacing
) {

    float dx = spacing[0];
    float dz = spacing[1];

    auto vp = models[0];
    auto vs = models[1];
    auto rho = models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = adjoint_sources_loc.size(1);

    int B = N * C;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext solver{2, nx, 0, nz, B, dt, nt, M, abcn, false, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto f_this = torch::zeros_like(vp); // for gradient calculation
    
    ElasticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Generate pointer views
    auto adj_view = adjoint.view();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(pml_vals, 2);
    auto cpml_view = cpml.view();

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );


    for (int it = nt - 1; it >= 1; --it) {

        LAUNCH_ELASTIC_VELOCITY_ADJOINT(
            order,
            adj_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_STRESS_ADJOINT(
            order,
            adj_view,
            rho.data_ptr<float>(),
            cpml_view,
            solver
        );

        add_source<<<B, adjoint_nsrc>>>(
            adj_view.vz,
            adjoint_source.data_ptr<float>(),
            adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            solver
        );

        calculate_elastic_grad<<<grid, block>>>(
            adj_view,

            u_forward.select(0, it).select(0, 0).data_ptr<float>(), // vx_x
            u_forward.select(0, it).select(0, 1).data_ptr<float>(), // vx_z
            u_forward.select(0, it).select(0, 2).data_ptr<float>(), // vz_x
            u_forward.select(0, it).select(0, 3).data_ptr<float>(), // vz_z

            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),

            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),

            solver
        );


    }

    return std::make_tuple(grad_vp, grad_vs, grad_rho);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
elastic_backward_boundary_saving_cuda(
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
    float dz = spacing[1];

    auto vp = models[0];
    auto vs = models[1];
    auto rho = models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = adjoint_sources_loc.size(1);
    int forward_nsrc = forward_sources_loc.size(1);
    int B = N * C;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext solver{2, nx, 0, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto f_this = torch::zeros_like(vp); // for gradient calculation
    
    ElasticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2);
    ElasticWavefieldTensor forward;
    forward.allocate(vp, 2);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Copy last step of forward wavefield from u_last_two
    forward.allocate(vp, 2, false);
    forward.vx_t.copy_(u_last_two.select(0,0).select(0,0));
    forward.vz_t.copy_(u_last_two.select(0,1).select(0,0));
    forward.sxx_t.copy_(u_last_two.select(0,2).select(0,0));
    forward.szz_t.copy_(u_last_two.select(0,3).select(0,0));
    forward.sxz_t.copy_(u_last_two.select(0,4).select(0,0));

    auto neg_forward_source = -forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(pml_vals, 2);
    auto cpml_view = cpml.view();

    GeneralBoundarySaver boundary_saver;
    boundary_saver.allocate(true, 2, 5, solver, vp, solver.M+1);
    boundary_saver.load_from_vector(u_boundary);
    auto bs = boundary_saver.view();

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );

    // Set boundarys of the last frame to be zeors
    set_boundary_zeros<<<grid, block>>>(for_view.vx, solver.abcn+solver.M, nx, nz);
    set_boundary_zeros<<<grid, block>>>(for_view.vz, solver.abcn+solver.M, nx, nz);
    set_boundary_zeros<<<grid, block>>>(for_view.szz, solver.abcn+solver.M, nx, nz);
    set_boundary_zeros<<<grid, block>>>(for_view.sxx, solver.abcn+solver.M, nx, nz);
    set_boundary_zeros<<<grid, block>>>(for_view.sxz, solver.abcn+solver.M, nx, nz);

    auto fvz_prev = torch::zeros_like(vp);
    auto fvx_prev = torch::zeros_like(vp);

    auto u_allt = torch::zeros({nt, B, C, nz, nx}, vp.options()); // check
    auto u_allt_allw = torch::zeros({5, nt, B, C, nz, nx}, vp.options()); // check

    for (int it = nt - 1; it >= 1; --it) {
        
        LAUNCH_ELASTIC_VELOCITY_ADJOINT(
            order,
            adj_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_STRESS_ADJOINT(
            order,
            adj_view,
            rho.data_ptr<float>(),
            cpml_view,
            solver
        );

        add_source<<<B, adjoint_nsrc>>>(
            adj_view.vz,
            adjoint_source.data_ptr<float>(),
            adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            solver
        );

        // Wavefield reconstruction
        add_source<<<B, forward_nsrc>>>(
            for_view.vz,
            neg_forward_source.data_ptr<float>(),
            forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            solver
        );

        // Update Stress components
        LAUNCH_ELASTIC_STRESS_NOPML(
            order,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            solver
        );

        float *field2[3] = {for_view.sxx, for_view.szz, for_view.sxz};

        for (int f = 2; f < 5; ++f)
            restore_boundary_kernel<<<grid, block>>>(
                field2[f-2],
                boundary_saver.top_t[f].data_ptr<float>(),
                boundary_saver.bottom_t[f].data_ptr<float>(),
                boundary_saver.left_t[f].data_ptr<float>(),
                boundary_saver.right_t[f].data_ptr<float>(),
                it-1,
                solver.M+1,
                solver
            );    

        // Gradient calculation
        LAUNCH_CALCULATE_GRAD_ELASTIC(
            order,
            for_view,
            adj_view,

            fvx_prev.data_ptr<float>(),
            fvz_prev.data_ptr<float>(),

            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),

            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),

            solver
        );

        fvz_prev.copy_(forward.vz_t);
        fvx_prev.copy_(forward.vx_t);

        // Update Velocity components
        LAUNCH_ELASTIC_VELOCITY_NOPML(
            order,
            for_view,
            rho.data_ptr<float>(),
            solver
        );

        float *field1[2] = {for_view.vx, for_view.vz};

        for (int f = 0; f < 2; ++f)
            restore_boundary_kernel<<<grid, block>>>(
                field1[f],
                boundary_saver.top_t[f].data_ptr<float>(),
                boundary_saver.bottom_t[f].data_ptr<float>(),
                boundary_saver.left_t[f].data_ptr<float>(),
                boundary_saver.right_t[f].data_ptr<float>(),
                it-1,
                solver.M+1,
                solver
            );
        u_allt[it].copy_(forward.vz_t); // check

        u_allt_allw.select(1, it).select(0, 0).copy_(adjoint.vx_t);
        u_allt_allw.select(1, it).select(0, 1).copy_(adjoint.vz_t);
        u_allt_allw.select(1, it).select(0, 2).copy_(adjoint.sxx_t);
        u_allt_allw.select(1, it).select(0, 3).copy_(adjoint.szz_t);
        u_allt_allw.select(1, it).select(0, 4).copy_(adjoint.sxz_t);
    }

    return std::make_tuple(u_allt, u_allt_allw,grad_vp, grad_vs, grad_rho);
}
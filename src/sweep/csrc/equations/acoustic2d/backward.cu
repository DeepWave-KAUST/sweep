#include <torch/extension.h>

#include "kernels.cuh"
#include "../../operators/laplace2d.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/boundarysaver.h"
#include "../../launch/config.h"

namespace acoustic2d {

std::tuple<torch::Tensor>
backward(
    torch::Tensor u_forward,     // (nt, B, nz, nx)
    const std::vector<torch::Tensor>& models,
    torch::Tensor adjoint_source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,       // FD coefficients c[0..M]
    torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
    int M,            // half order (order = 2*M)
    int abcn,                 // number of ABC layers
    torch::Tensor sources_loc,   // (B, nsrc, 2) int32
    const std::vector<torch::Tensor>& pml_vals,
    unsigned int nt,
    float dt,
    std::vector<float> spacing
) {

    auto vp = models[0];

    float dx = spacing[0];
    float dz = spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int nsrc = sources_loc.size(1);
    int B = N * C;

    AcousticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2, true);

    auto grad = torch::zeros_like(vp);

    float* u_thist = nullptr;

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, nt, M, abcn, true, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    for (int it = nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        LAUNCH_FORWARD(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );
        
        add_source<<<source_config.grid, source_config.block>>>(
            adj_view.u_next,
            adjoint_source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        // rotate pointers: u_prev <- u_now <- u_next
        adjoint.swap();

        calculate_grad<<<launch_config.grid, launch_config.block>>>(
            u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz
        );

    }

    return std::make_tuple(grad);
}

std::tuple<torch::Tensor>
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
    float dz = spacing[1];

    auto vp = models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = adjoint_sources_loc.size(1);
    int forward_nsrc = forward_sources_loc.size(1);
    int B = N * C;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2, true);
    AcousticWavefieldTensor forward;
    forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(u_last_two.select(1,1).squeeze(0));
    forward.u_now_t.copy_(u_last_two.select(1,0).squeeze(0));

    auto grad = torch::zeros_like(vp);

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(pml_vals, 2);
    auto cpml = cpml_tensor.view();

    // Boundary wavefields (for saving all wavefields)
    GeneralBoundarySaver boundary_saver;
    boundary_saver.allocate(true, 2, 1, ctx, vp);
    boundary_saver.load_from_vector(u_boundary, vp);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto for_view = forward.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.abcn+ctx.M, nx, nz);
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.abcn+ctx.M, nx, nz);

    for (int it = nt - 1; it >= 1; --it) {

        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        // u_allt[it].copy_(forward.u_now_t);

        // adjoint modeling
        LAUNCH_FORWARD(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );
        
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            adjoint_source.data_ptr<float>(),
            adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();
        
        LAUNCH_FORWARD_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            vp.data_ptr<float>(),
            ctx
        );

        // Reconstruct the forward wavefield
        restore_boundary_kernel<<<launch_config.grid, launch_config.block>>>(
            for_view.u_next,
            bs.top,
            bs.bottom,
            bs.left,
            bs.right,
            it-1,
            ctx.M,
            ctx
        );

        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            forward_source.data_ptr<float>(),
            forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );
        
        forward.swap();
        
        calculate_grad_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_next_t.data_ptr<float>(),
            forward.u_now_t.data_ptr<float>(),
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz, dt
        );


    }

    return std::make_tuple(grad);
}

}
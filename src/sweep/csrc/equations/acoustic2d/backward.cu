#include <torch/extension.h>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace acoustic2d {

BackwardOutput backward(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int nsrc = p.adjoint_source.size(1);
    int B = N * C;

    int M = p.M;
    float dt = p.dt;

    AcousticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2, true);

    auto grad = torch::zeros_like(vp);

    float* u_thist = nullptr;

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, M, p.abcn, true, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            u_thist,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );
        
        add_source<<<source_config.grid, source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        // rotate pointers: u_prev <- u_now <- u_next
        adjoint.swap();

        calculate_grad<<<launch_config.grid, launch_config.block>>>(
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz
        );


    }
    out.grads = {grad};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    adjoint.allocate(vp, 2, true);
    AcousticWavefieldTensor forward;
    forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1,1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1,0).squeeze(0));

    auto grad = torch::zeros_like(vp);

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    // Boundary wavefields (for saving all wavefields)
    int save_width = p.abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true);
    boundary_saver.load_from_vector(p.u_boundary, vp);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto for_view = forward.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.abcn+ctx.M, nx, nz, p.free_surface);
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.abcn+ctx.M, nx, nz, p.free_surface);
    
    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 1; --it) {

        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        // u_allt[it].copy_(forward.u_now_t);

        // adjoint modeling
        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );
        
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();
        
        ACOUSTIC2D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            vp.data_ptr<float>(),
            lap_ctx,
            ctx
        );

        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
            for_view.u_next,
            bs.top,
            bs.bottom,
            bs.left,
            bs.right,
            it-1,
            save_width,
            0,
            ctx,
            BOUNDARY_RESTORE
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

    out.grads = {grad};
    return out;

}

}
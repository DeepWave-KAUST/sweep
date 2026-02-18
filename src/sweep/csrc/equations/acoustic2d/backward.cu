#include <torch/extension.h>

#include "kernels.cuh"
#include "../../operators/laplace2d.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

std::tuple<torch::Tensor>
acoustic_backward_cuda(
    torch::Tensor u_forward,     // (nt, B, nz, nx)
    torch::Tensor vp,          // velocity (m/s)
    torch::Tensor source,      // (B, nsrc, nt)
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

    float dx = spacing[0];
    float dz = spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int nsrc = sources_loc.size(1);
    int B = N * C;

    auto u_prev = torch::zeros_like(vp);
    auto u_now  = torch::zeros_like(vp);
    auto u_next = torch::zeros_like(vp);

    auto psixn = torch::zeros_like(vp);
    auto psizn = torch::zeros_like(vp);
    auto zetax = torch::zeros_like(vp);
    auto zetaz = torch::zeros_like(vp);

    AcousticWavefield adjoint{

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

    auto grad = torch::zeros_like(vp);

    float* u_thist = nullptr;

    // PML coefficients
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

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{nx, 0, nz, B, dt, nt, M, abcn, true, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    for (int it = nt - 1; it >= 0; --it) {

        LAUNCH_FORWARD(
            order,
            adjoint,
            false,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );
        
        add_source<<<B, nsrc>>>(
            adjoint.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            nt,
            nx,
            nz
        );

        // rotate pointers: u_prev <- u_now <- u_next
        auto tmp = adjoint.u_prev;
        adjoint.u_prev = adjoint.u_now;
        adjoint.u_now = adjoint.u_next;
        adjoint.u_next = tmp;

        calculate_grad<<<grid, block>>>(
            u_forward[it].data_ptr<float>(),
            adjoint.u_now,
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz
        );

    }

    return std::make_tuple(grad);
}


std::tuple<torch::Tensor>
acoustic_backward_boundary_saving_cuda(
    const std::vector<torch::Tensor>& u_boundary,
    torch::Tensor u_last_two,     // (B, nz, nx)
    torch::Tensor vp,          // velocity (m/s)
    torch::Tensor source,      // (B, nsrc, nt)
    torch::Tensor lap_coes,       // FD coefficients c[0..M]
    torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
    int M,            // half order (order = 2*M)
    int abcn,                 // number of ABC layers
    torch::Tensor sources_loc,   // (B, nsrc, 2) int32
    const std::vector<torch::Tensor>& pml_vals,
    unsigned int nt,
    float dt,
    std::vector<float> spacing,
    bool free_surface
) {

    float dx = spacing[0];
    float dz = spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int nsrc = sources_loc.size(1);
    int B = N * C;

    // Forward wavefields
    auto f_prev = torch::zeros_like(vp);
    auto f_now  = torch::zeros_like(vp);
    auto f_next = torch::zeros_like(vp);

    auto f_this = torch::zeros_like(vp); // for gradient calculation

    f_prev.copy_(u_last_two[0]);
    f_now.copy_(u_last_two[1]);

    // Backward wavefields
    auto u_prev = torch::zeros_like(vp);
    auto u_now  = torch::zeros_like(vp);
    auto u_next = torch::zeros_like(vp);

    auto psixn = torch::zeros_like(vp);
    auto psizn = torch::zeros_like(vp);
    auto zetax = torch::zeros_like(vp);
    auto zetaz = torch::zeros_like(vp);

    AcousticWavefield adjoint{

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

    AcousticWavefield forward{

        f_prev.data_ptr<float>(), 
        f_now.data_ptr<float>(), 
        f_next.data_ptr<float>(), 

        nullptr, 
        nullptr,
        nullptr, 

        nullptr, 
        nullptr,
        nullptr
    };


    auto grad = torch::zeros_like(vp);

    float* u_thist = nullptr;

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    // PML coefficients
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

    // Boundary wavefields (for saving all wavefields)
    auto u_boundary_top = u_boundary[0];
    auto u_boundary_bottom = u_boundary[1];
    auto u_boundary_left = u_boundary[2];
    auto u_boundary_right = u_boundary[3];

    dim3 block(32, 8);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (nz + block.y - 1) / block.y,
        B
    );

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    // Assign the last two wavefields from forward to u_prev and u_now
    SolverContext ctx{nx, 0, nz, B, dt, nt, M, abcn, free_surface, lap_coes.data_ptr<float>(), grad_coes.data_ptr<float>(), dx, 0.f, dz};

    for (int it = nt - 1; it >= 1; --it) {

        // adjoint modeling
        LAUNCH_FORWARD(
            order,
            adjoint,
            false,
            u_thist,
            vp.data_ptr<float>(),
            cpml,
            ctx
        );
        
        add_source<<<B, nsrc>>>(
            adjoint.u_next,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            nt,
            nx,
            nz
        );

        // rotate pointers: u_prev <- u_now <- u_next
        auto tmp = adjoint.u_prev;
        adjoint.u_prev = adjoint.u_now;
        adjoint.u_now = adjoint.u_next;
        adjoint.u_next = tmp;
        
        LAUNCH_FORWARD_NOPML(
            order,
            forward,
            f_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            ctx
        );

        // Reconstruct the forward wavefield
        restore_boundary_kernel<<<grid, block>>>(
            forward.u_next,
            u_boundary_top.data_ptr<float>(),
            u_boundary_bottom.data_ptr<float>(),
            u_boundary_left.data_ptr<float>(),
            u_boundary_right.data_ptr<float>(),
            it-1,
            ctx
        );
        
        // rotate pointers for forward wavefields
        auto tmp_f = forward.u_prev;
        forward.u_prev = forward.u_now;
        forward.u_now = forward.u_next;
        forward.u_next = tmp_f;

        calculate_grad_utt<<<grid, block>>>(
            forward.u_next,
            forward.u_now,
            forward.u_prev,
            adjoint.u_now,
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz, dt
        );

        // u_allt[it].copy_(f_now);

    }

    return std::make_tuple(grad);
}
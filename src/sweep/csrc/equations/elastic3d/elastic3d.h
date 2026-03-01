#pragma once
#include <torch/extension.h>

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
);

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
// elastic_backward3d_cuda(
//     torch::Tensor u_forward,     // (nt, 4， B, nz, nx)
//     const std::vector<torch::Tensor>& models,
//     torch::Tensor adjoint_source,      // (B, nsrc, nt)
//     torch::Tensor lap_coes,       // FD coefficients c[0..M]
//     torch::Tensor grad_coes,      // Grad FD coefficients g[0..M-1]
//     int M,            // half order (order = 2*M)
//     int abcn,                 // number of ABC layers
//     torch::Tensor adjoint_sources_loc,   // (B, nsrc, 2) int32
//     const std::vector<torch::Tensor>& pml_vals,
//     unsigned int nt,
//     float dt,
//     std::vector<float> spacing
// );

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
);

}
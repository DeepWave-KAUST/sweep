#pragma once

#include <torch/extension.h>
#include <vector>


struct ForwardInput {

    std::vector<torch::Tensor> models;

    torch::Tensor source;
    torch::Tensor lap_coes;
    torch::Tensor grad_coes;

    int M;
    int abcn;

    torch::Tensor sources_loc;
    torch::Tensor receivers_loc;
    torch::Tensor source_field_indices;
    torch::Tensor receiver_field_indices;

    std::vector<torch::Tensor> pml_vals;  // Bind from python
    std::vector<torch::Tensor> wavefields; // Bind from python
    torch::Tensor last_two; // Bind from python

    std::vector<torch::Tensor> boundary_cpu; // Bind from python
    std::vector<torch::Tensor> boundary_gpu; // Bind from python

    bool save_all_wavefields;
    bool use_boundary_saving;
    bool boundary_on_cpu = false;
    bool use_pinned_memory = false;
    bool free_surface;

    unsigned int nt;
    float dt;

    std::vector<float> spacing;

    int transfer_interval = 1; // Transfer every time step by default
};

struct ForwardOutput {

    torch::Tensor wavefield;

    torch::Tensor last_two;

    torch::Tensor record;
};

struct BackwardOutput {

    std::vector<torch::Tensor> checkpoints;

    std::vector<torch::Tensor> grads;
};

struct BackwardInput {

    // forward wavefield (used in normal backward)
    torch::Tensor u_forward;

    // boundary-saving data (used in backward_bs)
    std::vector<torch::Tensor> u_boundary;
    torch::Tensor u_last_two;

    // Wavefields
    std::vector<torch::Tensor> adjoint_wavefields; // Bind from python
    std::vector<torch::Tensor> forward_wavefields; // Bind from python

    // Wavefields
    std::vector<torch::Tensor> boundary_cpu; // Bind from python
    std::vector<torch::Tensor> boundary_gpu; // Bind from python

    // models
    std::vector<torch::Tensor> models;

    // sources
    torch::Tensor adjoint_source;
    torch::Tensor forward_source;

    // finite-difference coefficients
    torch::Tensor lap_coes;
    torch::Tensor grad_coes;

    int M;
    int abcn;

    // source locations
    torch::Tensor adjoint_sources_loc;
    torch::Tensor forward_sources_loc;
    torch::Tensor source_field_indices;
    torch::Tensor receiver_field_indices;

    // pml
    std::vector<torch::Tensor> pml_vals;

    // time
    unsigned int nt;
    float dt;

    // grid
    std::vector<float> spacing;

    // options
    bool free_surface;
    bool boundary_on_cpu = false;
    bool use_pinned_memory = false;
    int transfer_interval = 1; // Transfer every time step by default

};

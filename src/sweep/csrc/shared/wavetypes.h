#pragma once

#include <torch/extension.h>
#include <string>
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
    std::vector<std::string> boundary_disk_files; // Bind from python
    std::vector<torch::Tensor> checkpoints; // Bind from python
    torch::Tensor checkpoint_steps;

    bool save_all_wavefields;
    bool use_boundary_saving;
    bool use_checkpoint = false;
    bool use_recursive_checkpoint = false;
    bool checkpoint_on_cpu = false;
    bool boundary_on_cpu = false;
    bool boundary_on_disk = false;
    bool boundary_disk_async_read = false;
    bool use_pinned_memory = false;
    bool free_surface;

    // Irregular free-surface topography (image method / vacuum staircase).
    // ``topo_rows`` is a 1-D ``int32`` tensor of length nx_runtime giving
    // the surface row index per column in runtime (PML-padded) coords; any
    // cell with ``iz < topo_rows[ix]`` is air.  Empty + ``has_topo=false``
    // for the flat / no-topo case.
    torch::Tensor topo_rows;
    bool has_topo = false;

    // APM (Cao & Chen 2018) per-cell category, runtime-padded int32 tensor
    // shape ``(nz_runtime, nx_runtime)``.  Empty unless ``use_apm`` is set.
    // Codes: INTERIOR=0, AIR=1, H=2, VL=3, VR=4, OC=5, IC=6 (see Python
    // ``sweep.equations._topography``).
    torch::Tensor topo_category;
    bool use_apm = false;

    unsigned int nt;
    float dt;

    std::vector<float> spacing;

    int transfer_interval = 1; // Transfer every time step by default
    int boundary_ring_buffers = 1;
    int checkpoint_interval = 1;
    int checkpoint_count = 0;
};

struct ForwardOutput {

    torch::Tensor wavefield;

    torch::Tensor last_two;

    torch::Tensor record;
};

struct BackwardOutput {

    std::vector<torch::Tensor> checkpoints;

    std::vector<torch::Tensor> grads;

    torch::Tensor source_illumination;

    torch::Tensor receiver_illumination;
};

struct RTMOutput {

    torch::Tensor image;

    torch::Tensor source_illumination;

    torch::Tensor receiver_illumination;
};

struct BackwardInput {

    // forward wavefield (used in normal backward)
    torch::Tensor u_forward;

    // boundary-saving data (used in backward_bs)
    std::vector<torch::Tensor> u_boundary;
    torch::Tensor u_last_two;
    std::vector<torch::Tensor> checkpoints;
    torch::Tensor checkpoint_steps;

    // Wavefields
    std::vector<torch::Tensor> adjoint_wavefields; // Bind from python
    std::vector<torch::Tensor> forward_wavefields; // Bind from python
    std::vector<torch::Tensor> adjoint_workspace; // Bind from python

    // Wavefields
    std::vector<torch::Tensor> boundary_cpu; // Bind from python
    std::vector<torch::Tensor> boundary_gpu; // Bind from python
    std::vector<std::string> boundary_disk_files; // Bind from python

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
    // See ForwardInput for semantics.  Mirrored by the propagator.
    torch::Tensor topo_rows;
    bool has_topo = false;
    torch::Tensor topo_category;
    bool use_apm = false;
    bool checkpoint_on_cpu = false;
    bool boundary_on_cpu = false;
    bool boundary_on_disk = false;
    bool boundary_disk_async_read = false;
    bool use_pinned_memory = false;
    int transfer_interval = 1; // Transfer every time step by default
    int boundary_ring_buffers = 1;
    int checkpoint_interval = 1;
    int checkpoint_count = 0;

    // When false, skip the per-timestep source/receiver illumination (RTM-image)
    // accumulation in backward(): it is not needed for a plain FWI vp gradient
    // and costs ~1/3 of the backward.  The vp gradient (calculate_grad) is
    // unaffected.  Default true = unchanged behaviour.
    bool compute_illumination = true;

};

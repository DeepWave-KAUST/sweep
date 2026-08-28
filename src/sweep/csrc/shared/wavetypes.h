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

    // Per-edge free surface / PML thickness (see SolverContext).  ``fs_faces``
    // is a bitmask (bit 2*axis = low face, 2*axis+1 = high; axis 0=z,1=y,2=x);
    // ``pad_lo``/``pad_hi`` are per-axis PML widths in C axis order [z,(y,)x]
    // (free-surface faces = 0).  Defaults (-1 / empty) => legacy single
    // ``free_surface`` (z-min only) + uniform ``abcn``.
    int fs_faces = -1;
    std::vector<int> pad_lo;
    std::vector<int> pad_hi;

    // Equation-specific auxiliary input tensors, opaque to the shared
    // plumbing (packed by the equation's ``c_eq_aux`` hook, mirrored to the
    // backward by the autograd wrapper).  visco_acoustic2d: {|k| grid}
    // (nz_runtime, nx_runtime float32) — present iff amplitude damping is
    // active.  Empty for every other equation.
    std::vector<torch::Tensor> eq_aux;

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
    // Boundary tail truncation (steady-state / frequency-selection FWI): when
    // > 0, only the LAST boundary_tail_steps time steps have their boundary
    // strips saved (forward) and back-propagated (backward).  The forward
    // physics is unchanged -- only the range of saved/reconstructed steps
    // shrinks.  0 = disabled (bit-exact legacy behaviour).  Both directions
    // MUST share the same value or the reconstruction misaligns.
    int boundary_tail_steps = 0;
    int checkpoint_interval = 1;
    int checkpoint_count = 0;

    // Stepped forward: run the time loop over [it_begin, it_end) instead of
    // [0, nt). it_end == -1 means nt. All per-step indexing stays absolute,
    // so repeated calls over consecutive ranges reproduce one full run
    // bit-for-bit (the caller re-binds wavefields with roles rotated by the
    // cumulative step count, see sweep/propagator/_stepped.py).
    int it_begin = 0;
    int it_end = -1;
    // Python-bound full-nt outputs so absolute-it writes from successive
    // stepped calls accumulate in one tensor. Undefined => internal
    // per-call allocation (legacy behaviour, full runs only).
    torch::Tensor record_out;   // (N, nrec, nt)
    torch::Tensor u_allt_out;   // (nt, B, nz[, ny], nx) when save_all_wavefields

    // ---- DD phase-split step (SPECFEM-style comm/compute overlap) ----
    // step_phase: 0 = legacy full step;
    //   1 = boundary phase: stencil (+ air-clear if topo) over the
    //       cut-adjacent M-wide physical edge strips ONLY — no boundary
    //       saving, no source, no record, no swap, no checkpoint;
    //   2 = interior phase: stencil over the strict complement of the
    //       phase-1 strips inside the normal compute region, then the full
    //       legacy tail (boundary saving, add_source, record, swap_pml,
    //       checkpoint).
    // Phases require a single step (it_end == it_begin + 1) and
    // cut_face_mask != 0.  acoustic2d/acoustic3d CUDA-only.
    int step_phase = 0;
    // Cut-face bitmask, same convention as BackwardInput::cut_face_mask:
    // bit0 = x_lo, bit1 = x_hi.  v1 supports x faces only.
    int cut_face_mask = 0;
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

    // Angle-domain common-image gather cube (space-lag extended imaging
    // condition).  Shape (nlag, N, C, nz, nx[, ny]) on the runtime-padded grid;
    // undefined unless compute_adcig was requested.  See BackwardInput below.
    torch::Tensor adcig;
};

struct RTMOutput {

    torch::Tensor image;

    torch::Tensor source_illumination;

    torch::Tensor receiver_illumination;

    // Space-lag ADCIG cube; see BackwardOutput::adcig.
    torch::Tensor adcig;
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

    // Equation-specific auxiliary tensors (see ForwardInput::eq_aux).
    std::vector<torch::Tensor> eq_aux;

    // time
    unsigned int nt;
    float dt;

    // grid
    std::vector<float> spacing;

    // options
    bool free_surface;
    // Per-edge free surface / PML thickness (see SolverContext / ForwardInput).
    int fs_faces = -1;
    std::vector<int> pad_lo;
    std::vector<int> pad_hi;
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
    int boundary_tail_steps = 0;   // see ForwardInput::boundary_tail_steps
    int checkpoint_interval = 1;
    int checkpoint_count = 0;

    // When false, skip the per-timestep source/receiver illumination (RTM-image)
    // accumulation in backward(): it is not needed for a plain FWI vp gradient
    // and costs ~1/3 of the backward.  The vp gradient (calculate_grad) is
    // unaffected.  Default true = unchanged behaviour.
    bool compute_illumination = true;
    // ---- stepped backward segment ----
    // A segment is the half-open interval [bw_it_end, bw_it_begin) of FORWARD
    // time indices, processed in DESCENDING order:
    //     it = bw_it_begin - 1, bw_it_begin - 2, ..., bw_it_end.
    // Segments must partition [0, nt) and be issued in descending order:
    //   * the FIRST segment issued has bw_it_begin == nt   (triggers seeding)
    //   * the segment with bw_it_end == 0 is the LAST       (runs the it==0 tail)
    // Defaults reproduce the monolithic call.  All per-it indexing stays
    // absolute, so repeated calls over consecutive descending ranges
    // reproduce one full backward bit-for-bit (the caller re-binds the
    // adjoint/reconstruction wavefield lists with roles rotated by the
    // cumulative step counts, see sweep/propagator/_stepped.py).
    int bw_it_begin = -1;   // -1 sentinel => nt
    int bw_it_end = 0;

    // Backward phase split (elastic backward_bs only; CUDA-only):
    //   0 = legacy full iteration;
    //   1 = adjoint-source inject + stress reconstruction (+restore) +
    //       gradient accumulation + STRESS-adjoint (prepare/apply);
    //   2 = VELOCITY-adjoint (prepare/apply) + fv_prev capture + velocity
    //       reconstruction (+restore).
    // Between 1 and 2 a DD driver exchanges the adjoint velocity halo and
    // the reconstruction stress halo; after 2 it exchanges the adjoint
    // stress halo and the reconstruction velocity halo (each M wide).
    // Phases require a single-step segment (bw_it_begin == bw_it_end + 1)
    // and Python-bound adjoint/recon/grads state.
    int step_phase = 0;

    // ---- bound accumulators (allocation moves to Python when stepping) ----
    // If non-empty, the driver accumulates "+=" into these instead of
    // allocating torch::zeros internally, and does NOT zero them.  Python
    // zeroes them once before the first segment.
    // grads_out order matches BackwardOutput.grads: slot 0 = grad_wavelet
    // (zeros_like(forward_source), raw CUDA layout (B,nsrc,nt)), slots
    // 1..N = model grads (zeros_like(models[i])).
    std::vector<torch::Tensor> grads_out;
    // illum_out = {source_illumination, receiver_illumination}, shapes
    // identical to what init_rtm_output_{2d,3d} allocates.
    std::vector<torch::Tensor> illum_out;

    // ---- DD cut faces ----
    // Bitmask of tile faces that are interior cuts (a neighbour tile
    // exists there):
    //   bit0 = x_lo (ix=0 side), bit1 = x_hi, bit2 = z_lo (top),
    //   bit3 = z_hi, bit4 = y_lo, bit5 = y_hi (3D only).
    // 0 = no cuts (single domain, legacy behaviour bit-identical).
    // On a cut face the backward_bs path (Design B):
    //   * skips the boundary-strip restore and the seed rim-zeroing, and
    //   * collapses the NOPML exclusion band / fused-adjoint pure_interior
    //     bound to the stencil halo M,
    // so cut-adjacent cells are computed by exactly the same kernel
    // expression as single-domain interior cells; the driver supplies the
    // M-halo of the reconstruction and adjoint u_now via per-step exchange.
    int cut_face_mask = 0;

    int bw_begin() const { return bw_it_begin < 0 ? static_cast<int>(nt) : bw_it_begin; }
    bool bw_stepped() const { return bw_begin() < static_cast<int>(nt) || bw_it_end > 0; }

    // Space-lag ADCIG (Sava & Fomel 2003): accumulate the horizontal
    // subsurface-offset extended imaging condition
    //   E(z,x,h) = sum_t u_s(x-h,t) * u_r(x+h,t),  h in [-adcig_max_lag, +adcig_max_lag]
    // into a (nlag=2*adcig_max_lag+1, N, C, nz, nx[, ny]) cube.  Off by default;
    // like compute_illumination it is an extra per-timestep grid pass and only
    // runs when requested.  The lag is along the contiguous x axis.
    bool compute_adcig = false;
    int adcig_max_lag = 0;

};

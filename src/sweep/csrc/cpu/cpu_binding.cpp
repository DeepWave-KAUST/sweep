#include "cpu_binding.h"

#include "equations/acoustic2d/acoustic2d_cpu.h"
#include "equations/acoustic3d/acoustic3d_cpu.h"
#include "equations/acoustic_lsrtm2d/acoustic_lsrtm2d_cpu.h"
#include "equations/acoustic_lsrtm3d/acoustic_lsrtm3d_cpu.h"
#include "equations/acoustic_vrz2d/acoustic_vrz2d_cpu.h"
#include "equations/acoustic_vrz3d/acoustic_vrz3d_cpu.h"
#include "equations/das2d/das2d_cpu.h"
#include "equations/das3d/das3d_cpu.h"
#include "equations/das_mu2d/das_mu2d_cpu.h"
#include "equations/das_mu3d/das_mu3d_cpu.h"
#include "equations/elastic2d/elastic2d_cpu.h"
#include "equations/elastic3d/elastic3d_cpu.h"
#include "equations/elastic_tti_sg2d/elastic_tti_sg2d_cpu.h"

#include <torch/extension.h>

#include <algorithm>

namespace sweep_cpu {

bool is_cpu_input(const ForwardInput& in)
{
    return engine::is_cpu_input(in);
}

bool is_cpu_input(const BackwardInput& in)
{
    return engine::is_cpu_input(in);
}

namespace {

// The Python side reads what it BOUND, not what an entry returns (51871c0f,
// "entries return nothing"): record_out, u_allt_out and last_two for a
// forward; grads_out, illum_out and adcig_out for a backward.  The CUDA core
// writes into those; the CPU engine below allocates its own outputs and
// returns them, which the caller then drops -- every CPU record, gradient
// and illumination came back as the zeros Python bound.  These put the CPU
// engine's outputs where the caller reads them.  The sizes must agree
// exactly: a disagreement is a layout mismatch between the engine and the
// Python-side shape, and is raised rather than reshaped.
void check_same_sizes(const torch::Tensor& bound, const torch::Tensor& got, const char* what)
{
    TORCH_CHECK(bound.sizes() == got.sizes(),
                "CPU engine ", what, ": the engine produced ", got.sizes(),
                " but the caller bound ", bound.sizes());
}

// Forward outputs are overwritten (absolute-it writes over a full run).
void write_into(const torch::Tensor& bound, const torch::Tensor& got, const char* what)
{
    if (!bound.defined() || bound.numel() == 0) return;       // nothing bound: nobody reads it
    TORCH_CHECK(got.defined() && got.numel() > 0,
                "CPU engine ", what, ": the caller bound ", bound.sizes(),
                " but the engine produced nothing -- the compiled CPU engine keeps no "
                "forward state for this configuration, so it has no backward for it "
                "(a free surface, for one, falls to its generic forward). Use "
                "impl='eager' on the CPU, or a CUDA device.");
    if (got.is_same(bound)) return;                            // the engine wrote in place
    check_same_sizes(bound, got, what);
    bound.copy_(got);
}

// Backward accumulators are added into (BackwardInput: "the driver
// accumulates += into these ... and does NOT zero them").
void add_into(const torch::Tensor& bound, const torch::Tensor& got, const char* what)
{
    if (!bound.defined() || bound.numel() == 0) return;
    if (!got.defined() || got.numel() == 0) return;            // the engine has none: leave the bound zeros
    if (got.is_same(bound)) return;
    check_same_sizes(bound, got, what);
    bound.add_(got);
}

void bind_forward_outputs(const ForwardInput& in, const ForwardOutput& out)
{
    write_into(in.record_out, out.record, "record (ForwardInput.record_out)");
    if (in.save_all_wavefields)
        write_into(in.u_allt_out, out.wavefield, "forward history (ForwardInput.u_allt_out)");
    if (in.use_boundary_saving)
        write_into(in.last_two, out.last_two, "boundary-saving last_two (ForwardInput.last_two)");
}

void bind_backward_outputs(const BackwardInput& in, const BackwardOutput& out)
{
    if (!in.grads_out.empty()) {
        // grads_out is [grad_wavelet?, *model grads] (cuda_layout.grads_out_has_wavelet);
        // the engine returns the same list, or -- acoustic_vrz* -- the model
        // gradients alone under a declared-but-unwritten wavelet slot.  Line
        // them up from the END (the model gradients); any other count is a
        // layout disagreement.
        const size_t nb = in.grads_out.size(), ng = out.grads.size(), nm = in.models.size();
        const bool same = nb == ng;
        const bool models_only = nb == nm + 1 && ng == nm;       // bound wavelet slot left zero
        const bool extra_wavelet = nb == nm && ng == nm + 1;     // engine's wavelet grad not bound
        TORCH_CHECK(same || models_only || extra_wavelet,
                    "CPU engine gradients: the engine produced ", ng,
                    " gradients but the caller bound ", nb, " for ", nm,
                    " models (BackwardInput.grads_out)");
        const size_t n = std::min(nb, ng);
        for (size_t i = 0; i < n; ++i)
            add_into(in.grads_out[nb - n + i], out.grads[ng - n + i],
                     "gradient (BackwardInput.grads_out)");
    }
    if (in.illum_out.size() >= 1)
        add_into(in.illum_out[0], out.source_illumination, "source illumination (BackwardInput.illum_out[0])");
    if (in.illum_out.size() >= 2)
        add_into(in.illum_out[1], out.receiver_illumination, "receiver illumination (BackwardInput.illum_out[1])");
    add_into(in.adcig_out, out.adcig, "ADCIG cube (BackwardInput.adcig_out)");
}

ForwardOutput forward_unbound(const ForwardInput& in, EquationKind kind)
{
    switch (kind) {
        case EquationKind::Acoustic2D:
            return acoustic2d::forward(in);
        case EquationKind::Acoustic3D:
            return acoustic3d::forward(in);
        case EquationKind::AcousticLSRTM2D:
            return acoustic_lsrtm2d::forward(in);
        case EquationKind::AcousticLSRTM3D:
            return acoustic_lsrtm3d::forward(in);
        case EquationKind::AcousticVRZ2D:
            return acoustic_vrz2d::forward(in);
        case EquationKind::AcousticVRZ3D:
            return acoustic_vrz3d::forward(in);
        case EquationKind::Elastic2D:
            return elastic2d::forward(in);
        case EquationKind::Elastic3D:
            return elastic3d::forward(in);
        case EquationKind::ElasticTTISG2D:
            return elastic_tti_sg2d::forward(in);
        case EquationKind::DAS2D:
            return das2d::forward(in);
        case EquationKind::DAS3D:
            return das3d::forward(in);
        case EquationKind::DASMu2D:
            return das_mu2d::forward(in);
        case EquationKind::DASMu3D:
            return das_mu3d::forward(in);
    }
    TORCH_CHECK(false, "Unsupported CPU equation kind");
}

BackwardOutput backward_unbound(const BackwardInput& in, EquationKind kind, BackwardMode mode)
{
    switch (kind) {
        case EquationKind::Acoustic2D:
            if (mode == BackwardMode::BoundarySaving) return acoustic2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic2d::backward_recursive_ckpt(in);
            return acoustic2d::backward(in);
        case EquationKind::Acoustic3D:
            if (mode == BackwardMode::BoundarySaving) return acoustic3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic3d::backward_recursive_ckpt(in);
            return acoustic3d::backward(in);
        case EquationKind::AcousticLSRTM2D:
            if (mode == BackwardMode::BoundarySaving) return acoustic_lsrtm2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic_lsrtm2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic_lsrtm2d::backward_recursive_ckpt(in);
            return acoustic_lsrtm2d::backward(in);
        case EquationKind::AcousticLSRTM3D:
            if (mode == BackwardMode::BoundarySaving) return acoustic_lsrtm3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic_lsrtm3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic_lsrtm3d::backward_recursive_ckpt(in);
            return acoustic_lsrtm3d::backward(in);
        case EquationKind::AcousticVRZ2D:
            if (mode == BackwardMode::BoundarySaving) return acoustic_vrz2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic_vrz2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic_vrz2d::backward_recursive_ckpt(in);
            return acoustic_vrz2d::backward(in);
        case EquationKind::AcousticVRZ3D:
            if (mode == BackwardMode::BoundarySaving) return acoustic_vrz3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return acoustic_vrz3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return acoustic_vrz3d::backward_recursive_ckpt(in);
            return acoustic_vrz3d::backward(in);
        case EquationKind::Elastic2D:
            if (mode == BackwardMode::BoundarySaving) return elastic2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return elastic2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return elastic2d::backward_recursive_ckpt(in);
            return elastic2d::backward(in);
        case EquationKind::Elastic3D:
            if (mode == BackwardMode::BoundarySaving) return elastic3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return elastic3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return elastic3d::backward_recursive_ckpt(in);
            return elastic3d::backward(in);
        case EquationKind::ElasticTTISG2D:
            if (mode == BackwardMode::BoundarySaving) return elastic_tti_sg2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return elastic_tti_sg2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return elastic_tti_sg2d::backward_recursive_ckpt(in);
            return elastic_tti_sg2d::backward(in);
        case EquationKind::DAS2D:
            if (mode == BackwardMode::BoundarySaving) return das2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return das2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return das2d::backward_recursive_ckpt(in);
            return das2d::backward(in);
        case EquationKind::DAS3D:
            if (mode == BackwardMode::BoundarySaving) return das3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return das3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return das3d::backward_recursive_ckpt(in);
            return das3d::backward(in);
        case EquationKind::DASMu2D:
            if (mode == BackwardMode::BoundarySaving) return das_mu2d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return das_mu2d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return das_mu2d::backward_recursive_ckpt(in);
            return das_mu2d::backward(in);
        case EquationKind::DASMu3D:
            if (mode == BackwardMode::BoundarySaving) return das_mu3d::backward_bs(in);
            if (mode == BackwardMode::Checkpoint) return das_mu3d::backward_ckpt(in);
            if (mode == BackwardMode::RecursiveCheckpoint) return das_mu3d::backward_recursive_ckpt(in);
            return das_mu3d::backward(in);
    }
    TORCH_CHECK(false, "Unsupported CPU equation kind");
}

} // namespace

ForwardOutput forward(const ForwardInput& in, EquationKind kind)
{
    ForwardOutput out = forward_unbound(in, kind);
    bind_forward_outputs(in, out);
    return out;
}

BackwardOutput backward(const BackwardInput& in, EquationKind kind, BackwardMode mode)
{
    BackwardOutput out = backward_unbound(in, kind, mode);
    bind_backward_outputs(in, out);
    return out;
}

} // namespace sweep_cpu

#include "cpu_binding.h"

#include "equations/acoustic2d/acoustic2d_cpu.h"
#include "equations/acoustic3d/acoustic3d_cpu.h"
#include "equations/acoustic_lsrtm2d/acoustic_lsrtm2d_cpu.h"
#include "equations/acoustic_lsrtm3d/acoustic_lsrtm3d_cpu.h"
#include "equations/acoustic_vrz2d/acoustic_vrz2d_cpu.h"
#include "equations/acoustic_vrz3d/acoustic_vrz3d_cpu.h"
#include "equations/das2d/das2d_cpu.h"
#include "equations/das3d/das3d_cpu.h"
#include "equations/elastic2d/elastic2d_cpu.h"
#include "equations/elastic3d/elastic3d_cpu.h"
#include "equations/elastic_tti_sg2d/elastic_tti_sg2d_cpu.h"

#include <torch/extension.h>

namespace sweep_cpu {

bool is_cpu_input(const ForwardInput& in)
{
    return engine::is_cpu_input(in);
}

bool is_cpu_input(const BackwardInput& in)
{
    return engine::is_cpu_input(in);
}

ForwardOutput forward(const ForwardInput& in, EquationKind kind)
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
    }
    TORCH_CHECK(false, "Unsupported CPU equation kind");
}

BackwardOutput backward(const BackwardInput& in, EquationKind kind, BackwardMode mode)
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
    }
    TORCH_CHECK(false, "Unsupported CPU equation kind");
}

} // namespace sweep_cpu

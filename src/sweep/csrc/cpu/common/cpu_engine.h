#pragma once

#include "../../shared/wavetypes.h"

namespace sweep_cpu {

enum class EquationKind {
    Acoustic2D,
    Acoustic3D,
    AcousticLSRTM2D,
    AcousticLSRTM3D,
    AcousticVRZ2D,
    AcousticVRZ3D,
    Elastic2D,
    Elastic3D,
    ElasticTTISG2D,
    DAS2D,
    DAS3D,
};

namespace engine {

bool is_cpu_input(const ForwardInput& in);
bool is_cpu_input(const BackwardInput& in);

ForwardOutput forward(const ForwardInput& in, EquationKind kind);
BackwardOutput backward(const BackwardInput& in, EquationKind kind);
BackwardOutput backward_full(const BackwardInput& in, EquationKind kind);
BackwardOutput backward_bs(const BackwardInput& in, EquationKind kind);
BackwardOutput backward_ckpt(const BackwardInput& in, EquationKind kind);
BackwardOutput backward_recursive_ckpt(const BackwardInput& in, EquationKind kind);

} // namespace engine
} // namespace sweep_cpu

#pragma once

#include "common/cpu_engine.h"

namespace sweep_cpu {

enum class BackwardMode {
    Full,
    BoundarySaving,
    Checkpoint,
    RecursiveCheckpoint,
};

bool is_cpu_input(const ForwardInput& in);
bool is_cpu_input(const BackwardInput& in);

ForwardOutput forward(const ForwardInput& in, EquationKind kind);
BackwardOutput backward(const BackwardInput& in, EquationKind kind, BackwardMode mode);

} // namespace sweep_cpu

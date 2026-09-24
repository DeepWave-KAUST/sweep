#pragma once
// core/runner.h -- the stepped runners' torch-free interface: what the C API's
// runner handles wrap (core/capi.h) and what the torch shim adapts to the
// binding's IForwardRunner / IBackwardRunner (shared/wavetypes.h).  A runner is
// built once from a core input (its spans must stay valid for the runner's
// life: the caller keeps the backing storage) and run() per segment.
#include <memory>
#include "outputs.h"

struct IForwardRunnerCore {
    virtual ~IForwardRunnerCore() = default;
    virtual ForwardOutputCore run(int it_begin, int it_end, int step_phase) = 0;
    virtual int device_index() const = 0;
};

struct IBackwardRunnerCore {
    virtual ~IBackwardRunnerCore() = default;
    virtual BackwardOutputCore run(int bw_it_begin, int bw_it_end, int step_phase) = 0;
    virtual int device_index() const = 0;
};

using ForwardRunnerCorePtr = std::shared_ptr<IForwardRunnerCore>;
using BackwardRunnerCorePtr = std::shared_ptr<IBackwardRunnerCore>;

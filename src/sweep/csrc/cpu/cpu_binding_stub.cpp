// ---------------------------------------------------------------------------
// GPU-only build stub for sweep_cpu::*
//
// Built INSTEAD of the full cpu/equations/* tree when the user opts out of
// CPU compilation via the `SWEEP_SKIP_CPU=1` environment variable.  This
// file provides the minimum symbols that `bindings/module.cpp` links
// against, so the extension compiles and the CUDA forward path runs
// without ever touching the heavy `*_cpu.cpp` files (~19k lines total).
//
// Behaviour:
//   - `is_cpu_input(...)` always returns false → the dispatcher routes every
//     call to the CUDA path.
//   - `forward(...)`/`backward(...)` raise a TORCH_CHECK error message with
//     a clear remediation hint.  If a user passes CPU tensors by accident
//     they get a one-line error explaining what to do.
//
// This file is NOT compiled in the standard build; it is conditionally
// added to the source list by `build_config.get_sources()` when
// `SWEEP_SKIP_CPU=1`.
// ---------------------------------------------------------------------------

#include "cpu_binding.h"
#include <torch/extension.h>

namespace sweep_cpu {

bool is_cpu_input(const ForwardInput& /*in*/)  { return false; }
bool is_cpu_input(const BackwardInput& /*in*/) { return false; }

[[noreturn]] static void raise_skipped(const char* role)
{
    TORCH_CHECK(
        false,
        "sweep was built with SWEEP_SKIP_CPU=1, so the compiled CPU ",
        role,
        " path is not available. Either: (a) rebuild without SWEEP_SKIP_CPU, "
        "(b) move your input tensors to .cuda() before calling, or "
        "(c) use the Python/JAX backend (backend='torch', impl='eager') "
        "which does not need this binding."
    );
}

ForwardOutput forward(const ForwardInput& /*in*/, EquationKind /*kind*/)
{
    raise_skipped("forward");
}

BackwardOutput backward(const BackwardInput& /*in*/, EquationKind /*kind*/,
                        BackwardMode /*mode*/)
{
    raise_skipped("backward");
}

}  // namespace sweep_cpu

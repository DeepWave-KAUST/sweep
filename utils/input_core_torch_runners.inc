
// The torch side of a stepped runner (IForwardRunner / IBackwardRunner are the
// binding's interfaces, shared/wavetypes.h): it holds the torch input and the
// arena the core runner's spans point into, adapts once at construction and
// maps every run()'s outputs back by pointer identity.  R is a core runner
// (IForwardRunnerCore / IBackwardRunnerCore, core/runner.h).
template <class R>
struct TorchForwardRunner final : IForwardRunner {
    ForwardInput in_;
    InputArena arena_;
    R core_;
    explicit TorchForwardRunner(const ForwardInput& in) : in_(in), core_(adapt_input(in_, arena_)) {}
    ForwardOutput run(int it_begin, int it_end, int step_phase) override
    { return to_torch(core_.run(it_begin, it_end, step_phase), in_); }
    int device_index() const override { return core_.device_index(); }
};

template <class R>
struct TorchBackwardRunner final : IBackwardRunner {
    BackwardInput in_;
    InputArena arena_;
    R core_;
    explicit TorchBackwardRunner(const BackwardInput& in) : in_(in), core_(adapt_input(in_, arena_)) {}
    BackwardOutput run(int bw_it_begin, int bw_it_end, int step_phase) override
    { return to_torch(core_.run(bw_it_begin, bw_it_end, step_phase), in_); }
    int device_index() const override { return core_.device_index(); }
};

#pragma once
// core/outputs.h -- the torch-free twins of ForwardOutput / BackwardOutput /
// RTMOutput (shared/wavetypes.h).  Every buffer a driver hands back is one the
// propagator bound on the input struct (record_out, u_allt_out, last_two,
// grads_out, illum_out, adcig_out), so these carry descriptors and the torch
// side maps each one back to its tensor by pointer identity
// (cuda/common/adapt_inputs.h, tensor_of).
#include <initializer_list>
#include <type_traits>
#include <vector>
#include "buf.h"
#include "check.h"

// The gradients a backward hands back: one descriptor per bound grads_out slot,
// in BackwardOutput.grads order.  Fixed capacity so the struct is standard
// layout (it crosses the C boundary); the drivers assign it from a brace list
// or a std::vector<Buf>, the adapter walks it.
constexpr int SWEEP_MAX_GRADS = 32;   // elastic_tti_sg3d hands back 22
struct GradList {
    Buf items[SWEEP_MAX_GRADS];
    int64_t n = 0;
    GradList& operator=(std::initializer_list<Buf> l) { assign(l.begin(), l.end()); return *this; }
    GradList& operator=(const std::vector<Buf>& v) { assign(v.data(), v.data() + v.size()); return *this; }
    template <class It> void assign(It b, It e) { n = 0; for (It it = b; it != e; ++it) push_back(*it); }
    void push_back(const Buf& b)
    {
        SWEEP_CHECK(n < SWEEP_MAX_GRADS, "a backward hands back at most ", SWEEP_MAX_GRADS, " gradients");
        items[n++] = b;
    }
    int64_t size() const { return n; }
    bool empty() const { return n == 0; }
    const Buf& operator[](int64_t i) const { return items[i]; }
    Buf& operator[](int64_t i) { return items[i]; }
    const Buf* begin() const { return items; }
    const Buf* end() const { return items + n; }
};

struct ForwardOutputCore {
    Buf wavefield;   // u_allt_out when save_all_wavefields
    Buf last_two;    // the bound last_two when boundary saving is on
    Buf record;      // record_out
};

struct RTMOutputCore {
    Buf source_illumination;
    Buf receiver_illumination;
    Buf adcig;
};

struct BackwardOutputCore {
    GradList grads;
    Buf source_illumination;
    Buf receiver_illumination;
    Buf adcig;
};

// These cross the C boundary (core/capi.h): plain memory, no std:: members.
static_assert(std::is_standard_layout<GradList>::value, "GradList crosses the C boundary");
static_assert(std::is_standard_layout<ForwardOutputCore>::value, "ForwardOutputCore crosses the C boundary");
static_assert(std::is_standard_layout<RTMOutputCore>::value, "RTMOutputCore crosses the C boundary");
static_assert(std::is_standard_layout<BackwardOutputCore>::value, "BackwardOutputCore crosses the C boundary");

#pragma once
// The torch side's handle on a core boundary-staging session (core/capi.h):
// what ForwardInput/BackwardInput.boundary_session carries and what Python's
// BoundarySession object is.  The core never sees this type; the adapter hands
// the raw handle over as the core's BoundarySession*.
#include <stdexcept>
#include "../core/capi.h"

struct BoundarySessionHandle {
    void* h = nullptr;
    BoundarySessionHandle() : h(sweep_session_create()) {}
    ~BoundarySessionHandle() { if (h) sweep_session_destroy(h); }
    BoundarySessionHandle(const BoundarySessionHandle&) = delete;
    BoundarySessionHandle& operator=(const BoundarySessionHandle&) = delete;
    void finish()
    {
        char err[1024];
        if (sweep_session_finish(h, err, sizeof err)) throw std::runtime_error(err);
    }
    bool used() const { return sweep_session_used(h) != 0; }
};

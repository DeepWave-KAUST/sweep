#pragma once

#include <memory>

// Handle for a boundary-staging session that outlives a single forward/backward
// call.  Declared here (plain C++, no CUDA) so the pybind translation unit --
// compiled by the host compiler -- can bind it; the implementation lives in
// cuda/common/boundary/session.cu.
//
// See cuda/common/boundary/session.cuh for WHY this exists: under domain
// decomposition every time step is a separate call into the extension, so a
// per-call copy stream can never keep a transfer in flight across steps.
class BoundarySessionImpl;

class BoundarySession {
public:
    BoundarySession();
    ~BoundarySession();
    BoundarySession(const BoundarySession&) = delete;
    BoundarySession& operator=(const BoundarySession&) = delete;

    // Let every outstanding copy land and close the current phase.  Python
    // calls this after a DD time loop; the destructor repeats it.
    void finish();

    // False means no call site ever bound this session -- i.e. the staging
    // silently fell back to the per-call path.  Tests assert on it.
    bool used() const;

    BoundarySessionImpl* impl() const { return impl_.get(); }

private:
    std::unique_ptr<BoundarySessionImpl> impl_;
};

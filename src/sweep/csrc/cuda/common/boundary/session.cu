#include "session.cuh"

// The public handle is a thin pimpl over BoundarySessionImpl so that the pybind
// translation unit (host compiler) never sees CUDA types.

BoundarySession::BoundarySession()
    : impl_(std::make_unique<BoundarySessionImpl>())
{
}

BoundarySession::~BoundarySession() = default;

void BoundarySession::finish()
{
    if (impl_)
        impl_->finish();
}

bool BoundarySession::used() const
{
    return impl_ && impl_->used();
}

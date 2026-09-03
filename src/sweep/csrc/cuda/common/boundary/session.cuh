#pragma once

#include <optional>
#include <string>
#include <vector>

#include "runtime.cuh"
#include "../cudautils.h"
#include "../../../shared/boundary_session.h"

// A boundary-staging session that OUTLIVES a single forward/backward call.
//
// WHY THIS EXISTS.  Under domain decomposition the time loop lives in Python
// (``dd_propagator``: ``for it in range(nt): runner.run_to(it+1); exchange()``),
// so every time step is a separate call into this extension.  Both the copy
// stream (``AsyncCopyContext``) and the ``BoundaryRuntime``'s events used to be
// locals of that call, which means every step ended with
//   * ``cudaStreamDestroy(copy_stream)`` -- an implicit barrier, and
//   * the forward's trailing ``boundary_runtime.synchronize()``.
// A copy therefore could never still be in flight when the next step ran, so
// ``transfer_interval`` / ``ring_buffers`` overlapped nothing under DD:
// ``storage='cpu'`` measured ~15x slower than boundary-in-VRAM on the 2-16 Hz
// band (29,475 steps), against +22% for the SAME staging on a single tile with
// interval=32 / ring=4.  The knobs were not the problem; the per-step teardown
// was.
//
// Holding the stream and the events in a Python-owned session removes both
// barriers: the copy issued in step ``it`` may still be in flight while step
// ``it+1`` computes, and the synchronize moves to the end of the phase.
//
// SAFETY.  Forgetting to end a phase must not corrupt results, so ``bind()``
// synchronizes whenever the phase changes (forward -> backward) and the
// destructor synchronizes as well.  A session bound with a DIFFERENT staging
// configuration than the one it was created with raises rather than silently
// reusing mismatched events/rings.
class BoundarySessionImpl {
public:
    enum class Phase { None, Forward, Backward };

    BoundarySessionImpl() = default;
    BoundarySessionImpl(const BoundarySessionImpl&) = delete;
    BoundarySessionImpl& operator=(const BoundarySessionImpl&) = delete;

    ~BoundarySessionImpl() { finish(); }

    // Bind the per-call objects and return the persistent runtime.  The first
    // call builds the stream + events; later calls only re-point the runtime at
    // this call's saver / disk-file list.
    BoundaryRuntime& bind(
        Phase phase,
        EffectiveBoundarySaver& saver,
        int dim,
        bool use_boundary_saving,
        bool boundary_on_cpu,
        bool boundary_on_disk,
        bool boundary_disk_async_read,
        int transfer_interval,
        int ring_buffers,
        const std::vector<std::string>& disk_files)
    {
        const bool staged = boundary_on_cpu || boundary_on_disk;
        if (!rt_.has_value()) {
            cfg_ = Cfg{dim, use_boundary_saving, boundary_on_cpu, boundary_on_disk,
                       boundary_disk_async_read, transfer_interval, ring_buffers};
            async_.emplace(staged);
            rt_.emplace(saver, dim, use_boundary_saving, boundary_on_cpu,
                        boundary_on_disk, boundary_disk_async_read,
                        transfer_interval, ring_buffers, disk_files,
                        async_->compute_stream, async_->copy_stream);
        } else {
            Cfg want{dim, use_boundary_saving, boundary_on_cpu, boundary_on_disk,
                     boundary_disk_async_read, transfer_interval, ring_buffers};
            TORCH_CHECK(
                want == cfg_,
                "BoundarySession reused with a different staging configuration "
                "(dim/use_bs/cpu/disk/async_read/transfer_interval/ring_buffers). "
                "The persistent events and ring bookkeeping are sized for the "
                "first binding; make a new session instead of rebinding.");
            // A phase change means the previous phase's copies must land before
            // the next phase reads the same buffers.  Doing it here (rather
            // than trusting the caller to call finish()) keeps a forgotten
            // end-of-phase from silently corrupting the reconstruction.
            if (phase != phase_)
                rt_->synchronize();
            rt_->rebind(saver, disk_files);
        }
        phase_ = phase;
        used_ = true;
        return *rt_;
    }

    // End the current phase: let every outstanding copy land.  Python calls
    // this after the DD time loop; the destructor repeats it so an early exit
    // cannot leave a copy in flight.
    void finish()
    {
        if (rt_.has_value())
            rt_->synchronize();
        phase_ = Phase::None;
    }

    // BoundaryScope needs the stream context; it is only valid after the
    // first bind() built it.
    AsyncCopyContext& async_ctx()
    {
        TORCH_CHECK(async_.has_value(),
                    "BoundarySession::async_ctx() before the first bind()");
        return *async_;
    }

    // Diagnostics for the tests: a session that was never bound means the call
    // site silently fell back to the per-call path.
    bool used() const { return used_; }

private:
    struct Cfg {
        int dim = 0;
        bool use_bs = false, on_cpu = false, on_disk = false, disk_async = false;
        int transfer_interval = 0, ring_buffers = 0;
        bool operator==(const Cfg& o) const
        {
            return dim == o.dim && use_bs == o.use_bs && on_cpu == o.on_cpu
                && on_disk == o.on_disk && disk_async == o.disk_async
                && transfer_interval == o.transfer_interval
                && ring_buffers == o.ring_buffers;
        }
    };

    std::optional<AsyncCopyContext> async_;
    std::optional<BoundaryRuntime> rt_;
    Cfg cfg_;
    Phase phase_ = Phase::None;
    bool used_ = false;
};

// Call-site helper: use the caller's session when one was supplied, otherwise
// build the per-call stream + runtime exactly as before.  ``owns()`` tells the
// call site whether the trailing synchronize is its responsibility (per-call
// path) or the session's (persistent path).
class BoundaryScope {
public:
    BoundaryScope(
        BoundarySessionImpl* session,
        BoundarySessionImpl::Phase phase,
        EffectiveBoundarySaver& saver,
        int dim,
        bool use_boundary_saving,
        bool boundary_on_cpu,
        bool boundary_on_disk,
        bool boundary_disk_async_read,
        int transfer_interval,
        int ring_buffers,
        const std::vector<std::string>& disk_files)
    {
        const bool staged = boundary_on_cpu || boundary_on_disk;
        if (session != nullptr) {
            rt_ = &session->bind(phase, saver, dim, use_boundary_saving,
                                 boundary_on_cpu, boundary_on_disk,
                                 boundary_disk_async_read, transfer_interval,
                                 ring_buffers, disk_files);
            async_ = &session->async_ctx();
            owns_ = false;
        } else {
            local_async_.emplace(staged);
            local_rt_.emplace(saver, dim, use_boundary_saving, boundary_on_cpu,
                              boundary_on_disk, boundary_disk_async_read,
                              transfer_interval, ring_buffers, disk_files,
                              local_async_->compute_stream,
                              local_async_->copy_stream);
            rt_ = &*local_rt_;
            async_ = &*local_async_;
            owns_ = true;
        }
    }

    BoundaryRuntime& runtime() { return *rt_; }
    AsyncCopyContext& async() { return *async_; }
    bool owns() const { return owns_; }

private:
    std::optional<AsyncCopyContext> local_async_;
    std::optional<BoundaryRuntime> local_rt_;
    BoundaryRuntime* rt_ = nullptr;
    AsyncCopyContext* async_ = nullptr;
    bool owns_ = true;
};

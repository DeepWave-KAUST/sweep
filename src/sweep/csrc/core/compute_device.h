#pragma once
// core/compute_device.h -- which memory tier THIS core computes on.
//
// The same source builds two cores: libsweep_core.so for CUDA (nvcc) and the
// host core (g++ -DSWEEP_HOST_CORE, csrc/host/).  A ``Buf`` says where its
// bytes live (``is_cuda()``: CUDA memory or not -- that meaning never changes,
// the memcpy direction logic depends on it), but a binding check that used to
// ask "is this buffer on the GPU?" really asks "is this buffer where my kernels
// can reach it?".  That is this predicate.  Spelling the check as
// ``on_compute_device(b)`` instead of ``b.is_cuda()`` is what lets the pool /
// wavefield / staging checks keep guarding against a buffer on the wrong tier
// on BOTH cores without a second copy of the check.
//
// Host-staging options (boundary_on_cpu / boundary_on_disk / checkpoint_on_cpu
// / pinned memory) move bytes between the compute tier and the host; on the
// host core the two are the same tier, so the entry prologue refuses them
// (``refuse_host_staging``) rather than running a memcpy ring onto itself.
#include "buf.h"
#include "check.h"

namespace sweep {

#if defined(SWEEP_HOST_CORE)
inline constexpr bool kCoreIsCuda = false;
inline constexpr const char* kComputeDeviceName = "host";
#else
inline constexpr bool kCoreIsCuda = true;
inline constexpr const char* kComputeDeviceName = "CUDA";
#endif

// Where the kernels of this core can read and write.  An undefined Buf is
// "not on the compute device" on both cores (its is_cuda_ is false).
inline bool on_compute_device(const Buf& b) { return b.defined() && b.is_cuda() == kCoreIsCuda; }

// The entry prologue's guard against host/disk staging on the host core.  A
// template so the one function serves ForwardInputCore and BackwardInputCore
// (both carry the same four flags under the same names).
template <class Input>
inline void refuse_host_staging(const Input& p, const char* entry)
{
#if !defined(SWEEP_HOST_CORE)
    (void)p;
    (void)entry;
#else
    SWEEP_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                entry, ": host/disk boundary staging is a CUDA-core option; on the host core "
                "the compute device is the host -- use boundary storage 'gpu' (on the compute device)");
    SWEEP_CHECK(!p.checkpoint_on_cpu,
                entry, ": checkpoint_on_cpu is a CUDA-core option; on the host core the "
                "checkpoints already live on the compute device");
    SWEEP_CHECK(!p.use_pinned_memory,
                entry, ": pinned host memory is a CUDA-core staging option; nothing to pin on the host core");
#endif
}

}  // namespace sweep

// The cuFFT plan cache of the visco-acoustic spectral step (ViscoFFT,
// declared in kernels.cuh) and the binding that sizes the pool's work-area
// slot from it.  One instance per process: forward.cu and backward.cu share
// the plans through ViscoFFT::get.
//
// Why a plan of our own and not at::fft_fft2: ATen allocates the work area
// (at::empty of workspace_size() bytes) and the output on EVERY transform,
// and its plan cache is private to libtorch.  Building the plan with the
// parameters ATen's _exec_fft derives for our operands (see the class note in
// kernels.cuh) and issuing the same cufftXtExec on the same data reproduces
// its bits exactly; only the allocations go away.
//
// The plan is built with the cuFFT API directly.  An earlier version reached
// for at::native::detail::CuFFTConfig to do it, which worked but put a PRIVATE
// ATen header on this tree's compile path -- the only such dependency it had,
// and one that a torch minor bump can break silently.  Same three arguments,
// same plan; what guarantees the bits is the gate, not the provenance of the
// constructor.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <map>
#include <tuple>

#include "kernels.cuh"
#include "visco_acoustic2d.h"

// cuFFT status check.  ATen has one (at::native::CUFFT_CHECK) but it is in the
// private header this file exists to stop including.
#define VISCO_CUFFT_CHECK(call)                                                   \
    do {                                                                          \
        const cufftResult _st = (call);                                           \
        TORCH_CHECK(_st == CUFFT_SUCCESS, "cuFFT error ", static_cast<int>(_st),   \
                    " from " #call);                                              \
    } while (0)

namespace {

using PlanKey = std::tuple<int, int64_t, int64_t, int64_t>;   // (device, B, nz, nx)

std::mutex& plan_cache_mutex()
{
    static std::mutex m;
    return m;
}

std::map<PlanKey, std::shared_ptr<ViscoFFT>>& plan_cache()
{
    static std::map<PlanKey, std::shared_ptr<ViscoFFT>> cache;
    return cache;
}

}  // namespace

std::shared_ptr<ViscoFFT> ViscoFFT::get(c10::DeviceIndex device, int64_t B, int64_t nz, int64_t nx)
{
    TORCH_CHECK(B >= 1 && nz >= 1 && nx >= 1,
                "ViscoFFT: the transform geometry must be positive, got B=", B, " nz=", nz, " nx=", nx);
    std::lock_guard<std::mutex> lock(plan_cache_mutex());
    auto& cache = plan_cache();
    const PlanKey key(static_cast<int>(device), B, nz, nx);
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, std::shared_ptr<ViscoFFT>(new ViscoFFT(device, B, nz, nx))).first;
    return it->second;
}

ViscoFFT::ViscoFFT(c10::DeviceIndex device, int64_t B, int64_t nz, int64_t nx)
    : device_(device), B_(B), nz_(nz), nx_(nx)
{
    // _fft_normalization_scale(fft_norm_mode::by_n, sizes, dims = {-2, -1}):
    // signal_numel = prod(sizes[dim]) as int64, scale = 1.0 / double(signal_numel).
    int64_t signal_numel = 1;
    signal_numel *= nz;
    signal_numel *= nx;
    inverse_scale_ = 1.0 / static_cast<double>(signal_numel);

    // _exec_fft on a contiguous complex64 (B, 1, nz, nx) tensor over dims
    // {2, 3}: the batch dims (0, 1) come first and are collapsed, so the input
    // it plans for is the (B, nz, nx) contiguous reshape and the output the
    // (B, nz, nx) contiguous resize -- strides {nz*nx, nx, 1} on both sides,
    // signal_size {B, nz, nx}, fft_type C2C, value_type Float.
    // sizes {B, nz, nx} give CuFFTConfig batch = sizes[0] = B, signal_ndim =
    // sizes.size() - 1 = 2 and signal_sizes {nz, nx}; kFloat + C2C gives
    // itype = otype = exec_type = CUDA_C_32F.  Both operands are contiguous, so
    // as_cufft_embed() reports `simple` (stride == 1, dist == nz*nx == the
    // signal numel, embed.back() == nx) and CuFFTConfig takes the branch that
    // passes inembed == onembed == nullptr, which tells cuFFT to assume the
    // unit-stride layout and IGNORE istride/idist/ostride/odist.  Passing the
    // {nz*nx, nx, 1} strides as an explicit embedding instead would be the
    // other branch -- a different plan, and no longer a claim about the same
    // bits.  Auto-allocation stays off, as there, so the work area is ours and
    // comes from the pool.
    long long signal_sizes[2] = {static_cast<long long>(nz), static_cast<long long>(nx)};

    // cufftXtMakePlanMany binds the plan to the current device.
    c10::cuda::CUDAGuard guard(device_);
    VISCO_CUFFT_CHECK(cufftCreate(&plan_));
    VISCO_CUFFT_CHECK(cufftSetAutoAllocation(plan_, /*autoAllocate=*/0));
    size_t ws_size_t = 0;
    VISCO_CUFFT_CHECK(cufftXtMakePlanMany(
        plan_, /*rank=*/2, signal_sizes,
        /*inembed=*/nullptr, /*istride=*/1, /*idist=*/1, CUDA_C_32F,
        /*onembed=*/nullptr, /*ostride=*/1, /*odist=*/1, CUDA_C_32F,
        /*batch=*/static_cast<long long>(B), &ws_size_t, /*executiontype=*/CUDA_C_32F));
    ws_bytes_ = static_cast<int64_t>(ws_size_t);
}

ViscoFFT::~ViscoFFT()
{
    if (plan_ != 0) {
        // Unchecked and UNGUARDED, which is what ATen's ~CuFFTHandle does
        // (a bare cufftDestroy, no CUDAGuard, no CUFFT_CHECK).  Both parts
        // matter: a destructor must not throw, and the plan cache is a
        // function-local static, so this runs during static destruction --
        // where a CUDAGuard would call cudaSetDevice on a runtime that may
        // already be torn down, and c10 turning that into an exception inside
        // a destructor is std::terminate.  Deleting the guard here is not a
        // simplification; it removes an exit-time crash the previous code
        // (which destroyed the handle through ATen) never had.
        cufftDestroy(plan_);
        plan_ = 0;
    }
}

int64_t ViscoFFT::workspace_bytes() const
{
    return ws_bytes_;
}

void ViscoFFT::exec(const torch::Tensor& in, const torch::Tensor& out, const torch::Tensor& work_area,
                    bool forward)
{
    auto check = [&](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.defined() && t.is_cuda() && t.device().index() == device_,
                    "ViscoFFT: ", name, " must live on CUDA device ", static_cast<int>(device_),
                    " (the plan's device)");
        TORCH_CHECK(t.scalar_type() == torch::kComplexFloat && t.is_contiguous(),
                    "ViscoFFT: ", name, " must be a contiguous complex64 tensor, got ",
                    t.scalar_type());
        TORCH_CHECK(t.dim() >= 2 && t.size(-2) == nz_ && t.size(-1) == nx_
                        && t.numel() == B_ * nz_ * nx_,
                    "ViscoFFT: ", name, " has shape ", t.sizes(), " but the plan is for ",
                    B_, " x (", nz_, ", ", nx_, ")");
    };
    check(in, "in");
    check(out, "out");
    TORCH_CHECK(in.data_ptr() != out.data_ptr(),
                "ViscoFFT: the transform is out of place (in and out must be distinct slots)");
    TORCH_CHECK(work_area.defined() && work_area.is_cuda() && work_area.device().index() == device_
                    && work_area.is_contiguous(),
                "ViscoFFT: the work area must be a contiguous CUDA tensor on the plan's device");
    const int64_t have = static_cast<int64_t>(work_area.nbytes());
    TORCH_CHECK(have >= workspace_bytes(),
                "ViscoFFT: the work area holds ", have, " bytes but the plan needs ",
                workspace_bytes(), " (size the slot with visco_acoustic2d_fft_workspace_bytes)");

    // ATen's sequence (_exec_fft): stream, work area, exec.  The plan is
    // shared, so the three plan-state calls are serialised per plan.  ATen
    // additionally re-binds a primary context that exists but is not current
    // (via its NVRTC stub, which the pip wheels do not ship); our call sites
    // always run after runtime calls on this thread (the CUDAGuard, the
    // stencil launches), so a current context is guaranteed here.
    c10::cuda::CUDAGuard guard(device_);
    std::lock_guard<std::mutex> lock(mutex_);
    VISCO_CUFFT_CHECK(cufftSetStream(plan_, at::cuda::getCurrentCUDAStream()));
    VISCO_CUFFT_CHECK(cufftSetWorkArea(plan_, work_area.data_ptr()));
    VISCO_CUFFT_CHECK(cufftXtExec(plan_, in.data_ptr(), out.data_ptr(),
                                  forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

void ViscoFFT::forward(const torch::Tensor& in, const torch::Tensor& out, const torch::Tensor& work_area)
{
    exec(in, out, work_area, /*forward=*/true);
}

void ViscoFFT::inverse(const torch::Tensor& in, const torch::Tensor& out, const torch::Tensor& work_area)
{
    exec(in, out, work_area, /*forward=*/false);
    // _fft_apply_normalization: (scale == 1.0) ? self : self.mul_(scale), the
    // double Scalar form of Tensor::mul_.
    if (inverse_scale_ != 1.0)
        out.mul_(inverse_scale_);
}

namespace visco_acoustic2d {

size_t fft_workspace_bytes(int64_t B, int64_t nz, int64_t nx)
{
    const auto plan = ViscoFFT::get(c10::cuda::current_device(), B, nz, nx);
    return static_cast<size_t>(plan->workspace_bytes());
}

}  // namespace visco_acoustic2d

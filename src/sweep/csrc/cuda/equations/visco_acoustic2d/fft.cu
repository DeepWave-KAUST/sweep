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
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>

#include <map>
#include <tuple>

#include "kernels.cuh"
#include "visco_acoustic2d.h"

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
    const int64_t strides[3] = {nz * nx, nx, 1};
    const int64_t sizes[3] = {B, nz, nx};
    const at::native::detail::CuFFTParams params(
        c10::IntArrayRef(strides, 3), c10::IntArrayRef(strides, 3), c10::IntArrayRef(sizes, 3),
        at::native::detail::CuFFTTransformType::C2C, at::ScalarType::Float);
    // cufftXtMakePlanMany binds the plan to the current device.
    c10::cuda::CUDAGuard guard(device_);
    config_ = std::make_unique<at::native::detail::CuFFTConfig>(params);
}

int64_t ViscoFFT::workspace_bytes() const
{
    return config_->workspace_size();
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
    auto& plan = config_->plan();
    at::native::CUFFT_CHECK(cufftSetStream(plan, at::cuda::getCurrentCUDAStream()));
    at::native::CUFFT_CHECK(cufftSetWorkArea(plan, work_area.data_ptr()));
    at::native::CUFFT_CHECK(cufftXtExec(plan, in.data_ptr(), out.data_ptr(),
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

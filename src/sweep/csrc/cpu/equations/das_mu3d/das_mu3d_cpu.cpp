#include "das_mu3d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <torch/extension.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace sweep_cpu {
namespace {

using torch::indexing::None;
using torch::indexing::Slice;

struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

int ndim_of(EquationKind kind)
{
    switch (kind) {
        case EquationKind::Acoustic3D:
        case EquationKind::AcousticLSRTM3D:
        case EquationKind::AcousticVRZ3D:
        case EquationKind::Elastic3D:
        case EquationKind::DAS3D:
        case EquationKind::DASMu3D:
            return 3;
        default:
            return 2;
    }
}

bool is_acoustic_family(EquationKind kind)
{
    return kind == EquationKind::Acoustic2D || kind == EquationKind::Acoustic3D;
}

bool is_lsrtm_family(EquationKind kind)
{
    return kind == EquationKind::AcousticLSRTM2D || kind == EquationKind::AcousticLSRTM3D;
}

bool is_vrz_family(EquationKind kind)
{
    return kind == EquationKind::AcousticVRZ2D || kind == EquationKind::AcousticVRZ3D;
}

int spatial_dim(const torch::Tensor& t, int ndim, int axis_from_front)
{
    return static_cast<int>(t.dim()) - ndim + axis_from_front;
}

torch::Tensor roll_axis(const torch::Tensor& x, int shift, int dim)
{
    return at::roll(x, {shift}, {dim});
}

struct SpatialLayout {
    int64_t leading;
    int64_t nz;
    int64_t ny;
    int64_t nx;
    int64_t spatial;
    int64_t stride;
    int64_t axis_size;
};

SpatialLayout spatial_layout(const torch::Tensor& u, int ndim, int axis_from_front)
{
    TORCH_CHECK(u.is_contiguous(), "CPU finite-difference kernels require contiguous tensors");
    TORCH_CHECK(u.scalar_type() == torch::kFloat32, "CPU finite-difference kernels currently support float32 tensors");
    TORCH_CHECK(ndim == 2 || ndim == 3, "CPU finite-difference kernels support only 2D/3D tensors");
    TORCH_CHECK(u.dim() >= ndim, "Tensor rank is smaller than the requested spatial rank");

    SpatialLayout s;
    s.nx = u.size(-1);
    s.ny = ndim == 3 ? u.size(-2) : 1;
    s.nz = ndim == 3 ? u.size(-3) : u.size(-2);
    s.spatial = s.nz * s.ny * s.nx;
    s.leading = u.numel() / s.spatial;
    if (ndim == 2) {
        s.axis_size = axis_from_front == 0 ? s.nz : s.nx;
        s.stride = axis_from_front == 0 ? s.nx : 1;
    } else {
        s.axis_size = axis_from_front == 0 ? s.nz : (axis_from_front == 1 ? s.ny : s.nx);
        s.stride = axis_from_front == 0 ? s.ny * s.nx : (axis_from_front == 1 ? s.nx : 1);
    }
    return s;
}

int64_t axis_coord(int64_t spatial_idx, const SpatialLayout& s, int ndim, int axis_from_front)
{
    const int64_t x = spatial_idx % s.nx;
    if (ndim == 2) {
        const int64_t z = spatial_idx / s.nx;
        return axis_from_front == 0 ? z : x;
    }
    const int64_t y = (spatial_idx / s.nx) % s.ny;
    const int64_t z = spatial_idx / (s.nx * s.ny);
    if (axis_from_front == 0) return z;
    if (axis_from_front == 1) return y;
    return x;
}

using sweep_cpu::ops::centered_coeff;
using sweep_cpu::ops::centered_grad_value;
using sweep_cpu::ops::laplace_value;
using sweep_cpu::ops::staggered_coeff;
using sweep_cpu::ops::staggered_grad_value;

template <int M, typename Eval>
void apply_axis_raw(torch::Tensor& out, const torch::Tensor& u, int ndim, int axis_from_front, Eval eval)
{
    const auto s = spatial_layout(u, ndim, axis_from_front);
    const float* in = u.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    std::fill(dst, dst + out.numel(), 0.0f);
    const int64_t total = u.numel();
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t spatial_idx = linear % s.spatial;
            const int64_t coord = axis_coord(spatial_idx, s, ndim, axis_from_front);
            if (coord < M || coord >= s.axis_size - M) continue;
            dst[linear] = eval(in, linear, s.stride);
        }
    });
}

template <int M>
void apply_laplace_raw(torch::Tensor& out, const torch::Tensor& u, int ndim, int axis_from_front, double h)
{
    const float inv_h2 = static_cast<float>(1.0 / (h * h));
    apply_axis_raw<M>(out, u, ndim, axis_from_front, [inv_h2](const float* in, int64_t idx, int64_t stride) {
        return laplace_value<M>(in, idx, stride, inv_h2);
    });
}

template <int M>
void apply_centered_grad_raw(torch::Tensor& out, const torch::Tensor& u, int ndim, int axis_from_front, double h)
{
    const float inv_h = static_cast<float>(1.0 / h);
    apply_axis_raw<M>(out, u, ndim, axis_from_front, [inv_h](const float* in, int64_t idx, int64_t stride) {
        return centered_grad_value<M>(in, idx, stride, inv_h);
    });
}

template <int M, bool Forward>
void apply_staggered_grad_raw(torch::Tensor& out, const torch::Tensor& u, int ndim, int axis_from_front, double h)
{
    const float inv_h = static_cast<float>(1.0 / h);
    apply_axis_raw<M>(out, u, ndim, axis_from_front, [inv_h](const float* in, int64_t idx, int64_t stride) {
        return staggered_grad_value<M, Forward>(in, idx, stride, inv_h);
    });
}

template <int M>
void apply_centered_grad_adjoint_raw(torch::Tensor& out, const torch::Tensor& grad, int ndim, int axis_from_front, double h)
{
    auto s = spatial_layout(grad, ndim, axis_from_front);
    const float* g = grad.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    std::fill(dst, dst + out.numel(), 0.0f);
    const float inv_h = static_cast<float>(1.0 / h);
    const int64_t total = grad.numel();
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t spatial_idx = linear % s.spatial;
            const int64_t coord = axis_coord(spatial_idx, s, ndim, axis_from_front);
            float acc = 0.0f;
            for (int m = 1; m <= M; ++m) {
                const float c = centered_coeff<M>(m);
                const int64_t left = coord - m;
                const int64_t right = coord + m;
                if (left >= M && left < s.axis_size - M) acc += c * g[linear - m * s.stride];
                if (right >= M && right < s.axis_size - M) acc -= c * g[linear + m * s.stride];
            }
            dst[linear] = acc * inv_h;
        }
    });
}

template <int M, bool Forward>
void apply_staggered_grad_adjoint_raw(torch::Tensor& out, const torch::Tensor& grad, int ndim, int axis_from_front, double h)
{
    auto s = spatial_layout(grad, ndim, axis_from_front);
    const float* g = grad.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    std::fill(dst, dst + out.numel(), 0.0f);
    const float inv_h = static_cast<float>(1.0 / h);
    const int64_t total = grad.numel();
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t spatial_idx = linear % s.spatial;
            const int64_t coord = axis_coord(spatial_idx, s, ndim, axis_from_front);
            float acc = 0.0f;
            for (int m = 0; m < M; ++m) {
                const float c = staggered_coeff<M>(m + 1);
                int64_t plus_coord;
                int64_t minus_coord;
                if constexpr (Forward) {
                    plus_coord = coord - (m + 1);
                    minus_coord = coord + m;
                    if (plus_coord >= M && plus_coord < s.axis_size - M) acc += c * g[linear - (m + 1) * s.stride];
                    if (minus_coord >= M && minus_coord < s.axis_size - M) acc -= c * g[linear + m * s.stride];
                } else {
                    plus_coord = coord - m;
                    minus_coord = coord + (m + 1);
                    if (plus_coord >= M && plus_coord < s.axis_size - M) acc += c * g[linear - m * s.stride];
                    if (minus_coord >= M && minus_coord < s.axis_size - M) acc -= c * g[linear + (m + 1) * s.stride];
                }
            }
            dst[linear] = acc * inv_h;
        }
    });
}

class LaplaceAxisFn : public torch::autograd::Function<LaplaceAxisFn> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor u, int64_t ndim, int64_t axis, int64_t M, double h)
    {
        ctx->saved_data["ndim"] = ndim;
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["M"] = M;
        ctx->saved_data["h"] = h;
        auto out = torch::empty_like(u);
        if (M == 1) apply_laplace_raw<1>(out, u, ndim, axis, h);
        else if (M == 2) apply_laplace_raw<2>(out, u, ndim, axis, h);
        else if (M == 3) apply_laplace_raw<3>(out, u, ndim, axis, h);
        else apply_laplace_raw<4>(out, u, ndim, axis, h);
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs)
    {
        auto grad = grad_outputs[0].contiguous();
        auto out = torch::empty_like(grad);
        const auto ndim = ctx->saved_data["ndim"].toInt();
        const auto axis = ctx->saved_data["axis"].toInt();
        const auto M = ctx->saved_data["M"].toInt();
        const auto h = ctx->saved_data["h"].toDouble();
        if (M == 1) apply_laplace_raw<1>(out, grad, ndim, axis, h);
        else if (M == 2) apply_laplace_raw<2>(out, grad, ndim, axis, h);
        else if (M == 3) apply_laplace_raw<3>(out, grad, ndim, axis, h);
        else apply_laplace_raw<4>(out, grad, ndim, axis, h);
        return {out, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

class CenteredGradAxisFn : public torch::autograd::Function<CenteredGradAxisFn> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor u, int64_t ndim, int64_t axis, int64_t M, double h)
    {
        ctx->saved_data["ndim"] = ndim;
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["M"] = M;
        ctx->saved_data["h"] = h;
        auto out = torch::empty_like(u);
        if (M == 1) apply_centered_grad_raw<1>(out, u, ndim, axis, h);
        else if (M == 2) apply_centered_grad_raw<2>(out, u, ndim, axis, h);
        else if (M == 3) apply_centered_grad_raw<3>(out, u, ndim, axis, h);
        else apply_centered_grad_raw<4>(out, u, ndim, axis, h);
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs)
    {
        auto grad = grad_outputs[0].contiguous();
        auto out = torch::empty_like(grad);
        const auto ndim = ctx->saved_data["ndim"].toInt();
        const auto axis = ctx->saved_data["axis"].toInt();
        const auto M = ctx->saved_data["M"].toInt();
        const auto h = ctx->saved_data["h"].toDouble();
        if (M == 1) apply_centered_grad_adjoint_raw<1>(out, grad, ndim, axis, h);
        else if (M == 2) apply_centered_grad_adjoint_raw<2>(out, grad, ndim, axis, h);
        else if (M == 3) apply_centered_grad_adjoint_raw<3>(out, grad, ndim, axis, h);
        else apply_centered_grad_adjoint_raw<4>(out, grad, ndim, axis, h);
        return {out, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

template <bool Forward>
class StaggeredGradAxisFn : public torch::autograd::Function<StaggeredGradAxisFn<Forward>> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor u, int64_t ndim, int64_t axis, int64_t M, double h)
    {
        ctx->saved_data["ndim"] = ndim;
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["M"] = M;
        ctx->saved_data["h"] = h;
        auto out = torch::empty_like(u);
        if (M == 1) apply_staggered_grad_raw<1, Forward>(out, u, ndim, axis, h);
        else if (M == 2) apply_staggered_grad_raw<2, Forward>(out, u, ndim, axis, h);
        else if (M == 3) apply_staggered_grad_raw<3, Forward>(out, u, ndim, axis, h);
        else apply_staggered_grad_raw<4, Forward>(out, u, ndim, axis, h);
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs)
    {
        auto grad = grad_outputs[0].contiguous();
        auto out = torch::empty_like(grad);
        const auto ndim = ctx->saved_data["ndim"].toInt();
        const auto axis = ctx->saved_data["axis"].toInt();
        const auto M = ctx->saved_data["M"].toInt();
        const auto h = ctx->saved_data["h"].toDouble();
        if (M == 1) apply_staggered_grad_adjoint_raw<1, Forward>(out, grad, ndim, axis, h);
        else if (M == 2) apply_staggered_grad_adjoint_raw<2, Forward>(out, grad, ndim, axis, h);
        else if (M == 3) apply_staggered_grad_adjoint_raw<3, Forward>(out, grad, ndim, axis, h);
        else apply_staggered_grad_adjoint_raw<4, Forward>(out, grad, ndim, axis, h);
        return {out, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor laplace_axis_torch_fallback(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h
)
{
    const int dim = spatial_dim(u, ndim, axis_from_front);
    auto out = torch::zeros_like(u);
    for (int k = 1; k <= M; ++k) {
        out = out + coeff.index({k}) * (roll_axis(u, -k, dim) + roll_axis(u, k, dim));
    }
    out = out - coeff.index({0}) * u;
    return out / (h * h);
}

torch::Tensor centered_grad_axis_torch_fallback(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h
)
{
    const int dim = spatial_dim(u, ndim, axis_from_front);
    auto out = torch::zeros_like(u);
    for (int k = 1; k <= M; ++k) {
        out = out + coeff.index({k}) * (roll_axis(u, -k, dim) - roll_axis(u, k, dim));
    }
    return out / h;
}

torch::Tensor staggered_grad_axis_torch_fallback(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h,
    bool forward
)
{
    const int dim = spatial_dim(u, ndim, axis_from_front);
    auto out = torch::zeros_like(u);
    for (int m = 0; m < M; ++m) {
        if (forward) {
            out = out + coeff.index({m}) * (roll_axis(u, -(m + 1), dim) - roll_axis(u, m, dim));
        } else {
            out = out + coeff.index({m}) * (roll_axis(u, -m, dim) - roll_axis(u, m + 1, dim));
        }
    }
    return out / h;
}

bool can_use_raw_stencil(const torch::Tensor& u, int M)
{
    return u.is_contiguous() && u.scalar_type() == torch::kFloat32 && M >= 1 && M <= 4;
}

torch::Tensor laplace_axis(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h
)
{
    if (!can_use_raw_stencil(u, M)) {
        return laplace_axis_torch_fallback(u, ndim, axis_from_front, M, coeff, h);
    }
    return LaplaceAxisFn::apply(u, static_cast<int64_t>(ndim), static_cast<int64_t>(axis_from_front), static_cast<int64_t>(M), h);
}

torch::Tensor centered_grad_axis(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h
)
{
    if (!can_use_raw_stencil(u, M)) {
        return centered_grad_axis_torch_fallback(u, ndim, axis_from_front, M, coeff, h);
    }
    return CenteredGradAxisFn::apply(u, static_cast<int64_t>(ndim), static_cast<int64_t>(axis_from_front), static_cast<int64_t>(M), h);
}

torch::Tensor staggered_grad_axis(
    const torch::Tensor& u,
    int ndim,
    int axis_from_front,
    int M,
    const torch::Tensor& coeff,
    double h,
    bool forward
)
{
    if (!can_use_raw_stencil(u, M)) {
        return staggered_grad_axis_torch_fallback(u, ndim, axis_from_front, M, coeff, h, forward);
    }
    if (forward) {
        return StaggeredGradAxisFn<true>::apply(u, static_cast<int64_t>(ndim), static_cast<int64_t>(axis_from_front), static_cast<int64_t>(M), h);
    }
    return StaggeredGradAxisFn<false>::apply(u, static_cast<int64_t>(ndim), static_cast<int64_t>(axis_from_front), static_cast<int64_t>(M), h);
}

torch::Tensor interior_mask_like(const torch::Tensor& x, int ndim, int M)
{
    auto mask = torch::zeros_like(x);
    std::vector<torch::indexing::TensorIndex> indices;
    indices.reserve(x.dim());
    for (int i = 0; i < x.dim() - ndim; ++i) {
        indices.emplace_back(Slice());
    }
    for (int axis = 0; axis < ndim; ++axis) {
        const int dim = spatial_dim(x, ndim, axis);
        indices.emplace_back(Slice(M, x.size(dim) - M));
    }
    mask.index_put_(indices, 1.0);
    return mask;
}

torch::Tensor apply_mask(const torch::Tensor& x, const torch::Tensor& mask)
{
    return x * mask;
}

torch::Tensor slice_axis(const torch::Tensor& x, int dim, int64_t start, int64_t stop)
{
    std::vector<torch::indexing::TensorIndex> indices;
    indices.reserve(x.dim());
    for (int i = 0; i < x.dim(); ++i) {
        indices.emplace_back(i == dim ? Slice(start, stop) : Slice());
    }
    return x.index(indices);
}

torch::Tensor slice_axis_from(const torch::Tensor& x, int dim, int64_t start)
{
    std::vector<torch::indexing::TensorIndex> indices;
    indices.reserve(x.dim());
    for (int i = 0; i < x.dim(); ++i) {
        indices.emplace_back(i == dim ? Slice(start, None) : Slice());
    }
    return x.index(indices);
}

torch::Tensor extend_top_free_surface_2d(const torch::Tensor& u, int M, bool odd)
{
    if (M <= 0) return u;
    const int dim = spatial_dim(u, 2, 0);
    auto ghost = slice_axis(u, dim, M + 1, 2 * M + 1).flip({dim});
    if (odd) ghost = -ghost;
    auto body = slice_axis_from(u, dim, M);
    return torch::cat({ghost, body}, dim).contiguous();
}

torch::Tensor top_free_surface_staggered_grad_z_2d(
    const torch::Tensor& u,
    int M,
    const torch::Tensor& coeff,
    double h,
    bool forward,
    bool odd
)
{
    return staggered_grad_axis(extend_top_free_surface_2d(u, M, odd), 2, 0, M, coeff, h, forward);
}

torch::Tensor zero_top_free_surface_row_2d(const torch::Tensor& u, int M)
{
    if (M < 0) return u;
    const int dim = spatial_dim(u, 2, 0);
    auto out = u.clone();
    std::vector<torch::indexing::TensorIndex> indices;
    indices.reserve(out.dim());
    for (int i = 0; i < out.dim(); ++i) {
        if (i == dim) {
            indices.emplace_back(M);
        } else {
            indices.emplace_back(Slice());
        }
    }
    out.index_put_(indices, 0.0);
    return out;
}

torch::Tensor extend_top_free_surface_3d(const torch::Tensor& u, int M, bool odd)
{
    if (M <= 0) return u;
    const int dim = spatial_dim(u, 3, 0);
    auto ghost = slice_axis(u, dim, M + 1, 2 * M + 1).flip({dim});
    if (odd) ghost = -ghost;
    auto body = slice_axis_from(u, dim, M);
    return torch::cat({ghost, body}, dim).contiguous();
}

torch::Tensor top_free_surface_staggered_grad_z_3d(
    const torch::Tensor& u,
    int M,
    const torch::Tensor& coeff,
    double h,
    bool forward,
    bool odd
)
{
    return staggered_grad_axis(extend_top_free_surface_3d(u, M, odd), 3, 0, M, coeff, h, forward);
}

torch::Tensor zero_top_free_surface_row_3d(const torch::Tensor& u, int M)
{
    if (M < 0) return u;
    const int dim = spatial_dim(u, 3, 0);
    auto out = u.clone();
    std::vector<torch::indexing::TensorIndex> indices;
    indices.reserve(out.dim());
    for (int i = 0; i < out.dim(); ++i) {
        if (i == dim) {
            indices.emplace_back(M);
        } else {
            indices.emplace_back(Slice());
        }
    }
    out.index_put_(indices, 0.0);
    return out;
}

std::vector<double> spacing_for(const ForwardInput& p, int ndim)
{
    if (ndim == 2) {
        return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])}; // z, x
    }
    return {
        static_cast<double>(p.spacing[2]),
        static_cast<double>(p.spacing[1]),
        static_cast<double>(p.spacing[0]),
    }; // z, y, x
}

[[maybe_unused]] std::vector<double> spacing_for(const BackwardInput& p, int ndim)
{
    if (ndim == 2) {
        return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])}; // z, x
    }
    return {
        static_cast<double>(p.spacing[2]),
        static_cast<double>(p.spacing[1]),
        static_cast<double>(p.spacing[0]),
    }; // z, y, x
}

std::vector<torch::Tensor> acoustic_pml(const std::vector<torch::Tensor>& pml, int ndim)
{
    if (ndim == 2) {
        return {pml[0], pml[1], pml[2], pml[3], pml[4], pml[5]}; // az,bz,dbzdz,ax,bx,dbxdx
    }
    return {
        pml[0], pml[1], pml[2],
        pml[3], pml[4], pml[5],
        pml[6], pml[7], pml[8],
    };
}

std::vector<torch::Tensor> elastic_pml(const std::vector<torch::Tensor>& pml, int ndim)
{
    if (ndim == 2) {
        return {pml[0], pml[1], pml[2], pml[3], pml[4], pml[5], pml[6], pml[7]};
    }
    return {
        pml[0], pml[1], pml[2], pml[3],
        pml[4], pml[5], pml[6], pml[7],
        pml[8], pml[9], pml[10], pml[11],
    };
}

torch::Tensor add_sources(
    const torch::Tensor& field,
    const torch::Tensor& source,
    const torch::Tensor& locations,
    int it,
    int ndim
)
{
    const int64_t B = field.size(0);
    const int64_t nsrc = locations.size(1);
    auto flat_count = B * nsrc;
    auto device = field.device();
    auto batch = torch::arange(B, torch::TensorOptions().device(device).dtype(torch::kLong))
        .repeat_interleave(nsrc);
    auto channel = torch::zeros({flat_count}, torch::TensorOptions().device(device).dtype(torch::kLong));
    auto loc = locations.to(torch::kLong);
    auto values = source.index({Slice(), Slice(), it}).reshape({flat_count});
    auto add = torch::zeros_like(field);

    if (ndim == 2) {
        auto x = loc.index({Slice(), Slice(), 0}).reshape({flat_count});
        auto z = loc.index({Slice(), Slice(), 1}).reshape({flat_count});
        add.index_put_({batch, channel, z, x}, values, true);
    } else {
        auto x = loc.index({Slice(), Slice(), 0}).reshape({flat_count});
        auto y = loc.index({Slice(), Slice(), 1}).reshape({flat_count});
        auto z = loc.index({Slice(), Slice(), 2}).reshape({flat_count});
        add.index_put_({batch, channel, z, y, x}, values, true);
    }
    return field + add;
}

torch::Tensor sample_field(
    const torch::Tensor& field,
    const torch::Tensor& locations,
    int ndim
)
{
    const int64_t B = field.size(0);
    const int64_t nrec = locations.size(1);
    auto device = field.device();
    auto flat_count = B * nrec;
    auto batch = torch::arange(B, torch::TensorOptions().device(device).dtype(torch::kLong))
        .repeat_interleave(nrec);
    auto channel = torch::zeros({flat_count}, torch::TensorOptions().device(device).dtype(torch::kLong));
    auto loc = locations.to(torch::kLong);

    if (ndim == 2) {
        auto x = loc.index({Slice(), Slice(), 0}).reshape({flat_count});
        auto z = loc.index({Slice(), Slice(), 1}).reshape({flat_count});
        return field.index({batch, channel, z, x}).reshape({B, nrec});
    }

    auto x = loc.index({Slice(), Slice(), 0}).reshape({flat_count});
    auto y = loc.index({Slice(), Slice(), 1}).reshape({flat_count});
    auto z = loc.index({Slice(), Slice(), 2}).reshape({flat_count});
    return field.index({batch, channel, z, y, x}).reshape({B, nrec});
}

std::vector<int> tensor_to_ints(const torch::Tensor& t)
{
    auto cpu = t.to(torch::kCPU).contiguous();
    std::vector<int> out(cpu.numel());
    auto ptr = cpu.data_ptr<int>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        out[i] = ptr[i];
    }
    return out;
}

torch::Tensor field_by_id_2d(const std::vector<torch::Tensor>& fields, int id)
{
    if (id >= 0 && id < static_cast<int>(fields.size())) return fields[id];
    return torch::Tensor();
}

void set_field_by_id_2d(std::vector<torch::Tensor>& fields, int id, const torch::Tensor& value)
{
    if (id >= 0 && id < static_cast<int>(fields.size())) fields[id] = value;
}

std::vector<torch::Tensor> active_boundary_tensors(const ForwardInput& p)
{
    if (!p.boundary_cpu.empty()) return p.boundary_cpu;
    if (!p.boundary_gpu.empty()) return p.boundary_gpu;
    return {};
}

std::vector<torch::Tensor> active_boundary_tensors(const BackwardInput& p)
{
    if (!p.boundary_cpu.empty()) return p.boundary_cpu;
    if (!p.boundary_gpu.empty()) return p.boundary_gpu;
    if (!p.u_boundary.empty()) return p.u_boundary;
    return {};
}

void save_das_mu3d_boundaries(
    std::vector<torch::Tensor>& boundary,
    const std::vector<torch::Tensor>& fields,
    int64_t it,
    int M,
    int abcn,
    bool free_surface
)
{
    if (boundary.empty()) return;
    TORCH_CHECK(boundary.size() == 6, "DAS-Mu 3D CPU boundary saving expects 6 boundary tensors");
    const int64_t nz = fields[0].size(2);
    const int64_t ny = fields[0].size(3);
    const int64_t nx = fields[0].size(4);
    const int64_t width = M + 1;
    const int64_t x0 = abcn + M;
    const int64_t x1 = nx - abcn - M;
    const int64_t y0 = abcn + M;
    const int64_t y1 = ny - abcn - M;
    const int64_t z0 = free_surface ? M : abcn + M;
    const int64_t z1 = nz - abcn - M;
    const int64_t top_start = z0 - M;
    const int64_t bottom_start = z1 - 1;
    const int64_t front_start = y0 - M;
    const int64_t back_start = y1 - 1;
    const int64_t left_start = x0 - M;
    const int64_t right_start = x1 - 1;
    const int64_t nsave = std::min<int64_t>(15, fields.size());
    for (int64_t f = 0; f < nsave; ++f) {
        const int64_t slot = it * nsave + f;
        auto src = fields[f];
        boundary[0].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(top_start, top_start + width), Slice(y0, y1), Slice(x0, x1)}));
        boundary[1].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(bottom_start, bottom_start + width), Slice(y0, y1), Slice(x0, x1)}));
        boundary[2].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(z0, z1), Slice(front_start, front_start + width), Slice(x0, x1)}));
        boundary[3].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(z0, z1), Slice(back_start, back_start + width), Slice(x0, x1)}));
        boundary[4].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(z0, z1), Slice(y0, y1), Slice(left_start, left_start + width)}));
        boundary[5].index_put_({slot, Slice(), Slice(), Slice(), Slice()}, src.index({Slice(), 0, Slice(z0, z1), Slice(y0, y1), Slice(right_start, right_start + width)}));
    }
}

void restore_das_mu3d_boundary_field(
    const std::vector<torch::Tensor>& boundary,
    torch::Tensor& field,
    int64_t field_id,
    int64_t it,
    int M,
    int abcn,
    bool free_surface
)
{
    TORCH_CHECK(boundary.size() == 6, "DAS-Mu 3D CPU boundary-saving backward expects 6 boundary tensors");
    const int64_t nz = field.size(2);
    const int64_t ny = field.size(3);
    const int64_t nx = field.size(4);
    const int64_t width = M + 1;
    const int64_t x0 = abcn + M;
    const int64_t x1 = nx - abcn - M;
    const int64_t y0 = abcn + M;
    const int64_t y1 = ny - abcn - M;
    const int64_t z0 = free_surface ? M : abcn + M;
    const int64_t z1 = nz - abcn - M;
    const int64_t top_start = z0 - M;
    const int64_t bottom_start = z1 - 1;
    const int64_t front_start = y0 - M;
    const int64_t back_start = y1 - 1;
    const int64_t left_start = x0 - M;
    const int64_t right_start = x1 - 1;
    const int64_t slot = it * 15 + field_id;
    field.index_put_({Slice(), 0, Slice(top_start, top_start + width), Slice(y0, y1), Slice(x0, x1)}, boundary[0].index({slot, Slice(), Slice(), Slice(), Slice()}));
    field.index_put_({Slice(), 0, Slice(bottom_start, bottom_start + width), Slice(y0, y1), Slice(x0, x1)}, boundary[1].index({slot, Slice(), Slice(), Slice(), Slice()}));
    field.index_put_({Slice(), 0, Slice(z0, z1), Slice(front_start, front_start + width), Slice(x0, x1)}, boundary[2].index({slot, Slice(), Slice(), Slice(), Slice()}));
    field.index_put_({Slice(), 0, Slice(z0, z1), Slice(back_start, back_start + width), Slice(x0, x1)}, boundary[3].index({slot, Slice(), Slice(), Slice(), Slice()}));
    field.index_put_({Slice(), 0, Slice(z0, z1), Slice(y0, y1), Slice(left_start, left_start + width)}, boundary[4].index({slot, Slice(), Slice(), Slice(), Slice()}));
    field.index_put_({Slice(), 0, Slice(z0, z1), Slice(y0, y1), Slice(right_start, right_start + width)}, boundary[5].index({slot, Slice(), Slice(), Slice(), Slice()}));
}

CpuForwardResult forward_das_mu3d(const ForwardInput& p)
{
    auto like = p.models[0];
    auto h = spacing_for(p, 3);
    auto pml = elastic_pml(p.pml_vals, 3);
    auto mask = interior_mask_like(like, 3, p.M);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);
    std::vector<torch::Tensor> f(33, torch::zeros_like(like));
    auto source_fields = tensor_to_ints(p.source_field_indices);
    auto receiver_fields = tensor_to_ints(p.receiver_field_indices);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);
    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        u_allt = torch::zeros(
            {static_cast<int64_t>(p.nt), 3, like.size(0), like.size(2), like.size(3), like.size(4)},
            like.options()
        );
    }
    auto boundary = active_boundary_tensors(p);

    for (unsigned int it = 0; it < p.nt; ++it) {
        auto vx = f[0], vy = f[1], vz = f[2];
        auto sxx = f[3], syy = f[4], szz = f[5], sxy = f[6], sxz = f[7], syz = f[8];
        auto exx = f[9], eyy = f[10], ezz = f[11], exy = f[12], exz = f[13], eyz = f[14];
        auto m_vxx = f[15], m_vxy = f[16], m_vxz = f[17];
        auto m_vyx = f[18], m_vyy = f[19], m_vyz = f[20];
        auto m_vzx = f[21], m_vzy = f[22], m_vzz = f[23];
        auto m_sxxx = f[24], m_szzz = f[25], m_sxyx = f[26], m_sxyy = f[27];
        auto m_sxzx = f[28], m_sxzz = f[29], m_syyy = f[30], m_syzy = f[31], m_syzz = f[32];

        auto dsxx_dx = staggered_grad_axis(sxx, 3, 2, p.M, p.grad_coes, h[2], true);
        auto dsxy_dy = staggered_grad_axis(sxy, 3, 1, p.M, p.grad_coes, h[1], false);
        auto dsxz_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(sxz, p.M, p.grad_coes, h[0], false, true)
            : staggered_grad_axis(sxz, 3, 0, p.M, p.grad_coes, h[0], false);

        auto dsxy_dx = staggered_grad_axis(sxy, 3, 2, p.M, p.grad_coes, h[2], false);
        auto dsyy_dy = staggered_grad_axis(syy, 3, 1, p.M, p.grad_coes, h[1], true);
        auto dsyz_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(syz, p.M, p.grad_coes, h[0], false, true)
            : staggered_grad_axis(syz, 3, 0, p.M, p.grad_coes, h[0], false);

        auto dsxz_dx = staggered_grad_axis(sxz, 3, 2, p.M, p.grad_coes, h[2], false);
        auto dsyz_dy = staggered_grad_axis(syz, 3, 1, p.M, p.grad_coes, h[1], false);
        auto dszz_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(szz, p.M, p.grad_coes, h[0], true, true)
            : staggered_grad_axis(szz, 3, 0, p.M, p.grad_coes, h[0], true);

        m_szzz = pml[2] * m_szzz + pml[3] * dszz_dz;
        dszz_dz = dszz_dz + m_szzz;
        m_sxzx = pml[8] * m_sxzx + pml[9] * dsxz_dx;
        dsxz_dx = dsxz_dx + m_sxzx;

        m_sxzz = pml[0] * m_sxzz + pml[1] * dsxz_dz;
        dsxz_dz = dsxz_dz + m_sxzz;
        m_sxxx = pml[10] * m_sxxx + pml[11] * dsxx_dx;
        dsxx_dx = dsxx_dx + m_sxxx;

        m_sxyy = pml[4] * m_sxyy + pml[5] * dsxy_dy;
        dsxy_dy = dsxy_dy + m_sxyy;
        m_sxyx = pml[8] * m_sxyx + pml[9] * dsxy_dx;
        dsxy_dx = dsxy_dx + m_sxyx;

        m_syyy = pml[6] * m_syyy + pml[7] * dsyy_dy;
        dsyy_dy = dsyy_dy + m_syyy;
        m_syzz = pml[0] * m_syzz + pml[1] * dsyz_dz;
        dsyz_dz = dsyz_dz + m_syzz;
        m_syzy = pml[4] * m_syzy + pml[5] * dsyz_dy;
        dsyz_dy = dsyz_dy + m_syzy;

        vx = vx + p.dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz);
        vy = vy + p.dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz);
        vz = vz + p.dt / rho * (dsxz_dx + dsyz_dy + dszz_dz);

        auto dvx_dx = staggered_grad_axis(vx, 3, 2, p.M, p.grad_coes, h[2], false);
        auto dvx_dy = staggered_grad_axis(vx, 3, 1, p.M, p.grad_coes, h[1], true);
        auto dvx_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(vx, p.M, p.grad_coes, h[0], true, false)
            : staggered_grad_axis(vx, 3, 0, p.M, p.grad_coes, h[0], true);

        auto dvy_dx = staggered_grad_axis(vy, 3, 2, p.M, p.grad_coes, h[2], true);
        auto dvy_dy = staggered_grad_axis(vy, 3, 1, p.M, p.grad_coes, h[1], false);
        auto dvy_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(vy, p.M, p.grad_coes, h[0], true, false)
            : staggered_grad_axis(vy, 3, 0, p.M, p.grad_coes, h[0], true);

        auto dvz_dx = staggered_grad_axis(vz, 3, 2, p.M, p.grad_coes, h[2], true);
        auto dvz_dy = staggered_grad_axis(vz, 3, 1, p.M, p.grad_coes, h[1], true);
        auto dvz_dz = p.free_surface
            ? top_free_surface_staggered_grad_z_3d(vz, p.M, p.grad_coes, h[0], false, true)
            : staggered_grad_axis(vz, 3, 0, p.M, p.grad_coes, h[0], false);

        m_vzz = pml[0] * m_vzz + pml[1] * dvz_dz;
        dvz_dz = dvz_dz + m_vzz;
        m_vyy = pml[4] * m_vyy + pml[5] * dvy_dy;
        dvy_dy = dvy_dy + m_vyy;
        m_vxx = pml[8] * m_vxx + pml[9] * dvx_dx;
        dvx_dx = dvx_dx + m_vxx;

        m_vxz = pml[2] * m_vxz + pml[3] * dvx_dz;
        dvx_dz = dvx_dz + m_vxz;
        m_vzx = pml[10] * m_vzx + pml[11] * dvz_dx;
        dvz_dx = dvz_dx + m_vzx;

        m_vxy = pml[6] * m_vxy + pml[7] * dvx_dy;
        dvx_dy = dvx_dy + m_vxy;
        m_vyx = pml[10] * m_vyx + pml[11] * dvy_dx;
        dvy_dx = dvy_dx + m_vyx;
        m_vyz = pml[2] * m_vyz + pml[3] * dvy_dz;
        dvy_dz = dvy_dz + m_vyz;
        m_vzy = pml[6] * m_vzy + pml[7] * dvz_dy;
        dvz_dy = dvz_dy + m_vzy;

        auto div_v = dvx_dx + dvy_dy + dvz_dz;
        sxx = sxx + p.dt * (lambda * div_v + 2.0 * mu * dvx_dx);
        syy = syy + p.dt * (lambda * div_v + 2.0 * mu * dvy_dy);
        szz = szz + p.dt * (lambda * div_v + 2.0 * mu * dvz_dz);
        sxy = sxy + p.dt * mu * (dvx_dy + dvy_dx);
        sxz = sxz + p.dt * mu * (dvx_dz + dvz_dx);
        syz = syz + p.dt * mu * (dvy_dz + dvz_dy);

        exx = exx + p.dt * dvx_dx;
        eyy = eyy + p.dt * dvy_dy;
        ezz = ezz + p.dt * dvz_dz;
        exy = exy + 0.5 * p.dt * (dvx_dy + dvy_dx);
        exz = exz + 0.5 * p.dt * (dvx_dz + dvz_dx);
        eyz = eyz + 0.5 * p.dt * (dvy_dz + dvz_dy);

        if (p.free_surface) {
            szz = zero_top_free_surface_row_3d(szz, p.M);
            sxz = zero_top_free_surface_row_3d(sxz, p.M);
            syz = zero_top_free_surface_row_3d(syz, p.M);
        }

        f = {
            vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
            exx, eyy, ezz, exy, exz, eyz,
            m_vxx, m_vxy, m_vxz, m_vyx, m_vyy, m_vyz, m_vzx, m_vzy, m_vzz,
            m_sxxx, m_szzz, m_sxyx, m_sxyy, m_sxzx, m_sxzz, m_syyy, m_syzy, m_syzz,
        };
        for (auto& field : f) field = apply_mask(field, mask);

        if (u_allt.defined()) {
            u_allt.index_put_({static_cast<int64_t>(it), 0}, f[0].select(1, 0));
            u_allt.index_put_({static_cast<int64_t>(it), 1}, f[1].select(1, 0));
            u_allt.index_put_({static_cast<int64_t>(it), 2}, f[2].select(1, 0));
        }
        for (int id : source_fields) {
            if (id >= 0 && id < static_cast<int>(f.size())) {
                f[id] = add_sources(f[id], p.source, p.sources_loc, static_cast<int>(it), 3);
            }
        }
        if (p.use_boundary_saving) {
            save_das_mu3d_boundaries(boundary, f, static_cast<int64_t>(it), p.M, p.abcn, p.free_surface);
        }
        std::vector<torch::Tensor> per_field;
        for (int id : receiver_fields) {
            if (id >= 0 && id < static_cast<int>(f.size())) {
                per_field.push_back(sample_field(f[id], p.receivers_loc, 3));
            } else {
                per_field.push_back(torch::zeros({like.size(0), p.receivers_loc.size(1)}, like.options()));
            }
        }
        records.push_back(torch::stack(per_field, 0));
    }
    torch::Tensor last_two = p.last_two.defined() ? p.last_two : torch::empty({0}, like.options());
    if (p.use_boundary_saving) {
        if (!last_two.defined() || last_two.numel() == 0) {
            last_two = torch::zeros({15, 1, like.size(0), 1, like.size(2), like.size(3), like.size(4)}, like.options());
        }
        for (int fidx = 0; fidx < 15; ++fidx) {
            last_two.index_put_({fidx, 0, Slice(), 0}, f[fidx].select(1, 0));
        }
    }
    return {torch::stack(records, 3), u_allt, last_two};
}
torch::Tensor interior_halo_mask_like(const torch::Tensor& x, int ndim, int halo, int top_halo)
{
    auto mask = torch::zeros_like(x);
    if (ndim == 2) {
        mask.index_put_({Slice(), Slice(), Slice(top_halo, x.size(2) - halo), Slice(halo, x.size(3) - halo)}, 1.0);
    } else {
        mask.index_put_({Slice(), Slice(), Slice(top_halo, x.size(2) - halo), Slice(halo, x.size(3) - halo), Slice(halo, x.size(4) - halo)}, 1.0);
    }
    return mask;
}

std::vector<torch::Tensor> make_zero_fields_like(const torch::Tensor& like, int nfields)
{
    return std::vector<torch::Tensor>(nfields, torch::zeros_like(like));
}

void add_adjoint_sources(std::vector<torch::Tensor>& adj, const BackwardInput& p, int64_t it, int ndim)
{
    auto receiver_fields = tensor_to_ints(p.receiver_field_indices);
    for (int64_t f = 0; f < static_cast<int64_t>(receiver_fields.size()); ++f) {
        const int id = receiver_fields[f];
        if (id >= 0 && id < static_cast<int>(adj.size())) {
            adj[id] = add_sources(adj[id], p.adjoint_source.select(0, f), p.adjoint_sources_loc, static_cast<int>(it), ndim);
        }
    }
}

void remove_forward_sources(std::vector<torch::Tensor>& fwd, const BackwardInput& p, const torch::Tensor& neg_source, int64_t it, int ndim)
{
    auto source_fields = tensor_to_ints(p.source_field_indices);
    for (int id : source_fields) {
        if (id >= 0 && id < static_cast<int>(fwd.size())) {
            fwd[id] = add_sources(fwd[id], neg_source, p.forward_sources_loc, static_cast<int>(it), ndim);
        }
    }
}

void das_mu3d_accumulate_grad(
    const std::vector<torch::Tensor>& adj,
    const torch::Tensor& vx,
    const torch::Tensor& vy,
    const torch::Tensor& vz,
    const torch::Tensor& vx_next,
    const torch::Tensor& vy_next,
    const torch::Tensor& vz_next,
    const BackwardInput& p,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_vs,
    torch::Tensor& grad_rho
)
{
    auto h = spacing_for(p, 3);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto fvx_x = staggered_grad_axis(vx, 3, 2, p.M, p.grad_coes, h[2], false);
    auto fvx_y = staggered_grad_axis(vx, 3, 1, p.M, p.grad_coes, h[1], true);
    auto fvx_z = p.free_surface ? top_free_surface_staggered_grad_z_3d(vx, p.M, p.grad_coes, h[0], true, false)
                                : staggered_grad_axis(vx, 3, 0, p.M, p.grad_coes, h[0], true);
    auto fvy_x = staggered_grad_axis(vy, 3, 2, p.M, p.grad_coes, h[2], true);
    auto fvy_y = staggered_grad_axis(vy, 3, 1, p.M, p.grad_coes, h[1], false);
    auto fvy_z = p.free_surface ? top_free_surface_staggered_grad_z_3d(vy, p.M, p.grad_coes, h[0], true, false)
                                : staggered_grad_axis(vy, 3, 0, p.M, p.grad_coes, h[0], true);
    auto fvz_x = staggered_grad_axis(vz, 3, 2, p.M, p.grad_coes, h[2], true);
    auto fvz_y = staggered_grad_axis(vz, 3, 1, p.M, p.grad_coes, h[1], true);
    auto fvz_z = p.free_surface ? top_free_surface_staggered_grad_z_3d(vz, p.M, p.grad_coes, h[0], false, true)
                                : staggered_grad_axis(vz, 3, 0, p.M, p.grad_coes, h[0], false);
    auto bar_szz = adj[5], bar_sxz = adj[7], bar_syz = adj[8];
    if (p.free_surface) {
        bar_szz = zero_top_free_surface_row_3d(bar_szz, p.M);
        bar_sxz = zero_top_free_surface_row_3d(bar_sxz, p.M);
        bar_syz = zero_top_free_surface_row_3d(bar_syz, p.M);
    }
    auto div_v = fvx_x + fvy_y + fvz_z;
    auto grad_lambda = (adj[3] + adj[4] + bar_szz) * div_v;
    auto grad_mu = 2.0 * (adj[3] * fvx_x + adj[4] * fvy_y + bar_szz * fvz_z)
                 + adj[6] * (fvx_y + fvy_x)
                 + bar_sxz * (fvx_z + fvz_x)
                 + bar_syz * (fvy_z + fvz_y);
    grad_vp = grad_vp + (-2.0 * rho * vp * grad_lambda * p.dt);
    grad_vs = grad_vs - (-4.0 * rho * vs * grad_lambda + 2.0 * rho * vs * grad_mu) * p.dt;
    grad_rho = grad_rho + (adj[0] * (vx - vx_next) + adj[1] * (vy - vy_next) + adj[2] * (vz - vz_next)) / rho;
    grad_rho = grad_rho - grad_lambda * (vp * vp - 2.0 * vs * vs) * p.dt - grad_mu * (vs * vs) * p.dt;
}

void das_mu3d_adjoint_step(std::vector<torch::Tensor>& adj, const BackwardInput& p)
{
    auto h = spacing_for(p, 3);
    auto pml = elastic_pml(p.pml_vals, 3);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);
    auto l2m = lambda + 2.0 * mu;
    auto bar_szz = adj[5], bar_sxz = adj[7], bar_syz = adj[8];
    if (p.free_surface) {
        bar_szz = zero_top_free_surface_row_3d(bar_szz, p.M);
        bar_sxz = zero_top_free_surface_row_3d(bar_sxz, p.M);
        bar_syz = zero_top_free_surface_row_3d(bar_syz, p.M);
        adj[5] = zero_top_free_surface_row_3d(adj[5], p.M);
        adj[7] = zero_top_free_surface_row_3d(adj[7], p.M);
        adj[8] = zero_top_free_surface_row_3d(adj[8], p.M);
    }
    auto bar_dvx_dx = p.dt * (l2m * adj[3] + lambda * adj[4] + lambda * bar_szz - adj[9]);
    auto bar_dvy_dy = p.dt * (lambda * adj[3] + l2m * adj[4] + lambda * bar_szz - adj[10]);
    auto bar_dvz_dz = p.dt * (lambda * adj[3] + lambda * adj[4] + l2m * bar_szz - adj[11]);
    auto bar_dvx_dy = p.dt * (mu * adj[6] - 0.5 * adj[12]);
    auto bar_dvy_dx = p.dt * (mu * adj[6] - 0.5 * adj[12]);
    auto bar_dvx_dz = p.dt * (mu * bar_sxz - 0.5 * adj[13]);
    auto bar_dvz_dx = p.dt * (mu * bar_sxz - 0.5 * adj[13]);
    auto bar_dvy_dz = p.dt * (mu * bar_syz - 0.5 * adj[14]);
    auto bar_dvz_dy = p.dt * (mu * bar_syz - 0.5 * adj[14]);

    auto tmp_vxx = adj[15] + bar_dvx_dx;
    auto tmp_vxy = adj[16] + bar_dvx_dy;
    auto tmp_vxz = adj[17] + bar_dvx_dz;
    auto tmp_vyx = adj[18] + bar_dvy_dx;
    auto tmp_vyy = adj[19] + bar_dvy_dy;
    auto tmp_vyz = adj[20] + bar_dvy_dz;
    auto tmp_vzx = adj[21] + bar_dvz_dx;
    auto tmp_vzy = adj[22] + bar_dvz_dy;
    auto tmp_vzz = adj[23] + bar_dvz_dz;
    auto qxx = bar_dvx_dx + pml[9] * tmp_vxx;
    auto qxy = bar_dvx_dy + pml[7] * tmp_vxy;
    auto qxz = bar_dvx_dz + pml[3] * tmp_vxz;
    auto qyx = bar_dvy_dx + pml[11] * tmp_vyx;
    auto qyy = bar_dvy_dy + pml[5] * tmp_vyy;
    auto qyz = bar_dvy_dz + pml[3] * tmp_vyz;
    auto qzx = bar_dvz_dx + pml[11] * tmp_vzx;
    auto qzy = bar_dvz_dy + pml[7] * tmp_vzy;
    auto qzz = bar_dvz_dz + pml[1] * tmp_vzz;
    adj[15] = pml[8] * tmp_vxx;
    adj[16] = pml[6] * tmp_vxy;
    adj[17] = pml[2] * tmp_vxz;
    adj[18] = pml[10] * tmp_vyx;
    adj[19] = pml[4] * tmp_vyy;
    adj[20] = pml[2] * tmp_vyz;
    adj[21] = pml[10] * tmp_vzx;
    adj[22] = pml[6] * tmp_vzy;
    adj[23] = pml[0] * tmp_vzz;

    adj[0] = adj[0] + staggered_grad_axis(qxx, 3, 2, p.M, p.grad_coes, h[2], true)
                    + staggered_grad_axis(qxy, 3, 1, p.M, p.grad_coes, h[1], false)
                    + staggered_grad_axis(qxz, 3, 0, p.M, p.grad_coes, h[0], false);
    adj[1] = adj[1] + staggered_grad_axis(qyx, 3, 2, p.M, p.grad_coes, h[2], false)
                    + staggered_grad_axis(qyy, 3, 1, p.M, p.grad_coes, h[1], true)
                    + staggered_grad_axis(qyz, 3, 0, p.M, p.grad_coes, h[0], false);
    adj[2] = adj[2] + staggered_grad_axis(qzx, 3, 2, p.M, p.grad_coes, h[2], false)
                    + staggered_grad_axis(qzy, 3, 1, p.M, p.grad_coes, h[1], false)
                    + staggered_grad_axis(qzz, 3, 0, p.M, p.grad_coes, h[0], true);

    auto inv_rho = 1.0 / rho;
    auto bar_dsxx_dx = p.dt * inv_rho * adj[0];
    auto bar_dsxy_dy = p.dt * inv_rho * adj[0];
    auto bar_dsxz_dz = p.dt * inv_rho * adj[0];
    auto bar_dsxy_dx = p.dt * inv_rho * adj[1];
    auto bar_dsyy_dy = p.dt * inv_rho * adj[1];
    auto bar_dsyz_dz = p.dt * inv_rho * adj[1];
    auto bar_dsxz_dx = p.dt * inv_rho * adj[2];
    auto bar_dsyz_dy = p.dt * inv_rho * adj[2];
    auto bar_dszz_dz = p.dt * inv_rho * adj[2];
    auto tmp_sxxx = adj[24] + bar_dsxx_dx;
    auto tmp_sxyy = adj[27] + bar_dsxy_dy;
    auto tmp_sxzz = adj[29] + bar_dsxz_dz;
    auto tmp_sxyx = adj[26] + bar_dsxy_dx;
    auto tmp_syyy = adj[30] + bar_dsyy_dy;
    auto tmp_syzz = adj[32] + bar_dsyz_dz;
    auto tmp_sxzx = adj[28] + bar_dsxz_dx;
    auto tmp_syzy = adj[31] + bar_dsyz_dy;
    auto tmp_szzz = adj[25] + bar_dszz_dz;
    auto pxx = bar_dsxx_dx + pml[11] * tmp_sxxx;
    auto pxy = bar_dsxy_dy + pml[5] * tmp_sxyy;
    auto pxz = bar_dsxz_dz + pml[1] * tmp_sxzz;
    auto pyx = bar_dsxy_dx + pml[9] * tmp_sxyx;
    auto pyy = bar_dsyy_dy + pml[7] * tmp_syyy;
    auto pyz = bar_dsyz_dz + pml[1] * tmp_syzz;
    auto pzx = bar_dsxz_dx + pml[9] * tmp_sxzx;
    auto pzy = bar_dsyz_dy + pml[5] * tmp_syzy;
    auto pzz = bar_dszz_dz + pml[3] * tmp_szzz;
    adj[24] = pml[10] * tmp_sxxx;
    adj[27] = pml[4] * tmp_sxyy;
    adj[29] = pml[0] * tmp_sxzz;
    adj[26] = pml[8] * tmp_sxyx;
    adj[30] = pml[6] * tmp_syyy;
    adj[32] = pml[0] * tmp_syzz;
    adj[28] = pml[8] * tmp_sxzx;
    adj[31] = pml[4] * tmp_syzy;
    adj[25] = pml[2] * tmp_szzz;
    adj[3] = adj[3] + staggered_grad_axis(pxx, 3, 2, p.M, p.grad_coes, h[2], false);
    adj[6] = adj[6] + staggered_grad_axis(pxy, 3, 1, p.M, p.grad_coes, h[1], true)
                    + staggered_grad_axis(pyx, 3, 2, p.M, p.grad_coes, h[2], true);
    adj[7] = adj[7] + staggered_grad_axis(pxz, 3, 0, p.M, p.grad_coes, h[0], true)
                    + staggered_grad_axis(pzx, 3, 2, p.M, p.grad_coes, h[2], true);
    adj[4] = adj[4] + staggered_grad_axis(pyy, 3, 1, p.M, p.grad_coes, h[1], false);
    adj[8] = adj[8] + staggered_grad_axis(pyz, 3, 0, p.M, p.grad_coes, h[0], true)
                    + staggered_grad_axis(pzy, 3, 1, p.M, p.grad_coes, h[1], true);
    adj[5] = adj[5] + staggered_grad_axis(pzz, 3, 0, p.M, p.grad_coes, h[0], false);
}

void das_mu3d_reverse_stress_nopml(std::vector<torch::Tensor>& f, const BackwardInput& p)
{
    auto h = spacing_for(p, 3);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);
    int halo = p.abcn + p.M + 1;
    int top_halo = p.free_surface ? p.M : halo;
    auto mask = interior_halo_mask_like(vp, 3, halo, top_halo);
    auto dvx_dx = staggered_grad_axis(f[0], 3, 2, p.M, p.grad_coes, h[2], false);
    auto dvx_dy = staggered_grad_axis(f[0], 3, 1, p.M, p.grad_coes, h[1], true);
    auto dvx_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[0], p.M, p.grad_coes, h[0], true, false)
                                : staggered_grad_axis(f[0], 3, 0, p.M, p.grad_coes, h[0], true);
    auto dvy_dx = staggered_grad_axis(f[1], 3, 2, p.M, p.grad_coes, h[2], true);
    auto dvy_dy = staggered_grad_axis(f[1], 3, 1, p.M, p.grad_coes, h[1], false);
    auto dvy_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[1], p.M, p.grad_coes, h[0], true, false)
                                : staggered_grad_axis(f[1], 3, 0, p.M, p.grad_coes, h[0], true);
    auto dvz_dx = staggered_grad_axis(f[2], 3, 2, p.M, p.grad_coes, h[2], true);
    auto dvz_dy = staggered_grad_axis(f[2], 3, 1, p.M, p.grad_coes, h[1], true);
    auto dvz_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[2], p.M, p.grad_coes, h[0], false, true)
                                : staggered_grad_axis(f[2], 3, 0, p.M, p.grad_coes, h[0], false);
    auto div_v = dvx_dx + dvy_dy + dvz_dz;
    f[3] = f[3] - mask * p.dt * (lambda * div_v + 2.0 * mu * dvx_dx);
    f[4] = f[4] - mask * p.dt * (lambda * div_v + 2.0 * mu * dvy_dy);
    f[5] = f[5] - mask * p.dt * (lambda * div_v + 2.0 * mu * dvz_dz);
    f[6] = f[6] - mask * p.dt * mu * (dvx_dy + dvy_dx);
    f[7] = f[7] - mask * p.dt * mu * (dvx_dz + dvz_dx);
    f[8] = f[8] - mask * p.dt * mu * (dvy_dz + dvz_dy);
}

void das_mu3d_reverse_velocity_nopml(std::vector<torch::Tensor>& f, const BackwardInput& p)
{
    auto h = spacing_for(p, 3);
    auto rho = p.models[2];
    int halo = p.abcn + p.M + 1;
    int top_halo = p.free_surface ? p.M : halo;
    auto mask = interior_halo_mask_like(rho, 3, halo, top_halo);
    auto dsxx_dx = staggered_grad_axis(f[3], 3, 2, p.M, p.grad_coes, h[2], true);
    auto dsxy_dy = staggered_grad_axis(f[6], 3, 1, p.M, p.grad_coes, h[1], false);
    auto dsxz_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[7], p.M, p.grad_coes, h[0], false, true)
                                : staggered_grad_axis(f[7], 3, 0, p.M, p.grad_coes, h[0], false);
    auto dsxy_dx = staggered_grad_axis(f[6], 3, 2, p.M, p.grad_coes, h[2], false);
    auto dsyy_dy = staggered_grad_axis(f[4], 3, 1, p.M, p.grad_coes, h[1], true);
    auto dsyz_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[8], p.M, p.grad_coes, h[0], false, true)
                                : staggered_grad_axis(f[8], 3, 0, p.M, p.grad_coes, h[0], false);
    auto dsxz_dx = staggered_grad_axis(f[7], 3, 2, p.M, p.grad_coes, h[2], false);
    auto dsyz_dy = staggered_grad_axis(f[8], 3, 1, p.M, p.grad_coes, h[1], false);
    auto dszz_dz = p.free_surface ? top_free_surface_staggered_grad_z_3d(f[5], p.M, p.grad_coes, h[0], true, true)
                                : staggered_grad_axis(f[5], 3, 0, p.M, p.grad_coes, h[0], true);
    f[0] = f[0] - mask * p.dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz);
    f[1] = f[1] - mask * p.dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz);
    f[2] = f[2] - mask * p.dt / rho * (dsxz_dx + dsyz_dy + dszz_dz);
}

BackwardOutput backward_das_mu3d_full_manual(const BackwardInput& p, bool skip_initial_time = false)
{
    TORCH_CHECK(p.u_forward.defined() && p.u_forward.numel() > 0, "DAS-Mu 3D full CPU backward requires saved forward velocities");
    auto like = p.models[0];
    auto grad_vp = torch::zeros_like(like);
    auto grad_vs = torch::zeros_like(like);
    auto grad_rho = torch::zeros_like(like);
    auto adj = make_zero_fields_like(like, 33);
    auto zero = torch::zeros_like(like);
    const int64_t min_it = skip_initial_time ? 1 : 0;
    for (int64_t it = static_cast<int64_t>(p.nt) - 1; it >= min_it; --it) {
        add_adjoint_sources(adj, p, it, 3);
        auto vx = p.u_forward.index({it, 0}).unsqueeze(1);
        auto vy = p.u_forward.index({it, 1}).unsqueeze(1);
        auto vz = p.u_forward.index({it, 2}).unsqueeze(1);
        auto vx_next = (it + 1 < static_cast<int64_t>(p.nt)) ? p.u_forward.index({it + 1, 0}).unsqueeze(1) : zero;
        auto vy_next = (it + 1 < static_cast<int64_t>(p.nt)) ? p.u_forward.index({it + 1, 1}).unsqueeze(1) : zero;
        auto vz_next = (it + 1 < static_cast<int64_t>(p.nt)) ? p.u_forward.index({it + 1, 2}).unsqueeze(1) : zero;
        das_mu3d_accumulate_grad(adj, vx, vy, vz, vx_next, vy_next, vz_next, p, grad_vp, grad_vs, grad_rho);
        if (it > 0) das_mu3d_adjoint_step(adj, p);
    }
    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_das_mu3d_bs_manual(const BackwardInput& p)
{
    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0, "DAS-Mu 3D boundary-saving CPU backward requires last_two");
    auto boundary = active_boundary_tensors(p);
    TORCH_CHECK(!boundary.empty(), "DAS-Mu 3D boundary-saving CPU backward requires saved boundaries");
    auto like = p.models[0];
    auto grad_vp = torch::zeros_like(like);
    auto grad_vs = torch::zeros_like(like);
    auto grad_rho = torch::zeros_like(like);
    auto adj = make_zero_fields_like(like, 33);
    auto fwd = make_zero_fields_like(like, 33);
    for (int i = 0; i < 15; ++i) fwd[i] = p.u_last_two.index({i, 0, Slice(), 0}).unsqueeze(1).clone();
    auto vx_next = torch::zeros_like(like);
    auto vy_next = torch::zeros_like(like);
    auto vz_next = torch::zeros_like(like);
    auto neg_source = -p.forward_source;
    for (int64_t it = static_cast<int64_t>(p.nt) - 1; it >= 1; --it) {
        add_adjoint_sources(adj, p, it, 3);
        remove_forward_sources(fwd, p, neg_source, it, 3);
        das_mu3d_reverse_stress_nopml(fwd, p);
        for (int f = 3; f < 9; ++f) restore_das_mu3d_boundary_field(boundary, fwd[f], f, it - 1, p.M, p.abcn, p.free_surface);
        das_mu3d_accumulate_grad(adj, fwd[0], fwd[1], fwd[2], vx_next, vy_next, vz_next, p, grad_vp, grad_vs, grad_rho);
        das_mu3d_adjoint_step(adj, p);
        vx_next = fwd[0].clone();
        vy_next = fwd[1].clone();
        vz_next = fwd[2].clone();
        das_mu3d_reverse_velocity_nopml(fwd, p);
        for (int f = 0; f < 3; ++f) restore_das_mu3d_boundary_field(boundary, fwd[f], f, it - 1, p.M, p.abcn, p.free_surface);
    }
    add_adjoint_sources(adj, p, 0, 3);
    das_mu3d_accumulate_grad(adj, fwd[0], fwd[1], fwd[2], vx_next, vy_next, vz_next, p, grad_vp, grad_vs, grad_rho);
    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_das_mu3d_autograd_replay(const BackwardInput& in)
{
    TORCH_CHECK(!in.models.empty() && in.models[0].device().is_cpu(), "DAS-Mu 3D CPU backward called with non-CPU tensors");
    torch::AutoGradMode enable_grad(true);

    ForwardInput fwd;
    fwd.models.reserve(in.models.size());
    for (const auto& model : in.models) {
        auto leaf = model.detach().clone();
        leaf.set_requires_grad(true);
        fwd.models.push_back(leaf);
    }
    fwd.source = in.forward_source.detach().clone();
    fwd.source.set_requires_grad(true);
    fwd.lap_coes = in.lap_coes;
    fwd.grad_coes = in.grad_coes;
    fwd.M = in.M;
    fwd.abcn = in.abcn;
    fwd.sources_loc = in.forward_sources_loc;
    fwd.receivers_loc = in.adjoint_sources_loc;
    fwd.source_field_indices = in.source_field_indices;
    fwd.receiver_field_indices = in.receiver_field_indices;
    fwd.pml_vals = in.pml_vals;
    fwd.free_surface = in.free_surface;
    fwd.nt = in.nt;
    fwd.dt = in.dt;
    fwd.spacing = in.spacing;
    fwd.save_all_wavefields = false;
    fwd.use_boundary_saving = false;
    fwd.use_checkpoint = false;
    fwd.use_recursive_checkpoint = false;

    auto record = forward_das_mu3d(fwd).record;
    std::vector<torch::Tensor> inputs;
    inputs.push_back(fwd.source);
    inputs.insert(inputs.end(), fwd.models.begin(), fwd.models.end());

    std::vector<torch::Tensor> grads;
    auto grad_out = in.adjoint_source;
    torch::autograd::variable_list outputs{record};
    torch::autograd::variable_list grad_outputs{grad_out};
    {
        pybind11::gil_scoped_release no_gil;
        grads = torch::autograd::grad(outputs, inputs, grad_outputs, false, false, false);
    }

    BackwardOutput out;
    out.grads = grads;
    out.source_illumination = torch::zeros_like(in.models[0]);
    out.receiver_illumination = torch::zeros_like(in.models[0]);
    return out;
}

} // namespace

namespace das_mu3d {

ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(!in.models.empty() && in.models[0].device().is_cpu(), "DAS-Mu 3D CPU forward called with non-CPU tensors");
    auto result = forward_das_mu3d(in);
    ForwardOutput out;
    out.record = result.record;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(!in.models.empty() && in.models[0].device().is_cpu(), "DAS-Mu 3D CPU backward called with non-CPU tensors");
    return backward_das_mu3d_full_manual(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    TORCH_CHECK(!in.models.empty() && in.models[0].device().is_cpu(), "DAS-Mu 3D CPU boundary-saving backward called with non-CPU tensors");
    return backward_das_mu3d_bs_manual(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return backward_das_mu3d_autograd_replay(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return backward_das_mu3d_autograd_replay(in);
}

} // namespace das_mu3d
} // namespace sweep_cpu

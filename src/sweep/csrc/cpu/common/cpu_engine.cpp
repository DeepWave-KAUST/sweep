#include "cpu_engine.h"

#include "../operators/fd.h"

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

torch::Tensor helix35(const torch::Tensor& exx, const torch::Tensor& eyy, const torch::Tensor& ezz)
{
    return exx + eyy + ezz;
}

torch::Tensor helix54(const torch::Tensor& exx, const torch::Tensor& eyy, const torch::Tensor& ezz, int core)
{
    if (core == 0) return 4.0 * exx + eyy + ezz;
    if (core == 1) return exx + 4.0 * eyy + ezz;
    return exx + eyy + 4.0 * ezz;
}

struct AcousticState2D {
    torch::Tensor u_prev, u_now, psix, psiz, zetax, zetaz;
};

AcousticState2D make_acoustic_state_2d(const torch::Tensor& like)
{
    return {
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
    };
}


AcousticState2D acoustic_step_2d(
    const AcousticState2D& s,
    const torch::Tensor& vp,
    const ForwardInput& p,
    const torch::Tensor& mask
)
{
    auto h = spacing_for(p, 2);
    auto pml = acoustic_pml(p.pml_vals, 2);
    auto lap_z = laplace_axis(s.u_now, 2, 0, p.M, p.lap_coes, h[0]);
    auto lap_x = laplace_axis(s.u_now, 2, 1, p.M, p.lap_coes, h[1]);
    auto dudz = centered_grad_axis(s.u_now, 2, 0, p.M, p.grad_coes, h[0]);
    auto dudx = centered_grad_axis(s.u_now, 2, 1, p.M, p.grad_coes, h[1]);
    auto dpsizdz = centered_grad_axis(pml[0] * s.psiz, 2, 0, p.M, p.grad_coes, h[0]);
    auto dpsixdx = centered_grad_axis(pml[3] * s.psix, 2, 1, p.M, p.grad_coes, h[1]);

    auto tmpz = (1.0 + pml[1]) * lap_z + pml[2] * dudz + dpsizdz;
    auto tmpx = (1.0 + pml[4]) * lap_x + pml[5] * dudx + dpsixdx;
    auto psiz = pml[1] * dudz + pml[0] * s.psiz;
    auto psix = pml[4] * dudx + pml[3] * s.psix;
    auto zetaz = pml[1] * tmpz + pml[0] * s.zetaz;
    auto zetax = pml[4] * tmpx + pml[3] * s.zetax;
    auto wsum = (1.0 + pml[1]) * tmpz + pml[0] * s.zetaz
              + (1.0 + pml[4]) * tmpx + pml[3] * s.zetax;
    auto u_next = 2.0 * s.u_now - s.u_prev + vp * vp * (p.dt * p.dt) * wsum;
    return {
        apply_mask(s.u_now, mask),
        apply_mask(u_next, mask),
        apply_mask(psix, mask),
        apply_mask(psiz, mask),
        apply_mask(zetax, mask),
        apply_mask(zetaz, mask),
    };
}

struct AcousticState3D {
    torch::Tensor u_prev, u_now, psix, psiy, psiz, zetax, zetay, zetaz;
};

AcousticState3D make_acoustic_state_3d(const torch::Tensor& like)
{
    return {
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
        torch::zeros_like(like),
    };
}

AcousticState3D acoustic_step_3d(
    const AcousticState3D& s,
    const torch::Tensor& vp,
    const ForwardInput& p,
    const torch::Tensor& mask
)
{
    auto h = spacing_for(p, 3);
    auto pml = acoustic_pml(p.pml_vals, 3);
    auto lap_z = laplace_axis(s.u_now, 3, 0, p.M, p.lap_coes, h[0]);
    auto lap_y = laplace_axis(s.u_now, 3, 1, p.M, p.lap_coes, h[1]);
    auto lap_x = laplace_axis(s.u_now, 3, 2, p.M, p.lap_coes, h[2]);
    auto dudz = centered_grad_axis(s.u_now, 3, 0, p.M, p.grad_coes, h[0]);
    auto dudy = centered_grad_axis(s.u_now, 3, 1, p.M, p.grad_coes, h[1]);
    auto dudx = centered_grad_axis(s.u_now, 3, 2, p.M, p.grad_coes, h[2]);
    auto tmpz = (1.0 + pml[1]) * lap_z + pml[2] * dudz + centered_grad_axis(pml[0] * s.psiz, 3, 0, p.M, p.grad_coes, h[0]);
    auto tmpy = (1.0 + pml[4]) * lap_y + pml[5] * dudy + centered_grad_axis(pml[3] * s.psiy, 3, 1, p.M, p.grad_coes, h[1]);
    auto tmpx = (1.0 + pml[7]) * lap_x + pml[8] * dudx + centered_grad_axis(pml[6] * s.psix, 3, 2, p.M, p.grad_coes, h[2]);
    auto psiz = pml[1] * dudz + pml[0] * s.psiz;
    auto psiy = pml[4] * dudy + pml[3] * s.psiy;
    auto psix = pml[7] * dudx + pml[6] * s.psix;
    auto zetaz = pml[1] * tmpz + pml[0] * s.zetaz;
    auto zetay = pml[4] * tmpy + pml[3] * s.zetay;
    auto zetax = pml[7] * tmpx + pml[6] * s.zetax;
    auto wsum = (1.0 + pml[1]) * tmpz + pml[0] * s.zetaz
              + (1.0 + pml[4]) * tmpy + pml[3] * s.zetay
              + (1.0 + pml[7]) * tmpx + pml[6] * s.zetax;
    auto u_next = 2.0 * s.u_now - s.u_prev + vp * vp * (p.dt * p.dt) * wsum;
    return {
        apply_mask(s.u_now, mask),
        apply_mask(u_next, mask),
        apply_mask(psix, mask),
        apply_mask(psiy, mask),
        apply_mask(psiz, mask),
        apply_mask(zetax, mask),
        apply_mask(zetay, mask),
        apply_mask(zetaz, mask),
    };
}

CpuForwardResult forward_acoustic(const ForwardInput& p, EquationKind kind)
{
    const int ndim = ndim_of(kind);
    auto vp = p.models[0];
    auto mask = interior_mask_like(vp, ndim, p.M);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    if (ndim == 2) {
        auto s = make_acoustic_state_2d(vp);
        for (unsigned int it = 0; it < p.nt; ++it) {
            s = acoustic_step_2d(s, vp, p, mask);
            s.u_now = add_sources(s.u_now, p.source, p.sources_loc, static_cast<int>(it), 2);
            records.push_back(sample_field(s.u_now, p.receivers_loc, 2));
        }
    } else {
        auto s = make_acoustic_state_3d(vp);
        for (unsigned int it = 0; it < p.nt; ++it) {
            s = acoustic_step_3d(s, vp, p, mask);
            s.u_now = add_sources(s.u_now, p.source, p.sources_loc, static_cast<int>(it), 3);
            records.push_back(sample_field(s.u_now, p.receivers_loc, 3));
        }
    }
    return {torch::stack(records, 2)};
}

CpuForwardResult forward_lsrtm(const ForwardInput& p, EquationKind kind)
{
    const int ndim = ndim_of(kind);
    auto vp = p.models[0];
    auto mp = p.models[1];
    auto mask = interior_mask_like(vp, ndim, p.M);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    if (ndim == 2) {
        auto bg = make_acoustic_state_2d(vp);
        auto sc = make_acoustic_state_2d(vp);
        for (unsigned int it = 0; it < p.nt; ++it) {
            auto bg_prev = bg;
            bg = acoustic_step_2d(bg, vp, p, mask);
            sc = acoustic_step_2d(sc, vp, p, mask);
            sc.u_now = sc.u_now + mp * (bg.u_now - 2.0 * bg_prev.u_now + bg_prev.u_prev);
            bg.u_now = add_sources(bg.u_now, p.source, p.sources_loc, static_cast<int>(it), 2);
            records.push_back(sample_field(sc.u_now, p.receivers_loc, 2));
        }
    } else {
        auto bg = make_acoustic_state_3d(vp);
        auto sc = make_acoustic_state_3d(vp);
        for (unsigned int it = 0; it < p.nt; ++it) {
            auto bg_prev = bg;
            bg = acoustic_step_3d(bg, vp, p, mask);
            sc = acoustic_step_3d(sc, vp, p, mask);
            sc.u_now = sc.u_now + mp * (bg.u_now - 2.0 * bg_prev.u_now + bg_prev.u_prev);
            bg.u_now = add_sources(bg.u_now, p.source, p.sources_loc, static_cast<int>(it), 3);
            records.push_back(sample_field(sc.u_now, p.receivers_loc, 3));
        }
    }
    return {torch::stack(records, 2)};
}

CpuForwardResult forward_vrz(const ForwardInput& p, EquationKind kind)
{
    const int ndim = ndim_of(kind);
    auto vp = p.models[0];
    auto z = p.models[1];
    auto b = vp / z;
    auto kappa = z * vp;
    auto mask = interior_mask_like(vp, ndim, p.M);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    if (ndim == 2) {
        auto s = make_acoustic_state_2d(vp);
        auto h = spacing_for(p, 2);
        auto pml = acoustic_pml(p.pml_vals, 2);
        auto dbdz = centered_grad_axis(b, 2, 0, p.M, p.grad_coes, h[0]);
        auto dbdx = centered_grad_axis(b, 2, 1, p.M, p.grad_coes, h[1]);
        for (unsigned int it = 0; it < p.nt; ++it) {
            auto lap_z = laplace_axis(s.u_now, 2, 0, p.M, p.lap_coes, h[0]);
            auto lap_x = laplace_axis(s.u_now, 2, 1, p.M, p.lap_coes, h[1]);
            auto dudz = centered_grad_axis(s.u_now, 2, 0, p.M, p.grad_coes, h[0]);
            auto dudx = centered_grad_axis(s.u_now, 2, 1, p.M, p.grad_coes, h[1]);
            auto tmpz = (1.0 + pml[1]) * lap_z + pml[2] * dudz + centered_grad_axis(pml[0] * s.psiz, 2, 0, p.M, p.grad_coes, h[0]);
            auto tmpx = (1.0 + pml[4]) * lap_x + pml[5] * dudx + centered_grad_axis(pml[3] * s.psix, 2, 1, p.M, p.grad_coes, h[1]);
            auto psiz = pml[1] * dudz + pml[0] * s.psiz;
            auto psix = pml[4] * dudx + pml[3] * s.psix;
            auto zetaz = pml[1] * tmpz + pml[0] * s.zetaz;
            auto zetax = pml[4] * tmpx + pml[3] * s.zetax;
            auto wsum = (1.0 + pml[1]) * tmpz + pml[0] * s.zetaz
                      + (1.0 + pml[4]) * tmpx + pml[3] * s.zetax;
            auto u_next = 2.0 * s.u_now - s.u_prev
                + (p.dt * p.dt) * kappa * (b * wsum + dbdx * (dudx + psix) + dbdz * (dudz + psiz));
            s = {apply_mask(s.u_now, mask), apply_mask(u_next, mask), apply_mask(psix, mask), apply_mask(psiz, mask), apply_mask(zetax, mask), apply_mask(zetaz, mask)};
            s.u_now = add_sources(s.u_now, p.source, p.sources_loc, static_cast<int>(it), 2);
            records.push_back(sample_field(s.u_now, p.receivers_loc, 2));
        }
    } else {
        auto s = make_acoustic_state_3d(vp);
        auto h = spacing_for(p, 3);
        auto pml = acoustic_pml(p.pml_vals, 3);
        auto dbdz = centered_grad_axis(b, 3, 0, p.M, p.grad_coes, h[0]);
        auto dbdy = centered_grad_axis(b, 3, 1, p.M, p.grad_coes, h[1]);
        auto dbdx = centered_grad_axis(b, 3, 2, p.M, p.grad_coes, h[2]);
        for (unsigned int it = 0; it < p.nt; ++it) {
            auto lap_z = laplace_axis(s.u_now, 3, 0, p.M, p.lap_coes, h[0]);
            auto lap_y = laplace_axis(s.u_now, 3, 1, p.M, p.lap_coes, h[1]);
            auto lap_x = laplace_axis(s.u_now, 3, 2, p.M, p.lap_coes, h[2]);
            auto dudz = centered_grad_axis(s.u_now, 3, 0, p.M, p.grad_coes, h[0]);
            auto dudy = centered_grad_axis(s.u_now, 3, 1, p.M, p.grad_coes, h[1]);
            auto dudx = centered_grad_axis(s.u_now, 3, 2, p.M, p.grad_coes, h[2]);
            auto tmpz = (1.0 + pml[1]) * lap_z + pml[2] * dudz + centered_grad_axis(pml[0] * s.psiz, 3, 0, p.M, p.grad_coes, h[0]);
            auto tmpy = (1.0 + pml[4]) * lap_y + pml[5] * dudy + centered_grad_axis(pml[3] * s.psiy, 3, 1, p.M, p.grad_coes, h[1]);
            auto tmpx = (1.0 + pml[7]) * lap_x + pml[8] * dudx + centered_grad_axis(pml[6] * s.psix, 3, 2, p.M, p.grad_coes, h[2]);
            auto psiz = pml[1] * dudz + pml[0] * s.psiz;
            auto psiy = pml[4] * dudy + pml[3] * s.psiy;
            auto psix = pml[7] * dudx + pml[6] * s.psix;
            auto zetaz = pml[1] * tmpz + pml[0] * s.zetaz;
            auto zetay = pml[4] * tmpy + pml[3] * s.zetay;
            auto zetax = pml[7] * tmpx + pml[6] * s.zetax;
            auto wsum = (1.0 + pml[1]) * tmpz + pml[0] * s.zetaz
                      + (1.0 + pml[4]) * tmpy + pml[3] * s.zetay
                      + (1.0 + pml[7]) * tmpx + pml[6] * s.zetax;
            auto u_next = 2.0 * s.u_now - s.u_prev
                + (p.dt * p.dt) * kappa * (b * wsum + dbdx * (dudx + psix) + dbdy * (dudy + psiy) + dbdz * (dudz + psiz));
            s = {apply_mask(s.u_now, mask), apply_mask(u_next, mask), apply_mask(psix, mask), apply_mask(psiy, mask), apply_mask(psiz, mask), apply_mask(zetax, mask), apply_mask(zetay, mask), apply_mask(zetaz, mask)};
            s.u_now = add_sources(s.u_now, p.source, p.sources_loc, static_cast<int>(it), 3);
            records.push_back(sample_field(s.u_now, p.receivers_loc, 3));
        }
    }
    return {torch::stack(records, 2)};
}

std::vector<torch::Tensor> elastic_step_2d(
    const std::vector<torch::Tensor>& f,
    const std::vector<torch::Tensor>& models,
    const ForwardInput& p,
    const torch::Tensor& mask
)
{
    auto h = spacing_for(p, 2);
    auto pml = elastic_pml(p.pml_vals, 2);
    auto vp = models[0], vs = models[1], rho = models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);

    auto vx = f[0], vz = f[1], sxx = f[2], szz = f[3], sxz = f[4];
    auto m_vxx = f[5], m_vxz = f[6], m_vzx = f[7], m_vzz = f[8];
    auto m_sxxx = f[9], m_sxxz = f[10], m_szzx = f[11], m_szzz = f[12], m_sxzx = f[13], m_sxzz = f[14];

    auto dszz_dz = staggered_grad_axis(szz, 2, 0, p.M, p.grad_coes, h[0], true);
    m_szzz = pml[2] * m_szzz + pml[3] * dszz_dz;
    dszz_dz = dszz_dz + m_szzz;
    auto dsxz_dx = staggered_grad_axis(sxz, 2, 1, p.M, p.grad_coes, h[1], false);
    m_sxzx = pml[4] * m_sxzx + pml[5] * dsxz_dx;
    dsxz_dx = dsxz_dx + m_sxzx;
    vz = vz + p.dt / rho * (dszz_dz + dsxz_dx);

    auto dsxz_dz = staggered_grad_axis(sxz, 2, 0, p.M, p.grad_coes, h[0], false);
    m_sxzz = pml[0] * m_sxzz + pml[1] * dsxz_dz;
    dsxz_dz = dsxz_dz + m_sxzz;
    auto dsxx_dx = staggered_grad_axis(sxx, 2, 1, p.M, p.grad_coes, h[1], true);
    m_sxxx = pml[6] * m_sxxx + pml[7] * dsxx_dx;
    dsxx_dx = dsxx_dx + m_sxxx;
    vx = vx + p.dt / rho * (dsxx_dx + dsxz_dz);

    auto dvx_dx = staggered_grad_axis(vx, 2, 1, p.M, p.grad_coes, h[1], false);
    m_vxx = pml[4] * m_vxx + pml[5] * dvx_dx;
    dvx_dx = dvx_dx + m_vxx;
    auto dvz_dz = staggered_grad_axis(vz, 2, 0, p.M, p.grad_coes, h[0], false);
    m_vzz = pml[0] * m_vzz + pml[1] * dvz_dz;
    dvz_dz = dvz_dz + m_vzz;
    sxx = sxx + p.dt * ((lambda + 2.0 * mu) * dvx_dx + lambda * dvz_dz);
    szz = szz + p.dt * ((lambda + 2.0 * mu) * dvz_dz + lambda * dvx_dx);

    auto dvx_dz = staggered_grad_axis(vx, 2, 0, p.M, p.grad_coes, h[0], true);
    m_vxz = pml[2] * m_vxz + pml[3] * dvx_dz;
    dvx_dz = dvx_dz + m_vxz;
    auto dvz_dx = staggered_grad_axis(vz, 2, 1, p.M, p.grad_coes, h[1], true);
    m_vzx = pml[6] * m_vzx + pml[7] * dvz_dx;
    dvz_dx = dvz_dx + m_vzx;
    sxz = sxz + p.dt * mu * (dvx_dz + dvz_dx);

    return {
        apply_mask(vx, mask), apply_mask(vz, mask), apply_mask(sxx, mask), apply_mask(szz, mask), apply_mask(sxz, mask),
        apply_mask(m_vxx, mask), apply_mask(m_vxz, mask), apply_mask(m_vzx, mask), apply_mask(m_vzz, mask),
        apply_mask(m_sxxx, mask), apply_mask(m_sxxz, mask), apply_mask(m_szzx, mask), apply_mask(m_szzz, mask),
        apply_mask(m_sxzx, mask), apply_mask(m_sxzz, mask),
    };
}

CpuForwardResult forward_elastic2d(const ForwardInput& p)
{
    auto like = p.models[0];
    auto mask = interior_mask_like(like, 2, p.M);
    std::vector<torch::Tensor> f(15, torch::zeros_like(like));
    auto source_fields = tensor_to_ints(p.source_field_indices);
    auto receiver_fields = tensor_to_ints(p.receiver_field_indices);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    for (unsigned int it = 0; it < p.nt; ++it) {
        f = elastic_step_2d(f, p.models, p, mask);
        for (int id : source_fields) {
            auto field = field_by_id_2d(f, id);
            if (field.defined()) set_field_by_id_2d(f, id, add_sources(field, p.source, p.sources_loc, static_cast<int>(it), 2));
        }
        std::vector<torch::Tensor> per_field;
        for (int id : receiver_fields) {
            auto field = field_by_id_2d(f, id);
            per_field.push_back(field.defined() ? sample_field(field, p.receivers_loc, 2) : torch::zeros({like.size(0), p.receivers_loc.size(1)}, like.options()));
        }
        records.push_back(torch::stack(per_field, 0));
    }
    return {torch::stack(records, 3)};
}

CpuForwardResult forward_das2d(const ForwardInput& p)
{
    auto like = p.models[0];
    auto h = spacing_for(p, 2);
    auto pml = elastic_pml(p.pml_vals, 2);
    auto mask = interior_mask_like(like, 2, p.M);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);
    auto mu = rho * vs * vs;
    std::vector<torch::Tensor> f(17, torch::zeros_like(like));
    auto source_fields = tensor_to_ints(p.source_field_indices);
    auto receiver_fields = tensor_to_ints(p.receiver_field_indices);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    auto d2_cpml = [&](const torch::Tensor& u, torch::Tensor& mf, torch::Tensor& mb, int axis, double dh,
                       const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& ah, const torch::Tensor& bh) {
        auto first = staggered_grad_axis(u, 2, axis, p.M, p.grad_coes, dh, true);
        mf = ah * mf + bh * first;
        first = first + mf;
        auto second = staggered_grad_axis(first, 2, axis, p.M, p.grad_coes, dh, false);
        mb = a * mb + b * second;
        return second + mb;
    };

    for (unsigned int it = 0; it < p.nt; ++it) {
        auto dxx_sxx = d2_cpml(f[2], f[6], f[7], 1, h[1], pml[4], pml[5], pml[6], pml[7]);
        auto dzz_szz = d2_cpml(f[3], f[8], f[9], 0, h[0], pml[0], pml[1], pml[2], pml[3]);
        auto dzz_txx = d2_cpml(f[4], f[10], f[11], 0, h[0], pml[0], pml[1], pml[2], pml[3]);
        auto dxx_tzz = d2_cpml(f[5], f[12], f[13], 1, h[1], pml[4], pml[5], pml[6], pml[7]);
        auto shear = dzz_txx + dxx_tzz;
        f[0] = apply_mask(f[0] + p.dt / rho * (dxx_sxx + shear), mask);
        f[1] = apply_mask(f[1] + p.dt / rho * (dzz_szz + shear), mask);
        f[2] = apply_mask(f[2] + p.dt * ((lambda + 2.0 * mu) * f[0] + lambda * f[1]), mask);
        f[3] = apply_mask(f[3] + p.dt * ((lambda + 2.0 * mu) * f[1] + lambda * f[0]), mask);
        f[4] = apply_mask(f[4] + p.dt * mu * f[0], mask);
        f[5] = apply_mask(f[5] + p.dt * mu * f[1], mask);
        f[14] = helix35(f[0], torch::zeros_like(f[0]), f[1]);
        f[15] = helix54(f[0], torch::zeros_like(f[0]), f[1], 0);
        f[16] = helix54(f[0], torch::zeros_like(f[0]), f[1], 2);

        for (int id : source_fields) {
            auto field = field_by_id_2d(f, id);
            if (field.defined()) set_field_by_id_2d(f, id, add_sources(field, p.source, p.sources_loc, static_cast<int>(it), 2));
        }
        std::vector<torch::Tensor> per_field;
        for (int id : receiver_fields) {
            auto field = field_by_id_2d(f, id);
            per_field.push_back(field.defined() ? sample_field(field, p.receivers_loc, 2) : torch::zeros({like.size(0), p.receivers_loc.size(1)}, like.options()));
        }
        records.push_back(torch::stack(per_field, 0));
    }
    return {torch::stack(records, 3)};
}

CpuForwardResult forward_elastic3d_like(const ForwardInput& p, bool das)
{
    auto like = p.models[0];
    auto h = spacing_for(p, 3);
    auto pml = elastic_pml(p.pml_vals, 3);
    auto mask = interior_mask_like(like, 3, p.M);
    auto vp = p.models[0], vs = p.models[1], rho = p.models[2];
    auto lambda = rho * (vp * vp - 2.0 * vs * vs);
    auto mu = rho * vs * vs;
    const int nfields = das ? 31 : 36;
    std::vector<torch::Tensor> f(nfields, torch::zeros_like(like));
    auto source_fields = tensor_to_ints(p.source_field_indices);
    auto receiver_fields = tensor_to_ints(p.receiver_field_indices);
    std::vector<torch::Tensor> records;
    records.reserve(p.nt);

    for (unsigned int it = 0; it < p.nt; ++it) {
        if (!das) {
            auto vx = f[0], vy = f[1], vz = f[2], sxx = f[3], syy = f[4], szz = f[5], sxy = f[6], sxz = f[7], syz = f[8];
            auto dsxx_dx = staggered_grad_axis(sxx, 3, 2, p.M, p.grad_coes, h[2], true);
            auto dsxy_dy = staggered_grad_axis(sxy, 3, 1, p.M, p.grad_coes, h[1], false);
            auto dsxz_dz = staggered_grad_axis(sxz, 3, 0, p.M, p.grad_coes, h[0], false);
            auto dsxy_dx = staggered_grad_axis(sxy, 3, 2, p.M, p.grad_coes, h[2], false);
            auto dsyy_dy = staggered_grad_axis(syy, 3, 1, p.M, p.grad_coes, h[1], true);
            auto dsyz_dz = staggered_grad_axis(syz, 3, 0, p.M, p.grad_coes, h[0], false);
            auto dsxz_dx = staggered_grad_axis(sxz, 3, 2, p.M, p.grad_coes, h[2], false);
            auto dsyz_dy = staggered_grad_axis(syz, 3, 1, p.M, p.grad_coes, h[1], false);
            auto dszz_dz = staggered_grad_axis(szz, 3, 0, p.M, p.grad_coes, h[0], true);
            vx = apply_mask(vx + p.dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz), mask);
            vy = apply_mask(vy + p.dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz), mask);
            vz = apply_mask(vz + p.dt / rho * (dsxz_dx + dsyz_dy + dszz_dz), mask);
            auto dvx_dx = staggered_grad_axis(vx, 3, 2, p.M, p.grad_coes, h[2], false);
            auto dvy_dy = staggered_grad_axis(vy, 3, 1, p.M, p.grad_coes, h[1], false);
            auto dvz_dz = staggered_grad_axis(vz, 3, 0, p.M, p.grad_coes, h[0], false);
            auto div = dvx_dx + dvy_dy + dvz_dz;
            sxx = apply_mask(sxx + p.dt * (lambda * div + 2.0 * mu * dvx_dx), mask);
            syy = apply_mask(syy + p.dt * (lambda * div + 2.0 * mu * dvy_dy), mask);
            szz = apply_mask(szz + p.dt * (lambda * div + 2.0 * mu * dvz_dz), mask);
            sxy = apply_mask(sxy + p.dt * mu * (staggered_grad_axis(vx, 3, 1, p.M, p.grad_coes, h[1], true) + staggered_grad_axis(vy, 3, 2, p.M, p.grad_coes, h[2], true)), mask);
            sxz = apply_mask(sxz + p.dt * mu * (staggered_grad_axis(vx, 3, 0, p.M, p.grad_coes, h[0], true) + staggered_grad_axis(vz, 3, 2, p.M, p.grad_coes, h[2], true)), mask);
            syz = apply_mask(syz + p.dt * mu * (staggered_grad_axis(vy, 3, 0, p.M, p.grad_coes, h[0], true) + staggered_grad_axis(vz, 3, 1, p.M, p.grad_coes, h[1], true)), mask);
            f[0] = vx; f[1] = vy; f[2] = vz; f[3] = sxx; f[4] = syy; f[5] = szz; f[6] = sxy; f[7] = sxz; f[8] = syz;
        } else {
            auto dxx_sxx = staggered_grad_axis(staggered_grad_axis(f[3], 3, 2, p.M, p.grad_coes, h[2], true), 3, 2, p.M, p.grad_coes, h[2], false);
            auto dyy_syy = staggered_grad_axis(staggered_grad_axis(f[4], 3, 1, p.M, p.grad_coes, h[1], true), 3, 1, p.M, p.grad_coes, h[1], false);
            auto dzz_szz = staggered_grad_axis(staggered_grad_axis(f[5], 3, 0, p.M, p.grad_coes, h[0], true), 3, 0, p.M, p.grad_coes, h[0], false);
            auto dyy_txx = staggered_grad_axis(staggered_grad_axis(f[6], 3, 1, p.M, p.grad_coes, h[1], true), 3, 1, p.M, p.grad_coes, h[1], false);
            auto dzz_txx = staggered_grad_axis(staggered_grad_axis(f[6], 3, 0, p.M, p.grad_coes, h[0], true), 3, 0, p.M, p.grad_coes, h[0], false);
            auto dxx_tyy = staggered_grad_axis(staggered_grad_axis(f[7], 3, 2, p.M, p.grad_coes, h[2], true), 3, 2, p.M, p.grad_coes, h[2], false);
            auto dzz_tyy = staggered_grad_axis(staggered_grad_axis(f[7], 3, 0, p.M, p.grad_coes, h[0], true), 3, 0, p.M, p.grad_coes, h[0], false);
            auto dxx_tzz = staggered_grad_axis(staggered_grad_axis(f[8], 3, 2, p.M, p.grad_coes, h[2], true), 3, 2, p.M, p.grad_coes, h[2], false);
            auto dyy_tzz = staggered_grad_axis(staggered_grad_axis(f[8], 3, 1, p.M, p.grad_coes, h[1], true), 3, 1, p.M, p.grad_coes, h[1], false);
            f[0] = apply_mask(f[0] + p.dt / rho * (dxx_sxx + dyy_txx + dxx_tyy + dzz_txx + dxx_tzz), mask);
            f[1] = apply_mask(f[1] + p.dt / rho * (dyy_syy + dyy_txx + dxx_tyy + dzz_tyy + dyy_tzz), mask);
            f[2] = apply_mask(f[2] + p.dt / rho * (dzz_szz + dzz_txx + dxx_tzz + dzz_tyy + dyy_tzz), mask);
            auto div = f[0] + f[1] + f[2];
            f[3] = apply_mask(f[3] + p.dt * (lambda * div + 2.0 * mu * f[0]), mask);
            f[4] = apply_mask(f[4] + p.dt * (lambda * div + 2.0 * mu * f[1]), mask);
            f[5] = apply_mask(f[5] + p.dt * (lambda * div + 2.0 * mu * f[2]), mask);
            f[6] = apply_mask(f[6] + p.dt * mu * f[0], mask);
            f[7] = apply_mask(f[7] + p.dt * mu * f[1], mask);
            f[8] = apply_mask(f[8] + p.dt * mu * f[2], mask);
            f[27] = helix35(f[0], f[1], f[2]);
            f[28] = helix54(f[0], f[1], f[2], 0);
            f[29] = helix54(f[0], f[1], f[2], 1);
            f[30] = helix54(f[0], f[1], f[2], 2);
        }

        for (int id : source_fields) {
            if (id >= 0 && id < static_cast<int>(f.size())) {
                f[id] = add_sources(f[id], p.source, p.sources_loc, static_cast<int>(it), 3);
            }
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
    return {torch::stack(records, 3)};
}

CpuForwardResult run_forward_internal(const ForwardInput& p, EquationKind kind)
{
    if (is_acoustic_family(kind)) return forward_acoustic(p, kind);
    if (is_lsrtm_family(kind)) return forward_lsrtm(p, kind);
    if (is_vrz_family(kind)) return forward_vrz(p, kind);
    if (kind == EquationKind::Elastic2D) return forward_elastic2d(p);
    if (kind == EquationKind::DAS2D) return forward_das2d(p);
    if (kind == EquationKind::Elastic3D) return forward_elastic3d_like(p, false);
    if (kind == EquationKind::DAS3D) return forward_elastic3d_like(p, true);
    TORCH_CHECK(false, "Unsupported CPU equation kind");
}

} // namespace

namespace engine {

bool is_cpu_input(const ForwardInput& in)
{
    return !in.models.empty() && in.models[0].device().is_cpu();
}

bool is_cpu_input(const BackwardInput& in)
{
    return !in.models.empty() && in.models[0].device().is_cpu();
}

ForwardOutput forward(const ForwardInput& in, EquationKind kind)
{
    TORCH_CHECK(is_cpu_input(in), "sweep_cpu::forward called with non-CPU tensors");
    ForwardOutput out;
    auto result = run_forward_internal(in, kind);
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward_full(const BackwardInput& in, EquationKind kind)
{
    TORCH_CHECK(is_cpu_input(in), "sweep_cpu::backward called with non-CPU tensors");
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

    auto record = run_forward_internal(fwd, kind).record;
    std::vector<torch::Tensor> inputs;
    inputs.push_back(fwd.source);
    inputs.insert(inputs.end(), fwd.models.begin(), fwd.models.end());

    auto grad_out = in.adjoint_source;
    std::vector<torch::Tensor> grads;
    {
        pybind11::gil_scoped_release no_gil;
        grads = torch::autograd::grad(
            {record},
            inputs,
            {grad_out},
            false,
            false,
            false
        );
    }

    BackwardOutput out;
    out.grads = grads;
    out.source_illumination = torch::zeros_like(in.models[0]);
    out.receiver_illumination = torch::zeros_like(in.models[0]);
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in, EquationKind kind)
{
    return backward_full(in, kind);
}

BackwardOutput backward_ckpt(const BackwardInput& in, EquationKind kind)
{
    return backward_full(in, kind);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in, EquationKind kind)
{
    return backward_full(in, kind);
}

BackwardOutput backward(const BackwardInput& in, EquationKind kind)
{
    return backward_full(in, kind);
}

} // namespace engine
} // namespace sweep_cpu

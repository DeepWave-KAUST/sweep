#include "elastic_tti_sg2d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace sweep_cpu::elastic_tti_sg2d {
namespace {

using sweep_cpu::ops::StencilCoefficients;
using sweep_cpu::ops::sgrad_backward;
using sweep_cpu::ops::sgrad_forward;

constexpr int kNumFields = 8;
constexpr int kNumState = 20;
constexpr int kNumModels = 16;

struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

struct State {
    std::vector<float> vx, vy, vz, sxx, szz, syz, sxz, sxy;
    std::vector<float> m_vxx, m_vxz, m_vyx, m_vyz, m_vzx, m_vzz;
    std::vector<float> m_txxx, m_txzz, m_txyx, m_tyzz, m_txzx, m_tzzz;

    State() = default;
    explicit State(int64_t total)
        : vx(total, 0.0f), vy(total, 0.0f), vz(total, 0.0f),
          sxx(total, 0.0f), szz(total, 0.0f), syz(total, 0.0f), sxz(total, 0.0f), sxy(total, 0.0f),
          m_vxx(total, 0.0f), m_vxz(total, 0.0f), m_vyx(total, 0.0f), m_vyz(total, 0.0f),
          m_vzx(total, 0.0f), m_vzz(total, 0.0f),
          m_txxx(total, 0.0f), m_txzz(total, 0.0f), m_txyx(total, 0.0f), m_tyzz(total, 0.0f),
          m_txzx(total, 0.0f), m_tzzz(total, 0.0f)
    {}
};

struct Workspace {
    std::array<std::vector<float>, 6> q;

    explicit Workspace(int64_t total)
    {
        for (auto& v : q) v.assign(total, 0.0f);
    }
};

struct ModelPtr {
    const float* rho;
    const float* C11;
    const float* C13;
    const float* C14;
    const float* C15;
    const float* C16;
    const float* C33;
    const float* C34;
    const float* C35;
    const float* C36;
    const float* C44;
    const float* C45;
    const float* C46;
    const float* C55;
    const float* C56;
    const float* C66;
};

struct GradPtr {
    float* rho;
    float* C11;
    float* C13;
    float* C14;
    float* C15;
    float* C16;
    float* C33;
    float* C34;
    float* C35;
    float* C36;
    float* C44;
    float* C45;
    float* C46;
    float* C55;
    float* C56;
    float* C66;
};

std::vector<double> spacing_for(const ForwardInput& p)
{
    return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
}

std::vector<double> spacing_for(const BackwardInput& p)
{
    return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
}

ModelPtr model_ptr(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == kNumModels, "ElasticTTISG CPU expects rho plus 15 prepared stiffness tensors");
    int i = 0;
    return {
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
        models[i++].data_ptr<float>(),
    };
}

GradPtr grad_ptr(std::vector<torch::Tensor>& grads)
{
    TORCH_CHECK(grads.size() == kNumModels, "ElasticTTISG CPU expects 16 prepared model gradients");
    int i = 0;
    return {
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
        grads[i++].data_ptr<float>(),
    };
}

bool can_use_forward(const ForwardInput& p)
{
    if (p.models.size() != kNumModels || p.pml_vals.size() < 8) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, false)) return false;
    for (const auto& model : p.models) {
        if (!model.is_contiguous() || model.scalar_type() != torch::kFloat32 || model.dim() != 4) return false;
        if (model.sizes() != p.models[0].sizes()) return false;
    }
    if (!p.source.is_contiguous() || p.source.scalar_type() != torch::kFloat32 || p.source.dim() != 3) return false;
    if (!p.sources_loc.is_contiguous() || !p.receivers_loc.is_contiguous()) return false;
    if (p.sources_loc.scalar_type() != torch::kInt32 || p.receivers_loc.scalar_type() != torch::kInt32) return false;
    if (!p.source_field_indices.is_contiguous() || !p.receiver_field_indices.is_contiguous()) return false;
    if (p.source_field_indices.scalar_type() != torch::kInt32 || p.receiver_field_indices.scalar_type() != torch::kInt32) return false;
    for (const auto& t : p.pml_vals) {
        if (!t.is_contiguous() || t.scalar_type() != torch::kFloat32) return false;
    }
    const int64_t nz = p.models[0].size(2);
    const int64_t nx = p.models[0].size(3);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz || p.pml_vals[3].numel() != nz) return false;
    if (p.pml_vals[4].numel() != nx || p.pml_vals[5].numel() != nx || p.pml_vals[6].numel() != nx || p.pml_vals[7].numel() != nx) return false;
    return true;
}

bool can_use_backward(const BackwardInput& p)
{
    if (p.models.size() != kNumModels || p.pml_vals.size() < 8) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, false)) return false;
    for (const auto& model : p.models) {
        if (!model.is_contiguous() || model.scalar_type() != torch::kFloat32 || model.dim() != 4) return false;
        if (model.sizes() != p.models[0].sizes()) return false;
    }
    if (!p.forward_source.is_contiguous() || p.forward_source.scalar_type() != torch::kFloat32 || p.forward_source.dim() != 3) return false;
    if (!p.adjoint_source.is_contiguous() || p.adjoint_source.scalar_type() != torch::kFloat32 || p.adjoint_source.dim() != 4) return false;
    if (!p.forward_sources_loc.is_contiguous() || !p.adjoint_sources_loc.is_contiguous()) return false;
    if (p.forward_sources_loc.scalar_type() != torch::kInt32 || p.adjoint_sources_loc.scalar_type() != torch::kInt32) return false;
    if (!p.source_field_indices.is_contiguous() || !p.receiver_field_indices.is_contiguous()) return false;
    if (p.source_field_indices.scalar_type() != torch::kInt32 || p.receiver_field_indices.scalar_type() != torch::kInt32) return false;
    for (const auto& t : p.pml_vals) {
        if (!t.is_contiguous() || t.scalar_type() != torch::kFloat32) return false;
    }
    return true;
}

std::vector<float*> mutable_field_ptrs(State& s)
{
    return {s.vx.data(), s.vy.data(), s.vz.data(), s.sxx.data(), s.szz.data(), s.syz.data(), s.sxz.data(), s.sxy.data()};
}

std::vector<const float*> const_field_ptrs(const State& s)
{
    return {s.vx.data(), s.vy.data(), s.vz.data(), s.sxx.data(), s.szz.data(), s.syz.data(), s.sxz.data(), s.sxy.data()};
}

std::array<const std::vector<float>*, kNumState> state_vectors(const State& s)
{
    return {
        &s.vx, &s.vy, &s.vz, &s.sxx, &s.szz, &s.syz, &s.sxz, &s.sxy,
        &s.m_vxx, &s.m_vxz, &s.m_vyx, &s.m_vyz, &s.m_vzx, &s.m_vzz,
        &s.m_txxx, &s.m_txzz, &s.m_txyx, &s.m_tyzz, &s.m_txzx, &s.m_tzzz,
    };
}

void copy_vector_to_tensor(const std::vector<float>& src, torch::Tensor tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(src.size()) == tensor.numel(), "Tensor/vector size mismatch");
    std::copy(src.begin(), src.end(), tensor.data_ptr<float>());
}

void copy_tensor_to_vector(const torch::Tensor& tensor, std::vector<float>& dst)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(dst.size()) == tensor.numel(), "Tensor/vector size mismatch");
    const float* ptr = tensor.data_ptr<float>();
    std::copy(ptr, ptr + tensor.numel(), dst.begin());
}

void save_state_to_checkpoint(const std::vector<torch::Tensor>& checkpoints, int checkpoint_idx, const State& s)
{
    if (checkpoint_idx < 0 || checkpoints.empty()) return;
    TORCH_CHECK(checkpoints.size() == kNumState, "ElasticTTISG CPU checkpointing expects 20 checkpoint tensors");
    auto vectors = state_vectors(s);
    for (int i = 0; i < kNumState; ++i) {
        copy_vector_to_tensor(*vectors[i], checkpoints[i].select(0, checkpoint_idx));
    }
}

void load_state_from_checkpoint(const std::vector<torch::Tensor>& checkpoints, int checkpoint_idx, State& s)
{
    TORCH_CHECK(checkpoints.size() == kNumState, "ElasticTTISG CPU checkpointing expects 20 checkpoint tensors");
    std::array<std::vector<float>*, kNumState> vectors = {
        &s.vx, &s.vy, &s.vz, &s.sxx, &s.szz, &s.syz, &s.sxz, &s.sxy,
        &s.m_vxx, &s.m_vxz, &s.m_vyx, &s.m_vyz, &s.m_vzx, &s.m_vzz,
        &s.m_txxx, &s.m_txzz, &s.m_txyx, &s.m_tyzz, &s.m_txzx, &s.m_tzzz,
    };
    for (int i = 0; i < kNumState; ++i) {
        copy_tensor_to_vector(checkpoints[i].select(0, checkpoint_idx), *vectors[i]);
    }
}

int checkpoint_index(int it, int nt, int interval)
{
    if (interval < 1) return -1;
    if (((it + 1) % interval == 0) && (it + 1 < nt)) return (it + 1) / interval;
    return -1;
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

int phys_x0(int abcn, int M) { return abcn + M; }
int phys_x1(int64_t nx, int abcn, int M) { return static_cast<int>(nx) - abcn - M; }
int phys_z0(int abcn, int M, bool free_surface) { return free_surface ? M : abcn + M; }
int phys_z1(int64_t nz, int abcn, int M) { return static_cast<int>(nz) - abcn - M; }

template <int Order>
float top_fs_value_z(
    const float* u,
    int64_t b,
    int64_t z,
    int64_t x,
    int64_t nz,
    int64_t nx,
    int top,
    bool odd
)
{
    if (z < top) {
        z = 2 * top - z;
        const float v = u[b * nz * nx + z * nx + x];
        return odd ? -v : v;
    }
    return u[b * nz * nx + z * nx + x];
}

template <int Order, bool Forward>
float top_fs_sgrad_z(
    const float* u,
    int64_t idx,
    int64_t b,
    int64_t z,
    int64_t x,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    bool free_surface,
    bool odd,
    const StencilCoefficients& stencil
)
{
    if (!free_surface) {
        if constexpr (Forward) {
            return sgrad_forward<Order>(u, idx, nx, inv_dz, stencil);
        } else {
            return sgrad_backward<Order>(u, idx, nx, inv_dz, stencil);
        }
    }

    const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int top = M;
    float gz = 0.0f;
    for (int m = 0; m < M; ++m) {
        const float c = sweep_cpu::ops::staggered_coeff<Order>(m + 1, stencil);
        if constexpr (Forward) {
            const float up = top_fs_value_z<Order>(u, b, z + m + 1, x, nz, nx, top, odd);
            const float um = top_fs_value_z<Order>(u, b, z - m, x, nz, nx, top, odd);
            gz += c * (up - um);
        } else {
            const float up = top_fs_value_z<Order>(u, b, z + m, x, nz, nx, top, odd);
            const float um = top_fs_value_z<Order>(u, b, z - m - 1, x, nz, nx, top, odd);
            gz += c * (up - um);
        }
    }
    return gz * inv_dz;
}

template <int Order, bool ForwardOp>
float top_fs_adjoint_sgrad_z(
    const float* q,
    int64_t b,
    int64_t z,
    int64_t x,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    bool free_surface,
    bool odd,
    const StencilCoefficients& stencil
)
{
    const int64_t idx = b * nz * nx + z * nx + x;
    if (!free_surface) {
        if constexpr (ForwardOp) {
            return sgrad_backward<Order>(q, idx, nx, inv_dz, stencil);
        } else {
            return sgrad_forward<Order>(q, idx, nx, inv_dz, stencil);
        }
    }

    const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int top = M;
    const float parity = odd ? -1.0f : 1.0f;
    const int64_t base = b * nz * nx;
    float gz = 0.0f;
    for (int m = 0; m < M; ++m) {
        const float c = sweep_cpu::ops::staggered_coeff<Order>(m + 1, stencil);
        if constexpr (ForwardOp) {
            const int64_t jp = z + m;
            if (jp >= top && jp < nz) gz += c * q[base + jp * nx + x];
            const int64_t jm = z - m - 1;
            if (jm >= top) gz -= c * q[base + jm * nx + x];
            const int64_t jg = 2 * top + m - z;
            if (z > top && z <= top + m && jg >= top && jg < nz) {
                gz += c * parity * q[base + jg * nx + x];
            }
        } else {
            const int64_t jp = z + m + 1;
            if (jp >= top && jp < nz) gz += c * q[base + jp * nx + x];
            const int64_t jm = z - m;
            if (jm >= top) gz -= c * q[base + jm * nx + x];
            const int64_t jg = 2 * top + m + 1 - z;
            if (z > top && z <= top + m + 1 && jg >= top && jg < nz) {
                gz += c * parity * q[base + jg * nx + x];
            }
        }
    }
    return gz * inv_dz;
}

inline bool solve3x3(
    float a00, float a01, float a02,
    float a10, float a11, float a12,
    float a20, float a21, float a22,
    float r0, float r1, float r2,
    float& x0, float& x1, float& x2)
{
    const float det =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    if (std::fabs(det) <= 1.0e-20f) return false;
    x0 = ((a11 * a22 - a12 * a21) * r0 + (a02 * a21 - a01 * a22) * r1 + (a01 * a12 - a02 * a11) * r2) / det;
    x1 = ((a12 * a20 - a10 * a22) * r0 + (a00 * a22 - a02 * a20) * r1 + (a02 * a10 - a00 * a12) * r2) / det;
    x2 = ((a10 * a21 - a11 * a20) * r0 + (a01 * a20 - a00 * a21) * r1 + (a00 * a11 - a01 * a10) * r2) / det;
    return true;
}

inline bool apply_top_traction_solve(
    ModelPtr m,
    int64_t idx,
    float dvx_dx,
    float dvy_dx,
    float dvz_dx,
    float& dvx_dz,
    float& dvy_dz,
    float& dvz_dz)
{
    const float r0 = -(m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * dvz_dx);
    const float r1 = -(m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * dvz_dx);
    const float r2 = -(m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * dvz_dx);
    return solve3x3(
        m.C35[idx], m.C34[idx], m.C33[idx],
        m.C45[idx], m.C44[idx], m.C34[idx],
        m.C55[idx], m.C45[idx], m.C35[idx],
        r0, r1, r2,
        dvx_dz, dvy_dz, dvz_dz);
}

inline bool add_top_traction_solve_adjoint_h(
    ModelPtr m,
    int64_t idx,
    float bar_dvx_dz,
    float bar_dvy_dz,
    float bar_dvz_dz,
    float& bar_dvx_dx,
    float& bar_dvy_dx,
    float& bar_dvz_dx)
{
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
    const bool ok = solve3x3(
        m.C35[idx], m.C45[idx], m.C55[idx],
        m.C34[idx], m.C44[idx], m.C45[idx],
        m.C33[idx], m.C34[idx], m.C35[idx],
        bar_dvx_dz, bar_dvy_dz, bar_dvz_dz,
        l0, l1, l2);
    if (!ok) return false;
    bar_dvx_dx += -(m.C13[idx] * l0 + m.C14[idx] * l1 + m.C15[idx] * l2);
    bar_dvy_dx += -(m.C36[idx] * l0 + m.C46[idx] * l1 + m.C56[idx] * l2);
    bar_dvz_dx += -(m.C35[idx] * l0 + m.C45[idx] * l1 + m.C55[idx] * l2);
    return true;
}

inline void accumulate_top_traction_solve_model_grad(
    ModelPtr m,
    GradPtr grad,
    int64_t idx,
    float dvx_dx,
    float dvy_dx,
    float dvz_dx,
    float dvx_dz,
    float dvy_dz,
    float dvz_dz,
    float bar_dvx_dz,
    float bar_dvy_dz,
    float bar_dvz_dz)
{
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
    const bool ok = solve3x3(
        m.C35[idx], m.C45[idx], m.C55[idx],
        m.C34[idx], m.C44[idx], m.C45[idx],
        m.C33[idx], m.C34[idx], m.C35[idx],
        bar_dvx_dz, bar_dvy_dz, bar_dvz_dz,
        l0, l1, l2);
    if (!ok) return;

    grad.C35[idx] += -l0 * dvx_dz - l2 * dvz_dz - l0 * dvz_dx;
    grad.C34[idx] += -l0 * dvy_dz - l1 * dvz_dz;
    grad.C33[idx] += -l0 * dvz_dz;
    grad.C45[idx] += -l1 * dvx_dz - l2 * dvy_dz - l1 * dvz_dx;
    grad.C44[idx] += -l1 * dvy_dz;
    grad.C55[idx] += -l2 * dvx_dz - l2 * dvz_dx;

    grad.C13[idx] += -l0 * dvx_dx;
    grad.C36[idx] += -l0 * dvy_dx;
    grad.C14[idx] += -l1 * dvx_dx;
    grad.C46[idx] += -l1 * dvy_dx;
    grad.C15[idx] += -l2 * dvx_dx;
    grad.C56[idx] += -l2 * dvy_dx;
}

void add_source_to_field(
    float* field,
    const float* source,
    const int32_t* loc,
    int64_t B,
    int64_t nsrc,
    int64_t nt,
    int64_t it,
    int64_t nz,
    int64_t nx,
    float scale = 1.0f
)
{
    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        float* base = field + b * spatial;
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t off = (b * nsrc + isrc) * 2;
            const int64_t x = loc[off];
            const int64_t z = loc[off + 1];
            if (x >= 0 && x < nx && z >= 0 && z < nz) {
                base[z * nx + x] += scale * source[(b * nsrc + isrc) * nt + it];
            }
        }
    }
}

void save_boundary_field(
    const std::vector<torch::Tensor>& boundary,
    int field_id,
    int it,
    const float* field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    if (boundary.empty()) return;
    TORCH_CHECK(boundary.size() == 4, "ElasticTTISG CPU boundary saving expects 4 boundary tensors");
    const int x0 = phys_x0(abcn, M);
    const int x1 = phys_x1(nx, abcn, M);
    const int z0 = phys_z0(abcn, M, free_surface);
    const int z1 = phys_z1(nz, abcn, M);
    const int nx_boundary = x1 - x0;
    const int nz_boundary = z1 - z0;
    auto top = boundary[0];
    auto bottom = boundary[1];
    auto left = boundary[2];
    auto right = boundary[3];
    TORCH_CHECK(top.is_contiguous() && bottom.is_contiguous() && left.is_contiguous() && right.is_contiguous(),
                "ElasticTTISG CPU boundary tensors must be contiguous");
    const int64_t top_stride = static_cast<int64_t>(top.size(1)) * top.size(2) * top.size(3) * top.size(4);
    const int64_t side_stride = static_cast<int64_t>(left.size(1)) * left.size(2) * left.size(3) * left.size(4);
    float* top_ptr = top.data_ptr<float>() + static_cast<int64_t>(field_id) * top_stride;
    float* bottom_ptr = bottom.data_ptr<float>() + static_cast<int64_t>(field_id) * top_stride;
    float* left_ptr = left.data_ptr<float>() + static_cast<int64_t>(field_id) * side_stride;
    float* right_ptr = right.data_ptr<float>() + static_cast<int64_t>(field_id) * side_stride;
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        const float* base = field + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * width + w) * nx_boundary + x);
                top_ptr[bd_idx] = base[(z0 + w) * nx + (x0 + x)];
                bottom_ptr[bd_idx] = base[(z1 - width + w) * nx + (x0 + x)];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * nz_boundary + z) * width + w);
                left_ptr[bd_idx] = base[(z0 + z) * nx + (x0 + w)];
                right_ptr[bd_idx] = base[(z0 + z) * nx + (x1 - width + w)];
            }
        }
    }
}

void restore_boundary_field(
    const std::vector<torch::Tensor>& boundary,
    int field_id,
    int it,
    float* field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    TORCH_CHECK(!boundary.empty(), "ElasticTTISG CPU boundary-saving backward requires saved boundaries");
    TORCH_CHECK(boundary.size() == 4, "ElasticTTISG CPU boundary saving expects 4 boundary tensors");
    const int x0 = phys_x0(abcn, M);
    const int x1 = phys_x1(nx, abcn, M);
    const int z0 = phys_z0(abcn, M, free_surface);
    const int z1 = phys_z1(nz, abcn, M);
    const int nx_boundary = x1 - x0;
    const int nz_boundary = z1 - z0;
    auto top = boundary[0];
    auto bottom = boundary[1];
    auto left = boundary[2];
    auto right = boundary[3];
    TORCH_CHECK(top.is_contiguous() && bottom.is_contiguous() && left.is_contiguous() && right.is_contiguous(),
                "ElasticTTISG CPU boundary tensors must be contiguous");
    const int64_t top_stride = static_cast<int64_t>(top.size(1)) * top.size(2) * top.size(3) * top.size(4);
    const int64_t side_stride = static_cast<int64_t>(left.size(1)) * left.size(2) * left.size(3) * left.size(4);
    const float* top_ptr = top.data_ptr<float>() + static_cast<int64_t>(field_id) * top_stride;
    const float* bottom_ptr = bottom.data_ptr<float>() + static_cast<int64_t>(field_id) * top_stride;
    const float* left_ptr = left.data_ptr<float>() + static_cast<int64_t>(field_id) * side_stride;
    const float* right_ptr = right.data_ptr<float>() + static_cast<int64_t>(field_id) * side_stride;
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        float* base = field + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * width + w) * nx_boundary + x);
                base[(z0 + w) * nx + (x0 + x)] = top_ptr[bd_idx];
                base[(z1 - width + w) * nx + (x0 + x)] = bottom_ptr[bd_idx];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * nz_boundary + z) * width + w);
                base[(z0 + z) * nx + (x0 + w)] = left_ptr[bd_idx];
                base[(z0 + z) * nx + (x1 - width + w)] = right_ptr[bd_idx];
            }
        }
    }
}

void write_boundary_file(
    const std::string& path,
    size_t offset_elems,
    const std::vector<float>& data)
{
    std::ofstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    TORCH_CHECK(out.good(), "Failed to open ElasticTTISG CPU boundary file for writing: ", path);
    out.seekp(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    TORCH_CHECK(out.good(), "Failed to write ElasticTTISG CPU boundary file: ", path);
}

void read_boundary_file(
    const std::string& path,
    size_t offset_elems,
    std::vector<float>& data)
{
    std::ifstream in(path, std::ios::binary);
    TORCH_CHECK(in.good(), "Failed to open ElasticTTISG CPU boundary file for reading: ", path);
    in.seekg(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    TORCH_CHECK(in.good(), "Failed to read ElasticTTISG CPU boundary file: ", path);
}

void save_boundary_field_disk(
    const std::vector<std::string>& files,
    int field_id,
    int it,
    int nt,
    const float* field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width)
{
    TORCH_CHECK(files.size() == 4, "ElasticTTISG CPU disk boundary-saving expects 4 files");
    const int x0 = phys_x0(abcn, M);
    const int x1 = phys_x1(nx, abcn, M);
    const int z0 = phys_z0(abcn, M, free_surface);
    const int z1 = phys_z1(nz, abcn, M);
    const int nx_boundary = x1 - x0;
    const int nz_boundary = z1 - z0;
    const size_t top_elems = static_cast<size_t>(B) * width * nx_boundary;
    const size_t side_elems = static_cast<size_t>(B) * nz_boundary * width;
    std::vector<float> top(top_elems), bottom(top_elems), left(side_elems), right(side_elems);
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        const float* base = field + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t idx = ((b * width + w) * nx_boundary + x);
                top[idx] = base[(z0 + w) * nx + (x0 + x)];
                bottom[idx] = base[(z1 - width + w) * nx + (x0 + x)];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t idx = ((b * nz_boundary + z) * width + w);
                left[idx] = base[(z0 + z) * nx + (x0 + w)];
                right[idx] = base[(z0 + z) * nx + (x1 - width + w)];
            }
        }
    }

    write_boundary_file(files[0], (static_cast<size_t>(field_id) * nt + it) * top_elems, top);
    write_boundary_file(files[1], (static_cast<size_t>(field_id) * nt + it) * top_elems, bottom);
    write_boundary_file(files[2], (static_cast<size_t>(field_id) * nt + it) * side_elems, left);
    write_boundary_file(files[3], (static_cast<size_t>(field_id) * nt + it) * side_elems, right);
}

void restore_boundary_field_disk(
    const std::vector<std::string>& files,
    int field_id,
    int saved_it,
    int nt,
    float* field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width)
{
    TORCH_CHECK(files.size() == 4, "ElasticTTISG CPU disk boundary-saving expects 4 files");
    const int x0 = phys_x0(abcn, M);
    const int x1 = phys_x1(nx, abcn, M);
    const int z0 = phys_z0(abcn, M, free_surface);
    const int z1 = phys_z1(nz, abcn, M);
    const int nx_boundary = x1 - x0;
    const int nz_boundary = z1 - z0;
    const size_t top_elems = static_cast<size_t>(B) * width * nx_boundary;
    const size_t side_elems = static_cast<size_t>(B) * nz_boundary * width;
    std::vector<float> top(top_elems), bottom(top_elems), left(side_elems), right(side_elems);

    read_boundary_file(files[0], (static_cast<size_t>(field_id) * nt + saved_it) * top_elems, top);
    read_boundary_file(files[1], (static_cast<size_t>(field_id) * nt + saved_it) * top_elems, bottom);
    read_boundary_file(files[2], (static_cast<size_t>(field_id) * nt + saved_it) * side_elems, left);
    read_boundary_file(files[3], (static_cast<size_t>(field_id) * nt + saved_it) * side_elems, right);

    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        float* base = field + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t idx = ((b * width + w) * nx_boundary + x);
                base[(z0 + w) * nx + (x0 + x)] = top[idx];
                base[(z1 - width + w) * nx + (x0 + x)] = bottom[idx];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t idx = ((b * nz_boundary + z) * width + w);
                base[(z0 + z) * nx + (x0 + w)] = left[idx];
                base[(z0 + z) * nx + (x1 - width + w)] = right[idx];
            }
        }
    }
}

void record_field(
    const float* field,
    float* record,
    const int32_t* loc,
    int64_t B,
    int64_t nrec,
    int64_t nt,
    int64_t it,
    int64_t nz,
    int64_t nx
)
{
    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        const float* base = field + b * spatial;
        for (int64_t irec = 0; irec < nrec; ++irec) {
            const int64_t off = (b * nrec + irec) * 2;
            const int64_t x = loc[off];
            const int64_t z = loc[off + 1];
            record[(b * nrec + irec) * nt + it] =
                (x >= 0 && x < nx && z >= 0 && z < nz) ? base[z * nx + x] : 0.0f;
        }
    }
}

template <int Order>
void velocity_step(
    State& s,
    ModelPtr m,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t nx,
    float inv_dz, float inv_dx, float dt,
    bool free_surface,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int64_t spatial = nz * nx;
    const int64_t row_count = B * (nz - 2 * M);
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                float dsxx_dx = sgrad_forward<Order>(s.sxx.data(), idx, 1, inv_dx, stencil);
                float dsxz_dz = top_fs_sgrad_z<Order, false>(s.sxz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                float dsxy_dx = sgrad_backward<Order>(s.sxy.data(), idx, 1, inv_dx, stencil);
                float dsyz_dz = top_fs_sgrad_z<Order, false>(s.syz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                float dsxz_dx = sgrad_backward<Order>(s.sxz.data(), idx, 1, inv_dx, stencil);
                float dszz_dz = top_fs_sgrad_z<Order, true>(s.szz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);

                s.m_txxx[idx] = axh[x] * s.m_txxx[idx] + bxh[x] * dsxx_dx;
                s.m_txzz[idx] = az[z] * s.m_txzz[idx] + bz[z] * dsxz_dz;
                s.m_txyx[idx] = ax[x] * s.m_txyx[idx] + bx[x] * dsxy_dx;
                s.m_tyzz[idx] = az[z] * s.m_tyzz[idx] + bz[z] * dsyz_dz;
                s.m_txzx[idx] = ax[x] * s.m_txzx[idx] + bx[x] * dsxz_dx;
                s.m_tzzz[idx] = azh[z] * s.m_tzzz[idx] + bzh[z] * dszz_dz;

                dsxx_dx += s.m_txxx[idx];
                dsxz_dz += s.m_txzz[idx];
                dsxy_dx += s.m_txyx[idx];
                dsyz_dz += s.m_tyzz[idx];
                dsxz_dx += s.m_txzx[idx];
                dszz_dz += s.m_tzzz[idx];

                const float scale = dt / m.rho[idx];
                s.vx[idx] += scale * (dsxx_dx + dsxz_dz);
                s.vy[idx] += scale * (dsxy_dx + dsyz_dz);
                s.vz[idx] += scale * (dsxz_dx + dszz_dz);
            }
        }
    });
}

template <int Order>
void stress_step(
    State& s,
    ModelPtr m,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t nx,
    float inv_dz, float inv_dx, float dt,
    bool free_surface,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int64_t spatial = nz * nx;
    const int64_t row_count = B * (nz - 2 * M);
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                float dvx_dx = sgrad_backward<Order>(s.vx.data(), idx, 1, inv_dx, stencil);
                float dvy_dx = sgrad_forward<Order>(s.vy.data(), idx, 1, inv_dx, stencil);
                float dvz_dx = sgrad_forward<Order>(s.vz.data(), idx, 1, inv_dx, stencil);
                float dvz_dz = top_fs_sgrad_z<Order, false>(s.vz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                float dvx_dz = top_fs_sgrad_z<Order, true>(s.vx.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);
                float dvy_dz = top_fs_sgrad_z<Order, true>(s.vy.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);

                s.m_vxx[idx] = ax[x] * s.m_vxx[idx] + bx[x] * dvx_dx;
                s.m_vxz[idx] = azh[z] * s.m_vxz[idx] + bzh[z] * dvx_dz;
                s.m_vyx[idx] = axh[x] * s.m_vyx[idx] + bxh[x] * dvy_dx;
                s.m_vyz[idx] = azh[z] * s.m_vyz[idx] + bzh[z] * dvy_dz;
                s.m_vzx[idx] = axh[x] * s.m_vzx[idx] + bxh[x] * dvz_dx;
                s.m_vzz[idx] = az[z] * s.m_vzz[idx] + bz[z] * dvz_dz;

                dvx_dx += s.m_vxx[idx];
                dvx_dz += s.m_vxz[idx];
                dvy_dx += s.m_vyx[idx];
                dvy_dz += s.m_vyz[idx];
                dvz_dx += s.m_vzx[idx];
                dvz_dz += s.m_vzz[idx];

                if (free_surface && z == M) {
                    apply_top_traction_solve(m, idx, dvx_dx, dvy_dx, dvz_dx, dvx_dz, dvy_dz, dvz_dz);
                }

                const float shear_xz = dvz_dx + dvx_dz;
                s.sxx[idx] += dt * (m.C11[idx] * dvx_dx + m.C16[idx] * dvy_dx + m.C15[idx] * shear_xz + m.C14[idx] * dvy_dz + m.C13[idx] * dvz_dz);
                s.szz[idx] += dt * (m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * shear_xz + m.C34[idx] * dvy_dz + m.C33[idx] * dvz_dz);
                s.syz[idx] += dt * (m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * shear_xz + m.C44[idx] * dvy_dz + m.C34[idx] * dvz_dz);
                s.sxz[idx] += dt * (m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * shear_xz + m.C45[idx] * dvy_dz + m.C35[idx] * dvz_dz);
                s.sxy[idx] += dt * (m.C16[idx] * dvx_dx + m.C66[idx] * dvy_dx + m.C56[idx] * shear_xz + m.C46[idx] * dvy_dz + m.C36[idx] * dvz_dz);

                if (free_surface && z == M) {
                    s.szz[idx] = 0.0f;
                    s.syz[idx] = 0.0f;
                    s.sxz[idx] = 0.0f;
                }
            }
        }
    });
}

template <int Order>
void velocity_step_nopml(
    State& s,
    ModelPtr m,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int abcn,
    float inv_dz,
    float inv_dx,
    float dt,
    bool free_surface,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int halo = abcn + M + 1;
    const int top_halo = free_surface ? M : halo;
    const int64_t spatial = nz * nx;
    const int64_t z_count = std::max<int64_t>(0, nz - top_halo - halo);
    const int64_t row_count = B * z_count;
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / z_count;
            const int64_t z = top_halo + row - b * z_count;
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = halo; x < nx - halo; ++x) {
                const int64_t idx = base + x;
                const float dsxx_dx = sgrad_forward<Order>(s.sxx.data(), idx, 1, inv_dx, stencil);
                const float dsxz_dz = top_fs_sgrad_z<Order, false>(s.sxz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                const float dsxy_dx = sgrad_backward<Order>(s.sxy.data(), idx, 1, inv_dx, stencil);
                const float dsyz_dz = top_fs_sgrad_z<Order, false>(s.syz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                const float dsxz_dx = sgrad_backward<Order>(s.sxz.data(), idx, 1, inv_dx, stencil);
                const float dszz_dz = top_fs_sgrad_z<Order, true>(s.szz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                const float scale = dt / m.rho[idx];
                s.vx[idx] -= scale * (dsxx_dx + dsxz_dz);
                s.vy[idx] -= scale * (dsxy_dx + dsyz_dz);
                s.vz[idx] -= scale * (dsxz_dx + dszz_dz);
            }
        }
    });
}

template <int Order>
void stress_step_nopml(
    State& s,
    ModelPtr m,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int abcn,
    float inv_dz,
    float inv_dx,
    float dt,
    bool free_surface,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int halo = abcn + M + 1;
    const int top_halo = free_surface ? M : halo;
    const int64_t spatial = nz * nx;
    const int64_t z_count = std::max<int64_t>(0, nz - top_halo - halo);
    const int64_t row_count = B * z_count;
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / z_count;
            const int64_t z = top_halo + row - b * z_count;
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = halo; x < nx - halo; ++x) {
                const int64_t idx = base + x;
                const float dvx_dx = sgrad_backward<Order>(s.vx.data(), idx, 1, inv_dx, stencil);
                const float dvy_dx = sgrad_forward<Order>(s.vy.data(), idx, 1, inv_dx, stencil);
                const float dvz_dx = sgrad_forward<Order>(s.vz.data(), idx, 1, inv_dx, stencil);
                const float dvz_dz = top_fs_sgrad_z<Order, false>(s.vz.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                const float dvx_dz = top_fs_sgrad_z<Order, true>(s.vx.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);
                const float dvy_dz = top_fs_sgrad_z<Order, true>(s.vy.data(), idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);
                const float shear_xz = dvz_dx + dvx_dz;
                s.sxx[idx] -= dt * (m.C11[idx] * dvx_dx + m.C16[idx] * dvy_dx + m.C15[idx] * shear_xz + m.C14[idx] * dvy_dz + m.C13[idx] * dvz_dz);
                s.szz[idx] -= dt * (m.C13[idx] * dvx_dx + m.C36[idx] * dvy_dx + m.C35[idx] * shear_xz + m.C34[idx] * dvy_dz + m.C33[idx] * dvz_dz);
                s.syz[idx] -= dt * (m.C14[idx] * dvx_dx + m.C46[idx] * dvy_dx + m.C45[idx] * shear_xz + m.C44[idx] * dvy_dz + m.C34[idx] * dvz_dz);
                s.sxz[idx] -= dt * (m.C15[idx] * dvx_dx + m.C56[idx] * dvy_dx + m.C55[idx] * shear_xz + m.C45[idx] * dvy_dz + m.C35[idx] * dvz_dz);
                s.sxy[idx] -= dt * (m.C16[idx] * dvx_dx + m.C66[idx] * dvy_dx + m.C56[idx] * shear_xz + m.C46[idx] * dvy_dz + m.C36[idx] * dvz_dz);
            }
        }
    });
}

template <int Order>
void adjoint_step(
    State& adj,
    Workspace& work,
    ModelPtr m,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t nx,
    float inv_dz, float inv_dx, float dt,
    bool free_surface,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    auto& q0 = work.q[0];
    auto& q1 = work.q[1];
    auto& q2 = work.q[2];
    auto& q3 = work.q[3];
    auto& q4 = work.q[4];
    auto& q5 = work.q[5];

    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t z = (idx / nx) % nz;
            const bool top_row = free_surface && z == M;
            const float bar_sxx = adj.sxx[idx];
            const float bar_szz = top_row ? 0.0f : adj.szz[idx];
            const float bar_syz = top_row ? 0.0f : adj.syz[idx];
            const float bar_sxz = top_row ? 0.0f : adj.sxz[idx];
            const float bar_sxy = adj.sxy[idx];
            if (top_row) {
                adj.szz[idx] = 0.0f;
                adj.syz[idx] = 0.0f;
                adj.sxz[idx] = 0.0f;
            }

            const float bar_shear = dt * (m.C15[idx] * bar_sxx + m.C35[idx] * bar_szz + m.C45[idx] * bar_syz + m.C55[idx] * bar_sxz + m.C56[idx] * bar_sxy);
            float bar_dvx_dx = dt * (m.C11[idx] * bar_sxx + m.C13[idx] * bar_szz + m.C14[idx] * bar_syz + m.C15[idx] * bar_sxz + m.C16[idx] * bar_sxy);
            float bar_dvy_dx = dt * (m.C16[idx] * bar_sxx + m.C36[idx] * bar_szz + m.C46[idx] * bar_syz + m.C56[idx] * bar_sxz + m.C66[idx] * bar_sxy);
            float bar_dvz_dx = bar_shear;
            float bar_dvx_dz = bar_shear;
            float bar_dvy_dz = dt * (m.C14[idx] * bar_sxx + m.C34[idx] * bar_szz + m.C44[idx] * bar_syz + m.C45[idx] * bar_sxz + m.C46[idx] * bar_sxy);
            float bar_dvz_dz = dt * (m.C13[idx] * bar_sxx + m.C33[idx] * bar_szz + m.C34[idx] * bar_syz + m.C35[idx] * bar_sxz + m.C36[idx] * bar_sxy);
            if (top_row) {
                add_top_traction_solve_adjoint_h(m, idx, bar_dvx_dz, bar_dvy_dz, bar_dvz_dz, bar_dvx_dx, bar_dvy_dx, bar_dvz_dx);
                bar_dvx_dz = 0.0f;
                bar_dvy_dz = 0.0f;
                bar_dvz_dz = 0.0f;
            }

            const float tmp_vxx = adj.m_vxx[idx] + bar_dvx_dx;
            const float tmp_vxz = adj.m_vxz[idx] + bar_dvx_dz;
            const float tmp_vyx = adj.m_vyx[idx] + bar_dvy_dx;
            const float tmp_vyz = adj.m_vyz[idx] + bar_dvy_dz;
            const float tmp_vzx = adj.m_vzx[idx] + bar_dvz_dx;
            const float tmp_vzz = adj.m_vzz[idx] + bar_dvz_dz;

            q0[idx] = bar_dvx_dx + bx[x] * tmp_vxx;
            q1[idx] = bar_dvy_dx + bxh[x] * tmp_vyx;
            q2[idx] = bar_dvz_dx + bxh[x] * tmp_vzx;
            q3[idx] = bar_dvx_dz + bzh[z] * tmp_vxz;
            q4[idx] = bar_dvy_dz + bzh[z] * tmp_vyz;
            q5[idx] = bar_dvz_dz + bz[z] * tmp_vzz;

            adj.m_vxx[idx] = ax[x] * tmp_vxx;
            adj.m_vxz[idx] = azh[z] * tmp_vxz;
            adj.m_vyx[idx] = axh[x] * tmp_vyx;
            adj.m_vyz[idx] = azh[z] * tmp_vyz;
            adj.m_vzx[idx] = axh[x] * tmp_vzx;
            adj.m_vzz[idx] = az[z] * tmp_vzz;
        }
    });

    const int64_t row_count = B * (nz - 2 * M);
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                adj.vx[idx] += sgrad_forward<Order>(q0.data(), idx, 1, inv_dx, stencil) + sgrad_backward<Order>(q3.data(), idx, nx, inv_dz, stencil);
                adj.vy[idx] += sgrad_backward<Order>(q1.data(), idx, 1, inv_dx, stencil) + sgrad_backward<Order>(q4.data(), idx, nx, inv_dz, stencil);
                adj.vz[idx] += sgrad_backward<Order>(q2.data(), idx, 1, inv_dx, stencil) + sgrad_forward<Order>(q5.data(), idx, nx, inv_dz, stencil);
                adj.vx[idx] += top_fs_adjoint_sgrad_z<Order, true>(q3.data(), b, z, x, nz, nx, inv_dz, free_surface, false, stencil)
                             - sgrad_backward<Order>(q3.data(), idx, nx, inv_dz, stencil);
                adj.vy[idx] += top_fs_adjoint_sgrad_z<Order, true>(q4.data(), b, z, x, nz, nx, inv_dz, free_surface, false, stencil)
                             - sgrad_backward<Order>(q4.data(), idx, nx, inv_dz, stencil);
                adj.vz[idx] += top_fs_adjoint_sgrad_z<Order, false>(q5.data(), b, z, x, nz, nx, inv_dz, free_surface, true, stencil)
                             - sgrad_forward<Order>(q5.data(), idx, nx, inv_dz, stencil);
            }
        }
    });

    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t z = (idx / nx) % nz;
            const float scale = dt / m.rho[idx];
            const float bar_dsxx_dx = scale * adj.vx[idx];
            const float bar_dsxz_dz = scale * adj.vx[idx];
            const float bar_dsxy_dx = scale * adj.vy[idx];
            const float bar_dsyz_dz = scale * adj.vy[idx];
            const float bar_dsxz_dx = scale * adj.vz[idx];
            const float bar_dszz_dz = scale * adj.vz[idx];

            const float tmp_txxx = adj.m_txxx[idx] + bar_dsxx_dx;
            const float tmp_txzz = adj.m_txzz[idx] + bar_dsxz_dz;
            const float tmp_txyx = adj.m_txyx[idx] + bar_dsxy_dx;
            const float tmp_tyzz = adj.m_tyzz[idx] + bar_dsyz_dz;
            const float tmp_txzx = adj.m_txzx[idx] + bar_dsxz_dx;
            const float tmp_tzzz = adj.m_tzzz[idx] + bar_dszz_dz;

            q0[idx] = bar_dsxx_dx + bxh[x] * tmp_txxx;
            q1[idx] = bar_dsxz_dz + bz[z] * tmp_txzz;
            q2[idx] = bar_dsxy_dx + bx[x] * tmp_txyx;
            q3[idx] = bar_dsyz_dz + bz[z] * tmp_tyzz;
            q4[idx] = bar_dsxz_dx + bx[x] * tmp_txzx;
            q5[idx] = bar_dszz_dz + bzh[z] * tmp_tzzz;

            adj.m_txxx[idx] = axh[x] * tmp_txxx;
            adj.m_txzz[idx] = az[z] * tmp_txzz;
            adj.m_txyx[idx] = ax[x] * tmp_txyx;
            adj.m_tyzz[idx] = az[z] * tmp_tyzz;
            adj.m_txzx[idx] = ax[x] * tmp_txzx;
            adj.m_tzzz[idx] = azh[z] * tmp_tzzz;
        }
    });

    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                adj.sxx[idx] += sgrad_backward<Order>(q0.data(), idx, 1, inv_dx, stencil);
                adj.sxz[idx] += top_fs_adjoint_sgrad_z<Order, false>(q1.data(), b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                adj.sxy[idx] += sgrad_forward<Order>(q2.data(), idx, 1, inv_dx, stencil);
                adj.syz[idx] += top_fs_adjoint_sgrad_z<Order, false>(q3.data(), b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                adj.sxz[idx] += sgrad_forward<Order>(q4.data(), idx, 1, inv_dx, stencil);
                adj.szz[idx] += top_fs_adjoint_sgrad_z<Order, true>(q5.data(), b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
            }
        }
    });
}

template <int Order>
void accumulate_grad(
    const State& adj,
    ModelPtr model,
    GradPtr grad,
    const float* fvx,
    const float* fvy,
    const float* fvz,
    const float* fvx_prev,
    const float* fvy_prev,
    const float* fvz_prev,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float dt,
    bool free_surface,
    bool do_stiffness,
    bool do_rho,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);
    const int64_t spatial = nz * nx;
    const int64_t row_count = B * (nz - 2 * M);
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float dvx_dx = sgrad_backward<Order>(fvx, idx, 1, inv_dx, stencil);
                const float dvy_dx = sgrad_forward<Order>(fvy, idx, 1, inv_dx, stencil);
                const float dvz_dx = sgrad_forward<Order>(fvz, idx, 1, inv_dx, stencil);
                float dvz_dz = top_fs_sgrad_z<Order, false>(fvz, idx, b, z, x, nz, nx, inv_dz, free_surface, true, stencil);
                float dvx_dz = top_fs_sgrad_z<Order, true>(fvx, idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);
                float dvy_dz = top_fs_sgrad_z<Order, true>(fvy, idx, b, z, x, nz, nx, inv_dz, free_surface, false, stencil);

                const float bar_sxx = adj.sxx[idx];
                const bool top_row = free_surface && z == M;
                const float bar_szz = top_row ? 0.0f : adj.szz[idx];
                const float bar_syz = top_row ? 0.0f : adj.syz[idx];
                const float bar_sxz = top_row ? 0.0f : adj.sxz[idx];
                const float bar_sxy = adj.sxy[idx];
                const float scale = -dt;
                if (top_row) {
                    apply_top_traction_solve(model, idx, dvx_dx, dvy_dx, dvz_dx, dvx_dz, dvy_dz, dvz_dz);
                }
                const float shear_xz = dvz_dx + dvx_dz;

                if (do_stiffness) {
                    grad.C11[idx] += scale * bar_sxx * dvx_dx;
                    grad.C13[idx] += scale * (bar_sxx * dvz_dz + bar_szz * dvx_dx);
                    grad.C14[idx] += scale * (bar_sxx * dvy_dz + bar_syz * dvx_dx);
                    grad.C15[idx] += scale * (bar_sxx * shear_xz + bar_sxz * dvx_dx);
                    grad.C16[idx] += scale * (bar_sxx * dvy_dx + bar_sxy * dvx_dx);
                    grad.C33[idx] += scale * bar_szz * dvz_dz;
                    grad.C34[idx] += scale * (bar_szz * dvy_dz + bar_syz * dvz_dz);
                    grad.C35[idx] += scale * (bar_szz * shear_xz + bar_sxz * dvz_dz);
                    grad.C36[idx] += scale * (bar_szz * dvy_dx + bar_sxy * dvz_dz);
                    grad.C44[idx] += scale * bar_syz * dvy_dz;
                    grad.C45[idx] += scale * (bar_syz * shear_xz + bar_sxz * dvy_dz);
                    grad.C46[idx] += scale * (bar_syz * dvy_dx + bar_sxy * dvy_dz);
                    grad.C55[idx] += scale * bar_sxz * shear_xz;
                    grad.C56[idx] += scale * (bar_sxz * dvy_dx + bar_sxy * shear_xz);
                    grad.C66[idx] += scale * bar_sxy * dvy_dx;
                    if (top_row) {
                        const float bar_dvx_dz = scale * (model.C15[idx] * bar_sxx + model.C56[idx] * bar_sxy);
                        const float bar_dvy_dz = scale * (model.C14[idx] * bar_sxx + model.C46[idx] * bar_sxy);
                        const float bar_dvz_dz = scale * (model.C13[idx] * bar_sxx + model.C36[idx] * bar_sxy);
                        accumulate_top_traction_solve_model_grad(model, grad, idx, dvx_dx, dvy_dx, dvz_dx, dvx_dz, dvy_dz, dvz_dz, bar_dvx_dz, bar_dvy_dz, bar_dvz_dz);
                    }
                }
                if (do_rho) {
                    grad.rho[idx] += (
                        adj.vx[idx] * (fvx_prev[idx] - fvx[idx]) +
                        adj.vy[idx] * (fvy_prev[idx] - fvy[idx]) +
                        adj.vz[idx] * (fvz_prev[idx] - fvz[idx])
                    ) / model.rho[idx];
                }
            }
        }
    });
}

template <int Order>
CpuForwardResult forward_raw_impl(const ForwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    auto rho_t = p.models[0];
    const int64_t B = rho_t.size(0) * rho_t.size(1);
    const int64_t nz = rho_t.size(2);
    const int64_t nx = rho_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nsrc = p.sources_loc.size(1);
    const int64_t nrec = p.receivers_loc.size(1);
    const int64_t nsrc_fields = p.source_field_indices.numel();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t nt = static_cast<int64_t>(p.nt);

    auto record = torch::zeros({nrec_fields, B, nrec, nt}, rho_t.options());
    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = torch::zeros({nt, kNumFields, B, nz, nx}, rho_t.options());

    State state(total);
    const auto m = model_ptr(p.models);
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ax = pml[4].data_ptr<float>();
    const float* bx = pml[5].data_ptr<float>();
    const float* axh = pml[6].data_ptr<float>();
    const float* bxh = pml[7].data_ptr<float>();
    const auto h = spacing_for(p);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float dt = static_cast<float>(p.dt);
    const auto source_fields_cpu = p.source_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const auto receiver_fields_cpu = p.receiver_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* source_fields = source_fields_cpu.data_ptr<int32_t>();
    const int32_t* receiver_fields = receiver_fields_cpu.data_ptr<int32_t>();
    const int32_t* sources = p.sources_loc.data_ptr<int32_t>();
    const int32_t* receivers = p.receivers_loc.data_ptr<int32_t>();
    const float* source = p.source.data_ptr<float>();
    float* rec = record.data_ptr<float>();
    auto boundary = active_boundary_tensors(p);
    const int save_width = p.M + 1;
    if (p.use_boundary_saving) {
        if (p.boundary_on_disk) {
            TORCH_CHECK(p.boundary_disk_files.size() == 4, "ElasticTTISG CPU disk boundary-saving forward requires 4 files");
        } else {
            TORCH_CHECK(!boundary.empty(), "ElasticTTISG CPU boundary-saving forward requires boundary tensors");
        }
    }

    for (int64_t it = 0; it < nt; ++it) {
        velocity_step<Order>(state, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
        stress_step<Order>(state, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
        if (u_allt.defined()) {
            auto fields = const_field_ptrs(state);
            for (int f = 0; f < kNumFields; ++f) {
                std::copy(fields[f], fields[f] + total, u_allt.select(0, it).select(0, f).data_ptr<float>());
            }
        }
        auto fields = mutable_field_ptrs(state);
        for (int64_t f = 0; f < nsrc_fields; ++f) {
            const int field_id = source_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field(fields[field_id], source, sources, B, nsrc, nt, it, nz, nx);
            }
        }
        if (p.use_boundary_saving) {
            auto saved_fields = const_field_ptrs(state);
            for (int f = 0; f < kNumFields; ++f) {
                if (p.boundary_on_disk) {
                    save_boundary_field_disk(p.boundary_disk_files, f, static_cast<int>(it), static_cast<int>(nt), saved_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                } else {
                    save_boundary_field(boundary, f, static_cast<int>(it), saved_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                }
            }
        }
        auto cfields = const_field_ptrs(state);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(cfields.size())) {
                record_field(cfields[field_id], rec + f * B * nrec * nt, receivers, B, nrec, nt, it, nz, nx);
            }
        }
        if (p.use_checkpoint) {
            save_state_to_checkpoint(p.checkpoints, checkpoint_index(static_cast<int>(it), static_cast<int>(nt), p.checkpoint_interval), state);
        }
    }

    torch::Tensor last_two = p.last_two.defined() ? p.last_two : torch::empty({0}, rho_t.options());
    if (p.use_boundary_saving) {
        if (!last_two.defined() || last_two.numel() == 0) last_two = torch::zeros({kNumFields, 1, B, 1, nz, nx}, rho_t.options());
        auto fields = const_field_ptrs(state);
        for (int f = 0; f < kNumFields; ++f) {
            std::copy(fields[f], fields[f] + total, last_two.select(0, f).select(0, 0).data_ptr<float>());
        }
    }
    return {record, u_allt, last_two};
}

CpuForwardResult forward_raw(const ForwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, forward_raw_impl, p);
}

template <int Order>
BackwardOutput backward_full_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    auto rho_t = p.models[0];
    const int64_t B = rho_t.size(0) * rho_t.size(1);
    const int64_t nz = rho_t.size(2);
    const int64_t nx = rho_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nt = static_cast<int64_t>(p.nt);
    TORCH_CHECK(p.u_forward.defined() && p.u_forward.dim() == 5 && p.u_forward.size(0) == nt && p.u_forward.size(1) == kNumFields,
                "ElasticTTISG full CPU backward requires forward wavefields with shape (nt, 8, B, nz, nx)");

    std::vector<torch::Tensor> grads;
    grads.reserve(kNumModels);
    for (const auto& model : p.models) grads.push_back(torch::zeros_like(model));
    auto grad = grad_ptr(grads);
    const auto m = model_ptr(p.models);
    State adj(total);
    Workspace work(total);
    std::vector<float> zero(total, 0.0f);
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ax = pml[4].data_ptr<float>();
    const float* bx = pml[5].data_ptr<float>();
    const float* axh = pml[6].data_ptr<float>();
    const float* bxh = pml[7].data_ptr<float>();
    const auto h = spacing_for(p);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float dt = static_cast<float>(p.dt);
    const auto receiver_fields_cpu = p.receiver_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* receiver_fields = receiver_fields_cpu.data_ptr<int32_t>();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    const float* adj_source = p.adjoint_source.data_ptr<float>();

    for (int64_t it = nt - 1; it >= 0; --it) {
        auto fields = mutable_field_ptrs(adj);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field(fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources, B, adjoint_nsrc, nt, it, nz, nx);
            }
        }

        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vy_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 2).data_ptr<float>();
        const float* vx_prev = (it > 0) ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>() : zero.data();
        const float* vy_prev = (it > 0) ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>() : zero.data();
        const float* vz_prev = (it > 0) ? p.u_forward.select(0, it - 1).select(0, 2).data_ptr<float>() : zero.data();
        accumulate_grad<Order>(adj, m, grad, vx_now, vy_now, vz_now, vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, true, false, stencil);
        adjoint_step<Order>(adj, work, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
        accumulate_grad<Order>(adj, m, grad, vx_now, vy_now, vz_now, vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, false, true, stencil);
    }
    BackwardOutput out;
    out.grads = grads;
    return out;
}

template <int Order>
BackwardOutput backward_ckpt_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    auto rho_t = p.models[0];
    const int64_t B = rho_t.size(0) * rho_t.size(1);
    const int64_t nz = rho_t.size(2);
    const int64_t nx = rho_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nt = static_cast<int64_t>(p.nt);
    const int chunk_size = std::max(1, p.checkpoint_interval);
    const int num_chunks = (static_cast<int>(nt) + chunk_size - 1) / chunk_size;
    TORCH_CHECK(p.checkpoints.size() == kNumState, "ElasticTTISG CPU checkpoint backward expects 20 checkpoint tensors");

    std::vector<torch::Tensor> grads;
    grads.reserve(kNumModels);
    for (const auto& model : p.models) grads.push_back(torch::zeros_like(model));
    auto grad = grad_ptr(grads);
    const auto m = model_ptr(p.models);
    State adj(total);
    Workspace work(total);
    std::vector<float> zero(total, 0.0f);
    std::vector<float> start_prev_vx(total, 0.0f), start_prev_vy(total, 0.0f), start_prev_vz(total, 0.0f);

    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ax = pml[4].data_ptr<float>();
    const float* bx = pml[5].data_ptr<float>();
    const float* axh = pml[6].data_ptr<float>();
    const float* bxh = pml[7].data_ptr<float>();
    const auto h = spacing_for(p);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float dt = static_cast<float>(p.dt);

    const auto receiver_fields_cpu = p.receiver_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* receiver_fields = receiver_fields_cpu.data_ptr<int32_t>();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t forward_nsrc = p.forward_sources_loc.size(1);
    const int64_t nsrc_fields = p.source_field_indices.numel();
    const auto source_fields_cpu = p.source_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* source_fields = source_fields_cpu.data_ptr<int32_t>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    const int32_t* forward_sources = p.forward_sources_loc.data_ptr<int32_t>();
    const float* adj_source = p.adjoint_source.data_ptr<float>();
    const float* forward_source = p.forward_source.data_ptr<float>();

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        const int start = chunk_id * chunk_size;
        const int end = std::min(static_cast<int>(nt), start + chunk_size);
        const int len = end - start;
        State fwd(total);
        if (chunk_id > 0) {
            load_state_from_checkpoint(p.checkpoints, chunk_id, fwd);
        }
        if (start > 0) {
            std::copy(fwd.vx.begin(), fwd.vx.end(), start_prev_vx.begin());
            std::copy(fwd.vy.begin(), fwd.vy.end(), start_prev_vy.begin());
            std::copy(fwd.vz.begin(), fwd.vz.end(), start_prev_vz.begin());
        } else {
            std::fill(start_prev_vx.begin(), start_prev_vx.end(), 0.0f);
            std::fill(start_prev_vy.begin(), start_prev_vy.end(), 0.0f);
            std::fill(start_prev_vz.begin(), start_prev_vz.end(), 0.0f);
        }

        std::vector<float> chunk_wavefields(static_cast<size_t>(len) * kNumFields * total, 0.0f);
        for (int it = start; it < end; ++it) {
            velocity_step<Order>(fwd, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
            stress_step<Order>(fwd, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
            auto fields = const_field_ptrs(fwd);
            float* base = chunk_wavefields.data() + static_cast<size_t>(it - start) * kNumFields * total;
            for (int f = 0; f < kNumFields; ++f) {
                std::copy(fields[f], fields[f] + total, base + static_cast<size_t>(f) * total);
            }
            auto mfields = mutable_field_ptrs(fwd);
            for (int64_t f = 0; f < nsrc_fields; ++f) {
                const int field_id = source_fields[f];
                if (field_id >= 0 && field_id < static_cast<int>(mfields.size())) {
                    add_source_to_field(mfields[field_id], forward_source, forward_sources, B, forward_nsrc, nt, it, nz, nx);
                }
            }
        }

        for (int it = end - 1; it >= start; --it) {
            auto fields = mutable_field_ptrs(adj);
            for (int64_t f = 0; f < nrec_fields; ++f) {
                const int field_id = receiver_fields[f];
                if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                    add_source_to_field(fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources, B, adjoint_nsrc, nt, it, nz, nx);
                }
            }

            const float* base = chunk_wavefields.data() + static_cast<size_t>(it - start) * kNumFields * total;
            const float* vx_now = base + 0 * total;
            const float* vy_now = base + 1 * total;
            const float* vz_now = base + 2 * total;
            const float* vx_prev = nullptr;
            const float* vy_prev = nullptr;
            const float* vz_prev = nullptr;
            if (it > start) {
                const float* prev_base = chunk_wavefields.data() + static_cast<size_t>(it - 1 - start) * kNumFields * total;
                vx_prev = prev_base + 0 * total;
                vy_prev = prev_base + 1 * total;
                vz_prev = prev_base + 2 * total;
            } else if (it > 0) {
                vx_prev = start_prev_vx.data();
                vy_prev = start_prev_vy.data();
                vz_prev = start_prev_vz.data();
            } else {
                vx_prev = zero.data();
                vy_prev = zero.data();
                vz_prev = zero.data();
            }
            accumulate_grad<Order>(adj, m, grad, vx_now, vy_now, vz_now, vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, true, false, stencil);
            adjoint_step<Order>(adj, work, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
            accumulate_grad<Order>(adj, m, grad, vx_now, vy_now, vz_now, vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, false, true, stencil);
        }
    }

    BackwardOutput out;
    out.grads = grads;
    return out;
}

template <int Order>
BackwardOutput backward_bs_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    auto rho_t = p.models[0];
    const int64_t B = rho_t.size(0) * rho_t.size(1);
    const int64_t nz = rho_t.size(2);
    const int64_t nx = rho_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nt = static_cast<int64_t>(p.nt);
    const int save_width = p.M + 1;
    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() >= kNumFields * total,
                "ElasticTTISG CPU boundary-saving backward requires last_two");
    auto boundary = active_boundary_tensors(p);
    if (p.boundary_on_disk) {
        TORCH_CHECK(p.boundary_disk_files.size() == 4, "ElasticTTISG CPU disk boundary-saving backward requires 4 files");
    } else {
        TORCH_CHECK(!boundary.empty(), "ElasticTTISG CPU boundary-saving backward requires boundary tensors");
    }

    std::vector<torch::Tensor> grads;
    grads.reserve(kNumModels);
    for (const auto& model : p.models) grads.push_back(torch::zeros_like(model));
    auto grad = grad_ptr(grads);
    const auto m = model_ptr(p.models);
    State adj(total);
    State fwd(total);
    Workspace work(total);
    std::vector<float> zero(total, 0.0f);
    std::vector<float> fvx_now(total, 0.0f), fvy_now(total, 0.0f), fvz_now(total, 0.0f);

    std::array<std::vector<float>*, kNumFields> base_fields = {
        &fwd.vx, &fwd.vy, &fwd.vz, &fwd.sxx, &fwd.szz, &fwd.syz, &fwd.sxz, &fwd.sxy,
    };
    for (int f = 0; f < kNumFields; ++f) {
        copy_tensor_to_vector(p.u_last_two.select(0, f).select(0, 0), *base_fields[f]);
    }

    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ax = pml[4].data_ptr<float>();
    const float* bx = pml[5].data_ptr<float>();
    const float* axh = pml[6].data_ptr<float>();
    const float* bxh = pml[7].data_ptr<float>();
    const auto h = spacing_for(p);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float dt = static_cast<float>(p.dt);

    const auto receiver_fields_cpu = p.receiver_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* receiver_fields = receiver_fields_cpu.data_ptr<int32_t>();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t forward_nsrc = p.forward_sources_loc.size(1);
    const int64_t nsrc_fields = p.source_field_indices.numel();
    const auto source_fields_cpu = p.source_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* source_fields = source_fields_cpu.data_ptr<int32_t>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    const int32_t* forward_sources = p.forward_sources_loc.data_ptr<int32_t>();
    const float* adj_source = p.adjoint_source.data_ptr<float>();
    const float* forward_source = p.forward_source.data_ptr<float>();

    for (int64_t it = nt - 1; it >= 0; --it) {
        auto adj_fields = mutable_field_ptrs(adj);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(adj_fields.size())) {
                add_source_to_field(adj_fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources, B, adjoint_nsrc, nt, it, nz, nx);
            }
        }

        auto fwd_fields = mutable_field_ptrs(fwd);
        for (int64_t f = 0; f < nsrc_fields; ++f) {
            const int field_id = source_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fwd_fields.size())) {
                add_source_to_field(fwd_fields[field_id], forward_source, forward_sources, B, forward_nsrc, nt, it, nz, nx, -1.0f);
            }
        }

        stress_step_nopml<Order>(fwd, m, B, nz, nx, p.abcn, inv_dz, inv_dx, dt, p.free_surface, stencil);
        fwd_fields = mutable_field_ptrs(fwd);
        if (it > 0) {
            for (int f = 3; f < kNumFields; ++f) {
                if (p.boundary_on_disk) {
                    restore_boundary_field_disk(p.boundary_disk_files, f, static_cast<int>(it - 1), static_cast<int>(nt), fwd_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                } else {
                    restore_boundary_field(boundary, f, static_cast<int>(it - 1), fwd_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                }
            }
        }

        std::copy(fwd.vx.begin(), fwd.vx.end(), fvx_now.begin());
        std::copy(fwd.vy.begin(), fwd.vy.end(), fvy_now.begin());
        std::copy(fwd.vz.begin(), fwd.vz.end(), fvz_now.begin());

        const float* vx_prev = zero.data();
        const float* vy_prev = zero.data();
        const float* vz_prev = zero.data();
        if (it > 0) {
            velocity_step_nopml<Order>(fwd, m, B, nz, nx, p.abcn, inv_dz, inv_dx, dt, p.free_surface, stencil);
            fwd_fields = mutable_field_ptrs(fwd);
            for (int f = 0; f < 3; ++f) {
                if (p.boundary_on_disk) {
                    restore_boundary_field_disk(p.boundary_disk_files, f, static_cast<int>(it - 1), static_cast<int>(nt), fwd_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                } else {
                    restore_boundary_field(boundary, f, static_cast<int>(it - 1), fwd_fields[f], B, nz, nx, p.M, p.abcn, p.free_surface, save_width);
                }
            }
            vx_prev = fwd.vx.data();
            vy_prev = fwd.vy.data();
            vz_prev = fwd.vz.data();
        }

        accumulate_grad<Order>(adj, m, grad, fvx_now.data(), fvy_now.data(), fvz_now.data(), vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, true, false, stencil);
        adjoint_step<Order>(adj, work, m, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, stencil);
        accumulate_grad<Order>(adj, m, grad, fvx_now.data(), fvy_now.data(), fvz_now.data(), vx_prev, vy_prev, vz_prev, B, nz, nx, inv_dz, inv_dx, dt, p.free_surface, false, true, stencil);
    }

    BackwardOutput out;
    out.grads = grads;
    return out;
}

template <int Order>
BackwardOutput backward_replay_impl(const BackwardInput& p)
{
    ForwardInput fwd;
    fwd.models = p.models;
    fwd.source = p.forward_source;
    fwd.lap_coes = p.lap_coes;
    fwd.grad_coes = p.grad_coes;
    fwd.M = p.M;
    fwd.abcn = p.abcn;
    fwd.sources_loc = p.forward_sources_loc;
    fwd.receivers_loc = p.adjoint_sources_loc;
    fwd.source_field_indices = p.source_field_indices;
    fwd.receiver_field_indices = p.receiver_field_indices;
    fwd.pml_vals = p.pml_vals;
    fwd.free_surface = p.free_surface;
    fwd.nt = p.nt;
    fwd.dt = p.dt;
    fwd.spacing = p.spacing;
    fwd.save_all_wavefields = true;
    fwd.use_boundary_saving = false;
    fwd.use_checkpoint = false;
    fwd.use_recursive_checkpoint = false;
    auto forward_result = forward_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_full_impl<Order>(replay);
}

BackwardOutput backward_raw(const BackwardInput& p)
{
    const bool has_full_wavefield = p.u_forward.defined() && p.u_forward.numel() > 0;
    #define SWEEP_CPU_TTI_SG_BACKWARD_CASE(ORDER) \
        if (has_full_wavefield) return backward_full_impl<ORDER>(p); \
        return backward_replay_impl<ORDER>(p);
    SWEEP_CPU_DISPATCH_STENCIL_BODY(p.M, SWEEP_CPU_TTI_SG_BACKWARD_CASE);
    #undef SWEEP_CPU_TTI_SG_BACKWARD_CASE
}

BackwardOutput backward_bs_raw(const BackwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, backward_bs_impl, p);
}

BackwardOutput backward_ckpt_raw(const BackwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, backward_ckpt_impl, p);
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic_tti_sg2d::forward called with non-CPU tensors");
    TORCH_CHECK(can_use_forward(in), "ElasticTTISG CPU forward requires contiguous float32 prepared models and cpmls PML.");
    auto result = forward_raw(in);
    ForwardOutput out;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic_tti_sg2d::backward called with non-CPU tensors");
    TORCH_CHECK(can_use_backward(in), "ElasticTTISG CPU backward requires handwritten raw float32 path inputs.");
    return backward_raw(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic_tti_sg2d::backward_bs called with non-CPU tensors");
    TORCH_CHECK(can_use_backward(in), "ElasticTTISG CPU boundary-saving backward requires handwritten raw float32 path inputs.");
    return backward_bs_raw(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic_tti_sg2d::backward_ckpt called with non-CPU tensors");
    TORCH_CHECK(can_use_backward(in), "ElasticTTISG CPU checkpoint backward requires handwritten raw float32 path inputs.");
    return backward_ckpt_raw(in);
}
BackwardOutput backward_recursive_ckpt(const BackwardInput& in) { return backward(in); }

} // namespace sweep_cpu::elastic_tti_sg2d

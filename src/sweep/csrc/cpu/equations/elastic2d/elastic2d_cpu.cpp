#include "elastic2d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace sweep_cpu::elastic2d {
namespace {

using sweep_cpu::ops::StencilCoefficients;
struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

struct Elastic2DRawState {
    std::vector<float> vx;
    std::vector<float> vz;
    std::vector<float> sxx;
    std::vector<float> szz;
    std::vector<float> sxz;
    std::vector<float> m_vxx;
    std::vector<float> m_vxz;
    std::vector<float> m_vzx;
    std::vector<float> m_vzz;
    std::vector<float> m_sxxx;
    std::vector<float> m_sxxz;
    std::vector<float> m_szzx;
    std::vector<float> m_szzz;
    std::vector<float> m_sxzx;
    std::vector<float> m_sxzz;

    Elastic2DRawState() = default;

    explicit Elastic2DRawState(int64_t total)
        : vx(total, 0.0f),
          vz(total, 0.0f),
          sxx(total, 0.0f),
          szz(total, 0.0f),
          sxz(total, 0.0f),
          m_vxx(total, 0.0f),
          m_vxz(total, 0.0f),
          m_vzx(total, 0.0f),
          m_vzz(total, 0.0f),
          m_sxxx(total, 0.0f),
          m_sxxz(total, 0.0f),
          m_szzx(total, 0.0f),
          m_szzz(total, 0.0f),
          m_sxzx(total, 0.0f),
          m_sxzz(total, 0.0f)
    {}
};

struct Elastic2DWorkspace {
    std::vector<float> qxx;
    std::vector<float> qzz;
    std::vector<float> qxz;
    std::vector<float> qzx;
    std::vector<float> pxx;
    std::vector<float> pzz;
    std::vector<float> pxz;
    std::vector<float> pzx;

    explicit Elastic2DWorkspace(int64_t total)
        : qxx(total, 0.0f), qzz(total, 0.0f), qxz(total, 0.0f), qzx(total, 0.0f),
          pxx(total, 0.0f), pzz(total, 0.0f), pxz(total, 0.0f), pzx(total, 0.0f)
    {}
};

std::vector<double> spacing_for(const ForwardInput& p)
{
    return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
}

std::vector<double> spacing_for(const BackwardInput& p)
{
    return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
}

using sweep_cpu::ops::sgrad_backward;
using sweep_cpu::ops::sgrad_forward;

bool can_use_elastic2d_raw_forward(const ForwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 3 || p.pml_vals.size() < 8) return false;
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

bool can_use_elastic2d_raw_backward(const BackwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 3 || p.pml_vals.size() < 8) return false;
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
    const int64_t nz = p.models[0].size(2);
    const int64_t nx = p.models[0].size(3);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz || p.pml_vals[3].numel() != nz) return false;
    if (p.pml_vals[4].numel() != nx || p.pml_vals[5].numel() != nx || p.pml_vals[6].numel() != nx || p.pml_vals[7].numel() != nx) return false;
    return true;
}

void copy_vector_to_tensor(const std::vector<float>& src, torch::Tensor tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(src.size()) == tensor.numel(), "Tensor/vector size mismatch");
    std::copy(src.begin(), src.end(), tensor.data_ptr<float>());
}

void save_checkpoint(const std::vector<torch::Tensor>& checkpoints, int checkpoint_idx, const Elastic2DRawState& s)
{
    if (checkpoint_idx < 0) return;
    if (checkpoints.empty()) return;
    TORCH_CHECK(checkpoints.size() == 15, "Elastic2D CPU checkpointing expects 15 checkpoint tensors");
    copy_vector_to_tensor(s.vx, checkpoints[0].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.vz, checkpoints[1].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.sxx, checkpoints[2].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.szz, checkpoints[3].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.sxz, checkpoints[4].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_vxx, checkpoints[5].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_vxz, checkpoints[6].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_vzx, checkpoints[7].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_vzz, checkpoints[8].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_sxxx, checkpoints[9].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_sxxz, checkpoints[10].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_szzx, checkpoints[11].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_szzz, checkpoints[12].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_sxzx, checkpoints[13].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.m_sxzz, checkpoints[14].select(0, checkpoint_idx));
}

std::vector<int> checkpoint_steps_vector(const torch::Tensor& steps)
{
    std::vector<int> out;
    if (!steps.defined() || steps.numel() == 0) return out;
    auto cpu_steps = steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* ptr = cpu_steps.data_ptr<int32_t>();
    out.reserve(cpu_steps.numel());
    for (int64_t i = 0; i < cpu_steps.numel(); ++i) out.push_back(static_cast<int>(ptr[i]));
    return out;
}

int checkpoint_index(int it, int nt, int interval)
{
    if (interval < 1) return -1;
    if (((it + 1) % interval == 0) && (it + 1 < nt)) return (it + 1) / interval;
    return -1;
}

int recursive_checkpoint_index(int it, const std::vector<int>& steps, int& next_idx)
{
    if (next_idx < static_cast<int>(steps.size()) && steps[next_idx] == it + 1) return next_idx++;
    return -1;
}

std::vector<float*> mutable_field_ptrs(Elastic2DRawState& s)
{
    return {s.vx.data(), s.vz.data(), s.sxx.data(), s.szz.data(), s.sxz.data()};
}

const std::vector<const float*> const_field_ptrs(const Elastic2DRawState& s)
{
    return {s.vx.data(), s.vz.data(), s.sxx.data(), s.szz.data(), s.sxz.data()};
}

void build_lame(const float* vp, const float* vs, const float* rho, std::vector<float>& lambda, std::vector<float>& mu, int64_t total)
{
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float vs2 = vs[i] * vs[i];
            mu[i] = rho[i] * vs2;
            lambda[i] = rho[i] * (vp[i] * vp[i] - 2.0f * vs2);
        }
    });
}

template <int Order>
void elastic2d_velocity_step(
    Elastic2DRawState& s,
    const float* rho,
    const float* az,
    const float* bz,
    const float* azh,
    const float* bzh,
    const float* ax,
    const float* bx,
    const float* axh,
    const float* bxh,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float dt
,
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
                float dsxz_dz = sgrad_backward<Order>(s.sxz.data(), idx, nx, inv_dz, stencil);
                float dsxz_dx = sgrad_backward<Order>(s.sxz.data(), idx, 1, inv_dx, stencil);
                float dszz_dz = sgrad_forward<Order>(s.szz.data(), idx, nx, inv_dz, stencil);

                s.m_szzz[idx] = azh[z] * s.m_szzz[idx] + bzh[z] * dszz_dz;
                dszz_dz += s.m_szzz[idx];
                s.m_sxzx[idx] = ax[x] * s.m_sxzx[idx] + bx[x] * dsxz_dx;
                dsxz_dx += s.m_sxzx[idx];
                s.m_sxzz[idx] = az[z] * s.m_sxzz[idx] + bz[z] * dsxz_dz;
                dsxz_dz += s.m_sxzz[idx];
                s.m_sxxx[idx] = axh[x] * s.m_sxxx[idx] + bxh[x] * dsxx_dx;
                dsxx_dx += s.m_sxxx[idx];

                const float inv_rho = 1.0f / rho[idx];
                s.vx[idx] += dt * inv_rho * (dsxx_dx + dsxz_dz);
                s.vz[idx] += dt * inv_rho * (dsxz_dx + dszz_dz);
            }
        }
    });
}

template <int Order>
void elastic2d_stress_step(
    Elastic2DRawState& s,
    const float* lambda,
    const float* mu,
    const float* az,
    const float* bz,
    const float* azh,
    const float* bzh,
    const float* ax,
    const float* bx,
    const float* axh,
    const float* bxh,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float dt
,
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
                float dvz_dz = sgrad_backward<Order>(s.vz.data(), idx, nx, inv_dz, stencil);
                float dvx_dz = sgrad_forward<Order>(s.vx.data(), idx, nx, inv_dz, stencil);
                float dvz_dx = sgrad_forward<Order>(s.vz.data(), idx, 1, inv_dx, stencil);

                s.m_vzz[idx] = az[z] * s.m_vzz[idx] + bz[z] * dvz_dz;
                dvz_dz += s.m_vzz[idx];
                s.m_vxx[idx] = ax[x] * s.m_vxx[idx] + bx[x] * dvx_dx;
                dvx_dx += s.m_vxx[idx];

                const float lam = lambda[idx];
                const float mu0 = mu[idx];
                s.sxx[idx] += dt * ((lam + 2.0f * mu0) * dvx_dx + lam * dvz_dz);
                s.szz[idx] += dt * ((lam + 2.0f * mu0) * dvz_dz + lam * dvx_dx);

                s.m_vxz[idx] = azh[z] * s.m_vxz[idx] + bzh[z] * dvx_dz;
                dvx_dz += s.m_vxz[idx];
                s.m_vzx[idx] = axh[x] * s.m_vzx[idx] + bxh[x] * dvz_dx;
                dvz_dx += s.m_vzx[idx];
                s.sxz[idx] += dt * mu0 * (dvx_dz + dvz_dx);
            }
        }
    });
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
    int64_t nx
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
                base[z * nx + x] += source[(b * nsrc + isrc) * nt + it];
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
void elastic2d_adjoint_step(
    Elastic2DRawState& adj,
    Elastic2DWorkspace& work,
    const float* lambda,
    const float* mu,
    const float* rho,
    const float* az,
    const float* bz,
    const float* azh,
    const float* bzh,
    const float* ax,
    const float* bx,
    const float* axh,
    const float* bxh,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float dt
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t z = (idx / nx) % nz;
            const float lam = lambda[idx];
            const float mu0 = mu[idx];
            const float bar_sxx = adj.sxx[idx];
            const float bar_szz = adj.szz[idx];
            const float bar_sxz = adj.sxz[idx];

            const float bar_dvx_dx = dt * ((lam + 2.0f * mu0) * bar_sxx + lam * bar_szz);
            const float bar_dvz_dz = dt * ((lam + 2.0f * mu0) * bar_szz + lam * bar_sxx);
            const float bar_dvx_dz = dt * mu0 * bar_sxz;
            const float bar_dvz_dx = dt * mu0 * bar_sxz;

            const float tmp_vxx = adj.m_vxx[idx] + bar_dvx_dx;
            const float tmp_vzz = adj.m_vzz[idx] + bar_dvz_dz;
            const float tmp_vxz = adj.m_vxz[idx] + bar_dvx_dz;
            const float tmp_vzx = adj.m_vzx[idx] + bar_dvz_dx;

            work.qxx[idx] = bar_dvx_dx + bx[x] * tmp_vxx;
            work.qzz[idx] = bar_dvz_dz + bz[z] * tmp_vzz;
            work.qxz[idx] = bar_dvx_dz + bzh[z] * tmp_vxz;
            work.qzx[idx] = bar_dvz_dx + bxh[x] * tmp_vzx;

            adj.m_vxx[idx] = ax[x] * tmp_vxx;
            adj.m_vzz[idx] = az[z] * tmp_vzz;
            adj.m_vxz[idx] = azh[z] * tmp_vxz;
            adj.m_vzx[idx] = axh[x] * tmp_vzx;
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
                const float dqxx_dx = sgrad_forward<Order>(work.qxx.data(), idx, 1, inv_dx, stencil);
                const float dqxz_dz = sgrad_backward<Order>(work.qxz.data(), idx, nx, inv_dz, stencil);
                const float dqzx_dx = sgrad_backward<Order>(work.qzx.data(), idx, 1, inv_dx, stencil);
                const float dqzz_dz = sgrad_forward<Order>(work.qzz.data(), idx, nx, inv_dz, stencil);
                adj.vx[idx] += dqxx_dx + dqxz_dz;
                adj.vz[idx] += dqzx_dx + dqzz_dz;
            }
        }
    });

    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t z = (idx / nx) % nz;
            const float inv_rho = 1.0f / rho[idx];
            const float bar_dsxx_dx = dt * inv_rho * adj.vx[idx];
            const float bar_dsxz_dz = dt * inv_rho * adj.vx[idx];
            const float bar_dsxz_dx = dt * inv_rho * adj.vz[idx];
            const float bar_dszz_dz = dt * inv_rho * adj.vz[idx];

            const float tmp_sxxx = adj.m_sxxx[idx] + bar_dsxx_dx;
            const float tmp_sxzz = adj.m_sxzz[idx] + bar_dsxz_dz;
            const float tmp_sxzx = adj.m_sxzx[idx] + bar_dsxz_dx;
            const float tmp_szzz = adj.m_szzz[idx] + bar_dszz_dz;

            work.pxx[idx] = bar_dsxx_dx + bxh[x] * tmp_sxxx;
            work.pxz[idx] = bar_dsxz_dz + bz[z] * tmp_sxzz;
            work.pzx[idx] = bar_dsxz_dx + bx[x] * tmp_sxzx;
            work.pzz[idx] = bar_dszz_dz + bzh[z] * tmp_szzz;

            adj.m_sxxx[idx] = axh[x] * tmp_sxxx;
            adj.m_sxzz[idx] = az[z] * tmp_sxzz;
            adj.m_sxzx[idx] = ax[x] * tmp_sxzx;
            adj.m_szzz[idx] = azh[z] * tmp_szzz;
        }
    });

    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float dpxx_dx = sgrad_backward<Order>(work.pxx.data(), idx, 1, inv_dx, stencil);
                const float dpzz_dz = sgrad_backward<Order>(work.pzz.data(), idx, nx, inv_dz, stencil);
                const float dpxz_dz = sgrad_forward<Order>(work.pxz.data(), idx, nx, inv_dz, stencil);
                const float dpzx_dx = sgrad_forward<Order>(work.pzx.data(), idx, 1, inv_dx, stencil);
                adj.sxx[idx] += dpxx_dx;
                adj.szz[idx] += dpzz_dz;
                adj.sxz[idx] += dpxz_dz + dpzx_dx;
            }
        }
    });
}

template <int Order>
void accumulate_elastic2d_grad(
    const Elastic2DRawState& adj,
    const float* fvx,
    const float* fvz,
    const float* fvx_next,
    const float* fvz_next,
    const float* vp,
    const float* vs,
    const float* rho,
    float* grad_vp,
    float* grad_vs,
    float* grad_rho,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float dt
,
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
                const float fvx_x = sgrad_backward<Order>(fvx, idx, 1, inv_dx, stencil);
                const float fvz_z = sgrad_backward<Order>(fvz, idx, nx, inv_dz, stencil);
                const float fvx_z = sgrad_forward<Order>(fvx, idx, nx, inv_dz, stencil);
                const float fvz_x = sgrad_forward<Order>(fvz, idx, 1, inv_dx, stencil);
                const float grad_lambda = (adj.sxx[idx] + adj.szz[idx]) * (fvx_x + fvz_z);
                const float grad_mu = 2.0f * (adj.sxx[idx] * fvx_x + adj.szz[idx] * fvz_z) + adj.sxz[idx] * (fvx_z + fvz_x);

                grad_vp[idx] += -2.0f * rho[idx] * vp[idx] * grad_lambda * dt;
                grad_vs[idx] += -(-4.0f * rho[idx] * vs[idx] * grad_lambda + 2.0f * rho[idx] * vs[idx] * grad_mu) * dt;
                grad_rho[idx] += (adj.vx[idx] * (fvx[idx] - fvx_next[idx]) + adj.vz[idx] * (fvz[idx] - fvz_next[idx])) / rho[idx];
                grad_rho[idx] -= grad_lambda * (vp[idx] * vp[idx] - 2.0f * vs[idx] * vs[idx]) * dt + grad_mu * (vs[idx] * vs[idx]) * dt;
            }
        }
    });
}

template <int Order>
CpuForwardResult forward_elastic2d_raw_impl(const ForwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto vs_t = p.models[1];
    auto rho_t = p.models[2];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nsrc = p.sources_loc.size(1);
    const int64_t nrec = p.receivers_loc.size(1);
    const int64_t nsrc_fields = p.source_field_indices.numel();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t nt = static_cast<int64_t>(p.nt);
    TORCH_CHECK(vp_t.size(1) == 1, "Elastic2D raw CPU forward expects one model channel");
    auto record = torch::zeros({nrec_fields, B, nrec, nt}, vp_t.options());
    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = torch::zeros({nt, 2, B, nz, nx}, vp_t.options());

    std::vector<float> lambda(total, 0.0f), mu(total, 0.0f);
    build_lame(vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(), lambda, mu, total);
    Elastic2DRawState state(total);
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
    const std::vector<int> recursive_steps = checkpoint_steps_vector(p.checkpoint_steps);
    int next_recursive_checkpoint = 0;

    for (int64_t it = 0; it < nt; ++it) {
        elastic2d_velocity_step<Order>(state, rho_t.data_ptr<float>(), az, bz, azh, bzh, ax, bx, axh, bxh,
                                   B, nz, nx, inv_dz, inv_dx, dt, stencil);
        elastic2d_stress_step<Order>(state, lambda.data(), mu.data(), az, bz, azh, bzh, ax, bx, axh, bxh,
                                 B, nz, nx, inv_dz, inv_dx, dt, stencil);
        if (u_allt.defined()) {
            copy_vector_to_tensor(state.vx, u_allt.select(0, it).select(0, 0));
            copy_vector_to_tensor(state.vz, u_allt.select(0, it).select(0, 1));
        }
        auto fields = mutable_field_ptrs(state);
        for (int64_t f = 0; f < nsrc_fields; ++f) {
            const int field_id = source_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field(fields[field_id], source, sources, B, nsrc, nt, it, nz, nx);
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
            const int ckpt_idx = p.use_recursive_checkpoint
                ? recursive_checkpoint_index(static_cast<int>(it), recursive_steps, next_recursive_checkpoint)
                : checkpoint_index(static_cast<int>(it), static_cast<int>(nt), p.checkpoint_interval);
            save_checkpoint(p.checkpoints, ckpt_idx, state);
        }
    }

    torch::Tensor last_two = p.last_two.defined() ? p.last_two : torch::empty({0}, vp_t.options());
    if (p.use_boundary_saving) {
        if (!last_two.defined() || last_two.numel() == 0) last_two = torch::zeros({5, 1, B, 1, nz, nx}, vp_t.options());
        copy_vector_to_tensor(state.vx, last_two.select(0, 0).select(0, 0));
        copy_vector_to_tensor(state.vz, last_two.select(0, 1).select(0, 0));
        copy_vector_to_tensor(state.sxx, last_two.select(0, 2).select(0, 0));
        copy_vector_to_tensor(state.szz, last_two.select(0, 3).select(0, 0));
        copy_vector_to_tensor(state.sxz, last_two.select(0, 4).select(0, 0));
    }
    return {record, u_allt, last_two};
}

CpuForwardResult forward_elastic2d_raw(const ForwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, forward_elastic2d_raw_impl, p);
}

template <int Order>
BackwardOutput backward_elastic2d_full_impl(const BackwardInput& p, bool skip_initial_time = false)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto vs_t = p.models[1];
    auto rho_t = p.models[2];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t nt = static_cast<int64_t>(p.nt);
    TORCH_CHECK(p.u_forward.defined() && p.u_forward.dim() == 5 && p.u_forward.size(0) == nt && p.u_forward.size(1) == 2,
                "Elastic2D full CPU backward requires forward wavefields with shape (nt, 2, B, nz, nx)");
    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_vs = torch::zeros_like(vs_t);
    auto grad_rho = torch::zeros_like(rho_t);
    std::vector<float> lambda(total, 0.0f), mu(total, 0.0f);
    build_lame(vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(), lambda, mu, total);
    Elastic2DRawState adj(total);
    Elastic2DWorkspace work(total);
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

    const int64_t min_it = skip_initial_time ? 1 : 0;
    for (int64_t it = nt - 1; it >= min_it; --it) {
        auto fields = mutable_field_ptrs(adj);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field(fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources,
                                    B, adjoint_nsrc, nt, it, nz, nx);
            }
        }
        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vx_next = (it + 1 < nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero.data();
        const float* vz_next = (it + 1 < nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero.data();
        accumulate_elastic2d_grad<Order>(adj, vx_now, vz_now, vx_next, vz_next, vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(),
                                     grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_rho.data_ptr<float>(),
                                     B, nz, nx, inv_dz, inv_dx, dt, stencil);
        if (it == 0) continue;
        elastic2d_adjoint_step<Order>(adj, work, lambda.data(), mu.data(), rho_t.data_ptr<float>(),
                                  az, bz, azh, bzh, ax, bx, axh, bxh,
                                  B, nz, nx, inv_dz, inv_dx, dt, stencil);
    }
    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

template <int Order>
BackwardOutput backward_elastic2d_raw_impl(const BackwardInput& p)
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
    auto forward_result = forward_elastic2d_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_elastic2d_full_impl<Order>(replay);
}

template <int Order>
BackwardOutput backward_elastic2d_bs_impl(const BackwardInput& p)
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
    auto forward_result = forward_elastic2d_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_elastic2d_full_impl<Order>(replay, true);
}

BackwardOutput backward_elastic2d_raw(const BackwardInput& p)
{
    const bool has_full_wavefield = p.u_forward.defined() && p.u_forward.numel() > 0;
    const bool has_boundary_state = p.u_last_two.defined() && p.u_last_two.numel() > 0;

    #define SWEEP_CPU_ELASTIC2D_BACKWARD_CASE(ORDER) \
        if (has_full_wavefield) return backward_elastic2d_full_impl<ORDER>(p); \
        if (has_boundary_state) return backward_elastic2d_bs_impl<ORDER>(p); \
        return backward_elastic2d_raw_impl<ORDER>(p);
    SWEEP_CPU_DISPATCH_STENCIL_BODY(p.M, SWEEP_CPU_ELASTIC2D_BACKWARD_CASE);
    #undef SWEEP_CPU_ELASTIC2D_BACKWARD_CASE
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic2d::forward called with non-CPU tensors");
    if (!can_use_elastic2d_raw_forward(in)) {
        return engine::forward(in, EquationKind::Elastic2D);
    }
    auto result = forward_elastic2d_raw(in);
    ForwardOutput out;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic2d::backward called with non-CPU tensors");
    TORCH_CHECK(
        can_use_elastic2d_raw_backward(in),
        "Elastic2D CPU backward requires the handwritten raw float32 path; unsupported inputs will not fall back to torch autograd."
    );
    return backward_elastic2d_raw(in);
}

BackwardOutput backward_bs(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_ckpt(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_recursive_ckpt(const BackwardInput& in) { return backward(in); }

} // namespace sweep_cpu::elastic2d

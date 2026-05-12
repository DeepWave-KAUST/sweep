#include "das2d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sweep_cpu::das2d {
namespace {

using sweep_cpu::ops::StencilCoefficients;
struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

struct Das2DRawState {
    std::vector<float> exx, ezz, sxx, szz, txx, tzz;
    std::vector<float> m_sxx_xf, m_sxx_xb, m_szz_zf, m_szz_zb;
    std::vector<float> m_txx_zf, m_txx_zb, m_tzz_xf, m_tzz_xb;
    std::vector<float> das35, das54x, das54z;

    Das2DRawState() = default;

    explicit Das2DRawState(int64_t total)
        : exx(total, 0.0f), ezz(total, 0.0f), sxx(total, 0.0f), szz(total, 0.0f),
          txx(total, 0.0f), tzz(total, 0.0f),
          m_sxx_xf(total, 0.0f), m_sxx_xb(total, 0.0f),
          m_szz_zf(total, 0.0f), m_szz_zb(total, 0.0f),
          m_txx_zf(total, 0.0f), m_txx_zb(total, 0.0f),
          m_tzz_xf(total, 0.0f), m_tzz_xb(total, 0.0f),
          das35(total, 0.0f), das54x(total, 0.0f), das54z(total, 0.0f)
    {}
};

struct Das2DAdjointWork {
    std::vector<float> bar_dxx_sxx, bar_dzz_szz, bar_dzz_txx, bar_dxx_tzz;
    std::vector<float> bar_sxx_x, bar_szz_z, bar_txx_z, bar_tzz_x;

    explicit Das2DAdjointWork(int64_t total)
        : bar_dxx_sxx(total, 0.0f), bar_dzz_szz(total, 0.0f),
          bar_dzz_txx(total, 0.0f), bar_dxx_tzz(total, 0.0f),
          bar_sxx_x(total, 0.0f), bar_szz_z(total, 0.0f),
          bar_txx_z(total, 0.0f), bar_tzz_x(total, 0.0f)
    {}

    void zero_second()
    {
        std::fill(bar_dxx_sxx.begin(), bar_dxx_sxx.end(), 0.0f);
        std::fill(bar_dzz_szz.begin(), bar_dzz_szz.end(), 0.0f);
        std::fill(bar_dzz_txx.begin(), bar_dzz_txx.end(), 0.0f);
        std::fill(bar_dxx_tzz.begin(), bar_dxx_tzz.end(), 0.0f);
    }

    void zero_first()
    {
        std::fill(bar_sxx_x.begin(), bar_sxx_x.end(), 0.0f);
        std::fill(bar_szz_z.begin(), bar_szz_z.end(), 0.0f);
        std::fill(bar_txx_z.begin(), bar_txx_z.end(), 0.0f);
        std::fill(bar_tzz_x.begin(), bar_tzz_x.end(), 0.0f);
    }
};

using sweep_cpu::ops::scatter_sgradient_adjoint;
using sweep_cpu::ops::sgrad_backward;
using sweep_cpu::ops::sgrad_forward;

bool can_use_das2d_raw_forward(const ForwardInput& p)
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

bool can_use_das2d_raw_backward(const BackwardInput& p)
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
    return true;
}

void copy_vector_to_tensor(const std::vector<float>& src, torch::Tensor tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(src.size()) == tensor.numel(), "Tensor/vector size mismatch");
    std::copy(src.begin(), src.end(), tensor.data_ptr<float>());
}

std::vector<float*> mutable_field_ptrs(Das2DRawState& s)
{
    return {
        s.exx.data(), s.ezz.data(), s.sxx.data(), s.szz.data(), s.txx.data(), s.tzz.data(),
        s.m_sxx_xf.data(), s.m_sxx_xb.data(), s.m_szz_zf.data(), s.m_szz_zb.data(),
        s.m_txx_zf.data(), s.m_txx_zb.data(), s.m_tzz_xf.data(), s.m_tzz_xb.data(),
        s.das35.data(), s.das54x.data(), s.das54z.data()
    };
}

std::vector<const float*> const_field_ptrs(const Das2DRawState& s)
{
    return {
        s.exx.data(), s.ezz.data(), s.sxx.data(), s.szz.data(), s.txx.data(), s.tzz.data(),
        s.m_sxx_xf.data(), s.m_sxx_xb.data(), s.m_szz_zf.data(), s.m_szz_zb.data(),
        s.m_txx_zf.data(), s.m_txx_zb.data(), s.m_tzz_xf.data(), s.m_tzz_xb.data(),
        s.das35.data(), s.das54x.data(), s.das54z.data()
    };
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
void das2d_first_derivatives(
    Das2DRawState& s,
    std::vector<float>& tmp_sxx_x,
    std::vector<float>& tmp_szz_z,
    std::vector<float>& tmp_txx_z,
    std::vector<float>& tmp_tzz_x,
    const float* azh,
    const float* bzh,
    const float* axh,
    const float* bxh,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx
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
                s.m_sxx_xf[idx] = axh[x] * s.m_sxx_xf[idx] + bxh[x] * dsxx_dx;
                tmp_sxx_x[idx] = dsxx_dx + s.m_sxx_xf[idx];

                float dszz_dz = sgrad_forward<Order>(s.szz.data(), idx, nx, inv_dz, stencil);
                s.m_szz_zf[idx] = azh[z] * s.m_szz_zf[idx] + bzh[z] * dszz_dz;
                tmp_szz_z[idx] = dszz_dz + s.m_szz_zf[idx];

                float dtxx_dz = sgrad_forward<Order>(s.txx.data(), idx, nx, inv_dz, stencil);
                s.m_txx_zf[idx] = azh[z] * s.m_txx_zf[idx] + bzh[z] * dtxx_dz;
                tmp_txx_z[idx] = dtxx_dz + s.m_txx_zf[idx];

                float dtzz_dx = sgrad_forward<Order>(s.tzz.data(), idx, 1, inv_dx, stencil);
                s.m_tzz_xf[idx] = axh[x] * s.m_tzz_xf[idx] + bxh[x] * dtzz_dx;
                tmp_tzz_x[idx] = dtzz_dx + s.m_tzz_xf[idx];
            }
        }
    });
}

template <int Order>
void das2d_update(
    Das2DRawState& s,
    const std::vector<float>& tmp_sxx_x,
    const std::vector<float>& tmp_szz_z,
    const std::vector<float>& tmp_txx_z,
    const std::vector<float>& tmp_tzz_x,
    const float* rho,
    const float* lambda,
    const float* mu,
    const float* az,
    const float* bz,
    const float* ax,
    const float* bx,
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
    const int update_halo = 2 * M;
    const int64_t row_count = B * (nz - 2 * update_halo);
    if (row_count <= 0) return;
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * update_halo);
            const int64_t z = update_halo + row - b * (nz - 2 * update_halo);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = update_halo; x < nx - update_halo; ++x) {
                const int64_t idx = base + x;
                float dxx_sxx = sgrad_backward<Order>(tmp_sxx_x.data(), idx, 1, inv_dx, stencil);
                s.m_sxx_xb[idx] = ax[x] * s.m_sxx_xb[idx] + bx[x] * dxx_sxx;
                dxx_sxx += s.m_sxx_xb[idx];

                float dzz_szz = sgrad_backward<Order>(tmp_szz_z.data(), idx, nx, inv_dz, stencil);
                s.m_szz_zb[idx] = az[z] * s.m_szz_zb[idx] + bz[z] * dzz_szz;
                dzz_szz += s.m_szz_zb[idx];

                float dzz_txx = sgrad_backward<Order>(tmp_txx_z.data(), idx, nx, inv_dz, stencil);
                s.m_txx_zb[idx] = az[z] * s.m_txx_zb[idx] + bz[z] * dzz_txx;
                dzz_txx += s.m_txx_zb[idx];

                float dxx_tzz = sgrad_backward<Order>(tmp_tzz_x.data(), idx, 1, inv_dx, stencil);
                s.m_tzz_xb[idx] = ax[x] * s.m_tzz_xb[idx] + bx[x] * dxx_tzz;
                dxx_tzz += s.m_tzz_xb[idx];

                const float inv_rho = 1.0f / rho[idx];
                const float shear_xz = dzz_txx + dxx_tzz;
                const float exx_new = s.exx[idx] + dt * inv_rho * (dxx_sxx + shear_xz);
                const float ezz_new = s.ezz[idx] + dt * inv_rho * (dzz_szz + shear_xz);
                s.exx[idx] = exx_new;
                s.ezz[idx] = ezz_new;

                const float lam = lambda[idx];
                const float mu0 = mu[idx];
                s.sxx[idx] += dt * ((lam + 2.0f * mu0) * exx_new + lam * ezz_new);
                s.szz[idx] += dt * ((lam + 2.0f * mu0) * ezz_new + lam * exx_new);
                s.txx[idx] += dt * mu0 * exx_new;
                s.tzz[idx] += dt * mu0 * ezz_new;
                s.das35[idx] = exx_new + ezz_new;
                s.das54x[idx] = 4.0f * exx_new + ezz_new;
                s.das54z[idx] = exx_new + 4.0f * ezz_new;
            }
        }
    });
}

void add_source_to_field(float* field, const float* source, const int32_t* loc, int64_t B, int64_t nsrc, int64_t nt, int64_t it, int64_t nz, int64_t nx)
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

void record_field(const float* field, float* record, const int32_t* loc, int64_t B, int64_t nrec, int64_t nt, int64_t it, int64_t nz, int64_t nx)
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
void das2d_second_adjoint(
    const std::vector<float>& bar_out,
    std::vector<float>& bar_tmp,
    std::vector<float>& adj_memory,
    bool x_direction,
    const float* az,
    const float* bz,
    const float* ax,
    const float* bx,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    const int update_halo = 2 * M;
    for (int64_t b = 0; b < B; ++b) {
        const int64_t bbase = b * spatial;
        for (int64_t z = update_halo; z < nz - update_halo; ++z) {
            const int64_t base = bbase + z * nx;
            for (int64_t x = update_halo; x < nx - update_halo; ++x) {
                const int64_t idx = base + x;
                const float q = bar_out[idx];
                const float acoef = x_direction ? ax[x] : az[z];
                const float bcoef = x_direction ? bx[x] : bz[z];
                const float total_memory_bar = adj_memory[idx] + q;
                const float derivative_bar = q + bcoef * total_memory_bar;
                adj_memory[idx] = acoef * total_memory_bar;
                if (x_direction) {
                    scatter_sgradient_adjoint<Order, false>(derivative_bar, bar_tmp.data(), idx, 1, inv_dx, stencil);
                } else {
                    scatter_sgradient_adjoint<Order, false>(derivative_bar, bar_tmp.data(), idx, nx, inv_dz, stencil);
                }
            }
        }
    }
}

template <int Order>
void das2d_first_adjoint(
    const std::vector<float>& bar_tmp,
    std::vector<float>& adj_memory,
    std::vector<float>& adj_field,
    bool x_direction,
    const float* azh,
    const float* bzh,
    const float* axh,
    const float* bxh,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        const int64_t bbase = b * spatial;
        for (int64_t z = M; z < nz - M; ++z) {
            const int64_t base = bbase + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float q = bar_tmp[idx];
                const float acoef = x_direction ? axh[x] : azh[z];
                const float bcoef = x_direction ? bxh[x] : bzh[z];
                const float total_memory_bar = adj_memory[idx] + q;
                const float derivative_bar = q + bcoef * total_memory_bar;
                adj_memory[idx] = acoef * total_memory_bar;
                if (x_direction) {
                    scatter_sgradient_adjoint<Order, true>(derivative_bar, adj_field.data(), idx, 1, inv_dx, stencil);
                } else {
                    scatter_sgradient_adjoint<Order, true>(derivative_bar, adj_field.data(), idx, nx, inv_dz, stencil);
                }
            }
        }
    }
}

template <int Order>
void das2d_project_model_grad(
    Das2DRawState& adj,
    const float* exx_now,
    const float* ezz_now,
    const float* exx_prev,
    const float* ezz_prev,
    const float* vp,
    const float* vs,
    const float* rho,
    float* grad_vp,
    float* grad_vs,
    float* grad_rho,
    Das2DAdjointWork& work,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float dt,
    int M
)
{
    const int64_t spatial = nz * nx;
    const int update_halo = 2 * M;
    const int64_t total = B * spatial;
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t z = (idx / nx) % nz;
            const bool active = x >= update_halo && x < nx - update_halo &&
                                z >= update_halo && z < nz - update_halo;
            if (!active) {
                adj.das35[idx] = 0.0f;
                adj.das54x[idx] = 0.0f;
                adj.das54z[idx] = 0.0f;
                continue;
            }

            float bar_exx = adj.exx[idx] + adj.das35[idx] + 4.0f * adj.das54x[idx] + adj.das54z[idx];
            float bar_ezz = adj.ezz[idx] + adj.das35[idx] + adj.das54x[idx] + 4.0f * adj.das54z[idx];
            adj.das35[idx] = 0.0f;
            adj.das54x[idx] = 0.0f;
            adj.das54z[idx] = 0.0f;

            const float exx = exx_now[idx];
            const float ezz = ezz_now[idx];
            const float vp0 = vp[idx];
            const float vs0 = vs[idx];
            const float rho0 = rho[idx];
            const float lambda = rho0 * (vp0 * vp0 - 2.0f * vs0 * vs0);
            const float mu = rho0 * vs0 * vs0;
            const float bar_sxx = adj.sxx[idx];
            const float bar_szz = adj.szz[idx];
            const float bar_txx = adj.txx[idx];
            const float bar_tzz = adj.tzz[idx];
            const float grad_lambda = dt * (bar_sxx + bar_szz) * (exx + ezz);
            const float grad_mu = dt * (2.0f * bar_sxx * exx + 2.0f * bar_szz * ezz + bar_txx * exx + bar_tzz * ezz);

            bar_exx += dt * ((lambda + 2.0f * mu) * bar_sxx + lambda * bar_szz + mu * bar_txx);
            bar_ezz += dt * (lambda * bar_sxx + (lambda + 2.0f * mu) * bar_szz + mu * bar_tzz);

            const float dexx = exx - exx_prev[idx];
            const float dezz = ezz - ezz_prev[idx];
            grad_vp[idx] += 2.0f * rho0 * vp0 * grad_lambda;
            grad_vs[idx] += -4.0f * rho0 * vs0 * grad_lambda + 2.0f * rho0 * vs0 * grad_mu;
            grad_rho[idx] += (vp0 * vp0 - 2.0f * vs0 * vs0) * grad_lambda +
                             (vs0 * vs0) * grad_mu -
                             (bar_exx * dexx + bar_ezz * dezz) / rho0;

            adj.exx[idx] = bar_exx;
            adj.ezz[idx] = bar_ezz;
            const float common = dt / rho0 * (bar_exx + bar_ezz);
            work.bar_dxx_sxx[idx] = dt / rho0 * bar_exx;
            work.bar_dzz_szz[idx] = dt / rho0 * bar_ezz;
            work.bar_dzz_txx[idx] = common;
            work.bar_dxx_tzz[idx] = common;
        }
    });
}

template <int Order>
void das2d_adjoint_step(
    Das2DRawState& adj,
    Das2DAdjointWork& work,
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
    float inv_dx
,
    const StencilCoefficients& stencil)
{
    das2d_second_adjoint<Order>(work.bar_dxx_sxx, work.bar_sxx_x, adj.m_sxx_xb, true, az, bz, ax, bx, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_second_adjoint<Order>(work.bar_dzz_szz, work.bar_szz_z, adj.m_szz_zb, false, az, bz, ax, bx, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_second_adjoint<Order>(work.bar_dzz_txx, work.bar_txx_z, adj.m_txx_zb, false, az, bz, ax, bx, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_second_adjoint<Order>(work.bar_dxx_tzz, work.bar_tzz_x, adj.m_tzz_xb, true, az, bz, ax, bx, B, nz, nx, inv_dz, inv_dx, stencil);

    das2d_first_adjoint<Order>(work.bar_sxx_x, adj.m_sxx_xf, adj.sxx, true, azh, bzh, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_first_adjoint<Order>(work.bar_szz_z, adj.m_szz_zf, adj.szz, false, azh, bzh, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_first_adjoint<Order>(work.bar_txx_z, adj.m_txx_zf, adj.txx, false, azh, bzh, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
    das2d_first_adjoint<Order>(work.bar_tzz_x, adj.m_tzz_xf, adj.tzz, true, azh, bzh, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
}

template <int Order>
CpuForwardResult forward_das2d_raw_impl(const ForwardInput& p)
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
    TORCH_CHECK(vp_t.size(1) == 1, "DAS2D raw CPU forward expects one model channel");

    auto record = torch::zeros({nrec_fields, B, nrec, nt}, vp_t.options());
    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = torch::zeros({nt, 2, B, nz, nx}, vp_t.options());

    std::vector<float> lambda(total, 0.0f), mu(total, 0.0f);
    build_lame(vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(), lambda, mu, total);
    Das2DRawState state(total);
    std::vector<float> tmp_sxx_x(total, 0.0f), tmp_szz_z(total, 0.0f), tmp_txx_z(total, 0.0f), tmp_tzz_x(total, 0.0f);
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ax = pml[4].data_ptr<float>();
    const float* bx = pml[5].data_ptr<float>();
    const float* axh = pml[6].data_ptr<float>();
    const float* bxh = pml[7].data_ptr<float>();
    const float inv_dx = static_cast<float>(1.0 / p.spacing[0]);
    const float inv_dz = static_cast<float>(1.0 / p.spacing[1]);
    const float dt = static_cast<float>(p.dt);
    const auto source_fields_cpu = p.source_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const auto receiver_fields_cpu = p.receiver_field_indices.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* source_fields = source_fields_cpu.data_ptr<int32_t>();
    const int32_t* receiver_fields = receiver_fields_cpu.data_ptr<int32_t>();
    const int32_t* sources = p.sources_loc.data_ptr<int32_t>();
    const int32_t* receivers = p.receivers_loc.data_ptr<int32_t>();
    const float* source = p.source.data_ptr<float>();
    float* rec = record.data_ptr<float>();

    for (int64_t it = 0; it < nt; ++it) {
        std::fill(tmp_sxx_x.begin(), tmp_sxx_x.end(), 0.0f);
        std::fill(tmp_szz_z.begin(), tmp_szz_z.end(), 0.0f);
        std::fill(tmp_txx_z.begin(), tmp_txx_z.end(), 0.0f);
        std::fill(tmp_tzz_x.begin(), tmp_tzz_x.end(), 0.0f);
        das2d_first_derivatives<Order>(state, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x, azh, bzh, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
        das2d_update<Order>(state, tmp_sxx_x, tmp_szz_z, tmp_txx_z, tmp_tzz_x,
                        rho_t.data_ptr<float>(), lambda.data(), mu.data(), az, bz, ax, bx,
                        B, nz, nx, inv_dz, inv_dx, dt, stencil);
        auto fields = mutable_field_ptrs(state);
        for (int64_t f = 0; f < nsrc_fields; ++f) {
            const int field_id = source_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field(fields[field_id], source, sources, B, nsrc, nt, it, nz, nx);
            }
        }
        if (u_allt.defined()) {
            copy_vector_to_tensor(state.exx, u_allt.select(0, it).select(0, 0));
            copy_vector_to_tensor(state.ezz, u_allt.select(0, it).select(0, 1));
        }
        auto cfields = const_field_ptrs(state);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(cfields.size())) {
                record_field(cfields[field_id], rec + f * B * nrec * nt, receivers, B, nrec, nt, it, nz, nx);
            }
        }
    }

    torch::Tensor last_two = p.last_two.defined() ? p.last_two : torch::empty({0}, vp_t.options());
    if (p.use_boundary_saving) {
        if (!last_two.defined() || last_two.numel() == 0) last_two = torch::zeros({9, 1, B, 1, nz, nx}, vp_t.options());
        copy_vector_to_tensor(state.exx, last_two.select(0, 0).select(0, 0));
        copy_vector_to_tensor(state.ezz, last_two.select(0, 1).select(0, 0));
        copy_vector_to_tensor(state.sxx, last_two.select(0, 2).select(0, 0));
        copy_vector_to_tensor(state.szz, last_two.select(0, 3).select(0, 0));
        copy_vector_to_tensor(state.txx, last_two.select(0, 4).select(0, 0));
        copy_vector_to_tensor(state.tzz, last_two.select(0, 5).select(0, 0));
        copy_vector_to_tensor(state.das35, last_two.select(0, 6).select(0, 0));
        copy_vector_to_tensor(state.das54x, last_two.select(0, 7).select(0, 0));
        copy_vector_to_tensor(state.das54z, last_two.select(0, 8).select(0, 0));
    }
    return {record, u_allt, last_two};
}

CpuForwardResult forward_das2d_raw(const ForwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, forward_das2d_raw_impl, p);
}

template <int Order>
BackwardOutput backward_das2d_full_impl(const BackwardInput& p)
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
                "DAS2D full CPU backward requires saved exx/ezz wavefields with shape (nt, 2, B, nz, nx)");

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_vs = torch::zeros_like(vs_t);
    auto grad_rho = torch::zeros_like(rho_t);
    Das2DRawState adj(total);
    Das2DAdjointWork work(total);
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
    const float inv_dx = static_cast<float>(1.0 / p.spacing[0]);
    const float inv_dz = static_cast<float>(1.0 / p.spacing[1]);
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
                add_source_to_field(fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources,
                                    B, adjoint_nsrc, nt, it, nz, nx);
            }
        }

        work.zero_second();
        work.zero_first();
        const float* exx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* ezz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* exx_prev = (it > 0) ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>() : zero.data();
        const float* ezz_prev = (it > 0) ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>() : zero.data();
        das2d_project_model_grad<Order>(adj, exx_now, ezz_now, exx_prev, ezz_prev,
                                    vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(),
                                    grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_rho.data_ptr<float>(),
                                    work, B, nz, nx, dt, M);
        if (it == 0) continue;
        das2d_adjoint_step<Order>(adj, work, az, bz, azh, bzh, ax, bx, axh, bxh, B, nz, nx, inv_dz, inv_dx, stencil);
    }
    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

template <int Order>
BackwardOutput backward_das2d_recompute_impl(const BackwardInput& p)
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
    auto forward_result = forward_das2d_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_das2d_full_impl<Order>(replay);
}

BackwardOutput backward_das2d_raw(const BackwardInput& p)
{
    const bool has_full_wavefield = p.u_forward.defined() && p.u_forward.numel() > 0;

    #define SWEEP_CPU_DAS2D_BACKWARD_CASE(ORDER) \
        if (has_full_wavefield) return backward_das2d_full_impl<ORDER>(p); \
        return backward_das2d_recompute_impl<ORDER>(p);
    SWEEP_CPU_DISPATCH_STENCIL_BODY(p.M, SWEEP_CPU_DAS2D_BACKWARD_CASE);
    #undef SWEEP_CPU_DAS2D_BACKWARD_CASE
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::das2d::forward called with non-CPU tensors");
    if (!can_use_das2d_raw_forward(in)) {
        return engine::forward(in, EquationKind::DAS2D);
    }
    auto result = forward_das2d_raw(in);
    ForwardOutput out;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::das2d::backward called with non-CPU tensors");
    TORCH_CHECK(
        can_use_das2d_raw_backward(in),
        "DAS2D CPU backward requires the handwritten raw float32 path; unsupported inputs will not fall back to torch autograd."
    );
    return backward_das2d_raw(in);
}

BackwardOutput backward_bs(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_ckpt(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_recursive_ckpt(const BackwardInput& in) { return backward(in); }

} // namespace sweep_cpu::das2d

#include "elastic3d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sweep_cpu::elastic3d {
namespace {

using sweep_cpu::ops::StencilCoefficients;
struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

struct Elastic3DRawState {
    std::vector<float> vx, vy, vz;
    std::vector<float> sxx, syy, szz, sxy, sxz, syz;
    std::vector<float> m_vxx, m_vxy, m_vxz, m_vyx, m_vyy, m_vyz, m_vzx, m_vzy, m_vzz;
    std::vector<float> m_sxxx, m_sxxy, m_sxxz, m_syyx, m_syyy, m_syyz, m_szzx, m_szzy, m_szzz;
    std::vector<float> m_sxyx, m_sxyy, m_sxyz, m_sxzx, m_sxzy, m_sxzz, m_syzx, m_syzy, m_syzz;

    Elastic3DRawState() = default;

    explicit Elastic3DRawState(int64_t total)
        : vx(total, 0.0f), vy(total, 0.0f), vz(total, 0.0f),
          sxx(total, 0.0f), syy(total, 0.0f), szz(total, 0.0f),
          sxy(total, 0.0f), sxz(total, 0.0f), syz(total, 0.0f),
          m_vxx(total, 0.0f), m_vxy(total, 0.0f), m_vxz(total, 0.0f),
          m_vyx(total, 0.0f), m_vyy(total, 0.0f), m_vyz(total, 0.0f),
          m_vzx(total, 0.0f), m_vzy(total, 0.0f), m_vzz(total, 0.0f),
          m_sxxx(total, 0.0f), m_sxxy(total, 0.0f), m_sxxz(total, 0.0f),
          m_syyx(total, 0.0f), m_syyy(total, 0.0f), m_syyz(total, 0.0f),
          m_szzx(total, 0.0f), m_szzy(total, 0.0f), m_szzz(total, 0.0f),
          m_sxyx(total, 0.0f), m_sxyy(total, 0.0f), m_sxyz(total, 0.0f),
          m_sxzx(total, 0.0f), m_sxzy(total, 0.0f), m_sxzz(total, 0.0f),
          m_syzx(total, 0.0f), m_syzy(total, 0.0f), m_syzz(total, 0.0f)
    {}
};

struct Elastic3DWorkspace {
    std::vector<float> qxx, qxy, qxz, qyx, qyy, qyz, qzx, qzy, qzz;
    std::vector<float> pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz;

    explicit Elastic3DWorkspace(int64_t total)
        : qxx(total, 0.0f), qxy(total, 0.0f), qxz(total, 0.0f),
          qyx(total, 0.0f), qyy(total, 0.0f), qyz(total, 0.0f),
          qzx(total, 0.0f), qzy(total, 0.0f), qzz(total, 0.0f),
          pxx(total, 0.0f), pxy(total, 0.0f), pxz(total, 0.0f),
          pyx(total, 0.0f), pyy(total, 0.0f), pyz(total, 0.0f),
          pzx(total, 0.0f), pzy(total, 0.0f), pzz(total, 0.0f)
    {}
};

using sweep_cpu::ops::sgrad_backward;
using sweep_cpu::ops::sgrad_forward;

inline bool is_elastic3d_interior(int64_t x, int64_t y, int64_t z, int64_t nx, int64_t ny, int64_t nz, int M, int abcn)
{
    const int64_t x0 = abcn + M;
    const int64_t x1 = nx - abcn - M;
    const int64_t y0 = abcn + M;
    const int64_t y1 = ny - abcn - M;
    const int64_t z0 = abcn + M;
    const int64_t z1 = nz - abcn - M;
    return x >= x0 + 1 && x < x1 - 1 &&
           y >= y0 + 1 && y < y1 - 1 &&
           z >= z0 + 1 && z < z1 - 1;
}

bool can_use_elastic3d_raw_forward(const ForwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 3 || p.pml_vals.size() < 12) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, false)) return false;
    for (const auto& model : p.models) {
        if (!model.is_contiguous() || model.scalar_type() != torch::kFloat32 || model.dim() != 5) return false;
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
    const int64_t ny = p.models[0].size(3);
    const int64_t nx = p.models[0].size(4);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz || p.pml_vals[3].numel() != nz) return false;
    if (p.pml_vals[4].numel() != ny || p.pml_vals[5].numel() != ny || p.pml_vals[6].numel() != ny || p.pml_vals[7].numel() != ny) return false;
    if (p.pml_vals[8].numel() != nx || p.pml_vals[9].numel() != nx || p.pml_vals[10].numel() != nx || p.pml_vals[11].numel() != nx) return false;
    return true;
}

bool can_use_elastic3d_raw_backward(const BackwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 3 || p.pml_vals.size() < 12) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, false)) return false;
    for (const auto& model : p.models) {
        if (!model.is_contiguous() || model.scalar_type() != torch::kFloat32 || model.dim() != 5) return false;
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
    const int64_t ny = p.models[0].size(3);
    const int64_t nx = p.models[0].size(4);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz || p.pml_vals[3].numel() != nz) return false;
    if (p.pml_vals[4].numel() != ny || p.pml_vals[5].numel() != ny || p.pml_vals[6].numel() != ny || p.pml_vals[7].numel() != ny) return false;
    if (p.pml_vals[8].numel() != nx || p.pml_vals[9].numel() != nx || p.pml_vals[10].numel() != nx || p.pml_vals[11].numel() != nx) return false;
    return true;
}

void copy_vector_to_tensor(const std::vector<float>& src, torch::Tensor tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(src.size()) == tensor.numel(), "Tensor/vector size mismatch");
    std::copy(src.begin(), src.end(), tensor.data_ptr<float>());
}

std::vector<float*> mutable_field_ptrs(Elastic3DRawState& s)
{
    return {s.vx.data(), s.vy.data(), s.vz.data(), s.sxx.data(), s.syy.data(), s.szz.data(), s.sxy.data(), s.sxz.data(), s.syz.data()};
}

std::vector<const float*> const_field_ptrs(const Elastic3DRawState& s)
{
    return {s.vx.data(), s.vy.data(), s.vz.data(), s.sxx.data(), s.syy.data(), s.szz.data(), s.sxy.data(), s.sxz.data(), s.syz.data()};
}

std::vector<const std::vector<float>*> checkpoint_vectors(const Elastic3DRawState& s)
{
    return {
        &s.vx, &s.vy, &s.vz, &s.sxx, &s.syy, &s.szz, &s.sxy, &s.sxz, &s.syz,
        &s.m_vxx, &s.m_vxy, &s.m_vxz, &s.m_vyx, &s.m_vyy, &s.m_vyz, &s.m_vzx, &s.m_vzy, &s.m_vzz,
        &s.m_sxxx, &s.m_sxxy, &s.m_sxxz, &s.m_syyx, &s.m_syyy, &s.m_syyz, &s.m_szzx, &s.m_szzy, &s.m_szzz,
        &s.m_sxyx, &s.m_sxyy, &s.m_sxyz, &s.m_sxzx, &s.m_sxzy, &s.m_sxzz, &s.m_syzx, &s.m_syzy, &s.m_syzz
    };
}

void save_checkpoint(const std::vector<torch::Tensor>& checkpoints, int checkpoint_idx, const Elastic3DRawState& s)
{
    if (checkpoint_idx < 0 || checkpoints.empty()) return;
    TORCH_CHECK(checkpoints.size() == 36, "Elastic3D CPU checkpointing expects 36 checkpoint tensors");
    const auto vectors = checkpoint_vectors(s);
    for (int i = 0; i < 36; ++i) {
        copy_vector_to_tensor(*vectors[i], checkpoints[i].select(0, checkpoint_idx));
    }
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
void elastic3d_velocity_step(
    Elastic3DRawState& s, const float* rho,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ay, const float* by, const float* ayh, const float* byh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t ny, int64_t nx, int abcn,
    float inv_dz, float inv_dy, float inv_dx, float dt
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * ny * nx;
    const int64_t yz_count = B * (nz - 2 * M) * (ny - 2 * M);
    at::parallel_for(0, yz_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / ((nz - 2 * M) * (ny - 2 * M));
            const int64_t rem = row - b * ((nz - 2 * M) * (ny - 2 * M));
            const int64_t z = M + rem / (ny - 2 * M);
            const int64_t y = M + rem % (ny - 2 * M);
            const int64_t base = b * spatial + z * ny * nx + y * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                float dsxx_dx = sgrad_forward<Order>(s.sxx.data(), idx, 1, inv_dx, stencil);
                float dsxy_dy = sgrad_backward<Order>(s.sxy.data(), idx, nx, inv_dy, stencil);
                float dsxz_dz = sgrad_backward<Order>(s.sxz.data(), idx, ny * nx, inv_dz, stencil);
                float dsxy_dx = sgrad_backward<Order>(s.sxy.data(), idx, 1, inv_dx, stencil);
                float dsyy_dy = sgrad_forward<Order>(s.syy.data(), idx, nx, inv_dy, stencil);
                float dsyz_dz = sgrad_backward<Order>(s.syz.data(), idx, ny * nx, inv_dz, stencil);
                float dsxz_dx = sgrad_backward<Order>(s.sxz.data(), idx, 1, inv_dx, stencil);
                float dsyz_dy = sgrad_backward<Order>(s.syz.data(), idx, nx, inv_dy, stencil);
                float dszz_dz = sgrad_forward<Order>(s.szz.data(), idx, ny * nx, inv_dz, stencil);

                if (!is_elastic3d_interior(x, y, z, nx, ny, nz, M, abcn)) {
                    s.m_szzz[idx] = azh[z] * s.m_szzz[idx] + bzh[z] * dszz_dz;
                    dszz_dz += s.m_szzz[idx];
                    s.m_sxzx[idx] = ax[x] * s.m_sxzx[idx] + bx[x] * dsxz_dx;
                    dsxz_dx += s.m_sxzx[idx];
                    s.m_sxzz[idx] = az[z] * s.m_sxzz[idx] + bz[z] * dsxz_dz;
                    dsxz_dz += s.m_sxzz[idx];
                    s.m_sxxx[idx] = axh[x] * s.m_sxxx[idx] + bxh[x] * dsxx_dx;
                    dsxx_dx += s.m_sxxx[idx];
                    s.m_sxyy[idx] = ay[y] * s.m_sxyy[idx] + by[y] * dsxy_dy;
                    dsxy_dy += s.m_sxyy[idx];
                    s.m_sxyx[idx] = ax[x] * s.m_sxyx[idx] + bx[x] * dsxy_dx;
                    dsxy_dx += s.m_sxyx[idx];
                    s.m_syyy[idx] = ayh[y] * s.m_syyy[idx] + byh[y] * dsyy_dy;
                    dsyy_dy += s.m_syyy[idx];
                    s.m_syzz[idx] = az[z] * s.m_syzz[idx] + bz[z] * dsyz_dz;
                    dsyz_dz += s.m_syzz[idx];
                    s.m_syzy[idx] = ay[y] * s.m_syzy[idx] + by[y] * dsyz_dy;
                    dsyz_dy += s.m_syzy[idx];
                }

                const float inv_rho = 1.0f / rho[idx];
                s.vx[idx] += dt * inv_rho * (dsxx_dx + dsxy_dy + dsxz_dz);
                s.vy[idx] += dt * inv_rho * (dsxy_dx + dsyy_dy + dsyz_dz);
                s.vz[idx] += dt * inv_rho * (dsxz_dx + dsyz_dy + dszz_dz);
            }
        }
    });
}

template <int Order>
void elastic3d_stress_step(
    Elastic3DRawState& s, const float* lambda, const float* mu,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ay, const float* by, const float* ayh, const float* byh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t ny, int64_t nx, int abcn,
    float inv_dz, float inv_dy, float inv_dx, float dt
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * ny * nx;
    const int64_t yz_count = B * (nz - 2 * M) * (ny - 2 * M);
    at::parallel_for(0, yz_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / ((nz - 2 * M) * (ny - 2 * M));
            const int64_t rem = row - b * ((nz - 2 * M) * (ny - 2 * M));
            const int64_t z = M + rem / (ny - 2 * M);
            const int64_t y = M + rem % (ny - 2 * M);
            const int64_t base = b * spatial + z * ny * nx + y * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                float dvx_dx = sgrad_backward<Order>(s.vx.data(), idx, 1, inv_dx, stencil);
                float dvx_dy = sgrad_forward<Order>(s.vx.data(), idx, nx, inv_dy, stencil);
                float dvx_dz = sgrad_forward<Order>(s.vx.data(), idx, ny * nx, inv_dz, stencil);
                float dvy_dx = sgrad_forward<Order>(s.vy.data(), idx, 1, inv_dx, stencil);
                float dvy_dy = sgrad_backward<Order>(s.vy.data(), idx, nx, inv_dy, stencil);
                float dvy_dz = sgrad_forward<Order>(s.vy.data(), idx, ny * nx, inv_dz, stencil);
                float dvz_dx = sgrad_forward<Order>(s.vz.data(), idx, 1, inv_dx, stencil);
                float dvz_dy = sgrad_forward<Order>(s.vz.data(), idx, nx, inv_dy, stencil);
                float dvz_dz = sgrad_backward<Order>(s.vz.data(), idx, ny * nx, inv_dz, stencil);

                if (!is_elastic3d_interior(x, y, z, nx, ny, nz, M, abcn)) {
                    s.m_vzz[idx] = az[z] * s.m_vzz[idx] + bz[z] * dvz_dz;
                    dvz_dz += s.m_vzz[idx];
                    s.m_vyy[idx] = ay[y] * s.m_vyy[idx] + by[y] * dvy_dy;
                    dvy_dy += s.m_vyy[idx];
                    s.m_vxx[idx] = ax[x] * s.m_vxx[idx] + bx[x] * dvx_dx;
                    dvx_dx += s.m_vxx[idx];
                    s.m_vxz[idx] = azh[z] * s.m_vxz[idx] + bzh[z] * dvx_dz;
                    dvx_dz += s.m_vxz[idx];
                    s.m_vzx[idx] = axh[x] * s.m_vzx[idx] + bxh[x] * dvz_dx;
                    dvz_dx += s.m_vzx[idx];
                    s.m_vxy[idx] = ayh[y] * s.m_vxy[idx] + byh[y] * dvx_dy;
                    dvx_dy += s.m_vxy[idx];
                    s.m_vyx[idx] = axh[x] * s.m_vyx[idx] + bxh[x] * dvy_dx;
                    dvy_dx += s.m_vyx[idx];
                    s.m_vyz[idx] = azh[z] * s.m_vyz[idx] + bzh[z] * dvy_dz;
                    dvy_dz += s.m_vyz[idx];
                    s.m_vzy[idx] = ayh[y] * s.m_vzy[idx] + byh[y] * dvz_dy;
                    dvz_dy += s.m_vzy[idx];
                }

                const float lam = lambda[idx];
                const float mu0 = mu[idx];
                const float div_v = dvx_dx + dvy_dy + dvz_dz;
                s.sxx[idx] += dt * (lam * div_v + 2.0f * mu0 * dvx_dx);
                s.syy[idx] += dt * (lam * div_v + 2.0f * mu0 * dvy_dy);
                s.szz[idx] += dt * (lam * div_v + 2.0f * mu0 * dvz_dz);
                s.sxy[idx] += dt * mu0 * (dvx_dy + dvy_dx);
                s.sxz[idx] += dt * mu0 * (dvx_dz + dvz_dx);
                s.syz[idx] += dt * mu0 * (dvy_dz + dvz_dy);
            }
        }
    });
}

void add_source_to_field_3d(float* field, const float* source, const int32_t* loc, int64_t B, int64_t nsrc, int64_t nt, int64_t it, int64_t nz, int64_t ny, int64_t nx)
{
    const int64_t spatial = nz * ny * nx;
    for (int64_t b = 0; b < B; ++b) {
        float* base = field + b * spatial;
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t off = (b * nsrc + isrc) * 3;
            const int64_t x = loc[off];
            const int64_t y = loc[off + 1];
            const int64_t z = loc[off + 2];
            if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
                base[z * ny * nx + y * nx + x] += source[(b * nsrc + isrc) * nt + it];
            }
        }
    }
}

void record_field_3d(const float* field, float* record, const int32_t* loc, int64_t B, int64_t nrec, int64_t nt, int64_t it, int64_t nz, int64_t ny, int64_t nx)
{
    const int64_t spatial = nz * ny * nx;
    for (int64_t b = 0; b < B; ++b) {
        const float* base = field + b * spatial;
        for (int64_t irec = 0; irec < nrec; ++irec) {
            const int64_t off = (b * nrec + irec) * 3;
            const int64_t x = loc[off];
            const int64_t y = loc[off + 1];
            const int64_t z = loc[off + 2];
            record[(b * nrec + irec) * nt + it] =
                (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) ? base[z * ny * nx + y * nx + x] : 0.0f;
        }
    }
}

template <int Order>
void elastic3d_adjoint_step(
    Elastic3DRawState& adj, Elastic3DWorkspace& work, const float* lambda, const float* mu, const float* rho,
    const float* az, const float* bz, const float* azh, const float* bzh,
    const float* ay, const float* by, const float* ayh, const float* byh,
    const float* ax, const float* bx, const float* axh, const float* bxh,
    int64_t B, int64_t nz, int64_t ny, int64_t nx, int abcn,
    float inv_dz, float inv_dy, float inv_dx, float dt
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * ny * nx;
    const int64_t total = B * spatial;
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t y = (idx / nx) % ny;
            const int64_t z = (idx / (nx * ny)) % nz;
            const float lam = lambda[idx];
            const float mu0 = mu[idx];
            const float l2m = lam + 2.0f * mu0;

            const float bar_sxx = adj.sxx[idx];
            const float bar_syy = adj.syy[idx];
            const float bar_szz = adj.szz[idx];
            const float bar_sxy = adj.sxy[idx];
            const float bar_sxz = adj.sxz[idx];
            const float bar_syz = adj.syz[idx];

            const float bar_dvx_dx = dt * (l2m * bar_sxx + lam * bar_syy + lam * bar_szz);
            const float bar_dvx_dy = dt * mu0 * bar_sxy;
            const float bar_dvx_dz = dt * mu0 * bar_sxz;
            const float bar_dvy_dx = dt * mu0 * bar_sxy;
            const float bar_dvy_dy = dt * (lam * bar_sxx + l2m * bar_syy + lam * bar_szz);
            const float bar_dvy_dz = dt * mu0 * bar_syz;
            const float bar_dvz_dx = dt * mu0 * bar_sxz;
            const float bar_dvz_dy = dt * mu0 * bar_syz;
            const float bar_dvz_dz = dt * (lam * bar_sxx + lam * bar_syy + l2m * bar_szz);

            const float tmp_vxx = adj.m_vxx[idx] + bar_dvx_dx;
            const float tmp_vxy = adj.m_vxy[idx] + bar_dvx_dy;
            const float tmp_vxz = adj.m_vxz[idx] + bar_dvx_dz;
            const float tmp_vyx = adj.m_vyx[idx] + bar_dvy_dx;
            const float tmp_vyy = adj.m_vyy[idx] + bar_dvy_dy;
            const float tmp_vyz = adj.m_vyz[idx] + bar_dvy_dz;
            const float tmp_vzx = adj.m_vzx[idx] + bar_dvz_dx;
            const float tmp_vzy = adj.m_vzy[idx] + bar_dvz_dy;
            const float tmp_vzz = adj.m_vzz[idx] + bar_dvz_dz;

            if (is_elastic3d_interior(x, y, z, nx, ny, nz, M, abcn)) {
                work.qxx[idx] = bar_dvx_dx;
                work.qxy[idx] = bar_dvx_dy;
                work.qxz[idx] = bar_dvx_dz;
                work.qyx[idx] = bar_dvy_dx;
                work.qyy[idx] = bar_dvy_dy;
                work.qyz[idx] = bar_dvy_dz;
                work.qzx[idx] = bar_dvz_dx;
                work.qzy[idx] = bar_dvz_dy;
                work.qzz[idx] = bar_dvz_dz;
            } else {
                work.qxx[idx] = bar_dvx_dx + bx[x] * tmp_vxx;
                work.qxy[idx] = bar_dvx_dy + byh[y] * tmp_vxy;
                work.qxz[idx] = bar_dvx_dz + bzh[z] * tmp_vxz;
                work.qyx[idx] = bar_dvy_dx + bxh[x] * tmp_vyx;
                work.qyy[idx] = bar_dvy_dy + by[y] * tmp_vyy;
                work.qyz[idx] = bar_dvy_dz + bzh[z] * tmp_vyz;
                work.qzx[idx] = bar_dvz_dx + bxh[x] * tmp_vzx;
                work.qzy[idx] = bar_dvz_dy + byh[y] * tmp_vzy;
                work.qzz[idx] = bar_dvz_dz + bz[z] * tmp_vzz;
                adj.m_vxx[idx] = ax[x] * tmp_vxx;
                adj.m_vxy[idx] = ayh[y] * tmp_vxy;
                adj.m_vxz[idx] = azh[z] * tmp_vxz;
                adj.m_vyx[idx] = axh[x] * tmp_vyx;
                adj.m_vyy[idx] = ay[y] * tmp_vyy;
                adj.m_vyz[idx] = azh[z] * tmp_vyz;
                adj.m_vzx[idx] = axh[x] * tmp_vzx;
                adj.m_vzy[idx] = ayh[y] * tmp_vzy;
                adj.m_vzz[idx] = az[z] * tmp_vzz;
            }
        }
    });

    const int64_t yz_count = B * (nz - 2 * M) * (ny - 2 * M);
    at::parallel_for(0, yz_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / ((nz - 2 * M) * (ny - 2 * M));
            const int64_t rem = row - b * ((nz - 2 * M) * (ny - 2 * M));
            const int64_t z = M + rem / (ny - 2 * M);
            const int64_t y = M + rem % (ny - 2 * M);
            const int64_t base = b * spatial + z * ny * nx + y * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float dqxx_dx = sgrad_forward<Order>(work.qxx.data(), idx, 1, inv_dx, stencil);
                const float dqxy_dy = sgrad_backward<Order>(work.qxy.data(), idx, nx, inv_dy, stencil);
                const float dqxz_dz = sgrad_backward<Order>(work.qxz.data(), idx, ny * nx, inv_dz, stencil);
                const float dqyx_dx = sgrad_backward<Order>(work.qyx.data(), idx, 1, inv_dx, stencil);
                const float dqyy_dy = sgrad_forward<Order>(work.qyy.data(), idx, nx, inv_dy, stencil);
                const float dqyz_dz = sgrad_backward<Order>(work.qyz.data(), idx, ny * nx, inv_dz, stencil);
                const float dqzx_dx = sgrad_backward<Order>(work.qzx.data(), idx, 1, inv_dx, stencil);
                const float dqzy_dy = sgrad_backward<Order>(work.qzy.data(), idx, nx, inv_dy, stencil);
                const float dqzz_dz = sgrad_forward<Order>(work.qzz.data(), idx, ny * nx, inv_dz, stencil);
                adj.vx[idx] += dqxx_dx + dqxy_dy + dqxz_dz;
                adj.vy[idx] += dqyx_dx + dqyy_dy + dqyz_dz;
                adj.vz[idx] += dqzx_dx + dqzy_dy + dqzz_dz;
            }
        }
    });

    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t x = idx % nx;
            const int64_t y = (idx / nx) % ny;
            const int64_t z = (idx / (nx * ny)) % nz;
            const float inv_rho = 1.0f / rho[idx];
            const float bar_dsxx_dx = dt * inv_rho * adj.vx[idx];
            const float bar_dsxy_dy = dt * inv_rho * adj.vx[idx];
            const float bar_dsxz_dz = dt * inv_rho * adj.vx[idx];
            const float bar_dsxy_dx = dt * inv_rho * adj.vy[idx];
            const float bar_dsyy_dy = dt * inv_rho * adj.vy[idx];
            const float bar_dsyz_dz = dt * inv_rho * adj.vy[idx];
            const float bar_dsxz_dx = dt * inv_rho * adj.vz[idx];
            const float bar_dsyz_dy = dt * inv_rho * adj.vz[idx];
            const float bar_dszz_dz = dt * inv_rho * adj.vz[idx];

            const float tmp_sxxx = adj.m_sxxx[idx] + bar_dsxx_dx;
            const float tmp_sxyy = adj.m_sxyy[idx] + bar_dsxy_dy;
            const float tmp_sxzz = adj.m_sxzz[idx] + bar_dsxz_dz;
            const float tmp_sxyx = adj.m_sxyx[idx] + bar_dsxy_dx;
            const float tmp_syyy = adj.m_syyy[idx] + bar_dsyy_dy;
            const float tmp_syzz = adj.m_syzz[idx] + bar_dsyz_dz;
            const float tmp_sxzx = adj.m_sxzx[idx] + bar_dsxz_dx;
            const float tmp_syzy = adj.m_syzy[idx] + bar_dsyz_dy;
            const float tmp_szzz = adj.m_szzz[idx] + bar_dszz_dz;

            if (is_elastic3d_interior(x, y, z, nx, ny, nz, M, abcn)) {
                work.pxx[idx] = bar_dsxx_dx;
                work.pxy[idx] = bar_dsxy_dy;
                work.pxz[idx] = bar_dsxz_dz;
                work.pyx[idx] = bar_dsxy_dx;
                work.pyy[idx] = bar_dsyy_dy;
                work.pyz[idx] = bar_dsyz_dz;
                work.pzx[idx] = bar_dsxz_dx;
                work.pzy[idx] = bar_dsyz_dy;
                work.pzz[idx] = bar_dszz_dz;
            } else {
                work.pxx[idx] = bar_dsxx_dx + bxh[x] * tmp_sxxx;
                work.pxy[idx] = bar_dsxy_dy + by[y] * tmp_sxyy;
                work.pxz[idx] = bar_dsxz_dz + bz[z] * tmp_sxzz;
                work.pyx[idx] = bar_dsxy_dx + bx[x] * tmp_sxyx;
                work.pyy[idx] = bar_dsyy_dy + byh[y] * tmp_syyy;
                work.pyz[idx] = bar_dsyz_dz + bz[z] * tmp_syzz;
                work.pzx[idx] = bar_dsxz_dx + bx[x] * tmp_sxzx;
                work.pzy[idx] = bar_dsyz_dy + by[y] * tmp_syzy;
                work.pzz[idx] = bar_dszz_dz + bzh[z] * tmp_szzz;
                adj.m_sxxx[idx] = axh[x] * tmp_sxxx;
                adj.m_sxyy[idx] = ay[y] * tmp_sxyy;
                adj.m_sxzz[idx] = az[z] * tmp_sxzz;
                adj.m_sxyx[idx] = ax[x] * tmp_sxyx;
                adj.m_syyy[idx] = ayh[y] * tmp_syyy;
                adj.m_syzz[idx] = az[z] * tmp_syzz;
                adj.m_sxzx[idx] = ax[x] * tmp_sxzx;
                adj.m_syzy[idx] = ay[y] * tmp_syzy;
                adj.m_szzz[idx] = azh[z] * tmp_szzz;
            }
        }
    });

    at::parallel_for(0, yz_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / ((nz - 2 * M) * (ny - 2 * M));
            const int64_t rem = row - b * ((nz - 2 * M) * (ny - 2 * M));
            const int64_t z = M + rem / (ny - 2 * M);
            const int64_t y = M + rem % (ny - 2 * M);
            const int64_t base = b * spatial + z * ny * nx + y * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float dpxx_dx = sgrad_backward<Order>(work.pxx.data(), idx, 1, inv_dx, stencil);
                const float dpxy_dy = sgrad_forward<Order>(work.pxy.data(), idx, nx, inv_dy, stencil);
                const float dpxz_dz = sgrad_forward<Order>(work.pxz.data(), idx, ny * nx, inv_dz, stencil);
                const float dpyx_dx = sgrad_forward<Order>(work.pyx.data(), idx, 1, inv_dx, stencil);
                const float dpyy_dy = sgrad_backward<Order>(work.pyy.data(), idx, nx, inv_dy, stencil);
                const float dpyz_dz = sgrad_forward<Order>(work.pyz.data(), idx, ny * nx, inv_dz, stencil);
                const float dpzx_dx = sgrad_forward<Order>(work.pzx.data(), idx, 1, inv_dx, stencil);
                const float dpzy_dy = sgrad_forward<Order>(work.pzy.data(), idx, nx, inv_dy, stencil);
                const float dpzz_dz = sgrad_backward<Order>(work.pzz.data(), idx, ny * nx, inv_dz, stencil);
                adj.sxx[idx] += dpxx_dx;
                adj.sxy[idx] += dpxy_dy + dpyx_dx;
                adj.sxz[idx] += dpxz_dz + dpzx_dx;
                adj.syy[idx] += dpyy_dy;
                adj.syz[idx] += dpyz_dz + dpzy_dy;
                adj.szz[idx] += dpzz_dz;
            }
        }
    });
}

template <int Order>
void accumulate_elastic3d_grad(
    const Elastic3DRawState& adj,
    const float* fvx, const float* fvy, const float* fvz,
    const float* fvx_next, const float* fvy_next, const float* fvz_next,
    const float* vp, const float* vs, const float* rho,
    float* grad_vp, float* grad_vs, float* grad_rho,
    int64_t B, int64_t nz, int64_t ny, int64_t nx,
    float inv_dz, float inv_dy, float inv_dx, float dt
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * ny * nx;
    const int64_t yz_count = B * (nz - 2 * M) * (ny - 2 * M);
    at::parallel_for(0, yz_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / ((nz - 2 * M) * (ny - 2 * M));
            const int64_t rem = row - b * ((nz - 2 * M) * (ny - 2 * M));
            const int64_t z = M + rem / (ny - 2 * M);
            const int64_t y = M + rem % (ny - 2 * M);
            const int64_t base = b * spatial + z * ny * nx + y * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float fvx_x = sgrad_backward<Order>(fvx, idx, 1, inv_dx, stencil);
                const float fvx_y = sgrad_forward<Order>(fvx, idx, nx, inv_dy, stencil);
                const float fvx_z = sgrad_forward<Order>(fvx, idx, ny * nx, inv_dz, stencil);
                const float fvy_x = sgrad_forward<Order>(fvy, idx, 1, inv_dx, stencil);
                const float fvy_y = sgrad_backward<Order>(fvy, idx, nx, inv_dy, stencil);
                const float fvy_z = sgrad_forward<Order>(fvy, idx, ny * nx, inv_dz, stencil);
                const float fvz_x = sgrad_forward<Order>(fvz, idx, 1, inv_dx, stencil);
                const float fvz_y = sgrad_forward<Order>(fvz, idx, nx, inv_dy, stencil);
                const float fvz_z = sgrad_backward<Order>(fvz, idx, ny * nx, inv_dz, stencil);

                const float grad_lambda = (adj.sxx[idx] + adj.syy[idx] + adj.szz[idx]) * (fvx_x + fvy_y + fvz_z);
                const float grad_mu =
                    2.0f * (adj.sxx[idx] * fvx_x + adj.syy[idx] * fvy_y + adj.szz[idx] * fvz_z) +
                    adj.sxz[idx] * (fvx_z + fvz_x) +
                    adj.sxy[idx] * (fvx_y + fvy_x) +
                    adj.syz[idx] * (fvy_z + fvz_y);

                grad_vp[idx] += -2.0f * rho[idx] * vp[idx] * grad_lambda * dt;
                grad_vs[idx] += -(-4.0f * rho[idx] * vs[idx] * grad_lambda + 2.0f * rho[idx] * vs[idx] * grad_mu) * dt;
                grad_rho[idx] += (adj.vx[idx] * (fvx[idx] - fvx_next[idx]) +
                                  adj.vy[idx] * (fvy[idx] - fvy_next[idx]) +
                                  adj.vz[idx] * (fvz[idx] - fvz_next[idx])) / rho[idx];
                grad_rho[idx] -= grad_lambda * (vp[idx] * vp[idx] - 2.0f * vs[idx] * vs[idx]) * dt +
                                 grad_mu * (vs[idx] * vs[idx]) * dt;
            }
        }
    });
}

template <int Order>
CpuForwardResult forward_elastic3d_raw_impl(const ForwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto vs_t = p.models[1];
    auto rho_t = p.models[2];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t ny = vp_t.size(3);
    const int64_t nx = vp_t.size(4);
    const int64_t spatial = nz * ny * nx;
    const int64_t total = B * spatial;
    const int64_t nsrc = p.sources_loc.size(1);
    const int64_t nrec = p.receivers_loc.size(1);
    const int64_t nsrc_fields = p.source_field_indices.numel();
    const int64_t nrec_fields = p.receiver_field_indices.numel();
    const int64_t nt = static_cast<int64_t>(p.nt);
    TORCH_CHECK(vp_t.size(1) == 1, "Elastic3D raw CPU forward expects one model channel");

    auto record = torch::zeros({nrec_fields, B, nrec, nt}, vp_t.options());
    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = torch::zeros({nt, 3, B, nz, ny, nx}, vp_t.options());

    std::vector<float> lambda(total, 0.0f), mu(total, 0.0f);
    build_lame(vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(), lambda, mu, total);
    Elastic3DRawState state(total);
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ay = pml[4].data_ptr<float>();
    const float* by = pml[5].data_ptr<float>();
    const float* ayh = pml[6].data_ptr<float>();
    const float* byh = pml[7].data_ptr<float>();
    const float* ax = pml[8].data_ptr<float>();
    const float* bx = pml[9].data_ptr<float>();
    const float* axh = pml[10].data_ptr<float>();
    const float* bxh = pml[11].data_ptr<float>();
    const float inv_dx = static_cast<float>(1.0 / p.spacing[0]);
    const float inv_dy = static_cast<float>(1.0 / p.spacing[1]);
    const float inv_dz = static_cast<float>(1.0 / p.spacing[2]);
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
        elastic3d_velocity_step<Order>(state, rho_t.data_ptr<float>(), az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
                                   B, nz, ny, nx, p.abcn, inv_dz, inv_dy, inv_dx, dt, stencil);
        elastic3d_stress_step<Order>(state, lambda.data(), mu.data(), az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
                                 B, nz, ny, nx, p.abcn, inv_dz, inv_dy, inv_dx, dt, stencil);
        if (u_allt.defined()) {
            copy_vector_to_tensor(state.vx, u_allt.select(0, it).select(0, 0));
            copy_vector_to_tensor(state.vy, u_allt.select(0, it).select(0, 1));
            copy_vector_to_tensor(state.vz, u_allt.select(0, it).select(0, 2));
        }
        auto fields = mutable_field_ptrs(state);
        for (int64_t f = 0; f < nsrc_fields; ++f) {
            const int field_id = source_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(fields.size())) {
                add_source_to_field_3d(fields[field_id], source, sources, B, nsrc, nt, it, nz, ny, nx);
            }
        }
        auto cfields = const_field_ptrs(state);
        for (int64_t f = 0; f < nrec_fields; ++f) {
            const int field_id = receiver_fields[f];
            if (field_id >= 0 && field_id < static_cast<int>(cfields.size())) {
                record_field_3d(cfields[field_id], rec + f * B * nrec * nt, receivers, B, nrec, nt, it, nz, ny, nx);
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
        if (!last_two.defined() || last_two.numel() == 0) last_two = torch::zeros({9, 1, B, 1, nz, ny, nx}, vp_t.options());
        copy_vector_to_tensor(state.vx, last_two.select(0, 0).select(0, 0));
        copy_vector_to_tensor(state.vy, last_two.select(0, 1).select(0, 0));
        copy_vector_to_tensor(state.vz, last_two.select(0, 2).select(0, 0));
        copy_vector_to_tensor(state.sxx, last_two.select(0, 3).select(0, 0));
        copy_vector_to_tensor(state.syy, last_two.select(0, 4).select(0, 0));
        copy_vector_to_tensor(state.szz, last_two.select(0, 5).select(0, 0));
        copy_vector_to_tensor(state.sxy, last_two.select(0, 6).select(0, 0));
        copy_vector_to_tensor(state.sxz, last_two.select(0, 7).select(0, 0));
        copy_vector_to_tensor(state.syz, last_two.select(0, 8).select(0, 0));
    }
    return {record, u_allt, last_two};
}

CpuForwardResult forward_elastic3d_raw(const ForwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, forward_elastic3d_raw_impl, p);
}

template <int Order>
BackwardOutput backward_elastic3d_full_impl(const BackwardInput& p, bool skip_initial_time = false)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto vs_t = p.models[1];
    auto rho_t = p.models[2];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t ny = vp_t.size(3);
    const int64_t nx = vp_t.size(4);
    const int64_t spatial = nz * ny * nx;
    const int64_t total = B * spatial;
    const int64_t nt = static_cast<int64_t>(p.nt);
    TORCH_CHECK(p.u_forward.defined() && p.u_forward.dim() == 6 && p.u_forward.size(0) == nt && p.u_forward.size(1) == 3,
                "Elastic3D full CPU backward requires forward wavefields with shape (nt, 3, B, nz, ny, nx)");

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_vs = torch::zeros_like(vs_t);
    auto grad_rho = torch::zeros_like(rho_t);
    std::vector<float> lambda(total, 0.0f), mu(total, 0.0f);
    build_lame(vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(), lambda, mu, total);
    Elastic3DRawState adj(total);
    Elastic3DWorkspace work(total);
    std::vector<float> zero(total, 0.0f);
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* azh = pml[2].data_ptr<float>();
    const float* bzh = pml[3].data_ptr<float>();
    const float* ay = pml[4].data_ptr<float>();
    const float* by = pml[5].data_ptr<float>();
    const float* ayh = pml[6].data_ptr<float>();
    const float* byh = pml[7].data_ptr<float>();
    const float* ax = pml[8].data_ptr<float>();
    const float* bx = pml[9].data_ptr<float>();
    const float* axh = pml[10].data_ptr<float>();
    const float* bxh = pml[11].data_ptr<float>();
    const float inv_dx = static_cast<float>(1.0 / p.spacing[0]);
    const float inv_dy = static_cast<float>(1.0 / p.spacing[1]);
    const float inv_dz = static_cast<float>(1.0 / p.spacing[2]);
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
                add_source_to_field_3d(fields[field_id], adj_source + f * B * adjoint_nsrc * nt, adjoint_sources,
                                       B, adjoint_nsrc, nt, it, nz, ny, nx);
            }
        }
        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vy_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 2).data_ptr<float>();
        const float* vx_next = (it + 1 < nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero.data();
        const float* vy_next = (it + 1 < nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero.data();
        const float* vz_next = (it + 1 < nt) ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>() : zero.data();
        accumulate_elastic3d_grad<Order>(adj, vx_now, vy_now, vz_now, vx_next, vy_next, vz_next,
                                     vp_t.data_ptr<float>(), vs_t.data_ptr<float>(), rho_t.data_ptr<float>(),
                                     grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_rho.data_ptr<float>(),
                                     B, nz, ny, nx, inv_dz, inv_dy, inv_dx, dt, stencil);
        if (it == 0) continue;
        elastic3d_adjoint_step<Order>(adj, work, lambda.data(), mu.data(), rho_t.data_ptr<float>(),
                                  az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
                                  B, nz, ny, nx, p.abcn, inv_dz, inv_dy, inv_dx, dt, stencil);
    }
    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

template <int Order>
BackwardOutput backward_elastic3d_raw_impl(const BackwardInput& p)
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
    auto forward_result = forward_elastic3d_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_elastic3d_full_impl<Order>(replay);
}

template <int Order>
BackwardOutput backward_elastic3d_bs_impl(const BackwardInput& p)
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
    auto forward_result = forward_elastic3d_raw(fwd);
    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_elastic3d_full_impl<Order>(replay, true);
}

BackwardOutput backward_elastic3d_raw(const BackwardInput& p)
{
    const bool has_full_wavefield = p.u_forward.defined() && p.u_forward.numel() > 0;
    const bool has_boundary_state = p.u_last_two.defined() && p.u_last_two.numel() > 0;

    #define SWEEP_CPU_ELASTIC3D_BACKWARD_CASE(ORDER) \
        if (has_full_wavefield) return backward_elastic3d_full_impl<ORDER>(p); \
        if (has_boundary_state) return backward_elastic3d_bs_impl<ORDER>(p); \
        return backward_elastic3d_raw_impl<ORDER>(p);
    SWEEP_CPU_DISPATCH_STENCIL_BODY(p.M, SWEEP_CPU_ELASTIC3D_BACKWARD_CASE);
    #undef SWEEP_CPU_ELASTIC3D_BACKWARD_CASE
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic3d::forward called with non-CPU tensors");
    if (!can_use_elastic3d_raw_forward(in)) {
        return engine::forward(in, EquationKind::Elastic3D);
    }
    auto result = forward_elastic3d_raw(in);
    ForwardOutput out;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::elastic3d::backward called with non-CPU tensors");
    TORCH_CHECK(
        can_use_elastic3d_raw_backward(in),
        "Elastic3D CPU backward requires the handwritten raw float32 path; unsupported inputs will not fall back to torch autograd."
    );
    return backward_elastic3d_raw(in);
}

BackwardOutput backward_bs(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_ckpt(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_recursive_ckpt(const BackwardInput& in) { return backward(in); }

} // namespace sweep_cpu::elastic3d

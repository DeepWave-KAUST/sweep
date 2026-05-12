#include "acoustic_vrz2d_cpu.h"

#include "../../common/cpu_engine.h"
#include "../../operators/fd.h"

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <vector>

namespace sweep_cpu::acoustic_vrz2d {
namespace {

using sweep_cpu::ops::StencilCoefficients;
struct CpuForwardResult {
    torch::Tensor record;
    torch::Tensor wavefield;
    torch::Tensor last_two;
};

std::vector<double> spacing_for(const ForwardInput& p, int ndim)
{
    if (ndim == 2) {
        return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
    }
    return {
        static_cast<double>(p.spacing[2]),
        static_cast<double>(p.spacing[1]),
        static_cast<double>(p.spacing[0]),
    };
}

std::vector<double> spacing_for(const BackwardInput& p, int ndim)
{
    if (ndim == 2) {
        return {static_cast<double>(p.spacing[1]), static_cast<double>(p.spacing[0])};
    }
    return {
        static_cast<double>(p.spacing[2]),
        static_cast<double>(p.spacing[1]),
        static_cast<double>(p.spacing[0]),
    };
}

using sweep_cpu::ops::centered_coeff;
using sweep_cpu::ops::centered_grad_1d;
using sweep_cpu::ops::centered_grad_value;
using sweep_cpu::ops::centered_grad_weighted_value;
using sweep_cpu::ops::laplace_value;

bool can_use_acoustic_vrz2d_raw_forward(const ForwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 2 || p.pml_vals.size() < 6) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, true)) return false;
    if (!p.models[0].is_contiguous() || p.models[0].scalar_type() != torch::kFloat32 || p.models[0].dim() != 4) return false;
    if (!p.models[1].is_contiguous() || p.models[1].scalar_type() != torch::kFloat32 || p.models[1].dim() != 4) return false;
    if (p.models[1].sizes() != p.models[0].sizes()) return false;
    if (!p.source.is_contiguous() || p.source.scalar_type() != torch::kFloat32 || p.source.dim() != 3) return false;
    if (!p.sources_loc.is_contiguous() || !p.receivers_loc.is_contiguous()) return false;
    if (p.sources_loc.scalar_type() != torch::kInt32 || p.receivers_loc.scalar_type() != torch::kInt32) return false;
    for (const auto& t : p.pml_vals) {
        if (!t.is_contiguous() || t.scalar_type() != torch::kFloat32) return false;
    }
    const int64_t nz = p.models[0].size(2);
    const int64_t nx = p.models[0].size(3);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz) return false;
    if (p.pml_vals[3].numel() != nx || p.pml_vals[4].numel() != nx || p.pml_vals[5].numel() != nx) return false;
    return true;
}

void zero_acoustic_vrz2d_boundary_width(
    std::vector<float>& field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int width
)
{
    if (width <= 0) return;
    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        float* base = field.data() + b * spatial;
        const int64_t z_width = std::min<int64_t>(width, nz);
        const int64_t x_width = std::min<int64_t>(width, nx);
        std::fill(base, base + z_width * nx, 0.0f);
        std::fill(base + std::max<int64_t>(nz - z_width, 0) * nx, base + nz * nx, 0.0f);
        for (int64_t z = z_width; z < nz - z_width; ++z) {
            float* row = base + z * nx;
            std::fill(row, row + x_width, 0.0f);
            std::fill(row + nx - x_width, row + nx, 0.0f);
        }
    }
}

void zero_acoustic_vrz2d_boundary(
    std::vector<float>& field,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M
)
{
    zero_acoustic_vrz2d_boundary_width(field, B, nz, nx, M);
}

struct AcousticVRZ2DRawState {
    std::vector<float> u_prev;
    std::vector<float> u_now;
    std::vector<float> u_next;
    std::vector<float> psix;
    std::vector<float> psiz;
    std::vector<float> zetax;
    std::vector<float> zetaz;
    std::vector<float> psix_next;
    std::vector<float> psiz_next;
    std::vector<float> zetax_next;
    std::vector<float> zetaz_next;

    AcousticVRZ2DRawState() = default;

    explicit AcousticVRZ2DRawState(int64_t total)
        : u_prev(total, 0.0f),
          u_now(total, 0.0f),
          u_next(total, 0.0f),
          psix(total, 0.0f),
          psiz(total, 0.0f),
          zetax(total, 0.0f),
          zetaz(total, 0.0f),
          psix_next(total, 0.0f),
          psiz_next(total, 0.0f),
          zetax_next(total, 0.0f),
          zetaz_next(total, 0.0f)
    {}

    void swap()
    {
        std::swap(u_prev, u_now);
        std::swap(u_now, u_next);
        std::swap(psix, psix_next);
        std::swap(psiz, psiz_next);
        std::swap(zetax, zetax_next);
        std::swap(zetaz, zetaz_next);
    }
};

void copy_tensor_to_vector(const torch::Tensor& tensor, std::vector<float>& dst)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(dst.size()) == tensor.numel(), "Tensor/vector size mismatch");
    const float* ptr = tensor.data_ptr<float>();
    std::copy(ptr, ptr + tensor.numel(), dst.begin());
}

void copy_vector_to_tensor(const std::vector<float>& src, torch::Tensor tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(static_cast<int64_t>(src.size()) == tensor.numel(), "Tensor/vector size mismatch");
    std::copy(src.begin(), src.end(), tensor.data_ptr<float>());
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

struct AcousticVRZ2DBoundaryGeometry {
    int phys_x0;
    int phys_x1;
    int phys_z0;
    int phys_z1;
    int nx_boundary;
    int nz_boundary;
    int top_z0;
    int bottom_z0;
    int left_x0;
    int right_x0;
};

AcousticVRZ2DBoundaryGeometry make_acoustic_vrz2d_boundary_geometry(
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    // Match CUDA boundary_kernel2d's compact layout: save normal halos around
    // SolverContext's physical box, without tangential padding in the index math.
    const int phys_x0 = abcn + M;
    const int phys_x1 = static_cast<int>(nx) - abcn - M;
    const int phys_z0 = free_surface ? M : abcn + M;
    const int phys_z1 = static_cast<int>(nz) - abcn - M;
    const int nx_boundary = phys_x1 - phys_x0;
    const int nz_boundary = phys_z1 - phys_z0;

    TORCH_CHECK(width > 0, "AcousticVRZ2D boundary width must be positive");
    TORCH_CHECK(nx_boundary > 0 && nz_boundary > 0,
                "AcousticVRZ2D boundary-saving physical domain is empty");

    AcousticVRZ2DBoundaryGeometry g{
        phys_x0,
        phys_x1,
        phys_z0,
        phys_z1,
        nx_boundary,
        nz_boundary,
        phys_z0 - M,
        phys_z1 + M - width,
        phys_x0 - M,
        phys_x1 + M - width,
    };
    TORCH_CHECK(g.top_z0 >= 0 && g.bottom_z0 + width <= nz,
                "AcousticVRZ2D z boundary geometry is out of bounds");
    TORCH_CHECK(g.left_x0 >= 0 && g.right_x0 + width <= nx,
                "AcousticVRZ2D x boundary geometry is out of bounds");
    return g;
}

int acoustic_vrz2d_checkpoint_index(int it, int nt, int interval)
{
    if (interval < 1) return -1;
    if (((it + 1) % interval == 0) && (it + 1 < nt)) {
        return (it + 1) / interval;
    }
    return -1;
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

int acoustic_vrz2d_recursive_checkpoint_index(int it, const std::vector<int>& steps, int& next_idx)
{
    if (next_idx < static_cast<int>(steps.size()) && steps[next_idx] == it + 1) {
        return next_idx++;
    }
    return -1;
}

void save_acoustic_vrz2d_checkpoint(
    const std::vector<torch::Tensor>& checkpoints,
    int checkpoint_idx,
    const AcousticVRZ2DRawState& s
)
{
    if (checkpoint_idx < 0) return;
    TORCH_CHECK(checkpoints.size() == 6, "AcousticVRZ2D CPU checkpointing expects 6 checkpoint tensors");
    copy_vector_to_tensor(s.u_prev, checkpoints[0].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.u_now, checkpoints[1].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.psix, checkpoints[2].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.psiz, checkpoints[3].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.zetax, checkpoints[4].select(0, checkpoint_idx));
    copy_vector_to_tensor(s.zetaz, checkpoints[5].select(0, checkpoint_idx));
}

void load_acoustic_vrz2d_checkpoint(
    const std::vector<torch::Tensor>& checkpoints,
    int checkpoint_idx,
    AcousticVRZ2DRawState& s
)
{
    TORCH_CHECK(checkpoints.size() == 6, "AcousticVRZ2D CPU checkpointing expects 6 checkpoint tensors");
    copy_tensor_to_vector(checkpoints[0].select(0, checkpoint_idx), s.u_prev);
    copy_tensor_to_vector(checkpoints[1].select(0, checkpoint_idx), s.u_now);
    copy_tensor_to_vector(checkpoints[2].select(0, checkpoint_idx), s.psix);
    copy_tensor_to_vector(checkpoints[3].select(0, checkpoint_idx), s.psiz);
    copy_tensor_to_vector(checkpoints[4].select(0, checkpoint_idx), s.zetax);
    copy_tensor_to_vector(checkpoints[5].select(0, checkpoint_idx), s.zetaz);
    std::fill(s.u_next.begin(), s.u_next.end(), 0.0f);
    std::fill(s.psix_next.begin(), s.psix_next.end(), 0.0f);
    std::fill(s.psiz_next.begin(), s.psiz_next.end(), 0.0f);
    std::fill(s.zetax_next.begin(), s.zetax_next.end(), 0.0f);
    std::fill(s.zetaz_next.begin(), s.zetaz_next.end(), 0.0f);
}

void save_acoustic_vrz2d_boundary(
    const std::vector<torch::Tensor>& boundary,
    int it,
    const std::vector<float>& u,
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
    TORCH_CHECK(boundary.size() == 4, "AcousticVRZ2D boundary saving expects 4 boundary tensors");
    const auto geom = make_acoustic_vrz2d_boundary_geometry(nz, nx, M, abcn, free_surface, width);
    const int nx_boundary = geom.nx_boundary;
    const int nz_boundary = geom.nz_boundary;
    auto top = boundary[0];
    auto bottom = boundary[1];
    auto left = boundary[2];
    auto right = boundary[3];
    TORCH_CHECK(top.is_contiguous() && bottom.is_contiguous() && left.is_contiguous() && right.is_contiguous(),
                "AcousticVRZ2D boundary tensors must be contiguous");
    float* top_ptr = top.data_ptr<float>();
    float* bottom_ptr = bottom.data_ptr<float>();
    float* left_ptr = left.data_ptr<float>();
    float* right_ptr = right.data_ptr<float>();
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        const float* base = u.data() + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * width + w) * nx_boundary + x);
                top_ptr[bd_idx] = base[(geom.top_z0 + w) * nx + (geom.phys_x0 + x)];
                bottom_ptr[bd_idx] = base[(geom.bottom_z0 + w) * nx + (geom.phys_x0 + x)];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t bd_idx = (((static_cast<int64_t>(it) * B + b) * nz_boundary + z) * width + w);
                left_ptr[bd_idx] = base[(geom.phys_z0 + z) * nx + (geom.left_x0 + w)];
                right_ptr[bd_idx] = base[(geom.phys_z0 + z) * nx + (geom.right_x0 + w)];
            }
        }
    }
}

void restore_acoustic_vrz2d_boundary(
    const std::vector<torch::Tensor>& boundary,
    int saved_it,
    std::vector<float>& u,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    TORCH_CHECK(!boundary.empty(), "AcousticVRZ2D boundary-saving backward requires saved boundaries");
    TORCH_CHECK(boundary.size() == 4, "AcousticVRZ2D boundary saving expects 4 boundary tensors");
    const auto geom = make_acoustic_vrz2d_boundary_geometry(nz, nx, M, abcn, free_surface, width);
    const int nx_boundary = geom.nx_boundary;
    const int nz_boundary = geom.nz_boundary;
    auto top = boundary[0];
    auto bottom = boundary[1];
    auto left = boundary[2];
    auto right = boundary[3];
    TORCH_CHECK(top.is_contiguous() && bottom.is_contiguous() && left.is_contiguous() && right.is_contiguous(),
                "AcousticVRZ2D boundary tensors must be contiguous");
    const float* top_ptr = top.data_ptr<float>();
    const float* bottom_ptr = bottom.data_ptr<float>();
    const float* left_ptr = left.data_ptr<float>();
    const float* right_ptr = right.data_ptr<float>();
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        float* base = u.data() + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t bd_idx = (((static_cast<int64_t>(saved_it) * B + b) * width + w) * nx_boundary + x);
                base[(geom.top_z0 + w) * nx + (geom.phys_x0 + x)] = top_ptr[bd_idx];
                base[(geom.bottom_z0 + w) * nx + (geom.phys_x0 + x)] = bottom_ptr[bd_idx];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t bd_idx = (((static_cast<int64_t>(saved_it) * B + b) * nz_boundary + z) * width + w);
                base[(geom.phys_z0 + z) * nx + (geom.left_x0 + w)] = left_ptr[bd_idx];
                base[(geom.phys_z0 + z) * nx + (geom.right_x0 + w)] = right_ptr[bd_idx];
            }
        }
    }
}

void write_acoustic_vrz2d_boundary_file(
    const std::string& path,
    size_t offset_elems,
    const std::vector<float>& data
)
{
    std::ofstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    TORCH_CHECK(out.good(), "Failed to open boundary file for writing: ", path);
    out.seekp(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    TORCH_CHECK(out.good(), "Failed to write boundary file: ", path);
}

void read_acoustic_vrz2d_boundary_file(
    const std::string& path,
    size_t offset_elems,
    std::vector<float>& data
)
{
    std::ifstream in(path, std::ios::binary);
    TORCH_CHECK(in.good(), "Failed to open boundary file for reading: ", path);
    in.seekg(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    TORCH_CHECK(in.good(), "Failed to read boundary file: ", path);
}

void save_acoustic_vrz2d_boundary_disk(
    const std::vector<std::string>& files,
    int it,
    const std::vector<float>& u,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    TORCH_CHECK(files.size() == 4, "AcousticVRZ2D disk boundary saving expects 4 files");
    const auto geom = make_acoustic_vrz2d_boundary_geometry(nz, nx, M, abcn, free_surface, width);
    const int nx_boundary = geom.nx_boundary;
    const int nz_boundary = geom.nz_boundary;
    const size_t top_elems = static_cast<size_t>(B) * width * nx_boundary;
    const size_t left_elems = static_cast<size_t>(B) * nz_boundary * width;
    std::vector<float> top(top_elems), bottom(top_elems), left(left_elems), right(left_elems);
    const int64_t spatial = nz * nx;

    for (int64_t b = 0; b < B; ++b) {
        const float* base = u.data() + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t idx = ((b * width + w) * nx_boundary + x);
                top[idx] = base[(geom.top_z0 + w) * nx + (geom.phys_x0 + x)];
                bottom[idx] = base[(geom.bottom_z0 + w) * nx + (geom.phys_x0 + x)];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t idx = ((b * nz_boundary + z) * width + w);
                left[idx] = base[(geom.phys_z0 + z) * nx + (geom.left_x0 + w)];
                right[idx] = base[(geom.phys_z0 + z) * nx + (geom.right_x0 + w)];
            }
        }
    }

    write_acoustic_vrz2d_boundary_file(files[0], static_cast<size_t>(it) * top_elems, top);
    write_acoustic_vrz2d_boundary_file(files[1], static_cast<size_t>(it) * top_elems, bottom);
    write_acoustic_vrz2d_boundary_file(files[2], static_cast<size_t>(it) * left_elems, left);
    write_acoustic_vrz2d_boundary_file(files[3], static_cast<size_t>(it) * left_elems, right);
}

void restore_acoustic_vrz2d_boundary_disk(
    const std::vector<std::string>& files,
    int saved_it,
    std::vector<float>& u,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int M,
    int abcn,
    bool free_surface,
    int width
)
{
    TORCH_CHECK(files.size() == 4, "AcousticVRZ2D disk boundary saving expects 4 files");
    const auto geom = make_acoustic_vrz2d_boundary_geometry(nz, nx, M, abcn, free_surface, width);
    const int nx_boundary = geom.nx_boundary;
    const int nz_boundary = geom.nz_boundary;
    const size_t top_elems = static_cast<size_t>(B) * width * nx_boundary;
    const size_t left_elems = static_cast<size_t>(B) * nz_boundary * width;
    std::vector<float> top(top_elems), bottom(top_elems), left(left_elems), right(left_elems);
    read_acoustic_vrz2d_boundary_file(files[0], static_cast<size_t>(saved_it) * top_elems, top);
    read_acoustic_vrz2d_boundary_file(files[1], static_cast<size_t>(saved_it) * top_elems, bottom);
    read_acoustic_vrz2d_boundary_file(files[2], static_cast<size_t>(saved_it) * left_elems, left);
    read_acoustic_vrz2d_boundary_file(files[3], static_cast<size_t>(saved_it) * left_elems, right);

    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        float* base = u.data() + b * spatial;
        for (int w = 0; w < width; ++w) {
            for (int x = 0; x < nx_boundary; ++x) {
                const int64_t idx = ((b * width + w) * nx_boundary + x);
                base[(geom.top_z0 + w) * nx + (geom.phys_x0 + x)] = top[idx];
                base[(geom.bottom_z0 + w) * nx + (geom.phys_x0 + x)] = bottom[idx];
            }
        }
        for (int z = 0; z < nz_boundary; ++z) {
            for (int w = 0; w < width; ++w) {
                const int64_t idx = ((b * nz_boundary + z) * width + w);
                base[(geom.phys_z0 + z) * nx + (geom.left_x0 + w)] = left[idx];
                base[(geom.phys_z0 + z) * nx + (geom.right_x0 + w)] = right[idx];
            }
        }
    }
}

template <int Order>
void advance_acoustic_vrz2d_cpml(
    AcousticVRZ2DRawState& s,
    const float* vp,
    const float* z_model,
    const float* inv_z_model,
    const float* az,
    const float* bz,
    const float* dbzdz,
    const float* ax,
    const float* bx,
    const float* dbxdx,
    int64_t B,
    int64_t nz,
    int64_t nx,
    float inv_dz,
    float inv_dx,
    float inv_dz2,
    float inv_dx2,
    float dt2
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int64_t row_count = B * (nz - 2 * M);
    const float* u_prev_ptr = s.u_prev.data();
    const float* u_now_ptr = s.u_now.data();
    const float* psix_ptr = s.psix.data();
    const float* psiz_ptr = s.psiz.data();
    const float* zetax_ptr = s.zetax.data();
    const float* zetaz_ptr = s.zetaz.data();
    float* u_next_ptr = s.u_next.data();
    float* psix_next_ptr = s.psix_next.data();
    float* psiz_next_ptr = s.psiz_next.data();
    float* zetax_next_ptr = s.zetax_next.data();
    float* zetaz_next_ptr = s.zetaz_next.data();

    at::parallel_for(0, total, 65536, [&](int64_t begin, int64_t end) {
        std::fill(u_next_ptr + begin, u_next_ptr + end, 0.0f);
        std::fill(psix_next_ptr + begin, psix_next_ptr + end, 0.0f);
        std::fill(psiz_next_ptr + begin, psiz_next_ptr + end, 0.0f);
        std::fill(zetax_next_ptr + begin, zetax_next_ptr + end, 0.0f);
        std::fill(zetaz_next_ptr + begin, zetaz_next_ptr + end, 0.0f);
    });

    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            const float az_z = az[z];
            const float bz_z = bz[z];
            const float dbzdz_z = dbzdz[z];
            const float dazdz = centered_grad_1d<Order>(az, z, inv_dz, stencil);
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float ax_x = ax[x];
                const float bx_x = bx[x];
                const float dbxdx_x = dbxdx[x];
                const float daxdx = centered_grad_1d<Order>(ax, x, inv_dx, stencil);
                const float lap_z = laplace_value<Order>(u_now_ptr, idx, nx, inv_dz2, stencil);
                const float lap_x = laplace_value<Order>(u_now_ptr, idx, 1, inv_dx2, stencil);
                const float dudz = centered_grad_value<Order>(u_now_ptr, idx, nx, inv_dz, stencil);
                const float dudx = centered_grad_value<Order>(u_now_ptr, idx, 1, inv_dx, stencil);
                const float dpsizdz = centered_grad_value<Order>(psiz_ptr, idx, nx, inv_dz, stencil);
                const float dpsixdx = centered_grad_value<Order>(psix_ptr, idx, 1, inv_dx, stencil);
                const float dvpdz = centered_grad_value<Order>(vp, idx, nx, inv_dz, stencil);
                const float dvpdx = centered_grad_value<Order>(vp, idx, 1, inv_dx, stencil);
                const float z1z = centered_grad_value<Order>(inv_z_model, idx, nx, inv_dz, stencil);
                const float z1x = centered_grad_value<Order>(inv_z_model, idx, 1, inv_dx, stencil);

                const float tmpz = (1.0f + bz_z) * lap_z + dbzdz_z * dudz + dazdz * psiz_ptr[idx] + az_z * dpsizdz;
                const float tmpx = (1.0f + bx_x) * lap_x + dbxdx_x * dudx + daxdx * psix_ptr[idx] + ax_x * dpsixdx;
                const float psiz_new = bz_z * dudz + az_z * psiz_ptr[idx];
                const float psix_new = bx_x * dudx + ax_x * psix_ptr[idx];
                const float zetaz_new = bz_z * tmpz + az_z * zetaz_ptr[idx];
                const float zetax_new = bx_x * tmpx + ax_x * zetax_ptr[idx];
                const float pz = dudz + psiz_new;
                const float px = dudx + psix_new;
                const float wsum = (1.0f + bz_z) * tmpz + az_z * zetaz_ptr[idx]
                                 + (1.0f + bx_x) * tmpx + ax_x * zetax_ptr[idx];

                const float v = vp[idx];
                const float inv_z0 = inv_z_model[idx];
                const float beta = v * inv_z0;
                const float kappa = v * z_model[idx];
                const float dbdz = dvpdz * inv_z0 + v * z1z;
                const float dbdx = dvpdx * inv_z0 + v * z1x;
                const float rhs = kappa * (beta * wsum + dbdx * px + dbdz * pz);

                u_next_ptr[idx] = 2.0f * u_now_ptr[idx] - u_prev_ptr[idx] + dt2 * rhs;
                psiz_next_ptr[idx] = psiz_new;
                psix_next_ptr[idx] = psix_new;
                zetaz_next_ptr[idx] = zetaz_new;
                zetax_next_ptr[idx] = zetax_new;
            }
        }
    });
}

template <int Order>
void advance_acoustic_vrz2d_nopml(
    AcousticVRZ2DRawState& s,
    const float* vp,
    const float* z_model,
    const float* inv_z_model,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int abcn,
    float inv_dz,
    float inv_dx,
    float inv_dz2,
    float inv_dx2,
    float dt2
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    const int64_t total = B * spatial;
    const int x_start = abcn + M + 1;
    const int x_end = static_cast<int>(nx) - abcn - M - 1;
    const int z_start = abcn + M + 1;
    const int z_end = static_cast<int>(nz) - abcn - M - 1;
    const float* u_prev_ptr = s.u_prev.data();
    const float* u_now_ptr = s.u_now.data();
    float* u_next_ptr = s.u_next.data();

    at::parallel_for(0, total, 65536, [&](int64_t begin, int64_t end) {
        std::fill(u_next_ptr + begin, u_next_ptr + end, 0.0f);
    });
    if (x_start >= x_end || z_start >= z_end) return;

    const int64_t row_count = B * (z_end - z_start);
    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (z_end - z_start);
            const int64_t z = z_start + row - b * (z_end - z_start);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = x_start; x < x_end; ++x) {
                const int64_t idx = base + x;
                const float lap_z = laplace_value<Order>(u_now_ptr, idx, nx, inv_dz2, stencil);
                const float lap_x = laplace_value<Order>(u_now_ptr, idx, 1, inv_dx2, stencil);
                const float dudz = centered_grad_value<Order>(u_now_ptr, idx, nx, inv_dz, stencil);
                const float dudx = centered_grad_value<Order>(u_now_ptr, idx, 1, inv_dx, stencil);
                const float dvpdz = centered_grad_value<Order>(vp, idx, nx, inv_dz, stencil);
                const float dvpdx = centered_grad_value<Order>(vp, idx, 1, inv_dx, stencil);
                const float z1z = centered_grad_value<Order>(inv_z_model, idx, nx, inv_dz, stencil);
                const float z1x = centered_grad_value<Order>(inv_z_model, idx, 1, inv_dx, stencil);
                const float v = vp[idx];
                const float inv_z0 = inv_z_model[idx];
                const float beta = v * inv_z0;
                const float kappa = v * z_model[idx];
                const float dbdz = dvpdz * inv_z0 + v * z1z;
                const float dbdx = dvpdx * inv_z0 + v * z1x;
                const float rhs = kappa * (beta * (lap_x + lap_z) + dbdx * dudx + dbdz * dudz);
                u_next_ptr[idx] = 2.0f * u_now_ptr[idx] - u_prev_ptr[idx] + dt2 * rhs;
            }
        }
    });
}

void add_acoustic_vrz2d_source(
    std::vector<float>& u,
    const float* source,
    const int32_t* sources,
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
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t loc = (b * nsrc + isrc) * 2;
            const int64_t x = sources[loc];
            const int64_t z = sources[loc + 1];
            if (x >= 0 && x < nx && z >= 0 && z < nz) {
                u[b * spatial + z * nx + x] += scale * source[(b * nsrc + isrc) * nt + it];
            }
        }
    }
}

void record_acoustic_vrz2d_receivers(
    const std::vector<float>& u,
    float* rec,
    const int32_t* receivers,
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
        for (int64_t irec = 0; irec < nrec; ++irec) {
            const int64_t loc = (b * nrec + irec) * 2;
            const int64_t x = receivers[loc];
            const int64_t z = receivers[loc + 1];
            rec[(b * nrec + irec) * nt + it] =
                (x >= 0 && x < nx && z >= 0 && z < nz) ? u[b * spatial + z * nx + x] : 0.0f;
        }
    }
}

[[maybe_unused]] void accumulate_acoustic_vrz2d_source_grad(
    float* grad_wavelet,
    const std::vector<float>& adj_now,
    const int32_t* sources,
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
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t loc = (b * nsrc + isrc) * 2;
            const int64_t x = sources[loc];
            const int64_t z = sources[loc + 1];
            if (x >= 0 && x < nx && z >= 0 && z < nz) {
                grad_wavelet[(b * nsrc + isrc) * nt + it] += adj_now[b * spatial + z * nx + x];
            }
        }
    }
}

template <int Order>
void build_acoustic_vrz2d_kappa_lambda(
    const std::vector<float>& lambda_now,
    const float* vp,
    const float* z_model,
    std::vector<float>& kappa_lambda,
    int64_t total
)
{
    at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            kappa_lambda[i] = vp[i] * z_model[i] * lambda_now[i];
        }
    });
}

template <int Order>
void accumulate_acoustic_vrz2d_model_grad(
    const float* f_u_now,
    const std::vector<float>& lambda_now,
    const std::vector<float>& kappa_lambda,
    const float* vp,
    const float* z_model,
    const float* inv_z_model,
    float* grad_vp,
    float* grad_z,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int abcn,
    float inv_dz,
    float inv_dx,
    float inv_dz2,
    float inv_dx2,
    float dt2
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    const int64_t spatial = nz * nx;
    const int64_t row_count = B * (nz - 2 * M);
    const int phys_x0 = abcn + M;
    const int phys_x1 = static_cast<int>(nx) - abcn - M;
    const int phys_z0 = abcn + M;
    const int phys_z1 = static_cast<int>(nz) - abcn - M;
    const float* lambda = lambda_now.data();
    const float* q = kappa_lambda.data();

    at::parallel_for(0, row_count, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t b = row / (nz - 2 * M);
            const int64_t z = M + row - b * (nz - 2 * M);
            const int64_t base = b * spatial + z * nx;
            for (int64_t x = M; x < nx - M; ++x) {
                const int64_t idx = base + x;
                const float lap_z = laplace_value<Order>(f_u_now, idx, nx, inv_dz2, stencil);
                const float lap_x = laplace_value<Order>(f_u_now, idx, 1, inv_dx2, stencil);
                const float dpdz = centered_grad_value<Order>(f_u_now, idx, nx, inv_dz, stencil);
                const float dpdx = centered_grad_value<Order>(f_u_now, idx, 1, inv_dx, stencil);
                const float dvpdz = centered_grad_value<Order>(vp, idx, nx, inv_dz, stencil);
                const float dvpdx = centered_grad_value<Order>(vp, idx, 1, inv_dx, stencil);
                const float z1z = centered_grad_value<Order>(inv_z_model, idx, nx, inv_dz, stencil);
                const float z1x = centered_grad_value<Order>(inv_z_model, idx, 1, inv_dx, stencil);
                const float v = vp[idx];
                const float inv_z0 = inv_z_model[idx];
                const float beta = v * inv_z0;
                const float dbdz = dvpdz * inv_z0 + v * z1z;
                const float dbdx = dvpdx * inv_z0 + v * z1x;
                const float div_b_grad_p = beta * (lap_x + lap_z) + dbdx * dpdx + dbdz * dpdz;

                float d_q_dpdx = 0.0f;
                float d_q_dpdz = 0.0f;
                if (x >= phys_x0 && x < phys_x1 && z >= phys_z0 && z < phys_z1) {
                    d_q_dpdx = centered_grad_value<Order>(q, idx, 1, inv_dx, stencil) * dpdx;
                    d_q_dpdz = centered_grad_value<Order>(q, idx, nx, inv_dz, stencil) * dpdz;
                }

                const float g_kappa = -dt2 * lambda[idx] * div_b_grad_p;
                const float g_beta = dt2 * (d_q_dpdx + d_q_dpdz);
                grad_vp[idx] += z_model[idx] * g_kappa + inv_z0 * g_beta;
                grad_z[idx] += v * g_kappa - beta * inv_z0 * g_beta;
            }
        }
    });
}

template <int Order>
void acoustic_vrz2d_adjoint_step(
    AcousticVRZ2DRawState& adj,
    const float* vp,
    const float* z_model,
    const float* inv_z_model,
    const float* adjoint_source,
    const int32_t* adjoint_sources,
    const float* az,
    const float* bz,
    const float* dbzdz,
    const float* ax,
    const float* bx,
    const float* dbxdx,
    int64_t B,
    int64_t nz,
    int64_t nx,
    int64_t adjoint_nsrc,
    int64_t nt,
    int64_t it,
    float inv_dz,
    float inv_dx,
    float inv_dz2,
    float inv_dx2,
    float dt2
,
    const StencilCoefficients& stencil)
{
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    advance_acoustic_vrz2d_cpml<Order>(
        adj,
        vp,
        z_model,
        inv_z_model,
        az,
        bz,
        dbzdz,
        ax,
        bx,
        dbxdx,
        B,
        nz,
        nx,
        inv_dz,
        inv_dx,
        inv_dz2,
        inv_dx2,
        dt2
    , stencil);
    add_acoustic_vrz2d_source(adj.u_next, adjoint_source, adjoint_sources, B, adjoint_nsrc, nt, it, nz, nx, -1.0f);
    adj.swap();
    zero_acoustic_vrz2d_boundary(adj.u_prev, B, nz, nx, M);
}

template <int Order>
CpuForwardResult forward_acoustic_vrz2d_raw_impl(const ForwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto z_t = p.models[1];
    auto source_t = p.source;
    auto sources_t = p.sources_loc;
    auto receivers_t = p.receivers_loc;
    const auto& pml = p.pml_vals;

    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t spatial = nz * nx;
    const int64_t nsrc = sources_t.size(1);
    const int64_t nrec = receivers_t.size(1);
    const int64_t nt = static_cast<int64_t>(p.nt);

    TORCH_CHECK(vp_t.size(1) == 1, "AcousticVRZ2D raw CPU forward expects one model channel");
    TORCH_CHECK(z_t.sizes() == vp_t.sizes(), "AcousticVRZ2D raw CPU forward expects vp and z to have the same shape");
    TORCH_CHECK(static_cast<int64_t>(source_t.size(0)) == B, "Source batch size does not match model batch size");
    TORCH_CHECK(static_cast<int64_t>(source_t.size(1)) == nsrc, "Source count does not match source locations");
    TORCH_CHECK(static_cast<int64_t>(source_t.size(2)) == nt, "Source time length does not match nt");
    TORCH_CHECK(nz > 2 * M && nx > 2 * M, "AcousticVRZ2D raw CPU forward grid is too small for stencil radius");

    auto record = torch::empty({B, nrec, nt}, vp_t.options());
    const float* vp = vp_t.data_ptr<float>();
    const float* z_model = z_t.data_ptr<float>();
    const float* source = source_t.data_ptr<float>();
    const int32_t* sources = sources_t.data_ptr<int32_t>();
    const int32_t* receivers = receivers_t.data_ptr<int32_t>();
    float* rec = record.data_ptr<float>();
    std::vector<float> inv_z(B * spatial, 0.0f);
    at::parallel_for(0, B * spatial, 4096, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) inv_z[i] = 1.0f / z_model[i];
    });

    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* dbzdz = pml[2].data_ptr<float>();
    const float* ax = pml[3].data_ptr<float>();
    const float* bx = pml[4].data_ptr<float>();
    const float* dbxdx = pml[5].data_ptr<float>();

    const auto h = spacing_for(p, 2);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float dt2 = static_cast<float>(p.dt * p.dt);

    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        u_allt = torch::zeros({nt, 5, B, 1, nz, nx}, vp_t.options());
    }

    auto boundary = active_boundary_tensors(p);
    const int save_width = M + 1;
    if (p.use_boundary_saving) {
        if (p.boundary_on_disk) {
            TORCH_CHECK(p.boundary_disk_files.size() == 4, "AcousticVRZ2D CPU disk boundary-saving forward requires 4 files");
        } else {
            TORCH_CHECK(!boundary.empty(), "AcousticVRZ2D CPU boundary-saving forward requires boundary tensors");
        }
    }
    if (p.use_checkpoint) {
        TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ2D CPU checkpointing expects 6 checkpoint tensors");
    }

    AcousticVRZ2DRawState state(B * spatial);
    std::vector<int> recursive_steps = checkpoint_steps_vector(p.checkpoint_steps);
    int next_recursive_checkpoint = 0;
    for (int64_t it = 0; it < nt; ++it) {
        [[maybe_unused]] float* imaging = u_allt.defined()
            ? u_allt.select(0, it).data_ptr<float>()
            : nullptr;
        advance_acoustic_vrz2d_cpml<Order>(
            state,
            vp,
            z_model,
            inv_z.data(),
            az,
            bz,
            dbzdz,
            ax,
            bx,
            dbxdx,
            B,
            nz,
            nx,
            inv_dz,
            inv_dx,
            inv_dz2,
            inv_dx2,
            dt2
        , stencil);

        if (p.use_boundary_saving) {
            if (p.boundary_on_disk) {
                save_acoustic_vrz2d_boundary_disk(
                    p.boundary_disk_files,
                    static_cast<int>(it),
                    state.u_now,
                    B,
                    nz,
                    nx,
                    M,
                    p.abcn,
                    p.free_surface,
                    save_width
                );
            } else {
                save_acoustic_vrz2d_boundary(
                    boundary,
                    static_cast<int>(it),
                    state.u_now,
                    B,
                    nz,
                    nx,
                    M,
                    p.abcn,
                    p.free_surface,
                    save_width
                );
            }
        }

        add_acoustic_vrz2d_source(state.u_next, source, sources, B, nsrc, nt, it, nz, nx);
        record_acoustic_vrz2d_receivers(state.u_next, rec, receivers, B, nrec, nt, it, nz, nx);

        state.swap();
        if (u_allt.defined()) {
            copy_vector_to_tensor(state.u_now, u_allt.select(0, it).select(0, 0).squeeze(1));
            copy_vector_to_tensor(state.psix, u_allt.select(0, it).select(0, 1).squeeze(1));
            copy_vector_to_tensor(state.psiz, u_allt.select(0, it).select(0, 2).squeeze(1));
            copy_vector_to_tensor(state.zetax, u_allt.select(0, it).select(0, 3).squeeze(1));
            copy_vector_to_tensor(state.zetaz, u_allt.select(0, it).select(0, 4).squeeze(1));
        }

        if (p.use_checkpoint) {
            const int ckpt_idx = p.use_recursive_checkpoint
                ? acoustic_vrz2d_recursive_checkpoint_index(static_cast<int>(it), recursive_steps, next_recursive_checkpoint)
                : acoustic_vrz2d_checkpoint_index(static_cast<int>(it), static_cast<int>(nt), p.checkpoint_interval);
            save_acoustic_vrz2d_checkpoint(p.checkpoints, ckpt_idx, state);
        }
    }

    torch::Tensor last_two = p.last_two.defined() ? p.last_two : torch::empty({0}, vp_t.options());
    if (p.use_boundary_saving) {
        if (!last_two.defined() || last_two.numel() == 0) {
            last_two = torch::zeros({1, 2, B, 1, nz, nx}, vp_t.options());
        }
        TORCH_CHECK(last_two.is_contiguous(), "AcousticVRZ2D CPU boundary-saving last_two must be contiguous");
        copy_vector_to_tensor(state.u_prev, last_two.select(0, 0).select(0, 0));
        copy_vector_to_tensor(state.u_now, last_two.select(0, 0).select(0, 1));
    }

    return {record, u_allt, last_two};
}

CpuForwardResult forward_acoustic_vrz2d_raw(const ForwardInput& p)
{
    SWEEP_CPU_DISPATCH_STENCIL(p.M, forward_acoustic_vrz2d_raw_impl, p);
}

bool can_use_acoustic_vrz2d_raw_backward(const BackwardInput& p)
{
    if (p.free_surface) return false;
    if (p.models.size() != 2 || p.pml_vals.size() < 6) return false;
    if (p.M < 1) return false;
    if (!sweep_cpu::ops::runtime_stencil_coefficients_are_valid(p.M, p.lap_coes, p.grad_coes, true)) return false;
    if (!p.models[0].is_contiguous() || p.models[0].scalar_type() != torch::kFloat32 || p.models[0].dim() != 4) return false;
    if (!p.models[1].is_contiguous() || p.models[1].scalar_type() != torch::kFloat32 || p.models[1].dim() != 4) return false;
    if (p.models[1].sizes() != p.models[0].sizes()) return false;
    if (!p.forward_source.is_contiguous() || p.forward_source.scalar_type() != torch::kFloat32 || p.forward_source.dim() != 3) return false;
    if (!p.adjoint_source.is_contiguous() || p.adjoint_source.scalar_type() != torch::kFloat32 || p.adjoint_source.dim() != 3) return false;
    if (!p.forward_sources_loc.is_contiguous() || !p.adjoint_sources_loc.is_contiguous()) return false;
    if (p.forward_sources_loc.scalar_type() != torch::kInt32 || p.adjoint_sources_loc.scalar_type() != torch::kInt32) return false;
    for (const auto& t : p.pml_vals) {
        if (!t.is_contiguous() || t.scalar_type() != torch::kFloat32) return false;
    }
    const int64_t nz = p.models[0].size(2);
    const int64_t nx = p.models[0].size(3);
    if (p.pml_vals[0].numel() != nz || p.pml_vals[1].numel() != nz || p.pml_vals[2].numel() != nz) return false;
    if (p.pml_vals[3].numel() != nx || p.pml_vals[4].numel() != nx || p.pml_vals[5].numel() != nx) return false;
    return true;
}


template <typename Input>
std::vector<float> make_inv_z_vector(const torch::Tensor& z_t)
{
    std::vector<float> inv_z(z_t.numel(), 0.0f);
    const float* z_model = z_t.data_ptr<float>();
    at::parallel_for(0, z_t.numel(), 4096, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) inv_z[i] = 1.0f / z_model[i];
    });
    return inv_z;
}

template <int Order>
BackwardOutput backward_acoustic_vrz2d_full_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto z_t = p.models[1];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t total = B * nz * nx;
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t nt = static_cast<int64_t>(p.nt);

    TORCH_CHECK(p.u_forward.defined() && p.u_forward.dim() == 6 && p.u_forward.size(0) == nt && p.u_forward.size(1) == 5,
                "AcousticVRZ2D full CPU backward requires saved forward wavefields with shape (nt, 5, B, 1, nz, nx)");
    TORCH_CHECK(p.u_forward.is_contiguous(), "AcousticVRZ2D full CPU forward wavefields must be contiguous");

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_z = torch::zeros_like(z_t);

    const float* vp = vp_t.data_ptr<float>();
    const float* z_model = z_t.data_ptr<float>();
    auto inv_z = make_inv_z_vector<BackwardInput>(z_t);
    const float* adjoint_source = p.adjoint_source.data_ptr<float>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    float* grad_vp_ptr = grad_vp.data_ptr<float>();
    float* grad_z_ptr = grad_z.data_ptr<float>();
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* dbzdz = pml[2].data_ptr<float>();
    const float* ax = pml[3].data_ptr<float>();
    const float* bx = pml[4].data_ptr<float>();
    const float* dbxdx = pml[5].data_ptr<float>();
    const auto h = spacing_for(p, 2);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float dt2 = static_cast<float>(p.dt * p.dt);

    AcousticVRZ2DRawState adj(total);
    std::vector<float> kappa_lambda(total, 0.0f);
    for (int64_t it = nt - 1; it >= 0; --it) {
        acoustic_vrz2d_adjoint_step<Order>(
            adj, vp, z_model, inv_z.data(), adjoint_source, adjoint_sources,
            az, bz, dbzdz, ax, bx, dbxdx,
            B, nz, nx, adjoint_nsrc, nt, it,
            inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
        , stencil);
        build_acoustic_vrz2d_kappa_lambda<Order>(adj.u_now, vp, z_model, kappa_lambda, total);
        auto fwd_u = p.u_forward.select(0, it).select(0, 0).squeeze(1);
        accumulate_acoustic_vrz2d_model_grad<Order>(
            fwd_u.data_ptr<float>(), adj.u_now, kappa_lambda, vp, z_model, inv_z.data(),
            grad_vp_ptr, grad_z_ptr, B, nz, nx, p.abcn,
            inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
        , stencil);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_z};
    return out;
}

template <int Order>
BackwardOutput backward_acoustic_vrz2d_ckpt_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto z_t = p.models[1];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t total = B * nz * nx;
    const int64_t forward_nsrc = p.forward_sources_loc.size(1);
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t nt = static_cast<int64_t>(p.nt);
    const int chunk_size = std::max(1, p.checkpoint_interval);
    const int num_chunks = (static_cast<int>(nt) + chunk_size - 1) / chunk_size;
    TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ2D checkpoint CPU backward expects 6 checkpoint tensors");

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_z = torch::zeros_like(z_t);

    const float* vp = vp_t.data_ptr<float>();
    const float* z_model = z_t.data_ptr<float>();
    auto inv_z = make_inv_z_vector<BackwardInput>(z_t);
    const float* forward_source = p.forward_source.data_ptr<float>();
    const float* adjoint_source = p.adjoint_source.data_ptr<float>();
    const int32_t* forward_sources = p.forward_sources_loc.data_ptr<int32_t>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    float* grad_vp_ptr = grad_vp.data_ptr<float>();
    float* grad_z_ptr = grad_z.data_ptr<float>();
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* dbzdz = pml[2].data_ptr<float>();
    const float* ax = pml[3].data_ptr<float>();
    const float* bx = pml[4].data_ptr<float>();
    const float* dbxdx = pml[5].data_ptr<float>();
    const auto h = spacing_for(p, 2);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float dt2 = static_cast<float>(p.dt * p.dt);

    AcousticVRZ2DRawState adj(total);
    AcousticVRZ2DRawState fwd(total);
    std::vector<float> chunk_forward(static_cast<size_t>(chunk_size) * total, 0.0f);
    std::vector<float> kappa_lambda(total, 0.0f);

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        const int start = chunk_id * chunk_size;
        const int end = std::min(static_cast<int>(nt), start + chunk_size);
        load_acoustic_vrz2d_checkpoint(p.checkpoints, chunk_id, fwd);

        for (int it = start; it < end; ++it) {
            advance_acoustic_vrz2d_cpml<Order>(
                fwd, vp, z_model, inv_z.data(), az, bz, dbzdz, ax, bx, dbxdx,
                B, nz, nx, inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
            , stencil);
            add_acoustic_vrz2d_source(fwd.u_next, forward_source, forward_sources, B, forward_nsrc, nt, it, nz, nx);
            fwd.swap();
            std::copy(fwd.u_now.begin(), fwd.u_now.end(), chunk_forward.begin() + static_cast<size_t>(it - start) * total);
        }

        for (int it = end - 1; it >= start; --it) {
            acoustic_vrz2d_adjoint_step<Order>(
                adj, vp, z_model, inv_z.data(), adjoint_source, adjoint_sources,
                az, bz, dbzdz, ax, bx, dbxdx,
                B, nz, nx, adjoint_nsrc, nt, it,
                inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
            , stencil);
            build_acoustic_vrz2d_kappa_lambda<Order>(adj.u_now, vp, z_model, kappa_lambda, total);
            const float* fwd_ptr = chunk_forward.data() + static_cast<size_t>(it - start) * total;
            accumulate_acoustic_vrz2d_model_grad<Order>(
                fwd_ptr, adj.u_now, kappa_lambda, vp, z_model, inv_z.data(),
                grad_vp_ptr, grad_z_ptr, B, nz, nx, p.abcn,
                inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
            , stencil);
        }
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_z};
    return out;
}

template <int Order>
BackwardOutput backward_acoustic_vrz2d_recursive_ckpt_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto z_t = p.models[1];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t total = B * nz * nx;
    const int64_t forward_nsrc = p.forward_sources_loc.size(1);
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t nt = static_cast<int64_t>(p.nt);
    const std::vector<int> steps = checkpoint_steps_vector(p.checkpoint_steps);
    TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ2D recursive checkpoint CPU backward expects 6 checkpoint tensors");

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_z = torch::zeros_like(z_t);

    const float* vp = vp_t.data_ptr<float>();
    const float* z_model = z_t.data_ptr<float>();
    auto inv_z = make_inv_z_vector<BackwardInput>(z_t);
    const float* forward_source = p.forward_source.data_ptr<float>();
    const float* adjoint_source = p.adjoint_source.data_ptr<float>();
    const int32_t* forward_sources = p.forward_sources_loc.data_ptr<int32_t>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    float* grad_vp_ptr = grad_vp.data_ptr<float>();
    float* grad_z_ptr = grad_z.data_ptr<float>();
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* dbzdz = pml[2].data_ptr<float>();
    const float* ax = pml[3].data_ptr<float>();
    const float* bx = pml[4].data_ptr<float>();
    const float* dbxdx = pml[5].data_ptr<float>();
    const auto h = spacing_for(p, 2);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float dt2 = static_cast<float>(p.dt * p.dt);

    AcousticVRZ2DRawState adj(total);
    std::vector<float> fwd_now(total, 0.0f);
    std::vector<float> kappa_lambda(total, 0.0f);

    auto advance_interval = [&](AcousticVRZ2DRawState& state, int start, int end) {
        for (int it = start; it < end; ++it) {
            advance_acoustic_vrz2d_cpml<Order>(
                state, vp, z_model, inv_z.data(), az, bz, dbzdz, ax, bx, dbxdx,
                B, nz, nx, inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
            , stencil);
            add_acoustic_vrz2d_source(state.u_next, forward_source, forward_sources, B, forward_nsrc, nt, it, nz, nx);
            state.swap();
        }
    };

    std::function<void(int, int, AcousticVRZ2DRawState&)> process_interval =
        [&](int start, int end, AcousticVRZ2DRawState& start_state) {
            if (start >= end) return;
            if (end - start == 1) {
                advance_acoustic_vrz2d_cpml<Order>(
                    start_state, vp, z_model, inv_z.data(), az, bz, dbzdz, ax, bx, dbxdx,
                    B, nz, nx, inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
                , stencil);
                add_acoustic_vrz2d_source(start_state.u_next, forward_source, forward_sources, B, forward_nsrc, nt, start, nz, nx);
                start_state.swap();
                std::copy(start_state.u_now.begin(), start_state.u_now.end(), fwd_now.begin());

                acoustic_vrz2d_adjoint_step<Order>(
                    adj, vp, z_model, inv_z.data(), adjoint_source, adjoint_sources,
                    az, bz, dbzdz, ax, bx, dbxdx,
                    B, nz, nx, adjoint_nsrc, nt, start,
                    inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
                , stencil);
                build_acoustic_vrz2d_kappa_lambda<Order>(adj.u_now, vp, z_model, kappa_lambda, total);
                accumulate_acoustic_vrz2d_model_grad<Order>(
                    fwd_now.data(), adj.u_now, kappa_lambda, vp, z_model, inv_z.data(),
                    grad_vp_ptr, grad_z_ptr, B, nz, nx, p.abcn,
                    inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
                , stencil);
                return;
            }

            const int mid = start + (end - start) / 2;
            AcousticVRZ2DRawState mid_state = start_state;
            advance_interval(mid_state, start, mid);
            process_interval(mid, end, mid_state);
            process_interval(start, mid, start_state);
        };

    for (int segment_idx = static_cast<int>(steps.size()); segment_idx >= 0; --segment_idx) {
        const int start = segment_idx == 0 ? 0 : steps[segment_idx - 1];
        const int end = segment_idx == static_cast<int>(steps.size()) ? static_cast<int>(nt) : steps[segment_idx];
        AcousticVRZ2DRawState start_state(total);
        if (segment_idx > 0) {
            load_acoustic_vrz2d_checkpoint(p.checkpoints, segment_idx - 1, start_state);
        }
        process_interval(start, end, start_state);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_z};
    return out;
}

template <int Order>
BackwardOutput backward_acoustic_vrz2d_raw_impl(const BackwardInput& p);

template <int Order>
BackwardOutput backward_acoustic_vrz2d_bs_impl(const BackwardInput& p)
{
    const StencilCoefficients stencil{p.M, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>()};
    [[maybe_unused]] const int M = sweep_cpu::ops::stencil_half_order<Order>(stencil);

    auto vp_t = p.models[0];
    auto z_t = p.models[1];
    const int64_t B = vp_t.size(0);
    const int64_t nz = vp_t.size(2);
    const int64_t nx = vp_t.size(3);
    const int64_t total = B * nz * nx;
    const int64_t forward_nsrc = p.forward_sources_loc.size(1);
    const int64_t adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int64_t nt = static_cast<int64_t>(p.nt);

    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0,
                "AcousticVRZ2D CPU boundary-saving backward requires u_last_two");
    if (p.boundary_on_disk) {
        TORCH_CHECK(p.boundary_disk_files.size() == 4,
                    "AcousticVRZ2D CPU disk boundary-saving backward requires 4 files");
    } else {
        TORCH_CHECK(!active_boundary_tensors(p).empty(),
                    "AcousticVRZ2D CPU boundary-saving backward requires saved boundary tensors");
    }

    auto grad_vp = torch::zeros_like(vp_t);
    auto grad_z = torch::zeros_like(z_t);

    const float* vp = vp_t.data_ptr<float>();
    const float* z_model = z_t.data_ptr<float>();
    auto inv_z = make_inv_z_vector<BackwardInput>(z_t);
    const float* forward_source = p.forward_source.data_ptr<float>();
    const float* adjoint_source = p.adjoint_source.data_ptr<float>();
    const int32_t* forward_sources = p.forward_sources_loc.data_ptr<int32_t>();
    const int32_t* adjoint_sources = p.adjoint_sources_loc.data_ptr<int32_t>();
    float* grad_vp_ptr = grad_vp.data_ptr<float>();
    float* grad_z_ptr = grad_z.data_ptr<float>();
    const auto& pml = p.pml_vals;
    const float* az = pml[0].data_ptr<float>();
    const float* bz = pml[1].data_ptr<float>();
    const float* dbzdz = pml[2].data_ptr<float>();
    const float* ax = pml[3].data_ptr<float>();
    const float* bx = pml[4].data_ptr<float>();
    const float* dbxdx = pml[5].data_ptr<float>();
    const auto h = spacing_for(p, 2);
    const float inv_dz = static_cast<float>(1.0 / h[0]);
    const float inv_dx = static_cast<float>(1.0 / h[1]);
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float dt2 = static_cast<float>(p.dt * p.dt);
    const int save_width = M + 1;

    AcousticVRZ2DRawState adj(total);
    AcousticVRZ2DRawState fwd(total);
    std::vector<float> kappa_lambda(total, 0.0f);
    auto boundary = active_boundary_tensors(p);

    copy_tensor_to_vector(p.u_last_two.select(1, 1).squeeze(0), fwd.u_prev);
    copy_tensor_to_vector(p.u_last_two.select(1, 0).squeeze(0), fwd.u_now);

    for (int64_t it = nt - 1; it >= 1; --it) {
        acoustic_vrz2d_adjoint_step<Order>(
            adj, vp, z_model, inv_z.data(), adjoint_source, adjoint_sources,
            az, bz, dbzdz, ax, bx, dbxdx,
            B, nz, nx, adjoint_nsrc, nt, it,
            inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
        , stencil);

        advance_acoustic_vrz2d_nopml<Order>(
            fwd, vp, z_model, inv_z.data(), B, nz, nx, p.abcn,
            inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
        , stencil);
        add_acoustic_vrz2d_source(fwd.u_next, forward_source, forward_sources, B, forward_nsrc, nt, it, nz, nx);
        if (p.boundary_on_disk) {
            restore_acoustic_vrz2d_boundary_disk(
                p.boundary_disk_files, static_cast<int>(it), fwd.u_next,
                B, nz, nx, M, p.abcn, p.free_surface, save_width
            );
        } else {
            restore_acoustic_vrz2d_boundary(
                boundary, static_cast<int>(it), fwd.u_next,
                B, nz, nx, M, p.abcn, p.free_surface, save_width
            );
        }
        fwd.swap();

        build_acoustic_vrz2d_kappa_lambda<Order>(adj.u_now, vp, z_model, kappa_lambda, total);
        accumulate_acoustic_vrz2d_model_grad<Order>(
            fwd.u_now.data(), adj.u_now, kappa_lambda, vp, z_model, inv_z.data(),
            grad_vp_ptr, grad_z_ptr, B, nz, nx, p.abcn,
            inv_dz, inv_dx, inv_dz2, inv_dx2, dt2
        , stencil);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_z};
    return out;
}

template <int Order>
BackwardOutput backward_acoustic_vrz2d_raw_impl(const BackwardInput& p)
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
    auto forward_result = forward_acoustic_vrz2d_raw(fwd);

    BackwardInput replay = p;
    replay.u_forward = forward_result.wavefield;
    return backward_acoustic_vrz2d_full_impl<Order>(replay);
}

BackwardOutput backward_acoustic_vrz2d_raw(const BackwardInput& p)
{
    const bool has_full_wavefield = p.u_forward.defined() && p.u_forward.numel() > 0;
    const bool has_boundary_state = p.u_last_two.defined() && p.u_last_two.numel() > 0;
    const bool has_checkpoints = !p.checkpoints.empty();
    const bool has_recursive_steps = p.checkpoint_steps.defined() && p.checkpoint_steps.numel() > 0;

    #define SWEEP_CPU_VRZ2D_BACKWARD_CASE(ORDER) \
        if (has_full_wavefield) return backward_acoustic_vrz2d_full_impl<ORDER>(p); \
        if (has_boundary_state) return backward_acoustic_vrz2d_bs_impl<ORDER>(p); \
        if (has_checkpoints && has_recursive_steps) return backward_acoustic_vrz2d_recursive_ckpt_impl<ORDER>(p); \
        if (has_checkpoints) return backward_acoustic_vrz2d_ckpt_impl<ORDER>(p); \
        return backward_acoustic_vrz2d_raw_impl<ORDER>(p);
    SWEEP_CPU_DISPATCH_STENCIL_BODY(p.M, SWEEP_CPU_VRZ2D_BACKWARD_CASE);
    #undef SWEEP_CPU_VRZ2D_BACKWARD_CASE
}

} // namespace


ForwardOutput forward(const ForwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::acoustic_vrz2d::forward called with non-CPU tensors");
    if (!can_use_acoustic_vrz2d_raw_forward(in)) {
        return engine::forward(in, EquationKind::AcousticVRZ2D);
    }

    auto result = forward_acoustic_vrz2d_raw(in);
    ForwardOutput out;
    out.wavefield = result.wavefield.defined() ? result.wavefield : torch::empty({0}, in.models[0].options());
    out.last_two = result.last_two.defined() ? result.last_two : torch::empty({0}, in.models[0].options());
    out.record = result.record;
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    TORCH_CHECK(engine::is_cpu_input(in), "sweep_cpu::acoustic_vrz2d::backward called with non-CPU tensors");
    TORCH_CHECK(
        can_use_acoustic_vrz2d_raw_backward(in),
        "AcousticVRZ2D CPU backward requires the handwritten raw float32 path; "
        "unsupported inputs will not fall back to torch autograd."
    );
    return backward_acoustic_vrz2d_raw(in);
}

BackwardOutput backward_bs(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_ckpt(const BackwardInput& in) { return backward(in); }
BackwardOutput backward_recursive_ckpt(const BackwardInput& in) { return backward(in); }

} // namespace sweep_cpu::acoustic_vrz2d

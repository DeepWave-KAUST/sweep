#include <torch/extension.h>
#include "cuda/equations/acoustic2d/acoustic2d.h"
#include "cuda/equations/acoustic_lsrtm2d/acoustic_lsrtm2d.h"
#include "cuda/equations/acoustic_lsrtm3d/acoustic_lsrtm3d.h"
#include "cuda/equations/acoustic_vrz2d/acoustic_vrz2d.h"
#include "cuda/equations/acoustic_vrz3d/acoustic_vrz3d.h"
#include "cuda/equations/acoustic3d/acoustic3d.h"
#include "cuda/equations/das2d/das2d.h"
#include "cuda/equations/das3d/das3d.h"
#include "cuda/equations/elastic2d/elastic2d.h"
#include "cuda/equations/elastic3d/elastic3d.h"
#include "cpu/cpu_binding.h"
#include "bindings_utils.h"

// Some CUDA 12.x / libstdc++ header combinations emit a reference to this
// glibc 2.32 symbol even when building on older glibc hosts. Defining the weak
// fallback as 0 keeps libstdc++ on the conservative multi-threaded path.
extern "C" {
__attribute__((weak)) char __libc_single_threaded = 0;
}

template <typename Func>
auto dispatch_forward(Func cuda_func, sweep_cpu::EquationKind kind)
{
    return [cuda_func, kind](const ForwardInput& in) {
        if (sweep_cpu::is_cpu_input(in)) {
            return sweep_cpu::forward(in, kind);
        }
        return cuda_func(in);
    };
}

template <typename Func>
auto dispatch_backward(Func cuda_func, sweep_cpu::EquationKind kind, sweep_cpu::BackwardMode mode)
{
    return [cuda_func, kind, mode](const BackwardInput& in) {
        if (sweep_cpu::is_cpu_input(in)) {
            return sweep_cpu::backward(in, kind, mode);
        }
        return cuda_func(in);
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using EK = sweep_cpu::EquationKind;
    using BM = sweep_cpu::BackwardMode;
    m.def("acoustic2d_forward", wrap_forward(dispatch_forward(acoustic2d::forward, EK::Acoustic2D)));
    m.def("acoustic2d_backward", wrap_backward(dispatch_backward(acoustic2d::backward, EK::Acoustic2D, BM::Full)), "Acoustic backward (CUDA/CPU)");
    m.def("acoustic2d_backward_bs", wrap_backward(dispatch_backward(acoustic2d::backward_bs, EK::Acoustic2D, BM::BoundarySaving)), "Acoustic backward with boundary saving (CUDA/CPU)");
    m.def("acoustic2d_backward_ckpt", wrap_backward(dispatch_backward(acoustic2d::backward_ckpt, EK::Acoustic2D, BM::Checkpoint)), "Acoustic backward with checkpointing (CUDA/CPU)");
    m.def("acoustic2d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic2d::backward_recursive_ckpt, EK::Acoustic2D, BM::RecursiveCheckpoint)), "Acoustic backward with recursive checkpointing (CUDA/CPU)");
    m.def("acoustic2d_rtm", wrap_rtm(acoustic2d::rtm), "Acoustic RTM 2D (CUDA)");
    m.def("acoustic_lsrtm2d_forward", wrap_forward(dispatch_forward(acoustic_lsrtm2d::forward, EK::AcousticLSRTM2D)), "Acoustic LSRTM forward 2D (CUDA/CPU)");
    m.def("acoustic_lsrtm2d_backward", wrap_backward(dispatch_backward(acoustic_lsrtm2d::backward, EK::AcousticLSRTM2D, BM::Full)), "Acoustic LSRTM backward 2D (CUDA/CPU)");
    m.def("acoustic_lsrtm2d_backward_bs", wrap_backward(dispatch_backward(acoustic_lsrtm2d::backward_bs, EK::AcousticLSRTM2D, BM::BoundarySaving)), "Acoustic LSRTM backward with boundary saving 2D (CUDA/CPU)");
    m.def("acoustic_lsrtm2d_backward_ckpt", wrap_backward(dispatch_backward(acoustic_lsrtm2d::backward_ckpt, EK::AcousticLSRTM2D, BM::Checkpoint)), "Acoustic LSRTM backward with checkpointing 2D (CUDA/CPU)");
    m.def("acoustic_lsrtm2d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic_lsrtm2d::backward_recursive_ckpt, EK::AcousticLSRTM2D, BM::RecursiveCheckpoint)), "Acoustic LSRTM backward with recursive checkpointing 2D (CUDA/CPU)");
    m.def("acoustic_lsrtm3d_forward", wrap_forward(dispatch_forward(acoustic_lsrtm3d::forward, EK::AcousticLSRTM3D)), "Acoustic LSRTM forward 3D (CUDA/CPU)");
    m.def("acoustic_lsrtm3d_backward", wrap_backward(dispatch_backward(acoustic_lsrtm3d::backward, EK::AcousticLSRTM3D, BM::Full)), "Acoustic LSRTM backward 3D (CUDA/CPU)");
    m.def("acoustic_lsrtm3d_backward_bs", wrap_backward(dispatch_backward(acoustic_lsrtm3d::backward_bs, EK::AcousticLSRTM3D, BM::BoundarySaving)), "Acoustic LSRTM backward with boundary saving 3D (CUDA/CPU)");
    m.def("acoustic_lsrtm3d_backward_ckpt", wrap_backward(dispatch_backward(acoustic_lsrtm3d::backward_ckpt, EK::AcousticLSRTM3D, BM::Checkpoint)), "Acoustic LSRTM backward with checkpointing 3D (CUDA/CPU)");
    m.def("acoustic_lsrtm3d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic_lsrtm3d::backward_recursive_ckpt, EK::AcousticLSRTM3D, BM::RecursiveCheckpoint)), "Acoustic LSRTM backward with recursive checkpointing 3D (CUDA/CPU)");
    m.def("acoustic_vrz2d_forward", wrap_forward(dispatch_forward(acoustic_vrz2d::forward, EK::AcousticVRZ2D)), "Acoustic VRZ forward 2D (CUDA/CPU)");
    m.def("acoustic_vrz2d_backward", wrap_backward(dispatch_backward(acoustic_vrz2d::backward, EK::AcousticVRZ2D, BM::Full)), "Acoustic VRZ backward 2D (CUDA/CPU)");
    m.def("acoustic_vrz2d_backward_bs", wrap_backward(dispatch_backward(acoustic_vrz2d::backward_bs, EK::AcousticVRZ2D, BM::BoundarySaving)), "Acoustic VRZ backward with boundary saving 2D (CUDA/CPU)");
    m.def("acoustic_vrz2d_backward_ckpt", wrap_backward(dispatch_backward(acoustic_vrz2d::backward_ckpt, EK::AcousticVRZ2D, BM::Checkpoint)), "Acoustic VRZ backward with checkpointing 2D (CUDA/CPU)");
    m.def("acoustic_vrz2d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic_vrz2d::backward_recursive_ckpt, EK::AcousticVRZ2D, BM::RecursiveCheckpoint)), "Acoustic VRZ backward with recursive checkpointing 2D (CUDA/CPU)");
    m.def("acoustic_vrz3d_forward", wrap_forward(dispatch_forward(acoustic_vrz3d::forward, EK::AcousticVRZ3D)), "Acoustic VRZ forward 3D (CUDA/CPU)");
    m.def("acoustic_vrz3d_backward", wrap_backward(dispatch_backward(acoustic_vrz3d::backward, EK::AcousticVRZ3D, BM::Full)), "Acoustic VRZ backward 3D (CUDA/CPU)");
    m.def("acoustic_vrz3d_backward_bs", wrap_backward(dispatch_backward(acoustic_vrz3d::backward_bs, EK::AcousticVRZ3D, BM::BoundarySaving)), "Acoustic VRZ backward with boundary saving 3D (CUDA/CPU)");
    m.def("acoustic_vrz3d_backward_ckpt", wrap_backward(dispatch_backward(acoustic_vrz3d::backward_ckpt, EK::AcousticVRZ3D, BM::Checkpoint)), "Acoustic VRZ backward with checkpointing 3D (CUDA/CPU)");
    m.def("acoustic_vrz3d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic_vrz3d::backward_recursive_ckpt, EK::AcousticVRZ3D, BM::RecursiveCheckpoint)), "Acoustic VRZ backward with recursive checkpointing 3D (CUDA/CPU)");
    m.def("acoustic3d_forward", wrap_forward(dispatch_forward(acoustic3d::forward, EK::Acoustic3D)), "Acoustic forward 3D (CUDA/CPU)");
    m.def("acoustic3d_backward", wrap_backward(dispatch_backward(acoustic3d::backward, EK::Acoustic3D, BM::Full)), "Acoustic backward 3D (CUDA/CPU)");
    m.def("acoustic3d_backward_bs", wrap_backward(dispatch_backward(acoustic3d::backward_bs, EK::Acoustic3D, BM::BoundarySaving)), "Acoustic backward with boundary saving 3D (CUDA/CPU)");
    m.def("acoustic3d_backward_ckpt", wrap_backward(dispatch_backward(acoustic3d::backward_ckpt, EK::Acoustic3D, BM::Checkpoint)), "Acoustic backward with checkpointing 3D (CUDA/CPU)");
    m.def("acoustic3d_backward_recursive_ckpt", wrap_backward(dispatch_backward(acoustic3d::backward_recursive_ckpt, EK::Acoustic3D, BM::RecursiveCheckpoint)), "Acoustic backward with recursive checkpointing 3D (CUDA/CPU)");
    m.def("acoustic3d_rtm", wrap_rtm(acoustic3d::rtm), "Acoustic RTM 3D (CUDA)");
    m.def("elastic2d_forward", wrap_forward(dispatch_forward(elastic2d::forward, EK::Elastic2D)), "Elastic forward 2D (CUDA/CPU)");
    m.def("elastic2d_backward", wrap_backward(dispatch_backward(elastic2d::backward, EK::Elastic2D, BM::Full)), "Elastic backward 2D (CUDA/CPU)");
    m.def("elastic2d_backward_bs", wrap_backward(dispatch_backward(elastic2d::backward_bs, EK::Elastic2D, BM::BoundarySaving)), "Elastic backward with boundary saving 2D (CUDA/CPU)");
    m.def("elastic2d_backward_ckpt", wrap_backward(dispatch_backward(elastic2d::backward_ckpt, EK::Elastic2D, BM::Checkpoint)), "Elastic backward with checkpointing 2D (CUDA/CPU)");
    m.def("elastic2d_backward_recursive_ckpt", wrap_backward(dispatch_backward(elastic2d::backward_recursive_ckpt, EK::Elastic2D, BM::RecursiveCheckpoint)), "Elastic backward with recursive checkpointing 2D (CUDA/CPU)");
    m.def("elastic3d_forward", wrap_forward(dispatch_forward(elastic3d::forward, EK::Elastic3D)), "Elastic forward 3D (CUDA/CPU)");
    m.def("elastic3d_backward_bs", wrap_backward(dispatch_backward(elastic3d::backward_bs, EK::Elastic3D, BM::BoundarySaving)), "Elastic backward with boundary saving 3D (CUDA/CPU)");
    m.def("elastic3d_backward_ckpt", wrap_backward(dispatch_backward(elastic3d::backward_ckpt, EK::Elastic3D, BM::Checkpoint)), "Elastic backward with checkpointing 3D (CUDA/CPU)");
    m.def("elastic3d_backward_recursive_ckpt", wrap_backward(dispatch_backward(elastic3d::backward_recursive_ckpt, EK::Elastic3D, BM::RecursiveCheckpoint)), "Elastic backward with recursive checkpointing 3D (CUDA/CPU)");
    m.def("elastic3d_backward", wrap_backward(dispatch_backward(elastic3d::backward, EK::Elastic3D, BM::Full)), "Elastic backward 3D (CUDA/CPU)");
    m.def("das2d_forward", wrap_forward(dispatch_forward(das2d::forward, EK::DAS2D)), "DAS forward 2D (CUDA/CPU)");
    m.def("das2d_backward", wrap_backward(dispatch_backward(das2d::backward, EK::DAS2D, BM::Full)), "DAS backward 2D (CUDA/CPU)");
    m.def("das2d_backward_bs", wrap_backward(dispatch_backward(das2d::backward_bs, EK::DAS2D, BM::BoundarySaving)), "DAS backward with boundary saving 2D (CUDA/CPU)");
    m.def("das2d_backward_ckpt", wrap_backward(dispatch_backward(das2d::backward_ckpt, EK::DAS2D, BM::Checkpoint)), "DAS backward with checkpointing 2D (CUDA/CPU)");
    m.def("das2d_backward_recursive_ckpt", wrap_backward(dispatch_backward(das2d::backward_recursive_ckpt, EK::DAS2D, BM::RecursiveCheckpoint)), "DAS backward with recursive checkpointing 2D (CUDA/CPU)");
    m.def("das3d_forward", wrap_forward(dispatch_forward(das3d::forward, EK::DAS3D)), "DAS forward 3D (CUDA/CPU)");
    m.def("das3d_backward", wrap_backward(dispatch_backward(das3d::backward, EK::DAS3D, BM::Full)), "DAS backward 3D (CUDA/CPU)");
    m.def("das3d_backward_bs", wrap_backward(dispatch_backward(das3d::backward_bs, EK::DAS3D, BM::BoundarySaving)), "DAS backward with boundary saving 3D (CUDA/CPU)");
    m.def("das3d_backward_ckpt", wrap_backward(dispatch_backward(das3d::backward_ckpt, EK::DAS3D, BM::Checkpoint)), "DAS backward with checkpointing 3D (CUDA/CPU)");
    m.def("das3d_backward_recursive_ckpt", wrap_backward(dispatch_backward(das3d::backward_recursive_ckpt, EK::DAS3D, BM::RecursiveCheckpoint)), "DAS backward with recursive checkpointing 3D (CUDA/CPU)");

    py::class_<ForwardInput>(m, "ForwardInput")
        .def(py::init<>())
        .def_readwrite("models", &ForwardInput::models)
        .def_readwrite("source", &ForwardInput::source)
        .def_readwrite("lap_coes", &ForwardInput::lap_coes)
        .def_readwrite("grad_coes", &ForwardInput::grad_coes)
        .def_readwrite("M", &ForwardInput::M)
        .def_readwrite("abcn", &ForwardInput::abcn)
        .def_readwrite("sources_loc", &ForwardInput::sources_loc)
        .def_readwrite("receivers_loc", &ForwardInput::receivers_loc)
        .def_readwrite("source_field_indices", &ForwardInput::source_field_indices)
        .def_readwrite("receiver_field_indices", &ForwardInput::receiver_field_indices)
        .def_readwrite("pml_vals", &ForwardInput::pml_vals)
        .def_readwrite("last_two", &ForwardInput::last_two)
        .def_readwrite("save_all_wavefields", &ForwardInput::save_all_wavefields)
        .def_readwrite("use_boundary_saving", &ForwardInput::use_boundary_saving)
        .def_readwrite("use_checkpoint", &ForwardInput::use_checkpoint)
        .def_readwrite("use_recursive_checkpoint", &ForwardInput::use_recursive_checkpoint)
        .def_readwrite("checkpoint_on_cpu", &ForwardInput::checkpoint_on_cpu)
        .def_readwrite("boundary_on_cpu", &ForwardInput::boundary_on_cpu)
        .def_readwrite("boundary_on_disk", &ForwardInput::boundary_on_disk)
        .def_readwrite("boundary_disk_async_read", &ForwardInput::boundary_disk_async_read)
        .def_readwrite("use_pinned_memory", &ForwardInput::use_pinned_memory)
        .def_readwrite("free_surface", &ForwardInput::free_surface)
        .def_readwrite("nt", &ForwardInput::nt)
        .def_readwrite("dt", &ForwardInput::dt)
        .def_readwrite("spacing", &ForwardInput::spacing)
        .def_readwrite("transfer_interval", &ForwardInput::transfer_interval)
        .def_readwrite("boundary_ring_buffers", &ForwardInput::boundary_ring_buffers)
        .def_readwrite("checkpoint_interval", &ForwardInput::checkpoint_interval)
        .def_readwrite("checkpoint_count", &ForwardInput::checkpoint_count)
        .def_readwrite("wavefields", &ForwardInput::wavefields)
        .def_readwrite("boundary_cpu", &ForwardInput::boundary_cpu)
        .def_readwrite("boundary_gpu", &ForwardInput::boundary_gpu)
        .def_readwrite("boundary_disk_files", &ForwardInput::boundary_disk_files)
        .def_readwrite("checkpoints", &ForwardInput::checkpoints)
        .def_readwrite("checkpoint_steps", &ForwardInput::checkpoint_steps);

    py::class_<BackwardInput>(m, "BackwardInput")
        .def(py::init<>())
        .def_readwrite("u_forward", &BackwardInput::u_forward)
        .def_readwrite("u_boundary", &BackwardInput::u_boundary)
        .def_readwrite("u_last_two", &BackwardInput::u_last_two)
        .def_readwrite("checkpoints", &BackwardInput::checkpoints)
        .def_readwrite("models", &BackwardInput::models)
        .def_readwrite("adjoint_source", &BackwardInput::adjoint_source)
        .def_readwrite("forward_source", &BackwardInput::forward_source)
        .def_readwrite("lap_coes", &BackwardInput::lap_coes)
        .def_readwrite("grad_coes", &BackwardInput::grad_coes)
        .def_readwrite("M", &BackwardInput::M)
        .def_readwrite("abcn", &BackwardInput::abcn)
        .def_readwrite("adjoint_sources_loc", &BackwardInput::adjoint_sources_loc)
        .def_readwrite("forward_sources_loc", &BackwardInput::forward_sources_loc)
        .def_readwrite("source_field_indices", &BackwardInput::source_field_indices)
        .def_readwrite("receiver_field_indices", &BackwardInput::receiver_field_indices)
        .def_readwrite("pml_vals", &BackwardInput::pml_vals)
        .def_readwrite("nt", &BackwardInput::nt)
        .def_readwrite("dt", &BackwardInput::dt)
        .def_readwrite("spacing", &BackwardInput::spacing)
        .def_readwrite("free_surface", &BackwardInput::free_surface)
        .def_readwrite("checkpoint_on_cpu", &BackwardInput::checkpoint_on_cpu)
        .def_readwrite("boundary_on_cpu", &BackwardInput::boundary_on_cpu)
        .def_readwrite("boundary_on_disk", &BackwardInput::boundary_on_disk)
        .def_readwrite("boundary_disk_async_read", &BackwardInput::boundary_disk_async_read)
        .def_readwrite("use_pinned_memory", &BackwardInput::use_pinned_memory)
        .def_readwrite("transfer_interval", &BackwardInput::transfer_interval)
        .def_readwrite("boundary_ring_buffers", &BackwardInput::boundary_ring_buffers)
        .def_readwrite("checkpoint_interval", &BackwardInput::checkpoint_interval)
        .def_readwrite("checkpoint_count", &BackwardInput::checkpoint_count)
        .def_readwrite("forward_wavefields", &BackwardInput::forward_wavefields)
        .def_readwrite("adjoint_wavefields", &BackwardInput::adjoint_wavefields)
        .def_readwrite("adjoint_workspace", &BackwardInput::adjoint_workspace)
        .def_readwrite("boundary_cpu", &BackwardInput::boundary_cpu)
        .def_readwrite("boundary_gpu", &BackwardInput::boundary_gpu)
        .def_readwrite("boundary_disk_files", &BackwardInput::boundary_disk_files)
        .def_readwrite("checkpoint_steps", &BackwardInput::checkpoint_steps);


}

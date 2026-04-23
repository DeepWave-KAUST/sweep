#include <torch/extension.h>
#include "equations/acoustic2d/acoustic2d.h"
#include "equations/acoustic_lsrtm2d/acoustic_lsrtm2d.h"
#include "equations/acoustic_lsrtm3d/acoustic_lsrtm3d.h"
#include "equations/acoustic_vrz2d/acoustic_vrz2d.h"
#include "equations/acoustic3d/acoustic3d.h"
#include "equations/elastic2d/elastic2d.h"
#include "equations/elastic3d/elastic3d.h"
#include "bindings_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("acoustic2d_forward", wrap_forward(acoustic2d::forward));
    m.def("acoustic2d_backward", wrap_backward(acoustic2d::backward), "Acoustic backward (CUDA)");
    m.def("acoustic2d_backward_bs", wrap_backward(acoustic2d::backward_bs), "Acoustic backward with boundary saving (CUDA)");
    m.def("acoustic2d_backward_ckpt", wrap_backward(acoustic2d::backward_ckpt), "Acoustic backward with checkpointing (CUDA)");
    m.def("acoustic2d_backward_recursive_ckpt", wrap_backward(acoustic2d::backward_recursive_ckpt), "Acoustic backward with recursive checkpointing (CUDA)");
    m.def("acoustic2d_rtm", wrap_rtm(acoustic2d::rtm), "Acoustic RTM 2D (CUDA)");
    m.def("acoustic_lsrtm2d_forward", wrap_forward(acoustic_lsrtm2d::forward), "Acoustic LSRTM forward 2D (CUDA)");
    m.def("acoustic_lsrtm2d_backward", wrap_backward(acoustic_lsrtm2d::backward), "Acoustic LSRTM backward 2D (CUDA)");
    m.def("acoustic_lsrtm2d_backward_bs", wrap_backward(acoustic_lsrtm2d::backward_bs), "Acoustic LSRTM backward with boundary saving 2D (CUDA)");
    m.def("acoustic_lsrtm2d_backward_ckpt", wrap_backward(acoustic_lsrtm2d::backward_ckpt), "Acoustic LSRTM backward with checkpointing 2D (CUDA)");
    m.def("acoustic_lsrtm2d_backward_recursive_ckpt", wrap_backward(acoustic_lsrtm2d::backward_recursive_ckpt), "Acoustic LSRTM backward with recursive checkpointing 2D (CUDA)");
    m.def("acoustic_lsrtm3d_forward", wrap_forward(acoustic_lsrtm3d::forward), "Acoustic LSRTM forward 3D (CUDA)");
    m.def("acoustic_lsrtm3d_backward", wrap_backward(acoustic_lsrtm3d::backward), "Acoustic LSRTM backward 3D (CUDA)");
    m.def("acoustic_lsrtm3d_backward_bs", wrap_backward(acoustic_lsrtm3d::backward_bs), "Acoustic LSRTM backward with boundary saving 3D (CUDA)");
    m.def("acoustic_lsrtm3d_backward_ckpt", wrap_backward(acoustic_lsrtm3d::backward_ckpt), "Acoustic LSRTM backward with checkpointing 3D (CUDA)");
    m.def("acoustic_lsrtm3d_backward_recursive_ckpt", wrap_backward(acoustic_lsrtm3d::backward_recursive_ckpt), "Acoustic LSRTM backward with recursive checkpointing 3D (CUDA)");
    m.def("acoustic_vrz2d_forward", wrap_forward(acoustic_vrz2d::forward), "Acoustic VRZ forward 2D (CUDA)");
    m.def("acoustic_vrz2d_backward", wrap_backward(acoustic_vrz2d::backward), "Acoustic VRZ backward 2D (CUDA)");
    m.def("acoustic_vrz2d_backward_bs", wrap_backward(acoustic_vrz2d::backward_bs), "Acoustic VRZ backward with boundary saving 2D (CUDA)");
    m.def("acoustic_vrz2d_backward_ckpt", wrap_backward(acoustic_vrz2d::backward_ckpt), "Acoustic VRZ backward with checkpointing 2D (CUDA)");
    m.def("acoustic_vrz2d_backward_recursive_ckpt", wrap_backward(acoustic_vrz2d::backward_recursive_ckpt), "Acoustic VRZ backward with recursive checkpointing 2D (CUDA)");
    m.def("acoustic3d_forward", wrap_forward(acoustic3d::forward), "Acoustic forward 3D (CUDA)");
    m.def("acoustic3d_backward", wrap_backward(acoustic3d::backward), "Acoustic backward 3D (CUDA)");
    m.def("acoustic3d_backward_bs", wrap_backward(acoustic3d::backward_bs), "Acoustic backward with boundary saving 3D (CUDA)");
    m.def("acoustic3d_backward_ckpt", wrap_backward(acoustic3d::backward_ckpt), "Acoustic backward with checkpointing 3D (CUDA)");
    m.def("acoustic3d_backward_recursive_ckpt", wrap_backward(acoustic3d::backward_recursive_ckpt), "Acoustic backward with recursive checkpointing 3D (CUDA)");
    m.def("acoustic3d_rtm", wrap_rtm(acoustic3d::rtm), "Acoustic RTM 3D (CUDA)");
    m.def("elastic2d_forward", wrap_forward(elastic2d::forward), "Elastic forward 2D (CUDA)");
    m.def("elastic2d_backward", wrap_backward(elastic2d::backward), "Elastic backward 2D (CUDA)");
    m.def("elastic2d_backward_bs", wrap_backward(elastic2d::backward_bs), "Elastic backward with boundary saving 2D (CUDA)");
    m.def("elastic2d_backward_ckpt", wrap_backward(elastic2d::backward_ckpt), "Elastic backward with checkpointing 2D (CUDA)");
    m.def("elastic2d_backward_recursive_ckpt", wrap_backward(elastic2d::backward_recursive_ckpt), "Elastic backward with recursive checkpointing 2D (CUDA)");
    m.def("elastic3d_forward", wrap_forward(elastic3d::forward), "Elastic forward 3D (CUDA)");
    m.def("elastic3d_backward_bs", wrap_backward(elastic3d::backward_bs), "Elastic backward with boundary saving 3D (CUDA)");
    m.def("elastic3d_backward_ckpt", wrap_backward(elastic3d::backward_ckpt), "Elastic backward with checkpointing 3D (CUDA)");
    m.def("elastic3d_backward_recursive_ckpt", wrap_backward(elastic3d::backward_recursive_ckpt), "Elastic backward with recursive checkpointing 3D (CUDA)");
    m.def("elastic3d_backward", wrap_backward(elastic3d::backward), "Elastic backward 3D (CUDA)");
    
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
        .def_readwrite("boundary_on_cpu", &ForwardInput::boundary_on_cpu)
        .def_readwrite("use_pinned_memory", &ForwardInput::use_pinned_memory)
        .def_readwrite("free_surface", &ForwardInput::free_surface)
        .def_readwrite("nt", &ForwardInput::nt)
        .def_readwrite("dt", &ForwardInput::dt)
        .def_readwrite("spacing", &ForwardInput::spacing)
        .def_readwrite("transfer_interval", &ForwardInput::transfer_interval)
        .def_readwrite("checkpoint_interval", &ForwardInput::checkpoint_interval)
        .def_readwrite("checkpoint_count", &ForwardInput::checkpoint_count)
        .def_readwrite("wavefields", &ForwardInput::wavefields)
        .def_readwrite("boundary_cpu", &ForwardInput::boundary_cpu)
        .def_readwrite("boundary_gpu", &ForwardInput::boundary_gpu)
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
        .def_readwrite("boundary_on_cpu", &BackwardInput::boundary_on_cpu)
        .def_readwrite("use_pinned_memory", &BackwardInput::use_pinned_memory)
        .def_readwrite("transfer_interval", &BackwardInput::transfer_interval)
        .def_readwrite("checkpoint_interval", &BackwardInput::checkpoint_interval)
        .def_readwrite("checkpoint_count", &BackwardInput::checkpoint_count)
        .def_readwrite("forward_wavefields", &BackwardInput::forward_wavefields)
        .def_readwrite("adjoint_wavefields", &BackwardInput::adjoint_wavefields)
        .def_readwrite("adjoint_workspace", &BackwardInput::adjoint_workspace)
        .def_readwrite("boundary_cpu", &BackwardInput::boundary_cpu)
        .def_readwrite("boundary_gpu", &BackwardInput::boundary_gpu)
        .def_readwrite("checkpoint_steps", &BackwardInput::checkpoint_steps);


}

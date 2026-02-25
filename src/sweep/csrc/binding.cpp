#include <torch/extension.h>
#include "equations/acoustic2d/acoustic2d.h"
#include "equations/acoustic3d/acoustic3d.h"
#include "equations/elastic2d/elastic2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("acoustic_forward", &acoustic_forward_cuda, "Acoustic forward (CUDA)");
    m.def("acoustic_backward", &acoustic_backward_cuda, "Acoustic backward (CUDA)");
    m.def("acoustic_backward_bs", &acoustic_backward_boundary_saving_cuda, "Acoustic backward with boundary saving (CUDA)");
    m.def("acoustic_forward3d", &acoustic_forward3d_cuda, "Acoustic forward 3D (CUDA)");
    m.def("acoustic_backward3d", &acoustic_backward3d_cuda, "Acoustic backward 3D (CUDA)");
    m.def("acoustic_backward3d_bs", &acoustic_backward3d_boundary_saving_cuda, "Acoustic backward with boundary saving 3D (CUDA)");
    m.def("elastic_forward", &elastic_forward_cuda, "Elastic forward (CUDA)");
    m.def("elastic_backward", &elastic_backward_cuda, "Elastic backward with boundary saving (CUDA)");
    m.def("elastic_backward_bs", &elastic_backward_boundary_saving_cuda, "Elastic backward with boundary saving (CUDA)");

}
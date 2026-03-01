#include <torch/extension.h>
#include "equations/acoustic2d/acoustic2d.h"
#include "equations/acoustic3d/acoustic3d.h"
#include "equations/elastic2d/elastic2d.h"
#include "equations/elastic3d/elastic3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("acoustic2d_forward", &acoustic2d::forward, "Acoustic forward 2D (CUDA)");
    m.def("acoustic2d_backward", &acoustic2d::backward, "Acoustic backward (CUDA)");
    m.def("acoustic2d_backward_bs", &acoustic2d::backward_bs, "Acoustic backward with boundary saving (CUDA)");
    m.def("acoustic3d_forward", &acoustic3d::forward, "Acoustic forward 3D (CUDA)");
    m.def("acoustic3d_backward", &acoustic3d::backward, "Acoustic backward 3D (CUDA)");
    m.def("acoustic3d_backward_bs", &acoustic3d::backward_bs, "Acoustic backward with boundary saving 3D (CUDA)");
    m.def("elastic2d_forward", &elastic2d::forward, "Elastic forward 2D (CUDA)");
    m.def("elastic2d_backward", &elastic2d::backward, "Elastic backward 2D (CUDA)");
    m.def("elastic2d_backward_bs", &elastic2d::backward_bs, "Elastic backward with boundary saving 2D (CUDA)");
    m.def("elastic3d_forward", &elastic3d::forward, "Elastic forward 3D (CUDA)");
    m.def("elastic3d_backward_bs", &elastic3d::backward_bs, "Elastic backward with boundary saving 3D (CUDA)");
}
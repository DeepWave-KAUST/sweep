#pragma once

#include <torch/extension.h>

#include "kernels.cuh"

namespace elastic_tti_sg3d {

inline StiffnessPointer stiffness_view(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == 22, "ElasticTTISG3D CUDA expects prepared models: rho plus 21 stiffness tensors");
    StiffnessPointer out{};
    int i = 0;
    out.rho = models[i++].data_ptr<float>();
    out.C11 = models[i++].data_ptr<float>();
    out.C12 = models[i++].data_ptr<float>();
    out.C13 = models[i++].data_ptr<float>();
    out.C14 = models[i++].data_ptr<float>();
    out.C15 = models[i++].data_ptr<float>();
    out.C16 = models[i++].data_ptr<float>();
    out.C22 = models[i++].data_ptr<float>();
    out.C23 = models[i++].data_ptr<float>();
    out.C24 = models[i++].data_ptr<float>();
    out.C25 = models[i++].data_ptr<float>();
    out.C26 = models[i++].data_ptr<float>();
    out.C33 = models[i++].data_ptr<float>();
    out.C34 = models[i++].data_ptr<float>();
    out.C35 = models[i++].data_ptr<float>();
    out.C36 = models[i++].data_ptr<float>();
    out.C44 = models[i++].data_ptr<float>();
    out.C45 = models[i++].data_ptr<float>();
    out.C46 = models[i++].data_ptr<float>();
    out.C55 = models[i++].data_ptr<float>();
    out.C56 = models[i++].data_ptr<float>();
    out.C66 = models[i++].data_ptr<float>();
    return out;
}

inline std::vector<torch::Tensor> zero_model_grads(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == 22, "ElasticTTISG3D CUDA backward expects 22 prepared models");
    std::vector<torch::Tensor> grads;
    grads.reserve(models.size());
    for (const auto& model : models)
        grads.push_back(torch::zeros_like(model));
    return grads;
}

inline StiffnessGradPointer stiffness_grad_view(std::vector<torch::Tensor>& grads)
{
    TORCH_CHECK(grads.size() == 22, "ElasticTTISG3D CUDA backward expects 22 prepared model gradients");
    StiffnessGradPointer out{};
    int i = 0;
    out.rho = grads[i++].data_ptr<float>();
    out.C11 = grads[i++].data_ptr<float>();
    out.C12 = grads[i++].data_ptr<float>();
    out.C13 = grads[i++].data_ptr<float>();
    out.C14 = grads[i++].data_ptr<float>();
    out.C15 = grads[i++].data_ptr<float>();
    out.C16 = grads[i++].data_ptr<float>();
    out.C22 = grads[i++].data_ptr<float>();
    out.C23 = grads[i++].data_ptr<float>();
    out.C24 = grads[i++].data_ptr<float>();
    out.C25 = grads[i++].data_ptr<float>();
    out.C26 = grads[i++].data_ptr<float>();
    out.C33 = grads[i++].data_ptr<float>();
    out.C34 = grads[i++].data_ptr<float>();
    out.C35 = grads[i++].data_ptr<float>();
    out.C36 = grads[i++].data_ptr<float>();
    out.C44 = grads[i++].data_ptr<float>();
    out.C45 = grads[i++].data_ptr<float>();
    out.C46 = grads[i++].data_ptr<float>();
    out.C55 = grads[i++].data_ptr<float>();
    out.C56 = grads[i++].data_ptr<float>();
    out.C66 = grads[i++].data_ptr<float>();
    return out;
}

} // namespace elastic_tti_sg3d

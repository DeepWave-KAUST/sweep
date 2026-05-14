#pragma once

#include <torch/extension.h>

#include "kernels.cuh"

namespace elastic_tti_sg2d {

struct WavefieldTensor {
    torch::Tensor vx_t, vy_t, vz_t, sxx_t, szz_t, syz_t, sxz_t, sxy_t;
    torch::Tensor m_vxx_t, m_vxz_t, m_vyx_t, m_vyz_t, m_vzx_t, m_vzz_t;
    torch::Tensor m_txxx_t, m_txzz_t, m_txyx_t, m_tyzz_t, m_txzx_t, m_tzzz_t;

    void allocate(const torch::Tensor& like)
    {
        vx_t = torch::zeros_like(like);
        vy_t = torch::zeros_like(like);
        vz_t = torch::zeros_like(like);
        sxx_t = torch::zeros_like(like);
        szz_t = torch::zeros_like(like);
        syz_t = torch::zeros_like(like);
        sxz_t = torch::zeros_like(like);
        sxy_t = torch::zeros_like(like);
        m_vxx_t = torch::zeros_like(like);
        m_vxz_t = torch::zeros_like(like);
        m_vyx_t = torch::zeros_like(like);
        m_vyz_t = torch::zeros_like(like);
        m_vzx_t = torch::zeros_like(like);
        m_vzz_t = torch::zeros_like(like);
        m_txxx_t = torch::zeros_like(like);
        m_txzz_t = torch::zeros_like(like);
        m_txyx_t = torch::zeros_like(like);
        m_tyzz_t = torch::zeros_like(like);
        m_txzx_t = torch::zeros_like(like);
        m_tzzz_t = torch::zeros_like(like);
    }

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        TORCH_CHECK(tensors.size() == 20, "ElasticTTISG 2D expects 20 wavefield tensors");
        int i = 0;
        vx_t = tensors[i++];
        vy_t = tensors[i++];
        vz_t = tensors[i++];
        sxx_t = tensors[i++];
        szz_t = tensors[i++];
        syz_t = tensors[i++];
        sxz_t = tensors[i++];
        sxy_t = tensors[i++];
        m_vxx_t = tensors[i++];
        m_vxz_t = tensors[i++];
        m_vyx_t = tensors[i++];
        m_vyz_t = tensors[i++];
        m_vzx_t = tensors[i++];
        m_vzz_t = tensors[i++];
        m_txxx_t = tensors[i++];
        m_txzz_t = tensors[i++];
        m_txyx_t = tensors[i++];
        m_tyzz_t = tensors[i++];
        m_txzx_t = tensors[i++];
        m_tzzz_t = tensors[i++];
    }

    WavefieldPointer view() const
    {
        WavefieldPointer out{};
        out.vx = vx_t.data_ptr<float>();
        out.vy = vy_t.data_ptr<float>();
        out.vz = vz_t.data_ptr<float>();
        out.sxx = sxx_t.data_ptr<float>();
        out.szz = szz_t.data_ptr<float>();
        out.syz = syz_t.data_ptr<float>();
        out.sxz = sxz_t.data_ptr<float>();
        out.sxy = sxy_t.data_ptr<float>();
        out.m_vxx = m_vxx_t.data_ptr<float>();
        out.m_vxz = m_vxz_t.data_ptr<float>();
        out.m_vyx = m_vyx_t.data_ptr<float>();
        out.m_vyz = m_vyz_t.data_ptr<float>();
        out.m_vzx = m_vzx_t.data_ptr<float>();
        out.m_vzz = m_vzz_t.data_ptr<float>();
        out.m_txxx = m_txxx_t.data_ptr<float>();
        out.m_txzz = m_txzz_t.data_ptr<float>();
        out.m_txyx = m_txyx_t.data_ptr<float>();
        out.m_tyzz = m_tyzz_t.data_ptr<float>();
        out.m_txzx = m_txzx_t.data_ptr<float>();
        out.m_tzzz = m_tzzz_t.data_ptr<float>();
        return out;
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return {
            vx_t, vy_t, vz_t, sxx_t, szz_t, syz_t, sxz_t, sxy_t,
            m_vxx_t, m_vxz_t, m_vyx_t, m_vyz_t, m_vzx_t, m_vzz_t,
            m_txxx_t, m_txzz_t, m_txyx_t, m_tyzz_t, m_txzx_t, m_tzzz_t,
        };
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        return state_tensors();
    }
};

inline StiffnessPointer stiffness_view(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == 16, "ElasticTTISG CUDA expects prepared models: rho plus 15 stiffness tensors");
    StiffnessPointer out{};
    int i = 0;
    out.rho = models[i++].data_ptr<float>();
    out.C11 = models[i++].data_ptr<float>();
    out.C13 = models[i++].data_ptr<float>();
    out.C14 = models[i++].data_ptr<float>();
    out.C15 = models[i++].data_ptr<float>();
    out.C16 = models[i++].data_ptr<float>();
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
    TORCH_CHECK(models.size() == 16, "ElasticTTISG CUDA backward expects 16 prepared models");
    std::vector<torch::Tensor> grads;
    grads.reserve(models.size());
    for (const auto& model : models)
        grads.push_back(torch::zeros_like(model));
    return grads;
}

inline StiffnessGradPointer stiffness_grad_view(std::vector<torch::Tensor>& grads)
{
    TORCH_CHECK(grads.size() == 16, "ElasticTTISG CUDA backward expects 16 prepared model gradients");
    StiffnessGradPointer out{};
    int i = 0;
    out.rho = grads[i++].data_ptr<float>();
    out.C11 = grads[i++].data_ptr<float>();
    out.C13 = grads[i++].data_ptr<float>();
    out.C14 = grads[i++].data_ptr<float>();
    out.C15 = grads[i++].data_ptr<float>();
    out.C16 = grads[i++].data_ptr<float>();
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

} // namespace elastic_tti_sg2d

#pragma once

#include <torch/extension.h>

#include "kernels.cuh"

namespace elastic_tti_2nd2d {

struct WavefieldTensor {
    torch::Tensor ux_t, uz_t, ux_pre_t, uz_pre_t, ux_nxt_t, uz_nxt_t;
    torch::Tensor m_gxux_t, m_gzux_t, m_gxuz_t, m_gzuz_t;
    torch::Tensor m_sxxx_t, m_sxzz_t, m_sxzx_t, m_szzz_t;

    void allocate(const torch::Tensor& like)
    {
        ux_t = torch::zeros_like(like);
        uz_t = torch::zeros_like(like);
        ux_pre_t = torch::zeros_like(like);
        uz_pre_t = torch::zeros_like(like);
        ux_nxt_t = torch::zeros_like(like);
        uz_nxt_t = torch::zeros_like(like);
        m_gxux_t = torch::zeros_like(like);
        m_gzux_t = torch::zeros_like(like);
        m_gxuz_t = torch::zeros_like(like);
        m_gzuz_t = torch::zeros_like(like);
        m_sxxx_t = torch::zeros_like(like);
        m_sxzz_t = torch::zeros_like(like);
        m_sxzx_t = torch::zeros_like(like);
        m_szzz_t = torch::zeros_like(like);
    }

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        TORCH_CHECK(tensors.size() == 14, "ElasticTTI2nd expects 14 wavefield tensors");
        int i = 0;
        ux_t = tensors[i++];
        uz_t = tensors[i++];
        ux_pre_t = tensors[i++];
        uz_pre_t = tensors[i++];
        ux_nxt_t = tensors[i++];
        uz_nxt_t = tensors[i++];
        m_gxux_t = tensors[i++];
        m_gzux_t = tensors[i++];
        m_gxuz_t = tensors[i++];
        m_gzuz_t = tensors[i++];
        m_sxxx_t = tensors[i++];
        m_sxzz_t = tensors[i++];
        m_sxzx_t = tensors[i++];
        m_szzz_t = tensors[i++];
    }

    // Rotate the (now, pre, next) displacement triple buffer: next becomes
    // now, now becomes pre, the old pre tensor is recycled as next.
    void swap_u()
    {
        auto tmp_x = ux_pre_t;
        auto tmp_z = uz_pre_t;
        ux_pre_t = ux_t;
        uz_pre_t = uz_t;
        ux_t = ux_nxt_t;
        uz_t = uz_nxt_t;
        ux_nxt_t = tmp_x;
        uz_nxt_t = tmp_z;
    }

    WavefieldPointer view() const
    {
        WavefieldPointer out{};
        out.ux = ux_t.data_ptr<float>();
        out.uz = uz_t.data_ptr<float>();
        out.ux_pre = ux_pre_t.data_ptr<float>();
        out.uz_pre = uz_pre_t.data_ptr<float>();
        out.ux_nxt = ux_nxt_t.data_ptr<float>();
        out.uz_nxt = uz_nxt_t.data_ptr<float>();
        out.m_gxux = m_gxux_t.data_ptr<float>();
        out.m_gzux = m_gzux_t.data_ptr<float>();
        out.m_gxuz = m_gxuz_t.data_ptr<float>();
        out.m_gzuz = m_gzuz_t.data_ptr<float>();
        out.m_sxxx = m_sxxx_t.data_ptr<float>();
        out.m_sxzz = m_sxzz_t.data_ptr<float>();
        out.m_sxzx = m_sxzx_t.data_ptr<float>();
        out.m_szzz = m_szzz_t.data_ptr<float>();
        return out;
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return {
            ux_t, uz_t, ux_pre_t, uz_pre_t, ux_nxt_t, uz_nxt_t,
            m_gxux_t, m_gzux_t, m_gxuz_t, m_gzuz_t,
            m_sxxx_t, m_sxzz_t, m_sxzx_t, m_szzz_t,
        };
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        return state_tensors();
    }
};

inline StiffnessPointer stiffness_view(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == 7, "ElasticTTI2nd CUDA expects prepared models: rho plus 6 stiffness tensors");
    StiffnessPointer out{};
    int i = 0;
    out.rho = models[i++].data_ptr<float>();
    out.C11 = models[i++].data_ptr<float>();
    out.C33 = models[i++].data_ptr<float>();
    out.C13 = models[i++].data_ptr<float>();
    out.C55 = models[i++].data_ptr<float>();
    out.C15 = models[i++].data_ptr<float>();
    out.C35 = models[i++].data_ptr<float>();
    return out;
}

inline std::vector<torch::Tensor> zero_model_grads(const std::vector<torch::Tensor>& models)
{
    TORCH_CHECK(models.size() == 7, "ElasticTTI2nd CUDA backward expects 7 prepared models");
    std::vector<torch::Tensor> grads;
    grads.reserve(models.size());
    for (const auto& model : models)
        grads.push_back(torch::zeros_like(model));
    return grads;
}

inline StiffnessGradPointer stiffness_grad_view(std::vector<torch::Tensor>& grads)
{
    TORCH_CHECK(grads.size() == 7, "ElasticTTI2nd CUDA backward expects 7 prepared model gradients");
    StiffnessGradPointer out{};
    int i = 0;
    out.rho = grads[i++].data_ptr<float>();
    out.C11 = grads[i++].data_ptr<float>();
    out.C33 = grads[i++].data_ptr<float>();
    out.C13 = grads[i++].data_ptr<float>();
    out.C55 = grads[i++].data_ptr<float>();
    out.C15 = grads[i++].data_ptr<float>();
    out.C35 = grads[i++].data_ptr<float>();
    return out;
}

} // namespace elastic_tti_2nd2d

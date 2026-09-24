#pragma once

#include <torch/extension.h>

#include "kernels.cuh"
#include "../../common/cudautils.h"   // ptr_or_null

namespace elastic_tti_sg2d {

struct WavefieldTensor {
    torch::Tensor vx_t, vy_t, vz_t, sxx_t, szz_t, syz_t, sxz_t, sxy_t;
    torch::Tensor m_vxx_t, m_vxz_t, m_vyx_t, m_vyz_t, m_vzx_t, m_vzz_t;
    torch::Tensor m_txxx_t, m_txzz_t, m_txyx_t, m_tyzz_t, m_txzx_t, m_tzzz_t;

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        SWEEP_CHECK(tensors.size() == 20, "ElasticTTISG 2D expects 20 wavefield tensors");
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

    // Boundary-saving reconstruction bind: the 8 physical fields only (the
    // first 8 slots of the bind() order); the 12 CPML memory tensors stay
    // undefined, so view() hands the kernels nullptr for them.  Only valid
    // for the NOPML reverse reconstruction, which never touches m_*.
    void bind_physical(const std::vector<torch::Tensor>& tensors)
    {
        SWEEP_CHECK(tensors.size() == 8,
                    "ElasticTTISG 2D expects 8 physical wavefield tensors "
                    "[vx, vy, vz, sxx, szz, syz, sxz, sxy]; got ", tensors.size());
        int i = 0;
        vx_t = tensors[i++];
        vy_t = tensors[i++];
        vz_t = tensors[i++];
        sxx_t = tensors[i++];
        szz_t = tensors[i++];
        syz_t = tensors[i++];
        sxz_t = tensors[i++];
        sxy_t = tensors[i++];
        m_vxx_t = m_vxz_t = m_vyx_t = m_vyz_t = m_vzx_t = m_vzz_t = torch::Tensor();
        m_txxx_t = m_txzz_t = m_txyx_t = m_tyzz_t = m_txzx_t = m_tzzz_t = torch::Tensor();
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
        // CPML memory: nullptr after bind_physical (bs reconstruction).
        out.m_vxx = ptr_or_null(m_vxx_t);
        out.m_vxz = ptr_or_null(m_vxz_t);
        out.m_vyx = ptr_or_null(m_vyx_t);
        out.m_vyz = ptr_or_null(m_vyz_t);
        out.m_vzx = ptr_or_null(m_vzx_t);
        out.m_vzz = ptr_or_null(m_vzz_t);
        out.m_txxx = ptr_or_null(m_txxx_t);
        out.m_txzz = ptr_or_null(m_txzz_t);
        out.m_txyx = ptr_or_null(m_txyx_t);
        out.m_tyzz = ptr_or_null(m_tyzz_t);
        out.m_txzx = ptr_or_null(m_txzx_t);
        out.m_tzzz = ptr_or_null(m_tzzz_t);
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
    SWEEP_CHECK(models.size() == 16, "ElasticTTISG CUDA expects prepared models: rho plus 15 stiffness tensors");
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

inline StiffnessGradPointer stiffness_grad_view(std::vector<torch::Tensor>& grads)
{
    SWEEP_CHECK(grads.size() == 16, "ElasticTTISG CUDA backward expects 16 prepared model gradients");
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

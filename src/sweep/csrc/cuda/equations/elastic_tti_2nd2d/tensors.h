#pragma once
#include <cuda_runtime.h>


#include "kernels.cuh"
#include "../../common/cudautils.h"

namespace elastic_tti_2nd2d {

struct WavefieldTensor {
    Buf ux_t, uz_t, ux_pre_t, uz_pre_t, ux_nxt_t, uz_nxt_t;
    Buf m_gxux_t, m_gzux_t, m_gxuz_t, m_gzuz_t;
    Buf m_sxxx_t, m_sxzz_t, m_sxzx_t, m_szzz_t;

    // Boundary-saving reconstruction state (bind_recon): the six displacement
    // slots only, in bind() order -- ElasticTTI2nd.cuda_layout.bs_reconstruction_nvar.
    // The eight CPML memory members stay undefined; the nopml reverse kernels
    // never read them and view() packs them as nullptr.
    static constexpr int RECON_WF_COUNT = 6;
    static constexpr const char* RECON_LIST_DESC =
        "[ux, uz, ux_pre, uz_pre, ux_nxt, uz_nxt]";

    // No allocate(): every state this struct carries -- the forward, the
    // adjoint, the bs reconstruction and the checkpoint replay -- is bound by
    // the propagator, so the driver owns no wavefield storage.

    void bind(const std::vector<Buf>& tensors)
    {
        SWEEP_CHECK(tensors.size() == 14, "ElasticTTI2nd expects 14 wavefield tensors");
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

    void bind_recon(const std::vector<Buf>& tensors)
    {
        SWEEP_CHECK(static_cast<int>(tensors.size()) == RECON_WF_COUNT,
                    "ElasticTTI2nd backward_bs reconstruction expects ", RECON_WF_COUNT,
                    " wavefield tensors ", RECON_LIST_DESC, ", got ", tensors.size());
        int i = 0;
        ux_t = tensors[i++];
        uz_t = tensors[i++];
        ux_pre_t = tensors[i++];
        uz_pre_t = tensors[i++];
        ux_nxt_t = tensors[i++];
        uz_nxt_t = tensors[i++];
        m_gxux_t = m_gzux_t = m_gxuz_t = m_gzuz_t = Buf{};
        m_sxxx_t = m_sxzz_t = m_sxzx_t = m_szzz_t = Buf{};
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
        // CPML memory: nullptr after bind_recon (backward_bs reconstruction),
        // where only the nopml kernels run on this view.
        out.m_gxux = ptr_or_null(m_gxux_t);
        out.m_gzux = ptr_or_null(m_gzux_t);
        out.m_gxuz = ptr_or_null(m_gxuz_t);
        out.m_gzuz = ptr_or_null(m_gzuz_t);
        out.m_sxxx = ptr_or_null(m_sxxx_t);
        out.m_sxzz = ptr_or_null(m_sxzz_t);
        out.m_sxzx = ptr_or_null(m_sxzx_t);
        out.m_szzz = ptr_or_null(m_szzz_t);
        return out;
    }

    std::vector<Buf> state_tensors() const
    {
        return {
            ux_t, uz_t, ux_pre_t, uz_pre_t, ux_nxt_t, uz_nxt_t,
            m_gxux_t, m_gzux_t, m_gxuz_t, m_gzuz_t,
            m_sxxx_t, m_sxzz_t, m_sxzx_t, m_szzz_t,
        };
    }

    std::vector<Buf> checkpoint_tensors() const
    {
        return state_tensors();
    }
};

inline StiffnessPointer stiffness_view(const std::vector<Buf>& models)
{
    SWEEP_CHECK(models.size() == 7, "ElasticTTI2nd CUDA expects prepared models: rho plus 6 stiffness tensors");
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

inline StiffnessGradPointer stiffness_grad_view(std::vector<Buf>& grads)
{
    SWEEP_CHECK(grads.size() == 7, "ElasticTTI2nd CUDA backward expects 7 prepared model gradients");
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

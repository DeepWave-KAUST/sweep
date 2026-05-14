#pragma once

#include <torch/extension.h>

#include "elastic.h"

struct DasMuWavefieldPointer2D {
    float* __restrict__ vx;
    float* __restrict__ vz;
    float* __restrict__ sxx;
    float* __restrict__ szz;
    float* __restrict__ sxz;
    float* __restrict__ exx;
    float* __restrict__ ezz;
    float* __restrict__ exz;

    float* __restrict__ m_vxx;
    float* __restrict__ m_vxz;
    float* __restrict__ m_vzx;
    float* __restrict__ m_vzz;
    float* __restrict__ m_sxxx;
    float* __restrict__ m_sxxz;
    float* __restrict__ m_szzx;
    float* __restrict__ m_szzz;
    float* __restrict__ m_sxzx;
    float* __restrict__ m_sxzz;

    __device__ DasMuWavefieldPointer2D offset(int b, int spatial_size) const
    {
        DasMuWavefieldPointer2D out = *this;
        const int shift = b * spatial_size;

        out.vx += shift;
        out.vz += shift;
        out.sxx += shift;
        out.szz += shift;
        out.sxz += shift;
        out.exx += shift;
        out.ezz += shift;
        out.exz += shift;
        out.m_vxx += shift;
        out.m_vxz += shift;
        out.m_vzx += shift;
        out.m_vzz += shift;
        out.m_sxxx += shift;
        out.m_sxxz += shift;
        out.m_szzx += shift;
        out.m_szzz += shift;
        out.m_sxzx += shift;
        out.m_sxzz += shift;
        return out;
    }
};

struct DasMuWavefieldTensor2D {
    torch::Tensor vx_t;
    torch::Tensor vz_t;
    torch::Tensor sxx_t;
    torch::Tensor szz_t;
    torch::Tensor sxz_t;
    torch::Tensor exx_t;
    torch::Tensor ezz_t;
    torch::Tensor exz_t;
    torch::Tensor m_vxx_t;
    torch::Tensor m_vxz_t;
    torch::Tensor m_vzx_t;
    torch::Tensor m_vzz_t;
    torch::Tensor m_sxxx_t;
    torch::Tensor m_sxxz_t;
    torch::Tensor m_szzx_t;
    torch::Tensor m_szzz_t;
    torch::Tensor m_sxzx_t;
    torch::Tensor m_sxzz_t;

    bool allocated = false;

    void allocate(const torch::Tensor& like, bool use_pml = true)
    {
        if (allocated) return;
        vx_t = torch::zeros_like(like);
        vz_t = torch::zeros_like(like);
        sxx_t = torch::zeros_like(like);
        szz_t = torch::zeros_like(like);
        sxz_t = torch::zeros_like(like);
        exx_t = torch::zeros_like(like);
        ezz_t = torch::zeros_like(like);
        exz_t = torch::zeros_like(like);

        if (use_pml) {
            m_vxx_t = torch::zeros_like(like);
            m_vxz_t = torch::zeros_like(like);
            m_vzx_t = torch::zeros_like(like);
            m_vzz_t = torch::zeros_like(like);
            m_sxxx_t = torch::zeros_like(like);
            m_sxxz_t = torch::zeros_like(like);
            m_szzx_t = torch::zeros_like(like);
            m_szzz_t = torch::zeros_like(like);
            m_sxzx_t = torch::zeros_like(like);
            m_sxzz_t = torch::zeros_like(like);
        }
        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors, bool use_pml = true)
    {
        int i = 0;
        TORCH_CHECK(tensors.size() == (use_pml ? 18 : 8), "DAS Mu 2D wavefield expects 8 base tensors plus 10 CPML tensors");
        vx_t = tensors[i++];
        vz_t = tensors[i++];
        sxx_t = tensors[i++];
        szz_t = tensors[i++];
        sxz_t = tensors[i++];
        exx_t = tensors[i++];
        ezz_t = tensors[i++];
        exz_t = tensors[i++];
        if (use_pml) {
            m_vxx_t = tensors[i++];
            m_vxz_t = tensors[i++];
            m_vzx_t = tensors[i++];
            m_vzz_t = tensors[i++];
            m_sxxx_t = tensors[i++];
            m_sxxz_t = tensors[i++];
            m_szzx_t = tensors[i++];
            m_szzz_t = tensors[i++];
            m_sxzx_t = tensors[i++];
            m_sxzz_t = tensors[i++];
        }
        allocated = true;
    }

    DasMuWavefieldPointer2D view()
    {
        DasMuWavefieldPointer2D v{};
        v.vx = vx_t.data_ptr<float>();
        v.vz = vz_t.data_ptr<float>();
        v.sxx = sxx_t.data_ptr<float>();
        v.szz = szz_t.data_ptr<float>();
        v.sxz = sxz_t.data_ptr<float>();
        v.exx = exx_t.data_ptr<float>();
        v.ezz = ezz_t.data_ptr<float>();
        v.exz = exz_t.data_ptr<float>();
        v.m_vxx = m_vxx_t.data_ptr<float>();
        v.m_vxz = m_vxz_t.data_ptr<float>();
        v.m_vzx = m_vzx_t.data_ptr<float>();
        v.m_vzz = m_vzz_t.data_ptr<float>();
        v.m_sxxx = m_sxxx_t.data_ptr<float>();
        v.m_sxxz = m_sxxz_t.data_ptr<float>();
        v.m_szzx = m_szzx_t.data_ptr<float>();
        v.m_szzz = m_szzz_t.data_ptr<float>();
        v.m_sxzx = m_sxzx_t.data_ptr<float>();
        v.m_sxzz = m_sxzz_t.data_ptr<float>();
        return v;
    }

    ElasticWavefieldPointer elastic_view()
    {
        ElasticWavefieldPointer v{};
        v.vx = vx_t.data_ptr<float>();
        v.vy = nullptr;
        v.vz = vz_t.data_ptr<float>();
        v.sxx = sxx_t.data_ptr<float>();
        v.syy = nullptr;
        v.szz = szz_t.data_ptr<float>();
        v.sxy = nullptr;
        v.sxz = sxz_t.data_ptr<float>();
        v.syz = nullptr;
        v.m_vxx = m_vxx_t.data_ptr<float>();
        v.m_vxy = nullptr;
        v.m_vxz = m_vxz_t.data_ptr<float>();
        v.m_vyx = nullptr;
        v.m_vyy = nullptr;
        v.m_vyz = nullptr;
        v.m_vzx = m_vzx_t.data_ptr<float>();
        v.m_vzy = nullptr;
        v.m_vzz = m_vzz_t.data_ptr<float>();
        v.m_sxxx = m_sxxx_t.data_ptr<float>();
        v.m_sxxy = nullptr;
        v.m_sxxz = m_sxxz_t.data_ptr<float>();
        v.m_syyx = nullptr;
        v.m_syyy = nullptr;
        v.m_syyz = nullptr;
        v.m_szzx = m_szzx_t.data_ptr<float>();
        v.m_szzy = nullptr;
        v.m_szzz = m_szzz_t.data_ptr<float>();
        v.m_sxyx = nullptr;
        v.m_sxyy = nullptr;
        v.m_sxyz = nullptr;
        v.m_sxzx = m_sxzx_t.data_ptr<float>();
        v.m_sxzy = nullptr;
        v.m_sxzz = m_sxzz_t.data_ptr<float>();
        v.m_syzx = nullptr;
        v.m_syzy = nullptr;
        v.m_syzz = nullptr;
        return v;
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        return {
            vx_t, vz_t, sxx_t, szz_t, sxz_t, exx_t, ezz_t, exz_t,
            m_vxx_t, m_vxz_t, m_vzx_t, m_vzz_t,
            m_sxxx_t, m_sxxz_t, m_szzx_t, m_szzz_t, m_sxzx_t, m_sxzz_t
        };
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return checkpoint_tensors();
    }
};

inline float* das_mu2d_field_ptr(DasMuWavefieldPointer2D& wf, int idx)
{
    switch (idx) {
        case 0: return wf.vx;
        case 1: return wf.vz;
        case 2: return wf.sxx;
        case 3: return wf.szz;
        case 4: return wf.sxz;
        case 5: return wf.exx;
        case 6: return wf.ezz;
        case 7: return wf.exz;
        default: return nullptr;
    }
}

struct DasMuWavefieldPointer3D {
    float* __restrict__ vx;
    float* __restrict__ vy;
    float* __restrict__ vz;
    float* __restrict__ sxx;
    float* __restrict__ syy;
    float* __restrict__ szz;
    float* __restrict__ sxy;
    float* __restrict__ sxz;
    float* __restrict__ syz;
    float* __restrict__ exx;
    float* __restrict__ eyy;
    float* __restrict__ ezz;
    float* __restrict__ exy;
    float* __restrict__ exz;
    float* __restrict__ eyz;

    float* __restrict__ m_vxx;
    float* __restrict__ m_vxy;
    float* __restrict__ m_vxz;
    float* __restrict__ m_vyx;
    float* __restrict__ m_vyy;
    float* __restrict__ m_vyz;
    float* __restrict__ m_vzx;
    float* __restrict__ m_vzy;
    float* __restrict__ m_vzz;
    float* __restrict__ m_sxxx;
    float* __restrict__ m_szzz;
    float* __restrict__ m_sxyx;
    float* __restrict__ m_sxyy;
    float* __restrict__ m_sxzx;
    float* __restrict__ m_sxzz;
    float* __restrict__ m_syyy;
    float* __restrict__ m_syzy;
    float* __restrict__ m_syzz;

    __device__ DasMuWavefieldPointer3D offset(int b, int spatial_size) const
    {
        DasMuWavefieldPointer3D out = *this;
        const int shift = b * spatial_size;

        out.vx += shift;
        out.vy += shift;
        out.vz += shift;
        out.sxx += shift;
        out.syy += shift;
        out.szz += shift;
        out.sxy += shift;
        out.sxz += shift;
        out.syz += shift;
        out.exx += shift;
        out.eyy += shift;
        out.ezz += shift;
        out.exy += shift;
        out.exz += shift;
        out.eyz += shift;
        if (out.m_vxx) out.m_vxx += shift;
        if (out.m_vxy) out.m_vxy += shift;
        if (out.m_vxz) out.m_vxz += shift;
        if (out.m_vyx) out.m_vyx += shift;
        if (out.m_vyy) out.m_vyy += shift;
        if (out.m_vyz) out.m_vyz += shift;
        if (out.m_vzx) out.m_vzx += shift;
        if (out.m_vzy) out.m_vzy += shift;
        if (out.m_vzz) out.m_vzz += shift;
        if (out.m_sxxx) out.m_sxxx += shift;
        if (out.m_szzz) out.m_szzz += shift;
        if (out.m_sxyx) out.m_sxyx += shift;
        if (out.m_sxyy) out.m_sxyy += shift;
        if (out.m_sxzx) out.m_sxzx += shift;
        if (out.m_sxzz) out.m_sxzz += shift;
        if (out.m_syyy) out.m_syyy += shift;
        if (out.m_syzy) out.m_syzy += shift;
        if (out.m_syzz) out.m_syzz += shift;
        return out;
    }
};

struct DasMuWavefieldTensor3D {
    torch::Tensor vx_t;
    torch::Tensor vy_t;
    torch::Tensor vz_t;
    torch::Tensor sxx_t;
    torch::Tensor syy_t;
    torch::Tensor szz_t;
    torch::Tensor sxy_t;
    torch::Tensor sxz_t;
    torch::Tensor syz_t;
    torch::Tensor exx_t;
    torch::Tensor eyy_t;
    torch::Tensor ezz_t;
    torch::Tensor exy_t;
    torch::Tensor exz_t;
    torch::Tensor eyz_t;
    torch::Tensor m_vxx_t;
    torch::Tensor m_vxy_t;
    torch::Tensor m_vxz_t;
    torch::Tensor m_vyx_t;
    torch::Tensor m_vyy_t;
    torch::Tensor m_vyz_t;
    torch::Tensor m_vzx_t;
    torch::Tensor m_vzy_t;
    torch::Tensor m_vzz_t;
    torch::Tensor m_sxxx_t;
    torch::Tensor m_szzz_t;
    torch::Tensor m_sxyx_t;
    torch::Tensor m_sxyy_t;
    torch::Tensor m_sxzx_t;
    torch::Tensor m_sxzz_t;
    torch::Tensor m_syyy_t;
    torch::Tensor m_syzy_t;
    torch::Tensor m_syzz_t;

    bool allocated = false;

    void allocate(const torch::Tensor& like, bool use_pml = true)
    {
        if (allocated) return;
        vx_t = torch::zeros_like(like);
        vy_t = torch::zeros_like(like);
        vz_t = torch::zeros_like(like);
        sxx_t = torch::zeros_like(like);
        syy_t = torch::zeros_like(like);
        szz_t = torch::zeros_like(like);
        sxy_t = torch::zeros_like(like);
        sxz_t = torch::zeros_like(like);
        syz_t = torch::zeros_like(like);
        exx_t = torch::zeros_like(like);
        eyy_t = torch::zeros_like(like);
        ezz_t = torch::zeros_like(like);
        exy_t = torch::zeros_like(like);
        exz_t = torch::zeros_like(like);
        eyz_t = torch::zeros_like(like);

        if (use_pml) {
            m_vxx_t = torch::zeros_like(like);
            m_vxy_t = torch::zeros_like(like);
            m_vxz_t = torch::zeros_like(like);
            m_vyx_t = torch::zeros_like(like);
            m_vyy_t = torch::zeros_like(like);
            m_vyz_t = torch::zeros_like(like);
            m_vzx_t = torch::zeros_like(like);
            m_vzy_t = torch::zeros_like(like);
            m_vzz_t = torch::zeros_like(like);
            m_sxxx_t = torch::zeros_like(like);
            m_szzz_t = torch::zeros_like(like);
            m_sxyx_t = torch::zeros_like(like);
            m_sxyy_t = torch::zeros_like(like);
            m_sxzx_t = torch::zeros_like(like);
            m_sxzz_t = torch::zeros_like(like);
            m_syyy_t = torch::zeros_like(like);
            m_syzy_t = torch::zeros_like(like);
            m_syzz_t = torch::zeros_like(like);
        }
        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors, bool use_pml = true)
    {
        int i = 0;
        TORCH_CHECK(tensors.size() == (use_pml ? 33 : 15), "DAS Mu 3D wavefield expects 15 base tensors plus 18 CPML tensors");
        vx_t = tensors[i++];
        vy_t = tensors[i++];
        vz_t = tensors[i++];
        sxx_t = tensors[i++];
        syy_t = tensors[i++];
        szz_t = tensors[i++];
        sxy_t = tensors[i++];
        sxz_t = tensors[i++];
        syz_t = tensors[i++];
        exx_t = tensors[i++];
        eyy_t = tensors[i++];
        ezz_t = tensors[i++];
        exy_t = tensors[i++];
        exz_t = tensors[i++];
        eyz_t = tensors[i++];
        if (use_pml) {
            m_vxx_t = tensors[i++];
            m_vxy_t = tensors[i++];
            m_vxz_t = tensors[i++];
            m_vyx_t = tensors[i++];
            m_vyy_t = tensors[i++];
            m_vyz_t = tensors[i++];
            m_vzx_t = tensors[i++];
            m_vzy_t = tensors[i++];
            m_vzz_t = tensors[i++];
            m_sxxx_t = tensors[i++];
            m_szzz_t = tensors[i++];
            m_sxyx_t = tensors[i++];
            m_sxyy_t = tensors[i++];
            m_sxzx_t = tensors[i++];
            m_sxzz_t = tensors[i++];
            m_syyy_t = tensors[i++];
            m_syzy_t = tensors[i++];
            m_syzz_t = tensors[i++];
        }
        allocated = true;
    }

    DasMuWavefieldPointer3D view()
    {
        DasMuWavefieldPointer3D v{};
        v.vx = vx_t.data_ptr<float>();
        v.vy = vy_t.data_ptr<float>();
        v.vz = vz_t.data_ptr<float>();
        v.sxx = sxx_t.data_ptr<float>();
        v.syy = syy_t.data_ptr<float>();
        v.szz = szz_t.data_ptr<float>();
        v.sxy = sxy_t.data_ptr<float>();
        v.sxz = sxz_t.data_ptr<float>();
        v.syz = syz_t.data_ptr<float>();
        v.exx = exx_t.data_ptr<float>();
        v.eyy = eyy_t.data_ptr<float>();
        v.ezz = ezz_t.data_ptr<float>();
        v.exy = exy_t.data_ptr<float>();
        v.exz = exz_t.data_ptr<float>();
        v.eyz = eyz_t.data_ptr<float>();
        v.m_vxx = m_vxx_t.defined() ? m_vxx_t.data_ptr<float>() : nullptr;
        v.m_vxy = m_vxy_t.defined() ? m_vxy_t.data_ptr<float>() : nullptr;
        v.m_vxz = m_vxz_t.defined() ? m_vxz_t.data_ptr<float>() : nullptr;
        v.m_vyx = m_vyx_t.defined() ? m_vyx_t.data_ptr<float>() : nullptr;
        v.m_vyy = m_vyy_t.defined() ? m_vyy_t.data_ptr<float>() : nullptr;
        v.m_vyz = m_vyz_t.defined() ? m_vyz_t.data_ptr<float>() : nullptr;
        v.m_vzx = m_vzx_t.defined() ? m_vzx_t.data_ptr<float>() : nullptr;
        v.m_vzy = m_vzy_t.defined() ? m_vzy_t.data_ptr<float>() : nullptr;
        v.m_vzz = m_vzz_t.defined() ? m_vzz_t.data_ptr<float>() : nullptr;
        v.m_sxxx = m_sxxx_t.defined() ? m_sxxx_t.data_ptr<float>() : nullptr;
        v.m_szzz = m_szzz_t.defined() ? m_szzz_t.data_ptr<float>() : nullptr;
        v.m_sxyx = m_sxyx_t.defined() ? m_sxyx_t.data_ptr<float>() : nullptr;
        v.m_sxyy = m_sxyy_t.defined() ? m_sxyy_t.data_ptr<float>() : nullptr;
        v.m_sxzx = m_sxzx_t.defined() ? m_sxzx_t.data_ptr<float>() : nullptr;
        v.m_sxzz = m_sxzz_t.defined() ? m_sxzz_t.data_ptr<float>() : nullptr;
        v.m_syyy = m_syyy_t.defined() ? m_syyy_t.data_ptr<float>() : nullptr;
        v.m_syzy = m_syzy_t.defined() ? m_syzy_t.data_ptr<float>() : nullptr;
        v.m_syzz = m_syzz_t.defined() ? m_syzz_t.data_ptr<float>() : nullptr;
        return v;
    }

    ElasticWavefieldPointer elastic_view()
    {
        ElasticWavefieldPointer v{};
        v.vx = vx_t.data_ptr<float>();
        v.vy = vy_t.data_ptr<float>();
        v.vz = vz_t.data_ptr<float>();
        v.sxx = sxx_t.data_ptr<float>();
        v.syy = syy_t.data_ptr<float>();
        v.szz = szz_t.data_ptr<float>();
        v.sxy = sxy_t.data_ptr<float>();
        v.sxz = sxz_t.data_ptr<float>();
        v.syz = syz_t.data_ptr<float>();
        v.m_vxx = m_vxx_t.defined() ? m_vxx_t.data_ptr<float>() : nullptr;
        v.m_vxy = m_vxy_t.defined() ? m_vxy_t.data_ptr<float>() : nullptr;
        v.m_vxz = m_vxz_t.defined() ? m_vxz_t.data_ptr<float>() : nullptr;
        v.m_vyx = m_vyx_t.defined() ? m_vyx_t.data_ptr<float>() : nullptr;
        v.m_vyy = m_vyy_t.defined() ? m_vyy_t.data_ptr<float>() : nullptr;
        v.m_vyz = m_vyz_t.defined() ? m_vyz_t.data_ptr<float>() : nullptr;
        v.m_vzx = m_vzx_t.defined() ? m_vzx_t.data_ptr<float>() : nullptr;
        v.m_vzy = m_vzy_t.defined() ? m_vzy_t.data_ptr<float>() : nullptr;
        v.m_vzz = m_vzz_t.defined() ? m_vzz_t.data_ptr<float>() : nullptr;
        v.m_sxxx = m_sxxx_t.defined() ? m_sxxx_t.data_ptr<float>() : nullptr;
        v.m_sxxy = nullptr;
        v.m_sxxz = nullptr;
        v.m_syyx = nullptr;
        v.m_syyy = m_syyy_t.defined() ? m_syyy_t.data_ptr<float>() : nullptr;
        v.m_syyz = nullptr;
        v.m_szzx = nullptr;
        v.m_szzy = nullptr;
        v.m_szzz = m_szzz_t.defined() ? m_szzz_t.data_ptr<float>() : nullptr;
        v.m_sxyx = m_sxyx_t.defined() ? m_sxyx_t.data_ptr<float>() : nullptr;
        v.m_sxyy = m_sxyy_t.defined() ? m_sxyy_t.data_ptr<float>() : nullptr;
        v.m_sxyz = nullptr;
        v.m_sxzx = m_sxzx_t.defined() ? m_sxzx_t.data_ptr<float>() : nullptr;
        v.m_sxzy = nullptr;
        v.m_sxzz = m_sxzz_t.defined() ? m_sxzz_t.data_ptr<float>() : nullptr;
        v.m_syzx = nullptr;
        v.m_syzy = m_syzy_t.defined() ? m_syzy_t.data_ptr<float>() : nullptr;
        v.m_syzz = m_syzz_t.defined() ? m_syzz_t.data_ptr<float>() : nullptr;
        return v;
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        return {
            vx_t, vy_t, vz_t, sxx_t, syy_t, szz_t, sxy_t, sxz_t, syz_t,
            exx_t, eyy_t, ezz_t, exy_t, exz_t, eyz_t,
            m_vxx_t, m_vxy_t, m_vxz_t, m_vyx_t, m_vyy_t, m_vyz_t, m_vzx_t, m_vzy_t, m_vzz_t,
            m_sxxx_t, m_szzz_t, m_sxyx_t, m_sxyy_t, m_sxzx_t, m_sxzz_t, m_syyy_t, m_syzy_t, m_syzz_t
        };
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return checkpoint_tensors();
    }
};

inline float* das_mu3d_field_ptr(DasMuWavefieldPointer3D& wf, int idx)
{
    switch (idx) {
        case 0: return wf.vx;
        case 1: return wf.vy;
        case 2: return wf.vz;
        case 3: return wf.sxx;
        case 4: return wf.syy;
        case 5: return wf.szz;
        case 6: return wf.sxy;
        case 7: return wf.sxz;
        case 8: return wf.syz;
        case 9: return wf.exx;
        case 10: return wf.eyy;
        case 11: return wf.ezz;
        case 12: return wf.exy;
        case 13: return wf.exz;
        case 14: return wf.eyz;
        default: return nullptr;
    }
}

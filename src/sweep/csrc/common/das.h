#pragma once

#include <torch/extension.h>

struct DasWavefieldPointer2D {
    float* __restrict__ exx;
    float* __restrict__ ezz;
    float* __restrict__ sxx;
    float* __restrict__ szz;
    float* __restrict__ txx;
    float* __restrict__ tzz;
    float* __restrict__ m_sxx_xf;
    float* __restrict__ m_sxx_xb;
    float* __restrict__ m_szz_zf;
    float* __restrict__ m_szz_zb;
    float* __restrict__ m_txx_zf;
    float* __restrict__ m_txx_zb;
    float* __restrict__ m_tzz_xf;
    float* __restrict__ m_tzz_xb;
    float* __restrict__ das35;
    float* __restrict__ das54x;
    float* __restrict__ das54z;

    __device__ DasWavefieldPointer2D offset(int b, int spatial_size) const
    {
        DasWavefieldPointer2D out = *this;
        const int shift = b * spatial_size;
        out.exx += shift;
        out.ezz += shift;
        out.sxx += shift;
        out.szz += shift;
        out.txx += shift;
        out.tzz += shift;
        out.m_sxx_xf += shift;
        out.m_sxx_xb += shift;
        out.m_szz_zf += shift;
        out.m_szz_zb += shift;
        out.m_txx_zf += shift;
        out.m_txx_zb += shift;
        out.m_tzz_xf += shift;
        out.m_tzz_xb += shift;
        out.das35 += shift;
        out.das54x += shift;
        out.das54z += shift;
        return out;
    }
};

struct DasWavefieldTensor2D {
    torch::Tensor exx_t, ezz_t, sxx_t, szz_t, txx_t, tzz_t;
    torch::Tensor m_sxx_xf_t, m_sxx_xb_t, m_szz_zf_t, m_szz_zb_t;
    torch::Tensor m_txx_zf_t, m_txx_zb_t, m_tzz_xf_t, m_tzz_xb_t;
    torch::Tensor das35_t, das54x_t, das54z_t;
    bool allocated = false;

    void allocate(const torch::Tensor& like)
    {
        if (allocated) return;
        exx_t = torch::zeros_like(like);
        ezz_t = torch::zeros_like(like);
        sxx_t = torch::zeros_like(like);
        szz_t = torch::zeros_like(like);
        txx_t = torch::zeros_like(like);
        tzz_t = torch::zeros_like(like);
        m_sxx_xf_t = torch::zeros_like(like);
        m_sxx_xb_t = torch::zeros_like(like);
        m_szz_zf_t = torch::zeros_like(like);
        m_szz_zb_t = torch::zeros_like(like);
        m_txx_zf_t = torch::zeros_like(like);
        m_txx_zb_t = torch::zeros_like(like);
        m_tzz_xf_t = torch::zeros_like(like);
        m_tzz_xb_t = torch::zeros_like(like);
        das35_t = torch::zeros_like(like);
        das54x_t = torch::zeros_like(like);
        das54z_t = torch::zeros_like(like);
        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        TORCH_CHECK(tensors.size() == 17, "DAS 2D expects 17 wavefield tensors");
        int i = 0;
        exx_t = tensors[i++];
        ezz_t = tensors[i++];
        sxx_t = tensors[i++];
        szz_t = tensors[i++];
        txx_t = tensors[i++];
        tzz_t = tensors[i++];
        m_sxx_xf_t = tensors[i++];
        m_sxx_xb_t = tensors[i++];
        m_szz_zf_t = tensors[i++];
        m_szz_zb_t = tensors[i++];
        m_txx_zf_t = tensors[i++];
        m_txx_zb_t = tensors[i++];
        m_tzz_xf_t = tensors[i++];
        m_tzz_xb_t = tensors[i++];
        das35_t = tensors[i++];
        das54x_t = tensors[i++];
        das54z_t = tensors[i++];
        allocated = true;
    }

    DasWavefieldPointer2D view()
    {
        return {
            exx_t.data_ptr<float>(),
            ezz_t.data_ptr<float>(),
            sxx_t.data_ptr<float>(),
            szz_t.data_ptr<float>(),
            txx_t.data_ptr<float>(),
            tzz_t.data_ptr<float>(),
            m_sxx_xf_t.data_ptr<float>(),
            m_sxx_xb_t.data_ptr<float>(),
            m_szz_zf_t.data_ptr<float>(),
            m_szz_zb_t.data_ptr<float>(),
            m_txx_zf_t.data_ptr<float>(),
            m_txx_zb_t.data_ptr<float>(),
            m_tzz_xf_t.data_ptr<float>(),
            m_tzz_xb_t.data_ptr<float>(),
            das35_t.data_ptr<float>(),
            das54x_t.data_ptr<float>(),
            das54z_t.data_ptr<float>(),
        };
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return {
            exx_t, ezz_t, sxx_t, szz_t, txx_t, tzz_t,
            m_sxx_xf_t, m_sxx_xb_t, m_szz_zf_t, m_szz_zb_t,
            m_txx_zf_t, m_txx_zb_t, m_tzz_xf_t, m_tzz_xb_t,
            das35_t, das54x_t, das54z_t
        };
    }
};

inline float* das2d_field_ptr(DasWavefieldPointer2D wf, int field)
{
    switch (field) {
        case 0: return wf.exx;
        case 1: return wf.ezz;
        case 2: return wf.sxx;
        case 3: return wf.szz;
        case 4: return wf.txx;
        case 5: return wf.tzz;
        case 6: return wf.m_sxx_xf;
        case 7: return wf.m_sxx_xb;
        case 8: return wf.m_szz_zf;
        case 9: return wf.m_szz_zb;
        case 10: return wf.m_txx_zf;
        case 11: return wf.m_txx_zb;
        case 12: return wf.m_tzz_xf;
        case 13: return wf.m_tzz_xb;
        case 14: return wf.das35;
        case 15: return wf.das54x;
        case 16: return wf.das54z;
        default: return nullptr;
    }
}

struct DasWavefieldPointer3D {
    float* __restrict__ exx;
    float* __restrict__ eyy;
    float* __restrict__ ezz;
    float* __restrict__ sxx;
    float* __restrict__ syy;
    float* __restrict__ szz;
    float* __restrict__ txx;
    float* __restrict__ tyy;
    float* __restrict__ tzz;
    float* __restrict__ m_sxx_xf;
    float* __restrict__ m_sxx_xb;
    float* __restrict__ m_syy_yf;
    float* __restrict__ m_syy_yb;
    float* __restrict__ m_szz_zf;
    float* __restrict__ m_szz_zb;
    float* __restrict__ m_txx_yf;
    float* __restrict__ m_txx_yb;
    float* __restrict__ m_txx_zf;
    float* __restrict__ m_txx_zb;
    float* __restrict__ m_tyy_xf;
    float* __restrict__ m_tyy_xb;
    float* __restrict__ m_tyy_zf;
    float* __restrict__ m_tyy_zb;
    float* __restrict__ m_tzz_xf;
    float* __restrict__ m_tzz_xb;
    float* __restrict__ m_tzz_yf;
    float* __restrict__ m_tzz_yb;
    float* __restrict__ das35;
    float* __restrict__ das54x;
    float* __restrict__ das54y;
    float* __restrict__ das54z;

    __device__ DasWavefieldPointer3D offset(int b, int spatial_size) const
    {
        DasWavefieldPointer3D out = *this;
        const int shift = b * spatial_size;
        out.exx += shift;
        out.eyy += shift;
        out.ezz += shift;
        out.sxx += shift;
        out.syy += shift;
        out.szz += shift;
        out.txx += shift;
        out.tyy += shift;
        out.tzz += shift;
        out.m_sxx_xf += shift;
        out.m_sxx_xb += shift;
        out.m_syy_yf += shift;
        out.m_syy_yb += shift;
        out.m_szz_zf += shift;
        out.m_szz_zb += shift;
        out.m_txx_yf += shift;
        out.m_txx_yb += shift;
        out.m_txx_zf += shift;
        out.m_txx_zb += shift;
        out.m_tyy_xf += shift;
        out.m_tyy_xb += shift;
        out.m_tyy_zf += shift;
        out.m_tyy_zb += shift;
        out.m_tzz_xf += shift;
        out.m_tzz_xb += shift;
        out.m_tzz_yf += shift;
        out.m_tzz_yb += shift;
        out.das35 += shift;
        out.das54x += shift;
        out.das54y += shift;
        out.das54z += shift;
        return out;
    }
};

struct DasWavefieldTensor3D {
    torch::Tensor exx_t, eyy_t, ezz_t, sxx_t, syy_t, szz_t, txx_t, tyy_t, tzz_t;
    torch::Tensor m_sxx_xf_t, m_sxx_xb_t, m_syy_yf_t, m_syy_yb_t, m_szz_zf_t, m_szz_zb_t;
    torch::Tensor m_txx_yf_t, m_txx_yb_t, m_txx_zf_t, m_txx_zb_t;
    torch::Tensor m_tyy_xf_t, m_tyy_xb_t, m_tyy_zf_t, m_tyy_zb_t;
    torch::Tensor m_tzz_xf_t, m_tzz_xb_t, m_tzz_yf_t, m_tzz_yb_t;
    torch::Tensor das35_t, das54x_t, das54y_t, das54z_t;
    bool allocated = false;

    void allocate(const torch::Tensor& like)
    {
        if (allocated) return;
        exx_t = torch::zeros_like(like);
        eyy_t = torch::zeros_like(like);
        ezz_t = torch::zeros_like(like);
        sxx_t = torch::zeros_like(like);
        syy_t = torch::zeros_like(like);
        szz_t = torch::zeros_like(like);
        txx_t = torch::zeros_like(like);
        tyy_t = torch::zeros_like(like);
        tzz_t = torch::zeros_like(like);
        m_sxx_xf_t = torch::zeros_like(like);
        m_sxx_xb_t = torch::zeros_like(like);
        m_syy_yf_t = torch::zeros_like(like);
        m_syy_yb_t = torch::zeros_like(like);
        m_szz_zf_t = torch::zeros_like(like);
        m_szz_zb_t = torch::zeros_like(like);
        m_txx_yf_t = torch::zeros_like(like);
        m_txx_yb_t = torch::zeros_like(like);
        m_txx_zf_t = torch::zeros_like(like);
        m_txx_zb_t = torch::zeros_like(like);
        m_tyy_xf_t = torch::zeros_like(like);
        m_tyy_xb_t = torch::zeros_like(like);
        m_tyy_zf_t = torch::zeros_like(like);
        m_tyy_zb_t = torch::zeros_like(like);
        m_tzz_xf_t = torch::zeros_like(like);
        m_tzz_xb_t = torch::zeros_like(like);
        m_tzz_yf_t = torch::zeros_like(like);
        m_tzz_yb_t = torch::zeros_like(like);
        das35_t = torch::zeros_like(like);
        das54x_t = torch::zeros_like(like);
        das54y_t = torch::zeros_like(like);
        das54z_t = torch::zeros_like(like);
        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        TORCH_CHECK(tensors.size() == 31, "DAS 3D expects 31 wavefield tensors");
        int i = 0;
        exx_t = tensors[i++];
        eyy_t = tensors[i++];
        ezz_t = tensors[i++];
        sxx_t = tensors[i++];
        syy_t = tensors[i++];
        szz_t = tensors[i++];
        txx_t = tensors[i++];
        tyy_t = tensors[i++];
        tzz_t = tensors[i++];
        m_sxx_xf_t = tensors[i++];
        m_sxx_xb_t = tensors[i++];
        m_syy_yf_t = tensors[i++];
        m_syy_yb_t = tensors[i++];
        m_szz_zf_t = tensors[i++];
        m_szz_zb_t = tensors[i++];
        m_txx_yf_t = tensors[i++];
        m_txx_yb_t = tensors[i++];
        m_txx_zf_t = tensors[i++];
        m_txx_zb_t = tensors[i++];
        m_tyy_xf_t = tensors[i++];
        m_tyy_xb_t = tensors[i++];
        m_tyy_zf_t = tensors[i++];
        m_tyy_zb_t = tensors[i++];
        m_tzz_xf_t = tensors[i++];
        m_tzz_xb_t = tensors[i++];
        m_tzz_yf_t = tensors[i++];
        m_tzz_yb_t = tensors[i++];
        das35_t = tensors[i++];
        das54x_t = tensors[i++];
        das54y_t = tensors[i++];
        das54z_t = tensors[i++];
        allocated = true;
    }

    DasWavefieldPointer3D view()
    {
        return {
            exx_t.data_ptr<float>(),
            eyy_t.data_ptr<float>(),
            ezz_t.data_ptr<float>(),
            sxx_t.data_ptr<float>(),
            syy_t.data_ptr<float>(),
            szz_t.data_ptr<float>(),
            txx_t.data_ptr<float>(),
            tyy_t.data_ptr<float>(),
            tzz_t.data_ptr<float>(),
            m_sxx_xf_t.data_ptr<float>(),
            m_sxx_xb_t.data_ptr<float>(),
            m_syy_yf_t.data_ptr<float>(),
            m_syy_yb_t.data_ptr<float>(),
            m_szz_zf_t.data_ptr<float>(),
            m_szz_zb_t.data_ptr<float>(),
            m_txx_yf_t.data_ptr<float>(),
            m_txx_yb_t.data_ptr<float>(),
            m_txx_zf_t.data_ptr<float>(),
            m_txx_zb_t.data_ptr<float>(),
            m_tyy_xf_t.data_ptr<float>(),
            m_tyy_xb_t.data_ptr<float>(),
            m_tyy_zf_t.data_ptr<float>(),
            m_tyy_zb_t.data_ptr<float>(),
            m_tzz_xf_t.data_ptr<float>(),
            m_tzz_xb_t.data_ptr<float>(),
            m_tzz_yf_t.data_ptr<float>(),
            m_tzz_yb_t.data_ptr<float>(),
            das35_t.data_ptr<float>(),
            das54x_t.data_ptr<float>(),
            das54y_t.data_ptr<float>(),
            das54z_t.data_ptr<float>(),
        };
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return {
            exx_t, eyy_t, ezz_t, sxx_t, syy_t, szz_t, txx_t, tyy_t, tzz_t,
            m_sxx_xf_t, m_sxx_xb_t, m_syy_yf_t, m_syy_yb_t, m_szz_zf_t, m_szz_zb_t,
            m_txx_yf_t, m_txx_yb_t, m_txx_zf_t, m_txx_zb_t,
            m_tyy_xf_t, m_tyy_xb_t, m_tyy_zf_t, m_tyy_zb_t,
            m_tzz_xf_t, m_tzz_xb_t, m_tzz_yf_t, m_tzz_yb_t,
            das35_t, das54x_t, das54y_t, das54z_t
        };
    }
};

inline float* das3d_field_ptr(DasWavefieldPointer3D wf, int field)
{
    switch (field) {
        case 0: return wf.exx;
        case 1: return wf.eyy;
        case 2: return wf.ezz;
        case 3: return wf.sxx;
        case 4: return wf.syy;
        case 5: return wf.szz;
        case 6: return wf.txx;
        case 7: return wf.tyy;
        case 8: return wf.tzz;
        case 9: return wf.m_sxx_xf;
        case 10: return wf.m_sxx_xb;
        case 11: return wf.m_syy_yf;
        case 12: return wf.m_syy_yb;
        case 13: return wf.m_szz_zf;
        case 14: return wf.m_szz_zb;
        case 15: return wf.m_txx_yf;
        case 16: return wf.m_txx_yb;
        case 17: return wf.m_txx_zf;
        case 18: return wf.m_txx_zb;
        case 19: return wf.m_tyy_xf;
        case 20: return wf.m_tyy_xb;
        case 21: return wf.m_tyy_zf;
        case 22: return wf.m_tyy_zb;
        case 23: return wf.m_tzz_xf;
        case 24: return wf.m_tzz_xb;
        case 25: return wf.m_tzz_yf;
        case 26: return wf.m_tzz_yb;
        case 27: return wf.das35;
        case 28: return wf.das54x;
        case 29: return wf.das54y;
        case 30: return wf.das54z;
        default: return nullptr;
    }
}

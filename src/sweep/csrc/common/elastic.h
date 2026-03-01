#pragma once
#include <torch/extension.h>
#include "context.h"

struct ElasticCPMLPointer{

    const float* __restrict__  ax;
    const float* __restrict__  bx;
    const float* __restrict__  axh;
    const float* __restrict__  bxh;

    const float* __restrict__  az;
    const float* __restrict__  bz;
    const float* __restrict__  azh;
    const float* __restrict__  bzh;

    const float* __restrict__  ay; // 3D
    const float* __restrict__  by; // 3D
    const float* __restrict__  ayh; // 3D
    const float* __restrict__  byh; // 3D

};

struct ElasticCPMLTensor{

    // =========================
    // Tensor ownership
    // =========================
    torch::Tensor ax_t, bx_t, axh_t, bxh_t;
    torch::Tensor az_t, bz_t, azh_t, bzh_t;
    torch::Tensor ay_t, by_t, ayh_t, byh_t; // 3D

    int dim = 3;
    bool allocated = false;

    // =========================
    // Allocate (from pml_vals)
    // =========================
    void allocate(
        const std::vector<torch::Tensor>& pml_vals,
        int dim_
    )
    {
        dim = dim_;

        int idx = 0;

        az_t   = pml_vals[idx++];
        bz_t   = pml_vals[idx++];
        azh_t  = pml_vals[idx++];
        bzh_t  = pml_vals[idx++];

        if (dim == 3) {
            ay_t   = pml_vals[idx++];
            by_t   = pml_vals[idx++];
            ayh_t  = pml_vals[idx++];
            byh_t  = pml_vals[idx++];
        }

        ax_t   = pml_vals[idx++];
        bx_t   = pml_vals[idx++];
        axh_t  = pml_vals[idx++];
        bxh_t  = pml_vals[idx++];

        allocated = true;
    }

    // =========================
    // Generate View
    // =========================
    ElasticCPMLPointer view() const
    {
        ElasticCPMLPointer v;
        v.az = az_t.data_ptr<float>();
        v.bz = bz_t.data_ptr<float>();
        v.azh = azh_t.data_ptr<float>();
        v.bzh = bzh_t.data_ptr<float>();

        if (dim == 3) {
            v.ay = ay_t.data_ptr<float>();
            v.by = by_t.data_ptr<float>();
            v.ayh = ayh_t.data_ptr<float>();
            v.byh = byh_t.data_ptr<float>();
        } else {
            v.ay = v.by = v.ayh = v.byh = nullptr;
        }
        v.ax = ax_t.data_ptr<float>();
        v.bx = bx_t.data_ptr<float>();
        v.axh = axh_t.data_ptr<float>();
        v.bxh = bxh_t.data_ptr<float>();

        return v;
    }
};

struct ElasticWavefieldPointer {

    // For forward/backward wavefield update
    float* __restrict__ vx;
    float* __restrict__ vy; // 3D
    float* __restrict__ vz;

    float* __restrict__ sxx;
    float* __restrict__ syy; // 3D
    float* __restrict__ szz;
    float* __restrict__ sxy; // 3D
    float* __restrict__ sxz;
    float* __restrict__ syz; // 3D

    // For PML Vx
    float* __restrict__ m_vxx;
    float* __restrict__ m_vxy;
    float* __restrict__ m_vxz;

    // For PML Vy
    float* __restrict__ m_vyx;
    float* __restrict__ m_vyy;
    float* __restrict__ m_vyz;

    // For PML Vz
    float* __restrict__ m_vzx;
    float* __restrict__ m_vzy;
    float* __restrict__ m_vzz;

    // For PML sxx
    float* __restrict__ m_sxxx;
    float* __restrict__ m_sxxy;
    float* __restrict__ m_sxxz;

    // For PML syy
    float* __restrict__ m_syyx;
    float* __restrict__ m_syyy;
    float* __restrict__ m_syyz;

    // For PML szz
    float* __restrict__ m_szzx;
    float* __restrict__ m_szzy;
    float* __restrict__ m_szzz;

    // For PML sxy
    float* __restrict__ m_sxyx;
    float* __restrict__ m_sxyy;
    float* __restrict__ m_sxyz;

    // For PML sxz
    float* __restrict__ m_sxzx;
    float* __restrict__ m_sxzy;
    float* __restrict__ m_sxzz;

    // For PML syz
    float* __restrict__ m_syzx;
    float* __restrict__ m_syzy;
    float* __restrict__ m_syzz;

    __device__ ElasticWavefieldPointer offset(
        int b,
        int spatial_size
    ) const {
        ElasticWavefieldPointer out = *this;

        int shift = b * spatial_size;

        out.vx += shift;
        if (out.vy) out.vy += shift;
        out.vz += shift;

        out.sxx += shift;
        if (out.syy) out.syy += shift;
        out.szz += shift;
        if (out.sxy) out.sxy += shift;
        out.sxz += shift;
        if (out.syz) out.syz += shift;

        // PML Vx
        out.m_vxx += shift;
        if (out.m_vxy) out.m_vxy += shift;
        out.m_vxz += shift;

        // PML Vy
        if (out.m_vyx) out.m_vyx += shift;
        if (out.m_vyy) out.m_vyy += shift;
        if (out.m_vyz) out.m_vyz += shift;

        // PML Vz
        out.m_vzx += shift;
        if (out.m_vzy) out.m_vzy += shift;
        out.m_vzz += shift;

        // PML sxx
        out.m_sxxx += shift;
        if (out.m_sxxy) out.m_sxxy += shift;
        out.m_sxxz += shift;

        // PML syy
        if (out.m_syyx) out.m_syyx += shift;
        if (out.m_syyy) out.m_syyy += shift;
        if (out.m_syyz) out.m_syyz += shift;

        // PML szz
        out.m_szzx += shift;
        if (out.m_szzy) out.m_szzy += shift;
        out.m_szzz += shift;

        // PML sxy
        if (out.m_sxyx) out.m_sxyx += shift;
        if (out.m_sxyy) out.m_sxyy += shift;
        if (out.m_sxyz) out.m_sxyz += shift;

        // PML sxz
        out.m_sxzx += shift;
        if (out.m_sxzy) out.m_sxzy += shift;
        out.m_sxzz += shift;

        // PML syz
        if (out.m_syzx) out.m_syzx += shift;
        if (out.m_syzy) out.m_syzy += shift;
        if (out.m_syzz) out.m_syzz += shift;

        return out;
    }

};

struct ElasticWavefieldTensor {

    // =========================
    // Tensor ownership
    // =========================
    torch::Tensor vx_t, vy_t, vz_t;
    torch::Tensor sxx_t, syy_t, szz_t, sxy_t, sxz_t, syz_t;

    // For boundary conditions
    torch::Tensor m_vxx_t, m_vxy_t, m_vxz_t; // Vx
    torch::Tensor m_vyx_t, m_vyy_t, m_vyz_t; // Vy
    torch::Tensor m_vzx_t, m_vzy_t, m_vzz_t; // Vz
    torch::Tensor m_sxxx_t, m_sxxy_t, m_sxxz_t; // sxx
    torch::Tensor m_syyx_t, m_syyy_t, m_syyz_t; // syy
    torch::Tensor m_szzx_t, m_szzy_t, m_szzz_t; // szz
    torch::Tensor m_sxyx_t, m_sxyy_t, m_sxyz_t; // sxy
    torch::Tensor m_sxzx_t, m_sxzy_t, m_sxzz_t; // sxz
    torch::Tensor m_syzx_t, m_syzy_t, m_syzz_t; // syz

    int dim = 2;
    bool use_pml = true;
    bool allocated = false;

    // =========================
    // Allocate (only once)
    // =========================
    void allocate(
        const torch::Tensor& vp,
        int dim_,
        bool use_pml_ = true
    )
    {
        if (allocated) return;

        dim = dim_;
        use_pml = use_pml_;

        // Always allocate wavefield
        vx_t = torch::zeros_like(vp);
        vz_t = torch::zeros_like(vp);

        sxx_t = torch::zeros_like(vp);
        szz_t = torch::zeros_like(vp);
        sxz_t = torch::zeros_like(vp);

        if (dim == 3) {
            // Wavefield
            vy_t = torch::zeros_like(vp);
            syy_t = torch::zeros_like(vp);
            sxy_t = torch::zeros_like(vp);
            syz_t = torch::zeros_like(vp);
        }

        if (use_pml) {

            // Vx
            m_vxx_t = torch::zeros_like(vp);
            m_vxz_t = torch::zeros_like(vp);
            // Vz
            m_vzx_t = torch::zeros_like(vp);
            m_vzz_t = torch::zeros_like(vp);
            // sxx
            m_sxxx_t = torch::zeros_like(vp);
            m_sxxz_t = torch::zeros_like(vp);
            // sxz
            m_sxzx_t = torch::zeros_like(vp);
            m_sxzz_t = torch::zeros_like(vp);
            // szz
            m_szzx_t = torch::zeros_like(vp);
            m_szzz_t = torch::zeros_like(vp);

            if (dim == 3) {
                // Boundary conditions

                // Vx
                m_vxy_t = torch::zeros_like(vp);
                // Vz
                m_vzy_t = torch::zeros_like(vp);

                // Vy
                m_vyx_t = torch::zeros_like(vp);
                m_vyy_t = torch::zeros_like(vp);
                m_vyz_t = torch::zeros_like(vp);

                // sxy
                m_sxyx_t = torch::zeros_like(vp);
                m_sxyy_t = torch::zeros_like(vp);
                m_sxyz_t = torch::zeros_like(vp);

                // syy
                m_syyx_t = torch::zeros_like(vp); // Adjoint
                m_syyy_t = torch::zeros_like(vp); // Adjoint
                m_syyz_t = torch::zeros_like(vp); // Adjoint

                // syz
                // m_syzx_t = torch::zeros_like(vp);
                m_syzy_t = torch::zeros_like(vp);
                m_syzz_t = torch::zeros_like(vp);

                // for adjoint
                m_sxxy_t = torch::zeros_like(vp);
                m_szzy_t = torch::zeros_like(vp);
            }
        }

        allocated = true;
    }

    // =========================
    // Generate View
    // =========================
    ElasticWavefieldPointer view()
    {
        ElasticWavefieldPointer v;

        v.vx = vx_t.data_ptr<float>();
        v.vz = vz_t.data_ptr<float>();

        v.sxx = sxx_t.data_ptr<float>();
        v.szz = szz_t.data_ptr<float>();
        v.sxz = sxz_t.data_ptr<float>();

        if (dim == 3) {
            v.vy = vy_t.data_ptr<float>();
            v.syy = syy_t.data_ptr<float>();
            v.sxy = sxy_t.data_ptr<float>();
            v.syz = syz_t.data_ptr<float>();
        }
        else {
            v.vy = nullptr;
            v.syy = nullptr;
            v.sxy = nullptr;
            v.syz = nullptr;
        }

        if (use_pml) {

            // Vx
            v.m_vxx = m_vxx_t.data_ptr<float>();
            v.m_vxz = m_vxz_t.data_ptr<float>();

            // Vz
            v.m_vzx = m_vzx_t.data_ptr<float>();
            v.m_vzz = m_vzz_t.data_ptr<float>();

            // sxx
            v.m_sxxx = m_sxxx_t.data_ptr<float>();
            v.m_sxxz = m_sxxz_t.data_ptr<float>();

            // sxz
            v.m_sxzx = m_sxzx_t.data_ptr<float>();
            v.m_sxzz = m_sxzz_t.data_ptr<float>();
            
            // szz
            v.m_szzx = m_szzx_t.data_ptr<float>();
            v.m_szzz = m_szzz_t.data_ptr<float>();

            // For adjoint
            v.m_sxxy = m_sxxy_t.data_ptr<float>();
            v.m_szzy = m_szzy_t.data_ptr<float>();

            if (dim == 3) {
                // Boundary conditions

                // Vx
                v.m_vxy = m_vxy_t.data_ptr<float>();

                // Vz
                v.m_vzy = m_vzy_t.data_ptr<float>();

                // Vy
                v.m_vyx = m_vyx_t.data_ptr<float>();
                v.m_vyy = m_vyy_t.data_ptr<float>();
                v.m_vyz = m_vyz_t.data_ptr<float>();

                // sxy
                v.m_sxyx = m_sxyx_t.data_ptr<float>();
                v.m_sxyy = m_sxyy_t.data_ptr<float>();
                v.m_sxyz = m_sxyz_t.data_ptr<float>();

                // syy
                v.m_syyx = m_syyx_t.data_ptr<float>(); // Adjoint
                v.m_syyy = m_syyy_t.data_ptr<float>(); // Adjoint
                v.m_syyz = m_syyz_t.data_ptr<float>();

                // syz
                // v.m_syzx = m_syzx_t.data_ptr<float>();
                v.m_syzy = m_syzy_t.data_ptr<float>();
                v.m_syzz = m_syzz_t.data_ptr<float>();
            }

        }
        else {
            v.m_vxx = v.m_vxz = nullptr;
            v.m_vzx = v.m_vzz = nullptr;
            v.m_sxxx = v.m_sxxz = nullptr;
            v.m_sxzx = v.m_sxzz = nullptr;
            v.m_szzx = v.m_szzz = nullptr;
        }

        return v;
    }

};
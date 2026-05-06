#pragma once

#include <torch/extension.h>

#include "context.h"

struct ElasticCPMLPointer {
    const float* __restrict__ ax;
    const float* __restrict__ bx;
    const float* __restrict__ axh;
    const float* __restrict__ bxh;

    const float* __restrict__ az;
    const float* __restrict__ bz;
    const float* __restrict__ azh;
    const float* __restrict__ bzh;

    const float* __restrict__ ay;   // 3D only
    const float* __restrict__ by;   // 3D only
    const float* __restrict__ ayh;  // 3D only
    const float* __restrict__ byh;  // 3D only
};

struct ElasticCPMLTensor {
    torch::Tensor ax_t, bx_t, axh_t, bxh_t;
    torch::Tensor az_t, bz_t, azh_t, bzh_t;
    torch::Tensor ay_t, by_t, ayh_t, byh_t;  // 3D only

    int dim = 3;
    bool allocated = false;

    void allocate(const std::vector<torch::Tensor>& pml_vals, int dim_)
    {
        dim = dim_;

        int idx = 0;
        az_t = pml_vals[idx++];
        bz_t = pml_vals[idx++];
        azh_t = pml_vals[idx++];
        bzh_t = pml_vals[idx++];

        if (dim == 3) {
            ay_t = pml_vals[idx++];
            by_t = pml_vals[idx++];
            ayh_t = pml_vals[idx++];
            byh_t = pml_vals[idx++];
        } else {
            ay_t = torch::Tensor();
            by_t = torch::Tensor();
            ayh_t = torch::Tensor();
            byh_t = torch::Tensor();
        }

        ax_t = pml_vals[idx++];
        bx_t = pml_vals[idx++];
        axh_t = pml_vals[idx++];
        bxh_t = pml_vals[idx++];
        allocated = true;
    }

    ElasticCPMLPointer view() const
    {
        ElasticCPMLPointer v{};
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
            v.ay = nullptr;
            v.by = nullptr;
            v.ayh = nullptr;
            v.byh = nullptr;
        }

        v.ax = ax_t.data_ptr<float>();
        v.bx = bx_t.data_ptr<float>();
        v.axh = axh_t.data_ptr<float>();
        v.bxh = bxh_t.data_ptr<float>();
        return v;
    }
};

struct ElasticWavefieldPointer {
    float* __restrict__ vx;
    float* __restrict__ vy;  // 3D only
    float* __restrict__ vz;

    float* __restrict__ sxx;
    float* __restrict__ syy;  // 3D only
    float* __restrict__ szz;
    float* __restrict__ sxy;  // 3D only
    float* __restrict__ sxz;
    float* __restrict__ syz;  // 3D only

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
    float* __restrict__ m_sxxy;
    float* __restrict__ m_sxxz;

    float* __restrict__ m_syyx;
    float* __restrict__ m_syyy;
    float* __restrict__ m_syyz;

    float* __restrict__ m_szzx;
    float* __restrict__ m_szzy;
    float* __restrict__ m_szzz;

    float* __restrict__ m_sxyx;
    float* __restrict__ m_sxyy;
    float* __restrict__ m_sxyz;

    float* __restrict__ m_sxzx;
    float* __restrict__ m_sxzy;
    float* __restrict__ m_sxzz;

    float* __restrict__ m_syzx;
    float* __restrict__ m_syzy;
    float* __restrict__ m_syzz;

    __device__ ElasticWavefieldPointer offset(int b, int spatial_size) const
    {
        ElasticWavefieldPointer out = *this;
        const int shift = b * spatial_size;

        out.vx += shift;
        if (out.vy) out.vy += shift;
        out.vz += shift;

        out.sxx += shift;
        if (out.syy) out.syy += shift;
        out.szz += shift;
        if (out.sxy) out.sxy += shift;
        out.sxz += shift;
        if (out.syz) out.syz += shift;

        out.m_vxx += shift;
        if (out.m_vxy) out.m_vxy += shift;
        out.m_vxz += shift;

        if (out.m_vyx) out.m_vyx += shift;
        if (out.m_vyy) out.m_vyy += shift;
        if (out.m_vyz) out.m_vyz += shift;

        out.m_vzx += shift;
        if (out.m_vzy) out.m_vzy += shift;
        out.m_vzz += shift;

        out.m_sxxx += shift;
        if (out.m_sxxy) out.m_sxxy += shift;
        out.m_sxxz += shift;

        if (out.m_syyx) out.m_syyx += shift;
        if (out.m_syyy) out.m_syyy += shift;
        if (out.m_syyz) out.m_syyz += shift;

        out.m_szzx += shift;
        if (out.m_szzy) out.m_szzy += shift;
        out.m_szzz += shift;

        if (out.m_sxyx) out.m_sxyx += shift;
        if (out.m_sxyy) out.m_sxyy += shift;
        if (out.m_sxyz) out.m_sxyz += shift;

        out.m_sxzx += shift;
        if (out.m_sxzy) out.m_sxzy += shift;
        out.m_sxzz += shift;

        if (out.m_syzx) out.m_syzx += shift;
        if (out.m_syzy) out.m_syzy += shift;
        if (out.m_syzz) out.m_syzz += shift;

        return out;
    }
};

struct ElasticWavefieldTensor {
    torch::Tensor vx_t, vy_t, vz_t;
    torch::Tensor sxx_t, syy_t, szz_t, sxy_t, sxz_t, syz_t;

    torch::Tensor m_vxx_t, m_vxy_t, m_vxz_t;
    torch::Tensor m_vyx_t, m_vyy_t, m_vyz_t;
    torch::Tensor m_vzx_t, m_vzy_t, m_vzz_t;

    torch::Tensor m_sxxx_t, m_sxxy_t, m_sxxz_t;
    torch::Tensor m_syyx_t, m_syyy_t, m_syyz_t;
    torch::Tensor m_szzx_t, m_szzy_t, m_szzz_t;
    torch::Tensor m_sxyx_t, m_sxyy_t, m_sxyz_t;
    torch::Tensor m_sxzx_t, m_sxzy_t, m_sxzz_t;
    torch::Tensor m_syzx_t, m_syzy_t, m_syzz_t;

    int dim = 2;
    bool use_pml = true;
    bool allocated = false;

    void allocate(const torch::Tensor& vp, int dim_, bool use_pml_ = true)
    {
        if (allocated) return;

        dim = dim_;
        use_pml = use_pml_;

        vx_t = torch::zeros_like(vp);
        vz_t = torch::zeros_like(vp);
        sxx_t = torch::zeros_like(vp);
        szz_t = torch::zeros_like(vp);
        sxz_t = torch::zeros_like(vp);

        reset_optional_3d();
        reset_optional_pml();

        if (dim == 3) {
            vy_t = torch::zeros_like(vp);
            syy_t = torch::zeros_like(vp);
            sxy_t = torch::zeros_like(vp);
            syz_t = torch::zeros_like(vp);
        }

        if (use_pml) {
            allocate_common_pml(vp);

            if (dim == 3) {
                allocate_3d_only_pml(vp);
            }
        }

        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors, bool use_pml_ = true)
    {
        int i = 0;
        use_pml = use_pml_;

        reset_optional_3d();
        reset_optional_pml();

        if (tensors.size() == 15) {
            dim = 2;

            vx_t = tensors[i++];
            vz_t = tensors[i++];
            sxx_t = tensors[i++];
            szz_t = tensors[i++];
            sxz_t = tensors[i++];

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
        } else {
            dim = 3;

            vx_t = tensors[i++];
            vy_t = tensors[i++];
            vz_t = tensors[i++];
            sxx_t = tensors[i++];
            syy_t = tensors[i++];
            szz_t = tensors[i++];
            sxy_t = tensors[i++];
            sxz_t = tensors[i++];
            syz_t = tensors[i++];

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
                m_sxxy_t = tensors[i++];
                m_sxxz_t = tensors[i++];

                m_syyx_t = tensors[i++];
                m_syyy_t = tensors[i++];
                m_syyz_t = tensors[i++];

                m_szzx_t = tensors[i++];
                m_szzy_t = tensors[i++];
                m_szzz_t = tensors[i++];

                m_sxyx_t = tensors[i++];
                m_sxyy_t = tensors[i++];
                m_sxyz_t = tensors[i++];

                m_sxzx_t = tensors[i++];
                m_sxzy_t = tensors[i++];
                m_sxzz_t = tensors[i++];

                m_syzx_t = tensors[i++];
                m_syzy_t = tensors[i++];
                m_syzz_t = tensors[i++];
            }
        }

        allocated = true;
    }

    ElasticWavefieldPointer view()
    {
        ElasticWavefieldPointer v{};

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
        } else {
            v.vy = nullptr;
            v.syy = nullptr;
            v.sxy = nullptr;
            v.syz = nullptr;
        }

        if (use_pml) {
            bind_common_pml_view(v);

            if (dim == 3) {
                bind_3d_only_pml_view(v);
            }
        } else {
            clear_pml_view(v);
        }

        return v;
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        if (dim == 3) {
            return {
                vx_t, vy_t, vz_t, sxx_t, syy_t, szz_t, sxy_t, sxz_t, syz_t,
                m_vxx_t, m_vxy_t, m_vxz_t, m_vyx_t, m_vyy_t, m_vyz_t, m_vzx_t, m_vzy_t, m_vzz_t,
                m_sxxx_t, m_sxxy_t, m_sxxz_t, m_syyx_t, m_syyy_t, m_syyz_t,
                m_szzx_t, m_szzy_t, m_szzz_t, m_sxyx_t, m_sxyy_t, m_sxyz_t,
                m_sxzx_t, m_sxzy_t, m_sxzz_t, m_syzx_t, m_syzy_t, m_syzz_t
            };
        }

        return {
            vx_t, vz_t, sxx_t, szz_t, sxz_t,
            m_vxx_t, m_vxz_t, m_vzx_t, m_vzz_t,
            m_sxxx_t, m_sxxz_t, m_szzx_t, m_szzz_t, m_sxzx_t, m_sxzz_t
        };
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        return checkpoint_tensors();
    }

private:
    void reset_optional_3d()
    {
        vy_t = torch::Tensor();
        syy_t = torch::Tensor();
        sxy_t = torch::Tensor();
        syz_t = torch::Tensor();
    }

    void reset_optional_pml()
    {
        m_vxx_t = torch::Tensor();
        m_vxy_t = torch::Tensor();
        m_vxz_t = torch::Tensor();
        m_vyx_t = torch::Tensor();
        m_vyy_t = torch::Tensor();
        m_vyz_t = torch::Tensor();
        m_vzx_t = torch::Tensor();
        m_vzy_t = torch::Tensor();
        m_vzz_t = torch::Tensor();

        m_sxxx_t = torch::Tensor();
        m_sxxy_t = torch::Tensor();
        m_sxxz_t = torch::Tensor();
        m_syyx_t = torch::Tensor();
        m_syyy_t = torch::Tensor();
        m_syyz_t = torch::Tensor();
        m_szzx_t = torch::Tensor();
        m_szzy_t = torch::Tensor();
        m_szzz_t = torch::Tensor();
        m_sxyx_t = torch::Tensor();
        m_sxyy_t = torch::Tensor();
        m_sxyz_t = torch::Tensor();
        m_sxzx_t = torch::Tensor();
        m_sxzy_t = torch::Tensor();
        m_sxzz_t = torch::Tensor();
        m_syzx_t = torch::Tensor();
        m_syzy_t = torch::Tensor();
        m_syzz_t = torch::Tensor();
    }

    void allocate_common_pml(const torch::Tensor& like)
    {
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

    void allocate_3d_only_pml(const torch::Tensor& like)
    {
        m_vxy_t = torch::zeros_like(like);
        m_vzy_t = torch::zeros_like(like);

        m_vyx_t = torch::zeros_like(like);
        m_vyy_t = torch::zeros_like(like);
        m_vyz_t = torch::zeros_like(like);

        m_sxyx_t = torch::zeros_like(like);
        m_sxyy_t = torch::zeros_like(like);
        m_sxyz_t = torch::zeros_like(like);

        m_syyx_t = torch::zeros_like(like);
        m_syyy_t = torch::zeros_like(like);
        m_syyz_t = torch::zeros_like(like);

        m_sxzy_t = torch::zeros_like(like);
        m_syzx_t = torch::zeros_like(like);
        m_syzy_t = torch::zeros_like(like);
        m_syzz_t = torch::zeros_like(like);

        m_sxxy_t = torch::zeros_like(like);
        m_szzy_t = torch::zeros_like(like);
    }

    void bind_common_pml_view(ElasticWavefieldPointer& v)
    {
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
    }

    void bind_3d_only_pml_view(ElasticWavefieldPointer& v)
    {
        v.m_vxy = m_vxy_t.data_ptr<float>();
        v.m_vyx = m_vyx_t.data_ptr<float>();
        v.m_vyy = m_vyy_t.data_ptr<float>();
        v.m_vyz = m_vyz_t.data_ptr<float>();
        v.m_vzy = m_vzy_t.data_ptr<float>();

        v.m_sxxy = m_sxxy_t.data_ptr<float>();
        v.m_syyx = m_syyx_t.data_ptr<float>();
        v.m_syyy = m_syyy_t.data_ptr<float>();
        v.m_syyz = m_syyz_t.data_ptr<float>();
        v.m_szzy = m_szzy_t.data_ptr<float>();

        v.m_sxyx = m_sxyx_t.data_ptr<float>();
        v.m_sxyy = m_sxyy_t.data_ptr<float>();
        v.m_sxyz = m_sxyz_t.data_ptr<float>();

        v.m_sxzy = m_sxzy_t.data_ptr<float>();
        v.m_syzx = m_syzx_t.data_ptr<float>();
        v.m_syzy = m_syzy_t.data_ptr<float>();
        v.m_syzz = m_syzz_t.data_ptr<float>();
    }

    void clear_pml_view(ElasticWavefieldPointer& v)
    {
        v.m_vxx = nullptr;
        v.m_vxy = nullptr;
        v.m_vxz = nullptr;
        v.m_vyx = nullptr;
        v.m_vyy = nullptr;
        v.m_vyz = nullptr;
        v.m_vzx = nullptr;
        v.m_vzy = nullptr;
        v.m_vzz = nullptr;

        v.m_sxxx = nullptr;
        v.m_sxxy = nullptr;
        v.m_sxxz = nullptr;
        v.m_syyx = nullptr;
        v.m_syyy = nullptr;
        v.m_syyz = nullptr;
        v.m_szzx = nullptr;
        v.m_szzy = nullptr;
        v.m_szzz = nullptr;
        v.m_sxyx = nullptr;
        v.m_sxyy = nullptr;
        v.m_sxyz = nullptr;
        v.m_sxzx = nullptr;
        v.m_sxzy = nullptr;
        v.m_sxzz = nullptr;
        v.m_syzx = nullptr;
        v.m_syzy = nullptr;
        v.m_syzz = nullptr;
    }
};

struct ElasticAdjointWorkspaceTensor {
    torch::Tensor qxx_t, qxy_t, qxz_t, qyx_t, qyy_t, qyz_t, qzx_t, qzy_t, qzz_t;
    torch::Tensor pxx_t, pxy_t, pxz_t, pyx_t, pyy_t, pyz_t, pzx_t, pzy_t, pzz_t;

    int dim = 2;
    bool allocated = false;

    void allocate(const torch::Tensor& like, int dim_ = 2)
    {
        if (allocated) return;

        dim = dim_;
        qxx_t = torch::zeros_like(like);
        pxx_t = torch::zeros_like(like);

        if (dim == 2) {
            qzz_t = torch::zeros_like(like);
            qxz_t = torch::zeros_like(like);
            qzx_t = torch::zeros_like(like);
            pzz_t = torch::zeros_like(like);
            pxz_t = torch::zeros_like(like);
            pzx_t = torch::zeros_like(like);
        } else {
            qxy_t = torch::zeros_like(like);
            qxz_t = torch::zeros_like(like);
            qyx_t = torch::zeros_like(like);
            qyy_t = torch::zeros_like(like);
            qyz_t = torch::zeros_like(like);
            qzx_t = torch::zeros_like(like);
            qzy_t = torch::zeros_like(like);
            qzz_t = torch::zeros_like(like);

            pxy_t = torch::zeros_like(like);
            pxz_t = torch::zeros_like(like);
            pyx_t = torch::zeros_like(like);
            pyy_t = torch::zeros_like(like);
            pyz_t = torch::zeros_like(like);
            pzx_t = torch::zeros_like(like);
            pzy_t = torch::zeros_like(like);
            pzz_t = torch::zeros_like(like);
        }
        allocated = true;
    }

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        int i = 0;

        if (tensors.size() == 8) {
            dim = 2;
            qxx_t = tensors[i++];
            qzz_t = tensors[i++];
            qxz_t = tensors[i++];
            qzx_t = tensors[i++];
            pxx_t = tensors[i++];
            pzz_t = tensors[i++];
            pxz_t = tensors[i++];
            pzx_t = tensors[i++];
        } else if (tensors.size() == 18) {
            dim = 3;
            qxx_t = tensors[i++];
            qxy_t = tensors[i++];
            qxz_t = tensors[i++];
            qyx_t = tensors[i++];
            qyy_t = tensors[i++];
            qyz_t = tensors[i++];
            qzx_t = tensors[i++];
            qzy_t = tensors[i++];
            qzz_t = tensors[i++];
            pxx_t = tensors[i++];
            pxy_t = tensors[i++];
            pxz_t = tensors[i++];
            pyx_t = tensors[i++];
            pyy_t = tensors[i++];
            pyz_t = tensors[i++];
            pzx_t = tensors[i++];
            pzy_t = tensors[i++];
            pzz_t = tensors[i++];
        } else {
            TORCH_CHECK(false, "Elastic adjoint workspace expects 8 tensors (2D) or 18 tensors (3D)");
        }

        allocated = true;
    }
};

inline void init_adjoint_workspace(
    ElasticAdjointWorkspaceTensor& workspace,
    const std::vector<torch::Tensor>& tensors,
    const torch::Tensor& like,
    int dim
)
{
    if (!tensors.empty())
        workspace.bind(tensors);
    else
        workspace.allocate(like, dim);
}

inline void zero_wavefield_state(ElasticWavefieldTensor& wf)
{
    if (wf.dim == 3 && !wf.m_syzx_t.defined())
        wf.m_syzx_t = torch::zeros_like(wf.vx_t);

    wf.vx_t.zero_();
    wf.vz_t.zero_();
    wf.sxx_t.zero_();
    wf.szz_t.zero_();
    wf.sxz_t.zero_();

    if (wf.dim == 3) {
        wf.vy_t.zero_();
        wf.syy_t.zero_();
        wf.sxy_t.zero_();
        wf.syz_t.zero_();
    }

    if (!wf.use_pml)
        return;

    wf.m_vxx_t.zero_();
    wf.m_vxz_t.zero_();
    wf.m_vzx_t.zero_();
    wf.m_vzz_t.zero_();
    wf.m_sxxx_t.zero_();
    wf.m_sxxz_t.zero_();
    wf.m_szzx_t.zero_();
    wf.m_szzz_t.zero_();
    wf.m_sxzx_t.zero_();
    wf.m_sxzz_t.zero_();

    if (wf.dim == 3) {
        wf.m_vxy_t.zero_();
        wf.m_vyx_t.zero_();
        wf.m_vyy_t.zero_();
        wf.m_vyz_t.zero_();
        wf.m_vzy_t.zero_();
        wf.m_sxxy_t.zero_();
        wf.m_syyx_t.zero_();
        wf.m_syyy_t.zero_();
        wf.m_syyz_t.zero_();
        wf.m_szzy_t.zero_();
        wf.m_sxyx_t.zero_();
        wf.m_sxyy_t.zero_();
        wf.m_sxyz_t.zero_();
        wf.m_sxzy_t.zero_();
        if (wf.m_syzx_t.defined()) wf.m_syzx_t.zero_();
        wf.m_syzy_t.zero_();
        wf.m_syzz_t.zero_();
    }
}

inline float* elastic_field_ptr(ElasticWavefieldPointer& wf, int dim, int idx)
{
    if (dim == 2) {
        switch (idx) {
            case 0: return wf.vx;
            case 1: return wf.vz;
            case 2: return wf.sxx;
            case 3: return wf.szz;
            case 4: return wf.sxz;
            case 5: return wf.m_vxx;
            case 6: return wf.m_vxz;
            case 7: return wf.m_vzx;
            case 8: return wf.m_vzz;
            case 9: return wf.m_sxxx;
            case 10: return wf.m_sxxz;
            case 11: return wf.m_szzx;
            case 12: return wf.m_szzz;
            case 13: return wf.m_sxzx;
            case 14: return wf.m_sxzz;
            default: return nullptr;
        }
    }

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
        case 9: return wf.m_vxx;
        case 10: return wf.m_vxy;
        case 11: return wf.m_vxz;
        case 12: return wf.m_vyx;
        case 13: return wf.m_vyy;
        case 14: return wf.m_vyz;
        case 15: return wf.m_vzx;
        case 16: return wf.m_vzy;
        case 17: return wf.m_vzz;
        case 18: return wf.m_sxxx;
        case 19: return wf.m_szzz;
        case 20: return wf.m_sxyx;
        case 21: return wf.m_sxyy;
        case 22: return wf.m_sxzx;
        case 23: return wf.m_sxzz;
        case 24: return wf.m_syyy;
        case 25: return wf.m_syzy;
        case 26: return wf.m_syzz;
        default: return nullptr;
    }
}

#pragma once
#include <torch/extension.h>
#include "context.h"

struct AcousticCPMLPointer {

    const float* __restrict__ ax;
    const float* __restrict__ bx;
    const float* __restrict__ dbxdx;

    const float* __restrict__ ay;
    const float* __restrict__ by;
    const float* __restrict__ dbydy;

    const float* __restrict__ az;
    const float* __restrict__ bz;
    const float* __restrict__ dbzdz;
};

struct AcousticCPMLTensor {

    // =========================
    // Tensor ownership
    // =========================
    torch::Tensor ax_t, bx_t, dbxdx_t;
    torch::Tensor ay_t, by_t, dbydy_t;
    torch::Tensor az_t, bz_t, dbzdz_t;

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

        az_t    = pml_vals[idx++];
        bz_t    = pml_vals[idx++];
        dbzdz_t = pml_vals[idx++];

        if (dim == 3) {
            ay_t    = pml_vals[idx++];
            by_t    = pml_vals[idx++];
            dbydy_t = pml_vals[idx++];
        }

        ax_t    = pml_vals[idx++];
        bx_t    = pml_vals[idx++];
        dbxdx_t = pml_vals[idx++];
    }

    // =========================
    // Generate View
    // =========================
    AcousticCPMLPointer view() const
    {
        AcousticCPMLPointer v;

        v.ax     = ax_t.data_ptr<float>();
        v.bx     = bx_t.data_ptr<float>();
        v.dbxdx  = dbxdx_t.data_ptr<float>();

        if (dim == 3) {
            v.ay     = ay_t.data_ptr<float>();
            v.by     = by_t.data_ptr<float>();
            v.dbydy  = dbydy_t.data_ptr<float>();
        } else {
            v.ay = v.by = v.dbydy = nullptr;
        }

        v.az     = az_t.data_ptr<float>();
        v.bz     = bz_t.data_ptr<float>();
        v.dbzdz  = dbzdz_t.data_ptr<float>();

        return v;
    }
};


struct AcousticWavefieldPointer {

    float* __restrict__ u_prev;
    float* __restrict__ u_now;
    float* __restrict__ u_next;

    float* __restrict__ psix;
    float* __restrict__ psiy;
    float* __restrict__ psiz;

    float* __restrict__ zetax;
    float* __restrict__ zetay;
    float* __restrict__ zetaz;

    __device__ AcousticWavefieldPointer offset(
        int b,
        int spatial_size
    ) const {

        AcousticWavefieldPointer out = *this;
 
        int shift = b * spatial_size;

        out.u_prev += shift;
        out.u_now  += shift;
        out.u_next += shift;

        if (out.psix)  out.psix  += shift;
        if (out.psiy)  out.psiy  += shift;
        if (out.psiz)  out.psiz  += shift;

        if (out.zetax) out.zetax += shift;
        if (out.zetay) out.zetay += shift;
        if (out.zetaz) out.zetaz += shift;

        return out;
    }

};

struct AcousticWavefieldTensor {

    // =========================
    // Tensor ownership
    // =========================
    torch::Tensor u_prev_t;
    torch::Tensor u_now_t;
    torch::Tensor u_next_t;

    torch::Tensor psix_t;
    torch::Tensor psiy_t;
    torch::Tensor psiz_t;

    torch::Tensor zetax_t;
    torch::Tensor zetay_t;
    torch::Tensor zetaz_t;

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
        u_prev_t = torch::zeros_like(vp);
        u_now_t  = torch::zeros_like(vp);
        u_next_t = torch::zeros_like(vp);

        if (use_pml) {

            psix_t  = torch::zeros_like(vp);
            psiz_t  = torch::zeros_like(vp);
            zetax_t = torch::zeros_like(vp);
            zetaz_t = torch::zeros_like(vp);

            if (dim == 3) {
                psiy_t  = torch::zeros_like(vp);
                zetay_t = torch::zeros_like(vp);
            }
        }

        allocated = true;
    }

    // =========================
    // Generate View
    // =========================
    AcousticWavefieldPointer view()
    {
        AcousticWavefieldPointer v;

        v.u_prev = u_prev_t.data_ptr<float>();
        v.u_now  = u_now_t.data_ptr<float>();
        v.u_next = u_next_t.data_ptr<float>();

        if (use_pml) {
            v.psix  = psix_t.data_ptr<float>();
            v.psiz  = psiz_t.data_ptr<float>();
            v.zetax = zetax_t.data_ptr<float>();
            v.zetaz = zetaz_t.data_ptr<float>();

            if (dim == 3) {
                v.psiy  = psiy_t.data_ptr<float>();
                v.zetay = zetay_t.data_ptr<float>();
            } else {
                v.psiy = nullptr;
                v.zetay = nullptr;
            }
        } else {
            v.psix = v.psiy = v.psiz = nullptr;
            v.zetax = v.zetay = v.zetaz = nullptr;
        }

        return v;
    }

    void swap()
    {
        std::swap(u_prev_t, u_now_t);
        std::swap(u_now_t,  u_next_t);
    }

};
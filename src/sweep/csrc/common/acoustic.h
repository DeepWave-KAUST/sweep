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

struct AcousticBoundaryPointer {

    float* __restrict__ left;
    float* __restrict__ right;

    float* __restrict__ front;
    float* __restrict__ back;

    float* __restrict__ bottom;
    float* __restrict__ top;

    float* __restrict__ last_two;
};

struct AcousticBoundarySaver {

    torch::Tensor left_t, right_t;
    torch::Tensor front_t, back_t;
    torch::Tensor bottom_t, top_t;
    torch::Tensor last_two_t;

    bool enabled = false;
    int dim = 3;

    void allocate(
        bool use_boundary_saving,
        int dim_,
        SolverContext ctx,
        const torch::Tensor& vp
    )
    {
        enabled = use_boundary_saving;
        dim = dim_;

        if (!enabled) return;

        auto options = vp.options();

        if (dim == 3) {

            left_t  = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.ny, ctx.M}, options);
            right_t = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.ny, ctx.M}, options);

            front_t = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.M, ctx.nx}, options);
            back_t  = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.M, ctx.nx}, options);

            bottom_t = torch::zeros({ctx.nt, ctx.B, ctx.M, ctx.ny, ctx.nx}, options);
            top_t    = torch::zeros({ctx.nt, ctx.B, ctx.M, ctx.ny, ctx.nx}, options);

            last_two_t = torch::zeros({2, ctx.B, 1, ctx.nz, ctx.ny, ctx.nx}, options);

        } else {

            left_t  = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.M}, options);
            right_t = torch::zeros({ctx.nt, ctx.B, ctx.nz, ctx.M}, options);

            bottom_t = torch::zeros({ctx.nt, ctx.B, ctx.M, ctx.nx}, options);
            top_t    = torch::zeros({ctx.nt, ctx.B, ctx.M, ctx.nx}, options);

            last_two_t = torch::zeros({2, ctx.B, 1, ctx.nz, ctx.nx}, options);

            front_t = torch::Tensor();
            back_t  = torch::Tensor();
        }
    }

    AcousticBoundaryPointer view()
    {
        AcousticBoundaryPointer v{};

        if (!enabled) return v;

        v.left  = left_t.data_ptr<float>();
        v.right = right_t.data_ptr<float>();

        if (dim == 3) {
            v.front = front_t.data_ptr<float>();
            v.back  = back_t.data_ptr<float>();
        } else {
            v.front = nullptr;
            v.back  = nullptr;
        }

        v.bottom = bottom_t.data_ptr<float>();
        v.top    = top_t.data_ptr<float>();

        v.last_two = last_two_t.data_ptr<float>();

        return v;

    }

    void load_from_vector(const std::vector<torch::Tensor>& u_boundary)
        {
            if (!enabled)
                throw std::runtime_error("Boundary saving not enabled.");

            if (dim == 2) {

                if (u_boundary.size() != 4)
                    throw std::runtime_error("2D boundary expects 4 tensors.");

                top_t.copy_(u_boundary[0]);
                bottom_t.copy_(u_boundary[1]);
                left_t.copy_(u_boundary[2]);
                right_t.copy_(u_boundary[3]);

            } else { // 3D

                if (u_boundary.size() != 6)
                    throw std::runtime_error("3D boundary expects 6 tensors.");

                top_t.copy_(u_boundary[0]);
                bottom_t.copy_(u_boundary[1]);

                front_t.copy_(u_boundary[2]);
                back_t.copy_(u_boundary[3]);

                left_t.copy_(u_boundary[4]);
                right_t.copy_(u_boundary[5]);
            }
        }

};

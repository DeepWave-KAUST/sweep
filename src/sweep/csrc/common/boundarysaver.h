#pragma once
#include <torch/extension.h>
#include "context.h"

struct GeneralBoundaryPointer {

    float* __restrict__ left;
    float* __restrict__ right;

    float* __restrict__ front;
    float* __restrict__ back;

    float* __restrict__ bottom;
    float* __restrict__ top;

    float* __restrict__ last_two;
};

struct GeneralBoundarySaver {

    torch::Tensor left_t, right_t;
    torch::Tensor front_t, back_t;
    torch::Tensor bottom_t, top_t;
    torch::Tensor last_two_t;

    bool enabled = false;
    int dim = 3;
    int nvar = 1;

    void allocate(
        bool use_boundary_saving,
        int dim_,
        int nvar_,
        SolverContext ctx,
        const torch::Tensor& ref_tensor,
        int width = -1
    )
    {
        enabled = use_boundary_saving;
        dim = dim_;
        nvar = nvar_;

        if (!enabled) return;

        if (width < 0) {
            width = ctx.M;
        }

        auto options = ref_tensor.options();

        if (dim == 3) {

            left_t  = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, ctx.ny, width}, options);
            right_t = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, ctx.ny, width}, options);

            front_t = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, width, ctx.nx}, options);
            back_t  = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, width, ctx.nx}, options);

            bottom_t = torch::zeros({nvar, ctx.nt, ctx.B, width, ctx.ny, ctx.nx}, options);
            top_t    = torch::zeros({nvar, ctx.nt, ctx.B, width, ctx.ny, ctx.nx}, options);

            last_two_t = torch::zeros({nvar, 2, ctx.B, 1, ctx.nz, ctx.ny, ctx.nx}, options);

        } else {

            left_t  = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, width}, options);
            right_t = torch::zeros({nvar, ctx.nt, ctx.B, ctx.nz, width}, options);

            bottom_t = torch::zeros({nvar, ctx.nt, ctx.B, width, ctx.nx}, options);
            top_t    = torch::zeros({nvar, ctx.nt, ctx.B, width, ctx.nx}, options);

            last_two_t = torch::zeros({nvar, 2, ctx.B, 1, ctx.nz, ctx.nx}, options);

            front_t = torch::Tensor();
            back_t  = torch::Tensor();
        }
    }

    GeneralBoundaryPointer view()
    {
        GeneralBoundaryPointer v{};

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

    void load_from_vector(
        const std::vector<torch::Tensor>& u_boundary,
        const torch::Tensor& ref_tensor
        )
        {
            if (!enabled)
                throw std::runtime_error("Boundary saving not enabled.");

            auto device = ref_tensor.device();
            
            auto move = [&](const torch::Tensor& t) {
                return t.device() == device ?
                    t :
                    t.to(device, /*non_blocking=*/true);
            };

            if (dim == 2) {

                if (u_boundary.size() != 4)
                    throw std::runtime_error("2D boundary expects 4 tensors.");

                top_t.copy_(move(u_boundary[0]));
                bottom_t.copy_(move(u_boundary[1]));
                left_t.copy_(move(u_boundary[2]));
                right_t.copy_(move(u_boundary[3]));

            } else { // 3D

                if (u_boundary.size() != 6)
                    throw std::runtime_error("3D boundary expects 6 tensors.");

                top_t.copy_(move(u_boundary[0]));
                bottom_t.copy_(move(u_boundary[1]));

                front_t.copy_(move(u_boundary[2]));
                back_t.copy_(move(u_boundary[3]));

                left_t.copy_(move(u_boundary[4]));
                right_t.copy_(move(u_boundary[5]));
            }
        }

};


__global__ void save_boundary_kernel(
    const float* __restrict__ u,   // (B, nz, nx)
    float* __restrict__ top,       // (nt, B, n, nx)
    float* __restrict__ bottom,    // (nt, B, n, nx)
    float* __restrict__ left,      // (nt, B, nz, n)
    float* __restrict__ right,     // (nt, B, nz, n)
    int it,
    int width,
    SolverContext solver
);

__global__ void restore_boundary_kernel(
    float* __restrict__ u,        // (B, nz, nx)
    const float* __restrict__ top,
    const float* __restrict__ bottom,
    const float* __restrict__ left,
    const float* __restrict__ right,
    int it,
    int width,
    SolverContext solver
);


__global__ void save_boundary_kernel_3d(
    const float* __restrict__ u,    // (B, nz, ny, nx)

    float* __restrict__ front,      // (nt, B, M, ny, nx)
    float* __restrict__ back,

    float* __restrict__ top,        // (nt, B, nz, M, nx)
    float* __restrict__ bottom,

    float* __restrict__ left,       // (nt, B, nz, ny, M)
    float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
);

__global__ void restore_boundary_kernel_3d(
    float* __restrict__ u,        // (B, nz, ny, nx)

    const float* __restrict__ top,   // (nt, B, n, ny, nx)
    const float* __restrict__ bottom,

    const float* __restrict__ front,     // (nt, B, nz, n, nx)
    const float* __restrict__ back,

    const float* __restrict__ left,    // (nt, B, nz, ny, n)
    const float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
);
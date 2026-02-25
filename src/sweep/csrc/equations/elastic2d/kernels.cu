#include "kernels.cuh"
#include "../../common/context.h"

__global__ void calculate_elastic_grad(

    ElasticWavefieldPointer adjoint,

    const float* __restrict__ vx_x,
    const float* __restrict__ vx_z,
    const float* __restrict__ vz_x,
    const float* __restrict__ vz_z,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,             // (B, nz, nx)
    float* __restrict__ grad_vs,             // (B, nz, nx)
    float* __restrict__ grad_rho,            // (B, nz, nx)

    SolverContext solver
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* fvx_x  = vx_x + b * spatial_size;
    const float* fvx_z  = vx_z + b * spatial_size;
    const float* fvz_x  = vz_x + b * spatial_size;
    const float* fvz_z  = vz_z + b * spatial_size;

    float*       grad_vp_b       = grad_vp       + b * spatial_size;
    float*       grad_vs_b       = grad_vs       + b * spatial_size;
    float*       grad_rho_b      = grad_rho      + b * spatial_size;

    const float* vp_b         = vp         + b * spatial_size;
    const float* vs_b         = vs         + b * spatial_size;
    const float* rho_b        = rho        + b * spatial_size;

    auto a = adjoint.offset(b, spatial_size);

    float grad_lambda = (a.sxx[idx] + a.szz[idx]) * (fvx_x[idx] + fvz_z[idx]);
    float grad_mu = 2*(a.sxx[idx] * fvx_x[idx] + a.szz[idx] * fvz_z[idx]) + a.sxz[idx] * (fvx_z[idx] + fvz_x[idx]);
    
    grad_vp_b[idx] += -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    grad_vs_b[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                         2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;


}
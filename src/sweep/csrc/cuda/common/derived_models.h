#pragma once
// Model coefficients the drivers derive from the bound models once per call
// (Lame parameters from vp/vs/rho, VTI stiffness from vp/eps/delta/rho, 1/z
// for VRZ).  The propagator hands the driver ``cuda_layout.derived_model_nvar``
// model-shaped slots as ``derived_models`` (``torch.empty``: every cell is
// written here before anything reads it); one fused kernel per family fills
// them, reading each model once.  Unbound slots (callers that predate the
// contract) are allocated here, uninitialised, and filled the same way -- the
// kernel is the only implementation of the formulas.
//
// Bit-exactness: each kernel evaluates the formula in the order and rounding
// of the torch expression it replaced (see derived_models.cu), so records and
// gradients are unchanged to the bit.
#include <torch/extension.h>
#include <vector>
#include "cudautils.h"

namespace derived {

enum LameSlot : int { MU = 0, LAMBDA, N_LAME };
enum VtiSlot : int { C11 = 0, C13, C33, INV_RHO, N_VTI };
enum VrzSlot : int { INV_Z = 0, N_VRZ };

struct Lame { torch::Tensor mu, lambda; };
struct VtiStiffness { torch::Tensor c11, c13, c33, inv_rho; };

// mu = rho*vs*vs, lambda = rho*(vp*vp - 2*vs*vs)
void derive_lame(const torch::Tensor& vp, const torch::Tensor& vs, const torch::Tensor& rho,
                 torch::Tensor& mu, torch::Tensor& lambda);
// c33 = rho*vp^2, c11 = c33*(1+2*eps), c13 = c33*sqrt(1+2*delta), inv_rho = 1/rho
void derive_vti_stiffness(const torch::Tensor& vp, const torch::Tensor& epsilon,
                          const torch::Tensor& delta, const torch::Tensor& rho,
                          torch::Tensor& c11, torch::Tensor& c13, torch::Tensor& c33,
                          torch::Tensor& inv_rho);
// inv_z = 1/z
void derive_reciprocal(const torch::Tensor& z, torch::Tensor& inv_z);

// The declared slots, bound from ``derived_models`` when the propagator passed
// them (count checked at entry, shape/dtype per slot) or allocated here.
inline std::vector<torch::Tensor> slots(const std::vector<torch::Tensor>& bound, int n,
                                        const torch::Tensor& like, const char* what)
{
    TORCH_CHECK(bound.empty() || static_cast<int>(bound.size()) == n,
                what, ": derived_models must be empty or hold ", n,
                " tensors (cuda_layout.derived_model_nvar), got ", bound.size());
    std::vector<torch::Tensor> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        out.push_back(pool_or_empty(bound, i, like, "derived_models"));
    return out;
}

template <class P>
inline Lame lame(const P& p, const torch::Tensor& vp, const torch::Tensor& vs,
                 const torch::Tensor& rho, const char* what)
{
    auto s = slots(p.derived_models, N_LAME, vp, what);
    derive_lame(vp, vs, rho, s[MU], s[LAMBDA]);
    return {s[MU], s[LAMBDA]};
}

template <class P>
inline VtiStiffness vti_stiffness(const P& p, const torch::Tensor& vp, const torch::Tensor& epsilon,
                                  const torch::Tensor& delta, const torch::Tensor& rho,
                                  const char* what)
{
    auto s = slots(p.derived_models, N_VTI, vp, what);
    derive_vti_stiffness(vp, epsilon, delta, rho, s[C11], s[C13], s[C33], s[INV_RHO]);
    return {s[C11], s[C13], s[C33], s[INV_RHO]};
}

template <class P>
inline torch::Tensor reciprocal(const P& p, const torch::Tensor& z, const char* what)
{
    auto s = slots(p.derived_models, N_VRZ, z, what);
    derive_reciprocal(z, s[INV_Z]);
    return s[INV_Z];
}

}  // namespace derived

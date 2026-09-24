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

// visco_acoustic2d: the spectral-term tables, each one model scaled by a float
// (Gp = A*dt, dt2A = A*dt^2, Gd1 = B1*dt^2, Gd2 = B2*dt^2 with the prepared
// models [vp_step, B1, B2, A]).  Which tables a call derives depends on the
// constructor flags (a = amplitude damping, d = phase dispersion) AND on the
// mode -- the forward damps with dt2A, the reverse sweep with Gp, and a
// checkpoint replay needs both -- so the slot index of a table is given by
// ``visco_tables`` (the one (a, d, mode) -> index mapping; the Python side's
// ``ViscoAcoustic.cuda_layout.derived_model_nvar`` counts the same list).
enum ViscoTable : int { GP = 0, DT2A, GD1, GD2, N_VISCO_TABLES };
enum class ViscoMode : int { Forward, Full, BoundarySaving, Checkpoint, Recursive };

// Slot index of each table in ``derived_models`` for one call, -1 when the
// mode does not derive it; ``count`` is the pool size the caller must bind.
struct ViscoTables {
    int gp = -1, dt2a = -1, gd1 = -1, gd2 = -1;
    int count = 0;
    bool has(ViscoTable t) const { return index(t) >= 0; }
    int index(ViscoTable t) const {
        switch (t) {
            case GP:   return gp;
            case DT2A: return dt2a;
            case GD1:  return gd1;
            case GD2:  return gd2;
            default:   return -1;
        }
    }
};

// Slot order per mode (the Python layout, verbatim):
//   forward:            [dt2A if a, Gd1 if d, Gd2 if d]
//   full / bs backward: [Gp if a,   Gd1 if d, Gd2 if d]
//   ckpt / recursive:   [Gp if a, dt2A if a, Gd1 if d, Gd2 if d]
inline ViscoTables visco_tables(bool damping, bool dispersion, ViscoMode mode)
{
    ViscoTables t;
    const bool forward = (mode == ViscoMode::Forward);
    const bool replay = (mode == ViscoMode::Checkpoint || mode == ViscoMode::Recursive);
    if (damping && !forward) t.gp = t.count++;
    if (damping && (forward || replay)) t.dt2a = t.count++;
    if (dispersion) {
        t.gd1 = t.count++;
        t.gd2 = t.count++;
    }
    return t;
}

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
// out = in * s, one float multiply per cell (__fmul_rn): the bits of
// ``tensor * float_scalar`` (the visco tables Gp / dt2A / Gd1 / Gd2)
void derive_scale(const torch::Tensor& in, float s, torch::Tensor& out);

// The declared slots, bound from ``derived_models`` (the propagator always
// passes cuda_layout.derived_model_nvar of them; count checked here, shape/
// dtype per slot).  Nothing allocates: an empty list is the error.
inline std::vector<torch::Tensor> slots(const std::vector<torch::Tensor>& bound, int n,
                                        const torch::Tensor& like, const char* what)
{
    SWEEP_CHECK(static_cast<int>(bound.size()) == n,
                what, ": derived_models must hold ", n,
                " tensors (cuda_layout.derived_model_nvar), got ", bound.size());
    std::vector<torch::Tensor> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        out.push_back(pool_required(bound, i, like, "derived_models"));
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

// The visco tables of one call: the mode's slots (``visco_tables``) bound from
// ``derived_models`` or allocated here, each filled by ``derive_scale``.
// ``dt2`` is dt*dt as the caller computed it (float, on the host: the scalar
// the replaced ``model * (dt * dt)`` expression used).  Tables the mode does
// not derive stay undefined.
struct ViscoCoefficients {
    ViscoTables tables;
    torch::Tensor gp, dt2a, gd1, gd2;
};

inline ViscoCoefficients visco_coefficients(const std::vector<torch::Tensor>& bound,
                                            const torch::Tensor& B1, const torch::Tensor& B2,
                                            const torch::Tensor& A, bool damping, bool dispersion,
                                            ViscoMode mode, float dt, float dt2, const char* what)
{
    ViscoCoefficients c;
    c.tables = visco_tables(damping, dispersion, mode);
    auto s = slots(bound, c.tables.count, A, what);
    if (c.tables.gp >= 0)   { c.gp   = s[c.tables.gp];   derive_scale(A,  dt,  c.gp); }
    if (c.tables.dt2a >= 0) { c.dt2a = s[c.tables.dt2a]; derive_scale(A,  dt2, c.dt2a); }
    if (c.tables.gd1 >= 0)  { c.gd1  = s[c.tables.gd1];  derive_scale(B1, dt2, c.gd1); }
    if (c.tables.gd2 >= 0)  { c.gd2  = s[c.tables.gd2];  derive_scale(B2, dt2, c.gd2); }
    return c;
}

}  // namespace derived

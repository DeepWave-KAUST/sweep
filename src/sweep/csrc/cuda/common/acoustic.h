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


// Host-side: derive the per-axis CPML aux slab geometry from the bound psi
// tensor shapes and install it on the SolverContext.  Full-length tensors
// select the identity (legacy full-domain) layout; slab-length tensors select
// the strip layout; anything else is a Python/C++ layout drift and throws.
// Call after BOTH the SolverContext (incl. set_per_edge/cut_mask) and the
// PML-carrying wavefield tensor struct exist, before any kernel launch.
template <typename WF>
inline void acoustic_init_aux_slabs(SolverContext& ctx, const WF& wf) {
    long lz = -1, ly = -1, lx = -1;
    if (wf.use_pml && wf.psix_t.defined() && wf.psix_t.numel() > 0) {
        lx = wf.psix_t.size(-1);
        lz = wf.psiz_t.size(2);
        if (wf.dim == 3 && wf.psiy_t.defined() && wf.psiy_t.numel() > 0)
            ly = wf.psiy_t.size(3);
    }
    TORCH_CHECK(ctx.init_aux_slabs(lz, ly, lx),
                "CPML aux tensor axis lengths match neither the full grid nor "
                "the strip layout: z=", lz, " y=", ly, " x=", lx,
                " grid (", ctx.nz, ",", ctx.ny, ",", ctx.nx, ") M=", ctx.M);
}

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

    // Optional double-buffer "next" psi (race-free forward: read psi*, write psi*n).
    // nullptr when the equation does not opt into psi double-buffering.
    float* __restrict__ psixn = nullptr;
    float* __restrict__ psiyn = nullptr;
    float* __restrict__ psizn = nullptr;

    // Optional double-buffer "next" zeta (fused adjoint: read zeta at neighbours
    // via g_tmp, write next zeta to a SEPARATE buffer).  nullptr unless the bound
    // tensor list opts into the aux (zeta) double-buffer.
    float* __restrict__ zetaxn = nullptr;
    float* __restrict__ zetayn = nullptr;
    float* __restrict__ zetazn = nullptr;

    // Per-batch numel of one aux field of each axis family, filled by view()
    // from the bound tensor shapes.  Equals spatial_size for full-domain
    // (legacy) tensors, the slab numel for strip tensors — so the batch
    // offset below is correct for both layouts without any driver flag.
    long aux_bn_x = -1;
    long aux_bn_y = -1;
    long aux_bn_z = -1;

    __device__ AcousticWavefieldPointer offset(
        int b,
        int spatial_size
    ) const {

        AcousticWavefieldPointer out = *this;

        int shift = b * spatial_size;
        long sx = aux_bn_x >= 0 ? (long)b * aux_bn_x : (long)shift;
        long sy = aux_bn_y >= 0 ? (long)b * aux_bn_y : (long)shift;
        long sz = aux_bn_z >= 0 ? (long)b * aux_bn_z : (long)shift;

        out.u_prev += shift;
        out.u_now  += shift;
        out.u_next += shift;

        if (out.psix)  out.psix  += sx;
        if (out.psiy)  out.psiy  += sy;
        if (out.psiz)  out.psiz  += sz;

        if (out.zetax) out.zetax += sx;
        if (out.zetay) out.zetay += sy;
        if (out.zetaz) out.zetaz += sz;

        if (out.psixn) out.psixn += sx;
        if (out.psiyn) out.psiyn += sy;
        if (out.psizn) out.psizn += sz;

        if (out.zetaxn) out.zetaxn += sx;
        if (out.zetayn) out.zetayn += sy;
        if (out.zetazn) out.zetazn += sz;

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

    // Optional double-buffer "next" psi (set only when the bound tensor list
    // includes them, i.e. the equation opts into psi double-buffering).
    torch::Tensor psixn_t;
    torch::Tensor psiyn_t;
    torch::Tensor psizn_t;

    // Optional double-buffer "next" zeta (fused adjoint only).
    torch::Tensor zetaxn_t;
    torch::Tensor zetayn_t;
    torch::Tensor zetazn_t;

    int dim = 2;
    bool use_pml = true;
    bool allocated = false;
    bool double_buffer_psi = false;   // true when psi*n_t are present
    bool double_buffer_aux = false;   // true when zeta*n_t are present (fused adjoint)

    // =========================
    // Allocate (only once)
    // =========================
    // double_buffer_psi_: pass true at call sites whose stepping pairs with
    // swap_pml() (the forward time loops).  The stencil kernels neighbour-read
    // psi* in the same launch that writes the new psi, so the write must land
    // in psi*n (read-old/write-new, swap_pml() per step) — matching the
    // Python-bound 9/12-tensor layout.  Without it the kernel falls back to
    // the legacy in-place write, an intra-launch RAW race (nondeterministic
    // at ulp level in 3-D).  Call sites that pair with the u-only swap()
    // (checkpoint recompute, adjoint states) must keep the default: they
    // never swap psi, so a psi*n write would be lost.
    void allocate(
        const torch::Tensor& vp,
        int dim_,
        bool use_pml_ = true,
        bool double_buffer_psi_ = false
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

            if (double_buffer_psi_) {
                psixn_t = torch::zeros_like(vp);
                psizn_t = torch::zeros_like(vp);
                if (dim == 3) psiyn_t = torch::zeros_like(vp);
                double_buffer_psi = true;
            }
        }

        allocated = true;
    }

    // Allocate state whose aux shapes follow the Python-allocated checkpoint
    // snapshot slots (checkpoint_tensors() order, each [n_ckpt, B, 1, ...]).
    // Used by the recursive-checkpoint driver, which has no bound forward
    // wavefield to copy the (possibly slab-shaped) aux layout from.
    void allocate_from_snapshots(const torch::Tensor& vp,
                                 const std::vector<torch::Tensor>& snaps,
                                 int dim_)
    {
        if (allocated) return;
        dim = dim_;
        use_pml = true;
        u_prev_t = torch::zeros_like(vp);
        u_now_t  = torch::zeros_like(vp);
        u_next_t = torch::zeros_like(vp);
        auto zl = [&](const torch::Tensor& t) {
            auto sizes = t.sizes().vec();
            sizes.erase(sizes.begin());          // drop the n_ckpt axis
            return torch::zeros(sizes, vp.options());
        };
        if (dim == 2) {
            TORCH_CHECK(snaps.size() == 6, "acoustic 2D checkpoint set expects 6 tensors");
            psix_t = zl(snaps[2]); psiz_t = zl(snaps[3]);
            zetax_t = zl(snaps[4]); zetaz_t = zl(snaps[5]);
        } else {
            TORCH_CHECK(snaps.size() == 8, "acoustic 3D checkpoint set expects 8 tensors");
            psix_t = zl(snaps[2]); psiy_t = zl(snaps[3]); psiz_t = zl(snaps[4]);
            zetax_t = zl(snaps[5]); zetay_t = zl(snaps[6]); zetaz_t = zl(snaps[7]);
        }
        allocated = true;
    }

    // Like allocate(), but clones the aux-field SHAPES from a reference
    // wavefield (which may carry slab-shaped CPML aux tensors).  Physical
    // fields stay model-shaped.  Use for driver-internal scratch/segment
    // state so it agrees with the Python-chosen aux layout.
    void allocate_like(const torch::Tensor& vp, const AcousticWavefieldTensor& ref)
    {
        if (allocated) return;
        dim = ref.dim;
        use_pml = ref.use_pml;

        u_prev_t = torch::zeros_like(vp);
        u_now_t  = torch::zeros_like(vp);
        u_next_t = torch::zeros_like(vp);

        if (use_pml) {
            psix_t  = torch::zeros_like(ref.psix_t);
            psiz_t  = torch::zeros_like(ref.psiz_t);
            zetax_t = torch::zeros_like(ref.zetax_t);
            zetaz_t = torch::zeros_like(ref.zetaz_t);
            if (dim == 3) {
                psiy_t  = torch::zeros_like(ref.psiy_t);
                zetay_t = torch::zeros_like(ref.zetay_t);
            }
        }
        allocated = true;
    }

    void bind(
        const std::vector<torch::Tensor>& tensors,
        int dim_,
        bool use_pml_ = true
    )
    {
        int i = 0;
        dim = dim_;
        use_pml = use_pml_;

        if (dim == 2) {
            TORCH_CHECK(
                !use_pml ? tensors.size() == 3
                         : (tensors.size() == 7 || tensors.size() == 9 || tensors.size() == 11),
                "Acoustic 2D wavefields expect 3 (no PML), 7 (PML), 9 (PML+psi double-buffer), or 11 (PML+psi+zeta double-buffer) tensors"
            );
        } else {
            TORCH_CHECK(
                !use_pml ? tensors.size() == 3
                         : (tensors.size() == 9 || tensors.size() == 12 || tensors.size() == 15),
                "Acoustic 3D wavefields expect 3 (no PML), 9 (PML), 12 (PML+psi double-buffer), or 15 (PML+psi+zeta double-buffer) tensors"
            );
        }

        u_prev_t = tensors[i++];
        u_now_t  = tensors[i++];
        u_next_t = tensors[i++];

        if (use_pml) {
            psix_t  = tensors[i++];
            psiz_t  = tensors[i++];
            zetax_t = tensors[i++];
            zetaz_t = tensors[i++];

            if (dim == 3) {
                psiy_t  = tensors[i++];
                zetay_t = tensors[i++];
            } else {
                psiy_t = torch::Tensor();
                zetay_t = torch::Tensor();
            }

            // Optional psi double-buffer: extra "next" psi tensors appended
            // after the standard PML set (2D: +2 -> 9, 3D: +3 -> 12).  The fused
            // adjoint adds a zeta double-buffer on top (2D: +2 -> 11, 3D: +3 -> 15);
            // tail order is psixn,psizn[,psiyn], zetaxn,zetazn[,zetayn].
            double_buffer_aux = (dim == 2 ? tensors.size() == 11
                                          : tensors.size() == 15);
            double_buffer_psi = double_buffer_aux ||
                                (dim == 2 ? tensors.size() == 9
                                          : tensors.size() == 12);
            if (double_buffer_psi) {
                psixn_t = tensors[i++];
                psizn_t = tensors[i++];
                if (dim == 3) psiyn_t = tensors[i++];
            } else {
                psixn_t = torch::Tensor();
                psizn_t = torch::Tensor();
                psiyn_t = torch::Tensor();
            }
            if (double_buffer_aux) {
                zetaxn_t = tensors[i++];
                zetazn_t = tensors[i++];
                if (dim == 3) zetayn_t = tensors[i++];
            } else {
                zetaxn_t = torch::Tensor();
                zetazn_t = torch::Tensor();
                zetayn_t = torch::Tensor();
            }
        } else {
            psix_t = torch::Tensor();
            psiy_t = torch::Tensor();
            psiz_t = torch::Tensor();
            zetax_t = torch::Tensor();
            zetay_t = torch::Tensor();
            zetaz_t = torch::Tensor();
            double_buffer_psi = false;
            double_buffer_aux = false;
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

            if (double_buffer_psi) {
                v.psixn = psixn_t.data_ptr<float>();
                v.psizn = psizn_t.data_ptr<float>();
                v.psiyn = (dim == 3) ? psiyn_t.data_ptr<float>() : nullptr;
            } else {
                v.psixn = v.psiyn = v.psizn = nullptr;
            }

            if (double_buffer_aux) {
                v.zetaxn = zetaxn_t.data_ptr<float>();
                v.zetazn = zetazn_t.data_ptr<float>();
                v.zetayn = (dim == 3) ? zetayn_t.data_ptr<float>() : nullptr;
            } else {
                v.zetaxn = v.zetayn = v.zetazn = nullptr;
            }
        } else {
            v.psix = v.psiy = v.psiz = nullptr;
            v.zetax = v.zetay = v.zetaz = nullptr;
            v.psixn = v.psiyn = v.psizn = nullptr;
            v.zetaxn = v.zetayn = v.zetazn = nullptr;
        }

        if (use_pml) {
            auto bn = [](const torch::Tensor& t) -> long {
                return t.defined() && t.numel() > 0 ? t.numel() / t.size(0) : -1;
            };
            v.aux_bn_x = bn(psix_t);
            v.aux_bn_z = bn(psiz_t);
            if (dim == 3) v.aux_bn_y = bn(psiy_t);
        }

        return v;
    }

    std::vector<torch::Tensor> checkpoint_tensors() const
    {
        if (dim == 3)
            return {u_prev_t, u_now_t, psix_t, psiy_t, psiz_t, zetax_t, zetay_t, zetaz_t};
        return {u_prev_t, u_now_t, psix_t, psiz_t, zetax_t, zetaz_t};
    }

    std::vector<torch::Tensor> state_tensors() const
    {
        if (!use_pml)
            return {u_prev_t, u_now_t, u_next_t};

        if (dim == 3)
            return {u_prev_t, u_now_t, u_next_t, psix_t, psiy_t, psiz_t, zetax_t, zetay_t, zetaz_t};
        return {u_prev_t, u_now_t, u_next_t, psix_t, psiz_t, zetax_t, zetaz_t};
    }

    std::vector<torch::Tensor> next_tensors() const
    {
        return {u_next_t};
    }

    void swap()
    {
        std::swap(u_prev_t, u_now_t);
        std::swap(u_now_t,  u_next_t);
    }

    // Forward double-buffer swap: rotate u AND swap psi <-> psin so the next
    // step reads the freshly-written psi (race-free forward).  Use this only
    // when the forward kernel writes psi*n; otherwise use swap() (u-only).
    // No-op on psi if not double-buffering, so it is safe to call either way.
    void swap_pml()
    {
        std::swap(u_prev_t, u_now_t);
        std::swap(u_now_t,  u_next_t);
        if (double_buffer_psi) {
            std::swap(psix_t, psixn_t);
            std::swap(psiz_t, psizn_t);
            if (dim == 3) std::swap(psiy_t, psiyn_t);
        }
    }

    // Fused-adjoint double-buffer swap: rotate u AND psi<->psin AND zeta<->zetan.
    // Use ONLY with the fused adjoint kernel (which writes psi*n AND zeta*n).
    void swap_aux()
    {
        std::swap(u_prev_t, u_now_t);
        std::swap(u_now_t,  u_next_t);
        if (double_buffer_psi) {
            std::swap(psix_t, psixn_t);
            std::swap(psiz_t, psizn_t);
            if (dim == 3) std::swap(psiy_t, psiyn_t);
        }
        if (double_buffer_aux) {
            std::swap(zetax_t, zetaxn_t);
            std::swap(zetaz_t, zetazn_t);
            if (dim == 3) std::swap(zetay_t, zetayn_t);
        }
    }

};

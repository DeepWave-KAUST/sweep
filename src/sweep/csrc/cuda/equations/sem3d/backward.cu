// sweep.sem CUDA (impl='c') backend — 3-D acoustic SEM backward (P5).
// Adjoint-state, save-all or boundary-free reverse march. The step is
// u_next = 2u - u_prev + dt2 inv_M (pat w - C K0 u), C = diag(c^2) and K0 the
// symmetric c^2-free stiffness, so the reverse leapfrog applies
// (inv_M C K0)^T = K0 C inv_M with the same stiffness kernel. Grads: c2_node
// (in the grad_c2_jx slot), inv_M, wavelet — PyTorch chains through the eager
// build() to vp_e.
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

#include "sem3d.h"
#include "kernels.cuh"
#include "../sem2d/kernels.cuh"

namespace sem3d {

void acoustic3d_backward(const Sem3dBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt, B = in.batch;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    SWEEP_CHECK(B == 1, "sem3d driver runs batch=1: the ELEMENT kernels are "
                "batch-ready but the NODAL ones are not (shared per-node arrays + a "
                "non-atomic shared gradient). See the batch note in sem3d/kernels.cuh.");

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const bool def = in.deformed != 0;
    const float* jx = def ? nullptr : in.c2_jx.data_ptr<float>();
    const float* jy = def ? nullptr : in.c2_jy.data_ptr<float>();
    const float* jz = def ? nullptr : in.c2_jz.data_ptr<float>();
    const float* c2n = in.c2_node.data_ptr<float>();     // (n_global,) c^2
    const float* wJ = def ? in.wJ.data_ptr<float>() : nullptr;
    const float* Ixx = def ? in.ixx.data_ptr<float>() : nullptr;
    const float* Ixy = def ? in.ixy.data_ptr<float>() : nullptr;
    const float* Ixz = def ? in.ixz.data_ptr<float>() : nullptr;
    const float* Iex = def ? in.iex.data_ptr<float>() : nullptr;
    const float* Iey = def ? in.iey.data_ptr<float>() : nullptr;
    const float* Iez = def ? in.iez.data_ptr<float>() : nullptr;
    const float* Igx = def ? in.igx.data_ptr<float>() : nullptr;
    const float* Igy = def ? in.igy.data_ptr<float>() : nullptr;
    const float* Igz = def ? in.igz.data_ptr<float>() : nullptr;
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* pattern = in.src_pattern.data_ptr<float>();
    const float* wavelet = in.wavelet.data_ptr<float>();
    const long*  didx = kdir > 0 ? in.dirichlet_idx.data_ptr<int64_t>() : nullptr;
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* grad_syn = nrec > 0 ? in.grad_syn.data_ptr<float>() : nullptr;
    const bool bsm = in.bs != 0;
    const float* save_wf = bsm ? nullptr : in.save_wavefield.data_ptr<float>();
    float* u_hi = bsm ? in.u_hi.data_ptr<float>() : nullptr;
    float* u_lo = bsm ? in.u_lo.data_ptr<float>() : nullptr;
    float* u_km1 = bsm ? in.u_km1.data_ptr<float>() : nullptr;

    float* a_p1 = in.a_p1.data_ptr<float>();
    float* a_p2 = in.a_p2.data_ptr<float>();
    float* a_m  = in.a_m.data_ptr<float>();
    float* tmp  = in.tmp.data_ptr<float>();
    float* Ktmp = in.Ktmp.data_ptr<float>();
    float* Ku   = in.Ku.data_ptr<float>();
    float* adj  = in.adj_src.data_ptr<float>();
    float* gC2 = in.grad_c2_jx.data_ptr<float>();   // (n_global,) dL/dc2_node
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();

    const size_t sh_stiff = static_cast<size_t>(4 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    if (bsm) {   // seed the reverse march with the final two u levels
        cudaMemcpyAsync(u_hi, in.fin_u.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_lo, in.fin_u_prev.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
    }

    for (int mm = nt; mm >= 1; --mm) {
        // reverse leapfrog: a_m = 2 a_p1 - a_p2 - dt2 K0(c^2 o inv_M o a_p1)
        // + R^T g  (adj holds c^2 o inv_M o a_p1 until the adjoint source)
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(a_p1, inv_M, 1.f, tmp, ng);
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(tmp, c2n, 1.f, adj, ng);
        cudaMemsetAsync(Ktmp, 0, ngb, stream);
        if (def)
            sem3d_acoustic_stiffness_deformed<<<grid, m, sh_stiff, stream>>>(
                adj, ibool, D, wJ, Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz,
                Ktmp, ngll, E, ng);
        else
            sem3d_acoustic_stiffness<<<grid, m, sh_stiff, stream>>>(
                adj, ibool, D, jx, jy, jz, Ktmp, ngll, E, ng);
        cudaMemsetAsync(adj, 0, ngb, stream);
        if (nrec > 0)
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(
                grad_syn, rec_G, rec_W, adj, mm - 1, nt, nrec, m);
        sem2d::sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(a_p1, a_p2, Ktmp, adj, dt2, a_m, ng);
        if (kdir > 0)
            sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(a_m, didx, kdir);

        // gradients at forward line k = mm-1 (uses u^{mm-1} = save[mm-2])
        const int km1 = mm - 1;
        const bool u_is_zero = (mm == 1);
        const float* u_line = bsm ? u_lo : (u_is_zero ? nullptr
                              : save_wf + static_cast<size_t>(mm - 2) * ng);
        // Ku = K0 u^{mm-1}, then Ktmp (free after the leapfrog) = c^2 o Ku:
        // the c^2 gradient takes the former, inv_M's and the march the latter
        cudaMemsetAsync(Ku, 0, ngb, stream);
        if (!u_is_zero) {
            if (def)
                sem3d_acoustic_stiffness_deformed<<<grid, m, sh_stiff, stream>>>(
                    u_line, ibool, D, wJ, Ixx, Ixy, Ixz, Iex, Iey, Iez,
                    Igx, Igy, Igz, Ku, ngll, E, ng);
            else
                sem3d_acoustic_stiffness<<<grid, m, sh_stiff, stream>>>(
                    u_line, ibool, D, jx, jy, jz, Ku, ngll, E, ng);
            sem3d_accum_grad_c2<<<nb, TPB, 0, stream>>>(a_m, inv_M, Ku, dt2, gC2, ng);
        }
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(Ku, c2n, 1.f, Ktmp, ng);
        sem2d::sem2d_accum_gradM<<<nb, TPB, 0, stream>>>(
            a_m, pattern, Ktmp, wavelet, km1, dt2, gM, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(
            pattern, inv_M, a_m, dt2, km1, gW, ng);
        if (bsm && !u_is_zero) {
            // reverse-march: u^{mm-2} = 2 u^{mm-1} - u^{mm}
            //   + dt2 invM (pat wav[mm-1] - c^2 K0 u^{mm-1});  Ktmp already
            // holds c^2 K0 u^{mm-1} from the gradient step above.
            sem2d::sem2d_acoustic_newmark<<<nb, TPB, 0, stream>>>(
                u_lo, u_hi, Ktmp, pattern, inv_M, wavelet, km1, dt2, u_km1, ng);
            if (kdir > 0)
                sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(u_km1, didx, kdir);
            float* t2 = u_hi; u_hi = u_lo; u_lo = u_km1; u_km1 = t2;
        }
        float* t = a_p2; a_p2 = a_p1; a_p1 = a_m; a_m = t;
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- 3-D acoustic backward WITH C-PML (P6) ----------------------------------
// 3-axis version of the validated 2-D P3b recursion: gA/gB u-adjoint carry,
// psi/zeta adjoint recursions, per-step recompute (grads + step_T) from the
// saved states, grads_T for the u/apsi adjoints. Only vp2_node and the
// wavelet carry gradients (all other weights are geometric).
// psi_slab: refuse a table / array set that does not describe this grid
// (the forward's twin; here the restores and a_d*psi_d are the slab arrays)
static void sem3d_check_slab(const Sem3dCpmlBackwardInputCore& in) {
    if (in.psi_slab == 0) return;
    SWEEP_CHECK(static_cast<long>(in.nx) * in.ny * in.nz == in.n_global,
                "psi_slab: nx*ny*nz = ", static_cast<long>(in.nx) * in.ny * in.nz,
                " != n_global ", in.n_global);
    SWEEP_CHECK(in.tab_x.numel() == in.nx && in.tab_y.numel() == in.ny
                && in.tab_z.numel() == in.nz, "psi_slab: table lengths != nx/ny/nz");
    SWEEP_CHECK(in.planes_x.numel() == in.nsx && in.planes_y.numel() == in.nsy
                && in.planes_z.numel() == in.nsz, "psi_slab: plane lists != nsx/nsy/nsz");
    SWEEP_CHECK(in.tab_x.element_size() == 4 && in.planes_x.element_size() == 4
                && in.tab_y.element_size() == 4 && in.tab_z.element_size() == 4,
                "psi_slab: tables must be int32");
    SWEEP_CHECK(in.ring_slot_x.element_size() == 8
                && in.ring_slot_y.element_size() == 8
                && in.ring_slot_z.element_size() == 8,
                "psi_slab: slot lists must be int64");
    const long n[3] = {sem3d_slab_len(0, in), sem3d_slab_len(1, in), sem3d_slab_len(2, in)};
    const Buf* f[30] = {&in.r_psi_x, &in.r_psi_y, &in.r_psi_z,
                                  &in.r_zeta_x, &in.r_zeta_y, &in.r_zeta_z,
                                  &in.apx, &in.apy, &in.apz,
                                  &in.lpsi_x, &in.lpsi_y, &in.lpsi_z,
                                  &in.lzeta_x, &in.lzeta_y, &in.lzeta_z,
                                  &in.lpsi_x_n, &in.lpsi_y_n, &in.lpsi_z_n,
                                  &in.lzeta_x_n, &in.lzeta_y_n, &in.lzeta_z_n,
                                  &in.lapx, &in.lapy, &in.lapz,
                                  &in.gx, &in.gy, &in.gz,
                                  &in.kx, &in.ky, &in.kz};
    for (int j = 0; j < 30; ++j) {
        if (j < 9 && in.bs == 0) continue;   // restores / a*psi: boundary saving only
        SWEEP_CHECK(f[j]->numel() == n[j % 3], "psi_slab: backward array ", j, " has ",
                    f[j]->numel(), " entries, its slab has ", n[j % 3]);
    }
}

// ---- 3-D elastic backward (P7, bare; K self-adjoint) ------------------------
// ---- DD stepped backward (3-D acoustic C-PML) -------------------------------
// Mirrors the 2-D acoustic split: two assemblies per reverse step, so two
// exchanges. batch == 1, save-all (boundary saving on the 3-D DD path is a
// later phase, refused Python-side). Carried-buffer rotation is the caller's.
//   pre : adjoint source, dirichlet, gw, a*psi, grads      [exchange 9]
//   mid : step_T + grads_T                                 [exchange 4]
//   end : combine + wavelet gradient + gB
// Zero the lu_G / lap_x / lap_y / lap_z family. lu_G is per node; under the
// slab layout each lap_d is its AXIS SLAB long, because that is where grads_T
// writes it and where combine reads it. Two entries zero this family and they
// take the sizes from here so they cannot drift -- they did, and a
// full-length memset of a slab-length array is an out-of-bounds
// cudaMemsetAsync, i.e. cudaErrorInvalidValue at the next launch.
template <typename In>
static inline void sem3d_zero_l4(const In& in, bool slab, cudaStream_t stream) {
    float* l[4] = {in.lu_G.template data_ptr<float>(),
                   in.lapx.template data_ptr<float>(),
                   in.lapy.template data_ptr<float>(),
                   in.lapz.template data_ptr<float>()};
    size_t lb[4];
    lb[0] = static_cast<size_t>(in.n_global) * sizeof(float);
    for (int j = 1; j < 4; ++j)
        lb[j] = (slab ? static_cast<size_t>(sem3d_slab_len(j - 1, in))
                      : static_cast<size_t>(in.n_global)) * sizeof(float);
    bool c4 = true;
    for (int j = 1; j < 4; ++j)
        c4 = c4 && (reinterpret_cast<char*>(l[j])
                    == reinterpret_cast<char*>(l[j - 1]) + lb[j - 1]);
    if (c4) {
        cudaMemsetAsync(l[0], 0, lb[0] + lb[1] + lb[2] + lb[3], stream);
    } else {
        for (int j = 0; j < 4; ++j) cudaMemsetAsync(l[j], 0, lb[j], stream);
    }
}

void acoustic3d_cpml_dd_bwd_pre(const Sem3dCpmlBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const bool bs = in.bs != 0;
    const bool def = in.deformed != 0;
    const float ihx = static_cast<float>(in.ihx), ihy = static_cast<float>(in.ihy),
                ihz = static_cast<float>(in.ihz);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const bool slab = in.psi_slab != 0;
    const bool wt = !bs && in.save_w_sum.numel() > 0;     // w_sum tape: no recompute
    SWEEP_CHECK(!(slab && !bs && !wt), "psi_slab: not with the full save-all "
                "tapes (they are full-length); boundary saving or the w_sum tape");
    sem3d_check_slab(in);
    const SemSlabs sl = sem3d_slabs(in);
    const int nsl[3] = {slab ? sem3d_slab_len(0, in) : ng,
                        slab ? sem3d_slab_len(1, in) : ng,
                        slab ? sem3d_slab_len(2, in) : ng};
    // Dirichlet on an adjoint carry: the slab's own node list under psi_slab
    const long* dsl[3] = {in.dir_slot_x.numel() ? in.dir_slot_x.data_ptr<int64_t>() : nullptr,
                          in.dir_slot_y.numel() ? in.dir_slot_y.data_ptr<int64_t>() : nullptr,
                          in.dir_slot_z.numel() ? in.dir_slot_z.data_ptr<int64_t>() : nullptr};
    const int kdsl[3] = {static_cast<int>(in.dir_slot_x.numel()),
                         static_cast<int>(in.dir_slot_y.numel()),
                         static_cast<int>(in.dir_slot_z.numel())};
    float* gA = in.gA.data_ptr<float>();
    if (nrec > 0) {
        const int rj = (nrec * m + TPB - 1) / TPB;
        sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(
            in.grad_syn.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), gA, static_cast<int>(k), nt, nrec, m);
    }
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        const long* didx = in.dirichlet_idx.data_ptr<int64_t>();
        sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(gA, didx, kdir);
        float* lf[6] = {in.lpsi_x.data_ptr<float>(), in.lpsi_y.data_ptr<float>(),
                        in.lpsi_z.data_ptr<float>(), in.lzeta_x.data_ptr<float>(),
                        in.lzeta_y.data_ptr<float>(), in.lzeta_z.data_ptr<float>()};
        for (int j = 0; j < 6; ++j) {
            if (slab) {
                const int kd = kdsl[j % 3];
                if (kd > 0)
                    sem2d::sem2d_dirichlet_fill<<<(kd + TPB - 1) / TPB, TPB, 0, stream>>>(
                        lf[j], dsl[j % 3], kd);
            } else {
                sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(lf[j], didx, kdir);
            }
        }
    }
    sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(
        in.vp2_node.data_ptr<float>(), gA, dt2, in.gw.data_ptr<float>(), ng);
    float* g[9] = {in.gx.data_ptr<float>(), in.gy.data_ptr<float>(),
                   in.gz.data_ptr<float>(), in.kx.data_ptr<float>(),
                   in.ky.data_ptr<float>(), in.kz.data_ptr<float>(),
                   in.mx.data_ptr<float>(), in.my.data_ptr<float>(),
                   in.mz.data_ptr<float>()};
    if (k >= 1 && !wt) {
        const size_t o = static_cast<size_t>(k - 1) * ng;
        float* apx = in.apx.data_ptr<float>();
        float* apy = in.apy.data_ptr<float>();
        float* apz = in.apz.data_ptr<float>();
        // state^k comes from the save-all tape, or -- under boundary saving
        // -- is reconstructed: u_lo carries the reverse march, and the C-PML
        // memories are zero outside the band and restored from the ring
        // inside it (a cut plane is outside, so both strips agree there).
        const float* psx = bs ? nullptr : in.save_psi_x.data_ptr<float>() + o;
        const float* psy = bs ? nullptr : in.save_psi_y.data_ptr<float>() + o;
        const float* psz = bs ? nullptr : in.save_psi_z.data_ptr<float>() + o;
        if (bs) {
            float* r_px = in.r_psi_x.data_ptr<float>();
            float* r_py = in.r_psi_y.data_ptr<float>();
            float* r_pz = in.r_psi_z.data_ptr<float>();
            float* r_zx = in.r_zeta_x.data_ptr<float>();
            float* r_zy = in.r_zeta_y.data_ptr<float>();
            float* r_zz = in.r_zeta_z.data_ptr<float>();
            const int nr[6] = {nsl[0], nsl[1], nsl[2], nsl[0], nsl[1], nsl[2]};
            const bool r6 = (r_py == r_px + nr[0]) && (r_pz == r_py + nr[1])
                         && (r_zx == r_pz + nr[2]) && (r_zy == r_zx + nr[3])
                         && (r_zz == r_zy + nr[4]);
            if (r6) {
                const size_t nrb = static_cast<size_t>(nr[0] + nr[1] + nr[2]) * 2 * sizeof(float);
                if (nrb > 0) cudaMemsetAsync(r_px, 0, nrb, stream);
            } else {
                float* r[6] = {r_px, r_py, r_pz, r_zx, r_zy, r_zz};
                for (int j = 0; j < 6; ++j)
                    if (nr[j] > 0)
                        cudaMemsetAsync(r[j], 0, static_cast<size_t>(nr[j]) * sizeof(float), stream);
            }
            const int nband = static_cast<int>(in.band_idx.numel());
            if (nband > 0) {
                const long* bg = in.band_idx.data_ptr<int64_t>();
                const int bk = in.band_kind;
                float* r[6] = {r_px, r_py, r_pz, r_zx, r_zy, r_zz};
                const Buf* rg[6] = {&in.band_psi_x, &in.band_psi_y,
                                              &in.band_psi_z, &in.band_zeta_x,
                                              &in.band_zeta_y, &in.band_zeta_z};
                const bool bsplit = in.band_split != 0;
                SWEEP_CHECK(!(slab && !bsplit), "psi_slab needs band_split: the "
                            "union ring index addresses nodes off the slab");
                const sem2d::SemBandIdx bi_x = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_x : in.band_idx_x, bg, nband);
                const sem2d::SemBandIdx bi_y = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_y : in.band_idx_y, bg, nband);
                const sem2d::SemBandIdx bi_z = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_z : in.band_idx_z, bg, nband);
                const sem2d::SemBandIdx bi[6] = {bi_x, bi_y, bi_z, bi_x, bi_y, bi_z};
                for (int j = 0; j < 6; ++j)   // ring slots 1..6 (0 is u)
                    sem2d::sem2d_band_load(
                        sem2d::sem_hist(*rg[j], in.band_scale, bk, j + 1, nt),
                        static_cast<int>(k) - 1, bi[j].idx, r[j], bi[j].n, TPB, stream);
            }
            psx = r_px; psy = r_py; psz = r_pz;
        }
        {
            const float* ps[3] = {psx, psy, psz};
            const float* as[3] = {in.a_x.data_ptr<float>(), in.a_y.data_ptr<float>(),
                                  in.a_z.data_ptr<float>()};
            float* aps[3] = {apx, apy, apz};
            for (int axis = 0; axis < 3; ++axis) {
                if (slab) {
                    const int n_ = nsl[axis];
                    if (n_ > 0)
                        sem3d_slab_scale<<<(n_ + TPB - 1) / TPB, TPB, 0, stream>>>(
                            as[axis], ps[axis], aps[axis], axis, sl, n_);
                } else {
                    sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(as[axis], ps[axis], 1.f, aps[axis], ng);
                }
            }
        }
        sem3d_zero_g9(in, slab, stream, g);
    // columnar metric (sem3d_types.h). Set only by prop_c.metric_consts, and
    // read only by the deformed kernels below; 0 is the full 3x3 path.
    const bool col = in.columnar != 0;
    const float c_ixx = static_cast<float>(in.col_ixx);
    const float c_iey = static_cast<float>(in.col_iey);
        const size_t sh_g = static_cast<size_t>(7 * m) * sizeof(float);
        // the a*psi skip (sem3d_types.h): empty tensor = every element flagged
        const int* eslab = in.elem_slab.numel() ? in.elem_slab.data_ptr<int32_t>() : nullptr;
        const float* u_k = bs ? in.u_lo.data_ptr<float>()
                              : in.save_u.data_ptr<float>() + o;
        if (def)
            SEM3D_LAUNCH_SLAB2(slab, col, sem3d_acoustic_cpml_grads_deformed, E, m, sh_g, stream,
                u_k, apx, apy, apz, in.ibool.data_ptr<int32_t>(),
                in.D.data_ptr<float>(), in.w_jac.data_ptr<float>(),
                in.ixx.data_ptr<float>(), in.ixy.data_ptr<float>(),
                in.ixz.data_ptr<float>(), in.iex.data_ptr<float>(),
                in.iey.data_ptr<float>(), in.iez.data_ptr<float>(),
                in.igx.data_ptr<float>(), in.igy.data_ptr<float>(),
                in.igz.data_ptr<float>(), c_ixx, c_iey,
                g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], ngll, E, ng,
                sl, eslab);
        else
            SEM3D_LAUNCH_SLAB(slab, sem3d_acoustic_cpml_grads, E, m, sh_g, stream,
                u_k, apx, apy, apz, in.ibool.data_ptr<int32_t>(),
                in.D.data_ptr<float>(), in.w_jac.data_ptr<float>(),
                in.w_jx.data_ptr<float>(), in.w_jy.data_ptr<float>(),
                in.w_jz.data_ptr<float>(), ihx, ihy, ihz,
                g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], ngll, E, ng,
                sl, eslab);
    } else if (k == 0) {
        cudaMemsetAsync(in.w_sum.data_ptr<float>(), 0, ngb, stream);
        sem3d_zero_l4(in, slab, stream);
    }
    // (k >= 1 under the w_sum tape: nothing to rebuild -- step_T in bwd_mid
    // runs the adjoint algebra alone and combine reads the tape row)
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic3d_cpml_dd_bwd_mid(const Sem3dCpmlBackwardInputCore& in, int64_t k) {
    if (k < 1) return;
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll;
    const bool def = in.deformed != 0;
    const float ihx = static_cast<float>(in.ihx), ihy = static_cast<float>(in.ihy),
                ihz = static_cast<float>(in.ihz);
    const bool bs = in.bs != 0;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t o = static_cast<size_t>(k - 1) * ng;
    const bool slab = in.psi_slab != 0;
    const bool wt = !bs && in.save_w_sum.numel() > 0;     // w_sum tape: no recompute
    SWEEP_CHECK(!(slab && !bs && !wt), "psi_slab: not with the full save-all tapes");
    sem3d_check_slab(in);
    const SemSlabs sl = sem3d_slabs(in);
    // zeta^k: tape, or the ring copies bwd_pre already restored this step
    // (unread under the w_sum tape)
    const float* zsx = wt ? nullptr : bs ? in.r_zeta_x.data_ptr<float>()
                          : in.save_zeta_x.data_ptr<float>() + o;
    const float* zsy = wt ? nullptr : bs ? in.r_zeta_y.data_ptr<float>()
                          : in.save_zeta_y.data_ptr<float>() + o;
    const float* zsz = wt ? nullptr : bs ? in.r_zeta_z.data_ptr<float>()
                          : in.save_zeta_z.data_ptr<float>() + o;
    SEM3D_LAUNCH_SLAB2(slab, wt, sem3d_acoustic_cpml_step_T, nb, TPB, 0, stream,
        in.gx.data_ptr<float>(), in.gy.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.ky.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.mx.data_ptr<float>(), in.my.data_ptr<float>(), in.mz.data_ptr<float>(),
        in.inv_M.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(), in.dbx_dx.data_ptr<float>(),
        in.a_y.data_ptr<float>(), in.b_y.data_ptr<float>(), in.dby_dy.data_ptr<float>(),
        in.a_z.data_ptr<float>(), in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        in.band_node.data_ptr<uint8_t>(),
        zsx, zsy, zsz, in.gw.data_ptr<float>(),
        in.lpsi_x.data_ptr<float>(), in.lpsi_y.data_ptr<float>(),
        in.lpsi_z.data_ptr<float>(), in.lzeta_x.data_ptr<float>(),
        in.lzeta_y.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
        in.gbx.data_ptr<float>(), in.gby.data_ptr<float>(), in.gbz.data_ptr<float>(),
        in.gbkx.data_ptr<float>(), in.gbky.data_ptr<float>(), in.gbkz.data_ptr<float>(),
        in.gbax.data_ptr<float>(), in.gbay.data_ptr<float>(), in.gbaz.data_ptr<float>(),
        in.w_sum.data_ptr<float>(), ng, sl);
    sem3d_zero_l4(in, slab, stream);
    float* l[4] = {in.lu_G.data_ptr<float>(), in.lapx.data_ptr<float>(),
                   in.lapy.data_ptr<float>(), in.lapz.data_ptr<float>()};
    // columnar metric (sem3d_types.h). Set only by prop_c.metric_consts, and
    // read only by the deformed kernels below; 0 is the full 3x3 path.
    const bool col = in.columnar != 0;
    const float c_ixx = static_cast<float>(in.col_ixx);
    const float c_iey = static_cast<float>(in.col_iey);
    const size_t sh_t = static_cast<size_t>(6 * m) * sizeof(float);
    // the transpose skips lap_d for an element off slab d ONLY under the slab
    // layout, where combine reads the carries on their slab alone; the full
    // layout reads them everywhere (dead, but read), so it keeps every element
    const int* eslab_T = (in.psi_slab != 0 && in.elem_slab.numel())
        ? in.elem_slab.data_ptr<int32_t>() : nullptr;
    if (def)
        SEM3D_LAUNCH_SLAB2(slab, col, sem3d_acoustic_cpml_grads_T_deformed,
                          E, m, sh_t, stream,
            in.gbx.data_ptr<float>(), in.gby.data_ptr<float>(), in.gbz.data_ptr<float>(),
            in.gbkx.data_ptr<float>(), in.gbky.data_ptr<float>(), in.gbkz.data_ptr<float>(),
            in.gbax.data_ptr<float>(), in.gbay.data_ptr<float>(), in.gbaz.data_ptr<float>(),
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.w_jac.data_ptr<float>(),
            in.ixx.data_ptr<float>(), in.ixy.data_ptr<float>(), in.ixz.data_ptr<float>(),
            in.iex.data_ptr<float>(), in.iey.data_ptr<float>(), in.iez.data_ptr<float>(),
            in.igx.data_ptr<float>(), in.igy.data_ptr<float>(), in.igz.data_ptr<float>(), c_ixx, c_iey,
            l[0], l[1], l[2], l[3], ngll, E, ng, eslab_T, sl);
    else
        SEM3D_LAUNCH_SLAB(slab, sem3d_acoustic_cpml_grads_T,
                          E, m, sh_t, stream,
            in.gbx.data_ptr<float>(), in.gby.data_ptr<float>(), in.gbz.data_ptr<float>(),
            in.gbkx.data_ptr<float>(), in.gbky.data_ptr<float>(), in.gbkz.data_ptr<float>(),
            in.gbax.data_ptr<float>(), in.gbay.data_ptr<float>(), in.gbaz.data_ptr<float>(),
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.w_jac.data_ptr<float>(), in.w_jx.data_ptr<float>(),
            in.w_jy.data_ptr<float>(), in.w_jz.data_ptr<float>(), ihx, ihy, ihz,
            l[0], l[1], l[2], l[3], ngll, E, ng, eslab_T, sl);
    if (bs) {
        // Reverse-march u one level: u^{k-1} = 2 u^k - u^{k+1}
        // + dt2 (vp2 w_sum^k + invM pat wav[k]), then dirichlet + the exact
        // band restore. Every operation is POINTWISE and w_sum came out of
        // the already-exchanged assembly (step_T above), so the march needs
        // NO extra interface exchange: both copies of a cut-plane node carry
        // the same inputs and reproduce the same value. src_pattern is the
        // full slice here, not the ownership-masked one -- this rebuilds the
        // physical field, it is not the wavelet's global reduction. The
        // caller rotates u_hi/u_lo/u_km1 after this call.
        const float dt2 = static_cast<float>(in.dt * in.dt);
        const int nt = in.nt;
        float* u_km1 = in.u_km1.data_ptr<float>();
        if (k == 1) {
            cudaMemsetAsync(u_km1, 0, ngb, stream);          // u^0 = 0
        } else {
            sem2d::sem2d_acoustic_cpml_newmark<<<nb, TPB, 0, stream>>>(
                in.u_lo.data_ptr<float>(), in.u_hi.data_ptr<float>(),
                in.vp2_node.data_ptr<float>(), in.w_sum.data_ptr<float>(),
                in.inv_M.data_ptr<float>(), in.src_pattern.data_ptr<float>(),
                in.wavelet.data_ptr<float>(), static_cast<int>(k), dt2,
                u_km1, ng, 0, 0);
            const int kdir = static_cast<int>(in.dirichlet_idx.numel());
            if (kdir > 0) {
                const int db = (kdir + TPB - 1) / TPB;
                sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(
                    u_km1, in.dirichlet_idx.data_ptr<int64_t>(), kdir);
            }
            const int nband = static_cast<int>(in.band_idx.numel());
            if (nband > 0)
                sem2d::sem2d_band_load(
                    sem2d::sem_hist(in.band_u, in.band_scale, in.band_kind,
                                    0, nt),
                    static_cast<int>(k) - 2, in.band_idx.data_ptr<int64_t>(),
                    u_km1, nband, TPB, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic3d_cpml_dd_bwd_end(const Sem3dCpmlBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const bool slab = in.psi_slab != 0;
    sem3d_check_slab(in);
    const SemSlabs sl = sem3d_slabs(in);
    // w_sum^k: the forward's tape row, or what bwd_mid recomputed
    const bool wt = in.bs == 0 && in.save_w_sum.numel() > 0;
    const float* wsum_k = (wt && k >= 1)
        ? in.save_w_sum.data_ptr<float>() + static_cast<size_t>(k) * ng
        : in.w_sum.data_ptr<float>();
    SEM3D_LAUNCH_SLAB(slab, sem3d_acoustic_cpml_combine, nb, TPB, 0, stream,
        in.gA.data_ptr<float>(), in.gB.data_ptr<float>(), in.lu_G.data_ptr<float>(),
        in.vp2_node.data_ptr<float>(), wsum_k,
        in.lpsi_x.data_ptr<float>(), in.lpsi_y.data_ptr<float>(),
        in.lpsi_z.data_ptr<float>(), in.lzeta_x.data_ptr<float>(),
        in.lzeta_y.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
        in.lapx.data_ptr<float>(), in.lapy.data_ptr<float>(), in.lapz.data_ptr<float>(),
        in.gw.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.a_y.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.gu_k.data_ptr<float>(),
        in.lpsi_x_n.data_ptr<float>(), in.lpsi_y_n.data_ptr<float>(),
        in.lpsi_z_n.data_ptr<float>(), in.lzeta_x_n.data_ptr<float>(),
        in.lzeta_y_n.data_ptr<float>(), in.lzeta_z_n.data_ptr<float>(),
        in.grad_vp2.data_ptr<float>(), dt2, ng, sl);
    // wavelet gradient: a global reduction, so it reads the ownership-masked
    // pattern (a cut plane is complete on BOTH strips after the exchange)
    const float* pat_w = in.src_pattern_own.numel()
        ? in.src_pattern_own.data_ptr<float>() : in.src_pattern.data_ptr<float>();
    sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(
        pat_w, in.inv_M.data_ptr<float>(), in.gA.data_ptr<float>(), dt2,
        static_cast<int>(k), in.grad_wavelet.data_ptr<float>(), ng);
    sem2d::sem2d_scale1<<<nb, TPB, 0, stream>>>(
        in.gA.data_ptr<float>(), -1.f, in.gB.data_ptr<float>(), ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic3d_backward(const Sem3dElasticBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt, B = in.batch;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    SWEEP_CHECK(B == 1, "sem3d driver runs batch=1 -- see the batch note in "
                "sem3d/kernels.cuh before lifting this.");

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const bool def = in.deformed != 0;
    const float* lam = in.lam.data_ptr<float>();
    const float* mu = in.mu.data_ptr<float>();
    const float* l2m = in.lam2mu.data_ptr<float>();
    const float* w3 = in.w3.data_ptr<float>();
    const float* jac = in.jac.data_ptr<float>();
    const float* wJ = def ? in.wJ.data_ptr<float>() : nullptr;
    const float* Ixx = def ? in.ixx.data_ptr<float>() : nullptr;
    const float* Ixy = def ? in.ixy.data_ptr<float>() : nullptr;
    const float* Ixz = def ? in.ixz.data_ptr<float>() : nullptr;
    const float* Iex = def ? in.iex.data_ptr<float>() : nullptr;
    const float* Iey = def ? in.iey.data_ptr<float>() : nullptr;
    const float* Iez = def ? in.iez.data_ptr<float>() : nullptr;
    const float* Igx = def ? in.igx.data_ptr<float>() : nullptr;
    const float* Igy = def ? in.igy.data_ptr<float>() : nullptr;
    const float* Igz = def ? in.igz.data_ptr<float>() : nullptr;
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* px_ = in.pat_x.data_ptr<float>();
    const float* py_ = in.pat_y.data_ptr<float>();
    const float* pz_ = in.pat_z.data_ptr<float>();
    const float* wav = in.wavelet.data_ptr<float>();
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* gsx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* gsy = nrec > 0 ? in.grad_sy.data_ptr<float>() : nullptr;
    const float* gsz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;
    const float* sux = in.save_ux.data_ptr<float>();
    const float* suy = in.save_uy.data_ptr<float>();
    const float* suz = in.save_uz.data_ptr<float>();
    const float Gxx = (float)in.Gxx, Gyy = (float)in.Gyy, Gzz = (float)in.Gzz;
    const float Gxy = (float)in.Gxy, Gxz = (float)in.Gxz, Gyz = (float)in.Gyz;

    float* ax1 = in.ax1.data_ptr<float>(); float* ay1 = in.ay1.data_ptr<float>();
    float* az1 = in.az1.data_ptr<float>();
    float* ax2 = in.ax2.data_ptr<float>(); float* ay2 = in.ay2.data_ptr<float>();
    float* az2 = in.az2.data_ptr<float>();
    float* axm = in.axm.data_ptr<float>(); float* aym = in.aym.data_ptr<float>();
    float* azm = in.azm.data_ptr<float>();
    float* tx = in.tmp_x.data_ptr<float>(); float* ty = in.tmp_y.data_ptr<float>();
    float* tz = in.tmp_z.data_ptr<float>();
    float* Kx = in.Kx.data_ptr<float>(); float* Ky = in.Ky.data_ptr<float>();
    float* Kz = in.Kz.data_ptr<float>();
    float* adx = in.adj_x.data_ptr<float>(); float* ady = in.adj_y.data_ptr<float>();
    float* adz = in.adj_z.data_ptr<float>();
    float* yx = in.yx.data_ptr<float>(); float* yy = in.yy.data_ptr<float>();
    float* yz = in.yz.data_ptr<float>();
    float* gLam = in.grad_lam.data_ptr<float>();
    float* gMu = in.grad_mu.data_ptr<float>();
    float* gL2m = in.grad_lam2mu.data_ptr<float>();
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();

    const size_t sh_st = static_cast<size_t>(12 * m) * sizeof(float);
    const size_t sh_im = static_cast<size_t>(6 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int mm = nt; mm >= 1; --mm) {
        const int k = mm - 1;
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(ax1, inv_M, 1.f, tx, ng);
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(ay1, inv_M, 1.f, ty, ng);
    // columnar metric (sem3d_types.h). Set only by prop_c.metric_consts, and
    // read only by the deformed kernels below; 0 is the full 3x3 path.
    const bool col = in.columnar != 0;
    const float c_ixx = static_cast<float>(in.col_ixx);
    const float c_iey = static_cast<float>(in.col_iey);
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(az1, inv_M, 1.f, tz, ng);
        cudaMemsetAsync(Kx, 0, ngb, stream);
        cudaMemsetAsync(Ky, 0, ngb, stream);
        cudaMemsetAsync(Kz, 0, ngb, stream);
        if (def)
            SEM3D_LAUNCH_SLAB(col, sem3d_elastic_stiffness_deformed,
                              grid, m, sh_st, stream,
                tx, ty, tz, ibool, D, lam, mu, l2m, wJ,
                Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz, c_ixx, c_iey,
                Kx, Ky, Kz, ngll, E, ng);
        else
            sem3d_elastic_stiffness<<<grid, m, sh_st, stream>>>(
                tx, ty, tz, ibool, D, lam, mu, l2m, w3, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz,
                jac, Kx, Ky, Kz, ngll, E, ng);
        cudaMemsetAsync(adx, 0, ngb, stream);
        cudaMemsetAsync(ady, 0, ngb, stream);
        cudaMemsetAsync(adz, 0, ngb, stream);
        if (nrec > 0) {
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsx, rec_G, rec_W, adx, k, nt, nrec, m);
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsy, rec_G, rec_W, ady, k, nt, nrec, m);
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsz, rec_G, rec_W, adz, k, nt, nrec, m);
        }
        sem2d::sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(ax1, ax2, Kx, adx, dt2, axm, ng);
        sem2d::sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(ay1, ay2, Ky, ady, dt2, aym, ng);
        sem2d::sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(az1, az2, Kz, adz, dt2, azm, ng);

        const bool u_is_zero = (mm == 1);
        cudaMemsetAsync(Kx, 0, ngb, stream);
        cudaMemsetAsync(Ky, 0, ngb, stream);
        cudaMemsetAsync(Kz, 0, ngb, stream);
        const float* uxk = u_is_zero ? nullptr : sux + static_cast<size_t>(mm - 2) * ng;
        const float* uyk = u_is_zero ? nullptr : suy + static_cast<size_t>(mm - 2) * ng;
        const float* uzk = u_is_zero ? nullptr : suz + static_cast<size_t>(mm - 2) * ng;
        if (!u_is_zero) {
            if (def)
                SEM3D_LAUNCH_SLAB(col, sem3d_elastic_stiffness_deformed,
                                  grid, m, sh_st, stream,
                    uxk, uyk, uzk, ibool, D, lam, mu, l2m, wJ,
                    Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz, c_ixx, c_iey,
                    Kx, Ky, Kz, ngll, E, ng);
            else
                sem3d_elastic_stiffness<<<grid, m, sh_st, stream>>>(
                    uxk, uyk, uzk, ibool, D, lam, mu, l2m, w3, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz,
                    jac, Kx, Ky, Kz, ngll, E, ng);
        }
        sem2d::sem2d_accum_gradM<<<nb, TPB, 0, stream>>>(axm, px_, Kx, wav, k, dt2, gM, ng);
        sem2d::sem2d_accum_gradM<<<nb, TPB, 0, stream>>>(aym, py_, Ky, wav, k, dt2, gM, ng);
        sem2d::sem2d_accum_gradM<<<nb, TPB, 0, stream>>>(azm, pz_, Kz, wav, k, dt2, gM, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(px_, inv_M, axm, dt2, k, gW, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(py_, inv_M, aym, dt2, k, gW, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(pz_, inv_M, azm, dt2, k, gW, ng);
        if (!u_is_zero) {
            sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, axm, -dt2, yx, ng);
            sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, aym, -dt2, yy, ng);
            sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, azm, -dt2, yz, ng);
            if (def)
                SEM3D_LAUNCH_SLAB(col, sem3d_elastic_imaging_deformed,
                                  grid, m, sh_im, stream,
                    uxk, uyk, uzk, yx, yy, yz, ibool, D, wJ,
                    Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz, c_ixx, c_iey,
                    gL2m, gLam, gMu, ngll, E, ng);
            else
                sem3d_elastic_imaging<<<grid, m, sh_im, stream>>>(
                    uxk, uyk, uzk, yx, yy, yz, ibool, D, w3, jac,
                    Gxx, Gyy, Gzz, Gxy, Gxz, Gyz, gL2m, gLam, gMu, ngll, E, ng);
        }
        float* t;
        t = ax2; ax2 = ax1; ax1 = axm; axm = t;
        t = ay2; ay2 = ay1; ay1 = aym; aym = t;
        t = az2; az2 = az1; az1 = azm; azm = t;
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- 3-D velocity-stress elastic backward (P8) ------------------------------
// Reverse loop k = nt-1..0 over the staggered leapfrog. Per step:
// record^T -> stress_update^T (imaging + mvel adjoints + Grad^T into lv) ->
// vel_update^T (grad_inv_M, lR) + grad_wav -> sigma_div^T (msig adjoints,
// lsig update). lv/lsig/lmsig/lmvel carry across steps in place.
void elastic_vs3d_backward(const Sem3dVsBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt, B = in.batch;
    const float dt = static_cast<float>(in.dt);
    const float ihx = (float)in.ihx, ihy = (float)in.ihy, ihz = (float)in.ihz;
    const bool cp = in.a_x.numel() > 0;
    SWEEP_CHECK(B == 1, "sem3d driver runs batch=1 -- see the batch note in "
                "sem3d/kernels.cuh before lifting this.");

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* lam = in.lam.data_ptr<float>();
    const float* mu = in.mu.data_ptr<float>();
    const float* l2m = in.lam2mu.data_ptr<float>();
    const float* wHx = in.wHx.data_ptr<float>();
    const float* wHy = in.wHy.data_ptr<float>();
    const float* wHz = in.wHz.data_ptr<float>();
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* ax = cp ? in.a_x.data_ptr<float>() : nullptr;
    const float* bx = cp ? in.b_x.data_ptr<float>() : nullptr;
    const float* ay = cp ? in.a_y.data_ptr<float>() : nullptr;
    const float* by = cp ? in.b_y.data_ptr<float>() : nullptr;
    const float* az = cp ? in.a_z.data_ptr<float>() : nullptr;
    const float* bz = cp ? in.b_z.data_ptr<float>() : nullptr;
    const float* px_ = in.pat_x.data_ptr<float>();
    const float* py_ = in.pat_y.data_ptr<float>();
    const float* pz_ = in.pat_z.data_ptr<float>();
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* gsx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* gsy = nrec > 0 ? in.grad_sy.data_ptr<float>() : nullptr;
    const float* gsz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;
    const float* svx = in.save_vx.data_ptr<float>();
    const float* svy = in.save_vy.data_ptr<float>();
    const float* svz = in.save_vz.data_ptr<float>();
    const float* sgv = in.save_gv.data_ptr<float>();

    float* lvx = in.lvx.data_ptr<float>();
    float* lvy = in.lvy.data_ptr<float>();
    float* lvz = in.lvz.data_ptr<float>();
    float* lRx = in.lRx.data_ptr<float>();
    float* lRy = in.lRy.data_ptr<float>();
    float* lRz = in.lRz.data_ptr<float>();
    float* lsig = in.lsig.data_ptr<float>();
    float* lmsig = in.lmsig.data_ptr<float>();
    float* lmvel = in.lmvel.data_ptr<float>();
    float* gLam = in.grad_lam.data_ptr<float>();
    float* gMu = in.grad_mu.data_ptr<float>();
    float* gL2m = in.grad_lam2mu.data_ptr<float>();
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();
    const long Em = static_cast<long>(E) * m;

    const size_t sh9 = static_cast<size_t>(9 * m) * sizeof(float);
    const size_t sh3 = static_cast<size_t>(3 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;

    for (int k = nt - 1; k >= 0; --k) {
        if (nrec > 0) {
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsx, rec_G, rec_W, lvx, k, nt, nrec, m);
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsy, rec_G, rec_W, lvy, k, nt, nrec, m);
            sem2d::sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsz, rec_G, rec_W, lvz, k, nt, nrec, m);
        }
        sem3d_vs_stress_update_T<<<grid, m, sh9, stream>>>(
            lsig, sgv + static_cast<size_t>(k) * 9 * Em, lmvel,
            ax, bx, ay, by, az, bz, lam, mu, l2m, ibool, D,
            ihx, ihy, ihz, dt, lvx, lvy, lvz, gLam, gMu, gL2m, ngll, E, ng);
        const size_t ok = static_cast<size_t>(k) * ng;
        const float* vk1x = k >= 1 ? svx + (ok - ng) : nullptr;
        const float* vk1y = k >= 1 ? svy + (ok - ng) : nullptr;
        const float* vk1z = k >= 1 ? svz + (ok - ng) : nullptr;
        sem3d_vs_vel_update_T<<<nb, TPB, 0, stream>>>(
            lvx, lvy, lvz, svx + ok, svy + ok, svz + ok, vk1x, vk1y, vk1z,
            inv_M, dt, lRx, lRy, lRz, gM, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(px_, inv_M, lvx, dt, k, gW, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(py_, inv_M, lvy, dt, k, gW, ng);
        sem2d::sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(pz_, inv_M, lvz, dt, k, gW, ng);
        sem3d_vs_sigma_div_T<<<grid, m, sh3, stream>>>(
            lRx, lRy, lRz, lsig, lmsig, ax, bx, ay, by, az, bz,
            wHx, wHy, wHz, ibool, D, ngll, E, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem3d

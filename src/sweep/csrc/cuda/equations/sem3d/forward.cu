// sweep.sem CUDA (impl='c') backend — 3-D acoustic SEM forward (P5).
// Reuses the dimension-agnostic nodal kernels from sem2d (newmark, dirichlet,
// record — they operate on flat arrays; record takes m = ngll^3).
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "sem3d.h"
#include "kernels.cuh"
#include "../sem2d/kernels.cuh"

namespace sem3d {

void acoustic3d_forward(const Sem3dForwardInputCore& in) {
    sweep::DeviceGuard guard(in.u.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt, B = in.batch;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const long ngB = static_cast<long>(ng) * B;

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
    float* seis = nrec > 0 ? in.seis.data_ptr<float>() : nullptr;

    float* u = in.u.data_ptr<float>();
    float* u_prev = in.u_prev.data_ptr<float>();
    float* u_next = in.u_next.data_ptr<float>();
    float* K = in.K.data_ptr<float>();
    const bool save = in.save_wavefield.numel() > 0;
    float* save_wf = save ? in.save_wavefield.data_ptr<float>() : nullptr;

    // batch = 1 today for the nodal kernels (newmark's pattern is (n_global,));
    // the element kernels already take grid.y = batch.
    SWEEP_CHECK(B == 1, "sem3d driver runs batch=1: the ELEMENT kernels are "
                "batch-ready but the NODAL ones are not (shared per-node arrays + a "
                "non-atomic shared gradient). See the batch note in sem3d/kernels.cuh.");

    const size_t shmem = static_cast<size_t>(4 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    // Sized for ng, NOT ngB: every nodal kernel below is passed `ng` as its
    // bound, so an ngB-wide grid would just early-return the extra threads and
    // silently never step batches 1..B-1. See the batch note in kernels.cuh.
    const int nb = static_cast<int>((ng + TPB - 1) / TPB);
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ngB) * sizeof(float);

    for (int n = 0; n < nt; ++n) {
        // u_next = 2u - u_prev + dt2 inv_M (pat w - c^2 K0 u): K0 u lands in
        // u_next (scratch until newmark writes it), then K = c^2 o K0 u
        cudaMemsetAsync(u_next, 0, ngb, stream);
        if (def)
            sem3d_acoustic_stiffness_deformed<<<grid, m, shmem, stream>>>(
                u, ibool, D, wJ, Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz,
                u_next, ngll, E, ng);
        else
            sem3d_acoustic_stiffness<<<grid, m, shmem, stream>>>(
                u, ibool, D, jx, jy, jz, u_next, ngll, E, ng);
        sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(u_next, c2n, 1.f, K, ng);
        sem2d::sem2d_acoustic_newmark<<<nb, TPB, 0, stream>>>(
            u, u_prev, K, pattern, inv_M, wavelet, n, dt2, u_next, ng);
        if (kdir > 0)
            sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(u_next, didx, kdir);
        if (nrec > 0)
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(
                u_next, rec_G, rec_W, seis, n, nt, nrec, m, ng);
        if (save)
            cudaMemcpyAsync(save_wf + static_cast<size_t>(n) * ngB, u_next, ngb,
                            cudaMemcpyDeviceToDevice, stream);
        float* t = u_prev; u_prev = u; u = u_next; u_next = t;
    }
    if (in.bs) {
        cudaMemcpyAsync(in.fin_u.data_ptr<float>(), u, ngb,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_u_prev.data_ptr<float>(), u_prev, ngb,
                        cudaMemcpyDeviceToDevice, stream);
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- 3-D acoustic forward WITH C-PML (P6) -----------------------------------
// psi_slab: refuse a table / array set that does not describe this grid.
// Python builds both sides; a mismatch would address the wrong node silently.
static void sem3d_check_slab(const Sem3dCpmlForwardInputCore& in) {
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
    SWEEP_CHECK(in.ring_slot_x.element_size() == 8 && in.dir_slot_x.element_size() == 8
                && in.ring_slot_y.element_size() == 8 && in.ring_slot_z.element_size() == 8,
                "psi_slab: slot lists must be int64");
    const long n[3] = {sem3d_slab_len(0, in), sem3d_slab_len(1, in), sem3d_slab_len(2, in)};
    const Buf* f[21] = {&in.psi_x, &in.psi_y, &in.psi_z, &in.zeta_x, &in.zeta_y, &in.zeta_z,
                                  &in.psi_x_n, &in.psi_y_n, &in.psi_z_n,
                                  &in.zeta_x_n, &in.zeta_y_n, &in.zeta_z_n,
                                  &in.apsi_x, &in.apsi_y, &in.apsi_z,
                                  &in.gx, &in.gy, &in.gz,
                                  &in.kx, &in.ky, &in.kz};
    for (int j = 0; j < 21; ++j)
        SWEEP_CHECK(f[j]->numel() == n[j % 3], "psi_slab: memory field ", j, " has ",
                    f[j]->numel(), " entries, its slab has ", n[j % 3]);
}

// ---- 3-D elastic forward (P7, bare) -----------------------------------------
// ---- DD stepped forward (3-D acoustic C-PML) --------------------------------
// The 3-D step has the same shape as its 2-D twin: ONE assembly (the C-PML
// grads, nine assembled fields) and pointwise algebra after it. A cut is a
// whole YZ PLANE of nodes rather than a column, but the global numbering is
// ((iz*ny_g + iy)*nx_g + ix) -- x fastest -- so a cut plane is still a
// strided index set the caller builds once, and the exchange is the same
// "sum the shared nodes" operation. batch == 1; rotation is the caller's.
void acoustic3d_cpml_dd_asm(const Sem3dCpmlForwardInputCore& in) {
    sweep::DeviceGuard guard(in.u.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll;
    SWEEP_CHECK(in.batch == 1, "acoustic3d_cpml_dd: batch=1");
    const float ihx = static_cast<float>(in.ihx), ihy = static_cast<float>(in.ihy),
                ihz = static_cast<float>(in.ihz);
    const bool def = in.deformed != 0;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    float* apx = in.apsi_x.data_ptr<float>();
    float* apy = in.apsi_y.data_ptr<float>();
    float* apz = in.apsi_z.data_ptr<float>();
    const bool slab = in.psi_slab != 0;
    sem3d_check_slab(in);
    const SemSlabs sl = sem3d_slabs(in);
    const float* ps[3] = {in.psi_x.data_ptr<float>(), in.psi_y.data_ptr<float>(),
                          in.psi_z.data_ptr<float>()};
    const float* as[3] = {in.a_x.data_ptr<float>(), in.a_y.data_ptr<float>(),
                          in.a_z.data_ptr<float>()};
    float* aps[3] = {apx, apy, apz};
    for (int axis = 0; axis < 3; ++axis) {
        if (slab) {
            const int n_ = sem3d_slab_len(axis, in);
            if (n_ > 0)
                sem3d_slab_scale<<<(n_ + TPB - 1) / TPB, TPB, 0, stream>>>(
                    as[axis], ps[axis], aps[axis], axis, sl, n_);
        } else {
            sem2d::sem2d_scale2<<<nb, TPB, 0, stream>>>(as[axis], ps[axis], 1.f, aps[axis], ng);
        }
    }
    float* g[9] = {in.gx.data_ptr<float>(), in.gy.data_ptr<float>(),
                   in.gz.data_ptr<float>(), in.kx.data_ptr<float>(),
                   in.ky.data_ptr<float>(), in.kz.data_ptr<float>(),
                   in.mx.data_ptr<float>(), in.my.data_ptr<float>(),
                   in.mz.data_ptr<float>()};
    sem3d_zero_g9(in, slab, stream, g);
    // columnar metric (sem3d_types.h). Set only by prop_c.metric_consts, and
    // read only by the deformed kernels below; 0 is the full 3x3 path.
    const bool col = in.columnar != 0;
    const float c_ixx = static_cast<float>(in.col_ixx);
    const float c_iey = static_cast<float>(in.col_iey);
    const size_t shmem = static_cast<size_t>(7 * m) * sizeof(float);
    // the a*psi skip (sem3d_types.h): empty tensor = every element flagged
    const int* eslab = in.elem_slab.numel() ? in.elem_slab.data_ptr<int32_t>() : nullptr;
    if (def)
        SEM3D_LAUNCH_SLAB2(slab, col, sem3d_acoustic_cpml_grads_deformed, E, m, shmem, stream,
            in.u.data_ptr<float>(), apx, apy, apz,
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.w_jac.data_ptr<float>(),
            in.ixx.data_ptr<float>(), in.ixy.data_ptr<float>(),
            in.ixz.data_ptr<float>(), in.iex.data_ptr<float>(),
            in.iey.data_ptr<float>(), in.iez.data_ptr<float>(),
            in.igx.data_ptr<float>(), in.igy.data_ptr<float>(),
            in.igz.data_ptr<float>(), c_ixx, c_iey,
            g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], ngll, E, ng,
            sl, eslab);
    else
        SEM3D_LAUNCH_SLAB(slab, sem3d_acoustic_cpml_grads, E, m, shmem, stream,
            in.u.data_ptr<float>(), apx, apy, apz,
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.w_jac.data_ptr<float>(), in.w_jx.data_ptr<float>(),
            in.w_jy.data_ptr<float>(), in.w_jz.data_ptr<float>(),
            ihx, ihy, ihz,
            g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], ngll, E, ng,
            sl, eslab);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic3d_cpml_dd_step(const Sem3dCpmlForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.u.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const bool slab = in.psi_slab != 0;
    SWEEP_CHECK(!(slab && in.save_u.numel() > 0), "psi_slab: the save-all tape "
                "copies the full-length memory fields; use the full layout with tapes");
    sem3d_check_slab(in);
    const SemSlabs sl = sem3d_slabs(in);
    SEM3D_LAUNCH_SLAB(slab, sem3d_acoustic_cpml_step_newmark, nb, TPB, 0, stream,
        in.gx.data_ptr<float>(), in.gy.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.ky.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.mx.data_ptr<float>(), in.my.data_ptr<float>(), in.mz.data_ptr<float>(),
        in.inv_M.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(), in.dbx_dx.data_ptr<float>(),
        in.a_y.data_ptr<float>(), in.b_y.data_ptr<float>(), in.dby_dy.data_ptr<float>(),
        in.a_z.data_ptr<float>(), in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        in.band_node.data_ptr<uint8_t>(),
        in.psi_x.data_ptr<float>(), in.psi_y.data_ptr<float>(), in.psi_z.data_ptr<float>(),
        in.zeta_x.data_ptr<float>(), in.zeta_y.data_ptr<float>(), in.zeta_z.data_ptr<float>(),
        in.u.data_ptr<float>(), in.u_prev.data_ptr<float>(),
        in.vp2_node.data_ptr<float>(), in.src_pattern.data_ptr<float>(),
        in.wavelet.data_ptr<float>(), static_cast<int>(n), dt2,
        in.w_sum.data_ptr<float>(), in.u_next.data_ptr<float>(),
        in.psi_x_n.data_ptr<float>(), in.psi_y_n.data_ptr<float>(),
        in.psi_z_n.data_ptr<float>(), in.zeta_x_n.data_ptr<float>(),
        in.zeta_y_n.data_ptr<float>(), in.zeta_z_n.data_ptr<float>(), ng, sl);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        const long* didx = in.dirichlet_idx.data_ptr<int64_t>();
        sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(in.u_next.data_ptr<float>(), didx, kdir);
        float* f[6] = {in.psi_x_n.data_ptr<float>(), in.psi_y_n.data_ptr<float>(),
                       in.psi_z_n.data_ptr<float>(), in.zeta_x_n.data_ptr<float>(),
                       in.zeta_y_n.data_ptr<float>(), in.zeta_z_n.data_ptr<float>()};
        if (slab) {   // the slab's own Dirichlet nodes, in slot coordinates
            const Buf* ds[3] = {&in.dir_slot_x, &in.dir_slot_y, &in.dir_slot_z};
            for (int j = 0; j < 6; ++j) {
                const int kd = static_cast<int>(ds[j % 3]->numel());
                if (kd > 0)
                    sem2d::sem2d_dirichlet_fill<<<(kd + TPB - 1) / TPB, TPB, 0, stream>>>(
                        f[j], ds[j % 3]->data_ptr<int64_t>(), kd);
            }
        } else {
            for (int j = 0; j < 6; ++j)
                sem2d::sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(f[j], didx, kdir);
        }
    }
    if (nrec > 0) {
        const int rb = (nrec + TPB - 1) / TPB;
        sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(
            in.u_next.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.seis.data_ptr<float>(),
            static_cast<int>(n), nt, nrec, m, ng);
    }
    if (in.save_u.numel() > 0) {
        const size_t o = static_cast<size_t>(n) * ng;
        const float* src[7] = {in.u_next.data_ptr<float>(),
                               in.psi_x_n.data_ptr<float>(),
                               in.psi_y_n.data_ptr<float>(),
                               in.psi_z_n.data_ptr<float>(),
                               in.zeta_x_n.data_ptr<float>(),
                               in.zeta_y_n.data_ptr<float>(),
                               in.zeta_z_n.data_ptr<float>()};
        float* dst[7] = {in.save_u.data_ptr<float>(),
                         in.save_psi_x.data_ptr<float>(),
                         in.save_psi_y.data_ptr<float>(),
                         in.save_psi_z.data_ptr<float>(),
                         in.save_zeta_x.data_ptr<float>(),
                         in.save_zeta_y.data_ptr<float>(),
                         in.save_zeta_z.data_ptr<float>()};
        for (int j = 0; j < 7; ++j)
            cudaMemcpyAsync(dst[j] + o, src[j], ngb,
                            cudaMemcpyDeviceToDevice, stream);
    }
    if (in.save_w_sum.numel() > 0)      // the w_sum tape (see the struct)
        cudaMemcpyAsync(in.save_w_sum.data_ptr<float>() + static_cast<size_t>(n) * ng,
                        in.w_sum.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
    if (in.bs != 0) {
        // Boundary saving under DD: the ring is this STRIP's own C-PML band.
        // The cut plane is often INSIDE that band (the y and bottom rims
        // reach it), which is fine: the profile is a slice of one global
        // build, so both strips give the plane the same membership and
        // therefore reconstruct it the same way -- from the ring, or by the
        // pointwise march whose inputs were already exchanged. Either way
        // BS composes with the decomposition at zero extra exchange.
        const int nband = static_cast<int>(in.band_idx.numel());
        if (nband > 0) {
            const long* bg = in.band_idx.data_ptr<int64_t>();
            const int bkind = in.band_kind;
            float* f[7] = {in.u_next.data_ptr<float>(),
                           in.psi_x_n.data_ptr<float>(),
                           in.psi_y_n.data_ptr<float>(),
                           in.psi_z_n.data_ptr<float>(),
                           in.zeta_x_n.data_ptr<float>(),
                           in.zeta_y_n.data_ptr<float>(),
                           in.zeta_z_n.data_ptr<float>()};
            const Buf* rg[7] = {&in.band_u, &in.band_psi_x,
                                          &in.band_psi_y, &in.band_psi_z,
                                          &in.band_zeta_x, &in.band_zeta_y,
                                          &in.band_zeta_z};
            const bool bsplit = in.band_split != 0;
            SWEEP_CHECK(!(slab && !bsplit), "psi_slab needs band_split: the "
                        "union ring index addresses nodes off the slab");
            const sem2d::SemBandIdx bi_x = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_x : in.band_idx_x, bg, nband);
            const sem2d::SemBandIdx bi_y = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_y : in.band_idx_y, bg, nband);
            const sem2d::SemBandIdx bi_z = sem2d::sem_band_idx(bsplit, slab ? in.ring_slot_z : in.band_idx_z, bg, nband);
            const sem2d::SemBandIdx bi[7] = {{bg, nband}, bi_x, bi_y, bi_z, bi_x, bi_y, bi_z};
            for (int j = 0; j < 7; ++j)
                sem2d::sem2d_band_save(
                    f[j], bi[j].idx,
                    sem2d::sem_hist(*rg[j], in.band_scale, bkind, j, nt),
                    static_cast<int>(n), bi[j].n, TPB, stream);
        }
        if (n == nt - 1) {   // seed the backward's reverse march
            cudaMemcpyAsync(in.fin_u.data_ptr<float>(),
                            in.u_next.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(in.fin_u_prev.data_ptr<float>(),
                            in.u.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic3d_forward(const Sem3dElasticForwardInputCore& in) {
    sweep::DeviceGuard guard(in.ux.device_index());
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
    float* sx = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* sy = nrec > 0 ? in.seis_y.data_ptr<float>() : nullptr;
    float* sz = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;
    const float Gxx = (float)in.Gxx, Gyy = (float)in.Gyy, Gzz = (float)in.Gzz;
    const float Gxy = (float)in.Gxy, Gxz = (float)in.Gxz, Gyz = (float)in.Gyz;

    float* ux = in.ux.data_ptr<float>(); float* uy = in.uy.data_ptr<float>();
    float* uz = in.uz.data_ptr<float>();
    float* uxp = in.ux_prev.data_ptr<float>(); float* uyp = in.uy_prev.data_ptr<float>();
    float* uzp = in.uz_prev.data_ptr<float>();
    float* uxn = in.ux_next.data_ptr<float>(); float* uyn = in.uy_next.data_ptr<float>();
    float* uzn = in.uz_next.data_ptr<float>();
    float* Kx = in.Kx.data_ptr<float>(); float* Ky = in.Ky.data_ptr<float>();
    float* Kz = in.Kz.data_ptr<float>();
    const bool save = in.save_ux.numel() > 0;
    float* s_ux = save ? in.save_ux.data_ptr<float>() : nullptr;
    float* s_uy = save ? in.save_uy.data_ptr<float>() : nullptr;
    float* s_uz = save ? in.save_uz.data_ptr<float>() : nullptr;

    const size_t shmem = static_cast<size_t>(12 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int n = 0; n < nt; ++n) {
    // columnar metric (sem3d_types.h). Set only by prop_c.metric_consts, and
    // read only by the deformed kernels below; 0 is the full 3x3 path.
    const bool col = in.columnar != 0;
    const float c_ixx = static_cast<float>(in.col_ixx);
    const float c_iey = static_cast<float>(in.col_iey);
        cudaMemsetAsync(Kx, 0, ngb, stream);
        cudaMemsetAsync(Ky, 0, ngb, stream);
        cudaMemsetAsync(Kz, 0, ngb, stream);
        if (def)
            SEM3D_LAUNCH_SLAB(col, sem3d_elastic_stiffness_deformed,
                              grid, m, shmem, stream,
                ux, uy, uz, ibool, D, lam, mu, l2m, wJ,
                Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz, c_ixx, c_iey,
                Kx, Ky, Kz, ngll, E, ng);
        else
            sem3d_elastic_stiffness<<<grid, m, shmem, stream>>>(
                ux, uy, uz, ibool, D, lam, mu, l2m, w3, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz,
                jac, Kx, Ky, Kz, ngll, E, ng);
        sem2d::sem2d_acoustic_newmark<<<nb, TPB, 0, stream>>>(
            ux, uxp, Kx, px_, inv_M, wav, n, dt2, uxn, ng);
        sem2d::sem2d_acoustic_newmark<<<nb, TPB, 0, stream>>>(
            uy, uyp, Ky, py_, inv_M, wav, n, dt2, uyn, ng);
        sem2d::sem2d_acoustic_newmark<<<nb, TPB, 0, stream>>>(
            uz, uzp, Kz, pz_, inv_M, wav, n, dt2, uzn, ng);
        if (nrec > 0) {
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(uxn, rec_G, rec_W, sx, n, nt, nrec, m, ng);
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(uyn, rec_G, rec_W, sy, n, nt, nrec, m, ng);
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(uzn, rec_G, rec_W, sz, n, nt, nrec, m, ng);
        }
        if (save) {
            const size_t o = static_cast<size_t>(n) * ng;
            cudaMemcpyAsync(s_ux + o, uxn, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_uy + o, uyn, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_uz + o, uzn, ngb, cudaMemcpyDeviceToDevice, stream);
        }
        float* t;
        t = uxp; uxp = ux; ux = uxn; uxn = t;
        t = uyp; uyp = uy; uy = uyn; uyn = t;
        t = uzp; uzp = uz; uz = uzn; uzn = t;
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- 3-D velocity-stress elastic forward (P8, bare or CPML) -----------------
void elastic_vs3d_forward(const Sem3dVsForwardInputCore& in) {
    sweep::DeviceGuard guard(in.vx.device_index());
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
    const float* wav = in.wavelet.data_ptr<float>();
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    float* sx = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* sy = nrec > 0 ? in.seis_y.data_ptr<float>() : nullptr;
    float* sz = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;

    float* vx = in.vx.data_ptr<float>();
    float* vy = in.vy.data_ptr<float>();
    float* vz = in.vz.data_ptr<float>();
    float* Rx = in.Rx.data_ptr<float>();
    float* Ry = in.Ry.data_ptr<float>();
    float* Rz = in.Rz.data_ptr<float>();
    float* sig = in.sig.data_ptr<float>();
    float* msig = in.msig.data_ptr<float>();
    float* mvel = in.mvel.data_ptr<float>();
    const bool save = in.save_vx.numel() > 0;
    float* s_vx = save ? in.save_vx.data_ptr<float>() : nullptr;
    float* s_vy = save ? in.save_vy.data_ptr<float>() : nullptr;
    float* s_vz = save ? in.save_vz.data_ptr<float>() : nullptr;
    float* s_gv = save ? in.save_gv.data_ptr<float>() : nullptr;
    const long Em = static_cast<long>(E) * m;

    const size_t sh_v = static_cast<size_t>(9 * m) * sizeof(float);
    const size_t sh_g = static_cast<size_t>(3 * m) * sizeof(float);
    const dim3 grid(E, B);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int n = 0; n < nt; ++n) {
        cudaMemsetAsync(Rx, 0, ngb, stream);
        cudaMemsetAsync(Ry, 0, ngb, stream);
        cudaMemsetAsync(Rz, 0, ngb, stream);
        sem3d_vs_sigma_div<<<grid, m, sh_v, stream>>>(
            sig, msig, ax, bx, ay, by, az, bz, wHx, wHy, wHz,
            ibool, D, Rx, Ry, Rz, ngll, E, ng);
        const size_t o = static_cast<size_t>(n) * ng;
        sem3d_vs_vel_update<<<nb, TPB, 0, stream>>>(
            vx, vy, vz, Rx, Ry, Rz, inv_M, px_, py_, pz_, wav, n, dt,
            save ? s_vx + o : nullptr, save ? s_vy + o : nullptr,
            save ? s_vz + o : nullptr, ng);
        sem3d_vs_stress_update<<<grid, m, sh_g, stream>>>(
            vx, vy, vz, sig, mvel, ax, bx, ay, by, az, bz, lam, mu, l2m,
            ibool, D, ihx, ihy, ihz, dt,
            save ? s_gv + static_cast<size_t>(n) * 9 * Em : nullptr, ngll, E, ng);
        if (nrec > 0) {
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(vx, rec_G, rec_W, sx, n, nt, nrec, m, ng);
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(vy, rec_G, rec_W, sy, n, nt, nrec, m, ng);
            sem2d::sem2d_record<<<rb, TPB, 0, stream>>>(vz, rec_G, rec_W, sz, n, nt, nrec, m, ng);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem3d

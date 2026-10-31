// sweep.sem CUDA (impl='c') backend -- the 3-D Biot SEM drivers (BiotVSSEM3DDeformed):
// the 2-D drivers' step (sem2d/biot2d.cu) with y, the flux copies one launch per group
// of nodes with the same number of copies. Kernels: sem3d/biot3d_kernels.cuh.
#include "../../../core/device.h"
#include "../../launch/launch.h"   // SWEEP_LAUNCH
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "sem3d.h"
#include "kernels.cuh"
#include "../sem2d/kernels.cuh"
#include "../sem2d/vs2d_kernels.cuh"
#include "../sem2d/biot2d_kernels.cuh"
#include "biot3d_kernels.cuh"

namespace sem3d {

namespace {

struct Group { int k, n, coff, noff; long eoff; };

struct Biot3dCommon {
    int E, ng, ngll, m, nrec, nt, npts;
    long Em;
    float dt, decay, gain;
    bool cp, vmode, visc;
    std::vector<Group> groups;
    const int* ibool; const float* D;
    const float *wJ, *I[9], *qmask;
    const float *inv_M, *inv_Mp, *inv_M1, *inv_Mm, *pin_n;
    const float *lam_d, *mu, *alpha, *b;
    const int *perm, *node_order; const float *wr, *share, *dragE, *dragP;
    const float *ax, *bx, *ay, *by, *az, *bz, *a_n;
    const long *ifc_idx; const float *ifc_w, *ifc_nx, *ifc_ny, *ifc_nz;
    const float *pv[3], *pp, *pq[3], *wav;
    const long *rec_G, *rec_Ge; const float* rec_W;
};

template <class In>
Biot3dCommon biot3d_common(const In& in)
{
    Biot3dCommon c{};
    c.E = in.n_elements; c.ng = in.n_global; c.ngll = in.ngll;
    c.m = c.ngll * c.ngll * c.ngll; c.Em = static_cast<long>(c.E) * c.m;
    c.nrec = in.n_rec; c.nt = in.nt;
    c.dt = static_cast<float>(in.dt);
    c.decay = static_cast<float>(in.decay); c.gain = static_cast<float>(in.gain);
    SWEEP_CHECK(in.batch == 1 && in.ndim == 3, "sem3d Biot driver: 3-D, batch=1");
    c.cp = in.a_x.numel() > 0;
    c.vmode = in.velocity_mode != 0; c.visc = in.viscous != 0;
    c.npts = static_cast<int>(in.ifc_idx.numel());
    const int gk[4] = {in.g0_k, in.g1_k, in.g2_k, in.g3_k};
    const int gn[4] = {in.g0_n, in.g1_n, in.g2_n, in.g3_n};
    int coff = 0, noff = 0; long eoff = 0;
    for (int g = 0; g < 4; ++g) {
        if (gk[g] <= 0 || gn[g] <= 0) continue;
        SWEEP_CHECK(gk[g] <= 8, "sem3d Biot: a node has at most 8 flux copies");
        c.groups.push_back({gk[g], gn[g], coff, noff, eoff});
        coff += gk[g] * gn[g]; noff += gn[g];
        eoff += static_cast<long>(gn[g]) * gk[g] * gk[g];
    }
    SWEEP_CHECK(coff == c.Em && noff == c.ng, "sem3d Biot: the flux groups must cover every "
                "element node (", coff, " of ", c.Em, ") and every node (", noff, " of ", c.ng, ")");
    c.ibool = in.ibool.template data_ptr<int32_t>();
    c.D = in.D.template data_ptr<float>();
    c.wJ = in.wJ.template data_ptr<float>();
    c.I[0] = in.ixx.template data_ptr<float>(); c.I[1] = in.ixy.template data_ptr<float>();
    c.I[2] = in.ixz.template data_ptr<float>(); c.I[3] = in.iex.template data_ptr<float>();
    c.I[4] = in.iey.template data_ptr<float>(); c.I[5] = in.iez.template data_ptr<float>();
    c.I[6] = in.igx.template data_ptr<float>(); c.I[7] = in.igy.template data_ptr<float>();
    c.I[8] = in.igz.template data_ptr<float>();
    c.qmask = in.qmask.template data_ptr<float>();
    c.inv_M = in.inv_M.template data_ptr<float>();
    c.inv_Mp = in.inv_Mp.template data_ptr<float>();
    c.inv_M1 = in.inv_M1.template data_ptr<float>();
    c.inv_Mm = in.inv_Mm.template data_ptr<float>();
    c.pin_n = in.pin_n.template data_ptr<float>();
    c.lam_d = in.lam_d.template data_ptr<float>();
    c.mu = in.mu.template data_ptr<float>();
    c.alpha = in.alpha.template data_ptr<float>();
    c.b = c.visc ? in.b.template data_ptr<float>() : nullptr;
    c.perm = in.perm.template data_ptr<int32_t>();
    c.node_order = in.node_order.template data_ptr<int32_t>();
    c.wr = in.wr.template data_ptr<float>();
    c.share = in.share.template data_ptr<float>();
    c.dragE = in.dragE.template data_ptr<float>();
    c.dragP = in.dragP.template data_ptr<float>();
    c.ax = c.cp ? in.a_x.template data_ptr<float>() : nullptr;
    c.bx = c.cp ? in.b_x.template data_ptr<float>() : nullptr;
    c.ay = c.cp ? in.a_y.template data_ptr<float>() : nullptr;
    c.by = c.cp ? in.b_y.template data_ptr<float>() : nullptr;
    c.az = c.cp ? in.a_z.template data_ptr<float>() : nullptr;
    c.bz = c.cp ? in.b_z.template data_ptr<float>() : nullptr;
    c.a_n = c.cp ? in.a_n.template data_ptr<float>() : nullptr;
    c.ifc_idx = c.npts ? in.ifc_idx.template data_ptr<int64_t>() : nullptr;
    c.ifc_w = c.npts ? in.ifc_w.template data_ptr<float>() : nullptr;
    c.ifc_nx = c.npts ? in.ifc_nx.template data_ptr<float>() : nullptr;
    c.ifc_ny = c.npts ? in.ifc_ny.template data_ptr<float>() : nullptr;
    c.ifc_nz = c.npts ? in.ifc_nz.template data_ptr<float>() : nullptr;
    c.pv[0] = in.pat_vx.template data_ptr<float>(); c.pv[1] = in.pat_vy.template data_ptr<float>();
    c.pv[2] = in.pat_vz.template data_ptr<float>();
    c.pp = in.pat_p.template data_ptr<float>();
    c.pq[0] = in.pat_qx.template data_ptr<float>(); c.pq[1] = in.pat_qy.template data_ptr<float>();
    c.pq[2] = in.pat_qz.template data_ptr<float>();
    c.wav = in.wavelet.template data_ptr<float>();
    c.rec_G = c.nrec > 0 ? in.rec_G.template data_ptr<int64_t>() : nullptr;
    c.rec_Ge = c.nrec > 0 ? in.rec_Ge.template data_ptr<int64_t>() : nullptr;
    c.rec_W = c.nrec > 0 ? in.rec_W.template data_ptr<float>() : nullptr;
    return c;
}

std::vector<float> host_wavelet(const float* wav, int nt, cudaStream_t stream)
{
    std::vector<float> h(static_cast<size_t>(nt));
    cudaMemcpyAsync(h.data(), wav, sizeof(float) * nt, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

} // namespace

void biot3d_forward(const Sem3dBiotForwardInputCore& in) {
    sweep::DeviceGuard guard(in.vx.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Biot3dCommon c = biot3d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const long Em = c.Em;
    const float dt = c.dt;
    const float* const* I = c.I;
    float* vx = in.vx.data_ptr<float>();
    float* vy = in.vy.data_ptr<float>();
    float* vz = in.vz.data_ptr<float>();
    float* p = in.p.data_ptr<float>();
    float* q = in.q.data_ptr<float>();
    float* sig = in.sig.data_ptr<float>();
    float* r = in.r.data_ptr<float>();
    float* mp = c.cp ? in.mp.data_ptr<float>() : nullptr;
    float* mv = c.cp ? in.mv.data_ptr<float>() : nullptr;
    float* msig = c.cp ? in.msig.data_ptr<float>() : nullptr;
    float* mq = c.cp ? in.mq.data_ptr<float>() : nullptr;
    float* Rx = in.Rx.data_ptr<float>();
    float* Ry = in.Ry.data_ptr<float>();
    float* Rz = in.Rz.data_ptr<float>();
    float* P = c.cp ? in.P.data_ptr<float>() : nullptr;
    float* Pq = c.cp ? in.Pq.data_ptr<float>() : nullptr;
    float* gp = in.gp.data_ptr<float>();
    float* f = in.f.data_ptr<float>();
    float* adiv = in.adiv.data_ptr<float>();
    float* divq = in.divq.data_ptr<float>();
    float* dp = in.dp.data_ptr<float>();
    float* seis = nrec > 0 ? in.seis.data_ptr<float>() : nullptr;
    const bool save = in.save_f.numel() > 0;
    float* s_f = save ? in.save_f.data_ptr<float>() : nullptr;
    float* s_qs = save ? in.save_qs.data_ptr<float>() : nullptr;
    float* s_rs = save ? in.save_rs.data_ptr<float>() : nullptr;
    float* s_rhs = save ? in.save_rhs.data_ptr<float>() : nullptr;
    float* s_gv = save ? in.save_gv.data_ptr<float>() : nullptr;
    float* s_q = (save && c.visc) ? in.save_q.data_ptr<float>() : nullptr;
    const std::vector<float> wav = host_wavelet(c.wav, nt, stream);

    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int eb = static_cast<int>((Em + TPB - 1) / TPB);
    const int pb = c.npts > 0 ? (c.npts + TPB - 1) / TPB : 0;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t sh1 = static_cast<size_t>(m) * sizeof(float);
    const int t0 = in.t0, t1 = in.t1 > 0 ? in.t1 : nt;
    SWEEP_CHECK(0 <= t0 && t0 < t1 && t1 <= nt, "sem3d Biot: bad step range");

    for (int n = t0; n < t1; ++n) {
        const float wv = wav[n];
        const size_t row = static_cast<size_t>(n - t0);
        SWEEP_LAUNCH_ON(sem2d::biot2d_inject_p, nb, TPB, 0, stream, p, c.inv_M1, c.pin_n, c.pp, wv, dt, ng);
        SWEEP_LAUNCH_ON(biot3d_grad_p, E, m, sh1, stream, p, gp, mp, c.ax, c.bx, c.ay, c.by, c.az, c.bz,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, ngll, E);
        cudaMemsetAsync(Rx, 0, ngb, stream);
        cudaMemsetAsync(Ry, 0, ngb, stream);
        cudaMemsetAsync(Rz, 0, ngb, stream);
        if (c.cp) cudaMemsetAsync(P, 0, 9 * ngb, stream);
        SWEEP_LAUNCH_ON(sem3d_vs_sigma_div_deformed, dim3(E, 1), m, 9 * sh1, stream,
            sig, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.wJ,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8],
            c.ibool, c.D, Rx, Ry, Rz, P, ngll, E, ng);
        if (c.cp)
            SWEEP_LAUNCH_ON(sem3d_vs_cpml_nodal, nb, TPB, 0, stream, c.a_n, P, msig, Rx, Ry, Rz, ng);
        SWEEP_LAUNCH_ON(biot3d_force, nb, TPB, 0, stream, Rx, Ry, Rz, f, c.pv[0], c.pv[1], c.pv[2], wv,
                                             !c.vmode, ng);
        if (c.npts)
            SWEEP_LAUNCH_ON(biot3d_ifc_el, pb, TPB, 0, stream, p, c.ifc_idx, c.ifc_w, c.ifc_nx, c.ifc_ny,
                                                  c.ifc_nz, f, c.npts, ng);
        for (const Group& g : c.groups) {
            const int gb = (g.n + TPB - 1) / TPB;
            BIOT3D_LAUNCH_K(g.k, biot3d_flux, gb, TPB, 0, stream,
                vx, vy, vz, q, r, gp, f, c.wJ, c.qmask, c.inv_M, c.inv_Mm, c.perm,
                c.node_order, c.wr, c.share, c.dragE, c.dragP,
                c.pv[0], c.pv[1], c.pv[2], c.pq[0], c.pq[1], c.pq[2], wv, c.vmode ? 1 : 0, dt,
                save ? s_f + row * 3 * ng : nullptr, save ? s_qs + row * 3 * Em : nullptr,
                save ? s_rs + row * 3 * Em : nullptr, g.n, g.coff, g.eoff, g.noff, m, Em, ng);
        }
        cudaMemsetAsync(adiv, 0, ngb, stream);
        cudaMemsetAsync(divq, 0, ngb, stream);
        if (c.cp) cudaMemsetAsync(Pq, 0, 3 * ngb, stream);
        SWEEP_LAUNCH_ON(biot3d_stress, E, m, 3 * sh1, stream,
            vx, vy, vz, sig, mv, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.lam_d, c.mu, c.alpha,
            c.wJ, I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, dt, adiv,
            save ? s_gv + row * 9 * Em : nullptr, ngll, E);
        SWEEP_LAUNCH_ON(biot3d_div_q, E, m, 9 * sh1, stream,
            q, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.wJ,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, divq, Pq,
            ngll, E, ng);
        if (c.cp)
            SWEEP_LAUNCH_ON(biot3d_cpml_q, nb, TPB, 0, stream, c.a_n, Pq, mq, divq, ng);
        if (c.npts)
            SWEEP_LAUNCH_ON(biot3d_ifc_ac, pb, TPB, 0, stream, vx, vy, vz, c.ifc_idx, c.ifc_w, c.ifc_nx,
                                                  c.ifc_ny, c.ifc_nz, divq, c.npts);
        SWEEP_LAUNCH_ON(sem2d::biot2d_pressure, nb, TPB, 0, stream, p, divq, adiv, c.inv_Mp, c.pin_n, dt, dp,
                                                       save ? s_rhs + row * ng : nullptr, ng);
        SWEEP_LAUNCH_ON(biot3d_stress2, eb, TPB, 0, stream, sig, r, q, c.alpha, dp, c.b, c.decay, c.gain,
            c.visc ? 1 : 0, s_q ? s_q + row * 3 * Em : nullptr, c.ibool, Em);
        if (nrec > 0) {
            const size_t fb = static_cast<size_t>(nrec) * nt;
            const float* nodal[4] = {vx, vy, vz, p};
            for (int k = 0; k < 4; ++k)
                SWEEP_LAUNCH_ON(sem2d::sem2d_record, rb, TPB, 0, stream, nodal[k], c.rec_G, c.rec_W,
                                                            seis + k * fb, n, nt, nrec, m, ng);
            for (int k = 0; k < 9; ++k) {
                const float* fld = k < 3 ? q + k * Em : sig + (k - 3) * Em;
                SWEEP_LAUNCH_ON(sem2d::sem2d_record, rb, TPB, 0, stream, fld, c.rec_Ge, c.rec_W,
                                                            seis + (4 + k) * fb, n, nt, nrec, m,
                                                            static_cast<int>(Em));
            }
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void biot3d_backward(const Sem3dBiotBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Biot3dCommon c = biot3d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const long Em = c.Em;
    const float dt = c.dt;
    const float* const* I = c.I;
    const float* gseis = nrec > 0 ? in.grad_seis.data_ptr<float>() : nullptr;
    const float* s_f = in.save_f.data_ptr<float>();
    const float* s_qs = in.save_qs.data_ptr<float>();
    const float* s_rs = in.save_rs.data_ptr<float>();
    const float* s_rhs = in.save_rhs.data_ptr<float>();
    const float* s_gv = in.save_gv.data_ptr<float>();
    const float* s_q = c.visc ? in.save_q.data_ptr<float>() : nullptr;
    float* lvx = in.lvx.data_ptr<float>();
    float* lvy = in.lvy.data_ptr<float>();
    float* lvz = in.lvz.data_ptr<float>();
    float* lp = in.lp.data_ptr<float>();
    float* lq = in.lq.data_ptr<float>();
    float* lsig = in.lsig.data_ptr<float>();
    float* lr = in.lr.data_ptr<float>();
    float* lmp = c.cp ? in.lmp.data_ptr<float>() : nullptr;
    float* lmv = c.cp ? in.lmv.data_ptr<float>() : nullptr;
    float* lmsig = c.cp ? in.lmsig.data_ptr<float>() : nullptr;
    float* lmq = c.cp ? in.lmq.data_ptr<float>() : nullptr;
    float* lRx = in.lRx.data_ptr<float>();
    float* lRy = in.lRy.data_ptr<float>();
    float* lRz = in.lRz.data_ptr<float>();
    float* lP = c.cp ? in.lP.data_ptr<float>() : nullptr;
    float* lPq = c.cp ? in.lPq.data_ptr<float>() : nullptr;
    float* lf = in.lf.data_ptr<float>();
    float* ldp = in.ldp.data_ptr<float>();
    float* ldivq = in.ldivq.data_ptr<float>();
    float* lgp = in.lgp.data_ptr<float>();
    float* lgv = in.lgv.data_ptr<float>();
    float* lg = in.lg.data_ptr<float>();
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gMp = in.grad_inv_Mp.data_ptr<float>();
    float* gM1 = in.grad_inv_M1.data_ptr<float>();
    float* gMm = in.grad_inv_Mm.data_ptr<float>();
    float* gLd = in.grad_lam_d.data_ptr<float>();
    float* gMu = in.grad_mu.data_ptr<float>();
    float* gAl = in.grad_alpha.data_ptr<float>();
    float* gB = c.visc ? in.grad_b.data_ptr<float>() : nullptr;
    float* gwr = in.grad_wr.data_ptr<float>();
    float* gsh = in.grad_share.data_ptr<float>();
    float* gE = in.grad_dragE.data_ptr<float>();
    float* gP = in.grad_dragP.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();
    const std::vector<float> wav = host_wavelet(c.wav, nt, stream);

    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int eb = static_cast<int>((Em + TPB - 1) / TPB);
    const int pb = c.npts > 0 ? (c.npts + TPB - 1) / TPB : 0;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t sh1 = static_cast<size_t>(m) * sizeof(float);
    const int t0 = in.t0, t1 = in.t1 > 0 ? in.t1 : nt;
    SWEEP_CHECK(0 <= t0 && t0 < t1 && t1 <= nt, "sem3d Biot: bad step range");

    for (int k = t1 - 1; k >= t0; --k) {
        const float wv = wav[k];
        const size_t row = static_cast<size_t>(k - t0);
        if (nrec > 0) {
            const size_t fb = static_cast<size_t>(nrec) * nt;
            float* nodal[4] = {lvx, lvy, lvz, lp};
            for (int f2 = 0; f2 < 4; ++f2)
                SWEEP_LAUNCH_ON(sem2d::sem2d_adjoint_source, rj, TPB, 0, stream, gseis + f2 * fb, c.rec_G,
                                                                    c.rec_W, nodal[f2], k, nt,
                                                                    nrec, m);
            for (int f2 = 0; f2 < 9; ++f2) {
                float* fld = f2 < 3 ? lq + f2 * Em : lsig + (f2 - 3) * Em;
                SWEEP_LAUNCH_ON(sem2d::sem2d_adjoint_source, rj, TPB, 0, stream, gseis + (4 + f2) * fb,
                                                                    c.rec_Ge, c.rec_W, fld, k,
                                                                    nt, nrec, m);
            }
        }
        const float* rhs_k = s_rhs + row * ng;
        const float* gv_k = s_gv + row * 9 * Em;
        cudaMemcpyAsync(ldp, lp, ngb, cudaMemcpyDeviceToDevice, stream);
        SWEEP_LAUNCH_ON(biot3d_stress_T, eb, TPB, 0, stream,
            lsig, lr, lq, gv_k, s_q ? s_q + row * 3 * Em : nullptr, rhs_k, c.inv_Mp, c.pin_n, dt,
            c.lam_d, c.mu, c.alpha, c.b, c.decay, c.gain, c.visc ? 1 : 0, c.ibool, lgv, ldp,
            gLd, gMu, gAl, gB, Em);
        SWEEP_LAUNCH_ON(sem2d::biot2d_pressure_T, nb, TPB, 0, stream, ldp, rhs_k, c.inv_Mp, c.pin_n, dt,
                                                         ldivq, gMp, ng);
        if (c.npts)
            SWEEP_LAUNCH_ON(biot3d_ifc_ac_T, pb, TPB, 0, stream, ldivq, c.ifc_idx, c.ifc_w, c.ifc_nx,
                                                    c.ifc_ny, c.ifc_nz, lvx, lvy, lvz, c.npts);
        SWEEP_LAUNCH_ON(biot3d_stress_grad_T, E, m, 9 * sh1, stream,
            lgv, gv_k, ldivq, lmv, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.alpha, c.wJ,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D,
            lvx, lvy, lvz, gAl, ngll, E);
        if (c.cp)
            SWEEP_LAUNCH_ON(biot3d_cpml_q_T, nb, TPB, 0, stream, c.a_n, ldivq, lmq, lPq, ng);
        SWEEP_LAUNCH_ON(biot3d_div_q_T, E, m, 3 * sh1, stream,
            ldivq, lPq, lq, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.wJ,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, ngll, E, ng);
        for (const Group& g : c.groups) {
            const int gb = (g.n + TPB - 1) / TPB;
            BIOT3D_LAUNCH_K(g.k, biot3d_flux_T, gb, TPB, 0, stream,
                lvx, lvy, lvz, lq, s_f + row * 3 * ng, s_qs + row * 3 * Em, s_rs + row * 3 * Em,
                c.qmask, c.inv_M, c.inv_Mm, c.perm, c.node_order, c.wr, c.share, c.dragE,
                c.dragP, c.pv[0], c.pv[1], c.pv[2], c.pq[0], c.pq[1], c.pq[2], wv,
                c.vmode ? 1 : 0, dt, lf, lg, gE, gP, gwr, gsh, gM, gMm, gW, k,
                g.n, g.coff, g.eoff, g.noff, m, Em, ng);
        }
        SWEEP_LAUNCH_ON(biot3d_g_T, eb, TPB, 0, stream, lg, lr, lgp, c.wJ, c.qmask, m, Em);
        SWEEP_LAUNCH_ON(biot3d_force_T, nb, TPB, 0, stream, lf, lRx, lRy, lRz, c.pv[0], c.pv[1], c.pv[2],
                                               !c.vmode, gW, k, ng);
        if (c.npts)
            SWEEP_LAUNCH_ON(biot3d_ifc_el_T, pb, TPB, 0, stream, lf, c.ifc_idx, c.ifc_w, c.ifc_nx, c.ifc_ny,
                                                    c.ifc_nz, lp, c.npts, ng);
        if (c.cp)
            SWEEP_LAUNCH_ON(sem3d_vs_cpml_nodal_T, nb, TPB, 0, stream, c.a_n, lRx, lRy, lRz, lmsig, lP, ng);
        SWEEP_LAUNCH_ON(sem3d_vs_sigma_div_T_deformed, dim3(E, 1), m, 9 * sh1, stream,
            lRx, lRy, lRz, lP, lsig, c.ax, c.bx, c.ay, c.by, c.az, c.bz, c.wJ,
            I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, ngll, E, ng);
        SWEEP_LAUNCH_ON(biot3d_grad_p_T, E, m, 3 * sh1, stream, lgp, lmp, c.ax, c.bx, c.ay, c.by, c.az,
            c.bz, I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7], I[8], c.ibool, c.D, lp, ngll, E);
        SWEEP_LAUNCH_ON(sem2d::biot2d_inject_p_T, nb, TPB, 0, stream, lp, c.inv_M1, c.pin_n, c.pp, wv, dt,
                                                         gM1, gW, k, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem3d

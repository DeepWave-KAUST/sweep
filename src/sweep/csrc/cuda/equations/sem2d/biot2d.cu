// sweep.sem CUDA (impl='c') backend -- the 2-D Biot SEM drivers
// (BiotVSSEM2DDeformed, BiotAnisoPermSEM2DDeformed): the eager step of
// sweep.sem.equations.biot_vs_deformed, bare or with the C-PML, fluid / porous /
// dry regions and their interfaces, the open-pore surface, the viscous memory;
// save-all or in chunks [t0, t1) (sweep.sem.drivers.vs_chunks). Kernels and the
// step's sub-steps: sem2d/biot2d_kernels.cuh.
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "sem2d.h"
#include "kernels.cuh"
#include "vs2d_kernels.cuh"
#include "biot2d_kernels.cuh"

namespace sem2d {

namespace {

struct Biot2dCommon {
    int E, ng, ngll, m, nrec, nt, kmax, npts, nd;
    long Em;
    float dt, decay, gain;
    bool cp, tensor, vmode, visc;
    const int* ibool; const float* D;
    const float *wJ, *Ixx, *Ixz, *Iex, *Iez, *qmask;
    const float *inv_M, *inv_Mp, *inv_M1, *inv_Mm, *pin_n;
    const float *lam_d, *mu, *alpha;
    const int* slot_elem; const float *wr, *share, *dragE, *dragP;
    const float *bxx, *bxz, *bzz;
    const float *ax, *bx, *az, *bz, *a_n;
    const long *ifc_idx; const float *ifc_w, *ifc_nx, *ifc_nz;
    const float *pvx, *pvz, *pp, *pqx, *pqz, *wav;
    int nsls; const float *ptau, *mtau, *gsls_w;
    const long *rec_G, *rec_Ge; const float* rec_W;
};

template <class In>
Biot2dCommon biot2d_common(const In& in)
{
    Biot2dCommon c{};
    c.E = in.n_elements; c.ng = in.n_global; c.ngll = in.ngll;
    c.m = c.ngll * c.ngll; c.Em = static_cast<long>(c.E) * c.m;
    c.nrec = in.n_rec; c.nt = in.nt; c.kmax = in.kmax;
    c.dt = static_cast<float>(in.dt);
    c.decay = static_cast<float>(in.decay); c.gain = static_cast<float>(in.gain);
    SWEEP_CHECK(in.batch == 1 && in.ndim == 2, "sem2d Biot driver: 2-D, batch=1");
    SWEEP_CHECK(c.kmax >= 1 && c.kmax <= BIOT2D_KMAX, "sem2d Biot: kmax must be 1..",
                BIOT2D_KMAX);
    c.cp = in.a_x.numel() > 0;
    c.tensor = in.tensor_drag != 0; c.vmode = in.velocity_mode != 0; c.visc = in.viscous != 0;
    c.npts = static_cast<int>(in.ifc_idx.numel());
    c.nd = static_cast<int>(in.dirichlet_idx.numel());
    c.ibool = in.ibool.template data_ptr<int32_t>();
    c.D = in.D.template data_ptr<float>();
    c.wJ = in.wJ.template data_ptr<float>();
    c.Ixx = in.ixx.template data_ptr<float>(); c.Ixz = in.ixz.template data_ptr<float>();
    c.Iex = in.iex.template data_ptr<float>(); c.Iez = in.iez.template data_ptr<float>();
    c.qmask = in.qmask.template data_ptr<float>();
    c.inv_M = in.inv_M.template data_ptr<float>();
    c.inv_Mp = in.inv_Mp.template data_ptr<float>();
    c.inv_M1 = in.inv_M1.template data_ptr<float>();
    c.inv_Mm = in.inv_Mm.template data_ptr<float>();
    c.pin_n = in.pin_n.template data_ptr<float>();
    c.lam_d = in.lam_d.template data_ptr<float>();
    c.mu = in.mu.template data_ptr<float>();
    c.alpha = in.alpha.template data_ptr<float>();
    c.slot_elem = in.slot_elem.template data_ptr<int32_t>();
    c.wr = in.wr.template data_ptr<float>();
    c.share = in.share.template data_ptr<float>();
    c.dragE = in.dragE.template data_ptr<float>();
    c.dragP = in.dragP.template data_ptr<float>();
    c.bxx = c.visc ? in.b_xx.template data_ptr<float>() : nullptr;
    c.bxz = c.visc ? in.b_xz.template data_ptr<float>() : nullptr;
    c.bzz = c.visc ? in.b_zz.template data_ptr<float>() : nullptr;
    c.ax = c.cp ? in.a_x.template data_ptr<float>() : nullptr;
    c.bx = c.cp ? in.b_x.template data_ptr<float>() : nullptr;
    c.az = c.cp ? in.a_z.template data_ptr<float>() : nullptr;
    c.bz = c.cp ? in.b_z.template data_ptr<float>() : nullptr;
    c.a_n = c.cp ? in.a_n.template data_ptr<float>() : nullptr;
    c.ifc_idx = c.npts ? in.ifc_idx.template data_ptr<int64_t>() : nullptr;
    c.ifc_w = c.npts ? in.ifc_w.template data_ptr<float>() : nullptr;
    c.ifc_nx = c.npts ? in.ifc_nx.template data_ptr<float>() : nullptr;
    c.ifc_nz = c.npts ? in.ifc_nz.template data_ptr<float>() : nullptr;
    c.pvx = in.pat_vx.template data_ptr<float>(); c.pvz = in.pat_vz.template data_ptr<float>();
    c.pp = in.pat_p.template data_ptr<float>();
    c.pqx = in.pat_qx.template data_ptr<float>(); c.pqz = in.pat_qz.template data_ptr<float>();
    c.wav = in.wavelet.template data_ptr<float>();
    c.nsls = in.nsls;
    c.ptau = c.nsls ? in.ptau.template data_ptr<float>() : nullptr;
    c.mtau = c.nsls ? in.mtau.template data_ptr<float>() : nullptr;
    c.gsls_w = c.nsls ? in.gsls_w.template data_ptr<float>() : nullptr;
    c.rec_G = c.nrec > 0 ? in.rec_G.template data_ptr<int64_t>() : nullptr;
    c.rec_Ge = c.nrec > 0 ? in.rec_Ge.template data_ptr<int64_t>() : nullptr;
    c.rec_W = c.nrec > 0 ? in.rec_W.template data_ptr<float>() : nullptr;
    return c;
}

// The wavelet on the host side of the loop: one scalar per step (the kernels take
// it by value, read once from the device copy).
inline float wavelet_at(const float* wav_host, int n) { return wav_host[n]; }

} // namespace

void biot2d_forward(const Sem2dBiotForwardInputCore& in) {
    sweep::DeviceGuard guard(in.vx.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Biot2dCommon c = biot2d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const long Em = c.Em;
    const float dt = c.dt;
    float* vx = in.vx.data_ptr<float>();
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
    float* rmem = c.nsls ? in.rmem.data_ptr<float>() : nullptr;
    const int kmax = c.kmax;

    // the wavelet, on the host once: the kernels take each step's value by value
    std::vector<float> wav(static_cast<size_t>(nt));
    cudaMemcpyAsync(wav.data(), c.wav, sizeof(float) * nt, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int eb = static_cast<int>((Em + TPB - 1) / TPB);
    const int pb = c.npts > 0 ? (c.npts + TPB - 1) / TPB : 0;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t sh1 = static_cast<size_t>(m) * sizeof(float);
    const size_t sh2 = 2 * sh1, sh4 = 4 * sh1;
    const int t0 = in.t0, t1 = in.t1 > 0 ? in.t1 : nt;
    SWEEP_CHECK(0 <= t0 && t0 < t1 && t1 <= nt, "sem2d Biot: bad step range");

    for (int n = t0; n < t1; ++n) {
        const float wv = wavelet_at(wav.data(), n);
        const size_t row = static_cast<size_t>(n - t0);
        biot2d_inject_p<<<nb, TPB, 0, stream>>>(p, c.inv_M1, c.pin_n, c.pp, wv, dt, ng);
        biot2d_grad_p<<<E, m, sh1, stream>>>(p, gp, mp, c.ax, c.bx, c.az, c.bz,
            c.Ixx, c.Ixz, c.Iex, c.Iez, c.ibool, c.D, ngll, E);
        cudaMemsetAsync(Rx, 0, ngb, stream);
        cudaMemsetAsync(Rz, 0, ngb, stream);
        if (c.cp) cudaMemsetAsync(P, 0, 4 * ngb, stream);
        sem2d_vs_sigma_div<<<E, m, sh4, stream>>>(
            sig, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, Rx, Rz, P, ngll, E, ng);
        if (c.cp)
            sem2d_vs_cpml_nodal<<<nb, TPB, 0, stream>>>(c.a_n, P, msig, Rx, Rz, ng);
        biot2d_force<<<nb, TPB, 0, stream>>>(Rx, Rz, f, c.pvx, c.pvz, wv, !c.vmode, ng);
        if (c.npts)
            biot2d_ifc_el<<<pb, TPB, 0, stream>>>(p, c.ifc_idx, c.ifc_w, c.ifc_nx, c.ifc_nz,
                                                  f, c.npts, ng);
        BIOT2D_LAUNCH_T(c.tensor, biot2d_flux, nb, TPB, 0, stream,
            vx, vz, q, r, gp, f, c.wJ, c.qmask, c.inv_M, c.inv_Mm, c.slot_elem, c.wr,
            c.share, c.dragE, c.dragP, c.pvx, c.pvz, c.pqx, c.pqz, wv, c.vmode ? 1 : 0, dt,
            save ? s_f + row * 2 * ng : nullptr,
            save ? s_qs + row * 2 * static_cast<size_t>(ng) * kmax : nullptr,
            save ? s_rs + row * 2 * static_cast<size_t>(ng) * kmax : nullptr,
            kmax, m, Em, ng);
        cudaMemsetAsync(adiv, 0, ngb, stream);
        cudaMemsetAsync(divq, 0, ngb, stream);
        if (c.cp) cudaMemsetAsync(Pq, 0, 2 * ngb, stream);
        biot2d_stress<<<E, m, sh2, stream>>>(
            vx, vz, sig, mv, c.ax, c.bx, c.az, c.bz, c.lam_d, c.mu, c.alpha, c.wJ,
            c.nsls, c.ptau, c.mtau, c.gsls_w, rmem,
            c.Ixx, c.Ixz, c.Iex, c.Iez, c.ibool, c.D, dt, adiv,
            save ? s_gv + row * 4 * Em : nullptr, ngll, E);
        biot2d_div_q<<<E, m, sh4, stream>>>(
            q, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, divq, Pq, ngll, E, ng);
        if (c.cp)
            biot2d_cpml_q<<<nb, TPB, 0, stream>>>(c.a_n, Pq, mq, divq, ng);
        if (c.npts)
            biot2d_ifc_ac<<<pb, TPB, 0, stream>>>(vx, vz, c.ifc_idx, c.ifc_w, c.ifc_nx,
                                                  c.ifc_nz, divq, c.npts);
        biot2d_pressure<<<nb, TPB, 0, stream>>>(p, divq, adiv, c.inv_Mp, c.pin_n, dt, dp,
                                                save ? s_rhs + row * ng : nullptr, ng);
        biot2d_stress2<<<eb, TPB, 0, stream>>>(sig, r, q, c.alpha, dp, c.bxx, c.bxz, c.bzz,
            c.decay, c.gain, c.visc ? 1 : 0, s_q ? s_q + row * 2 * Em : nullptr, c.ibool, Em);
        if (nrec > 0) {
            const size_t fb = static_cast<size_t>(nrec) * nt;
            const float* nodal[3] = {vx, vz, p};
            for (int k = 0; k < 3; ++k)
                sem2d_record<<<rb, TPB, 0, stream>>>(nodal[k], c.rec_G, c.rec_W, seis + k * fb,
                                                     n, nt, nrec, m, ng);
            const float* elem[5] = {q, q + Em, sig, sig + Em, sig + 2 * Em};
            for (int k = 0; k < 5; ++k)
                sem2d_record<<<rb, TPB, 0, stream>>>(elem[k], c.rec_Ge, c.rec_W,
                                                     seis + (3 + k) * fb, n, nt, nrec, m,
                                                     static_cast<int>(Em));
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void biot2d_backward(const Sem2dBiotBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Biot2dCommon c = biot2d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const long Em = c.Em;
    const float dt = c.dt;
    const int kmax = c.kmax;
    const float* gseis = nrec > 0 ? in.grad_seis.data_ptr<float>() : nullptr;
    const float* s_f = in.save_f.data_ptr<float>();
    const float* s_qs = in.save_qs.data_ptr<float>();
    const float* s_rs = in.save_rs.data_ptr<float>();
    const float* s_rhs = in.save_rhs.data_ptr<float>();
    const float* s_gv = in.save_gv.data_ptr<float>();
    const float* s_q = c.visc ? in.save_q.data_ptr<float>() : nullptr;
    float* lvx = in.lvx.data_ptr<float>();
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
    float* gwr = in.grad_wr.data_ptr<float>();
    float* gsh = in.grad_share.data_ptr<float>();
    float* gE = in.grad_dragE.data_ptr<float>();
    float* gP = in.grad_dragP.data_ptr<float>();
    float* gBxx = c.visc ? in.grad_b_xx.data_ptr<float>() : nullptr;
    float* gBxz = c.visc ? in.grad_b_xz.data_ptr<float>() : nullptr;
    float* gBzz = c.visc ? in.grad_b_zz.data_ptr<float>() : nullptr;
    float* gW = in.grad_wavelet.data_ptr<float>();
    float* lrmem = c.nsls ? in.lrmem.data_ptr<float>() : nullptr;
    float* gPt = c.nsls ? in.grad_ptau.data_ptr<float>() : nullptr;
    float* gMt = c.nsls ? in.grad_mtau.data_ptr<float>() : nullptr;

    std::vector<float> wav(static_cast<size_t>(nt));
    cudaMemcpyAsync(wav.data(), c.wav, sizeof(float) * nt, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int eb = static_cast<int>((Em + TPB - 1) / TPB);
    const int pb = c.npts > 0 ? (c.npts + TPB - 1) / TPB : 0;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t sh1 = static_cast<size_t>(m) * sizeof(float);
    const size_t sh2 = 2 * sh1, sh4 = 4 * sh1;
    const int t0 = in.t0, t1 = in.t1 > 0 ? in.t1 : nt;
    SWEEP_CHECK(0 <= t0 && t0 < t1 && t1 <= nt, "sem2d Biot: bad step range");

    for (int k = t1 - 1; k >= t0; --k) {
        const float wv = wavelet_at(wav.data(), k);
        const size_t row = static_cast<size_t>(k - t0);
        if (nrec > 0) {
            const size_t fb = static_cast<size_t>(nrec) * nt;
            float* nodal[3] = {lvx, lvz, lp};
            for (int f2 = 0; f2 < 3; ++f2)
                sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gseis + f2 * fb, c.rec_G, c.rec_W,
                                                             nodal[f2], k, nt, nrec, m);
            float* elem[5] = {lq, lq + Em, lsig, lsig + Em, lsig + 2 * Em};
            for (int f2 = 0; f2 < 5; ++f2)
                sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gseis + (3 + f2) * fb, c.rec_Ge,
                                                             c.rec_W, elem[f2], k, nt, nrec, m);
        }
        const float* rhs_k = s_rhs + row * ng;
        const float* gv_k = s_gv + row * 4 * Em;
        // K10^T / K11^T / K6^T: p' = p1 + dp, so dp's adjoint starts as lp
        cudaMemcpyAsync(ldp, lp, ngb, cudaMemcpyDeviceToDevice, stream);
        biot2d_stress_T<<<eb, TPB, 0, stream>>>(
            lsig, lr, lq, gv_k, s_q ? s_q + row * 2 * Em : nullptr, rhs_k, c.inv_Mp, c.pin_n,
            dt, c.lam_d, c.mu, c.alpha, c.bxx, c.bxz, c.bzz, c.decay, c.gain, c.visc ? 1 : 0,
            c.ibool, c.nsls, c.ptau, c.mtau, c.gsls_w, lrmem, gPt, gMt,
            lgv, ldp, gLd, gMu, gAl, gBxx, gBxz, gBzz, Em);
        biot2d_pressure_T<<<nb, TPB, 0, stream>>>(ldp, rhs_k, c.inv_Mp, c.pin_n, dt, ldivq,
                                                  gMp, ng);
        if (c.npts)
            biot2d_ifc_ac_T<<<pb, TPB, 0, stream>>>(ldivq, c.ifc_idx, c.ifc_w, c.ifc_nx,
                                                    c.ifc_nz, lvx, lvz, c.npts);
        biot2d_stress_grad_T<<<E, m, sh4, stream>>>(
            lgv, gv_k, ldivq, lmv, c.ax, c.bx, c.az, c.bz, c.alpha, c.wJ,
            c.Ixx, c.Ixz, c.Iex, c.Iez, c.ibool, c.D, lvx, lvz, gAl, ngll, E);
        if (c.cp)
            biot2d_cpml_q_T<<<nb, TPB, 0, stream>>>(c.a_n, ldivq, lmq, lPq, ng);
        biot2d_div_q_T<<<E, m, sh2, stream>>>(
            ldivq, lPq, lq, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, ngll, E, ng);
        // K5^T: the copies and v
        BIOT2D_LAUNCH_T(c.tensor, biot2d_flux_T, nb, TPB, 0, stream,
            lvx, lvz, lq, s_f + row * 2 * ng,
            s_qs + row * 2 * static_cast<size_t>(ng) * kmax,
            s_rs + row * 2 * static_cast<size_t>(ng) * kmax,
            c.qmask, c.inv_M, c.inv_Mm, c.slot_elem, c.wr, c.share, c.dragE, c.dragP,
            c.pvx, c.pvz, c.pqx, c.pqz, wv, c.vmode ? 1 : 0, dt, lf, lg,
            gE, gP, gwr, gsh, gM, gMm, gW, k, kmax, m, Em, ng);
        biot2d_g_T<<<eb, TPB, 0, stream>>>(lg, lr, lgp, c.wJ, c.qmask, m, Em);
        // K4^T, K3^T
        biot2d_force_T<<<nb, TPB, 0, stream>>>(lf, lRx, lRz, c.pvx, c.pvz, !c.vmode, gW, k, ng);
        if (c.npts)
            biot2d_ifc_el_T<<<pb, TPB, 0, stream>>>(lf, c.ifc_idx, c.ifc_w, c.ifc_nx, c.ifc_nz,
                                                    lp, c.npts, ng);
        if (c.cp)
            sem2d_vs_cpml_nodal_T<<<nb, TPB, 0, stream>>>(c.a_n, lRx, lRz, lmsig, lP, ng);
        sem2d_vs_sigma_div_T<<<E, m, sh4, stream>>>(
            lRx, lRz, lP, lsig, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, ngll, E, ng);
        // K2^T, K1^T
        biot2d_grad_p_T<<<E, m, sh2, stream>>>(lgp, lmp, c.ax, c.bx, c.az, c.bz,
            c.Ixx, c.Ixz, c.Iex, c.Iez, c.ibool, c.D, lp, ngll, E);
        biot2d_inject_p_T<<<nb, TPB, 0, stream>>>(lp, c.inv_M1, c.pin_n, c.pp, wv, dt, gM1,
                                                  gW, k, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem2d

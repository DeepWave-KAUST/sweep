// sweep.sem CUDA (impl='c') backend — 2-D acoustic-potential SEM backward.
//
// Adjoint-state gradient of the eager AcousticPotentialSEM2D.step (bare, no CPML)
// w.r.t. the leaf constants inv_M, inv_rho, and the wavelet, in fp32. Save-all
// mode: the forward saved u^1..u^nt, and this reverse loop co-propagates the
// adjoint field (reverse leap-frog; K is self-adjoint, so the same forward
// stiffness kernel is reused) and correlates it with the forward field. PyTorch
// then chains grad_inv_M/grad_inv_rho through the eager build() graph to vp_e/rho_e.
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>

#include "sem2d.h"
#include "kernels.cuh"

namespace sem2d {



void acoustic_backward(const SemBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();

    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());

    const int*  ibool   = in.ibool.data_ptr<int32_t>();
    const float* D       = in.D.data_ptr<float>();
    const float* inv_rho = in.inv_rho.data_ptr<float>();
    const float* wJ      = in.wJ.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>();
    const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>();
    const float* iez = in.iez.data_ptr<float>();
    const float* inv_M   = in.inv_M.data_ptr<float>();
    const float* mask_e  = in.mask_e.data_ptr<float>();
    const long*  didx    = kdir > 0 ? in.dirichlet_idx.data_ptr<int64_t>() : nullptr;
    const float* pattern = in.src_pattern.data_ptr<float>();
    const float* wavelet = in.wavelet.data_ptr<float>();
    const long*  rec_G   = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W   = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* grad_syn = nrec > 0 ? in.grad_syn.data_ptr<float>() : nullptr;
    const float* save_wf = in.save_wavefield.data_ptr<float>();

    float* a_p1 = in.a_p1.data_ptr<float>();
    float* a_p2 = in.a_p2.data_ptr<float>();
    float* a_m  = in.a_m.data_ptr<float>();
    float* tmp  = in.tmp.data_ptr<float>();
    float* Ktmp = in.Ktmp.data_ptr<float>();
    float* Ku   = in.Ku.data_ptr<float>();
    float* adj  = in.adj_src.data_ptr<float>();

    float* gradM = in.grad_inv_M.data_ptr<float>();
    float* gradR = in.grad_inv_rho.data_ptr<float>();
    float* gradW = in.grad_wavelet.data_ptr<float>();

    const size_t sh_stiff = static_cast<size_t>(3 * m) * sizeof(float);
    const size_t sh_imag  = static_cast<size_t>(2 * m) * sizeof(float);
    const int TPB = 256;
    const int nb  = (ng + TPB - 1) / TPB;
    const int dir_b = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const int rj_b  = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ng_bytes = static_cast<size_t>(ng) * sizeof(float);

    for (int mm = nt; mm >= 1; --mm) {
        // reverse leap-frog: a_m = 2 a_p1 - a_p2 - dt2 K(inv_M o a_p1) + R^T g[mm-1]
        sem2d_scale2<<<nb, TPB, 0, stream>>>(a_p1, inv_M, 1.f, tmp, ng);      // tmp = inv_M o a_p1
        cudaMemsetAsync(Ktmp, 0, ng_bytes, stream);
        sem2d_acoustic_stiffness<<<E, m, sh_stiff, stream>>>(
            tmp, ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, mask_e, Ktmp, ngll, E);
        cudaMemsetAsync(adj, 0, ng_bytes, stream);
        if (nrec > 0)
            sem2d_adjoint_source<<<rj_b, TPB, 0, stream>>>(
                grad_syn, rec_G, rec_W, adj, mm - 1, nt, nrec, m);
        sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(a_p1, a_p2, Ktmp, adj, dt2, a_m, ng);
        if (kdir > 0)
            sem2d_dirichlet_fill<<<dir_b, TPB, 0, stream>>>(a_m, didx, kdir);

        // gradients for forward line k = mm-1 (produces u^{mm}), field u^{mm-1}
        const int km1 = mm - 1;                          // wavelet/source time index
        const bool u_is_zero = (mm == 1);                // u^0 = 0
        cudaMemsetAsync(Ku, 0, ng_bytes, stream);
        if (!u_is_zero) {
            const float* u_km1 = save_wf + static_cast<size_t>(mm - 2) * ng;  // u^{mm-1}
            sem2d_acoustic_stiffness<<<E, m, sh_stiff, stream>>>(
                u_km1, ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, mask_e, Ku, ngll, E);
        }
        sem2d_accum_gradM<<<nb, TPB, 0, stream>>>(a_m, pattern, Ku, wavelet, km1, dt2, gradM, ng);
        sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(pattern, inv_M, a_m, dt2, km1, gradW, ng);
        if (!u_is_zero) {                                // u^{mm-1}=0 -> no imaging term
            const float* u_km1 = save_wf + static_cast<size_t>(mm - 2) * ng;
            sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, a_m, -dt2, tmp, ng);  // tmp = y = -dt2 inv_M o a_m
            sem2d_imaging_invrho<<<E, m, sh_imag, stream>>>(
                u_km1, tmp, ibool, D, wJ, ixx, ixz, iex, iez, mask_e, gradR, ngll, E);
        }

        // rotate: (a_p1,a_p2,a_m) -> (a_m,a_p1, reuse old a_p2 as scratch)
        float* t = a_p2; a_p2 = a_p1; a_p1 = a_m; a_m = t;
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- elastic backward (2-component adjoint + lam/mu/mass imaging) ------------
void elastic_backward(const SemElasticBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();

    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* lam = in.lam.data_ptr<float>();
    const float* mu = in.mu.data_ptr<float>();
    const float* lam2mu = in.lam2mu.data_ptr<float>();
    const float* wJ = in.wJ.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>();
    const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>();
    const float* iez = in.iez.data_ptr<float>();
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* mask_e = in.mask_e.data_ptr<float>();
    const float* pat_x = in.pat_x.data_ptr<float>();
    const float* pat_z = in.pat_z.data_ptr<float>();
    const float* wavelet = in.wavelet.data_ptr<float>();
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* grad_sx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* grad_sz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;
    const float* sux = in.save_ux.data_ptr<float>();
    const float* suz = in.save_uz.data_ptr<float>();

    float* ax1 = in.ax1.data_ptr<float>(); float* az1 = in.az1.data_ptr<float>();
    float* ax2 = in.ax2.data_ptr<float>(); float* az2 = in.az2.data_ptr<float>();
    float* axm = in.axm.data_ptr<float>(); float* azm = in.azm.data_ptr<float>();
    float* tmp_x = in.tmp_x.data_ptr<float>(); float* tmp_z = in.tmp_z.data_ptr<float>();
    float* Kx = in.Kx.data_ptr<float>(); float* Kz = in.Kz.data_ptr<float>();
    float* adx = in.adj_x.data_ptr<float>(); float* adz = in.adj_z.data_ptr<float>();
    float* gLam = in.grad_lam.data_ptr<float>();
    float* gMu = in.grad_mu.data_ptr<float>();
    float* gL2m = in.grad_lam2mu.data_ptr<float>();
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();

    const size_t sh_st = static_cast<size_t>(6 * m) * sizeof(float);
    const size_t sh_im = static_cast<size_t>(4 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int mm = nt; mm >= 1; --mm) {
        // reverse leap-frog (2-component): a? = 2 a?1 - a?2 - dt2 K(inv_M o a1) + R^T g
        sem2d_scale2<<<nb, TPB, 0, stream>>>(ax1, inv_M, 1.f, tmp_x, ng);
        sem2d_scale2<<<nb, TPB, 0, stream>>>(az1, inv_M, 1.f, tmp_z, ng);
        cudaMemsetAsync(Kx, 0, ngb, stream);
        cudaMemsetAsync(Kz, 0, ngb, stream);
        sem2d_elastic_stiffness<<<E, m, sh_st, stream>>>(
            tmp_x, tmp_z, ibool, D, lam, mu, lam2mu, wJ, ixx, ixz, iex, iez, mask_e, Kx, Kz, ngll, E);
        cudaMemsetAsync(adx, 0, ngb, stream);
        cudaMemsetAsync(adz, 0, ngb, stream);
        if (nrec > 0) {
            sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(grad_sx, rec_G, rec_W, adx, mm-1, nt, nrec, m);
            sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(grad_sz, rec_G, rec_W, adz, mm-1, nt, nrec, m);
        }
        sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(ax1, ax2, Kx, adx, dt2, axm, ng);
        sem2d_adj_leapfrog<<<nb, TPB, 0, stream>>>(az1, az2, Kz, adz, dt2, azm, ng);

        // gradients for forward line k = mm-1, using u^{mm-1}
        const int km1 = mm - 1;
        const bool u_is_zero = (mm == 1);
        cudaMemsetAsync(Kx, 0, ngb, stream);   // reuse Kx,Kz as Kux,Kuz of u^{mm-1}
        cudaMemsetAsync(Kz, 0, ngb, stream);
        const float* uxm1 = u_is_zero ? nullptr : sux + static_cast<size_t>(mm-2) * ng;
        const float* uzm1 = u_is_zero ? nullptr : suz + static_cast<size_t>(mm-2) * ng;
        if (!u_is_zero)
            sem2d_elastic_stiffness<<<E, m, sh_st, stream>>>(
                uxm1, uzm1, ibool, D, lam, mu, lam2mu, wJ, ixx, ixz, iex, iez, mask_e, Kx, Kz, ngll, E);
        sem2d_elastic_accum_gradM<<<nb, TPB, 0, stream>>>(
            axm, azm, pat_x, pat_z, Kx, Kz, wavelet, km1, dt2, gM, ng);
        sem2d_elastic_accum_gradW<<<nb, TPB, 0, stream>>>(
            pat_x, pat_z, inv_M, axm, azm, dt2, km1, gW, ng, 0);
        if (!u_is_zero) {
            sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, axm, -dt2, tmp_x, ng);   // yx
            sem2d_scale2<<<nb, TPB, 0, stream>>>(inv_M, azm, -dt2, tmp_z, ng);   // yz
            sem2d_elastic_imaging<<<E, m, sh_im, stream>>>(
                uxm1, uzm1, tmp_x, tmp_z, ibool, D, wJ, ixx, ixz, iex, iez, mask_e,
                gL2m, gLam, gMu, ngll, E);
        }
        // rotate both components
        float* tx = ax2; ax2 = ax1; ax1 = axm; axm = tx;
        float* tz = az2; az2 = az1; az1 = azm; azm = tz;
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- acoustic-potential C-PML backward (adjoint through psi/zeta) (P3b) ------
void acoustic_cpml_backward(const SemAcousticCpmlBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    // shot batch (see sem_types.h): adjoint state is (B, ng), histories are
    // (nt, B, ng), grad_syn is (B, nrec, nt); gradients come out PER-SHOT.
    const int B = in.B, Bm = in.Bm;
    SWEEP_CHECK(B >= 1 && (Bm == 1 || Bm == B),
                "acoustic_cpml_backward: B=", B, " Bm=", Bm);
    SWEEP_CHECK(in.bs == 0
                || in.band_chi.numel()
                   == static_cast<int64_t>(in.nt) * B * in.band_idx.numel(),
                "acoustic_cpml_backward: band histories must be (nt, B, nband)");
    SWEEP_CHECK(in.bs == 0 || in.band_kind != 2
                || in.band_scale.numel() == static_cast<int64_t>(5) * in.nt * B,
                "acoustic_cpml_backward: int8 band_scale must be (5, nt, B)");
    SWEEP_CHECK(in.gA.numel() == static_cast<int64_t>(B) * ng,
                "acoustic_cpml_backward: gA holds ", in.gA.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(in.grad_kappa.numel() == static_cast<int64_t>(B) * ng
                && in.grad_inv_rho.numel() == static_cast<int64_t>(B) * m * E
                && in.grad_wavelet.numel() == static_cast<int64_t>(B) * nt,
                "acoustic_cpml_backward: gradient buffers must be per-shot (B, ...)");
    const int msn = Bm > 1 ? ng : 0;
    const int msr = Bm > 1 ? m * E : 0;
    const int ws = in.wavelet.numel() == nt ? 0 : nt;

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* inv_rho = in.inv_rho.data_ptr<float>();
    const float* wJ = in.wJ.data_ptr<float>();
    const float* w1d = in.w1d.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>(); const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>(); const float* iez = in.iez.data_ptr<float>();
    const float* mask_e = in.mask_e.data_ptr<float>();
    const float* kappa = in.kappa_node.data_ptr<float>();
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* inv_Mg = in.inv_Mg.data_ptr<float>();
    const float* ax = in.a_x.data_ptr<float>(); const float* bx = in.b_x.data_ptr<float>();
    const float* dbxdx = in.dbx_dx.data_ptr<float>();
    const float* az = in.a_z.data_ptr<float>(); const float* bz = in.b_z.data_ptr<float>();
    const float* dbzdz = in.dbz_dz.data_ptr<float>();
    const float* pattern = in.src_pattern.data_ptr<float>();
    const float* wavelet = in.wavelet.data_ptr<float>();
    const long*  didx = kdir > 0 ? in.dirichlet_idx.data_ptr<int64_t>() : nullptr;
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* grad_syn = nrec > 0 ? in.grad_syn.data_ptr<float>() : nullptr;
    const bool bs = in.bs != 0;
    // the w_sum tape (sem_types.h): at k >= 1 nothing is rebuilt and the
    // psi/zeta tapes are not read. Never under boundary saving -- the chi
    // reverse march needs the rebuilt w_sum^k.
    const bool wt = !bs && in.save_w_sum.numel() > 0;
    SWEEP_CHECK(bs || wt || in.save_psi_x.numel() > 0,
                "acoustic_cpml_backward: save-all needs the psi/zeta tapes or "
                "the w_sum tape");
    const float* sch = bs ? nullptr : in.save_chi.data_ptr<float>();
    const float* spx = (bs || wt) ? nullptr : in.save_psi_x.data_ptr<float>();
    const float* spz = (bs || wt) ? nullptr : in.save_psi_z.data_ptr<float>();
    const float* szx = (bs || wt) ? nullptr : in.save_zeta_x.data_ptr<float>();
    const float* szz = (bs || wt) ? nullptr : in.save_zeta_z.data_ptr<float>();
    const float* sws = wt ? in.save_w_sum.data_ptr<float>() : nullptr;
    const int nband = bs ? static_cast<int>(in.band_idx.numel()) : 0;
    const long* band_g = nband > 0 ? in.band_idx.data_ptr<int64_t>() : nullptr;
    const int bkind = bs ? in.band_kind : SEM_BAND_F32;
    const bool bsplit = bs && in.band_split != 0;
    const SemBandIdx bi_x = sem_band_idx(bsplit, in.band_idx_x, band_g, nband);
    const SemBandIdx bi_z = sem_band_idx(bsplit, in.band_idx_z, band_g, nband);
    const SemBandHist h_chi = bs ? sem_hist(in.band_chi, in.band_scale, bkind, 0, nt, B) : SemBandHist();
    const SemBandHist h_px = bs ? sem_hist(in.band_psi_x, in.band_scale, bkind, 1, nt, B) : SemBandHist();
    const SemBandHist h_pz = bs ? sem_hist(in.band_psi_z, in.band_scale, bkind, 2, nt, B) : SemBandHist();
    const SemBandHist h_zx = bs ? sem_hist(in.band_zeta_x, in.band_scale, bkind, 3, nt, B) : SemBandHist();
    const SemBandHist h_zz = bs ? sem_hist(in.band_zeta_z, in.band_scale, bkind, 4, nt, B) : SemBandHist();
    float* chi_hi = bs ? in.chi_hi.data_ptr<float>() : nullptr;
    float* chi_lo = bs ? in.chi_lo.data_ptr<float>() : nullptr;
    float* chi_km1 = bs ? in.chi_km1.data_ptr<float>() : nullptr;
    float* r_px = bs ? in.r_psi_x.data_ptr<float>() : nullptr;
    float* r_pz = bs ? in.r_psi_z.data_ptr<float>() : nullptr;
    float* r_zx = bs ? in.r_zeta_x.data_ptr<float>() : nullptr;
    float* r_zz = bs ? in.r_zeta_z.data_ptr<float>() : nullptr;

    float* gA = in.gA.data_ptr<float>(); float* gB = in.gB.data_ptr<float>();
    float* gchi_k = in.gchi_k.data_ptr<float>();
    float* lpx = in.lpsi_x.data_ptr<float>(); float* lpz = in.lpsi_z.data_ptr<float>();
    float* lzx = in.lzeta_x.data_ptr<float>(); float* lzz = in.lzeta_z.data_ptr<float>();
    float* lpxn = in.lpsi_x_n.data_ptr<float>(); float* lpzn = in.lpsi_z_n.data_ptr<float>();
    float* lzxn = in.lzeta_x_n.data_ptr<float>(); float* lzzn = in.lzeta_z_n.data_ptr<float>();
    float* apx = in.apx.data_ptr<float>(); float* apz = in.apz.data_ptr<float>();
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    float* w_sum = in.w_sum.data_ptr<float>(); float* gw = in.gw.data_ptr<float>();
    float* gbx = in.gbx.data_ptr<float>(); float* gbz = in.gbz.data_ptr<float>();
    float* gbkx = in.gbkx.data_ptr<float>(); float* gbkz = in.gbkz.data_ptr<float>();
    float* gbax = in.gbax.data_ptr<float>(); float* gbaz = in.gbaz.data_ptr<float>();
    float* lchi_G = in.lchi_G.data_ptr<float>();
    float* lapx = in.lapx.data_ptr<float>(); float* lapz = in.lapz.data_ptr<float>();
    float* gLam = nullptr; (void)gLam;
    float* gInvM = in.grad_inv_M.data_ptr<float>();
    float* gInvRho = in.grad_inv_rho.data_ptr<float>();
    float* gKappa = in.grad_kappa.data_ptr<float>();
    float* gInvMg = in.grad_inv_Mg.data_ptr<float>();
    float* gWav = in.grad_wavelet.data_ptr<float>();

    const size_t sh_g = static_cast<size_t>(7 * m) * sizeof(float);   // cpml_grads
    const size_t sh_t = static_cast<size_t>(9 * m) * sizeof(float);   // cpml_grads_T
    // Per-step zeroing: collapse the runs Python hands over contiguously.
    // lchi_G/lapx/lapz are re-zeroed TWICE per reverse step; the BS r_psi/r_zeta
    // quartet once. None of them rotate -- the rotating carries are gA, lpsi*,
    // lzeta* and chi_hi -- so a single sized memset stays valid every step.
    const size_t ngB = static_cast<size_t>(B) * ng;      // per-buffer batch span
    const bool l3 = (lapx == lchi_G + ngB) && (lapz == lchi_G + 2 * ngB);
    const bool r4c = bs && (r_pz == r_px + ngB) && (r_zx == r_px + 2 * ngB)
                  && (r_zz == r_px + 3 * ngB);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const size_t ngb = ngB * sizeof(float);
    const dim3 gE(E, B), gN(nb, B), gJ(rj, B), gD(db, B);

    const bool contig6 = (gz == gx + ngB && kx == gx + 2 * ngB && kz == gx + 3 * ngB
                          && gax == gx + 4 * ngB && gaz == gx + 5 * ngB);
    const int bb = nband > 0 ? (nband + TPB - 1) / TPB : 0;
    if (bs) {   // seed the reverse march with the final two chi levels
        cudaMemcpyAsync(chi_hi, in.fin_chi.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(chi_lo, in.fin_chi_prev.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
    }

    for (int mm = nt; mm >= 1; --mm) {
        const int k = mm - 1;
        // data + dirichlet^T on gA and the carried memory adjoints
        if (nrec > 0)
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(grad_syn, rec_G, rec_W, gA, k, nt, nrec, m, ng);
        if (kdir > 0)
            sem2d_dirichlet_fill5<<<gD, TPB, 0, stream>>>(
                gA, lpx, lpz, lzx, lzz,
                nullptr, nullptr, nullptr, nullptr, nullptr, didx, kdir, ng);
        sem2d_scale2_b<<<gN, TPB, 0, stream>>>(kappa, gA, dt2, gw, ng, msn);   // gw = dt2 kappa gA

        if (k >= 1) {
            const size_t o = static_cast<size_t>(k - 1) * ngB;   // (nt, B, ng)
            // chi^k: the tape row (state after step k-1) or the marched level
            const float* chi_k = bs ? chi_lo : sch + o;
            if (wt) {
                // the w_sum tape: the adjoint algebra alone (the assembly,
                // zeta, w_sum_out and grad_inv_Mg are neither read nor written)
                sem2d_acoustic_cpml_step_T<true><<<gN, TPB, 0, stream>>>(
                    gx, gz, kx, kz, gax, gaz, inv_Mg, ax, bx, dbxdx, az, bz, dbzdz,
                    nullptr, nullptr, gw, lpx, lpz, lzx, lzz,
                    gbx, gbz, gbkx, gbkz, gbax, gbaz, w_sum, gInvMg, ng, msn);
            } else {
                const float* psx; const float* psz;
                const float* zex; const float* zez;
                if (bs) {
                    // rebuild psi^{k-1}/zeta^{k-1}: zero outside the band (exact
                    // there) + the saved band values.
                    if (r4c) {
                        cudaMemsetAsync(r_px, 0, 4 * ngb, stream);
                    } else {
                        cudaMemsetAsync(r_px, 0, ngb, stream);
                        cudaMemsetAsync(r_pz, 0, ngb, stream);
                        cudaMemsetAsync(r_zx, 0, ngb, stream);
                        cudaMemsetAsync(r_zz, 0, ngb, stream);
                    }
                    if (nband > 0) {
                        sem2d_band_load(h_px, k - 1, bi_x.idx, r_px, bi_x.n, TPB, stream, B, ng);
                        sem2d_band_load(h_pz, k - 1, bi_z.idx, r_pz, bi_z.n, TPB, stream, B, ng);
                        sem2d_band_load(h_zx, k - 1, bi_x.idx, r_zx, bi_x.n, TPB, stream, B, ng);
                        sem2d_band_load(h_zz, k - 1, bi_z.idx, r_zz, bi_z.n, TPB, stream, B, ng);
                    }
                    psx = r_px; psz = r_pz; zex = r_zx; zez = r_zz;
                } else {
                    psx = spx + o; psz = spz + o;
                    zex = szx + o; zez = szz + o;
                }
                if (contig6) {
                    cudaMemsetAsync(gx, 0, 6 * ngb, stream);
                } else {
                    cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
                    cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
                    cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
                }
                sem2d_acoustic_cpml_grads_f<<<gE, m, sh_g, stream>>>(
                    chi_k, psx, psz, ax, az, ibool, D, inv_rho, w1d, ixx, ixz, iex, iez,
                    mask_e, gx, gz, kx, kz, gax, gaz, ngll, E, ng, msr);
                sem2d_acoustic_cpml_step_T<false><<<gN, TPB, 0, stream>>>(
                    gx, gz, kx, kz, gax, gaz, inv_Mg, ax, bx, dbxdx, az, bz, dbzdz,
                    zex, zez, gw, lpx, lpz, lzx, lzz, gbx, gbz, gbkx, gbkz, gbax, gbaz,
                    w_sum, gInvMg, ng, msn);
            }
            if (l3) {
                cudaMemsetAsync(lchi_G, 0, 3 * ngb, stream);
            } else {
                cudaMemsetAsync(lchi_G, 0, ngb, stream);
                cudaMemsetAsync(lapx, 0, ngb, stream);
                cudaMemsetAsync(lapz, 0, ngb, stream);
            }
            sem2d_acoustic_cpml_grads_T<<<gE, m, sh_t, stream>>>(
                chi_k, gbx, gbz, gbkx, gbkz, gbax, gbaz, ibool, D, inv_rho, w1d,
                ixx, ixz, iex, iez, mask_e, lchi_G, lapx, lapz, gInvRho, ngll, E,
                ng, msr);
            if (bs) {
                // reverse-march chi: chi^{k-1} = 2 chi^k - chi^{k+1}
                //   + dt2 (kappa w_sum^k + invM pat wav[k])  (exact in the
                // reversible interior), then dirichlet re-zero + exact band
                // restore from the saved history.
                if (k == 1) {
                    cudaMemsetAsync(chi_km1, 0, ngb, stream);   // chi^0 = 0
                } else {
                    sem2d_acoustic_cpml_newmark<<<gN, TPB, 0, stream>>>(
                        chi_lo, chi_hi, kappa, w_sum, inv_M, pattern, wavelet,
                        k, dt2, chi_km1, ng, msn, ws);
                    if (kdir > 0)
                        sem2d_dirichlet_fill_b<<<gD, TPB, 0, stream>>>(chi_km1, didx, kdir, ng);
                    if (nband > 0)
                        sem2d_band_load(h_chi, k - 2, band_g, chi_km1, nband, TPB, stream, B, ng);
                }
            }
        } else {
            cudaMemsetAsync(w_sum, 0, ngb, stream);
            if (l3) {
                cudaMemsetAsync(lchi_G, 0, 3 * ngb, stream);
            } else {
                cudaMemsetAsync(lchi_G, 0, ngb, stream);
                cudaMemsetAsync(lapx, 0, ngb, stream);
                cudaMemsetAsync(lapz, 0, ngb, stream);
            }
        }
        // w_sum^k: the forward's tape row k (k >= 1), else the recompute or
        // the k == 0 zeros. Row 0 is never read (w_sum^0 == 0: chi/psi/zeta^0 = 0).
        const float* wsum_k = (wt && k >= 1)
            ? sws + static_cast<size_t>(k) * ngB : w_sum;
        sem2d_acoustic_cpml_combine_f<<<gN, TPB, 0, stream>>>(
            gA, gB, lchi_G, kappa, wsum_k, pattern, wavelet, k, lpx, lpz, lzx, lzz,
            lapx, lapz, gw, ax, az, gchi_k, lpxn, lpzn, lzxn, lzzn, gKappa, gInvM,
            gB, dt2, ng, msn, ws);                 // gB_out aliases gB (safe: RaW per thread)
        sem2d_accum_gradW_b<<<gN, TPB, 0, stream>>>(pattern, inv_M, gA, dt2, k, gWav, ng, nt, msn);
        // rotate
        float* t;
        t = gA; gA = gchi_k; gchi_k = t;
        t = lpx; lpx = lpxn; lpxn = t;
        t = lpz; lpz = lpzn; lpzn = t;
        t = lzx; lzx = lzxn; lzxn = t;
        t = lzz; lzz = lzzn; lzzn = t;
        if (bs && k >= 1) {
            t = chi_hi; chi_hi = chi_lo; chi_lo = chi_km1; chi_km1 = t;
        }
    }

    // grad_wavelet accumulated per step above: grad_wav[k] = dt2 sum_i
    // pattern inv_M gA (the source enters chi_next as dt2 inv_M pattern wav[k]).
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped backward (acoustic C-PML): reverse step k in three phases ---
// The reverse step has TWO assembly points, so DD splits it around two
// interface exchanges (see the forward's dd_grads/dd_step note):
//   pre:  adjoint source + dirichlet + gw, then (k>=1) the grads_f recompute
//         on the SAVED state k -- exchange the six fields after it returns
//   mid:  step_T (pointwise) + the grads_T transpose -- exchange
//         lchi_G/lapx/lapz after it returns
//   end:  combine_f + the wavelet-gradient reduction (which reads the
//         ownership-masked src_pattern_own -- see sem_types.h)
// gA<->gchi_k and the lpsi/lzeta pair rotation are the caller's, done by
// re-pointing struct fields between steps. B == 1, save-all only.
void acoustic_cpml_dd_bwd_pre(const SemAcousticCpmlBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    SWEEP_CHECK(in.B == 1, "acoustic_cpml_dd_bwd: DD is single-shot");
    const bool bs = in.bs != 0;
    // the w_sum tape: nothing to rebuild at k >= 1 (bwd_mid runs step_T<true>,
    // the adjoint algebra alone, and bwd_end reads the row); no g6 to exchange
    const bool wt = !bs && in.save_w_sum.numel() > 0;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    float* gA = in.gA.data_ptr<float>();
    if (nrec > 0) {
        const int rj = (nrec * m + TPB - 1) / TPB;
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_syn.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), gA, static_cast<int>(k), nt, nrec, m, ng);
    }
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill5<<<db, TPB, 0, stream>>>(
            gA, in.lpsi_x.data_ptr<float>(), in.lpsi_z.data_ptr<float>(),
            in.lzeta_x.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
            nullptr, nullptr, nullptr, nullptr, nullptr,
            in.dirichlet_idx.data_ptr<int64_t>(), kdir, ng);
    }
    sem2d_scale2_b<<<nb, TPB, 0, stream>>>(
        in.kappa_node.data_ptr<float>(), gA, dt2, in.gw.data_ptr<float>(), ng, 0);
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    float* lchi_G = in.lchi_G.data_ptr<float>();
    float* lapx = in.lapx.data_ptr<float>(); float* lapz = in.lapz.data_ptr<float>();
    const bool l3 = (lapx == lchi_G + ng) && (lapz == lchi_G + 2L * ng);
    if (k >= 1 && !wt) {
        const size_t o = static_cast<size_t>(k - 1) * ng;
        const bool contig6 = (gz == gx + ng && kx == gx + 2L * ng && kz == gx + 3L * ng
                              && gax == gx + 4L * ng && gaz == gx + 5L * ng);
        if (contig6) {
            cudaMemsetAsync(gx, 0, 6 * ngb, stream);
        } else {
            cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
            cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
            cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
        }
        // state^k: from the save-all tape, or -- under boundary saving --
        // reconstructed (chi_lo carries the reverse march, the memories are
        // zero outside the band and restored from the ring inside it)
        const float* chi_k = in.save_chi.data_ptr<float>() + o;
        const float* psx = in.save_psi_x.data_ptr<float>() + o;
        const float* psz = in.save_psi_z.data_ptr<float>() + o;
        if (bs) {
            float* r_px = in.r_psi_x.data_ptr<float>();
            float* r_pz = in.r_psi_z.data_ptr<float>();
            float* r_zx = in.r_zeta_x.data_ptr<float>();
            float* r_zz = in.r_zeta_z.data_ptr<float>();
            const bool r4 = (r_pz == r_px + ng) && (r_zx == r_px + 2L * ng)
                         && (r_zz == r_px + 3L * ng);
            if (r4) {
                cudaMemsetAsync(r_px, 0, 4 * ngb, stream);
            } else {
                cudaMemsetAsync(r_px, 0, ngb, stream);
                cudaMemsetAsync(r_pz, 0, ngb, stream);
                cudaMemsetAsync(r_zx, 0, ngb, stream);
                cudaMemsetAsync(r_zz, 0, ngb, stream);
            }
            const int nband = static_cast<int>(in.band_idx.numel());
            if (nband > 0) {
                const long* bg = in.band_idx.data_ptr<int64_t>();
                const int bk = in.band_kind;
                const bool bsplit = in.band_split != 0;
                const SemBandIdx bi_x = sem_band_idx(bsplit, in.band_idx_x, bg, nband);
                const SemBandIdx bi_z = sem_band_idx(bsplit, in.band_idx_z, bg, nband);
                sem2d_band_load(sem_hist(in.band_psi_x, in.band_scale, bk, 1, nt),
                                k - 1, bi_x.idx, r_px, bi_x.n, TPB, stream);
                sem2d_band_load(sem_hist(in.band_psi_z, in.band_scale, bk, 2, nt),
                                k - 1, bi_z.idx, r_pz, bi_z.n, TPB, stream);
                sem2d_band_load(sem_hist(in.band_zeta_x, in.band_scale, bk, 3, nt),
                                k - 1, bi_x.idx, r_zx, bi_x.n, TPB, stream);
                sem2d_band_load(sem_hist(in.band_zeta_z, in.band_scale, bk, 4, nt),
                                k - 1, bi_z.idx, r_zz, bi_z.n, TPB, stream);
            }
            chi_k = in.chi_lo.data_ptr<float>();
            psx = r_px; psz = r_pz;
        }
        const size_t sh_g = static_cast<size_t>(7 * m) * sizeof(float);
        sem2d_acoustic_cpml_grads_f<<<E, m, sh_g, stream>>>(
            chi_k, psx, psz,
            in.a_x.data_ptr<float>(), in.a_z.data_ptr<float>(),
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.inv_rho.data_ptr<float>(), in.w1d.data_ptr<float>(),
            in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
            in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
            in.mask_e.data_ptr<float>(), gx, gz, kx, kz, gax, gaz, ngll, E, ng, 0);
    } else if (k == 0) {
        cudaMemsetAsync(in.w_sum.data_ptr<float>(), 0, ngb, stream);
        if (l3) {
            cudaMemsetAsync(lchi_G, 0, 3 * ngb, stream);
        } else {
            cudaMemsetAsync(lchi_G, 0, ngb, stream);
            cudaMemsetAsync(lapx, 0, ngb, stream);
            cudaMemsetAsync(lapz, 0, ngb, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic_cpml_dd_bwd_mid(const SemAcousticCpmlBackwardInputCore& in, int64_t k) {
    if (k < 1) return;
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nt = in.nt;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const bool bs = in.bs != 0;
    const bool wt = !bs && in.save_w_sum.numel() > 0;    // see acoustic_cpml_dd_bwd_pre
    const size_t o = static_cast<size_t>(k - 1) * ng;
    float* lchi_G = in.lchi_G.data_ptr<float>();
    float* lapx = in.lapx.data_ptr<float>(); float* lapz = in.lapz.data_ptr<float>();
    const float* chi_k = bs ? in.chi_lo.data_ptr<float>()
                            : in.save_chi.data_ptr<float>() + o;
    // zeta^k: the tape, or the ring copies bwd_pre restored this step -- unread
    // under the w_sum tape, where bwd_pre rebuilt nothing
    const float* zex = wt ? nullptr
                          : bs ? in.r_zeta_x.data_ptr<float>()
                               : in.save_zeta_x.data_ptr<float>() + o;
    const float* zez = wt ? nullptr
                          : bs ? in.r_zeta_z.data_ptr<float>()
                               : in.save_zeta_z.data_ptr<float>() + o;
    SEM2D_LAUNCH_WT(wt, sem2d_acoustic_cpml_step_T, nb, TPB, 0, stream,
        in.gx.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.gax.data_ptr<float>(), in.gaz.data_ptr<float>(),
        in.inv_Mg.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(), in.dbx_dx.data_ptr<float>(),
        in.a_z.data_ptr<float>(), in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        zex, zez,
        in.gw.data_ptr<float>(),
        in.lpsi_x.data_ptr<float>(), in.lpsi_z.data_ptr<float>(),
        in.lzeta_x.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
        in.gbx.data_ptr<float>(), in.gbz.data_ptr<float>(),
        in.gbkx.data_ptr<float>(), in.gbkz.data_ptr<float>(),
        in.gbax.data_ptr<float>(), in.gbaz.data_ptr<float>(),
        in.w_sum.data_ptr<float>(), in.grad_inv_Mg.data_ptr<float>(), ng, 0);
    const bool l3 = (lapx == lchi_G + ng) && (lapz == lchi_G + 2L * ng);
    if (l3) {
        cudaMemsetAsync(lchi_G, 0, 3 * ngb, stream);
    } else {
        cudaMemsetAsync(lchi_G, 0, ngb, stream);
        cudaMemsetAsync(lapx, 0, ngb, stream);
        cudaMemsetAsync(lapz, 0, ngb, stream);
    }
    const size_t sh_t = static_cast<size_t>(9 * m) * sizeof(float);
    sem2d_acoustic_cpml_grads_T<<<E, m, sh_t, stream>>>(
        chi_k,
        in.gbx.data_ptr<float>(), in.gbz.data_ptr<float>(),
        in.gbkx.data_ptr<float>(), in.gbkz.data_ptr<float>(),
        in.gbax.data_ptr<float>(), in.gbaz.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.inv_rho.data_ptr<float>(), in.w1d.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.mask_e.data_ptr<float>(), lchi_G, lapx, lapz,
        in.grad_inv_rho.data_ptr<float>(), ngll, E, ng, 0);
    if (bs) {
        // Reverse-march chi one level: chi^{k-1} = 2 chi^k - chi^{k+1}
        // + dt2 (kappa w_sum^k + invM pat wav[k]), then dirichlet + the
        // exact band restore. Every operation is POINTWISE, and w_sum came
        // out of the already-exchanged assembly, so the march needs no
        // extra interface exchange: both copies of a cut column carry the
        // same inputs and therefore reproduce the same value. The caller
        // rotates chi_hi/chi_lo/chi_km1 after this call.
        const float dt2m = static_cast<float>(in.dt * in.dt);
        float* chi_km1 = in.chi_km1.data_ptr<float>();
        const size_t ngbm = static_cast<size_t>(ng) * sizeof(float);
        if (k == 1) {
            cudaMemsetAsync(chi_km1, 0, ngbm, stream);      // chi^0 = 0
        } else {
            sem2d_acoustic_cpml_newmark<<<nb, TPB, 0, stream>>>(
                in.chi_lo.data_ptr<float>(), in.chi_hi.data_ptr<float>(),
                in.kappa_node.data_ptr<float>(), in.w_sum.data_ptr<float>(),
                in.inv_M.data_ptr<float>(), in.src_pattern.data_ptr<float>(),
                in.wavelet.data_ptr<float>(), static_cast<int>(k), dt2m,
                chi_km1, ng, 0, 0);
            const int kdirm = static_cast<int>(in.dirichlet_idx.numel());
            if (kdirm > 0) {
                const int dbm = (kdirm + TPB - 1) / TPB;
                sem2d_dirichlet_fill_b<<<dbm, TPB, 0, stream>>>(
                    chi_km1, in.dirichlet_idx.data_ptr<int64_t>(), kdirm, ng);
            }
            const int nbandm = static_cast<int>(in.band_idx.numel());
            if (nbandm > 0)
                sem2d_band_load(
                    sem_hist(in.band_chi, in.band_scale, in.band_kind, 0, nt),
                    k - 2, in.band_idx.data_ptr<int64_t>(), chi_km1,
                    nbandm, TPB, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic_cpml_dd_bwd_end(const SemAcousticCpmlBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    // w_sum^k: the forward's tape row k, or what bwd_mid recomputed (k == 0:
    // the zeros bwd_pre memset)
    const bool wt = in.bs == 0 && in.save_w_sum.numel() > 0;
    const float* wsum_k = (wt && k >= 1)
        ? in.save_w_sum.data_ptr<float>() + static_cast<size_t>(k) * ng
        : in.w_sum.data_ptr<float>();
    sem2d_acoustic_cpml_combine_f<<<nb, TPB, 0, stream>>>(
        in.gA.data_ptr<float>(), in.gB.data_ptr<float>(), in.lchi_G.data_ptr<float>(),
        in.kappa_node.data_ptr<float>(), wsum_k,
        in.src_pattern.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(k),
        in.lpsi_x.data_ptr<float>(), in.lpsi_z.data_ptr<float>(),
        in.lzeta_x.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
        in.lapx.data_ptr<float>(), in.lapz.data_ptr<float>(),
        in.gw.data_ptr<float>(), in.a_x.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.gchi_k.data_ptr<float>(),
        in.lpsi_x_n.data_ptr<float>(), in.lpsi_z_n.data_ptr<float>(),
        in.lzeta_x_n.data_ptr<float>(), in.lzeta_z_n.data_ptr<float>(),
        in.grad_kappa.data_ptr<float>(), in.grad_inv_M.data_ptr<float>(),
        in.gB.data_ptr<float>(), dt2, ng, 0, 0);
    const float* pat_w = in.src_pattern_own.numel()
        ? in.src_pattern_own.data_ptr<float>()
        : in.src_pattern.data_ptr<float>();
    sem2d_accum_gradW_b<<<nb, TPB, 0, stream>>>(
        pat_w, in.inv_M.data_ptr<float>(), in.gA.data_ptr<float>(), dt2,
        static_cast<int>(k), in.grad_wavelet.data_ptr<float>(), ng, nt, 0);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- elastic C-PML backward (P3e) -------------------------------------------
// Reverse loop n = nt-1 .. 0 over forward iterations. Carried state: lam1 =
// lambda(u^{n+1}), gprev = P lambda(u^{n+2}), sc_prev carry, and the 6
// memory-field adjoints (in the mu_* buffers, updated in place by the element
// transpose kernel).
void elastic_cpml_backward(const SemElasticCpmlBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    // shot batch: shared model, per-shot (B, ...) gradient outputs
    const int B = in.B;
    SWEEP_CHECK(B >= 1, "elastic_cpml_backward: B=", B);
    SWEEP_CHECK(in.lam1_x.numel() == static_cast<int64_t>(B) * ng,
                "elastic_cpml_backward: lam1_x holds ", in.lam1_x.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(B == 1 || (in.grad_inv_M.numel() == static_cast<int64_t>(B) * ng
                           && in.grad_wavelet.numel() == static_cast<int64_t>(B) * nt),
                "elastic_cpml_backward: gradient buffers must be per-shot (B, ...)");
    SWEEP_CHECK(in.bs == 0 || in.wavelet.numel() == nt
                || in.wavelet.numel() == static_cast<int64_t>(B) * nt,
                "elastic_cpml_backward: wavelet must be (nt,) or (B, nt)");
    const int wst = (in.bs != 0 && in.wavelet.numel() != nt) ? nt : 0;

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* lam = in.lam.data_ptr<float>();
    const float* muv = in.mu.data_ptr<float>();
    const float* lam2mu = in.lam2mu.data_ptr<float>();
    const float* wJ = in.wJ.data_ptr<float>();
    const float* wrhoJ = in.w_rho_J.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>(); const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>(); const float* iez = in.iez.data_ptr<float>();
    const float* mask_e = in.mask_e.data_ptr<float>();
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* pat_x = in.pat_x.data_ptr<float>();
    const float* pat_z = in.pat_z.data_ptr<float>();
    const long*  pin_idx = kpin > 0 ? in.outer_edge_idx.data_ptr<int64_t>() : nullptr;
    // same wall as pin_idx, as a per-node 0/1 mask so it can be applied inside a
    // fused per-node kernel instead of a separate scatter launch
    const float* pin_mask = in.pin_mask.numel() > 0 ? in.pin_mask.data_ptr<float>() : nullptr;
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* grad_sx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* grad_sz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;

    // the same packing the helper does -- these two entries kept a
    // hand-copied copy, so a field added to ElCpmlCoefs reached them
    // UNINITIALISED (an indeterminate device pointer, which is an
    // illegal access at the first launch that reads it)
    ElCpmlCoefs cf = sem2d_el_coefs(in);

    const bool bs = in.bs != 0;
    const float* sux = bs ? nullptr : in.save_ux.data_ptr<float>();
    const float* suz = bs ? nullptr : in.save_uz.data_ptr<float>();
    const float* wavelet = bs ? in.wavelet.data_ptr<float>() : nullptr;
    const int nbn = bs ? static_cast<int>(in.band_node_idx.numel()) : 0;
    const int nbe = bs ? static_cast<int>(in.band_elem_idx.numel()) : 0;
    const long* bnode = bs ? in.band_node_idx.data_ptr<int64_t>() : nullptr;
    const int bkind = in.band_kind;
    const long* belem = bs ? in.band_elem_idx.data_ptr<int64_t>() : nullptr;
    float* u_hi_x = bs ? in.u_hi_x.data_ptr<float>() : nullptr;
    float* u_hi_z = bs ? in.u_hi_z.data_ptr<float>() : nullptr;
    float* u_lo_x = bs ? in.u_lo_x.data_ptr<float>() : nullptr;
    float* u_lo_z = bs ? in.u_lo_z.data_ptr<float>() : nullptr;
    float* u_km1_x = bs ? in.u_km1_x.data_ptr<float>() : nullptr;
    float* u_km1_z = bs ? in.u_km1_z.data_ptr<float>() : nullptr;
    float* r_sdxx = bs ? in.rec_sd_xx.data_ptr<float>() : nullptr;
    float* r_sdzx = bs ? in.rec_sd_zx.data_ptr<float>() : nullptr;
    float* r_sdxz = bs ? in.rec_sd_xz.data_ptr<float>() : nullptr;
    float* r_sdzz = bs ? in.rec_sd_zz.data_ptr<float>() : nullptr;
    float* r_apx = bs ? in.rec_apx.data_ptr<float>() : nullptr;
    float* r_apz = bs ? in.rec_apz.data_ptr<float>() : nullptr;
    float* r_rxx = bs ? in.rec_rxx.data_ptr<float>() : nullptr;
    float* r_rxz = bs ? in.rec_rxz.data_ptr<float>() : nullptr;
    float* r_rzx = bs ? in.rec_rzx.data_ptr<float>() : nullptr;
    float* r_rzz = bs ? in.rec_rzz.data_ptr<float>() : nullptr;
    float* r_rdx = bs ? in.rec_rdx.data_ptr<float>() : nullptr;
    float* r_rdz = bs ? in.rec_rdz.data_ptr<float>() : nullptr;
    float* r_Acc_x = bs ? in.rec_Acc_x.data_ptr<float>() : nullptr;
    float* r_Acc_z = bs ? in.rec_Acc_z.data_ptr<float>() : nullptr;
    const float* sdxx = in.save_sd_xx.data_ptr<float>();
    const float* sdzx = in.save_sd_zx.data_ptr<float>();
    const float* sdxz = in.save_sd_xz.data_ptr<float>();
    const float* sdzz = in.save_sd_zz.data_ptr<float>();
    const float* sapx = in.save_apx.data_ptr<float>();
    const float* sapz = in.save_apz.data_ptr<float>();

    float* lam1_x = in.lam1_x.data_ptr<float>(); float* lam1_z = in.lam1_z.data_ptr<float>();
    float* lamn_x = in.lamn_x.data_ptr<float>(); float* lamn_z = in.lamn_z.data_ptr<float>();
    float* g_x = in.g_x.data_ptr<float>();       float* g_z = in.g_z.data_ptr<float>();
    float* gprev_x = in.gprev_x.data_ptr<float>(); float* gprev_z = in.gprev_z.data_ptr<float>();
    float* y_x = in.y_x.data_ptr<float>();       float* y_z = in.y_z.data_ptr<float>();
    float* adj_x = in.adj_x.data_ptr<float>();   float* adj_z = in.adj_z.data_ptr<float>();
    float* sc_now_x = in.sc_now_x.data_ptr<float>(); float* sc_now_z = in.sc_now_z.data_ptr<float>();
    float* scp_a_x = in.sc_prev_a_x.data_ptr<float>(); float* scp_a_z = in.sc_prev_a_z.data_ptr<float>();
    float* scp_b_x = in.sc_prev_b_x.data_ptr<float>(); float* scp_b_z = in.sc_prev_b_z.data_ptr<float>();
    float* mu_xx = in.mu_xx.data_ptr<float>(); float* mu_zx = in.mu_zx.data_ptr<float>();
    float* mu_xz = in.mu_xz.data_ptr<float>(); float* mu_zz = in.mu_zz.data_ptr<float>();
    float* mu_dx = in.mu_dx.data_ptr<float>(); float* mu_dz = in.mu_dz.data_ptr<float>();
    float* gLam = in.grad_lam.data_ptr<float>();
    float* gMu = in.grad_mu.data_ptr<float>();
    float* gL2m = in.grad_lam2mu.data_ptr<float>();
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gWrhoJ = in.grad_w_rho_J.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();

    const size_t shmem = static_cast<size_t>(8 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const int pb = kpin > 0 ? (kpin + TPB - 1) / TPB : 0;
    // batch-wide spans (every field/scratch buffer is (B, ng) / (B, E*m[*2]))
    const size_t ngB = static_cast<size_t>(B) * ng;
    const size_t ngb = ngB * sizeof(float);
    const size_t Em = static_cast<size_t>(E) * m;
    const dim3 gE(E, B), gN(nb, B), gJ(rj, B), gP(pb, B);

    // lambda(u^{nt}) = Rec^T grad_seis[:, nt-1]   (u^{nt} has no later use)
    if (nrec > 0) {
        sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(grad_sx, rec_G, rec_W, lam1_x, nt - 1, nt, nrec, m, ng);
        sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(grad_sz, rec_G, rec_W, lam1_z, nt - 1, nt, nrec, m, ng);
    }

    const size_t em2b = static_cast<size_t>(B) * Em * 2 * sizeof(float);
    // Per-step zeroing: memsets were the largest GPU-time item in this loop, so
    // collapse them where the buffers are laid out contiguously.
    //   z4  : sc_now_{x,z} + adj_{x,z} -- these never rotate.
    //   zpa : the sc_prev_a pair. sc_prev_a/sc_prev_b SWAP every step, so they
    //         must NOT be folded into z4 (that would wipe the carry); but the
    //         "a" half stays two adjacent rows in either state, so a 2*ng
    //         memset at sc_prev_a_x is valid throughout.
    // Both fall back to per-buffer memsets if the pointers are not contiguous.
    const bool z4 = (sc_now_z == sc_now_x + ngB) && (adj_x == sc_now_x + 2 * ngB)
                 && (adj_z == sc_now_x + 3 * ngB);
    const size_t em2 = static_cast<size_t>(B) * Em * 2;
    const bool r6 = bs && (r_rxz == r_rxx + em2) && (r_rzx == r_rxx + 2 * em2)
                 && (r_rzz == r_rxx + 3 * em2) && (r_rdx == r_rxx + 4 * em2)
                 && (r_rdz == r_rxx + 5 * em2);
    const dim3 gBN(bs ? (nbn + TPB - 1) / TPB : 0, B);
    const dim3 gBE(bs ? (nbe * m + TPB - 1) / TPB : 0, B);
    if (bs) {   // seed: u_hi = u^{nt}, u_lo = u^{nt-1}
        cudaMemcpyAsync(u_hi_x, in.fin_ux.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_hi_z, in.fin_uz.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_lo_x, in.fin_ux_prev.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_lo_z, in.fin_uz_prev.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
    }

    for (int n = nt - 1; n >= 0; --n) {
        if (bs) {
            // swap2nd reverse of forward iteration n: feed u^{n+1} as the "prev"
            // level with the CPML memories zeroed. Exact in the reversible
            // interior (there the memories are 0, Sx0 = 1 and aPML vanishes, so
            // the recomputed stretched derivatives ARE the raw ones); inside the
            // band both u and the element histories are then overwritten from
            // the per-step band history.
            // boundary saving is the production path, so it gets the same
            // memset consolidation as the save-all branch above
            if (r6) {
                cudaMemsetAsync(r_rxx, 0, 6 * em2b, stream);
            } else {
                cudaMemsetAsync(r_rxx, 0, em2b, stream);
                cudaMemsetAsync(r_rxz, 0, em2b, stream);
                cudaMemsetAsync(r_rzx, 0, em2b, stream);
                cudaMemsetAsync(r_rzz, 0, em2b, stream);
                cudaMemsetAsync(r_rdx, 0, em2b, stream);
                cudaMemsetAsync(r_rdz, 0, em2b, stream);
            }
            if (r_Acc_z == r_Acc_x + ngB) {
                cudaMemsetAsync(r_Acc_x, 0, 2 * ngb, stream);
            } else {
                cudaMemsetAsync(r_Acc_x, 0, ngb, stream);
                cudaMemsetAsync(r_Acc_z, 0, ngb, stream);
            }
            sem2d_elastic_cpml_element<<<gE, m, shmem, stream>>>(
                u_lo_x, u_lo_z, u_hi_x, u_hi_z, ibool, D, lam, muv, lam2mu,
                wJ, wrhoJ, ixx, ixz, iex, iez, mask_e, cf,
                r_rxx, r_rxz, r_rzx, r_rzz, r_rdx, r_rdz,
                inv_dt, r_Acc_x, r_Acc_z,
                r_sdxx, r_sdzx, r_sdxz, r_sdzz, r_apx, r_apz, 0L, ngll, E, ng);
            sem2d_elastic_newmark<<<gN, TPB, 0, stream>>>(
                u_lo_x, u_lo_z, u_hi_x, u_hi_z, r_Acc_x, r_Acc_z,
                pat_x, pat_z, inv_M, wavelet, n, dt2, u_km1_x, u_km1_z, ng, wst);
            if (kpin > 0) {
                sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(u_km1_x, pin_idx, kpin, ng);
                sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(u_km1_z, pin_idx, kpin, ng);
            }
            if (n >= 2) {   // band_ux[n-2] holds u^{n-1}
                const size_t no = static_cast<size_t>(n - 2) * B * nbn;
                sem2d_band_load(
                    sem_hist(in.band_ux, in.band_scale, bkind, 0, nt, B),
                    n - 2, bnode, u_km1_x, nbn, TPB, stream, B, ng);
                sem2d_band_load(
                    sem_hist(in.band_uz, in.band_scale, bkind, 1, nt, B),
                    n - 2, bnode, u_km1_z, nbn, TPB, stream, B, ng);
            }
            float* escr[6] = {r_sdxx, r_sdzx, r_sdxz, r_sdzz, r_apx, r_apz};
            const Buf* erg[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                           &in.band_sd_xz, &in.band_sd_zz,
                                           &in.band_apx, &in.band_apz};
            for (int j = 0; j < 6; ++j)
                sem2d_band_load_elem(
                    sem_hist(*erg[j], in.band_scale_e, in.band_kind_e, j, nt, B, nbe),
                    n, belem, escr[j], nbe, m, Em, TPB, stream, B);
        }
        // g = P lambda(u^{n+1});  y = -dt^2 inv_M o g   -- one pass, was six
        // launches over 2*n_global (2 copies + 2 pin scatters + 2 scales).
        sem2d_elastic_adj_carry<<<gN, TPB, 0, stream>>>(
            lam1_x, lam1_z, pin_mask, inv_M, -dt2, g_x, g_z, y_x, y_z, ng);

        // element transpose at forward iteration n
        const float* uxn = n >= 1 ? (bs ? u_lo_x : sux + static_cast<size_t>(n - 1) * ngB) : nullptr;
        const float* uzn = n >= 1 ? (bs ? u_lo_z : suz + static_cast<size_t>(n - 1) * ngB) : nullptr;
        if (z4) {                       // clears sc_now_{x,z} AND adj_{x,z}
            cudaMemsetAsync(sc_now_x, 0, 4 * ngb, stream);
        } else {
            cudaMemsetAsync(sc_now_x, 0, ngb, stream);
            cudaMemsetAsync(sc_now_z, 0, ngb, stream);
        }
        if (scp_a_z == scp_a_x + ngB) {  // holds in both rotation states
            cudaMemsetAsync(scp_a_x, 0, 2 * ngb, stream);
        } else {
            cudaMemsetAsync(scp_a_x, 0, ngb, stream);
            cudaMemsetAsync(scp_a_z, 0, ngb, stream);
        }
        const size_t so = bs ? 0 : static_cast<size_t>(n) * B * Em;
        sem2d_elastic_cpml_element_T<<<gE, m, shmem, stream>>>(
            y_x, y_z, uxn, uzn, ibool, D, lam, muv, lam2mu, wJ, wrhoJ,
            ixx, ixz, iex, iez, mask_e, cf,
            (bs ? r_sdxx : sdxx + so), (bs ? r_sdzx : sdzx + so),
            (bs ? r_sdxz : sdxz + so), (bs ? r_sdzz : sdzz + so),
            (bs ? r_apx : sapx + so), (bs ? r_apz : sapz + so),
            mu_xx, mu_zx, mu_xz, mu_zz, mu_dx, mu_dz, inv_dt,
            sc_now_x, sc_now_z, scp_a_x, scp_a_z,
            gLam, gMu, gL2m, gWrhoJ, ngll, E, ng);

        // grad_inv_M via the leapfrog identity + grad_wavelet
        const float* uxm1 = n >= 2 ? (bs ? u_km1_x : sux + static_cast<size_t>(n - 2) * ngB) : nullptr;
        const float* uzm1 = n >= 2 ? (bs ? u_km1_z : suz + static_cast<size_t>(n - 2) * ngB) : nullptr;
        sem2d_elastic_cpml_gradM<<<gN, TPB, 0, stream>>>(
            g_x, g_z,
            bs ? u_hi_x : sux + static_cast<size_t>(n) * ngB,
            bs ? u_hi_z : suz + static_cast<size_t>(n) * ngB,
            uxn, uzn, uxm1, uzm1, inv_M, gM, ng);
        sem2d_elastic_accum_gradW<<<gN, TPB, 0, stream>>>(
            pat_x, pat_z, inv_M, g_x, g_z, dt2, n, gW, ng, nt);

        // lambda(u^n) = 2g - gprev + adj_src(n-1) + sc_now + sc_prev_carry
        if (!z4) {   // already covered by the single 4*ng memset above; nothing
            cudaMemsetAsync(adj_x, 0, ngb, stream);   // writes adj between there
            cudaMemsetAsync(adj_z, 0, ngb, stream);   // and the adjoint source
        }
        if (n >= 1 && nrec > 0) {
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(grad_sx, rec_G, rec_W, adj_x, n - 1, nt, nrec, m, ng);
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(grad_sz, rec_G, rec_W, adj_z, n - 1, nt, nrec, m, ng);
        }
        sem2d_elastic_cpml_combine<<<gN, TPB, 0, stream>>>(
            g_x, gprev_x, adj_x, sc_now_x, scp_b_x, lamn_x, ng);
        sem2d_elastic_cpml_combine<<<gN, TPB, 0, stream>>>(
            g_z, gprev_z, adj_z, sc_now_z, scp_b_z, lamn_z, ng);

        // rotate carries
        float* t;
        t = lam1_x; lam1_x = lamn_x; lamn_x = t;
        t = lam1_z; lam1_z = lamn_z; lamn_z = t;
        t = gprev_x; gprev_x = g_x; g_x = t;
        t = gprev_z; gprev_z = g_z; g_z = t;
        t = scp_b_x; scp_b_x = scp_a_x; scp_a_x = t;
        t = scp_b_z; scp_b_z = scp_a_z; scp_a_z = t;
        if (bs) {   // (u^{n+1}, u^n, u^{n-1}) -> (u^n, u^{n-1}, scratch)
            t = u_hi_x; u_hi_x = u_lo_x; u_lo_x = u_km1_x; u_km1_x = t;
            t = u_hi_z; u_hi_z = u_lo_z; u_lo_z = u_km1_z; u_km1_z = t;
        }
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped backward (coupled fluid-solid C-PML) ------------------------
// Four phases, THREE interface exchanges per reverse step (save-all path):
//   a : adjoint sources, pin, y, ifc_ac(y -> lchidd), ifc_ac(u^k -> s_ac)
//                                            [exchange 2: lchidd, s_ac]
//   b : axpy + dirichlet + gw + grads_f      [exchange 6: gx..gaz]
//   c : step_T + grads_T                     [exchange 3: lchi_G, lapx, lapz]
//   d : gradMK, gradW, element_T, ifc_el(yf -> sc_now)
//                                            [exchange 4: sc_now, sc_prev_a]
//   e : gradM, elastic gradW, combines, gB
// The seabed interface crossing a cut needs no special handling: the
// exchange sums the whole cut COLUMN, and the interface node on it is one of
// those nodes. B == 1, save-all (bs is a later phase).
void fluid_solid_dd_bwd_a(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, ngll = in.ngll, m = ngll * ngll;
    const int nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    SWEEP_CHECK(in.B == 1, "fluid_solid_dd_bwd: DD is single-shot");
    SWEEP_CHECK(in.has_cpml != 0,
                "fluid_solid_dd_bwd: DD needs the coupled C-PML");
    const bool bs = in.bs != 0;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    if (nrec > 0) {
        const int rj = (nrec * m + TPB - 1) / TPB;
        const long* rg = in.rec_G.data_ptr<int64_t>();
        const float* rw = in.rec_W.data_ptr<float>();
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_schi.data_ptr<float>(), rg, rw, in.gA.data_ptr<float>(),
            static_cast<int>(k), nt, nrec, m, ng);
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sx.data_ptr<float>(), rg, rw, in.lam1_x.data_ptr<float>(),
            static_cast<int>(k), nt, nrec, m, ng);
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sz.data_ptr<float>(), rg, rw, in.lam1_z.data_ptr<float>(),
            static_cast<int>(k), nt, nrec, m, ng);
    }
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill5<<<db, TPB, 0, stream>>>(
            in.gA.data_ptr<float>(), in.lpsi_x.data_ptr<float>(),
            in.lpsi_z.data_ptr<float>(), in.lzeta_x.data_ptr<float>(),
            in.lzeta_z.data_ptr<float>(),
            nullptr, nullptr, nullptr, nullptr, nullptr,
            in.dirichlet_idx.data_ptr<int64_t>(), kdir, ng);
    }
    cudaMemcpyAsync(in.g_x.data_ptr<float>(), in.lam1_x.data_ptr<float>(), ngb,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(in.g_z.data_ptr<float>(), in.lam1_z.data_ptr<float>(), ngb,
                    cudaMemcpyDeviceToDevice, stream);
    if (kpin > 0) {
        const int pb = (kpin + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.g_x.data_ptr<float>(), in.outer_edge_idx.data_ptr<int64_t>(), kpin);
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.g_z.data_ptr<float>(), in.outer_edge_idx.data_ptr<int64_t>(), kpin);
    }
    sem2d_scale2<<<nb, TPB, 0, stream>>>(
        in.el_inv_M.data_ptr<float>(), in.g_x.data_ptr<float>(), -dt2,
        in.y_x.data_ptr<float>(), ng);
    sem2d_scale2<<<nb, TPB, 0, stream>>>(
        in.el_inv_M.data_ptr<float>(), in.g_z.data_ptr<float>(), -dt2,
        in.y_z.data_ptr<float>(), ng);
    cudaMemsetAsync(in.lchidd.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.s_ac.data_ptr<float>(), 0, ngb, stream);
    if (nifc > 0) {
        const int ib = (nifc + TPB - 1) / TPB;
        const long* ig = in.ifc_idx.data_ptr<int64_t>();
        const float* iw = in.ifc_w.data_ptr<float>();
        const float* inx = in.ifc_nx.data_ptr<float>();
        const float* inz = in.ifc_nz.data_ptr<float>();
        sem2d_ifc_ac<<<ib, TPB, 0, stream>>>(
            in.y_x.data_ptr<float>(), in.y_z.data_ptr<float>(), ig, iw, inx, inz,
            in.lchidd.data_ptr<float>(), nifc, ng);
        if (k >= 1) {
            // u^k: the save-all tape, or the reverse march's lower level
            const size_t o = static_cast<size_t>(k - 1) * ng;
            const float* uxk = bs ? in.u_lo_x.data_ptr<float>()
                                  : in.save_ux.data_ptr<float>() + o;
            const float* uzk = bs ? in.u_lo_z.data_ptr<float>()
                                  : in.save_uz.data_ptr<float>() + o;
            sem2d_ifc_ac<<<ib, TPB, 0, stream>>>(
                uxk, uzk, ig, iw, inx, inz,
                in.s_ac.data_ptr<float>(), nifc, ng);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_b(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    sem2d_fs_axpy<<<nb, TPB, 0, stream>>>(
        in.gA.data_ptr<float>(), dt2, in.lchidd.data_ptr<float>(), ng);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(
            in.lchidd.data_ptr<float>(),
            in.dirichlet_idx.data_ptr<int64_t>(), kdir);
    }
    sem2d_scale2<<<nb, TPB, 0, stream>>>(
        in.ac_kappa.data_ptr<float>(), in.lchidd.data_ptr<float>(), 1.f,
        in.gw.data_ptr<float>(), ng);
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    if (k >= 1) {
        const size_t o = static_cast<size_t>(k - 1) * ng;
        cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
        cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
        cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
        const size_t sh_ac = static_cast<size_t>(7 * m) * sizeof(float);
        // state^k: the tape, or -- under boundary saving -- the reverse
        // march's lower level plus the memories restored from the rings
        const bool bs = in.bs != 0;
        sem2d_acoustic_cpml_grads_f<<<E, m, sh_ac, stream>>>(
            bs ? in.chi_lo.data_ptr<float>()
               : in.save_chi.data_ptr<float>() + o,
            bs ? in.r_psi_x.data_ptr<float>()
               : in.save_psi_x.data_ptr<float>() + o,
            bs ? in.r_psi_z.data_ptr<float>()
               : in.save_psi_z.data_ptr<float>() + o,
            in.a_x.data_ptr<float>(), in.a_z.data_ptr<float>(),
            in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
            in.ac_inv_rho.data_ptr<float>(), in.w1d.data_ptr<float>(),
            in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
            in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
            in.ac_mask_e.data_ptr<float>(), gx, gz, kx, kz, gax, gaz,
            ngll, E, ng, 0);
    } else {
        cudaMemsetAsync(in.w_sum.data_ptr<float>(), 0, ngb, stream);
        cudaMemsetAsync(in.lchi_G.data_ptr<float>(), 0, ngb, stream);
        cudaMemsetAsync(in.lapx.data_ptr<float>(), 0, ngb, stream);
        cudaMemsetAsync(in.lapz.data_ptr<float>(), 0, ngb, stream);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_c(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    if (k < 1) return;
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t o = static_cast<size_t>(k - 1) * ng;
    const bool bs = in.bs != 0;
    sem2d_acoustic_cpml_step_T<false><<<nb, TPB, 0, stream>>>(
        in.gx.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.gax.data_ptr<float>(), in.gaz.data_ptr<float>(),
        in.ac_inv_Mg.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(),
        in.dbx_dx.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        bs ? in.r_zeta_x.data_ptr<float>()
           : in.save_zeta_x.data_ptr<float>() + o,
        bs ? in.r_zeta_z.data_ptr<float>()
           : in.save_zeta_z.data_ptr<float>() + o,
        in.gw.data_ptr<float>(),
        in.lpsi_x.data_ptr<float>(), in.lpsi_z.data_ptr<float>(),
        in.lzeta_x.data_ptr<float>(), in.lzeta_z.data_ptr<float>(),
        in.gbx.data_ptr<float>(), in.gbz.data_ptr<float>(),
        in.gbkx.data_ptr<float>(), in.gbkz.data_ptr<float>(),
        in.gbax.data_ptr<float>(), in.gbaz.data_ptr<float>(),
        in.w_sum.data_ptr<float>(), in.grad_ac_inv_Mg.data_ptr<float>(), ng, 0);
    cudaMemsetAsync(in.lchi_G.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.lapx.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.lapz.data_ptr<float>(), 0, ngb, stream);
    const size_t sh_t = static_cast<size_t>(9 * m) * sizeof(float);
    sem2d_acoustic_cpml_grads_T<<<E, m, sh_t, stream>>>(
        bs ? in.chi_lo.data_ptr<float>()
           : in.save_chi.data_ptr<float>() + o,
        in.gbx.data_ptr<float>(), in.gbz.data_ptr<float>(),
        in.gbkx.data_ptr<float>(), in.gbkz.data_ptr<float>(),
        in.gbax.data_ptr<float>(), in.gbaz.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.ac_inv_rho.data_ptr<float>(), in.w1d.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.ac_mask_e.data_ptr<float>(), in.lchi_G.data_ptr<float>(),
        in.lapx.data_ptr<float>(), in.lapz.data_ptr<float>(),
        in.grad_ac_inv_rho.data_ptr<float>(), ngll, E, ng, 0);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_d(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nt = in.nt;
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    // fluid material grads + the wavelet reduction (ownership-masked pattern)
    sem2d_fs_ac_gradMK<<<nb, TPB, 0, stream>>>(
        in.lchidd.data_ptr<float>(), in.w_sum.data_ptr<float>(), nullptr,
        in.pat_chi.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(k), in.s_ac.data_ptr<float>(),
        in.grad_ac_kappa.data_ptr<float>(), in.grad_ac_inv_M.data_ptr<float>(),
        ng, 0);
    const float* pat_w = in.pat_chi_own.numel()
        ? in.pat_chi_own.data_ptr<float>() : in.pat_chi.data_ptr<float>();
    sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(
        pat_w, in.ac_inv_M.data_ptr<float>(), in.lchidd.data_ptr<float>(),
        1.f, static_cast<int>(k), in.grad_wavelet.data_ptr<float>(), ng);
    // solid element transpose + the coupling scatter into sc_now
    cudaMemsetAsync(in.sc_now_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.sc_now_z.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.sc_prev_a_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.sc_prev_a_z.data_ptr<float>(), 0, ngb, stream);
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    const bool bs = in.bs != 0;
    const size_t so = static_cast<size_t>(k) * E * m;
    // u^k and this step's element histories: the tape, or the reverse
    // march's lower level and the histories march_asm just recomputed
    const float* uxn = k >= 1 ? (bs ? in.u_lo_x.data_ptr<float>()
                                    : in.save_ux.data_ptr<float>()
                                      + static_cast<size_t>(k - 1) * ng)
                              : nullptr;
    const float* uzn = k >= 1 ? (bs ? in.u_lo_z.data_ptr<float>()
                                    : in.save_uz.data_ptr<float>()
                                      + static_cast<size_t>(k - 1) * ng)
                              : nullptr;
    const size_t sh_el = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element_T<<<E, m, sh_el, stream>>>(
        in.y_x.data_ptr<float>(), in.y_z.data_ptr<float>(), uxn, uzn,
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.el_lam.data_ptr<float>(), in.el_mu.data_ptr<float>(),
        in.el_lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.el_w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.el_mask_e.data_ptr<float>(), cf,
        bs ? in.rec_sd_xx.data_ptr<float>() : in.save_sd_xx.data_ptr<float>() + so,
        bs ? in.rec_sd_zx.data_ptr<float>() : in.save_sd_zx.data_ptr<float>() + so,
        bs ? in.rec_sd_xz.data_ptr<float>() : in.save_sd_xz.data_ptr<float>() + so,
        bs ? in.rec_sd_zz.data_ptr<float>() : in.save_sd_zz.data_ptr<float>() + so,
        bs ? in.rec_apx.data_ptr<float>() : in.save_apx.data_ptr<float>() + so,
        bs ? in.rec_apz.data_ptr<float>() : in.save_apz.data_ptr<float>() + so,
        in.mu_xx.data_ptr<float>(), in.mu_zx.data_ptr<float>(),
        in.mu_xz.data_ptr<float>(), in.mu_zz.data_ptr<float>(),
        in.mu_dx.data_ptr<float>(), in.mu_dz.data_ptr<float>(), inv_dt,
        in.sc_now_x.data_ptr<float>(), in.sc_now_z.data_ptr<float>(),
        in.sc_prev_a_x.data_ptr<float>(), in.sc_prev_a_z.data_ptr<float>(),
        in.grad_el_lam.data_ptr<float>(), in.grad_el_mu.data_ptr<float>(),
        in.grad_el_lam2mu.data_ptr<float>(),
        in.grad_el_w_rho_J.data_ptr<float>(), ngll, E, ng);
    sem2d_scale2<<<nb, TPB, 0, stream>>>(
        in.ac_inv_M.data_ptr<float>(), in.lchidd.data_ptr<float>(), -1.f,
        in.yf.data_ptr<float>(), ng);
    if (nifc > 0) {
        const int ib = (nifc + TPB - 1) / TPB;
        sem2d_ifc_el<<<ib, TPB, 0, stream>>>(
            in.yf.data_ptr<float>(), in.ifc_idx.data_ptr<int64_t>(),
            in.ifc_w.data_ptr<float>(), in.ifc_nx.data_ptr<float>(),
            in.ifc_nz.data_ptr<float>(), in.sc_now_x.data_ptr<float>(),
            in.sc_now_z.data_ptr<float>(), nifc, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_e(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const bool bs = in.bs != 0;
    // the three leapfrog levels u^{k+1}, u^k, u^{k-1}: the tape, or the
    // reverse march's hi / lo / km1 (which is why march_fin must run before
    // this entry, and why the caller rotates them only afterwards)
    const float* uxn = k >= 1 ? (bs ? in.u_lo_x.data_ptr<float>()
                                    : in.save_ux.data_ptr<float>()
                                      + static_cast<size_t>(k - 1) * ng)
                              : nullptr;
    const float* uzn = k >= 1 ? (bs ? in.u_lo_z.data_ptr<float>()
                                    : in.save_uz.data_ptr<float>()
                                      + static_cast<size_t>(k - 1) * ng)
                              : nullptr;
    const float* uxm1 = k >= 2 ? (bs ? in.u_km1_x.data_ptr<float>()
                                     : in.save_ux.data_ptr<float>()
                                       + static_cast<size_t>(k - 2) * ng)
                               : nullptr;
    const float* uzm1 = k >= 2 ? (bs ? in.u_km1_z.data_ptr<float>()
                                     : in.save_uz.data_ptr<float>()
                                       + static_cast<size_t>(k - 2) * ng)
                               : nullptr;
    sem2d_elastic_cpml_gradM<<<nb, TPB, 0, stream>>>(
        in.g_x.data_ptr<float>(), in.g_z.data_ptr<float>(),
        bs ? in.u_hi_x.data_ptr<float>()
           : in.save_ux.data_ptr<float>() + static_cast<size_t>(k) * ng,
        bs ? in.u_hi_z.data_ptr<float>()
           : in.save_uz.data_ptr<float>() + static_cast<size_t>(k) * ng,
        uxn, uzn, uxm1, uzm1, in.el_inv_M.data_ptr<float>(),
        in.grad_el_inv_M.data_ptr<float>(), ng);
    const float* pwx = in.pat_x_own.numel() ? in.pat_x_own.data_ptr<float>()
                                            : in.pat_x.data_ptr<float>();
    const float* pwz = in.pat_z_own.numel() ? in.pat_z_own.data_ptr<float>()
                                            : in.pat_z.data_ptr<float>();
    sem2d_elastic_accum_gradW<<<nb, TPB, 0, stream>>>(
        pwx, pwz, in.el_inv_M.data_ptr<float>(),
        in.g_x.data_ptr<float>(), in.g_z.data_ptr<float>(), dt2,
        static_cast<int>(k), in.grad_wavelet.data_ptr<float>(), ng, 0);
    sem2d_elastic_cpml_combine<<<nb, TPB, 0, stream>>>(
        in.g_x.data_ptr<float>(), in.gprev_x.data_ptr<float>(),
        in.zeros.data_ptr<float>(), in.sc_now_x.data_ptr<float>(),
        in.sc_prev_b_x.data_ptr<float>(), in.lamn_x.data_ptr<float>(), ng);
    sem2d_elastic_cpml_combine<<<nb, TPB, 0, stream>>>(
        in.g_z.data_ptr<float>(), in.gprev_z.data_ptr<float>(),
        in.zeros.data_ptr<float>(), in.sc_now_z.data_ptr<float>(),
        in.sc_prev_b_z.data_ptr<float>(), in.lamn_z.data_ptr<float>(), ng);
    sem2d_fs_ac_combine_cpml<<<nb, TPB, 0, stream>>>(
        in.gA.data_ptr<float>(), in.gB.data_ptr<float>(),
        in.lchi_G.data_ptr<float>(), in.lpsi_x.data_ptr<float>(),
        in.lpsi_z.data_ptr<float>(), in.lzeta_x.data_ptr<float>(),
        in.lzeta_z.data_ptr<float>(), in.lapx.data_ptr<float>(),
        in.lapz.data_ptr<float>(), in.gw.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.gchi_k.data_ptr<float>(), in.lpsi_x_n.data_ptr<float>(),
        in.lpsi_z_n.data_ptr<float>(), in.lzeta_x_n.data_ptr<float>(),
        in.lzeta_z_n.data_ptr<float>(), ng);
    sem2d_scale1<<<nb, TPB, 0, stream>>>(
        in.gA.data_ptr<float>(), -1.f, in.gB.data_ptr<float>(), ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD boundary saving for the coupled path --------------------------------
// Three extra entries wrap the reverse step when bs != 0:
//   recon     : re-anchor on the forward's checkpoint (the coupled reverse
//               march accumulates error the acoustic/elastic ones do not --
//               see AcousticElasticSEM2D) and restore the C-PML memories
//               of step k-1 from the rings. Purely local -- no exchange.
//   march_asm : recompute chi_dd^k, the solid force and this step's element
//               histories + Acc. Acc is an ASSEMBLY and ifc_el writes the
//               interface nodes, so this is the one place boundary saving
//               ADDS an exchange to the coupled DD backward (4 fields, the
//               same set the forward exchanges after dd_mid: f_x, f_z, Acc).
//   march_fin : leapfrog both fields back one level and restore the band.
// Both strips run identical arithmetic on inputs that were already summed, so
// a shared cut node comes out the same on either side.
void fluid_solid_dd_bwd_recon(const SemFluidSolidBackwardInputCore& in, int64_t k) {
    if (in.bs == 0) return;
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nt = in.nt;
    const int TPB = 256;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const int Kck = in.ckpt_every;
    if (Kck > 0 && (k + 1) % Kck == 0) {
        const int slot = static_cast<int>((k + 1) / Kck) - 1;
        if (slot >= 0 && slot < static_cast<int>(in.ck_chi.size(0))) {
            const size_t co = static_cast<size_t>(slot) * ng;
            const float* srcv[6] = {in.ck_chi.data_ptr<float>() + co,
                                    in.ck_chi_prev.data_ptr<float>() + co,
                                    in.ck_ux.data_ptr<float>() + co,
                                    in.ck_uz.data_ptr<float>() + co,
                                    in.ck_ux_prev.data_ptr<float>() + co,
                                    in.ck_uz_prev.data_ptr<float>() + co};
            float* dstv[6] = {in.chi_hi.data_ptr<float>(),
                              in.chi_lo.data_ptr<float>(),
                              in.u_hi_x.data_ptr<float>(),
                              in.u_hi_z.data_ptr<float>(),
                              in.u_lo_x.data_ptr<float>(),
                              in.u_lo_z.data_ptr<float>()};
            for (int j = 0; j < 6; ++j)
                cudaMemcpyAsync(dstv[j], srcv[j], ngb,
                                cudaMemcpyDeviceToDevice, stream);
        }
    }
    float* r[4] = {in.r_psi_x.data_ptr<float>(), in.r_psi_z.data_ptr<float>(),
                   in.r_zeta_x.data_ptr<float>(), in.r_zeta_z.data_ptr<float>()};
    const bool r4 = (r[1] == r[0] + ng) && (r[2] == r[0] + 2L * ng)
                 && (r[3] == r[0] + 3L * ng);
    if (r4) {
        cudaMemsetAsync(r[0], 0, 4 * ngb, stream);
    } else {
        for (int j = 0; j < 4; ++j) cudaMemsetAsync(r[j], 0, ngb, stream);
    }
    const int nband = static_cast<int>(in.band_idx.numel());
    if (k >= 1 && nband > 0) {
        const long* bg = in.band_idx.data_ptr<int64_t>();
        const Buf* rg[4] = {&in.band_psi_x, &in.band_psi_z,
                                      &in.band_zeta_x, &in.band_zeta_z};
        for (int j = 0; j < 4; ++j)   // ring slots 1..4 (0 is chi)
            sem2d_band_load(
                sem_hist(*rg[j], in.band_scale, in.band_kind, j + 1, nt),
                k - 1, bg, r[j], nband, TPB, stream);
    }
    if (k == 0) {
        // step 0's element histories are identically zero (u starts at rest)
        // and there is no march at k == 0: clear what element_T will read
        const size_t emb = static_cast<size_t>(E) * m * sizeof(float);
        float* sc[6] = {in.rec_sd_xx.data_ptr<float>(),
                        in.rec_sd_zx.data_ptr<float>(),
                        in.rec_sd_xz.data_ptr<float>(),
                        in.rec_sd_zz.data_ptr<float>(),
                        in.rec_apx.data_ptr<float>(),
                        in.rec_apz.data_ptr<float>()};
        for (int j = 0; j < 6; ++j) cudaMemsetAsync(sc[j], 0, emb, stream);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_march_asm(const SemFluidSolidBackwardInputCore& in,
                                  int64_t k) {
    if (in.bs == 0 || k < 1) return;
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    // chi_dd^k = kappa w_sum^k + invM (pat wav[k] + s_ac^k): w_sum came out of
    // bwd_c's step_T and s_ac out of bwd_a, both already exchanged
    sem2d_fs_chidd<<<nb, TPB, 0, stream>>>(
        in.ac_kappa.data_ptr<float>(), in.w_sum.data_ptr<float>(),
        in.ac_inv_M.data_ptr<float>(), in.pat_chi.data_ptr<float>(),
        in.wavelet.data_ptr<float>(), static_cast<int>(k),
        in.s_ac.data_ptr<float>(), in.rec_chi_dd.data_ptr<float>(), ng, 0);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(
            in.rec_chi_dd.data_ptr<float>(),
            in.dirichlet_idx.data_ptr<int64_t>(), kdir);
    }
    sem2d_fs_pat_wav<<<nb, TPB, 0, stream>>>(
        in.pat_x.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(k), in.rec_f_x.data_ptr<float>(), ng, 0);
    sem2d_fs_pat_wav<<<nb, TPB, 0, stream>>>(
        in.pat_z.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(k), in.rec_f_z.data_ptr<float>(), ng, 0);
    if (nifc > 0) {
        const int ib = (nifc + TPB - 1) / TPB;
        sem2d_ifc_el<<<ib, TPB, 0, stream>>>(
            in.rec_chi_dd.data_ptr<float>(), in.ifc_idx.data_ptr<int64_t>(),
            in.ifc_w.data_ptr<float>(), in.ifc_nx.data_ptr<float>(),
            in.ifc_nz.data_ptr<float>(), in.rec_f_x.data_ptr<float>(),
            in.rec_f_z.data_ptr<float>(), nifc, ng);
    }
    const size_t em2b = static_cast<size_t>(E) * m * 2 * sizeof(float);
    float* rr[6] = {in.rec_rxx.data_ptr<float>(), in.rec_rxz.data_ptr<float>(),
                    in.rec_rzx.data_ptr<float>(), in.rec_rzz.data_ptr<float>(),
                    in.rec_rdx.data_ptr<float>(), in.rec_rdz.data_ptr<float>()};
    const size_t em2 = static_cast<size_t>(E) * m * 2;
    bool r6 = true;
    for (int j = 1; j < 6; ++j)
        r6 = r6 && (rr[j] == rr[0] + static_cast<size_t>(j) * em2);
    if (r6) {
        cudaMemsetAsync(rr[0], 0, 6 * em2b, stream);
    } else {
        for (int j = 0; j < 6; ++j) cudaMemsetAsync(rr[j], 0, em2b, stream);
    }
    cudaMemsetAsync(in.rec_Acc_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.rec_Acc_z.data_ptr<float>(), 0, ngb, stream);
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    const size_t sh_el = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element<<<E, m, sh_el, stream>>>(
        in.u_lo_x.data_ptr<float>(), in.u_lo_z.data_ptr<float>(),
        in.u_hi_x.data_ptr<float>(), in.u_hi_z.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.el_lam.data_ptr<float>(), in.el_mu.data_ptr<float>(),
        in.el_lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.el_w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.el_mask_e.data_ptr<float>(), cf,
        rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], inv_dt,
        in.rec_Acc_x.data_ptr<float>(), in.rec_Acc_z.data_ptr<float>(),
        in.rec_sd_xx.data_ptr<float>(), in.rec_sd_zx.data_ptr<float>(),
        in.rec_sd_xz.data_ptr<float>(), in.rec_sd_zz.data_ptr<float>(),
        in.rec_apx.data_ptr<float>(), in.rec_apz.data_ptr<float>(),
        0L, ngll, E, ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_bwd_march_fin(const SemFluidSolidBackwardInputCore& in,
                                  int64_t k) {
    if (in.bs == 0 || k < 1) return;
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    sem2d_elastic_newmark_force<<<nb, TPB, 0, stream>>>(
        in.u_lo_x.data_ptr<float>(), in.u_lo_z.data_ptr<float>(),
        in.u_hi_x.data_ptr<float>(), in.u_hi_z.data_ptr<float>(),
        in.rec_Acc_x.data_ptr<float>(), in.rec_Acc_z.data_ptr<float>(),
        in.rec_f_x.data_ptr<float>(), in.rec_f_z.data_ptr<float>(),
        in.el_inv_M.data_ptr<float>(), dt2,
        in.u_km1_x.data_ptr<float>(), in.u_km1_z.data_ptr<float>(), ng);
    if (kpin > 0) {
        const int pb = (kpin + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.u_km1_x.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.u_km1_z.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
    }
    const int nbn = static_cast<int>(in.band_node_idx.numel());
    const int nbe = static_cast<int>(in.band_elem_idx.numel());
    const int bk = in.band_kind;
    if (k >= 2 && nbn > 0) {     // band_ux[n] holds u^{n+1} -> u^{k-1} is row k-2
        const long* bnode = in.band_node_idx.data_ptr<int64_t>();
        sem2d_band_load(sem_hist(in.band_ux, in.band_scale, bk, 5, nt),
                        k - 2, bnode, in.u_km1_x.data_ptr<float>(), nbn,
                        TPB, stream);
        sem2d_band_load(sem_hist(in.band_uz, in.band_scale, bk, 6, nt),
                        k - 2, bnode, in.u_km1_z.data_ptr<float>(), nbn,
                        TPB, stream);
    }
    if (nbe > 0) {
        const long* belem = in.band_elem_idx.data_ptr<int64_t>();
        const long Em = static_cast<long>(E) * m;
        const Buf* src[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                       &in.band_sd_xz, &in.band_sd_zz,
                                       &in.band_apx, &in.band_apz};
        float* dst[6] = {in.rec_sd_xx.data_ptr<float>(),
                         in.rec_sd_zx.data_ptr<float>(),
                         in.rec_sd_xz.data_ptr<float>(),
                         in.rec_sd_zz.data_ptr<float>(),
                         in.rec_apx.data_ptr<float>(),
                         in.rec_apz.data_ptr<float>()};
        for (int j = 0; j < 6; ++j)
            sem2d_band_load_elem(
                sem_hist(*src[j], in.band_scale_e, in.band_kind_e, j, nt, 1, nbe),
                k, belem, dst[j], nbe, m, Em, TPB, stream);
    }
    if (k == 1) {
        cudaMemsetAsync(in.chi_km1.data_ptr<float>(), 0, ngb, stream);
    } else {
        sem2d_fs_chi_leapfrog<<<nb, TPB, 0, stream>>>(
            in.chi_lo.data_ptr<float>(), in.chi_hi.data_ptr<float>(),
            in.rec_chi_dd.data_ptr<float>(), dt2,
            in.chi_km1.data_ptr<float>(), ng);
        if (kdir > 0) {
            const int db = (kdir + TPB - 1) / TPB;
            sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(
                in.chi_km1.data_ptr<float>(),
                in.dirichlet_idx.data_ptr<int64_t>(), kdir);
        }
        const int nband = static_cast<int>(in.band_idx.numel());
        if (nband > 0)
            sem2d_band_load(
                sem_hist(in.band_chi, in.band_scale, bk, 0, nt),
                k - 2, in.band_idx.data_ptr<int64_t>(),
                in.chi_km1.data_ptr<float>(), nband, TPB, stream);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped backward (elastic C-PML) ------------------------------------
// The reverse step has ONE assembly (element_T's sc_now/sc_prev scatter), so
// DD splits around one 4-field exchange -- plus, under boundary saving, a
// second 2-field exchange for the reverse march's Acc recompute:
//   march_asm (bs only): element on (u_lo, u_hi) -> r_Acc      [exchange 2]
//   march_fin (bs only): newmark_force -> u_km1, dirichlet, band restores
//   pre : adjoint source at nt-1 (caller-seeded), adj_carry, element_T
//                                                             [exchange 4]
//   end : gradM + wavelet grad + adjoint source(n-1) + combine
// Carried-buffer rotation (lam1/lamn, g/gprev, sc_prev a/b, u_hi/lo/km1) is
// the caller's. B == 1.
void elastic_cpml_dd_bwd_march_asm(const SemElasticCpmlBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    SWEEP_CHECK(in.B == 1, "elastic_cpml_dd_bwd: DD is single-shot");
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const size_t em2b = static_cast<size_t>(E) * m * 2 * sizeof(float);
    float* r_rxx = in.rec_rxx.data_ptr<float>();
    const size_t em2 = static_cast<size_t>(E) * m * 2;
    const bool r6 = (in.rec_rxz.data_ptr<float>() == r_rxx + em2)
                 && (in.rec_rzx.data_ptr<float>() == r_rxx + 2 * em2)
                 && (in.rec_rzz.data_ptr<float>() == r_rxx + 3 * em2)
                 && (in.rec_rdx.data_ptr<float>() == r_rxx + 4 * em2)
                 && (in.rec_rdz.data_ptr<float>() == r_rxx + 5 * em2);
    if (r6) {
        cudaMemsetAsync(r_rxx, 0, 6 * em2b, stream);
    } else {
        cudaMemsetAsync(r_rxx, 0, em2b, stream);
        cudaMemsetAsync(in.rec_rxz.data_ptr<float>(), 0, em2b, stream);
        cudaMemsetAsync(in.rec_rzx.data_ptr<float>(), 0, em2b, stream);
        cudaMemsetAsync(in.rec_rzz.data_ptr<float>(), 0, em2b, stream);
        cudaMemsetAsync(in.rec_rdx.data_ptr<float>(), 0, em2b, stream);
        cudaMemsetAsync(in.rec_rdz.data_ptr<float>(), 0, em2b, stream);
    }
    cudaMemsetAsync(in.rec_Acc_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.rec_Acc_z.data_ptr<float>(), 0, ngb, stream);
    const size_t shmem = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element<<<E, m, shmem, stream>>>(
        in.u_lo_x.data_ptr<float>(), in.u_lo_z.data_ptr<float>(),
        in.u_hi_x.data_ptr<float>(), in.u_hi_z.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.lam.data_ptr<float>(), in.mu.data_ptr<float>(),
        in.lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.mask_e.data_ptr<float>(), cf,
        r_rxx, in.rec_rxz.data_ptr<float>(), in.rec_rzx.data_ptr<float>(),
        in.rec_rzz.data_ptr<float>(), in.rec_rdx.data_ptr<float>(),
        in.rec_rdz.data_ptr<float>(), inv_dt,
        in.rec_Acc_x.data_ptr<float>(), in.rec_Acc_z.data_ptr<float>(),
        in.rec_sd_xx.data_ptr<float>(), in.rec_sd_zx.data_ptr<float>(),
        in.rec_sd_xz.data_ptr<float>(), in.rec_sd_zz.data_ptr<float>(),
        in.rec_apx.data_ptr<float>(), in.rec_apz.data_ptr<float>(),
        0L, ngll, E, ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic_cpml_dd_bwd_march_fin(const SemElasticCpmlBackwardInputCore& in,
                                   int64_t n) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, ngll = in.ngll, m = ngll * ngll;
    const int nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    sem2d_elastic_newmark<<<nb, TPB, 0, stream>>>(
        in.u_lo_x.data_ptr<float>(), in.u_lo_z.data_ptr<float>(),
        in.u_hi_x.data_ptr<float>(), in.u_hi_z.data_ptr<float>(),
        in.rec_Acc_x.data_ptr<float>(), in.rec_Acc_z.data_ptr<float>(),
        in.pat_x.data_ptr<float>(), in.pat_z.data_ptr<float>(),
        in.inv_M.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(n), dt2,
        in.u_km1_x.data_ptr<float>(), in.u_km1_z.data_ptr<float>(), ng, 0);
    if (kpin > 0) {
        const int pb = (kpin + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.u_km1_x.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.u_km1_z.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
    }
    const int nbn = static_cast<int>(in.band_node_idx.numel());
    const int nbe = static_cast<int>(in.band_elem_idx.numel());
    const long* bnode = in.band_node_idx.data_ptr<int64_t>();
    const long* belem = in.band_elem_idx.data_ptr<int64_t>();
    const int bnb = (nbn + TPB - 1) / TPB;
    const int beb = (nbe * m + TPB - 1) / TPB;
    if (n >= 2) {   // band_ux[n-2] holds u^{n-1}
        const size_t no = static_cast<size_t>(n - 2) * nbn;
        const int bkind = in.band_kind;
        sem2d_band_load(sem_hist(in.band_ux, in.band_scale, bkind, 0, nt),
                        n - 2, bnode, in.u_km1_x.data_ptr<float>(), nbn,
                        TPB, stream);
        sem2d_band_load(sem_hist(in.band_uz, in.band_scale, bkind, 1, nt),
                        n - 2, bnode, in.u_km1_z.data_ptr<float>(), nbn,
                        TPB, stream);
    }
    const Buf* ring[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                    &in.band_sd_xz, &in.band_sd_zz,
                                    &in.band_apx, &in.band_apz};
    float* scr[6] = {in.rec_sd_xx.data_ptr<float>(),
                     in.rec_sd_zx.data_ptr<float>(),
                     in.rec_sd_xz.data_ptr<float>(),
                     in.rec_sd_zz.data_ptr<float>(),
                     in.rec_apx.data_ptr<float>(),
                     in.rec_apz.data_ptr<float>()};
    for (int j = 0; j < 6; ++j)
        sem2d_band_load_elem(
            sem_hist(*ring[j], in.band_scale_e, in.band_kind_e, j, nt, 1, nbe),
            n, belem, scr[j], nbe, m, 0L, TPB, stream);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic_cpml_dd_bwd_pre(const SemElasticCpmlBackwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    SWEEP_CHECK(in.B == 1, "elastic_cpml_dd_bwd: DD is single-shot");
    const bool bs = in.bs != 0;
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const int nrec = in.n_rec, nt = in.nt;
    if (n == nt - 1 && nrec > 0) {
        // seed lambda(u^nt) = Rec^T grad_seis[:, nt-1] (the fused driver's
        // pre-loop injection). Cut-column receivers are registered on both
        // strips and their grad rows were patched to the owner's, so both
        // copies of a shared DOF take the SAME injection.
        const int rj = (nrec * m + TPB - 1) / TPB;
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sx.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.lam1_x.data_ptr<float>(),
            nt - 1, nt, nrec, m, ng);
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sz.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.lam1_z.data_ptr<float>(),
            nt - 1, nt, nrec, m, ng);
    }
    sem2d_elastic_adj_carry<<<nb, TPB, 0, stream>>>(
        in.lam1_x.data_ptr<float>(), in.lam1_z.data_ptr<float>(),
        in.pin_mask.numel() ? in.pin_mask.data_ptr<float>() : nullptr,
        in.inv_M.data_ptr<float>(), -dt2,
        in.g_x.data_ptr<float>(), in.g_z.data_ptr<float>(),
        in.y_x.data_ptr<float>(), in.y_z.data_ptr<float>(), ng);
    float* sc_now_x = in.sc_now_x.data_ptr<float>();
    const bool z4 = (in.sc_now_z.data_ptr<float>() == sc_now_x + ng)
                 && (in.adj_x.data_ptr<float>() == sc_now_x + 2L * ng)
                 && (in.adj_z.data_ptr<float>() == sc_now_x + 3L * ng);
    if (z4) {
        cudaMemsetAsync(sc_now_x, 0, 4 * ngb, stream);
    } else {
        cudaMemsetAsync(sc_now_x, 0, ngb, stream);
        cudaMemsetAsync(in.sc_now_z.data_ptr<float>(), 0, ngb, stream);
    }
    float* scp_a_x = in.sc_prev_a_x.data_ptr<float>();
    if (in.sc_prev_a_z.data_ptr<float>() == scp_a_x + ng) {
        cudaMemsetAsync(scp_a_x, 0, 2 * ngb, stream);
    } else {
        cudaMemsetAsync(scp_a_x, 0, ngb, stream);
        cudaMemsetAsync(in.sc_prev_a_z.data_ptr<float>(), 0, ngb, stream);
    }
    const size_t so = bs ? 0 : static_cast<size_t>(n) * E * m;
    const float* uxn = n >= 1 ? (bs ? in.u_lo_x.data_ptr<float>()
                                    : in.save_ux.data_ptr<float>()
                                      + static_cast<size_t>(n - 1) * ng)
                              : nullptr;
    const float* uzn = n >= 1 ? (bs ? in.u_lo_z.data_ptr<float>()
                                    : in.save_uz.data_ptr<float>()
                                      + static_cast<size_t>(n - 1) * ng)
                              : nullptr;
    const size_t shmem = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element_T<<<E, m, shmem, stream>>>(
        in.y_x.data_ptr<float>(), in.y_z.data_ptr<float>(), uxn, uzn,
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.lam.data_ptr<float>(), in.mu.data_ptr<float>(),
        in.lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.mask_e.data_ptr<float>(), cf,
        bs ? in.rec_sd_xx.data_ptr<float>() : in.save_sd_xx.data_ptr<float>() + so,
        bs ? in.rec_sd_zx.data_ptr<float>() : in.save_sd_zx.data_ptr<float>() + so,
        bs ? in.rec_sd_xz.data_ptr<float>() : in.save_sd_xz.data_ptr<float>() + so,
        bs ? in.rec_sd_zz.data_ptr<float>() : in.save_sd_zz.data_ptr<float>() + so,
        bs ? in.rec_apx.data_ptr<float>() : in.save_apx.data_ptr<float>() + so,
        bs ? in.rec_apz.data_ptr<float>() : in.save_apz.data_ptr<float>() + so,
        in.mu_xx.data_ptr<float>(), in.mu_zx.data_ptr<float>(),
        in.mu_xz.data_ptr<float>(), in.mu_zz.data_ptr<float>(),
        in.mu_dx.data_ptr<float>(), in.mu_dz.data_ptr<float>(), inv_dt,
        sc_now_x, in.sc_now_z.data_ptr<float>(),
        scp_a_x, in.sc_prev_a_z.data_ptr<float>(),
        in.grad_lam.data_ptr<float>(), in.grad_mu.data_ptr<float>(),
        in.grad_lam2mu.data_ptr<float>(), in.grad_w_rho_J.data_ptr<float>(),
        ngll, E, ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic_cpml_dd_bwd_end(const SemElasticCpmlBackwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int ng = in.n_global, ngll = in.ngll, m = ngll * ngll;
    const int nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const bool bs = in.bs != 0;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const float* uxn = n >= 1 ? (bs ? in.u_lo_x.data_ptr<float>()
                                    : in.save_ux.data_ptr<float>()
                                      + static_cast<size_t>(n - 1) * ng)
                              : nullptr;
    const float* uzn = n >= 1 ? (bs ? in.u_lo_z.data_ptr<float>()
                                    : in.save_uz.data_ptr<float>()
                                      + static_cast<size_t>(n - 1) * ng)
                              : nullptr;
    const float* uxm1 = n >= 2 ? (bs ? in.u_km1_x.data_ptr<float>()
                                     : in.save_ux.data_ptr<float>()
                                       + static_cast<size_t>(n - 2) * ng)
                               : nullptr;
    const float* uzm1 = n >= 2 ? (bs ? in.u_km1_z.data_ptr<float>()
                                     : in.save_uz.data_ptr<float>()
                                       + static_cast<size_t>(n - 2) * ng)
                               : nullptr;
    sem2d_elastic_cpml_gradM<<<nb, TPB, 0, stream>>>(
        in.g_x.data_ptr<float>(), in.g_z.data_ptr<float>(),
        bs ? in.u_hi_x.data_ptr<float>()
           : in.save_ux.data_ptr<float>() + static_cast<size_t>(n) * ng,
        bs ? in.u_hi_z.data_ptr<float>()
           : in.save_uz.data_ptr<float>() + static_cast<size_t>(n) * ng,
        uxn, uzn, uxm1, uzm1, in.inv_M.data_ptr<float>(),
        in.grad_inv_M.data_ptr<float>(), ng);
    // wavelet gradient: a global reduction, so it reads the ownership-masked
    // source pattern (interface columns are complete on both strips)
    const float* pw_x = in.pat_x_own.numel() ? in.pat_x_own.data_ptr<float>()
                                             : in.pat_x.data_ptr<float>();
    const float* pw_z = in.pat_z_own.numel() ? in.pat_z_own.data_ptr<float>()
                                             : in.pat_z.data_ptr<float>();
    sem2d_elastic_accum_gradW<<<nb, TPB, 0, stream>>>(
        pw_x, pw_z, in.inv_M.data_ptr<float>(),
        in.g_x.data_ptr<float>(), in.g_z.data_ptr<float>(), dt2,
        static_cast<int>(n), in.grad_wavelet.data_ptr<float>(), ng, 0);
    float* adj_x = in.adj_x.data_ptr<float>();
    float* sc_now_x = in.sc_now_x.data_ptr<float>();
    const bool z4 = (in.sc_now_z.data_ptr<float>() == sc_now_x + ng)
                 && (adj_x == sc_now_x + 2L * ng)
                 && (in.adj_z.data_ptr<float>() == sc_now_x + 3L * ng);
    if (!z4) {
        cudaMemsetAsync(adj_x, 0, ngb, stream);
        cudaMemsetAsync(in.adj_z.data_ptr<float>(), 0, ngb, stream);
    }
    if (n >= 1 && nrec > 0) {
        const int rj = (nrec * m + TPB - 1) / TPB;
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sx.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), adj_x, static_cast<int>(n - 1), nt,
            nrec, m, ng);
        sem2d_adjoint_source_b<<<dim3(rj, 1), TPB, 0, stream>>>(
            in.grad_sz.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.adj_z.data_ptr<float>(),
            static_cast<int>(n - 1), nt, nrec, m, ng);
    }
    sem2d_elastic_cpml_combine<<<nb, TPB, 0, stream>>>(
        in.g_x.data_ptr<float>(), in.gprev_x.data_ptr<float>(), adj_x,
        sc_now_x, in.sc_prev_b_x.data_ptr<float>(),
        in.lamn_x.data_ptr<float>(), ng);
    sem2d_elastic_cpml_combine<<<nb, TPB, 0, stream>>>(
        in.g_z.data_ptr<float>(), in.gprev_z.data_ptr<float>(),
        in.adj_z.data_ptr<float>(), in.sc_now_z.data_ptr<float>(),
        in.sc_prev_b_z.data_ptr<float>(), in.lamn_z.data_ptr<float>(), ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- coupled fluid-solid backward (P4) --------------------------------------
// Reverse loop over forward iterations k = nt-1 .. 0. Fluid side follows the
// P3b recursion (gA/gB, psi/zeta adjoints, step_T/grads_T recomputes), solid
// side the P3e recursion (g/gprev, 6 memory adjoints, element transpose); the
// coupling transposes are: lchidd = Dir(dt2 gA + Ifc_ac(y_el)) and
// sc_now += Ifc_el(-inv_M_ac o lchidd)  (the interface kernels are mutual
// transposes, so both reuse the forward kernels).
void fluid_solid_backward(const SemFluidSolidBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.ac_inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const bool cp = in.has_cpml != 0;
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int kpin = cp ? static_cast<int>(in.outer_edge_idx.numel()) : 0;
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    // shot batch (see sem_types.h): shared model, per-shot (B, ...) gradient
    // outputs (the caller sums). C-PML flavour only.
    const int B = in.B;
    SWEEP_CHECK(B >= 1, "fluid_solid_backward: B=", B);
    SWEEP_CHECK(B == 1 || cp,
                "fluid_solid_backward: the shot batch needs the coupled C-PML");
    SWEEP_CHECK(in.gA.numel() == static_cast<int64_t>(B) * ng,
                "fluid_solid_backward: gA holds ", in.gA.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(in.wavelet.numel() == nt
                || in.wavelet.numel() == static_cast<int64_t>(B) * nt,
                "fluid_solid_backward: wavelet must be (nt,) or (B, nt)");
    SWEEP_CHECK(B == 1 || (in.grad_ac_inv_M.numel() == static_cast<int64_t>(B) * ng
                           && in.grad_wavelet.numel() == static_cast<int64_t>(B) * nt),
                "fluid_solid_backward: gradient buffers must be per-shot (B, ...)");
    const int wst = in.wavelet.numel() == nt ? 0 : nt;

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* wJ = in.wJ.data_ptr<float>();
    const float* w1d = in.w1d.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>(); const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>(); const float* iez = in.iez.data_ptr<float>();
    const float* ac_ir = in.ac_inv_rho.data_ptr<float>();
    const float* ac_me = in.ac_mask_e.data_ptr<float>();
    const float* ac_kp = in.ac_kappa.data_ptr<float>();
    const float* ac_iM = in.ac_inv_M.data_ptr<float>();
    const float* ac_iMg = cp ? in.ac_inv_Mg.data_ptr<float>() : nullptr;
    const float* ax = cp ? in.a_x.data_ptr<float>() : nullptr;
    const float* bx = cp ? in.b_x.data_ptr<float>() : nullptr;
    const float* dbxdx = cp ? in.dbx_dx.data_ptr<float>() : nullptr;
    const float* az = cp ? in.a_z.data_ptr<float>() : nullptr;
    const float* bz = cp ? in.b_z.data_ptr<float>() : nullptr;
    const float* dbzdz = cp ? in.dbz_dz.data_ptr<float>() : nullptr;
    const long*  didx = kdir > 0 ? in.dirichlet_idx.data_ptr<int64_t>() : nullptr;
    const float* el_lam = in.el_lam.data_ptr<float>();
    const float* el_mu = in.el_mu.data_ptr<float>();
    const float* el_l2m = in.el_lam2mu.data_ptr<float>();
    const float* el_iM = in.el_inv_M.data_ptr<float>();
    const float* el_me = in.el_mask_e.data_ptr<float>();
    const float* el_wrJ = cp ? in.el_w_rho_J.data_ptr<float>() : nullptr;
    const long*  pin_idx = kpin > 0 ? in.outer_edge_idx.data_ptr<int64_t>() : nullptr;
    const long*  ifc_g = nifc > 0 ? in.ifc_idx.data_ptr<int64_t>() : nullptr;
    const float* ifc_w = nifc > 0 ? in.ifc_w.data_ptr<float>() : nullptr;
    const float* ifc_nx = nifc > 0 ? in.ifc_nx.data_ptr<float>() : nullptr;
    const float* ifc_nz = nifc > 0 ? in.ifc_nz.data_ptr<float>() : nullptr;
    const float* pat_chi = in.pat_chi.data_ptr<float>();
    const float* pat_x = in.pat_x.data_ptr<float>();
    const float* pat_z = in.pat_z.data_ptr<float>();
    const float* wav = in.wavelet.data_ptr<float>();
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    const float* g_schi = nrec > 0 ? in.grad_schi.data_ptr<float>() : nullptr;
    const float* g_sx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* g_sz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;

    ElCpmlCoefs cf{};
    if (cp) {
        // the helper, not a hand-copied packing: a field added to
        // ElCpmlCoefs used to reach this site SILENTLY NULL, which is not
        // a crash -- the coupled path simply never took the interior fast
        // path the elastic one had just gained.
        cf = sem2d_el_coefs(in);
    }

    const float* sch = in.save_chi.data_ptr<float>();
    const float* spx = cp ? in.save_psi_x.data_ptr<float>() : nullptr;
    const float* spz = cp ? in.save_psi_z.data_ptr<float>() : nullptr;
    const float* szx = cp ? in.save_zeta_x.data_ptr<float>() : nullptr;
    const float* szz = cp ? in.save_zeta_z.data_ptr<float>() : nullptr;
    const float* sux = in.save_ux.data_ptr<float>();
    const float* suz = in.save_uz.data_ptr<float>();
    const float* sdxx = cp ? in.save_sd_xx.data_ptr<float>() : nullptr;
    const float* sdzx = cp ? in.save_sd_zx.data_ptr<float>() : nullptr;
    const float* sdxz = cp ? in.save_sd_xz.data_ptr<float>() : nullptr;
    const float* sdzz = cp ? in.save_sd_zz.data_ptr<float>() : nullptr;
    const float* sapx = cp ? in.save_apx.data_ptr<float>() : nullptr;
    const float* sapz = cp ? in.save_apz.data_ptr<float>() : nullptr;
    // ---- boundary saving: rings + reverse-march state (see sem_types.h) ----
    const bool bs = in.bs != 0;
    SWEEP_CHECK(!bs || cp, "fluid_solid_backward: boundary saving needs C-PML");
    const int nband = bs ? static_cast<int>(in.band_idx.numel()) : 0;
    const long* band_g = nband > 0 ? in.band_idx.data_ptr<int64_t>() : nullptr;
    const int bkind = in.band_kind;
    const SemBandHist h_chi = bs ? sem_hist(in.band_chi, in.band_scale, bkind, 0, nt, B) : SemBandHist();
    const SemBandHist h_px = bs ? sem_hist(in.band_psi_x, in.band_scale, bkind, 1, nt, B) : SemBandHist();
    const SemBandHist h_pz = bs ? sem_hist(in.band_psi_z, in.band_scale, bkind, 2, nt, B) : SemBandHist();
    const SemBandHist h_zx = bs ? sem_hist(in.band_zeta_x, in.band_scale, bkind, 3, nt, B) : SemBandHist();
    const SemBandHist h_zz = bs ? sem_hist(in.band_zeta_z, in.band_scale, bkind, 4, nt, B) : SemBandHist();
    const int nbn = bs ? static_cast<int>(in.band_node_idx.numel()) : 0;
    const int nbe = bs ? static_cast<int>(in.band_elem_idx.numel()) : 0;
    const long* bnode = bs ? in.band_node_idx.data_ptr<int64_t>() : nullptr;
    const long* belem = bs ? in.band_elem_idx.data_ptr<int64_t>() : nullptr;
    float* chi_hi = bs ? in.chi_hi.data_ptr<float>() : nullptr;
    float* chi_lo = bs ? in.chi_lo.data_ptr<float>() : nullptr;
    float* chi_km1 = bs ? in.chi_km1.data_ptr<float>() : nullptr;
    float* rec_chi_dd = bs ? in.rec_chi_dd.data_ptr<float>() : nullptr;
    float* r_px = bs ? in.r_psi_x.data_ptr<float>() : nullptr;
    float* r_pz = bs ? in.r_psi_z.data_ptr<float>() : nullptr;
    float* r_zx = bs ? in.r_zeta_x.data_ptr<float>() : nullptr;
    float* r_zz = bs ? in.r_zeta_z.data_ptr<float>() : nullptr;
    float* u_hi_x = bs ? in.u_hi_x.data_ptr<float>() : nullptr;
    float* u_hi_z = bs ? in.u_hi_z.data_ptr<float>() : nullptr;
    float* u_lo_x = bs ? in.u_lo_x.data_ptr<float>() : nullptr;
    float* u_lo_z = bs ? in.u_lo_z.data_ptr<float>() : nullptr;
    float* u_km1_x = bs ? in.u_km1_x.data_ptr<float>() : nullptr;
    float* u_km1_z = bs ? in.u_km1_z.data_ptr<float>() : nullptr;
    float* r_sdxx = bs ? in.rec_sd_xx.data_ptr<float>() : nullptr;
    float* r_sdzx = bs ? in.rec_sd_zx.data_ptr<float>() : nullptr;
    float* r_sdxz = bs ? in.rec_sd_xz.data_ptr<float>() : nullptr;
    float* r_sdzz = bs ? in.rec_sd_zz.data_ptr<float>() : nullptr;
    float* r_apx = bs ? in.rec_apx.data_ptr<float>() : nullptr;
    float* r_apz = bs ? in.rec_apz.data_ptr<float>() : nullptr;
    float* r_rxx = bs ? in.rec_rxx.data_ptr<float>() : nullptr;
    float* r_rxz = bs ? in.rec_rxz.data_ptr<float>() : nullptr;
    float* r_rzx = bs ? in.rec_rzx.data_ptr<float>() : nullptr;
    float* r_rzz = bs ? in.rec_rzz.data_ptr<float>() : nullptr;
    float* r_rdx = bs ? in.rec_rdx.data_ptr<float>() : nullptr;
    float* r_rdz = bs ? in.rec_rdz.data_ptr<float>() : nullptr;
    float* r_Acc_x = bs ? in.rec_Acc_x.data_ptr<float>() : nullptr;
    float* r_Acc_z = bs ? in.rec_Acc_z.data_ptr<float>() : nullptr;
    float* r_f_x = bs ? in.rec_f_x.data_ptr<float>() : nullptr;
    float* r_f_z = bs ? in.rec_f_z.data_ptr<float>() : nullptr;
    const int Kck = bs ? in.ckpt_every : 0;
    const int nck = Kck > 0 ? static_cast<int>(in.ck_chi.size(0)) : 0;

    float* gA = in.gA.data_ptr<float>(); float* gB = in.gB.data_ptr<float>();
    float* gchi_k = in.gchi_k.data_ptr<float>();
    float* lchidd = in.lchidd.data_ptr<float>();
    float* s_ac = in.s_ac.data_ptr<float>();
    float* Kchi = in.Kchi.data_ptr<float>();
    float* yf = in.yf.data_ptr<float>();
    float* zeros = in.zeros.data_ptr<float>();
    float* lpx = cp ? in.lpsi_x.data_ptr<float>() : nullptr;
    float* lpz = cp ? in.lpsi_z.data_ptr<float>() : nullptr;
    float* lzx = cp ? in.lzeta_x.data_ptr<float>() : nullptr;
    float* lzz = cp ? in.lzeta_z.data_ptr<float>() : nullptr;
    float* lpxn = cp ? in.lpsi_x_n.data_ptr<float>() : nullptr;
    float* lpzn = cp ? in.lpsi_z_n.data_ptr<float>() : nullptr;
    float* lzxn = cp ? in.lzeta_x_n.data_ptr<float>() : nullptr;
    float* lzzn = cp ? in.lzeta_z_n.data_ptr<float>() : nullptr;
    float* gxp = cp ? in.gx.data_ptr<float>() : nullptr;
    float* gzp = cp ? in.gz.data_ptr<float>() : nullptr;
    float* kxp = cp ? in.kx.data_ptr<float>() : nullptr;
    float* kzp = cp ? in.kz.data_ptr<float>() : nullptr;
    float* gaxp = cp ? in.gax.data_ptr<float>() : nullptr;
    float* gazp = cp ? in.gaz.data_ptr<float>() : nullptr;
    float* w_sum = cp ? in.w_sum.data_ptr<float>() : nullptr;
    float* gw = cp ? in.gw.data_ptr<float>() : nullptr;
    float* gbxp = cp ? in.gbx.data_ptr<float>() : nullptr;
    float* gbzp = cp ? in.gbz.data_ptr<float>() : nullptr;
    float* gbkx = cp ? in.gbkx.data_ptr<float>() : nullptr;
    float* gbkz = cp ? in.gbkz.data_ptr<float>() : nullptr;
    float* gbax = cp ? in.gbax.data_ptr<float>() : nullptr;
    float* gbaz = cp ? in.gbaz.data_ptr<float>() : nullptr;
    float* lchi_G = in.lchi_G.data_ptr<float>();
    float* lapx = cp ? in.lapx.data_ptr<float>() : nullptr;
    float* lapz = cp ? in.lapz.data_ptr<float>() : nullptr;

    float* lam1_x = in.lam1_x.data_ptr<float>(); float* lam1_z = in.lam1_z.data_ptr<float>();
    float* lamn_x = in.lamn_x.data_ptr<float>(); float* lamn_z = in.lamn_z.data_ptr<float>();
    float* g_x = in.g_x.data_ptr<float>();       float* g_z = in.g_z.data_ptr<float>();
    float* gprev_x = in.gprev_x.data_ptr<float>(); float* gprev_z = in.gprev_z.data_ptr<float>();
    float* y_x = in.y_x.data_ptr<float>();       float* y_z = in.y_z.data_ptr<float>();
    float* adj_x = in.adj_x.data_ptr<float>();   float* adj_z = in.adj_z.data_ptr<float>();
    float* tmp_x = in.tmp_x.data_ptr<float>();   float* tmp_z = in.tmp_z.data_ptr<float>();
    float* sc_now_x = in.sc_now_x.data_ptr<float>(); float* sc_now_z = in.sc_now_z.data_ptr<float>();
    float* scp_a_x = in.sc_prev_a_x.data_ptr<float>(); float* scp_a_z = in.sc_prev_a_z.data_ptr<float>();
    float* scp_b_x = in.sc_prev_b_x.data_ptr<float>(); float* scp_b_z = in.sc_prev_b_z.data_ptr<float>();
    float* mu_xx = cp ? in.mu_xx.data_ptr<float>() : nullptr;
    float* mu_zx = cp ? in.mu_zx.data_ptr<float>() : nullptr;
    float* mu_xz = cp ? in.mu_xz.data_ptr<float>() : nullptr;
    float* mu_zz = cp ? in.mu_zz.data_ptr<float>() : nullptr;
    float* mu_dx = cp ? in.mu_dx.data_ptr<float>() : nullptr;
    float* mu_dz = cp ? in.mu_dz.data_ptr<float>() : nullptr;

    float* gAcIR = in.grad_ac_inv_rho.data_ptr<float>();
    float* gAcIM = in.grad_ac_inv_M.data_ptr<float>();
    float* gAcKp = in.grad_ac_kappa.data_ptr<float>();
    float* gAcIMg = cp ? in.grad_ac_inv_Mg.data_ptr<float>() : nullptr;
    float* gElLam = in.grad_el_lam.data_ptr<float>();
    float* gElMu = in.grad_el_mu.data_ptr<float>();
    float* gElL2m = in.grad_el_lam2mu.data_ptr<float>();
    float* gElIM = in.grad_el_inv_M.data_ptr<float>();
    float* gElWrJ = in.grad_el_w_rho_J.data_ptr<float>();
    float* gWav = in.grad_wavelet.data_ptr<float>();

    const size_t sh_ac = static_cast<size_t>((cp ? 7 : 3) * m) * sizeof(float);
    const size_t sh_acT = static_cast<size_t>(9 * m) * sizeof(float);
    const size_t sh_el = static_cast<size_t>((cp ? 8 : 6) * m) * sizeof(float);
    const size_t sh_im2 = static_cast<size_t>(2 * m) * sizeof(float);
    const size_t sh_im4 = static_cast<size_t>(4 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const int pb = kpin > 0 ? (kpin + TPB - 1) / TPB : 0;
    const int ib = nifc > 0 ? (nifc + TPB - 1) / TPB : 0;
    // batch-wide spans (every field/scratch buffer is (B, ng) / (B, E*m[*2]))
    const size_t ngB = static_cast<size_t>(B) * ng;
    const size_t ngb = ngB * sizeof(float);
    const size_t Em = static_cast<size_t>(E) * m;
    const dim3 gE(E, B), gN(nb, B), gJ(rj, B), gD(db, B), gP(pb, B), gI(ib, B);

    const dim3 gBN(bs ? (nbn + TPB - 1) / TPB : 0, B);
    const dim3 gBE(bs ? (nbe * m + TPB - 1) / TPB : 0, B);
    const size_t em1b = static_cast<size_t>(B) * Em * sizeof(float);
    const size_t em2b = static_cast<size_t>(B) * Em * 2 * sizeof(float);
    if (bs) {   // seed: the final two leapfrog levels of chi and u
        cudaMemcpyAsync(chi_hi, in.fin_chi.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(chi_lo, in.fin_chi_prev.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_hi_x, in.fin_ux.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_hi_z, in.fin_uz.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_lo_x, in.fin_ux_prev.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(u_lo_z, in.fin_uz_prev.data_ptr<float>(), ngb, cudaMemcpyDeviceToDevice, stream);
    }

    for (int mm = nt; mm >= 1; --mm) {
        const int k = mm - 1;
        if (bs) {
            // Re-anchor the reverse march on the forward's checkpoint: the
            // coupled march accumulates error the acoustic/elastic ones do not
            // (see AcousticElasticSEM2D), so the two leapfrog levels of each
            // field are reset every ckpt_every steps.
            if (Kck > 0 && (k + 1) % Kck == 0) {
                const int slot = (k + 1) / Kck - 1;
                if (slot < nck) {
                    const size_t co = static_cast<size_t>(slot) * ngB;   // (nc, B, ng)
                    cudaMemcpyAsync(chi_hi, in.ck_chi.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(chi_lo, in.ck_chi_prev.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(u_hi_x, in.ck_ux.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(u_hi_z, in.ck_uz.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(u_lo_x, in.ck_ux_prev.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(u_lo_z, in.ck_uz_prev.data_ptr<float>() + co, ngb, cudaMemcpyDeviceToDevice, stream);
                }
            }
            // rebuild the acoustic memories at level k: zero outside the band
            // (exact there), band values inside. state^k = ring row k-1.
            const bool r4c2 = (r_pz == r_px + ngB) && (r_zx == r_px + 2 * ngB)
                           && (r_zz == r_px + 3 * ngB);
            if (r4c2) {
                cudaMemsetAsync(r_px, 0, 4 * ngb, stream);
            } else {
                cudaMemsetAsync(r_px, 0, ngb, stream);
                cudaMemsetAsync(r_pz, 0, ngb, stream);
                cudaMemsetAsync(r_zx, 0, ngb, stream);
                cudaMemsetAsync(r_zz, 0, ngb, stream);
            }
            if (k >= 1 && nband > 0) {
                sem2d_band_load(h_px, k - 1, band_g, r_px, nband, TPB, stream, B, ng);
                sem2d_band_load(h_pz, k - 1, band_g, r_pz, nband, TPB, stream, B, ng);
                sem2d_band_load(h_zx, k - 1, band_g, r_zx, nband, TPB, stream, B, ng);
                sem2d_band_load(h_zz, k - 1, band_g, r_zz, nband, TPB, stream, B, ng);
            }
            if (k == 0) {
                // step 0's element histories are identically zero (u starts
                // at rest); there is no march at k == 0, so clear the scratch
                // the element transpose will read.
                cudaMemsetAsync(r_sdxx, 0, em1b, stream);
                cudaMemsetAsync(r_sdzx, 0, em1b, stream);
                cudaMemsetAsync(r_sdxz, 0, em1b, stream);
                cudaMemsetAsync(r_sdzz, 0, em1b, stream);
                cudaMemsetAsync(r_apx, 0, em1b, stream);
                cudaMemsetAsync(r_apz, 0, em1b, stream);
            }
        }
        // ---- loop top: record adjoints into the carried lambdas ---- //
        if (nrec > 0) {
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(g_schi, rec_G, rec_W, gA, k, nt, nrec, m, ng);
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(g_sx, rec_G, rec_W, lam1_x, k, nt, nrec, m, ng);
            sem2d_adjoint_source_b<<<gJ, TPB, 0, stream>>>(g_sz, rec_G, rec_W, lam1_z, k, nt, nrec, m, ng);
        }
        if (kdir > 0) {
            if (cp) {   // gA + the four memory adjoints in ONE launch (was 5)
                sem2d_dirichlet_fill5<<<gD, TPB, 0, stream>>>(
                    gA, lpx, lpz, lzx, lzz,
                    nullptr, nullptr, nullptr, nullptr, nullptr, didx, kdir, ng);
            } else {
                sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(gA, didx, kdir);
            }
        }
        // ---- solid: g = Pin(lam1), y = -dt2 invM g, lambda(force) = -y ---- //
        cudaMemcpyAsync(g_x, lam1_x, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(g_z, lam1_z, ngb, cudaMemcpyDeviceToDevice, stream);
        if (kpin > 0) {
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(g_x, pin_idx, kpin, ng);
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(g_z, pin_idx, kpin, ng);
        }
        sem2d_scale2_b<<<gN, TPB, 0, stream>>>(el_iM, g_x, -dt2, y_x, ng, 0);
        sem2d_scale2_b<<<gN, TPB, 0, stream>>>(el_iM, g_z, -dt2, y_z, ng, 0);

        // ---- fluid accel adjoint: lchidd = Dir(dt2 gA + Ifc_ac(y_el)) ---- //
        cudaMemsetAsync(lchidd, 0, ngb, stream);
        if (nifc > 0)     // -Ifc_ac(lambda_f) with lambda_f = -y  ->  +Ifc_ac(y)
            sem2d_ifc_ac<<<gI, TPB, 0, stream>>>(y_x, y_z, ifc_g, ifc_w, ifc_nx, ifc_nz, lchidd, nifc, ng);
        sem2d_fs_axpy<<<gN, TPB, 0, stream>>>(gA, dt2, lchidd, ng);
        if (kdir > 0)
            sem2d_dirichlet_fill_b<<<gD, TPB, 0, stream>>>(lchidd, didx, kdir, ng);

        // ---- recompute s_ac^k (from u^k) and the fluid stiffness/CPML ---- //
        const float* uxn = k >= 1 ? (bs ? u_lo_x : sux + static_cast<size_t>(k - 1) * ngB) : nullptr;
        const float* uzn = k >= 1 ? (bs ? u_lo_z : suz + static_cast<size_t>(k - 1) * ngB) : nullptr;
        cudaMemsetAsync(s_ac, 0, ngb, stream);
        if (nifc > 0 && k >= 1)
            sem2d_ifc_ac<<<gI, TPB, 0, stream>>>(uxn, uzn, ifc_g, ifc_w, ifc_nx, ifc_nz, s_ac, nifc, ng);
        if (cp) {
            sem2d_scale2_b<<<gN, TPB, 0, stream>>>(ac_kp, lchidd, 1.f, gw, ng, 0);
            if (k >= 1) {
                const size_t o = static_cast<size_t>(k - 1) * ngB;   // (nt, B, ng)
                // fused gather (see forward.cu): drops the two scale2 launches
                cudaMemsetAsync(gxp, 0, ngb, stream);  cudaMemsetAsync(gzp, 0, ngb, stream);
                cudaMemsetAsync(kxp, 0, ngb, stream);  cudaMemsetAsync(kzp, 0, ngb, stream);
                cudaMemsetAsync(gaxp, 0, ngb, stream); cudaMemsetAsync(gazp, 0, ngb, stream);
                const float* chi_k = bs ? chi_lo : sch + o;
                const float* psx_k = bs ? r_px : spx + o;
                const float* psz_k = bs ? r_pz : spz + o;
                const float* zex_k = bs ? r_zx : szx + o;
                const float* zez_k = bs ? r_zz : szz + o;
                sem2d_acoustic_cpml_grads_f<<<gE, m, sh_ac, stream>>>(
                    chi_k, psx_k, psz_k, ax, az, ibool, D, ac_ir, w1d,
                    ixx, ixz, iex, iez, ac_me,
                    gxp, gzp, kxp, kzp, gaxp, gazp, ngll, E, ng, 0);
                sem2d_acoustic_cpml_step_T<false><<<gN, TPB, 0, stream>>>(
                    gxp, gzp, kxp, kzp, gaxp, gazp, ac_iMg, ax, bx, dbxdx, az, bz, dbzdz,
                    zex_k, zez_k, gw, lpx, lpz, lzx, lzz,
                    gbxp, gbzp, gbkx, gbkz, gbax, gbaz, w_sum, gAcIMg, ng, 0);
                cudaMemsetAsync(lchi_G, 0, ngb, stream);
                cudaMemsetAsync(lapx, 0, ngb, stream);
                cudaMemsetAsync(lapz, 0, ngb, stream);
                sem2d_acoustic_cpml_grads_T<<<gE, m, sh_acT, stream>>>(
                    chi_k, gbxp, gbzp, gbkx, gbkz, gbax, gbaz, ibool, D, ac_ir, w1d,
                    ixx, ixz, iex, iez, ac_me, lchi_G, lapx, lapz, gAcIR, ngll, E,
                    ng, 0);
            } else {
                cudaMemsetAsync(w_sum, 0, ngb, stream);
                cudaMemsetAsync(lchi_G, 0, ngb, stream);
                cudaMemsetAsync(lapx, 0, ngb, stream);
                cudaMemsetAsync(lapz, 0, ngb, stream);
            }
            sem2d_fs_ac_gradMK<<<gN, TPB, 0, stream>>>(
                lchidd, w_sum, nullptr, pat_chi, wav, k, s_ac, gAcKp, gAcIM, ng, wst);
        } else {
            cudaMemsetAsync(Kchi, 0, ngb, stream);
            if (k >= 1)
                sem2d_acoustic_stiffness<<<E, m, sh_ac, stream>>>(
                    sch + static_cast<size_t>(k - 1) * ng, ibool, D, ac_ir, wJ,
                    ixx, ixz, iex, iez, ac_me, Kchi, ngll, E);
            // K^T path: lchi_G = K(yf), yf = -invM o lchidd; imaging for inv_rho
            sem2d_scale2<<<nb, TPB, 0, stream>>>(ac_iM, lchidd, -1.f, yf, ng);
            cudaMemsetAsync(lchi_G, 0, ngb, stream);
            sem2d_acoustic_stiffness<<<E, m, sh_ac, stream>>>(
                yf, ibool, D, ac_ir, wJ, ixx, ixz, iex, iez, ac_me, lchi_G, ngll, E);
            if (k >= 1)
                sem2d_imaging_invrho<<<E, m, sh_im2, stream>>>(
                    sch + static_cast<size_t>(k - 1) * ng, yf, ibool, D, wJ,
                    ixx, ixz, iex, iez, ac_me, gAcIR, ngll, E);
            sem2d_fs_ac_gradMK<<<nb, TPB, 0, stream>>>(
                lchidd, nullptr, Kchi, pat_chi, wav, k, s_ac, nullptr, gAcIM, ng, 0);
        }
        sem2d_accum_gradW_b<<<gN, TPB, 0, stream>>>(pat_chi, ac_iM, lchidd, 1.f, k, gWav, ng, nt, 0);

        // ---- boundary saving: the coupled reverse march. Recompute forward
        // step k's accelerations from the reconstructed state (memories from
        // the rings, zero in the reversible interior where they truly vanish)
        // and leapfrog both fields back one level. The coupling is recomputed
        // at the SAME reconstructed time level: chi_dd^k feeds the solid
        // force, u^k fed s_ac^k above -- exactly the forward's ordering. ---- //
        if (bs && k >= 1) {
            // chi_dd^k = kappa w_sum^k + invM (pat wav[k] + s_ac^k), dirichlet
            sem2d_fs_chidd<<<gN, TPB, 0, stream>>>(
                ac_kp, w_sum, ac_iM, pat_chi, wav, k, s_ac, rec_chi_dd, ng, wst);
            if (kdir > 0)
                sem2d_dirichlet_fill_b<<<gD, TPB, 0, stream>>>(rec_chi_dd, didx, kdir, ng);
            // solid force^k = pat wav[k] + Ifc_el(chi_dd^k)
            sem2d_fs_pat_wav<<<gN, TPB, 0, stream>>>(pat_x, wav, k, r_f_x, ng, wst);
            sem2d_fs_pat_wav<<<gN, TPB, 0, stream>>>(pat_z, wav, k, r_f_z, ng, wst);
            if (nifc > 0)
                sem2d_ifc_el<<<gI, TPB, 0, stream>>>(rec_chi_dd, ifc_g, ifc_w, ifc_nx, ifc_nz,
                                                     r_f_x, r_f_z, nifc, ng);
            // solid: recompute Acc^k + this step's element histories with the
            // memories zeroed (exact outside the band; the band histories are
            // then overwritten from the ring), u^{n+1} standing in as "prev"
            const bool r6c = (r_rxz == r_rxx + static_cast<size_t>(B) * Em * 2)
                          && (r_rzx == r_rxx + 2 * static_cast<size_t>(B) * Em * 2)
                          && (r_rzz == r_rxx + 3 * static_cast<size_t>(B) * Em * 2)
                          && (r_rdx == r_rxx + 4 * static_cast<size_t>(B) * Em * 2)
                          && (r_rdz == r_rxx + 5 * static_cast<size_t>(B) * Em * 2);
            if (r6c) {
                cudaMemsetAsync(r_rxx, 0, 6 * em2b, stream);
            } else {
                cudaMemsetAsync(r_rxx, 0, em2b, stream);
                cudaMemsetAsync(r_rxz, 0, em2b, stream);
                cudaMemsetAsync(r_rzx, 0, em2b, stream);
                cudaMemsetAsync(r_rzz, 0, em2b, stream);
                cudaMemsetAsync(r_rdx, 0, em2b, stream);
                cudaMemsetAsync(r_rdz, 0, em2b, stream);
            }
            cudaMemsetAsync(r_Acc_x, 0, ngb, stream);
            cudaMemsetAsync(r_Acc_z, 0, ngb, stream);
            sem2d_elastic_cpml_element<<<gE, m, sh_el, stream>>>(
                u_lo_x, u_lo_z, u_hi_x, u_hi_z, ibool, D, el_lam, el_mu, el_l2m,
                wJ, el_wrJ, ixx, ixz, iex, iez, el_me, cf,
                r_rxx, r_rxz, r_rzx, r_rzz, r_rdx, r_rdz,
                inv_dt, r_Acc_x, r_Acc_z,
                r_sdxx, r_sdzx, r_sdxz, r_sdzz, r_apx, r_apz, 0L, ngll, E, ng);
            sem2d_elastic_newmark_force<<<gN, TPB, 0, stream>>>(
                u_lo_x, u_lo_z, u_hi_x, u_hi_z, r_Acc_x, r_Acc_z, r_f_x, r_f_z,
                el_iM, dt2, u_km1_x, u_km1_z, ng);
            if (kpin > 0) {
                sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(u_km1_x, pin_idx, kpin, ng);
                sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(u_km1_z, pin_idx, kpin, ng);
            }
            if (k >= 2) {   // band_ux[n] holds u^{n+1} -> u^{k-1} = row k-2
                const size_t no = static_cast<size_t>(k - 2) * B * nbn;
                sem2d_band_load(
                    sem_hist(in.band_ux, in.band_scale, bkind, 5, nt, B),
                    k - 2, bnode, u_km1_x, nbn, TPB, stream, B, ng);
                sem2d_band_load(
                    sem_hist(in.band_uz, in.band_scale, bkind, 6, nt, B),
                    k - 2, bnode, u_km1_z, nbn, TPB, stream, B, ng);
            }
            float* escr[6] = {r_sdxx, r_sdzx, r_sdxz, r_sdzz, r_apx, r_apz};
            const Buf* erg[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                           &in.band_sd_xz, &in.band_sd_zz,
                                           &in.band_apx, &in.band_apz};
            for (int q = 0; q < 6; ++q)
                sem2d_band_load_elem(
                    sem_hist(*erg[q], in.band_scale_e, in.band_kind_e, q, nt, B, nbe),
                    k, belem, escr[q], nbe, m, Em, TPB, stream, B);
            // fluid: chi^{k-1} = 2 chi^k - chi^{k+1} + dt2 chi_dd^k, dirichlet,
            // then the exact band restore
            if (k == 1) {
                cudaMemsetAsync(chi_km1, 0, ngb, stream);   // chi^0 = 0
            } else {
                sem2d_fs_chi_leapfrog<<<gN, TPB, 0, stream>>>(
                    chi_lo, chi_hi, rec_chi_dd, dt2, chi_km1, ng);
                if (kdir > 0)
                    sem2d_dirichlet_fill_b<<<gD, TPB, 0, stream>>>(chi_km1, didx, kdir, ng);
                if (nband > 0)
                    sem2d_band_load(h_chi, k - 2, band_g, chi_km1, nband, TPB, stream, B, ng);
            }
        }

        // ---- solid element transpose + coupling scatter into sc_now ---- //
        cudaMemsetAsync(sc_now_x, 0, ngb, stream);
        cudaMemsetAsync(sc_now_z, 0, ngb, stream);
        cudaMemsetAsync(scp_a_x, 0, ngb, stream);
        cudaMemsetAsync(scp_a_z, 0, ngb, stream);
        if (cp) {
            const size_t so = bs ? 0 : static_cast<size_t>(k) * B * Em;
            sem2d_elastic_cpml_element_T<<<gE, m, sh_el, stream>>>(
                y_x, y_z, uxn, uzn, ibool, D, el_lam, el_mu, el_l2m, wJ, el_wrJ,
                ixx, ixz, iex, iez, el_me, cf,
                (bs ? r_sdxx : sdxx + so), (bs ? r_sdzx : sdzx + so),
                (bs ? r_sdxz : sdxz + so), (bs ? r_sdzz : sdzz + so),
                (bs ? r_apx : sapx + so), (bs ? r_apz : sapz + so),
                mu_xx, mu_zx, mu_xz, mu_zz, mu_dx, mu_dz, inv_dt,
                sc_now_x, sc_now_z, scp_a_x, scp_a_z,
                gElLam, gElMu, gElL2m, gElWrJ, ngll, E, ng);
        } else {
            // bare solid: sc_now = -dt2 K_el(invM o g); imaging for lam/mu/l2m
            sem2d_scale2<<<nb, TPB, 0, stream>>>(el_iM, g_x, 1.f, tmp_x, ng);
            sem2d_scale2<<<nb, TPB, 0, stream>>>(el_iM, g_z, 1.f, tmp_z, ng);
            cudaMemsetAsync(adj_x, 0, ngb, stream);
            cudaMemsetAsync(adj_z, 0, ngb, stream);
            sem2d_elastic_stiffness<<<E, m, sh_el, stream>>>(
                tmp_x, tmp_z, ibool, D, el_lam, el_mu, el_l2m, wJ, ixx, ixz, iex, iez,
                el_me, adj_x, adj_z, ngll, E);
            sem2d_fs_axpy<<<nb, TPB, 0, stream>>>(adj_x, -dt2, sc_now_x, ng);
            sem2d_fs_axpy<<<nb, TPB, 0, stream>>>(adj_z, -dt2, sc_now_z, ng);
            if (k >= 1)
                sem2d_elastic_imaging<<<E, m, sh_im4, stream>>>(
                    uxn, uzn, y_x, y_z, ibool, D, wJ, ixx, ixz, iex, iez, el_me,
                    gElL2m, gElLam, gElMu, ngll, E);
        }
        // coupling: sc_now += Ifc_el(-invM_ac o lchidd)  (Ifc_ac^T)
        // In the bare branch the K^T path above already left exactly this in yf
        // and nothing since writes yf / lchidd / ac_iM, so only the CPML branch
        // has to recompute it.
        if (cp)
            sem2d_scale2_b<<<gN, TPB, 0, stream>>>(ac_iM, lchidd, -1.f, yf, ng, 0);
        if (nifc > 0)
            sem2d_ifc_el<<<gI, TPB, 0, stream>>>(yf, ifc_g, ifc_w, ifc_nx, ifc_nz,
                                                 sc_now_x, sc_now_z, nifc, ng);

        // ---- solid grads (identity + wavelet) ---- //
        const float* uxm1 = k >= 2 ? (bs ? u_km1_x : sux + static_cast<size_t>(k - 2) * ngB) : nullptr;
        const float* uzm1 = k >= 2 ? (bs ? u_km1_z : suz + static_cast<size_t>(k - 2) * ngB) : nullptr;
        sem2d_elastic_cpml_gradM<<<gN, TPB, 0, stream>>>(
            g_x, g_z,
            bs ? u_hi_x : sux + static_cast<size_t>(k) * ngB,
            bs ? u_hi_z : suz + static_cast<size_t>(k) * ngB,
            uxn, uzn, uxm1, uzm1, el_iM, gElIM, ng);
        sem2d_elastic_accum_gradW<<<gN, TPB, 0, stream>>>(
            pat_x, pat_z, el_iM, g_x, g_z, dt2, k, gWav, ng, nt);

        // ---- combines ---- //
        sem2d_elastic_cpml_combine<<<gN, TPB, 0, stream>>>(
            g_x, gprev_x, zeros, sc_now_x, scp_b_x, lamn_x, ng);
        sem2d_elastic_cpml_combine<<<gN, TPB, 0, stream>>>(
            g_z, gprev_z, zeros, sc_now_z, scp_b_z, lamn_z, ng);
        if (cp)
            sem2d_fs_ac_combine_cpml<<<gN, TPB, 0, stream>>>(
                gA, gB, lchi_G, lpx, lpz, lzx, lzz, lapx, lapz, gw, ax, az,
                gchi_k, lpxn, lpzn, lzxn, lzzn, ng);
        else
            sem2d_fs_combine3<<<nb, TPB, 0, stream>>>(gA, gB, lchi_G, gchi_k, ng);
        sem2d_scale1<<<gN, TPB, 0, stream>>>(gA, -1.f, gB, ng);

        // ---- rotate ---- //
        float* t;
        t = gA; gA = gchi_k; gchi_k = t;
        if (cp) {
            t = lpx; lpx = lpxn; lpxn = t;
            t = lpz; lpz = lpzn; lpzn = t;
            t = lzx; lzx = lzxn; lzxn = t;
            t = lzz; lzz = lzzn; lzzn = t;
        }
        t = lam1_x; lam1_x = lamn_x; lamn_x = t;
        t = lam1_z; lam1_z = lamn_z; lamn_z = t;
        t = gprev_x; gprev_x = g_x; g_x = t;
        t = gprev_z; gprev_z = g_z; g_z = t;
        t = scp_b_x; scp_b_x = scp_a_x; scp_a_x = t;
        t = scp_b_z; scp_b_z = scp_a_z; scp_a_z = t;
        if (bs && k >= 1) {   // (chi^{k+1}, chi^k) -> (chi^k, chi^{k-1}); same for u
            t = chi_hi; chi_hi = chi_lo; chi_lo = chi_km1; chi_km1 = t;
            t = u_hi_x; u_hi_x = u_lo_x; u_lo_x = u_km1_x; u_km1_x = t;
            t = u_hi_z; u_hi_z = u_lo_z; u_lo_z = u_km1_z; u_km1_z = t;
        }
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem2d

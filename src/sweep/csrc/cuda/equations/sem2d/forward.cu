// sweep.sem CUDA (impl='c') backend — 2-D acoustic-potential SEM forward.
//
// Drives the full NSTEP Newmark loop on device over Python-allocated buffers,
// reproducing AcousticPotentialSEM2D's eager `step` (bare, no CPML) in fp32.
// Ported from SPECFEM2D's device kernels but hosted in sweep.sem's native
// (E, ngll, ngll) / (a=eta, b=xi) layout; see kernels.cuh.
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "sem2d.h"
#include "kernels.cuh"

namespace sem2d {



void acoustic_forward(const SemForwardInputCore& in) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();

    const int E   = in.n_elements;
    const int ng  = in.n_global;
    const int ngll = in.ngll;
    const int m    = ngll * ngll;         // GLL nodes per element
    const int nrec = in.n_rec;
    const int nt   = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());

    // constant device pointers
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
    float* seis          = nrec > 0 ? in.seis.data_ptr<float>() : nullptr;

    // rotating wavefield buffers (bind Python-allocated tensors; never cudaMalloc)
    float* chi_now  = in.chi.data_ptr<float>();
    float* chi_prev = in.chi_prev.data_ptr<float>();
    float* chi_next = in.chi_next.data_ptr<float>();
    float* K        = in.K.data_ptr<float>();
    const bool save = in.save_wavefield.numel() > 0;
    float* save_wf  = save ? in.save_wavefield.data_ptr<float>() : nullptr;

    const size_t shmem = static_cast<size_t>(3 * m) * sizeof(float);
    const int TPB = 256;
    const int nm_blocks  = (ng + TPB - 1) / TPB;
    const int dir_blocks = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const int rec_blocks = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;

    for (int n = 0; n < nt; ++n) {
        cudaMemsetAsync(K, 0, static_cast<size_t>(ng) * sizeof(float), stream);
        sem2d_acoustic_stiffness<<<E, m, shmem, stream>>>(
            chi_now, ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, mask_e, K, ngll, E);
        sem2d_acoustic_newmark<<<nm_blocks, TPB, 0, stream>>>(
            chi_now, chi_prev, K, pattern, inv_M, wavelet, n, dt2, chi_next, ng);
        if (kdir > 0)
            sem2d_dirichlet_fill<<<dir_blocks, TPB, 0, stream>>>(chi_next, didx, kdir);
        if (nrec > 0)
            sem2d_record<<<rec_blocks, TPB, 0, stream>>>(
                chi_next, rec_G, rec_W, seis, n, nt, nrec, m, ng);
        if (save)   // save-all forward wavefield: save_wf[n] = chi_next = u^{n+1}
            cudaMemcpyAsync(save_wf + static_cast<size_t>(n) * ng, chi_next,
                            static_cast<size_t>(ng) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        // 3-way rotation: prev <- now, now <- next, next <- (old prev, reused scratch)
        float* tmp = chi_prev;
        chi_prev = chi_now;
        chi_now  = chi_next;
        chi_next = tmp;
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- elastic forward (2-component, bare) ------------------------------------
void elastic_forward(const SemElasticForwardInputCore& in) {
    sweep::DeviceGuard guard(in.ux.device_index());
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
    float* seis_x = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* seis_z = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;

    float* ux = in.ux.data_ptr<float>();
    float* uz = in.uz.data_ptr<float>();
    float* ux_prev = in.ux_prev.data_ptr<float>();
    float* uz_prev = in.uz_prev.data_ptr<float>();
    float* ux_next = in.ux_next.data_ptr<float>();
    float* uz_next = in.uz_next.data_ptr<float>();
    float* Kux = in.Kux.data_ptr<float>();
    float* Kuz = in.Kuz.data_ptr<float>();
    const bool save = in.save_ux.numel() > 0;
    float* sux = save ? in.save_ux.data_ptr<float>() : nullptr;
    float* suz = save ? in.save_uz.data_ptr<float>() : nullptr;

    const size_t shmem = static_cast<size_t>(6 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int n = 0; n < nt; ++n) {
        cudaMemsetAsync(Kux, 0, ngb, stream);
        cudaMemsetAsync(Kuz, 0, ngb, stream);
        sem2d_elastic_stiffness<<<E, m, shmem, stream>>>(
            ux, uz, ibool, D, lam, mu, lam2mu, wJ, ixx, ixz, iex, iez, mask_e, Kux, Kuz, ngll, E);
        sem2d_elastic_newmark<<<nb, TPB, 0, stream>>>(
            ux, uz, ux_prev, uz_prev, Kux, Kuz, pat_x, pat_z, inv_M, wavelet,
            n, dt2, ux_next, uz_next, ng, 0);
        if (nrec > 0) {
            sem2d_record<<<rb, TPB, 0, stream>>>(ux_next, rec_G, rec_W, seis_x, n, nt, nrec, m, ng);
            sem2d_record<<<rb, TPB, 0, stream>>>(uz_next, rec_G, rec_W, seis_z, n, nt, nrec, m, ng);
        }
        if (save) {
            cudaMemcpyAsync(sux + static_cast<size_t>(n) * ng, ux_next, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(suz + static_cast<size_t>(n) * ng, uz_next, ngb, cudaMemcpyDeviceToDevice, stream);
        }
        float* tx = ux_prev; ux_prev = ux; ux = ux_next; ux_next = tx;
        float* tz = uz_prev; uz_prev = uz; uz = uz_next; uz_next = tz;
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- acoustic-potential forward WITH C-PML (P3a) ----------------------------
void acoustic_cpml_forward(const SemAcousticCpmlForwardInputCore& in) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    // shot batch (see sem_types.h). Every field/scratch buffer is (B, ng),
    // materials are broadcast (Bm == 1) or per-shot (Bm == B).
    const int B = in.B, Bm = in.Bm;
    SWEEP_CHECK(B >= 1 && (Bm == 1 || Bm == B),
                "acoustic_cpml_forward: B=", B, " Bm=", Bm);
    SWEEP_CHECK(in.chi.numel() == static_cast<int64_t>(B) * ng,
                "acoustic_cpml_forward: chi holds ", in.chi.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(in.src_pattern.numel() == static_cast<int64_t>(B) * ng,
                "acoustic_cpml_forward: src_pattern must be (B, n_global)");
    SWEEP_CHECK(in.wavelet.numel() == nt
                || in.wavelet.numel() == static_cast<int64_t>(B) * nt,
                "acoustic_cpml_forward: wavelet must be (nt,) or (B, nt)");
    SWEEP_CHECK(in.bs == 0
                || in.band_chi.numel()
                   == static_cast<int64_t>(in.nt) * B * in.band_idx.numel(),
                "acoustic_cpml_forward: band histories must be (nt, B, nband)");
    SWEEP_CHECK(in.bs == 0 || in.band_kind != 2
                || in.band_scale.numel() == static_cast<int64_t>(5) * in.nt * B,
                "acoustic_cpml_forward: int8 band_scale must be (5, nt, B)");
    const int msn = Bm > 1 ? ng : 0;            // node materials
    const int msr = Bm > 1 ? m * E : 0;         // element materials (inv_rho)
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
    float* seis = nrec > 0 ? in.seis.data_ptr<float>() : nullptr;

    float* chi = in.chi.data_ptr<float>();
    float* chi_prev = in.chi_prev.data_ptr<float>();
    float* chi_next = in.chi_next.data_ptr<float>();
    float* psi_x = in.psi_x.data_ptr<float>(); float* psi_z = in.psi_z.data_ptr<float>();
    float* zeta_x = in.zeta_x.data_ptr<float>(); float* zeta_z = in.zeta_z.data_ptr<float>();
    float* psi_x_n = in.psi_x_n.data_ptr<float>(); float* psi_z_n = in.psi_z_n.data_ptr<float>();
    float* zeta_x_n = in.zeta_x_n.data_ptr<float>(); float* zeta_z_n = in.zeta_z_n.data_ptr<float>();
    // apsi_x / apsi_z / w_sum: struct fields kept for the ABI, never read or
    // written by this entry (the fused kernels fold a*psi and w_sum in
    // registers) -- proven by NaN-poisoning them.
    // The drivers hand empty tensors.
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    // save-all: the chi tape always; then either the four state tapes (the
    // old path) or the w_sum tape (sem_types.h) -- keyed off numel, nothing else
    const bool save = in.save_wavefield.numel() > 0;
    const bool ssave = save && in.save_psi_x.numel() > 0;
    const bool wsave = in.save_w_sum.numel() > 0;
    SWEEP_CHECK(!save || ssave || wsave,
                "acoustic_cpml_forward: save-all needs the psi/zeta tapes or "
                "the w_sum tape");
    float* save_wf = save ? in.save_wavefield.data_ptr<float>() : nullptr;
    float* save_px = ssave ? in.save_psi_x.data_ptr<float>() : nullptr;
    float* save_pz = ssave ? in.save_psi_z.data_ptr<float>() : nullptr;
    float* save_zx = ssave ? in.save_zeta_x.data_ptr<float>() : nullptr;
    float* save_zz = ssave ? in.save_zeta_z.data_ptr<float>() : nullptr;
    float* save_ws = wsave ? in.save_w_sum.data_ptr<float>() : nullptr;
    const bool bs = in.bs != 0;
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
    const int bb = nband > 0 ? (nband + 255) / 256 : 0;

    const size_t shmem = static_cast<size_t>(7 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const size_t ngB = static_cast<size_t>(B) * ng;      // per-buffer batch span
    const size_t ngb = ngB * sizeof(float);
    const dim3 gE(E, B), gN(nb, B), gR(rb, B), gD(db, B);

    // one memset for all 6 grad fields when Python allocated them contiguously
    const bool contig6 = (gz == gx + ngB && kx == gx + 2 * ngB && kz == gx + 3 * ngB
                          && gax == gx + 4 * ngB && gaz == gx + 5 * ngB);

    for (int n = 0; n < nt; ++n) {
        if (contig6) {
            cudaMemsetAsync(gx, 0, 6 * ngb, stream);
        } else {
            cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
            cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
            cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
        }
        sem2d_acoustic_cpml_grads_f<<<gE, m, shmem, stream>>>(
            chi, psi_x, psi_z, ax, az, ibool, D, inv_rho, w1d, ixx, ixz, iex, iez,
            mask_e, gx, gz, kx, kz, gax, gaz, ngll, E, ng, msr);
        const size_t o = static_cast<size_t>(n) * ngB;   // (nt, B, ng) histories
        SEM2D_LAUNCH_WS(wsave, sem2d_acoustic_cpml_step_newmark_f,
            gN, TPB, 0, stream,
            gx, gz, kx, kz, gax, gaz, inv_Mg, ax, bx, dbxdx, az, bz, dbzdz,
            psi_x, psi_z, zeta_x, zeta_z, chi, chi_prev, kappa, inv_M,
            pattern, wavelet, n, dt2,
            chi_next, psi_x_n, psi_z_n, zeta_x_n, zeta_z_n,
            save ? save_wf + o : nullptr, ssave ? save_px + o : nullptr,
            ssave ? save_pz + o : nullptr, ssave ? save_zx + o : nullptr,
            ssave ? save_zz + o : nullptr,
            wsave ? save_ws + o : nullptr,          // row n = w_sum^n
            ng, msn, ws);
        if (kdir > 0)
            sem2d_dirichlet_fill5<<<gD, TPB, 0, stream>>>(
                chi_next, psi_x_n, psi_z_n, zeta_x_n, zeta_z_n,
                save ? save_wf + o : nullptr, ssave ? save_px + o : nullptr,
                ssave ? save_pz + o : nullptr, ssave ? save_zx + o : nullptr,
                ssave ? save_zz + o : nullptr, didx, kdir, ng);
        if (nrec > 0)
            sem2d_record<<<gR, TPB, 0, stream>>>(chi_next, rec_G, rec_W, seis, n, nt, nrec, m, ng);
        if (bs && nband > 0) {
            sem2d_band_save(chi_next, band_g, h_chi, n, nband, TPB, stream, B, ng);
            sem2d_band_save(psi_x_n, bi_x.idx, h_px, n, bi_x.n, TPB, stream, B, ng);
            sem2d_band_save(psi_z_n, bi_z.idx, h_pz, n, bi_z.n, TPB, stream, B, ng);
            sem2d_band_save(zeta_x_n, bi_x.idx, h_zx, n, bi_x.n, TPB, stream, B, ng);
            sem2d_band_save(zeta_z_n, bi_z.idx, h_zz, n, bi_z.n, TPB, stream, B, ng);
        }
        // rotate chi (3-way) and swap psi/zeta (2-way)
        float* t = chi_prev; chi_prev = chi; chi = chi_next; chi_next = t;
        float* a; a = psi_x; psi_x = psi_x_n; psi_x_n = a;
        a = psi_z; psi_z = psi_z_n; psi_z_n = a;
        a = zeta_x; zeta_x = zeta_x_n; zeta_x_n = a;
        a = zeta_z; zeta_z = zeta_z_n; zeta_z_n = a;
    }
    if (bs) {   // final two chi levels seed the backward's reverse march
        cudaMemcpyAsync(in.fin_chi.data_ptr<float>(), chi, ngb,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_chi_prev.data_ptr<float>(), chi_prev, ngb,
                        cudaMemcpyDeviceToDevice, stream);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped forward (acoustic C-PML): ONE step, split at the exchange ----
// Domain decomposition runs the nt loop in PYTHON so the interface-column sum
// can ride torch.distributed between the two phases (the FD stepped pattern):
//   phase A (dd_grads): memset + the step's ONLY assembly (grads_f, 6 fields)
//   [caller sums the interface columns across ranks]
//   phase B (dd_step):  step_newmark_f + dirichlet + record + save-all row n
// Buffer rotation (chi/chi_prev/chi_next, psi/zeta pairs) is the caller's:
// the struct fields are re-pointed between calls. B == 1, no boundary saving
// (DD phase 1 keeps the save-all tape per rank).
void acoustic_cpml_dd_grads(const SemAcousticCpmlForwardInputCore& in) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    SWEEP_CHECK(in.B == 1, "acoustic_cpml_dd_grads: DD is single-shot");
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const bool contig6 = (gz == gx + ng && kx == gx + 2L * ng && kz == gx + 3L * ng
                          && gax == gx + 4L * ng && gaz == gx + 5L * ng);
    if (contig6) {
        cudaMemsetAsync(gx, 0, 6 * ngb, stream);
    } else {
        cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
        cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
        cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
    }
    const size_t shmem = static_cast<size_t>(7 * m) * sizeof(float);
    sem2d_acoustic_cpml_grads_f<<<E, m, shmem, stream>>>(
        in.chi.data_ptr<float>(), in.psi_x.data_ptr<float>(), in.psi_z.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.inv_rho.data_ptr<float>(), in.w1d.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.mask_e.data_ptr<float>(), gx, gz, kx, kz, gax, gaz, ngll, E, ng, 0);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void acoustic_cpml_dd_step(const SemAcousticCpmlForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int nrec = in.n_rec, nt = in.nt;
    const int m = ngll * ngll;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    SWEEP_CHECK(in.B == 1, "acoustic_cpml_dd_step: DD is single-shot");
    const bool save = in.save_wavefield.numel() > 0;
    const bool ssave = save && in.save_psi_x.numel() > 0;
    const bool wsave = in.save_w_sum.numel() > 0;
    const size_t o = static_cast<size_t>(n) * ng;
    float* swf = save ? in.save_wavefield.data_ptr<float>() + o : nullptr;
    float* spx = ssave ? in.save_psi_x.data_ptr<float>() + o : nullptr;
    float* spz = ssave ? in.save_psi_z.data_ptr<float>() + o : nullptr;
    float* szx = ssave ? in.save_zeta_x.data_ptr<float>() + o : nullptr;
    float* szz = ssave ? in.save_zeta_z.data_ptr<float>() + o : nullptr;
    // the w_sum tape row: written AFTER the caller's g6 exchange, so each
    // strip's row is its own post-exchange w_sum (a cut column agrees on both
    // strips -- the exchange is a commutative sum and this step is pointwise)
    float* sws = wsave ? in.save_w_sum.data_ptr<float>() + o : nullptr;
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    SEM2D_LAUNCH_WS(wsave, sem2d_acoustic_cpml_step_newmark_f,
        nb, TPB, 0, stream,
        in.gx.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.gax.data_ptr<float>(), in.gaz.data_ptr<float>(),
        in.inv_Mg.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(), in.dbx_dx.data_ptr<float>(),
        in.a_z.data_ptr<float>(), in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        in.psi_x.data_ptr<float>(), in.psi_z.data_ptr<float>(),
        in.zeta_x.data_ptr<float>(), in.zeta_z.data_ptr<float>(),
        in.chi.data_ptr<float>(), in.chi_prev.data_ptr<float>(),
        in.kappa_node.data_ptr<float>(), in.inv_M.data_ptr<float>(),
        in.src_pattern.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(n), dt2,
        in.chi_next.data_ptr<float>(),
        in.psi_x_n.data_ptr<float>(), in.psi_z_n.data_ptr<float>(),
        in.zeta_x_n.data_ptr<float>(), in.zeta_z_n.data_ptr<float>(),
        swf, spx, spz, szx, szz, sws, ng, 0, 0);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill5<<<db, TPB, 0, stream>>>(
            in.chi_next.data_ptr<float>(),
            in.psi_x_n.data_ptr<float>(), in.psi_z_n.data_ptr<float>(),
            in.zeta_x_n.data_ptr<float>(), in.zeta_z_n.data_ptr<float>(),
            swf, spx, spz, szx, szz,
            in.dirichlet_idx.data_ptr<int64_t>(), kdir, ng);
    }
    if (nrec > 0) {
        const int rb = (nrec + TPB - 1) / TPB;
        sem2d_record<<<rb, TPB, 0, stream>>>(
            in.chi_next.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.seis.data_ptr<float>(),
            static_cast<int>(n), nt, nrec, m, ng);
    }
    if (in.bs != 0) {
        // Boundary saving under DD: the ring is this STRIP's C-PML band
        // (internal cut columns carry no PML, so they are outside it and
        // the interior reverse march recomputes them exactly -- the same
        // property the single-domain path relies on).
        const int nband = static_cast<int>(in.band_idx.numel());
        if (nband > 0) {
            const long* bg = in.band_idx.data_ptr<int64_t>();
            const int bkind = in.band_kind;
            sem2d_band_save(in.chi_next.data_ptr<float>(), bg,
                            sem_hist(in.band_chi, in.band_scale, bkind, 0, nt),
                            n, nband, TPB, stream);
            const bool bsplit = in.band_split != 0;
            const SemBandIdx bi_x = sem_band_idx(bsplit, in.band_idx_x, bg, nband);
            const SemBandIdx bi_z = sem_band_idx(bsplit, in.band_idx_z, bg, nband);
            sem2d_band_save(in.psi_x_n.data_ptr<float>(), bi_x.idx,
                            sem_hist(in.band_psi_x, in.band_scale, bkind, 1, nt),
                            n, bi_x.n, TPB, stream);
            sem2d_band_save(in.psi_z_n.data_ptr<float>(), bi_z.idx,
                            sem_hist(in.band_psi_z, in.band_scale, bkind, 2, nt),
                            n, bi_z.n, TPB, stream);
            sem2d_band_save(in.zeta_x_n.data_ptr<float>(), bi_x.idx,
                            sem_hist(in.band_zeta_x, in.band_scale, bkind, 3, nt),
                            n, bi_x.n, TPB, stream);
            sem2d_band_save(in.zeta_z_n.data_ptr<float>(), bi_z.idx,
                            sem_hist(in.band_zeta_z, in.band_scale, bkind, 4, nt),
                            n, bi_z.n, TPB, stream);
        }
        if (n == nt - 1) {   // seed the backward's reverse march
            const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
            cudaMemcpyAsync(in.fin_chi.data_ptr<float>(),
                            in.chi_next.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(in.fin_chi_prev.data_ptr<float>(),
                            in.chi.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- elastic forward WITH SPECFEM2D-style 2nd-order C-PML (P3e) -------------
void elastic_cpml_forward(const SemElasticCpmlForwardInputCore& in) {
    sweep::DeviceGuard guard(in.ux.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    // shot batch (see sem_types.h): B shots, ONE shared material set
    const int B = in.B;
    SWEEP_CHECK(B >= 1, "elastic_cpml_forward: B=", B);
    SWEEP_CHECK(in.ux.numel() == static_cast<int64_t>(B) * ng,
                "elastic_cpml_forward: ux holds ", in.ux.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(in.pat_x.numel() == static_cast<int64_t>(B) * ng,
                "elastic_cpml_forward: pat_x must be (B, n_global)");
    SWEEP_CHECK(in.wavelet.numel() == nt
                || in.wavelet.numel() == static_cast<int64_t>(B) * nt,
                "elastic_cpml_forward: wavelet must be (nt,) or (B, nt)");
    const int wst = in.wavelet.numel() == nt ? 0 : nt;

    const int*  ibool = in.ibool.data_ptr<int32_t>();
    const float* D = in.D.data_ptr<float>();
    const float* lam = in.lam.data_ptr<float>();
    const float* mu = in.mu.data_ptr<float>();
    const float* lam2mu = in.lam2mu.data_ptr<float>();
    const float* wJ = in.wJ.data_ptr<float>();
    const float* wrhoJ = in.w_rho_J.data_ptr<float>();
    const float* ixx = in.ixx.data_ptr<float>(); const float* ixz = in.ixz.data_ptr<float>();
    const float* iex = in.iex.data_ptr<float>(); const float* iez = in.iez.data_ptr<float>();
    const float* mask_e = in.mask_e.data_ptr<float>();
    const float* inv_M = in.inv_M.data_ptr<float>();
    const float* pat_x = in.pat_x.data_ptr<float>();
    const float* pat_z = in.pat_z.data_ptr<float>();
    const float* wavelet = in.wavelet.data_ptr<float>();
    const long*  pin_idx = kpin > 0 ? in.outer_edge_idx.data_ptr<int64_t>() : nullptr;
    const long*  rec_G = nrec > 0 ? in.rec_G.data_ptr<int64_t>() : nullptr;
    const float* rec_W = nrec > 0 ? in.rec_W.data_ptr<float>() : nullptr;
    float* seis_x = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* seis_z = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;

    // the same packing the helper does -- these two entries kept a
    // hand-copied copy, so a field added to ElCpmlCoefs reached them
    // UNINITIALISED (an indeterminate device pointer, which is an
    // illegal access at the first launch that reads it)
    ElCpmlCoefs cf = sem2d_el_coefs(in);

    float* ux = in.ux.data_ptr<float>();       float* uz = in.uz.data_ptr<float>();
    float* ux_prev = in.ux_prev.data_ptr<float>(); float* uz_prev = in.uz_prev.data_ptr<float>();
    float* ux_next = in.ux_next.data_ptr<float>(); float* uz_next = in.uz_next.data_ptr<float>();
    float* Acc_x = in.Acc_x.data_ptr<float>(); float* Acc_z = in.Acc_z.data_ptr<float>();
    float* r_dux_dx = in.r_dux_dx.data_ptr<float>();
    float* r_dux_dz = in.r_dux_dz.data_ptr<float>();
    float* r_duz_dx = in.r_duz_dx.data_ptr<float>();
    float* r_duz_dz = in.r_duz_dz.data_ptr<float>();
    float* r_displ_x = in.r_displ_x.data_ptr<float>();
    float* r_displ_z = in.r_displ_z.data_ptr<float>();
    const bool save = in.save_ux.numel() > 0;
    const bool bs = in.bs != 0;
    float* sux = save ? in.save_ux.data_ptr<float>() : nullptr;
    float* suz = save ? in.save_uz.data_ptr<float>() : nullptr;
    // Element-kernel history sink: the (nt, E*m) save-all slice, or -- under
    // boundary saving -- one per-step scratch that is then band-gathered.
    float* sd_xx = save ? in.save_sd_xx.data_ptr<float>()
                        : (bs ? in.sd_scr_xx.data_ptr<float>() : nullptr);
    float* sd_zx = save ? in.save_sd_zx.data_ptr<float>()
                        : (bs ? in.sd_scr_zx.data_ptr<float>() : nullptr);
    float* sd_xz = save ? in.save_sd_xz.data_ptr<float>()
                        : (bs ? in.sd_scr_xz.data_ptr<float>() : nullptr);
    float* sd_zz = save ? in.save_sd_zz.data_ptr<float>()
                        : (bs ? in.sd_scr_zz.data_ptr<float>() : nullptr);
    float* s_apx = save ? in.save_apx.data_ptr<float>()
                        : (bs ? in.ap_scr_x.data_ptr<float>() : nullptr);
    float* s_apz = save ? in.save_apz.data_ptr<float>()
                        : (bs ? in.ap_scr_z.data_ptr<float>() : nullptr);
    const int nbn = bs ? static_cast<int>(in.band_node_idx.numel()) : 0;
    const int nbe = bs ? static_cast<int>(in.band_elem_idx.numel()) : 0;
    const long* bnode = bs ? in.band_node_idx.data_ptr<int64_t>() : nullptr;
    const long* belem = bs ? in.band_elem_idx.data_ptr<int64_t>() : nullptr;

    const size_t shmem = static_cast<size_t>(8 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const int pb = kpin > 0 ? (kpin + TPB - 1) / TPB : 0;
    // batch-wide spans: every field/memory buffer is (B, ng) / (B, E*m*2)
    const size_t ngB = static_cast<size_t>(B) * ng;
    const size_t ngb = ngB * sizeof(float);
    const long Em = static_cast<long>(E) * m;
    const dim3 gE(E, B), gN(nb, B), gR(rb, B), gP(pb, B);

    for (int n = 0; n < nt; ++n) {
        cudaMemsetAsync(Acc_x, 0, ngb, stream);
        cudaMemsetAsync(Acc_z, 0, ngb, stream);
        sem2d_elastic_cpml_element<<<gE, m, shmem, stream>>>(
            ux, uz, ux_prev, uz_prev, ibool, D, lam, mu, lam2mu, wJ, wrhoJ,
            ixx, ixz, iex, iez, mask_e, cf,
            r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z,
            inv_dt, Acc_x, Acc_z,
            sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz,
            bs ? 0L : static_cast<long>(n) * B * Em, ngll, E, ng);
        sem2d_elastic_newmark<<<gN, TPB, 0, stream>>>(
            ux, uz, ux_prev, uz_prev, Acc_x, Acc_z, pat_x, pat_z, inv_M, wavelet,
            n, dt2, ux_next, uz_next, ng, wst);
        if (kpin > 0) {   // rigid-wall PML termination: pin u=0 on the outer edge
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(ux_next, pin_idx, kpin, ng);
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(uz_next, pin_idx, kpin, ng);
        }
        if (nrec > 0) {
            sem2d_record<<<gR, TPB, 0, stream>>>(ux_next, rec_G, rec_W, seis_x, n, nt, nrec, m, ng);
            sem2d_record<<<gR, TPB, 0, stream>>>(uz_next, rec_G, rec_W, seis_z, n, nt, nrec, m, ng);
        }
        if (save) {          // (nt, B, ng) histories
            cudaMemcpyAsync(sux + static_cast<size_t>(n) * ngB, ux_next, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(suz + static_cast<size_t>(n) * ngB, uz_next, ngb, cudaMemcpyDeviceToDevice, stream);
        } else if (bs) {
            // band_ux[n] holds u^{n+1} -- same indexing as the save-all
            // history. Rings are (nt, B, row): host offset n*B*row, kernel
            // adds bat*row.
            const dim3 gBN((nbn + TPB - 1) / TPB, B);
            const dim3 gBE((nbe * m + TPB - 1) / TPB, B);
            const size_t no = static_cast<size_t>(n) * B * nbn;
            const size_t eo = static_cast<size_t>(n) * B * nbe * m;
            // ring slots: u on nodes (0,1), the six element histories (2..7)
            const int bk = in.band_kind;
            sem2d_band_save(ux_next, bnode,
                sem_hist(in.band_ux, in.band_scale, bk, 0, nt, B),
                n, nbn, TPB, stream, B, ng);
            sem2d_band_save(uz_next, bnode,
                sem_hist(in.band_uz, in.band_scale, bk, 1, nt, B),
                n, nbn, TPB, stream, B, ng);
            const float* efld[6] = {sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz};
            const Buf* erg[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                           &in.band_sd_xz, &in.band_sd_zz,
                                           &in.band_apx, &in.band_apz};
            for (int j = 0; j < 6; ++j)
                sem2d_band_save_elem(efld[j], belem,
                    sem_hist(*erg[j], in.band_scale_e, in.band_kind_e, j, nt, B, nbe),
                    n, nbe, m, Em, TPB, stream, B);
        }
        float* tx = ux_prev; ux_prev = ux; ux = ux_next; ux_next = tx;
        float* tz = uz_prev; uz_prev = uz; uz = uz_next; uz_next = tz;
    }
    if (bs) {   // after the final rotate: ux = u^{nt}, ux_prev = u^{nt-1}
        cudaMemcpyAsync(in.fin_ux.data_ptr<float>(), ux, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_uz.data_ptr<float>(), uz, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_ux_prev.data_ptr<float>(), ux_prev, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_uz_prev.data_ptr<float>(), uz_prev, ngb, cudaMemcpyDeviceToDevice, stream);
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped forward (coupled fluid-solid C-PML) -------------------------
// Three phases, TWO interface exchanges per step. The coupled step has three
// assembly points -- the fluid grads, the two interface scatters, and the
// solid element kernel -- but they group:
//   asm  : grads_f + ifc_ac(u -> s_ac)          [exchange 7: gx..gaz, s_ac]
//   mid  : cpml_step, chi_dd, dirichlet, pat*wav, ifc_el, element
//                                               [exchange 4: f_x, f_z, Acc]
//   step : newmark_force, pin, chi leapfrog, dirichlet5, record, save/ring
// The seabed interface necessarily CROSSES a vertical cut; no special
// machinery is needed for that, because the exchange sums the whole cut
// COLUMN and the interface node on it is just one of those nodes -- faces on
// either side of the cut contribute exactly as elements do.
// B == 1; buffer rotation is the caller's.
void fluid_solid_dd_asm(const SemFluidSolidForwardInputCore& in) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    SWEEP_CHECK(in.B == 1, "fluid_solid_dd_asm: DD is single-shot");
    SWEEP_CHECK(in.has_cpml != 0, "fluid_solid_dd: DD needs the coupled C-PML");
    const int TPB = 256;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    float* gx = in.gx.data_ptr<float>(); float* gz = in.gz.data_ptr<float>();
    float* kx = in.kx.data_ptr<float>(); float* kz = in.kz.data_ptr<float>();
    float* gax = in.gax.data_ptr<float>(); float* gaz = in.gaz.data_ptr<float>();
    cudaMemsetAsync(gx, 0, ngb, stream);  cudaMemsetAsync(gz, 0, ngb, stream);
    cudaMemsetAsync(kx, 0, ngb, stream);  cudaMemsetAsync(kz, 0, ngb, stream);
    cudaMemsetAsync(gax, 0, ngb, stream); cudaMemsetAsync(gaz, 0, ngb, stream);
    const size_t sh_ac = static_cast<size_t>(7 * m) * sizeof(float);
    sem2d_acoustic_cpml_grads_f<<<E, m, sh_ac, stream>>>(
        in.chi.data_ptr<float>(), in.psi_x.data_ptr<float>(),
        in.psi_z.data_ptr<float>(), in.a_x.data_ptr<float>(),
        in.a_z.data_ptr<float>(), in.ibool.data_ptr<int32_t>(),
        in.D.data_ptr<float>(), in.ac_inv_rho.data_ptr<float>(),
        in.w1d.data_ptr<float>(), in.ixx.data_ptr<float>(),
        in.ixz.data_ptr<float>(), in.iex.data_ptr<float>(),
        in.iez.data_ptr<float>(), in.ac_mask_e.data_ptr<float>(),
        gx, gz, kx, kz, gax, gaz, ngll, E, ng, 0);
    cudaMemsetAsync(in.s_ac.data_ptr<float>(), 0, ngb, stream);
    if (nifc > 0) {
        const int ib = (nifc + TPB - 1) / TPB;
        sem2d_ifc_ac<<<ib, TPB, 0, stream>>>(
            in.ux.data_ptr<float>(), in.uz.data_ptr<float>(),
            in.ifc_idx.data_ptr<int64_t>(), in.ifc_w.data_ptr<float>(),
            in.ifc_nx.data_ptr<float>(), in.ifc_nz.data_ptr<float>(),
            in.s_ac.data_ptr<float>(), nifc, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_mid(const SemFluidSolidForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    const bool save = in.save_chi.numel() > 0;
    const bool bs = in.bs != 0;
    sem2d_acoustic_cpml_step<<<nb, TPB, 0, stream>>>(
        in.gx.data_ptr<float>(), in.gz.data_ptr<float>(),
        in.kx.data_ptr<float>(), in.kz.data_ptr<float>(),
        in.gax.data_ptr<float>(), in.gaz.data_ptr<float>(),
        in.ac_inv_Mg.data_ptr<float>(),
        in.a_x.data_ptr<float>(), in.b_x.data_ptr<float>(),
        in.dbx_dx.data_ptr<float>(), in.a_z.data_ptr<float>(),
        in.b_z.data_ptr<float>(), in.dbz_dz.data_ptr<float>(),
        in.psi_x.data_ptr<float>(), in.psi_z.data_ptr<float>(),
        in.zeta_x.data_ptr<float>(), in.zeta_z.data_ptr<float>(),
        in.w_sum.data_ptr<float>(), in.psi_x_n.data_ptr<float>(),
        in.psi_z_n.data_ptr<float>(), in.zeta_x_n.data_ptr<float>(),
        in.zeta_z_n.data_ptr<float>(), ng);
    sem2d_fs_chidd<<<nb, TPB, 0, stream>>>(
        in.ac_kappa.data_ptr<float>(), in.w_sum.data_ptr<float>(),
        in.ac_inv_M.data_ptr<float>(), in.pat_chi.data_ptr<float>(),
        in.wavelet.data_ptr<float>(), static_cast<int>(n),
        in.s_ac.data_ptr<float>(), in.chi_dd.data_ptr<float>(), ng, 0);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(
            in.chi_dd.data_ptr<float>(),
            in.dirichlet_idx.data_ptr<int64_t>(), kdir);
    }
    sem2d_fs_pat_wav<<<nb, TPB, 0, stream>>>(
        in.pat_x.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(n), in.f_x.data_ptr<float>(), ng, 0);
    sem2d_fs_pat_wav<<<nb, TPB, 0, stream>>>(
        in.pat_z.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(n), in.f_z.data_ptr<float>(), ng, 0);
    if (nifc > 0) {
        const int ib = (nifc + TPB - 1) / TPB;
        sem2d_ifc_el<<<ib, TPB, 0, stream>>>(
            in.chi_dd.data_ptr<float>(), in.ifc_idx.data_ptr<int64_t>(),
            in.ifc_w.data_ptr<float>(), in.ifc_nx.data_ptr<float>(),
            in.ifc_nz.data_ptr<float>(), in.f_x.data_ptr<float>(),
            in.f_z.data_ptr<float>(), nifc, ng);
    }
    cudaMemsetAsync(in.Acc_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.Acc_z.data_ptr<float>(), 0, ngb, stream);
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    float* sd_xx = save ? in.save_sd_xx.data_ptr<float>()
                        : (bs ? in.sd_scr_xx.data_ptr<float>() : nullptr);
    float* sd_zx = save ? in.save_sd_zx.data_ptr<float>()
                        : (bs ? in.sd_scr_zx.data_ptr<float>() : nullptr);
    float* sd_xz = save ? in.save_sd_xz.data_ptr<float>()
                        : (bs ? in.sd_scr_xz.data_ptr<float>() : nullptr);
    float* sd_zz = save ? in.save_sd_zz.data_ptr<float>()
                        : (bs ? in.sd_scr_zz.data_ptr<float>() : nullptr);
    float* s_apx = save ? in.save_apx.data_ptr<float>()
                        : (bs ? in.ap_scr_x.data_ptr<float>() : nullptr);
    float* s_apz = save ? in.save_apz.data_ptr<float>()
                        : (bs ? in.ap_scr_z.data_ptr<float>() : nullptr);
    const size_t sh_el = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element<<<E, m, sh_el, stream>>>(
        in.ux.data_ptr<float>(), in.uz.data_ptr<float>(),
        in.ux_prev.data_ptr<float>(), in.uz_prev.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.el_lam.data_ptr<float>(), in.el_mu.data_ptr<float>(),
        in.el_lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.el_w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.el_mask_e.data_ptr<float>(), cf,
        in.r_dux_dx.data_ptr<float>(), in.r_dux_dz.data_ptr<float>(),
        in.r_duz_dx.data_ptr<float>(), in.r_duz_dz.data_ptr<float>(),
        in.r_displ_x.data_ptr<float>(), in.r_displ_z.data_ptr<float>(),
        inv_dt, in.Acc_x.data_ptr<float>(), in.Acc_z.data_ptr<float>(),
        sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz,
        save ? static_cast<long>(n) * E * m : 0L, ngll, E, ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void fluid_solid_dd_step(const SemFluidSolidForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    sem2d_elastic_newmark_force<<<nb, TPB, 0, stream>>>(
        in.ux.data_ptr<float>(), in.uz.data_ptr<float>(),
        in.ux_prev.data_ptr<float>(), in.uz_prev.data_ptr<float>(),
        in.Acc_x.data_ptr<float>(), in.Acc_z.data_ptr<float>(),
        in.f_x.data_ptr<float>(), in.f_z.data_ptr<float>(),
        in.el_inv_M.data_ptr<float>(), dt2,
        in.ux_next.data_ptr<float>(), in.uz_next.data_ptr<float>(), ng);
    if (kpin > 0) {
        const int pb = (kpin + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.ux_next.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.uz_next.data_ptr<float>(),
            in.outer_edge_idx.data_ptr<int64_t>(), kpin);
    }
    sem2d_fs_chi_leapfrog<<<nb, TPB, 0, stream>>>(
        in.chi.data_ptr<float>(), in.chi_prev.data_ptr<float>(),
        in.chi_dd.data_ptr<float>(), dt2, in.chi_next.data_ptr<float>(), ng);
    if (kdir > 0) {
        const int db = (kdir + TPB - 1) / TPB;
        sem2d_dirichlet_fill5<<<db, TPB, 0, stream>>>(
            in.chi_next.data_ptr<float>(), in.psi_x_n.data_ptr<float>(),
            in.psi_z_n.data_ptr<float>(), in.zeta_x_n.data_ptr<float>(),
            in.zeta_z_n.data_ptr<float>(),
            nullptr, nullptr, nullptr, nullptr, nullptr,
            in.dirichlet_idx.data_ptr<int64_t>(), kdir, ng);
    }
    if (nrec > 0) {
        const int rb = (nrec + TPB - 1) / TPB;
        const long* rg = in.rec_G.data_ptr<int64_t>();
        const float* rw = in.rec_W.data_ptr<float>();
        sem2d_record<<<rb, TPB, 0, stream>>>(in.chi_next.data_ptr<float>(),
            rg, rw, in.seis_chi.data_ptr<float>(), static_cast<int>(n), nt, nrec, m, ng);
        sem2d_record<<<rb, TPB, 0, stream>>>(in.ux_next.data_ptr<float>(),
            rg, rw, in.seis_x.data_ptr<float>(), static_cast<int>(n), nt, nrec, m, ng);
        sem2d_record<<<rb, TPB, 0, stream>>>(in.uz_next.data_ptr<float>(),
            rg, rw, in.seis_z.data_ptr<float>(), static_cast<int>(n), nt, nrec, m, ng);
    }
    const bool save = in.save_chi.numel() > 0;
    if (save) {
        const size_t o = static_cast<size_t>(n) * ng;
        cudaMemcpyAsync(in.save_chi.data_ptr<float>() + o, in.chi_next.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_ux.data_ptr<float>() + o, in.ux_next.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_uz.data_ptr<float>() + o, in.uz_next.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_psi_x.data_ptr<float>() + o, in.psi_x_n.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_psi_z.data_ptr<float>() + o, in.psi_z_n.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_zeta_x.data_ptr<float>() + o, in.zeta_x_n.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_zeta_z.data_ptr<float>() + o, in.zeta_z_n.data_ptr<float>(),
                        ngb, cudaMemcpyDeviceToDevice, stream);
    }
    if (in.bs != 0) {
        // Boundary saving under DD: each strip rings its OWN band -- the
        // acoustic node band (chi + 4 memories), the elastic node band (u)
        // and the elastic ELEMENT band (six memory/history buffers, written
        // this step into the sd_scr_*/ap_scr_* scratch by dd_mid). fp32
        // rings only, like the fused coupled driver.
        //
        // Called BEFORE the caller rotates, so chi_next == chi^{n+1} and
        // chi == chi^n; the fused driver takes its checkpoints and finals
        // after rotating, which is the same pair.
        const int bk = in.band_kind;
        const int nband = static_cast<int>(in.band_idx.numel());
        if (nband > 0) {
            const long* bg = in.band_idx.data_ptr<int64_t>();
            float* f[5] = {in.chi_next.data_ptr<float>(),
                           in.psi_x_n.data_ptr<float>(),
                           in.psi_z_n.data_ptr<float>(),
                           in.zeta_x_n.data_ptr<float>(),
                           in.zeta_z_n.data_ptr<float>()};
            const Buf* rg[5] = {&in.band_chi, &in.band_psi_x,
                                          &in.band_psi_z, &in.band_zeta_x,
                                          &in.band_zeta_z};
            for (int j = 0; j < 5; ++j)   // fluid node rings, slots 0..4
                sem2d_band_save(f[j], bg,
                                sem_hist(*rg[j], in.band_scale, bk, j, nt),
                                n, nband, TPB, stream);
        }
        const int nbn = static_cast<int>(in.band_node_idx.numel());
        const int nbe = static_cast<int>(in.band_elem_idx.numel());
        if (nbn > 0) {                    // solid node rings, slots 5..6
            const long* bnode = in.band_node_idx.data_ptr<int64_t>();
            sem2d_band_save(in.ux_next.data_ptr<float>(), bnode,
                            sem_hist(in.band_ux, in.band_scale, bk, 5, nt),
                            n, nbn, TPB, stream);
            sem2d_band_save(in.uz_next.data_ptr<float>(), bnode,
                            sem_hist(in.band_uz, in.band_scale, bk, 6, nt),
                            n, nbn, TPB, stream);
        }
        if (nbe > 0) {                    // solid ELEMENT rings, own table
            const long* belem = in.band_elem_idx.data_ptr<int64_t>();
            const long Em = static_cast<long>(E) * m;
            const float* src[6] = {in.sd_scr_xx.data_ptr<float>(),
                                   in.sd_scr_zx.data_ptr<float>(),
                                   in.sd_scr_xz.data_ptr<float>(),
                                   in.sd_scr_zz.data_ptr<float>(),
                                   in.ap_scr_x.data_ptr<float>(),
                                   in.ap_scr_z.data_ptr<float>()};
            const Buf* dst[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                           &in.band_sd_xz, &in.band_sd_zz,
                                           &in.band_apx, &in.band_apz};
            for (int j = 0; j < 6; ++j)
                sem2d_band_save_elem(src[j], belem,
                    sem_hist(*dst[j], in.band_scale_e, in.band_kind_e, j, nt, 1, nbe),
                    n, nbe, m, Em, TPB, stream);
        }
        // full-state re-anchors: the coupled reverse march accumulates error
        // the acoustic/elastic ones do not (see AcousticElasticSEM2D)
        const int Kck = in.ckpt_every;
        if (Kck > 0 && (n + 1) % Kck == 0) {
            const int slot = static_cast<int>((n + 1) / Kck) - 1;
            if (slot < static_cast<int>(in.ck_chi.size(0))) {
                const size_t co = static_cast<size_t>(slot) * ng;
                const float* srcv[6] = {in.chi_next.data_ptr<float>(),
                                        in.chi.data_ptr<float>(),
                                        in.ux_next.data_ptr<float>(),
                                        in.uz_next.data_ptr<float>(),
                                        in.ux.data_ptr<float>(),
                                        in.uz.data_ptr<float>()};
                float* dstv[6] = {in.ck_chi.data_ptr<float>(),
                                  in.ck_chi_prev.data_ptr<float>(),
                                  in.ck_ux.data_ptr<float>(),
                                  in.ck_uz.data_ptr<float>(),
                                  in.ck_ux_prev.data_ptr<float>(),
                                  in.ck_uz_prev.data_ptr<float>()};
                for (int j = 0; j < 6; ++j)
                    cudaMemcpyAsync(dstv[j] + co, srcv[j], ngb,
                                    cudaMemcpyDeviceToDevice, stream);
            }
        }
        if (n == nt - 1) {          // seed the backward's reverse march
            const float* srcv[6] = {in.chi_next.data_ptr<float>(),
                                    in.chi.data_ptr<float>(),
                                    in.ux_next.data_ptr<float>(),
                                    in.uz_next.data_ptr<float>(),
                                    in.ux.data_ptr<float>(),
                                    in.uz.data_ptr<float>()};
            float* dstv[6] = {in.fin_chi.data_ptr<float>(),
                              in.fin_chi_prev.data_ptr<float>(),
                              in.fin_ux.data_ptr<float>(),
                              in.fin_uz.data_ptr<float>(),
                              in.fin_ux_prev.data_ptr<float>(),
                              in.fin_uz_prev.data_ptr<float>()};
            for (int j = 0; j < 6; ++j)
                cudaMemcpyAsync(dstv[j], srcv[j], ngb,
                                cudaMemcpyDeviceToDevice, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- DD stepped forward (elastic C-PML) -------------------------------------
// Same split as the acoustic DD pair, one step at a time so the interface
// columns can be summed across ranks between the phases. The elastic step has
// exactly ONE assembly (the element kernel's Acc_x/Acc_z scatter), so DD needs
// one 2-field exchange per step -- everything the element kernel writes on top
// of that (the six memory/history buffers) is element-local and never crosses
// a cut. B == 1; buffer rotation is the caller's.
void elastic_cpml_dd_asm(const SemElasticCpmlForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.ux.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll;
    SWEEP_CHECK(in.B == 1, "elastic_cpml_dd_asm: DD is single-shot");
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    ElCpmlCoefs cf = sem2d_el_coefs(in);
    const bool save = in.save_ux.numel() > 0;
    const bool bs = in.bs != 0;
    float* sd_xx = save ? in.save_sd_xx.data_ptr<float>()
                        : (bs ? in.sd_scr_xx.data_ptr<float>() : nullptr);
    float* sd_zx = save ? in.save_sd_zx.data_ptr<float>()
                        : (bs ? in.sd_scr_zx.data_ptr<float>() : nullptr);
    float* sd_xz = save ? in.save_sd_xz.data_ptr<float>()
                        : (bs ? in.sd_scr_xz.data_ptr<float>() : nullptr);
    float* sd_zz = save ? in.save_sd_zz.data_ptr<float>()
                        : (bs ? in.sd_scr_zz.data_ptr<float>() : nullptr);
    float* s_apx = save ? in.save_apx.data_ptr<float>()
                        : (bs ? in.ap_scr_x.data_ptr<float>() : nullptr);
    float* s_apz = save ? in.save_apz.data_ptr<float>()
                        : (bs ? in.ap_scr_z.data_ptr<float>() : nullptr);
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    cudaMemsetAsync(in.Acc_x.data_ptr<float>(), 0, ngb, stream);
    cudaMemsetAsync(in.Acc_z.data_ptr<float>(), 0, ngb, stream);
    const size_t shmem = static_cast<size_t>(8 * m) * sizeof(float);
    sem2d_elastic_cpml_element<<<E, m, shmem, stream>>>(
        in.ux.data_ptr<float>(), in.uz.data_ptr<float>(),
        in.ux_prev.data_ptr<float>(), in.uz_prev.data_ptr<float>(),
        in.ibool.data_ptr<int32_t>(), in.D.data_ptr<float>(),
        in.lam.data_ptr<float>(), in.mu.data_ptr<float>(),
        in.lam2mu.data_ptr<float>(), in.wJ.data_ptr<float>(),
        in.w_rho_J.data_ptr<float>(),
        in.ixx.data_ptr<float>(), in.ixz.data_ptr<float>(),
        in.iex.data_ptr<float>(), in.iez.data_ptr<float>(),
        in.mask_e.data_ptr<float>(), cf,
        in.r_dux_dx.data_ptr<float>(), in.r_dux_dz.data_ptr<float>(),
        in.r_duz_dx.data_ptr<float>(), in.r_duz_dz.data_ptr<float>(),
        in.r_displ_x.data_ptr<float>(), in.r_displ_z.data_ptr<float>(),
        inv_dt, in.Acc_x.data_ptr<float>(), in.Acc_z.data_ptr<float>(),
        sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz,
        // save-all: THIS step's slice of the (nt, E*m) element history. A
        // fixed 0 offset would pile every step into slot 0 and the backward,
        // which reads slot n, would image garbage (bs uses a per-step
        // scratch, hence 0 there).
        save ? static_cast<long>(n) * E * m : 0L, ngll, E, ng);
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

void elastic_cpml_dd_step(const SemElasticCpmlForwardInputCore& in, int64_t n) {
    sweep::DeviceGuard guard(in.ux.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const int kpin = static_cast<int>(in.outer_edge_idx.numel());
    SWEEP_CHECK(in.B == 1, "elastic_cpml_dd_step: DD is single-shot");
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);
    sem2d_elastic_newmark<<<nb, TPB, 0, stream>>>(
        in.ux.data_ptr<float>(), in.uz.data_ptr<float>(),
        in.ux_prev.data_ptr<float>(), in.uz_prev.data_ptr<float>(),
        in.Acc_x.data_ptr<float>(), in.Acc_z.data_ptr<float>(),
        in.pat_x.data_ptr<float>(), in.pat_z.data_ptr<float>(),
        in.inv_M.data_ptr<float>(), in.wavelet.data_ptr<float>(),
        static_cast<int>(n), dt2,
        in.ux_next.data_ptr<float>(), in.uz_next.data_ptr<float>(), ng, 0);
    if (kpin > 0) {
        const int pb = (kpin + TPB - 1) / TPB;
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.ux_next.data_ptr<float>(), in.outer_edge_idx.data_ptr<int64_t>(), kpin);
        sem2d_dirichlet_fill<<<pb, TPB, 0, stream>>>(
            in.uz_next.data_ptr<float>(), in.outer_edge_idx.data_ptr<int64_t>(), kpin);
    }
    if (nrec > 0) {
        const int rb = (nrec + TPB - 1) / TPB;
        sem2d_record<<<rb, TPB, 0, stream>>>(
            in.ux_next.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.seis_x.data_ptr<float>(),
            static_cast<int>(n), nt, nrec, m, ng);
        sem2d_record<<<rb, TPB, 0, stream>>>(
            in.uz_next.data_ptr<float>(), in.rec_G.data_ptr<int64_t>(),
            in.rec_W.data_ptr<float>(), in.seis_z.data_ptr<float>(),
            static_cast<int>(n), nt, nrec, m, ng);
    }
    const bool save = in.save_ux.numel() > 0;
    if (save) {
        const size_t o = static_cast<size_t>(n) * ng;
        cudaMemcpyAsync(in.save_ux.data_ptr<float>() + o,
                        in.ux_next.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.save_uz.data_ptr<float>() + o,
                        in.uz_next.data_ptr<float>(), ngb,
                        cudaMemcpyDeviceToDevice, stream);
    } else if (in.bs != 0) {
        // this strip's own band ring (cut columns carry no PML, so they are
        // outside it -- the interior reverse march recomputes them exactly)
        const int nbn = static_cast<int>(in.band_node_idx.numel());
        const int nbe = static_cast<int>(in.band_elem_idx.numel());
        const long* bnode = in.band_node_idx.data_ptr<int64_t>();
        const long* belem = in.band_elem_idx.data_ptr<int64_t>();
        const int bnb = (nbn + TPB - 1) / TPB;
        const int beb = (nbe * m + TPB - 1) / TPB;
        const size_t no = static_cast<size_t>(n) * nbn;
        const size_t eo = static_cast<size_t>(n) * nbe * m;
        const int bk = in.band_kind;
        sem2d_band_save(in.ux_next.data_ptr<float>(), bnode,
            sem_hist(in.band_ux, in.band_scale, bk, 0, nt),
            n, nbn, TPB, stream);
        sem2d_band_save(in.uz_next.data_ptr<float>(), bnode,
            sem_hist(in.band_uz, in.band_scale, bk, 1, nt),
            n, nbn, TPB, stream);
        const float* scr[6] = {in.sd_scr_xx.data_ptr<float>(),
                               in.sd_scr_zx.data_ptr<float>(),
                               in.sd_scr_xz.data_ptr<float>(),
                               in.sd_scr_zz.data_ptr<float>(),
                               in.ap_scr_x.data_ptr<float>(),
                               in.ap_scr_z.data_ptr<float>()};
        const Buf* ring[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                        &in.band_sd_xz, &in.band_sd_zz,
                                        &in.band_apx, &in.band_apz};
        for (int j = 0; j < 6; ++j)
            sem2d_band_save_elem(scr[j], belem,
                sem_hist(*ring[j], in.band_scale_e, in.band_kind_e, j, nt, 1, nbe),
                n, nbe, m, 0L, TPB, stream);
        if (n == nt - 1) {   // seed the reverse march (post-rotate levels)
            cudaMemcpyAsync(in.fin_ux.data_ptr<float>(),
                            in.ux_next.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(in.fin_uz.data_ptr<float>(),
                            in.uz_next.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(in.fin_ux_prev.data_ptr<float>(),
                            in.ux.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(in.fin_uz_prev.data_ptr<float>(),
                            in.uz.data_ptr<float>(), ngb,
                            cudaMemcpyDeviceToDevice, stream);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// ---- coupled fluid-solid forward (P4) ---------------------------------------
void fluid_solid_forward(const SemFluidSolidForwardInputCore& in) {
    sweep::DeviceGuard guard(in.chi.device_index());
    cudaStream_t stream = sweep::current_stream();
    const int E = in.n_elements, ng = in.n_global, ngll = in.ngll;
    const int m = ngll * ngll, nrec = in.n_rec, nt = in.nt;
    const float dt2 = static_cast<float>(in.dt * in.dt);
    const float inv_dt = static_cast<float>(1.0 / in.dt);
    const bool cp = in.has_cpml != 0;
    const int kdir = static_cast<int>(in.dirichlet_idx.numel());
    const int kpin = cp ? static_cast<int>(in.outer_edge_idx.numel()) : 0;
    const int nifc = static_cast<int>(in.ifc_idx.numel());
    // shot batch (see sem_types.h): B shots in ONE launch sequence, ONE shared
    // model. Only the coupled C-PML flavour is batched -- the bare-box kernels
    // carry no batch axis.
    const int B = in.B;
    SWEEP_CHECK(B >= 1, "fluid_solid_forward: B=", B);
    SWEEP_CHECK(B == 1 || cp,
                "fluid_solid_forward: the shot batch needs the coupled C-PML");
    SWEEP_CHECK(in.chi.numel() == static_cast<int64_t>(B) * ng,
                "fluid_solid_forward: chi holds ", in.chi.numel(),
                " values, expected B*n_global = ", static_cast<int64_t>(B) * ng);
    SWEEP_CHECK(in.pat_chi.numel() == static_cast<int64_t>(B) * ng,
                "fluid_solid_forward: pat_chi must be (B, n_global)");
    SWEEP_CHECK(in.wavelet.numel() == nt
                || in.wavelet.numel() == static_cast<int64_t>(B) * nt,
                "fluid_solid_forward: wavelet must be (nt,) or (B, nt)");
    const int wst = in.wavelet.numel() == nt ? 0 : nt;   // per-shot wavelet stride

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
    float* seis_chi = nrec > 0 ? in.seis_chi.data_ptr<float>() : nullptr;
    float* seis_x = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* seis_z = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;

    ElCpmlCoefs cf{};
    if (cp) {
        // the helper, not a hand-copied packing: a field added to
        // ElCpmlCoefs used to reach this site SILENTLY NULL, which is not
        // a crash -- the coupled path simply never took the interior fast
        // path the elastic one had just gained.
        cf = sem2d_el_coefs(in);
    }

    float* chi = in.chi.data_ptr<float>();
    float* chi_prev = in.chi_prev.data_ptr<float>();
    float* chi_next = in.chi_next.data_ptr<float>();
    float* s_ac = in.s_ac.data_ptr<float>();
    float* chi_dd = in.chi_dd.data_ptr<float>();
    float* Kchi = in.Kchi.data_ptr<float>();
    float* psi_x = cp ? in.psi_x.data_ptr<float>() : nullptr;
    float* psi_z = cp ? in.psi_z.data_ptr<float>() : nullptr;
    float* zeta_x = cp ? in.zeta_x.data_ptr<float>() : nullptr;
    float* zeta_z = cp ? in.zeta_z.data_ptr<float>() : nullptr;
    float* psi_x_n = cp ? in.psi_x_n.data_ptr<float>() : nullptr;
    float* psi_z_n = cp ? in.psi_z_n.data_ptr<float>() : nullptr;
    float* zeta_x_n = cp ? in.zeta_x_n.data_ptr<float>() : nullptr;
    float* zeta_z_n = cp ? in.zeta_z_n.data_ptr<float>() : nullptr;
    float* w_sum = cp ? in.w_sum.data_ptr<float>() : nullptr;
    float* gxp = cp ? in.gx.data_ptr<float>() : nullptr;
    float* gzp = cp ? in.gz.data_ptr<float>() : nullptr;
    float* kxp = cp ? in.kx.data_ptr<float>() : nullptr;
    float* kzp = cp ? in.kz.data_ptr<float>() : nullptr;
    float* gaxp = cp ? in.gax.data_ptr<float>() : nullptr;
    float* gazp = cp ? in.gaz.data_ptr<float>() : nullptr;
    float* ux = in.ux.data_ptr<float>();       float* uz = in.uz.data_ptr<float>();
    float* ux_prev = in.ux_prev.data_ptr<float>(); float* uz_prev = in.uz_prev.data_ptr<float>();
    float* ux_next = in.ux_next.data_ptr<float>(); float* uz_next = in.uz_next.data_ptr<float>();
    float* Acc_x = in.Acc_x.data_ptr<float>(); float* Acc_z = in.Acc_z.data_ptr<float>();
    float* f_x = in.f_x.data_ptr<float>();     float* f_z = in.f_z.data_ptr<float>();
    float* r_dux_dx = cp ? in.r_dux_dx.data_ptr<float>() : nullptr;
    float* r_dux_dz = cp ? in.r_dux_dz.data_ptr<float>() : nullptr;
    float* r_duz_dx = cp ? in.r_duz_dx.data_ptr<float>() : nullptr;
    float* r_duz_dz = cp ? in.r_duz_dz.data_ptr<float>() : nullptr;
    float* r_displ_x = cp ? in.r_displ_x.data_ptr<float>() : nullptr;
    float* r_displ_z = cp ? in.r_displ_z.data_ptr<float>() : nullptr;

    const bool save = in.save_chi.numel() > 0;
    const bool bs = in.bs != 0;
    SWEEP_CHECK(!bs || cp, "fluid_solid_forward: boundary saving needs the "
                "coupled C-PML (bare boxes keep the full tape)");
    SWEEP_CHECK(!(bs && save), "fluid_solid_forward: bs and save-all are exclusive");
    float* s_chi = save ? in.save_chi.data_ptr<float>() : nullptr;
    float* s_px = (save && cp) ? in.save_psi_x.data_ptr<float>() : nullptr;
    float* s_pz = (save && cp) ? in.save_psi_z.data_ptr<float>() : nullptr;
    float* s_zx = (save && cp) ? in.save_zeta_x.data_ptr<float>() : nullptr;
    float* s_zz = (save && cp) ? in.save_zeta_z.data_ptr<float>() : nullptr;
    float* s_ux = save ? in.save_ux.data_ptr<float>() : nullptr;
    float* s_uz = save ? in.save_uz.data_ptr<float>() : nullptr;
    // element-kernel history sinks: the (nt, E*m) save-all slice, or -- under
    // boundary saving -- a per-step scratch that is then band-gathered
    float* sd_xx = (save && cp) ? in.save_sd_xx.data_ptr<float>()
                 : (bs ? in.sd_scr_xx.data_ptr<float>() : nullptr);
    float* sd_zx = (save && cp) ? in.save_sd_zx.data_ptr<float>()
                 : (bs ? in.sd_scr_zx.data_ptr<float>() : nullptr);
    float* sd_xz = (save && cp) ? in.save_sd_xz.data_ptr<float>()
                 : (bs ? in.sd_scr_xz.data_ptr<float>() : nullptr);
    float* sd_zz = (save && cp) ? in.save_sd_zz.data_ptr<float>()
                 : (bs ? in.sd_scr_zz.data_ptr<float>() : nullptr);
    float* s_apx = (save && cp) ? in.save_apx.data_ptr<float>()
                 : (bs ? in.ap_scr_x.data_ptr<float>() : nullptr);
    float* s_apz = (save && cp) ? in.save_apz.data_ptr<float>()
                 : (bs ? in.ap_scr_z.data_ptr<float>() : nullptr);
    // bs rings: acoustic node band (fp32 handles) + elastic node/element bands
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
    const int Kck = bs ? in.ckpt_every : 0;
    const int nck = Kck > 0 ? static_cast<int>(in.ck_chi.size(0)) : 0;

    const size_t sh_ac = static_cast<size_t>((cp ? 7 : 3) * m) * sizeof(float);
    const size_t sh_el = static_cast<size_t>((cp ? 8 : 6) * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const int db = kdir > 0 ? (kdir + TPB - 1) / TPB : 0;
    const int pb = kpin > 0 ? (kpin + TPB - 1) / TPB : 0;
    const int ib = nifc > 0 ? (nifc + TPB - 1) / TPB : 0;
    // batch-wide spans: every field/scratch buffer is (B, ng), so the per-step
    // memsets/copies below cover the whole batch in one call
    const size_t ngB = static_cast<size_t>(B) * ng;
    const size_t ngb = ngB * sizeof(float);
    const long Em = static_cast<long>(E) * m;
    const dim3 gE(E, B), gN(nb, B), gR(rb, B), gD(db, B), gP(pb, B), gI(ib, B);

    for (int n = 0; n < nt; ++n) {
        // 1. fluid residual: stretched (CPML) or bare stiffness, + S^ac
        if (cp) {
            // fused gather: a*psi is formed in-register, dropping the two
            // scale2 launches. sh_ac is already 7*m in the cp branch, which is
            // exactly what _grads_f needs. Bit-exact: same single fp32 multiply.
            cudaMemsetAsync(gxp, 0, ngb, stream);  cudaMemsetAsync(gzp, 0, ngb, stream);
            cudaMemsetAsync(kxp, 0, ngb, stream);  cudaMemsetAsync(kzp, 0, ngb, stream);
            cudaMemsetAsync(gaxp, 0, ngb, stream); cudaMemsetAsync(gazp, 0, ngb, stream);
            sem2d_acoustic_cpml_grads_f<<<gE, m, sh_ac, stream>>>(
                chi, psi_x, psi_z, ax, az, ibool, D, ac_ir, w1d, ixx, ixz, iex, iez,
                ac_me, gxp, gzp, kxp, kzp, gaxp, gazp, ngll, E, ng, 0);
            sem2d_acoustic_cpml_step<<<gN, TPB, 0, stream>>>(
                gxp, gzp, kxp, kzp, gaxp, gazp, ac_iMg, ax, bx, dbxdx, az, bz, dbzdz,
                psi_x, psi_z, zeta_x, zeta_z, w_sum, psi_x_n, psi_z_n, zeta_x_n, zeta_z_n, ng);
        } else {
            cudaMemsetAsync(Kchi, 0, ngb, stream);
            sem2d_acoustic_stiffness<<<E, m, sh_ac, stream>>>(
                chi, ibool, D, ac_ir, wJ, ixx, ixz, iex, iez, ac_me, Kchi, ngll, E);
        }
        cudaMemsetAsync(s_ac, 0, ngb, stream);
        if (nifc > 0)
            sem2d_ifc_ac<<<gI, TPB, 0, stream>>>(ux, uz, ifc_g, ifc_w, ifc_nx, ifc_nz, s_ac, nifc, ng);
        // 2-3. chi_dd (post-inversion), free surface wins at the triple point
        if (cp)
            sem2d_fs_chidd<<<gN, TPB, 0, stream>>>(
                ac_kp, w_sum, ac_iM, pat_chi, wav, n, s_ac, chi_dd, ng, wst);
        else
            sem2d_fs_chidd_bare<<<nb, TPB, 0, stream>>>(
                Kchi, ac_iM, pat_chi, wav, n, s_ac, chi_dd, ng);
        if (kdir > 0)
            sem2d_dirichlet_fill_b<<<gD, TPB, 0, stream>>>(chi_dd, didx, kdir, ng);
        // 4. solid force = pat*wav + S^el(-chi_dd), then the solid step
        sem2d_fs_pat_wav<<<gN, TPB, 0, stream>>>(pat_x, wav, n, f_x, ng, wst);
        sem2d_fs_pat_wav<<<gN, TPB, 0, stream>>>(pat_z, wav, n, f_z, ng, wst);
        if (nifc > 0)
            sem2d_ifc_el<<<gI, TPB, 0, stream>>>(chi_dd, ifc_g, ifc_w, ifc_nx, ifc_nz, f_x, f_z, nifc, ng);
        cudaMemsetAsync(Acc_x, 0, ngb, stream);
        cudaMemsetAsync(Acc_z, 0, ngb, stream);
        if (cp)
            sem2d_elastic_cpml_element<<<gE, m, sh_el, stream>>>(
                ux, uz, ux_prev, uz_prev, ibool, D, el_lam, el_mu, el_l2m, wJ, el_wrJ,
                ixx, ixz, iex, iez, el_me, cf,
                r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z,
                inv_dt, Acc_x, Acc_z,
                sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz,
                save ? static_cast<long>(n) * B * Em : 0L, ngll, E, ng);
        else
            sem2d_elastic_stiffness<<<E, m, sh_el, stream>>>(
                ux, uz, ibool, D, el_lam, el_mu, el_l2m, wJ, ixx, ixz, iex, iez,
                el_me, Acc_x, Acc_z, ngll, E);
        sem2d_elastic_newmark_force<<<gN, TPB, 0, stream>>>(
            ux, uz, ux_prev, uz_prev, Acc_x, Acc_z, f_x, f_z, el_iM, dt2,
            ux_next, uz_next, ng);
        if (kpin > 0) {
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(ux_next, pin_idx, kpin, ng);
            sem2d_dirichlet_fill_b<<<gP, TPB, 0, stream>>>(uz_next, pin_idx, kpin, ng);
        }
        // 5. fluid update + free surface (chi and, with CPML, its memories)
        sem2d_fs_chi_leapfrog<<<gN, TPB, 0, stream>>>(chi, chi_prev, chi_dd, dt2, chi_next, ng);
        if (kdir > 0) {
            if (cp) {   // chi + the four CPML memories in ONE launch (was 5)
                sem2d_dirichlet_fill5<<<gD, TPB, 0, stream>>>(
                    chi_next, psi_x_n, psi_z_n, zeta_x_n, zeta_z_n,
                    nullptr, nullptr, nullptr, nullptr, nullptr, didx, kdir, ng);
            } else {
                sem2d_dirichlet_fill<<<db, TPB, 0, stream>>>(chi_next, didx, kdir);
            }
        }
        if (nrec > 0) {
            sem2d_record<<<gR, TPB, 0, stream>>>(chi_next, rec_G, rec_W, seis_chi, n, nt, nrec, m, ng);
            sem2d_record<<<gR, TPB, 0, stream>>>(ux_next, rec_G, rec_W, seis_x, n, nt, nrec, m, ng);
            sem2d_record<<<gR, TPB, 0, stream>>>(uz_next, rec_G, rec_W, seis_z, n, nt, nrec, m, ng);
        }
        if (save) {
            const size_t o = static_cast<size_t>(n) * ngB;   // (nt, B, ng)
            cudaMemcpyAsync(s_chi + o, chi_next, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_ux + o, ux_next, ngb, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_uz + o, uz_next, ngb, cudaMemcpyDeviceToDevice, stream);
            if (cp) {
                cudaMemcpyAsync(s_px + o, psi_x_n, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s_pz + o, psi_z_n, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s_zx + o, zeta_x_n, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(s_zz + o, zeta_z_n, ngb, cudaMemcpyDeviceToDevice, stream);
            }
        }
        if (bs) {
            // acoustic ring: state^{n+1}; elastic ring: u^{n+1} on band nodes
            // + this step's element histories on band elements. All rings are
            // (nt, B, nband)-shaped: host offset n*B*row, kernel adds bat*row.
            sem2d_band_save(chi_next, band_g, h_chi, n, nband, TPB, stream, B, ng);
            sem2d_band_save(psi_x_n, band_g, h_px, n, nband, TPB, stream, B, ng);
            sem2d_band_save(psi_z_n, band_g, h_pz, n, nband, TPB, stream, B, ng);
            sem2d_band_save(zeta_x_n, band_g, h_zx, n, nband, TPB, stream, B, ng);
            sem2d_band_save(zeta_z_n, band_g, h_zz, n, nband, TPB, stream, B, ng);
            const dim3 gBN((nbn + TPB - 1) / TPB, B);
            const dim3 gBE((nbe * m + TPB - 1) / TPB, B);
            const size_t no = static_cast<size_t>(n) * B * nbn;
            const size_t eo = static_cast<size_t>(n) * B * nbe * m;
            sem2d_band_save(ux_next, bnode,
                sem_hist(in.band_ux, in.band_scale, bkind, 5, nt, B),
                n, nbn, TPB, stream, B, ng);
            sem2d_band_save(uz_next, bnode,
                sem_hist(in.band_uz, in.band_scale, bkind, 6, nt, B),
                n, nbn, TPB, stream, B, ng);
            const float* efld[6] = {sd_xx, sd_zx, sd_xz, sd_zz, s_apx, s_apz};
            const Buf* erg[6] = {&in.band_sd_xx, &in.band_sd_zx,
                                           &in.band_sd_xz, &in.band_sd_zz,
                                           &in.band_apx, &in.band_apz};
            for (int q = 0; q < 6; ++q)
                sem2d_band_save_elem(efld[q], belem,
                    sem_hist(*erg[q], in.band_scale_e, in.band_kind_e, q, nt, B, nbe),
                    n, nbe, m, Em, TPB, stream, B);
        }
        // rotate
        float* t = chi_prev; chi_prev = chi; chi = chi_next; chi_next = t;
        float* tx = ux_prev; ux_prev = ux; ux = ux_next; ux_next = tx;
        float* tz = uz_prev; uz_prev = uz; uz = uz_next; uz_next = tz;
        if (cp) {
            float* a;
            a = psi_x; psi_x = psi_x_n; psi_x_n = a;
            a = psi_z; psi_z = psi_z_n; psi_z_n = a;
            a = zeta_x; zeta_x = zeta_x_n; zeta_x_n = a;
            a = zeta_z; zeta_z = zeta_z_n; zeta_z_n = a;
        }
        // full-state re-anchor checkpoint at level n+1: the two leapfrog
        // levels of each field (memories are rebuilt from the rings exactly,
        // so they need no checkpointing)
        if (bs && Kck > 0 && (n + 1) % Kck == 0) {
            const int slot = (n + 1) / Kck - 1;
            if (slot < nck) {
                const size_t co = static_cast<size_t>(slot) * ngB;   // (nc, B, ng)
                cudaMemcpyAsync(in.ck_chi.data_ptr<float>() + co, chi, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(in.ck_chi_prev.data_ptr<float>() + co, chi_prev, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(in.ck_ux.data_ptr<float>() + co, ux, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(in.ck_uz.data_ptr<float>() + co, uz, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(in.ck_ux_prev.data_ptr<float>() + co, ux_prev, ngb, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(in.ck_uz_prev.data_ptr<float>() + co, uz_prev, ngb, cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
    if (bs) {   // final two levels of each field seed the reverse march
        cudaMemcpyAsync(in.fin_chi.data_ptr<float>(), chi, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_chi_prev.data_ptr<float>(), chi_prev, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_ux.data_ptr<float>(), ux, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_uz.data_ptr<float>(), uz, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_ux_prev.data_ptr<float>(), ux_prev, ngb, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(in.fin_uz_prev.data_ptr<float>(), uz_prev, ngb, cudaMemcpyDeviceToDevice, stream);
    }

    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem2d

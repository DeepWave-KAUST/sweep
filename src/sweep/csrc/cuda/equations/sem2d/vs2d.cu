// sweep.sem CUDA (impl='c') backend -- the 2-D velocity-stress SEM drivers
// (ElasticVSSEM2DDeformed, ElasticVTISEM2DDeformed, ElasticTTISEM2DDeformed,
// ViscoElasticVSSEM2DDeformed): the staggered leapfrog of the eager step, bare or
// with the C-PML, the natural free surface, save-all. The 3-D velocity-stress
// drivers (sem3d/forward.cu, backward.cu) with two components.
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "sem2d.h"
#include "kernels.cuh"
#include "vs2d_kernels.cuh"

namespace sem2d {

namespace {

// What both directions read off their input the same way.
struct Vs2dCommon {
    int E, ng, ngll, m, nrec, nt, nc, nsls;
    float dt;
    bool cp;
    const int* ibool; const float* D;
    const float *lam, *mu, *l2m, *cij;
    const float *wJ, *Ixx, *Ixz, *Iex, *Iez, *inv_M;
    const float *ax, *bx, *az, *bz, *a_n;
    const float *px, *pz, *ptau, *mtau, *gsls_w;
    const long* rec_G; const float* rec_W;
};

template <class In>
Vs2dCommon vs2d_common(const In& in)
{
    Vs2dCommon c{};
    c.E = in.n_elements; c.ng = in.n_global; c.ngll = in.ngll;
    c.m = c.ngll * c.ngll; c.nrec = in.n_rec; c.nt = in.nt;
    c.dt = static_cast<float>(in.dt);
    SWEEP_CHECK(in.batch == 1, "sem2d velocity-stress driver runs batch=1");
    SWEEP_CHECK(in.ndim == 2, "sem2d velocity-stress driver is 2-D");
    c.nc = in.aniso;
    SWEEP_CHECK(c.nc == 0 || c.nc == 4 || c.nc == 6,
                "sem2d VS aniso must be 0, 4 or 6 stiffnesses");
    c.nsls = in.nsls;
    SWEEP_CHECK(c.nsls == 0 || c.nc == 0, "sem2d VS: GSLS memories need the isotropic moduli");
    c.cp = in.a_x.numel() > 0;
    c.ibool = in.ibool.template data_ptr<int32_t>();
    c.D = in.D.template data_ptr<float>();
    c.lam = c.nc ? nullptr : in.lam.template data_ptr<float>();
    c.mu = c.nc ? nullptr : in.mu.template data_ptr<float>();
    c.l2m = c.nc ? nullptr : in.lam2mu.template data_ptr<float>();
    c.cij = c.nc ? in.cij.template data_ptr<float>() : nullptr;
    c.wJ = in.wJ.template data_ptr<float>();
    c.Ixx = in.ixx.template data_ptr<float>();
    c.Ixz = in.ixz.template data_ptr<float>();
    c.Iex = in.iex.template data_ptr<float>();
    c.Iez = in.iez.template data_ptr<float>();
    c.inv_M = in.inv_M.template data_ptr<float>();
    c.ax = c.cp ? in.a_x.template data_ptr<float>() : nullptr;
    c.bx = c.cp ? in.b_x.template data_ptr<float>() : nullptr;
    c.az = c.cp ? in.a_z.template data_ptr<float>() : nullptr;
    c.bz = c.cp ? in.b_z.template data_ptr<float>() : nullptr;
    c.a_n = c.cp ? in.a_n.template data_ptr<float>() : nullptr;
    c.px = in.pat_x.template data_ptr<float>();
    c.pz = in.pat_z.template data_ptr<float>();
    c.ptau = c.nsls ? in.ptau.template data_ptr<float>() : nullptr;
    c.mtau = c.nsls ? in.mtau.template data_ptr<float>() : nullptr;
    c.gsls_w = c.nsls ? in.gsls_w.template data_ptr<float>() : nullptr;
    c.rec_G = c.nrec > 0 ? in.rec_G.template data_ptr<int64_t>() : nullptr;
    c.rec_W = c.nrec > 0 ? in.rec_W.template data_ptr<float>() : nullptr;
    return c;
}

} // namespace

// Per step: R = div(sig) (P by direction where the C-PML reaches) -> C-PML on the
// divergence -> v update (+ record at (n+1/2) dt) -> stretched gradients and the
// stress update (+ the GSLS memories).
void elastic_vs2d_forward(const Sem2dVsForwardInputCore& in) {
    sweep::DeviceGuard guard(in.vx.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Vs2dCommon c = vs2d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const float dt = c.dt;
    const float* wav = in.wavelet.data_ptr<float>();
    float* sx = nrec > 0 ? in.seis_x.data_ptr<float>() : nullptr;
    float* sz = nrec > 0 ? in.seis_z.data_ptr<float>() : nullptr;
    float* vx = in.vx.data_ptr<float>();
    float* vz = in.vz.data_ptr<float>();
    float* Rx = in.Rx.data_ptr<float>();
    float* Rz = in.Rz.data_ptr<float>();
    float* P = c.cp ? in.P.data_ptr<float>() : nullptr;
    float* msig = c.cp ? in.msig.data_ptr<float>() : nullptr;
    float* sig = in.sig.data_ptr<float>();
    float* mvel = in.mvel.data_ptr<float>();
    float* rmem = c.nsls ? in.rmem.data_ptr<float>() : nullptr;
    const bool save = in.save_vx.numel() > 0;
    float* s_vx = save ? in.save_vx.data_ptr<float>() : nullptr;
    float* s_vz = save ? in.save_vz.data_ptr<float>() : nullptr;
    float* s_gv = save ? in.save_gv.data_ptr<float>() : nullptr;
    const long Em = static_cast<long>(E) * m;

    const size_t sh4 = static_cast<size_t>(4 * m) * sizeof(float);
    const size_t sh2 = static_cast<size_t>(2 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rb = nrec > 0 ? (nrec + TPB - 1) / TPB : 0;
    const size_t ngb = static_cast<size_t>(ng) * sizeof(float);

    for (int n = 0; n < nt; ++n) {
        cudaMemsetAsync(Rx, 0, ngb, stream);
        cudaMemsetAsync(Rz, 0, ngb, stream);
        if (c.cp) cudaMemsetAsync(P, 0, 4 * ngb, stream);
        sem2d_vs_sigma_div<<<E, m, sh4, stream>>>(
            sig, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, Rx, Rz, P, ngll, E, ng);
        if (c.cp)
            sem2d_vs_cpml_nodal<<<nb, TPB, 0, stream>>>(c.a_n, P, msig, Rx, Rz, ng);
        const size_t o = static_cast<size_t>(n) * ng;
        sem2d_vs_vel_update<<<nb, TPB, 0, stream>>>(
            vx, vz, Rx, Rz, c.inv_M, c.px, c.pz, wav, n, dt,
            save ? s_vx + o : nullptr, save ? s_vz + o : nullptr, ng);
        float* gv_n = save ? s_gv + static_cast<size_t>(n) * 4 * Em : nullptr;
        SEM2D_VS_LAUNCH_NC(c.nc, sem2d_vs_stress_update, E, m, sh2, stream,
            vx, vz, sig, mvel, c.ax, c.bx, c.az, c.bz, c.lam, c.mu, c.l2m, c.cij,
            c.nsls, c.ptau, c.mtau, c.gsls_w, rmem, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, dt, gv_n, ngll, E);
        if (nrec > 0) {
            sem2d_record<<<rb, TPB, 0, stream>>>(vx, c.rec_G, c.rec_W, sx, n, nt, nrec, m, ng);
            sem2d_record<<<rb, TPB, 0, stream>>>(vz, c.rec_G, c.rec_W, sz, n, nt, nrec, m, ng);
        }
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

// Reverse loop k = nt-1..0: record^T -> stress_update^T (imaging, the memories'
// adjoints, Grad^T into lv) -> vel_update^T (grad_inv_M, lR) + grad_wavelet ->
// cpml_nodal^T (lmsig, lP) -> sigma_div^T (lsig). lv / lsig / lmsig / lmvel /
// lrmem carry across steps in place.
void elastic_vs2d_backward(const Sem2dVsBackwardInputCore& in) {
    sweep::DeviceGuard guard(in.inv_M.device_index());
    cudaStream_t stream = sweep::current_stream();
    const Vs2dCommon c = vs2d_common(in);
    const int E = c.E, ng = c.ng, ngll = c.ngll, m = c.m, nrec = c.nrec, nt = c.nt;
    const float dt = c.dt;
    const float* gsx = nrec > 0 ? in.grad_sx.data_ptr<float>() : nullptr;
    const float* gsz = nrec > 0 ? in.grad_sz.data_ptr<float>() : nullptr;
    const float* svx = in.save_vx.data_ptr<float>();
    const float* svz = in.save_vz.data_ptr<float>();
    const float* sgv = in.save_gv.data_ptr<float>();
    float* lvx = in.lvx.data_ptr<float>();
    float* lvz = in.lvz.data_ptr<float>();
    float* lRx = in.lRx.data_ptr<float>();
    float* lRz = in.lRz.data_ptr<float>();
    float* lsig = in.lsig.data_ptr<float>();
    float* lmsig = c.cp ? in.lmsig.data_ptr<float>() : nullptr;
    float* lP = c.cp ? in.lP.data_ptr<float>() : nullptr;
    float* lmvel = in.lmvel.data_ptr<float>();
    float* lrmem = c.nsls ? in.lrmem.data_ptr<float>() : nullptr;
    float* gLam = c.nc ? nullptr : in.grad_lam.data_ptr<float>();
    float* gMu = c.nc ? nullptr : in.grad_mu.data_ptr<float>();
    float* gL2m = c.nc ? nullptr : in.grad_lam2mu.data_ptr<float>();
    float* gC = c.nc ? in.grad_cij.data_ptr<float>() : nullptr;
    float* gPt = c.nsls ? in.grad_ptau.data_ptr<float>() : nullptr;
    float* gMt = c.nsls ? in.grad_mtau.data_ptr<float>() : nullptr;
    float* gM = in.grad_inv_M.data_ptr<float>();
    float* gW = in.grad_wavelet.data_ptr<float>();
    const long Em = static_cast<long>(E) * m;

    const size_t sh4 = static_cast<size_t>(4 * m) * sizeof(float);
    const int TPB = 256;
    const int nb = (ng + TPB - 1) / TPB;
    const int rj = nrec > 0 ? (nrec * m + TPB - 1) / TPB : 0;

    for (int k = nt - 1; k >= 0; --k) {
        if (nrec > 0) {
            sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsx, c.rec_G, c.rec_W, lvx, k, nt, nrec, m);
            sem2d_adjoint_source<<<rj, TPB, 0, stream>>>(gsz, c.rec_G, c.rec_W, lvz, k, nt, nrec, m);
        }
        const float* gv_k = sgv + static_cast<size_t>(k) * 4 * Em;
        SEM2D_VS_LAUNCH_NC(c.nc, sem2d_vs_stress_update_T, E, m, sh4, stream,
            lsig, gv_k, lmvel, c.ax, c.bx, c.az, c.bz, c.lam, c.mu, c.l2m, c.cij,
            c.nsls, c.ptau, c.mtau, c.gsls_w, lrmem, gPt, gMt,
            c.Ixx, c.Ixz, c.Iex, c.Iez, c.ibool, c.D, dt, lvx, lvz,
            gLam, gMu, gL2m, gC, ngll, E);
        const size_t ok = static_cast<size_t>(k) * ng;
        const float* vk1x = k >= 1 ? svx + (ok - ng) : nullptr;
        const float* vk1z = k >= 1 ? svz + (ok - ng) : nullptr;
        sem2d_vs_vel_update_T<<<nb, TPB, 0, stream>>>(
            lvx, lvz, svx + ok, svz + ok, vk1x, vk1z, c.inv_M, dt, lRx, lRz, gM, ng);
        sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(c.px, c.inv_M, lvx, dt, k, gW, ng);
        sem2d_accum_gradW<<<nb, TPB, 0, stream>>>(c.pz, c.inv_M, lvz, dt, k, gW, ng);
        if (c.cp)
            sem2d_vs_cpml_nodal_T<<<nb, TPB, 0, stream>>>(c.a_n, lRx, lRz, lmsig, lP, ng);
        sem2d_vs_sigma_div_T<<<E, m, sh4, stream>>>(
            lRx, lRz, lP, lsig, c.ax, c.bx, c.az, c.bz, c.wJ, c.Ixx, c.Ixz, c.Iex, c.Iez,
            c.ibool, c.D, ngll, E, ng);
    }
    SWEEP_CUDA_CHECK(cudaGetLastError());
}

} // namespace sem2d

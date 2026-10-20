#pragma once
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"

namespace sem2d {

// Phase 0: acoustic-potential SEM forward (bare, no CPML). Runs the full NSTEP
// Newmark loop on device over the Python-allocated buffers in `in`, fills
// `in.seis`, and returns it. Buffers must be pre-zeroed by the caller.
void acoustic_forward(const SemForwardInputCore& in);

// Phase 1: acoustic-potential SEM backward (adjoint-state, save-all forward
// wavefield). Returns (grad_inv_M, grad_inv_rho, grad_wavelet) — the eager
// build() graph then chains these to vp_e/rho_e. Buffers/outputs in `in` must
// be pre-zeroed by the caller.
void acoustic_backward(const SemBackwardInputCore& in);

// P2: elastic SEM forward (bare, free surface = natural). Fills seis_x, seis_z
// and returns them; if save_ux/save_uz are non-empty, saves the forward
// displacement history for the backward pass.
void elastic_forward(const SemElasticForwardInputCore& in);

// P3a: acoustic-potential SEM forward WITH C-PML. Fills seis and returns it.
void acoustic_cpml_forward(const SemAcousticCpmlForwardInputCore& in);

// DD stepped entry points (see forward.cu / backward.cu): one time step split
// around the Python-side interface-column exchanges. B == 1, no bs.
void acoustic_cpml_dd_grads(const SemAcousticCpmlForwardInputCore& in);
void acoustic_cpml_dd_step(const SemAcousticCpmlForwardInputCore& in, int64_t n);
void acoustic_cpml_dd_bwd_pre(const SemAcousticCpmlBackwardInputCore& in, int64_t k);
void acoustic_cpml_dd_bwd_mid(const SemAcousticCpmlBackwardInputCore& in, int64_t k);
void acoustic_cpml_dd_bwd_end(const SemAcousticCpmlBackwardInputCore& in, int64_t k);

// Elastic C-PML DD stepped entry points (see forward.cu / backward.cu).
void elastic_cpml_dd_asm(const SemElasticCpmlForwardInputCore& in, int64_t n);
void elastic_cpml_dd_step(const SemElasticCpmlForwardInputCore& in, int64_t n);
void elastic_cpml_dd_bwd_march_asm(const SemElasticCpmlBackwardInputCore& in);
void elastic_cpml_dd_bwd_march_fin(const SemElasticCpmlBackwardInputCore& in, int64_t n);
void elastic_cpml_dd_bwd_pre(const SemElasticCpmlBackwardInputCore& in, int64_t n);
void elastic_cpml_dd_bwd_end(const SemElasticCpmlBackwardInputCore& in, int64_t n);

// Coupled fluid-solid DD stepped entry points (see forward.cu / backward.cu).
void fluid_solid_dd_asm(const SemFluidSolidForwardInputCore& in);
void fluid_solid_dd_mid(const SemFluidSolidForwardInputCore& in, int64_t n);
void fluid_solid_dd_step(const SemFluidSolidForwardInputCore& in, int64_t n);
void fluid_solid_dd_bwd_a(const SemFluidSolidBackwardInputCore& in, int64_t k);
void fluid_solid_dd_bwd_b(const SemFluidSolidBackwardInputCore& in, int64_t k);
void fluid_solid_dd_bwd_c(const SemFluidSolidBackwardInputCore& in, int64_t k);
void fluid_solid_dd_bwd_d(const SemFluidSolidBackwardInputCore& in, int64_t k);
void fluid_solid_dd_bwd_e(const SemFluidSolidBackwardInputCore& in, int64_t k);
// boundary saving on the coupled DD path (no-ops when bs == 0)
void fluid_solid_dd_bwd_recon(const SemFluidSolidBackwardInputCore& in, int64_t k);
void fluid_solid_dd_bwd_march_asm(const SemFluidSolidBackwardInputCore& in,
                                  int64_t k);
void fluid_solid_dd_bwd_march_fin(const SemFluidSolidBackwardInputCore& in,
                                  int64_t k);

// P3b: acoustic-potential SEM backward WITH C-PML (adjoint through psi/zeta).
// Returns (grad_inv_M, grad_inv_rho, grad_kappa, grad_inv_Mg, grad_wavelet).
void acoustic_cpml_backward(const SemAcousticCpmlBackwardInputCore& in);

// P2: elastic SEM backward (adjoint-state, save-all). Returns
// (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
void elastic_backward(const SemElasticBackwardInputCore& in);

// P3e: elastic SEM forward WITH SPECFEM2D-style 2nd-order C-PML. Fills
// seis_x, seis_z and returns them; the optional save buffers record the
// displacement history + stretched derivatives + accel paren terms.
void elastic_cpml_forward(const SemElasticCpmlForwardInputCore& in);

// P3e: elastic SEM backward WITH C-PML (adjoint through the 6 memory
// recursions). Returns (grad_lam, grad_mu, grad_lam2mu, grad_inv_M,
// grad_w_rho_J, grad_wavelet).
void elastic_cpml_backward(const SemElasticCpmlBackwardInputCore& in);

// P4: coupled fluid-solid SEM forward (bare or dual-CPML via has_cpml). Fills
// (seis_chi, seis_x, seis_z) and returns them; optional save-all histories.
void fluid_solid_forward(const SemFluidSolidForwardInputCore& in);

// P4: coupled fluid-solid SEM backward. Returns (grad_ac_inv_rho,
// grad_ac_inv_M, grad_ac_kappa, grad_el_lam, grad_el_mu, grad_el_lam2mu,
// grad_el_inv_M, grad_el_w_rho_J, grad_wavelet).
void fluid_solid_backward(const SemFluidSolidBackwardInputCore& in);

} // namespace sem2d

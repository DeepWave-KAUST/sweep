#pragma once
#include "../../../core/device.h"
#include "../../../core/sem_input_core.h"

namespace sem3d {

// P5: 3-D acoustic SEM forward (bare, velocity form, Hex8). Runs the NSTEP
// Newmark loop on device over the Python-allocated buffers, fills in.seis and
// returns it; save_wavefield optionally records u^{n+1} per step.
void acoustic3d_forward(const Sem3dForwardInputCore& in);

// P5: 3-D acoustic SEM backward (adjoint-state, save-all; K self-adjoint).
// Returns (grad_c2_jx, grad_c2_jy, grad_c2_jz, grad_inv_M, grad_wavelet).
void acoustic3d_backward(const Sem3dBackwardInputCore& in);

// P6: 3-D acoustic C-PML forward / backward (Komatitsch-Martin psi/zeta,
// 3-axis). Backward returns (grad_vp2_node, grad_wavelet).

// DD stepped entry points (see forward.cu / backward.cu): one time step split
// around the Python-side interface-plane exchanges. batch == 1, save-all.
void acoustic3d_cpml_dd_asm(const Sem3dCpmlForwardInputCore& in);
void acoustic3d_cpml_dd_step(const Sem3dCpmlForwardInputCore& in, int64_t n);
void acoustic3d_cpml_dd_bwd_pre(const Sem3dCpmlBackwardInputCore& in, int64_t k);
void acoustic3d_cpml_dd_bwd_mid(const Sem3dCpmlBackwardInputCore& in, int64_t k);
void acoustic3d_cpml_dd_bwd_end(const Sem3dCpmlBackwardInputCore& in, int64_t k);

// P7: 3-D elastic (bare). Backward returns
// (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
void elastic3d_forward(const Sem3dElasticForwardInputCore& in);
void elastic3d_backward(const Sem3dElasticBackwardInputCore& in);

// P8: 3-D velocity-stress elastic (bare/CPML). Backward returns
// (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
void elastic_vs3d_forward(const Sem3dVsForwardInputCore& in);
void elastic_vs3d_backward(const Sem3dVsBackwardInputCore& in);

} // namespace sem3d

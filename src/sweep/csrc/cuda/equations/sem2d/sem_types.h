// The torch-typed SPEC of these structs: utils/gen_input_core.py turns it into the
// torch-free twins the drivers take (core/sem_input_core.h), their ctypes mirror
// and the layout probe; it is not compiled into the core.  Add fields here (never
// rename or remove one) and re-run the generator.
//
// ABI structs for the sweep.sem CUDA (impl='c') backend.
//
// Carries the eager SEM equation's already-built device constants (mesh metric,
// derivative matrix, lumped inverse mass, material) plus the Python-allocated
// wavefield/output buffers, in sweep.sem-native (E, ngll, ...) layout — NO
// 32-padding, NO 1-based ibool, NO axis transpose. Scalars `ndim`/`ngll` are
// carried explicitly so the same struct serves a later 3-D backend.
#pragma once

#include <torch/extension.h>

// Forward input for the 2-D acoustic-potential SEM step (Phase 0, forward-only).
// All tensors are on the same CUDA device; float32 (per the fp32-only decision),
// except the index tensors (ibool, rec_G, dirichlet_idx) which are int64.
struct SemForwardInput {
    // --- mesh + operator constants (from the eager equation's build()) --------
    torch::Tensor ibool;          // (E, ngll, ngll) int64, 0-based, (a=eta,b=xi)
    torch::Tensor D;              // (ngll, ngll) f32 derivative matrix D[i,j]=l'_j(xi_i)
    torch::Tensor inv_rho;        // (E, ngll, ngll) f32  1/rho at GLL
    torch::Tensor wJ;             // (E, ngll, ngll) f32  w_a w_b |J|
    torch::Tensor ixx, ixz, iex, iez;  // (E, ngll, ngll) f32  inverse-Jacobian metric
    torch::Tensor inv_M;          // (n_global,) f32  reciprocal lumped mass (masked)
    torch::Tensor mask_e;         // (E,) f32  element (domain) mask, 1 = active
    torch::Tensor dirichlet_idx;  // (k,) int64  free-surface pinned nodes (may be empty)

    // --- source / receiver ----------------------------------------------------
    torch::Tensor src_pattern;    // (n_global,) f32  pre-scattered source spatial pattern
    torch::Tensor wavelet;        // (nt,) f32  shared source time function
    torch::Tensor rec_G;          // (n_rec, ngll*ngll) int64  receiver gather node indices
    torch::Tensor rec_W;          // (n_rec, ngll*ngll) f32    receiver Lagrange weights

    // --- Python-allocated buffers (in/out; the .cu binds, never cudaMallocs) ---
    torch::Tensor chi;            // (n_global,) f32  wavefield now   (zeroed)
    torch::Tensor chi_prev;       // (n_global,) f32  wavefield prev  (zeroed)
    torch::Tensor chi_next;       // (n_global,) f32  scratch/next    (zeroed)
    torch::Tensor K;              // (n_global,) f32  stiffness accumulator scratch
    torch::Tensor seis;           // (n_rec, nt) f32  output seismogram (filled)
    torch::Tensor save_wavefield; // (nt, n_global) f32, OPTIONAL: chi_next per step
                                  // (save-all forward wavefield for the backward pass;
                                  //  empty -> forward-only, nothing saved)

    // --- scalars --------------------------------------------------------------
    double dt = 0.0;
    int nt = 0;
    int ngll = 0;
    int n_elements = 0;
    int n_global = 0;
    int n_rec = 0;
    int ndim = 2;
};

// Backward input for the 2-D acoustic-potential SEM adjoint (Phase 1, save-all).
// Reuses the forward constants; adds the saved forward wavefield and upstream
// grad_syn; carries Python-allocated adjoint/scratch buffers and the (zeroed)
// gradient outputs. Returns (grad_inv_M, grad_inv_rho, grad_wavelet).
struct SemBackwardInput {
    // --- constants (same layout as forward) -----------------------------------
    torch::Tensor ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, inv_M, mask_e;
    torch::Tensor dirichlet_idx;
    torch::Tensor src_pattern;    // (n_global,)
    torch::Tensor wavelet;        // (nt,)
    torch::Tensor rec_G, rec_W;   // (n_rec, ngll*ngll)

    // --- forward record + upstream grad ---------------------------------------
    torch::Tensor save_wavefield; // (nt, n_global) f32  u^1..u^nt from the forward
    torch::Tensor grad_syn;       // (n_rec, nt) f32  dL/dseis

    // --- Python-allocated scratch buffers (bound, not cudaMalloc'd) -----------
    torch::Tensor a_p1, a_p2, a_m;// (n_global,) adjoint states (3-way ping-pong; zeroed)
    torch::Tensor tmp;            // (n_global,) scratch (inv_M o a, and y)
    torch::Tensor Ktmp;           // (n_global,) stiffness scratch
    torch::Tensor Ku;             // (n_global,) stiffness of forward field scratch
    torch::Tensor adj_src;        // (n_global,) adjoint-source scratch (zeroed per step)

    // --- gradient outputs (Python-allocated, pre-zeroed) ----------------------
    torch::Tensor grad_inv_M;     // (n_global,)
    torch::Tensor grad_inv_rho;   // (E, ngll, ngll)
    torch::Tensor grad_wavelet;   // (nt,)

    // --- scalars --------------------------------------------------------------
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
};

// Forward input for the 2-D elastic SEM step (P2, bare, free surface = natural).
// 2-component displacement; material is lam/mu/lam2mu (built in Python from
// vp/vs/rho so PyTorch chains gradients back to those leaves).
struct SemElasticForwardInput {
    torch::Tensor ibool, D, wJ, ixx, ixz, iex, iez, mask_e;   // geometry
    torch::Tensor lam, mu, lam2mu;                            // material (E,ngll,ngll)
    torch::Tensor inv_M;                                      // (n_global,)
    torch::Tensor pat_x, pat_z;                               // source patterns (n_global,)
    torch::Tensor wavelet;                                    // (nt,)
    torch::Tensor rec_G, rec_W;                               // (n_rec, ngll*ngll)
    // buffers (Python-allocated, bound)
    torch::Tensor ux, uz, ux_prev, uz_prev, ux_next, uz_next, Kux, Kuz;
    torch::Tensor seis_x, seis_z;                             // (n_rec, nt) outputs
    torch::Tensor save_ux, save_uz;                           // (nt, n_global) OPTIONAL for backward
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
};

// Forward input for the 2-D acoustic-potential SEM step WITH C-PML (P3a).
// Komatitsch-Martin double convolution; carries the psi/zeta memory state.
struct SemAcousticCpmlForwardInput {
    torch::Tensor ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, mask_e;
    torch::Tensor w1d;   // (ngll,) 1-D GLL weights; w_J is derived
    torch::Tensor kappa_node, inv_M, inv_Mg;              // node kappa, mass, geo-mass
    torch::Tensor a_x, b_x, dbx_dx, a_z, b_z, dbz_dz;     // CPML coeffs (n_global,)
    torch::Tensor src_pattern, wavelet, rec_G, rec_W, dirichlet_idx;
    // buffers (Python-allocated)
    torch::Tensor chi, chi_prev, chi_next;
    torch::Tensor psi_x, psi_z, zeta_x, zeta_z;           // memory state (read)
    torch::Tensor psi_x_n, psi_z_n, zeta_x_n, zeta_z_n;   // memory state (write, ping-pong)
    torch::Tensor apsi_x, apsi_z, w_sum;                  // scratch
    torch::Tensor gx, gz, kx, kz, gax, gaz;               // scratch (assembled grads)
    torch::Tensor seis;                                  // (n_rec, nt) output
    torch::Tensor save_wavefield;                        // (nt, n_global) OPTIONAL: chi_next
    torch::Tensor save_psi_x, save_psi_z;                // (nt, n_global) OPTIONAL: psi_?_n
    torch::Tensor save_zeta_x, save_zeta_z;              // (nt, n_global) OPTIONAL: zeta_?_n
    // save_w_sum (nt, B, n_global) OPTIONAL: w_sum^n of forward step n (the
    // `ws` register of step_newmark_f, stored through a save slice like the
    // five fields above; NOT re-zeroed on the Dirichlet nodes -- the row must
    // hold what the forward actually used, and every consumer multiplies it
    // by an adjoint the backward zeroes there first). The backward reads
    // psi^k / zeta^k ONLY to rebuild w_sum^k, and chi^k is the one forward
    // field its adjoint element kernel images from (grad_inv_rho), so with
    // this tape the drivers take save_wavefield + save_w_sum and leave the
    // four psi/zeta tapes empty. The entries key off numel(). Additive.
    torch::Tensor save_w_sum;
    // boundary-saving mode (bs=1): per-step CPML-band histories instead of the
    // full save-all fields, + the final two chi levels for the reverse march.
    torch::Tensor band_idx;                              // (nband,) int64
    torch::Tensor band_chi, band_psi_x, band_psi_z;      // (nt, nband)
    torch::Tensor band_zeta_x, band_zeta_z;              // (nt, nband)
    // band_split=1: psi_d / zeta_d ring on band_idx_d (band nodes with
    // b_d != 0) instead of band_idx; chi stays on band_idx. Additive.
    int band_split = 0;
    torch::Tensor band_idx_x, band_idx_z;
    torch::Tensor fin_chi, fin_chi_prev;                 // (n_global,) chi^nt, chi^{nt-1}
    // ring storage precision: 0 fp32, 1 bf16, 2 int8. int8 also needs
    // band_scale (5, nt) fp32 -- one amax per (field, step).
    int band_kind = 0;
    torch::Tensor band_scale;
    int bs = 0;
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot batch (additive wire fields). B > 1 runs B shots in ONE kernel
    // sequence: every field/scratch buffer, src_pattern and the save histories
    // gain a leading B (histories are (nt, B, ng)), seis becomes (B, n_rec,
    // nt). Bm is the MODEL batch: Bm == 1 broadcasts one material set over all
    // shots (stride-0 reads, the "same model, different sources" case), Bm ==
    // B gives every shot its own inv_rho/kappa/inv_M/inv_Mg. The C-PML profile
    // and the mesh geometry are per-mesh, never per-shot.
    int B = 1, Bm = 1;
};

// Backward input for the 2-D acoustic-potential SEM adjoint WITH C-PML (P3b).
// save-all of the 5 state histories (chi,psi_x,psi_z,zeta_x,zeta_z at each step).
// Returns (grad_inv_M, grad_inv_rho, grad_kappa, grad_inv_Mg, grad_wavelet).
struct SemAcousticCpmlBackwardInput {
    torch::Tensor ibool, D, inv_rho, wJ, ixx, ixz, iex, iez, mask_e;
    torch::Tensor w1d;   // (ngll,) 1-D GLL weights; w_J is derived
    torch::Tensor kappa_node, inv_M, inv_Mg;
    torch::Tensor a_x, b_x, dbx_dx, a_z, b_z, dbz_dz;
    torch::Tensor src_pattern, wavelet, rec_G, rec_W, dirichlet_idx;
    // saved forward state histories (output state at each step; state^k = save[k-1], ^0=0)
    torch::Tensor save_chi, save_psi_x, save_psi_z, save_zeta_x, save_zeta_z;
    // see the forward struct. Non-empty AND bs == 0 => the w_sum-tape path:
    // reverse step k >= 1 rebuilds nothing (no g6 memset, no grads_f, no
    // psi/zeta read; step_T<WTAPE=true> runs the adjoint algebra alone) and
    // combine_f reads row k. grad_inv_Mg is then never accumulated (it needs
    // the rebuilt assembly) and stays at its pre-zeroed value; both drivers
    // discard it. Under bs the field is ignored: the reverse march needs
    // w_sum^k rebuilt on the fly. Additive.
    torch::Tensor save_w_sum;
    // boundary-saving inputs (bs=1): band histories + final chi levels; the
    // full chi/psi/zeta states are reconstructed on the fly (reverse leapfrog
    // in the reversible interior + exact band restore).
    torch::Tensor band_idx;
    torch::Tensor band_chi, band_psi_x, band_psi_z, band_zeta_x, band_zeta_z;
    torch::Tensor fin_chi, fin_chi_prev;
    int band_split = 0;                                  // see the forward struct
    torch::Tensor band_idx_x, band_idx_z;
    int band_kind = 0;                                   // 0 fp32, 1 bf16, 2 int8
    torch::Tensor band_scale;                            // (5, nt) fp32, int8 only
    torch::Tensor chi_hi, chi_lo, chi_km1;               // rolling chi levels
    torch::Tensor r_psi_x, r_psi_z, r_zeta_x, r_zeta_z;  // rebuilt memory fields
    int bs = 0;
    torch::Tensor grad_syn;                              // (n_rec, nt)
    // adjoint state buffers (Python-allocated, zeroed)
    torch::Tensor gA, gB, gchi_k;                        // chi adjoint (2-history + work)
    torch::Tensor lpsi_x, lpsi_z, lzeta_x, lzeta_z;      // memory adjoints (carried)
    torch::Tensor lpsi_x_n, lpsi_z_n, lzeta_x_n, lzeta_z_n;  // memory adjoints (write)
    // per-step recompute + transpose scratch
    torch::Tensor apx, apz, gx, gz, kx, kz, gax, gaz;    // forward recompute
    torch::Tensor w_sum, gw;                             // forward w_sum + its adjoint
    torch::Tensor gbx, gbz, gbkx, gbkz, gbax, gbaz;      // adjoint of the 6 grad fields
    torch::Tensor lchi_G, lapx, lapz;                    // G^T outputs
    // gradient outputs (pre-zeroed). Under B > 1 they are PER-SHOT -- shaped
    // (B, ...) -- regardless of Bm; a shared model's gradient is the Python
    // side's sum over the leading axis. Keeping the kernels' plain +=
    // accumulation (no cross-shot atomics) preserves the B == 1 bit-path.
    torch::Tensor grad_inv_M, grad_inv_rho, grad_kappa, grad_inv_Mg, grad_wavelet;
    // Domain decomposition only (additive): ownership-masked source pattern
    // for the WAVELET gradient. Interface columns hold complete values on
    // BOTH ranks after the exchange, so the per-rank grad_wavelet reduction
    // would double-count them under the cross-rank all-reduce; this copy
    // zeroes the non-owned interface column. Empty = use src_pattern.
    torch::Tensor src_pattern_own;
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot/model batch, same semantics as the forward struct (additive).
    int B = 1, Bm = 1;
};

// Forward input for the 2-D elastic SEM WITH SPECFEM2D-style 2nd-order C-PML
// (P3e). All CPML coefficient fields are element-local (E, ngll, ngll); the 6
// memory fields are (E, ngll, ngll, 2) state carried across steps. The
// optional save buffers additionally record the 4 stretched derivatives and
// the 2 acceleration paren terms per step for the backward imaging pass.
struct SemElasticCpmlForwardInput {
    torch::Tensor ibool, D, wJ, w_rho_J, ixx, ixz, iex, iez, mask_e;  // geometry
    torch::Tensor lam, mu, lam2mu;                            // material (E,ngll,ngll)
    torch::Tensor inv_M;                                      // (n_global,)
    // ElasticCPMLProfile, flattened (all (E, ngll, ngll) f32)
    torch::Tensor Sx0, Sx1, Sx2, zx1_0, zx1_1, zx1_2, zx2_0, zx2_1, zx2_2;
    torch::Tensor Sz0, Sz1, Sz2, xz1_0, xz1_1, xz1_2, xz2_0, xz2_1, xz2_2;
    torch::Tensor La1, La2, La3, La4, l1_0, l1_1, l1_2, l2_0, l2_1, l2_2;
    torch::Tensor outer_edge_idx;                             // (k,) int64 rigid wall
    torch::Tensor pat_x, pat_z, wavelet, rec_G, rec_W;        // source / receivers
    // state (Python-allocated, zeroed): displacements + 6 CPML memories
    torch::Tensor ux, uz, ux_prev, uz_prev, ux_next, uz_next, Acc_x, Acc_z;
    torch::Tensor r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz;     // (E,ngll,ngll,2)
    torch::Tensor r_displ_x, r_displ_z;                       // (E,ngll,ngll,2)
    torch::Tensor seis_x, seis_z;                             // (n_rec, nt) outputs
    // OPTIONAL save-all for backward (empty = skip)
    torch::Tensor save_ux, save_uz;                           // (nt, n_global)
    torch::Tensor save_sd_xx, save_sd_zx, save_sd_xz, save_sd_zz;  // (nt, E*ngll*ngll)
    torch::Tensor save_apx, save_apz;                         // (nt, E*ngll*ngll)
    // OPTIONAL boundary saving (bs=1): the save_* arrays above stay empty and
    // only the C-PML band is stored per step -- the displacement on band NODES
    // and the six element-local histories on band ELEMENTS. Outside the band
    // the stretched derivatives equal the raw ones and aPML is 0, so backward
    // recomputes the interior from the reconstructed displacement.
    torch::Tensor band_node_idx, band_elem_idx;                    // int64 (nbn,) (nbe,)
    // (E,) int32, 1 where the element touches the C-PML band. Outside it
    // every C-PML coefficient is exactly 0 and the stretch exactly 1
    // (measured), so an interior block runs the bare weak form. An
    // EMPTY tensor means every element is treated as band, which is
    // exactly the pre-flag behaviour.
    torch::Tensor elem_band;
    torch::Tensor band_ux, band_uz;                                // (nt, nbn)
    torch::Tensor band_sd_xx, band_sd_zx, band_sd_xz, band_sd_zz;  // (nt, nbe*m)
    torch::Tensor band_apx, band_apz;                              // (nt, nbe*m)
    torch::Tensor sd_scr_xx, sd_scr_zx, sd_scr_xz, sd_scr_zz;      // (E*m) per-step
    torch::Tensor ap_scr_x, ap_scr_z;                              // (E*m)
    torch::Tensor fin_ux, fin_uz, fin_ux_prev, fin_uz_prev;        // (n_global,)
    int bs = 0;
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot batch (additive; same semantics as the fluid-solid structs): B > 1
    // runs B shots in one kernel sequence over ONE shared material set --
    // fields/memories/histories/rings gain a leading B, pat_x/pat_z are
    // (B, ng), each seis is (B, n_rec, nt), the wavelet is (nt,) or (B, nt).
    int B = 1;
    // ring storage precision (additive): 0 fp32, 1 bf16, 2 int8.
    // band_scale is (n_fields, nt) fp32, int8 only -- one amax per
    // (field, step); the ELEMENT rings quantise through the same
    // contract as the node rings, via sem2d_band_save/load_elem.
    int band_kind = 0;
    torch::Tensor band_scale;
    // the ELEMENT rings keep one int8 amax per band element:
    // (n_elem_fields, nt, B, nbe) fp32, int8 only.
    torch::Tensor band_scale_e;
    // ELEMENT rings take their OWN kind. Bisecting the ring families
    // showed the elastic int8 cost lives entirely in the u NODE rings
    // (2.5e-2) while all six element rings together cost 3.6e-4 -- and
    // the element rings are 82% of the ring memory. Splitting the two
    // buys most of the memory at the cheap error. Defaults to
    // band_kind, so an unset field keeps the old all-or-nothing.
    int band_kind_e = 0;
};

// Forward input for the 2-D coupled fluid-solid SEM (P4): SPECFEM potential
// fluid + 2nd-order displacement solid on ONE mesh, interface quadrature
// (node_idx/weight/normal per face-GLL point), optional C-PML on BOTH domains
// (has_cpml=1: acoustic psi/zeta double convolution + elastic 6-memory CPML;
// has_cpml=0: bare stiffness both sides, natural solid edges). The load-
// bearing ordering matches AcousticElasticSEM2D.step: fluid residual -> chi_dd
// (post-inversion) -> dirichlet(chi_dd) -> S^el(-chi_dd) -> solid step ->
// fluid leapfrog + dirichlet.
struct SemFluidSolidForwardInput {
    torch::Tensor ibool, D, wJ, ixx, ixz, iex, iez;
    torch::Tensor w1d;   // (ngll,) 1-D GLL weights; w_J is derived           // shared geometry
    // fluid (acoustic potential)
    torch::Tensor ac_inv_rho, ac_mask_e, ac_kappa, ac_inv_M, ac_inv_Mg;
    torch::Tensor a_x, b_x, dbx_dx, a_z, b_z, dbz_dz;         // empty when bare
    torch::Tensor dirichlet_idx;
    // solid (elastic) + its CPML profile (empty tensors when bare)
    torch::Tensor el_lam, el_mu, el_lam2mu, el_inv_M, el_mask_e, el_w_rho_J;
    torch::Tensor Sx0, Sx1, Sx2, zx1_0, zx1_1, zx1_2, zx2_0, zx2_1, zx2_2;
    torch::Tensor Sz0, Sz1, Sz2, xz1_0, xz1_1, xz1_2, xz2_0, xz2_1, xz2_2;
    torch::Tensor La1, La2, La3, La4, l1_0, l1_1, l1_2, l2_0, l2_1, l2_2;
    torch::Tensor outer_edge_idx;
    // interface quadrature, flat (n_face*ngll,)
    torch::Tensor ifc_idx, ifc_w, ifc_nx, ifc_nz;
    // sources / receivers
    torch::Tensor pat_chi, pat_x, pat_z, wavelet, rec_G, rec_W;
    // fluid state + scratch
    torch::Tensor chi, chi_prev, chi_next, s_ac, chi_dd, Kchi;
    torch::Tensor psi_x, psi_z, zeta_x, zeta_z, psi_x_n, psi_z_n, zeta_x_n, zeta_z_n;
    torch::Tensor apsi_x, apsi_z, w_sum, gx, gz, kx, kz, gax, gaz;
    // solid state + scratch
    torch::Tensor ux, uz, ux_prev, uz_prev, ux_next, uz_next, Acc_x, Acc_z, f_x, f_z;
    torch::Tensor r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z;
    // outputs
    torch::Tensor seis_chi, seis_x, seis_z;                   // (n_rec, nt)
    // OPTIONAL save-all (empty = skip)
    torch::Tensor save_chi, save_psi_x, save_psi_z, save_zeta_x, save_zeta_z;
    torch::Tensor save_ux, save_uz;
    torch::Tensor save_sd_xx, save_sd_zx, save_sd_xz, save_sd_zz, save_apx, save_apz;
    // OPTIONAL boundary saving (bs=1, requires has_cpml): BOTH domains'
    // C-PML rings -- the acoustic (chi, psi, zeta) on band NODES like the
    // standalone potential path, the elastic u on band nodes + the six
    // element-local histories on band ELEMENTS like the standalone elastic
    // path -- plus the final two leapfrog levels of each field to seed the
    // reverse march, plus FULL-STATE re-anchor checkpoints every ckpt_every
    // steps: the coupled reverse march accumulates error the acoustic and
    // elastic ones do not (measured; see the numbers on
    // acoustic_elastic2d.bs_frame_checkpoint_every). fp32 rings only.
    int bs = 0;
    torch::Tensor band_idx;                                  // acoustic nodes
    torch::Tensor band_chi, band_psi_x, band_psi_z;          // (nt, nband)
    torch::Tensor band_zeta_x, band_zeta_z;
    torch::Tensor fin_chi, fin_chi_prev;
    torch::Tensor band_node_idx, band_elem_idx;              // elastic
    // (E,) int32, 1 where the element touches the C-PML band. Outside it
    // every C-PML coefficient is exactly 0 and the stretch exactly 1
    // (measured), so an interior block runs the bare weak form. An
    // EMPTY tensor means every element is treated as band, which is
    // exactly the pre-flag behaviour.
    torch::Tensor elem_band;
    torch::Tensor band_ux, band_uz;                          // (nt, nbn)
    torch::Tensor band_sd_xx, band_sd_zx, band_sd_xz, band_sd_zz;  // (nt, nbe*m)
    torch::Tensor band_apx, band_apz;
    torch::Tensor sd_scr_xx, sd_scr_zx, sd_scr_xz, sd_scr_zz;      // (E*m)
    torch::Tensor ap_scr_x, ap_scr_z;
    torch::Tensor fin_ux, fin_uz, fin_ux_prev, fin_uz_prev;
    int ckpt_every = 0;
    torch::Tensor ck_chi, ck_chi_prev;                       // (nc, n_global)
    torch::Tensor ck_ux, ck_uz, ck_ux_prev, ck_uz_prev;
    double dt = 0.0;
    int has_cpml = 0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot batch (additive wire field). B > 1 runs B shots in ONE kernel
    // sequence: every field/scratch/memory buffer, the three source patterns
    // and the histories gain a leading B (save-all (nt, B, ng) / (nt, B, E*m),
    // rings (nt, B, nband)-style, checkpoints (nc, B, ng)), each seis becomes
    // (B, n_rec, nt), the wavelet is (nt,) shared or (B, nt). The MODEL is
    // always shared across the batch (no Bm here -- the coupled path is the
    // FWI inner loop, where one model serves every shot); material gradients
    // come out per-shot (B, ...) and the caller sums. Requires has_cpml.
    int B = 1;
    // ring storage precision (additive): 0 fp32, 1 bf16, 2 int8.
    // band_scale is (7, nt, B) -- one amax per (field, step) for the
    // five fluid node rings (slots 0..4) and the two solid ones
    // (5,6). band_scale_e is (6, nt, B, nbe): the ELEMENT rings get
    // one amax per BAND ELEMENT, their range being far wider.
    // The fin_*/ck_* full-state buffers stay fp32 -- they are the
    // march's seeds and re-anchors, not ring rows.
    int band_kind = 0;
    torch::Tensor band_scale, band_scale_e;
    // ELEMENT rings take their OWN kind. Bisecting the ring families
    // showed the elastic int8 cost lives entirely in the u NODE rings
    // (2.5e-2) while all six element rings together cost 3.6e-4 -- and
    // the element rings are 82% of the ring memory. Splitting the two
    // buys most of the memory at the cheap error. Defaults to
    // band_kind, so an unset field keeps the old all-or-nothing.
    int band_kind_e = 0;
};

// Backward input for the coupled fluid-solid adjoint (P4). Interleaves the
// P3b fluid recursion (gA/gB + psi/zeta adjoints) with the P3e solid recursion
// (g/gprev + 6 memory adjoints); the coupling transposes reuse the interface
// kernels (Ifc_el^T = Ifc_ac and vice versa). Returns
// (grad_ac_inv_rho, grad_ac_inv_M, grad_ac_kappa, grad_el_lam, grad_el_mu,
//  grad_el_lam2mu, grad_el_inv_M, grad_el_w_rho_J, grad_wavelet).
struct SemFluidSolidBackwardInput {
    torch::Tensor ibool, D, wJ, ixx, ixz, iex, iez;
    torch::Tensor w1d;   // (ngll,) 1-D GLL weights; w_J is derived
    torch::Tensor ac_inv_rho, ac_mask_e, ac_kappa, ac_inv_M, ac_inv_Mg;
    torch::Tensor a_x, b_x, dbx_dx, a_z, b_z, dbz_dz, dirichlet_idx;
    torch::Tensor el_lam, el_mu, el_lam2mu, el_inv_M, el_mask_e, el_w_rho_J;
    torch::Tensor Sx0, Sx1, Sx2, zx1_0, zx1_1, zx1_2, zx2_0, zx2_1, zx2_2;
    torch::Tensor Sz0, Sz1, Sz2, xz1_0, xz1_1, xz1_2, xz2_0, xz2_1, xz2_2;
    torch::Tensor La1, La2, La3, La4, l1_0, l1_1, l1_2, l2_0, l2_1, l2_2;
    torch::Tensor outer_edge_idx, ifc_idx, ifc_w, ifc_nx, ifc_nz;
    torch::Tensor pat_chi, pat_x, pat_z, wavelet, rec_G, rec_W;
    // saved forward histories (as written by fluid_solid_forward)
    torch::Tensor save_chi, save_psi_x, save_psi_z, save_zeta_x, save_zeta_z;
    torch::Tensor save_ux, save_uz;
    torch::Tensor save_sd_xx, save_sd_zx, save_sd_xz, save_sd_zz, save_apx, save_apz;
    torch::Tensor grad_schi, grad_sx, grad_sz;               // (n_rec, nt) upstream
    // fluid adjoint work (all (n_global,), zeroed)
    torch::Tensor gA, gB, gchi_k, lchidd, s_ac, Kchi, yf, zeros;
    torch::Tensor lpsi_x, lpsi_z, lzeta_x, lzeta_z;
    torch::Tensor lpsi_x_n, lpsi_z_n, lzeta_x_n, lzeta_z_n;
    torch::Tensor apx, apz, gx, gz, kx, kz, gax, gaz, w_sum, gw;
    torch::Tensor gbx, gbz, gbkx, gbkz, gbax, gbaz, lchi_G, lapx, lapz;
    // solid adjoint work
    torch::Tensor lam1_x, lam1_z, lamn_x, lamn_z, g_x, g_z, gprev_x, gprev_z;
    torch::Tensor y_x, y_z, adj_x, adj_z, tmp_x, tmp_z;
    torch::Tensor sc_now_x, sc_now_z, sc_prev_a_x, sc_prev_a_z, sc_prev_b_x, sc_prev_b_z;
    torch::Tensor mu_xx, mu_zx, mu_xz, mu_zz, mu_dx, mu_dz;  // (E,ngll,ngll,2)
    // outputs (pre-zeroed)
    torch::Tensor grad_ac_inv_rho, grad_ac_inv_M, grad_ac_kappa, grad_ac_inv_Mg;
    torch::Tensor grad_el_lam, grad_el_mu, grad_el_lam2mu, grad_el_inv_M;
    torch::Tensor grad_el_w_rho_J, grad_wavelet;
    // OPTIONAL boundary saving (bs=1): ring inputs (as written by the
    // forward) + reverse-march state. See the forward struct's note.
    int bs = 0;
    torch::Tensor band_idx, band_chi, band_psi_x, band_psi_z;
    torch::Tensor band_zeta_x, band_zeta_z, fin_chi, fin_chi_prev;
    torch::Tensor chi_hi, chi_lo, chi_km1, rec_chi_dd;       // fluid march
    torch::Tensor r_psi_x, r_psi_z, r_zeta_x, r_zeta_z;      // rebuilt memories
    torch::Tensor band_node_idx, band_elem_idx, band_ux, band_uz;
    // (E,) int32, 1 where the element touches the C-PML band. Outside it
    // every C-PML coefficient is exactly 0 and the stretch exactly 1
    // (measured), so an interior block runs the bare weak form. An
    // EMPTY tensor means every element is treated as band, which is
    // exactly the pre-flag behaviour.
    torch::Tensor elem_band;
    torch::Tensor band_sd_xx, band_sd_zx, band_sd_xz, band_sd_zz;
    torch::Tensor band_apx, band_apz;
    torch::Tensor fin_ux, fin_uz, fin_ux_prev, fin_uz_prev;
    torch::Tensor u_hi_x, u_hi_z, u_lo_x, u_lo_z, u_km1_x, u_km1_z;
    torch::Tensor rec_sd_xx, rec_sd_zx, rec_sd_xz, rec_sd_zz;      // (E*m)
    torch::Tensor rec_apx, rec_apz;
    torch::Tensor rec_rxx, rec_rxz, rec_rzx, rec_rzz, rec_rdx, rec_rdz;  // (E*m*2)
    torch::Tensor rec_Acc_x, rec_Acc_z, rec_f_x, rec_f_z;
    int ckpt_every = 0;
    torch::Tensor ck_chi, ck_chi_prev, ck_ux, ck_uz, ck_ux_prev, ck_uz_prev;
    double dt = 0.0;
    int has_cpml = 0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot batch, same semantics as the forward struct (additive). grad_schi/
    // grad_sx/grad_sz are (B, n_rec, nt); every gradient output is per-shot
    // (B, ...) and the caller sums for the (always-shared) model.
    int B = 1;
    // Domain decomposition only (additive): ownership-masked source patterns
    // for the WAVELET gradient (a global reduction, and an interface column
    // is complete on BOTH strips after the adjoint exchange). Empty = use
    // pat_chi / pat_x / pat_z.
    torch::Tensor pat_chi_own, pat_x_own, pat_z_own;
    // ring storage precision (additive): 0 fp32, 1 bf16, 2 int8.
    // band_scale is (7, nt, B) -- one amax per (field, step) for the
    // five fluid node rings (slots 0..4) and the two solid ones
    // (5,6). band_scale_e is (6, nt, B, nbe): the ELEMENT rings get
    // one amax per BAND ELEMENT, their range being far wider.
    // The fin_*/ck_* full-state buffers stay fp32 -- they are the
    // march's seeds and re-anchors, not ring rows.
    int band_kind = 0;
    torch::Tensor band_scale, band_scale_e;
    // ELEMENT rings take their OWN kind. Bisecting the ring families
    // showed the elastic int8 cost lives entirely in the u NODE rings
    // (2.5e-2) while all six element rings together cost 3.6e-4 -- and
    // the element rings are 82% of the ring memory. Splitting the two
    // buys most of the memory at the cheap error. Defaults to
    // band_kind, so an unset field keeps the old all-or-nothing.
    int band_kind_e = 0;
};

// Backward input for the 2-D elastic SEM adjoint WITH C-PML (P3e). The mu_*
// buffers carry the memory-field adjoints across reverse steps; g/gprev carry
// the pin-masked displacement adjoints; sc_prev_a/b double-buffer the
// "previous-step" scatter contributions. Returns
// (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_w_rho_J, grad_wavelet).
struct SemElasticCpmlBackwardInput {
    torch::Tensor ibool, D, wJ, w_rho_J, ixx, ixz, iex, iez, mask_e;
    torch::Tensor lam, mu, lam2mu, inv_M;
    torch::Tensor Sx0, Sx1, Sx2, zx1_0, zx1_1, zx1_2, zx2_0, zx2_1, zx2_2;
    torch::Tensor Sz0, Sz1, Sz2, xz1_0, xz1_1, xz1_2, xz2_0, xz2_1, xz2_2;
    torch::Tensor La1, La2, La3, La4, l1_0, l1_1, l1_2, l2_0, l2_1, l2_2;
    torch::Tensor outer_edge_idx, pat_x, pat_z, rec_G, rec_W;
    torch::Tensor pin_mask;  // (n_global,) 0/1, outer_edge_idx as a per-node mask
    torch::Tensor wavelet;   // needed by the BS reverse march (source term)
    // saved forward histories
    torch::Tensor save_ux, save_uz;                                   // (nt, n_global)
    torch::Tensor save_sd_xx, save_sd_zx, save_sd_xz, save_sd_zz;     // (nt, E*m)
    torch::Tensor save_apx, save_apz;                                 // (nt, E*m)
    torch::Tensor grad_sx, grad_sz;                                   // (n_rec, nt)
    // work buffers (Python-allocated, zeroed)
    torch::Tensor lam1_x, lam1_z, lamn_x, lamn_z;      // pinned-state adjoints
    torch::Tensor g_x, g_z, gprev_x, gprev_z;          // pin-masked adjoints
    torch::Tensor y_x, y_z, adj_x, adj_z;              // accel adjoint + adj source
    torch::Tensor sc_now_x, sc_now_z;                  // element scatter (this step)
    torch::Tensor sc_prev_a_x, sc_prev_a_z, sc_prev_b_x, sc_prev_b_z;
    torch::Tensor mu_xx, mu_zx, mu_xz, mu_zz, mu_dx, mu_dz;  // (E,ngll,ngll,2)
    // outputs (pre-zeroed)
    torch::Tensor grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_w_rho_J, grad_wavelet;
    // boundary saving (bs=1): band histories + the final two displacement levels
    // + reverse-march scratch. The march is swap2nd (feed u^{n+1} as the "prev"
    // level with the CPML memories zeroed): exact in the reversible interior,
    // wrong inside the band -- which is then overwritten from the band history.
    torch::Tensor band_node_idx, band_elem_idx;
    // (E,) int32, 1 where the element touches the C-PML band. Outside it
    // every C-PML coefficient is exactly 0 and the stretch exactly 1
    // (measured), so an interior block runs the bare weak form. An
    // EMPTY tensor means every element is treated as band, which is
    // exactly the pre-flag behaviour.
    torch::Tensor elem_band;
    torch::Tensor band_ux, band_uz;
    torch::Tensor band_sd_xx, band_sd_zx, band_sd_xz, band_sd_zz;
    torch::Tensor band_apx, band_apz;
    torch::Tensor fin_ux, fin_uz, fin_ux_prev, fin_uz_prev;
    torch::Tensor u_hi_x, u_hi_z, u_lo_x, u_lo_z, u_km1_x, u_km1_z;
    torch::Tensor rec_sd_xx, rec_sd_zx, rec_sd_xz, rec_sd_zz, rec_apx, rec_apz;
    torch::Tensor rec_rxx, rec_rxz, rec_rzx, rec_rzz, rec_rdx, rec_rdz;
    torch::Tensor rec_Acc_x, rec_Acc_z;
    int bs = 0;
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
    // Shot batch (additive; see the forward struct): grad_sx/grad_sz are
    // (B, n_rec, nt), every gradient output is per-shot (B, ...) and the
    // caller sums for the (always-shared) model.
    int B = 1;    // Domain decomposition only (additive): ownership-masked source patterns
    // for the WAVELET gradient. An interface column holds complete values on
    // BOTH strips after the adjoint exchange, so the per-rank reduction would
    // double-count it. Empty = use pat_x / pat_z.
    torch::Tensor pat_x_own, pat_z_own;
    // ring storage precision (additive): 0 fp32, 1 bf16, 2 int8.
    // band_scale is (n_fields, nt) fp32, int8 only -- one amax per
    // (field, step); the ELEMENT rings quantise through the same
    // contract as the node rings, via sem2d_band_save/load_elem.
    int band_kind = 0;
    torch::Tensor band_scale;
    // the ELEMENT rings keep one int8 amax per band element:
    // (n_elem_fields, nt, B, nbe) fp32, int8 only.
    torch::Tensor band_scale_e;
    // ELEMENT rings take their OWN kind. Bisecting the ring families
    // showed the elastic int8 cost lives entirely in the u NODE rings
    // (2.5e-2) while all six element rings together cost 3.6e-4 -- and
    // the element rings are 82% of the ring memory. Splitting the two
    // buys most of the memory at the cheap error. Defaults to
    // band_kind, so an unset field keeps the old all-or-nothing.
    int band_kind_e = 0;
};

// Backward input for the 2-D elastic SEM adjoint (P2, save-all). Returns
// (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
struct SemElasticBackwardInput {
    torch::Tensor ibool, D, wJ, ixx, ixz, iex, iez, mask_e;
    torch::Tensor lam, mu, lam2mu, inv_M;
    torch::Tensor pat_x, pat_z, wavelet;
    torch::Tensor rec_G, rec_W;
    torch::Tensor save_ux, save_uz;   // (nt, n_global) forward displacement history
    torch::Tensor grad_sx, grad_sz;   // (n_rec, nt) upstream
    torch::Tensor ax1, az1, ax2, az2, axm, azm;      // adjoint states (ping-pong)
    torch::Tensor tmp_x, tmp_z, Kx, Kz, adj_x, adj_z; // scratch
    torch::Tensor grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet;  // outputs
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0, ndim = 2;
};

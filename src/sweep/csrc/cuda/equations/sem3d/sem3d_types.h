// The torch-typed SPEC of these structs: utils/gen_input_core.py turns it into the
// torch-free twins the drivers take (core/sem_input_core.h), their ctypes mirror
// and the layout probe; it is not compiled into the core.  Add fields here (never
// rename or remove one) and re-run the generator.
//
// sweep.sem CUDA (impl='c') backend — 3-D acoustic SEM (Hex8, axis-aligned)
// I/O structs. All buffers are Python-allocated (hard rule). Unlike the 2-D
// structs these carry an explicit ``batch`` dim: field buffers are
// (batch, n_global) and the kernels launch grid (E, batch) — the driver runs
// batch=1 today, but every new kernel is batch-ready by design.
#pragma once
#include <torch/extension.h>

namespace sem3d {

// Forward input for the 3-D acoustic SEM (velocity form: c^2 baked into the
// separable stiffness weights c2_j{x,y,z} = vp^2 w3 G_{x,y,z}).
struct Sem3dForwardInput {
    torch::Tensor ibool;              // (E, ngll^3) int64, flattened (kz,ky,kx)
    torch::Tensor D;                  // (ngll, ngll) f32
    torch::Tensor c2_jx, c2_jy, c2_jz;  // (E, ngll^3) f32 stiffness weights
    torch::Tensor inv_M;              // (n_global,) f32
    torch::Tensor dirichlet_idx;      // (k,) int64 (may be empty)
    torch::Tensor src_pattern;        // (n_global,) f32
    torch::Tensor wavelet;            // (nt,) f32
    torch::Tensor rec_G;              // (n_rec, ngll^3) int64
    torch::Tensor rec_W;              // (n_rec, ngll^3) f32
    // state (batch, n_global), zeroed
    torch::Tensor u, u_prev, u_next, K;
    torch::Tensor seis;               // (n_rec, nt) f32 output
    torch::Tensor save_wavefield;     // (nt, batch, n_global) OPTIONAL
    torch::Tensor fin_u, fin_u_prev;  // (n_global,) OPTIONAL (bs=1)
    int bs = 0;
    // deformed=1: Hex27 topography. c2_node (single per-node c^2) + w_J +
    // the full per-node 3x3 inv_J drive sem3d_acoustic_stiffness_deformed;
    // c2_j{x,y,z} are ignored. Empty otherwise (axis-aligned Hex8).
    int deformed = 0;
    torch::Tensor c2_node, wJ;        // (E, ngll^3)
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;  // (E, ngll^3)
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// Backward input (save-all adjoint). Returns
// (grad_c2_jx, grad_c2_jy, grad_c2_jz, grad_inv_M, grad_wavelet); when
// deformed=1 the single c^2 grad is returned in grad_c2_jx (grad_c2_node).
struct Sem3dBackwardInput {
    torch::Tensor ibool, D, c2_jx, c2_jy, c2_jz, inv_M, dirichlet_idx;
    torch::Tensor src_pattern, wavelet, rec_G, rec_W;
    torch::Tensor save_wavefield;     // (nt, batch, n_global) from the forward
    torch::Tensor fin_u, fin_u_prev;  // (bs=1) final levels for the reverse march
    torch::Tensor u_hi, u_lo, u_km1;  // (bs=1) rolling reconstruction levels
    int bs = 0;
    int deformed = 0;
    torch::Tensor c2_node, wJ;        // (E, ngll^3)
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;  // (E, ngll^3)
    torch::Tensor grad_syn;           // (n_rec, nt)
    torch::Tensor a_p1, a_p2, a_m, tmp, Ktmp, Ku, adj_src;   // (batch, n_global)
    torch::Tensor grad_c2_jx, grad_c2_jy, grad_c2_jz;        // (E, ngll^3)
    torch::Tensor grad_inv_M;         // (n_global,)
    torch::Tensor grad_wavelet;       // (nt,)
    double dt = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// Forward input for the 3-D acoustic SEM WITH C-PML (P6). Node coefficient
// arrays a/b/db per axis, geometric weights w_jac / w_j{x,y,z}, vp2_node the
// per-node c^2 (the ONLY material tensor). State: u 3-level + psi/zeta x 3.
struct Sem3dCpmlForwardInput {
    torch::Tensor ibool, D;
    torch::Tensor w_jac, w_jx, w_jy, w_jz;        // (E, ngll^3) geometric
    torch::Tensor vp2_node, inv_M;                // (n_global,)
    torch::Tensor a_x, b_x, dbx_dx, a_y, b_y, dby_dy, a_z, b_z, dbz_dz;
    // uint8, 1 where any of b_d or db_d above is non-zero: the node kernels'
    // interior fast-path test, one byte instead of six float loads per node
    torch::Tensor band_node;
    torch::Tensor dirichlet_idx, src_pattern, wavelet, rec_G, rec_W;
    torch::Tensor u, u_prev, u_next, w_sum;
    torch::Tensor psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z;
    torch::Tensor psi_x_n, psi_y_n, psi_z_n, zeta_x_n, zeta_y_n, zeta_z_n;
    torch::Tensor apsi_x, apsi_y, apsi_z;
    // scratch. m_d = ga_d - k_d is what the node kernels read on the interior
    // (62% of nodes, 50% of warps, at nex=48), where they enter only as that
    // difference;
    // k_d is kept for the b_d*k_d term inside the band and lives on its slab.
    // gax/gay/gaz are RETIRED -- read by nothing, kept because a pybind field
    // is ABI and the driver hands them the empty tensor.
    torch::Tensor gx, gy, gz, kx, ky, kz, gax, gay, gaz;
    torch::Tensor mx, my, mz;
    torch::Tensor seis;
    torch::Tensor save_u;                          // (nt, n_global) OPTIONAL
    torch::Tensor save_psi_x, save_psi_y, save_psi_z;
    torch::Tensor save_zeta_x, save_zeta_y, save_zeta_z;
    // save_w_sum (nt, n_global) OPTIONAL: the step's w_sum. The backward
    // reads the forward state only to recompute w_sum^k (u^k and a*psi^k
    // through grads, zeta^k through step_T), and the 3-D material gradient
    // enters through w_sum alone, so with this tape the seven above are
    // unnecessary: the drivers leave them empty. Additive.
    torch::Tensor save_w_sum;
    // boundary saving (bs=1): band histories + final two u levels
    torch::Tensor band_idx;                        // (nband,) int64
    torch::Tensor band_u, band_psi_x, band_psi_y, band_psi_z;
    torch::Tensor band_zeta_x, band_zeta_y, band_zeta_z;
    // band_split=1: psi_d / zeta_d are ringed on band_idx_d (the nodes of the
    // band where b_d != 0) instead of on band_idx; u stays on band_idx. See
    // SemBandIdx in sem2d/kernels.cuh. Additive: unset keeps the union.
    int band_split = 0;
    torch::Tensor band_idx_x, band_idx_y, band_idx_z;
    // ring storage precision: 0 fp32, 1 bf16, 2 int8. int8 also needs
    // band_scale (7, nt) fp32 -- one amax per (field, step).
    int band_kind = 0;
    torch::Tensor band_scale;   // (nt, nband)
    torch::Tensor fin_u, fin_u_prev;
    int bs = 0;
    // psi_slab=1: the C-PML memory fields psi_d / zeta_d / apsi_d are stored
    // only on the index SLAB of their axis -- the planes i_d on which some
    // node has b_d != 0 -- instead of on all n_global nodes. They are exactly
    // +0.0 everywhere else (psi_d' = b_d*q + a_d*psi_d from a zero start with
    // b_d == +0.0 there), so nothing is lost and no arithmetic changes: a
    // node off the slab reads the literal 0.0f the full array held. The node
    // numbering g = (iz*ny + iy)*nx + ix gives the slot arithmetically from
    // one table per axis (tab_d[i_d] = plane index inside the slab, or -1;
    // planes_d = its inverse), so there is no per-node map. Field lengths:
    // x: nz*ny*nsx, y: nz*nsy*nx, z: nsz*ny*nx. ring_slot_d re-expresses
    // band_idx_d in slot coordinates for the rings; dir_slot_d the Dirichlet
    // nodes that lie on the slab. Additive; unset keeps the full layout.
    // Not valid together with the save-all tape (save_u), which copies the
    // full arrays. See SemSlabs in kernels.cuh.
    int psi_slab = 0;
    int nx = 0, ny = 0, nz = 0, nsx = 0, nsy = 0, nsz = 0;
    torch::Tensor tab_x, tab_y, tab_z, planes_x, planes_y, planes_z;   // int32
    torch::Tensor ring_slot_x, ring_slot_y, ring_slot_z;              // int64
    torch::Tensor dir_slot_x, dir_slot_y, dir_slot_z;                 // int64
    // elem_slab (E,) int32, bit d set where element e may carry a non-zero
    // psi_d (slab.py). psi_d grows only through b_d, so an element whose
    // nodes all have b_d == 0 contributes exactly +0.0 to the a*psi half of
    // the assembly: both grads kernels then skip that axis for it -- most of
    // the mesh at production sizes (79% of the element-axis pairs at nex=32).
    // Empty = every element flagged, i.e. the pre-flag path. Additive.
    torch::Tensor elem_slab;
    // deformed=1: Hex27 topography. The full per-node 3x3 inv_J drives the
    // deformed cpml grads/grads_T; w_jx/w_jy/w_jz are ignored (w_jac reused).
    int deformed = 0;
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;  // (E, ngll^3)
    // columnar=1: the topography builder maps x(xi), y(eta), z(xi,eta,gamma)
    // -- X depends on ix alone and Y on iy alone (sem/mesh/hex27.py), so J is
    // lower triangular and inv_J has FOUR identically-zero components (ixy,
    // ixz, iex, iez) and TWO mesh-wide constants (ixx -> col_ixx, iey ->
    // col_iey). The deformed kernels then read wJ + igx/igy/igz instead of ten
    // (E, ngll^3) streams: 6 of 10 removed, measured 1.25x. prop_c.metric_consts
    // derives the flag from the mesh's CONTROL NODES -- where a separable map
    // leaves x bitwise constant along eta and gamma at any mesh size, unlike
    // inv_J, whose structural zeros are fp32 round-off that grows with the
    // domain -- and never takes a caller's word for it; 0 keeps the full 3x3
    // path, which is always correct.
    int columnar = 0;
    double col_ixx = 0.0, col_iey = 0.0;
    double dt = 0.0; double ihx = 0.0, ihy = 0.0, ihz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// Backward input (P6). Returns (grad_vp2_node, grad_wavelet) — the mass and
// all CPML weights are geometric in 3-D, so vp only enters via vp2_node.
struct Sem3dCpmlBackwardInput {
    torch::Tensor ibool, D, w_jac, w_jx, w_jy, w_jz, vp2_node, inv_M;
    int deformed = 0;
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;  // (E, ngll^3)
    // columnar=1: the topography builder maps x(xi), y(eta), z(xi,eta,gamma)
    // -- X depends on ix alone and Y on iy alone (sem/mesh/hex27.py), so J is
    // lower triangular and inv_J has FOUR identically-zero components (ixy,
    // ixz, iex, iez) and TWO mesh-wide constants (ixx -> col_ixx, iey ->
    // col_iey). The deformed kernels then read wJ + igx/igy/igz instead of ten
    // (E, ngll^3) streams: 6 of 10 removed, measured 1.25x. prop_c.metric_consts
    // derives the flag from the mesh's CONTROL NODES -- where a separable map
    // leaves x bitwise constant along eta and gamma at any mesh size, unlike
    // inv_J, whose structural zeros are fp32 round-off that grows with the
    // domain -- and never takes a caller's word for it; 0 keeps the full 3x3
    // path, which is always correct.
    int columnar = 0;
    double col_ixx = 0.0, col_iey = 0.0;
    torch::Tensor a_x, b_x, dbx_dx, a_y, b_y, dby_dy, a_z, b_z, dbz_dz;
    // uint8, 1 where any of b_d or db_d above is non-zero: the node kernels'
    // interior fast-path test, one byte instead of six float loads per node
    torch::Tensor band_node;
    torch::Tensor dirichlet_idx, src_pattern, wavelet, rec_G, rec_W;
    torch::Tensor save_u, save_psi_x, save_psi_y, save_psi_z;
    torch::Tensor save_zeta_x, save_zeta_y, save_zeta_z;
    torch::Tensor save_w_sum;     // see the forward struct; non-empty => no recompute
    torch::Tensor band_idx;
    torch::Tensor band_u, band_psi_x, band_psi_y, band_psi_z;
    torch::Tensor band_zeta_x, band_zeta_y, band_zeta_z;
    int band_split = 0;                            // see the forward struct
    torch::Tensor band_idx_x, band_idx_y, band_idx_z;
    // ring storage precision: 0 fp32, 1 bf16, 2 int8. int8 also needs
    // band_scale (7, nt) fp32 -- one amax per (field, step).
    int band_kind = 0;
    torch::Tensor band_scale;
    torch::Tensor fin_u, fin_u_prev, u_hi, u_lo, u_km1;
    torch::Tensor r_psi_x, r_psi_y, r_psi_z, r_zeta_x, r_zeta_y, r_zeta_z;
    int bs = 0;
    // psi_slab=1 (see the forward struct): r_psi_d / r_zeta_d and apx/apy/apz
    // live on their axis slab; the ring restores go through ring_slot_d.
    // Only meaningful with bs=1 -- the save-all tapes are full-length.
    int psi_slab = 0;
    int nx = 0, ny = 0, nz = 0, nsx = 0, nsy = 0, nsz = 0;
    torch::Tensor tab_x, tab_y, tab_z, planes_x, planes_y, planes_z;   // int32
    torch::Tensor ring_slot_x, ring_slot_y, ring_slot_z;              // int64
    torch::Tensor dir_slot_x, dir_slot_y, dir_slot_z;                 // int64
    // elem_slab (E,) int32, bit d set where element e may carry a non-zero
    // psi_d (slab.py). psi_d grows only through b_d, so an element whose
    // nodes all have b_d == 0 contributes exactly +0.0 to the a*psi half of
    // the assembly: both grads kernels then skip that axis for it -- most of
    // the mesh at production sizes (79% of the element-axis pairs at nex=32).
    // Empty = every element flagged, i.e. the pre-flag path. Additive.
    torch::Tensor elem_slab;
    torch::Tensor grad_syn;
    torch::Tensor gA, gB, gu_k, gw, w_sum, lu_G;
    torch::Tensor lpsi_x, lpsi_y, lpsi_z, lzeta_x, lzeta_y, lzeta_z;
    torch::Tensor lpsi_x_n, lpsi_y_n, lpsi_z_n, lzeta_x_n, lzeta_y_n, lzeta_z_n;
    torch::Tensor apx, apy, apz;
    torch::Tensor gx, gy, gz, kx, ky, kz, gax, gay, gaz;
    torch::Tensor mx, my, mz;                   // see the forward struct
    torch::Tensor gbx, gby, gbz, gbkx, gbky, gbkz, gbax, gbay, gbaz;
    torch::Tensor lapx, lapy, lapz;
    torch::Tensor grad_vp2, grad_wavelet;
    double dt = 0.0; double ihx = 0.0, ihy = 0.0, ihz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
    // Domain decomposition only (additive): ownership-masked source pattern
    // for the WAVELET gradient (a global reduction; a cut PLANE is complete
    // on both strips after the adjoint exchange). Empty = use src_pattern.
    torch::Tensor src_pattern_own;
};

// P7: 3-D elastic (bare, natural free surface). G* are the (2/da)(2/db)
// factors WITHOUT |J| (jac carried as an (E, ngll^3) array so deformed meshes
// stay possible); material lam/mu/lam2mu chain to vp/vs/rho via build().
struct Sem3dElasticForwardInput {
    torch::Tensor ibool, D, lam, mu, lam2mu, w3, jac, inv_M;
    torch::Tensor pat_x, pat_y, pat_z, wavelet, rec_G, rec_W;
    torch::Tensor ux, uy, uz, ux_prev, uy_prev, uz_prev;
    torch::Tensor ux_next, uy_next, uz_next, Kx, Ky, Kz;
    torch::Tensor seis_x, seis_y, seis_z;
    torch::Tensor save_ux, save_uy, save_uz;      // (nt, n_global) OPTIONAL
    // deformed=1: Hex27 topography. w_J + the full per-node 3x3 inv_J drive
    // sem3d_elastic_stiffness_deformed; w3/jac/G* are ignored.
    int deformed = 0;
    torch::Tensor wJ;                             // (E, ngll^3) = w3 |J|
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;
    // columnar=1: see the C-PML structs above. prop_c.metric_consts derives it
    // from the mesh's control nodes and _elastic3d_deformed_consts already
    // carries it; 0 keeps the full 3x3 path, which is always correct.
    int columnar = 0;
    double col_ixx = 0.0, col_iey = 0.0;
    double dt = 0.0;
    double Gxx = 0.0, Gyy = 0.0, Gzz = 0.0, Gxy = 0.0, Gxz = 0.0, Gyz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// Returns (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
struct Sem3dElasticBackwardInput {
    torch::Tensor ibool, D, lam, mu, lam2mu, w3, jac, inv_M;
    torch::Tensor pat_x, pat_y, pat_z, wavelet, rec_G, rec_W;
    torch::Tensor save_ux, save_uy, save_uz;
    torch::Tensor grad_sx, grad_sy, grad_sz;
    torch::Tensor ax1, ay1, az1, ax2, ay2, az2, axm, aym, azm;
    torch::Tensor tmp_x, tmp_y, tmp_z, Kx, Ky, Kz, adj_x, adj_y, adj_z;
    torch::Tensor yx, yy, yz;
    torch::Tensor grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet;
    int deformed = 0;
    torch::Tensor wJ;                             // (E, ngll^3) = w3 |J|
    torch::Tensor ixx, ixy, ixz, iex, iey, iez, igx, igy, igz;
    // columnar=1: see the C-PML structs above. prop_c.metric_consts derives it
    // from the mesh's control nodes and _elastic3d_deformed_consts already
    // carries it; 0 keeps the full 3x3 path, which is always correct.
    int columnar = 0;
    double col_ixx = 0.0, col_iey = 0.0;
    double dt = 0.0;
    double Gxx = 0.0, Gyy = 0.0, Gzz = 0.0, Gxy = 0.0, Gxz = 0.0, Gyz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// P8: 3-D velocity-stress elastic WITH simple per-axis CPML (ElasticVSSEM3D).
// 27-field staggered leapfrog: v x3 (global), sigma x6 + 18 memories
// (element-local). Coefficients a/b per axis are element-local. Optional
// save-all: v history (adjoint propagation) + the 9 STRETCHED velocity
// gradients per step (what the lam/mu imaging needs).
struct Sem3dVsForwardInput {
    torch::Tensor ibool, D, lam, mu, lam2mu;         // (E, ngll^3)
    torch::Tensor wHx, wHy, wHz, inv_M;              // weights + 1/mass_rho
    torch::Tensor a_x, b_x, a_y, b_y, a_z, b_z;      // (E, ngll^3); empty = bare
    torch::Tensor pat_x, pat_y, pat_z, wavelet, rec_G, rec_W;
    torch::Tensor vx, vy, vz, Rx, Ry, Rz;            // (n_global,)
    torch::Tensor sig;                               // (6, E*ngll^3) stress
    torch::Tensor msig, mvel;                        // (9, E*ngll^3) memories
    torch::Tensor seis_x, seis_y, seis_z;
    torch::Tensor save_vx, save_vy, save_vz;         // (nt, n_global) OPTIONAL
    torch::Tensor save_gv;                           // (nt, 9, E*ngll^3) OPTIONAL
    double dt = 0.0; double ihx = 0.0, ihy = 0.0, ihz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

// Returns (grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet).
struct Sem3dVsBackwardInput {
    torch::Tensor ibool, D, lam, mu, lam2mu, wHx, wHy, wHz, inv_M;
    torch::Tensor a_x, b_x, a_y, b_y, a_z, b_z;
    torch::Tensor pat_x, pat_y, pat_z, wavelet, rec_G, rec_W;
    torch::Tensor save_vx, save_vy, save_vz, save_gv;
    torch::Tensor grad_sx, grad_sy, grad_sz;
    // adjoint state: lv x3 (global), lsig (6, E*m), lmsig/lmvel (9, E*m)
    torch::Tensor lvx, lvy, lvz, lRx, lRy, lRz, adj_x, adj_y, adj_z;
    torch::Tensor lsig, lmsig, lmvel, lgv;           // lgv scratch (9, E*m)
    torch::Tensor grad_lam, grad_mu, grad_lam2mu, grad_inv_M, grad_wavelet;
    double dt = 0.0; double ihx = 0.0, ihy = 0.0, ihz = 0.0;
    int nt = 0, ngll = 0, n_elements = 0, n_global = 0, n_rec = 0;
    int ndim = 3, batch = 1;
};

} // namespace sem3d

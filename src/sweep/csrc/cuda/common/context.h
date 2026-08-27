#pragma once
#include <vector>
#include <type_traits>

// ============================================================================
// SolverContext -- launch-invariant solver geometry, passed BY VALUE to every
// kernel of every equation.
//
// WHY THE PHYSICAL BOUNDS ARE CACHED
// ----------------------------------
// phys_x0() ... nz_phys() are launch-invariant, yet every thread of every
// kernel evaluates them.  Stacking the DD cut-select on top of the per-edge pad
// sentinels made each face a 3-deep nested conditional (x 6 faces); ptxas gives
// up on if-conversion and emits a serial chain of warp-uniform branches in the
// kernel prologue.  For a thin kernel -- a boundary saver, a no-PML update --
// that prologue IS the kernel, which is how a purely geometric refactor came to
// cost 93% on boundary_kernel2d and 47% on acoustic2nd_nopml.
//
// So they are resolved ONCE, on the host, in refresh(), and stored as plain
// ints; the device accessors are then bare loads from the kernel parameter
// bank.  One place to fix, every equation benefits.
//
// WHY THE CACHE CANNOT GO STALE
// -----------------------------
// Enforced by the compiler, not by discipline:
//   * every input to refresh() is either ``const`` (a post-construction write
//     is a compile error) or ``private`` (reachable only through a setter that
//     re-runs refresh());
//   * there is deliberately NO default constructor, so a SolverContext cannot
//     exist without having gone through refresh();
//   * copy-assignment is implicitly deleted by the const members, so a
//     refreshed context can never be overwritten wholesale (copy-CONSTRUCTION
//     stays trivial, which is what the by-value kernel launches need).
// The remaining public members (topo_rows, has_topo, topo_category, use_apm,
// x_base, x_limit, aux_*) are free to assign because none of them feeds
// refresh().
// IF YOU ADD A MEMBER THAT phys_*() DEPENDS ON, IT MUST GO IN THE PRIVATE
// SECTION WITH A REFRESHING SETTER, OR BE const.
// ============================================================================
// A read-only view of just the launch geometry a stencil accessor needs.
//
// SolverContext is 240 bytes and grew there honestly (per-edge free surface, DD
// cut faces, CPML aux slabs, the cached bounds).  That is fine for a kernel
// parameter -- it is passed once and amortised over the whole kernel.  It is
// NOT fine inside a functor that gets inlined once per stencil tap: the VRZ
// gradient kernels build six accessors and each one is expanded 2M times, so
// every byte of state is paid for 24 times over at order 4.  Those accessors
// read six-to-twelve ints; this is that subset.
struct GridBounds {
    int nx, ny, nz, B, M;
    int x_end;
    int phys_lo[3], phys_hi[3];   // [z, y, x], same axis order as SolverContext

    __host__ __device__ inline int phys_x0() const { return phys_lo[2]; }
    __host__ __device__ inline int phys_x1() const { return phys_hi[2]; }
    __host__ __device__ inline int phys_y0() const { return phys_lo[1]; }
    __host__ __device__ inline int phys_y1() const { return phys_hi[1]; }
    __host__ __device__ inline int phys_z0() const { return phys_lo[0]; }
    __host__ __device__ inline int phys_z1() const { return phys_hi[0]; }
};

struct SolverContext {

    // ---- Geometry & physics: immutable after construction ----------------
    const int ndim;

    const int nx;
    const int ny;
    const int nz;

    const int B;

    const float dt;
    const unsigned int nt;

    const int M;
    const int abcn;

    const bool free_surface;

    const float* const lap_coeff;
    const float* const grad_coeff;

    const float dx;
    const float dy;
    const float dz;

    // ---- Trailing fields (default-initialised, so existing brace-init
    // sites that pass the first 13 positional args don't need to change).
    //
    // Irregular free-surface topography (image method / vacuum staircase).
    // ``topo_rows`` is a device pointer of length nx giving the surface
    // row index per column in runtime (PML-padded) coords; any cell with
    // ``iz < topo_rows[ix]`` is air.  ``nullptr`` + ``has_topo=false`` for
    // the flat / no-topo case.
    const int* topo_rows = nullptr;
    bool has_topo = false;

    // APM (Cao & Chen 2018) per-cell category, runtime-padded ``int*``
    // of length ``nz * nx`` (row-major).  Codes:
    //   INTERIOR=0, AIR=1, H=2, VL=3, VR=4, OC=5, IC=6
    // ``nullptr`` + ``use_apm=false`` for non-APM runs.
    const int* topo_category = nullptr;
    bool use_apm = false;

    // ---- Per-edge free surface / PML thickness --------------------------
    // ``fs_faces`` bitmask: bit (2*axis) = the LOW face of that axis is a free
    // surface, (2*axis+1) = the HIGH face; axis 0=z, 1=y, 2=x.  ``pad_lo`` /
    // ``pad_hi`` are per-axis PML pad widths (free-surface faces = 0).
    // ``fs_faces = -1`` and ``pad_* < 0`` (the defaults) reproduce the legacy
    // single ``free_surface`` (z-min only) + uniform ``abcn`` layout, so every
    // untouched brace-init / call site is bit-exact.
    // (private: see the bottom of the struct -- writes must go through
    // set_per_edge()/set_cut_mask() so the cached bounds stay in sync.)

    __host__ __device__
    inline bool fsLo(int axis) const {
        return fs_faces_ < 0 ? (axis == 0 && free_surface) : ((fs_faces_ >> (2 * axis)) & 1);
    }
    __host__ __device__
    inline bool fsHi(int axis) const {
        return fs_faces_ < 0 ? false : ((fs_faces_ >> (2 * axis + 1)) & 1);
    }
    __host__ __device__ inline int fs_faces() const { return fs_faces_; }
    // Per-face PML pad.  An explicit ``pad_lo``/``pad_hi`` (>= 0, per-edge
    // thickness) wins; otherwise a free-surface face has 0 pad and every other
    // face the uniform ``abcn``.  With ``fs_faces = -1`` this is exactly the
    // legacy ``free_surface ? 0 : abcn`` on z-min and ``abcn`` elsewhere.
    __host__ __device__
    inline int padLo(int axis) const {
        return pad_lo_[axis] >= 0 ? pad_lo_[axis] : (fsLo(axis) ? 0 : abcn);
    }
    __host__ __device__
    inline int padHi(int axis) const {
        return pad_hi_[axis] >= 0 ? pad_hi_[axis] : (fsHi(axis) ? 0 : abcn);
    }

    // Host-only: copy per-edge fields from a bound input's ``fs_faces`` +
    // ``pad_lo``/``pad_hi`` vectors (in C axis order [z,(y,)x]).  Empty vectors
    // leave the -1 sentinels => legacy layout.  Call right after the positional
    // brace-init of a SolverContext at every driver.
    inline void set_per_edge(int fs, const std::vector<int>& plo, const std::vector<int>& phi) {
        fs_faces_ = fs;
        for (int a = 0; a < 3; ++a) {
            pad_lo_[a] = (a < (int)plo.size()) ? plo[a] : -1;
            pad_hi_[a] = (a < (int)phi.size()) ? phi[a] : -1;
        }
        refresh();
    }

    // ---- Domain-decomposition cut faces ---------------------------------
    // ORTHOGONAL to the per-edge pad above, and deliberately so: a 0 pad has
    // two possible causes that demand OPPOSITE kernel behaviour.
    //   * free surface  -> pad 0, but the face's halo holds the image mirror,
    //                      so those cells must still be updated and saved.
    //   * DD cut face   -> pad 0, but the halo holds NEIGHBOUR data supplied
    //                      by HaloExchange, so the PML branch must be skipped
    //                      entirely (an algebraically-equal zero-coefficient
    //                      PML update reorders the FMAs and seeds ulp drift,
    //                      breaking DD bit-exactness) and nothing is saved to
    //                      or restored from its boundary buffer.
    // So a cut face can never be inferred from ``padLo()/padHi() == 0`` — it
    // carries its own bitmask (BackwardInput::cut_face_mask):
    //   bit0 = x_lo, bit1 = x_hi, bit2 = z_lo, bit3 = z_hi,
    //   bit4 = y_lo, bit5 = y_hi.
    // 0 (default) = single domain — every kernel below behaves exactly as
    // before.  Backward drivers copy BackwardInput::cut_face_mask here.
    // (private ``cut_mask_``; assign through set_cut_mask().)

    __host__ __device__ inline int  cut_mask() const { return cut_mask_; }
    __host__ __device__ inline bool cut_x_lo() const { return cut_mask_ & 1; }
    __host__ __device__ inline bool cut_x_hi() const { return cut_mask_ & 2; }
    __host__ __device__ inline bool cut_z_lo() const { return cut_mask_ & 4; }
    __host__ __device__ inline bool cut_z_hi() const { return cut_mask_ & 8; }
    __host__ __device__ inline bool cut_y_lo() const { return cut_mask_ & 16; }
    __host__ __device__ inline bool cut_y_hi() const { return cut_mask_ & 32; }

    // The ONLY way to change the DD cut faces: re-runs refresh().
    __host__ __device__ inline void set_cut_mask(int mask) { cut_mask_ = mask; refresh(); }

    // Ranged x stencil launch (DD phase-split forward): the kernel adds
    // ``x_base`` to its block/thread-derived ix and early-returns at
    // ``x_end()``.  Defaults (0, -1 = nx) leave every legacy launch
    // bit-identical.  Only the acoustic2d/3d forward stencil + air-clear
    // kernels honour these.
    int x_base = 0;
    int x_limit = -1;

    __host__ __device__ inline int x_end() const {
        return x_limit < 0 ? nx : x_limit;
    }


    // ===============================
    // Physical domain (computed)
    // ===============================

    // Physical bounds, aware of BOTH reasons a face can carry no PML pad:
    //   * a DD cut (neighbour-facing) face carries only the stencil halo M,
    //     HaloExchange fills it;
    //   * a per-edge free-surface / thin-PML face carries ``padLo/padHi``,
    //     which is 0 on a free surface (its halo holds the image mirror).
    // Cut wins, because a cut face is always exactly M regardless of what
    // pad the per-edge layout would otherwise assign it.  Degenerations:
    //   cut_mask == 0, fs_faces == -1  -> ``abcn + M`` (legacy, bit-exact)
    //   cut_mask == 0, per-edge set    -> pure dev per-edge behaviour
    //   cut face                       -> ``M`` (legacy DD behaviour)
    // The z-low legacy ``free_surface ? M : abcn + M`` is reproduced via
    // ``padLo(0)``, whose fs_faces == -1 branch is ``free_surface ? 0 : abcn``.
    // Bare loads: refresh() resolved the branch chain on the host.
    // Axis order of the cache is [z, y, x] (0, 1, 2), as everywhere else here.
    __host__ __device__ inline int phys_x0() const { return phys_lo_[2]; }
    __host__ __device__ inline int phys_x1() const { return phys_hi_[2]; }
    __host__ __device__ inline int phys_y0() const { return phys_lo_[1]; }
    __host__ __device__ inline int phys_y1() const { return phys_hi_[1]; }
    __host__ __device__ inline int phys_z0() const { return phys_lo_[0]; }
    __host__ __device__ inline int phys_z1() const { return phys_hi_[0]; }

    // Hand a stencil accessor only what it reads (see GridBounds above).
    __host__ __device__ inline GridBounds bounds() const {
        GridBounds g;
        g.nx = nx; g.ny = ny; g.nz = nz; g.B = B; g.M = M; g.x_end = x_end();
        for (int a = 0; a < 3; ++a) { g.phys_lo[a] = phys_lo_[a]; g.phys_hi[a] = phys_hi_[a]; }
        return g;
    }

    __host__ __device__ inline int nx_phys() const { return phys_hi_[2] - phys_lo_[2]; }
    __host__ __device__ inline int ny_phys() const { return phys_hi_[1] - phys_lo_[1]; }
    __host__ __device__ inline int nz_phys() const { return phys_hi_[0] - phys_lo_[0]; }

    // Per-column surface row (2-D propagator).  Returns the row index of
    // the first SOLID cell in column ``ix`` (matches Python ``topo_rows``
    // semantics: cells with ``iz < surface_row(ix)`` are air).  Falls
    // back to the constant ``phys_z0()`` when there's no topography.
    __host__ __device__
    inline int surface_row(int ix) const {
        return has_topo ? topo_rows[ix] : phys_z0();
    }

    // ---- CPML aux-field strip (slab) storage ----------------------------
    // psi/zeta (acoustic) and m_* (elastic) auxiliary fields are only ever
    // nonzero inside the PML bands of their own differencing axis, and are
    // only ever stencil-tapped ALONG that axis.  They can therefore live in
    // two per-axis slabs instead of the full grid.  Width per side covers
    // the profile support (pad + M), the widest adjoint write band
    // (pad + 2M, acoustic 3-D), the stencil tap reach (+M) and the
    // staggered half-node cell (+1).  A DD cut face carries no aux at all.
    // When the two slabs meet, the axis degenerates to full coverage.
    //
    // The slab a kernel actually uses is chosen HOST-side per bound tensor:
    // a full-length tensor selects the identity layout (legacy behaviour,
    // bit-exact), a slab-length tensor selects the strip layout.  Kernels
    // must consult these ONLY behind their per-axis PML gates.
    struct AuxSlab {
        int lo = 0, hi = 0, n = 0;   // side widths + full axis length
        long bnumel = 0;             // per-batch numel of one field of this axis
        __host__ __device__ inline int tot() const { return lo + hi; }
        __host__ __device__ inline bool full() const { return tot() >= n; }
        // Is axis index i backed by storage?
        __host__ __device__ inline bool stored(int i) const {
            return full() || i < lo || i >= n - hi;
        }
        // Slab coordinate of axis index i.  Only valid when stored(i) —
        // callers gate on the per-axis PML band, which the slab covers.
        __host__ __device__ inline int map(int i) const {
            if (full() || i < lo) return i;
            return lo + (i - (n - hi));
        }
        // Read coordinate for UNGATED accesses: stored positions map
        // normally; unstored positions read slab element 0.  Every such
        // read carries an exactly-zero profile weight, so the value is
        // irrelevant — this keeps the kernels' original single-path
        // expression tree (and hence FMA contraction) intact while the
        // address moves into the slab.  tot() >= 1 is guaranteed by
        // init_aux_slabs, so element 0 always exists.
        __host__ __device__ inline int rd(int i) const {
            return stored(i) ? map(i) : 0;
        }
    };

    AuxSlab aux_z, aux_y, aux_x;

    // Formula-side slab for one axis (0=z, 1=y, 2=x).  Host-only helper;
    // drivers compare it against the bound tensor's axis length and either
    // adopt it (slab tensors) or fall back to identity (full tensors).
    inline AuxSlab aux_slab_formula(int axis) const {
        AuxSlab s;
        s.n = axis == 0 ? nz : (axis == 1 ? ny : nx);
        bool cl = axis == 0 ? cut_z_lo() : (axis == 1 ? cut_y_lo() : cut_x_lo());
        bool ch = axis == 0 ? cut_z_hi() : (axis == 1 ? cut_y_hi() : cut_x_hi());
        s.lo = cl ? 0 : padLo(axis) + 3 * M + 1;
        s.hi = ch ? 0 : padHi(axis) + 3 * M + 1;
        if (s.lo + s.hi == 0) s.lo = 1;   // dummy column so rd() has a target
        if (s.lo + s.hi >= s.n) { s.lo = s.n; s.hi = 0; }
        return s;
    }

    inline AuxSlab aux_identity(int axis) const {
        AuxSlab s;
        s.n = axis == 0 ? nz : (axis == 1 ? ny : nx);
        s.lo = s.n; s.hi = 0;
        return s;
    }

    // Fill aux_z/aux_y/aux_x from the axis lengths of the bound aux tensors
    // (pass -1 for an axis the equation has no aux for / 2-D y).  Returns
    // false when a length matches neither the full axis nor the formula slab
    // (caller should raise).  Also fills per-batch numels.
    inline bool init_aux_slabs(long len_z, long len_y, long len_x) {
        const long lens[3] = {len_z, len_y, len_x};
        AuxSlab* out[3] = {&aux_z, &aux_y, &aux_x};
        for (int axis = 0; axis < 3; ++axis) {
            if (lens[axis] < 0) { *out[axis] = aux_identity(axis); continue; }
            AuxSlab s = aux_slab_formula(axis);
            if (lens[axis] == (long)s.n) s = aux_identity(axis);
            else if (lens[axis] != (long)s.tot()) return false;
            *out[axis] = s;
        }
        long nyy = ndim == 3 ? ny : 1;
        aux_z.bnumel = (long)aux_z.tot() * nyy * nx;
        aux_y.bnumel = (long)nz * aux_y.tot() * nx;
        aux_x.bnumel = (long)nz * nyy * aux_x.tot();
        return true;
    }

    // rd() variants for ungated reads (see AuxSlab::rd).
    __host__ __device__ inline long aux_rd_x2(int iz, int ix) const {
        return (long)iz * aux_x.tot() + aux_x.rd(ix);
    }
    __host__ __device__ inline long aux_rd_z2(int iz, int ix) const {
        return (long)aux_z.rd(iz) * nx + ix;
    }
    __host__ __device__ inline long aux_rd_x3(int iz, int iy, int ix) const {
        return ((long)iz * ny + iy) * aux_x.tot() + aux_x.rd(ix);
    }
    __host__ __device__ inline long aux_rd_y3(int iz, int iy, int ix) const {
        return ((long)iz * aux_y.tot() + aux_y.rd(iy)) * nx + ix;
    }
    __host__ __device__ inline long aux_rd_z3(int iz, int iy, int ix) const {
        return ((long)aux_z.rd(iz) * ny + iy) * nx + ix;
    }

    // Linear index of (iz[,iy],ix) inside an axis family's slab tensor.
    __host__ __device__ inline long aux_idx_x2(int iz, int ix) const {
        return (long)iz * aux_x.tot() + aux_x.map(ix);
    }
    __host__ __device__ inline long aux_idx_z2(int iz, int ix) const {
        return (long)aux_z.map(iz) * nx + ix;
    }
    __host__ __device__ inline long aux_idx_x3(int iz, int iy, int ix) const {
        return ((long)iz * ny + iy) * aux_x.tot() + aux_x.map(ix);
    }
    __host__ __device__ inline long aux_idx_y3(int iz, int iy, int ix) const {
        return ((long)iz * aux_y.tot() + aux_y.map(iy)) * nx + ix;
    }
    __host__ __device__ inline long aux_idx_z3(int iz, int iy, int ix) const {
        return ((long)aux_z.map(iz) * ny + iy) * nx + ix;
    }

    // Per-(iy, ix)-column surface row (3-D propagator).  ``topo_rows`` is
    // a flat row-major (ny, nx) int array on the runtime grid.  Returns
    // the row index of the first SOLID cell at (iy, ix); cells with
    // ``iz < surface_row(ix, iy)`` are air.  Falls back to ``phys_z0()``
    // when there's no topography.
    __host__ __device__
    inline int surface_row(int ix, int iy) const {
        return has_topo ? topo_rows[iy * nx + ix] : phys_z0();
    }

    // ---- Sole constructor ------------------------------------------------
    // Its parameter list is exactly the 15 leading members, in order, so every
    // existing brace-init site
    //     SolverContext ctx{2, nx, 0, nz, B, dt, nt, M, abcn, fs,
    //                       lap, grad, dx, 0.f, dz};
    // still compiles verbatim -- it is now a constructor call rather than
    // aggregate initialisation, with identical narrowing rules.  There is no
    // default constructor on purpose: it would let a context exist with an
    // unrefreshed cache.
    __host__ __device__
    SolverContext(int ndim_, int nx_, int ny_, int nz_, int B_,
                  float dt_, unsigned int nt_, int M_, int abcn_,
                  bool free_surface_,
                  const float* lap_coeff_, const float* grad_coeff_,
                  float dx_, float dy_, float dz_)
        : ndim(ndim_), nx(nx_), ny(ny_), nz(nz_), B(B_),
          dt(dt_), nt(nt_), M(M_), abcn(abcn_), free_surface(free_surface_),
          lap_coeff(lap_coeff_), grad_coeff(grad_coeff_),
          dx(dx_), dy(dy_), dz(dz_)
    {
        refresh();
    }

private:
    // ---- Refresh inputs: private, so the only writers are the setters -----
    int fs_faces_ = -1;
    int pad_lo_[3] = {-1, -1, -1};   // [z, y, x]
    int pad_hi_[3] = {-1, -1, -1};
    int cut_mask_ = 0;

    // ---- The cache -------------------------------------------------------
    int phys_lo_[3] = {0, 0, 0};      // [z, y, x]
    int phys_hi_[3] = {0, 0, 0};

    // Resolve the per-face bounds once.  A DD cut face is exactly M regardless
    // of the per-edge pad (its halo holds neighbour data); otherwise the face
    // carries padLo/padHi, which is 0 on a free surface.  This is the former
    // phys_*() expression, evaluated on the host instead of per thread.
    __host__ __device__ void refresh()
    {
        const int extent[3] = {nz, ny, nx};
        for (int a = 0; a < 3; ++a) {
            // cut_mask bit layout: 0=x_lo 1=x_hi 2=z_lo 3=z_hi 4=y_lo 5=y_hi
            const int lo_bit = (a == 0) ? 4 : (a == 1) ? 16 : 1;
            const int hi_bit = (a == 0) ? 8 : (a == 1) ? 32 : 2;
            const bool cl = (cut_mask_ & lo_bit) != 0;
            const bool ch = (cut_mask_ & hi_bit) != 0;
            phys_lo_[a] = cl ? M : padLo(a) + M;
            phys_hi_[a] = extent[a] - (ch ? M : padHi(a) + M);
        }
    }
};

// Kernels take SolverContext BY VALUE, so trivial copyability is not
// negotiable.  The const members delete copy-ASSIGNMENT (which is what stops a
// refreshed context being overwritten wholesale) but leave copy-CONSTRUCTION
// trivial, which is what the launches actually use.
static_assert(std::is_trivially_copyable<SolverContext>::value,
              "SolverContext must stay trivially copyable: it is passed by value to kernels.");
static_assert(!std::is_default_constructible<SolverContext>::value,
              "SolverContext must not be default-constructible: the bounds cache would be unrefreshed.");
static_assert(!std::is_copy_assignable<SolverContext>::value,
              "SolverContext must not be copy-assignable: that would bypass refresh().");

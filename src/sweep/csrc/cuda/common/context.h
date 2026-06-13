#pragma once
struct SolverContext {

    int ndim;

    int nx;
    int ny;
    int nz;

    int B;

    float dt;
    unsigned int nt;

    int M;
    int abcn;

    bool free_surface;

    const float* lap_coeff;
    const float* grad_coeff;

    float dx;
    float dy;
    float dz;

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

    // Domain-decomposition cut-face bitmask (BackwardInput::cut_face_mask):
    //   bit0 = x_lo, bit1 = x_hi, bit2 = z_lo, bit3 = z_hi,
    //   bit4 = y_lo, bit5 = y_hi.
    // 0 (default) = single domain — every kernel below behaves exactly as
    // before.  Backward drivers copy BackwardInput::cut_face_mask here.
    int cut_mask = 0;

    __host__ __device__ inline bool cut_x_lo() const { return cut_mask & 1; }
    __host__ __device__ inline bool cut_x_hi() const { return cut_mask & 2; }
    __host__ __device__ inline bool cut_z_lo() const { return cut_mask & 4; }
    __host__ __device__ inline bool cut_z_hi() const { return cut_mask & 8; }
    __host__ __device__ inline bool cut_y_lo() const { return cut_mask & 16; }
    __host__ __device__ inline bool cut_y_hi() const { return cut_mask & 32; }

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

    // Cut-aware physical bounds: a cut (neighbour-facing) face carries only
    // the stencil halo M (HaloExchange fills it) instead of a full abcn PML
    // pad, so the interior starts/ends M from the buffer edge there. With
    // cut_mask == 0 every expression collapses to the legacy ``abcn + M``.
    __host__ __device__
    inline int phys_x0() const { return cut_x_lo() ? M : abcn + M; }

    __host__ __device__
    inline int phys_x1() const { return nx - (cut_x_hi() ? M : abcn + M); }

    __host__ __device__
    inline int phys_y0() const { return cut_y_lo() ? M : abcn + M; }

    __host__ __device__
    inline int phys_y1() const { return ny - (cut_y_hi() ? M : abcn + M); }

    __host__ __device__
    inline int phys_z0() const { return free_surface ? M : (cut_z_lo() ? M : abcn + M); }

    __host__ __device__
    inline int phys_z1() const { return nz - (cut_z_hi() ? M : abcn + M); }

    __host__ __device__
    inline int nx_phys() const { return phys_x1() - phys_x0(); }

    __host__ __device__
    inline int ny_phys() const { return phys_y1() - phys_y0(); }

    __host__ __device__
    inline int nz_phys() const { return phys_z1() - phys_z0(); }

    // Per-column surface row (2-D propagator).  Returns the row index of
    // the first SOLID cell in column ``ix`` (matches Python ``topo_rows``
    // semantics: cells with ``iz < surface_row(ix)`` are air).  Falls
    // back to the constant ``phys_z0()`` when there's no topography.
    __host__ __device__
    inline int surface_row(int ix) const {
        return has_topo ? topo_rows[ix] : phys_z0();
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

};

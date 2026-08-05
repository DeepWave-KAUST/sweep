#pragma once
#include <vector>
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

    // ---- Per-edge free surface / PML thickness --------------------------
    // ``fs_faces`` bitmask: bit (2*axis) = the LOW face of that axis is a free
    // surface, (2*axis+1) = the HIGH face; axis 0=z, 1=y, 2=x.  ``pad_lo`` /
    // ``pad_hi`` are per-axis PML pad widths (free-surface faces = 0).
    // ``fs_faces = -1`` and ``pad_* < 0`` (the defaults) reproduce the legacy
    // single ``free_surface`` (z-min only) + uniform ``abcn`` layout, so every
    // untouched brace-init / call site is bit-exact.
    int fs_faces = -1;
    int pad_lo[3] = {-1, -1, -1};   // [z, y, x]
    int pad_hi[3] = {-1, -1, -1};

    __host__ __device__
    inline bool fsLo(int axis) const {
        return fs_faces < 0 ? (axis == 0 && free_surface) : ((fs_faces >> (2 * axis)) & 1);
    }
    __host__ __device__
    inline bool fsHi(int axis) const {
        return fs_faces < 0 ? false : ((fs_faces >> (2 * axis + 1)) & 1);
    }
    // Per-face PML pad.  An explicit ``pad_lo``/``pad_hi`` (>= 0, per-edge
    // thickness) wins; otherwise a free-surface face has 0 pad and every other
    // face the uniform ``abcn``.  With ``fs_faces = -1`` this is exactly the
    // legacy ``free_surface ? 0 : abcn`` on z-min and ``abcn`` elsewhere.
    __host__ __device__
    inline int padLo(int axis) const {
        return pad_lo[axis] >= 0 ? pad_lo[axis] : (fsLo(axis) ? 0 : abcn);
    }
    __host__ __device__
    inline int padHi(int axis) const {
        return pad_hi[axis] >= 0 ? pad_hi[axis] : (fsHi(axis) ? 0 : abcn);
    }

    // Host-only: copy per-edge fields from a bound input's ``fs_faces`` +
    // ``pad_lo``/``pad_hi`` vectors (in C axis order [z,(y,)x]).  Empty vectors
    // leave the -1 sentinels => legacy layout.  Call right after the positional
    // brace-init of a SolverContext at every driver.
    inline void set_per_edge(int fs, const std::vector<int>& plo, const std::vector<int>& phi) {
        fs_faces = fs;
        for (int a = 0; a < 3; ++a) {
            pad_lo[a] = (a < (int)plo.size()) ? plo[a] : -1;
            pad_hi[a] = (a < (int)phi.size()) ? phi[a] : -1;
        }
    }


    // ===============================
    // Physical domain (computed)
    // ===============================

    __host__ __device__
    inline int phys_x0() const { return padLo(2) + M; }

    __host__ __device__
    inline int phys_x1() const { return nx - padHi(2) - M; }

    __host__ __device__
    inline int phys_y0() const { return padLo(1) + M; }

    __host__ __device__
    inline int phys_y1() const { return ny - padHi(1) - M; }

    __host__ __device__
    inline int phys_z0() const { return padLo(0) + M; }

    __host__ __device__
    inline int phys_z1() const { return nz - padHi(0) - M; }

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

#pragma once

#include <cstdint>
#include <vector>

// ====================================================================
// Boundary-saving restore strips of the CPU engine -- ONE description of
// which cells restore_*_boundary / restore_*_boundary_disk overwrite, shared
// by those restores and by sub_source_in_restore_strip_* below
// (acoustic2d, acoustic3d, acoustic_lsrtm2d, acoustic_lsrtm3d).
//
// Geometry: the physical box [z0, z1) x [y0, y1) x [x0, x1) of the padded
// grid (z0 = M under a free surface, abcn + M otherwise; every other face
// abcn + M in from the grid edge) carries a band ``width`` cells deep inside
// each face.  The restores walk those bands face by face:
//   top / bottom : z in [z0, z0 + width) / [z1 - width, z1), the box's full
//                  y and x extent;
//   front / back : y in [y0, y0 + width) / [y1 - width, y1), full z and x;
//   left / right : x in [x0, x0 + width) / [x1 - width, x1), full z and y;
// so the set they overwrite is exactly the cells of the box that lie within
// ``width`` of one of its faces -- the predicate below.  The restores take
// their bounds from the same BsStripBox*, so the two cannot drift apart.
//
// The CUDA twin is csrc/cuda/common/boundary/strip.cuh (bs_in_restore_strip_*,
// offset 0, tangent_pad 0 for these equations).
// ====================================================================

namespace sweep_cpu::bs_strip {

struct BsStripBox2D {
    int x0, x1, z0, z1;   // physical box, half-open
    int width;            // band depth inside each face
};

struct BsStripBox3D {
    int x0, x1, y0, y1, z0, z1;
    int width;
};

inline bool in_restore_strip(const BsStripBox2D& g, int64_t z, int64_t x)
{
    if (z < g.z0 || z >= g.z1 || x < g.x0 || x >= g.x1)
        return false;
    return z < g.z0 + g.width || z >= g.z1 - g.width
        || x < g.x0 + g.width || x >= g.x1 - g.width;
}

inline bool in_restore_strip(const BsStripBox3D& g, int64_t z, int64_t y, int64_t x)
{
    if (z < g.z0 || z >= g.z1 || y < g.y0 || y >= g.y1 || x < g.x0 || x >= g.x1)
        return false;
    return z < g.z0 + g.width || z >= g.z1 - g.width
        || y < g.y0 + g.width || y >= g.y1 - g.width
        || x < g.x0 + g.width || x >= g.x1 - g.width;
}

// Boundary-saving reverse reconstruction.  The time-reversed NOPML step leaves
// w^{it-1} - s^{it} in u (the source is the one term it cannot reverse); the
// restore then overwrites the strips with the saved TRUE w^{it-1}, which
// already carries s^{it}.  Call this right after the restore, with the
// add_source that follows its own (source, sources, it): for every source
// whose cell the restore just overwrote it subtracts that sample once, so the
// strip's u_tt imaging and the add_source that completes w^{it-1} see what
// they see at any other source cell.  Sources outside the strips are not
// touched: a run with none in a strip is bit-identical with or without it.
// Indexing and bounds test are add_*_source's, line for line.
inline void sub_source_in_restore_strip_2d(
    std::vector<float>& u,
    const float* source,          // (B, nsrc, nt)
    const int32_t* sources,       // (B, nsrc, 2) as (x, z)
    int64_t B,
    int64_t nsrc,
    int64_t nt,
    int64_t it,
    int64_t nz,
    int64_t nx,
    const BsStripBox2D& g)
{
    const int64_t spatial = nz * nx;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t loc = (b * nsrc + isrc) * 2;
            const int64_t x = sources[loc];
            const int64_t z = sources[loc + 1];
            if (x >= 0 && x < nx && z >= 0 && z < nz && in_restore_strip(g, z, x)) {
                u[b * spatial + z * nx + x] -= source[(b * nsrc + isrc) * nt + it];
            }
        }
    }
}

inline void sub_source_in_restore_strip_3d(
    std::vector<float>& u,
    const float* source,          // (B, nsrc, nt)
    const int32_t* sources,       // (B, nsrc, 3) as (x, y, z)
    int64_t B,
    int64_t nsrc,
    int64_t nt,
    int64_t it,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    const BsStripBox3D& g)
{
    const int64_t spatial = nz * ny * nx;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t isrc = 0; isrc < nsrc; ++isrc) {
            const int64_t loc = (b * nsrc + isrc) * 3;
            const int64_t x = sources[loc];
            const int64_t y = sources[loc + 1];
            const int64_t z = sources[loc + 2];
            if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz
                && in_restore_strip(g, z, y, x)) {
                u[b * spatial + z * ny * nx + y * nx + x] -= source[(b * nsrc + isrc) * nt + it];
            }
        }
    }
}

} // namespace sweep_cpu::bs_strip

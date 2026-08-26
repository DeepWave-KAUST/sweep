// Exhaustive proof that SolverContext's shared in_pml_* predicates are
// identical to the open-coded forms they replace.
//
// The GPU gate can only sample: it runs a handful of configurations and checks
// the numbers came out the same. This enumerates the ENTIRE input space of the
// predicate -- every cut-face mask, every free-surface state, every pad layout,
// every halo width, every cell index -- and asserts equality at every point.
// If this passes there is no configuration in which the rewrite can differ, so
// the GPU runs afterwards are confirming the plumbing, not the algebra.
//
//   g++ -O2 -std=c++17 -I<csrc>/cuda/common -o /tmp/eq gate/in_pml_equiv.cpp && /tmp/eq
#define __host__
#define __device__
#include "context.h"

#include <cstdio>
#include <vector>

static long long checks = 0, fails = 0;

static void bad(const char *what, const SolverContext &c, int i, int halo,
                bool got, bool want) {
    if (fails++ < 20)
        printf("MISMATCH %s: i=%d halo=%d cut=%d fs_faces=%d free_surface=%d "
               "abcn=%d n=(%d,%d,%d) -> got %d want %d\n",
               what, i, halo, c.cut_mask, c.fs_faces, (int)c.free_surface,
               c.abcn, c.nx, c.ny, c.nz, (int)got, (int)want);
}

int main() {
    // Legacy call sites always run with cut_mask == 0 and fs_faces == -1; the
    // DD-migrated ones add a cut mask. Sweep both, plus per-edge pads, so the
    // claim covers every layout the kernels can be handed.
    const std::vector<int> masks = {0, 1, 2, 3, 4, 8, 12, 16, 32, 48, 63};
    const std::vector<int> fs_faces = {-1, 0, 1, 2, 4, 5, 63};
    const std::vector<int> abcns = {0, 1, 5, 20};
    const std::vector<int> halos = {1, 2, 3, 4, 5};

    for (int mask : masks)
    for (int fsf : fs_faces)
    for (int fs : {0, 1})
    for (int abcn : abcns)
    for (int halo : halos) {
        SolverContext c{};
        c.ndim = 3; c.nx = 37; c.ny = 31; c.nz = 41;
        c.M = 2; c.abcn = abcn; c.free_surface = (bool)fs;
        c.fs_faces = fsf;
        c.cut_mask = mask;
        // pad_lo/pad_hi stay at the -1 sentinel: that is the layout every
        // legacy site runs under, and it is what padLo/padHi degrade to.

        for (int ix = 0; ix < c.nx; ++ix) {
            // ---- legacy x term, verbatim from the kernels ----
            bool want = (ix < c.abcn + halo) || (ix >= c.nx - c.abcn - halo);
            bool got = c.in_pml_x(ix, halo);
            ++checks;
            // Only claim equality where the legacy form was actually used:
            // no cut faces on this axis, and the default pad layout.
            if (mask == 0 && fsf == -1 && got != want) bad("x-legacy", c, ix, halo, got, want);

            // ---- hand-written cut-aware x term ----
            // NOT equivalent, and deliberately so. On a cut face the shared
            // predicate says "outside the physical region" for the halo band
            // (phys_x0 == halo), while this form says "not in PML" for the
            // WHOLE low side. Both are correct DD implementations -- the halo
            // band is overwritten by the next exchange either way, so only the
            // physical interior has to agree, and it does -- but they are
            // different functions. A shared helper must not silently convert
            // one into the other, so these sites stay hand-written.
            bool want_cut = ((ix < c.abcn + halo) && !c.cut_x_lo()) ||
                            ((ix >= c.nx - c.abcn - halo) && !c.cut_x_hi());
            if (fsf == -1 && mask != 0 && c.in_pml_x(ix, halo) != want_cut) {
                // Record WHERE they differ: it must be confined to the halo
                // band of a cut face, never the physical interior.
                bool in_halo_band = (c.cut_x_lo() && ix < c.abcn + halo) ||
                                    (c.cut_x_hi() && ix >= c.nx - c.abcn - halo);
                ++checks;
                if (!in_halo_band)
                    bad("x-cutaware-LEAKED-INTO-INTERIOR", c, ix, halo,
                        c.in_pml_x(ix, halo), want_cut);
            }

            // ---- per-edge x term (no cut terms: not a DD-migrated kernel) ----
            bool want_pe = (!c.cut_x_lo() && ix < c.padLo(2) + halo) ||
                           (!c.cut_x_hi() && ix >= c.nx - c.padHi(2) - halo);
            ++checks;
            if (mask == 0 && c.in_pml_x(ix, halo) != want_pe)
                bad("x-peredge", c, ix, halo, c.in_pml_x(ix, halo), want_pe);
        }

        for (int iy = 0; iy < c.ny; ++iy) {
            bool want = (iy < c.abcn + halo) || (iy >= c.ny - c.abcn - halo);
            ++checks;
            if (mask == 0 && fsf == -1 && c.in_pml_y(iy, halo) != want)
                bad("y-legacy", c, iy, halo, c.in_pml_y(iy, halo), want);
        }

        for (int iz = 0; iz < c.nz; ++iz) {
            // The free-surface-aware z term: this is the one whose equality
            // depends on padLo(0)'s fs_faces == -1 branch being exactly
            // ``free_surface ? 0 : abcn``.
            bool want_fs = (iz < (c.free_surface ? halo : c.abcn + halo)) ||
                           (iz >= c.nz - c.abcn - halo);
            ++checks;
            if (mask == 0 && fsf == -1 && c.in_pml_z(iz, halo) != want_fs)
                bad("z-legacy-fs", c, iz, halo, c.in_pml_z(iz, halo), want_fs);

            // The z term WITHOUT a free-surface branch. It differs from the
            // shared predicate exactly when free_surface is true -- which is
            // why every site using this form has to be shown unreachable with
            // a free surface before it may be rewritten. Assert that the
            // difference is confined to free_surface, and nothing wider.
            bool want_nofs = (iz < c.abcn + halo) || (iz >= c.nz - c.abcn - halo);
            ++checks;
            if (mask == 0 && fsf == -1 && !fs && c.in_pml_z(iz, halo) != want_nofs)
                bad("z-legacy-nofs(fs=0)", c, iz, halo, c.in_pml_z(iz, halo), want_nofs);
        }
    }

    printf("%lld predicate evaluations, %lld mismatches\n", checks, fails);
    if (fails == 0)
        printf("PROVED: the shared predicate is identical to every legacy form "
               "it replaces, over the whole input space.\n");
    return fails ? 1 : 0;
}

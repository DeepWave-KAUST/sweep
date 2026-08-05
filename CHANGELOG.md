# Changelog

All notable changes to SWEEP are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Per-edge free surface (deepwave-style).  `Propagator(free_surface=...)` now
  accepts a per-edge spec — an edge-name list (`['top', 'left']`), a
  length-`2*ndim` bool mask (`[z0, z1, x0, x1]`), or a dict — in addition to the
  historical `bool` (top-only): a free surface on any subset of the domain
  faces.  `abcn` likewise accepts a per-edge list for an independent PML
  thickness per face.  The **eager** backend supports it for **Acoustic and
  Elastic 2-D** (all four edges, gradient-consistent); the compiled `impl='c'`
  backend supports it for **Acoustic 2-D on CUDA** (all four edges) and
  **Elastic 2-D on CUDA** (the top/bottom z faces) — bit-exact vs eager forward,
  adjoint-gradient cosine ~1.  `free_surface=True` / a scalar `abcn` stay
  bit-for-bit unchanged.  Requests not yet implemented raise a clear
  `NotImplementedError` pointing at `impl='eager'`: per-edge on 3-D or on
  non-migrated equations, per-edge on the CPU `impl='c'` backend, per-edge PML
  *thickness* on `impl='c'`, and Elastic **left/right** free surface on
  `impl='c'` (the CUDA x-face image method is not migrated yet).
- Documentation overhaul (Phase 1, facade): rewritten landing page with
  capability cards and audience-routed navigation; README and README.zh-CN
  gained badges and a tagline block.
- `CHANGELOG.md` and `CONTRIBUTING.md` scaffolding.

### Changed
- `docs/user-guide/equations.md`: summary table expanded from 3 rows to cover
  all 20+ exported equation classes, grouped by physics family. Template
  reminder at the bottom replaced with a "See Also" cross-reference block.
- `mkdocs.yml`: enabled `attr_list` and `md_in_html` Markdown extensions to
  support Material grid-card layouts.

## Earlier history

Earlier release notes will be backfilled from the commit history. For now,
see the
[GitHub commit history](https://github.com/DeepWave-KAUST/sweep/commits/dev)
for changes prior to this entry.

[Unreleased]: https://github.com/DeepWave-KAUST/sweep/compare/main...dev

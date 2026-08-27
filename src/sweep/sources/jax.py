

from sweep.sources.base import SourceBase
import jax.numpy as jnp

class SourceJax(SourceBase):
    def __init__(self, coords, shape, source_encoding=False, adj=False,
                 spread_kernel=None):
        """Source class for the wave equation

        Args:
            coords (jnp.ndarray): Source coordinates (nshots, 2)
            shape (jnp.ndarray): Wavefield shape for generating source mask(batch, 1, nz, nx)
            spread_kernel: Small 2-D stencil (e.g. a 3x3 binomial) that
                spreads each injected sample over its neighbourhood.
                Equations whose collocated stencils have a checkerboard null
                space (the rotated staggered grid) declare it via
                ``equation.source_receiver_stencil`` so a point source does
                not excite the spurious mode. ``None`` keeps the plain
                single-cell injection. Mirrors ``SourceTorch``.
        """
        super().__init__()
        self.mask = jnp.zeros(shape, dtype=jnp.float32)
        self.se = source_encoding
        self.coords = coords
        self._coords = zip(*coords)
        self.adj = adj
        self.coords_r = [c.flatten() for c in jnp.split(jnp.flip(coords, -1), coords.shape[-1], axis=-1)]
        self.spread_offsets = None
        if spread_kernel is not None:
            if len(self.coords_r) != 2:
                raise NotImplementedError("source spread_kernel is only supported for 2-D wavefields")
            k = jnp.asarray(spread_kernel, dtype=jnp.float32)
            half = k.shape[-1] // 2
            self.spread_offsets = [
                (dz, dx, float(k[dz + half, dx + half]))
                for dz in range(-half, half + 1)
                for dx in range(-half, half + 1)
                if float(k[dz + half, dx + half]) != 0.0
            ]
        # for i in range(coords.shape[0]): # Loop over each source
        #     index = 0 if source_encoding else i
        #     self.mask = self.mask.at[index, 0,  *jax.numpy.flip(coords, [-1])[i]].set(1.)
    
    def forward_source_encoding(self, wavefield, wavelet):
        wavefield = wavefield.at[(..., *self.coords_r)].add(wavelet)
        return wavefield
    
    def forward_adjoint_modeling(self, wavefield, wavelet):

        wavefield = wavefield.at[jnp.repeat(jnp.arange(self.coords.shape[0]), self.coords.shape[1]), 
                                 0, 
                                 self.coords[..., 1].reshape(-1), 
                                 self.coords[..., 0].reshape(-1)].add(wavelet.reshape(-1))
        return wavefield
    
    def multiwavelet(self, wavefield, wavelet):
        # Handles every input mode, including source encoding (mode B): with
        # batch=1 and nsrc points, the (1,) shots index broadcasts against the
        # nsrc-long coords_r under fancy indexing, and .at[].add accumulates
        # overlapping points — verified to match SourceTorch's explicit
        # forward_source_encoding branch (rel ~5e-7, grad cosine 1.0).
        shots = jnp.arange(wavefield.shape[0])
        if self.adj:
            shots = jnp.repeat(shots, self.coords.shape[1])
            wavelet = wavelet.reshape(-1)
        if self.spread_offsets is None:
            return wavefield.at[(shots, 0, *self.coords_r)].add(wavelet)

        zc, xc = self.coords_r
        for dz, dx, w in self.spread_offsets:
            wavefield = wavefield.at[(shots, 0, zc + dz, xc + dx)].add(w * wavelet)
        return wavefield

    def __call__(self, *args):
        # Always dispatches to multiwavelet — the forward_source_encoding /
        # forward_adjoint_modeling methods above are currently unwired (kept
        # for API compatibility; multiwavelet covers their cases).
        return self.multiwavelet(*args)
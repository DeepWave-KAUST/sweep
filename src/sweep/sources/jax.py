

from sweep.sources.base import SourceBase
import jax.numpy as jnp

class SourceJax(SourceBase):
    def __init__(self, coords, shape, source_encoding=False, adj=False):
        """Source class for the wave equation

        Args:
            coords (jnp.ndarray): Source coordinates (nshots, 2)
            shape (jnp.ndarray): Wavefield shape for generating source mask(batch, 1, nz, nx)
        """
        super().__init__()
        self.mask = jnp.zeros(shape, dtype=jnp.float32)
        self.se = source_encoding
        self.coords = coords
        self._coords = zip(*coords)
        self.adj = adj
        self.coords_r = [c.flatten() for c in jnp.split(jnp.flip(coords, -1), coords.shape[-1], axis=-1)]
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
        shots = jnp.arange(wavefield.shape[0])
        if self.adj:
            shots = jnp.repeat(shots, self.coords.shape[1])
            wavelet = wavelet.reshape(-1)
        wavefield = wavefield.at[(shots, 0, *self.coords_r)].add(wavelet)
        return wavefield

    def __call__(self, *args):
        return self.multiwavelet(*args)
import jax, torch
import jax.numpy as jnp

class SourceBase:

    def __init__(self, **kwargs):
        pass

    def forward(self, wavefield, wavelet):
        """Forward pass of the source

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)
            wavelet (torch.Tensor): Wavelet tensor (1,)

        Returns:
            torch.Tensor: The wavefield with the source injected
        """
        return wavefield + self.mask * wavelet

        
class SourceTorch(SourceBase, torch.nn.Module):
    def __init__(self, coords, shape, dev, source_encoding=False, adj=False):
        """Source class for the wave equation

        Args:
            coords (torch.Tensor): Source coordinates (nshots, 2)
            shape (torch.Tensor): Wavefield shape for generating source mask(batch, 1, nz, nx)
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        self.mask = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.se = source_encoding
        self.coords = coords
        self.adj = adj
        for i in range(coords.shape[0]):
            index = 0 if source_encoding else i
            self.mask[index, :, *torch.flip(coords, [-1])[i]] = 1.

    def forward_source_encoding(self, wavefield, wavelet):
        z = self.coords[..., 1]
        x = self.coords[..., 0]
        wavefield[0, 0, z, x] = wavefield[0, 0, z, x] + wavelet
        return wavefield
    
    def forward_adjoint_modeling(self, wavefield, wavelet):

        # for i in range(self.coords.shape[0]): # Loop over each shot
        #     wavefield = wavefield.at[i, 0, self.coords[i, :, 1], self.coords[i, :, 0]].add(wavelet[i])
        indices = [torch.arange(self.coords.shape[0]).repeat(self.coords.shape[1]), 0, self.coords[..., 1].reshape(-1), self.coords[..., 0].reshape(-1)]
        wavefield[indices] = wavefield[indices] + wavelet.reshape(-1)
        return wavefield

    def forward(self, *args):
        if self.se:
            return self.forward_source_encoding(*args)
        else:
            if self.adj:
                return self.forward_adjoint_modeling(*args)
            else:
                return super().forward(*args)
    
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
        self.adj = adj
        self.coords_r = [c.flatten() for c in jnp.split(jnp.flip(coords, -1), coords.shape[-1], axis=-1)]
        for i in range(coords.shape[0]): # Loop over each source
            index = 0 if source_encoding else i
            self.mask = self.mask.at[index, 0,  *jax.numpy.flip(coords, [-1])[i]].set(1.)
    
    def forward_source_encoding(self, wavefield, wavelet):
        wavefield = wavefield.at[..., *self.coords_r].add(wavelet)
        return wavefield
    
    def forward_adjoint_modeling(self, wavefield, wavelet):

        # for i in range(self.coords.shape[0]): # Loop over each shot
        #     wavefield = wavefield.at[i, 0, self.coords[i, :, 1], self.coords[i, :, 0]].add(wavelet[i])
        wavefield = wavefield.at[jnp.repeat(jnp.arange(self.coords.shape[0]), self.coords.shape[1]), 
                                 0, 
                                 self.coords[..., 1].reshape(-1), 
                                 self.coords[..., 0].reshape(-1)].add(wavelet.reshape(-1))
        return wavefield

    def __call__(self, *args):
        if self.se:
            return self.forward_source_encoding(*args)
        else:
            if self.adj:
                return self.forward_adjoint_modeling(*args)
            else:
                return super().forward(*args)
        
        
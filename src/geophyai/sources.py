import jax, torch

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
    def __init__(self, coords, shape, dev, source_encoding=False):
        """Source class for the wave equation

        Args:
            coords (torch.Tensor): Source coordinates (nshots, 2)
            shape (torch.Tensor): Wavefield shape for generating source mask(batch, 1, nz, nx)
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        self.mask = torch.zeros(shape, dtype=torch.float32, device=dev)
        for i in range(coords.shape[0]):
            index = 0 if source_encoding else i
            self.mask[index, :, coords[i, 1], coords[i, 0]] = 1.

    def forward(self, *args):
        return super().forward(*args)
    
class SourceJax(SourceBase):
    def __init__(self, coords, shape, source_encoding=False):
        """Source class for the wave equation

        Args:
            coords (jax.numpy.ndarray): Source coordinates (nshots, 2)
            shape (jax.numpy.ndarray): Wavefield shape for generating source mask(batch, 1, nz, nx)
        """
        super().__init__()
        self.mask = jax.numpy.zeros(shape, dtype=jax.numpy.float32)
        for i in range(coords.shape[0]):
            index = 0 if source_encoding else i
            self.mask = self.mask.at[index, 0, coords[i, 1], coords[i, 0]].set(1.)

    def __call__(self, *args):
        return super().forward(*args)
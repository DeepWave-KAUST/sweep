import torch

class Source(torch.nn.Module):
    def __init__(self, coords, shape, dev):
        """Source class for the wave equation

        Args:
            coords (torch.Tensor): Source coordinates (nshots, 2)
            shape (torch.Tensor): Wavefield shape for generating source mask(batch, 1, nz, nx)
        """
        super(Source, self).__init__()
        self.mask = torch.zeros(shape, dtype=torch.float32, device=dev)
        for i in range(coords.shape[0]):
            self.mask[i, :, coords[i, 1], coords[i, 0]] = 1.
        

    def forward(self, wavefield, wavelet):
        """Forward pass of the source

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)
            wavelet (torch.Tensor): Wavelet tensor (1,)

        Returns:
            torch.Tensor: The wavefield with the source injected
        """
        return wavefield + self.mask * wavelet
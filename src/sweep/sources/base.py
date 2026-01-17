
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

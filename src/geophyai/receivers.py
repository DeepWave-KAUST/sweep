import torch

class Receiver(torch.nn.Module):

    def __init__(self, coords):
        """Receiver class for the wave equation

        Args:
            coords (torch.Tensor): Receiver coordinates (nshots, nreceivers, 2)
        """
        super(Receiver, self).__init__()
        batch, nreceivers, _ = coords.shape
        self.x, self.z = coords[..., 0].flatten().to(torch.int64), coords[..., 1].flatten().to(torch.int64)
        self.bidx = torch.tensor([[i]*nreceivers for i in range(batch)], dtype=torch.int64).flatten()

    def forward(self, wavefield):
        """Forward pass of the receiver

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)

        Returns:
            torch.Tensor: The wavefield at the receiver locations
        """
        return wavefield[self.bidx, :, self.z, self.x]

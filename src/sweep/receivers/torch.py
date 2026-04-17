from sweep.receivers.base import ReceiverBase
import torch


class ReceiverTorch(ReceiverBase, torch.nn.Module):

    def __init__(self, coords):
        """Receiver class for the wave equation

        Args:
            coords (torch.Tensor): Receiver coordinates (nshots, nreceivers, 2)
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        batch, nreceivers, _ = coords.shape
        self.coords_r = [c.flatten().to(torch.int64) for c in torch.split(torch.flip(coords, (-1,)), 1, dim=-1)]
        self.bidx = torch.arange(batch, device=coords.device, dtype=torch.int64).repeat_interleave(nreceivers)

    def forward(self, wavefield):
        """Forward pass of the receiver

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)

        Returns:
            torch.Tensor: The wavefield at the receiver locations
        """
        return super().forward(wavefield)

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
        self.batch = batch
        self.nreceivers = nreceivers
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

    def sample_fields(self, wavefields):
        """Sample multiple receiver fields in one gather operation.

        Args:
            wavefields (Sequence[torch.Tensor] | torch.Tensor): Either a list of
                `(batch, 1, ...)` tensors or one `(batch, nfields, ...)` tensor.

        Returns:
            torch.Tensor: Receiver samples with shape
                `(batch, nreceivers, nfields)`.
        """
        if isinstance(wavefields, (list, tuple)):
            if len(wavefields) == 1:
                gathered = self.forward(wavefields[0])
                return gathered.view(self.batch, self.nreceivers, 1)
            wavefields = torch.cat(wavefields, dim=1)

        if wavefields.ndim != 4 and wavefields.ndim != 5:
            raise ValueError(
                f"sample_fields expects stacked wavefields with ndim 4 or 5, got {wavefields.ndim}"
            )

        gathered = wavefields[(self.bidx, slice(None), *self.coords_r)]
        return gathered.view(self.batch, self.nreceivers, wavefields.shape[1])

from sweep.receivers.base import ReceiverBase
import torch


class ReceiverTorch(ReceiverBase, torch.nn.Module):

    def __init__(self, coords, gather_kernel=None):
        """Receiver class for the wave equation

        Args:
            coords (torch.Tensor): Receiver coordinates (nshots, nreceivers, 2)
            gather_kernel (torch.Tensor, optional): Small 2-D stencil (e.g. a
                3x3 binomial) applied as a weighted gather around each
                receiver cell. Equations whose collocated stencils have a
                checkerboard null space (the rotated staggered grid) declare
                it via ``equation.source_receiver_stencil`` so point sampling
                does not pick up the spurious mode. ``None`` keeps the plain
                single-cell gather.
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        batch, nreceivers, _ = coords.shape
        self.batch = batch
        self.nreceivers = nreceivers
        self.coords_r = [c.flatten().to(torch.int64) for c in torch.split(torch.flip(coords, (-1,)), 1, dim=-1)]
        self.bidx = torch.arange(batch, device=coords.device, dtype=torch.int64).repeat_interleave(nreceivers)
        self.gather_offsets = None
        if gather_kernel is not None:
            if len(self.coords_r) != 2:
                raise NotImplementedError("receiver gather_kernel is only supported for 2-D wavefields")
            half = gather_kernel.shape[-1] // 2
            offsets = []
            for dz in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    w = float(gather_kernel[dz + half, dx + half])
                    if w != 0.0:
                        offsets.append((dz, dx, w))
            self.gather_offsets = offsets

    def _gather(self, wavefields):
        if self.gather_offsets is None:
            return wavefields[(self.bidx, slice(None), *self.coords_r)]
        zc, xc = self.coords_r
        out = None
        for dz, dx, w in self.gather_offsets:
            part = w * wavefields[(self.bidx, slice(None), zc + dz, xc + dx)]
            out = part if out is None else out + part
        return out

    def forward(self, wavefield):
        """Forward pass of the receiver

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)

        Returns:
            torch.Tensor: The wavefield at the receiver locations
        """
        if self.gather_offsets is None:
            return super().forward(wavefield)
        return self._gather(wavefield)

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

        gathered = self._gather(wavefields)
        return gathered.view(self.batch, self.nreceivers, wavefields.shape[1])

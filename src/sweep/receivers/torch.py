from sweep.receivers.base import ReceiverBase
import numpy as np
import torch


class ReceiverTorch(ReceiverBase, torch.nn.Module):

    def __init__(self, coords, gather_kernel=None, shape=None):
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
            shape (tuple, optional): Wavefield shape ``(B, 1, *spatial)``.
                Given, a field is sampled by a flat ``index_select``, whose
                backward is an ``index_add``; advanced indexing's backward is an
                ``index_put(accumulate=True)``, which sorts its indices -- about
                0.25 ms per backward step on a GPU. Several fields are gathered
                one by one instead of concatenated (a full copy of each) first.
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
        self.taps = None if shape is None else self._flat_taps(coords, shape)

    def _flat_taps(self, coords, shape):
        """``[(flat index, weight or None), ...]``, one per gather offset."""
        spatial = tuple(int(n) for n in shape[2:])
        cells = torch.flip(coords, (-1,)).detach().to("cpu", torch.long).reshape(-1, len(spatial)).numpy()
        shots = np.repeat(np.arange(self.batch), self.nreceivers)
        strides = np.cumprod((1,) + spatial[:0:-1])[::-1]
        if self.gather_offsets is None:
            offsets = [(np.zeros(len(spatial), np.int64), None)]
        else:
            offsets = [(np.array([dz, dx], np.int64), w) for dz, dx, w in self.gather_offsets]
        taps = []
        for off, w in offsets:
            at = cells + off
            outside = ((at < 0) | (at >= np.array(spatial))).any(axis=1)
            if outside.any():
                raise ValueError(
                    f"receiver sample at {tuple(int(v) for v in at[outside][0][::-1])} lies outside "
                    f"the {spatial[::-1]} (x, [y,] z) runtime grid")
            flat = shots * int(np.prod(spatial)) + at @ strides
            taps.append((torch.as_tensor(flat, device=coords.device), w))
        return taps

    def _gather(self, wavefields):
        if self.gather_offsets is None:
            return wavefields[(self.bidx, slice(None), *self.coords_r)]
        zc, xc = self.coords_r
        out = None
        for dz, dx, w in self.gather_offsets:
            part = w * wavefields[(self.bidx, slice(None), zc + dz, xc + dx)]
            out = part if out is None else out + part
        return out

    def _gather_flat(self, wavefield):
        """``(B * nrec,)`` samples of one ``(B, 1, *spatial)`` field."""
        flat = wavefield.reshape(-1)
        out = None
        for index, w in self.taps:
            part = flat.index_select(0, index)
            if w is not None:
                part = w * part
            out = part if out is None else out + part
        return out

    def forward(self, wavefield):
        """Forward pass of the receiver

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)

        Returns:
            torch.Tensor: The wavefield at the receiver locations
        """
        if self.taps is not None:
            return self._gather_flat(wavefield)
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
            if self.taps is not None:
                gathered = torch.stack([self._gather_flat(w) for w in wavefields], dim=-1)
                return gathered.view(self.batch, self.nreceivers, len(wavefields))
            wavefields = torch.cat(wavefields, dim=1)

        if wavefields.ndim != 4 and wavefields.ndim != 5:
            raise ValueError(
                f"sample_fields expects stacked wavefields with ndim 4 or 5, got {wavefields.ndim}"
            )

        gathered = self._gather(wavefields)
        return gathered.view(self.batch, self.nreceivers, wavefields.shape[1])

import torch
import torch.nn.functional as F
from sweep.sources.base import SourceBase

class SourceTorch(SourceBase, torch.nn.Module):
    def __init__(self, coords, shape, dev, source_encoding=False, adj=False,
                 spread_kernel=None):
        """Source class for the wave equation

        Args:
            coords (torch.Tensor): Source coordinates (nshots, 2)
            shape (torch.Tensor): Wavefield shape for generating source mask(batch, 1, nz, nx)
            spread_kernel (torch.Tensor, optional): Small 2-D stencil (e.g. a
                3x3 binomial) convolved into the injection mask. Equations
                whose collocated stencils have a checkerboard null space
                (the rotated staggered grid) declare it via
                ``equation.source_receiver_stencil`` so a point source does
                not excite the spurious mode. ``None`` keeps the plain
                single-cell injection.
        """
        torch.nn.Module.__init__(self)
        super().__init__()
        self.mask = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.se = source_encoding
        self.coords = coords
        self.adj = adj
        self.spread_kernel = None
        if spread_kernel is not None:
            if len(shape) != 4:
                raise NotImplementedError("source spread_kernel is only supported for 2-D wavefields")
            k = torch.as_tensor(spread_kernel, dtype=torch.float32, device=dev)
            self.spread_kernel = k.view(1, 1, *k.shape)

        flipped = torch.flip(coords, [-1])
        batch_idx = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        if not source_encoding:
            batch_idx = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)
        index = (batch_idx, slice(None), *flipped.unbind(-1))
        self.mask[index] = 1.
        if self.spread_kernel is not None:
            pad = self.spread_kernel.shape[-1] // 2
            self.mask = F.conv2d(self.mask, self.spread_kernel, padding=pad)

    def _source_values(self, wavelet, batch_size, nsrc):
        values = wavelet.reshape(-1)
        expected = batch_size * nsrc
        if values.numel() == 1:
            return values.expand(expected)
        if values.numel() == batch_size:
            return values.repeat_interleave(nsrc)
        if values.numel() != expected:
            raise ValueError(
                f"Wavelet time slice has {values.numel()} values, expected 1, "
                f"{batch_size}, or {expected} for {batch_size} batch(es) and {nsrc} source(s)."
            )
        return values

    def _add_indexed_sources(self, wavefield, wavelet, *, encoded=False):
        coords = self.coords
        batch_size, nsrc, ndim = coords.shape
        values = self._source_values(wavelet, batch_size, nsrc)

        if encoded:
            batch_idx = torch.zeros(batch_size * nsrc, dtype=torch.long, device=coords.device)
        else:
            batch_idx = torch.arange(batch_size, dtype=torch.long, device=coords.device).repeat_interleave(nsrc)
        channel_idx = torch.zeros(batch_size * nsrc, dtype=torch.long, device=coords.device)
        spatial = [c.reshape(-1).to(torch.long) for c in torch.flip(coords, [-1]).unbind(-1)]

        if self.spread_kernel is not None:
            delta = torch.zeros_like(wavefield)
            delta.index_put_((batch_idx, channel_idx, *spatial), values, accumulate=True)
            pad = self.spread_kernel.shape[-1] // 2
            return wavefield + F.conv2d(delta, self.spread_kernel, padding=pad)

        out = wavefield.clone()
        out.index_put_((batch_idx, channel_idx, *spatial), values, accumulate=True)
        return out

    def forward_source_encoding(self, wavefield, wavelet):
        return self._add_indexed_sources(wavefield, wavelet, encoded=True)
    
    def forward_adjoint_modeling(self, wavefield, wavelet):
        return self._add_indexed_sources(wavefield, wavelet, encoded=False)

    def forward(self, *args):
        if self.se:
            return self.forward_source_encoding(*args)
        else:
            if self.adj:
                return self.forward_adjoint_modeling(*args)
            else:
                return super().forward(*args)
    

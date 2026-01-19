import torch
from sweep.sources.base import SourceBase

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
        out = wavefield.clone()
        out[0, 0, z, x] = wavefield[0, 0, z, x] + wavelet
        return out
    
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
    
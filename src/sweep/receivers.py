import jax
import torch
import jax.numpy as jnp

class ReceiverBase:
    def __init__(self, **kwargs):
        pass

    def forward(self, wavefield):
        return wavefield[self.bidx, :, *self.coords_r]


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
        self.bidx = torch.tensor([[i]*nreceivers for i in range(batch)], dtype=torch.int64).flatten()

    def forward(self, wavefield):
        """Forward pass of the receiver

        Args:
            wavefield (torch.Tensor): Wavefield tensor (batch, 1, nz, nx)

        Returns:
            torch.Tensor: The wavefield at the receiver locations
        """
        return super().forward(wavefield)
    
class ReceiverJax(ReceiverBase):

    def __init__(self, coords):
        """Receiver class for the wave equation

        Args:
            coords (jax.numpy.ndarray): Receiver coordinates (nshots, nreceivers, 2)
        """
        super().__init__()
        batch, nreceivers, _ = coords.shape
        self.coords_r = [c.flatten() for c in jnp.split(jnp.flip(coords, -1), coords.shape[-1], axis=-1)]
        self.bidx = jnp.array([[i]*nreceivers for i in range(batch)], dtype=jnp.int32).flatten()

    def __call__(self, *args):
        return super().forward(*args)

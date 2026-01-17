from sweep.receivers.base import ReceiverBase
import jax.numpy as jnp

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
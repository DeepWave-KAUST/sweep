from sweep.receivers.base import ReceiverBase
import jax.numpy as jnp

class ReceiverJax(ReceiverBase):

    def __init__(self, coords, gather_kernel=None):
        """Receiver class for the wave equation

        Args:
            coords (jax.numpy.ndarray): Receiver coordinates (nshots, nreceivers, 2)
            gather_kernel: Small 2-D stencil (e.g. a 3x3 binomial) applied as
                a weighted gather around each receiver cell. Equations whose
                collocated stencils have a checkerboard null space (the
                rotated staggered grid) declare it via
                ``equation.source_receiver_stencil`` so point sampling does
                not pick up the spurious mode. ``None`` keeps the plain
                single-cell gather. Mirrors ``ReceiverTorch``.
        """
        super().__init__()
        batch, nreceivers, _ = coords.shape
        self.coords_r = [c.flatten() for c in jnp.split(jnp.flip(coords, -1), coords.shape[-1], axis=-1)]
        self.bidx = jnp.array([[i]*nreceivers for i in range(batch)], dtype=jnp.int32).flatten()
        self.gather_offsets = None
        if gather_kernel is not None:
            if len(self.coords_r) != 2:
                raise NotImplementedError("receiver gather_kernel is only supported for 2-D wavefields")
            k = jnp.asarray(gather_kernel, dtype=jnp.float32)
            half = k.shape[-1] // 2
            self.gather_offsets = [
                (dz, dx, float(k[dz + half, dx + half]))
                for dz in range(-half, half + 1)
                for dx in range(-half, half + 1)
                if float(k[dz + half, dx + half]) != 0.0
            ]

    def __call__(self, wavefield):
        if self.gather_offsets is None:
            return super().forward(wavefield)
        zc, xc = self.coords_r
        out = None
        for dz, dx, w in self.gather_offsets:
            part = w * wavefield[(self.bidx, slice(None), zc + dz, xc + dx)]
            out = part if out is None else out + part
        return out
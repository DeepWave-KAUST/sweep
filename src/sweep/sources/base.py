
class SourceBase:

    def __init__(self, **kwargs):
        pass

    def forward(self, wavefield, wavelet):
        """Forward pass of the source.

        Args:
            wavefield: Wavefield tensor ``(B, 1, nz, nx)`` (2-D) or
                ``(B, 1, nz, ny, nx)`` (3-D).
            wavelet: Per-timestep wavelet sample. Scalar for shared-wavelet
                mode (A1), or shape ``(B,)`` for per-shot mode (A2). The
                shape is expanded over the spatial axes for broadcasting.

        Returns:
            Wavefield with the source injected.
        """
        if wavelet.ndim >= 1:
            # Reshape (B,) -> (B, 1, 1, ...) so it broadcasts over channel +
            # spatial dims of (B, 1, ...) wavefields.
            wavelet = wavelet.reshape(wavelet.shape + (1,) * (wavefield.ndim - wavelet.ndim))
        return wavefield + self.mask * wavelet

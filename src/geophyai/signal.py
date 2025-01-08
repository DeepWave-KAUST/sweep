import numpy as np

def ricker(t, f=10.):
    """Ricker wavelet.

    Args:
        t (Array): Discrete time.
        f (float, optional): Dominant frequency. Defaults to 10..

    Returns:
        Array: Ricker wavelet series.
    """
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return r.astype(np.float32)
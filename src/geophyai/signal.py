import torch
import numpy as np
from scipy import signal
from torchaudio.functional import filtfilt

def decide_filter_type(freq):
    """Summary: Decide the filter type

    Args:
        freq (in, float, list): The frequency of the filter.

    Returns:
        str : The filter type.
    """
    if isinstance(freq, (int, float)): filter_mode = "lowpass"
    if isinstance(freq, list): 
        if len(freq) == 1: filter_mode = "lowpass"
        if len(freq) == 2: filter_mode = "bandpass"
    if freq == "all": filter_mode = "all"
    return filter_mode

def filter(d, freqs, dt=0.001, forder=3, btype=None, **kwargs):
    """Butterworth filter.

    Args:
        d (torch.Tensor): Data to be filtered. (nbatch, nsamples, ntraces, nchannels)
        freqs (int, float, list): The frequency of the filter.
        dt (float, optional): Sampling interval. Defaults to 0.001.
        forder (int, optional): Filter order. Defaults to 3.
        btype (str, optional): Filter type. Defaults to None, which will be decided by the function.
    """
    assert d.ndim == 4, "The input data must be 4D (nbatch, nsamples, ntraces, nchannels)."
    
    filter_mode = decide_filter_type(freqs) if btype is None else btype

    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    wn = [2*freq/(1/dt) for freq in list(freqs)]
    wn = wn[0] if len(wn)==1 else wn
    b, a = signal.butter(forder, Wn=wn, btype=filter_mode)

    b = torch.from_numpy(b).double().to(d.device)
    a = torch.from_numpy(a).double().to(d.device)

    # Move the time axis to the last axis
    d = d.permute(0, 3, 2, 1)
    d = filtfilt(d.double(), a, b, clamp=False)
    d = d.float()
    # Move the time axis back to the original position
    d = d.permute(0, 3, 2, 1)
    return d

def frequency_analysis(d, dt=0.001, axis=0):
    """Frequency analysis of a signal.

    Args:
        d (torch.Tensor): Data to be analyzed.
        dt (float, optional): Sampling interval. Defaults to 0.001.
        axis (int, optional): Axis to analyze. Defaults to 0.

    Returns:
        torch.Tensor: Frequency spectrum.
    """
    size = d.shape[axis]//2
    # if size % 2 == 0: size += 1
    backend = torch if isinstance(d, torch.Tensor) else np
    fft_freq = backend.fft.fft(d, axis=axis)
    freqs = backend.fft.fftfreq(d.shape[axis], dt)[:size] 
    #sum the absolute value of the frequency spectrum excluding the specified axis
    sum_axis = tuple(i for i in range(d.ndim) if i != axis)
    amp = backend.sum(backend.abs(fft_freq), axis=sum_axis)[:size] 
    return amp, freqs

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


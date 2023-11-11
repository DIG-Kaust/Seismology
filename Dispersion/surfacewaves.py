import numpy as np
import scipy as sp
import warnings
from pylops.signalprocessing import FFT2D


def _tcrop(t):
    """Crop time axis with even number of samples"""
    if len(t) % 2 == 0:
        t = t[:-1]
        warnings.warn("one sample removed from time axis...")
    return t


def ormsby(t, f=(5.0, 10.0, 45.0, 50.0), taper=None):
    r"""Ormsby wavelet

    Create a Ormsby wavelet given time axis ``t`` and frequency range
    defined by four frequencies which parametrize a trapezoidal shape in
    the frequency spectrum.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f : :obj:`tuple`, optional
        Frequency range
    taper : :obj:`func`, optional
        Taper to apply to wavelet (must be a function that
        takes the size of the window as input

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """

    def numerator(f, t):
        """The numerator of the Ormsby wavelet"""
        return (np.sinc(f * t) ** 2) * ((np.pi * f) ** 2)

    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    f1, f2, f3, f4 = f

    pf43 = (np.pi * f4) - (np.pi * f3)
    pf21 = (np.pi * f2) - (np.pi * f1)
    w = (
        (numerator(f4, t) / pf43)
        - (numerator(f3, t) / pf43)
        - (numerator(f2, t) / pf21)
        + (numerator(f1, t) / pf21)
    )
    w = w / np.amax(w)
    wcenter = np.argmax(np.abs(w))

    # apply taper
    if taper is not None:
        w *= taper(len(t))

    return w, t, wcenter



def surfacewavedata(nt, dt, nx, dx, nfft, fdisp, vdisp, wav):
    r"""Surface wave data

    Synthetise surface wave only seismic data from dispersion relation

    Parameters
    ----------
    nt : :obj:`int`
        number of time samples
    dt : :obj:`float`
        time sampling
    nx : :obj:`int`
        number of spatial samples
    dx : :obj:`float`
        spatial sampling
    nx : :obj:`int`
        number of fft samples
    fdisp : :obj:`int`
        frequency axis of dispersion relation
    vdisp : :obj:`int`
        velocity axis of dispersion relation in km/s
    wav : :obj:`numpy.ndarray`
        source wavelet
    
    Returns
    -------
    dshift : :obj:`numpy.ndarray`
        Data

    """

    # Axes and gridded phase velocity
    t, x = np.arange(nt)*dt, np.arange(nx)*dx
    f = np.fft.rfftfreq(nfft, dt)
    vf = np.interp(f, fdisp, vdisp)

    # Create data
    data = np.outer(wav, np.ones(nx)).T
    D = np.fft.rfft(data, n=nfft, axis=1)

    # Define and apply freq-dependant shifts
    shifts = np.outer(x, 1e-3/vf)
    shifts = np.exp(-1j * 2 * np.pi * f[np.newaxis, :] * shifts)

    Dshift = D * shifts
    dshift = np.fft.irfft(Dshift, n=nfft, axis=1)[:, :nt]

    return dshift, f, vf
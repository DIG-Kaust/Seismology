import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal.windows import gaussian as spgauss


def ricker(t, f0=10, plotflag=False):
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f0 : :obj:`float`, optional
        Central frequency

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if len(t)%2 == 0:
        t = t[:-1]
        warnings.warn('one sample removed from time axis...')

    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)

    w = np.concatenate((np.flipud(w[1:]), w), axis=0)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    if plotflag:
        plt.figure(figsize=(7, 3))
        plt.plot(t, w, 'k', lw=2)
        plt.title('Ricker wavelet')
        plt.xlabel('t')

    return w, t, wcenter


def gaussian(t, std=1, plotflag=False):
    r"""Gaussian wavelet

    Create a Gaussian wavelet given time axis ``t``
    and standard deviation ``std`` using
    :py:func:`scipy.signal.windows.gaussian`.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    std : :obj:`float`, optional
        Standard deviation of gaussian

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if len(t)%2 == 0:
        t = t[:-1]
        warnings.warn('one sample removed from time axis...')

    w = spgauss(len(t)*2-1, std=std)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    if plotflag:
        plt.figure(figsize=(7, 3))
        plt.plot(t, w, 'k', lw=2)
        plt.title('Gaussian wavelet')
        plt.xlabel('t')

    return w, t, wcenter


def cosine(t, extent=5, square=False, plotflag=False):
    r"""Cosine/Cosine square wavelet

    Create a Cosine or Cosine square wavelet given time axis ``t``
    and extent ``extent`` and square ``flag``
    :py:func:`scipy.signal.windows.gaussian`.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    extent : :obj:`int`, optional
        Half-width of wavelet in number of samples
    square : :obj:`bool`, optional
        square cosine (``True``) or not (``False``)
    plotflag : :obj:`bool`, optional
        Quickplot

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if t.size % 2 == 0:
        t = t[:-1]
        warnings.warn('one sample removed from time axis...')

    nt = t.size
    if extent > nt:
        raise ValueError('extent={} is bigger than time '
                         'axis size of {} samples...'.format(extent, nt))

    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    w = np.zeros(t.size)

    wtmp = np.cos(np.arange(extent+1)*np.pi/extent) + 1.
    if square:
        wtmp = wtmp**2
    wtmp = np.hstack([wtmp[::-1], wtmp[1:]]) / wtmp.max()

    w[nt-extent:nt+extent+1] = wtmp
    wcenter = np.argmax(np.abs(w))

    if plotflag:
        plt.figure(figsize=(7, 3))
        plt.plot(t, w, 'k', lw=2)
        plt.title('Cosine wavelet')
        plt.xlabel('t')

    return w, t, wcenter

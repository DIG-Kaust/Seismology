import numpy as np

def moving_average(signal, length):
    """Moving average.

    Compute average in slidings window handling reduction in samples at the
    edges of the input ``signal``

    Parameters
    ----------
    signal : :obj:`np.ndarray`
        Input signal
    length : :obj:`int`
        Length of averaging window
    Returns
    -------
    smooth : :obj:`np.ndarray`
        Output smoothed signal


    """
    padded = np.pad(signal, int(length/2), mode='edge')
    boxcar = np.ones(int(length))/length
    smooth = np.convolve(padded, boxcar, mode='same')
    return smooth[int(length/2):-int(length/2)]

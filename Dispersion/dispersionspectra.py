import numpy as np
import scipy as sp
from pylops.signalprocessing import FFT2D


def parkdispersion(data, dx, dt, cmin, cmax, dc, fmax):
    """Dispersion panel
    
    Calculate dispersion curves using the method of
    Park et al. 1998
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size `nx x nt`
    dx : :obj:`float`
        Spatial sampling
    dt : :obj:`float`
        Time sampling
    cmin : :obj:`float`
        Minimum velocity
    cmax : :obj:`float`
        Maximum velocity
    dc : :obj:`float`
        Velocity sampling
    fmax : :obj:`float`
        Maximum frequency
        
    Returns
    -------
    f : :obj:`numpy.ndarray`
        Frequency axis
    c : :obj:`numpy.ndarray`
        Velocity axis`
    disp : :obj:`numpy.ndarray`
        Dispersion panel of size `nc x nf`
    """
    nr, nt = data.shape
    
    # Axes
    t = np.linspace(0.0, nt*dt, nt)

    f = sp.fftpack.fftfreq(nt, dt)[:nt//2]
    df = f[1] - f[0]
    fmax_idx = int(fmax//df)

    c = np.arange(cmin, cmax, dc)  # set phase velocity range
    x = np.linspace(0.0, (nr-1)*dx, nr)

    # Fx spectrum
    U = sp.fftpack.fft(data)[:, :nt//2]
    
    # Dispersion panel
    disp = np.zeros((len(c), fmax_idx))
    for fi in range(fmax_idx):
        for ci in range(len(c)):
            k = 2.0*np.pi*f[fi]/(c[ci])
            disp[ci, fi] = np.abs(
                np.dot(dx * np.exp(1.0j*k*x), U[:, fi]/np.abs(U[:, fi])))

    return f[:fmax_idx], c, disp
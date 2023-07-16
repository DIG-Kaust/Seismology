import numpy as np
import scipy as sp
from pylops.signalprocessing import FFT2D


def fkdispersion(data, nfft, dx, dt, vdisp):
    
    nx, nt = data.shape
    nvel = vdisp.size
    
    # Fk spectrum
    F2Op = FFT2D(dims=(nx, nt), dirs=(-2, -1), nffts=(nfft, nfft), sampling=(dx, dt), real=True)
    f = F2Op.f2
    kx = np.flipud(F2Op.f1[:nfft//2-1])
    D = (F2Op @ data.ravel()).reshape(nfft, nfft//2+1)
    D = np.fft.fftshift(np.abs(D), axes=0)[:nfft//2-1]
    
    # Resample v axis
    Ddisp = np.zeros((nvel, nfft//2+1))
    for iif in np.arange(1, nfft//2+1):
        vdisp_sampled = f[iif]/kx
        vdisp_sampled[0] = vdisp_sampled[1]
        Ddisp[:, iif] = np.interp(vdisp, vdisp_sampled, D[:, iif])
    return Ddisp


def get_fft(traces, dt, nt):
    # Get temporal Fourier transform for each of the traces
    # f = np.linspace(0.0, 1.0/(2.0*dt), nt//2)
    f = sp.fftpack.fftfreq(nt, dt)
    U = sp.fftpack.fft(traces)
    if np.size(U.shape) > 1:
        return U[:, 0:nt//2], f[0:nt//2]
    else:
        return U[0:nt//2], f[0:nt//2]


def parkdispersion(traces, dx, dt, cmin, cmax, dc, fmax):
    """ calculate dispersion curves after Park et al. 1998
    INPUTS
    traces: SU traces
    dx: distance between stations (m)
    cmax: upper velocity limit (m/s)
    fmax: upper frequency limit (Hz)
    OUTPUTS
    f: 1d array frequency vector
    c: 1d array phase velocity vector
    img: 2d array (c x f) dispersion image
    fmax_idx: integer index corresponding to the given fmax
    U: 2d array (nr x npts//2) Fourier transform of traces
    t: 1d array time vector
    """
    nr, nt = traces.shape
    t = np.linspace(0.0, nt*dt, nt)
    
    # Fx spectrum
    U, f = get_fft(traces, dt, nt)
    c = np.arange(cmin, cmax, dc)  # set phase velocity range
    df = f[1] - f[0]
    fmax_idx = int(fmax//df)
    
    img = np.zeros((len(c), fmax_idx))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    for fi in range(fmax_idx):  # loop over frequency range
        for ci in range(len(c)):  # loop over phase velocity range
            k = 2.0*np.pi*f[fi]/(c[ci])
            img[ci, fi] = np.abs(
                np.dot(dx * np.exp(1.0j*k*x), U[:, fi]/np.abs(U[:, fi])))

    return f, c, img, fmax_idx, U, t
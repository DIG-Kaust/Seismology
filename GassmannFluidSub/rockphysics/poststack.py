import logging
import numpy as np

from scipy.signal import fftconvolve
from .elastic import backus


def zerooffset_mod(ai, wav, wavcenter=None):
    r"""Create zero-offset seismic trace from 1-, 2- or 3-dimensional acoustic
    impedence profile

    Parameters
    ----------
    ai : :obj:`np.ndarray`
        acoustic impedence profile of size :math:`n_z (\times n_y \times n_x)`
    wav : :obj:`np.ndarray`
        Wavelet
    wavcenter : :obj:`int`, optional
        Index of wavelet center (if ``None`` using middle value of ``wav``)

    Returns
    -------
    seismic : :obj:`np.ndarray`
        Zero-offset seismic of size :math:`n_z (\times n_y \times n_x)`
    reflectivity : :obj:`np.ndarray`
        Reflectivity of size :math:`n_z-1 (\times n_y \times n_x)`

    """
    # Normal incidence reflectivity and corresponding depth axis
    reflectivity = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
    reflectivity[np.isnan(reflectivity)] = 0.
    if ai.ndim == 2:
        reflectivity = np.pad(reflectivity, ((1, 0), (0, 0)), mode='constant')
    else:
        reflectivity = np.pad(reflectivity, (1, 0), mode='constant')

    # Extract center of wavelet if not provided
    if wavcenter is None:
        wavcenter = len(wav) // 2

    # Convolve with wavelet, adjust length of trace to start from wavelet peak
    if reflectivity.ndim == 1:
        seismic = np.convolve(reflectivity, wav, 'full')
    else:
        if reflectivity.ndim == 2:
            wav = wav[:, np.newaxis]
        else:
            wav = wav[:, np.newaxis, np.newaxis]
        seismic = fftconvolve(reflectivity, wav, mode='full')
    seismic = seismic[wavcenter:-wavcenter]
    return seismic, reflectivity


def zerooffset_wellmod(logs, zaxis, sampling, wav, ax=None, zaxisshift=0.0,
                       wavcenter=None, ai=None, vp=None, vs=None, rho=None,
                       mask=None, zlim=None, plotflag=False, figsize=(12, 9),
                       title=None, savefig=None):
    """Create zero-offset seismic trace from well logs

    Parameters
    ----------
    logs : :obj:`Logs`
        Logs object
    zaxis : :obj:`str`
        Label of log to use as z-axis
    sampling : :obj:`float`
        Sampling step to be used for depth axis. Logs will be
        decimated/interpolated to this sampling
    wav : :obj:`np.ndarray`
        Wavelet
    ax : :obj:`plt.axes`, optional
        Axes handle (if ``None`` create a new figure)
    zaxisshift : :obj:`float`, optional
        Label of log to use as z-axis
    wavcenter : :obj:`int`, optional
        Index of wavelet center (if ``None`` using middle value of ``wav``)
    ai : :obj:`str`, optional
        Label of log to use as acoustic impedance
    vp : :obj:`str`, optional
        Label of log to use as P-wave velocity (if ``ai`` is not provided)
    vs : :obj:`str`, optional
        Label of log to use as S-wave velocity (to apply Backus average from
        well scale sampling to provided ``sampling``)
    rho : :obj:`str`, optional
        Label of log to use as density (if ``ai`` is not provided)
    mask : :obj:`numpy.ndarray`, optional
        Mask to apply to logs prior to modelling
    zlim : :obj:`tuple` or :obj:`list`
        z-axis extremes
    plotflag : :obj:`bool`, optional
            Quickplot
    figsize : :obj:`tuple`, optional
             Size of figure
    title : :obj:`str`, optional
         Title of figure
    savefig : :obj:`str`, optional
         Figure filename, including path of location where to save plot
         (if ``None``, figure is not saved)

    Returns
    -------
    trace : :obj:`np.ndarray`
        Trace
    zaxisreglog : :obj:`np.ndarray`
        z-axis at choosen ``sampling``
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
        Axes handle

    """
    # Get curves for the seismic part of the exercise
    zaxislog = logs.logs[zaxis] + zaxisshift
    zaxisreglog = np.arange(np.nanmin(zaxislog), np.nanmax(zaxislog),
                            sampling)
    if vs is None:
        if ai is None:
            vplog, zreg = logs.resample_curve(vp, zaxis=zaxis, mask=mask)
            rholog, zreg = logs.resample_curve(rho, zaxis=zaxis, mask=mask)
            ailog = vplog * rholog
            ailog = np.interp(zaxisreglog, zreg, ailog)
        else:
            ailog, zreg = logs.resample_curve(ai, zaxis=zaxis, mask=mask)
            ailog = np.interp(zaxisreglog, zreg, ailog)
    else:
        # backus averaging
        vplog, zreg = logs.resample_curve(vp, zaxis=zaxis, mask=mask)
        vslog, zreg = logs.resample_curve(vs, zaxis=zaxis, mask=mask)
        rholog, zreg = logs.resample_curve(rho, zaxis=zaxis, mask=mask)
        # Backus averaging
        vpback, vsback, rhoback = \
            backus(vplog, vslog, rholog,
                   dz=zreg[1]-zreg[0], lb=sampling)
        ailog = vpback * rhoback
        ailog = np.interp(zaxisreglog, zreg, ailog)

    # Create synthetic
    rzaxisreglog = zaxisreglog[1:]
    trace, reflectivity = zerooffset_mod(ailog, wav, wavcenter=wavcenter)

    # Make the plots
    if plotflag:
        # plotting removed due to error in loading _wiggletrace
        logging.warning('plotting is deprecated')
    fig, ax = None, None
    return trace, zaxisreglog, fig, ax


def zerooffset_geomod(geomodel, intervalnames, aveprops,
                      wav, wavcenter=None,
                      ai=None, plotflag=False, savefig=None, **kwargs_view):
    """Create zero-offset seismic trace from geomodel

    Parameters
    ----------
    geomod : :obj:`ptcpy.objects.Seismic` or :obj:`ptcpy.objects.SeismicIrregular`
        Geomodel created via :func:`ptcpy.proc.geomod.geomod.create_geomodel`
        filled with discrete numbers corresponding to different intervals
    intervalnames : :obj:`list`
        Name of intervals to associate to discrete numbers in ``geomod``
    aveprops : :obj:`dict`
        Average elastic properties for each interval computed via
        :func:`ptcpy.objects.Project.create_averageprops_intervals`
        to use to create the
        acoustic impedence model used to model synthetic seismic
    wav : :obj:`np.ndarray`
        Wavelet
    wavcenter : :obj:`int`, optional
        Index of wavelet center (if ``None`` using middle value of ``wav``)
    ai : :obj:`str`, optional
        Label of log to use as acoustic impedance in ``aveprops``
    plotflag : :obj:`bool`, optional
            Quickplot
    figsize : :obj:`tuple`, optional
             Size of figure
    savefig : :obj:`str`, optional
         Figure filename, including path of location where to save plot
         (if ``None``, figure is not saved)
    kwargs_view : :obj:`dict`, optional
         Additional parameters to pass to :func:`ptcpy.objects.Seismic.view`

    Returns
    -------
    seismod : :obj:`ptcpy.objects.Seismic` or :obj:`ptcpy.objects.SeismicIrregular`
        Zero-offset seismic object
    aigeomod : :obj:`ptcpy.objects.Seismic` or :obj:`ptcpy.objects.SeismicIrregular`
        Acoustic impedence model object
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
        Axes handle

    """
    aigeomod = geomodel.copy(empty=True)

    for iinterval, interval in enumerate(intervalnames):
        # average properties
        aigeomod.data[geomodel.data == iinterval] = \
            aveprops[ai][interval]['mean']

    ai = np.swapaxes(aigeomod.data, -1, 0)
    seismic = zerooffset_mod(ai, wav, wavcenter)[0]
    seismic = np.swapaxes(seismic, 0, -1)

    del ai
    seismod = geomodel.copy(empty=True)
    seismod.data = seismic

    # Make the plots
    if plotflag:
        fig, axs = seismod.view(**kwargs_view)
        if savefig is not None:
            fig.savefig(savefig, dpi=300)
    else:
        fig, axs = None, None

    return seismod, aigeomod, fig, axs


def timeshift(z, vbase, vmoni, vupbase=None, vupmoni=None):
    r"""Two-way Timeshift between monitor and baseline

    Compute timeshift between a monitor and baseline velocity profile. If only
    ``vbase`` and ``vmoni`` are provided the timeshift is computed as
    :math:`2 (\int \frac{dz}{vmoni} - \int \frac{dz}{vbase})`. On the other hand,
    if ``vupbase`` and ``vupmoni`` are also provided, timeshift is computed as
    :math:`(\int \frac{dz}{vmoni} + \int \frac{dz}{vupmoni}) - (\int \frac{dz}{vbase} + \int \frac{dz}{vupbase})`

    """
    dz = np.diff(z)
    dz = np.hstack([dz, dz[-1]])
    slowbase = 1. / vbase
    slowmoni = 1. / vmoni
    mask = np.isnan(slowbase) | np.isnan(slowmoni)
    if vupbase is not None and vupmoni is not None:
        slowupbase = 1. / vupbase
        slowupmoni = 1. / vupmoni
        mask = mask | np.isnan(slowupbase) | np.isnan(slowupmoni)
        slowupbase[mask] = 0
        slowupmoni[mask] = 0
    slowbase[mask] = 0
    slowmoni[mask] = 0

    time = np.cumsum(dz * slowbase) * 1000.
    time1 = np.cumsum(dz * slowmoni) * 1000.
    if vupbase is not None and vupmoni is not None:
        time += np.cumsum(dz * slowupbase) * 1000.
        time1 += np.cumsum(dz * slowupmoni) * 1000.
        tshift = time1 - time
    else:
        tshift = 2 * (time1 - time)

    return tshift
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from scipy.signal import fftconvolve

from .elastic import backus
from .visualutils import _wiggletracecomb


def _prepare(func):
    """Make sure the inputs are arrays and add a dimension
    to theta to make the functions work in an outer product way.

    Parameters
    ----------
    func : :obj:`def`
        Any of the methods used :func:`angle_reflectivity`

    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):
        vp1 = np.asanyarray(vp1, dtype=float)
        vs1 = np.asanyarray(vs1, dtype=float) + 1e-12
        rho1 = np.asanyarray(rho1, dtype=float)
        vp2 = np.asanyarray(vp2, dtype=float)
        vs2 = np.asanyarray(vs2, dtype=float) + 1e-12
        rho2 = np.asanyarray(rho2, dtype=float)
        theta1 = np.asanyarray(theta1).reshape((-1, 1))
        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper


##################
# AVO modelling  #
##################
def critical_angles(vp1, vp2, vs2=None):
    """Compute critical angle at an inteface.

    Compute critical angle given P-wave velocity (``vp1``) above an interface
    and P- (``vp2``) and S-wave velocities (``vs2``) below an interface.

    Parameters
    ----------
    vp1 : :obj:`float`
        P-wave velocity above an interface
    vp2 : :obj:`float`
        P-wave velocity below an interface
    vs2 : :obj:`float`
        S-wave velocity above an interface (if ``None``
        PS critical angle is not estimated)

    Args:
        vp1 (ndarray): Upper layer P-wave velocity.
        vp2 (ndarray): Lower layer P-wave velocity.
        vs2 (ndarray): Lower layer S-wave velocity (optional).

    Returns
    -------
    capp : :obj:`float`
        PP critical angle
    caps : :obj:`float`
        PS critical angle (return)

    """
    capp = caps = None

    if vp1 < vp2:
        capp = np.rad2deg(np.arcsin(vp1/vp2))

    if vs2 is not None and vp1 < vs2:
        caps = np.rad2deg(np.arcsin(vp1/vs2))
    return capp, caps


@_prepare
def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """Zoeppritz equation.

    Approximate calculation of PP reflection from the Zoeppritz
    scattering matrix for a set of incident angles [1]_.

    .. [1] Dvorkin et al. Seismic Reflections of Rock Properties.
       Cambridge. 2014.

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    """
    theta1 = np.radians(theta1).astype(complex)

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2)**2.) - rho1 * (1 - 2 * np.sin(phi1)**2.)
    b = rho2 * (1 - 2 * np.sin(phi2)**2.) + 2 * rho1 * np.sin(phi1)**2.
    c = rho1 * (1 - 2 * np.sin(phi1)**2.) + 2 * rho2 * np.sin(phi2)**2.
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1)/vp1 * np.cos(phi2)/vs2
    H = a - d * np.cos(theta2)/vp2 * np.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1/D) * (F*(b*(np.cos(theta1)/vp1) - c*(np.cos(theta2)/vp2)) \
                   - H*p**2 * (a + d*(np.cos(theta1)/vp1)*(np.cos(phi2)/vs2)))
    return np.squeeze(rpp)


@_prepare
def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    r"""Three terms Aki-Richards approximation.

    Approximate calculation of PP reflection from the Aki-Richards
    scattering matrix for a set of incident angles [1]_.

    .. [1] Avseth et al., Quantitative seismic
       interpretation, Cambridge University Press, 2006.

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    """
    theta1 = np.radians(theta1).astype(complex)

    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    meantheta = (theta1+theta2) / 2.0
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0

    # Compute the coefficients
    w = 0.5 * drho/rho
    x = 2 * (vs/vp1)**2 * drho/rho
    y = 0.5 * (dvp/vp)
    z = 4 * (vs/vp1)**2 * (dvs/vs)

    # Compute the terms
    term1 = w
    term2 = -1 * x * np.sin(theta1)**2
    term3 = y / np.cos(meantheta)**2
    term4 = -1 * z * np.sin(theta1)**2

    return np.squeeze(term1 + term2 + term3 + term4)


@_prepare
def akirichards_alt(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    r"""Alternative three terms Aki-Richards approximation.

    Approximate calculation of PP reflection from the Aki-Richards
    scattering matrix for a set of incident angles [1]_.

    .. [1] http://subsurfwiki.org/wiki/Aki-Richards_equation

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    """
    theta1 = np.radians(theta1).astype(complex)

    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    theta = (theta1+theta2)/2.0
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute the three terms
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = (0.5*dvp/vp-2*(vs/vp)**2*(drho/rho+2*dvs/vs)) * np.sin(theta)**2
    term3 = 0.5 * dvp/vp * (np.tan(theta)**2 - np.sin(theta)**2)

    return np.squeeze(term1 + term2 + term3)


@_prepare
def fatti(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    r"""Fatti approximation.

    Approximate calculation of PP reflection from the  Fatti's formulation
    of the Aki-Richards equation, which does not account for the critical
    angle [1]_.

    .. [1] Fatti et al., 1994, Geophysics 59 (9).

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    """
    theta1 = np.radians(theta1)

    drho = rho2-rho1
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0
    dip = (vp2*rho2 - vp1*rho1)/(vp2*rho2 + vp1*rho1)
    dis = (vs2*rho2 - vs1*rho1)/(vs2*rho2 + vs1*rho1)
    d = drho/rho

    # Compute the three terms
    term1 = (1 + np.tan(theta1)**2) * dip
    term2 = -8 * (vs/vp)**2 * dis * np.sin(theta1)**2
    term3 = -1 * (0.5 * np.tan(theta1)**2 - 2 * (vs/vp)**2 * np.sin(theta1)**2) * d

    return np.squeeze(term1 + term2 + term3)


@_prepare
def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, return_gradient=False):
    r"""Shuey approximation.

    Approximate calculation of PP reflection from the Shuey's formulation [1]_.

    .. [1] http://subsurfwiki.org/wiki/Shuey_equation

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
    return_gradient : :obj:`bool`
        Return a tuple with the intercept and gradient
        (i.e. the second term divided by :math:`sin^2(\theta)`).
    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles
    gradint : :obj:`dict`
        Gradient and Intercept

    """
    theta1 = np.radians(theta1)

    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute three-term reflectivity
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2/vp**2) * (drho/rho + 2 * dvs/vs)
    f = 0.5 * dvp/vp

    term1 = r0
    term2 = g * np.sin(theta1)**2
    term3 = f * (np.tan(theta1)**2 - np.sin(theta1)**2)

    if return_gradient:
        gradint = dict(intercept=np.squeeze(r0), gradient=np.squeeze(g))
    else:
        gradint = (None, None)

    return np.squeeze(term1 + term2 + term3), gradint


@_prepare
def bortfeld(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    r"""Bortfeld approximation.

    Approximate calculation of PP reflection from the Bortfeld's formulation [1]_.

    .. [1] http://sepwww.stanford.edu/public/docs/sep111/marie2/paper_html/node2.html

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho2 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
        (i.e. the second term divided by :math:`sin^2(\theta)`).
    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles

    """
    theta1 = np.radians(theta1)

    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0
    k = (2 * vs/vp)**2
    rsh = 0.5 * (dvp/vp - k*drho/rho - 2*k*dvs/vs)

    # Compute three-term reflectivity
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = rsh * np.sin(theta1)**2
    term3 = 0.5 * dvp/vp * np.tan(theta1)**2 * np.sin(theta1)**2

    return np.squeeze(term1 + term2 + term3)


_methods = {'zoeppritz': zoeppritz,
            'akirichards': akirichards,
            'akirichards_alt': akirichards_alt,
            'fatti': fatti,
            'shuey': shuey,
            'bortfeld': bortfeld}

def angle_reflectivity(vp, vs, rho, theta=0, method='zoeppritz'):
    r"""Angle dependent reflectivity.

    Compute angle dependent reflectivity given P-wave (``vp``), S-wave (``vs``),
    and density (``rho``) profiles. Different formulas can be used to
    compute the amplitude-varition-with-angle (AVA) response at each interface
    depending on the choice of the ``method``.

    Parameters
    ----------
    vp : :obj:`np.ndarray`
        P-wave velocity profile
    vs : :obj:`np.ndarray`
        S-wave velocity profile
    rho : :obj:`np.ndarray`
        Density profile
    theta : :obj:`np.ndarray` or :obj:`floar`, optional
        Angles in degrees
    method : :obj:`np.ndarray`, optional
        Method to use (``zoeppritz``, ``akirichards``, ``akirichards_alt``,
        ``fatti``, ``shuey``, and ``bortfeld``

    Returns
    -------
    refl : :obj:`np.ndarray`
        Reflectivity of size :math:`[n_{theta}, n_{z}]` where :math:`n_{z}`
        is the size of the profiles

    Notes
    -------
    Various implementation of the  amplitude-varition-with-angle (AVA) response
    are available in the literature. Here we attempt to provide a unique
    interface to an handful of them:

    * ``zoeppritz``: zoeppritz
    * ``zoeppritz_rpp``: zoeppritz_rpp
    * ``akirichards``: akirichards
    * ``akirichards_alt``: akirichards alternative
    * ``fatti``: fatti
    * ``shuey``: shuey
    * ``bortfeld``': bortfeld

    """
    func = _methods[method.lower()]
    vp = np.asanyarray(vp, dtype=float)
    vs = np.asanyarray(vs, dtype=float)
    rho = np.asanyarray(rho, dtype=float)

    vp1, vp2 = vp[:-1], vp[1:]
    vs1, vs2 = vs[:-1], vs[1:]
    rho1, rho2 = rho[:-1], rho[1:]
    refl = func(vp1, vs1, rho1, vp2, vs2, rho2, theta)
    return refl


def angle_reflectivity_multidims(vp, vs, rho, theta=0, method='zoeppritz'):
    r"""Angle dependent reflectivity for multi-dimensional elastic property
    profiles. Refer to :func:`ptcpy.proc.seismicmod.avo.angle_reflectivity`
    for detailed documentation.

    Parameters
    ----------
    vp : :obj:`np.ndarray`
        P-wave velocity profile of size :math:`(n_y \times n_x) \times n_z`
    vs : :obj:`np.ndarray`
        S-wave velocity profile of size :math:`(n_y \times n_x) \times n_z`
    rho : :obj:`np.ndarray`
        Density profile of size :math:`(n_y \times n_x) \times n_z`
    theta : :obj:`np.ndarray` or :obj:`floar`, optional
        Angles in degrees
    method : :obj:`np.ndarray`, optional
        Method to use (``zoeppritz``, ``akirichards``, ``akirichards_alt``,
        ``fatti``, ``shuey``, and ``bortfeld``

    Returns
    -------
    refl : :obj:`np.ndarray`
        Reflectivity of size :math:`n_{theta} \times
        (n_y \times n_x) \times n_z`

    Notes
    -------
    Various implementation of the  amplitude-varition-with-angle (AVA) response
    are available in the literature. Here we attempt to provide a unique
    interface to an handful of them:

    * ``zoeppritz``: zoeppritz
    * ``zoeppritz_rpp``: zoeppritz_rpp
    * ``akirichards``: akirichards
    * ``akirichards_alt``: akirichards alternative
    * ``fatti``: fatti
    * ``shuey``: shuey
    * ``bortfeld``': bortfeld

    """
    dims = vp.shape
    ntheta = 1 if isinstance(theta, (float, int)) else len(theta)
    refl = angle_reflectivity(vp.flatten(), vs.flatten(), rho.flatten(), theta,
                              method=method)
    if ntheta == 1:
        refl = np.hstack((refl, np.zeros(1)))
        refl = refl.reshape(dims)[..., :-1]

    else:
        refl = np.hstack((refl, np.zeros((ntheta, 1))))
        refl = refl.reshape([ntheta]+list(dims))[..., :-1]
    return refl


def prestack_mod(vp, vs, rho, theta, wav, wavcenter=None, method='zoeppritz'):
    r"""Create angle-dependent seismic gathers from 1-, 2- or 3-dimensional
    vp, vs and rho profiles

    Parameters
    ----------
    vp : :obj:`str`, optional
        P-wave velocity of size :math:`n_{theta} (\times n_y \times n_x) \times n_z`
    vs : :obj:`str`, optional
        S-wave velocity of size :math:`n_{theta} (\times n_y \times n_x) \times n_z`
    rho : :obj:`str`, optional
        Density of size :math:`n_{theta} (\times n_y \times n_x) \times n_z`
    theta : :obj:`np.ndarray`
        Angles in degrees
    wav : :obj:`np.ndarray`
        Wavelet
    wavcenter : :obj:`int`, optional
        Index of wavelet center (if ``None`` using middle value of ``wav``)
    method : :obj:`np.ndarray`, optional
        Method to use (``zoeppritz``, ``akirichards``, ``akirichards_alt``,
        ``fatti``, ``shuey``, and ``bortfeld``)

    Returns
    -------
    seismic : :obj:`np.ndarray`
        Zero-offset seismic of size :math:`n_{theta} (\times n_y \times n_x)
        \times n_z`
    reflectivity : :obj:`np.ndarray`
        Reflectivity of size :math:`n_{theta} (\times n_y \times n_x)
        \times n_z-1`

    """
    # Normal incidence reflectivity and corresponding depth axis
    if vp.ndim == 1:
        refl = np.real(angle_reflectivity(vp, vs, rho,
                                          theta, method=method))
    else:
        refl = np.real(angle_reflectivity_multidims(vp, vs, rho,
                                                    theta, method=method))
    refl[np.isnan(refl)] = 0.

    # Extract center of wavelet if not provided
    if wavcenter is None:
        wavcenter = len(wav) // 2

    # Convolve with wavelet, adjust length of trace to start from wavelet peak
    if refl.ndim == 1:
        seismic = np.convolve(refl, wav, 'full')
    else:
        for i in range(refl.ndim - 1):
            wav = wav[np.newaxis, :]
        seismic = fftconvolve(refl, wav, mode='full')
    seismic = seismic[..., wavcenter:-wavcenter + 1]
    return seismic, refl


def prestack_wellmod(logs, zaxis, theta, sampling, wav, ax=None, wavcenter=None,
                     vp=None, vs=None, rho=None, mask=None,
                     method='zoeppritz', zlim=None,
                     plotflag=False, scaling=1, figsize=(12, 9),
                     title=None, savefig=None, **kwargs_plot):
    """Create angle-dependent seismic gather from well logs

    Parameters
    ----------
    logs : :obj:`ptcpy.objects.Logs.Logs`
        Logs object
    zaxis : :obj:`str`
        Label of log to use as z-axis
    theta : :obj:`np.ndarray`
        Angles in degrees
    sampling : :obj:`float`
        Sampling step to be used for depth axis. Logs will be
        decimated/interpolated to this sampling
    wav : :obj:`float`
        Sampling step to be used for depth axis. Logs will be
        decimated/interpolated to this sampling
    wavcenter : :obj:`float`
        Sampling step to be used for depth axis. Logs will be
        decimated/interpolated to this sampling
    vp : :obj:`str`, optional
        Label of log to use as P-wave velocity
    vs : :obj:`str`, optional
        Label of log to use as S-wave velocity
    rho : :obj:`str`, optional
        Label of log to use as density
    mask : :obj:`numpy.ndarray`, optional
        Mask to apply to logs prior to modelling
    method : :obj:`np.ndarray`, optional
        Method to use (``zoeppritz``, ``akirichards``, ``akirichards_alt``,
        ``fatti``, ``shuey``, and ``bortfeld``)
    zlim : :obj:`tuple` or :obj:`list`
        z-axis extremes
    plotflag : :obj:`bool`, optional
            Quickplot
    scaling : :obj:`float`, optional
        Scaling to apply to all traces
    figsize : :obj:`tuple`, optional
         Size of figure
    title : :obj:`str`, optional
         Title of figure
    savefig : :obj:`str`, optional
         Figure filename, including path of location where to save plot
         (if ``None``, figure is not saved)
    kwargs_plot : :obj:`dict`, optional
        Additional parameters to be passed to
        :func:`ptcpy.visual.utils._wiggletracecomb`

    Returns
    -------
    traces : :obj:`np.ndarray`
        Traces
    zaxisreglog : :obj:`np.ndarray`
        z-axis at choosen ``sampling``
    fig : :obj:`plt.figure`
        Figure handle (``None`` if ``axs`` are passed by user)
    ax : :obj:`plt.axes`
        Axes handle

    """
    # Get curves for the seismic part of the exercise
    zaxislog = logs.logs[zaxis]

    # Create regular axis
    zaxisreglog = np.arange(np.nanmin(zaxislog), np.nanmax(zaxislog), sampling)

    # Backus averaging and resample the logs to the new axis using linear interp
    vplog, zreg = logs.resample_curve(vp, zaxis=zaxis, mask=mask)
    vslog, zreg = logs.resample_curve(vs, zaxis=zaxis, mask=mask)
    rholog, zreg = logs.resample_curve(rho, zaxis=zaxis, mask=mask)

    vplog, vslog, rholog = \
        backus(vplog, vslog, rholog,
               dz=zreg[1] - zreg[0], lb=sampling)

    vplog = np.interp(zaxisreglog, zreg, vplog)
    vslog = np.interp(zaxisreglog, zreg, vslog)
    rholog = np.interp(zaxisreglog, zreg, rholog)

    traces, refl = prestack_mod(vplog, vslog, rholog, theta, wav,
                                wavcenter=wavcenter, method=method)

    # Make the plots
    if plotflag:
        if ax is None:
            fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
            fig.suptitle('' if title is None else title, y=0.99,
                         fontsize=16, fontweight='bold')
            axs[0].plot(vplog*rholog, zaxisreglog, 'k', lw=2)
            axs[0].set_ylabel(zaxis)
            axs[0].set_title('AI')
            axs[1].plot(vplog/vslog, zaxisreglog, 'k', lw=2)
            axs[1].set_ylabel(zaxis)
            axs[1].set_title('VP/VS')
            axs[2] = _wiggletracecomb(axs[2], zaxisreglog, theta,
                                      traces, **kwargs_plot)
            axs[2].set_ylabel(zaxis)
            axs[2].set_title('Seismic')
            if zlim is not None:
                axs[2].set_ylim(zlim[1], zlim[0])
            else:
                axs[2].set_ylim(zaxisreglog[-1], zaxisreglog[0])
        else:
            fig = None
            ax = _wiggletracecomb(ax, zaxisreglog,  theta,
                                  traces, scaling=scaling, **kwargs_plot)
            ax.set_title('Seismic' if title is None else title)
            if zlim is not None:
                ax.set_ylim(zlim[1], zlim[0])
            else:
                ax.set_ylim(zaxisreglog[-1], zaxisreglog[0])
            if title is not None:
                ax.set_title(title)

        if savefig is not None:
            fig.savefig(savefig, dpi=300)
    else:
        fig, ax = None, None

    return traces, zaxisreglog, fig, ax


def prestack_geomod(geomodel, intervalnames, aveprops, wav, theta,
                    wavcenter=None, vp=None, vs=None,
                    rho=None, method='zoeppritz',
                    plotflag=False, savefig=None, **kwargs_view):
    """Create angle-dependent seismic gather from geomodel

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
    theta : :obj:`np.ndarray`
        Angles in degrees
    wavcenter : :obj:`int`, optional
        Index of wavelet center (if ``None`` using middle value of ``wav``)
    ai : :obj:`str`, optional
        Label of log to use as acoustic impedance in ``aveprops``
    method : :obj:`np.ndarray`, optional
        Method to use (``zoeppritz``, ``akirichards``, ``akirichards_alt``,
        ``fatti``, ``shuey``, and ``bortfeld``)

    Returns
    -------
    seismic : :obj:`np.ndarray`
        Zero-offset seismic data of size
        :math:`n_{theta} (\times n_y \times n_x) \times n_z`

    """
    vpgeomod = geomodel.copy(empty=True)
    vpgeomod = vpgeomod.data
    vsgeomod = vpgeomod.copy()
    rhogeomod = vpgeomod.copy()

    for iinterval, interval in enumerate(intervalnames):
        # average properties
        vpgeomod[geomodel.data == iinterval] = \
            aveprops[vp][interval]['mean']
        vsgeomod[geomodel.data == iinterval] = \
            aveprops[vs][interval]['mean']
        rhogeomod[geomodel.data == iinterval] = \
            aveprops[rho][interval]['mean']

    seismic, refl = prestack_mod(vpgeomod, vsgeomod, rhogeomod, theta, wav,
                           wavcenter=wavcenter, method=method)
    del vpgeomod, vsgeomod, rhogeomod
    return seismic

import logging
import numpy as np


def _percent_check(percs, message='Percentages'):
    """Check that percentages sum to 1, otherwise rescale them

    Parameters
    ----------
    percs : :obj:`np.ndarray`
        Percentages
    percs : :obj:`str`, optional
        First part of logging message if percentages do not sum to 1

    Returns
    -------
    percs : :obj:`np.ndarray`
        Rescaled Percentages

    """
    sums = np.sum(percs, axis=0)
    if isinstance(sums, np.ndarray):
        sums[np.isnan(sums)] = 1.
    if not np.allclose(sums,
                       np.ones(1 if percs.ndim == 1 else percs.shape[1])):
        logging.warning('{} do not sum to 1, '
                        'will be normalized'.format(message))
        percs = percs / np.sum(percs, axis=0)
    return percs


def voigt_bound(f, m):
    """Voigt bound.

    The upper bound on the effective elastic modulus, mv of a
    mixture of N material phases. This is defined as the arithmetic
    average of the constituents.

    Parameters
    ----------
    f : :obj:`list` or :obj:`np.ndarray`
        Volume fractions of N constituents
    m : :obj:`list` or :obj:`np.ndarray`
        Elastic modulus of N constituents

    Returns
    -------
    mv : :obj:`list` or :obj:`np.ndarray`
        Voigt upper bound.

    """
    if not isinstance(f, np.ndarray):
        f = np.array(f)
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    _percent_check(f, message='Volume fractions')
    mv = np.sum(f.T * m.T, axis=1 if max(f.ndim, m.ndim) > 1 else 0)
    return mv


def reuss_bound(f, m):
    """Reuss bound.

    The lower bound on the effective elastic modulus, mv of a
    mixture of N material phases. This is defined as the harmonic
    average of the constituents.

    Parameters
    ----------
    f : :obj:`list` or :obj:`np.ndarray`
        Volume fractions of N constituents
    m : :obj:`list` or :obj:`np.ndarray`
        Elastic modulus of N constituents

    Returns
    -------
    mv : :obj:`list` or :obj:`np.ndarray`
        Reuss lower bound.

    """
    if not isinstance(f, np.ndarray):
        f = np.array(f)
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    _percent_check(f, message='Volume fractions')
    mr = 1.0 / np.sum(f.T / m.T, axis=1 if max(f.ndim, m.ndim) > 1 else 0)
    return mr


def voigt_reuss_hill_average(f, m):
    """Voigt-Reuss-Hill average.

    This is defined as the simple average of the Reuss (lower)
     and Voigt (upper) bounds.

    Parameters
    ----------
    f : :obj:`list` or :obj:`np.ndarray`
        Volume fractions of N constituents
    m : :obj:`list` or :obj:`np.ndarray`
        Elastic modulus of N constituents

    Returns
    -------
    mv : :obj:`list` or :obj:`np.ndarray`
        Voigt-Reuss-Hill average

    """
    mv = voigt_bound(f, m)
    mr = reuss_bound(f, m)
    mh = (mv + mr) / 2.0

    return mh


def hashin_shtrikman(f, k, mu, modulus='bulk'):
    """Hashin-Shtrikman bounds

    Hashin-Shtrikman bounds for a mixture of two constituents.
    The best bounds for an isotropic elastic mixture, which give
    the narrowest possible range of elastic modulus without
    specifying anything about the geometries of the constituents.

    Parameters
    ----------
    f : :obj:`list` or :obj:`np.ndarray`
        Volume fractions of N constituents
    k : :obj:`list` or :obj:`np.ndarray`
        Bulk modulus of N constituents
    mu : :obj:`list` or :obj:`np.ndarray`
        Sheader modulus of N constituents
    modulus: :obj:`str`, optional
        A string specifying whether to return either the ``bulk``
        or ``shear`` HS bound.

    Returns
    -------
    lower : :obj:`list` or :obj:`np.ndarray`
        Lower bound
    upper : :obj:`list` or :obj:`np.ndarray`
        Upper bound

    """
    def _z_bulk(k, mu):
        return (4/3.) * mu

    def _z_shear(k, mu):
        return mu * (9 * k + 8 * mu) / (k + 2 * mu) / 6

    def _bound(f, k, z):
        return 1 / sum(f / (k + z)) - z

    func = {'shear': _z_shear,
            'bulk': _z_bulk}

    if not isinstance(f, np.ndarray):
        f = np.array(f)
    if not isinstance(k, np.ndarray):
        k = np.array(k)
    if not isinstance(mu, np.ndarray):
        mu = np.array(mu)
    _percent_check(f, message='Volume fractions')
    z_min = func[modulus](np.amin(k), np.amin(mu))
    z_max = func[modulus](np.amax(k), np.amax(mu))
    lower, upper = _bound(f, k, z_min), _bound(f, k, z_max)
    return lower, upper

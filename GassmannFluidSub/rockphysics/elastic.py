import numpy as np

from .signals import moving_average


def _compute_vp(k, mu, rho):
    """P-wave velocity given parameter given bulk modulus, shear modulus and
    density


    Parameters
    ----------
    k : :obj:`float` or :obj:`np.ndarray`
        Bulk modulus in Pa
    mu : :obj:`float` or :obj:`np.ndarray`
        Shear modulus in Pa
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    Returns
    -------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s

    """
    vp = np.sqrt((k + 4./3.*mu)/rho)
    return vp


def _compute_vs(mu, rho):
    """S-wave velocity given parameter given shear modulus and
    density


    Parameters
    ----------
    mu : :obj:`float` or :obj:`np.ndarray`
        Shear modulus in Pa
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    Returns
    -------
    vs : :obj:`float` or :obj:`np.ndarray`
        S-wave velocity in m/s

    """
    vs = np.sqrt(mu/rho)
    return vs


def _compute_bulk(vp, vs, rho):
    """Bulk modulus given S-wave velocity, and density

    Parameters
    ----------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s
    vs : :obj:`float` or :obj:`np.ndarray`
        S-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    Returns
    -------
    bulk : :obj:`float` or :obj:`np.ndarray`
        Bulk modulus in Pa

    """
    bulk = rho * (vp ** 2 - (4. / 3.) * (vs ** 2))
    return bulk


def _compute_mu(vs, rho):
    """Shear modulus given S-wave velocity, and density

    Parameters
    ----------
    vs : :obj:`float` or :obj:`np.ndarray`
        S-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    Returns
    -------
    mu : :obj:`float` or :obj:`np.ndarray`
        Shear modulus in Pa

    """
    mu = rho * vs**2
    return mu


def _compute_lambda(vp, vs, rho):
    """Lambda parameter given either Vp, Vs, and density

    Parameters
    ----------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s
    vs : :obj:`float` or :obj:`np.ndarray`
        S-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    Returns
    -------
    lam : :obj:`float` or :obj:`np.ndarray`
        Lambda parameter in Pa

    """
    lam = rho * (vp ** 2 - 2. * vs ** 2.)
    return lam


def _backus_parameters(vp, vs, rho, dz, lb):
    """Parameters for Backus averaging.

    Parameters
    ----------
    vp : :obj:`np.ndarray`
        P-wave velocity in m/s
    vs : :obj:`np.ndarray`
        S-wave velocity in m/s
    rho : :obj:`np.ndarray`
        Density in kg/m3
    dz : :obj:`float`
        Sampling of elastic properties in m
    lb : :obj:`float`
        Backus averaging length in m

    Returns
    -------
    A, C, F, L, M : :obj:`float` or :obj:`np.ndarray`
        Intermaediate arameters for backus averaging

    """
    lam = _compute_lambda(vp, vs, rho)
    mu = _compute_mu(vs, rho)

    # Compute the layer parameters from Liner (2014) equation 2:
    a = rho * np.power(vp, 2.0)  # Acoustic impedance

    # Compute the Backus parameters from Liner (2014) equation 4:
    lenght = lb // dz
    A1 = 4 * moving_average(mu*(lam+mu)/a, lenght)
    A = A1 + np.power(moving_average(lam/a, lenght), 2.0)\
        / moving_average(1.0/a, lenght)
    C = 1.0 / moving_average(1.0/a, lenght)
    F = moving_average(lam/a, lenght)\
        / moving_average(1.0/a, lenght)
    L = 1.0 / moving_average(1.0/mu, lenght)
    M = moving_average(mu, lenght)
    return A, C, F, L, M


def backus(vp, vs, rho, dz, lb):
    """Backus averaging

    Apply Backus averaging as in [1]_ to elastic properties to upscal from fine
    ``dz`` to coarser ``lb`` vertical sampling

    .. [1] Liner, C (2014), Long-wave elastic attenuation produced by horizontal
       layering. The Leading Edge, June 2014, p 634-638.

     Parameters
    ----------
    vp : :obj:`np.ndarray`
        P-wave velocity in m/s
    vs : :obj:`np.ndarray`
        S-wave velocity in m/s
    rho : :obj:`np.ndarray`
        Density in kg/m3
    dz : :obj:`float`
        Sampling of elastic properties in m
    lb : :obj:`float`
        Backus averaging length in m

    Returns
    -------
    vpbackus : :obj:`np.ndarray`
        Backus averaged P-wave velocity in m/s
    vsbackus : :obj:`np.ndarray`
        Backus averaged S-wave velocity in m/s
    rhobackus : :obj:`np.ndarray`
        Backus averaged density in kg/m3

    """
    # Compute the Backus parameters:
    A, C, F, L, M = _backus_parameters(vp, vs, rho, dz, lb)

    # Compute the vertical velocities from Liner (2014) equation 5:
    rhobackus = moving_average(rho, lb//dz)
    vpbackus = np.sqrt(C / rhobackus)
    vsbackus = np.sqrt(L / rhobackus)

    return vpbackus, vsbackus, rhobackus


class Elastic:
    """Elastic object.

    An Elastic object is a container for an elastic medium. It contains an
    exahustive set of properties for the medium.

    The Elastic object is initialized by providing the triplet P-wave velocity,
    S-wave velocity, and density.

    Parameters
    ----------
    vp : :obj:`float` or :obj:`numpy.ndarray`
        P-wave velocity in m/s
    vs : :obj:`float` or :obj:`numpy.ndarray`
        S-wave velocity in m/s
    rho : :obj:`float` or :obj:`numpy.ndarray`
        Density in kg/m3

    Attributes
    ----------
    k : :obj:`float` or :obj:`numpy.ndarray`
        Bulk modulus in Pa
    mu : :obj:`float` or :obj:`np.ndarray`
        Shear modulus in Pa
    lam : :obj:`float` or :obj:`np.ndarray`
        Lambda parameter in Pa

    """
    def __init__(self, vp, vs, rho):
        self.vp = vp
        self.vs = vs
        self.rho = rho

    @property
    def ai(self):
        self._ai = self.vp*self.rho
        return self._ai

    @property
    def vpvs(self):
        self._vpvs = self.vp / self.vs
        return self._vpvs


    @property
    def k(self):
        self._k = _compute_bulk(self.vp, self.vs, self.rho)
        return self._k

    @property
    def mu(self):
        self._mu = _compute_mu(self.vs, self.rho)
        return self._mu

    @property
    def lam(self):
        self._lam = _compute_lambda(self.vp, self.vs, self.rho)
        return self._lam

    def __str__(self):
        descr = 'Elastic medium:\n' + \
                '---------------\n' + \
                'VP: {}\n'.format(self.vp) + \
                'VS: {}\n'.format(self.vs) + \
                'Rho: {}\n'.format(self.rho)
        return descr



class ElasticLimited:
    def __init__(self, k, rho):
        self.k = k
        self.rho = rho
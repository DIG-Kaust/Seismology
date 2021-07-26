import numpy as np

from .elastic import Elastic
from .bounds import _percent_check, voigt_bound, \
    voigt_reuss_hill_average


class Matrix():
    """Matrix object.

    A Matrix object is a container for properties of a mineral matrix.
    It takes the mineral composition of multiple rock types in
    :obj:`elastic.Elastic` objects and their fractions
    in the matrix and combines them.

    Parameters
    ----------
    minerals : :obj:`dict`
        Set of dictionaries of bulk modulus and density of minerals and
        their fractions

    """
    def __init__(self, minerals):
        self.names = minerals.keys()
        self.nmins = len(self.names)
        fracs = list(minerals.values())
        self.ks = np.array([value['k'] for value in fracs])
        self.rhos = np.array([value['rho'] for value in fracs])
        self.fracs = np.array([value['frac'] for value in fracs])
        self.fracs = _percent_check(self.fracs, message='Mineral fractions')

    def __str__(self):
        descr = 'Matrix\n' \
                '------\n'

        for name, frac in zip(self.names, self.fracs):
            descr += '{} : {}\n'.format(name, frac)

        return descr

    @property
    def k(self):
        self._k = voigt_reuss_hill_average(self.fracs, self.ks)
        return self._k

    @property
    def rho(self):
        self._rho = voigt_bound(self.fracs, self.rhos)
        return self._rho


class Rock(Elastic):
    """Rock object.

    A Rock object is a container for an elastic properties of a rock.

    Given knowledge of the rock elastic properties, matrix, fluids and porosity,
    it can return information about the Dry component.

    Parameters
    ----------
    vp : :obj:`float` or :obj:`numpy.ndarray`
        P-wave velocity in m/s
    vs : :obj:`float` or :obj:`numpy.ndarray`
        S-wave velocity in m/s
    rho : :obj:`float` or :obj:`numpy.ndarray`
        Density in kg/m3
    matrix : :obj:`elastic.Matrix`
        Matrix
    fluid : :obj:`fluid.Fluid`
        Fluid definition
    poro : :obj:`float` or :obj:`numpy.ndarray`
        Porosity

    Attributes
    ----------
    k : :obj:`float` or :obj:`numpy.ndarray`, optional
        Bulk modulus in Pa
    mu : :obj:`float` or :obj:`np.ndarray`
        Shear modulus in Pa
    lam : :obj:`float` or :obj:`np.ndarray`
        Lambda parameter in Pa
    k_dry : :obj:`float` or :obj:`np.ndarray`
        Lambda parameter in Pa
    mu_dry : :obj:`float` or :obj:`np.ndarray`
        Lambda parameter in Pa

    """

    def __init__(self, vp, vs, rho, matrix, fluid, poro):
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.matrix = matrix
        self.fluid = fluid
        self.poro = poro

    @property
    def kdry(self):
        self._kdry = (self.k * (self.poro * self.matrix.k / self.fluid.k + 1 - self.poro) - self.matrix.k) / \
            (self.poro * self.matrix.k / self.fluid.k + self.k / self.matrix.k - 1 - self.poro)
        return self._kdry

    @property
    def mudry(self):
        self._mudry = self.mu
        return self._mudry
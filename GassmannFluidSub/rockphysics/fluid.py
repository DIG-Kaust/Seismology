import numpy as np

from .units import kg_m3_to_g_cm3, g_cm3_to_kg_m3
from .elastic import _compute_vp, Elastic
from .bounds import _percent_check, voigt_bound, \
    reuss_bound


_W_batze_wang = np.array([[1402.85, 1.524, 3.437e-3, -1.197e-5],
                          [4.871, -0.0111, 1.739e-4, -1.628e-6],
                          [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
                          [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
                          [-2.197e-7, 7.987e-10, 5.23e-11, -4.614e-13]])

_r = 8.314 # gas constant


def _batze_wang_brine(temp, pres, sal):
    """Batze and Wang brine density and velocity

    Parameters
    ----------
    temp : :obj:`float` or :obj:`np.ndarray`
        Temperature in degrees C
    pres : :obj:`float` or :obj:`np.ndarray`
        In-situ pressure in MPa
    sal : :obj:`float` or :obj:`np.ndarray`
        Salinity in percentage

    Returns
    -------
    vp_brine : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity of brine in m/s
    rho_brine : :obj:`float` or :obj:`np.ndarray`
        Density of brine in kg/m^3

    """
    rho_w = 1 + 1e-6 * (-80*temp - 3.3*temp**2 + 0.00175*temp**3 +
                        489*pres - 2*temp*pres + 0.016*temp**2*pres -
                        1.3e-5*temp**3*pres - 0.333*pres**2 - 0.002*temp*pres**2)
    rho_brine = rho_w + 0.668*sal + 0.44*sal**2 + 1e-6 * sal * \
                (300*pres - 2400*pres*sal + temp*(80 + 3*temp - 3300*sal -
                                                  13*pres + 47*pres*sal))
    rho_brine = g_cm3_to_kg_m3(rho_brine)

    temps = np.vstack((temp ** 0, temp ** 1, temp ** 2, temp ** 3, temp ** 4)).T
    press = np.vstack((pres ** 0, pres ** 1, pres ** 2, pres ** 3))

    vp_w = np.dot(temps, np.dot(_W_batze_wang, press)).squeeze()
    if vp_w.ndim == 2:
        vp_w = np.diag(vp_w)
    vp_brine = vp_w + sal*(1170 - 9.6*temp + 0.055*temp**2 - 8.5e-5*temp**3 +
                           2.6*pres - 0.0029*temp*pres - 0.0476*pres**2) + \
               sal**1.5*(780 - 10*pres + 0.16*pres**2) - 1820*sal**2
    return vp_brine, rho_brine

def _batze_wang_oil(temp, pres, grav, gor, rho0):
    """Batze and Wang oil density and velocity

    Parameters
    ----------
    temp : :obj:`float` or :obj:`np.ndarray`
        Temperature in degrees C
    pres : :obj:`float` or :obj:`np.ndarray`
        In-situ pressure in MPa
    grav : :obj:`float` or :obj:`np.ndarray`
        Specific gravity in API (ratio of gas density to air density
        at 15.6 degrees)
    gor : :obj:`float` or :obj:`np.ndarray`
        Gas-oil ratio
    rho0 : :obj:`float` or :obj:`np.ndarray`
        Reference density of oul at 15.8 degrees C and atmosferic pressure in kg/m^3

    Returns
    -------
    vp_oil : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity of brine in m/s
    rho_oil : :obj:`float` or :obj:`np.ndarray`
        Density of brine in kg/m^3

    """
    rho0 = kg_m3_to_g_cm3(rho0)
    b0 = 0.972 + 0.00038 * (2.495 * gor * np.sqrt(grav / rho0) + temp + 17.8)**1.175
    rho_s = (rho0 + 0.0012 * gor * grav) / b0
    rho_ps = rho0 / ((1. + 0.001*gor) * b0)

    rho_oil = (rho_s + (0.00277*pres - 1.71e-7*pres**3)*(rho_s - 1.15)**2 + 3.49e-4 * pres) / \
              (0.972 + 3.81e-4*(temp + 17.78)**1.175)
    rho_oil = g_cm3_to_kg_m3(rho_oil)
    vp_oil = 2096. * np.sqrt(rho_ps/(2.6 - rho_ps)) - 3.7*temp + 4.64*pres + \
             0.0115*(np.sqrt(18.33/rho_ps - 16.97) - 1)*temp*pres
    return vp_oil, rho_oil

def _batze_wang_gas(temp, pres, grav):
        """Batze and Wang gas density and velocity

        Parameters
        ----------
        temp : :obj:`float` or :obj:`np.ndarray`
            Temperature in degrees C
        pres : :obj:`float` or :obj:`np.ndarray`
            In-situ pressure in MPa
        grav : :obj:`float` or :obj:`np.ndarray`
            Specific gravity in API (ratio of gas density to air density
            at 15.6 degrees)
        Returns
        -------
        rho_gas : :obj:`float` or :obj:`np.ndarray`
            Density of gas in kg/m^3

        """
        tpr = (temp + 273.15) / (94.72 + 170.75 * grav)
        ppr = pres / (4.892 - 0.4048 * grav)

        e = 0.109 * (3.85 - tpr) ** 2 * \
            np.exp(-(0.45 + 8. * (0.56 - 1. / tpr)**2) * ppr**1.2 / tpr)
        z = (0.03 + 0.00527 * (3.5 - tpr) ** 3) * ppr + \
                 (0.642 * tpr - 0.007 * tpr ** 4 - 0.52) + e
        rho_gas = 28.8 * grav * pres / (z * _r * (temp + 273.15))
        rho_gas = g_cm3_to_kg_m3(rho_gas)
        return rho_gas, tpr, ppr, z


class Brine(Elastic):
    """Brine object.

    An Brine object is a container for brine properties.

    Parameters
    ----------
    temp : :obj:`float` or :obj:`np.ndarray`
        Temperature in degrees C
    pres : :obj:`float` or :obj:`np.ndarray`
        In-situ pressure in MPa
    sal : :obj:`float` or :obj:`np.ndarray`
        Salinity in ppm

    Attributes
    ----------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3
    k : :obj:`float` or :obj:`np.ndarray`
        Bulk modulus in Pa

    """
    def __init__(self, temp, pres, sal):
        self.temp = temp
        self.pres = pres
        self.sal = sal / 1e6
        self.vp, self.rho = _batze_wang_brine(self.temp, self.pres, self.sal)
        self.vs = 0 if not isinstance(self.vp, np.ndarray) \
            else np.zeros_like(self.vp)
    @property
    def k(self):
        self._k = kg_m3_to_g_cm3(self.rho) * (self.vp**2) * 1e3
        return self._k


class Oil(Elastic):
    """Oil object.

    An Oil object is a container for oil properties.

     Parameters
    ----------
    temp : :obj:`float` or :obj:`np.ndarray`
        Temperature in degrees C
    pres : :obj:`float` or :obj:`np.ndarray`
        In-situ pressure in MPa
    oilgrav : :obj:`float` or :obj:`np.ndarray`
        Specific gravity of oil in API (ratio of oil density to air density
        at 15.6 degrees)
    grav : :obj:`float` or :obj:`np.ndarray`
        Specific gravity of gas in API (ratio of gas density to air density
        at 15.6 degrees)
    gor : :obj:`float` or :obj:`np.ndarray`
        Gas-oil ratio

    Attributes
    ----------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3
    k : :obj:`float` or :obj:`np.ndarray`
        Bulk modulus in Pa

    """
    def __init__(self, temp, pres, oilgrav, gasgrav, gor):
        self.temp = temp
        self.pres = pres
        self.gasgrav = gasgrav
        self.gor = gor
        self.rho0 = g_cm3_to_kg_m3(141.5 / (oilgrav + 131.5))
        self.vp, self.rho = _batze_wang_oil(temp, pres, gasgrav, gor, self.rho0)
        self.vs = 0 if not isinstance(self.vp, np.ndarray) \
            else np.zeros_like(self.vp)

    @property
    def k(self):
        self._k = kg_m3_to_g_cm3(self.rho) * (self.vp ** 2) * 1e3
        return self._k


class Gas(Elastic):
    """Gas object.

    An Gas object is a container for gas properties.

    Parameters
    ----------
    temp : :obj:`float` or :obj:`np.ndarray`
        Temperature in degrees C
    pres : :obj:`float` or :obj:`np.ndarray`
        In-situ pressure in MPa
    grav : :obj:`float` or :obj:`np.ndarray`
        Specific gravity in API (ratio of gas density to air density
        at 15.6 degrees)

    Attributes
    ----------
    vp : :obj:`float` or :obj:`np.ndarray`
        P-wave velocity in m/s
    rho : :obj:`float` or :obj:`np.ndarray`
        Density in kg/m3

    """
    def __init__(self, temp, pres, grav):
        self.temp = temp
        self.pres = pres
        self.grav = grav
        self.rho, self.tpr, self.ppr, self.z = \
            _batze_wang_gas(self.temp, self.pres, self.grav)
        self.vs = 0 if not isinstance(self.rho, np.ndarray) \
            else np.zeros_like(self.rho)

    @property
    def k(self):
        gamma0 = 0.85 + 5.6 / (self.ppr + 2) + 27.1 / (self.ppr + 3.5) ** 2 - \
                 8.7 * np.exp(-0.65 * (self.ppr + 1))
        exponent = (0.45 + 8 * (0.56 - 1. / self.tpr) ** 2)

        f = -1.2 * self.ppr ** 0.2 / self.tpr * exponent * \
            np.exp(-exponent * self.ppr ** 1.2 / self.tpr)
        dz_dppr = 0.03 + 0.00527 * (3.5 - self.tpr) ** 3 + \
                  0.109 * (3.85 - self.tpr) ** 2 * f
        self._k = self.pres * gamma0 / ((1 - self.ppr / self.z * dz_dppr) * 1000)
        self._k *= 1e9  # convert from GPa to Pa
        return self._k

    @property
    def vp(self):
        self._vp = _compute_vp(self.k, self.mu, self.rho)
        return self._vp


class Fluid(Elastic):
    """Fluid object.

    A Fluid object is a container for properties of a fluids to be mixed.
    It takes the fluid composition of multiple fluids in
    :obj:`Oil` or :obj:`Brine` or
    :obj:`Gas` objects and their fractions
    and combines them.

    Parameters
    ----------
    fluids : :obj:`dict`
        Set of tuples of Fluids in :obj:`Oil` or :obj:`Brine` or
        :obj:`Gas` objects objects and their fractions

    """
    def __init__(self, fluids):
        self.names = fluids.keys()
        self.nmins = len(self.names)
        fracs = fluids.values()
        self.comps = [value[0] for value in fracs]
        self.fracs = np.array([value[1] for value in fracs])
        self.fracs = _percent_check(self.fracs, message='Fluid fractions')

    def __str__(self):
        descr = 'Fluid\n' \
                '------\n'
        for name, frac in zip(self.names, self.fracs):
            descr += '{} : {}\n'.format(name, frac)

        return descr

    @property
    def k(self):
        self._k = reuss_bound(self.fracs,
                              [comp.k for comp in self.comps])
        return self._k

    @property
    def rho(self):
        self._rho = voigt_bound(self.fracs, [comp.rho for comp in self.comps])
        return self._rho

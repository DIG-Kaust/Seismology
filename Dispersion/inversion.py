import numpy as np
from disba import PhaseDispersion
import matplotlib.pyplot as plt



def fun(x, nlayers, t, vdispobs):
    r"""Surface wave inversion misfit function

    Create a dispersion curve from the input parameters 
    (thicknesses and shear wave velocities) and compute
    misfit

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Set of thicknesses and shear wave velocities
    t : :obj:`numpy.ndarray`
        Period
    vdispobs : :obj:`numpy.ndarray`
        Observed dispersion curve
        
    Returns
    -------
    loss : :obj:`float`
        Loss function

    """
    # Create model
    thick = x[:nlayers]
    vs = x[nlayers:]
    vp = vs * 4
    rho = 1. * np.ones(nlayers)
    model = np.vstack([thick, vp, vs, rho]).T
    
    # Compute the Rayleigh-wave modal dispersion curves
    pd = PhaseDispersion(*model.T)
    cpr = pd(t, mode=0, wave="rayleigh") 
    vdisp = cpr[1]

    return np.linalg.norm(vdisp-vdispobs)
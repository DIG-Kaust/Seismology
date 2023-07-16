import numpy as np
from disba import PhaseDispersion
import matplotlib.pyplot as plt



def fun(x, nlayers, t, vdispobs):
    thick = x[:nlayers]
    vs = x[nlayers:]
    vp = vs * 4
    rho = 1. * np.ones(nlayers)
    
    model = np.vstack([thick, vp, vs, rho]).T
    #print(model)
    
    # Compute the Rayleigh-wave modal dispersion curves
    pd = PhaseDispersion(*model.T)
    cpr = pd(t, mode=0, wave="rayleigh") 
    vdisp = cpr[1]
    #print(vdisp, vdispobs)
    
    #plt.figure()
    #plt.plot(t, vdisp, 'r')
    #plt.plot(t, vdispobs, 'k')

    return np.linalg.norm(vdisp-vdispobs)
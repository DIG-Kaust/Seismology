import numpy as np

from .elastic import _compute_vp, _compute_vs
from .solid import Rock


def _mask_unchanged(fluid0, fluid1):
    """Create mask for values where saturation is unchanged
    """
    fluid0_fracs = fluid0.fracs
    fluid1_fracs = fluid1.fracs
    changes = np.sum(np.abs(fluid0_fracs - fluid1_fracs), axis=0)
    mask = changes > 0
    return mask


class Gassmann:
    """Gassmann fluid substitution

    Parameters
    ----------
    medium0 : :obj:`solid.Rock`
        Initial saturated medium
    fluid1 : :obj:`fluid.Fluid`
        Final fluid
    mask : :obj:`bool`, optional
        Apply mask where no fluid change is provided from state0 to state1

    """
    def __init__(self, medium0, fluid1, mask=False):
        self.medium0 = medium0
        self.fluid1 = fluid1
        self.mask = mask
        self._compute()

    def _compute(self, ):
        """Compute fluid substitution
        """
        kdry = self.medium0.kdry
        kmatrix = self.medium0.matrix.k
        ksat1 = kdry + (1 - kdry / kmatrix) ** 2 / (
                self.medium0.poro / self.fluid1.k + (1 - self.medium0.poro) /
                kmatrix - kdry / kmatrix ** 2)

        musat1 = self.medium0.mu
        rhosat1 = self.medium0.matrix.rho + self.medium0.poro * (self.fluid1.rho - self.medium0.fluid.rho)
        vpsat1 = _compute_vp(ksat1, musat1, rhosat1)
        vssat1 = _compute_vs(musat1, rhosat1)

        if self.mask:
            mask = _mask_unchanged(self.medium0.fluid, self.fluid1)
            vpsat1 = mask*vpsat1 + (1 - mask)*self.medium0.vp
            vssat1 = mask*vssat1 + (1 - mask)*self.medium0.vs
            rhosat1 = mask*rhosat1 + (1 - mask)*self.medium0.rho

        self.medium1 = Rock(vpsat1, vssat1, rhosat1,
                            self.medium0.matrix, self.fluid1,
                            self.medium0.poro)

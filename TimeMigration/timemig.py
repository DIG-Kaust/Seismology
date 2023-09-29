import os
import numpy as np

from typing import Optional, Tuple, Union

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, NDArray
from pylops.signalprocessing import Convolve1D

jit_message = deps.numba_import("the kirchhoff module")

if jit_message is None:
    from numba import jit, prange

    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
else:
    prange = range


class TimeKirchhoff(LinearOperator):
    """Post-stack, time-domain Kirchhoff migration
    
    Parameters
    ----------
    t0 : np.ndarray
        Time axis 
    x : np.ndarray
        Spatial axis
    vrms : np.ndarray
        Root-mean-square velocity of events
    wav : np.ndarray
        Wavelet
    wavcenter : int
        Center of wavelet
    engine : str, optional
        Engine used for computations (``numpy`` or ``numba``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    """
    def __init__(
        self,
        t0: NDArray,
        x: NDArray,
        vrms: NDArray,
        wav: NDArray,
        wavcenter: int,
        engine: str = "numpy",
        dtype: DTypeLike = "float64",
        name: str = "K",
    ) -> None:

        self.t0, self.x = t0, x
        self.dt0 = self.t0[1] - self.t0[0]
        self.nt0, self.nx = t0.size, x.size
        self.vrms = vrms
        self.trav = self._traveltime_table(self.t0, self.x, self.vrms)
        self.cop = Convolve1D((self.nx, self.nt0), h=wav, offset=wavcenter, axis=1)
        
        super().__init__(dtype=np.dtype(dtype), dims=(self.nx, self.nt0), dimsd=(self.nx, self.nt0), name=name)
        self._register_multiplications(engine)
        
    @staticmethod
    def _traveltime_table(
        t0: NDArray,
        x: NDArray,
        vrms: NDArray,
    ) -> NDArray:
        r"""Traveltime table
        """
        nx, nt0 = x.size, t0.size
        trav = np.zeros((nx, nx, nt0))

        for ixsurf in range(nx): # loop over sources/receivers
            trav[ixsurf] = np.sqrt(t0**2 + 4*((x-x[ixsurf])[:, np.newaxis]/vrms)**2)
        return trav
    
    def _register_multiplications(self, engine: str) -> None:
        if engine not in ["numpy", "numba"]:
            raise KeyError("engine must be numpy or numba")
        if engine == "numba" and jit_message is None:
            # numba
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,
            self._kirch_matvec = jit(**numba_opts)(self._trav_matvec)
            self._kirch_rmatvec = jit(**numba_opts)(self._trav_rmatvec)
        else:
            if engine == "numba" and jit_message is not None:
                logging.warning(jit_message)
            self._kirch_matvec = self._trav_matvec
            self._kirch_rmatvec = self._trav_rmatvec
            
    @staticmethod
    def _trav_matvec(x, y, nx, nt0, dt0, trav):
        for ixsurf in prange(nx): # loop over receivers
            for ix in range(nx): # loop over x coordinate of image point
                for it0 in range(nt0): # loop over t0 coordinate of image point
                    tt = trav[ixsurf, ix, it0] / dt0
                    itt = int(tt)
                    dtt = tt - itt                    
                    if 0 <= itt < nt0 - 1:
                        y[ixsurf, itt] += x[ix, it0] * (1 - dtt)
                        y[ixsurf, itt + 1] += x[ix, it0] * dtt
        return y
    
    @staticmethod
    def _trav_rmatvec(x, y, nx, nt0, dt0, trav):
        for ix in prange(nx): # loop over x coordinate of image point
            for ixsurf in range(nx): # loop over receivers
                for it0 in range(nt0): # loop over t0 coordinate of image point
                    tt = trav[ixsurf, ix, it0] / dt0
                    itt = int(tt)
                    dtt = tt - itt                    
                    if 0 <= itt < nt0 - 1:
                        y[ix, it0] += x[ixsurf, itt] * (1 - dtt) +  x[ixsurf, itt + 1] * dtt
        return y
    
    @reshaped
    def _matvec(self, x):
        y = np.zeros((self.nx, self.nt0), dtype=self.dtype)
        y = self._kirch_matvec(x, y, self.nx, self.nt0, self.dt0, self.trav)
        y = self.cop @ y
        return y
    
    @reshaped
    def _rmatvec(self, x):
        x = self.cop.H @ x
        y = np.zeros((self.nx, self.nt0), dtype=self.dtype)
        y = self._kirch_rmatvec(x, y, self.nx, self.nt0, self.dt0, self.trav)
        return y
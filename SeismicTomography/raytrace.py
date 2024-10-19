import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from scipy.sparse import csc_matrix


def _raytrace_vertical(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Vertical ray

    Compute vertical ray and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (lenght of rays in grid cells)

    """
    # Define x-index and extreme z-indeces of ray
    xray, zray = s[0], s[1]
    ix = int((xray + dx // 2 - x[0]) // dx)
    izin = int((zray + dz // 2 - z[0]) // dz)
    izend = int((r[1] + dz // 2 - z[0]) // dz)

    # Create tomographic matrix
    R = np.zeros(nx * nz)
    R[ix * nz + izin + 1: ix * nz + izend] = dz # middle cells
    R[ix * nz + izin] = z[izin] + dz / 2 - s[1] # source side cell
    R[ix * nz + izend] = r[1] - z[izend] + dz / 2 # receiver side cell
    c = np.arange(ix * nz + izin, ix * nz + izend + 1)
    v = np.hstack((z[izin] + dz / 2 - s[1],
                   dz * np.ones(izend - izin - 1),
                   r[1] - z[izend] + dz / 2))
    return R, c, v


def _raytrace_generic(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Generic ray

    Compute generic ray and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (lenght of rays in grid cells)

    """
    # Define indices of source and parametric straigh ray
    xray, zray = s[0], s[1]
    m = (r[1] - s[1]) / (r[0] - s[0])
    q = s[1] - m * s[0]

    # Create tomographic matrix
    R = np.zeros(nx * nz)
    c, v = np.zeros(3 * (nx + nz)), np.zeros(3 * (nx + nz))
    ic = 0
    while xray < r[0]:
        # find grid points of source
        ix, iz = int((xray + dx / 2 - x[0]) / dx), int(
            (zray + dz / 2 - z[0]) / dz)
        # computing z intersecting x-edge of the grid point
        xedge = x[ix] + dx / 2
        zedge = xedge * m + q
        if xedge > r[0]:
            xedge = r[0]
            zedge = r[1]
        izedge = int((zedge + dz / 2 - z[0]) / dz)
        if izedge == iz:
            # find lenght of ray from x to x-edge
            rray = sqrt((xedge - xray) ** 2 + (zedge - zray) ** 2)
            R[ix * nz + iz] = rray
            c[ic], v[ic] = ix * nz + iz, rray
            ic += 1
        elif izedge > iz:
            # next zedge above current z cell: split ray from x to x-edge into
            # multiple rays passing through different z-cells
            nzcells = izedge - iz + 1
            zend = (x[ix + 1] - dx / 2) * m + q
            if zend > r[1]:
                zend = r[1]
            zs = z[iz] + np.arange(nzcells - 1) * dz + dz // 2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            rrays = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)
            R[ix * nz + iz: ix * nz + iz + nzcells] = rrays
            c[ic:ic + nzcells], v[ic:ic + nzcells] = np.arange(ix * nz + iz,
                                                               ix * nz + iz + nzcells), rrays
            ic += nzcells
        else:
            # next zedge below current z cell: split ray from x to x-edge into
            # multiple rays passing through different z-cells
            nzcells = iz - izedge + 1
            zend = (x[ix + 1] - dx / 2) * m + q
            if zend < r[1]:
                zend = r[1]
            zs = z[iz] - np.arange(nzcells - 1) * dz - dz / 2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            rrays = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)
            R[ix * nz + iz - nzcells + 1: ix * nz + iz + 1] = rrays[::-1]
            c[ic:ic + nzcells], v[ic:ic + nzcells] = np.arange(
                ix * nz + iz - nzcells + 1, ix * nz + iz + 1), rrays[::-1]
            ic += nzcells
        xray, zray = xedge, zedge
    return R, c[:ic].astype(np.int64), v[:ic]


def raytrace(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Raytrace

    Compute ray and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    oz : :obj:`float`
        Vertical axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    x : :obj:`np.ndarray`
        Horizontal axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`np.ndarray`
        Tomographic dense matrix
    c : :obj:`np.ndarray`
        Column indices (index of grid cells)
    v : :obj:`np.ndarray`
        Value (lenght of rays in grid cells)

    """
    # vertical ray
    if s[0] == r[0]:
        if r[1] > s[1]:
            # source below receiver
            return _raytrace_vertical(s, r, dx, dz, ox, oz, nx, nz, x, z)
        else:
            # source above receiver - swap them
            return _raytrace_vertical(r, s, dx, dz, ox, oz, nx, nz, x, z)

    # generic ray
    if r[0] > s[0]:
        # source on the left or receiver
        return _raytrace_generic(s, r, dx, dz, ox, oz, nx, nz, x, z)
    else:
        # source on the right or receiver - swap them
        return _raytrace_generic(r, s, dx, dz, ox, oz, nx, nz, x, z)


def straigh_ray(s, r, nray=10):
    """Analytical straight ray

    Compute analytical expression for straight ray given source and
    receiver pair

    Parameters
    ----------
    s : :obj:`list`
        Source location
    r : :obj:`list`
        Receiver location
    nray : :obj:`int`, optional
        Number of points per ray

    Returns
    -------
    ray : :obj:`tuple`
        Ray coordinates

    """
    # Find analytical expression of straigt line
    m = (r[1] - s[1]) / (r[0] - s[0])
    q = s[1] - m * s[0]
    rayx = np.linspace(s[0], r[0], nray)
    rayz = rayx * m + q
    ray = (rayx, rayz)
    return ray


def straigh_rays(s, r, nray=10):
    """Analytical straight rays

    Compute analytical expression for straight rays given set of sources and
    receivers

    Parameters
    ----------
    s : :obj:`np.ndarray`
        Source locations (array of size 2 x ns)
    r : :obj:`list`
        Receiver locations (array of size 2 x nr)
    nray : :obj:`int`, optional
        Number of points per ray

    Returns
    -------
    rays : :obj:`list`
        Ray coordinates

    """
    ns = s.shape[1]
    nr = r.shape[1]

    # Loop over sources
    rays = []
    for isrc in range(ns):
        # Loop over receivers
        for irec in range(nr):
            rays.append(straigh_ray(s[:, isrc], r[:, irec], nray=nray))
    return rays


def tomographic_matrix(s, r, dx, dz, ox, oz, nx, nz, x, z):
    """Tomographich matrix

    Compute set of rays and associated tomographic matrix

    Parameters
    ----------
    s : :obj:`np.ndarray`
        Source locations (array of size 2 x ns)
    r : :obj:`list`
        Receiver locations (array of size 2 x nr)
    dx : :obj:`float`
        Horizontal axis spacing
    dz : :obj:`float`
        Vertical axis spacing
    ox : :obj:`float`
        Horizontal axis origin
    oz : :obj:`float`
        Vertical axis origin
    nx : :obj:`int`
        Number of samples in horizontal axis
    nz : :obj:`float`
        Number of samples in vertical axis
    x : :obj:`np.ndarray`
        Horizontal axis
    z : :obj:`np.ndarray`
        Vertical axis

    Returns
    -------
    R : :obj:`scipy.sparse.csc_matrix`
        Tomographic sparse matrix

    """
    ns = s.shape[1]
    nr = r.shape[1]
    rows = []
    cols = []
    v = []

    # Loop over sources
    iray = 0
    for isrc in range(ns):
        # Loop over receivers
        for irec in range(nr):
            _, ctmp, vtmp = raytrace(s[:, isrc], r[:, irec],
                                     dx, dz, ox, oz, nx, nz, x, z)
            rows.append(iray * np.ones_like(ctmp))
            cols.append(ctmp)
            v.append(vtmp)
            iray += 1
    R = csc_matrix((np.hstack(v),
                    (np.hstack(rows), np.hstack(cols))),
                    shape=(ns * nr, nx * nz))
    return R

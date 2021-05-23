import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def raytrace(s, r, dx, dz, ox, oz, nx, nz, x, z):
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


def _raytrace_vertical(s, r, dx, dz, ox, oz, nx, nz, x, z):
    xray, zray = s[0], s[1]
    ix, izin = int((xray + dx//2 - x[0])//dx), int((zray + dz//2 - z[0])//dz)
    izend = int((r[1] + dz//2 - z[0])//dz)
    R = np.zeros(nx*nz)
    R[ix * nz + izin + 1: ix * nz + izend] = dz
    R[ix * nz + izin] = z[izin] + dz/2 - s[1]
    R[ix * nz + izend] = r[1] - z[izend] + dz/2
    c = np.arange(ix * nz + izin, ix * nz + izend + 1)
    v = np.hstack((z[izin] + dz/2 - s[1], dz * np.ones(izend - izin - 1), r[1] - z[izend] + dz/2))
    return R, c, v

    
def _raytrace_generic(s, r, dx, dz, ox, oz, nx, nz, x, z):
    xray, zray = s[0], s[1]
    m = (r[1]-s[1]) / (r[0]-s[0])
    q = s[1] - m*s[0]
    R = np.zeros(nx*nz)
    c, v = np.zeros(3*(nx + nz)), np.zeros(3*(nx + nz))
    ic = 0
    while xray < r[0]:
        # find grid points of source
        ix, iz = int((xray + dx/2 - x[0])/dx), int((zray + dz/2 - z[0])/dz)
        # computing z intersecting x-edge of the grid point
        #print('xray, x[ix]', xray, x[ix])
        xedge = x[ix] + dx / 2
        zedge = xedge * m + q
        if xedge > r[0]:
            xedge = r[0]
            zedge = r[1]
        #print('xedge, zedge', xedge, zedge)
        izedge = int((zedge + dz/2 - z[0])/dz)
        if izedge == iz:
            # find lenght of ray from x to x-edge
            rray = sqrt((xedge-xray)**2 + (zedge-zray)**2)
            R[ix * nz + iz] = rray
            #print(xedge, zedge)
            c[ic], v[ic] = ix * nz + iz, rray
            ic += 1
            #print('rray', rray)
        elif izedge > iz:
            # split ray from x to x-edge into multiple rays passing through different z-cells
            nzcells = izedge - iz + 1
            #print('nzcells', nzcells)
            zend = (x[ix + 1]-dx/2)*m + q
            if zend > r[1]:
                zend = r[1]
            #print('zray, zend', zray, zend)
            zs = z[iz] + np.arange(nzcells-1) * dz + dz//2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            #print('xs', xs, 'zs', zs)
            plt.plot(xs, zs, '.y', ms=10)
            rrays = np.sqrt(np.diff(xs)**2 + np.diff(zs)**2)
            R[ix * nz + iz: ix * nz + iz + nzcells] = rrays
            c[ic:ic+nzcells], v[ic:ic+nzcells] = np.arange(ix * nz + iz, ix * nz + iz + nzcells), rrays
            ic += nzcells
            #print('rrays', rrays)
        else:
            # split ray from x to x-edge into multiple rays passing through different z-cells
            nzcells = iz - izedge + 1
            #print('nzcells', nzcells)
            zend = (x[ix + 1]-dx/2)*m + q
            if zend < r[1]:
                zend = r[1]
            #print('zray, zend', zray, zend)
            zs = z[iz] - np.arange(nzcells-1) * dz - dz/2
            zs = np.insert(zs, 0, zray)
            zs = np.append(zs, zend)
            xs = (zs - q) / m
            #print('xs', xs, 'zs', zs)
            plt.plot(xs, zs, '.y', ms=10)
            rrays = np.sqrt(np.diff(xs)**2 + np.diff(zs)**2)
            #print(ix * nz + iz - nzcells, ix * nz + iz)
            R[ix * nz + iz - nzcells + 1: ix * nz + iz + 1] = rrays[::-1]
            c[ic:ic+nzcells], v[ic:ic+nzcells] = np.arange(ix * nz + iz - nzcells + 1, ix * nz + iz + 1), rrays[::-1]
            ic += nzcells
            #print('rrays', rrays)
            #break
        plt.plot([xray, xedge], [zray, zedge], '.-y', ms=10)
        xray, zray = xedge, zedge
        #print(xray, zray)
    return R, c[:ic].astype(np.int), v[:ic]
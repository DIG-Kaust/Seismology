import os
import getpass
import datetime

import numpy as np
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt

try:
    from IPython.display import display_html, display, Javascript
    displayflag=1
except:
    displayflag=0


###########
# private #
###########
def _rgb2hex(r,g,b):
    """From rgb to hexadecimal color

    Parameters
    ----------
    r : :obj:`str`
        Red
    g : :obj:`str`
        Green
    b : :obj:`str`
        Blue

    Returns
    -------
    hex : :obj:`str`
        Hexadecimal color
    """
    hex = f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'
    return hex


def _discrete_cmap(N, base_cmap='jet'):
    """Create an N-bin discrete colormap from the specified input map

    Parameters
    ----------
    N : :obj:`int`
        Number of elements
    base_cmap : :obj:`str`, optional
        Base colormap

    Returns
    -------
    cmap : :obj:`matplotlib.colors.LinearSegmentedColormap`
        Discrete colorbar

    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    cmap = base.from_list(cmap_name, color_list, N)
    return cmap


def _discrete_cmap_indexed(colors):
    """Create an N-bin discrete colormap from the set of colors

    Parameters
    ----------
    colors : :obj:`list` or :obj:`tuple`
        Colors

    Returns
    -------
    cmap : :obj:`matplotlib.colors.LinearSegmentedColormap`
        Discrete colorbar

    """
    cmap = pltcolors.ListedColormap(colors, 'indexed')
    return cmap


def _wiggletrace(ax, tz, trace, center=0, cpos='r', cneg='b'):
    """Plot a seismic wiggle trace onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    trace : :obj:`np.ndarray `
        Wiggle trace
    center : :obj:`float`, optional
        Center of x-axis where to switch color
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    ax.fill_betweenx(tz, center, trace, where=(trace > center),
                     color=cpos, interpolate=True)
    ax.fill_betweenx(tz, center, trace, where=(trace <= center),
                     color=cneg, interpolate=True)
    ax.plot(trace, tz, 'k', lw=1)
    return ax


def _wiggletracecomb(ax, tz, x, traces, scaling=None,
                     cpos='r', cneg='b'):
    """Plot a comb of seismic wiggle traces onto an axis

    Parameters
    ----------
    ax : :obj:`plt.axes`, optional
         Axes handle
    tz : :obj:`np.ndarray `
        Depth (or time) axis
    x : :obj:`np.ndarray `
        Lateral axis
    traces : :obj:`np.ndarray `
        Wiggle traces
    scaling : :obj:`float`, optional
        Scaling to apply to each trace
    cpos : :obj:`str`, optional
        Color of positive filling
    cneg : :obj:`str`, optional
        Color of negative filling

    Returns
    -------
    ax : :obj:`plt.axes`, optional
         Axes handle

    """
    dx = np.abs(x[1]-x[0])
    tracesmax = np.max(traces, axis=1)
    tracesmin = np.min(traces, axis=1)
    dynrange = tracesmax - tracesmin
    maxdynrange = np.max(dynrange)
    if scaling is None:
        scaling = 2*dx/maxdynrange
    else:
        scaling = scaling*2*dx/maxdynrange

    for ix, xx in enumerate(x):
        trace = traces[ix]
        _wiggletrace(ax, tz, xx + trace * scaling, center=xx,
                     cpos=cpos, cneg=cneg)
    ax.set_xlim(x[0]-1.5*dx, x[-1]+1.5*dx)
    ax.set_ylim(tz[-1], tz[0])
    ax.set_xticks(x)
    ax.set_xticklabels([str(xx) for xx in x])

    return ax
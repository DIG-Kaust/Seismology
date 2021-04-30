import logging

import copy
import lasio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cplt
import seaborn as sns

from scipy.interpolate import interp1d
from scipy.stats import norm

from .units import *
from .utils import *
from .solid import *
from .fluid import *
from .gassmann import *
from .poststack import zerooffset_wellmod
from .avo import prestack_wellmod
from .visualutils import _discrete_cmap, _discrete_cmap_indexed, _wiggletrace, _wiggletracecomb

try:
    from IPython.display import display
    ipython_flag = True
except:
    ipython_flag=False


def _threshold_curve(curve, thresh, greater=True):
    """Apply thresholding to a curve

    Parameters
    ----------
    curve : :obj:`np.ndarray`
        Curve to be thresholded
    thresh : :obj:`float`
        Maximum allowed value (values above will be set to non-valid
    greater : :obj:`bool`, optional
        Apply threshold for values greater than ``thresh`` (``True``) or
        smaller than ``thresh`` (``False``)

    Returns
    -------
    threshcurve : :obj:`np.ndarray`
        Thresholded curve

    """
    threshcurve = np.copy(curve)
    if thresh is not None:
        if greater:
            threshcurve[threshcurve > thresh] = np.nan
        else:
            threshcurve[threshcurve < thresh] = np.nan
    return threshcurve

def _filters_curves(curves, filters):
    """Apply conditional filters to a set of log curves

    Parameters
    ----------
    curves : :obj:`pd.DataFrame`
        Set of log curves
    filters : :obj:`list` or :obj:`tuple`
        Filters to be applied
        (each filter is a dictionary with logname and rule, e.g.
        logname='LFP_COAL', rule='<0.1' will keep all values where values
        in  LFP_COAL logs are <0.1)

    Returns
    -------
    filtered_curves : :obj:`pd.DataFrame`
        Filtered curves
    cond : :obj:`pd.DataFrame`
        Filtering mask
    """
    if isinstance(filters, dict):
        filters = (filters, )
    cond = eval("curves['" + filters[0]['logname'] + "']" +
                filters[0]['rule']).values
    cond = cond | (np.isnan(curves[filters[0]['logname']].values))
    for filter in filters[1:]:
        if filter['chain'] == 'and':
            cond = cond & eval("curves['" + filter['logname'] + "']" +
                               filter['rule']).values
        else:
            cond = cond | eval("curves['" + filter['logname'] + "']" +
                               filter['rule']).values
        cond = cond | (np.isnan(curves[filter['logname']].values))
    filtered_curves = curves[cond]
    return filtered_curves, cond

def _visualize_curve(ax, logs, curve, depth='MD', thresh=None, shift=None,
                     verticalshift=0., scale=1., color='k', lw=2,
                     logscale=False, grid=False, inverty=True, ylabel=True,
                     xlabelpos=0, xlim=None, ylim=None, title=None, **kwargs):
    """Visualize single curve track in axis ``ax``

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`str`
        Keyword of log curve to be visualized
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    thresh : :obj:`float`, optional
        Maximum allowed value (values above will be set to non-valid)
    shift : :obj:`np.ndarray`, optional
        Depth-dependent shift to apply to the curve to visualize
    verticalshift : :obj:`np.ndarray`, optional
        Bulk vertical shift to apply to the curve to visualize
    scale : :obj:`float`, optional
        Scaling to apply to log curve
    color : :obj:`str`, optional
        Curve color
    lw : :obj:`int`, optional
        Line width
    semilog : :obj:`bool`, optional
        Use log scale in log direction
    grid : :obj:`bool`, optional
        Add grid to plot
    inverty : :obj:`bool`, optional
        Invert y-axis
    ylabel : :obj:`str`, optional
        Show y-label
    xlabelpos : :obj:`str`, optional
        Position of xlabel outside of axes (if ``None`` keep it as original)
    xlim : :obj:`tuple`, optional
        x-axis extremes
    ylim : :obj:`tuple`, optional
        y-axis extremes
    title : :obj:`str`, optional
        Title of figure
    kwargs : :obj:`dict`, optional
        Additional plotting keywords

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    try:
        logcurve = _threshold_curve(logs[curve], thresh)
        logcurve *= scale

        if shift is not None:
            logcurve += shift

        plot = True
    except:
        logging.warning('logs object does not contain {}...'.format(curve))
        plot = False

    if plot:
        if grid:
            ax.grid()
        if logscale:
            ax.semilogx(logcurve, logs[depth] + verticalshift, c=color,
                        lw=lw, **kwargs)
        else:
            ax.plot(logcurve, logs[depth] + verticalshift, c=color,
                    lw=lw, **kwargs)
        if ylabel:
            ax.set_ylabel(depth)
        if xlabelpos is not None:
            ax.set_xlabel(title if title is not None else curve, color=color)
            ax.tick_params(direction='in', width=2, colors=color,
                           bottom=False, labelbottom=False, top=True, labeltop=True)
            ax.spines['top'].set_position(('outward', xlabelpos*80))

        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim[1], ylim[0])
        else:
            if inverty:
                ax.invert_yaxis()
    return ax

def _visualize_colorcodedcurve(ax, logs, curve, depth='MD',
                               thresh=None, shift=None, verticalshift=0.,
                               scale=1., envelope=None,
                               leftfill=True, cmap='seismic', clim=None,
                               step=100):
    """Visualize filling of single curve track in colorcode in axis ``ax``.
    Generally used in combination with _visualize_colorcode.

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`str`
        Keyword of log curve to be visualized
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    thresh : :obj:`float`, optional
        Maximum allowed value (values above will be set to non-valid)
    shift : :obj:`np.ndarray`, optional
        Depth-dependent shift to apply to the curve to visualize
    verticalshift : :obj:`np.ndarray`, optional
        Bulk vertical shift to apply to the curve to visualize
    scale : :obj:`float`, optional
        Scaling to apply to log curve
    envelope : :obj:`float`, optional
        Value to use as envelope in color-coded display (if ``None``
        use curve itself)
    leftfill : :obj:`bool`, optional
        Fill on left side of curve (``True``) or right side of curve (``False``)
    cmap : :obj:`str`, optional
        Colormap for colorcoding
    clim : :obj:`tuple`, optional
        Limits of colorbar (if ``None`` use min-max of curve)
    step : :obj:`str`, optional
        Step for colorcoding

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    try:
        logcurve = _threshold_curve(logs[curve], thresh)
        logcurve_color = logcurve.copy()
        logcurve *= scale

        if shift is not None:
            logcurve += shift
        plot = True
    except:
        logging.warning('logs object does not contain {}...'.format(curve))
        plot = False

    if plot:
        # get temporary curves and subsampled them
        x = logcurve[::step]
        y = logs[depth][::step] + verticalshift
        z = logcurve_color[::step]
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        if clim is None:
            normalize = mpl.colors.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
        else:
            normalize = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

        for i in range(x.size - 1):
            if leftfill:
                ax.fill_betweenx([y[i], y[i + 1]], shift,
                                 x2=[x[i], x[i + 1]] if envelope is None \
                                     else [shift + envelope, shift + envelope],
                                 color=cmap(normalize(z[i])))
            else:
                ax.fill_betweenx([y[i], y[i + 1]],
                                 [x[i], x[i + 1]] if envelope is None \
                                     else [np.nanmax(z) - envelope, np.nanmax(z) - envelope],
                                 x2=np.nanmax(z), color=cmap(normalize(z[i])))
        #if envelope is not None:
        #    ax.plot([shift + envelope]*2, [y[i], y[i + 1]], 'k', lw=1)
    return ax

def _visualize_filled(ax, logs, curves, colors, depth='MD', envelope=None,
                      grid=False, inverty=True, ylabel=True, xlim=None,
                      title=None, **kwargs):
    """Visualize filled set of curves

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curves : :obj:`tuple`
        Keywords of N log curve to be visualized
    colors : :obj:`tuple`
        N+1 colors to be used for filling between curves
        (last one used as complement)
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    envelope : :obj:`str`, optional
        keyword of log curve to be used as envelope
    grid : :obj:`bool`, optional
        Add grid to plot
    inverty : :obj:`bool`, optional
        Invert y-axis
    ylabel : :obj:`str`, optional
        Show y-label
    xlim : :obj:`tuple`, optional
        x-axis extremes
    title : :obj:`str`, optional
        Title of figure
    kwargs : :obj:`dict`, optional
        Additional plotting keywords

    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    # check that sum of volumes does not exceed 1
    filllogs = np.array([logs[curve] for curve in curves])
    cumfilllogs = np.cumsum(np.array(filllogs), axis=0)
    exceedvol = np.sum(cumfilllogs[-1][~np.isnan(cumfilllogs[-1])]>1.)
    if exceedvol > 0:
        logging.warning('Sum of volumes exceeds '
                        '1 for {} samples'.format(exceedvol))
    if envelope is not None: cumfilllogs = cumfilllogs * logs[envelope]

    # plotting
    if grid:
        ax.grid()
    ax.fill_betweenx(logs[depth], cumfilllogs[0], facecolor=colors[0])
    ax.plot(cumfilllogs[0], logs[depth], 'k', lw=0.5)
    for icurve in range(len(curves)-1):
        ax.fill_betweenx(logs[depth], cumfilllogs[icurve],
                         cumfilllogs[icurve+1],
                         facecolor=colors[icurve+1], **kwargs)
        ax.plot(cumfilllogs[icurve], logs[depth], 'k', lw=0.5)
    if envelope is None:
        ax.fill_betweenx(logs[depth], cumfilllogs[-1], 1, facecolor=colors[-1])
        ax.plot(cumfilllogs[-1], logs[depth], 'k', lw=0.5)
    else:
        ax.fill_betweenx(logs[depth], cumfilllogs[-1], logs[envelope],
                         facecolor=colors[-1])
        ax.plot(cumfilllogs[-1], logs[depth], 'k', lw=0.5)
        ax.plot(logs[envelope], logs[depth], 'k', lw=1.5)
    if ylabel:
        ax.set_ylabel(depth)
    ax.set_title(title if title is not None else '', pad=20)
    ax.tick_params(direction='in', width=2, colors='k',
                   bottom=False, labelbottom=False, top=True, labeltop=True)
    if xlim is not None and len(xlim)==2:
        ax.set_xlim(xlim[0], xlim[1])
    if inverty:
        ax.invert_yaxis()
    return ax


def _visualize_facies(ax, logs, curve, colors, names, depth='MD',
                      cbar=False, title=None):
    """Visualize facies curve as image

    Parameters
    ----------
    ax : :obj:`plt.axes`
        Axes handle (if ``None`` draw a new figure)
    logs : :obj:`lasio.las.LASFile`
        Lasio object containing logs
    curve : :obj:`tuple`
        Keywords oflog curve to be visualized
    colors : :obj:`tuple`
        Colors to be used for facies
    colors : :obj:`tuple`
        Names to be used for facies
    depth : :obj:`str`, optional
        Keyword of log curve to be used for vertical axis
    cbar : :obj:`bool`, optional
        Show colorbar (``True``) or not (``False``)
    title : :obj:`str`, optional
        Title of figure
    Returns
    -------
    axs : :obj:`plt.axes`
       Axes handles

    """
    nfacies = len(colors)
    faciesext, zfaciesest = \
        logs.resample_curve(curve, zaxis=depth)
    faciesext = np.repeat(np.expand_dims(faciesext, 1),
                          nfacies, 1)
    cmap_facies = cplt.ListedColormap(colors,
                                      'indexed')
    im = ax.imshow(faciesext, interpolation='none',
                       aspect='auto', origin='lower',
                       extent=(0, nfacies,
                               zfaciesest[0],
                               zfaciesest[-1]),
                       cmap=cmap_facies, vmin=-0.5,
                       vmax=nfacies - 0.5)
    ax.set_title(title if title is not None else '', pad=20)
    if cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(np.arange(0, nfacies))
        cbar.set_ticklabels(names)
    return ax


class Logs:
    """Log curves object.

    This object contains a set of log curves for a single well from a .LAS file

    Parameters
    ----------
    filename : :obj:`str`
        Name of file containing logs to be read
    wellname : :obj:`str`, optional
        Name of file containing logs to be read
    loadlogs : :obj:`bool`, optional
        Load data into ``self.logs`` variable during initialization (``True``)
        or not (``False``)
    kind : :obj:`str`, optional
        ``local`` when data are stored locally in a folder,'
        ``onprem`` when data are stored on-premise,
        ``omnia`` when data are stored in Omnia datalake storage
    ads : :obj:`azure.datalake.store.core.AzureDLFileSystem`, optional
        omnia datalake access pointer (needed only if ``kind='omnia'``)
    verb : :obj:`str`, optional
        Verbosity

    """
    def __init__(self, filename, wellname=None, loadlogs=True,
                 kind='local', ads=None, verb=False):
        self.filename = filename
        self.df = None
        self.wellname = filename if wellname is None else wellname
        self._loadlogs = loadlogs
        self._kind = kind
        self._ads = ads
        self._verb = verb
        if self._loadlogs:
            self._logs = self._read_logs()

    @property
    def logs(self):
        if not self._loadlogs:
            self._loadlogs = True
            self._logs = self._read_logs()

        return self._logs

    @property
    def avestep(self):
        return np.median(np.unique(np.diff(self.logs.index)))

    def __str__(self):
        descr = 'Logs {})\n\n'.format(self.wellname) + \
                'Curves: {}\n'.format(list(self.logs.keys()))
        return descr

    def _read_logs(self):
        """Read a set of logs from file
        """
        if self._verb:
            print('Reading {} logs...'.format(self.filename))
        if self._kind in ('local', 'onprem') or \
            (self._kind == 'omnia' and self._ads is None):
            logs = lasio.read(self.filename)
        elif self._kind == 'omnia':
            with self._ads.open(self.filename, 'rb') as f:
                logs = lasio.read(f.read().decode('utf-8'))
        else:
            raise NotImplementedError('kind must be local, onprem, or omnia')
        # ensure there is no TWT curve leaking in... we only want to get them
        # from TD curves or checkshots so we can keep track of them...
        if 'TWT' in logs.keys():
            logs.delete_curve('TWT')
        return logs

    def copy(self):
        """Return a copy of the object.

        Returns
        -------
        logscopy : :obj:`ptcpy.objects.Logs`
            Copy of Logs object

        """
        logscopy = copy.deepcopy(self)
        return logscopy

    def dataframe(self, resetindex=False):
        """Return log curves into a :obj:`pd.DataFrame`

        Parameters
        ----------
        resetindex : :obj:`bool`, optional
            Move index to curve DEPTH and reset index to consecutive numbers
        """
        self.df = self.logs.df()
        if resetindex:
            depth = self.df.index
            self.df.reset_index(inplace=True)
            self.df['DEPTH'] = depth

    def startsample(self, curve=None):
        "index of first available sample in log"
        if curve is None:
            curve = self.logs.curves[1].mnemonic
        mask = np.cumsum(~np.isnan(self.logs[curve]))
        return np.where(mask == 1)[0][0]

    def endsample(self, curve=None):
        "index of last available sample in log"
        if curve is None:
            curve = self.logs.curves[1].mnemonic
        mask = np.cumsum(np.flipud(~np.isnan(self.logs[curve])))
        return len(mask) - np.where(mask == 1)[0][0]

    def add_curve(self, curve, mnemonic, unit=None, descr=None, value=None,
                  delete=True):
        """Add curve to logset

        Parameters
        ----------
        curve : :obj:`np.ndarray`
            Curve to be added
        mnemonic : :obj:`str`
            Curve mnemonic
        unit : :obj:`str`, optional
            Curve unit
        descr : :obj:`str`, optional
            Curve description
        value : :obj:`int`, optional
            Curve value
        delete : :obj:`bool`, optional
            Delete curve with same name if present (``True``) or not (``False``)
        """
        if delete:
            self.delete_curve(mnemonic)
        self.logs.append_curve(mnemonic, curve,
                               unit='' if unit is None else unit,
                               descr='' if descr is None else descr,
                               value='' if value is None else value)
        self.dataframe()

    def add_tvdss(self, trajectory):
        """Add TVDSS curve (and interpolate from trajectory to logs sampling)

        Parameters
        ----------
        trajectory : :obj:`ptcpy.objects.Trajectory`
            Curve to be added

        """
        # create regular tvdss axis for mapping of picks
        md = trajectory.df['MD (meters)']
        tvdss = trajectory.df['TVDSS']

        f = interp1d(md, tvdss, kind='linear',
                     bounds_error=False, assume_sorted=True)
        tvdss_log = f(self.logs.index)
        self.logs.append_curve('TVDSS', tvdss_log, unit='m', descr='TVDSS')
        self.dataframe()

    def add_twt(self, tdcurve, name):
        """Add TWT curve (and interpolate from trajectory to logs sampling)

        Parameters
        ----------
        tdcurve : :obj:`ptcpy.objects.TDcurve`
            TD curve or checkshots
        tdcurve : :obj:`ptcpy.objects.TDcurve`
            name of TD or checkshot curve to be used within Logs object

        """
        # create regular tvdss axis for mapping of picks
        md = tdcurve.df['Md (meters)']
        twt = tdcurve.df['Time (ms)']

        f = interp1d(md, twt, kind='linear',
                     bounds_error=False, assume_sorted=True)
        twt_log = f(self.logs.index)
        self.logs.append_curve('TWT - {}'.format(name),
                               twt_log, unit='ms', descr='TWT')
        self.dataframe()

    def delete_curve(self, mnemonic, verb=False):
        """Delete curve to logset

        Parameters
        ----------
        mnemonic : :obj:`str`
            Curve mnemonic
        verb : :obj:`bool`, optional
            Verbosity

        """
        if mnemonic in self.logs.keys():
            if verb:
                print('Deleted {} from {} well'.format(mnemonic,
                                                       self.wellname))
            self.logs.delete_curve(mnemonic)
        else:
            if verb: print('Curve {} not present for '
                           '{} well'.format(mnemonic, self.wellname))

    def resample_curve(self, mnemonic, zaxis=None, mask=None, step=None):
        """Return resampled curve with constant step in depth axis.

        Parameters
        ----------
        mnemonic : :obj:`str`
            Curve mnemonic
        zaxis : :obj:`str`, optional
            Label of log to use as z-axis
        mask : :obj:`np.ndarray`, optional
            Mask to apply prior to resampling (values where mask is ``True``
            will be put to np.nan)s
        step : :obj:`float`, optional
            Step. If ``None`` estimated as median value of different steps
            in current depth axis

        Returns
        -------
        loginterp : :obj:`str`
            Interpolated log
        regaxis : :obj:`np.ndarray`
            Regularly sampled depth axis

        """
        if zaxis is None:
            start = self.logs.index[0]
            end = self.logs.index[-1]
        else:
            zaxis_nonan = self.logs[zaxis][~np.isnan(self.logs[zaxis])]
            start = zaxis_nonan[0]
            end = zaxis_nonan[-1]
        if step is None:
            if zaxis is None:
                step = self.avestep
            else:
                steps = np.unique(np.diff(self.logs[zaxis]))
                step = np.max(steps[~np.isnan(steps)])
        regaxis = np.arange(start, end + step, step)

        # Resample the logs to the new axis using linear interpolation
        log = self.logs[mnemonic].copy()
        if mask is not None:
            log[mask] = np.nan
        loginterp = np.interp(regaxis,
                              self.logs.index if zaxis is None else
                              self.logs[zaxis], log)
        return loginterp, regaxis

    def fluid_substitution(self, sand, shale, oil, water, changes, coal=None,
                           carb=None, gas=None, porocutoff=[0, 1.],
                           vshcutoff=[0., 1.], lfp=False,
                           phi='PHIT', vsh='VSH', vcoal='VCOAL', vcarb='VCARB',
                           vp='VP', vs='VS', rho='RHOB', ai='AI', vpvs='VPVS',
                           sot='SOT', sgt='SGT', savelogs=True, savedeltas=True,
                           logssuffix='fluidsub'):
        """Gassmann fluid substitution on well logs.

        Parameters
        ----------
        sand : :obj:`dict`
            Bulk modulus and density of sand in dictionary ``{'k': X, 'rho': X}``
        shale : :obj:`dict`
            Bulk modulus and density of shale in dictionary ``{'k': X, 'rho': X}``
        oil : :obj:`ptcpy.proc.rockphysics.fluid.Oil`
            Oil object
        water : :obj:`ptcpy.proc.rockphysics.fluid.Brine`
            Brine object
        change : :obj:`dict` or :obj:`list`
            Changes to be applied to saturation logs in dictionary(ies)
            ``{'zmin': X or pickname, 'zmax': X or pickname, 'sot': X, 'sgt': X}``
            where
        coal : :obj:`dict`, optional
            Bulk modulus and density of coal in dictionary ``{'k': X, 'rho': X}``
        carb : :obj:`dict`, optional
            Bulk modulus and density of carbonate/calcite in dictionary
            ``{'k': X, 'rho': X}``
        gas : :obj:`ptcpy.proc.rockphysics.fluid.Gas`, optional
            Gas object
        lfp : :obj:`bool`, optional
            Prepend `LFP_`` to every log (``True``) or not (``False``)
        phi : :obj:`str`, optional
            Name of Porosity log
        vsh : :obj:`str`, optional
            Name of gamma ray log
        vcoal : :obj:`str`, optional
            Name of Volume Coal log
        vcarb : :obj:`str`, optional
            Name of Volume Carbonate log
        vp : :obj:`str`, optional
            Name of P-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        rho : :obj:`str`, optional
            Name of Density log
        ai : :obj:`str`, optional
            Name of Acoustic Impedence log
        vpvs : :obj:`str`, optional
            Name of VP/VS log
        sot : :obj:`str`, optional
            Name of Total Oil Saturation log
        sgt : :obj:`str`, optional
            Name of Total Gas Saturation Ray log
        timeshiftpp : :obj:`bool`, optional
            Compute PP timeshift
        savelogs : :obj:`bool`, optional
            Save fluid substituted profiles as logs
        savedeltas : :obj:`bool`, optional
            Save differences as logs
        logssuffix : :obj:`str`, optional
            Suffix to add to log names if saved

        Returns
        -------
        vp1 : :obj:`numpy.ndarray`
            Fluid-substituted P-wave velocity
        vs1 : :obj:`numpy.ndarray`
            Fluid-substituted S-wave velocity
        rho1 : :obj:`numpy.ndarray`
            Fluid-substituted density
        so1 : :obj:`numpy.ndarray`
            Fluid-substituted oil saturation
        sg1 : :obj:`numpy.ndarray`
            Fluid-substituted gas saturation

        """
        # prepend lfp if lfp flag is True
        if lfp:
            vsh = 'LFP_' + vsh if lfp else vsh
            vcarb = 'LFP_' + vcarb if lfp else vcarb
            vcoal = 'LFP_' + vcoal if lfp else vcoal
            sgt = 'LFP_' + sgt if lfp else sgt
            sot = 'LFP_' + sot if lfp else sot
            phi = 'LFP_' + phi if lfp else phi
            vp = 'LFP_' + vp if lfp else vp
            vs = 'LFP_' + vs if lfp else vs
            rho = 'LFP_' + rho if lfp else rho
            ai = 'LFP_' + ai if lfp else ai
            vpvs = 'LFP_' + vpvs if lfp else vpvs
        vpname, vsname, rhoname, ainame, vpvsname, sotname, sgtname, = \
            vp, vs, rho, ai, vpvs, sot, sgt

        # extract logs
        z = self.logs.index
        phi = self.logs[phi].copy()
        vp = self.logs[vp].copy()
        vs = self.logs[vs].copy()
        rho = g_cm3_to_kg_m3(self.logs[rho].copy())
        vsh = self.logs[vsh].copy()
        vsand = 1. - vsh
        so0 = self.logs[sot].copy()
        sw0 = 1. - so0
        if carb is not None:
            vcarb = self.logs[vcarb]
            vsand -= vcarb
        else:
            vcarb = np.zeros_like(vsh)
        if coal is not None:
            vcoal = self.logs[vcoal]
            vsand -= vcoal
        else:
            vcoal = np.zeros_like(vsh)
        if sgt is not None:
            sg0 = self.logs[sgt]
            sw0 -= sg0
        else:
            sg0 = np.zeros_like(vsh)

        # cutoffs
        cutoff = np.zeros(len(z)).astype(bool)
        if porocutoff[0] > 0. or porocutoff[1] < 1.:
            cutoff = cutoff | (phi < porocutoff[0]) | (phi > porocutoff[1])
        if vshcutoff[0] > 0. or vshcutoff[1] < 1.:
            cutoff = cutoff | (vsh < vshcutoff[0]) | (vsh > vshcutoff[1])

        # fix nans fpr elastic params
        nans = cutoff | np.isnan(vp) | np.isnan(vs) | np.isnan(rho) | np.isnan(
            phi) | \
               np.isnan(vsh) | np.isnan(so0) | np.isnan(sg0)
        vp[nans] = 0
        vs[nans] = 0
        rho[nans] = 0
        phi[nans] = 0

        # apply changes to fluids
        sg1 = sg0.copy()
        so1 = so0.copy()
        if not isinstance(changes, list):
            changes = [changes]
        for change in changes:
            izmin = findclosest(z, change['zmin'])
            izmax = findclosest(z, change['zmax'])

            if 'sot' in change.keys():
                so1[izmin:izmax] = change['sot']
            if 'sgt' in change.keys():
                sg1[izmin:izmax] = change['sgt']
            sw1 = 1 - so1 - sg1
        so1_filled = so1.copy()
        sg1_filled = sg1.copy()
        sw1_filled = sw1.copy()
        so0[nans] = 0
        sg0[nans] = 0
        sw0[nans] = 1.
        so1_filled[nans] = 0
        sg1_filled[nans] = 0
        sw1_filled[nans] = 1.

        # create matrix and fluid
        sand['frac'] = vsand
        shale['frac'] = vsh
        minerals = {'sand': sand, 'shale': shale}

        if coal is not None:
            coal['frac'] = vcoal
            minerals['coal'] = coal
        if carb is not None:
            carb['frac'] = vcarb
            minerals['carb'] = carb
        mat = Matrix(minerals)

        if gas is None:
            fluid0 = Fluid({'oil': (oil, so0),
                            'water': (water, sw0)})
            fluid1 = Fluid({'oil': (oil, so1_filled),
                            'water': (water, sw1_filled)})
        else:
            fluid0 = Fluid({'gas': (gas, sg0),
                            'oil': (oil, so0),
                            'water': (water, sw0)})
            fluid1 = Fluid({'gas': (gas, sg1_filled),
                            'oil': (oil, so1_filled),
                            'water': (water, sw1_filled)})

        # fluid substitution
        medium0 = Rock(vp, vs, rho, mat, fluid0, poro=phi)
        fluidsub = Gassmann(medium0, fluid1, mask=True)

        # fill with original values in cutoff regions
        vp1, vs1, rho1 = fluidsub.medium1.vp, fluidsub.medium1.vs, fluidsub.medium1.rho
        vp1[cutoff] = self.logs[vpname][cutoff]
        vs1[cutoff] = self.logs[vsname][cutoff]
        rho1[cutoff] = g_cm3_to_kg_m3(self.logs[rhoname])[cutoff]

        # save logs
        if savelogs:
            self.add_curve(so1, '{}_{}'.format(sotname, logssuffix),
                           unit='frac',
                           descr='{} - {}'.format(sotname, logssuffix))
            self.add_curve(sg1, '{}_{}'.format(sgtname, logssuffix),
                           unit='frac',
                           descr='{} - {}'.format(sotname, logssuffix))
            self.add_curve(vp1, '{}_{}'.format(vpname, logssuffix),
                           unit='m/s',
                           descr='{} - {}'.format(vpname, logssuffix))
            self.add_curve(vs1, '{}_{}'.format(vsname, logssuffix),
                           unit='m/s',
                           descr='{} - {}'.format(vsname, logssuffix))
            self.add_curve(kg_m3_to_g_cm3(rho1),
                           '{}_{}'.format(rhoname, logssuffix),
                           unit='g/cm3',
                           descr='{} - {}'.format(rhoname, logssuffix))
            self.add_curve(vp1 * kg_m3_to_g_cm3(rho1),
                           '{}_{}'.format(ainame, logssuffix),
                           unit=None,
                           descr='{} - {}'.format(ainame, logssuffix))
            self.add_curve(vp1 / vs1, '{}_{}'.format(vpvsname, logssuffix),
                           unit=None,
                           descr='{} - {}'.format(vpvsname, logssuffix))
        if savedeltas:
            # differences
            vp = self.logs[vpname].copy()
            vs = self.logs[vsname].copy()
            rho = g_cm3_to_kg_m3(self.logs[rhoname].copy())

            self.add_curve(
                sg1 - self.df[sgtname].values.copy(),
                '{}diff_{}'.format(sgtname, logssuffix),
                unit='frac',
                descr='d{} - {}'.format(sgtname, logssuffix))
            self.add_curve(
                sg1 - self.df[sgtname].values.copy(),
                '{}diff_{}'.format(sgtname, logssuffix),
                unit='frac',
                descr='d{} - {}'.format(sgtname, logssuffix))
            self.add_curve(
                200 * (vp1 * kg_m3_to_g_cm3(rho1) - vp * kg_m3_to_g_cm3(rho)) / \
                (vp1 * kg_m3_to_g_cm3(rho1) + vp * kg_m3_to_g_cm3(rho)),
                '{}diff_{}'.format(ainame, logssuffix),
                unit='frac',
                descr='{} - {}'.format(ainame, logssuffix))
            self.add_curve(
                200 * (vp1 / vs1 - vp / vs) / (vp1 / vs1 + vp / vs),
                '{}diff_{}'.format(vpvsname, logssuffix),
                unit='frac',
                descr='d{} - {}'.format(vpvsname, logssuffix))
        return vp1, vs1, rho1, so1, sg1

    #########
    # Viewers
    #########
    def display(self, nrows=10):
        """Display logs as table

        nrows : :obj:`int`, optional
            Number of rows to display (if ``None`` display all)

        """
        self.dataframe()

        if ipython_flag:
            display(self.df.head(nrows))
        else:
            print(self.df.head(nrows))

    def describe(self):
        """Display statistics of logs
        """
        self.dataframe()

        if ipython_flag:
            display(self.df.describe())
        else:
            print(self.df.describe())

    def visualize_logcurve(self, curve, depth='MD',
                           thresh=None, shift=None, verticalshift=0.,
                           scale=1.,
                           color='k', lw=2,
                           grid=True, xlabelpos=0,
                           inverty=True, curveline=True,
                           colorcode=False, envelope=None,
                           cmap='seismic', clim=None,
                           step=40, leftfill=True,
                           ax=None, figsize=(4, 15), title=None,
                           savefig=None, **kwargs):
        """Visualize log track as function of certain depth curve

        curve : :obj:`str`
            Keyword of log curve to be visualized
        depth : :obj:`str`, optional
            Keyword of log curve to be used for vertical axis
        thresh : :obj:`float`, optional
            Maximum allowed value (values above will be set to non-valid)
        shift : :obj:`np.ndarray`, optional
            Depth-dependent lateral shift to apply to the curve to visualize
        verticalshift : :obj:`np.ndarray`, optional
            Bulk vertical shift to apply to the curve to visualize
        scale : :obj:`float`, optional
            Scaling to apply to log curve
        color : :obj:`str`, optional
            Curve color
        lw : :obj:`int`, optional
            Line width
        grid : :obj:`bool`, optional
            Add grid to plot
        xlabelpos : :obj:`str`, optional
            Position of xlabel outside of axes
        inverty : :obj:`bool`, optional
            Invert y-axis
        curveline : :obj:`bool`, optional
            Display curve as line between curve and max value of curve
        colorcode : :obj:`bool`, optional
            Display curve color-coded between well trajectory and ``envelope``
        envelope : :obj:`float`, optional
            Value to use as envelope in color-coded display (if ``None``
            use curve itself)
        cmap : :obj:`str`, optional
            Colormap for colorcoding
        clim : :obj:`tuple`, optional
            Limits of colorbar (if ``None`` use min-max of curve)
        step : :obj:`str`, optional
            Step for colorcoding
        leftfill : :obj:`bool`, optional
            Fill on left side of curve (``True``) or right side of curve (``False``)
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        if colorcode:
            ax = _visualize_colorcodedcurve(ax, self.logs, curve, depth=depth,
                                            thresh=thresh, shift=shift,
                                            verticalshift=verticalshift,
                                            scale=scale, envelope=envelope,
                                            cmap=cmap, clim=clim, step=step,
                                            leftfill=leftfill)
        if curveline:
            ax = _visualize_curve(ax, self.logs, curve, depth=depth,
                                  thresh=thresh, shift=shift,
                                  verticalshift=verticalshift, scale=scale,
                                  color=color, lw=lw,
                                  xlabelpos=xlabelpos, inverty=inverty,
                                  grid=grid, title=title, **kwargs)

        if savefig is not None:
            plt.subplots_adjust(bottom=0.2)
            fig.savefig(savefig, dpi=300)

        return fig, ax

    def visualize_logcurves(self, curves, depth='MD', ylim=None,
                            grid=True, ylabel=True, seisreverse=False,
                            prestack_wiggles=True,
                            axs=None, figsize=(9, 7),
                            title=None, savefig=None, **kwargs):
        """Visualize multiple logs curves using a common depth axis and
        different layouts (e.g., line curve, filled curve)
        depending on the name given to the curve.

        The parameter ``curves`` needs to be a dictionary of dictionaries
        whose keys can be:

        * 'Volume': volume plot with filled curves from ``xlim[0]`` to
          ``xlim[1]``. The internal dictionary needs to contain the
          following keys, ``logs`` as a list of log curves, ``colors`` as a
          list of colors for filling and ``xlim`` to be the limit of x-axis
        * 'Sat': saturatuion plot with filled curves from ``xlim[0]`` to
          ``xlim[1]`` (or envelope curve). The internal dictionary needs to
          contain the following keys, ``logs`` as a list of log curves,
          ``colors`` as a list of colors for filling, ``envelope`` as a string
          containing the name of the envolope curve and ``xlim`` to be the
          limit of x-axis.
        * 'Stack*': modelled seismic trace. The internal dictionary needs to
          contain the following keys, ``log`` as the AI log used to model
          seismic data, ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * 'Diff*': modelled difference between two seismic traces.
          The internal dictionary needs to contain the following keys,
          ``logs`` as the AI logs used to model seismic data (subtraction
          convention is first - second),
          ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * 'Prestack*': modelled pre-stack seismic gather. The internal
          dictionary needs to contain the following keys, ``vp`` and
          ``vs`` and ``rho`` as VP, VS and density logs used to model
          seismic data, ``theta`` for the angles to be modelled,
          ``sampling`` for the sampling of the trace and ``wav``
          for the wavelet to use in the modelling procedure.
        * Anything else: treated as single log curve or multiple overlayed
          log curves.  The internal dictionary needs to
          contain the following keys, ``logs`` as a list of log curves,
          ``colors`` as a list of colors for filling, ``lw`` as a list of
          line-widths, ``xlim`` to be the limit of x-axis

        curves : :obj:`dict`
            Dictionary of curve types and names
        depth : :obj:`str`, optional
            Keyword of log curve to be used for vertical axis
        ylim : :obj:`tuple`, optional
            Limits for depth axis
        grid : :obj:`bool`, optional
            Add grid to plots
        ylabel : :obj:`bool`, optional
            Add ylabel to first plot
        seisreverse : :obj:`bool`, optional
            Reverse colors for seismic plots
        prestack_wiggles : :obj:`bool`, optional
            Use wiggles to display pre-stack seismic (``True``) or imshow
            (``False``)
        axs : :obj:`plt.axes`
            Axes handles (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if seisreverse:
            cpos, cneg = 'b', 'r'
        else:
            cpos, cneg = 'r', 'b'

        N = len(curves)
        if axs is None:
            fig, axs = plt.subplots(1, N, sharey=True, figsize=figsize)
        else:
            fig = None

        # Plot the specified curves
        ncurves_max = 1
        for i, key in enumerate(curves.keys()):
            if key is 'Volume':
                axs[i] = _visualize_filled(axs[i], self.logs, curves[key]['logs'],
                                           colors=curves[key]['colors'], xlim=curves[key]['xlim'],
                                           depth=depth, grid=grid, inverty=False, title='Vol',
                                           ylabel=True if i==0 else False)
            elif 'Sat' in key:
                axs[i] = _visualize_filled(axs[i], self.logs, curves[key]['logs'],
                                           envelope = curves[key]['envelope'],
                                           colors=curves[key]['colors'], xlim=curves[key]['xlim'],
                                           depth=depth, grid=grid, inverty=False, title='Sat',
                                           ylabel=True if i==0 else False)
            elif 'Stack' in key:
                trace, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                       ai=curves[key]['log'])[:2]
                axs[i] = _wiggletrace(axs[i], tz, trace, cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Seismic' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=12)

                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Diff' in key:
                # identify common mask where samples from both logs are not nan
                ai1 = self.logs[curves[key]['logs'][0]]
                ai2 = self.logs[curves[key]['logs'][1]]
                mask = (np.isnan(ai1) | np.isnan(ai2))

                trace1, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                       ai=curves[key]['logs'][0], mask=mask)[:2]
                trace2, tz = \
                    zerooffset_wellmod(self, depth, curves[key]['sampling'],
                                       curves[key]['wav'],
                                       wavcenter=None if 'wavcenter' not in
                                                         curves[key].keys() else
                                       curves[key]['wavcenter'],
                                       ai=curves[key]['logs'][1], mask=mask)[:2]
                dtrace = trace1 - trace2
                axs[i] = _wiggletrace(axs[i], tz, dtrace, cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Seismic difference' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=7)
                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Prestack' in key:
                traces, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'], curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                     vp=curves[key]['vp'], vs=curves[key]['vs'],
                                     rho=curves[key]['rho'],
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     title='Modelled Pre-stack Seismic',
                                     plotflag=False)[0:2]
                if prestack_wiggles:
                    axs[i] = _wiggletracecomb(axs[i], tz, curves[key]['theta'],
                                              traces, scaling=curves[key]['scaling'],
                                              cpos=cpos, cneg=cneg)
                else:
                    axs[i].imshow(traces.T, vmin=-np.abs(traces).max(),
                                  vmax=np.abs(traces).max(),
                                  extent=(curves[key]['theta'][0],
                                          curves[key]['theta'][-1],
                                          tz[-1], tz[0]), cmap='seismic')
                    axs[i].axis('tight')
                axs[i].set_title('Modelled Pre-stack Seismic' if 'title' not in curves[key].keys()
                                 else curves[key]['title'], fontsize=7)
                if 'xlim' in curves[key].keys(): axs[i].set_xlim(curves[key]['xlim'])

            elif 'Prediff' in key:
                # identify common mask where samples from both logs are not nan
                vp1 = self.logs[curves[key]['vp'][0]]
                vp2 = self.logs[curves[key]['vp'][1]]
                vs1 = self.logs[curves[key]['vs'][0]]
                vs2 = self.logs[curves[key]['vs'][1]]
                rho1 = self.logs[curves[key]['rho'][0]]
                rho2 = self.logs[curves[key]['rho'][1]]
                mask = (np.isnan(vp1) | np.isnan(vp2) |
                        np.isnan(vs1) | np.isnan(vs2) |
                        np.isnan(rho1) | np.isnan(rho2))

                traces1, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'], curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else curves[key]['wavcenter'],
                                     vp=curves[key]['vp'][0], vs=curves[key]['vs'][0],
                                     rho=curves[key]['rho'][0], mask=mask,
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     plotflag=False)[0:2]
                traces2, tz = \
                    prestack_wellmod(self, depth, curves[key]['theta'],
                                     curves[key]['sampling'],
                                     curves[key]['wav'],
                                     wavcenter=None if 'wavcenter' not in curves[key].keys() else
                                     curves[key]['wavcenter'],
                                     vp=curves[key]['vp'][1], vs=curves[key]['vs'][1],
                                     rho=curves[key]['rho'][1], mask=mask,
                                     zlim=ylim, ax=axs[i],
                                     scaling=None if 'scaling' not in curves[key] else curves[key]['scaling'],
                                     plotflag=False)[0:2]
                axs[i] = _wiggletracecomb(axs[i], tz, curves[key]['theta'],
                                          traces1 - traces2,
                                          scaling=curves[key]['scaling'],
                                          cpos=cpos, cneg=cneg)
                axs[i].set_title('Modelled Pre-stack Seismic difference'
                                 if 'title' not in curves[key].keys() else curves[key]['title'],
                                 fontsize = 7)
                if 'xlim' in curves[key].keys():
                    axs[i].set_xlim(curves[key]['xlim'])

            elif 'Facies' in key:
                axs[i] = _visualize_facies(axs[i], self,
                                           curves[key]['log'],
                                           curves[key]['colors'],
                                           curves[key]['names'],
                                           depth=depth,
                                           cbar=False if 'cbar' not in \
                                                curves[key].keys() \
                                                else curves[key]['cbar'],
                                           title=key)

            else:
                ncurves = len(curves[key]['logs'])
                ncurves_max = ncurves if ncurves > ncurves_max else ncurves_max
                for icurve, (curve, color) in enumerate(zip(curves[key]['logs'],
                                                            curves[key]['colors'])):
                    if 'lw' not in curves[key].keys(): curves[key]['lw'] = [int(2)] * ncurves
                    if icurve == 0:
                        axs[i].tick_params(which='both', width=0, bottom=False,
                                           labelbottom=False, top=False, labeltop=False)

                    axs_tmp = axs[i].twiny()
                    axs_tmp = self.visualize_logcurve(curve, depth=depth, thresh=None,
                                                      color=color, lw=curves[key]['lw'][icurve],
                                                      xlim=curves[key]['xlim'],
                                                      logscale = False if 'logscale' not in curves[key] else curves[key]['logscale'],
                                                      grid=grid,
                                                      inverty=False, ylabel=True if i==0 else False,
                                                      xlabelpos=icurve/ncurves,
                                                      ax=axs_tmp, title=curve, **kwargs)
                axs[i].set_xlim(curves[key]['xlim'])
        axs[0].invert_yaxis()
        if ylabel:
            axs[0].set_ylabel(depth)
        if ylim is not None:
            axs[0].set_ylim(ylim[1], ylim[0])
        if fig is not None:
            fig.suptitle(self.filename if title is None else title, y=0.93+ncurves_max*0.02,
                         fontsize=20, fontweight='bold')

        if savefig is not None and fig is not None:
            fig.savefig(savefig, dpi=300)

        return fig, axs

    def visualize_histogram(self, curve, thresh=None, thresh1=None, bins=None, color='k',
                            grid=True, ax=None, figsize=(9, 7), title=None,
                            savefig=None):
        """Visualize histogram of log curve

        Parameters
        ----------
        curve : :obj:`str`
            Keyword of log curve to be visualized
        thresh : :obj:`float`, optional
            Maximum allowed value (values above will be set to non-valid)
        thresh1 : :obj:`float`, optional
            Minimum allowed value (values above will be set to non-valid)
        color : :obj:`str`, optional
            Curve color
        grid : :obj:`bool`, optional
            Add grid to plot
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        try:
            logcurve = _threshold_curve(self.logs[curve], thresh)
            logcurve = _threshold_curve(logcurve, thresh1, greater=False)
        except:
            raise ValueError('{} does not contain {}...'.format(self.filename,
                                                                curve))
        # remove nans
        logcurve = logcurve[~np.isnan(logcurve)]

        # plot samples
        if grid:
            ax.grid()
        sns.distplot(logcurve, fit=norm, rug=False, bins=bins,
                     hist_kws={'color': color, 'alpha': 0.5},
                     kde_kws={'color':color, 'lw': 3},
                     fit_kws={'color': color, 'lw': 3, 'ls':'--'},
                     ax=ax)
        ax.set_xlabel(curve)
        if title is not None: ax.set_title(title)
        if bins is not None: ax.set_xlim(bins[0], bins[-1])
        ax.text(0.95 * ax.get_xlim()[1], 0.85 * ax.get_ylim()[1],
                'mean: {0:%.3f},\nstd: {1:%.3f}' % (np.mean(logcurve),
                                                    np.std(logcurve)),
                fontsize=14,
                ha="right", va="center",
                bbox=dict(boxstyle="square",
                          ec=(0., 0., 0.),
                          fc=(1., 1., 1.)))

        if savefig is not None:
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            fig.savefig(savefig, dpi=300)
        return fig, ax

    def visualize_crossplot(self, curve1, curve2, curvecolor=None,
                            thresh1=None, thresh2=None, threshcolor=None,
                            cmap='jet', cbar=True, cbarlabels=None,
                            grid=True, ax=None, figsize=(9, 7),
                            title = None, savefig = None, **kwargs):
        """Crossplot two log curves (possibly color-coded using another curve)

        curve1 : :obj:`str`
            Keyword of log curve to be visualized along x-axis
        curve2 : :obj:`str`
            Keyword of log curve to be visualized along y-axis
        curvecolor : :obj:`str`
            Keyword of log curve to be color-coded
        thresh1 : :obj:`float`, optional
            Maximum allowed value for curve1
            (values above will be set to non-valid)
        thresh2 : :obj:`float`, optional
            Maximum allowed value for curve2
            (values above will be set to non-valid)
        threshcolor : :obj:`float`, optional
            Maximum allowed value for curvecolor
            (values above will be set to non-valid)
        cmap : :obj:`str` or :obj:`list`, optional
            Colormap name or list of colors for discrete map
        cbar : :obj:`bool`, optional
            Add colorbar
        cbarlabels : :obj:`list` or :obj:`tuple`, optional
            Labels to be added to colorbar. To be used for discrete colorbars
        grid : :obj:`bool`, optional
            Add grid to plot
        ax : :obj:`plt.axes`
            Axes handle (if ``None`` draw a new figure)
        figsize : :obj:`tuple`, optional
             Size of figure
        title : :obj:`str`, optional
             Title of figure
        savefig : :obj:`str`, optional
             Figure filename, including path of location where to save plot
             (if ``None``, figure is not saved)
        kwargs : :obj:`dict`, optional
             Additional plotting keywords

        Returns
        -------
        fig : :obj:`plt.figure`
            Figure handle (``None`` if ``axs`` are passed by user)
        ax : :obj:`plt.axes`
            Axes handle
        scatax : :obj:`matplotlib.collections.PathCollection`
            Scatterplot handle
        cbar : :obj:`matplotlib.colorbar.Colorbar`
            Colorbar handle
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = None
        try:
            logcurve1 = _threshold_curve(self.logs[curve1], thresh1)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curve1))
        try:
            logcurve2 = _threshold_curve(self.logs[curve2], thresh2)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curve2))
        try:
            if len(curvecolor):
                logcurvecolor = _threshold_curve(self.logs[curvecolor], threshcolor)
        except:
            raise ValueError('%s does not contain %s...' % (self.filename, curvecolor))

        if cbarlabels:
            if isinstance(cmap, str):
                cmap = _discrete_cmap(len(cbarlabels), cmap)
            else:
                cmap = _discrete_cmap_indexed(cmap)

        if grid:
            ax.grid()
        if curvecolor is not None:
            scatax = ax.scatter(logcurve1, logcurve2, c=logcurvecolor,
                                marker='o', edgecolors='none', alpha=0.7,
                                vmin=np.nanmin(logcurvecolor),
                                vmax=np.nanmax(logcurvecolor),
                                cmap=cmap)
        else:
            scatax = ax.scatter(logcurve1, logcurve2, marker='o',
                                edgecolors='none', alpha=0.7,
                                cmap=cmap, **kwargs)
        ax.set_xlabel(curve1), ax.set_ylabel(curve2)
        ax.set_title(title if title is None else '{} - {}'.format(curve1,
                                                                  curve2),
                     weight='bold')

        # add colorbar
        if cbar:
            cbar = plt.colorbar(scatax, ax=ax)
            if curvecolor is not None:
                cbar.ax.set_ylabel(curvecolor, rotation=270)
            if cbarlabels:
                scatax.set_clim(vmin=np.nanmin(logcurvecolor) - 0.5,
                                vmax=np.nanmax(logcurvecolor) + 0.5)
                cbar.set_ticks(np.arange(np.nanmin(logcurvecolor),
                                         np.nanmax(logcurvecolor)+1))
                cbar.set_ticklabels(cbarlabels)
        else:
            cbar = None

        if savefig is not None:
            fig.savefig(savefig, dpi=300)
        return fig, ax, scatax, cbar


    def view_logtrack(self, template='petro', lfp=False,
                      depth='MD', twtcurve=None,
                      cali='CALI', gr='GR', rt='RT', vsh='VSH',
                      vcarb='VCARB', vcoal='VCOAL',
                      sgt='SGT', sot='SOT', phi='PHIT',
                      vp='VP', vs='VS', rho='RHOB',
                      ai='AI', vpvs='VPVS', theta=np.arange(0, 40, 5),
                      seismic=None, wav=None, seissampling=1., seisshift=0.,
                      seisreverse=False, trace_in_seismic=False,
                      whichseismic='il', extendseismic=20, thetasub=1,
                      prestack_wiggles=True, horizonset=None,
                      intervals=None, facies=None,
                      faciesfromlog=False, scenario4d=None,
                      title=None, **kwargs_logs):
        """Display log track using any of the provided standard templates
        tailored to different disciplines and analysis

        Parameters
        ----------
        template : :obj:`str`, optional
            Template (``petro``: petrophysical analysis,
            ``rock``: rock-physics analysis,
            ``faciesclass``: facies-classification analysis,
            ``poststackmod``: seismic poststack modelling (with normal
            and averaged properties),
            ``prestackmod``: seismic prestack modelling (with normal
            and averaged properties),
            ``seismic``: 3D seismic intepretation,
            ``prestackseismic``: prestack/AVO seismic intepretation,
            ``4Dmod``: time-lapse seismic modelling)
        lfp : :obj:`bool`, optional
            Prepend `LFP_`` to every log (``True``) or not (``False``)
        depth : :obj:`str`, optional
            Name of depth log curve
        twtcurve : :obj:`str`, optional
            Name of TWT curve to used for y-axis when ``depth=TWT``
        cali : :obj:`str`, optional
            Name of Caliper log
        gr : :obj:`str`, optional
            Name of Gamma Ray log
        rt : :obj:`str`, optional
            Name of Resistivity log
        vsh : :obj:`str`, optional
            Name of gamma ray log
        vcarb : :obj:`str`, optional
            Name of Volume Carbonate log
        vcoal : :obj:`str`, optional
            Name of Volume Coal log
        sgt : :obj:`str`, optional
            Name of Total Gas Saturation Ray log
        sot : :obj:`str`, optional
            Name of Total Oil Saturation log
        phi : :obj:`str`, optional
            Name of Porosity log
        vp : :obj:`str`, optional
            Name of P-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        vs : :obj:`str`, optional
            Name of S-wave velocity log
        rho : :obj:`str`, optional
            Name of Density log
        ai : :obj:`str`, optional
            Name of Acoustic Impedence log
        vpvs : :obj:`str`, optional
            Name of VP/VS log
        theta : :obj:`np.ndarray`
            Angles in degrees (required for prestack modelling)
        seismic : :obj:`ptcpy.object.Seismic` or :obj:`ptcpy.object.SeismicIrregular` or :obj:`ptcpy.object.SeismicIrregular`, optional
            Name of seismic data to visualize when required by template
            (use ``None`` when not required)
        wav : :obj:`np.ndarray`, optional
            Wavelet to apply to synthetic seismic when required by template
        seissampling : :obj:`float`, optional
            Sampling along depth/time axis for seismic
        seisshift : :obj:`float`, optional
            Shift to apply to real seismic trace. If positive, shift downward,
            if negative shift upward (only available in ``template=seismic``
            or ``template=prestackseismic`` when ``trace_in_seismic=False``)
        seisreverse : :obj:`bool`, optional
            Reverse colors of seismic wavelet filling
        trace_in_seismic : :obj:`bool`, optional
            Display synthetic trace on top of real seismic (``True``) or
            side-by-side with extractec seismic trace (``False``)
        whichseismic : :obj:`str`, optional
            ``il``: display inline section passing through well,
            ``xl``: display crossline section passing through well.
            Note that if well is not vertical an arbitrary path along the well
            trajectory will be chosen
        extendseismic : :obj:`int`, optional
            Number of ilines and crosslines to add at the end of well toe when
            visualizing a deviated well
        thetasub : :obj:`int`, optional
            Susampling factor for angle axis if ``template='prestackseismic'``
            or ``template='prestackmod'``
        prestack_wiggles : :obj:`bool`, optional
            Use wiggles to display pre-stack seismic (``True``) or imshow
            (``False``)
        horizonset : :obj:`dict`, optional
            Horizon set to display if ``template='seismic'``
        intervals : :obj:`int`, optional
            level of intervals to be shown (if ``None``, intervals are not shown)
        facies : :obj:`dict`, optional
            Facies set
        faciesfromlog : :obj:`str`, optional
            Name of log curve with facies (if ``None`` estimate from ``facies``
            definition directly)
        scenario4d : :obj:`str`, optional
            Name of scenario to be used as suffix to select fluid substituted
            well logs for ``template='4D'``
        kwargs_logs : :obj:`dict`, optional
            additional input parameters to be provided to
            :func:`ptcpy.objects.Logs.visualize_logcurves`

        Returns
        -------
        fig : :obj:`plt.figure`
           Figure handle (``None`` if ``axs`` are passed by user)
        axs : :obj:`plt.axes`
           Axes handles

        """
        # prepend lfp if lfp flag is True
        if lfp:
            cali = 'LFP_'+cali if lfp else cali
            gr = 'LFP_'+gr if lfp else gr
            rt = 'LFP_'+rt if lfp else rt
            vsh = 'LFP_'+vsh if lfp else vsh
            vcarb = 'LFP_'+vcarb if lfp else vcarb
            vcoal = 'LFP_'+vcoal if lfp else vcoal
            sgt = 'LFP_'+sgt if lfp else sgt
            sot = 'LFP_'+sot if lfp else sot
            phi = 'LFP_'+phi if lfp else phi
            vp = 'LFP_'+vp if lfp else vp
            vs = 'LFP_'+vs if lfp else vs
            rho = 'LFP_'+rho if lfp else rho
            ai = 'LFP_'+ai if lfp else ai
            vpvs = 'LFP_'+vpvs if lfp else vpvs

        # define depth for logs
        depthlog = depth + ' - ' + twtcurve if twtcurve is not None else depth

        # plotting
        if template == 'petro':
            fig, axs = \
                self.visualize_logcurves(
                    dict(CALI=dict(logs=[cali],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.logs[cali]),
                                         np.nanmax(self.logs[cali]))),
                         GR=dict(logs=[gr],
                                 colors=['k'],
                                 xlim=(0, np.nanmax(self.logs[gr]))),
                         RT=dict(logs=[rt],
                                 colors=['k'],
                                 logscale=True,
                                 xlim=(np.nanmin(self.logs[rt]),
                                       np.nanmax(self.logs[rt]))),
                         RHOB=dict(logs=[rho],
                                   colors=['k'],
                                   xlim=((np.nanmin(self.logs[rho]),
                                          np.nanmax(self.logs[rho])))),
                         PHIT=dict(logs=[phi],
                                   colors=['k'],
                                   xlim=(0, 0.4)),
                         Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green', 'blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4))),
                    depth=depthlog, **kwargs_logs)

        elif template == 'rock':
            fig, axs = \
                self.visualize_logcurves(
                    dict(Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green','blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vp]),
                                       np.nanmax(self.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vs]),
                                       np.nanmax(self.logs[vs]))),
                         RHO=dict(logs=[rho],
                                  colors=['k'],
                                  xlim=(np.nanmin(self.logs[rho]),
                                        np.nanmax(self.logs[rho]))),
                         AI=dict(logs=[ai],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[ai]),
                                       np.nanmax(self.logs[ai]))),
                         VPVS=dict(logs=[vpvs],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.logs[vpvs]),
                                         np.nanmax(self.logs[vpvs])))),
                    depth=depthlog, **kwargs_logs)

        elif template == 'faciesclass':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 9, sharey=True, figsize=figsize)
            _, axs = \
                self.visualize_logcurves(
                    dict(GR=dict(logs=[gr],
                                 colors=['k'],
                                 xlim=(0, np.nanmax(self.logs[gr]))),
                         RT=dict(logs=[rt],
                                 colors=['k'],
                                 logscale=True,
                                 xlim=(np.nanmin(self.logs[rt]),
                                       np.nanmax(self.logs[rt]))),
                         RHOB=dict(logs=[rho],
                                   colors=['k'],
                                   xlim=((np.nanmin(self.logs[rho]),
                                          np.nanmax(self.logs[rho])))),
                         PHIT=dict(logs=[phi],
                                   colors=['k'],
                                   xlim=(0, 0.4)),
                         VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vp]),
                                       np.nanmax(self.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vs]),
                                       np.nanmax(self.logs[vs]))),
                         Volume = dict(logs=[vsh, vcarb, vcoal],
                                       colors=['green', '#94b8b8',
                                               '#4d4d4d', 'yellow'],
                                       xlim=(0, 1)),
                         Sat = dict(logs=[sgt, sot],
                                    colors=['red', 'green', 'blue'],
                                    envelope=phi,
                                    xlim=(0, 0.4))),
                    depth=depthlog, axs=axs, **kwargs_logs)

        elif template == 'poststackmod':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.visualize_logcurves(
                    dict(VP=dict(logs=[vp, vp + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[vp]),
                                       np.nanmax(self.logs[vp]))),
                         VS=dict(logs=[vs, vs + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[vs]),
                                       np.nanmax(self.logs[vs]))),
                         RHO=dict(logs=[rho, rho + '_mean'],
                                  colors=['k', '#8c8c8c'],
                                  lw=[2, 8],
                                  xlim=(np.nanmin(self.logs[rho]),
                                        np.nanmax(self.logs[rho]))),
                         AI=dict(logs=[ai, ai + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[ai]),
                                       np.nanmax(self.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_mean'],
                                   colors=['k', '#8c8c8c'],
                                   lw=[2, 8],
                                   xlim=(np.nanmin(self.logs[vpvs]),
                                         np.nanmax(self.logs[vpvs]))),
                         Stack=dict(log=ai, sampling=1., wav=wav, title='Modelled Seismic'),
                         Stack1=dict(log=ai + '_mean', sampling=seissampling,
                                     wav=wav, title='Modelled from blocky logs')),
                    depth=depthlog, seisreverse=seisreverse, axs=axs, **kwargs_logs)

        elif template == 'prestackmod':
            figsize = None if 'figsize' not in kwargs_logs.keys() \
                else kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.visualize_logcurves(
                    dict(VP=dict(logs=[vp, vp + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[vp]),
                                       np.nanmax(self.logs[vp]))),
                         VS=dict(logs=[vs, vs + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[vs]),
                                       np.nanmax(self.logs[vs]))),
                         RHO=dict(logs=[rho, rho + '_mean'],
                                  colors=['k', '#8c8c8c'],
                                  lw=[2, 8],
                                  xlim=(np.nanmin(self.logs[rho]),
                                        np.nanmax(self.logs[rho]))),
                         AI=dict(logs=[ai, ai + '_mean'],
                                 colors=['k', '#8c8c8c'],
                                 lw=[2, 8],
                                 xlim=(np.nanmin(self.logs[ai]),
                                       np.nanmax(self.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_mean'],
                                   colors=['k', '#8c8c8c'],
                                   lw=[2, 8],
                                   xlim=(np.nanmin(self.logs[vpvs]),
                                         np.nanmax(self.logs[vpvs]))),
                         Prestack=dict(theta=theta,
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=4),
                         Prestack1=dict(theta=theta,
                                        vp=vp + '_mean',
                                        vs=vs + '_mean',
                                        rho=rho + '_mean',
                                        sampling=seissampling,
                                        wav=wav,
                                        scaling=4)),
                    depth=depthlog, seisreverse=seisreverse,
                    prestack_wiggles=prestack_wiggles, axs=axs, **kwargs_logs)

        elif template == 'seismic':
            if not self.vertical:
                raise NotImplementedError('Cannot use template=seismic for non'
                                          'vertical wells')
            if seismic is None or wav is None:
                raise ValueError('Provide a seismic data and a wavelet when '
                                 'visualizing logs with seismic template')

            if trace_in_seismic and depth=='MD':
                trace_in_seismic=False
                logging.warning('Cannot view trace on seismic with depth=MD')

            figsize = None if 'figsize' not in kwargs_logs.keys() else \
                kwargs_logs['figsize']
            if trace_in_seismic:
                fig = plt.figure(figsize=figsize)
                axs = [plt.subplot2grid((1, 7), (0, i)) for i in range(6)]
                axs.append(plt.subplot2grid((1, 7), (0, 5), colspan=2))
            else:
                fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)

            logcurves_display = dict(VP=dict(logs=[vp],
                                             colors=['k'],
                                             xlim=(np.nanmin(self.logs[vp]),
                                                   np.nanmax(self.logs[vp]))),
                                     VS=dict(logs=[vs],
                                             colors=['k'],
                                             xlim=(np.nanmin(self.logs[vs]),
                                                   np.nanmax(self.logs[vs]))),
                                     RHO=dict(logs=[rho],
                                              colors=['k'],
                                              xlim=(np.nanmin(self.logs[rho]),
                                                    np.nanmax(self.logs[rho]))),
                                     AI=dict(logs=[ai],
                                         colors=['k'],
                                         xlim=(np.nanmin(self.logs[ai]),
                                               np.nanmax(self.logs[ai]))),
                                     VPVS=dict(logs=[vpvs],
                                               colors=['k'],
                                               xlim=(np.nanmin(self.logs[vpvs]),
                                                     np.nanmax(self.logs[vpvs]))))
            if not trace_in_seismic:
                logcurves_display['Stack'] = dict(log=ai,
                                                  sampling=seissampling,
                                                  wav=wav,
                                                  title='Modelled Seismic')
            _, axs = \
                self.visualize_logcurves(logcurves_display,
                                                  depth=depthlog,
                                                  axs=axs, seisreverse=seisreverse,
                                                  **kwargs_logs)

            # add real seismic trace
            realtrace = seismic['data'].extract_trace_verticalwell(self)

            if not trace_in_seismic:
                axs[-1] = _wiggletrace(axs[-1],
                                       seismic['data'].tz + seisshift,
                                       realtrace)
                axs[-1].set_xlim(axs[-2].get_xlim())
                axs[-1].set_title('Real Seismic (shift={})'.format(seisshift),
                                  fontsize=12)
            else:
                # find well in il-xl
                ilwell, xlwell = \
                    _findclosest_well_seismicsections(self, seismic['data'],
                                                      traj=False)

                if 'ylim' in kwargs_logs.keys() and \
                        kwargs_logs['ylim'] is not None:
                    axs[-1].set_ylim(kwargs_logs['ylim'])
                axs[-1].set_title('Real Seismic (shift={})'.format(seisshift),
                                  fontsize=12)
                axs[-1].invert_yaxis()

                # find out from first plot and set ylim for all plots
                if 'ylim' in kwargs_logs.keys():
                    zlim_seismic = kwargs_logs['ylim']
                else:
                    zlim_seismic = axs[0].get_ylim()
                for i in range(1, len(axs) - 1):
                    axs[i].set_ylim(zlim_seismic)
                    axs[i].invert_yaxis()
                    axs[i].set_yticks([])
                axs[-1].set_yticks([])

                if horizonset is None:
                    dictseis = {}
                else:
                    dictseis = dict(horizons=horizonset['data'],
                                    horcolors=horizonset['colors'],
                                    horlw=5)
                _, axs[-1] = \
                    self.view_in_seismicsection(seismic['data'], ax=axs[-1],
                                                which=whichseismic,
                                                display_wellname=False,
                                                picks=False,
                                                tzoom_index=False,
                                                tzoom=zlim_seismic,
                                                tzshift=seisshift,
                                                cmap='seismic',
                                                clip=1.,
                                                cbar=True,
                                                interp='sinc',
                                                title='Real Seismic',
                                                **dictseis)
                if whichseismic == 'il':
                    axs[-1].set_xlim(xlwell-extendseismic, xlwell+extendseismic)
                else:
                    axs[-1].set_xlim(ilwell-extendseismic, ilwell+extendseismic)

                trace, zaxisreglog = \
                    zerooffset_wellmod(self, depthlog,
                                       seissampling, wav,
                                       ai=ai, zlim=depthlog,
                                       ax=axs[-1])[:2]
                trace_center = xlwell if whichseismic == 'il' else ilwell
                _wiggletrace(axs[-1], zaxisreglog,
                             trace_center + (extendseismic/(4*np.nanmax(trace)))*trace,
                             center=trace_center)

        elif template == 'prestackseismic':
            if seismic is None or wav is None:
                raise ValueError('Provide a prestack seismic data and a wavelet '
                                 'when visualizing logs with seismic template')
            figsize = None if 'figsize' not in kwargs_logs.keys() else \
                kwargs_logs['figsize']
            fig, axs = plt.subplots(1, 7, sharey=True, figsize=figsize)
            _, axs = \
                self.visualize_logcurves(
                    dict(VP=dict(logs=[vp],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vp]),
                                       np.nanmax(self.logs[vp]))),
                         VS=dict(logs=[vs],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[vs]),
                                       np.nanmax(self.logs[vs]))),
                         RHO=dict(logs=[rho],
                                  colors=['k'],
                                  xlim=(np.nanmin(self.logs[rho]),
                                        np.nanmax(self.logs[rho]))),
                         AI=dict(logs=[ai],
                                 colors=['k'],
                                 xlim=(np.nanmin(self.logs[ai]),
                                       np.nanmax(self.logs[ai]))),
                         VPVS=dict(logs=[vpvs],
                                   colors=['k'],
                                   xlim=(np.nanmin(self.logs[vpvs]),
                                         np.nanmax(self.logs[vpvs]))),
                         Prestack=dict(theta=theta[::thetasub],
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=4)),
                    depth=depthlog, seisreverse=seisreverse, axs=axs, **kwargs_logs)

            # add real prestack seismic trace
            realgather = \
                seismic['data'].extract_gather_verticalwell(self, verb=True)
            realgather = realgather[::thetasub]
            axs[-1] = _wiggletracecomb(axs[-1], seismic['data'].tz + seisshift,
                                       theta[::thetasub], realgather,
                                       scaling=20)
            if 'ylim' in kwargs_logs.keys() and kwargs_logs['ylim'] is not None:
                axs[-1].set_ylim(kwargs_logs['ylim'])
            axs[-1].set_title('Real Seismic')
            axs[-1].invert_yaxis()

        elif template == '4Dmod':
            fig, axs = \
                self.visualize_logcurves(
                    dict(Volume=dict(logs=[vsh, vcarb, vcoal],
                                     colors=['green', '#94b8b8',
                                             '#4d4d4d', 'yellow'],
                                     xlim=(0, 1)),
                         Sat=dict(logs=[sgt, sot],
                                  colors=['red', 'green','blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         Sat1=dict(logs=[sgt+'_'+scenario4d, sot+'_'+scenario4d],
                                  colors=['red', 'green', 'blue'],
                                  envelope=phi,
                                  xlim=(0, 0.4)),
                         AI=dict(logs=[ai, ai+'_'+scenario4d],
                                 colors=['k', 'r'],
                                 lw=[1, 1],
                                 xlim=(np.nanmin(self.logs[ai]),
                                       np.nanmax(self.logs[ai]))),
                         VPVS=dict(logs=[vpvs, vpvs + '_' + scenario4d],
                                   colors=['k', 'r'],
                                   lw=[1, 1],
                                 xlim=(np.nanmin(self.logs[vpvs]),
                                       np.nanmax(self.logs[vpvs]))),
                         dAI=dict(logs=[ai+'diff_'+scenario4d],
                                  colors=['k'],
                                  xlim=(-50, 50)),
                         dVPVS=dict(logs=[vpvs+'diff_'+scenario4d],
                                    colors=['k'],
                                    xlim=(-50, 50)),
                         Stack=dict(log=ai,
                                    sampling=seissampling,
                                    wav=wav),
                         Diff=dict(logs=[ai+'_'+scenario4d, ai],
                                   sampling=1.,
                                   wav=wav),
                         Prestack=dict(theta=theta,
                                       vp=vp,
                                       vs=vs,
                                       rho=rho,
                                       sampling=seissampling,
                                       wav=wav,
                                       scaling=1.),
                         Prediff=dict(theta=theta,
                                      vp=[vp+'_'+scenario4d, vp],
                                      vs=[vs+'_'+scenario4d, vs],
                                      rho=[rho+'_'+scenario4d, rho],
                                      sampling=seissampling,
                                      wav=wav,
                                      scaling=1.)),
                    depth=depthlog,  seisreverse=seisreverse, ** kwargs_logs)
            xlims = np.array([-np.max(axs[-4].get_xlim()),
                              np.max(axs[-4].get_xlim())])
            axs[-4].set_xlim(xlims)
            axs[-3].set_xlim(xlims)

        else:
            raise ValueError('template={} does not exist'.format(template))

        if template == 'faciesclass':
            xlim_facies = axs[-1].get_xlim()
            faciesnames = list(facies.keys())
            faciescolors = [facies[faciesname].color for faciesname in
                            facies.keys()]

            if faciesfromlog:
                axs[-1] = _visualize_facies(axs[-1], self,
                                            faciesfromlog,
                                            faciescolors,
                                            faciesnames,
                                            depth=depth)

                axs[-1].set_xlim(xlim_facies)
        return fig, axs

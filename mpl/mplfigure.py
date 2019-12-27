# -*- coding: utf-8 -*-

""" Figures for matplotlib

"""

import re
import numpy as np
from numpy import ma
import pandas as pd

import matplotlib
from matplotlib import dates as mpldates
from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from ..utils import *


def _get_colormap(cmap):
    if isinstance(cmap, list) and len(cmap) == 1:
        cmap = cmap[0]
    return cmap


def get_point_size(s, dpi):
    return s*72/dpi


def create_colorbar_axes(axes, size=0.1, sep=0.2):
    axes.apply_aspect() # do this just in case
    axpos = axes.get_position()

    l = axpos.x0 + axpos.width + sep
    b = axpos.y0
    w = size
    h = axpos.height
    cbax = plt.axes([l, b, w, h])

    return cbax


class BaseFigure(object):
    def __init__(self, data, figure, axes, **options):
        self.data   = data
        self.figure = figure
        self.axes   = axes
        self.setup_options(options)
        self.setup_default_axes()

    def setup_options(self, options):
        opt_pixel_to_point = [
            'width',
            'height',
            'linewidth',
            'fontsize',
            'labelsize',
            'ticklength',
            'tickwidth',
            'tickpad',
            'colorbar_sep',
            'colorbar_size',
        ]
        self.opt = options.copy()
        # convert size in pixel to point
        if 'dpi' in self.opt:
            point = lambda s: get_point_size(s, self.opt['dpi'])
            for v in opt_pixel_to_point:
                if v in self.opt:
                    self.opt[v] = point(self.opt[v])

    def setup_default_axes(self):
        # tick options
        opt = {
            'labelsize' : self.opt['labelsize'],
            'pad'       : self.opt['tickpad'],
            'length'    : self.opt['ticklength'],
            'width'     : self.opt['tickwidth'],
            'top'       : True,
            'bottom'    : True,
            'left'      : True,
            'right'     : True,
        }
        self.axes.xaxis.set_tick_params(**opt)
        self.axes.yaxis.set_tick_params(**opt)
        for axis in ['top','bottom','left','right']:
            self.axes.spines[axis].set_linewidth(self.opt['linewidth'])

    def buildfigure(self):
        pass


class FigureLine(BaseFigure):
    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_option(self.data, key, val)
        data = self.data

        t = data.time.values
        x = pd_to_datetime(t)

        # ensure 2D array
        if data.values.ndim == 1:
            y = data.values[:,np.newaxis]
        elif data.values.ndim == 2:
            y = data.values
        else:
            print('Error: input must be either 1D or 2D array')
            return None

        legend_names = get_opt('legend')
        N = y.shape[1]
        for i in range(N):
            # line options
            opt = {
                'linewidth' : self.opt['linewidth'],
                'linestyle' : 'solid',
            }
            lc = get_opt('line_color')
            if lc is not None and len(lc) == N:
                opt['color'] = lc[i]
            else:
                opt['color'] = 'k'
            # legend
            if legend_names is not None:
                opt['label'] = legend_names[i]
            # plot
            plot = self.axes.plot(x, y[:,i], **opt)

        # update axes
        self.axes.set_ylabel(get_opt('ylabel', ''),
                             fontsize=self.opt['fontsize'])
        if get_opt('trange', None) is not None:
            self.axes.set_xlim(pd_to_datetime(get_opt('trange')))

        # legend
        if legend_names is not None:
            legend_opt = {
                'bbox_to_anchor' : (1.01, 1.0),
                'loc'            : 'upper left',
                'borderaxespad'  : 0,
                'frameon'        : False,
                'fontsize'       : self.opt['fontsize'],
            }
            self.axes.legend(**legend_opt)


class FigureSpec(BaseFigure):
    def setup_default_axes(self):
        # tick options
        opt = {
            'labelsize' : self.opt['labelsize'],
            'pad'       : self.opt['tickpad'],
            'length'    : self.opt['ticklength'],
            'width'     : self.opt['tickwidth'],
            'top'       : True,
            'bottom'    : True,
            'left'      : True,
            'right'     : True,
        }
        self.axes.xaxis.set_tick_params(**opt)
        self.axes.yaxis.set_tick_params(**opt)
        # colorbar size and sep should be fraction unit
        cbsep  = self.opt['colorbar_sep']  / self.opt['width']
        cbsize = self.opt['colorbar_size'] / self.opt['width']
        self.cbax = create_colorbar_axes(self.axes, size=cbsize, sep=cbsep)
        self.cbax.xaxis.set_tick_params(**opt)
        self.cbax.yaxis.set_tick_params(**opt)
        for ax in (self.axes, self.cbax):
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(self.opt['linewidth'])

    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_option(self.data, key, val)
        data = self.data

        t = data.time.values
        x = pd_to_datetime(t)
        y = data.coords['spec_bins'].values
        z = data.values

        if get_opt('ytype', 'linear') == 'log':
            ylog = True
            self.set_log_ticks(self.axes.yaxis)
        else:
            ylog = False
        zz, opt = interpolate_spectrogram(y, z, ylog=ylog)
        y0 = opt['y0']
        y1 = opt['y1']

        if get_opt('ztype', 'linear') == 'log':
            cond = np.logical_or(np.isnan(zz), np.less_equal(zz, 0.0))
            zz = np.log10(ma.masked_where(cond, zz))

        # colormap and range
        zmin, zmax = get_opt('zrange', [None, None])
        cmap = _get_colormap(get_opt('colormap'))

        zmin = np.floor(zz.min() if zmin is None else zmin)
        zmax = np.ceil (zz.max() if zmax is None else zmax)
        norm = matplotlib.colors.Normalize(vmin=zmin, vmax=zmax)

        # plot
        x0  = mpldates.date2num(x[ 0])
        x1  = mpldates.date2num(x[-1])
        ext = [x0, x1, np.log10(y0), np.log10(y1)]
        opt_imshow = {
            'origin'     : 'lower',
            'aspect'     : 'auto',
            'extent'     : ext,
            'norm'       : norm,
            'cmap'       : cmap,
            'rasterized' : True,
        }
        im = self.axes.imshow(zz.T, **opt_imshow)
        self.axes.set_xlim(ext[0], ext[1])
        self.axes.set_ylim(ext[2], ext[3])
        self.axes.xaxis_date()

        # update axes
        self.axes.set_ylabel(get_opt('ylabel', ''),
                             fontsize=self.opt['fontsize'])
        if get_opt('trange', None) is not None:
            self.axes.set_xlim(pd_to_datetime(get_opt('trange')))

        # colorbar
        cb = plt.colorbar(im, cax=self.cbax, drawedges=False)
        cb.outline.set_linewidth(self.opt['linewidth'])
        self.cbax.set_ylabel(get_opt('zlabel', ''),
                             fontsize=self.opt['fontsize'])

    def set_log_ticks(self, axis, dec=1):
        f  = lambda x, p: r'$\mathregular{10^{%d}}$' % (x)
        majorloc  = matplotlib.ticker.MultipleLocator(dec)
        formatter = matplotlib.ticker.FuncFormatter(f)
        axis.set_major_locator(majorloc)
        axis.set_major_formatter(formatter)


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

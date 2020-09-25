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


def _log_formatter(x, p):
    d = int(np.rint(np.log10(x)))
    return r'$\mathregular{10^{%+d}}$' % (d)


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


class DateFormatter(mpldates.AutoDateFormatter):
    def __init__(self, locator):
        super(DateFormatter, self).__init__(locator)

        # set datetime axis
        fmt = DateFormatter.formatter
        self.scaled = {
            365.0                 : '%Y',
            30.0                  : '%Y-%m',
            1.0                   : '%Y-%m-%d',
            1. / (24)             : fmt('%Y-%m-%d %H:%M', '%m-%d %H:%M'),
            1. / (24*60)          : fmt('%m-%d %H:%M', '%H:%M'),
            1. / (24*60*60)       : fmt('%H:%M:%S', '%M:%S'),
            1. / (24*60*60*1e+3)  : fmt('%H:%M:%S.%f', '%M:%S.%f', True),
        }

    @staticmethod
    def formatter(fmt1, fmt2, strip=False):
        def f(x, pos):
            x = mpldates.num2date(x)
            if pos == 0:
                r = x.strftime(fmt1)
            else:
                r = x.strftime(fmt2)
            if strip:
                r = r[:-3]
            return r

        return matplotlib.ticker.FuncFormatter(f)


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
            'line_width',
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
        self.opt['xtime']   = self.opt.get('xtime', True)
        self.opt['numplot'] = self.opt.get('numplot', 0)
        self.opt['numaxes'] = self.opt.get('numaxes', 0)

    def set_xdate(self):
        if not self.opt['xtime']:
            return

        def fmt(x, pos=None):
            x = mpldates.num2date(x)
            if pos == 1:
                fmt = '%H:%M:%S.%f'
            else:
                fmt = '%M:%S.%f'
            return x.strftime(fmt)[:-3]

        # set datetime axis
        majorloc = mpldates.AutoDateLocator()
        majorfmt = DateFormatter(majorloc)
        self.axes.xaxis.set_major_locator(majorloc)
        self.axes.xaxis.set_major_formatter(majorfmt)

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
        self.axes.tick_params(axis='x', which='major', **opt)
        self.axes.tick_params(axis='y', which='major', **opt)
        self.axes.tick_params(axis='x', which='minor', length=0, width=0)
        self.axes.tick_params(axis='y', which='minor', length=0, width=0)
        for axis in ['top','bottom','left','right']:
            self.axes.spines[axis].set_linewidth(self.opt['line_width'])
        self.set_xdate()

    def get_opt(self, key, val=None):
        return get_plot_option(self.data, key, val)

    def buildfigure(self):
        pass

    def update_axes(self):
        pass


class FigureLine(BaseFigure):
    def buildfigure(self):
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

        legend_names = self.get_opt('legend')
        N = y.shape[1]
        for i in range(N):
            # line options
            opt = {
                'linewidth' : self.opt['line_width'],
                'linestyle' : 'solid',
            }
            lc = self.get_opt('line_color')
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
        self.update_axes()

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

    def update_axes(self):
        if self.opt['numplot'] != 0:
            return

        if self.get_opt('trange', None) is not None:
            self.axes.set_xlim(pd_to_datetime(self.get_opt('trange')))

        if self.get_opt('yrange', None) is not None:
            self.axes.set_ylim(self.get_opt('yrange'))
        self.axes.set_ylabel(self.get_opt('ylabel', ''),
                             fontsize=self.opt['fontsize'])


class FigureSpec(BaseFigure):
    def create_background_axes(self, ax):
        bg = ax.twinx()
        for sp in bg.spines.values():
            sp.set_visible(False)
        bg.yaxis.set_ticks([])
        ax.set_zorder(bg.get_zorder()+1)
        ax.patch.set_visible(False)
        return bg

    def setup_default_axes(self):
        # tick options
        opt = {
            'labelsize' : self.opt['labelsize'],
            'pad'       : self.opt['tickpad'],
            'length'    : self.opt['ticklength'],
            'width'     : self.opt['tickwidth'],
        }

        # primary axes
        self.axes.xaxis.set_tick_params(top=True, bottom=True, **opt)
        self.axes.yaxis.set_tick_params(left=True, right=True, **opt)

        # colorbar axes; size and sep should be in fraction unit
        cbsep  = self.opt['colorbar_sep']  / self.opt['width']
        cbsize = self.opt['colorbar_size'] / self.opt['width']
        self.cbax = create_colorbar_axes(self.axes, size=cbsize, sep=cbsep)
        self.cbax.xaxis.set_tick_params(top=False, bottom=False, **opt)
        self.cbax.yaxis.set_tick_params(left=True, right=True, **opt)
        for ax in (self.axes, self.cbax):
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(self.opt['line_width'])
        self.set_xdate()

        # background axes
        self.axes_bg = self.create_background_axes(self.axes)
        self.axes_bg.xaxis.set_tick_params(top=False, bottom=False, **opt)
        self.axes_bg.yaxis.set_tick_params(left=True, right=True, **opt)
        self.cbax_bg = self.create_background_axes(self.cbax)
        self.cbax_bg.xaxis.set_tick_params(top=True, bottom=True, **opt)
        self.cbax_bg.yaxis.set_tick_params(left=True, right=True, **opt)
        self.cbax_bg.xaxis.set_ticks([])

        # colorbar tick and labels
        self.cbax_bg.yaxis.tick_left()
        self.cbax.yaxis.tick_right()
        self.cbax.yaxis.set_label_position('right')

    def buildfigure(self):
        data = self.data

        t = data.time.values
        x = pd_to_datetime(t)
        y = data.coords['spec_bins'].values
        z = data.values
        ylog = self.get_opt('ytype', 'linear') == 'log'
        zlog = self.get_opt('ztype', 'linear') == 'log'

        # colormap and range
        cmap = _get_colormap(self.get_opt('colormap'))
        zmin, zmax = self.get_opt('zrange', [None, None])

        # rasterized spectrogram
        kwargs = {
            'ylog' : ylog,
            'zlog' : zlog,
            'zmin' : zmin,
            'zmax' : zmax,
            'cmap' : cmap,
        }
        im_spectrogram, opt = get_raster_spectrogram(y, z, **kwargs)
        y0 = opt['y0']
        y1 = opt['y1']
        z0 = opt['zmin']
        z1 = opt['zmax']

        # plot
        x0 = mpldates.date2num(x[ 0] - 0.5*(x[+1] - x[ 0]))
        x1 = mpldates.date2num(x[-1] + 0.5*(x[-1] - x[-2]))
        opt_imshow = {
            'aspect' : 'auto',
            'extent' : [x0, x1, 0, 1],
        }
        # TODO: resizing may be needed depending on resolution
        im = self.axes_bg.imshow(np.asarray(im_spectrogram), **opt_imshow)

        # update axes
        self.xlim = [x0, x1]
        self.ylim = [y0, y1]
        self.zlim = [z0, z1]
        self.update_axes()

        # colorbar
        im_colorbar = get_raster_colorbar(cmap=cmap)
        opt_imshow = {
            'aspect' : 'auto',
            'extent' : [0, 1, 0, 1],
        }
        im = self.cbax_bg.imshow(np.asarray(im_colorbar), **opt_imshow)

        self.cbax.set_ylabel(self.get_opt('zlabel', ''),
                             fontsize=self.opt['fontsize'])
        self.set_colorbar_ticks()

    def update_axes(self):
        if self.opt['numplot'] != 0:
            return

        if self.get_opt('trange', None) is not None:
            self.axes.set_xlim(pd_to_datetime(self.get_opt('trange')))
            self.axes.xaxis_date()

        if self.get_opt('yrange', None) is not None:
            yrange = self.get_opt('yrange')
            self.axes.set_ylim(yrange)
        else:
            self.axes.set_ylim(self.ylim)

        if self.get_opt('ytype', 'linear') == 'log':
            self.axes.set_yscale('log')
            self.set_yticks(ylog=True)
        else:
            self.set_yticks(ylog=False)

        self.axes.set_ylabel(self.get_opt('ylabel', ''),
                             fontsize=self.opt['fontsize'])

        # TODO: minor ticks
        self.axes.tick_params(axis='x', which='minor', length=0, width=0)
        self.axes.tick_params(axis='y', which='minor', length=0, width=0)
        self.cbax.tick_params(axis='y', which='minor', length=0, width=0)

    def set_yticks(self, ylog=False):
        if ylog:
            self.set_ylog_ticks()
        else:
            self.set_ylinear_ticks()

    def set_ylinear_ticks(self):
        # get ticks for primary axes
        y0 = np.log10(self.ylim[0])
        y1 = np.log10(self.ylim[1])
        loc = self.axes.yaxis.get_major_locator()
        tickvals = loc.tick_values(y0, y1)

        # background axes
        tickvals = (ticks - y0)/(y1 - y0)
        loc_bg = matplotlib.ticker.FixedLocator(ticks)
        fmt_bg = matplotlib.ticker.NullFormatter()
        self.axes_bg.yaxis.set_major_locator(loc_bg)
        self.axes_bg.yaxis.set_major_formatter(fmt_bg)

    def set_ylog_ticks(self):
        y0 = np.log10(self.ylim[0])
        y1 = np.log10(self.ylim[1])
        ymin = np.ceil(y0)
        ymax = np.floor(y1)

        # TODO: need a cleverer way
        ntick = int(np.rint(y1 - y0))

        # primary axes
        tickvals = 10.0**(np.linspace(ymin, ymax, ntick))
        loc = matplotlib.ticker.FixedLocator(tickvals)
        fmt = matplotlib.ticker.FuncFormatter(_log_formatter)
        self.axes.yaxis.set_major_locator(loc)
        self.axes.yaxis.set_major_formatter(fmt)

        # background axes
        tickvals = (np.log10(tickvals) - y0)/(y1 - y0)
        loc_bg = matplotlib.ticker.FixedLocator(tickvals)
        fmt_bg = matplotlib.ticker.NullFormatter()
        self.axes_bg.yaxis.set_major_locator(loc_bg)
        self.axes_bg.yaxis.set_major_formatter(fmt_bg)

    def set_colorbar_ticks(self):
        if self.get_opt('ztype', 'linear') == 'log':
            self.cbax.set_yscale('log')

        if self.get_opt('colorbar_ticks', None) is None:
            # automatically determine ticks
            if self.get_opt('ztype', 'linear') == 'log':
                loc = self.cbax.yaxis.get_major_locator()
                fmt = matplotlib.ticker.FuncFormatter(_log_formatter)
            else:
                loc = matplotlib.ticker.AutoLocator()
                fmt = matplotlib.ticker.ScalarFormatter()
            tickvals = loc.tick_values(self.zlim[0], self.zlim[1])
            ticktext = []
            for v in tickvals:
                ticktext.append(fmt(v))
            self.cbax.yaxis.set_ticks(tickvals)
            self.cbax.yaxis.set_ticklabels(ticktext)
            self.cbax.set_ylim(self.zlim)
        else:
            # ticks are provided
            opt = self.get_opt('colorbar_ticks')
            if 'tickvals' in opt and 'ticktext' in opt:
                tickvals = np.atleast_1d(opt['tickvals'])
                ticktext = np.atleast_1d(opt['ticktext'])
                # check
                if tickvals.shape == ticktext.shape and tickvals.ndim == 1:
                    self.cbax.yaxis.set_ticks(tickvals)
                    self.cbax.yaxis.set_ticklabels(ticktext)
                    self.cbax.set_ylim(self.zlim)
                else:
                    print('Error: tickvals and ticktext are not consistent')
            else:
                print('Error: tickvals or ticktext are not given')


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

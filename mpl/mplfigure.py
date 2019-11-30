# -*- coding: utf-8 -*-

""" Figures for matplotlib

"""

import re
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from ..utils import get_plot_option


def _convert_color(color):
    colortable = {
        'r' : 'red',
        'g' : 'green',
        'b' : 'blue',
        'c' : 'cyan',
        'y' : 'yellow',
        'm' : 'magenta',
        'w' : 'white',
        'k' : 'black',
    }
    if not isinstance(color, str):
        return color

    if color in colortable:
        return colortable[color]
    elif re.fullmatch(r'^#([0-f][0-f][0-f]){1,2}', color):
        return color
    else:
        raise ValueError('unrecognized color format: %s' % (color))


def _get_colormap(cmap):
    cmaptable = {
        'jet'      : _mpl_jet,
        'bwr'      : _mpl_bwr,
        'seismic'  : _mpl_seismic,
    }
    if isinstance(cmap, list) and len(cmap) == 1:
        cmap = cmap[0]
    if isinstance(cmap, str) and cmap in cmaptable:
        cmap = cmaptable[cmap]
    return cmap


class BaseFigure(object):
    def __init__(self, data, figure, axes, **options):
        self.data   = data
        self.figure = figure
        self.axes   = axes
        self.setup_default_axes()
        self.setup_options(options)

    def setup_options(self, options):
        self.opt = dict()
        self.opt_point = {
            'linewidth'  : 1,
            'fontsize'   : 16,
            'labelsize'  : 16,
            'ticklength' : 6,
            'tickwidth'  : 2,
            'tickpad'    : 2,
        }
        for key in options:
            self.opt[key] = options[key]
        # convert size in pixel to point
        if 'dpi' in self.opt:
            point = lambda s: s*72/self.opt['dpi']
            for v in self.opt_point.keys():
                if v in self.opt:
                    self.opt[v] = point(self.opt[v])
                else:
                    self.opt[v] = point(self.opt_point[v])

    def setup_default_axes(self):
        pass

    def buildfigure(self):
        pass


class FigureLine(BaseFigure):
    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_option(self.data, key, val)
        data = self.data

        x = pd.to_datetime(data.time, unit='s')

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
                opt['color'] = _convert_color(lc[i])
            else:
                opt['color'] = _convert_color('k')
            # legend
            if legend_names is not None:
                opt['label'] = legend_names[i]
            # plot
            plot = plt.plot(x, y[:,i], **opt)
            # tick options
            opt = {
                'labelsize' : self.opt['labelsize'],
                'pad'       : self.opt['tickpad'],
                'length'    : self.opt['ticklength'],
                'width'     : self.opt['tickwidth'],
            }
            plt.tick_params(**opt)

        # update axes
        plt.ylabel(get_opt('ylabel', ''), fontsize=self.opt['fontsize'])

        # legend
        legend_opt = {
            'bbox_to_anchor' : (1.01, 1.0),
            'loc'            : 'upper left',
            'borderaxespad'  : 0,
            'frameon'        : False,
            'fontsize'       : self.opt['fontsize'],
        }
        plt.legend(**legend_opt)


class FigureSpec(BaseFigure):
    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_option(self.data, key, val)
        data = self.data

        x = pd.to_datetime(data.time, unit='s')
        y = data.coords['spec_bins']
        z = data.values.T

        # TODO
        if y.ndim == 2:
            y = y[0]

        # TODO
        if get_opt('ztype', 'linear') == 'log':
            z = np.log10(z)

        print('FigureSpec : buildfigure called')


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

# -*- coding: utf-8 -*-

""" Figures for plotly

"""

import re
import numpy as np
from numpy import ma
import pandas as pd

import plotly.graph_objects as go

from ..utils import *


_mpl_jet = \
[
    [ 0.0000, "rgb(  0,  0,127)"],
    [ 0.1100, "rgb(  0,  0,255)"],
    [ 0.1250, "rgb(  0,  0,254)"],
    [ 0.3400, "rgb(  0,219,255)"],
    [ 0.3500, "rgb(  0,229,246)"],
    [ 0.3750, "rgb( 20,255,226)"],
    [ 0.6400, "rgb(238,255,  8)"],
    [ 0.6500, "rgb(246,245,  0)"],
    [ 0.6600, "rgb(255,236,  0)"],
    [ 0.8900, "rgb(255, 18,  0)"],
    [ 0.9100, "rgb(231,  0,  0)"],
    [ 1.0000, "rgb(127,  0,  0)"],
]
_mpl_bwr = \
[
    [ 0.0000, "rgb(  0,  0,255)"],
    [ 0.5000, "rgb(255,255,255)"],
    [ 1.0000, "rgb(255,  0,  0)"],
]
_mpl_seismic = \
[
    [ 0.0000, "rgb(  0,  0, 76)"],
    [ 0.2500, "rgb(  0,  0,255)"],
    [ 0.5000, "rgb(255,255,255)"],
    [ 0.7500, "rgb(255,  0,  0)"],
    [ 1.0000, "rgb(127,  0,  0)"],
]


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


class Legend(object):
    def __init__(self, label, xpos, ypos, size, opts):
        self.label = label
        self.xpos = xpos
        self.ypos = ypos
        self.size = size
        self.opts = opts
        self._build()

    def _build(self):
        x0 = self.xpos
        y0 = self.ypos
        x1 = self.xpos+0.02
        y1 = self.ypos
        x2 = self.xpos+0.02
        y2 = self.ypos
        self.line = go.layout.Shape(
            type='line', x0=x0, x1=x1, y0=y0, y1=y1, xref='paper', yref='paper',
            **self.opts)
        self.text = go.layout.Annotation(font_size=self.size,
            text=self.label, x=x2, y=y2, xref='paper', yref='paper',
            xanchor='left', yanchor='middle', showarrow=False)

    def get_line(self):
        return self.line

    def get_text(self):
        return self.text


class BaseFigure(object):
    def __init__(self, data, figure, axes, **options):
        self.data   = data
        self.figure = figure
        self.axes   = axes
        self.setup_options(options)
        self.setup_default_axes()

    def setup_options(self, options):
        self.webgl = options.get('use_webgl', False)
        self.opt = options.copy()
        if not 'primary' in self.opt:
            self.opt['primary'] = False

    def setup_default_axes(self):
        xaxis = dict(
            linewidth=self.opt['linewidth'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
        )
        yaxis = dict(
            linewidth=self.opt['linewidth'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
        )
        self.figure.update_xaxes(**xaxis, selector=self.axes['xaxis'])
        self.figure.update_yaxes(**yaxis, selector=self.axes['yaxis'])

    def get_opt(self, key, val=None):
        return get_plot_option(self.data, key, val)

    def buildfigure(self):
        pass

    def update_axes(self):
        pass

class FigureLine(BaseFigure):
    def buildfigure(self):
        data = self.data

        # use WebGL version or not
        if self.webgl:
            scatter = go.Scattergl
        else:
            scatter = go.Scatter

        x = pd_to_datetime(data.time)

        # ensure 2D array
        if data.values.ndim == 1:
            y = data.values[:,np.newaxis]
        elif data.values.ndim == 2:
            y = data.values
        else:
            print('Error: input must be either 1D or 2D array')
            return None

        legend_names = self.get_opt('legend')
        layout = self.figure.layout
        legend = list()
        N = y.shape[1]
        for i in range(N):
            # line options
            lopt = dict(line_width=self.opt['linewidth'])
            lc = self.get_opt('line_color')
            if lc is not None and len(lc) == N:
                lopt['line_color'] = _convert_color(lc[i])
            else:
                lopt['line_color'] = _convert_color('k')
            # legend
            opt = dict(xaxis=self.axes['x'],
                       yaxis=self.axes['y'],
                       showlegend=False)
            if legend_names is not None:
                size = self.opt['fontsize']
                xdom = self.axes['xaxis']['domain']
                ydom = self.axes['yaxis']['domain']
                xpos = xdom[1] + 2*self.opt['ticklength']/self.opt['width']
                ypos = ydom[1] - (i + 0.5)*size/self.opt['width']
                legend.append(Legend(legend_names[i], xpos, ypos, size, lopt))
                opt['name'] = legend_names[i]
            # plot
            opt.update(lopt)
            plot = scatter(x=x, y=y[:,i], mode='lines', **opt)
            self.figure.add_trace(plot)

        # update axes
        self.update_axes()

        # legend
        text = list(layout.annotations)
        line = list(layout.shapes)
        for l in legend:
            text.append(l.get_text())
            line.append(l.get_line())
        self.figure.update_layout(annotations=text, shapes=line)

    def update_axes(self):
        if not self.opt['primary']:
            return

        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict(font)
        if self.get_opt('trange', None) is not None:
            xaxis['range'] = pd_to_datetime(self.get_opt('trange'))
        self.figure.update_xaxes(**xaxis, selector=self.axes['xaxis'])

        yaxis = dict(font, title_text=self.get_opt('ylabel'))
        if self.get_opt('yrange', None) is not None:
            yaxis['range'] = self.get_opt('yrange')
        self.figure.update_yaxes(**yaxis, selector=self.axes['yaxis'])


class FigureSpec(BaseFigure):
    def buildfigure(self):
        data = self.data
        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        t = data.time.values
        y = data.coords['spec_bins'].values
        z = data.values
        ylog = self.get_opt('ytype', 'linear') == 'log'
        zlog = self.get_opt('ztype', 'linear') == 'log'

        zz, opt = interpolate_spectrogram(y, z, ylog=ylog)
        if zlog:
            cond = np.logical_or(np.isnan(zz), np.less_equal(zz, 0.0))
            zz = np.log10(ma.masked_where(cond, zz))

        # colorbar
        layout = self.figure.layout
        xdom = self.axes['xaxis']['domain']
        ydom = self.axes['yaxis']['domain']
        xpos = xdom[1]
        ypos = ydom[0]
        xlen = self.opt['colorbar_size']
        ylen = ydom[1] - ydom[0]
        xpad = self.opt['colorbar_sep']
        ypad = 0

        cb = dict(font,
                  x=xpos, y=ypos, xpad=xpad, ypad=ypad, yanchor='bottom',
                  thickness=xlen, thicknessmode='pixels',
                  len=ylen, lenmode='fraction',
                  outlinewidth=self.opt['linewidth'],
                  title=self.get_opt('zlabel'),
                  titleside='right',
                  ticks='outside')

        # colormap and range
        zmin, zmax = self.get_opt('zrange', [None, None])
        cmap = _get_colormap(self.get_opt('colormap'))

        zmin = np.floor(zz.min() if zmin is None else zmin)
        zmax = np.ceil (zz.max() if zmax is None else zmax)

        # heatmap
        t0 = t[ 0] - 0.5*(t[+1] - t[ 0])
        t1 = t[-1] + 0.5*(t[-1] - t[-2])
        tt = np.linspace(t0, t1, zz.shape[0]+1)
        xx = pd_to_datetime(tt)
        yy = opt['bine']

        opt = dict(name='',
                   xaxis=self.axes['x'],
                   yaxis=self.axes['y'],
                   colorscale=cmap,
                   colorbar=cb,
                   zmin=zmin,
                   zmax=zmax)
        hm = go.Heatmap(z=zz.T.filled(-np.inf), x=xx, y=yy, **opt)
        self.figure.add_trace(hm)

        # update axes
        self.update_axes()

    def set_log_ticks(self, axis, dec=1):
        opt = dict(tickprefix='10<sup>',
                   ticksuffix='</sup>',
                   tickformat='d')
        self.figure.update_yaxes(**opt, selector=axis)

    def update_axes(self):
        if not self.opt['primary']:
            return

        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict(font)
        if self.get_opt('trange', None) is not None:
            xaxis['range'] = pd_to_datetime(self.get_opt('trange'))
        self.figure.update_xaxes(**xaxis, selector=self.axes['xaxis'])

        yaxis = dict(font, title_text=self.get_opt('ylabel'))
        if self.get_opt('ytype', 'linear') == 'log':
            yaxis['type'] = 'log'
            if self.get_opt('yrange', None) is not None:
                yaxis['range'] = [np.log10(yr) for yr in self.get_opt('yrange')]
        else:
            if self.get_opt('yrange', None) is not None:
                yaxis['range'] = self.get_opt('yrange')
        self.figure.update_yaxes(**yaxis, selector=self.axes['yaxis'])


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

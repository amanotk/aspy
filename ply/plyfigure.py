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


def _get_legend_label(x, y, text, **opts):
    r = go.layout.Annotation(x=x, y=y, text=text, showarrow=False,
                             xref='paper', yref='paper',
                             xanchor='left', yanchor='middle', **opts)
    return r


def _get_legend_line(x0, y0, x1, y1, **opts):
    r = go.layout.Shape(type='line', x0=x0, x1=x1, y0=y0, y1=y1,
                        xref='paper', yref='paper', **opts)
    return r


class BaseFigure(object):
    def __init__(self, data, figure, axes, **options):
        self.data   = data
        self.figure = figure
        self.axes   = axes
        self.setup_options(options)
        self.setup_default_axes()

    def setup_options(self, options):
        self.opt = options.copy()
        self.opt['webgl']   = self.opt.get('webgl', False)
        self.opt['xtime']   = self.opt.get('xtime', True)
        self.opt['numplot'] = self.opt.get('numplot', 0)
        self.opt['numaxes'] = self.opt.get('numaxes', 0)

    def setup_default_axes(self):
        xaxis = dict(
            linewidth=self.opt['line_width'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
        )
        yaxis = dict(
            linewidth=self.opt['line_width'],
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
            showticklabels=True,
        )
        if self.opt['xtime']:
            xaxis.update(self.get_date_options())
        self.figure.update_xaxes(**xaxis, selector=self.axes['xaxis'])
        self.figure.update_yaxes(**yaxis, selector=self.axes['yaxis'])

    def get_opt(self, key, val=None):
        return get_plot_option(self.data, key, val)

    def get_date_options(self):
        opt = {
            'tickformatstops' : [
                dict(dtickrange=[None, 1000], value="%M:%S.%L"),
                dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
                dict(dtickrange=[60000, 3600000], value="%H:%M"),
                dict(dtickrange=[3600000, 86400000], value="%Y-%m-%d %H"),
                dict(dtickrange=[86400000, 604800000], value="%Y-%m-%d"),
                dict(dtickrange=[604800000, "M1"], value="%Y-%m"),
                dict(dtickrange=["M1", "M12"], value="%Y"),
                dict(dtickrange=["M12", None], value="%Y")
            ]
        }
        return opt

    def get_ylog_options(self, tickvals):
        tick1 = np.ceil(np.log10(tickvals))
        tick2 = np.floor(np.log10(tickvals))
        ticks = np.sort(np.unique(np.array(np.concatenate([tick1, tick2]),
                                           dtype=np.int32)))
        nt = len(ticks)
        tickvals = [0]*nt
        ticktext = [0]*nt
        for i in range(nt):
            tickvals[i] = 10.0**ticks[i]
            ticktext[i] = '10<sup>%+d</sup>' % (ticks[i])
        opt = {
            'type' : 'log',
            'tickvals' : tickvals,
            'ticktext' : ticktext,
        }
        return opt

    def add_legend(self, label, line):
        # for the first legend
        na   = self.opt['numaxes']
        if self.figure._legend[na] is None:
            self.figure._legend[na] = []

        nl = len(self.figure._legend[na])
        ts = self.opt['ticklength']/self.opt['width']
        fs = self.opt['fontsize']/self.opt['width']
        xx = self.axes['xaxis']['domain'][1] + 2.0*ts
        yy = self.axes['yaxis']['domain'][1] - 0.5*fs
        x0 = xx
        y0 = yy - nl*fs
        x1 = x0 + 2.0*fs
        y1 = y0
        x2 = x1 + 0.5*fs
        y2 = y1
        r1 = _get_legend_line(x0, y0, x1, y1, **line)
        r2 = _get_legend_label(x2, y2, **label)
        self.figure._legend[na].append(dict(line=r1, label=r2))
        # render
        line = list(self.figure.layout.shapes)
        text = list(self.figure.layout.annotations)
        line.append(r1)
        text.append(r2)
        self.figure.update_layout(annotations=text, shapes=line)

    def buildfigure(self):
        pass

    def update_axes(self):
        pass


class FigureLine(BaseFigure):
    def buildfigure(self):
        data = self.data

        # use WebGL version or not
        if self.opt['webgl']:
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
            lopt = dict(line_width=self.opt['line_width'])
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
                label = {
                    'text'      : legend_names[i],
                    'font_size' : self.opt['fontsize'],
                }
                self.add_legend(label, lopt)
            # plot
            opt.update(lopt)
            plot = scatter(x=x, y=y[:,i], mode='lines', **opt)
            self.figure.add_trace(plot)

        # update axes
        self.update_axes()

    def update_axes(self):
        if self.opt['numplot'] != 0:
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
                  outlinewidth=self.opt['line_width'],
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

        if isinstance(zz, ma.MaskedArray):
            zz = zz.filled(-np.inf)

        opt = dict(name='',
                   xaxis=self.axes['x'],
                   yaxis=self.axes['y'],
                   colorscale=cmap,
                   colorbar=cb,
                   zmin=zmin,
                   zmax=zmax)
        hm = go.Heatmap(z=zz.T, x=xx, y=yy, **opt)
        self.figure.add_trace(hm)
        self.plotdata = dict(x=xx, y=yy, z=zz)

        # update axes
        self.update_axes()

    def update_axes(self):
        if self.opt['numplot'] != 0:
            return

        font = dict(titlefont_size=self.opt['fontsize'],
                    tickfont_size=self.opt['fontsize'])

        xaxis = dict(font)
        if self.get_opt('trange', None) is not None:
            xaxis['range'] = pd_to_datetime(self.get_opt('trange'))
        self.figure.update_xaxes(**xaxis, selector=self.axes['xaxis'])

        yaxis = dict(font, title_text=self.get_opt('ylabel'))
        if self.get_opt('ytype', 'linear') == 'log':
            # log scale in y
            if self.get_opt('yrange', None) is not None:
                yrange = [np.log10(yr) for yr in self.get_opt('yrange')]
            else:
                y = self.plotdata['y']
                yrange = [np.log10(y.min()), np.log10(y.max())]
            # ticks
            ylogmin = int(np.ceil(yrange[0]))
            ylogmax = int(np.floor(yrange[1]))
            ticks   = 10**np.arange(ylogmin, ylogmax+1)
            yaxis['range'] = yrange
            yaxis.update(self.get_ylog_options(ticks))
        else:
            # linear scale in y
            if self.get_opt('yrange', None) is not None:
                yaxis['range'] = self.get_opt('yrange')
            else:
                yaxis['range'] = [y.min(), y.max()]
        self.figure.update_yaxes(**yaxis, selector=self.axes['yaxis'])


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

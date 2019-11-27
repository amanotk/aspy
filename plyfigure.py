# -*- coding: utf-8 -*-

""" Figures for plotly

"""

import re
import numpy as np
import pandas as pd

import plotly.graph_objects as go

from .utils import get_plot_options

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
        cmap = cmap
    if isinstance(cmap, str) and cmap in cmaptable:
        cmap = cmaptable[cmap]
    return cmap

class Legend(object):
    def __init__(self, label, opts, xpos, ypos):
        self.label = label
        self.opts = opts
        self.xpos = xpos
        self.ypos = ypos
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
        self.text = go.layout.Annotation(
            text=self.label, x=x2, y=y2, xref='paper', yref='paper',
            xanchor='left', yanchor='middle', showarrow=False)

    def get_line(self):
        return self.line

    def get_text(self):
        return self.text


class BaseFigure(object):
    def __init__(self, data, figure, row, col, **options):
        self.data   = data
        self.figure = figure
        self.row    = row
        self.col    = col
        self.webgl  = False
        self.setup_default_axes()
        self.setup_options(options)

    def setup_options(self, options):
        self.webgl = options.get('use_webgl', False)

    def setup_default_axes(self):
        xaxis = dict(
            linewidth=1,
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
        )
        yaxis = dict(
            linewidth=1,
            linecolor='#000',
            ticks='outside',
            mirror='allticks',
            showline=True,
        )
        self.figure.update_xaxes(**xaxis, row=self.row, col=self.col)
        self.figure.update_yaxes(**yaxis, row=self.row, col=self.col)

    def buildfigure(self):
        pass


class Figure1D(BaseFigure):
    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_options(self.data, key, val)
        data = self.data

        # use WebGL version or not
        if self.webgl:
            scatter = go.Scattergl
        else:
            scatter = go.Scatter

        x = pd.to_datetime(data.time, unit='s')
        y = np.atleast_2d(data.values)
        N = y.shape[1]

        legend_names = get_opt('legend')
        layout = self.figure.layout
        legend = list()
        for i in range(N):
            # line options
            lopt = dict(line_width=1)
            lc = get_opt('line_color')
            if lc is not None and len(lc) == N:
                lopt['line_color'] = _convert_color(lc[i])
            # legend
            opt = dict(lopt, showlegend=False)
            if legend_names is not None:
                xaxis = 'xaxis%d' % (self.row)
                yaxis = 'yaxis%d' % (self.row)
                xpos = layout[xaxis].domain[1] + 0.01
                ypos = layout[yaxis].domain[1] - 0.02 - 0.04*i
                legend.append(Legend(legend_names[i], lopt, xpos, ypos))
                opt  = dict(opt, name=legend_names[i])
            # plot
            plot = scatter(x=x, y=y[:,i], mode='lines', **opt)
            self.figure.add_trace(plot, row=self.row, col=self.col)

        # update axes
        xaxis = dict()
        yaxis = dict(title=get_opt('ylabel'))
        self.figure.update_xaxes(**xaxis, row=self.row, col=self.col)
        self.figure.update_yaxes(**yaxis, row=self.row, col=self.col)

        # legend
        text = list(layout.annotations)
        line = list(layout.shapes)
        for l in legend:
            text.append(l.get_text())
            line.append(l.get_line())
        self.figure.update_layout(annotations=text, shapes=line)


class FigureSpec(BaseFigure):
    def buildfigure(self):
        get_opt = lambda key, val=None: get_plot_options(self.data, key, val)
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

        # colorbar
        layout = self.figure.layout
        xaxis = 'xaxis%d' % (self.row)
        yaxis = 'yaxis%d' % (self.row)
        xpos = layout[xaxis].domain[1] + 0.01
        ypos = layout[yaxis].domain[0]
        ylen = layout[yaxis].domain[1] - ypos
        ypad = 0
        cb = dict(x=xpos, y=ypos, len=ylen, ypad=ypad, yanchor='bottom',
                  title=get_opt('zlabel'),
                  titleside='right')

        # colormap and range
        zmin, zmax = get_opt('zrange', [None, None])
        cmap = _get_colormap(get_opt('colormap'))

        # heatmap
        opt = dict(name='',
                   colorscale=cmap,
                   colorbar=cb,
                   zmin=zmin,
                   zmax=zmax)
        hm = go.Heatmap(z=z, x=x, y=y, **opt)
        self.figure.add_trace(hm, row=self.row, col=self.col)

        # update axes
        xaxis = dict()
        yaxis = dict(title=get_opt('ylabel'),
                     type=get_opt('ytype'))
        self.figure.update_xaxes(**xaxis, row=self.row, col=self.col)
        self.figure.update_yaxes(**yaxis, row=self.row, col=self.col)


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

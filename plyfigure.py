# -*- coding: utf-8 -*-

""" Figures for plotly

 $Id$
"""

import re
import numpy as np
import pandas as pd

import plotly.graph_objects as go


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
    def __init__(self, data, figure, row, col):
        self.data   = data
        self.figure = figure
        self.row    = row
        self.col    = col
        self.setup_default_axes()

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
        data = self.data
        popt = data.attrs['plot_options']
        x = pd.to_datetime(data.time, unit='s')
        y = np.atleast_2d(data.values)
        N = y.shape[1]

        if 'legend_names' in popt['yaxis_opt']:
            legend_names = popt['yaxis_opt']['legend_names']
        else:
            legend_names = None

        layout = self.figure.layout
        legend = list()
        for i in range(N):
            # legend
            lopt = dict(line_width=1)
            if 'line_color' in popt['extras']:
                lopt['line_color'] = \
                    _convert_color(popt['extras']['line_color'][i])
            if legend_names is not None:
                xaxis = 'xaxis%d' % (self.row)
                yaxis = 'yaxis%d' % (self.row)
                xpos = layout[xaxis].domain[1] + 0.01
                ypos = layout[yaxis].domain[1] - 0.02 - 0.04*i
                legend.append(Legend(legend_names[i], lopt, xpos, ypos))
            # plot
            opt  = dict(lopt, name=legend_names[i], showlegend=False)
            plot = go.Scatter(x=x, y=y[:,i], mode='lines', **opt)
            self.figure.add_trace(plot, row=self.row, col=self.col)

        # update axes
        xaxis = dict()
        yaxis = dict(title=popt['yaxis_opt']['axis_label'])
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
        data = self.data
        popt = data.attrs['plot_options']
        x = pd.to_datetime(data.time, unit='s')
        y = data.coords['spec_bins'][0]
        z = np.log10(data.values.T)

        # colorbar
        layout = self.figure.layout
        xaxis = 'xaxis%d' % (self.row)
        yaxis = 'yaxis%d' % (self.row)
        xpos = layout[xaxis].domain[1] + 0.01
        ypos = layout[yaxis].domain[0]
        ylen = layout[yaxis].domain[1] - ypos
        ypad = 0
        cb = dict(x=xpos, y=ypos, len=ylen, ypad=ypad, yanchor='bottom',
                  title=popt['zaxis_opt']['axis_label'], titleside='right')

        # heatmap
        opt = dict(name='', colorscale='viridis')
        hm = go.Heatmap(z=z, x=x, y=y, colorbar=cb, **opt)
        self.figure.add_trace(hm, row=self.row, col=self.col)

        # update axes
        xaxis = dict()
        yaxis = dict(title=popt['yaxis_opt']['axis_label'], type='log')
        self.figure.update_xaxes(**xaxis, row=self.row, col=self.col)
        self.figure.update_yaxes(**yaxis, row=self.row, col=self.col)


class FigureAlt(BaseFigure):
    pass


class FigureMap(BaseFigure):
    pass

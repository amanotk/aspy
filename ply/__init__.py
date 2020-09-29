# -*- coding: utf-8 -*-

"""Plot Module via Plotly

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import *
from .plyfigure import FigureLine, FigureSpec, FigureAlt, FigureMap


def generate_stack(var, **options):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    var = cast_list(cast_xarray(var))

    # get figure layout
    layout = get_figure_layout(var, **options)
    options.update(layout)

    # create figure
    figure_layout = {
        'width'    : options['width'],
        'height'   : options['height'],
        'margin'   : dict(t=0, b=0, l=0, r=0, pad=0, autoexpand=False),
        'plot_bgcolor' : '#fff',
    }
    bbox = options['bbox_relative']

    num_plots = len(var)
    for j in range(num_plots):
        x0 = bbox['x0'][j]
        x1 = bbox['x1'][j]
        y0 = bbox['y0'][j]
        y1 = bbox['y1'][j]
        j0 = '%d' % (num_plots)
        jj = '%d' % (j+1)
        figure_layout['xaxis' + jj] = dict(domain=[x0, x1],
                                           anchor='y' + jj,
                                           matches='x' + j0)
        figure_layout['yaxis' + jj] = dict(domain=[y0, y1],
                                           anchor='x' + jj)

    figure = go.Figure(layout=figure_layout)
    figure.__dict__['_legend'] = [None]*num_plots # hack !

    # store axes
    axs = list()
    for j in range(num_plots):
        j0 = '%d' % (num_plots)
        jj = '%d' % (j+1)
        axs.append(dict(numaxes=j+1,
                        x='x' + jj,
                        y='y' + jj,
                        xaxis='xaxis' + jj,
                        yaxis='yaxis' + jj))

    # plot
    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            obj = cls(var[j], figure, axs[j],
                      numaxes=j, numplot=0, **options)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            dat = var[j]
            for k in range(len(dat)):
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j],
                          numaxes=j, numplot=k, **options)
                obj.buildfigure()

    # show ticks only for the bottom
    opt = dict()
    for j in range(num_plots-1):
        opt['xaxis%d' % (j+1)] = dict(showticklabels=False)
    figure.update_layout(**opt)

    if 'title' in options:
        title = {
            'text'      : options['title'],
            'font_size' : options['fontsize'],
            'x'         : 0.5,
            'y'         : figure.layout.yaxis.domain[1] * 1.01,
            'xanchor'   : 'center',
            'yanchor'   : 'bottom',
        }
        figure.update_layout(title=title)

    if 'trange' in options:
        trange = pd_to_datetime(options['trange'])
        layout = dict()
        for j in range(num_plots):
            layout['xaxis%d' % (j+1)] = {'range' : trange}
        figure.update_layout(**layout)

    if is_ipython():
        figure.show()

    return figure

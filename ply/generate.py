# -*- coding: utf-8 -*-

""" Generate stack of plots

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
        figure_layout['xaxis' + jj] = dict(domain=[x0, x1], anchor='y' + j0)
        figure_layout['yaxis' + jj] = dict(domain=[y0, y1])

    figure = go.Figure(layout=figure_layout)

    # store axes
    axs = list()
    for j in range(num_plots):
        j0 = '%d' % (num_plots)
        jj = '%d' % (j+1)
        axs.append(dict(x='x' + j0,
                        y='y' + jj,
                        xaxis=figure.layout['xaxis' + jj],
                        yaxis=figure.layout['yaxis' + jj]))

    # plot
    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            obj = cls(var[j], figure, axs[j], **options)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            dat = var[j]
            for k in range(len(dat)):
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j], **options)
                obj.buildfigure()

    if 'title' in options:
        title = {
            'text'      : options['title'],
            'font_size' : options['fontsize'],
            'x'         : 0.5,
            'y'         : figure.layout.yaxis.domain[1] * 1.01,
            'xanchor'   : 'center',
            'yanchor'   : 'bottom',
        }
        print(title)
        figure.update_layout(title=title)
    if 'trange' in options:
        figure.update_xaxes(range=pd_to_datetime(options['trange']))

    if is_ipython():
        figure.show()

    return figure

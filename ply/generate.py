# -*- coding: utf-8 -*-

""" Generate stack of plots

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import default_layout
from ..utils import cast_xarray
from ..utils import cast_list
from ..utils import get_figure_class
from ..utils import get_figure_layout
from ..utils import get_layout_option
from .plyfigure import FigureLine, FigureSpec, FigureAlt, FigureMap

try:
    import pytplot
except:
    pytplot = None


def generate_stack(var, layout=None, options=None):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    var = cast_list(cast_xarray(var))

    # get figure layout
    if layout is None:
        layout = dict()
    bbox_pixels, bbox_relative = get_figure_layout(var, **layout)

    # create figure
    figure_layout = {
        'width'    : get_layout_option(layout, 'width'),
        'height'   : get_layout_option(layout, 'height'),
        'margin'   : dict(t=0, b=0, l=0, r=0, pad=0, autoexpand=False),
    }

    num_plots = len(var)
    for j in range(num_plots):
        x0 = bbox_relative['x0'][j]
        x1 = bbox_relative['x1'][j]
        y0 = bbox_relative['y0'][j]
        y1 = bbox_relative['y1'][j]
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
    if options is None:
        options = default_layout.copy()

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

    return figure

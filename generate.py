# -*- coding: utf-8 -*-

""" Generate stack of plots

 $Id$
"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plyfigure import Figure1D, FigureSpec, FigureAlt, FigureMap

try:
    import pytplot
except:
    pytplot = None


def _cast_xarray(var):
    "cast input (scalar or sequence) into list of xarray's DataArray"
    if isinstance(var, str) and pytplot is not None:
        return list([pytplot.data_quants[var]])
    elif isinstance(var, xr.DataArray):
        return list([var])
    elif hasattr(var, '__iter__'):
        return list([_cast_xarray(v) for v in var])
    else:
        raise ValueError('Unrecognized input')


def _get_figure_class(var):
    extras = var.attrs['plot_options']['extras']

    if 'spec' in extras:
        cls = FigureSpec
    elif 'alt' in extras:
        cls = FigureAlt
    elif 'map' in extras:
        cls = FigureMap
    else:
        cls = Figure1D

    return cls


def generate_stack(var, figure=None, layout=None):
    if figure is None:
        figure = dict()
    if layout is None:
        layout = dict()
    var = _cast_xarray(var)
    num_plots = len(var)

    figure = make_subplots(rows=num_plots, cols=1, **figure)
    figure.update_layout(**layout)

    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = _get_figure_class(var[j])
            plt = cls(var[j], figure, j+1, 1)
            plt.buildfigure()
        elif hasattr(var[j], '__iter__'):
            data = var[j]
            for k in range(len(data)):
                cls = _get_figure_class(data[k])
                plt = cls(data[k], figure, j+1, 1)
                plt.buildfigure()

    return figure

# -*- coding: utf-8 -*-

""" Generate stack of plots

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import _cast_xarray
from ..utils import _cast_list
from ..utils import get_figure_class
from .plyfigure import FigureLine, FigureSpec, FigureAlt, FigureMap

try:
    import pytplot
except:
    pytplot = None


def generate_stack(var, figure=None, layout=None, options=None):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    if figure is None:
        figure = dict()
    if layout is None:
        layout = dict()
    var = _cast_list(_cast_xarray(var))
    num_plots = len(var)

    figure = make_subplots(rows=num_plots, cols=1, **figure)
    figure.update_layout(**layout)

    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            plt = cls(var[j], figure, j+1, 1, **options)
            plt.buildfigure()
        elif hasattr(var[j], '__iter__'):
            data = var[j]
            for k in range(len(data)):
                cls = _get_figure_class(data[k])
                plt = cls(data[k], figure, j+1, 1)
                plt.buildfigure()

    return figure

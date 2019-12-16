# -*- coding: utf-8 -*-

""" Generate stack of plots

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from ..utils import cast_xarray
from ..utils import cast_list
from ..utils import get_figure_class
from ..utils import get_figure_layout
from ..utils import bbox_to_rect
from .mplfigure import FigureLine, FigureSpec, FigureAlt, FigureMap


def matplotlib_change_style(params):
    current = matplotlib.rcParams.copy()
    params  = params.copy()
    if 'backend' in params:
        params.pop('backend')
    matplotlib.rcParams.update(params)
    return current


def generate_stack(var, **options):
    classdict = {
        'Line' : FigureLine,
        'Spec' : FigureSpec,
        'Alt'  : FigureAlt,
        'Map'  : FigureMap,
    }

    # use deafult matplotlib style
    style = matplotlib_change_style(matplotlib.rcParamsDefault)

    var = cast_list(cast_xarray(var))

    # get figure layout
    layout = get_figure_layout(var, **options)

    # create figure
    dpi    = layout['dpi']
    width  = layout['width']
    height = layout['height']
    bbox   = layout['bbox_relative']
    figure = plt.figure(dpi=dpi, figsize=(width/dpi, height/dpi))
    axs    = list()

    # create axes
    num_plots = len(var)
    for j in range(num_plots):
        bb = {
            'x0' : bbox['x0'][j],
            'x1' : bbox['x1'][j],
            'y0' : bbox['y0'][j],
            'y1' : bbox['y1'][j],
        }
        rect = bbox_to_rect(bb)
        axs.append(plt.axes(rect))
    axs[0].get_shared_x_axes().join(*tuple(axs))

    for j in range(num_plots-1):
        axs[j].xaxis.set_ticklabels([])

    # plot
    for j in range(num_plots):
        if isinstance(var[j], xr.DataArray):
            cls = get_figure_class(var[j], classdict)
            obj = cls(var[j], figure, axs[j], **layout)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            dat = var[j]
            for k in range(len(dat)):
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j], **layout)
                obj.buildfigure()
    if 'title' in layout:
        axs[0].set_title(layout['title'], fontsize=layout['fontsize'])

    # restore
    style = matplotlib_change_style(style)

    plt.show()
    return figure

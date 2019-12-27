# -*- coding: utf-8 -*-

""" Generate stack of plots

"""

import re
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

from ..utils import *
from .mplfigure import \
    FigureLine, FigureSpec, FigureAlt, FigureMap, get_point_size


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
    options.update(layout)

    # create figure
    dpi    = options['dpi']
    width  = options['width']
    height = options['height']
    bbox   = options['bbox_relative']
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
            obj = cls(var[j], figure, axs[j], **options)
            obj.buildfigure()
        elif hasattr(var[j], '__iter__'):
            dat = var[j]
            for k in range(len(dat)):
                cls = get_figure_class(dat[k], classdict)
                obj = cls(dat[k], figure, axs[j], **options)
                obj.buildfigure()

    if 'title' in options:
        fontsize = get_point_size(options['fontsize'], options['dpi'])
        title = {
            'label'    : options['title'],
            'pad'      : fontsize,
            'fontdict' : dict(fontsize=fontsize),
        }
        axs[0].set_title(**title)
    if 'trange' in options:
        axs[0].set_xlim(pd_to_datetime(options['trange']))

    # restore
    style = matplotlib_change_style(style)

    plt.show()
    return figure
